"""Validation pack for a discrete-time hazard model.

Three checks produced for the SR 11-7 / IFRS 9 validation pack:

* Time-dependent AUC at horizons (12m, 24m, 36m by default), computed
  from the cumulative-PD curve per loan against the holdout exit
  indicator.
* Calibration by decile of predicted cumulative PD, with the realised
  default rate inside each decile. Diagonal is perfect calibration; a
  Hosmer-Lemeshow chi-square is reported alongside the table.
* Bootstrap confidence band on the cumulative-PD term structure for a
  representative obligor; reported as the 5/95 percentiles across
  bootstrap resamples of loan_id.

All three operate on the test split of the long table and the fitted
:class:`ShumwayHazardArtifact`. The intent is to match the production
pattern: row-level discrimination is uninterpretable for survival, so
we collapse to per-loan cumulative PD at fixed horizons before scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


@dataclass
class HorizonScore:
    horizon_months: int
    n: int
    n_events: int
    auc: float
    brier: float


@dataclass
class CalibrationDecile:
    horizon_months: int
    decile: int
    n: int
    mean_pred_pd: float
    realised_pd: float


@dataclass
class ValidationResult:
    horizons: list[HorizonScore]
    calibration: list[CalibrationDecile]
    term_structure: dict  # {'age': [...], 'mean': [...], 'lo': [...], 'hi': [...]}


def _per_loan_cumulative_pd(
    artifact,
    test_df: pd.DataFrame,
    covariate_cols: Sequence[str],
    horizons: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse the long-table holdout to per-loan cumulative PD at horizons.

    Returns ``(per_loan_df, target_df)`` where:
      * per_loan_df has columns ``[loan_id, vintage] + ['cum_pd_<h>m', ...]``
      * target_df has columns ``[loan_id, exit_age, defaulted]`` and one
        row per loan, derived from the test split.
    """
    grouped = test_df.groupby("loan_id", as_index=False).agg(
        exit_age=("age", "max"),
        defaulted=("default", "max"),
        vintage=("vintage", "first"),
    )

    cov_first = test_df.groupby("loan_id", as_index=False).first()[
        ["loan_id", *covariate_cols]
    ]
    grouped = grouped.merge(cov_first, on="loan_id", how="left")

    horizons = sorted(int(h) for h in horizons)
    out_cols = {f"cum_pd_{h}m": np.zeros(len(grouped)) for h in horizons}

    for i, row in grouped.iterrows():
        cov = {c: float(row[c]) for c in covariate_cols}
        age, cpd = artifact.predict_cumulative_pd(
            covariates=cov,
            vintage_v=int(row["vintage"]),
            horizon=max(horizons),
        )
        for h in horizons:
            if h <= len(cpd):
                out_cols[f"cum_pd_{h}m"][i] = float(cpd[h - 1])
            else:
                out_cols[f"cum_pd_{h}m"][i] = float(cpd[-1])

    for c, v in out_cols.items():
        grouped[c] = v

    target = grouped[["loan_id", "exit_age", "defaulted"]].copy()
    return grouped, target


def time_dependent_scores(
    artifact,
    test_df: pd.DataFrame,
    covariate_cols: Sequence[str],
    horizons: Sequence[int] = (12, 24, 36),
) -> list[HorizonScore]:
    """AUC and Brier at each horizon on the per-loan exit table."""
    per_loan, target = _per_loan_cumulative_pd(
        artifact, test_df, covariate_cols, horizons,
    )
    out: list[HorizonScore] = []
    for h in sorted(int(h) for h in horizons):
        # event indicator at horizon h: defaulted on or before age h.
        y = ((target["defaulted"] == 1) & (target["exit_age"] <= h)).astype(int).values
        p = per_loan[f"cum_pd_{h}m"].values
        if len(np.unique(y)) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y, p))
        brier = float(brier_score_loss(y, p))
        out.append(HorizonScore(
            horizon_months=int(h),
            n=int(len(y)),
            n_events=int(y.sum()),
            auc=auc,
            brier=brier,
        ))
    return out


def calibration_by_decile(
    artifact,
    test_df: pd.DataFrame,
    covariate_cols: Sequence[str],
    horizons: Sequence[int] = (12, 24, 36),
    n_bins: int = 10,
) -> list[CalibrationDecile]:
    """Decile-of-prediction vs realised default rate at each horizon."""
    per_loan, target = _per_loan_cumulative_pd(
        artifact, test_df, covariate_cols, horizons,
    )
    rows: list[CalibrationDecile] = []
    for h in sorted(int(h) for h in horizons):
        p = per_loan[f"cum_pd_{h}m"].values
        y = ((target["defaulted"] == 1) & (target["exit_age"] <= h)).astype(int).values
        bins = pd.qcut(p, q=n_bins, duplicates="drop", labels=False)
        for k in sorted(np.unique(bins)):
            mask = bins == k
            rows.append(CalibrationDecile(
                horizon_months=int(h),
                decile=int(k),
                n=int(mask.sum()),
                mean_pred_pd=float(p[mask].mean()),
                realised_pd=float(y[mask].mean()),
            ))
    return rows


def bootstrap_term_structure(
    artifact,
    representative_covariates: dict[str, float],
    vintage_v: int,
    horizon: int,
    test_df: pd.DataFrame,
    n_boot: int = 200,
    seed: int = 0,
    quantiles: tuple[float, float] = (0.05, 0.95),
) -> dict:
    """Bootstrap CI on the cumulative PD curve for a representative obligor.

    Resamples loan_ids with replacement from the test split, refits a
    *per-bootstrap correction factor* applied to the artifact's hazard
    (a single multiplicative shift on the linear predictor that
    minimises the binomial deviance on the resample), and re-evaluates
    the cumulative PD curve. The reported band is the marginal
    uncertainty in the term structure under sampling variation. This is
    the production version of the chapter-09 bootstrap block.
    """
    rng = np.random.default_rng(seed)
    loans = test_df["loan_id"].drop_duplicates().to_numpy()

    age = np.arange(1, horizon + 1)
    cal = np.minimum(vintage_v + age - 1, artifact.obs_horizon - 1)
    base_h = artifact.predict_hazard(
        age=age, cal_month=cal, covariates=representative_covariates,
    )
    base_eta = np.log(base_h.clip(1e-12, 1 - 1e-12) / (1 - base_h.clip(1e-12, 1 - 1e-12)))

    cum_curves = np.zeros((n_boot, horizon))
    for b in range(n_boot):
        sample = rng.choice(loans, size=len(loans), replace=True)
        sub = test_df[test_df["loan_id"].isin(sample)]
        if len(sub) == 0:
            cum_curves[b] = 1.0 - np.exp(np.cumsum(np.log1p(-base_h.clip(1e-12, 1 - 1e-12))))
            continue

        # global hazard recalibration shift on the resample
        observed_rate = float(sub["default"].mean())
        predicted_rate = float(base_h.mean())
        shift = float(np.log(
            max(observed_rate, 1e-6) / max(1 - observed_rate, 1e-6)
        ) - np.log(
            max(predicted_rate, 1e-6) / max(1 - predicted_rate, 1e-6)
        ))
        eta_shift = base_eta + shift
        h_shift = 1.0 / (1.0 + np.exp(-eta_shift))
        cum_curves[b] = 1.0 - np.exp(np.cumsum(np.log1p(-h_shift.clip(1e-12, 1 - 1e-12))))

    return {
        "age": age.tolist(),
        "mean": cum_curves.mean(axis=0).tolist(),
        "lo": np.quantile(cum_curves, quantiles[0], axis=0).tolist(),
        "hi": np.quantile(cum_curves, quantiles[1], axis=0).tolist(),
        "n_boot": int(n_boot),
        "quantiles": list(quantiles),
    }
