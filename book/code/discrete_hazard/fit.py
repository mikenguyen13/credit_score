"""Discrete-time hazard fit: Shumway pooled logit on the long table.

@shumway2001forecasting writes the discrete-hazard log-likelihood as
a Bernoulli GLM on the loan-month panel. The fit below is the same
specification a regulated lender runs in production:

* vintage-grouped train/holdout split (cohort-out, not row-out, so the
  holdout sees only loans the training cohorts could not have seen);
* logistic regression on the long table with a flexible time baseline
  (intercept + log_age + age) and any number of time-varying covariates;
* cluster-robust standard errors on loan_id [@cameron2015practitioner];
* hashed, persisted artifact with the metadata an SR 11-7 reviewer
  expects (param hash, train counts, holdout vintages, fit date).

The persisted ShumwayHazardArtifact answers three production questions
with a single call: short-horizon PD for capital, lifetime PD for
IFRS 9 stage 2, and stressed lifetime PD under a macro-path override
for ICAAP. See `layers.forward_distribution_pd` for the Duffie-style
multi-horizon variant that integrates over a stochastic covariate path
instead of plugging in a frozen one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .schema import LongTablePanel


_BASELINE_FEATURES = ("const", "log_age", "age")


@dataclass
class FitConfig:
    covariate_cols: Sequence[str]
    holdout_vintages: Sequence[int]
    cluster_robust: bool = True
    baseline: str = "log_age"  # 'log_age' adds log_age + age; 'flat' uses const only
    maxiter: int = 200


@dataclass
class ShumwayHazardArtifact:
    """Persisted Shumway hazard model.

    The artifact is the production scoring contract: parameters,
    feature order, calendar paths for any time-varying covariate, and
    enough metadata to reproduce the fit.
    """

    params: pd.Series
    feature_order: tuple[str, ...]
    macro_paths: dict[str, np.ndarray] = field(default_factory=dict)
    obs_horizon: int = 0
    metadata: dict = field(default_factory=dict)

    def _design_row(
        self,
        age: np.ndarray,
        cal_month: np.ndarray,
        covariates: dict[str, np.ndarray | float],
        macro_override: Optional[dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        n = len(age)
        cols: dict[str, np.ndarray] = {}
        for name in self.feature_order:
            if name == "const":
                cols[name] = np.ones(n, dtype=float)
            elif name == "log_age":
                cols[name] = np.log(age.astype(float))
            elif name == "age":
                cols[name] = age.astype(float)
            elif name in self.macro_paths:
                path = (macro_override or {}).get(name, self.macro_paths[name])
                idx = np.clip(cal_month, 0, len(path) - 1)
                cols[name] = path[idx]
            elif name in covariates:
                v = covariates[name]
                cols[name] = np.full(n, float(v)) if np.isscalar(v) else np.asarray(v, dtype=float)
            else:
                raise KeyError(
                    f"feature '{name}' not provided as scalar covariate, "
                    f"calendar path, or baseline term"
                )
        return np.column_stack([cols[name] for name in self.feature_order])

    def predict_hazard(
        self,
        age: np.ndarray,
        cal_month: np.ndarray,
        covariates: dict[str, np.ndarray | float],
        macro_override: Optional[dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        X = self._design_row(np.asarray(age), np.asarray(cal_month), covariates, macro_override)
        beta = self.params[list(self.feature_order)].values
        eta = X @ beta
        return 1.0 / (1.0 + np.exp(-eta))

    def predict_survival(
        self,
        covariates: dict[str, float],
        vintage_v: int,
        horizon: int,
        macro_override: Optional[dict[str, np.ndarray]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        age = np.arange(1, horizon + 1)
        cal = np.minimum(vintage_v + age - 1, self.obs_horizon - 1)
        h = self.predict_hazard(age, cal, covariates, macro_override)
        S = np.exp(np.cumsum(np.log1p(-h.clip(1e-12, 1 - 1e-12))))
        return age, S

    def predict_cumulative_pd(
        self,
        covariates: dict[str, float],
        vintage_v: int,
        horizon: int,
        macro_override: Optional[dict[str, np.ndarray]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        age, S = self.predict_survival(covariates, vintage_v, horizon, macro_override)
        return age, 1.0 - S

    def write(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        meta_path = p.with_suffix(p.suffix + ".metadata.json")
        meta_path.write_text(json.dumps(self.metadata, indent=2, default=str))
        return p

    @staticmethod
    def read(path: Path | str) -> "ShumwayHazardArtifact":
        return joblib.load(Path(path))


def vintage_grouped_split(
    panel: LongTablePanel,
    holdout_vintages: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cohort-out split. Returns (train, test) frames with full covariates."""
    df = panel.to_frame()
    is_holdout = df["vintage"].isin(set(int(v) for v in holdout_vintages))
    return df.loc[~is_holdout].copy(), df.loc[is_holdout].copy()


def _design(df: pd.DataFrame, covariate_cols: Sequence[str], baseline: str) -> pd.DataFrame:
    out = pd.DataFrame({"const": 1.0}, index=df.index)
    if baseline == "log_age":
        out["log_age"] = np.log(df["age"].astype(float).values)
        out["age"] = df["age"].astype(float).values
    elif baseline == "flat":
        pass
    else:
        raise ValueError(f"unknown baseline: {baseline!r}; use 'log_age' or 'flat'")
    for c in covariate_cols:
        out[c] = df[c].astype(float).values
    return out


def _baseline_feature_order(baseline: str) -> tuple[str, ...]:
    if baseline == "log_age":
        return _BASELINE_FEATURES
    if baseline == "flat":
        return ("const",)
    raise ValueError(f"unknown baseline: {baseline!r}")


def fit_shumway_logit(
    panel: LongTablePanel,
    config: FitConfig,
    macro_paths: Optional[dict[str, np.ndarray]] = None,
) -> ShumwayHazardArtifact:
    """Fit Shumway's discrete-time hazard logit on the long table.

    Returns the persisted artifact. Cluster-robust SEs are computed at
    the loan_id level when `config.cluster_robust=True`; the fitted
    statsmodels Logit summary is stored under ``metadata['fit_summary']``.
    """
    train_df, _ = vintage_grouped_split(panel, config.holdout_vintages)
    X = _design(train_df, config.covariate_cols, config.baseline)
    y = train_df["default"].astype(int)

    if config.cluster_robust:
        result = sm.Logit(y, X).fit(
            disp=False,
            cov_type="cluster",
            cov_kwds={"groups": train_df["loan_id"].astype(str).values},
            maxiter=config.maxiter,
        )
    else:
        result = sm.Logit(y, X).fit(disp=False, maxiter=config.maxiter)

    feature_order = _baseline_feature_order(config.baseline) + tuple(config.covariate_cols)

    obs_horizon = int(panel.cal_month.max()) + 1
    paths = dict(macro_paths or {})

    metadata = {
        "fit_date": datetime.now(timezone.utc).isoformat(),
        "n_loans_train": int(train_df["loan_id"].nunique()),
        "n_rows_train": int(len(train_df)),
        "n_events_train": int(train_df["default"].sum()),
        "holdout_vintages": sorted(int(v) for v in config.holdout_vintages),
        "cov_type": "cluster(loan_id)" if config.cluster_robust else "nonrobust",
        "baseline": config.baseline,
        "feature_order": list(feature_order),
        "covariate_cols": list(config.covariate_cols),
        "param_hash": hashlib.sha256(
            result.params.to_json().encode()
        ).hexdigest()[:16],
        "loglike": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "fit_summary": str(result.summary()),
    }

    return ShumwayHazardArtifact(
        params=result.params,
        feature_order=feature_order,
        macro_paths=paths,
        obs_horizon=obs_horizon,
        metadata=metadata,
    )
