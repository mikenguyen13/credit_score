"""Covariate-distribution overlap across censoring causes.

If administratively-censored loans, prepaid loans, and lender-closed
loans look the same on observed covariates, conditioning on x buys the
T independent of C given x assumption. If they do not, x is too narrow.

Two diagnostics, both standard in the causal-inference and
matching literature:

* Two-sample Kolmogorov-Smirnov test on each numeric covariate, run
  pairwise across cause cohorts.

* Standardised mean difference (SMD) per covariate per pair. SMD > 0.2
  is the conventional flag in propensity-score balance assessments
  [@austin2009balance].

The aggregated result is a tidy DataFrame with one row per
(covariate, cause_a, cause_b) plus an `any_imbalanced` flag at the
pair level for downstream reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass
class CauseOverlapResult:
    table: pd.DataFrame
    any_imbalanced: bool
    smd_threshold: float
    ks_p_threshold: float


def _smd(a: np.ndarray, b: np.ndarray) -> float:
    sd = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
    if sd == 0.0:
        return 0.0
    return float((a.mean() - b.mean()) / sd)


def cause_overlap(
    cause: pd.Series,
    covariates: pd.DataFrame,
    smd_threshold: float = 0.2,
    ks_p_threshold: float = 0.01,
    causes_to_compare: tuple[str, ...] = ("admin", "prepay", "default", "lender_close"),
) -> CauseOverlapResult:
    """Pairwise covariate overlap across cause cohorts.

    Pairs with fewer than 30 rows on either side are skipped to avoid
    spurious small-sample KS rejections.
    """
    rows = []
    causes_present = [c for c in causes_to_compare if (cause == c).sum() >= 30]
    for a, b in combinations(causes_present, 2):
        ma = (cause == a).values
        mb = (cause == b).values
        for col in covariates.columns:
            xa = covariates.loc[ma, col].to_numpy()
            xb = covariates.loc[mb, col].to_numpy()
            ks = ks_2samp(xa, xb)
            rows.append({
                "covariate": col,
                "cause_a": a,
                "cause_b": b,
                "n_a": int(ma.sum()),
                "n_b": int(mb.sum()),
                "mean_a": float(xa.mean()),
                "mean_b": float(xb.mean()),
                "smd": _smd(xa, xb),
                "ks_d": float(ks.statistic),
                "ks_p": float(ks.pvalue),
            })

    table = pd.DataFrame(rows)
    if table.empty:
        return CauseOverlapResult(
            table=table, any_imbalanced=False,
            smd_threshold=smd_threshold, ks_p_threshold=ks_p_threshold,
        )
    table["imbalanced"] = (
        (table["smd"].abs() >= smd_threshold)
        | (table["ks_p"] <= ks_p_threshold)
    )
    return CauseOverlapResult(
        table=table.sort_values(["cause_a", "cause_b", "smd"], key=lambda s: s if s.name != "smd" else s.abs(), ascending=[True, True, False]).reset_index(drop=True),
        any_imbalanced=bool(table["imbalanced"].any()),
        smd_threshold=smd_threshold,
        ks_p_threshold=ks_p_threshold,
    )
