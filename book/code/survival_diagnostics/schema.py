"""Cohort schema for survival diagnostics.

A LoanCohort is the long-form input every other module in this package
consumes. One row per loan, observed at the analysis date. Time-varying
covariates are out of scope for this package; the cohort is a snapshot
and the diagnostics are vintage-level. For time-varying inputs, expand
the cohort to a counting-process panel upstream and feed the resulting
exit times here.

Required columns
----------------
loan_id      str          primary key
duration     float        observed time in months from origination to exit
event        int          1 = default, 0 = censored
cause        str          'default' | 'prepay' | 'admin' | 'lender_close'
                          (used to split censoring causes for diagnostics)
term_months  int          contractual term, used as the administrative cap
covariates   DataFrame    numeric covariate matrix (no NaNs); index aligned

Optional columns
----------------
vintage      str (YYYY-MM) origination month, used by holdout module
exposure     float        outstanding principal, used for exposure-weighted
                          aggregations in monitoring (not used in the core
                          diagnostics, which are sample-weighted)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


_VALID_CAUSES = frozenset({"default", "prepay", "admin", "lender_close"})


@dataclass
class LoanCohort:
    """Validated long-form loan-level cohort."""

    loan_id: pd.Series
    duration: np.ndarray
    event: np.ndarray
    cause: pd.Series
    term_months: int
    covariates: pd.DataFrame
    vintage: Optional[pd.Series] = None
    exposure: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        return len(self.duration)

    def cause_mask(self, name: str) -> np.ndarray:
        return (self.cause.values == name)


def validate_cohort(
    df: pd.DataFrame,
    covariate_cols: list[str],
    term_months: int,
) -> LoanCohort:
    """Strict validation of a cohort dataframe.

    Raises ValueError on any contract violation. Production callers
    should treat this as the boundary check; downstream modules trust
    the LoanCohort invariants.
    """
    required = {"loan_id", "duration", "event", "cause"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"cohort missing required columns: {sorted(missing)}")

    bad_cause = set(df["cause"].unique()) - _VALID_CAUSES
    if bad_cause:
        raise ValueError(
            f"cause column contains invalid values {sorted(bad_cause)}; "
            f"allowed: {sorted(_VALID_CAUSES)}"
        )

    if (df["duration"] <= 0).any():
        raise ValueError("duration must be strictly positive")
    if (df["duration"] > term_months + 1e-6).any():
        raise ValueError(
            "duration exceeds term_months; check administrative cap upstream"
        )

    ev = df["event"].astype(int).to_numpy()
    if not np.isin(ev, [0, 1]).all():
        raise ValueError("event must be in {0, 1}")
    if ((ev == 1) & (df["cause"] != "default")).any():
        raise ValueError("event=1 requires cause='default'")
    if ((ev == 0) & (df["cause"] == "default")).any():
        raise ValueError("event=0 must not have cause='default'")

    missing_cov = [c for c in covariate_cols if c not in df.columns]
    if missing_cov:
        raise ValueError(f"covariate columns not in cohort: {missing_cov}")
    X = df[covariate_cols].copy()
    if X.isna().any().any():
        raise ValueError("covariates contain NaNs; impute or drop upstream")
    if not all(np.issubdtype(X[c].dtype, np.number) for c in covariate_cols):
        raise ValueError("all covariate columns must be numeric")

    return LoanCohort(
        loan_id=df["loan_id"].astype(str).reset_index(drop=True),
        duration=df["duration"].astype(float).to_numpy(),
        event=ev,
        cause=df["cause"].astype(str).reset_index(drop=True),
        term_months=int(term_months),
        covariates=X.reset_index(drop=True),
        vintage=df["vintage"].astype(str).reset_index(drop=True)
            if "vintage" in df.columns else None,
        exposure=df["exposure"].astype(float).to_numpy()
            if "exposure" in df.columns else None,
    )
