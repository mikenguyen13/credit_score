"""Long-table panel schema for discrete-time (Shumway) hazard models.

A LongTablePanel is the loan-month (or firm-year) input every other
module in this package consumes. One row per (entity, age) period from
origination to exit. The discrete-hazard log-likelihood is identical
up to constants to a Bernoulli GLM on this table, so the schema
contract is intentionally tight: any deviation from the column set
below produces silently wrong PDs downstream.

Required columns
----------------
loan_id      str | int     entity primary key (loan, firm, account)
age          int           months (or years) since origination, >= 1
default      int           {0, 1}; 1 only on the exit row when the
                           entity defaults in that period
vintage      int           origination cohort index, used by the
                           vintage-grouped train/holdout split
cal_month    int           calendar month index, age + vintage - 1;
                           used for the layer-3 frailty and the
                           layer-1 CHS calendar join
covariates   DataFrame     numeric covariate matrix (no NaNs)

Optional columns
----------------
exposure     float         outstanding principal, used for exposure-
                           weighted aggregations in monitoring
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class LongTablePanel:
    """Validated long-form (entity, age) panel for a discrete hazard fit."""

    loan_id: pd.Series
    age: np.ndarray
    default: np.ndarray
    vintage: np.ndarray
    cal_month: np.ndarray
    covariates: pd.DataFrame
    exposure: Optional[np.ndarray] = None

    @property
    def n_rows(self) -> int:
        return int(len(self.age))

    @property
    def n_loans(self) -> int:
        return int(self.loan_id.nunique())

    @property
    def n_events(self) -> int:
        return int(self.default.sum())

    def to_frame(self) -> pd.DataFrame:
        out = pd.DataFrame({
            "loan_id": self.loan_id.values,
            "age": self.age,
            "default": self.default,
            "vintage": self.vintage,
            "cal_month": self.cal_month,
        })
        for c in self.covariates.columns:
            out[c] = self.covariates[c].values
        if self.exposure is not None:
            out["exposure"] = self.exposure
        return out


def validate_panel(
    df: pd.DataFrame,
    covariate_cols: list[str],
) -> LongTablePanel:
    """Strict validation of a long-table dataframe.

    Raises ValueError on contract violation. Downstream modules trust
    the LongTablePanel invariants: no NaNs in covariates, default in
    {0, 1}, age >= 1, cal_month consistent with age + vintage - 1, and
    at most one default = 1 row per loan_id.
    """
    required = {"loan_id", "age", "default", "vintage", "cal_month"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {sorted(missing)}")

    age = df["age"].astype(int).to_numpy()
    if (age < 1).any():
        raise ValueError("age must be >= 1 (origination month is 1, not 0)")

    d = df["default"].astype(int).to_numpy()
    if not np.isin(d, [0, 1]).all():
        raise ValueError("default must be in {0, 1}")

    vintage = df["vintage"].astype(int).to_numpy()
    cal = df["cal_month"].astype(int).to_numpy()
    expected_cal = vintage + age - 1
    if not np.array_equal(cal, expected_cal):
        bad = int((cal != expected_cal).sum())
        raise ValueError(
            f"cal_month must equal vintage + age - 1; "
            f"{bad} rows violate this invariant"
        )

    events_per_loan = (
        df.assign(_d=d).groupby("loan_id")["_d"].sum()
    )
    if (events_per_loan > 1).any():
        bad = int((events_per_loan > 1).sum())
        raise ValueError(
            f"each loan may default at most once on its exit row; "
            f"{bad} loan_ids have multiple default=1 rows"
        )

    missing_cov = [c for c in covariate_cols if c not in df.columns]
    if missing_cov:
        raise ValueError(f"covariate columns not in panel: {missing_cov}")
    X = df[covariate_cols].copy()
    if X.isna().any().any():
        raise ValueError("covariates contain NaNs; impute or drop upstream")
    if not all(np.issubdtype(X[c].dtype, np.number) for c in covariate_cols):
        raise ValueError("all covariate columns must be numeric")

    return LongTablePanel(
        loan_id=df["loan_id"].reset_index(drop=True),
        age=age,
        default=d,
        vintage=vintage,
        cal_month=cal,
        covariates=X.reset_index(drop=True),
        exposure=df["exposure"].astype(float).to_numpy()
            if "exposure" in df.columns else None,
    )
