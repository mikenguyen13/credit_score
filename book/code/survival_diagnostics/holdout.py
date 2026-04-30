"""Cohort holdout: prepay-heavy vs prepay-light vintage comparison.

If the bank has multiple vintages with materially different prepayment
intensity (a Tet-bonus vintage, a refi-wave vintage, a quiet-rate
vintage), the cohort holdout compares the lifetime PD on a low-prepay
("clean") subset to the full cohort. Persistent disagreement past what
covariates explain is informative-censoring evidence the IPCW model has
not absorbed.

In this package the split is purely operational: the user passes a
boolean mask flagging the clean cohort. Common rules:

* prepay_share <= q quantile of vintage-level prepay rate
* originated outside refi-pulse calendar window
* installment loans with strong prepay penalty (lower mechanical prepay)

The result is a structured comparison object suitable for the validation
pack table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lifelines import KaplanMeierFitter


@dataclass
class CohortHoldoutResult:
    pd_full: float
    pd_clean: float
    pd_full_minus_clean: float
    n_full: int
    n_clean: int
    prepay_share_full: float
    prepay_share_clean: float
    horizon_months: int


def cohort_holdout_compare(
    duration: np.ndarray,
    event_default: np.ndarray,
    cause: np.ndarray,
    clean_mask: np.ndarray,
    horizon_months: int,
) -> CohortHoldoutResult:
    """Compare lifetime PD on the clean vs full cohort.

    The "lifetime" horizon is typically the contractual term less one
    month, to keep the KM evaluation off the right edge of the support.
    """
    h = int(horizon_months)
    if clean_mask.sum() < 100:
        raise ValueError(
            f"clean cohort has only {int(clean_mask.sum())} rows; "
            f"too small for a stable KM at the lifetime horizon"
        )

    kmf_full = KaplanMeierFitter().fit(duration, event_default)
    kmf_clean = KaplanMeierFitter().fit(
        duration[clean_mask], event_default[clean_mask]
    )

    pd_full = 1.0 - float(kmf_full.predict(h))
    pd_clean = 1.0 - float(kmf_clean.predict(h))

    prepay_share_full = float((cause == "prepay").mean())
    prepay_share_clean = float((cause[clean_mask] == "prepay").mean())

    return CohortHoldoutResult(
        pd_full=pd_full,
        pd_clean=pd_clean,
        pd_full_minus_clean=pd_full - pd_clean,
        n_full=int(len(duration)),
        n_clean=int(clean_mask.sum()),
        prepay_share_full=prepay_share_full,
        prepay_share_clean=prepay_share_clean,
        horizon_months=h,
    )
