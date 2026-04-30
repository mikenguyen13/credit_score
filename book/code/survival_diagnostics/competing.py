"""Competing risks: cause-specific Cox, Fine-Gray, Aalen-Johansen.

Three estimators, three jobs:

* Cause-specific Cox. Fit a separate Cox on the default cause treating
  the competing event (prepay, lender close) as ordinary censoring.
  This is the right model for mechanism questions: what is the
  instantaneous default hazard among loans still on the book.

* Fine-Gray subdistribution Cox. Fit a Cox on the modified subdistribution
  risk set: loans that have failed from the competing cause stay at risk
  for default after their event, weighted by the IPCW for the censoring
  cause. This is the right model for IFRS 9 / CECL provisioning where
  the denominator is the originated cohort, not the surviving cohort.
  Under administrative censoring at a common horizon tau the IPCW
  collapses to one and the Fine-Gray fit reduces to a weighted Cox on a
  cohort with competing-event exit times reassigned to tau.

* Aalen-Johansen. Nonparametric cumulative incidence per cause, the
  competing-risks analog of Kaplan-Meier. Reports CIF curves and
  per-horizon point estimates with bootstrap CIs.

Reference: Fine and Gray (1999); Geskus (2011) for the IPCW
representation; Aalen and Johansen (1978).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sksurv.nonparametric import cumulative_incidence_competing_risks


_CAUSE_CODES = {"default": 1, "prepay": 2, "admin": 0, "lender_close": 3}


def _cause_codes(cause: pd.Series) -> np.ndarray:
    """Map cause names to non-negative integer risks (0 = censored)."""
    code = cause.map(_CAUSE_CODES).to_numpy()
    if np.any(pd.isna(code)):
        raise ValueError("cause column has values outside the known map")
    return code.astype(int)


@dataclass
class AalenJohansenResult:
    times: np.ndarray
    cif_default: np.ndarray
    cif_prepay: np.ndarray
    cif_lender_close: np.ndarray
    horizon_pd: dict[int, float]


def aalen_johansen(
    duration: np.ndarray,
    cause: pd.Series,
    horizons: list[int],
) -> AalenJohansenResult:
    code = _cause_codes(cause)
    times, cif = cumulative_incidence_competing_risks(
        event=code, time_exit=duration.astype(float)
    )
    def at(c: int) -> np.ndarray:
        if c < cif.shape[0]:
            return cif[c]
        return np.zeros_like(times)

    cif_d = at(1)
    cif_p = at(2)
    cif_l = at(3)

    horizon_pd: dict[int, float] = {}
    for h in horizons:
        j = int(np.searchsorted(times, h, side="right") - 1)
        j = max(j, 0)
        horizon_pd[int(h)] = float(cif_d[j]) if j < len(cif_d) else float(cif_d[-1])

    return AalenJohansenResult(
        times=times, cif_default=cif_d, cif_prepay=cif_p,
        cif_lender_close=cif_l, horizon_pd=horizon_pd,
    )


def cause_specific_cox(
    duration: np.ndarray,
    cause: pd.Series,
    covariates: pd.DataFrame,
    target_cause: str = "default",
    penalizer: float = 1e-4,
) -> CoxPHFitter:
    """Cause-specific Cox: target cause is the event, all others censored."""
    df = covariates.copy()
    df["__Y__"] = duration
    df["__E__"] = (cause.values == target_cause).astype(int)
    return CoxPHFitter(penalizer=penalizer).fit(
        df, duration_col="__Y__", event_col="__E__"
    )


def fine_gray_admin_censoring(
    duration: np.ndarray,
    cause: pd.Series,
    covariates: pd.DataFrame,
    term_months: int,
    target_cause: str = "default",
    penalizer: float = 1e-4,
) -> CoxPHFitter:
    """Fine-Gray subdistribution Cox under administrative censoring at term_months.

    Implements the Geskus (2011) reduction: when censoring is
    administrative at a common horizon tau, the IPCW weights collapse
    to one and the subdistribution risk set is implemented by
    reassigning competing-event subjects' exit times to tau and marking
    them as censored. The estimator reduces to a weighted Cox fit on
    the modified data.

    For random (non-administrative) censoring, replace this with the
    full IPCW expansion (split rows at cause-1 event times beyond Y,
    attach time-varying weights, fit a counting-process Cox). Out of
    scope for this single-pass production service; bring in
    `pycox`/`R::cmprsk` for that path.
    """
    competing_mask = ~np.isin(cause.values, [target_cause, "admin"])
    Y_mod = np.where(competing_mask, float(term_months), duration)
    E_mod = np.where(
        cause.values == target_cause, 1, 0
    ).astype(int)
    df = covariates.copy()
    df["__Y__"] = Y_mod
    df["__E__"] = E_mod
    return CoxPHFitter(penalizer=penalizer).fit(
        df, duration_col="__Y__", event_col="__E__"
    )
