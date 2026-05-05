"""Counterfactual risk minimisation for pre-announced policy changes.

When the lender announces a policy change with a known effective date
the retrain pipeline can estimate through-the-door PD under the new
policy *before* any data is observed under it. This shaves the lag
between policy go-live and the first defensible PD curve.

Two estimands:

new_policy_pd            mean PD under the candidate policy, weighted
                         by pi_new / pi_log on the funded slice.

ess_under_new_policy     effective sample size under the importance-
                         weighted estimator. When the candidate policy
                         is far from the logged policy the variance
                         of the estimator blows up; the ESS is the
                         honest statistic that says how far the new
                         policy can move before a live experiment is
                         the cleaner option.

The estimators are unbiased under support containment: every applicant
who would be funded under the new policy must have had positive logged
probability under the old policy. The pipeline's ``check_support``
helper raises if the assumption is violated on the holdout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass
class CFRMResult:
    pd_under_new_policy: float
    ess: float
    n_clipped: int
    weight_cap: float
    support_warning: bool


def counterfactual_pd(
    pi_logged: np.ndarray,
    pi_new: np.ndarray,
    s: np.ndarray,
    y_funded: np.ndarray,
    funded_mask: np.ndarray,
    weight_cap: float = 20.0,
) -> CFRMResult:
    """Off-policy PD estimate via clipped importance weights.

    Weights = pi_new / pi_logged. Clipped at ``weight_cap`` to bound
    variance. Only the funded slice contributes; declined applicants
    have y unobserved and contribute via the support check only.
    """
    if (pi_logged <= 0).any():
        raise ValueError("pi_logged must be strictly positive")
    raw_w = pi_new / pi_logged
    w = np.clip(raw_w, 0.0, weight_cap)
    n_clipped = int((raw_w > weight_cap).sum())

    support_warn = bool(((pi_new > 0.0) & (pi_logged < 1e-3)).any())

    funded = funded_mask
    yf = y_funded.astype(float)
    wf = w[funded]
    if wf.sum() <= 0:
        return CFRMResult(float("nan"), 0.0, n_clipped, weight_cap, support_warn)

    pd_new = float((wf * yf).sum() / wf.sum())
    ess = float((wf.sum()) ** 2 / max((wf ** 2).sum(), 1e-12))
    return CFRMResult(pd_new, ess, n_clipped, weight_cap, support_warn)


def reliability_index(
    cfrm: CFRMResult, raw_funded_n: int,
    ess_floor_share: float = 0.10,
) -> dict[str, float | bool]:
    """Decide whether a CFRM estimate is trustworthy enough to ship.

    The 10 percent ESS floor follows @swaminathan2015counterfactual's
    rule of thumb for clipped IPW. Below that, the off-policy estimate
    is dominated by the logged policy and a small live experiment is
    the cleaner option.
    """
    return {
        "ess": cfrm.ess,
        "ess_share": cfrm.ess / max(raw_funded_n, 1),
        "ess_share_floor": ess_floor_share,
        "trustworthy": cfrm.ess / max(raw_funded_n, 1) >= ess_floor_share,
        "support_warning": cfrm.support_warning,
        "n_clipped": cfrm.n_clipped,
    }
