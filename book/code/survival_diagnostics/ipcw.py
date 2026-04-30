"""Inverse-probability-of-censoring weighting (IPCW).

The censoring cause that matters in retail credit is prepayment, which
correlates with refreshed creditworthiness. If x captures the variation
that drives prepay, conditioning on x is enough; if not, IPCW corrects
the marginalisation by reweighting at-risk rows by 1 / S_C(Y_i^- | x_i),
where S_C is the survival function for the censoring time C.

Two production guards:

* Stabilised numerator. Replace the bare 1 / S_C with
  S_C^marg(Y_i^-) / S_C(Y_i^- | x_i) where S_C^marg is a covariate-free
  KM of the censoring cause. Stabilisation does not change the target
  estimand but shrinks weights to mean ~ 1 and tightens the bootstrap CI.

* Cap at the 99th percentile (or a user-specified quantile). A single
  row with weight 50 dominates the curve; capping trades a small bias
  for a large variance reduction. Report the cap and the share of rows
  affected in the model card.

Reference: Robins and Rotnitzky (1992); Fine and Gray (1999) for the
competing-risks adaptation; Cole and Hernan (2008) on stabilised weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter


@dataclass
class IpcwConfig:
    censoring_cause: str = "prepay"
    cap_quantile: float = 0.99
    stabilise: bool = True
    cox_penalizer: float = 1e-4
    s_c_floor: float = 0.05


@dataclass
class IpcwResult:
    weights: np.ndarray
    weights_capped: np.ndarray
    cap_value: float
    cap_share: float
    s_c_at_y: np.ndarray
    s_c_marg_at_y: Optional[np.ndarray]
    cox_summary: pd.DataFrame
    config: IpcwConfig


def compute_ipcw(
    duration: np.ndarray,
    cause: pd.Series,
    covariates: pd.DataFrame,
    config: Optional[IpcwConfig] = None,
) -> IpcwResult:
    """Cox-based IPCW for a single censoring cause.

    Treats `config.censoring_cause` as the event for the censoring Cox
    fit; everything else (default + admin + other causes) is
    right-censored from that perspective. The survival function S_C is
    evaluated at Y_i^- per row using a left-continuous lookup.
    """
    cfg = config or IpcwConfig()
    n = len(duration)
    event_C = (cause.values == cfg.censoring_cause).astype(int)
    if event_C.sum() < 5:
        raise ValueError(
            f"too few '{cfg.censoring_cause}' events ({event_C.sum()}) "
            f"to fit a censoring Cox; widen the cohort or fall back to a "
            f"covariate-free censoring KM"
        )

    df = covariates.copy()
    df["__Y__"] = duration
    df["__E_C__"] = event_C
    cph = CoxPHFitter(penalizer=cfg.cox_penalizer)
    cph.fit(df, duration_col="__Y__", event_col="__E_C__")

    times_grid = np.unique(np.append(duration, [0.0]))
    S_C = cph.predict_survival_function(covariates, times=times_grid)
    idx = np.searchsorted(times_grid, duration, side="right") - 1
    s_c_at_y = np.clip(S_C.values[idx, np.arange(n)], cfg.s_c_floor, 1.0)

    if cfg.stabilise:
        kmf = KaplanMeierFitter().fit(duration, event_C)
        s_c_marg_at_y = np.array([float(kmf.predict(t - 1e-9)) for t in duration])
        s_c_marg_at_y = np.clip(s_c_marg_at_y, cfg.s_c_floor, 1.0)
        w = s_c_marg_at_y / s_c_at_y
    else:
        s_c_marg_at_y = None
        w = 1.0 / s_c_at_y

    cap = float(np.quantile(w, cfg.cap_quantile))
    w_capped = np.minimum(w, cap)
    cap_share = float((w > cap).mean())

    return IpcwResult(
        weights=w,
        weights_capped=w_capped,
        cap_value=cap,
        cap_share=cap_share,
        s_c_at_y=s_c_at_y,
        s_c_marg_at_y=s_c_marg_at_y,
        cox_summary=cph.summary,
        config=cfg,
    )


def ipcw_kaplan_meier(
    duration: np.ndarray,
    event_default: np.ndarray,
    weights: np.ndarray,
    horizons: list[float],
) -> dict[str, float]:
    """Weighted KM of the default cause and PD readouts at fixed horizons.

    The weights argument is the (capped, possibly stabilised) IPCW from
    :func:`compute_ipcw`. The naive call passes weights = 1 to recover the
    baseline used in the headline curve.
    """
    kmf = KaplanMeierFitter().fit(duration, event_default, weights=weights)
    return {f"pd_{int(h)}m": 1.0 - float(kmf.predict(h)) for h in horizons}
