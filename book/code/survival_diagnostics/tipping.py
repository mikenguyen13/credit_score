"""Tipping-point sensitivity for non-administrative censoring.

The tipping-point method asks the dual question to IPCW: by how much
would censored rows have to default for the headline lifetime PD to
flip a decision? Replace the implicit assumption that censored rows
default at the residual at-risk rate with `rho` times that rate, sweep
rho over a grid, and report the lifetime-PD range.

Implementation note. The "naive baseline" S_hat(t) below is the
unweighted KM on the cohort treating all non-default exits as ordinary
censoring. The tipping curve is computed analytically off that baseline:
for a row censored at Y_i with conditional remainder S(M | x) / S(Y_i | x)
under hazard multiplier rho, the censored-row default contribution is
1 - (S(M) / S(Y_i))^rho.

The reported range over rho in [0.5, 2.0] is the conventional
robustness band; configure to widen for stress tests.

Reference: Greenland (1996); EMA / EFSPI sensitivity-analysis guidance
for missing-data principal stratification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d


@dataclass
class TippingConfig:
    rho_min: float = 0.25
    rho_max: float = 2.5
    n_grid: int = 19
    decision_band: tuple[float, float] = (0.5, 2.0)
    horizon_months: int = 36


@dataclass
class TippingResult:
    rho_grid: np.ndarray
    lifetime_pd: np.ndarray
    decision_band_min: float
    decision_band_max: float
    rho_at_naive: float
    rho_at_decision_low: float
    rho_at_decision_high: float
    config: TippingConfig


def tipping_point_sweep(
    duration: np.ndarray,
    event_default: np.ndarray,
    censored_mask: np.ndarray,
    config: TippingConfig,
) -> TippingResult:
    """Compute lifetime PD as a function of the censored-cohort hazard multiplier.

    The censored_mask flags rows treated as "potentially informative
    censoring" (typically prepay + lender-close); admin censoring at the
    contractual term is excluded since its censoring time is exogenous
    by construction.
    """
    kmf = KaplanMeierFitter().fit(duration, event_default, label="naive")
    base_S = kmf.survival_function_.iloc[:, 0]
    S_at = interp1d(
        base_S.index.values.astype(float),
        base_S.values,
        kind="previous",
        bounds_error=False,
        fill_value=(1.0, float(base_S.iloc[-1])),
    )

    horizon = config.horizon_months
    S_h = float(S_at(horizon))
    event_share = float((event_default == 1).mean())
    censored_share = float(censored_mask.mean())
    S_at_C = S_at(duration[censored_mask])
    S_at_C = np.clip(S_at_C, 1e-6, 1.0)

    rhos = np.linspace(config.rho_min, config.rho_max, config.n_grid)
    lifetime_pd = np.empty_like(rhos)
    for j, rho in enumerate(rhos):
        cond_surv = np.clip(S_h / S_at_C, 0.0, 1.0) ** rho
        pd_censored_contrib = float((1.0 - cond_surv).mean()) * censored_share
        lifetime_pd[j] = event_share + pd_censored_contrib

    band_lo, band_hi = config.decision_band
    band = (rhos >= band_lo) & (rhos <= band_hi)

    return TippingResult(
        rho_grid=rhos,
        lifetime_pd=lifetime_pd,
        decision_band_min=float(lifetime_pd[band].min()),
        decision_band_max=float(lifetime_pd[band].max()),
        rho_at_naive=float(rhos[np.argmin(np.abs(rhos - 1.0))]),
        rho_at_decision_low=band_lo,
        rho_at_decision_high=band_hi,
        config=config,
    )
