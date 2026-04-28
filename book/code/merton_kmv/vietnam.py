"""Vietnam-specific adapters for the Merton-KMV pipeline.

The chapter notes five concrete deviations from the developed-market
KMV reference that a Vietnamese deployment must handle:

1.  **Free-float wedge.** Reported market cap overstates economic
    equity because state and strategic stakes are not tradable. Use
    free-float-adjusted equity as the KMV input.
2.  **Volatility gaps.** Ex-dividend dates, trading-halt resumptions,
    and foreign-ownership-cap threshold hits create one-day jumps that
    a mechanical historical-volatility estimator treats as diffusion.
    Winsorise log-returns at a robust threshold before annualising.
3.  **Off-balance-sheet debt.** Conglomerate intra-group payables and
    parent-issued guarantees do not show up in DLTT or DLC. Augment
    the KMV face value with these items where disclosed.
4.  **Tet calendar.** HOSE/HNX close 5-9 trading days for Lunar New
    Year. Annualise on actual trading days observed, not 252.
5.  **PIT to TTC overlay.** SBV expects the IRB PD to be smoothed and
    macro-overlaid. The provided overlay applies a credit-cycle index
    multiplicatively, with a documented PIT->TTC mapping.

Every adapter returns plain NumPy / pandas objects so the rest of the
pipeline composes without modification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# Tet (Vietnamese Lunar New Year) HOSE/HNX closures, 2024-2027 reference.
# These are the published exchange-closure windows; the schedule is
# refreshed annually by SSC and the two exchanges.
TET_CLOSURES: dict[int, tuple[str, str]] = {
    2024: ("2024-02-08", "2024-02-14"),
    2025: ("2025-01-27", "2025-02-02"),
    2026: ("2026-02-16", "2026-02-22"),
    2027: ("2027-02-05", "2027-02-11"),
}


def vn_trading_calendar(start: str, end: str,
                        tet_closures: Optional[dict[int, tuple[str, str]]] = None) -> pd.DatetimeIndex:
    """Business days between ``start`` and ``end`` minus Tet closures."""
    tet_closures = tet_closures or TET_CLOSURES
    bdays = pd.bdate_range(start=start, end=end)
    blocked: list[pd.Timestamp] = []
    for lo, hi in tet_closures.values():
        blocked.extend(pd.bdate_range(lo, hi).tolist())
    blocked_idx = pd.DatetimeIndex(blocked)
    return bdays.difference(blocked_idx)


def free_float_equity(market_cap: np.ndarray, free_float_pct: float,
                      state_ownership_pct: float = 0.0) -> np.ndarray:
    """Adjust market cap to economic equity tradable on the open market.

    ``free_float_pct`` is the fraction of shares not locked by state,
    insider, or strategic holdings. ``state_ownership_pct`` is netted
    out separately when the data source reports it explicitly.
    """
    effective = max(min(free_float_pct - state_ownership_pct, 1.0), 0.05)
    return np.asarray(market_cap, dtype=float) * effective


def clean_vn_log_returns(prices: pd.Series,
                         dividend_dates: Optional[Iterable[str]] = None,
                         halt_dates: Optional[Iterable[str]] = None,
                         mad_k: float = 5.0) -> pd.Series:
    """Robust log-return cleaner for Vietnamese equity series.

    Drops the dividend-detachment day and trading-halt resumption day
    (their log-returns are events, not diffusion), then winsorises by
    a median-absolute-deviation rule at ``mad_k`` MADs.
    """
    prices = prices.sort_index()
    log_ret = np.log(prices).diff()
    if dividend_dates is not None:
        log_ret.loc[log_ret.index.intersection(pd.to_datetime(list(dividend_dates)))] = np.nan
    if halt_dates is not None:
        log_ret.loc[log_ret.index.intersection(pd.to_datetime(list(halt_dates)))] = np.nan
    log_ret = log_ret.dropna()
    med = log_ret.median()
    mad = (log_ret - med).abs().median()
    if mad > 0:
        thresh = mad_k * 1.4826 * mad
        log_ret = log_ret.clip(lower=med - thresh, upper=med + thresh)
    return log_ret


def annualise_sigma(log_returns: pd.Series, trading_days_per_year: int = 245) -> float:
    """Annualise daily log-return volatility using the actual VN trading-day count.

    The 245 default reflects HOSE / HNX losing roughly seven sessions
    per year to Tet plus public holidays, versus the 252 used in US
    deployments.
    """
    return float(log_returns.std() * np.sqrt(trading_days_per_year))


@dataclass
class VnDebtMapping:
    """KMV debt face value augmented for Vietnamese conglomerate structures."""
    short_term_debt: float
    long_term_debt: float
    off_bs_guarantees: float = 0.0
    intra_group_payables: float = 0.0
    half_long_term: bool = True

    def face_value(self) -> float:
        lt_factor = 0.5 if self.half_long_term else 1.0
        return float(
            self.short_term_debt
            + lt_factor * self.long_term_debt
            + self.off_bs_guarantees
            + self.intra_group_payables
        )


def pit_to_ttc_pd(pd_pit: np.ndarray, cycle_index: np.ndarray,
                   alpha: float = 0.5) -> np.ndarray:
    """Smooth PIT PD toward a TTC anchor using a credit-cycle index.

    ``cycle_index`` is centred at 1.0 (neutral cycle). Values above 1
    mark a loose cycle (observed PIT defaults are suppressed, so TTC
    adjusts the PD upward toward the long-run average); values below 1
    mark a tight cycle (PIT defaults are elevated, TTC pulls the PD
    downward). The smoother is

        TTC = (1 - alpha) * PIT + alpha * (PIT * cycle_index)
            = PIT * (1 + alpha * (cycle_index - 1))

    bounded to [1e-5, 0.5]. The convention matches the IFRS 9 PIT-TTC
    literature and the technical guidance in SBV Circular 13.
    """
    pit = np.asarray(pd_pit, dtype=float)
    cyc = np.asarray(cycle_index, dtype=float)
    overlay = pit * cyc
    ttc = (1.0 - alpha) * pit + alpha * overlay
    return np.clip(ttc, 1.0e-5, 0.5)


def peer_sigma_lite(target_accounting: pd.Series,
                    peer_panel: pd.DataFrame,
                    sector: str,
                    sigma_col: str = "sigma_V",
                    sector_col: str = "sector",
                    leverage_col: str = "leverage") -> float:
    """Borrow sigma_V from listed peers for an unlisted target.

    Uses the sector median ``sigma_V`` shrunk by a leverage adjustment.
    A higher leverage gap shrinks the borrowed sigma toward the sector
    floor, reflecting that high-leverage unlisted firms in Vietnam are
    typically opaque conglomerate subsidiaries whose true asset
    volatility is poorly identified.
    """
    peers = peer_panel[peer_panel[sector_col] == sector]
    if peers.empty:
        return float(peer_panel[sigma_col].median())
    median_sigma = float(peers[sigma_col].median())
    if leverage_col not in peers.columns or leverage_col not in target_accounting.index:
        return median_sigma
    median_lev = float(peers[leverage_col].median())
    target_lev = float(target_accounting[leverage_col])
    gap = abs(target_lev - median_lev)
    shrink = 1.0 / (1.0 + 2.0 * gap)
    floor = max(0.10, 0.6 * median_sigma)
    return float(shrink * median_sigma + (1.0 - shrink) * floor)


@dataclass
class VnSectorParams:
    """Sector parameter set for the Vietnamese listed universe."""
    sigma_A: float
    leverage: float
    mu_A: float
    free_float: float
    off_bs_load: float = 0.0


VN_LISTED_PARAMS: dict[str, VnSectorParams] = {
    # State-influenced utilities: low vol, high leverage, modest float.
    "Utilities_SOE": VnSectorParams(
        sigma_A=0.16, leverage=0.55, mu_A=0.05,
        free_float=0.25, off_bs_load=0.05,
    ),
    # Banks: high leverage, low asset volatility, moderate float,
    # heavy off-balance-sheet exposure (guarantees, LCs).
    "Banks": VnSectorParams(
        sigma_A=0.14, leverage=0.72, mu_A=0.07,
        free_float=0.30, off_bs_load=0.15,
    ),
    # Real estate conglomerates: large intra-group payables, high vol,
    # the sector that drove the 2022 corporate-bond freeze.
    "RealEstate": VnSectorParams(
        sigma_A=0.42, leverage=0.55, mu_A=0.09,
        free_float=0.35, off_bs_load=0.25,
    ),
    # Industrials: closer to developed-market parameters.
    "Industrials": VnSectorParams(
        sigma_A=0.30, leverage=0.40, mu_A=0.08,
        free_float=0.40, off_bs_load=0.05,
    ),
    # Consumer / retail: moderate leverage, moderate volatility.
    "Consumer": VnSectorParams(
        sigma_A=0.28, leverage=0.30, mu_A=0.09,
        free_float=0.45, off_bs_load=0.03,
    ),
}


def synthetic_vn_panel(
    n_firms_per_sector: int = 6,
    n_days: int = 252,
    seed: int = 20260428,
    start: str = "2025-04-28",
    macro_shock_window: Optional[tuple[str, str]] = ("2025-09-01", "2025-12-15"),
    macro_shock_vol_mult: float = 1.4,
):
    """Generate a synthetic VN30-style panel.

    Adds five Vietnam-specific behaviours on top of the developed-market
    synthetic generator:

    * Free-float-adjusted equity.
    * Off-balance-sheet debt loaded onto the KMV face value.
    * Tet trading-day gap.
    * One macro-shock window per panel (regime change in volatility).
    * One ex-dividend date and one trading-halt date per firm.
    """
    from .solver import equity_from_V

    rng = np.random.default_rng(seed)
    sector_keys = list(VN_LISTED_PARAMS.keys())
    dates = vn_trading_calendar(start=start,
                                end=str(pd.Timestamp(start) + pd.Timedelta(days=int(n_days * 1.6))))
    dates = dates[:n_days]
    if macro_shock_window is not None:
        shock_lo, shock_hi = pd.Timestamp(macro_shock_window[0]), pd.Timestamp(macro_shock_window[1])
        in_shock = (dates >= shock_lo) & (dates <= shock_hi)
    else:
        in_shock = np.zeros(len(dates), dtype=bool)

    equity_rows = []
    debt_rows = []
    meta_rows = []

    cfg_r = 0.04  # VN 1y T-bill anchor
    for sector in sector_keys:
        sp = VN_LISTED_PARAMS[sector]
        for k in range(n_firms_per_sector):
            firm_id = f"VN_{sector}_{k:02d}"
            V0 = 100.0 * float(rng.uniform(0.7, 1.4))
            D = sp.leverage * V0
            sigma_A = sp.sigma_A * float(rng.uniform(0.9, 1.1))
            mu_A = sp.mu_A
            dt = 1.0 / 245.0
            z = rng.standard_normal(len(dates))
            sigma_path = np.where(in_shock, sigma_A * macro_shock_vol_mult, sigma_A)
            V = np.empty(len(dates))
            V[0] = V0
            for t in range(1, len(dates)):
                V[t] = V[t - 1] * np.exp(
                    (mu_A - 0.5 * sigma_path[t] ** 2) * dt
                    + sigma_path[t] * np.sqrt(dt) * z[t]
                )
            E_full = equity_from_V(V, D, sigma_A, cfg_r, 1.0)
            E_full = np.maximum(E_full, 1.0e-3)
            # Observed equity is full market cap (price * total shares).
            # Free-float matters for liquidity and event-day amplitude,
            # not for the residual claim level. The cleaner below
            # winsorises the event days that thin float exaggerates.
            E_obs = E_full
            div_date = dates[len(dates) // 2]
            halt_date = dates[len(dates) // 3]
            E_obs = pd.Series(E_obs, index=dates)
            event_amp = 1.0 + (1.0 - sp.free_float)  # thinner float = larger event impact
            E_obs.loc[div_date] *= 1.0 - 0.06 * event_amp
            E_obs.loc[halt_date] *= 1.0 + 0.10 * event_amp
            equity_rows.append(pd.DataFrame({
                "firm_id": firm_id, "date": dates,
                "equity": E_obs.to_numpy(),
                "sector": sector,
            }))
            mapping = VnDebtMapping(
                short_term_debt=0.4 * D,
                long_term_debt=0.6 * D,
                off_bs_guarantees=sp.off_bs_load * D,
                intra_group_payables=0.5 * sp.off_bs_load * D,
            )
            debt_rows.append({
                "firm_id": firm_id,
                "debt": mapping.face_value(),
                "debt_naive": 0.4 * D + 0.5 * 0.6 * D,
            })
            meta_rows.append({
                "firm_id": firm_id, "sector": sector,
                "free_float": sp.free_float,
                "ex_div_date": div_date, "halt_date": halt_date,
                "true_sigma_A": sigma_A,
            })

    rate_df = pd.DataFrame({"date": dates, "r": cfg_r})
    return (
        pd.concat(equity_rows, ignore_index=True),
        pd.DataFrame(debt_rows),
        rate_df,
        pd.DataFrame(meta_rows),
    )
