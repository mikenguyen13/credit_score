"""Drift, calibration, and reconciliation monitors.

Each function returns a small DataFrame or a typed report so a routine
job (Airflow/cron) can serialize the output to a monitoring store and
alert on threshold breaches.

Monitors implemented:

* :func:`sigma_v_drift`           rolling z-score on a firm's sigma_V
* :func:`convergence_summary`     iteration-count and fallback rates
* :func:`pd_spread_rank_corr`     Spearman corr between PD and bond spread
* :func:`binomial_backtest`       per-bucket two-sided Binomial test
* :func:`hosmer_lemeshow`         decile chi-squared on PD calibration
* :func:`sector_recalibration`    median-pinned sector shrinkage of PD
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binom, chi2, spearmanr


def sigma_v_drift(history: pd.DataFrame, window: int = 90, z_thresh: float = 3.0) -> pd.DataFrame:
    """Rolling z-score on per-firm sigma_V.

    ``history`` must have columns ``(firm_id, asof_date, sigma_V)``.
    Returns one row per (firm_id, asof_date) with the rolling mean,
    standard deviation, z-score, and a boolean ``alert`` flag.
    """
    out = history.sort_values(["firm_id", "asof_date"]).copy()
    grp = out.groupby("firm_id")["sigma_V"]
    out["sigma_V_mean"] = grp.transform(lambda s: s.rolling(window, min_periods=10).mean())
    out["sigma_V_std"] = grp.transform(lambda s: s.rolling(window, min_periods=10).std())
    out["z"] = (out["sigma_V"] - out["sigma_V_mean"]) / out["sigma_V_std"]
    out["alert"] = out["z"].abs() >= z_thresh
    return out


def convergence_summary(diag_df: pd.DataFrame) -> dict:
    """Aggregate convergence diagnostics for a batch run."""
    n = len(diag_df)
    if n == 0:
        return {"n": 0}
    return {
        "n": int(n),
        "convergence_rate": float(diag_df["converged"].mean()),
        "fallback_rate": float(diag_df["fallback_used"].mean()),
        "mean_n_iter": float(diag_df.loc[diag_df["converged"], "n_iter"].mean()),
        "p95_n_iter": float(diag_df.loc[diag_df["converged"], "n_iter"].quantile(0.95)),
        "errors": int(diag_df["error"].notna().sum()),
    }


def pd_spread_rank_corr(pd_series: pd.Series, spread_series: pd.Series) -> float:
    """Spearman correlation between PD and bond spread on aligned indices."""
    df = pd.concat([pd_series, spread_series], axis=1, join="inner").dropna()
    if len(df) < 5:
        return float("nan")
    rho, _ = spearmanr(df.iloc[:, 0], df.iloc[:, 1])
    return float(rho)


def binomial_backtest(pd_forecast: np.ndarray, realized: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Per-bucket two-sided Binomial test of PD calibration.

    Each bucket holds firms with similar forecast PD. The expected
    default count is ``n * mean(PD)``; the realized count is the sum of
    the binary realized vector. The two-sided p-value is from a Binomial
    with ``mean(PD)`` as the success probability.
    """
    p = np.asarray(pd_forecast, dtype=float)
    y = np.asarray(realized, dtype=int)
    order = np.argsort(p)
    p_sorted = p[order]; y_sorted = y[order]
    edges = np.linspace(0, len(p), n_bins + 1, dtype=int)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        seg_p = p_sorted[lo:hi]; seg_y = y_sorted[lo:hi]
        n = len(seg_p); mean_p = float(seg_p.mean()); k = int(seg_y.sum())
        expected = n * mean_p
        # two-sided p-value via the test described in BCBS 2005 Annex A
        cdf_left = binom.cdf(k, n, mean_p)
        sf_right = binom.sf(k - 1, n, mean_p) if k > 0 else 1.0
        p_two = float(min(1.0, 2.0 * min(cdf_left, sf_right)))
        rows.append({
            "bucket": i, "n": n, "mean_PD": mean_p,
            "expected": expected, "realized": k, "p_value": p_two,
        })
    return pd.DataFrame(rows)


def hosmer_lemeshow(pd_forecast: np.ndarray, realized: np.ndarray, g: int = 10) -> dict:
    """Hosmer-Lemeshow chi-squared test on decile calibration."""
    p = np.asarray(pd_forecast, dtype=float)
    y = np.asarray(realized, dtype=int)
    order = np.argsort(p)
    p_s = p[order]; y_s = y[order]
    edges = np.linspace(0, len(p), g + 1, dtype=int)
    chi = 0.0; df_used = 0
    for i in range(g):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        n_g = hi - lo
        o1 = float(y_s[lo:hi].sum()); o0 = n_g - o1
        e1 = float(p_s[lo:hi].sum()); e0 = n_g - e1
        if e1 < 1.0e-9 or e0 < 1.0e-9:
            continue
        chi += (o1 - e1) ** 2 / e1 + (o0 - e0) ** 2 / e0
        df_used += 1
    df_ = max(df_used - 2, 1)
    return {"chi2": float(chi), "df": df_, "p_value": float(1.0 - chi2.cdf(chi, df_))}


def sector_recalibration(panel: pd.DataFrame, sector_col: str = "sector",
                         pd_col: str = "PD", anchor: float = 0.02,
                         shrinkage: float = 0.5) -> pd.DataFrame:
    """Pull each firm's PD toward a sector-level anchor.

    The recalibrated PD is

        PD_adj = (1 - lam) * PD + lam * (anchor / sector_median) * PD

    where ``lam = shrinkage``. This closes the structural gap noted in
    the chapter (utilities over-stated, tech under-stated) by tying each
    sector's median back to a target ``anchor``.
    """
    out = panel.copy()
    sector_med = out.groupby(sector_col)[pd_col].transform("median")
    factor = (anchor / sector_med).clip(lower=0.1, upper=10.0)
    out["PD_adj"] = (1.0 - shrinkage) * out[pd_col] + shrinkage * factor * out[pd_col]
    return out
