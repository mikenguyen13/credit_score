"""Equity, debt, and risk-free feed loaders.

Production deployments wire these adapters to Bloomberg/Refinitiv/IEX
(equity), Compustat (debt), and FRED (risk-free). For the book and for
unit tests we ship a synthetic generator that produces a panel with the
same schema, so the rest of the pipeline is exercised end-to-end without
licensed data.

Schema
------
Equity panel (long form):
    firm_id : str
    date    : datetime64[ns]
    equity  : float (market cap, in arbitrary currency units)
    sector  : str (one of {"Tech", "Utility", "Industrial", "Financial"})

Debt panel (per-firm scalar, refreshed quarterly):
    firm_id : str
    debt    : float (KMV mapping: short-term + 0.5 * long-term)

Rate panel:
    date    : datetime64[ns]
    r       : float (annualized continuously compounded risk-free)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class EquityRecord:
    firm_id: str
    date: pd.Timestamp
    equity: float
    sector: str


_SECTOR_PARAMS = {
    "Utility": {"sigma_A": 0.18, "leverage": 0.55, "mu_A": 0.05},
    "Industrial": {"sigma_A": 0.28, "leverage": 0.40, "mu_A": 0.08},
    "Financial": {"sigma_A": 0.18, "leverage": 0.65, "mu_A": 0.06},
    "Tech": {"sigma_A": 0.45, "leverage": 0.10, "mu_A": 0.12},
}


def synthetic_equity_panel(
    n_firms: int = 40,
    n_days: int = 252,
    seed: int = 20260428,
    start: str = "2025-04-28",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic Merton-consistent firm panel.

    Returns three DataFrames: equity, debt, rate. Asset values follow
    independent geometric Brownian motions per firm, equity is recovered
    from the Merton call, and the realized one-year default indicator is
    attached so back-tests can be wired up downstream.
    """
    from .solver import equity_from_V, MertonKMVConfig

    cfg = MertonKMVConfig()
    rng = np.random.default_rng(seed)
    sectors = list(_SECTOR_PARAMS.keys())
    dates = pd.bdate_range(start=start, periods=n_days)

    equity_rows = []
    debt_rows = []

    for k in range(n_firms):
        sector = sectors[k % len(sectors)]
        sp = _SECTOR_PARAMS[sector]
        V0 = 100.0 * float(rng.uniform(0.7, 1.4))
        D = sp["leverage"] * V0
        sigma_A = sp["sigma_A"] * float(rng.uniform(0.9, 1.1))
        mu_A = sp["mu_A"]
        dt = 1.0 / cfg.horizon_days
        z = rng.standard_normal(n_days)
        V = np.empty(n_days)
        V[0] = V0
        for t in range(1, n_days):
            V[t] = V[t - 1] * np.exp(
                (mu_A - 0.5 * sigma_A ** 2) * dt + sigma_A * np.sqrt(dt) * z[t]
            )
        E = equity_from_V(V, D, sigma_A, cfg.r, cfg.T)
        E = np.maximum(E, 1.0e-3)
        firm_id = f"SYN{k:03d}"
        equity_rows.append(pd.DataFrame({
            "firm_id": firm_id, "date": dates, "equity": E, "sector": sector,
        }))
        debt_rows.append({"firm_id": firm_id, "debt": D})

    equity_df = pd.concat(equity_rows, ignore_index=True)
    debt_df = pd.DataFrame(debt_rows)
    rate_df = pd.DataFrame({"date": dates, "r": cfg.r})
    return equity_df, debt_df, rate_df


def write_parquet_panel(df: pd.DataFrame, root: Path, partition_col: str = "date") -> Path:
    """Materialize a long-form panel to a date-partitioned Parquet lake."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    df.to_parquet(root, partition_cols=[partition_col], index=False)
    return root


def read_parquet_panel(root: Path, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Read a date-partitioned Parquet panel back."""
    return pd.read_parquet(Path(root), columns=columns)
