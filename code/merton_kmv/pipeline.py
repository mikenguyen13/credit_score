"""Per-firm orchestration for the Merton-KMV pipeline.

The job is embarrassingly parallel: each firm's KMV iteration is
independent of every other firm. We use joblib for portability across
laptops, Airflow workers, and Dagster ops; replacing the joblib backend
with a Spark/Dask/Ray runner is a one-line change.

The function :func:`run_panel` is the top-level entry point used by
both the batch ingestion job and the unit tests. It returns a long-form
EDF panel and a parallel diagnostics frame; downstream the EDF panel
goes to the serving store and the diagnostics frame goes to monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .solver import MertonKMVConfig, kmv_solve, distance_to_default
from .edf_map import dd_to_pd_normal, IsotonicEDF, rating_from_pd


@dataclass
class FirmDiagnostics:
    firm_id: str
    asof_date: pd.Timestamp
    n_iter: int
    final_residual: float
    max_damping_used: float
    fallback_used: bool
    converged: bool
    sigma_V: float
    DD: float
    error: Optional[str] = None


def _run_one_firm(
    firm_id: str,
    sector: str,
    equity: np.ndarray,
    dates: pd.DatetimeIndex,
    debt: float,
    cfg: MertonKMVConfig,
    edf_map: Optional[IsotonicEDF],
) -> tuple[pd.DataFrame, FirmDiagnostics]:
    """Run one firm. Errors are caught so a single bad firm cannot
    poison the rest of the panel."""
    asof = dates[-1]
    try:
        res = kmv_solve(equity, debt, cfg)
        dd = distance_to_default(res.V_path, res.sigma_V, debt, cfg.r, cfg.T)
        if edf_map is not None and edf_map.iso is not None:
            pd_hat = float(edf_map.predict(np.array([dd]))[0])
        else:
            pd_hat = float(dd_to_pd_normal(np.array([dd]))[0])
        edf_row = pd.DataFrame({
            "firm_id": [firm_id],
            "asof_date": [asof],
            "sector": [sector],
            "V": [float(res.V_path[-1])],
            "sigma_V": [res.sigma_V],
            "DD": [dd],
            "PD": [pd_hat],
            "rating": [rating_from_pd(pd_hat)],
        })
        diag = FirmDiagnostics(
            firm_id=firm_id, asof_date=asof,
            n_iter=res.n_iter, final_residual=res.final_residual,
            max_damping_used=res.max_damping_used,
            fallback_used=res.fallback_used, converged=res.converged,
            sigma_V=res.sigma_V, DD=dd, error=None,
        )
        return edf_row, diag
    except Exception as exc:
        empty = pd.DataFrame(columns=[
            "firm_id", "asof_date", "sector", "V", "sigma_V", "DD", "PD", "rating",
        ])
        diag = FirmDiagnostics(
            firm_id=firm_id, asof_date=asof, n_iter=-1,
            final_residual=float("nan"), max_damping_used=float("nan"),
            fallback_used=False, converged=False, sigma_V=float("nan"),
            DD=float("nan"), error=repr(exc),
        )
        return empty, diag


def run_panel(
    equity_df: pd.DataFrame,
    debt_df: pd.DataFrame,
    cfg: Optional[MertonKMVConfig] = None,
    edf_map: Optional[IsotonicEDF] = None,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the KMV solver across every firm in the panel.

    Parameters
    ----------
    equity_df : DataFrame with columns (firm_id, date, equity, sector)
    debt_df   : DataFrame with columns (firm_id, debt)
    cfg       : numerical configuration; defaults to ``MertonKMVConfig()``
    edf_map   : optional fitted DD->PD calibrator
    n_jobs    : joblib parallelism (1 = sequential)

    Returns
    -------
    (edf_df, diag_df)
        ``edf_df`` is the long-form EDF panel keyed by (firm_id,
        asof_date). ``diag_df`` is the parallel diagnostics frame.
    """
    cfg = cfg or MertonKMVConfig()
    debt_lookup = dict(zip(debt_df["firm_id"], debt_df["debt"]))

    work = []
    for firm_id, g in equity_df.sort_values("date").groupby("firm_id", sort=False):
        if firm_id not in debt_lookup:
            continue
        work.append((
            firm_id, g["sector"].iloc[0],
            g["equity"].to_numpy(dtype=float),
            pd.DatetimeIndex(g["date"].to_numpy()),
            float(debt_lookup[firm_id]),
        ))

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_run_one_firm)(fid, sec, eq, dt, debt, cfg, edf_map)
        for (fid, sec, eq, dt, debt) in work
    )

    edf_rows = [r[0] for r in results if not r[0].empty]
    diag_rows = [asdict(r[1]) for r in results]

    edf_df = pd.concat(edf_rows, ignore_index=True) if edf_rows else pd.DataFrame()
    diag_df = pd.DataFrame(diag_rows)
    return edf_df, diag_df
