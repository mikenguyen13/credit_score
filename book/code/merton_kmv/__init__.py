"""Production Merton-KMV pipeline.

Modules
-------
solver        Iterative KMV asset/volatility recovery, deterministic.
edf_map       Distance-to-default to PD calibration (Merton, isotonic).
feeds         Equity / debt / risk-free feed loaders (Parquet, synthetic).
pipeline      Per-firm orchestration with joblib parallelism.
monitoring    Drift, convergence, calibration, sector recalibration checks.
model_card    Markdown model-card generator (Mitchell et al., 2019).

The entry point used by the FastAPI service is :func:`pipeline.run_panel`,
which consumes a long-form equity panel and emits a long-form EDF panel
keyed by (firm_id, date).
"""

from .solver import (
    MertonKMVConfig,
    KmvResult,
    equity_from_V,
    kmv_solve,
    distance_to_default,
)
from .edf_map import dd_to_pd_normal, IsotonicEDF, rating_from_pd
from .feeds import EquityRecord, synthetic_equity_panel, write_parquet_panel, read_parquet_panel
from .pipeline import run_panel, FirmDiagnostics
from .monitoring import (
    sigma_v_drift,
    convergence_summary,
    pd_spread_rank_corr,
    binomial_backtest,
    hosmer_lemeshow,
    sector_recalibration,
)
from .model_card import render_model_card
from .vietnam import (
    TET_CLOSURES, vn_trading_calendar,
    free_float_equity, clean_vn_log_returns, annualise_sigma,
    VnDebtMapping, pit_to_ttc_pd, peer_sigma_lite,
    VnSectorParams, VN_LISTED_PARAMS, synthetic_vn_panel,
)

__all__ = [
    "MertonKMVConfig", "KmvResult",
    "equity_from_V", "kmv_solve", "distance_to_default",
    "dd_to_pd_normal", "IsotonicEDF", "rating_from_pd",
    "EquityRecord", "synthetic_equity_panel",
    "write_parquet_panel", "read_parquet_panel",
    "run_panel", "FirmDiagnostics",
    "sigma_v_drift", "convergence_summary", "pd_spread_rank_corr",
    "binomial_backtest", "hosmer_lemeshow", "sector_recalibration",
    "render_model_card",
    "TET_CLOSURES", "vn_trading_calendar",
    "free_float_equity", "clean_vn_log_returns", "annualise_sigma",
    "VnDebtMapping", "pit_to_ttc_pd", "peer_sigma_lite",
    "VnSectorParams", "VN_LISTED_PARAMS", "synthetic_vn_panel",
]
