"""Orchestrator: cohort -> diagnostics artifact.

The single entry point for the package. Consumes a validated LoanCohort
and a configuration object and returns a DiagnosticsArtifact with all
four defensibility checks plus competing-risks CIF curves and per-horizon
PD readouts. The artifact serialises to JSON for the validation pack.

Failure mode: a single corrupt diagnostic should not bring down the
artifact. Each step is wrapped in a try/except that records the error
under `errors[<step>]` and continues; the FastAPI service then surfaces
the partial result with a non-empty errors block, which is the right
contract for an SR 11-7 validation pack (silence is worse than partial).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .schema import LoanCohort
from .ipcw import IpcwConfig, compute_ipcw, ipcw_kaplan_meier
from .tipping import TippingConfig, tipping_point_sweep
from .holdout import cohort_holdout_compare
from .overlap import cause_overlap
from .competing import aalen_johansen, fine_gray_admin_censoring
from .truncation import TruncationConfig, detect_truncation


@dataclass
class DiagnosticsConfig:
    horizons_months: tuple[int, ...] = (12, 24, 36)
    lifetime_horizon_months: Optional[int] = None
    ipcw: IpcwConfig = field(default_factory=IpcwConfig)
    tipping: TippingConfig = field(default_factory=TippingConfig)
    truncation: TruncationConfig = field(default_factory=TruncationConfig)
    smd_threshold: float = 0.2
    ks_p_threshold: float = 0.01
    fit_fine_gray: bool = True
    fit_aalen_johansen: bool = True
    clean_cohort_mask: Optional[np.ndarray] = None
    entry_age_months: Optional[np.ndarray] = None
    vintage_age_at_cutoff_months: Optional[np.ndarray] = None


@dataclass
class DiagnosticsArtifact:
    cohort: dict[str, Any]
    pd_at_horizons: dict[str, dict[str, float]]
    pd_lifetime: dict[str, Any]
    cause_overlap: dict[str, Any]
    ipcw_weights: dict[str, float]
    competing_risks: dict[str, Any]
    holdout: Optional[dict[str, Any]]
    fine_gray_default_coefs: Optional[dict[str, float]]
    truncation: Optional[dict[str, Any]]
    errors: dict[str, str]

    def to_json(self) -> str:
        return json.dumps(self._sanitise(asdict(self)), indent=2, default=str)

    def write(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())
        return p

    @staticmethod
    def _sanitise(obj):
        if isinstance(obj, dict):
            return {k: DiagnosticsArtifact._sanitise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [DiagnosticsArtifact._sanitise(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        return obj


def _safe(step: str, errors: dict[str, str], fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        errors[step] = f"{type(exc).__name__}: {exc}"
        return None


def run_diagnostics(
    cohort: LoanCohort,
    config: Optional[DiagnosticsConfig] = None,
) -> DiagnosticsArtifact:
    cfg = config or DiagnosticsConfig()
    errors: dict[str, str] = {}

    Y = cohort.duration
    E_def = cohort.event
    cause = cohort.cause
    X = cohort.covariates
    term = cohort.term_months
    lt_h = cfg.lifetime_horizon_months or (term - 1)

    pd_naive = ipcw_kaplan_meier(Y, E_def, np.ones_like(Y, dtype=float),
                                 list(cfg.horizons_months) + [lt_h])

    trunc_cfg = cfg.truncation
    if trunc_cfg.horizons_months != tuple(cfg.horizons_months):
        trunc_cfg = TruncationConfig(
            horizons_months=tuple(cfg.horizons_months),
            event_share_high=cfg.truncation.event_share_high,
            event_share_low=cfg.truncation.event_share_low,
            entry_age_min_months=cfg.truncation.entry_age_min_months,
            bias_block_bps=cfg.truncation.bias_block_bps,
        )
    trunc_res = _safe(
        "truncation", errors, detect_truncation,
        Y, E_def,
        entry=cfg.entry_age_months,
        vintage_age_at_cutoff=cfg.vintage_age_at_cutoff_months,
        config=trunc_cfg,
    )

    overlap_res = _safe("cause_overlap", errors, cause_overlap, cause, X,
                        cfg.smd_threshold, cfg.ks_p_threshold)

    ipcw_res = _safe("ipcw", errors, compute_ipcw, Y, cause, X, cfg.ipcw)
    pd_ipcw = None
    if ipcw_res is not None:
        pd_ipcw = ipcw_kaplan_meier(Y, E_def, ipcw_res.weights_capped,
                                    list(cfg.horizons_months) + [lt_h])

    censored_for_tipping = (cause.values == cfg.ipcw.censoring_cause) | (cause.values == "lender_close")
    tipping_cfg = cfg.tipping
    tipping_cfg.horizon_months = lt_h
    tip = _safe("tipping", errors, tipping_point_sweep, Y, E_def, censored_for_tipping, tipping_cfg)

    holdout = None
    if cfg.clean_cohort_mask is not None:
        h_res = _safe("holdout", errors, cohort_holdout_compare,
                      Y, E_def, cause.values, cfg.clean_cohort_mask, lt_h)
        if h_res is not None:
            holdout = asdict(h_res)

    cr: dict[str, Any] = {}
    if cfg.fit_aalen_johansen:
        aj = _safe("aalen_johansen", errors, aalen_johansen,
                   Y, cause, list(cfg.horizons_months) + [lt_h])
        if aj is not None:
            cr["aalen_johansen_horizon_pd"] = aj.horizon_pd

    fg_coefs = None
    if cfg.fit_fine_gray:
        fg = _safe("fine_gray", errors, fine_gray_admin_censoring,
                   Y, cause, X, term)
        if fg is not None:
            fg_coefs = {k: float(v) for k, v in fg.params_.to_dict().items()}

    pd_at_h: dict[str, dict[str, float]] = {"naive": pd_naive}
    if pd_ipcw is not None:
        pd_at_h["ipcw"] = pd_ipcw
    if cr.get("aalen_johansen_horizon_pd"):
        pd_at_h["aalen_johansen"] = {f"pd_{k}m": v for k, v in cr["aalen_johansen_horizon_pd"].items()}

    pd_lifetime = {
        "horizon_months": lt_h,
        "naive": pd_naive[f"pd_{int(lt_h)}m"],
        "ipcw": pd_ipcw[f"pd_{int(lt_h)}m"] if pd_ipcw else None,
        "tipping": {
            "rho_grid": tip.rho_grid.tolist() if tip else None,
            "lifetime_pd": tip.lifetime_pd.tolist() if tip else None,
            "decision_band": [tip.rho_at_decision_low, tip.rho_at_decision_high] if tip else None,
            "decision_band_min": tip.decision_band_min if tip else None,
            "decision_band_max": tip.decision_band_max if tip else None,
        } if tip else None,
        "clean_cohort_pd": holdout["pd_clean"] if holdout else None,
    }

    overlap_payload = {
        "any_imbalanced": overlap_res.any_imbalanced if overlap_res else None,
        "smd_threshold": cfg.smd_threshold,
        "ks_p_threshold": cfg.ks_p_threshold,
        "table": overlap_res.table.to_dict(orient="records") if overlap_res else [],
    }

    ipcw_payload = {
        "min": float(ipcw_res.weights.min()) if ipcw_res else None,
        "median": float(np.median(ipcw_res.weights)) if ipcw_res else None,
        "p99": float(np.quantile(ipcw_res.weights, 0.99)) if ipcw_res else None,
        "max": float(ipcw_res.weights.max()) if ipcw_res else None,
        "cap_value": ipcw_res.cap_value if ipcw_res else None,
        "cap_share": ipcw_res.cap_share if ipcw_res else None,
        "stabilised": ipcw_res.config.stabilise if ipcw_res else None,
    }

    truncation_payload = trunc_res.to_dict() if trunc_res is not None else None
    if truncation_payload and truncation_payload.get("blocks"):
        errors.setdefault(
            "truncation_block",
            "truncation correction differs from naive PD by more than "
            f"{trunc_cfg.bias_block_bps} bps at one or more horizons",
        )

    return DiagnosticsArtifact(
        cohort={
            "n": int(cohort.n),
            "term_months": term,
            "horizons_months": list(cfg.horizons_months),
            "lifetime_horizon_months": int(lt_h),
            "cause_counts": cohort.cause.value_counts().to_dict(),
            "default_share": float((E_def == 1).mean()),
        },
        pd_at_horizons=pd_at_h,
        pd_lifetime=pd_lifetime,
        cause_overlap=overlap_payload,
        ipcw_weights=ipcw_payload,
        competing_risks=cr,
        holdout=holdout,
        fine_gray_default_coefs=fg_coefs,
        truncation=truncation_payload,
        errors=errors,
    )
