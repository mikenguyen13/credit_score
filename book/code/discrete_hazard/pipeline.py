"""Orchestrator: panel -> ShumwayHazardArtifact + validation pack.

Single entry point. Consumes a validated :class:`LongTablePanel` and a
:class:`ShumwayConfig`; returns a :class:`ShumwayPipelineArtifact`
that bundles the fitted hazard, the validation pack (time-dependent
AUC / Brier, decile calibration, bootstrap term-structure CIs), and
optional layer-2 forward-distribution PD readouts.

The artifact is JSON-serialisable (sans the fitted statsmodels object,
which lives only on the joblib pickle). The FastAPI service in
``deployment/discrete_hazard_app.py`` consumes the JSON.

Failure mode mirrors :mod:`survival_diagnostics.pipeline`: each step
is wrapped in a try/except that records the error under
``errors[<step>]`` and continues, so a single broken layer does not
take down the pack.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .fit import FitConfig, ShumwayHazardArtifact, fit_shumway_logit, vintage_grouped_split
from .layers import Ar1Process, forward_distribution_pd
from .schema import LongTablePanel
from .validation import (
    bootstrap_term_structure,
    calibration_by_decile,
    time_dependent_scores,
)


@dataclass
class ShumwayConfig:
    covariate_cols: Sequence[str]
    holdout_vintages: Sequence[int]
    horizons_months: tuple[int, ...] = (12, 24, 36)
    cluster_robust: bool = True
    baseline: str = "log_age"
    bootstrap_n: int = 200
    bootstrap_seed: int = 0
    representative_covariates: Optional[dict[str, float]] = None
    representative_vintage: Optional[int] = None
    representative_horizon: Optional[int] = None
    macro_paths: dict[str, np.ndarray] = field(default_factory=dict)
    forward_macro: Optional[str] = None  # name of macro path for layer-2 sim
    forward_n_paths: int = 2000


@dataclass
class ShumwayPipelineArtifact:
    fit_metadata: dict
    horizon_scores: list[dict]
    calibration_table: list[dict]
    term_structure: dict
    forward_distribution: Optional[dict]
    config: dict
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
            return {k: ShumwayPipelineArtifact._sanitise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [ShumwayPipelineArtifact._sanitise(v) for v in obj]
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


def run_shumway(
    panel: LongTablePanel,
    config: ShumwayConfig,
    artifact_path: Optional[Path | str] = None,
) -> tuple[ShumwayHazardArtifact, ShumwayPipelineArtifact]:
    """Fit the Shumway hazard, run the validation pack, persist the artifact."""
    errors: dict[str, str] = {}

    fit_cfg = FitConfig(
        covariate_cols=tuple(config.covariate_cols),
        holdout_vintages=tuple(config.holdout_vintages),
        cluster_robust=config.cluster_robust,
        baseline=config.baseline,
    )
    artifact = fit_shumway_logit(
        panel=panel, config=fit_cfg, macro_paths=dict(config.macro_paths),
    )

    train_df, test_df = vintage_grouped_split(panel, config.holdout_vintages)

    horizons = tuple(int(h) for h in config.horizons_months)

    scores = _safe(
        "horizon_scores", errors, time_dependent_scores,
        artifact, test_df, list(config.covariate_cols), horizons,
    ) or []
    calib = _safe(
        "calibration", errors, calibration_by_decile,
        artifact, test_df, list(config.covariate_cols), horizons,
    ) or []

    rep_cov = config.representative_covariates or {
        c: float(panel.covariates[c].median()) for c in config.covariate_cols
    }
    rep_v = (
        config.representative_vintage
        if config.representative_vintage is not None
        else int(np.median(np.unique(panel.vintage)))
    )
    rep_h = config.representative_horizon or int(panel.age.max())

    ts = _safe(
        "term_structure", errors, bootstrap_term_structure,
        artifact, rep_cov, rep_v, rep_h, test_df,
        config.bootstrap_n, config.bootstrap_seed,
    ) or {}

    fwd = None
    if config.forward_macro and config.forward_macro in artifact.macro_paths:
        macro_hist = artifact.macro_paths[config.forward_macro]
        proc = Ar1Process.from_path(macro_hist[: int(panel.cal_month.max()) + 1])
        u_today = float(macro_hist[min(rep_v, len(macro_hist) - 1)])
        fwd_age, fwd_mean, fwd_q = forward_distribution_pd(
            artifact=artifact,
            covariates=rep_cov,
            vintage_v=rep_v,
            horizon=rep_h,
            macro_name=config.forward_macro,
            process=proc,
            u_today=u_today,
            n_paths=config.forward_n_paths,
            seed=config.bootstrap_seed,
        )
        fwd = {
            "age": fwd_age.tolist(),
            "mean_cum_pd": fwd_mean.tolist(),
            "q05_cum_pd": fwd_q[0].tolist(),
            "q95_cum_pd": fwd_q[1].tolist(),
            "phi": proc.phi,
            "sigma": proc.sigma,
            "u_today": u_today,
            "n_paths": config.forward_n_paths,
        }

    pack = ShumwayPipelineArtifact(
        fit_metadata=artifact.metadata,
        horizon_scores=[asdict(s) for s in scores],
        calibration_table=[asdict(c) for c in calib],
        term_structure=ts,
        forward_distribution=fwd,
        config={
            "covariate_cols": list(config.covariate_cols),
            "holdout_vintages": sorted(int(v) for v in config.holdout_vintages),
            "horizons_months": list(horizons),
            "baseline": config.baseline,
            "cluster_robust": config.cluster_robust,
            "bootstrap_n": config.bootstrap_n,
            "forward_macro": config.forward_macro,
        },
        errors=errors,
    )

    if artifact_path is not None:
        ap = Path(artifact_path)
        artifact.write(ap)
        pack.write(ap.with_suffix(".pipeline.json"))

    return artifact, pack
