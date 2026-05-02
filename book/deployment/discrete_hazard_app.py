"""FastAPI service for the discrete-time (Shumway) hazard model.

Run locally::

    uvicorn discrete_hazard_app:app --host 0.0.0.0 --port 8003

The service reads a long-form (loan, age) panel (Parquet) and exposes::

    POST /shumway/fit                    -> fit + persist artifact + pack
    GET  /shumway/{vintage_tag}          -> read the persisted pipeline pack
    GET  /shumway/{vintage_tag}/card     -> markdown model card
    POST /shumway/{vintage_tag}/score    -> score a payload of obligor states
    GET  /healthz                        -> liveness probe
    GET  /version                        -> service / package version

The fit is read-mostly: the heavy logit + bootstrap + forward
distribution runs once per vintage in a batch job (Airflow / Dagster)
which writes the pickled artifact and the pipeline-pack JSON. The
score endpoint then produces hazards, survival, and cumulative PD on
demand from the persisted artifact.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_BOOK_CODE = Path(__file__).resolve().parents[1] / "code"
if str(_BOOK_CODE) not in sys.path:
    sys.path.insert(0, str(_BOOK_CODE))

from discrete_hazard import (
    ShumwayConfig,
    ShumwayHazardArtifact,
    ShumwayModelCard,
    add_calendar_covariates,
    render_card,
    run_shumway,
    validate_panel,
)

PANEL_ROOT = Path(os.environ.get("DH_PANEL_ROOT", "artifacts/panels"))
ARTIFACT_ROOT = Path(os.environ.get("DH_ARTIFACT_ROOT", "artifacts/discrete_hazard"))
PACKAGE_VERSION = "1.0.0"


class FitRequest(BaseModel):
    vintage_tag: str = Field(..., description="cohort tag, e.g. '2026-Q1'")
    covariate_cols: list[str] = Field(..., min_length=1)
    holdout_vintages: list[int] = Field(..., min_length=1)
    horizons_months: list[int] = Field(default=[12, 24, 36])
    cluster_robust: bool = True
    baseline: str = Field(default="log_age", pattern="^(log_age|flat)$")
    bootstrap_n: int = Field(default=200, ge=10, le=5000)
    forward_macro: Optional[str] = None
    forward_n_paths: int = Field(default=2000, ge=50, le=20000)
    macro_path_cols: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "name -> calendar-indexed parquet column file: "
            "loaded from PANEL_ROOT/<file>.parquet under column 'value'"
        ),
    )


class ScoreRequest(BaseModel):
    covariates: dict[str, float] = Field(...,
        description="scalar covariate values for one obligor")
    vintage_v: int = Field(..., ge=0)
    horizon: int = Field(..., gt=0, le=600)
    macro_override: Optional[dict[str, list[float]]] = Field(
        default=None,
        description="optional name -> calendar-indexed array, replaces fitted path",
    )


app = FastAPI(title="discrete-hazard", version=PACKAGE_VERSION)


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "ok",
        "panel_root": str(PANEL_ROOT),
        "artifact_root": str(ARTIFACT_ROOT),
        "ts": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/version")
def version() -> dict:
    return {
        "package": "discrete_hazard",
        "version": PACKAGE_VERSION,
        "card": render_card(ShumwayModelCard(version=PACKAGE_VERSION)),
    }


def _load_macro_paths(spec: dict[str, str]) -> dict[str, np.ndarray]:
    paths: dict[str, np.ndarray] = {}
    for name, file in spec.items():
        p = PANEL_ROOT / f"{file}.parquet"
        if not p.exists():
            raise HTTPException(404, f"macro path file not found: {p}")
        df = pd.read_parquet(p)
        if "value" not in df.columns:
            raise HTTPException(400, f"macro path file {p} missing 'value' column")
        paths[name] = df["value"].astype(float).to_numpy()
    return paths


@app.post("/shumway/fit")
def fit(req: FitRequest) -> dict:
    panel_path = PANEL_ROOT / f"{req.vintage_tag}.parquet"
    if not panel_path.exists():
        raise HTTPException(404, f"panel not found: {panel_path}")
    df = pd.read_parquet(panel_path)

    macro_paths = _load_macro_paths(req.macro_path_cols)
    if macro_paths:
        df = add_calendar_covariates(df, macro_paths)

    panel = validate_panel(df, req.covariate_cols)
    cfg = ShumwayConfig(
        covariate_cols=tuple(req.covariate_cols),
        holdout_vintages=tuple(req.holdout_vintages),
        horizons_months=tuple(req.horizons_months),
        cluster_robust=req.cluster_robust,
        baseline=req.baseline,
        bootstrap_n=req.bootstrap_n,
        macro_paths=macro_paths,
        forward_macro=req.forward_macro,
        forward_n_paths=req.forward_n_paths,
    )
    out = ARTIFACT_ROOT / f"{req.vintage_tag}.pkl"
    artifact, pack = run_shumway(panel, cfg, artifact_path=out)
    return {
        "vintage_tag": req.vintage_tag,
        "artifact_path": str(out),
        "pack_path": str(out.with_suffix(".pipeline.json")),
        "errors": pack.errors,
        "summary": {
            "n_loans": panel.n_loans,
            "n_rows": panel.n_rows,
            "n_events": panel.n_events,
            "param_hash": artifact.metadata["param_hash"],
            "horizon_scores": pack.horizon_scores,
            "lifetime_pd_mean_at_max_h": (
                pack.term_structure["mean"][-1] if pack.term_structure else None
            ),
        },
    }


@app.get("/shumway/{vintage_tag}")
def get_pack(vintage_tag: str) -> dict:
    p = ARTIFACT_ROOT / f"{vintage_tag}.pipeline.json"
    if not p.exists():
        raise HTTPException(404, f"pipeline pack not found: {p}")
    return json.loads(p.read_text())


@app.get("/shumway/{vintage_tag}/card")
def get_card(vintage_tag: str) -> dict:
    return {
        "vintage_tag": vintage_tag,
        "card_markdown": render_card(ShumwayModelCard(version=PACKAGE_VERSION)),
    }


@app.post("/shumway/{vintage_tag}/score")
def score(vintage_tag: str, req: ScoreRequest) -> dict:
    p = ARTIFACT_ROOT / f"{vintage_tag}.pkl"
    if not p.exists():
        raise HTTPException(404, f"artifact not found: {p}")
    artifact = ShumwayHazardArtifact.read(p)

    macro_override = None
    if req.macro_override:
        macro_override = {
            name: np.asarray(arr, dtype=float)
            for name, arr in req.macro_override.items()
        }

    age, S = artifact.predict_survival(
        covariates=req.covariates,
        vintage_v=req.vintage_v,
        horizon=req.horizon,
        macro_override=macro_override,
    )
    cum_pd = (1.0 - S).tolist()
    return {
        "vintage_tag": vintage_tag,
        "vintage_v": req.vintage_v,
        "horizon": req.horizon,
        "age": age.tolist(),
        "survival": S.tolist(),
        "cumulative_pd": cum_pd,
        "param_hash": artifact.metadata.get("param_hash"),
    }
