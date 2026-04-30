"""FastAPI service for the survival-censoring diagnostics artifact.

Run locally::

    uvicorn survival_diagnostics_app:app --host 0.0.0.0 --port 8002

The service reads a long-form loan cohort (Parquet) and exposes::

    POST /diagnostics/run            -> run diagnostics on a named cohort
    GET  /diagnostics/{vintage}      -> read the persisted artifact
    GET  /diagnostics/{vintage}/card -> markdown model card
    GET  /healthz                    -> liveness probe
    GET  /version                    -> service / package version

The diagnostics service is read-mostly: the heavy fit (Cox censoring
model, Aalen-Johansen, Fine-Gray) runs once per vintage in a batch job
(Airflow / Dagster) which writes the artifact JSON. The endpoints below
serve those artifacts and re-run on demand for ad-hoc validation
requests. Re-runs are O(n log n) in cohort size and complete in seconds
for vintages up to ~200k loans.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_BOOK_CODE = Path(__file__).resolve().parents[1] / "code"
if str(_BOOK_CODE) not in sys.path:
    sys.path.insert(0, str(_BOOK_CODE))

from survival_diagnostics import (
    DiagnosticsConfig,
    IpcwConfig,
    TippingConfig,
    TruncationConfig,
    SurvivalDiagnosticsCard,
    render_card,
    run_diagnostics,
    validate_cohort,
)

COHORT_ROOT = Path(os.environ.get("SD_COHORT_ROOT", "artifacts/cohorts"))
ARTIFACT_ROOT = Path(os.environ.get("SD_ARTIFACT_ROOT", "artifacts/survival_diagnostics"))
PACKAGE_VERSION = "1.0.0"


class RunRequest(BaseModel):
    vintage: str = Field(..., description="vintage tag, e.g. '2023-Q1'")
    covariate_cols: list[str] = Field(..., min_length=1)
    term_months: int = Field(..., gt=0, le=600)
    horizons_months: list[int] = Field(default=[12, 24, 36])
    censoring_cause: str = Field(default="prepay")
    cap_quantile: float = Field(default=0.99, ge=0.5, le=1.0)
    fit_fine_gray: bool = True
    fit_aalen_johansen: bool = True
    clean_cohort_query: Optional[str] = Field(
        default=None,
        description="pandas query string flagging the clean cohort (lifetime holdout)",
    )
    entry_age_col: Optional[str] = Field(
        default=None,
        description=(
            "column name (months) for delayed entry; if present and any row has "
            "entry > 0, the truncation diagnostic fits the left-truncated KM"
        ),
    )
    vintage_age_at_cutoff_col: Optional[str] = Field(
        default=None,
        description=(
            "column name (months) for tau_end - vintage; required to run the "
            "reverse-time KM correction when the cohort looks event-only"
        ),
    )
    truncation_bias_block_bps: float = Field(default=50.0, ge=0.0)


app = FastAPI(title="survival-diagnostics", version=PACKAGE_VERSION)


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "ok",
        "cohort_root": str(COHORT_ROOT),
        "artifact_root": str(ARTIFACT_ROOT),
        "ts": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/version")
def version() -> dict:
    return {
        "package": "survival_diagnostics",
        "version": PACKAGE_VERSION,
        "card": render_card(SurvivalDiagnosticsCard(version=PACKAGE_VERSION)),
    }


@app.post("/diagnostics/run")
def run(req: RunRequest) -> dict:
    cohort_path = COHORT_ROOT / f"{req.vintage}.parquet"
    if not cohort_path.exists():
        raise HTTPException(404, f"cohort not found: {cohort_path}")
    df = pd.read_parquet(cohort_path)

    cohort = validate_cohort(df, req.covariate_cols, req.term_months)

    clean_mask = None
    if req.clean_cohort_query:
        try:
            clean_idx = df.query(req.clean_cohort_query).index
            clean_mask = df.index.isin(clean_idx)
        except Exception as exc:
            raise HTTPException(400, f"bad clean_cohort_query: {exc}")

    entry_arr = None
    if req.entry_age_col:
        if req.entry_age_col not in df.columns:
            raise HTTPException(400, f"entry_age_col not in cohort: {req.entry_age_col}")
        entry_arr = df[req.entry_age_col].astype(float).to_numpy()

    cutoff_arr = None
    if req.vintage_age_at_cutoff_col:
        if req.vintage_age_at_cutoff_col not in df.columns:
            raise HTTPException(
                400,
                f"vintage_age_at_cutoff_col not in cohort: {req.vintage_age_at_cutoff_col}",
            )
        cutoff_arr = df[req.vintage_age_at_cutoff_col].astype(float).to_numpy()

    cfg = DiagnosticsConfig(
        horizons_months=tuple(req.horizons_months),
        ipcw=IpcwConfig(
            censoring_cause=req.censoring_cause,
            cap_quantile=req.cap_quantile,
        ),
        tipping=TippingConfig(),
        truncation=TruncationConfig(
            horizons_months=tuple(req.horizons_months),
            bias_block_bps=req.truncation_bias_block_bps,
        ),
        fit_fine_gray=req.fit_fine_gray,
        fit_aalen_johansen=req.fit_aalen_johansen,
        clean_cohort_mask=clean_mask,
        entry_age_months=entry_arr,
        vintage_age_at_cutoff_months=cutoff_arr,
    )

    artifact = run_diagnostics(cohort, cfg)
    out_path = ARTIFACT_ROOT / f"{req.vintage}.json"
    artifact.write(out_path)
    return {
        "vintage": req.vintage,
        "artifact_path": str(out_path),
        "errors": artifact.errors,
        "summary": {
            "n": artifact.cohort["n"],
            "default_share": artifact.cohort["default_share"],
            "pd_lifetime_naive": artifact.pd_lifetime["naive"],
            "pd_lifetime_ipcw": artifact.pd_lifetime["ipcw"],
            "tipping_decision_band": (
                [artifact.pd_lifetime["tipping"]["decision_band_min"],
                 artifact.pd_lifetime["tipping"]["decision_band_max"]]
                if artifact.pd_lifetime.get("tipping") else None
            ),
            "truncation": {
                "blocks": (artifact.truncation or {}).get("blocks"),
                "flags": (artifact.truncation or {}).get("flags"),
            } if artifact.truncation else None,
        },
    }


@app.get("/diagnostics/{vintage}")
def get_artifact(vintage: str) -> dict:
    p = ARTIFACT_ROOT / f"{vintage}.json"
    if not p.exists():
        raise HTTPException(404, f"artifact not found: {p}")
    return json.loads(p.read_text())


@app.get("/diagnostics/{vintage}/card")
def get_card(vintage: str) -> dict:
    p = ARTIFACT_ROOT / f"{vintage}.json"
    if not p.exists():
        raise HTTPException(404, f"artifact not found: {p}")
    return {
        "vintage": vintage,
        "card_markdown": render_card(SurvivalDiagnosticsCard(version=PACKAGE_VERSION)),
    }
