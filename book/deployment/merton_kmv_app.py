"""FastAPI service for the Merton-KMV EDF panel.

Run locally::

    uvicorn merton_kmv_app:app --host 0.0.0.0 --port 8001

The service reads a long-form EDF panel (the output of
``merton_kmv.pipeline.run_panel`` written to Parquet) and exposes::

    GET  /firm/{firm_id}/edf?date=YYYY-MM-DD  -> latest <= date EDF
    GET  /firm/{firm_id}/history              -> per-firm time series
    GET  /healthz                              -> liveness probe
    GET  /version                              -> model metadata + card

The endpoint is the contract used by the bank's RAROC engine and
wholesale limits system. The service is read-only: estimation runs in a
separate batch job (Airflow/Dagster) which writes the EDF Parquet store
that this service mounts. That separation keeps inference latency in
single-digit milliseconds and lets the bank version the EDF artifact
under MLflow without restarting the service.
"""

from __future__ import annotations

import os
import sys
from datetime import date as Date
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# Make the in-repo package importable when running this file directly.
_BOOK_CODE = Path(__file__).resolve().parents[1] / "code"
if str(_BOOK_CODE) not in sys.path:
    sys.path.insert(0, str(_BOOK_CODE))

from merton_kmv.model_card import ModelCard, render_model_card

EDF_PATH = Path(os.environ.get("EDF_PATH", "artifacts/edf_panel.parquet"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "merton_kmv_v1")


class EdfRow(BaseModel):
    firm_id: str
    asof_date: Date
    sector: Optional[str] = None
    V: float
    sigma_V: float
    DD: float
    PD: float
    rating: str
    model_version: str


app = FastAPI(title="Merton-KMV EDF service", version=MODEL_VERSION)
_PANEL: Optional[pd.DataFrame] = None


def _load_panel() -> Optional[pd.DataFrame]:
    if not EDF_PATH.exists():
        return None
    df = pd.read_parquet(EDF_PATH)
    df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.date
    df = df.sort_values(["firm_id", "asof_date"])
    return df


@app.on_event("startup")
def _warm() -> None:
    global _PANEL
    _PANEL = _load_panel()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok" if _PANEL is not None else "unloaded"}


@app.get("/version")
def version() -> dict[str, object]:
    return {
        "model_version": MODEL_VERSION,
        "edf_path": str(EDF_PATH),
        "rows": int(0 if _PANEL is None else len(_PANEL)),
        "model_card_md": render_model_card(ModelCard(version=MODEL_VERSION)),
    }


@app.get("/firm/{firm_id}/edf", response_model=EdfRow)
def edf(firm_id: str, date: Optional[Date] = Query(default=None)) -> EdfRow:
    if _PANEL is None:
        raise HTTPException(status_code=503, detail="EDF panel not loaded")
    df = _PANEL[_PANEL["firm_id"] == firm_id]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"firm_id {firm_id} not found")
    if date is not None:
        df = df[df["asof_date"] <= date]
        if df.empty:
            raise HTTPException(status_code=404, detail="No EDF on or before requested date")
    row = df.iloc[-1]
    return EdfRow(
        firm_id=str(row["firm_id"]),
        asof_date=row["asof_date"],
        sector=str(row.get("sector", "")) or None,
        V=float(row["V"]), sigma_V=float(row["sigma_V"]),
        DD=float(row["DD"]), PD=float(row["PD"]),
        rating=str(row["rating"]),
        model_version=MODEL_VERSION,
    )


@app.get("/firm/{firm_id}/history")
def history(firm_id: str) -> list[dict]:
    if _PANEL is None:
        raise HTTPException(status_code=503, detail="EDF panel not loaded")
    df = _PANEL[_PANEL["firm_id"] == firm_id]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"firm_id {firm_id} not found")
    return df.to_dict(orient="records")
