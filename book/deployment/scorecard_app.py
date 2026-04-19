"""FastAPI wrapper around a serialized credit scorecard.

Run locally::

    uvicorn scorecard_app:app --host 0.0.0.0 --port 8000

The app expects a pickled ``optbinning.Scorecard`` artifact at
``SCORECARD_PATH`` (defaults to ``./artifacts/scorecard.pkl``) and exposes::

    POST /score         -> returns {"points": int, "pd": float, "reason_codes": [...]}
    GET  /healthz       -> liveness probe
    GET  /version       -> model metadata

The goal is a minimum viable reference implementation that a risk team can
adapt to its own CI/CD. Reason codes follow the FCRA / Regulation B pattern:
the bins whose WoE contribution pushes the applicant furthest below the
approved-population average are returned, ranked worst first.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

SCORECARD_PATH = Path(os.environ.get("SCORECARD_PATH", "artifacts/scorecard.pkl"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "scorecard_v1")
CUTOFF_POINTS = int(os.environ.get("CUTOFF_POINTS", "580"))
NUM_REASON_CODES = int(os.environ.get("NUM_REASON_CODES", "4"))


class Application(BaseModel):
    """One applicant record. Extra fields are ignored."""

    features: dict[str, Any] = Field(..., description="Feature name to raw value.")


class Decision(BaseModel):
    points: int
    pd: float
    decision: str
    reason_codes: list[str]
    model_version: str


app = FastAPI(title="Credit Scorecard", version=MODEL_VERSION)


def _load_scorecard() -> Any:
    if not SCORECARD_PATH.exists():
        raise FileNotFoundError(
            f"Scorecard artifact missing at {SCORECARD_PATH}. "
            "Train with the Chapter 7 pipeline and pickle the fitted object."
        )
    with SCORECARD_PATH.open("rb") as fh:
        return pickle.load(fh)


_SCORECARD = None


@app.on_event("startup")
def _warm() -> None:
    global _SCORECARD
    try:
        _SCORECARD = _load_scorecard()
    except FileNotFoundError:
        _SCORECARD = None


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok" if _SCORECARD is not None else "unloaded"}


@app.get("/version")
def version() -> dict[str, str]:
    return {"model_version": MODEL_VERSION, "artifact": str(SCORECARD_PATH)}


def _reason_codes(scorecard: Any, row: pd.DataFrame, k: int) -> list[str]:
    """Return the k features contributing the fewest points (weakest bins)."""
    table = scorecard.table(style="detailed")
    points_col = "Points"
    contributions = []
    for feat in table["Variable"].unique():
        sub = table[table["Variable"] == feat]
        raw = row[feat].iloc[0]
        # match optbinning binning: use .transform
        try:
            woe_val = scorecard.binning_process_.get_binned_variable(feat).transform(
                row[feat].to_numpy(), metric="woe"
            )[0]
        except Exception:
            continue
        # find bin whose WoE matches (approx)
        bin_row = sub.iloc[(sub["WoE"] - woe_val).abs().argsort().iloc[0]]
        contributions.append((feat, float(bin_row[points_col]), str(bin_row["Bin"])))
    contributions.sort(key=lambda t: t[1])
    return [f"{feat}={binlabel} (points={pts:.1f})" for feat, pts, binlabel in contributions[:k]]


@app.post("/score", response_model=Decision)
def score(app_in: Application) -> Decision:
    if _SCORECARD is None:
        raise HTTPException(status_code=503, detail="Scorecard not loaded")
    row = pd.DataFrame([app_in.features])
    try:
        pts = float(_SCORECARD.score(row)[0])
        pd_hat = float(_SCORECARD.predict_proba(row)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Scoring failed: {exc}") from exc
    reasons = _reason_codes(_SCORECARD, row, NUM_REASON_CODES)
    decision = "APPROVE" if pts >= CUTOFF_POINTS else "DECLINE"
    return Decision(
        points=int(round(pts)),
        pd=pd_hat,
        decision=decision,
        reason_codes=reasons,
        model_version=MODEL_VERSION,
    )


if __name__ == "__main__":
    # For quick local sanity check: construct a dummy app definition.
    print(json.dumps({"app": "scorecard", "version": MODEL_VERSION, "cutoff": CUTOFF_POINTS}))
