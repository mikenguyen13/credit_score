"""FastAPI service for the reject-inference retrain pipeline.

Run locally::

    uvicorn reject_inference_app:app --host 0.0.0.0 --port 8003

Endpoints
---------

POST /retrain/observable      run observable-mode retrain on a parquet
                              cohort and persist the artifact
POST /retrain/unobservable    same, unobservable mode (Heckman stage-1)
POST /retrain/alt_data        per-lender hierarchical retrain
POST /promote                 evaluate a stored challenger against the
                              champion on the frozen holdout; emit the
                              SR 11-7 memo and gate decision
POST /cfrm                    counterfactual PD under a pre-announced
                              policy change
GET  /retrain/{snapshot}      read a persisted artifact JSON
GET  /version                 service / package version + model card
GET  /healthz                 liveness probe

The service is read-mostly: the heavy retrain runs nightly via a
scheduler (Airflow / Dagster / Prefect) which posts to /retrain. The
gate decision is consumed by the model registry to flip the champion
in lockstep across stage-1 and stage-2.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_BOOK_CODE = Path(__file__).resolve().parents[1] / "code"
if str(_BOOK_CODE) not in sys.path:
    sys.path.insert(0, str(_BOOK_CODE))

from reject_inference_pipeline import (
    PolicyVersion,
    RetrainConfig,
    counterfactual_pd, reliability_index,
    gated_promote, write_artifact,
    join_snapshot_outcomes,
    render_card, RejectInferenceCard,
    retrain_alt_data, retrain_observable, retrain_unobservable,
    validate_applicant_snapshot, validate_bureau_outcomes,
)
from reject_inference_pipeline.operational_state import (
    OperationalState, check_freeze, load_state,
    set_macro_shock, set_bureau_outage, set_iv_kill,
)


COHORT_ROOT = Path(os.environ.get("RI_COHORT_ROOT", "artifacts/ri_cohorts"))
OUTCOMES_ROOT = Path(os.environ.get("RI_OUTCOMES_ROOT", "artifacts/ri_outcomes"))
ARTIFACT_ROOT = Path(os.environ.get("RI_ARTIFACT_ROOT", "artifacts/ri_pipeline"))
PACKAGE_VERSION = "1.0.0"


class PolicyPayload(BaseModel):
    policy_version_id: str
    effective_from: str
    effective_to: Optional[str] = None
    propensity_mode: str = Field(..., description="observable|unobservable|alt_data")
    iv_columns: list[str]
    label_definition_id: str
    cutoff: Optional[float] = None
    override_quota: Optional[float] = None
    notes: str = ""


class RetrainRequest(BaseModel):
    snapshot_date: str = Field(..., description="ISO date, e.g. '2026-05-01'")
    cohort_name: str
    feature_cols: list[str] = Field(..., min_length=1)
    iv_cols: list[str] = Field(default_factory=list)
    performance_window_months: int = Field(..., gt=0, le=120)
    bootstrap_B: int = 0
    aipw_n_splits: int = 5
    iv_p_threshold: float = 0.05
    run_aipcw: bool = False
    seed: int = 20260504
    policy: PolicyPayload


class AltDataRetrainRequest(RetrainRequest):
    lender_id_col: str
    own_score_logged_col: Optional[str] = None
    shrinkage_lambda: float = 0.5


class PromoteRequest(BaseModel):
    snapshot_date: str
    challenger_artifact: str
    champion_artifact: str
    holdout_cohort_name: str
    feature_cols: list[str]
    iv_cols: list[str] = Field(default_factory=list)
    threshold: float
    reference_group: str
    drift_reason: str
    auc_pvalue: float = 0.05
    di_floor: float = 0.80
    shadow_psi: Optional[float] = None


class CFRMRequest(BaseModel):
    cohort_name: str
    feature_cols: list[str]
    iv_cols: list[str] = Field(default_factory=list)
    performance_window_months: int
    snapshot_date: str
    pi_logged_col: str = "pi_logged"
    pi_new_col: str = "pi_new"
    weight_cap: float = 20.0


app = FastAPI(title="reject-inference-pipeline", version=PACKAGE_VERSION)


def _build_policy(p: PolicyPayload) -> PolicyVersion:
    if p.propensity_mode not in ("observable", "unobservable", "alt_data"):
        raise HTTPException(400, f"bad propensity_mode: {p.propensity_mode}")
    return PolicyVersion(
        policy_version_id=p.policy_version_id,
        effective_from=pd.Timestamp(p.effective_from),
        effective_to=(pd.Timestamp(p.effective_to) if p.effective_to else None),
        propensity_mode=p.propensity_mode,  # type: ignore[arg-type]
        iv_columns=tuple(p.iv_columns),
        label_definition_id=p.label_definition_id,
        cutoff=p.cutoff,
        override_quota=p.override_quota,
        notes=p.notes,
    )


def _read_cohort(name: str) -> pd.DataFrame:
    path = COHORT_ROOT / f"{name}.parquet"
    if not path.exists():
        raise HTTPException(404, f"cohort not found: {path}")
    return pd.read_parquet(path)


def _read_outcomes(name: str) -> pd.DataFrame:
    path = OUTCOMES_ROOT / f"{name}.parquet"
    if not path.exists():
        raise HTTPException(404, f"outcomes not found: {path}")
    return pd.read_parquet(path)


def _build_join(req: RetrainRequest) -> Any:
    df_apps = _read_cohort(req.cohort_name)
    df_y = _read_outcomes(req.cohort_name)
    apps = validate_applicant_snapshot(
        df_apps,
        feature_cols=req.feature_cols,
        iv_cols=req.iv_cols,
        require_pi_logged=(req.policy.propensity_mode == "observable"),
    )
    outcomes = validate_bureau_outcomes(
        df_y, y_definition_id=req.policy.label_definition_id)
    return join_snapshot_outcomes(
        apps, outcomes,
        snapshot_date=pd.Timestamp(req.snapshot_date),
        performance_window_months=req.performance_window_months,
    )


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "ok",
        "cohort_root": str(COHORT_ROOT),
        "outcomes_root": str(OUTCOMES_ROOT),
        "artifact_root": str(ARTIFACT_ROOT),
        "ts": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/version")
def version() -> dict:
    return {
        "package": "reject_inference_pipeline",
        "version": PACKAGE_VERSION,
        "card": render_card(RejectInferenceCard(version=PACKAGE_VERSION)),
    }


@app.post("/retrain/observable")
def retrain_observable_endpoint(req: RetrainRequest) -> dict:
    if req.policy.propensity_mode != "observable":
        raise HTTPException(400, "propensity_mode must be 'observable'")
    joined = _build_join(req)
    policy = _build_policy(req.policy)
    cfg = RetrainConfig(
        snapshot_date=pd.Timestamp(req.snapshot_date),
        performance_window_months=req.performance_window_months,
        bootstrap_B=req.bootstrap_B,
        aipw_n_splits=req.aipw_n_splits,
        iv_p_threshold=req.iv_p_threshold,
        run_aipcw=req.run_aipcw,
        seed=req.seed,
    )
    art = retrain_observable(joined, cfg, policy)
    out = ARTIFACT_ROOT / f"{req.policy.policy_version_id}_{req.snapshot_date}_observable.json"
    write_artifact(art, out)
    return {
        "artifact": str(out),
        "iv_blocked": art.iv_diagnostic.iv_blocked if art.iv_diagnostic else None,
        "pd_through_door_aipw": (
            art.outcome_aipw.pd_through_door if art.outcome_aipw else None),
        "pd_through_door_heckman": (
            art.outcome_heckman.pd_through_door if art.outcome_heckman else None),
        "errors": art.errors,
    }


@app.post("/retrain/unobservable")
def retrain_unobservable_endpoint(req: RetrainRequest) -> dict:
    if req.policy.propensity_mode != "unobservable":
        raise HTTPException(400, "propensity_mode must be 'unobservable'")
    joined = _build_join(req)
    policy = _build_policy(req.policy)
    cfg = RetrainConfig(
        snapshot_date=pd.Timestamp(req.snapshot_date),
        performance_window_months=req.performance_window_months,
        bootstrap_B=req.bootstrap_B,
        aipw_n_splits=req.aipw_n_splits,
        iv_p_threshold=req.iv_p_threshold,
        run_aipcw=req.run_aipcw,
        seed=req.seed,
    )
    art = retrain_unobservable(joined, cfg, policy)
    out = ARTIFACT_ROOT / f"{req.policy.policy_version_id}_{req.snapshot_date}_unobservable.json"
    write_artifact(art, out)
    return {
        "artifact": str(out),
        "iv_blocked": art.iv_diagnostic.iv_blocked if art.iv_diagnostic else None,
        "selection_auc": (
            art.propensity.selection_auc if art.propensity else None),
        "pd_through_door_aipw": (
            art.outcome_aipw.pd_through_door if art.outcome_aipw else None),
        "pd_through_door_heckman": (
            art.outcome_heckman.pd_through_door if art.outcome_heckman else None),
        "errors": art.errors,
    }


@app.post("/retrain/alt_data")
def retrain_alt_data_endpoint(req: AltDataRetrainRequest) -> dict:
    if req.policy.propensity_mode != "alt_data":
        raise HTTPException(400, "propensity_mode must be 'alt_data'")
    joined = _build_join(req)
    df = _read_cohort(req.cohort_name)
    if req.lender_id_col not in df.columns:
        raise HTTPException(400, f"lender_id_col missing: {req.lender_id_col}")
    lender_id = df[req.lender_id_col].astype(str).reset_index(drop=True)
    own_score = (df[req.own_score_logged_col].astype(float).to_numpy()
                 if req.own_score_logged_col else None)
    policy = _build_policy(req.policy)
    cfg = RetrainConfig(
        snapshot_date=pd.Timestamp(req.snapshot_date),
        performance_window_months=req.performance_window_months,
        bootstrap_B=req.bootstrap_B,
        aipw_n_splits=req.aipw_n_splits,
        iv_p_threshold=req.iv_p_threshold,
        seed=req.seed,
    )
    art = retrain_alt_data(joined, lender_id, cfg, policy, own_score,
                           req.shrinkage_lambda)
    out = ARTIFACT_ROOT / f"{req.policy.policy_version_id}_{req.snapshot_date}_altdata.json"
    write_artifact(art, out)
    return {
        "artifact": str(out),
        "lenders": list(art.propensity_per_lender.per_lender.keys())
                    if art.propensity_per_lender else [],
        "cold_start": list(art.propensity_per_lender.cold_start_lenders)
                       if art.propensity_per_lender else [],
        "pd_through_door_aipw": (
            art.outcome_aipw.pd_through_door if art.outcome_aipw else None),
        "errors": art.errors,
    }


@app.post("/cfrm")
def cfrm_endpoint(req: CFRMRequest) -> dict:
    df = _read_cohort(req.cohort_name)
    for col in (req.pi_logged_col, req.pi_new_col):
        if col not in df.columns:
            raise HTTPException(400, f"missing column: {col}")
    pi_log = np.clip(df[req.pi_logged_col].astype(float).to_numpy(), 1e-3, 1)
    pi_new = np.clip(df[req.pi_new_col].astype(float).to_numpy(), 1e-3, 1)
    s = df["s"].astype(int).to_numpy()
    df_y = _read_outcomes(req.cohort_name)
    funded_mask = s == 1
    if "y" not in df_y.columns or "applicant_id" not in df_y.columns:
        raise HTTPException(400, "outcomes table requires 'applicant_id' and 'y'")
    aid_to_y = pd.Series(df_y["y"].astype(int).to_numpy(),
                          index=df_y["applicant_id"].astype(str).to_numpy())
    funded_apps = df["applicant_id"].astype(str).to_numpy()[funded_mask]
    y_funded = aid_to_y.reindex(funded_apps).dropna().astype(int).to_numpy()
    funded_with_y = aid_to_y.reindex(funded_apps).notna().to_numpy()
    if funded_with_y.sum() == 0:
        raise HTTPException(400, "no matured funded outcomes available")

    cf = counterfactual_pd(
        pi_log[funded_mask][funded_with_y],
        pi_new[funded_mask][funded_with_y],
        s[funded_mask][funded_with_y],
        y_funded,
        np.ones(funded_with_y.sum(), dtype=bool),
        weight_cap=req.weight_cap,
    )
    rel = reliability_index(cf, raw_funded_n=int(funded_mask.sum()))
    return {
        "pd_under_new_policy": cf.pd_under_new_policy,
        "ess": cf.ess,
        "n_clipped": cf.n_clipped,
        "support_warning": cf.support_warning,
        "reliability": rel,
    }


@app.get("/retrain/{snapshot}")
def get_artifact(snapshot: str) -> dict:
    p = ARTIFACT_ROOT / f"{snapshot}.json"
    if not p.exists():
        raise HTTPException(404, f"artifact not found: {p}")
    return json.loads(p.read_text())


class RegisterRequest(BaseModel):
    snapshot_date: str
    policy_version_id: str
    artifact_kind: str = Field("observable",
                                description="observable|unobservable|alt_data")
    registered_name: str = "reject_inference_pd"
    tracking_uri: Optional[str] = None
    experiment_name: str = "reject_inference"


class PromoteRegistryRequest(BaseModel):
    registered_name: str = "reject_inference_pd"
    stage1_version: int
    stage2_version: int
    target_stage: str = Field("Production",
                               description="Staging|Production|Archived")
    archive_existing: bool = True
    run_id: str = ""


class StateRequest(BaseModel):
    macro_shock_freeze: Optional[bool] = None
    bureau_outage: Optional[bool] = None
    iv_kill: Optional[bool] = None
    reason: str = ""
    actor: str = "operator"


def _load_artifact_for_registry(req: RegisterRequest):
    """Stub: in a real deployment this loads from artifact storage."""
    suffix_map = {"observable": "observable", "unobservable": "unobservable",
                  "alt_data": "altdata"}
    if req.artifact_kind not in suffix_map:
        raise HTTPException(400, f"bad artifact_kind: {req.artifact_kind}")
    p = (ARTIFACT_ROOT
         / f"{req.policy_version_id}_{req.snapshot_date}_{suffix_map[req.artifact_kind]}.json")
    if not p.exists():
        raise HTTPException(404, f"artifact metadata not found: {p}")
    return json.loads(p.read_text())


@app.post("/registry/register")
def register_endpoint(req: RegisterRequest) -> dict:
    """Register a previously-retrained challenger pair in MLflow.

    The retrain endpoints persist a JSON metadata sidecar; in a real
    deployment they would also pickle the in-memory artifacts to a
    blob store. This endpoint is a thin shim: it reads the metadata
    and delegates to ``mlflow_registry.register_pair``. For now the
    in-memory artifact must be re-built from the cohort if the caller
    did not pre-pickle; adapt to your storage layer.
    """
    try:
        from reject_inference_pipeline.mlflow_registry import register_pair
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    meta = _load_artifact_for_registry(req)
    raise HTTPException(
        501,
        f"register endpoint requires per-deployment artifact storage "
        f"(found metadata: keys={list(meta)[:3]}); call "
        f"mlflow_registry.register_pair() with the in-memory "
        f"RetrainArtifact returned by retrain_{req.artifact_kind}()."
    )


@app.post("/registry/promote")
def promote_pair_endpoint(req: PromoteRegistryRequest) -> dict:
    """Atomically transition stage-1 + stage-2 to a target MLflow stage.

    Honours the operational freeze flag: if macro_shock_freeze or
    iv_kill is active the request is rejected with 423 Locked.
    """
    freeze = check_freeze()
    if freeze.blocked:
        raise HTTPException(423, f"promotion frozen: {'; '.join(freeze.reasons)}")
    try:
        from reject_inference_pipeline.mlflow_registry import (
            promote_pair, RegistryRecord)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    record = RegistryRecord(
        registered_name=req.registered_name,
        stage1_version=req.stage1_version,
        stage2_version=req.stage2_version,
        run_id=req.run_id,
    )
    try:
        promote_pair(record=record, target_stage=req.target_stage,
                      archive_existing=req.archive_existing)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return {
        "registered_name": req.registered_name,
        "stage1_version": req.stage1_version,
        "stage2_version": req.stage2_version,
        "target_stage": req.target_stage,
    }


@app.post("/registry/rollback")
def rollback_pair_endpoint(req: PromoteRegistryRequest) -> dict:
    try:
        from reject_inference_pipeline.mlflow_registry import (
            rollback_pair, RegistryRecord)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    record = RegistryRecord(
        registered_name=req.registered_name,
        stage1_version=req.stage1_version,
        stage2_version=req.stage2_version,
        run_id=req.run_id,
    )
    rollback_pair(record=record)
    return {"rolled_back": True,
            "stage1_version": req.stage1_version,
            "stage2_version": req.stage2_version}


@app.get("/state")
def get_state_endpoint() -> dict:
    s = load_state()
    return s.to_dict()


@app.post("/state")
def set_state_endpoint(req: StateRequest) -> dict:
    state = load_state()
    if req.macro_shock_freeze is not None:
        state = set_macro_shock(req.macro_shock_freeze, req.reason, req.actor)
    if req.bureau_outage is not None:
        state = set_bureau_outage(req.bureau_outage, req.actor)
    if req.iv_kill is not None:
        state = set_iv_kill(req.iv_kill, req.actor)
    return state.to_dict()
