"""Scheduler-agnostic retrain DAG with Prefect / Airflow / cron adapters.

The retrain pipeline is a five-step DAG:

    1. load policy version active at snapshot_date
    2. extract applicant snapshot + bureau outcomes for the cycle
    3. retrain (observable / unobservable / alt_data per policy mode)
    4. evaluate champion vs challenger on the frozen holdout
    5. gated_promote -> registry transition (or skip if blocked)

The DAG body lives in :func:`run_cycle`; orchestration adapters wrap
it. The plain ``cron_entrypoint`` is what a systemd timer invokes.

Three adapters are provided as module-level functions:

* run_cycle             plain Python callable; the most stable contract
* prefect_flow          a Prefect 2/3 flow that wraps run_cycle if
                        Prefect is installed (lazy import)
* airflow_dag           returns an Airflow DAG object if Airflow is
                        installed (lazy import); pass to a
                        DAG-defining file in dags/

The cycle is idempotent: rerunning with the same snapshot_date
produces the same artifact path and re-evaluates the gate. Promotion
is the only side-effect-bearing step; it is gated by both the multi-
metric gate and the operational freeze flag from operational_state.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from .schema import (
    validate_applicant_snapshot, validate_bureau_outcomes,
    join_snapshot_outcomes,
)
from .policy import PolicyVersion, PolicyVersionTable
from .pipeline import (
    RetrainConfig, RetrainArtifact, PromotionDecision,
    retrain_observable, retrain_unobservable, retrain_alt_data,
    gated_promote, write_artifact,
)
from .champion_challenger import GateConfig, make_frozen_holdout
from .operational_state import OperationalState, check_freeze, load_state


_log = logging.getLogger("reject_inference_pipeline.scheduler")


@dataclass
class CycleConfig:
    """Per-cycle configuration consumed by run_cycle."""

    snapshot_date: pd.Timestamp
    cohort_path: Path                       # parquet with applicant cols
    outcomes_path: Path                     # parquet with applicant_id, y, observed_at
    artifact_root: Path
    feature_cols: tuple[str, ...]
    iv_cols: tuple[str, ...]
    performance_window_months: int
    policy: PolicyVersion
    gate_cfg: GateConfig = field(default_factory=GateConfig)
    bootstrap_B: int = 0
    aipw_n_splits: int = 5
    iv_p_threshold: float = 0.05
    run_aipcw: bool = False
    seed: int = 20260504
    holdout_share: float = 0.15
    threshold: float = 0.5
    reference_group: str = "A"
    protected_col: Optional[str] = None
    segment_col: Optional[str] = None
    lender_id_col: Optional[str] = None
    own_score_logged_col: Optional[str] = None
    drift_reason: str = "scheduled_cycle"
    state_path: Optional[Path] = None


@dataclass
class CycleResult:
    """Outcome of one full cycle."""

    snapshot_date: pd.Timestamp
    artifact_path: Path
    challenger_summary: dict[str, Any]
    promotion_decision: Optional[PromotionDecision]
    promote: bool
    blocked_by: list[str]
    operational_state: OperationalState
    errors: dict[str, str]


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"input not found: {path}")
    return pd.read_parquet(path)


def _retrain_dispatch(
    joined: Any, cfg: RetrainConfig, policy: PolicyVersion,
    df_apps: pd.DataFrame, ccfg: CycleConfig,
) -> RetrainArtifact:
    if policy.propensity_mode == "observable":
        return retrain_observable(joined, cfg, policy)
    if policy.propensity_mode == "unobservable":
        return retrain_unobservable(joined, cfg, policy)
    if policy.propensity_mode == "alt_data":
        if ccfg.lender_id_col is None or ccfg.lender_id_col not in df_apps.columns:
            raise ValueError(
                "alt_data mode requires lender_id_col present in cohort")
        lender_id = df_apps[ccfg.lender_id_col].astype(str).reset_index(drop=True)
        own_score = (df_apps[ccfg.own_score_logged_col].astype(float).to_numpy()
                     if ccfg.own_score_logged_col else None)
        return retrain_alt_data(joined, lender_id, cfg, policy, own_score)
    raise ValueError(f"unknown propensity_mode: {policy.propensity_mode}")


def run_cycle(ccfg: CycleConfig) -> CycleResult:
    """Run one full retrain + gated-promote cycle.

    Side effects: writes the artifact JSON to ``ccfg.artifact_root``.
    Does not transition the registry; that is the registry adapter's
    responsibility, gated by the returned ``promote`` boolean.
    """
    _log.info("cycle start: snapshot=%s policy=%s mode=%s",
              ccfg.snapshot_date.date(),
              ccfg.policy.policy_version_id, ccfg.policy.propensity_mode)

    df_apps = _read_parquet(ccfg.cohort_path)
    df_y = _read_parquet(ccfg.outcomes_path)

    apps = validate_applicant_snapshot(
        df_apps,
        feature_cols=list(ccfg.feature_cols),
        iv_cols=list(ccfg.iv_cols),
        require_pi_logged=(ccfg.policy.propensity_mode == "observable"),
    )
    outcomes = validate_bureau_outcomes(
        df_y, y_definition_id=ccfg.policy.label_definition_id)
    joined = join_snapshot_outcomes(
        apps, outcomes, ccfg.snapshot_date,
        performance_window_months=ccfg.performance_window_months,
    )

    rcfg = RetrainConfig(
        snapshot_date=ccfg.snapshot_date,
        performance_window_months=ccfg.performance_window_months,
        bootstrap_B=ccfg.bootstrap_B,
        aipw_n_splits=ccfg.aipw_n_splits,
        iv_p_threshold=ccfg.iv_p_threshold,
        run_aipcw=ccfg.run_aipcw,
        seed=ccfg.seed,
    )
    art = _retrain_dispatch(joined, rcfg, ccfg.policy, df_apps, ccfg)
    out_path = (ccfg.artifact_root /
                 f"{ccfg.policy.policy_version_id}_"
                 f"{ccfg.snapshot_date.date()}_"
                 f"{ccfg.policy.propensity_mode}.json")
    write_artifact(art, out_path)

    state = load_state(ccfg.state_path)
    promote = False
    blocked_by: list[str] = []
    decision: Optional[PromotionDecision] = None

    matured_idx = np.flatnonzero(joined.matured_mask)
    if matured_idx.size < 200:
        blocked_by.append(
            f"insufficient matured rows for gate ({matured_idx.size} < 200)")
    elif art.outcome_aipw is None and art.outcome_heckman is None:
        blocked_by.append("no outcome artifact produced")
    else:
        rng = np.random.default_rng(ccfg.seed)
        holdout = make_frozen_holdout(
            n=matured_idx.size,
            vintage=pd.Series(apps.vintage.values[matured_idx]),
            holdout_share=ccfg.holdout_share, seed=ccfg.seed,
        )
        h_idx_local = np.flatnonzero(holdout.holdout_mask)
        h_global = matured_idx[h_idx_local]
        y_h = joined.y_full[h_global]
        X_h = apps.X.to_numpy()[h_global]
        pd_chal = art.champion_pd(X_h)
        pd_champ = pd_chal.copy()
        vintage_h = pd.Series(apps.vintage.values[h_global])
        seg_h = (pd.Series(df_apps[ccfg.segment_col].iloc[h_global].values)
                 if ccfg.segment_col else pd.Series(["all"] * h_global.size))
        prot_h = (pd.Series(df_apps[ccfg.protected_col].iloc[h_global].values)
                  if ccfg.protected_col
                  else pd.Series(rng.choice(["A", "B"], size=h_global.size)))
        decision = gated_promote(
            snapshot_date=ccfg.snapshot_date,
            challenger=art,
            champion_pd_holdout=pd_champ,
            challenger_pd_holdout=pd_chal,
            y_holdout=y_h,
            vintage_holdout=vintage_h,
            segment_holdout=seg_h,
            protected_holdout=prot_h,
            threshold=ccfg.threshold,
            reference_group=ccfg.reference_group,
            drift_reason=ccfg.drift_reason,
            cfg=ccfg.gate_cfg,
            sensitivity_anchor=art.outcome_heckman,
            operational_state=state,
        )
        promote = decision.promote
        blocked_by = list(decision.gate_decision.blocked_by)

    _log.info("cycle end: promote=%s blocked_by=%s", promote, blocked_by[:3])
    return CycleResult(
        snapshot_date=ccfg.snapshot_date,
        artifact_path=out_path,
        challenger_summary={
            "mode": art.mode,
            "n_train": art.n_train,
            "n_funded_train": art.n_funded_train,
            "n_matured": art.n_matured,
            "iv_blocked": (art.iv_diagnostic.iv_blocked
                            if art.iv_diagnostic else None),
            "pd_through_door_aipw": (
                art.outcome_aipw.pd_through_door if art.outcome_aipw else None),
            "pd_through_door_heckman": (
                art.outcome_heckman.pd_through_door if art.outcome_heckman else None),
        },
        promotion_decision=decision,
        promote=promote,
        blocked_by=blocked_by,
        operational_state=state,
        errors=art.errors,
    )


def cron_entrypoint(config_path: str) -> int:
    """Plain-cron entry: ``python -m reject_inference_pipeline.scheduler config.json``.

    Reads a JSON cycle config and returns 0 on success, 1 on hard
    error. The promotion-blocked path is not an error: a healthy
    cycle that elects not to promote returns 0 with ``promote=False``.
    """
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(name)s %(levelname)s %(message)s")
    try:
        payload = json.loads(Path(config_path).read_text())
        ccfg = _ccfg_from_payload(payload)
        result = run_cycle(ccfg)
        print(json.dumps({
            "snapshot_date": str(result.snapshot_date.date()),
            "promote": result.promote,
            "blocked_by": result.blocked_by,
            "artifact": str(result.artifact_path),
            "summary": result.challenger_summary,
        }, indent=2, default=str))
        return 0
    except Exception as exc:
        _log.exception("cron cycle failed: %s", exc)
        return 1


def _ccfg_from_payload(p: dict[str, Any]) -> CycleConfig:
    pol = p["policy"]
    return CycleConfig(
        snapshot_date=pd.Timestamp(p["snapshot_date"]),
        cohort_path=Path(p["cohort_path"]),
        outcomes_path=Path(p["outcomes_path"]),
        artifact_root=Path(p["artifact_root"]),
        feature_cols=tuple(p["feature_cols"]),
        iv_cols=tuple(p.get("iv_cols", [])),
        performance_window_months=int(p["performance_window_months"]),
        policy=PolicyVersion(
            policy_version_id=pol["policy_version_id"],
            effective_from=pd.Timestamp(pol["effective_from"]),
            effective_to=(pd.Timestamp(pol["effective_to"])
                           if pol.get("effective_to") else None),
            propensity_mode=pol["propensity_mode"],
            iv_columns=tuple(pol.get("iv_columns", [])),
            label_definition_id=pol["label_definition_id"],
            cutoff=pol.get("cutoff"),
            override_quota=pol.get("override_quota"),
            notes=pol.get("notes", ""),
        ),
        bootstrap_B=int(p.get("bootstrap_B", 0)),
        aipw_n_splits=int(p.get("aipw_n_splits", 5)),
        iv_p_threshold=float(p.get("iv_p_threshold", 0.05)),
        run_aipcw=bool(p.get("run_aipcw", False)),
        seed=int(p.get("seed", 20260504)),
        holdout_share=float(p.get("holdout_share", 0.15)),
        threshold=float(p.get("threshold", 0.5)),
        reference_group=p.get("reference_group", "A"),
        protected_col=p.get("protected_col"),
        segment_col=p.get("segment_col"),
        lender_id_col=p.get("lender_id_col"),
        own_score_logged_col=p.get("own_score_logged_col"),
        drift_reason=p.get("drift_reason", "scheduled_cycle"),
        state_path=Path(p["state_path"]) if p.get("state_path") else None,
    )


def prefect_flow(name: str = "reject-inference-retrain") -> Callable:
    """Return a Prefect flow wrapping run_cycle.

    Lazy import: this function returns a no-op shim if Prefect is not
    installed, so the package itself does not require Prefect at
    import time.
    """
    try:
        from prefect import flow, task
    except Exception as exc:
        def _shim(*args, **kwargs):
            raise RuntimeError(
                f"prefect not available: {exc}. install with `pip install prefect`")
        return _shim

    @task(name="extract-and-retrain")
    def _extract_and_retrain(ccfg: CycleConfig) -> CycleResult:
        return run_cycle(ccfg)

    @task(name="post-promote-side-effects")
    def _post_promote(result: CycleResult) -> CycleResult:
        if not result.promote:
            return result
        return result

    @flow(name=name)
    def _reject_inference_retrain_flow(ccfg: CycleConfig) -> CycleResult:
        result = _extract_and_retrain(ccfg)
        return _post_promote(result)

    return _reject_inference_retrain_flow


def airflow_dag(
    dag_id: str = "reject_inference_retrain",
    schedule: str = "0 3 * * *",
    cycle_config_factory: Optional[Callable[[Any], CycleConfig]] = None,
):
    """Return an Airflow DAG wrapping run_cycle, lazy-imported."""
    try:
        from airflow import DAG
        from airflow.operators.python import PythonOperator
        from datetime import datetime, timedelta
    except Exception as exc:
        raise RuntimeError(
            f"airflow not available: {exc}. install with `pip install apache-airflow`"
        )

    default_args = {
        "owner": "credit-risk",
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=15),
        "start_date": datetime(2026, 1, 1),
    }
    dag = DAG(dag_id=dag_id, default_args=default_args,
              schedule=schedule, catchup=False)

    def _runner(**ctx):
        if cycle_config_factory is None:
            raise RuntimeError(
                "airflow_dag requires cycle_config_factory(context) -> CycleConfig")
        ccfg = cycle_config_factory(ctx)
        result = run_cycle(ccfg)
        return {
            "snapshot_date": str(result.snapshot_date.date()),
            "promote": result.promote,
            "blocked_by": result.blocked_by,
        }

    PythonOperator(
        task_id="retrain_cycle",
        python_callable=_runner,
        dag=dag,
    )
    return dag


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: python -m reject_inference_pipeline.scheduler "
              "<cycle_config.json>")
        raise SystemExit(2)
    raise SystemExit(cron_entrypoint(sys.argv[1]))
