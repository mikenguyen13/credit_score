"""MLflow registry integration with stage-1/stage-2 atomic promotion.

A reject-inference deployment is two coupled artifacts: the selection
propensity (stage 1) and the outcome model (stage 2 + IMR for the
Heckman path, or AIPW + cross-fit g_hat for the AIPW path). They must
move together: serving stage 2 against a stale stage 1 silently
miscalibrates the IMR. This module wraps MLflow so the two artifacts
are always logged inside the same parent run, registered as a coupled
pair, and transitioned in lockstep.

Three guarantees:

* register_pair                 logs both stages under a single
                                ``mlflow.start_run``; the registered-
                                model name is shared so a later
                                consumer reads the version pair as one
                                deployable unit.

* promote_pair                  transitions stage-1 and stage-2 to the
                                same MLflow stage (`Staging`,
                                `Production`, `Archived`) atomically;
                                rolls back stage-1 if stage-2
                                transition fails and vice versa.

* rollback                      moves the current Production pair back
                                to the previous version pair recorded
                                in the registry's tag history.

The integration is optional: if mlflow is not installed the module
imports lazily and surfaces a clear error. The pipeline does not
require MLflow to function; this is the bridge for production users.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .outcome import OutcomeArtifact
from .propensity import PropensityArtifact
from .pipeline import RetrainArtifact


_REGISTERED_NAME_DEFAULT = "reject_inference_pd"


@dataclass
class RegistryRecord:
    """Pointer to a registered stage-1/stage-2 pair."""

    registered_name: str
    stage1_version: int
    stage2_version: int
    run_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)


def _require_mlflow():
    try:
        import mlflow  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "mlflow not available; install with `pip install mlflow` "
            "or wire the pipeline to your registry directly"
        ) from exc
    import mlflow
    return mlflow


def _serialise_artifact(art: Any, path: Path) -> None:
    """Write a small JSON sidecar plus a joblib pickle of the object."""
    import joblib

    sidecar = {
        "type": type(art).__name__,
    }
    if isinstance(art, PropensityArtifact):
        sidecar.update({
            "mode": art.mode,
            "feature_names": list(art.feature_names),
            "selection_auc": art.selection_auc,
            "n_funded": art.n_funded,
            "n_total": art.n_total,
            "overlap_min": art.overlap_min,
            "overlap_max": art.overlap_max,
            "clip_share": art.clip_share,
        })
    elif isinstance(art, OutcomeArtifact):
        sidecar.update({
            "method": art.method,
            "feature_names": list(art.feature_names),
            "pd_through_door": art.pd_through_door,
            "pd_funded": art.pd_funded,
            "pd_declined": art.pd_declined,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, path / "artifact.joblib")
    (path / "card.json").write_text(json.dumps(sidecar, indent=2,
                                                default=str))


def register_pair(
    *,
    challenger: RetrainArtifact,
    registered_name: str = _REGISTERED_NAME_DEFAULT,
    metrics: Optional[dict[str, float]] = None,
    tags: Optional[dict[str, str]] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "reject_inference",
) -> RegistryRecord:
    """Log stage-1 + stage-2 inside one run and register both."""
    mlflow = _require_mlflow()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if challenger.propensity is None:
        raise ValueError("challenger has no stage-1 propensity artifact")
    if (challenger.outcome_aipw is None
            and challenger.outcome_heckman is None
            and challenger.outcome_aipcw is None):
        raise ValueError("challenger has no stage-2 outcome artifact")

    outcome = (challenger.outcome_aipcw
               or challenger.outcome_aipw
               or challenger.outcome_heckman)

    metrics = dict(metrics or {})
    metrics.setdefault("pd_through_door", float(outcome.pd_through_door))
    metrics.setdefault("pd_funded", float(outcome.pd_funded))
    metrics.setdefault("pd_declined", float(outcome.pd_declined))
    metrics.setdefault("n_train", float(challenger.n_train))
    metrics.setdefault("n_funded_train", float(challenger.n_funded_train))
    metrics.setdefault("n_matured", float(challenger.n_matured))

    tags = dict(tags or {})
    tags.setdefault("mode", challenger.mode)
    tags.setdefault("policy_version_id", challenger.policy_version_id)
    tags.setdefault("snapshot_date", str(challenger.snapshot_date.date()))
    tags.setdefault("outcome_method", outcome.method)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        stage1_dir = tmp_path / "stage1"
        stage2_dir = tmp_path / "stage2"
        _serialise_artifact(challenger.propensity, stage1_dir)
        _serialise_artifact(outcome, stage2_dir)

        with mlflow.start_run(tags=tags) as run:
            mlflow.log_metrics(metrics)
            mlflow.log_artifacts(str(stage1_dir), artifact_path="stage1")
            mlflow.log_artifacts(str(stage2_dir), artifact_path="stage2")

            stage1_uri = f"runs:/{run.info.run_id}/stage1"
            stage2_uri = f"runs:/{run.info.run_id}/stage2"
            stage1_name = f"{registered_name}__stage1"
            stage2_name = f"{registered_name}__stage2"
            mv1 = mlflow.register_model(stage1_uri, stage1_name)
            mv2 = mlflow.register_model(stage2_uri, stage2_name)

            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            client.set_model_version_tag(
                stage1_name, mv1.version, "paired_stage2_version", mv2.version)
            client.set_model_version_tag(
                stage2_name, mv2.version, "paired_stage1_version", mv1.version)
            for k, v in tags.items():
                client.set_model_version_tag(stage1_name, mv1.version, k, v)
                client.set_model_version_tag(stage2_name, mv2.version, k, v)

            return RegistryRecord(
                registered_name=registered_name,
                stage1_version=int(mv1.version),
                stage2_version=int(mv2.version),
                run_id=run.info.run_id,
                metrics=metrics,
                tags=tags,
            )


def _client():
    mlflow = _require_mlflow()
    from mlflow.tracking import MlflowClient
    return MlflowClient(), mlflow


def promote_pair(
    *,
    record: RegistryRecord,
    target_stage: str = "Production",
    archive_existing: bool = True,
) -> None:
    """Transition stage-1 and stage-2 atomically.

    On any failure the function rolls the partial transition back to
    the original stages so the registry is never left split between
    two model versions.
    """
    if target_stage not in {"Staging", "Production", "Archived"}:
        raise ValueError(f"bad target_stage: {target_stage}")
    client, _ = _client()

    s1_name = f"{record.registered_name}__stage1"
    s2_name = f"{record.registered_name}__stage2"

    s1_prev = client.get_model_version(s1_name, str(record.stage1_version))
    s2_prev = client.get_model_version(s2_name, str(record.stage2_version))
    s1_orig_stage = s1_prev.current_stage
    s2_orig_stage = s2_prev.current_stage

    try:
        client.transition_model_version_stage(
            s1_name, record.stage1_version, target_stage,
            archive_existing_versions=archive_existing,
        )
    except Exception as exc:
        raise RuntimeError(f"stage-1 transition failed: {exc}") from exc

    try:
        client.transition_model_version_stage(
            s2_name, record.stage2_version, target_stage,
            archive_existing_versions=archive_existing,
        )
    except Exception as exc:
        client.transition_model_version_stage(
            s1_name, record.stage1_version, s1_orig_stage,
            archive_existing_versions=False,
        )
        raise RuntimeError(
            f"stage-2 transition failed and stage-1 rolled back to "
            f"{s1_orig_stage}: {exc}"
        ) from exc

    client.set_model_version_tag(
        s1_name, record.stage1_version, "promoted_to", target_stage)
    client.set_model_version_tag(
        s2_name, record.stage2_version, "promoted_to", target_stage)
    client.set_model_version_tag(
        s1_name, record.stage1_version, "previous_stage", s1_orig_stage)
    client.set_model_version_tag(
        s2_name, record.stage2_version, "previous_stage", s2_orig_stage)


def rollback_pair(*, record: RegistryRecord) -> None:
    """Restore the pair to the stage recorded in `previous_stage` tag."""
    client, _ = _client()
    s1_name = f"{record.registered_name}__stage1"
    s2_name = f"{record.registered_name}__stage2"
    s1_meta = client.get_model_version(s1_name, str(record.stage1_version))
    s2_meta = client.get_model_version(s2_name, str(record.stage2_version))
    s1_back = s1_meta.tags.get("previous_stage", "Archived")
    s2_back = s2_meta.tags.get("previous_stage", "Archived")
    client.transition_model_version_stage(
        s1_name, record.stage1_version, s1_back,
        archive_existing_versions=False,
    )
    client.transition_model_version_stage(
        s2_name, record.stage2_version, s2_back,
        archive_existing_versions=False,
    )


def load_pair(
    registered_name: str = _REGISTERED_NAME_DEFAULT,
    stage: str = "Production",
) -> tuple[PropensityArtifact, OutcomeArtifact, RegistryRecord]:
    """Load the active stage-1/stage-2 pair from the registry.

    Verifies that the loaded versions are tagged as paired; raises if
    the registry is in a split state.
    """
    client, mlflow = _client()
    s1_name = f"{registered_name}__stage1"
    s2_name = f"{registered_name}__stage2"

    s1_versions = client.get_latest_versions(s1_name, stages=[stage])
    s2_versions = client.get_latest_versions(s2_name, stages=[stage])
    if not s1_versions or not s2_versions:
        raise RuntimeError(
            f"no models in stage {stage} for {registered_name}")
    s1 = s1_versions[0]
    s2 = s2_versions[0]
    if s1.tags.get("paired_stage2_version") != s2.version:
        raise RuntimeError(
            f"registry split: stage1 v{s1.version} pairs with "
            f"v{s1.tags.get('paired_stage2_version')} but stage2 v{s2.version} is live"
        )

    import joblib

    s1_local = client.download_artifacts(
        s1.run_id, "stage1/artifact.joblib")
    s2_local = client.download_artifacts(
        s2.run_id, "stage2/artifact.joblib")
    propensity = joblib.load(s1_local)
    outcome = joblib.load(s2_local)
    record = RegistryRecord(
        registered_name=registered_name,
        stage1_version=int(s1.version),
        stage2_version=int(s2.version),
        run_id=s1.run_id,
        tags=dict(s1.tags),
    )
    return propensity, outcome, record
