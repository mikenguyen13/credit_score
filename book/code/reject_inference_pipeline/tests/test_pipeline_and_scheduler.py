"""End-to-end pipeline + scheduler tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reject_inference_pipeline import (
    CycleConfig, GateConfig, PolicyVersion, RetrainConfig,
    gated_promote, retrain_observable, retrain_unobservable,
    retrain_alt_data, run_cycle, set_macro_shock, write_artifact,
)


def test_observable_retrain_returns_artifact(joined_snapshot, retrain_config,
                                                policy_observable):
    art = retrain_observable(joined_snapshot, retrain_config, policy_observable)
    assert art.mode == "observable"
    assert art.outcome_aipw is not None
    assert art.outcome_heckman is not None
    assert art.propensity is not None
    assert art.iv_diagnostic is not None


def test_unobservable_retrain_returns_artifact(joined_snapshot, retrain_config,
                                                  policy_unobservable):
    art = retrain_unobservable(joined_snapshot, retrain_config,
                                 policy_unobservable)
    assert art.mode == "unobservable"
    assert art.propensity.gamma is not None  # estimated stage 1


def test_alt_data_retrain_returns_per_lender(joined_snapshot, retrain_config,
                                              policy_alt_data,
                                              applicant_snapshot):
    rng = np.random.default_rng(0)
    lender = pd.Series(rng.choice(["bankA", "bankB"],
                                    size=applicant_snapshot.n))
    art = retrain_alt_data(joined_snapshot, lender, retrain_config,
                            policy_alt_data)
    assert art.mode == "alt_data"
    assert art.propensity_per_lender is not None
    assert set(art.propensity_per_lender.per_lender) == {"bankA", "bankB"}


def test_observable_rejects_wrong_policy_mode(joined_snapshot, retrain_config,
                                                policy_unobservable):
    with pytest.raises(ValueError, match="not observable"):
        retrain_observable(joined_snapshot, retrain_config,
                            policy_unobservable)


def test_write_artifact_serialises_to_json(observable_artifact, tmp_path: Path):
    p = write_artifact(observable_artifact, tmp_path / "art.json")
    payload = json.loads(p.read_text())
    assert payload["mode"] == "observable"
    assert "n_train" in payload


def test_gated_promote_blocks_under_freeze(observable_artifact, applicant_snapshot,
                                             joined_snapshot, tmp_path: Path):
    state_path = tmp_path / "state.json"
    set_macro_shock(True, "covid", path=state_path)
    matured_idx = np.flatnonzero(joined_snapshot.matured_mask)[:200]
    y_h = joined_snapshot.y_full[matured_idx]
    X_h = applicant_snapshot.X.to_numpy()[matured_idx]
    pd_chal = observable_artifact.champion_pd(X_h)
    pd_champ = pd_chal.copy()

    from reject_inference_pipeline import load_state
    state = load_state(state_path)
    decision = gated_promote(
        snapshot_date=pd.Timestamp("2026-05-01"),
        challenger=observable_artifact,
        champion_pd_holdout=pd_champ,
        challenger_pd_holdout=pd_chal,
        y_holdout=y_h,
        vintage_holdout=pd.Series(applicant_snapshot.vintage.values[matured_idx]),
        segment_holdout=pd.Series(applicant_snapshot.segment.values[matured_idx]),
        protected_holdout=pd.Series(["A"] * matured_idx.size),
        threshold=0.5, reference_group="A",
        drift_reason="test_freeze",
        operational_state=state,
    )
    assert decision.promote is False
    assert any("freeze" in b for b in decision.gate_decision.blocked_by)


def test_run_cycle_writes_artifact(cohort_df, tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    outcomes_path = tmp_path / "outcomes.parquet"
    artifact_root = tmp_path / "artifacts"
    state_path = tmp_path / "state.json"
    cohort_df.drop(columns=["_y_truth"]).to_parquet(cohort_path)

    funded = np.flatnonzero(cohort_df["s"].to_numpy() == 1)
    pd.DataFrame({
        "applicant_id": cohort_df["applicant_id"].iloc[funded].values,
        "observed_at": (cohort_df["as_of"].iloc[funded]
                         + pd.DateOffset(months=18)).values,
        "y": cohort_df["_y_truth"].iloc[funded].values,
    }).to_parquet(outcomes_path)

    ccfg = CycleConfig(
        snapshot_date=pd.Timestamp("2026-05-01"),
        cohort_path=cohort_path,
        outcomes_path=outcomes_path,
        artifact_root=artifact_root,
        feature_cols=("x1", "x2"),
        iv_cols=("z",),
        performance_window_months=18,
        policy=PolicyVersion(
            policy_version_id="P_2026_v1",
            effective_from=pd.Timestamp("2024-01-01"),
            effective_to=None,
            propensity_mode="observable",
            iv_columns=("z",),
            label_definition_id="dpd90_18m",
        ),
        gate_cfg=GateConfig(),
        aipw_n_splits=3,
        seed=42,
        segment_col="segment",
        state_path=state_path,
    )
    res = run_cycle(ccfg)
    assert res.artifact_path.exists()
    assert isinstance(res.promote, bool)
    assert isinstance(res.blocked_by, list)
    assert res.challenger_summary["mode"] == "observable"
