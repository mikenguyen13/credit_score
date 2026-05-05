"""Production retrain pipeline for reject-inference PD models.

Modules
-------
schema                ApplicantSnapshot, BureauOutcomeBatch, JoinedSnapshot
                      with strict validation and point-in-time joins.
policy                PolicyVersionTable, PolicyVersion, propensity-mode
                      classifier (observable / unobservable / alt_data).
propensity            Heckman stage-1 probit, observable-pi reader,
                      exclusion-restriction recheck and overlap.
outcome               Heckman stage-2, AIPW with cross-fitting, AIPCW
                      for the censored tail.
drift                 PSI, KL, three-kind drift classifier (covariate
                      vs concept vs selection), DriftTrigger with
                      hysteresis.
champion_challenger   Frozen holdout, multi-metric gate (DeLong AUC,
                      Brier, calibration slope, ECE, segment AUC,
                      ECOA disparate impact), shadow-mode PSI,
                      auto-rollback.
governance            SR 11-7 memo, ECOA disparate-impact diff, Basel
                      TTC multi-vintage gate.
alt_data              Per-lender hierarchical propensity with
                      shrinkage, cold-start pseudo-prior, feedback-
                      loop guard.
cfrm                  Counterfactual risk minimisation for off-policy
                      PD under a pre-announced policy change.
pipeline              Orchestrator: retrain_observable, retrain_
                      unobservable, retrain_alt_data, gated_promote,
                      write_artifact.
model_card            Auto-generated model card for SR 11-7.

Entry point used by ``deployment/reject_inference_app.py``:
:func:`pipeline.retrain_observable`, :func:`pipeline.retrain_
unobservable`, :func:`pipeline.retrain_alt_data` and
:func:`pipeline.gated_promote`.
"""

from .schema import (
    ApplicantSnapshot, BureauOutcomeBatch, JoinedSnapshot,
    validate_applicant_snapshot, validate_bureau_outcomes,
    join_snapshot_outcomes,
)
from .policy import (
    PolicyVersion, PolicyVersionTable, PropensityMode,
    policy_change_required_actions,
)
from .propensity import (
    PropensityArtifact, IVDiagnostic,
    fit_observable_propensity, fit_selection_probit,
    run_iv_diagnostics, overlap_summary,
)
from .outcome import (
    OutcomeArtifact,
    fit_heckman_outcome, fit_aipw_outcome, fit_aipcw_outcome,
    predict_pd,
)
from .drift import (
    DriftReport, DriftTrigger, DriftThresholds,
    psi, kl_divergence, classify_drift, compute_drift,
)
from .champion_challenger import (
    HoldoutSplit, ChallengerEvaluation, GateConfig, GateDecision,
    make_frozen_holdout, evaluate_challenger, gate,
    delong_auc_test, brier, calibration_slope,
    expected_calibration_error, per_segment_auc,
    disparate_impact_ratio, shadow_psi_score, rollback_check,
)
from .governance import (
    TTCResult, basel_ttc_multi_vintage_gate,
    ecoa_disparate_impact_diff, sr_117_memo,
)
from .alt_data import (
    HierarchicalPropensityArtifact, fit_hierarchical_propensity,
    cold_start_pseudoprior, feedback_loop_guard,
)
from .cfrm import CFRMResult, counterfactual_pd, reliability_index
from .pipeline import (
    RetrainConfig, RetrainArtifact, PromotionDecision,
    retrain_observable, retrain_unobservable, retrain_alt_data,
    gated_promote, write_artifact,
)
from .model_card import RejectInferenceCard, render_card
from .operational_state import (
    OperationalState, FreezeBlock,
    load_state, save_state, check_freeze,
    set_macro_shock, set_bureau_outage, set_iv_kill,
)
from .scheduler import (
    CycleConfig, CycleResult, run_cycle, cron_entrypoint,
    prefect_flow, airflow_dag,
)

__all__ = [
    "ApplicantSnapshot", "BureauOutcomeBatch", "JoinedSnapshot",
    "validate_applicant_snapshot", "validate_bureau_outcomes",
    "join_snapshot_outcomes",
    "PolicyVersion", "PolicyVersionTable", "PropensityMode",
    "policy_change_required_actions",
    "PropensityArtifact", "IVDiagnostic",
    "fit_observable_propensity", "fit_selection_probit",
    "run_iv_diagnostics", "overlap_summary",
    "OutcomeArtifact", "fit_heckman_outcome", "fit_aipw_outcome",
    "fit_aipcw_outcome", "predict_pd",
    "DriftReport", "DriftTrigger", "DriftThresholds",
    "psi", "kl_divergence", "classify_drift", "compute_drift",
    "HoldoutSplit", "ChallengerEvaluation", "GateConfig", "GateDecision",
    "make_frozen_holdout", "evaluate_challenger", "gate",
    "delong_auc_test", "brier", "calibration_slope",
    "expected_calibration_error", "per_segment_auc",
    "disparate_impact_ratio", "shadow_psi_score", "rollback_check",
    "TTCResult", "basel_ttc_multi_vintage_gate",
    "ecoa_disparate_impact_diff", "sr_117_memo",
    "HierarchicalPropensityArtifact", "fit_hierarchical_propensity",
    "cold_start_pseudoprior", "feedback_loop_guard",
    "CFRMResult", "counterfactual_pd", "reliability_index",
    "RetrainConfig", "RetrainArtifact", "PromotionDecision",
    "retrain_observable", "retrain_unobservable", "retrain_alt_data",
    "gated_promote", "write_artifact",
    "RejectInferenceCard", "render_card",
    "OperationalState", "FreezeBlock",
    "load_state", "save_state", "check_freeze",
    "set_macro_shock", "set_bureau_outage", "set_iv_kill",
    "CycleConfig", "CycleResult", "run_cycle", "cron_entrypoint",
    "prefect_flow", "airflow_dag",
]
