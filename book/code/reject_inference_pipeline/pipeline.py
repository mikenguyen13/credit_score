"""Orchestrator: snapshot -> retrain -> gated promotion.

Three entry points stitch the modules together:

retrain_observable        the inside-bank flow. The lender logs pi at
                          decision time. The retrain reads pi from the
                          snapshot, refits the outcome stage with
                          AIPW (production champion) and Heckman (SR
                          11-7 sensitivity anchor), and returns a
                          RetrainArtifact ready for the gate.

retrain_unobservable      the lender does not log pi. The retrain
                          fits Heckman stage-1 + stage-2 and re-runs
                          the IV diagnostic. AIPW also runs on the
                          estimated propensity for the doubly-robust
                          score.

retrain_alt_data          per-lender hierarchical stage-1 with
                          cold-start handling and the feedback-loop
                          guard. Each lender produces its own outcome
                          fit; the package returns the stack.

gated_promote             apply drift trigger -> challenger eval ->
                          ECOA diff -> TTC gate -> SR 11-7 memo.
                          Returns a PromotionDecision with a clear
                          ``promote`` boolean and the full reasoning.

The orchestrator is deliberately thin: every step delegates to a
module function, and every step is wrapped with ``_safe`` so a single
failure (e.g., one IV diagnostic blowing up) does not lose the entire
artifact. SR 11-7 reviewers get a partial result with a non-empty
``errors`` block instead of an opaque traceback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import JoinedSnapshot
from .policy import PolicyVersion
from .propensity import (
    PropensityArtifact, IVDiagnostic,
    fit_observable_propensity, fit_selection_probit,
    run_iv_diagnostics, overlap_summary,
)
from .outcome import (
    OutcomeArtifact,
    fit_heckman_outcome, fit_aipw_outcome, fit_aipcw_outcome,
)
from .drift import DriftThresholds
from .champion_challenger import (
    ChallengerEvaluation, GateConfig, GateDecision,
    evaluate_challenger, gate,
)
from .governance import (
    sr_117_memo, ecoa_disparate_impact_diff,
    basel_ttc_multi_vintage_gate,
)
from .alt_data import (
    HierarchicalPropensityArtifact, fit_hierarchical_propensity,
    feedback_loop_guard,
)
from .operational_state import OperationalState, check_freeze, load_state


@dataclass
class RetrainConfig:
    snapshot_date: pd.Timestamp
    performance_window_months: int
    bootstrap_B: int = 0
    cluster_key_col: Optional[str] = None
    aipw_n_splits: int = 5
    iv_p_threshold: float = 0.05
    drift_thresholds: DriftThresholds = field(default_factory=DriftThresholds)
    gate_config: GateConfig = field(default_factory=GateConfig)
    run_aipcw: bool = False
    seed: int = 20260504


@dataclass
class RetrainArtifact:
    """Output of any retrain entry point."""

    mode: str                                # 'observable'|'unobservable'|'alt_data'
    snapshot_date: pd.Timestamp
    policy_version_id: str
    propensity: Optional[PropensityArtifact]
    propensity_per_lender: Optional[HierarchicalPropensityArtifact] = None
    iv_diagnostic: Optional[IVDiagnostic] = None
    outcome_aipw: Optional[OutcomeArtifact] = None
    outcome_heckman: Optional[OutcomeArtifact] = None
    outcome_aipcw: Optional[OutcomeArtifact] = None
    overlap: dict[str, Any] = field(default_factory=dict)
    n_train: int = 0
    n_funded_train: int = 0
    n_matured: int = 0
    errors: dict[str, str] = field(default_factory=dict)
    seed: int = 0

    def champion_pd(self, X: np.ndarray) -> np.ndarray:
        from .outcome import predict_pd
        if self.outcome_aipcw is not None:
            return predict_pd(self.outcome_aipcw, X)
        if self.outcome_aipw is not None:
            return predict_pd(self.outcome_aipw, X)
        if self.outcome_heckman is not None:
            return predict_pd(self.outcome_heckman, X, self.propensity)
        raise RuntimeError("no outcome artifact available")


@dataclass
class PromotionDecision:
    promote: bool
    drift_reason: str
    gate_decision: GateDecision
    ttc_blocked: bool
    ecoa_blocked: bool
    memo_markdown: str
    eval_: ChallengerEvaluation


def _safe(step: str, errors: dict[str, str], fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        errors[step] = f"{type(exc).__name__}: {exc}"
        return None


def retrain_observable(
    joined: JoinedSnapshot,
    config: RetrainConfig,
    policy: PolicyVersion,
) -> RetrainArtifact:
    if policy.propensity_mode != "observable":
        raise ValueError(
            f"policy {policy.policy_version_id} is "
            f"{policy.propensity_mode}, not observable"
        )
    apps = joined.applicants
    errors: dict[str, str] = {}
    rng = np.random.default_rng(config.seed)

    prop = _safe("observable_propensity", errors,
                 fit_observable_propensity, apps)
    funded_matured = (apps.s == 1) & joined.matured_mask
    y_funded = (joined.y_full[funded_matured]
                if prop is not None else np.array([]))

    iv = _safe("iv_diagnostic", errors,
               run_iv_diagnostics, apps, y_funded, funded_matured,
               config.iv_p_threshold, 10.0,
               prop.imr if prop is not None else None)

    cluster_key = (apps.vintage.values if apps.vintage is not None else None)
    if config.cluster_key_col and apps.vintage is None:
        raise ValueError("cluster_key requested but no vintage on snapshot")

    heckman = _safe("heckman", errors,
                    fit_heckman_outcome,
                    apps, y_funded, funded_matured, prop,
                    config.bootstrap_B, cluster_key, rng)

    aipw = _safe("aipw", errors,
                 fit_aipw_outcome,
                 apps, y_funded, funded_matured, prop,
                 None, config.aipw_n_splits, True, rng)

    aipcw = None
    if config.run_aipcw:
        aipcw = _safe("aipcw", errors,
                      fit_aipcw_outcome, joined, prop, None, rng)

    return RetrainArtifact(
        mode="observable",
        snapshot_date=joined.snapshot_date,
        policy_version_id=policy.policy_version_id,
        propensity=prop,
        iv_diagnostic=iv,
        outcome_aipw=aipw,
        outcome_heckman=heckman,
        outcome_aipcw=aipcw,
        overlap=overlap_summary(prop) if prop else {},
        n_train=int(apps.n),
        n_funded_train=int(apps.n_funded),
        n_matured=int(joined.matured_mask.sum()),
        errors=errors,
        seed=config.seed,
    )


def retrain_unobservable(
    joined: JoinedSnapshot,
    config: RetrainConfig,
    policy: PolicyVersion,
) -> RetrainArtifact:
    if policy.propensity_mode != "unobservable":
        raise ValueError(
            f"policy {policy.policy_version_id} is "
            f"{policy.propensity_mode}, not unobservable"
        )
    apps = joined.applicants
    errors: dict[str, str] = {}
    rng = np.random.default_rng(config.seed)

    prop = _safe("selection_probit", errors,
                 fit_selection_probit, apps)
    funded_matured = (apps.s == 1) & joined.matured_mask
    y_funded = joined.y_full[funded_matured]

    iv = _safe("iv_diagnostic", errors,
               run_iv_diagnostics, apps, y_funded, funded_matured,
               config.iv_p_threshold, 10.0,
               prop.imr if prop is not None else None)

    cluster_key = apps.vintage.values if apps.vintage is not None else None

    heckman = _safe("heckman", errors,
                    fit_heckman_outcome,
                    apps, y_funded, funded_matured, prop,
                    config.bootstrap_B, cluster_key, rng)

    aipw = _safe("aipw", errors,
                 fit_aipw_outcome,
                 apps, y_funded, funded_matured, prop,
                 None, config.aipw_n_splits, True, rng)

    aipcw = None
    if config.run_aipcw:
        aipcw = _safe("aipcw", errors,
                      fit_aipcw_outcome, joined, prop, None, rng)

    return RetrainArtifact(
        mode="unobservable",
        snapshot_date=joined.snapshot_date,
        policy_version_id=policy.policy_version_id,
        propensity=prop,
        iv_diagnostic=iv,
        outcome_aipw=aipw,
        outcome_heckman=heckman,
        outcome_aipcw=aipcw,
        overlap=overlap_summary(prop) if prop else {},
        n_train=int(apps.n),
        n_funded_train=int(apps.n_funded),
        n_matured=int(joined.matured_mask.sum()),
        errors=errors,
        seed=config.seed,
    )


def retrain_alt_data(
    joined: JoinedSnapshot,
    lender_id: pd.Series,
    config: RetrainConfig,
    policy: PolicyVersion,
    own_score_logged: Optional[np.ndarray] = None,
    shrinkage_lambda: float = 0.5,
) -> RetrainArtifact:
    if policy.propensity_mode != "alt_data":
        raise ValueError(
            f"policy {policy.policy_version_id} is "
            f"{policy.propensity_mode}, not alt_data"
        )
    apps = joined.applicants
    errors: dict[str, str] = {}
    rng = np.random.default_rng(config.seed)

    if own_score_logged is not None:
        guard = _safe("feedback_loop_guard", errors,
                      feedback_loop_guard, apps, own_score_logged)
    else:
        guard = None

    hier = _safe("hierarchical_propensity", errors,
                 fit_hierarchical_propensity,
                 apps, lender_id, shrinkage_lambda)

    pooled_pi = np.zeros(apps.n)
    pooled_imr = np.zeros(apps.n)
    if hier is not None:
        for lid, art in hier.per_lender.items():
            idx = np.flatnonzero(lender_id.values == lid)
            pooled_pi[idx] = art.pi
            pooled_imr[idx] = art.imr
    pooled = PropensityArtifact(
        mode="estimated", pi=pooled_pi, imr=pooled_imr, gamma=None,
        feature_names=hier.feature_names if hier else (),
        overlap_min=float(pooled_pi.min()) if hier else 0.0,
        overlap_max=float(pooled_pi.max()) if hier else 1.0,
        clip_share=0.0,
        n_funded=int(apps.n_funded), n_total=int(apps.n),
    ) if hier is not None else None

    funded_matured = (apps.s == 1) & joined.matured_mask
    y_funded = joined.y_full[funded_matured]

    iv = _safe("iv_diagnostic", errors,
               run_iv_diagnostics, apps, y_funded, funded_matured,
               config.iv_p_threshold, 10.0,
               pooled.imr if pooled is not None else None)

    aipw = _safe("aipw", errors,
                 fit_aipw_outcome,
                 apps, y_funded, funded_matured, pooled,
                 None, config.aipw_n_splits, True, rng)

    art = RetrainArtifact(
        mode="alt_data",
        snapshot_date=joined.snapshot_date,
        policy_version_id=policy.policy_version_id,
        propensity=pooled,
        propensity_per_lender=hier,
        iv_diagnostic=iv,
        outcome_aipw=aipw,
        outcome_heckman=None,
        overlap=(overlap_summary(pooled) if pooled else {}),
        n_train=int(apps.n),
        n_funded_train=int(apps.n_funded),
        n_matured=int(joined.matured_mask.sum()),
        errors=errors,
        seed=config.seed,
    )
    if guard is not None:
        art.errors["feedback_loop_guard"] = json.dumps(guard, default=str)
    return art


def gated_promote(
    *,
    snapshot_date: pd.Timestamp,
    challenger: RetrainArtifact,
    champion_pd_holdout: np.ndarray,
    challenger_pd_holdout: np.ndarray,
    y_holdout: np.ndarray,
    vintage_holdout: pd.Series,
    segment_holdout: pd.Series,
    protected_holdout: pd.Series,
    threshold: float,
    reference_group: str,
    drift_reason: str,
    cfg: GateConfig = GateConfig(),
    shadow_psi: Optional[float] = None,
    sensitivity_anchor: Optional[OutcomeArtifact] = None,
    operational_state: Optional[OperationalState] = None,
) -> PromotionDecision:
    freeze = check_freeze(operational_state)
    eval_ = evaluate_challenger(
        y_holdout=y_holdout,
        p_champion=champion_pd_holdout,
        p_challenger=challenger_pd_holdout,
        segment=segment_holdout,
        protected=protected_holdout,
        threshold=threshold,
        reference_group=reference_group,
    )
    decision = gate(eval_, cfg, shadow_psi)
    ecoa = ecoa_disparate_impact_diff(
        eval_, floor=cfg.di_floor, max_drop=cfg.di_max_drop)
    ttc = basel_ttc_multi_vintage_gate(
        y_holdout, champion_pd_holdout, challenger_pd_holdout,
        vintage_holdout)

    promote = (decision.promote
               and (not ecoa["blocked"])
               and (not ttc.blocked)
               and (not freeze.blocked))
    if ecoa["blocked"]:
        decision.blocked_by.append("ECOA disparate impact: "
                                   + "; ".join(ecoa["reasons"]))
    if ttc.blocked:
        decision.blocked_by.append(f"Basel TTC: {ttc.reason}")
    if freeze.blocked:
        for r in freeze.reasons:
            decision.blocked_by.append(f"operational freeze: {r}")

    outcome_summary = {
        "pd_through_door": (
            challenger.outcome_aipw.pd_through_door if challenger.outcome_aipw
            else float("nan")),
        "pd_through_door_prev": float(np.mean(champion_pd_holdout)),
        "pd_funded": (
            challenger.outcome_aipw.pd_funded if challenger.outcome_aipw
            else float("nan")),
        "pd_declined": (
            challenger.outcome_aipw.pd_declined if challenger.outcome_aipw
            else float("nan")),
    }

    memo = sr_117_memo(
        snapshot_date=snapshot_date,
        champion_version="champion",
        challenger_version=f"challenger-{snapshot_date.date()}",
        drift_reason=drift_reason,
        propensity_summary=challenger.overlap,
        iv_diagnostic=(challenger.iv_diagnostic or IVDiagnostic(
            z_in_outcome_pvalues={}, z_in_outcome_coefs={},
            z_in_outcome_threshold=cfg.di_max_drop, iv_blocked=False,
            blocked_columns=())),
        outcome_summary=outcome_summary,
        gate_decision=GateDecision(
            promote=promote,
            reasons=decision.reasons,
            blocked_by=decision.blocked_by,
            notes=decision.notes,
        ),
        ecoa_diff=ecoa,
        ttc_result=ttc,
        sensitivity_anchor=sensitivity_anchor,
    )

    return PromotionDecision(
        promote=promote,
        drift_reason=drift_reason,
        gate_decision=GateDecision(
            promote=promote,
            reasons=decision.reasons,
            blocked_by=decision.blocked_by,
            notes=decision.notes,
        ),
        ttc_blocked=ttc.blocked,
        ecoa_blocked=ecoa["blocked"],
        memo_markdown=memo,
        eval_=eval_,
    )


def write_artifact(art: RetrainArtifact, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": art.mode,
        "snapshot_date": str(art.snapshot_date.date()),
        "policy_version_id": art.policy_version_id,
        "n_train": art.n_train,
        "n_funded_train": art.n_funded_train,
        "n_matured": art.n_matured,
        "overlap": art.overlap,
        "errors": art.errors,
        "seed": art.seed,
        "outcome_methods": [
            m for m, a in [
                ("aipw", art.outcome_aipw),
                ("heckman", art.outcome_heckman),
                ("aipcw", art.outcome_aipcw),
            ] if a is not None
        ],
        "iv_blocked": (
            art.iv_diagnostic.iv_blocked if art.iv_diagnostic else None),
    }
    p.write_text(json.dumps(payload, indent=2, default=str))
    return p
