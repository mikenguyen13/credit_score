"""Drift detection and retrain-trigger classifier.

The classifier separates three drift kinds because each demands a
different fix:

* covariate drift   P(X) moves; covariates differ from training.
                    Action: recalibrate or refresh feature scaler.

* concept drift     P(Y | X) moves; same X yields different defaults.
                    Action: full retrain. Most common cause is macro
                    shock.

* selection drift   P(S | X, Z) moves; the lender's accept rule
                    changes. Action: stage-1 refit only (when
                    unobservable) or no-op (when pi is logged and the
                    feature store reflects the new policy).

Conflating the three kinds is the canonical wrong-fix bug: a
covariate-only drift triggers a full outcome retrain, which then
overwrites a perfectly good model with one fit on noisy fresh labels.

DriftTrigger applies hysteresis: a single noisy day is not enough.
Retrain fires only when the same drift signal has crossed its
threshold for ``min_consecutive`` consecutive days, or when a manual
override fires (PolicyVersion bump or ad-hoc operator command).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd


DriftKind = Literal["covariate", "concept", "selection", "none", "ambiguous"]


@dataclass
class DriftThresholds:
    psi_breach: float = 0.25
    kl_breach: float = 0.10
    accept_rate_band: float = 0.03
    imr_p95_growth: float = 0.20      # 20% growth in IMR p95 vs baseline
    pred_dist_psi: float = 0.10       # PSI on predicted PD vs champion
    concept_default_rate_band: float = 0.005  # 50 bps absolute on funded


@dataclass
class DriftReport:
    """Snapshot of drift signals computed on a single comparison day."""

    feature_psi: dict[str, float]
    propensity_psi: float
    propensity_kl: float
    accept_rate_train: float
    accept_rate_observed: float
    imr_p95_train: float
    imr_p95_observed: float
    funded_default_rate_train: float
    funded_default_rate_observed: float
    pred_dist_psi: Optional[float]
    classified: DriftKind
    breaches: list[str]


def psi(
    base: np.ndarray, current: np.ndarray, n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Population Stability Index between two samples."""
    qs = np.quantile(base, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    counts_b, _ = np.histogram(base, bins=qs)
    counts_c, _ = np.histogram(current, bins=qs)
    p_b = (counts_b + eps) / (counts_b.sum() + n_bins * eps)
    p_c = (counts_c + eps) / (counts_c.sum() + n_bins * eps)
    return float(((p_c - p_b) * np.log(p_c / p_b)).sum())


def kl_divergence(base: np.ndarray, current: np.ndarray, n_bins: int = 20,
                  eps: float = 1e-6) -> float:
    qs = np.quantile(np.concatenate([base, current]),
                     np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    counts_b, _ = np.histogram(base, bins=qs)
    counts_c, _ = np.histogram(current, bins=qs)
    p_b = (counts_b + eps) / (counts_b.sum() + n_bins * eps)
    p_c = (counts_c + eps) / (counts_c.sum() + n_bins * eps)
    return float((p_c * np.log(p_c / p_b)).sum())


def classify_drift(
    feature_psi_max: float,
    propensity_psi: float,
    accept_rate_delta: float,
    funded_default_rate_delta: float,
    thresholds: DriftThresholds,
) -> tuple[DriftKind, list[str]]:
    """Map the four primary signals to a drift kind."""
    breaches: list[str] = []
    cov = feature_psi_max > thresholds.psi_breach
    sel = (propensity_psi > thresholds.psi_breach
           or abs(accept_rate_delta) > thresholds.accept_rate_band)
    con = abs(funded_default_rate_delta) > thresholds.concept_default_rate_band
    if cov:
        breaches.append("covariate")
    if sel:
        breaches.append("selection")
    if con:
        breaches.append("concept")
    if not breaches:
        return "none", []
    if len(breaches) == 1:
        return breaches[0], breaches  # type: ignore[return-value]
    return "ambiguous", breaches


def compute_drift(
    train_features: pd.DataFrame,
    current_features: pd.DataFrame,
    train_propensity: np.ndarray,
    current_propensity: np.ndarray,
    train_accept_rate: float,
    current_accept_rate: float,
    train_imr: np.ndarray,
    current_imr: np.ndarray,
    train_funded_default_rate: float,
    current_funded_default_rate: float,
    thresholds: DriftThresholds,
    train_predicted_pd: Optional[np.ndarray] = None,
    current_predicted_pd: Optional[np.ndarray] = None,
) -> DriftReport:
    feat_psi = {
        c: psi(train_features[c].to_numpy(), current_features[c].to_numpy())
        for c in train_features.columns
    }
    prop_psi = psi(train_propensity, current_propensity)
    prop_kl = kl_divergence(train_propensity, current_propensity)

    pred_psi = (psi(train_predicted_pd, current_predicted_pd)
                if (train_predicted_pd is not None
                    and current_predicted_pd is not None) else None)

    kind, breaches = classify_drift(
        feature_psi_max=max(feat_psi.values()) if feat_psi else 0.0,
        propensity_psi=prop_psi,
        accept_rate_delta=current_accept_rate - train_accept_rate,
        funded_default_rate_delta=current_funded_default_rate - train_funded_default_rate,
        thresholds=thresholds,
    )

    return DriftReport(
        feature_psi=feat_psi,
        propensity_psi=prop_psi,
        propensity_kl=prop_kl,
        accept_rate_train=train_accept_rate,
        accept_rate_observed=current_accept_rate,
        imr_p95_train=float(np.quantile(train_imr, 0.95)),
        imr_p95_observed=float(np.quantile(current_imr, 0.95)),
        funded_default_rate_train=train_funded_default_rate,
        funded_default_rate_observed=current_funded_default_rate,
        pred_dist_psi=pred_psi,
        classified=kind,
        breaches=breaches,
    )


@dataclass
class DriftTrigger:
    """Stateful trigger with hysteresis.

    Append daily DriftReports via observe(); the trigger fires only
    when the same drift kind has been classified ``min_consecutive``
    days in a row, or when ``manual_override`` is set.
    """

    thresholds: DriftThresholds = field(default_factory=DriftThresholds)
    min_consecutive: int = 3
    manual_override: bool = False
    _history: deque = field(default_factory=lambda: deque(maxlen=14))

    def observe(self, report: DriftReport) -> None:
        self._history.append(report)

    def should_retrain(self) -> tuple[bool, str]:
        if self.manual_override:
            return True, "manual_override"
        if len(self._history) < self.min_consecutive:
            return False, "insufficient_history"
        recent = list(self._history)[-self.min_consecutive:]
        kinds = {r.classified for r in recent}
        if kinds == {"none"}:
            return False, "stable"
        if len(kinds) > 1:
            return False, "non_stationary_signal"
        kind = recent[0].classified
        if kind == "ambiguous":
            return True, "ambiguous_drift_escalate"
        if kind == "concept":
            return True, "concept_drift_full_retrain"
        if kind == "covariate":
            return True, "covariate_drift_recalibrate"
        if kind == "selection":
            return True, "selection_drift_stage1_refit"
        return False, "no_decision"
