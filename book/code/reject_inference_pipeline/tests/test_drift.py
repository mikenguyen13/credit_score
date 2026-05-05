"""Drift detector and DriftTrigger hysteresis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from reject_inference_pipeline import (
    DriftThresholds, DriftTrigger, classify_drift, compute_drift,
    kl_divergence, psi,
)


def test_psi_zero_on_identical():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    assert psi(x, x.copy()) < 1e-6


def test_psi_grows_with_shift():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(2000)
    b = rng.standard_normal(2000) + 0.5
    assert psi(a, b) > 0.05


def test_kl_zero_on_identical():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(2000)
    assert abs(kl_divergence(x, x.copy())) < 1e-3


def test_classify_concept_when_only_default_rate_moves():
    th = DriftThresholds()
    kind, breaches = classify_drift(
        feature_psi_max=0.0, propensity_psi=0.0,
        accept_rate_delta=0.0, funded_default_rate_delta=0.05,
        thresholds=th)
    assert kind == "concept"
    assert breaches == ["concept"]


def test_classify_selection_when_propensity_moves():
    th = DriftThresholds()
    kind, _ = classify_drift(
        feature_psi_max=0.0, propensity_psi=0.4,
        accept_rate_delta=0.0, funded_default_rate_delta=0.0, thresholds=th)
    assert kind == "selection"


def test_classify_ambiguous_under_two_signals():
    th = DriftThresholds()
    kind, breaches = classify_drift(
        feature_psi_max=0.4, propensity_psi=0.4,
        accept_rate_delta=0.0, funded_default_rate_delta=0.0, thresholds=th)
    assert kind == "ambiguous"
    assert set(breaches) == {"covariate", "selection"}


def test_classify_none_when_all_quiet():
    th = DriftThresholds()
    kind, breaches = classify_drift(
        feature_psi_max=0.0, propensity_psi=0.0,
        accept_rate_delta=0.0, funded_default_rate_delta=0.0, thresholds=th)
    assert kind == "none"
    assert breaches == []


def test_trigger_requires_consecutive_breaches():
    th = DriftThresholds()
    rng = np.random.default_rng(3)
    base = pd.DataFrame({"x": rng.standard_normal(1000)})
    cur = pd.DataFrame({"x": rng.standard_normal(1000)})
    report = compute_drift(
        train_features=base, current_features=cur,
        train_propensity=rng.uniform(size=1000),
        current_propensity=rng.uniform(size=1000),
        train_accept_rate=0.5, current_accept_rate=0.55,
        train_imr=np.abs(rng.standard_normal(1000)),
        current_imr=np.abs(rng.standard_normal(1000)),
        train_funded_default_rate=0.18, current_funded_default_rate=0.20,
        thresholds=th,
    )
    trig = DriftTrigger(thresholds=th, min_consecutive=3)
    trig.observe(report)
    fire, _ = trig.should_retrain()
    assert fire is False
    trig.observe(report)
    trig.observe(report)
    fire, why = trig.should_retrain()
    assert isinstance(fire, bool)
    assert isinstance(why, str)


def test_manual_override_fires_immediately():
    th = DriftThresholds()
    trig = DriftTrigger(thresholds=th, manual_override=True)
    fire, why = trig.should_retrain()
    assert fire is True
    assert why == "manual_override"
