"""Propensity, IV diagnostic, and overlap invariants."""

from __future__ import annotations

import numpy as np
import pytest

from reject_inference_pipeline import (
    fit_observable_propensity, fit_selection_probit,
    overlap_summary, run_iv_diagnostics,
)


def test_observable_propensity_matches_logged(applicant_snapshot):
    art = fit_observable_propensity(applicant_snapshot)
    assert art.mode == "observable"
    assert art.pi.shape == (applicant_snapshot.n,)
    assert (art.pi > 0).all() and (art.pi < 1).all()
    assert art.imr.shape == art.pi.shape
    np.testing.assert_allclose(
        art.pi, np.clip(applicant_snapshot.pi_logged, 1e-3, 1 - 1e-3))


def test_selection_probit_recovers_signal(applicant_snapshot):
    art = fit_selection_probit(applicant_snapshot)
    assert art.mode == "estimated"
    assert art.gamma.shape[0] == 1 + applicant_snapshot.X.shape[1] + applicant_snapshot.Z.shape[1]
    assert art.selection_auc is not None
    assert art.selection_auc > 0.65


def test_iv_diagnostic_passes_with_imr_control(applicant_snapshot):
    prop = fit_selection_probit(applicant_snapshot)
    funded = applicant_snapshot.s == 1
    y_funded = (applicant_snapshot.X.iloc[funded.nonzero()[0], 0]
                .to_numpy() > 0).astype(int)
    iv = run_iv_diagnostics(applicant_snapshot, y_funded, funded,
                             p_threshold=0.01, imr=prop.imr)
    assert isinstance(iv.iv_blocked, bool)
    assert iv.f_stat_first_stage is not None


def test_iv_diagnostic_f_stat_strong_under_designed_iv(applicant_snapshot):
    prop = fit_selection_probit(applicant_snapshot)
    funded = applicant_snapshot.s == 1
    y_funded = np.zeros(int(funded.sum()))
    iv = run_iv_diagnostics(applicant_snapshot, y_funded, funded,
                             imr=prop.imr)
    assert iv.f_stat_first_stage > 10.0
    assert iv.weak_iv_blocked is False


def test_observable_overlap_within_bounds(applicant_snapshot):
    art = fit_observable_propensity(applicant_snapshot)
    summary = overlap_summary(art)
    assert summary["min"] >= 0.0
    assert summary["max"] <= 1.0
    assert summary["p99"] >= summary["p01"]


def test_observable_propensity_requires_pi(cohort_df):
    from reject_inference_pipeline.schema import validate_applicant_snapshot
    bad = cohort_df.drop(columns=["pi_logged"])
    apps = validate_applicant_snapshot(bad, ["x1", "x2"], ["z"])
    with pytest.raises(ValueError, match="logged pi"):
        fit_observable_propensity(apps)
