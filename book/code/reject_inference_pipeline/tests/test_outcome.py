"""Outcome estimators: Heckman, AIPW, AIPCW, prediction."""

from __future__ import annotations

import numpy as np

from reject_inference_pipeline import (
    fit_aipw_outcome, fit_heckman_outcome, predict_pd,
    fit_observable_propensity,
)


def _funded_y(applicant_snapshot, joined_snapshot):
    funded = (applicant_snapshot.s == 1) & joined_snapshot.matured_mask
    y_full = joined_snapshot.y_full
    return y_full[funded], funded


def test_heckman_returns_finite_pd(applicant_snapshot, joined_snapshot):
    prop = fit_observable_propensity(applicant_snapshot)
    y_funded, funded = _funded_y(applicant_snapshot, joined_snapshot)
    art = fit_heckman_outcome(applicant_snapshot, y_funded, funded, prop)
    assert art.method == "heckman"
    assert np.isfinite(art.pd_through_door)
    assert 0.0 <= art.pd_through_door <= 1.0
    assert art.beta.shape[0] == applicant_snapshot.X.shape[1] + 2  # intercept + X + IMR


def test_aipw_returns_finite_pd(applicant_snapshot, joined_snapshot):
    prop = fit_observable_propensity(applicant_snapshot)
    y_funded, funded = _funded_y(applicant_snapshot, joined_snapshot)
    art = fit_aipw_outcome(applicant_snapshot, y_funded, funded, prop,
                            n_splits=3)
    assert art.method == "aipw"
    assert 0.0 <= art.pd_through_door <= 1.0
    assert art.beta.shape[0] == applicant_snapshot.X.shape[1] + 1  # intercept + X


def test_predict_pd_shapes(applicant_snapshot, joined_snapshot):
    prop = fit_observable_propensity(applicant_snapshot)
    y_funded, funded = _funded_y(applicant_snapshot, joined_snapshot)
    aipw = fit_aipw_outcome(applicant_snapshot, y_funded, funded, prop,
                             n_splits=3)
    heck = fit_heckman_outcome(applicant_snapshot, y_funded, funded, prop)
    X = applicant_snapshot.X.to_numpy()[:50]
    pred_aipw = predict_pd(aipw, X)
    pred_heck = predict_pd(heck, X, prop)
    assert pred_aipw.shape == (50,)
    assert pred_heck.shape == (50,)
    assert ((pred_aipw >= 0) & (pred_aipw <= 1)).all()
    assert ((pred_heck >= 0) & (pred_heck <= 1)).all()


def test_heckman_and_aipw_are_in_same_ballpark(applicant_snapshot,
                                                 joined_snapshot):
    """Sensitivity-anchor sanity check: AIPW vs Heckman PD gap < 10pp."""
    prop = fit_observable_propensity(applicant_snapshot)
    y_funded, funded = _funded_y(applicant_snapshot, joined_snapshot)
    aipw = fit_aipw_outcome(applicant_snapshot, y_funded, funded, prop,
                             n_splits=3)
    heck = fit_heckman_outcome(applicant_snapshot, y_funded, funded, prop)
    assert abs(aipw.pd_through_door - heck.pd_through_door) < 0.10
