"""Schema validation invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reject_inference_pipeline import (
    validate_applicant_snapshot, validate_bureau_outcomes,
    join_snapshot_outcomes,
)


def test_validate_rejects_missing_required(cohort_df):
    bad = cohort_df.drop(columns=["s"])
    with pytest.raises(ValueError, match="missing"):
        validate_applicant_snapshot(bad, ["x1", "x2"], ["z"])


def test_validate_rejects_non_binary_s(cohort_df):
    bad = cohort_df.copy()
    bad.loc[0, "s"] = 2
    with pytest.raises(ValueError, match="s must be"):
        validate_applicant_snapshot(bad, ["x1", "x2"], ["z"])


def test_validate_rejects_nan_features(cohort_df):
    bad = cohort_df.copy()
    bad.loc[0, "x1"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_applicant_snapshot(bad, ["x1", "x2"], ["z"])


def test_validate_rejects_pi_outside_unit_interval(cohort_df):
    bad = cohort_df.copy()
    bad.loc[0, "pi_logged"] = 1.5
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_applicant_snapshot(bad, ["x1", "x2"], ["z"])


def test_require_pi_logged(cohort_df):
    bad = cohort_df.drop(columns=["pi_logged"])
    with pytest.raises(ValueError, match="pi_logged"):
        validate_applicant_snapshot(bad, ["x1", "x2"], ["z"],
                                     require_pi_logged=True)


def test_bureau_outcomes_reject_non_binary_y(cohort_df):
    funded = np.flatnonzero(cohort_df["s"].to_numpy() == 1)
    bad = pd.DataFrame({
        "applicant_id": cohort_df["applicant_id"].iloc[funded].values,
        "observed_at": cohort_df["as_of"].iloc[funded].values,
        "y": np.full(funded.size, 2),
    })
    with pytest.raises(ValueError, match="y must be"):
        validate_bureau_outcomes(bad, y_definition_id="dpd90_18m")


def test_join_drops_immature_applicants(applicant_snapshot, bureau_outcomes):
    early = pd.Timestamp("2024-09-01")
    joined = join_snapshot_outcomes(
        applicant_snapshot, bureau_outcomes, early,
        performance_window_months=18,
    )
    assert int(joined.matured_mask.sum()) == 0


def test_join_keeps_mature_applicants(joined_snapshot):
    n_mat = int(joined_snapshot.matured_mask.sum())
    assert n_mat > 0
    funded = (joined_snapshot.applicants.s == 1).sum()
    assert n_mat <= funded
