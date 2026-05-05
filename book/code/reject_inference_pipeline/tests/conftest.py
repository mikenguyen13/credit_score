"""Shared fixtures for the reject_inference_pipeline test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PKG_PARENT = Path(__file__).resolve().parents[2]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from reject_inference_pipeline import (
    PolicyVersion,
    RetrainConfig,
    join_snapshot_outcomes,
    retrain_observable,
    validate_applicant_snapshot, validate_bureau_outcomes,
)


SEED = 20260504
RHO = 0.6
N_PER_VINTAGE = 800

VINTAGE_BASE = {
    "2024-Q1": pd.Timestamp("2024-02-15"),
    "2024-Q2": pd.Timestamp("2024-05-15"),
    "2024-Q3": pd.Timestamp("2024-08-15"),
}


def _vintage(rng: np.random.Generator, vintage: str,
             n: int = N_PER_VINTAGE) -> pd.DataFrame:
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    z = rng.standard_normal(n)
    u = rng.standard_normal(n)
    v = RHO * u + np.sqrt(1 - RHO ** 2) * rng.standard_normal(n)
    y = ((-0.4 + 0.6 * x1 + 0.4 * x2 + u) > 0).astype(int)
    s = ((0.2 + 0.5 * x1 + 0.3 * x2 + 0.6 * z + v) > 0).astype(int)
    pi = 1.0 / (1.0 + np.exp(-(0.2 + 0.5 * x1 + 0.3 * x2 + 0.6 * z)))
    as_of = (VINTAGE_BASE[vintage]
             + pd.to_timedelta(rng.integers(0, 60, size=n), unit="D"))
    return pd.DataFrame({
        "applicant_id": [f"A{vintage}{i:05d}" for i in range(n)],
        "as_of": as_of, "x1": x1, "x2": x2, "z": z, "s": s,
        "policy_version_id": "P_2026_v1", "pi_logged": pi,
        "vintage": vintage,
        "segment": rng.choice(["digital", "branch"], size=n),
        "_y_truth": y,
    })


@pytest.fixture(scope="session")
def cohort_df() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    return pd.concat(
        [_vintage(rng, v) for v in ("2024-Q1", "2024-Q2", "2024-Q3")],
        ignore_index=True,
    )


@pytest.fixture(scope="session")
def applicant_snapshot(cohort_df: pd.DataFrame):
    return validate_applicant_snapshot(
        cohort_df, feature_cols=["x1", "x2"], iv_cols=["z"],
        require_pi_logged=True,
    )


@pytest.fixture(scope="session")
def bureau_outcomes(cohort_df: pd.DataFrame):
    funded = np.flatnonzero(cohort_df["s"].to_numpy() == 1)
    df = pd.DataFrame({
        "applicant_id": cohort_df["applicant_id"].iloc[funded].values,
        "observed_at": (cohort_df["as_of"].iloc[funded]
                         + pd.DateOffset(months=18)).values,
        "y": cohort_df["_y_truth"].iloc[funded].values,
    })
    return validate_bureau_outcomes(df, y_definition_id="dpd90_18m")


@pytest.fixture(scope="session")
def joined_snapshot(applicant_snapshot, bureau_outcomes):
    return join_snapshot_outcomes(
        applicant_snapshot, bureau_outcomes,
        snapshot_date=pd.Timestamp("2026-05-01"),
        performance_window_months=18,
    )


@pytest.fixture(scope="session")
def policy_observable():
    return PolicyVersion(
        policy_version_id="P_2026_v1",
        effective_from=pd.Timestamp("2024-01-01"),
        effective_to=None,
        propensity_mode="observable",
        iv_columns=("z",),
        label_definition_id="dpd90_18m",
        cutoff=0.0, override_quota=0.05,
    )


@pytest.fixture(scope="session")
def policy_unobservable():
    return PolicyVersion(
        policy_version_id="P_2026_v1u",
        effective_from=pd.Timestamp("2024-01-01"),
        effective_to=None,
        propensity_mode="unobservable",
        iv_columns=("z",),
        label_definition_id="dpd90_18m",
    )


@pytest.fixture(scope="session")
def policy_alt_data():
    return PolicyVersion(
        policy_version_id="P_2026_v1alt",
        effective_from=pd.Timestamp("2024-01-01"),
        effective_to=None,
        propensity_mode="alt_data",
        iv_columns=("z",),
        label_definition_id="dpd90_18m",
    )


@pytest.fixture(scope="session")
def retrain_config():
    return RetrainConfig(
        snapshot_date=pd.Timestamp("2026-05-01"),
        performance_window_months=18,
        bootstrap_B=0,
        cluster_key_col="vintage",
        aipw_n_splits=3,
        seed=SEED,
    )


@pytest.fixture(scope="session")
def observable_artifact(joined_snapshot, retrain_config, policy_observable):
    return retrain_observable(joined_snapshot, retrain_config, policy_observable)
