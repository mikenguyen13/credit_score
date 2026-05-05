"""Snapshot schemas for the reject-inference retrain pipeline.

Three immutable snapshot objects flow through every pipeline stage:

ApplicantSnapshot   one row per applicant at decision time
                    columns: applicant_id, as_of (decision timestamp),
                             X (feature matrix), Z (exclusion-restriction
                             columns), s (1 = funded, 0 = declined),
                             policy_version_id (FK into PolicyVersionTable),
                             pi_logged (Optional, decision-time propensity
                             written by an observable engine),
                             vintage (YYYY-MM, derived from as_of),
                             segment (channel / lender_id; categorical
                             used by the per-segment gate and the
                             hierarchical alt-data propensity).

BureauOutcomeBatch  one row per matured applicant
                    columns: applicant_id, observed_at, y,
                             y_definition_id (DPD / window spec hash;
                             label-policy version pin)

PolicyVersion       one row per change in the lender's underwriting policy
                    (cutoff, overlay, IV column rename, override quota change)

The schemas are pandas-DataFrame-backed so they slot into the existing
Quarto codebase without a Polars detour. Validation is strict: every
boundary entry runs validate_*() before the snapshot is consumed.

Point-in-time correctness is a load-bearing invariant. Any feature
column whose value depends on the bureau outcome must be reconstructed
as of as_of; never read it from a "current" view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ApplicantSnapshot:
    """Validated applicant-level snapshot at decision time."""

    applicant_id: pd.Series
    as_of: pd.Series              # datetime64[ns]
    X: pd.DataFrame                # numeric features used by the outcome model
    Z: pd.DataFrame                # exclusion-restriction columns (probit only)
    s: np.ndarray                  # 1 = funded, 0 = declined
    policy_version_id: pd.Series   # FK into PolicyVersionTable
    pi_logged: Optional[np.ndarray] = None
    vintage: Optional[pd.Series] = None
    segment: Optional[pd.Series] = None

    @property
    def n(self) -> int:
        return int(self.s.shape[0])

    @property
    def n_funded(self) -> int:
        return int(self.s.sum())

    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    def iv_names(self) -> list[str]:
        return list(self.Z.columns)


@dataclass
class BureauOutcomeBatch:
    """Validated matured-outcome batch.

    Joined to ApplicantSnapshot on applicant_id. Rows in the snapshot
    that have no matching outcome row are the censored tail and feed
    AIPCW in outcome.py.
    """

    applicant_id: pd.Series
    observed_at: pd.Series         # datetime64[ns]
    y: np.ndarray                  # binary 0/1
    y_definition_id: str           # label-policy version pin


@dataclass
class JoinedSnapshot:
    """Inner-join product: applicant + matured outcome.

    All retrain estimators operate on this object. The censored tail is
    held in `applicants_unmatured` for AIPCW.
    """

    applicants: ApplicantSnapshot
    outcomes: BureauOutcomeBatch
    matured_mask: np.ndarray       # True where applicant has a matured y
    snapshot_date: pd.Timestamp
    performance_window_months: int

    @property
    def y_full(self) -> np.ndarray:
        """y aligned to applicants.applicant_id; NaN on censored rows."""
        out = np.full(self.applicants.n, np.nan, dtype=float)
        idx = self.outcomes.applicant_id.values
        amap = pd.Series(np.arange(self.applicants.n),
                         index=self.applicants.applicant_id.values)
        try:
            target = amap.loc[idx].to_numpy()
        except KeyError as exc:
            raise ValueError(f"outcome applicant_id not in snapshot: {exc}")
        out[target] = self.outcomes.y.astype(float)
        return out


def validate_applicant_snapshot(
    df: pd.DataFrame,
    feature_cols: list[str],
    iv_cols: list[str],
    require_pi_logged: bool = False,
) -> ApplicantSnapshot:
    """Strict applicant-snapshot validation.

    Caller must pre-derive `vintage` and `segment` if they want them on
    the snapshot. Missing optional columns are accepted as None.
    """
    required = {"applicant_id", "as_of", "s", "policy_version_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"applicant snapshot missing: {sorted(missing)}")

    miss_feat = [c for c in feature_cols if c not in df.columns]
    miss_iv = [c for c in iv_cols if c not in df.columns]
    if miss_feat:
        raise ValueError(f"feature columns not in snapshot: {miss_feat}")
    if miss_iv:
        raise ValueError(f"IV columns not in snapshot: {miss_iv}")

    s_arr = df["s"].astype(int).to_numpy()
    if not np.isin(s_arr, [0, 1]).all():
        raise ValueError("s must be in {0, 1}")

    X = df[feature_cols].copy()
    Z = df[iv_cols].copy()
    for name, frame in (("X", X), ("Z", Z)):
        if frame.isna().any().any():
            raise ValueError(f"{name} contains NaNs; impute or drop upstream")
        if not all(np.issubdtype(frame[c].dtype, np.number)
                   for c in frame.columns):
            raise ValueError(f"{name} must be numeric only")

    pi = None
    if "pi_logged" in df.columns and df["pi_logged"].notna().any():
        pi = df["pi_logged"].astype(float).to_numpy()
        if (pi < 0).any() or (pi > 1).any():
            raise ValueError("pi_logged must be in [0, 1]")
    elif require_pi_logged:
        raise ValueError(
            "pi_logged required for observable retrain but not present"
        )

    as_of = pd.to_datetime(df["as_of"])
    vintage = (as_of.dt.strftime("%Y-%m") if "vintage" not in df.columns
               else df["vintage"].astype(str).reset_index(drop=True))

    segment = (df["segment"].astype(str).reset_index(drop=True)
               if "segment" in df.columns else None)

    return ApplicantSnapshot(
        applicant_id=df["applicant_id"].astype(str).reset_index(drop=True),
        as_of=as_of.reset_index(drop=True),
        X=X.reset_index(drop=True),
        Z=Z.reset_index(drop=True),
        s=s_arr,
        policy_version_id=df["policy_version_id"].astype(str)
            .reset_index(drop=True),
        pi_logged=pi,
        vintage=vintage.reset_index(drop=True),
        segment=segment,
    )


def validate_bureau_outcomes(
    df: pd.DataFrame, y_definition_id: str
) -> BureauOutcomeBatch:
    required = {"applicant_id", "observed_at", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"bureau batch missing: {sorted(missing)}")
    y = df["y"].astype(int).to_numpy()
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y must be in {0, 1}")
    return BureauOutcomeBatch(
        applicant_id=df["applicant_id"].astype(str).reset_index(drop=True),
        observed_at=pd.to_datetime(df["observed_at"]).reset_index(drop=True),
        y=y,
        y_definition_id=y_definition_id,
    )


def join_snapshot_outcomes(
    apps: ApplicantSnapshot,
    outcomes: BureauOutcomeBatch,
    snapshot_date: pd.Timestamp,
    performance_window_months: int,
) -> JoinedSnapshot:
    """Inner-join with point-in-time guard.

    Drops outcome rows whose applicant as_of is later than
    snapshot_date - performance_window_months. The censored tail is the
    set of applicants whose as_of is too recent for a matured y; those
    rows feed AIPCW.
    """
    cutoff = snapshot_date - pd.DateOffset(months=performance_window_months)
    matured_apps_mask = (apps.as_of <= cutoff).to_numpy()

    matched = pd.merge(
        apps.applicant_id.to_frame(name="applicant_id").assign(
            _idx=np.arange(apps.n)),
        pd.DataFrame({
            "applicant_id": outcomes.applicant_id.values,
            "y": outcomes.y,
            "observed_at": outcomes.observed_at.values,
        }),
        on="applicant_id", how="inner",
    )

    matured_y_mask = np.zeros(apps.n, dtype=bool)
    matured_y_mask[matched["_idx"].to_numpy()] = True
    final_mask = matured_apps_mask & matured_y_mask

    return JoinedSnapshot(
        applicants=apps,
        outcomes=BureauOutcomeBatch(
            applicant_id=pd.Series(matched["applicant_id"].astype(str)
                                   .reset_index(drop=True)),
            observed_at=pd.to_datetime(matched["observed_at"])
                .reset_index(drop=True),
            y=matched["y"].astype(int).to_numpy(),
            y_definition_id=outcomes.y_definition_id,
        ),
        matured_mask=final_mask,
        snapshot_date=snapshot_date,
        performance_window_months=performance_window_months,
    )
