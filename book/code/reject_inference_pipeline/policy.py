"""Policy version table and propensity-mode classification.

Every change to the lender's underwriting policy is a row in
PolicyVersionTable: cutoff move, overlay flip, IV column rename,
override-quota change, label-policy redefinition. Retrain triggers
fire on policy-version bumps even when drift signals are quiet (a
freshly-effective policy invalidates the previous selection model
before any data has been observed under the new regime).

Three propensity modes are recognised:

* observable          decision-time pi_i is logged on every applicant;
                      retrain reads it from the snapshot.
* unobservable        no logged pi; the retrain must fit a Heckman
                      stage-1 probit on (X, Z) and check the exclusion
                      restriction every cycle.
* alt_data            the modeller is outside the lender; the lender's
                      policy is opaque. Per-lender pi_k must be
                      re-estimated each cycle from accept/decline pairs
                      returned by the lender, with hierarchical
                      shrinkage across lenders (alt_data.py).

The mode is a function of the policy version, not a global flag: a
single provider may serve banks under all three modes simultaneously
(e.g., one observable fintech client and three unobservable banks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd


PropensityMode = Literal["observable", "unobservable", "alt_data"]


@dataclass(frozen=True)
class PolicyVersion:
    """Single immutable row in the policy-version log."""

    policy_version_id: str
    effective_from: pd.Timestamp
    effective_to: Optional[pd.Timestamp]   # None = currently active
    propensity_mode: PropensityMode
    iv_columns: tuple[str, ...]            # exclusion-restriction columns
    label_definition_id: str
    cutoff: Optional[float] = None          # decision-engine cutoff if known
    override_quota: Optional[float] = None  # random-override share, e.g. 0.05
    notes: str = ""

    def covers(self, ts: pd.Timestamp) -> bool:
        if ts < self.effective_from:
            return False
        if self.effective_to is not None and ts >= self.effective_to:
            return False
        return True


@dataclass
class PolicyVersionTable:
    """Append-only log of policy versions.

    Construction guarantees: rows are sorted by effective_from, no two
    rows overlap on the time axis, and there is exactly one row whose
    effective_to is None (the currently active policy).
    """

    rows: tuple[PolicyVersion, ...]

    def __post_init__(self) -> None:
        rs = sorted(self.rows, key=lambda r: r.effective_from)
        for a, b in zip(rs, rs[1:]):
            if a.effective_to is None:
                raise ValueError(
                    f"non-final policy {a.policy_version_id} has open end"
                )
            if a.effective_to > b.effective_from:
                raise ValueError(
                    f"policies {a.policy_version_id} and "
                    f"{b.policy_version_id} overlap in time"
                )
        if rs and rs[-1].effective_to is not None:
            raise ValueError("final policy must have effective_to=None")
        object.__setattr__(self, "rows", tuple(rs))

    def active(self, ts: pd.Timestamp) -> PolicyVersion:
        for r in self.rows:
            if r.covers(ts):
                return r
        raise KeyError(f"no policy active at {ts}")

    def get(self, policy_version_id: str) -> PolicyVersion:
        for r in self.rows:
            if r.policy_version_id == policy_version_id:
                return r
        raise KeyError(policy_version_id)

    def changed_since(
        self, last_retrain_at: pd.Timestamp,
    ) -> list[PolicyVersion]:
        return [r for r in self.rows
                if r.effective_from > last_retrain_at]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "policy_version_id": r.policy_version_id,
                "effective_from": r.effective_from,
                "effective_to": r.effective_to,
                "propensity_mode": r.propensity_mode,
                "iv_columns": ",".join(r.iv_columns),
                "label_definition_id": r.label_definition_id,
                "cutoff": r.cutoff,
                "override_quota": r.override_quota,
                "notes": r.notes,
            }
            for r in self.rows
        ])


def policy_change_required_actions(
    a: PolicyVersion, b: PolicyVersion
) -> list[str]:
    """Enumerate required pipeline actions when moving from a to b.

    The list is consumed by the orchestrator: each action maps to a
    function call that the retrain DAG executes before promotion. The
    point is to make policy-induced retraining explicit: a cutoff
    change can be handled by an outcome-only refit, but a label
    redefinition requires a full re-extraction.
    """
    actions: list[str] = []
    if a.label_definition_id != b.label_definition_id:
        actions.append("re_extract_labels")
        actions.append("full_retrain")
    if a.propensity_mode != b.propensity_mode:
        actions.append("switch_propensity_mode")
        actions.append("full_retrain")
    if set(a.iv_columns) != set(b.iv_columns):
        actions.append("rerun_iv_diagnostics")
        actions.append("full_retrain")
    if a.cutoff != b.cutoff:
        actions.append("refit_propensity_or_rdd")
    if a.override_quota != b.override_quota:
        actions.append("recheck_overlap")
    if not actions:
        actions.append("no_op")
    return actions
