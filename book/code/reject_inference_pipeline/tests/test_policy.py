"""PolicyVersionTable invariants and change-action enumeration."""

from __future__ import annotations

import pandas as pd
import pytest

from reject_inference_pipeline import (
    PolicyVersion, PolicyVersionTable, policy_change_required_actions,
)


def _v(pid, eff_from, eff_to, mode="observable", iv=("z",), label="dpd90_18m"):
    return PolicyVersion(
        policy_version_id=pid,
        effective_from=pd.Timestamp(eff_from),
        effective_to=pd.Timestamp(eff_to) if eff_to else None,
        propensity_mode=mode, iv_columns=iv, label_definition_id=label,
    )


def test_table_rejects_overlap():
    a = _v("A", "2024-01-01", "2024-06-30")
    b = _v("B", "2024-06-15", None)  # overlaps A
    with pytest.raises(ValueError, match="overlap"):
        PolicyVersionTable(rows=(a, b))


def test_table_requires_open_final_row():
    a = _v("A", "2024-01-01", "2024-06-30")
    b = _v("B", "2024-07-01", "2024-12-31")  # closed
    with pytest.raises(ValueError, match="effective_to=None"):
        PolicyVersionTable(rows=(a, b))


def test_active_returns_correct_row():
    a = _v("A", "2024-01-01", "2024-06-30")
    b = _v("B", "2024-07-01", None)
    t = PolicyVersionTable(rows=(a, b))
    assert t.active(pd.Timestamp("2024-03-01")).policy_version_id == "A"
    assert t.active(pd.Timestamp("2024-09-01")).policy_version_id == "B"


def test_active_raises_outside_coverage():
    a = _v("A", "2024-01-01", None)
    t = PolicyVersionTable(rows=(a,))
    with pytest.raises(KeyError):
        t.active(pd.Timestamp("2023-12-01"))


def test_changed_since_filters_by_effective_from():
    a = _v("A", "2024-01-01", "2024-06-30")
    b = _v("B", "2024-07-01", None)
    t = PolicyVersionTable(rows=(a, b))
    changes = t.changed_since(pd.Timestamp("2024-05-01"))
    assert [r.policy_version_id for r in changes] == ["B"]


def test_change_actions_label_redef_forces_full_retrain():
    a = _v("A", "2024-01-01", None, label="dpd90_18m")
    b = _v("B", "2024-01-01", None, label="dpd60_24m")
    actions = policy_change_required_actions(a, b)
    assert "re_extract_labels" in actions
    assert "full_retrain" in actions


def test_change_actions_iv_change_forces_recheck():
    a = _v("A", "2024-01-01", None, iv=("z",))
    b = _v("B", "2024-01-01", None, iv=("z2",))
    actions = policy_change_required_actions(a, b)
    assert "rerun_iv_diagnostics" in actions


def test_change_actions_quota_change_triggers_overlap_check():
    a = _v("A", "2024-01-01", None)
    a = PolicyVersion(**{**a.__dict__, "override_quota": 0.05})
    b = _v("B", "2024-01-01", None)
    b = PolicyVersion(**{**b.__dict__, "override_quota": 0.10})
    actions = policy_change_required_actions(a, b)
    assert "recheck_overlap" in actions


def test_change_actions_noop_when_identical():
    a = _v("A", "2024-01-01", None)
    actions = policy_change_required_actions(a, a)
    assert actions == ["no_op"]
