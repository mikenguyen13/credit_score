"""Governance, CFRM, alt-data, operational state."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reject_inference_pipeline import (
    basel_ttc_multi_vintage_gate, ecoa_disparate_impact_diff,
    counterfactual_pd, reliability_index,
    fit_hierarchical_propensity, feedback_loop_guard,
    OperationalState, check_freeze, load_state, save_state,
    set_macro_shock, set_iv_kill,
)
from reject_inference_pipeline.champion_challenger import (
    ChallengerEvaluation,
)


def test_basel_ttc_blocks_when_no_vintages_improve():
    rng = np.random.default_rng(0)
    n = 600
    y = (rng.uniform(size=n) < 0.3).astype(int)
    p_champ = rng.uniform(size=n)
    p_chal = p_champ.copy()
    vintage = pd.Series(rng.choice(["v1", "v2", "v3"], size=n))
    res = basel_ttc_multi_vintage_gate(y, p_champ, p_chal, vintage)
    assert res.blocked is True
    assert res.vintages_improved == 0


def test_basel_ttc_blocks_on_single_vintage_regression():
    rng = np.random.default_rng(1)
    n = 600
    y = (rng.uniform(size=n) < 0.3).astype(int)
    p_champ = rng.uniform(size=n)
    p_chal = p_champ.copy()
    p_chal[:200] += 0.5
    p_chal = np.clip(p_chal, 0, 1)
    vintage = pd.Series(np.repeat(["v1", "v2", "v3"], 200))
    res = basel_ttc_multi_vintage_gate(y, p_champ, p_chal, vintage,
                                         vintage_regression_max=0.001)
    assert res.blocked is True


def test_ecoa_blocks_below_floor():
    eval_ = ChallengerEvaluation(
        auc_test={"auc_a": 0.7, "auc_b": 0.7, "z": 0, "p": 1.0},
        brier_champion=0.1, brier_challenger=0.1,
        cal_slope_champion=1.0, cal_slope_challenger=1.0,
        ece_champion=0.05, ece_challenger=0.05,
        segment_auc_champion={}, segment_auc_challenger={},
        disparate_impact_champion={"B": 0.95},
        disparate_impact_challenger={"B": 0.7},
    )
    diff = ecoa_disparate_impact_diff(eval_, floor=0.8, max_drop=0.05)
    assert diff["blocked"] is True
    assert any("0.700" in r for r in diff["reasons"])


def test_cfrm_returns_reliable_estimate_under_small_shift():
    rng = np.random.default_rng(2)
    n = 2000
    pi_log = rng.uniform(0.3, 0.8, size=n)
    pi_new = np.clip(pi_log * 1.05, 0.0, 1.0)
    s = (rng.uniform(size=n) < pi_log).astype(int)
    funded_mask = s == 1
    y = (rng.uniform(size=int(funded_mask.sum())) < 0.3).astype(int)
    cf = counterfactual_pd(
        pi_log[funded_mask], pi_new[funded_mask], s[funded_mask], y,
        np.ones(int(funded_mask.sum()), dtype=bool), weight_cap=20.0,
    )
    rel = reliability_index(cf, raw_funded_n=int(funded_mask.sum()))
    assert np.isfinite(cf.pd_under_new_policy)
    assert rel["trustworthy"] is True


def test_cfrm_flags_unreliable_under_extreme_shift():
    rng = np.random.default_rng(3)
    n = 1000
    pi_log = rng.uniform(0.001, 0.01, size=n)
    pi_new = np.full(n, 0.95)
    s = np.ones(n, dtype=int)
    y = (rng.uniform(size=n) < 0.3).astype(int)
    cf = counterfactual_pd(pi_log, pi_new, s, y,
                            np.ones(n, dtype=bool), weight_cap=5.0)
    rel = reliability_index(cf, raw_funded_n=n)
    assert cf.n_clipped > 0


def test_alt_data_hierarchical_returns_per_lender(applicant_snapshot):
    rng = np.random.default_rng(4)
    lender = pd.Series(rng.choice(["bankA", "bankB", "bankC"],
                                    size=applicant_snapshot.n))
    art = fit_hierarchical_propensity(applicant_snapshot, lender)
    assert set(art.per_lender) == {"bankA", "bankB", "bankC"}
    for sub in art.per_lender.values():
        assert sub.pi.shape[0] > 0
        assert sub.gamma is not None


def test_feedback_loop_guard_detects_correlated_score(applicant_snapshot):
    rng = np.random.default_rng(5)
    own = (applicant_snapshot.s.astype(float)
           + 0.1 * rng.standard_normal(applicant_snapshot.n))
    out = feedback_loop_guard(applicant_snapshot, own, p_threshold=0.05)
    assert "p_own_score" in out
    assert "feedback_detected" in out


def test_macro_shock_freeze_round_trip(tmp_path: Path):
    p = tmp_path / "state.json"
    state = set_macro_shock(True, "covid_bis", "operator_a", p)
    assert state.macro_shock_freeze is True
    block = check_freeze(load_state(p))
    assert block.blocked is True
    state = set_macro_shock(False, path=p)
    assert state.macro_shock_freeze is False
    assert check_freeze(load_state(p)).blocked is False


def test_iv_kill_blocks_promotion(tmp_path: Path):
    p = tmp_path / "state.json"
    set_iv_kill(True, path=p)
    block = check_freeze(load_state(p))
    assert block.blocked is True
    assert any("IV kill" in r for r in block.reasons)


def test_default_state_unblocked():
    assert check_freeze(OperationalState()).blocked is False
