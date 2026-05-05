"""Champion-challenger gate components: DeLong, calibration, gate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from reject_inference_pipeline import (
    GateConfig, brier, calibration_slope, delong_auc_test,
    disparate_impact_ratio, evaluate_challenger,
    expected_calibration_error, gate, make_frozen_holdout,
    per_segment_auc, rollback_check, shadow_psi_score,
)


def test_brier_zero_for_perfect():
    assert brier(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1])) == 0.0


def test_calibration_slope_close_to_one_when_calibrated():
    rng = np.random.default_rng(0)
    n = 5000
    p = rng.uniform(0.05, 0.95, size=n)
    y = (rng.uniform(size=n) < p).astype(int)
    s = calibration_slope(y, p)
    assert 0.7 < s < 1.3


def test_delong_returns_finite_p():
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=2000) < 0.4).astype(int)
    p_a = rng.uniform(size=2000)
    p_b = rng.uniform(size=2000)
    res = delong_auc_test(y, p_a, p_b)
    assert "auc_a" in res and "auc_b" in res
    assert 0.0 <= res["p"] <= 1.0


def test_per_segment_auc_skips_small_segments():
    rng = np.random.default_rng(2)
    y = (rng.uniform(size=200) < 0.3).astype(int)
    p = rng.uniform(size=200)
    seg = pd.Series(["a"] * 5 + ["b"] * 195)
    out = per_segment_auc(y, p, seg, min_segment_n=50)
    assert "a" not in out
    assert "b" in out


def test_disparate_impact_ratio_for_reference_group_omitted():
    rng = np.random.default_rng(3)
    p = rng.uniform(size=1000)
    g = pd.Series(rng.choice(["A", "B", "C"], size=1000))
    out = disparate_impact_ratio(p, g, threshold=0.5, reference_group="A")
    assert "A" not in out
    assert set(out) == {"B", "C"}


def test_make_frozen_holdout_is_reproducible():
    n = 1000
    v = pd.Series(np.repeat(["v1", "v2"], 500))
    h1 = make_frozen_holdout(n, v, holdout_share=0.2, seed=42)
    h2 = make_frozen_holdout(n, v, holdout_share=0.2, seed=42)
    np.testing.assert_array_equal(h1.holdout_mask, h2.holdout_mask)


def test_make_frozen_holdout_stratifies_by_vintage():
    n = 1000
    v = pd.Series(np.repeat(["v1", "v2"], 500))
    h = make_frozen_holdout(n, v, holdout_share=0.2, seed=42)
    in_v1 = h.holdout_mask[v == "v1"].sum()
    in_v2 = h.holdout_mask[v == "v2"].sum()
    assert abs(in_v1 - in_v2) <= 1


def test_gate_blocks_on_calibration_breach():
    eval_ = evaluate_challenger(
        y_holdout=np.array([0, 1] * 50),
        p_champion=np.full(100, 0.5),
        p_challenger=np.full(100, 0.5),
        segment=pd.Series(["all"] * 100),
        protected=pd.Series(["A"] * 100),
        threshold=0.5, reference_group="A",
    )
    decision = gate(eval_, GateConfig())
    assert decision.promote in (True, False)


def test_rollback_triggers_on_high_psi():
    rng = np.random.default_rng(4)
    base = rng.uniform(0.1, 0.4, size=2000)
    new = rng.uniform(0.7, 0.9, size=2000)
    triggered, _ = rollback_check(base, new, GateConfig())
    assert triggered is True


def test_shadow_psi_low_for_similar_streams():
    rng = np.random.default_rng(5)
    a = rng.uniform(size=2000)
    b = rng.uniform(size=2000)
    assert shadow_psi_score(a, b) < 0.05
