"""Champion-challenger gating with shadow mode and auto-rollback.

A challenger never replaces the champion on a single metric. The gate
enforces:

1. *Frozen holdout*: a stratified vintage holdout reserved at the
   first ever retrain and never seen by any subsequent fit.

2. *Shadow mode*: the challenger scores in parallel with the champion
   for ``shadow_days`` of live traffic. Promotion proceeds only if
   prediction-distribution drift between champion and challenger on
   the live stream stays below ``shadow_psi_max``.

3. *Multi-metric gate*: AUC (DeLong), Brier, calibration slope, ECE,
   per-segment AUC, ECOA disparate-impact parity. Each metric must
   improve or be statistically indistinguishable from the champion;
   any single regression blocks the swap.

4. *Auto-rollback*: post-promotion live PSI on predicted PD vs the
   champion's pre-promotion baseline. A breach reverts the registry
   to the previous champion within one scoring cycle.

The gating logic is decoupled from MLflow / model registry. Plugging
into a registry is one function call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from .drift import psi


@dataclass
class HoldoutSplit:
    """Frozen holdout descriptor."""

    holdout_mask: np.ndarray
    train_mask: np.ndarray
    seed: int
    stratify_by: tuple[str, ...]


def make_frozen_holdout(
    n: int, vintage: pd.Series,
    holdout_share: float = 0.15, seed: int = 20260504,
) -> HoldoutSplit:
    """Stratified holdout by vintage, reproducible from seed.

    Stored once on disk; subsequent retrains read the same mask. The
    mask is a vector of length n_full_history; aligning it to the
    current snapshot is the orchestrator's job.
    """
    rng = np.random.default_rng(seed)
    holdout = np.zeros(n, dtype=bool)
    by_v = pd.Series(np.arange(n)).groupby(vintage.values)
    for _, idxs in by_v:
        idxs_arr = idxs.to_numpy()
        k = max(1, int(round(len(idxs_arr) * holdout_share)))
        chosen = rng.choice(idxs_arr, size=k, replace=False)
        holdout[chosen] = True
    return HoldoutSplit(
        holdout_mask=holdout,
        train_mask=~holdout,
        seed=seed,
        stratify_by=("vintage",),
    )


def delong_auc_test(
    y_true: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
) -> dict[str, float]:
    """DeLong z-test for paired AUC difference (model A vs model B).

    Returns auc_a, auc_b, z, p_two_sided. Paired structure avoided by
    using a closed-form covariance from the empirical distribution of
    midranks; this implementation uses the @sun2014fast variant which
    is O(n log n).
    """
    from scipy import stats as _st

    def _midrank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        x_sorted = x[order]
        ranks = np.empty_like(x, dtype=float)
        i = 0
        N = len(x)
        while i < N:
            j = i
            while j < N - 1 and x_sorted[j + 1] == x_sorted[i]:
                j += 1
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return ranks

    pos = y_true == 1
    neg = ~pos
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return {"auc_a": float("nan"), "auc_b": float("nan"),
                "z": float("nan"), "p": float("nan")}
    auc_a = float(((p_a[pos][:, None] > p_a[neg][None, :]).sum()
                  + 0.5 * (p_a[pos][:, None] == p_a[neg][None, :]).sum())
                  / (n_pos * n_neg))
    auc_b = float(((p_b[pos][:, None] > p_b[neg][None, :]).sum()
                  + 0.5 * (p_b[pos][:, None] == p_b[neg][None, :]).sum())
                  / (n_pos * n_neg))
    tx_a = (_midrank(p_a[pos]) - 0.5 * (n_pos + 1)) / n_neg
    tx_b = (_midrank(p_b[pos]) - 0.5 * (n_pos + 1)) / n_neg
    ty_a = 1.0 - (_midrank(p_a[neg]) - 0.5 * (n_neg + 1)) / n_pos
    ty_b = 1.0 - (_midrank(p_b[neg]) - 0.5 * (n_neg + 1)) / n_pos
    var_a = tx_a.var(ddof=1) / n_pos + ty_a.var(ddof=1) / n_neg
    var_b = tx_b.var(ddof=1) / n_pos + ty_b.var(ddof=1) / n_neg
    cov = (np.cov(tx_a, tx_b, ddof=1)[0, 1] / n_pos
           + np.cov(ty_a, ty_b, ddof=1)[0, 1] / n_neg)
    var_diff = max(var_a + var_b - 2 * cov, 1e-12)
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2.0 * (1.0 - _st.norm.cdf(abs(z)))
    return {"auc_a": auc_a, "auc_b": auc_b, "z": float(z), "p": float(p)}


def brier(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y_true) ** 2))


def calibration_slope(y_true: np.ndarray, p: np.ndarray) -> float:
    """Cox-style calibration slope: GLM y ~ logit(p) under a logit link.

    A well-calibrated model returns a slope close to 1. A slope below
    1 indicates over-fitting (predicted probabilities are too extreme);
    above 1 indicates under-fitting.
    """
    import statsmodels.api as sm
    p_clip = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p_clip / (1 - p_clip))
    Z = np.column_stack([np.ones(len(z)), z])
    try:
        fit = sm.GLM(y_true.astype(float), Z,
                     family=sm.families.Binomial()).fit(disp=False)
        return float(fit.params[1])
    except Exception:
        return float("nan")


def expected_calibration_error(
    y_true: np.ndarray, p: np.ndarray, n_bins: int = 10,
) -> float:
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    bins = np.clip(np.digitize(p, qs) - 1, 0, n_bins - 1)
    out = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.sum() == 0:
            continue
        out += (m.sum() / len(p)) * abs(p[m].mean() - y_true[m].mean())
    return float(out)


def per_segment_auc(
    y_true: np.ndarray, p: np.ndarray, segment: pd.Series,
    min_segment_n: int = 50,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, idx in segment.groupby(segment.values).groups.items():
        idx_arr = np.asarray(idx)
        if idx_arr.size < min_segment_n:
            continue
        if y_true[idx_arr].sum() in (0, idx_arr.size):
            continue
        from sklearn.metrics import roc_auc_score
        out[str(k)] = float(roc_auc_score(y_true[idx_arr], p[idx_arr]))
    return out


def disparate_impact_ratio(
    p: np.ndarray, group: pd.Series,
    threshold: float, reference_group: str,
) -> dict[str, float]:
    """Approval-rate ratio (group / reference) at a fixed threshold.

    The ECOA / EEOC four-fifths rule treats < 0.8 as a prima facie
    disparity. Output one ratio per non-reference group.
    """
    approve_full = (p < threshold)  # below = lower-risk = approved
    ref = approve_full[group == reference_group].mean()
    out: dict[str, float] = {}
    for k, idx in group.groupby(group.values).groups.items():
        if str(k) == reference_group:
            continue
        rate_k = approve_full[np.asarray(idx)].mean()
        out[str(k)] = float(rate_k / max(ref, 1e-9))
    return out


@dataclass
class ChallengerEvaluation:
    """All metrics needed for the gate, computed on the frozen holdout."""

    auc_test: dict[str, float]
    brier_champion: float
    brier_challenger: float
    cal_slope_champion: float
    cal_slope_challenger: float
    ece_champion: float
    ece_challenger: float
    segment_auc_champion: dict[str, float]
    segment_auc_challenger: dict[str, float]
    disparate_impact_champion: dict[str, float]
    disparate_impact_challenger: dict[str, float]


@dataclass
class GateConfig:
    auc_pvalue: float = 0.05
    brier_relative_max_regression: float = 0.02   # 2 percent
    cal_slope_target: tuple[float, float] = (0.85, 1.15)
    ece_relative_max_regression: float = 0.10
    segment_auc_max_regression: float = 0.02
    di_floor: float = 0.80
    di_max_drop: float = 0.02
    shadow_days: int = 7
    shadow_psi_max: float = 0.10
    rollback_psi: float = 0.20


def evaluate_challenger(
    y_holdout: np.ndarray,
    p_champion: np.ndarray,
    p_challenger: np.ndarray,
    segment: pd.Series,
    protected: pd.Series,
    threshold: float,
    reference_group: str,
) -> ChallengerEvaluation:
    return ChallengerEvaluation(
        auc_test=delong_auc_test(y_holdout, p_challenger, p_champion),
        brier_champion=brier(y_holdout, p_champion),
        brier_challenger=brier(y_holdout, p_challenger),
        cal_slope_champion=calibration_slope(y_holdout, p_champion),
        cal_slope_challenger=calibration_slope(y_holdout, p_challenger),
        ece_champion=expected_calibration_error(y_holdout, p_champion),
        ece_challenger=expected_calibration_error(y_holdout, p_challenger),
        segment_auc_champion=per_segment_auc(y_holdout, p_champion, segment),
        segment_auc_challenger=per_segment_auc(y_holdout, p_challenger, segment),
        disparate_impact_champion=disparate_impact_ratio(
            p_champion, protected, threshold, reference_group),
        disparate_impact_challenger=disparate_impact_ratio(
            p_challenger, protected, threshold, reference_group),
    )


@dataclass
class GateDecision:
    promote: bool
    reasons: list[str]
    blocked_by: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)


def gate(
    eval_: ChallengerEvaluation,
    cfg: GateConfig = GateConfig(),
    shadow_psi: Optional[float] = None,
) -> GateDecision:
    """Apply the multi-metric gate to a ChallengerEvaluation."""
    blocked: list[str] = []
    reasons: list[str] = []

    auc_a = eval_.auc_test["auc_a"]
    auc_b = eval_.auc_test["auc_b"]
    auc_p = eval_.auc_test["p"]
    if auc_a < auc_b and auc_p < cfg.auc_pvalue:
        blocked.append(
            f"AUC regression: challenger {auc_a:.4f} < champion {auc_b:.4f} "
            f"(p={auc_p:.4f})"
        )
    else:
        reasons.append(
            f"AUC: challenger {auc_a:.4f} vs champion {auc_b:.4f} "
            f"(p={auc_p:.4f})"
        )

    rel_brier = (eval_.brier_challenger - eval_.brier_champion) / max(
        eval_.brier_champion, 1e-12)
    if rel_brier > cfg.brier_relative_max_regression:
        blocked.append(f"Brier regression: +{rel_brier:.2%}")
    else:
        reasons.append(f"Brier delta: {rel_brier:+.2%}")

    lo, hi = cfg.cal_slope_target
    if not (lo <= eval_.cal_slope_challenger <= hi):
        blocked.append(
            f"calibration slope {eval_.cal_slope_challenger:.3f} "
            f"outside [{lo}, {hi}]"
        )

    rel_ece = ((eval_.ece_challenger - eval_.ece_champion)
               / max(eval_.ece_champion, 1e-12))
    if rel_ece > cfg.ece_relative_max_regression:
        blocked.append(f"ECE regression: +{rel_ece:.2%}")

    seg_keys = (set(eval_.segment_auc_champion)
                & set(eval_.segment_auc_challenger))
    seg_regress = [
        (k, eval_.segment_auc_champion[k] - eval_.segment_auc_challenger[k])
        for k in seg_keys
        if (eval_.segment_auc_champion[k] - eval_.segment_auc_challenger[k])
        > cfg.segment_auc_max_regression
    ]
    if seg_regress:
        blocked.append(
            "segment AUC regression: "
            + ", ".join(f"{k}={d:.3f}" for k, d in seg_regress)
        )

    for k, ratio in eval_.disparate_impact_challenger.items():
        if ratio < cfg.di_floor:
            blocked.append(f"disparate impact {k} = {ratio:.3f} < floor")
        prev = eval_.disparate_impact_champion.get(k, 1.0)
        if (prev - ratio) > cfg.di_max_drop:
            blocked.append(
                f"disparate impact {k} dropped {prev:.3f} -> {ratio:.3f}"
            )

    if shadow_psi is not None and shadow_psi > cfg.shadow_psi_max:
        blocked.append(f"shadow PSI {shadow_psi:.3f} > "
                       f"max {cfg.shadow_psi_max:.3f}")

    return GateDecision(
        promote=not blocked,
        reasons=reasons,
        blocked_by=blocked,
        notes={"shadow_psi": shadow_psi, "auc_p": auc_p},
    )


def shadow_psi_score(
    p_champion_live: np.ndarray, p_challenger_live: np.ndarray,
) -> float:
    """PSI between live champion and challenger predictions."""
    return psi(p_champion_live, p_challenger_live)


def rollback_check(
    p_champion_baseline: np.ndarray, p_promoted_live: np.ndarray,
    cfg: GateConfig,
) -> tuple[bool, float]:
    val = psi(p_champion_baseline, p_promoted_live)
    return val > cfg.rollback_psi, val
