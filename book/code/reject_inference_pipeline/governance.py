"""Auto-generated governance artifacts for SR 11-7, ECOA, Basel.

Each retrain produces three documents the model risk team consumes:

* sr_117_memo            markdown change memo: motivation, conceptual
                         framework delta, IV diagnostic, fair-lending
                         diff, validator action items.

* ecoa_disparate_impact_diff   structured diff of approval-rate ratios
                               (champion vs challenger) per protected
                               class. Hard-blocks promotion when a
                               class regresses by more than
                               ``max_drop`` or falls below the four-
                               fifths floor.

* basel_ttc_multi_vintage_gate  through-the-cycle anchor check: a
                                challenger must improve PD calibration
                                on at least ``min_vintages`` distinct
                                vintages without regressing on any
                                single vintage by more than the
                                ``vintage_regression_max`` threshold.
                                Auto-promotion is blocked if the
                                challenger fits a single vintage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from .champion_challenger import ChallengerEvaluation, GateDecision
from .propensity import IVDiagnostic
from .outcome import OutcomeArtifact


@dataclass
class TTCResult:
    per_vintage: pd.DataFrame      # columns: vintage, brier_champion,
                                    #          brier_challenger, n
    vintages_improved: int
    vintages_regressed: int
    blocked: bool
    reason: str


def basel_ttc_multi_vintage_gate(
    y: np.ndarray, p_champion: np.ndarray, p_challenger: np.ndarray,
    vintage: pd.Series,
    min_vintages: int = 3,
    vintage_regression_max: float = 0.005,
    min_n_per_vintage: int = 200,
) -> TTCResult:
    """Block auto-promote if the challenger only fits one vintage."""
    rows = []
    for v, idx in vintage.groupby(vintage.values).groups.items():
        idx_arr = np.asarray(idx)
        if idx_arr.size < min_n_per_vintage or y[idx_arr].sum() in (
                0, idx_arr.size):
            continue
        b_champ = float(np.mean((p_champion[idx_arr] - y[idx_arr]) ** 2))
        b_chal = float(np.mean((p_challenger[idx_arr] - y[idx_arr]) ** 2))
        rows.append({
            "vintage": str(v),
            "n": int(idx_arr.size),
            "brier_champion": b_champ,
            "brier_challenger": b_chal,
            "delta_brier": b_chal - b_champ,
            "improved": b_chal < b_champ,
            "regressed_beyond": (b_chal - b_champ) > vintage_regression_max,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return TTCResult(df, 0, 0, True, "no qualifying vintages")
    n_improved = int(df["improved"].sum())
    n_regressed = int(df["regressed_beyond"].sum())
    if n_regressed > 0:
        return TTCResult(
            df, n_improved, n_regressed, True,
            f"{n_regressed} vintage(s) regressed beyond "
            f"{vintage_regression_max:.4f}",
        )
    if n_improved < min_vintages:
        return TTCResult(
            df, n_improved, n_regressed, True,
            f"only {n_improved} vintage(s) improved; require "
            f"{min_vintages} for TTC promotion",
        )
    return TTCResult(df, n_improved, n_regressed, False,
                     "passes TTC multi-vintage gate")


def ecoa_disparate_impact_diff(
    eval_: ChallengerEvaluation,
    floor: float = 0.80, max_drop: float = 0.02,
) -> dict[str, Any]:
    """Structured diff: champion vs challenger by protected class."""
    diff: dict[str, Any] = {"by_class": {}, "blocked": False, "reasons": []}
    for k in (set(eval_.disparate_impact_champion)
              | set(eval_.disparate_impact_challenger)):
        prev = eval_.disparate_impact_champion.get(k, float("nan"))
        new = eval_.disparate_impact_challenger.get(k, float("nan"))
        diff["by_class"][k] = {
            "champion": prev, "challenger": new, "delta": new - prev,
        }
        if not np.isnan(new) and new < floor:
            diff["blocked"] = True
            diff["reasons"].append(f"{k}: challenger {new:.3f} < floor {floor}")
        if not np.isnan(prev) and not np.isnan(new) and (prev - new) > max_drop:
            diff["blocked"] = True
            diff["reasons"].append(
                f"{k}: champion {prev:.3f} -> challenger {new:.3f} "
                f"(drop > {max_drop})"
            )
    return diff


def sr_117_memo(
    *,
    snapshot_date: pd.Timestamp,
    champion_version: str,
    challenger_version: str,
    drift_reason: str,
    propensity_summary: dict[str, Any],
    iv_diagnostic: IVDiagnostic,
    outcome_summary: dict[str, Any],
    gate_decision: GateDecision,
    ecoa_diff: dict[str, Any],
    ttc_result: TTCResult,
    sensitivity_anchor: Optional[OutcomeArtifact] = None,
) -> str:
    """Render an SR 11-7 model-change memo as markdown."""
    lines = [
        f"# Model change memo: reject_inference_pd",
        "",
        f"- **Snapshot date:** {snapshot_date.date()}",
        f"- **Champion version:** {champion_version}",
        f"- **Challenger version:** {challenger_version}",
        f"- **Trigger:** {drift_reason}",
        "",
        "## 1. Conceptual framework",
        "Heckman two-step (sensitivity anchor) plus AIPW (production champion).",
        "Selection-stage features list and outcome-stage features list have not",
        "changed unless explicitly noted in the IV diagnostic below.",
        "",
        "## 2. Stage-1 selection model",
        f"- mode: {propensity_summary['mode']}",
        f"- selection AUC: {propensity_summary.get('selection_auc')}",
        f"- propensity overlap [{propensity_summary.get('p01')}, "
        f"{propensity_summary.get('p99')}]",
        f"- share clipped at boundary: {propensity_summary.get('share_clipped')}",
        "",
        "### Exclusion-restriction recheck",
        f"- threshold p < {iv_diagnostic.z_in_outcome_threshold}",
        "- p-values of Z in outcome equation:",
        *(f"  - `{k}`: p = {v:.4f}, coef = {iv_diagnostic.z_in_outcome_coefs[k]:.4f}"
          for k, v in iv_diagnostic.z_in_outcome_pvalues.items()),
        f"- first-stage F-stat: {iv_diagnostic.f_stat_first_stage}",
        f"- IV blocked: {iv_diagnostic.iv_blocked} "
        f"({list(iv_diagnostic.blocked_columns)})",
        f"- weak IV: {iv_diagnostic.weak_iv_blocked}",
        "",
        "## 3. Outcome model",
        f"- challenger PD through-the-door: {outcome_summary['pd_through_door']:.4f}",
        f"- champion  PD through-the-door: {outcome_summary['pd_through_door_prev']:.4f}",
        f"- challenger PD funded: {outcome_summary['pd_funded']:.4f}",
        f"- challenger PD declined: {outcome_summary['pd_declined']:.4f}",
        "",
        "## 4. Performance gate",
        "Gate decision: **{}**".format(
            "PROMOTE" if gate_decision.promote else "BLOCK"),
        "",
        "Pass conditions met:",
        *(f"- {r}" for r in gate_decision.reasons),
        "",
        "Blockers:",
        *(f"- {b}" for b in (gate_decision.blocked_by or ["(none)"])),
        "",
        "## 5. Fair lending (ECOA)",
        f"- floor: {0.80}, max_drop: {0.02}",
        f"- blocked: {ecoa_diff['blocked']}",
        *(f"- {k}: champion = {v['champion']:.3f}, challenger = "
          f"{v['challenger']:.3f}, delta = {v['delta']:+.3f}"
          for k, v in ecoa_diff["by_class"].items()),
        "",
        "## 6. Basel TTC multi-vintage gate",
        f"- vintages improved: {ttc_result.vintages_improved}",
        f"- vintages regressed beyond threshold: {ttc_result.vintages_regressed}",
        f"- blocked: {ttc_result.blocked}",
        f"- reason: {ttc_result.reason}",
        "",
        "## 7. Sensitivity anchor (Heckman)",
    ]
    if sensitivity_anchor is not None:
        lines.append(
            f"- Heckman PD through-the-door: "
            f"{sensitivity_anchor.pd_through_door:.4f}"
        )
        lines.append(
            f"- AIPW vs Heckman gap (challenger): "
            f"{outcome_summary['pd_through_door'] - sensitivity_anchor.pd_through_door:+.4f}"
        )
    else:
        lines.append("- not run this cycle")
    lines += [
        "",
        "## 8. Validator action items",
        "- review IV diagnostic and any blocked columns",
        "- confirm Heckman/AIPW gap is within tolerance documented in the model card",
        "- review per-segment AUC table for any newly-regressed segment",
        "- confirm shadow-mode PSI on the live stream is below threshold "
        "before flipping traffic",
        "",
    ]
    return "\n".join(lines)
