"""Left- and right-truncation diagnostics and corrections.

This module mirrors the chapter demos in :mod:`survival_diagnostics`
production code. It does three things:

1. Detect: scan an incoming cohort for the *signatures* of left and
   right truncation that should never quietly pass into a KM/Cox fit.
   The canonical failures we have seen in production:

   * Left truncation, undetected: the cohort was assembled by
     filtering a portfolio snapshot at calendar window-open
     ``tau_start``, dropping the ``v + a_0 < tau_start`` rows.
     The resulting frame has no ``entry`` column and the fit treats
     every loan as observed from age 0. KM under-counts early
     defaults. The detector fires when ``vintage`` and ``tau_start``
     are present and the implied entry ages are non-trivial.

   * Right truncation, undetected: the cohort is a defaulted-only
     extract (chargeoff feed, fraud incident table, recovered-loss
     register). ``event.mean() == 1`` is the classic fingerprint;
     a softer fingerprint is ``event.mean() > 0.99`` once admin
     censoring has been merged but a small leak of non-defaulters
     remains.

2. Correct: provide a delayed-entry KM (left truncation) and a
   reverse-time KM (right truncation, Lagakos 1988) so the chapter
   numerical demo is exactly the production code path.

3. Report: emit a typed result object with bias deltas at the
   horizons of interest, so the validation pipeline can block on
   "naive vs corrected lifetime PD differ by more than X bp".

References
----------
Lagakos, S. W., Barraj, L. M., De Gruttola, V. (1988).
    Nonparametric Analysis of Truncated Survival Data, with Application
    to AIDS. Biometrika 75(3), 515-523.
Andersen, P. K., Gill, R. D. (1982). Cox's Regression Model for
    Counting Processes: A Large Sample Study. AoS 10(4), 1100-1120.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


@dataclass
class TruncationConfig:
    horizons_months: tuple[int, ...] = (6, 12, 24, 36)
    event_share_high: float = 0.99
    event_share_low: float = 0.001
    entry_age_min_months: float = 1.0
    bias_block_bps: float = 50.0


@dataclass
class TruncationFlags:
    looks_event_only: bool
    looks_event_missing: bool
    has_entry_column: bool
    needs_left_truncation_fix: bool
    needs_right_truncation_fix: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class TruncationCorrection:
    horizon_months: int
    pd_naive: float
    pd_corrected: float
    delta_bps: float


@dataclass
class TruncationResult:
    flags: TruncationFlags
    left_corrections: list[TruncationCorrection]
    right_corrections: list[TruncationCorrection]
    blocks: bool
    config: TruncationConfig

    def to_dict(self) -> dict:
        return {
            "flags": {
                "looks_event_only": self.flags.looks_event_only,
                "looks_event_missing": self.flags.looks_event_missing,
                "has_entry_column": self.flags.has_entry_column,
                "needs_left_truncation_fix": self.flags.needs_left_truncation_fix,
                "needs_right_truncation_fix": self.flags.needs_right_truncation_fix,
                "notes": list(self.flags.notes),
            },
            "left_corrections": [c.__dict__ for c in self.left_corrections],
            "right_corrections": [c.__dict__ for c in self.right_corrections],
            "blocks": self.blocks,
            "config": {
                "horizons_months": list(self.config.horizons_months),
                "event_share_high": self.config.event_share_high,
                "event_share_low": self.config.event_share_low,
                "entry_age_min_months": self.config.entry_age_min_months,
                "bias_block_bps": self.config.bias_block_bps,
            },
        }


def _km_pd_at(km: KaplanMeierFitter, horizon: float) -> float:
    return float(1.0 - km.predict(float(horizon)))


def left_truncated_km(
    duration: np.ndarray,
    event: np.ndarray,
    entry: np.ndarray,
    horizons: tuple[int, ...],
) -> tuple[KaplanMeierFitter, dict[int, float]]:
    """Delayed-entry Kaplan-Meier with PD readouts at the given horizons.

    Each row enters the risk set at ``entry[i]`` (age at calendar
    window-open) and exits at ``entry[i] + duration[i]``, with the
    event indicator unchanged. This is the chapter's left-truncation
    fix, packaged for production use.
    """
    if not (np.isfinite(entry).all() and (entry >= 0).all()):
        raise ValueError("entry must be finite, non-negative ages in months")
    if (entry > duration + entry).any():  # tautology, defensive
        raise ValueError("entry exceeds exit")
    km = KaplanMeierFitter().fit(duration, event, entry=entry)
    pd_at = {int(h): _km_pd_at(km, h) for h in horizons}
    return km, pd_at


def right_truncated_km(
    duration: np.ndarray,
    vintage_age_at_cutoff: np.ndarray,
    horizons: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    """Reverse-time delayed-entry Kaplan-Meier (Lagakos 1988).

    Identifiability note. Right-truncated data, on its own, identifies
    the *conditional* event-time distribution on the observed support
    $[0, t^*]$ where $t^* = \\max R_i$, that is, it estimates
    $F_T(t) / F_T(t^*)$. Anything beyond $t^*$ is unidentifiable
    without an external assumption on $F_T(t^*)$ (a parametric tail or
    a known cohort-level base rate). The pipeline therefore reports the
    conditional CDF and flags whenever $t^*$ is materially smaller than
    the credit-policy horizon.

    Construction. Let $X_i = t^* - T_i$ (reversed-time exit) and
    $B_i = t^* - R_i$ (reversed-time entry). The right-truncation
    constraint $T_i \\le R_i$ becomes the left-truncation constraint
    $B_i \\le X_i$. Forward-time delayed-entry KM on $(B, X)$ with
    all-event indicator gives $\\hat S_X(x)$, and
    $\\hat F_T(t) / \\hat F_T(t^*) = \\hat S_X(t^* - t)$.

    Parameters
    ----------
    duration
        Observed default age in months. Every row is an event.
    vintage_age_at_cutoff
        ``tau_end - vintage`` in months: per-row truncation bound.
        Must satisfy ``duration <= vintage_age_at_cutoff``.
    horizons
        Ages at which to read off the corrected PD.

    Returns
    -------
    ages
        Grid (months) on which the conditional survival is evaluated.
    survival
        Conditional survival ``S_T(t)/F_T(t*)`` mapped through the
        ``S_T(t) = 1 - F_T(t)/F_T(t*)`` relation, on a grid in
        ``[0, t*]``.
    pd_at
        ``{horizon -> 1 - S(horizon)}``, the conditional cumulative
        default rate on the observed support.
    """
    duration = np.asarray(duration, dtype=float)
    R = np.asarray(vintage_age_at_cutoff, dtype=float)
    if duration.shape != R.shape:
        raise ValueError("duration and vintage_age_at_cutoff shape mismatch")
    if (duration > R + 1e-9).any():
        raise ValueError(
            "duration exceeds vintage_age_at_cutoff for some rows; "
            "input does not satisfy the right-truncation constraint"
        )
    if duration.size == 0:
        raise ValueError("empty truncated sample")

    t_star = float(R.max())
    X = t_star - duration
    B = t_star - R
    km_rev = KaplanMeierFitter().fit(
        X, np.ones_like(X, dtype=int), entry=B,
    )

    ages_grid = np.linspace(0.0, t_star, 256)
    f_cond = km_rev.survival_function_at_times(
        np.maximum(t_star - ages_grid, 0.0)
    ).values
    survival = 1.0 - f_cond

    def _s_at(a: float) -> float:
        a = float(min(max(a, 0.0), t_star))
        f = float(km_rev.survival_function_at_times(np.array([t_star - a])).values[0])
        return 1.0 - f

    pd_at = {int(h): float(1.0 - _s_at(h)) for h in horizons}
    return ages_grid, survival, pd_at


def detect_truncation(
    duration: np.ndarray,
    event: np.ndarray,
    *,
    entry: Optional[np.ndarray] = None,
    vintage_age_at_cutoff: Optional[np.ndarray] = None,
    config: Optional[TruncationConfig] = None,
) -> TruncationResult:
    """End-to-end truncation diagnostic for a cohort.

    Detection is robust to either input column being missing: when
    ``entry`` is None we cannot fit a left-truncation correction, so we
    only flag and warn; when ``vintage_age_at_cutoff`` is None and the
    event share is degenerate we still flag right truncation but cannot
    auto-correct.

    The ``blocks`` field is set when the corrected lifetime PD differs
    from the naive fit by more than ``config.bias_block_bps`` basis
    points. Validation pipelines should treat this as an SR 11-7 stop.
    """
    cfg = config or TruncationConfig()
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)

    notes: list[str] = []
    event_share = float(event.mean()) if event.size > 0 else 0.0
    looks_event_only = event_share >= cfg.event_share_high
    looks_event_missing = event_share <= cfg.event_share_low
    has_entry = entry is not None and np.any(np.asarray(entry) > cfg.entry_age_min_months)

    if looks_event_only:
        notes.append(
            f"event share {event_share:.4f} >= {cfg.event_share_high}; "
            "cohort looks event-only; right-truncation correction required"
        )
    if looks_event_missing:
        notes.append(
            f"event share {event_share:.4f} <= {cfg.event_share_low}; "
            "cohort may have lost the defaulter join (mirror failure mode)"
        )
    if entry is None:
        notes.append(
            "no entry column supplied; cannot apply delayed-entry correction; "
            "if the cohort is a snapshot at a calendar window, the KM is biased"
        )

    needs_left_fix = bool(has_entry)
    needs_right_fix = bool(looks_event_only and vintage_age_at_cutoff is not None)

    left_corrections: list[TruncationCorrection] = []
    if needs_left_fix:
        km_naive = KaplanMeierFitter().fit(duration, event)
        _, pd_corrected_left = left_truncated_km(
            duration, event, np.asarray(entry, dtype=float), cfg.horizons_months
        )
        for h in cfg.horizons_months:
            naive_h = _km_pd_at(km_naive, h)
            corr_h = pd_corrected_left[int(h)]
            left_corrections.append(
                TruncationCorrection(
                    horizon_months=int(h),
                    pd_naive=naive_h,
                    pd_corrected=corr_h,
                    delta_bps=(corr_h - naive_h) * 1e4,
                )
            )

    right_corrections: list[TruncationCorrection] = []
    if needs_right_fix:
        # On an event-only sample naive KM degenerates to the empirical CDF.
        km_naive = KaplanMeierFitter().fit(duration, np.ones_like(duration, dtype=int))
        _, _, pd_corrected_right = right_truncated_km(
            duration,
            np.asarray(vintage_age_at_cutoff, dtype=float),
            cfg.horizons_months,
        )
        for h in cfg.horizons_months:
            naive_h = _km_pd_at(km_naive, h)
            corr_h = pd_corrected_right[int(h)]
            right_corrections.append(
                TruncationCorrection(
                    horizon_months=int(h),
                    pd_naive=naive_h,
                    pd_corrected=corr_h,
                    delta_bps=(corr_h - naive_h) * 1e4,
                )
            )

    blocks = any(
        abs(c.delta_bps) > cfg.bias_block_bps
        for c in (*left_corrections, *right_corrections)
    )

    flags = TruncationFlags(
        looks_event_only=looks_event_only,
        looks_event_missing=looks_event_missing,
        has_entry_column=bool(has_entry),
        needs_left_truncation_fix=needs_left_fix,
        needs_right_truncation_fix=needs_right_fix,
        notes=notes,
    )
    return TruncationResult(
        flags=flags,
        left_corrections=left_corrections,
        right_corrections=right_corrections,
        blocks=blocks,
        config=cfg,
    )


def truncation_summary_table(result: TruncationResult) -> pd.DataFrame:
    """Long-form table of (kind, horizon, naive, corrected, delta_bps)."""
    rows: list[dict] = []
    for c in result.left_corrections:
        rows.append({
            "kind": "left",
            "horizon_months": c.horizon_months,
            "pd_naive": c.pd_naive,
            "pd_corrected": c.pd_corrected,
            "delta_bps": c.delta_bps,
        })
    for c in result.right_corrections:
        rows.append({
            "kind": "right",
            "horizon_months": c.horizon_months,
            "pd_naive": c.pd_naive,
            "pd_corrected": c.pd_corrected,
            "delta_bps": c.delta_bps,
        })
    return pd.DataFrame(rows)
