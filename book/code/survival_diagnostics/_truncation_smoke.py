"""Smoke test for the truncation diagnostic.

Reproduces the chapter's left- and right-truncation numerical demos
through the production code path (``detect_truncation`` +
``left_truncated_km`` + ``right_truncated_km``) and prints the
PD comparison table.

Invoke once after package install::

    python -m survival_diagnostics._truncation_smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from survival_diagnostics import (
    TruncationConfig,
    detect_truncation,
    truncation_summary_table,
)


def left_truncation_demo(seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    n = 6000
    T = 80.0 * rng.weibull(1.4, size=n)
    M = 60
    a0 = rng.uniform(0, 24, size=n)
    in_window = T > a0
    Y = np.minimum(T[in_window], M)
    E = (T[in_window] <= M).astype(int)
    entry = a0[in_window]

    cfg = TruncationConfig(
        horizons_months=(6, 12, 24, 36),
        bias_block_bps=50.0,
    )
    res = detect_truncation(Y, E, entry=entry, config=cfg)

    print("=== left-truncation smoke ===")
    print(f"flags = {res.flags}")
    print(truncation_summary_table(res).round(4).to_string(index=False))
    print(f"blocks = {res.blocks}")
    print()


def right_truncation_demo(seed: int = 11) -> None:
    rng = np.random.default_rng(seed)
    n = 8000
    v = rng.uniform(0, 24, size=n)
    T = 80.0 * rng.weibull(1.4, size=n)
    tau_end = 36.0
    keep = (v + T) <= tau_end

    Y = T[keep]
    E = np.ones_like(Y, dtype=int)
    cutoff = tau_end - v[keep]

    cfg = TruncationConfig(
        horizons_months=(6, 12, 24),
        bias_block_bps=50.0,
    )
    res = detect_truncation(Y, E, vintage_age_at_cutoff=cutoff, config=cfg)

    print("=== right-truncation smoke ===")
    print(f"flags = {res.flags}")
    print(truncation_summary_table(res).round(4).to_string(index=False))
    print(f"blocks = {res.blocks}")


def main() -> None:
    left_truncation_demo()
    right_truncation_demo()


if __name__ == "__main__":
    main()
