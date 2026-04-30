"""End-to-end smoke test on a synthetic informative-censoring cohort.

Generates a Weibull default-time cohort with prepay competing risk
correlated with a hidden score Z, runs the full diagnostics pipeline,
and prints the artifact summary. Intended to be invoked once after
package install:

    python -m survival_diagnostics._smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from survival_diagnostics import (
    DiagnosticsConfig,
    IpcwConfig,
    TippingConfig,
    run_diagnostics,
    validate_cohort,
)


def synthesise(n: int = 4000, term: int = 36, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=n)
    age = rng.uniform(20, 60, size=n)
    util = rng.beta(2, 5, size=n)

    lam_T = 50.0
    lam_P = 60.0
    k_w = 1.4
    alpha = 0.6

    T_lat = lam_T * np.exp(-alpha * Z) * rng.weibull(k_w, size=n)
    P_lat = lam_P * np.exp(+alpha * Z) * rng.weibull(k_w, size=n) * 0.6
    A_lat = np.full(n, float(term))

    times = np.column_stack([T_lat, P_lat, A_lat])
    which = np.argmin(times, axis=1)
    Y = times[np.arange(n), which]

    cause = np.where(which == 0, "default",
             np.where(which == 1, "prepay", "admin"))
    event = (cause == "default").astype(int)

    df = pd.DataFrame({
        "loan_id": [f"L{i:06d}" for i in range(n)],
        "duration": Y,
        "event": event,
        "cause": cause,
        "vintage": rng.choice(["2023-Q1", "2023-Q2", "2023-Q3"], size=n),
        "Z": Z,
        "age": age,
        "util": util,
    })
    return df


def main() -> None:
    df = synthesise()
    cohort = validate_cohort(df, covariate_cols=["Z", "age", "util"], term_months=36)
    clean_mask = (df["vintage"] == "2023-Q3").to_numpy()

    cfg = DiagnosticsConfig(
        horizons_months=(12, 24, 36),
        ipcw=IpcwConfig(censoring_cause="prepay", cap_quantile=0.99),
        tipping=TippingConfig(),
        fit_fine_gray=True,
        fit_aalen_johansen=True,
        clean_cohort_mask=clean_mask,
    )
    artifact = run_diagnostics(cohort, cfg)

    print("=== survival_diagnostics smoke test ===")
    print(f"n                = {artifact.cohort['n']}")
    print(f"cause counts     = {artifact.cohort['cause_counts']}")
    print(f"naive  PD@12m    = {artifact.pd_at_horizons['naive']['pd_12m']:.4f}")
    if "ipcw" in artifact.pd_at_horizons:
        print(f"ipcw   PD@12m    = {artifact.pd_at_horizons['ipcw']['pd_12m']:.4f}")
    if "aalen_johansen" in artifact.pd_at_horizons:
        aj = artifact.pd_at_horizons["aalen_johansen"]
        print(f"AJ CIF default@12m = {aj.get('pd_12m', float('nan')):.4f}")
    lt = artifact.pd_lifetime
    print(f"lifetime  naive  = {lt['naive']:.4f}")
    print(f"lifetime  ipcw   = {lt['ipcw']}")
    if lt.get("tipping"):
        t = lt["tipping"]
        print(f"tipping band PD  = [{t['decision_band_min']:.4f}, {t['decision_band_max']:.4f}]")
    if artifact.holdout:
        h = artifact.holdout
        print(f"clean cohort PD  = {h['pd_clean']:.4f}  vs full {h['pd_full']:.4f}")
    print(f"any imbalance    = {artifact.cause_overlap['any_imbalanced']}")
    print(f"ipcw cap value   = {artifact.ipcw_weights['cap_value']:.2f}  "
          f"share>cap = {artifact.ipcw_weights['cap_share']:.4f}")
    print(f"errors           = {artifact.errors}")


if __name__ == "__main__":
    main()
