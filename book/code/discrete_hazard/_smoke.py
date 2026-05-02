"""End-to-end smoke test for the discrete_hazard package.

Generates a Shumway-style synthetic vintage panel with a borrower
covariate, an AR(1) macro index joined on calendar month, and right-
censoring at the observation date. Runs the full pipeline (validated
panel -> fit -> validation pack -> persisted artifact) and prints the
artifact summary. Intended to be invoked once after package install:

    python -m discrete_hazard._smoke
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from discrete_hazard import (
    Ar1Process,
    ShumwayConfig,
    add_calendar_covariates,
    render_card,
    run_shumway,
    validate_panel,
)
from discrete_hazard.model_card import ShumwayModelCard


def synthesise_panel(
    n_loans: int = 6000,
    t_max: int = 36,
    n_vintages: int = 24,
    seed: int = 20260428,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs_horizon = n_vintages + t_max

    z = rng.normal(size=n_loans)
    vintage = rng.integers(0, n_vintages, size=n_loans)

    u = np.zeros(obs_horizon)
    for v in range(1, obs_horizon):
        u[v] = 0.85 * u[v - 1] + 0.25 * rng.normal()
    u += 0.6 * np.exp(-0.5 * ((np.arange(obs_horizon) - 18) / 3.0) ** 2)

    rows: list[tuple] = []
    for i in range(n_loans):
        v_i = int(vintage[i])
        for t in range(1, t_max + 1):
            cal = v_i + t - 1
            if cal >= obs_horizon:
                rows.append((i, t, 0, z[i], v_i, cal))
                break
            h = expit(-5.20 + 0.70 * z[i] + 0.025 * t + 0.40 * u[cal])
            d = int(rng.random() < h)
            rows.append((i, t, d, z[i], v_i, cal))
            if d:
                break

    df = pd.DataFrame(rows, columns=["loan_id", "age", "default", "z", "vintage", "cal_month"])
    df["loan_id"] = df["loan_id"].astype(int)
    return df, u


def main() -> None:
    df_raw, u_path = synthesise_panel()
    df = add_calendar_covariates(df_raw, {"u": u_path})

    panel = validate_panel(df, covariate_cols=["z", "u"])

    n_vintages = int(panel.vintage.max()) + 1
    holdout = list(range(n_vintages - 6, n_vintages))

    cfg = ShumwayConfig(
        covariate_cols=["z", "u"],
        holdout_vintages=holdout,
        horizons_months=(12, 24, 36),
        bootstrap_n=80,
        macro_paths={"u": u_path},
        forward_macro="u",
        forward_n_paths=400,
        representative_covariates={"z": 0.0, "u": float(u_path[holdout[0]])},
        representative_vintage=holdout[0],
        representative_horizon=36,
    )

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "shumway.pkl"
        artifact, pack = run_shumway(panel, cfg, artifact_path=out)

        print("=== discrete_hazard smoke test ===")
        print(f"loans = {panel.n_loans:,}  rows = {panel.n_rows:,}  events = {panel.n_events:,}")
        print(f"feature_order = {artifact.feature_order}")
        print(f"param_hash    = {artifact.metadata['param_hash']}")
        print(f"loglike       = {artifact.metadata['loglike']:.2f}   "
              f"AIC = {artifact.metadata['aic']:.1f}   "
              f"BIC = {artifact.metadata['bic']:.1f}")
        print()
        for hs in pack.horizon_scores:
            print(f"horizon {hs['horizon_months']:>3}m  "
                  f"n={hs['n']:,}  events={hs['n_events']:,}  "
                  f"AUC={hs['auc']:.4f}  Brier={hs['brier']:.4f}")

        if pack.term_structure:
            ts = pack.term_structure
            print(f"\nterm-structure 36m mean PD = {ts['mean'][-1]:.4f}  "
                  f"5/95 = [{ts['lo'][-1]:.4f}, {ts['hi'][-1]:.4f}]  "
                  f"(n_boot={ts['n_boot']})")

        if pack.forward_distribution:
            fd = pack.forward_distribution
            print(f"forward-dist 36m mean PD   = {fd['mean_cum_pd'][-1]:.4f}  "
                  f"5/95 = [{fd['q05_cum_pd'][-1]:.4f}, {fd['q95_cum_pd'][-1]:.4f}]  "
                  f"(phi={fd['phi']:+.3f}, sigma={fd['sigma']:.3f}, n_paths={fd['n_paths']})")

        print(f"\nerrors = {pack.errors}")
        print(f"artifact at {out}  ({out.stat().st_size / 1024:.1f} KB)")
        print(f"pipeline pack at {out.with_suffix('.pipeline.json').name}")
        print()
        print(render_card(ShumwayModelCard()).split('\n')[0])


if __name__ == "__main__":
    main()
