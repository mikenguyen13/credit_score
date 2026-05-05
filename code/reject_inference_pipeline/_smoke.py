"""End-to-end smoke for the reject-inference retrain pipeline.

Generates a synthetic three-vintage applicant base with a known
selection mechanism, runs the full retrain (observable + unobservable
+ alt_data flavours), the drift trigger, and the gated promote, and
prints the SR 11-7 memo summary.

    python -m reject_inference_pipeline._smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reject_inference_pipeline import (
    PolicyVersion, PolicyVersionTable,
    validate_applicant_snapshot, validate_bureau_outcomes,
    join_snapshot_outcomes,
    DriftThresholds, DriftTrigger, compute_drift,
    RetrainConfig,
    retrain_observable, retrain_unobservable, retrain_alt_data,
    gated_promote, write_artifact,
    counterfactual_pd, reliability_index,
    render_card, RejectInferenceCard,
)


SEED = 20260504
N_PER_VINTAGE = 1500
RHO = 0.6


_VINTAGE_BASE = {
    "2024-Q1": pd.Timestamp("2024-02-15"),
    "2024-Q2": pd.Timestamp("2024-05-15"),
    "2024-Q3": pd.Timestamp("2024-08-15"),
}


def _synthesise_vintage(rng: np.random.Generator, vintage: str,
                         n: int = N_PER_VINTAGE) -> pd.DataFrame:
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    z = rng.standard_normal(n)
    u = rng.standard_normal(n)
    v = RHO * u + np.sqrt(1 - RHO ** 2) * rng.standard_normal(n)
    y_latent = -0.4 + 0.6 * x1 + 0.4 * x2 + u
    y = (y_latent > 0).astype(int)
    sel_lin = 0.2 + 0.5 * x1 + 0.3 * x2 + 0.6 * z + v
    s = (sel_lin > 0).astype(int)
    pi_logged = (1.0 / (1.0 + np.exp(-(0.2 + 0.5 * x1 + 0.3 * x2 + 0.6 * z))))
    as_of = _VINTAGE_BASE[vintage] + pd.to_timedelta(
        rng.integers(0, 60, size=n), unit="D")
    return pd.DataFrame({
        "applicant_id": [f"A{vintage}{i:05d}" for i in range(n)],
        "as_of": as_of,
        "x1": x1, "x2": x2, "z": z,
        "s": s,
        "policy_version_id": "P_2026_v1",
        "pi_logged": pi_logged,
        "vintage": vintage,
        "segment": rng.choice(["digital", "branch"], size=n),
        "_y_truth": y,
    })


def main() -> None:
    rng = np.random.default_rng(SEED)
    parts = [_synthesise_vintage(rng, v) for v in
             ("2024-Q1", "2024-Q2", "2024-Q3")]
    df = pd.concat(parts, ignore_index=True)

    apps = validate_applicant_snapshot(
        df, feature_cols=["x1", "x2"], iv_cols=["z"],
        require_pi_logged=True,
    )

    funded_idx = np.flatnonzero(apps.s == 1)
    bureau_df = pd.DataFrame({
        "applicant_id": df["applicant_id"].iloc[funded_idx].values,
        "observed_at": (df["as_of"].iloc[funded_idx] +
                        pd.DateOffset(months=18)).values,
        "y": df["_y_truth"].iloc[funded_idx].values,
    })
    outcomes = validate_bureau_outcomes(bureau_df, y_definition_id="dpd90_18m")

    snapshot_date = pd.Timestamp("2026-05-01")
    joined = join_snapshot_outcomes(apps, outcomes, snapshot_date,
                                     performance_window_months=18)

    cfg = RetrainConfig(
        snapshot_date=snapshot_date,
        performance_window_months=18,
        bootstrap_B=0,
        cluster_key_col="vintage",
        aipw_n_splits=5,
        run_aipcw=False,
    )
    policies = PolicyVersionTable(rows=(
        PolicyVersion(
            policy_version_id="P_2026_v1",
            effective_from=pd.Timestamp("2024-01-01"),
            effective_to=None,
            propensity_mode="observable",
            iv_columns=("z",),
            label_definition_id="dpd90_18m",
            cutoff=0.0, override_quota=0.05,
        ),
    ))
    p_active = policies.active(snapshot_date)

    art_obs = retrain_observable(joined, cfg, p_active)

    p_unobs = PolicyVersion(
        policy_version_id="P_2026_v1u",
        effective_from=pd.Timestamp("2024-01-01"),
        effective_to=None,
        propensity_mode="unobservable",
        iv_columns=("z",),
        label_definition_id="dpd90_18m",
    )
    art_unobs = retrain_unobservable(joined, cfg, p_unobs)

    p_alt = PolicyVersion(
        policy_version_id="P_2026_v1alt",
        effective_from=pd.Timestamp("2024-01-01"),
        effective_to=None,
        propensity_mode="alt_data",
        iv_columns=("z",),
        label_definition_id="dpd90_18m",
    )
    lender_id = pd.Series(rng.choice(["bankA", "bankB", "bankC"], size=apps.n))
    art_alt = retrain_alt_data(joined, lender_id, cfg, p_alt)

    print("=== reject_inference_pipeline smoke ===")
    print(f"observable    | n={art_obs.n_train}, funded={art_obs.n_funded_train}, "
          f"matured={art_obs.n_matured}")
    print(f"  AIPW pd_ttd   = {art_obs.outcome_aipw.pd_through_door:.4f}")
    print(f"  Heckman pd_ttd= {art_obs.outcome_heckman.pd_through_door:.4f}")
    print(f"  IV blocked    = {art_obs.iv_diagnostic.iv_blocked}")
    print(f"  errors        = {art_obs.errors}")

    print(f"unobservable  | n={art_unobs.n_train}")
    print(f"  AIPW pd_ttd   = {art_unobs.outcome_aipw.pd_through_door:.4f}")
    print(f"  Heckman pd_ttd= {art_unobs.outcome_heckman.pd_through_door:.4f}")
    print(f"  IV blocked    = {art_unobs.iv_diagnostic.iv_blocked}")

    print(f"alt_data      | lenders={len(art_alt.propensity_per_lender.per_lender)}")
    print(f"  cold-start    = {art_alt.propensity_per_lender.cold_start_lenders}")
    if art_alt.outcome_aipw is not None:
        print(f"  AIPW pd_ttd   = {art_alt.outcome_aipw.pd_through_door:.4f}")

    holdout_idx = np.flatnonzero(joined.matured_mask)[: int(0.2 * joined.matured_mask.sum())]
    y_h = joined.y_full[holdout_idx]
    X_h = apps.X.to_numpy()[holdout_idx]
    pd_champ = art_obs.champion_pd(X_h)
    pd_chal = art_unobs.champion_pd(X_h)
    vintage_h = pd.Series(apps.vintage.values[holdout_idx])
    seg_h = pd.Series(apps.segment.values[holdout_idx])
    protected_h = pd.Series(rng.choice(["A", "B"], size=holdout_idx.size))

    decision = gated_promote(
        snapshot_date=snapshot_date,
        challenger=art_unobs,
        champion_pd_holdout=pd_champ,
        challenger_pd_holdout=pd_chal,
        y_holdout=y_h,
        vintage_holdout=vintage_h,
        segment_holdout=seg_h,
        protected_holdout=protected_h,
        threshold=0.5,
        reference_group="A",
        drift_reason="smoke_run",
        sensitivity_anchor=art_unobs.outcome_heckman,
    )
    print(f"promote       = {decision.promote}")
    print(f"blocked_by    = {decision.gate_decision.blocked_by}")
    print(f"ttc_blocked   = {decision.ttc_blocked}")
    print(f"ecoa_blocked  = {decision.ecoa_blocked}")

    drift_cfg = DriftThresholds()
    cur_idx = (apps.vintage == "2024-Q3").to_numpy()
    base_idx = ~cur_idx
    drift_report = compute_drift(
        train_features=apps.X.iloc[base_idx],
        current_features=apps.X.iloc[cur_idx],
        train_propensity=art_obs.propensity.pi[base_idx],
        current_propensity=art_obs.propensity.pi[cur_idx],
        train_accept_rate=float(apps.s[base_idx].mean()),
        current_accept_rate=float(apps.s[cur_idx].mean()),
        train_imr=art_obs.propensity.imr[base_idx],
        current_imr=art_obs.propensity.imr[cur_idx],
        train_funded_default_rate=0.18,
        current_funded_default_rate=0.20,
        thresholds=drift_cfg,
    )
    trigger = DriftTrigger(thresholds=drift_cfg, min_consecutive=2)
    for _ in range(2):
        trigger.observe(drift_report)
    fire, why = trigger.should_retrain()
    print(f"drift kind    = {drift_report.classified}; trigger fires = "
          f"{fire} ({why})")

    pi_log = apps.pi_logged
    pi_new = np.clip(pi_log * 1.10, 1e-3, 1 - 1e-3)
    funded_mask = apps.s == 1
    matured = joined.matured_mask
    fm = funded_mask & matured
    cf = counterfactual_pd(pi_log[fm], pi_new[fm],
                           apps.s[fm], joined.y_full[fm], np.ones(fm.sum(), dtype=bool),
                           weight_cap=10.0)
    rel = reliability_index(cf, raw_funded_n=int(fm.sum()))
    print(f"CFRM PD(new)  = {cf.pd_under_new_policy:.4f}, "
          f"ESS share = {rel['ess_share']:.3f}, trustworthy={rel['trustworthy']}")

    artifact_path = write_artifact(art_obs, "/tmp/ri_smoke/observable.json")
    print(f"artifact at   = {artifact_path}")
    print("--- model card preview ---")
    print(render_card(RejectInferenceCard(version="1.0.0"))[:400])


if __name__ == "__main__":
    main()
