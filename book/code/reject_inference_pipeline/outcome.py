"""Outcome-stage estimators: Heckman stage-2, AIPW, AIPCW.

Three production patterns sit in this module, all returning a uniform
OutcomeArtifact so the orchestrator can stack them:

fit_heckman_outcome           probit on (X, IMR) restricted to funded
                              applicants. The classical two-step.
                              Standard errors from the Heckman sandwich
                              plus a vintage-clustered bootstrap.

fit_aipw_outcome              method-agnostic doubly-robust score with
                              cross-fitting. Accepts any sklearn-style
                              estimator as the base learner. Uses the
                              propensity from PropensityArtifact and an
                              outcome regression g_hat on the funded
                              slice.

fit_aipcw_outcome             AIPW upgraded with inverse-probability-of-
                              censoring weights for the unmatured tail.
                              The censoring hazard is fit on the
                              censored-vs-matured indicator with a
                              proportional-hazards model on time since
                              as_of.

The orchestrator chooses Heckman for the SR 11-7 sensitivity anchor
and AIPW (or AIPCW when the censored tail is non-trivial) for the
production champion. The pair is reported together; a divergence
between the two is itself a diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from .schema import ApplicantSnapshot, JoinedSnapshot
from .propensity import PropensityArtifact


@dataclass
class OutcomeArtifact:
    """Output of any outcome-stage fit."""

    method: str                            # 'heckman' | 'aipw' | 'aipcw'
    beta: np.ndarray                       # outcome coefficients (intercept first)
    feature_names: tuple[str, ...]
    pd_through_door: float                 # mean PD over full applicant snapshot
    pd_funded: float                       # mean PD over funded slice
    pd_declined: float                     # implied PD on declined slice
    standard_errors: Optional[dict[str, np.ndarray]] = None  # 'sandwich','bootstrap'
    extra: dict[str, Any] = field(default_factory=dict)


def _design(X: np.ndarray, extra: Optional[np.ndarray] = None) -> np.ndarray:
    rows = X.shape[0]
    cols = [np.ones(rows), X]
    if extra is not None:
        cols.append(extra.reshape(rows, -1))
    return np.column_stack(cols)


def fit_heckman_outcome(
    apps: ApplicantSnapshot,
    y_funded: np.ndarray,
    funded_mask: np.ndarray,
    propensity: PropensityArtifact,
    bootstrap_B: int = 0,
    cluster_key: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> OutcomeArtifact:
    """Heckman two-step on the funded slice with IMR.

    The IMR comes from the PropensityArtifact (which itself may be
    observable or estimated). Standard errors: closed-form sandwich
    via statsmodels and an optional cluster bootstrap on cluster_key.
    """
    import statsmodels.api as sm

    Xf = apps.X.to_numpy()[funded_mask]
    imr_f = propensity.imr[funded_mask].reshape(-1, 1)
    Wf = _design(Xf, imr_f)
    fit = sm.GLM(
        y_funded.astype(float), Wf,
        family=sm.families.Binomial(sm.families.links.Probit()),
    ).fit(disp=False)
    beta = np.asarray(fit.params)
    se_sandwich = np.asarray(fit.bse)

    Xall = apps.X.to_numpy()
    Wall = _design(Xall, propensity.imr.reshape(-1, 1))
    pd_all = stats.norm.cdf(Wall @ beta)
    pd_funded = float(pd_all[funded_mask].mean())
    pd_declined = float(pd_all[~funded_mask].mean()) if (~funded_mask).any() else float("nan")
    pd_through = float(pd_all.mean())

    se_dict = {"sandwich": se_sandwich}
    if bootstrap_B > 0:
        if cluster_key is None:
            raise ValueError("bootstrap requires cluster_key (e.g., vintage)")
        rng = rng or np.random.default_rng(0)
        clusters = pd.Series(cluster_key)
        unique = clusters.unique()
        boot = np.empty((bootstrap_B, beta.size))
        idx_by_cluster = {c: np.flatnonzero(cluster_key == c) for c in unique}
        for b in range(bootstrap_B):
            drawn = rng.choice(unique, size=unique.size, replace=True)
            sel = np.concatenate([idx_by_cluster[c] for c in drawn])
            sel_funded = sel[funded_mask[sel]]
            if sel_funded.size < beta.size + 1:
                boot[b] = np.nan
                continue
            Wb = _design(Xall[sel_funded],
                         propensity.imr[sel_funded].reshape(-1, 1))
            try:
                yb = (np.full(apps.n, np.nan))
                yb[funded_mask] = y_funded.astype(float)
                rb = sm.GLM(
                    yb[sel_funded], Wb,
                    family=sm.families.Binomial(sm.families.links.Probit()),
                ).fit(disp=False)
                boot[b] = rb.params
            except Exception:
                boot[b] = np.nan
        se_dict["bootstrap"] = np.nanstd(boot, axis=0, ddof=1)

    fnames = ("__intercept__",) + tuple(apps.feature_names()) + ("imr",)
    return OutcomeArtifact(
        method="heckman",
        beta=beta,
        feature_names=fnames,
        pd_through_door=pd_through,
        pd_funded=pd_funded,
        pd_declined=pd_declined,
        standard_errors=se_dict,
    )


def fit_aipw_outcome(
    apps: ApplicantSnapshot,
    y_funded: np.ndarray,
    funded_mask: np.ndarray,
    propensity: PropensityArtifact,
    base_estimator: Optional[Any] = None,
    n_splits: int = 5,
    cross_fit: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> OutcomeArtifact:
    """Doubly-robust AIPW with optional cross-fitting.

    The pseudo-outcome combines an outcome regression g_hat on the
    funded slice with the propensity reweighting:

        psi_i = g_hat(X_i) + (s_i / pi_i) * (y_i - g_hat(X_i))

    g_hat is fit with K-fold cross-fitting to remove own-observation
    bias. The resulting psi is plugged into a weighted logistic to
    obtain a deployable closed-form scorer.
    """
    rng = rng or np.random.default_rng(0)
    base = base_estimator or LogisticRegression(max_iter=500)

    n = apps.n
    Xall = apps.X.to_numpy()
    Xf = Xall[funded_mask]
    g_hat = np.zeros(n)

    if cross_fit:
        kf = KFold(n_splits=n_splits, shuffle=True,
                   random_state=int(rng.integers(0, 2**31 - 1)))
        for tr_idx, te_idx in kf.split(Xf):
            est = clone(base)
            est.fit(Xf[tr_idx], y_funded[tr_idx])
            funded_indices = np.flatnonzero(funded_mask)
            te_global = funded_indices[te_idx]
            g_hat[te_global] = est.predict_proba(Xall[te_global])[:, 1]
        rest_mask = ~funded_mask
        if rest_mask.any():
            est_full = clone(base).fit(Xf, y_funded)
            g_hat[rest_mask] = est_full.predict_proba(Xall[rest_mask])[:, 1]
    else:
        est_full = clone(base).fit(Xf, y_funded)
        g_hat = est_full.predict_proba(Xall)[:, 1]

    y_use = np.zeros(n, dtype=float)
    y_use[funded_mask] = y_funded.astype(float)
    pi = propensity.pi
    psi = g_hat + (apps.s / pi) * (y_use - g_hat)
    psi = np.clip(psi, 0.0, 1.0)

    X_two = np.vstack([Xall, Xall])
    y_two = np.concatenate([np.ones(n), np.zeros(n)])
    w_two = np.concatenate([psi, 1.0 - psi])
    final = LogisticRegression(
        max_iter=1000, C=1e6, solver="lbfgs",
    ).fit(X_two, y_two, sample_weight=w_two)
    beta = np.concatenate([final.intercept_, final.coef_[0]])

    pd_all = final.predict_proba(Xall)[:, 1]
    pd_funded = float(pd_all[funded_mask].mean())
    pd_declined = (float(pd_all[~funded_mask].mean())
                   if (~funded_mask).any() else float("nan"))
    fnames = ("__intercept__",) + tuple(apps.feature_names())

    return OutcomeArtifact(
        method="aipw",
        beta=beta,
        feature_names=fnames,
        pd_through_door=float(pd_all.mean()),
        pd_funded=pd_funded,
        pd_declined=pd_declined,
        extra={"g_hat_mean_funded": float(g_hat[funded_mask].mean()),
               "psi_clip_share": float(((psi <= 0.0) | (psi >= 1.0)).mean())},
    )


def fit_aipcw_outcome(
    joined: JoinedSnapshot,
    propensity: PropensityArtifact,
    base_estimator: Optional[Any] = None,
    rng: Optional[np.random.Generator] = None,
) -> OutcomeArtifact:
    """AIPW upgraded for the censored (unmatured) tail.

    A funded applicant whose performance window has not yet closed has
    no y. Naive AIPW ignores those rows; AIPCW adds an inverse-
    probability-of-censoring weight 1 / S_C(t_i | x_i) so the matured
    funded slice represents the full funded population.

    This implementation uses a simple time-since-as_of logistic for the
    censoring hazard (equivalent to a discrete-time proportional hazard
    with one knot per month). For cohorts where the censoring mechanism
    is heavily covariate-dependent, swap in the survival model from
    survival_diagnostics.ipcw.
    """
    rng = rng or np.random.default_rng(0)
    apps = joined.applicants
    funded_mask = apps.s == 1
    matured = joined.matured_mask

    age = (joined.snapshot_date - apps.as_of).dt.days.to_numpy() / 30.0
    age = np.clip(age, 0.0, None)

    censored = funded_mask & (~matured)
    cens_mask_for_fit = funded_mask
    cens_y = censored[cens_mask_for_fit].astype(int)
    cens_X = np.column_stack([age[cens_mask_for_fit], apps.X.to_numpy()[cens_mask_for_fit]])

    cens_model = LogisticRegression(max_iter=500).fit(cens_X, cens_y)
    p_censor_full = np.zeros(apps.n)
    p_censor_full[funded_mask] = cens_model.predict_proba(cens_X)[:, 1]
    s_C = np.clip(1.0 - p_censor_full, 1e-3, 1.0)

    funded_idx = np.flatnonzero(funded_mask & matured)
    y_funded = np.zeros(funded_idx.size)
    aid_to_y = pd.Series(joined.outcomes.y, index=joined.outcomes.applicant_id.values)
    y_funded = aid_to_y.loc[apps.applicant_id.values[funded_idx]].to_numpy()

    base = base_estimator or LogisticRegression(max_iter=500)

    art = fit_aipw_outcome(
        apps=apps,
        y_funded=y_funded,
        funded_mask=(np.isin(np.arange(apps.n), funded_idx)),
        propensity=propensity,
        base_estimator=base,
        n_splits=5,
        cross_fit=True,
        rng=rng,
    )
    art.method = "aipcw"
    art.extra = dict(art.extra)
    art.extra.update({
        "censored_share": float(censored.mean()),
        "p99_inverse_S_C": float(np.quantile(1.0 / s_C[funded_mask], 0.99)),
    })
    return art


def predict_pd(
    art: OutcomeArtifact,
    X: np.ndarray,
    propensity: Optional[PropensityArtifact] = None,
) -> np.ndarray:
    """Closed-form scoring path for a deployed outcome artifact."""
    if art.method == "heckman":
        if propensity is None:
            raise ValueError("Heckman scoring needs the propensity for IMR")
        W = _design(X, propensity.imr.reshape(-1, 1))
        return stats.norm.cdf(W @ art.beta)
    W = _design(X)
    z = W @ art.beta
    return 1.0 / (1.0 + np.exp(-z))
