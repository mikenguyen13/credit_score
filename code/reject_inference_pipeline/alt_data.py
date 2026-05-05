"""Alt-data provider workflow: per-lender hierarchical propensity.

The provider sits outside any lender. The lender's underwriting policy
is opaque. The provider observes:

  - X_alt: the provider's own feature vector at decision time
  - lender_id: which bank scored this applicant
  - s: the bank's accept/decline (returned to the provider, possibly
       with delay)
  - y: bureau outcome (only on funded applicants)
  - own_score_logged: the provider's own score at decision time
       (recorded so the feedback-loop guard can detect when the bank
       starts using the provider's score in its own policy)

Three jobs live here:

fit_hierarchical_propensity   per-lender Heckman stage-1 with shrinkage
                              toward the cross-lender mean; cold-start
                              new lenders by setting the prior from
                              their lookalike peers.

cold_start_pseudoprior        empirical-Bayes pseudo-prior for a new
                              lender with insufficient observed
                              decisions; pulls strength from existing
                              lenders ranked by feature-distribution
                              similarity.

feedback_loop_guard           if the provider's own score becomes a
                              significant predictor of the lender's
                              decision, the system is training against
                              itself; the guard surfaces the warning
                              and recommends partial-out treatment of
                              own_score_logged in subsequent fits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from .schema import ApplicantSnapshot
from .propensity import PropensityArtifact


@dataclass
class HierarchicalPropensityArtifact:
    """Per-lender propensity stack."""

    per_lender: dict[str, PropensityArtifact]
    pooled_gamma: np.ndarray             # cross-lender mean
    shrinkage_lambda: float              # 0 = no pooling, 1 = full pooling
    feature_names: tuple[str, ...]
    cold_start_lenders: tuple[str, ...]


def _stack_design(apps: ApplicantSnapshot) -> np.ndarray:
    return np.column_stack([
        np.ones(apps.n),
        apps.X.to_numpy(),
        apps.Z.to_numpy(),
    ])


def _fit_lender_probit(W: np.ndarray, s: np.ndarray) -> Optional[np.ndarray]:
    if s.sum() in (0, len(s)):
        return None
    try:
        return np.asarray(sm.GLM(
            s.astype(float), W,
            family=sm.families.Binomial(sm.families.links.Probit()),
        ).fit(disp=False).params)
    except Exception:
        return None


def fit_hierarchical_propensity(
    apps: ApplicantSnapshot,
    lender_id: pd.Series,
    shrinkage_lambda: float = 0.5,
    min_n_per_lender: int = 200,
    eps: float = 1e-3,
) -> HierarchicalPropensityArtifact:
    """Per-lender Heckman stage-1 with shrinkage to the pooled mean.

    Lenders with fewer than ``min_n_per_lender`` observed decisions
    are flagged cold-start and assigned the pooled coefficient vector
    until they accumulate enough data for their own fit.
    """
    if not (0.0 <= shrinkage_lambda <= 1.0):
        raise ValueError("shrinkage_lambda must be in [0, 1]")
    W = _stack_design(apps)
    p = W.shape[1]

    pooled = _fit_lender_probit(W, apps.s)
    if pooled is None:
        raise RuntimeError("pooled stage-1 probit failed")

    fnames = (("__intercept__",) + tuple(apps.feature_names())
              + tuple(apps.iv_names()))

    per_lender: dict[str, PropensityArtifact] = {}
    cold: list[str] = []
    pi_full = np.zeros(apps.n)
    imr_full = np.zeros(apps.n)

    lender_arr = lender_id.values
    for lid in pd.unique(lender_arr):
        idx = np.flatnonzero(lender_arr == lid)
        if idx.size < min_n_per_lender:
            cold.append(str(lid))
            gamma_lid = pooled
        else:
            local = _fit_lender_probit(W[idx], apps.s[idx])
            gamma_lid = (pooled if local is None
                          else (1 - shrinkage_lambda) * local
                                + shrinkage_lambda * pooled)
        lin = W[idx] @ gamma_lid
        pi_raw = stats.norm.cdf(lin)
        pi = np.clip(pi_raw, eps, 1 - eps)
        imr = stats.norm.pdf(lin) / np.clip(stats.norm.cdf(lin), 1e-8, None)
        pi_full[idx] = pi
        imr_full[idx] = imr
        per_lender[str(lid)] = PropensityArtifact(
            mode="estimated",
            pi=pi, imr=imr, gamma=gamma_lid,
            feature_names=fnames,
            overlap_min=float(pi_raw.min()),
            overlap_max=float(pi_raw.max()),
            clip_share=float(((pi_raw < eps) | (pi_raw > 1 - eps)).mean()),
            n_funded=int(apps.s[idx].sum()),
            n_total=int(idx.size),
        )

    return HierarchicalPropensityArtifact(
        per_lender=per_lender,
        pooled_gamma=pooled,
        shrinkage_lambda=shrinkage_lambda,
        feature_names=fnames,
        cold_start_lenders=tuple(cold),
    )


def cold_start_pseudoprior(
    new_lender_features: pd.DataFrame,
    existing_per_lender: dict[str, PropensityArtifact],
    existing_lender_features: dict[str, pd.DataFrame],
    k_neighbours: int = 3,
) -> np.ndarray:
    """Pseudo-prior coefficients for a brand-new lender.

    Compute a Mahalanobis-like distance between the new lender's
    feature distribution and each existing lender's; average the
    coefficient vectors of the K closest peers.
    """
    if not existing_per_lender:
        raise ValueError("no existing lenders to borrow strength from")
    new_mu = new_lender_features.mean().to_numpy()
    new_var = new_lender_features.var().to_numpy()
    dists: list[tuple[str, float]] = []
    for lid, df in existing_lender_features.items():
        mu = df.reindex(columns=new_lender_features.columns).mean().to_numpy()
        var = df.reindex(columns=new_lender_features.columns).var().to_numpy()
        scale = np.sqrt(np.clip(0.5 * (new_var + var), 1e-9, None))
        d = float(np.linalg.norm((new_mu - mu) / scale))
        dists.append((lid, d))
    dists.sort(key=lambda x: x[1])
    chosen = [lid for lid, _ in dists[:k_neighbours]]
    coefs = np.stack([existing_per_lender[lid].gamma for lid in chosen
                      if existing_per_lender[lid].gamma is not None])
    return coefs.mean(axis=0)


def feedback_loop_guard(
    apps: ApplicantSnapshot,
    own_score_logged: np.ndarray,
    p_threshold: float = 0.05,
) -> dict[str, float]:
    """Detect whether the provider's own score has entered the lender's policy.

    Regress lender accept on (X, Z, own_score). If own_score is
    significant, the provider is training against its own predictions
    and the next fit must partial out own_score before estimating the
    selection coefficients.
    """
    W = np.column_stack([
        np.ones(apps.n), apps.X.to_numpy(), apps.Z.to_numpy(),
        own_score_logged.reshape(-1, 1),
    ])
    try:
        out = sm.GLM(
            apps.s.astype(float), W,
            family=sm.families.Binomial(sm.families.links.Probit()),
        ).fit(disp=False)
    except Exception as exc:
        return {"p_own_score": float("nan"), "coef_own_score": float("nan"),
                "feedback_detected": True, "error": str(exc)}
    return {
        "p_own_score": float(out.pvalues[-1]),
        "coef_own_score": float(out.params[-1]),
        "feedback_detected": bool(out.pvalues[-1] < p_threshold),
    }
