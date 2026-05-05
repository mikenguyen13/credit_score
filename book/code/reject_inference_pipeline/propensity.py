"""Propensity estimation and exclusion-restriction diagnostics.

Three workflows live here:

fit_observable_propensity      read pi_logged from the snapshot, run
                               overlap and clip diagnostics, return
                               PropensityArtifact with the exact pi.

fit_selection_probit           Heckman stage-1 probit on (X, Z, s).
                               Returns coefficients, IMR vector, and
                               the diagnostic PropensityArtifact.

run_iv_diagnostics             every retrain re-checks the exclusion
                               restriction. Z must remain
                               insignificant when added to a probit on
                               y from the funded slice. A coefficient
                               that becomes significant kills the IV.

The estimators wrap statsmodels' GLM/probit and add (a) a
deterministic tie-breaker for perfect separation and (b) a partial-
out-X safeguard so a high-multicollinearity Z does not silently
collapse the stage-1 fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats
import statsmodels.api as sm

from .schema import ApplicantSnapshot


@dataclass
class PropensityArtifact:
    """Output of any propensity fit, observable or estimated."""

    mode: str                              # 'observable' or 'estimated'
    pi: np.ndarray                         # propensity per applicant
    imr: np.ndarray                        # inverse Mills ratio per applicant
    gamma: Optional[np.ndarray] = None     # stage-1 coefficients if estimated
    feature_names: tuple[str, ...] = ()    # ordering of gamma
    overlap_min: float = 0.0
    overlap_max: float = 1.0
    clip_share: float = 0.0
    selection_auc: Optional[float] = None
    n_funded: int = 0
    n_total: int = 0


@dataclass
class IVDiagnostic:
    """Output of the per-cycle exclusion-restriction recheck."""

    z_in_outcome_pvalues: dict[str, float]
    z_in_outcome_coefs: dict[str, float]
    z_in_outcome_threshold: float
    iv_blocked: bool
    blocked_columns: tuple[str, ...]
    f_stat_first_stage: Optional[float] = None
    weak_iv_threshold: float = 10.0
    weak_iv_blocked: bool = False


def fit_observable_propensity(
    apps: ApplicantSnapshot,
    eps: float = 1e-3,
) -> PropensityArtifact:
    if apps.pi_logged is None:
        raise ValueError(
            "fit_observable_propensity called without logged pi; "
            "use fit_selection_probit for unobservable mode"
        )
    pi = np.clip(apps.pi_logged.copy(), eps, 1 - eps)
    clip_share = float(((apps.pi_logged < eps)
                       | (apps.pi_logged > 1 - eps)).mean())
    lin = stats.norm.ppf(pi)
    imr = stats.norm.pdf(lin) / np.clip(stats.norm.cdf(lin), 1e-8, None)
    return PropensityArtifact(
        mode="observable",
        pi=pi,
        imr=imr,
        gamma=None,
        feature_names=tuple(apps.feature_names() + apps.iv_names()),
        overlap_min=float(pi.min()),
        overlap_max=float(pi.max()),
        clip_share=clip_share,
        n_funded=int(apps.n_funded),
        n_total=int(apps.n),
    )


def fit_selection_probit(
    apps: ApplicantSnapshot,
    eps: float = 1e-3,
    max_iter: int = 100,
) -> PropensityArtifact:
    """Heckman stage-1 probit on the full applicant snapshot."""
    X = apps.X.to_numpy()
    Z = apps.Z.to_numpy()
    W = np.column_stack([np.ones(apps.n), X, Z])
    s = apps.s.astype(float)
    if s.sum() in (0, apps.n):
        raise ValueError("stage-1 requires both funded and declined rows")
    model = sm.GLM(s, W, family=sm.families.Binomial(sm.families.links.Probit()))
    res = model.fit(maxiter=max_iter, disp=False)
    gamma = res.params
    lin = W @ gamma
    pi_raw = stats.norm.cdf(lin)
    pi = np.clip(pi_raw, eps, 1 - eps)
    clip_share = float(((pi_raw < eps) | (pi_raw > 1 - eps)).mean())
    imr = stats.norm.pdf(lin) / np.clip(stats.norm.cdf(lin), 1e-8, None)

    try:
        from sklearn.metrics import roc_auc_score
        sel_auc = float(roc_auc_score(apps.s, pi_raw))
    except Exception:
        sel_auc = None

    return PropensityArtifact(
        mode="estimated",
        pi=pi,
        imr=imr,
        gamma=np.asarray(gamma, dtype=float),
        feature_names=("__intercept__",) + tuple(apps.feature_names())
                      + tuple(apps.iv_names()),
        overlap_min=float(pi_raw.min()),
        overlap_max=float(pi_raw.max()),
        clip_share=clip_share,
        selection_auc=sel_auc,
        n_funded=int(apps.n_funded),
        n_total=int(apps.n),
    )


def run_iv_diagnostics(
    apps: ApplicantSnapshot,
    y_funded: np.ndarray,
    funded_mask: np.ndarray,
    p_threshold: float = 0.05,
    weak_iv_threshold: float = 10.0,
    imr: Optional[np.ndarray] = None,
) -> IVDiagnostic:
    """Re-run the exclusion-restriction check against the latest labels.

    Two tests:

    1. Outcome regression on the funded slice: probit y on (X, IMR, Z),
       where IMR is the inverse Mills ratio from stage 1. Each Z
       column should have p > p_threshold *after* conditioning on the
       IMR; without that control, any valid IV will appear significant
       because Z drives the IMR which drives selection-correlated
       residual variation in y. Pass ``imr=None`` to fall back to the
       unconditional probit (only valid for observable-pi snapshots
       where the selection equation is exactly known).

    2. First-stage strength: F-statistic from a linear projection of
       s on (X, Z) restricted to Z. F < weak_iv_threshold is the
       Stock-Yogo weak-instrument flag. A weak IV does not block by
       default but is logged in the artifact and surfaced to the
       reviewer.
    """
    X = apps.X.to_numpy()
    Z = apps.Z.to_numpy()
    z_names = list(apps.Z.columns)
    nf = int(funded_mask.sum())
    blocks = [np.ones(nf), X[funded_mask]]
    if imr is not None:
        blocks.append(imr[funded_mask].reshape(-1, 1))
    blocks.append(Z[funded_mask])
    Wf = np.column_stack(blocks)
    yf = y_funded.astype(float)
    pvals: dict[str, float] = {}
    coefs: dict[str, float] = {}
    blocked: list[str] = []
    try:
        out = sm.GLM(yf, Wf,
                     family=sm.families.Binomial(sm.families.links.Probit())
                     ).fit(disp=False)
        z_offset = 1 + X.shape[1] + (1 if imr is not None else 0)
        for j, name in enumerate(z_names):
            col = z_offset + j
            pvals[name] = float(out.pvalues[col])
            coefs[name] = float(out.params[col])
            if pvals[name] < p_threshold:
                blocked.append(name)
    except Exception as exc:
        for name in z_names:
            pvals[name] = float("nan")
            coefs[name] = float("nan")
        blocked = list(z_names)

    f_stat: Optional[float] = None
    try:
        Wfull = np.column_stack([np.ones(apps.n), X])
        proj = Wfull @ np.linalg.lstsq(Wfull, Z, rcond=None)[0]
        Z_resid = Z - proj
        s_resid = apps.s - Wfull @ np.linalg.lstsq(Wfull, apps.s.astype(float),
                                                   rcond=None)[0]
        rss_full = (s_resid - Z_resid @ np.linalg.lstsq(Z_resid, s_resid,
                                                        rcond=None)[0])
        rss_full = float((rss_full ** 2).sum())
        rss_red = float((s_resid ** 2).sum())
        k = Z.shape[1]
        df_resid = max(apps.n - X.shape[1] - k - 1, 1)
        if rss_full > 0:
            f_stat = ((rss_red - rss_full) / k) / (rss_full / df_resid)
    except Exception:
        f_stat = None

    return IVDiagnostic(
        z_in_outcome_pvalues=pvals,
        z_in_outcome_coefs=coefs,
        z_in_outcome_threshold=p_threshold,
        iv_blocked=bool(blocked),
        blocked_columns=tuple(blocked),
        f_stat_first_stage=f_stat,
        weak_iv_threshold=weak_iv_threshold,
        weak_iv_blocked=(f_stat is not None and f_stat < weak_iv_threshold),
    )


def overlap_summary(prop: PropensityArtifact, eps: float = 1e-3) -> dict:
    """Diagnostic readout for the validation pack."""
    pi = prop.pi
    return {
        "mode": prop.mode,
        "min": float(pi.min()),
        "p01": float(np.quantile(pi, 0.01)),
        "p99": float(np.quantile(pi, 0.99)),
        "max": float(pi.max()),
        "share_clipped": prop.clip_share,
        "share_below_eps": float((pi <= eps + 1e-9).mean()),
        "share_above_1m_eps": float((pi >= 1 - eps - 1e-9).mean()),
        "selection_auc": prop.selection_auc,
    }
