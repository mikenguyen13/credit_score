"""Layered upgrades on top of Shumway's pooled logit.

Each function below maps to one layer in the state-of-the-art menu
discussed in chapter 09:

* Layer 1 (CHS market and macro covariates) -- @campbell2008search and
  @bellotti2009credit add equity volatility, excess return,
  cash-over-market-assets, market leverage, and macro indices to
  Shumway's accounting set. The operational addition is a calendar
  join, factored into :func:`add_calendar_covariates`.
* Layer 2 (multi-horizon with stochastic covariates) --
  @duffie2007multi integrate the hazard over the forward distribution
  of a stochastic covariate path, producing a term structure that does
  not under-price long-horizon risk under mean reversion.
  :func:`forward_distribution_pd` computes the integrated cumulative
  PD by Monte Carlo over AR(1) macro paths.
* Layer 3 (frailty / year FE / Bharath naive distance-to-default) --
  production analogs of the Duffie-Eckner-Horel-Saita filter
  [@duffie2009frailty], in increasing order of cost. The cheapest
  proxies are :func:`vintage_year_fe_columns` and
  :func:`profile_likelihood_frailty`; :func:`frailty_particle_filter`
  is the faithful bootstrap particle filter for the OU-driven latent
  intensity, returning a marginal log-likelihood that can be tested
  against the no-frailty base fit.
* Layer 4 (machine-learning hazards) -- :func:`boosted_long_table_clf`
  swaps the linear hazard index for an XGBoost classifier on the same
  long table; the survival contract is identical and the model
  documentation fits the same SR 11-7 template.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit


# ---------------------------------------------------------------------------
# Layer 1: CHS-style market and macro covariates
# ---------------------------------------------------------------------------

def add_calendar_covariates(
    panel_df: pd.DataFrame,
    covariate_paths: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Join calendar-indexed covariate paths onto a long-table panel.

    `covariate_paths` is a dict ``{name -> array of length obs_horizon}``.
    The value for loan i at age t is read at calendar month
    ``v_i + t - 1``. Out-of-range calendar months clamp to the last
    observed value, which is the production convention for scoring at
    the edge of the macro window.
    """
    out = panel_df.copy()
    cal = out["cal_month"].astype(int).to_numpy()
    for name, path in covariate_paths.items():
        idx = np.clip(cal, 0, len(path) - 1)
        out[name] = path[idx]
    return out


# ---------------------------------------------------------------------------
# Layer 2: forward-distribution PD via stochastic covariate paths
# ---------------------------------------------------------------------------

@dataclass
class Ar1Process:
    """AR(1) generator: u_{t+1} = phi * u_t + sigma * eps_t."""

    phi: float
    sigma: float

    @classmethod
    def from_path(cls, u_hist: np.ndarray) -> "Ar1Process":
        u_hist = np.asarray(u_hist, dtype=float)
        phi = float(np.corrcoef(u_hist[:-1], u_hist[1:])[0, 1])
        resid = u_hist[1:] - phi * u_hist[:-1]
        sigma = float(resid.std(ddof=1))
        return cls(phi=phi, sigma=sigma)

    def simulate(
        self,
        u_today: float,
        horizon: int,
        n: int,
        seed: int = 0,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        out = np.zeros((n, horizon))
        out[:, 0] = self.phi * u_today + self.sigma * rng.normal(size=n)
        for t in range(1, horizon):
            out[:, t] = self.phi * out[:, t - 1] + self.sigma * rng.normal(size=n)
        return out


def forward_distribution_pd(
    artifact,
    covariates: dict[str, float],
    vintage_v: int,
    horizon: int,
    macro_name: str,
    process: Ar1Process,
    u_today: float,
    n_paths: int = 2000,
    seed: int = 0,
    quantiles: Sequence[float] = (0.05, 0.95),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Duffie-style multi-horizon PD: integrate hazard over simulated paths.

    Returns ``(age_grid, mean_cum_pd, quantile_cum_pd)``. The plug-in
    cumulative PD with frozen ``u_today`` under-prices long-horizon
    risk when the macro process is mean-reverting and today's value is
    benign; the integrated PD pulls toward the unconditional level.
    """
    age = np.arange(1, horizon + 1)
    paths = process.simulate(u_today, horizon, n=n_paths, seed=seed)
    cum = np.zeros((n_paths, horizon))
    for p in range(n_paths):
        macro_override = {macro_name: np.tile(paths[p], (1,))}
        cal = np.minimum(vintage_v + age - 1, artifact.obs_horizon - 1)
        # Replace the macro path with the simulated one for this horizon
        sim_path = np.full(artifact.obs_horizon, paths[p, -1])
        valid_len = min(horizon, artifact.obs_horizon - vintage_v)
        if valid_len > 0:
            end = min(vintage_v + valid_len, artifact.obs_horizon)
            sim_path[vintage_v:end] = paths[p, :end - vintage_v]
        h = artifact.predict_hazard(
            age=age, cal_month=cal, covariates=covariates,
            macro_override={macro_name: sim_path},
        )
        cum[p] = 1.0 - np.exp(np.cumsum(np.log1p(-h.clip(1e-12, 1 - 1e-12))))
    mean = cum.mean(axis=0)
    qarr = np.quantile(cum, list(quantiles), axis=0)
    return age, mean, qarr


# ---------------------------------------------------------------------------
# Layer 3: year fixed effects, profile-likelihood frailty, naive DD
# ---------------------------------------------------------------------------

def vintage_year_fe_columns(
    vintages: np.ndarray,
    n_buckets: int = 4,
    drop_first: bool = True,
) -> pd.DataFrame:
    """Bucket vintages into year-like dummies for a crude frailty proxy."""
    bucket = pd.cut(vintages, bins=n_buckets, labels=False)
    return pd.get_dummies(bucket, prefix="yr", drop_first=drop_first).astype(float)


def profile_likelihood_frailty(
    eta_train: np.ndarray,
    cal_train: np.ndarray,
    default_train: np.ndarray,
    obs_horizon: int,
    f_cap: float = 8.0,
) -> np.ndarray:
    """Per-calendar-month frailty intercept via profile likelihood.

    Solves for each calendar bucket v the scalar f_v such that
    ``sum_{i in R(v)} sigma(eta_i + f_v) == sum_{i in R(v)} d_iv``.
    A practical, fast cousin of the Duffie-Eckner-Horel-Saita particle
    filter; the recovered factor tracks the cyclical macro signal that
    no covariate in the design absorbed.
    """
    f_hat = np.zeros(obs_horizon)
    for v in np.unique(cal_train):
        mask = cal_train == v
        eta_v = eta_train[mask]
        n_v = int(mask.sum())
        d_v = int(default_train[mask].sum())
        if d_v == 0:
            f_hat[v] = -f_cap
            continue
        if d_v >= n_v:
            f_hat[v] = +f_cap
            continue
        try:
            f_hat[v] = brentq(
                lambda f: float(expit(eta_v + f).sum() - d_v),
                -f_cap - 2.0, f_cap + 2.0,
            )
        except ValueError:
            f_hat[v] = 0.0
    return f_hat


@dataclass
class FrailtyOUPrior:
    """Prior for the latent OU-style frailty factor f_v.

    The discrete-time analog of the Duffie-Eckner-Horel-Saita
    Ornstein-Uhlenbeck specification is the AR(1) ``f_v = phi f_{v-1} +
    sigma_eta eps_v`` with initial draw ``f_0 ~ N(f0, f0_sd^2)``. The
    hazard at calendar bucket v takes ``eta_i + lam * f_v``, where
    ``lam`` is the loading that maps factor scale to log-odds. The
    chapter discussion at @sec-ch09-shumway-sota motivates the choice;
    sensible defaults match the persistence reported in
    @duffie2009frailty (phi around 0.85).
    """

    phi: float = 0.85
    sigma_eta: float = 0.30
    lam: float = 1.0
    f0: float = 0.0
    f0_sd: float = 1.0


@dataclass
class FrailtyFilterResult:
    """Posterior summaries from :func:`frailty_particle_filter`."""

    f_mean: np.ndarray
    f_q05: np.ndarray
    f_q95: np.ndarray
    log_marginal: float
    n_particles: int
    ess_min: float
    prior: FrailtyOUPrior


def frailty_particle_filter(
    eta: np.ndarray,
    cal: np.ndarray,
    default: np.ndarray,
    obs_horizon: int,
    prior: Optional[FrailtyOUPrior] = None,
    n_particles: int = 1000,
    seed: int = 0,
    resample_threshold: float = 0.5,
) -> FrailtyFilterResult:
    """Bootstrap particle filter for the Duffie-Eckner-Horel-Saita
    latent intensity factor.

    Filters a shared, OU-style latent factor ``f_v`` from the residuals
    of a base hazard fit. The state equation is the discrete AR(1) in
    :class:`FrailtyOUPrior`; the observation equation at calendar
    bucket v is the Bernoulli hazard
    ``p(d_iv = 1 | f_v) = sigma(eta_i + lam * f_v)``,
    with ``eta_i`` the linear predictor from a base Shumway logit fit
    on a design that excludes the macro / cycle covariate.
    Returns posterior mean and 5 / 95 quantiles per calendar bucket
    plus the marginal log-likelihood, which can be compared against
    the no-frailty base fit to test whether the latent factor adds
    significant explanatory power.

    This is the production analog of the Duffie filter at the top of
    the cost ladder; :func:`profile_likelihood_frailty` is the cheaper
    pointwise cousin.
    """
    if prior is None:
        prior = FrailtyOUPrior()

    eta = np.asarray(eta, dtype=float)
    cal = np.asarray(cal, dtype=int)
    default = np.asarray(default, dtype=int)

    rng = np.random.default_rng(seed)
    P = int(n_particles)
    f_mean = np.zeros(obs_horizon)
    f_q05 = np.zeros(obs_horizon)
    f_q95 = np.zeros(obs_horizon)
    ess_min = float(P)
    log_marginal = 0.0

    particles = prior.f0 + prior.f0_sd * rng.normal(size=P)
    log_w = np.full(P, -np.log(P))

    bucket_idx = [np.where(cal == v)[0] for v in range(obs_horizon)]

    for v in range(obs_horizon):
        if v == 0:
            particles = prior.f0 + prior.f0_sd * rng.normal(size=P)
        else:
            particles = prior.phi * particles + prior.sigma_eta * rng.normal(size=P)

        idx_v = bucket_idx[v]
        if idx_v.size > 0:
            eta_v = eta[idx_v]
            d_v = default[idx_v]
            D_v = float(d_v.sum())
            z = eta_v[None, :] + prior.lam * particles[:, None]
            log_lik = prior.lam * particles * D_v - np.logaddexp(0.0, z).sum(axis=1)
            log_lik += float(d_v @ eta_v)
            log_w_unnorm = log_w + log_lik
            m = log_w_unnorm.max()
            log_marginal_v = m + np.log(np.exp(log_w_unnorm - m).sum())
            log_marginal += log_marginal_v
            log_w = log_w_unnorm - log_marginal_v

        weights = np.exp(log_w)
        weights = weights / weights.sum()
        f_mean[v] = float(weights @ particles)
        order = np.argsort(particles)
        cw = np.cumsum(weights[order])
        f_q05[v] = float(particles[order][np.searchsorted(cw, 0.05)])
        f_q95[v] = float(particles[order][np.searchsorted(cw, 0.95, side="right").clip(max=P - 1)])

        ess = 1.0 / float(np.sum(weights ** 2))
        ess_min = min(ess_min, ess)
        if ess < resample_threshold * P:
            resample_idx = rng.choice(P, size=P, replace=True, p=weights)
            particles = particles[resample_idx]
            log_w = np.full(P, -np.log(P))

    return FrailtyFilterResult(
        f_mean=f_mean,
        f_q05=f_q05,
        f_q95=f_q95,
        log_marginal=log_marginal,
        n_particles=P,
        ess_min=ess_min,
        prior=prior,
    )


def bharath_naive_dd(
    equity: np.ndarray | float,
    debt: np.ndarray | float,
    equity_ret: np.ndarray | float,
    equity_vol: np.ndarray | float,
) -> np.ndarray:
    """Bharath-Shumway (2008) naive distance-to-default.

    Skips the Merton solve, plugs accounting debt and observed equity
    volatility. The closed-form approximation captures most of the lift
    that fully layered structural models add on a pure accounting
    panel, and is the cheapest single move to bring structural-model
    signal into a Shumway logit.
    """
    equity = np.asarray(equity, dtype=float)
    debt = np.asarray(debt, dtype=float)
    equity_ret = np.asarray(equity_ret, dtype=float)
    equity_vol = np.asarray(equity_vol, dtype=float)
    V = equity + debt
    sigma_V = (equity / V) * equity_vol + (debt / V) * (0.05 + 0.25 * equity_vol)
    mu = equity_ret
    return (np.log(V / debt) + (mu - 0.5 * sigma_V ** 2)) / sigma_V


# ---------------------------------------------------------------------------
# Layer 4: boosted long-table classifier
# ---------------------------------------------------------------------------

def boosted_long_table_clf(
    train_df: pd.DataFrame,
    covariate_cols: Sequence[str],
    baseline: str = "log_age",
    n_estimators: int = 400,
    max_depth: int = 4,
    learning_rate: float = 0.05,
):
    """Fit an XGBoost binary classifier on the long table.

    Same data shape as the Shumway logit; the linear hazard index
    ``x'beta`` is replaced by a boosted tree. To reconstruct survival
    from the boosted hazard, score every age row of a loan with
    ``clf.predict_proba(...)[:, 1]`` and apply the same
    cumulative-product as :meth:`ShumwayHazardArtifact.predict_survival`.
    """
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "boosted_long_table_clf requires xgboost; "
            "install with `pip install xgboost`"
        ) from exc

    feat: list[str] = []
    if baseline == "log_age":
        feat = ["log_age", "age"]
    feat = feat + list(covariate_cols)

    X = train_df.assign(log_age=np.log(train_df["age"].astype(float)))[feat]
    y = train_df["default"].astype(int)

    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="logloss",
    )
    clf.fit(X, y)
    return clf, feat
