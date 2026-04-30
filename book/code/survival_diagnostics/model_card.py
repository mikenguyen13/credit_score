"""Model card for the survival-diagnostics report.

The card documents intended use, scope limits, the four diagnostics
that constitute the artifact, and the decision rules for escalation.
A risk team should be able to override every section; the rendered
markdown is what auditors and SR 11-7 / IFRS 9 reviewers consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SurvivalDiagnosticsCard:
    name: str = "survival_censoring_diagnostics"
    version: str = "1.0.0"
    owner: str = "Retail Credit Risk / Model Validation"
    intended_use: str = (
        "Quantify the sensitivity of headline lifetime PD to the "
        "T independent of C given x assumption on a survival model "
        "fit to a retail loan vintage. Output is the validation-pack "
        "annex attached next to the headline KM / Cox / AFT curve."
    )
    out_of_scope: str = (
        "Portfolios where the dominant censoring cause is not prepayment "
        "or lender-initiated closure (e.g. fraud-driven write-offs); "
        "models with dynamic time-varying covariates entered as a "
        "counting-process panel (run a panel-aware variant upstream); "
        "credit-cycle stress horizons longer than the cohort observation "
        "window (extrapolation is the macro model's job, not this one's)."
    )
    diagnostics: list[str] = field(default_factory=lambda: [
        "Cause-overlap KS + SMD across {admin, prepay, default, lender_close}.",
        "IPCW Cox-based reweighting with stabilised, capped weights.",
        "Tipping-point sensitivity over rho in [0.5, 2.0].",
        "Clean-cohort holdout vs prepay-heavy vintage.",
        "Aalen-Johansen cumulative incidence per cause.",
        "Fine-Gray subdistribution Cox under administrative censoring.",
        "Truncation guard: detect event-only or no-defaulter cohorts; emit "
        "delayed-entry KM (left truncation) and reverse-time KM (right "
        "truncation, Lagakos 1988) with bias deltas in basis points.",
    ])
    escalation_rules: list[str] = field(default_factory=lambda: [
        "Naive vs IPCW 12m PD gap > 50 bps -> widen covariate set or escalate.",
        "Tipping range over rho in [0.5, 2.0] crosses any decision threshold -> escalate.",
        "Cause-overlap any_imbalanced=true on a covariate not in the survival model -> retrain with that covariate or run IPCW with it included.",
        "Clean-cohort lifetime PD differs from full-cohort by more than 25% relative -> investigate vintage selection bias.",
        "IPCW p99 weight > 10 -> check for positivity violations; consider widening cap or trimming the censoring tail.",
        "Truncation block triggered (corrected vs naive PD gap > 50 bps at any horizon) -> stop the validation run and re-extract the cohort with the at-risk denominator restored.",
    ])
    references: list[str] = field(default_factory=lambda: [
        "Robins, J.M. and Rotnitzky, A. (1992). Recovery of information and adjustment for dependent censoring using surrogate markers.",
        "Fine, J.P. and Gray, R.J. (1999). A proportional hazards model for the subdistribution of a competing risk.",
        "Geskus, R.B. (2011). Cause-specific cumulative incidence estimation and the Fine and Gray model under both left truncation and right censoring.",
        "Tsiatis, A.A. (1975). A nonidentifiability aspect of the problem of competing risks.",
        "Cole, S.R. and Hernan, M.A. (2008). Constructing inverse probability weights for marginal structural models.",
        "Austin, P.C. (2009). Balance diagnostics for comparing the distribution of baseline covariates between treatment groups.",
        "Lagakos, S.W., Barraj, L.M., De Gruttola, V. (1988). Nonparametric analysis of truncated survival data.",
        "Andersen, P.K. and Gill, R.D. (1982). Cox's regression model for counting processes.",
    ])


def render_card(card: Optional[SurvivalDiagnosticsCard] = None) -> str:
    c = card or SurvivalDiagnosticsCard()
    lines = [
        f"# Model Card: {c.name}",
        f"**Version:** {c.version}  ",
        f"**Owner:** {c.owner}",
        "",
        "## Intended Use",
        c.intended_use,
        "",
        "## Out of Scope",
        c.out_of_scope,
        "",
        "## Diagnostics in the Artifact",
        *(f"- {d}" for d in c.diagnostics),
        "",
        "## Escalation Rules",
        *(f"- {r}" for r in c.escalation_rules),
        "",
        "## References",
        *(f"- {r}" for r in c.references),
        "",
    ]
    return "\n".join(lines)
