"""Model card for the reject-inference retrain pipeline.

The card is the document attached to every artifact produced by the
pipeline. It captures intended use, scope limits, the diagnostic
contract, and the escalation rules that the gate enforces. Risk teams
override every section locally; the rendered markdown is what
auditors consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RejectInferenceCard:
    name: str = "reject_inference_pd"
    version: str = "1.0.0"
    owner: str = "Retail Credit Risk / Decision Science"
    intended_use: str = (
        "Estimate through-the-door PD on consumer-credit applications "
        "subject to selection bias from the underwriter's accept rule. "
        "Outputs a per-applicant PD for pricing, line management and "
        "regulatory PD reporting; the production champion is AIPW with "
        "Heckman as the SR 11-7 sensitivity anchor."
    )
    out_of_scope: str = (
        "Wholesale / SME credit (different selection structure); models "
        "where the lender's policy is unknown AND no exclusion "
        "restriction exists AND no bureau outcome is available on "
        "rejected applicants (the Hand-Henley impossibility regime; "
        "report bounds, not a point estimate); macro-shock periods "
        "where auto-promotion is frozen by policy."
    )
    diagnostics: list[str] = field(default_factory=lambda: [
        "Stage-1 selection AUC and propensity overlap (min, p01, p99, max).",
        "Exclusion-restriction recheck: Z in outcome equation p > p_thresh.",
        "First-stage F-statistic (weak-IV flag at F < 10).",
        "AIPW vs Heckman PD-through-the-door gap (sensitivity anchor).",
        "Per-vintage Brier and AUC (Basel TTC multi-vintage gate).",
        "Disparate impact ratio per protected class (ECOA four-fifths floor).",
        "Drift triple: covariate PSI, propensity PSI, funded default rate.",
        "Shadow-mode PSI between live champion and challenger predictions.",
    ])
    escalation_rules: list[str] = field(default_factory=lambda: [
        "IV diagnostic blocks any Z column at p < 0.05 -> abort promote, "
        "investigate exclusion restriction; either drop Z or document why.",
        "Weak-IV flag (F < 10) -> emit warning, allow promote only with "
        "validator sign-off.",
        "AIPW vs Heckman PD gap > documented tolerance -> escalate to "
        "model risk; do not auto-promote.",
        "Disparate-impact ratio < 0.80 OR drop > 0.02 vs champion on any "
        "protected class -> hard-block promotion.",
        "Basel TTC: challenger fits fewer than 3 vintages OR regresses on "
        "any vintage by > 50 bps Brier -> hard-block.",
        "Shadow PSI > 0.10 during the shadow window -> hold the swap, "
        "extend shadow.",
        "Post-promotion live PSI > 0.20 vs pre-promotion baseline -> "
        "auto-rollback to previous champion within one cycle.",
        "Macro-shock indicator (operator-set flag) -> freeze auto-promotion, "
        "require human approval.",
    ])
    references: list[str] = field(default_factory=lambda: [
        "Heckman, J.J. (1979). Sample Selection Bias as a Specification Error.",
        "Robins, J.M., Rotnitzky, A., Zhao, L.P. (1994). Estimation of "
        "regression coefficients when some regressors are not always observed.",
        "Chernozhukov, V. et al. (2018). Double/Debiased Machine Learning.",
        "Swaminathan, A., Joachims, T. (2015). Counterfactual Risk "
        "Minimization.",
        "Federal Reserve SR 11-7 (2011). Guidance on Model Risk Management.",
        "BCBS 128 (2006). Basel II IRB; downturn PD and TTC calibration.",
        "Hand, D.J., Henley, W.E. (1993/97). Statistical classification "
        "methods in consumer credit scoring.",
        "DeLong, E.R., DeLong, D.M., Clarke-Pearson, D.L. (1988). Comparing "
        "the areas under two or more correlated ROC curves.",
    ])


def render_card(card: Optional[RejectInferenceCard] = None) -> str:
    c = card or RejectInferenceCard()
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
