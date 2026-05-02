"""Model card for the Shumway discrete-time hazard production package.

The card documents intended use, scope limits, the layered upgrades
that this package supports, and the decision rules for SR 11-7
escalation. A risk team should be able to override every section; the
rendered markdown is what auditors and IFRS 9 / CECL reviewers consume
alongside the persisted hazard artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ShumwayModelCard:
    name: str = "discrete_hazard_shumway"
    version: str = "1.0.0"
    owner: str = "Credit Risk / Model Validation"
    intended_use: str = (
        "Discrete-time hazard PD model for retail loan-month or "
        "corporate firm-year panels. Output is a per-period hazard, "
        "a cumulative PD curve, and a stressed-macro lifetime PD for "
        "capital, IFRS 9 stage-2 ECL, and ICAAP. The fitted artifact "
        "is the production scoring contract."
    )
    out_of_scope: str = (
        "Continuous-time event data without a natural discretisation "
        "(use Cox PH instead); cohorts with no observed events on the "
        "training vintages (the logit will not converge); models that "
        "require a true counting-process panel with multiple events "
        "per loan (extend the schema or expand to competing risks)."
    )
    layers_supported: list[str] = field(default_factory=lambda: [
        "Layer 0: Shumway pooled logit on the long table with cluster-"
        "robust SEs and vintage-grouped train/holdout split.",
        "Layer 1 (CHS): time-varying market and macro covariates joined "
        "on calendar month (Campbell, Hilscher, Szilagyi 2008; "
        "Bellotti and Crook 2009).",
        "Layer 2 (Duffie multi-horizon): forward-distribution cumulative "
        "PD by Monte Carlo over AR(1) macro paths "
        "(Duffie, Saita, Wang 2007).",
        "Layer 3a (year FE): bucketed origination dummies as a crude "
        "frailty proxy.",
        "Layer 3b (profile-likelihood frailty): per-calendar-month "
        "intercept solved by Brent's method, the production cousin of "
        "the Duffie-Eckner-Horel-Saita filter.",
        "Layer 3c (Bharath naive distance-to-default): closed-form "
        "structural-model covariate (Bharath and Shumway 2008).",
        "Layer 4 (boosted long-table classifier): XGBoost on the same "
        "data shape as the SR 11-7 effective-challenger model.",
    ])
    escalation_rules: list[str] = field(default_factory=lambda: [
        "Holdout 12m AUC drops more than 5 points vs the prior fit -> "
        "investigate covariate drift or vintage-selection change.",
        "Calibration decile slope deviates from 1.0 by more than 0.10 "
        "at any horizon -> recalibrate the macro path or refit.",
        "Bootstrap term-structure 5/95 band at 36m exceeds the IFRS 9 "
        "decision band -> escalate to model risk; the lifetime PD is "
        "not pinned tightly enough for stage allocation.",
        "Forward-distribution lifetime PD differs from the frozen-"
        "covariate plug-in by more than 25 percent relative -> the "
        "macro process is materially mean-reverting and the layer-2 "
        "engine should drive provisioning, not the plug-in.",
    ])
    references: list[str] = field(default_factory=lambda: [
        "Shumway, T. (2001). Forecasting bankruptcy more accurately: "
        "a simple hazard model.",
        "Campbell, J.Y., Hilscher, J., Szilagyi, J. (2008). In search "
        "of distress risk.",
        "Duffie, D., Saita, L., Wang, K. (2007). Multi-period corporate "
        "default prediction with stochastic covariates.",
        "Duffie, D., Eckner, A., Horel, G., Saita, L. (2009). Frailty "
        "correlated default.",
        "Bellotti, T., Crook, J. (2009). Credit scoring with macro-"
        "economic variables using survival analysis.",
        "Bharath, S.T., Shumway, T. (2008). Forecasting default with "
        "the Merton distance to default model.",
        "Allison, P.D. (1982). Discrete-time methods for the analysis "
        "of event histories.",
        "Prentice, R.L., Gloeckler, L.A. (1978). Regression analysis of "
        "grouped survival data with application to breast cancer data.",
    ])


def render_card(card: Optional[ShumwayModelCard] = None) -> str:
    c = card or ShumwayModelCard()
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
        "## Layered Upgrades Supported",
        *(f"- {layer}" for layer in c.layers_supported),
        "",
        "## Escalation Rules",
        *(f"- {r}" for r in c.escalation_rules),
        "",
        "## References",
        *(f"- {r}" for r in c.references),
        "",
    ]
    return "\n".join(lines)
