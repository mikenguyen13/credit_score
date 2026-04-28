"""Model card renderer (Mitchell et al., 2019).

The card is intentionally minimal: a risk team should be able to fill
in or override every section, and the resulting markdown is what
auditors and SR 11-7 reviewers consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelCard:
    name: str = "merton_kmv_edf"
    version: str = "1.0.0"
    owner: str = "Wholesale Credit Risk"
    intended_use: str = (
        "Point-in-time one-year PD for listed corporate borrowers, used "
        "as a challenger to the IRB internal rating and as an input to "
        "the wholesale RAROC engine."
    )
    out_of_scope: str = (
        "Privately held firms (no equity feed); financial-firm "
        "subsidiaries with intra-group debt; firms in the first 60 "
        "trading days post-IPO; sovereign and project finance."
    )
    training_window: str = "Rolling 1-year daily equity history, refreshed daily."
    data_sources: list[str] = field(default_factory=lambda: [
        "Bloomberg or Refinitiv equity prices",
        "Compustat quarterly debt (DLTT + DLC, mapped via KMV 0.5x rule)",
        "FRED 1-year Treasury or OIS swap curve",
    ])
    known_failure_modes: list[str] = field(default_factory=lambda: [
        "Iterative solver oscillation for highly leveraged firms (logged via diagnostics).",
        "Spurious sigma_V jumps after corporate actions (split, spin-off, merger).",
        "Short-horizon PD undershoot from diffusion-only dynamics (no jumps).",
        "Sector mis-calibration: utilities over-stated, tech under-stated absent the recalibration layer.",
    ])
    metrics: dict[str, str] = field(default_factory=lambda: {
        "discrimination": "AUROC on 1y default flag, target >= 0.78",
        "calibration": "Hosmer-Lemeshow p > 0.05 across deciles",
        "stability": "Rolling 90d sigma_V z-score |z| < 3 for established firms",
    })
    challenger: str = "Bharath-Shumway naive DD; Altman Z'; Jarrow-Turnbull reduced form."


def render_model_card(card: Optional[ModelCard] = None) -> str:
    card = card or ModelCard()
    lines = [
        f"# Model Card: {card.name}",
        f"**Version:** {card.version}  ",
        f"**Owner:** {card.owner}",
        "",
        "## Intended Use",
        card.intended_use,
        "",
        "## Out of Scope",
        card.out_of_scope,
        "",
        "## Training Window",
        card.training_window,
        "",
        "## Data Sources",
        *(f"- {s}" for s in card.data_sources),
        "",
        "## Known Failure Modes",
        *(f"- {f}" for f in card.known_failure_modes),
        "",
        "## Metrics",
        *(f"- **{k}:** {v}" for k, v in card.metrics.items()),
        "",
        "## Challenger Models",
        card.challenger,
        "",
    ]
    return "\n".join(lines)
