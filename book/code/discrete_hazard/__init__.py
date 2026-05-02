"""Production discrete-time (Shumway) hazard package.

Modules
-------
schema       Long-table panel dataclass and validation.
fit          Vintage-grouped split + cluster-robust Shumway logit fit;
             hashed, persisted ShumwayHazardArtifact.
layers       CHS calendar covariates (layer 1), Duffie forward-
             distribution PD (layer 2), year FE / profile-likelihood
             frailty / Bharath naive distance-to-default (layer 3),
             boosted long-table classifier (layer 4).
validation   Time-dependent AUC, Brier, calibration-by-decile,
             bootstrap term-structure CIs.
pipeline     Orchestrator: panel -> ShumwayHazardArtifact +
             ShumwayPipelineArtifact (JSON validation pack).
model_card   Markdown card generator for the SR 11-7 / IFRS 9 review.

The entry point used by the FastAPI service in
``deployment/discrete_hazard_app.py`` is :func:`pipeline.run_shumway`,
which consumes a long-table panel and emits both the persisted hazard
artifact and a JSON validation pack keyed by vintage cohort.
"""

from .schema import LongTablePanel, validate_panel
from .fit import (
    FitConfig,
    ShumwayHazardArtifact,
    fit_shumway_logit,
    vintage_grouped_split,
)
from .layers import (
    Ar1Process,
    FrailtyFilterResult,
    FrailtyOUPrior,
    add_calendar_covariates,
    bharath_naive_dd,
    boosted_long_table_clf,
    forward_distribution_pd,
    frailty_particle_filter,
    profile_likelihood_frailty,
    vintage_year_fe_columns,
)
from .validation import (
    CalibrationDecile,
    HorizonScore,
    ValidationResult,
    bootstrap_term_structure,
    calibration_by_decile,
    time_dependent_scores,
)
from .pipeline import (
    ShumwayConfig,
    ShumwayPipelineArtifact,
    run_shumway,
)
from .model_card import ShumwayModelCard, render_card

__all__ = [
    "LongTablePanel", "validate_panel",
    "FitConfig", "ShumwayHazardArtifact",
    "fit_shumway_logit", "vintage_grouped_split",
    "Ar1Process", "FrailtyFilterResult", "FrailtyOUPrior",
    "add_calendar_covariates", "bharath_naive_dd",
    "boosted_long_table_clf", "forward_distribution_pd",
    "frailty_particle_filter",
    "profile_likelihood_frailty", "vintage_year_fe_columns",
    "CalibrationDecile", "HorizonScore", "ValidationResult",
    "bootstrap_term_structure", "calibration_by_decile",
    "time_dependent_scores",
    "ShumwayConfig", "ShumwayPipelineArtifact", "run_shumway",
    "ShumwayModelCard", "render_card",
]
