"""Production survival-censoring diagnostics for credit portfolios.

Modules
-------
schema       Long-form loan panel dataclass and validation.
ipcw         Inverse-probability-of-censoring weights with stabilisation and capping.
tipping      Tipping-point sensitivity sweep over the censored-cohort hazard multiplier.
holdout      Clean-cohort vs prepay-heavy vintage comparison.
overlap      Covariate-distribution overlap (KS, SMD) across censoring causes.
competing    Aalen-Johansen cumulative incidence and Fine-Gray subdistribution Cox.
pipeline     Orchestrator: cohort -> diagnostics artifact -> validation pack JSON.
model_card   Markdown card generator for the survival-diagnostics report.

The entry point used by the FastAPI service in
``deployment/survival_diagnostics_app.py`` is
:func:`pipeline.run_diagnostics`, which consumes a cohort and emits a
versioned JSON artifact suitable for an SR 11-7 / IFRS 9 model
validation pack.
"""

from .schema import LoanCohort, validate_cohort
from .ipcw import IpcwConfig, IpcwResult, compute_ipcw, ipcw_kaplan_meier
from .tipping import TippingConfig, TippingResult, tipping_point_sweep
from .holdout import CohortHoldoutResult, cohort_holdout_compare
from .overlap import CauseOverlapResult, cause_overlap
from .competing import (
    AalenJohansenResult,
    aalen_johansen,
    fine_gray_admin_censoring,
    cause_specific_cox,
)
from .pipeline import DiagnosticsConfig, DiagnosticsArtifact, run_diagnostics
from .model_card import SurvivalDiagnosticsCard, render_card

__all__ = [
    "LoanCohort", "validate_cohort",
    "IpcwConfig", "IpcwResult", "compute_ipcw", "ipcw_kaplan_meier",
    "TippingConfig", "TippingResult", "tipping_point_sweep",
    "CohortHoldoutResult", "cohort_holdout_compare",
    "CauseOverlapResult", "cause_overlap",
    "AalenJohansenResult", "aalen_johansen",
    "fine_gray_admin_censoring", "cause_specific_cox",
    "DiagnosticsConfig", "DiagnosticsArtifact", "run_diagnostics",
    "SurvivalDiagnosticsCard", "render_card",
]
