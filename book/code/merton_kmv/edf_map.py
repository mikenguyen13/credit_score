"""Distance-to-default to PD mapping.

Two implementations are provided:

* :func:`dd_to_pd_normal` is the closed-form Merton PD, ``Phi(-DD)``.
  Useful as a sanity check and as a fallback when the empirical EDF
  table is unavailable. Documented to under-state short-horizon PD.

* :class:`IsotonicEDF` fits a monotone non-parametric map from DD to
  realized one-year default rate. This is the modern KMV/EDF approach:
  the parametric Merton tail is replaced by an empirical curve fit on
  the firm-year panel. The class persists with joblib for use inside
  the FastAPI service.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression


def dd_to_pd_normal(dd) -> np.ndarray:
    """Closed-form Merton PD: ``Phi(-DD)``."""
    return norm.cdf(-np.asarray(dd, dtype=float))


@dataclass
class IsotonicEDF:
    """Monotone DD -> PD calibration."""
    iso: IsotonicRegression = None
    dd_floor: float = -10.0
    dd_cap: float = 15.0
    pd_floor: float = 1.0e-5
    pd_cap: float = 0.5

    def fit(self, dd: np.ndarray, default: np.ndarray) -> "IsotonicEDF":
        dd = np.clip(np.asarray(dd, dtype=float), self.dd_floor, self.dd_cap)
        y = np.asarray(default, dtype=float)
        self.iso = IsotonicRegression(
            increasing=False, y_min=self.pd_floor, y_max=self.pd_cap,
            out_of_bounds="clip",
        )
        self.iso.fit(dd, y)
        return self

    def predict(self, dd) -> np.ndarray:
        if self.iso is None:
            raise RuntimeError("IsotonicEDF.fit must be called before predict.")
        dd = np.clip(np.asarray(dd, dtype=float), self.dd_floor, self.dd_cap)
        return np.clip(self.iso.predict(dd), self.pd_floor, self.pd_cap)


# Master-scale used by many wholesale shops. This is illustrative; bank
# letter scales differ slightly in their mid-points and in whether they
# include modifiers like A1/A2/A3.
_LETTER_BANDS = [
    (0.00010, "AAA"), (0.00050, "AA"), (0.00200, "A"),
    (0.00800, "BBB"), (0.02500, "BB"), (0.07500, "B"),
    (0.20000, "CCC"), (1.00000, "CC"),
]


def rating_from_pd(pd_value: float) -> str:
    """Map a PD to a coarse letter grade. Edges are ``<=`` boundaries."""
    p = float(pd_value)
    for upper, letter in _LETTER_BANDS:
        if p <= upper:
            return letter
    return "D"
