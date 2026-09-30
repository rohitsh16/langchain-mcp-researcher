"""
Calibration, risk, and efficiency metrics package.
"""
from research.metrics.calibration import (
    compute_ece,
    compute_ace,
    compute_brier_score,
    compute_reliability_diagram,
    platt_scale,
)
from research.metrics.risk import (
    compute_empirical_coverage,
    compute_selective_risk,
    wilson_confidence_interval,
)

__all__ = [
    "compute_ece",
    "compute_ace",
    "compute_brier_score",
    "compute_reliability_diagram",
    "platt_scale",
    "compute_empirical_coverage",
    "compute_selective_risk",
    "wilson_confidence_interval",
]
