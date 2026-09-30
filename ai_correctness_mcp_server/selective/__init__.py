"""
Selective prediction package.
"""
from .abstention import SelectivePredictor, SelectiveDecision
from .risk_coverage import compute_risk_coverage_curve, compute_accuracy_at_coverage

__all__ = [
    "SelectivePredictor",
    "SelectiveDecision",
    "compute_risk_coverage_curve",
    "compute_accuracy_at_coverage",
]
