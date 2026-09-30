"""
Adversarial attacks and stress-testing subpackage for correctness evaluation.
"""
from .correlated_errors import simulate_correlated_failures
from .verifier_exploitation import generate_superficial_exploit, evaluate_verifier_susceptibility
from .evidence_conflict import create_conflicting_evidence_pair, evaluate_conflict_awareness
from .distribution_shift import generate_shifted_data, evaluate_conformal_breakdown

__all__ = [
    "simulate_correlated_failures",
    "generate_superficial_exploit",
    "evaluate_verifier_susceptibility",
    "create_conflicting_evidence_pair",
    "evaluate_conflict_awareness",
    "generate_shifted_data",
    "evaluate_conformal_breakdown",
]
