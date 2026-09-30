"""
Adversarial distribution shift generators to test conformal coverage breakdown.
"""
from typing import List, Dict, Any
import random


def generate_shifted_data(
    base_scores: List[float],
    shift_type: str = "mean_shift",
    shift_magnitude: float = 0.20,
    seed: int = 42,
) -> List[float]:
    """
    Induces covariate/concept shift on confidence or nonconformity scores:
    - 'mean_shift': shifts scores systematically by magnitude
    - 'variance_inflation': spreads scores towards extremes
    - 'adversarial_truncation': suppresses high confidence scores
    """
    rng = random.Random(seed)
    shifted = []

    for s in base_scores:
        if shift_type == "mean_shift":
            new_s = s - shift_magnitude + rng.gauss(0, 0.05)
        elif shift_type == "variance_inflation":
            centered = s - 0.5
            new_s = 0.5 + centered * (1.0 + shift_magnitude) + rng.gauss(0, 0.05)
        elif shift_type == "adversarial_truncation":
            new_s = s * (1.0 - shift_magnitude)
        else:
            new_s = s

        shifted.append(max(0.0, min(1.0, new_s)))

    return shifted


def evaluate_conformal_breakdown(
    conformal_predictor,
    reference_scores: List[float],
    shifted_scores: List[float],
) -> Dict[str, Any]:
    """
    Compares coverage under in-distribution calibration vs shifted test distribution.
    """
    ref_included = sum(1 for s in reference_scores if conformal_predictor.predict_inclusion(s))
    ref_coverage = ref_included / len(reference_scores) if reference_scores else 0.0

    shift_included = sum(1 for s in shifted_scores if conformal_predictor.predict_inclusion(s))
    shift_coverage = shift_included / len(shifted_scores) if shifted_scores else 0.0

    coverage_gap = ref_coverage - shift_coverage

    return {
        "reference_coverage": ref_coverage,
        "shifted_coverage": shift_coverage,
        "coverage_gap": coverage_gap,
        "significant_drop": coverage_gap > 0.05,
    }
