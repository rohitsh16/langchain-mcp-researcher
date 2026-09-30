"""
Risk and coverage metrics for statistical verification and conformal prediction.
"""
import math
from typing import List, Tuple


def compute_empirical_coverage(covered: List[bool]) -> float:
    """Computes empirical fraction of instances where ground truth is covered."""
    if not covered:
        return 0.0
    return sum(1.0 for c in covered if c) / len(covered)


def compute_selective_risk(losses: List[float], accepted: List[bool]) -> float:
    """
    Computes selective risk: mean loss conditioned on acceptance.
    Risk_sel = sum(L_i * Accept_i) / sum(Accept_i)
    """
    acc_losses = [l for l, a in zip(losses, accepted) if a]
    if not acc_losses:
        return 0.0
    return sum(acc_losses) / len(acc_losses)


def wilson_confidence_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes Wilson score interval for binomial proportions.
    Guaranteed coverage even for small sample sizes.
    """
    if total == 0:
        return 0.0, 1.0

    # Normal critical value z
    z_map = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    z = z_map.get(confidence, 1.960)

    p_hat = successes / total
    denominator = 1.0 + (z**2) / total
    centre_adj = p_hat + (z**2) / (2.0 * total)
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + (z**2) / (4.0 * total)) / total)

    lower = max(0.0, (centre_adj - margin) / denominator)
    upper = min(1.0, (centre_adj + margin) / denominator)
    return lower, upper


wilson_score_interval = wilson_confidence_interval
