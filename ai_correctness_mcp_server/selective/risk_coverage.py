"""
Risk-coverage curve computation and metrics (AURC, accuracy at coverage).
"""
from typing import List, Dict, Tuple, Any


def compute_risk_coverage_curve(
    confidences: List[float],
    losses: List[float],
) -> Dict[str, Any]:
    """
    Computes empirical risk-coverage curve given confidence scores and binary losses (0 = correct, 1 = error).
    Returns list of (coverage, risk) points and Area Under the Risk-Coverage Curve (AURC).
    """
    if len(confidences) != len(losses):
        raise ValueError("Confidences and losses must have the same length.")
    n = len(confidences)
    if n == 0:
        return {"coverages": [], "risks": [], "aurc": 0.0, "e_aurc": 0.0}

    # Sort descending by confidence
    paired = sorted(zip(confidences, losses), key=lambda x: x[0], reverse=True)

    coverages: List[float] = []
    risks: List[float] = []

    cum_loss = 0.0
    for k, (_, loss) in enumerate(paired, start=1):
        cum_loss += float(loss)
        coverage = k / n
        risk = cum_loss / k
        coverages.append(coverage)
        risks.append(risk)

    # Compute AURC using trapezoidal approximation or average over coverage steps
    # Standard AURC is 1/n * sum_{k=1}^n R(k)
    aurc = sum(risks) / n

    # Optimal baseline (oracle ordering: losses 0 first, then 1s)
    sorted_losses = sorted(losses)
    oracle_risks = []
    cum_opt = 0.0
    for k, loss in enumerate(sorted_losses, start=1):
        cum_opt += float(loss)
        oracle_risks.append(cum_opt / k)
    optimal_aurc = sum(oracle_risks) / n
    excess_aurc = max(0.0, aurc - optimal_aurc)

    return {
        "coverages": coverages,
        "risks": risks,
        "aurc": aurc,
        "optimal_aurc": optimal_aurc,
        "excess_aurc": excess_aurc,
    }


def compute_accuracy_at_coverage(
    confidences: List[float],
    correctness: List[int],
    target_coverages: List[float] = (0.5, 0.7, 0.8, 0.9, 1.0),
) -> Dict[float, float]:
    """
    Computes accuracy on the accepted subset for specific target coverage levels.
    """
    if len(confidences) != len(correctness):
        raise ValueError("Confidences and correctness must have the same length.")
    n = len(confidences)
    if n == 0:
        return {cov: 0.0 for cov in target_coverages}

    paired = sorted(zip(confidences, correctness), key=lambda x: x[0], reverse=True)
    results = {}

    for cov in target_coverages:
        k = max(1, min(n, int(round(cov * n))))
        top_k = paired[:k]
        acc = sum(c for _, c in top_k) / k
        results[cov] = acc

    return results
