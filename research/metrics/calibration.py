"""
Statistical calibration metrics: ECE, ACE, Brier score, and Platt scaling.
"""
import math
from typing import Dict, List, Tuple


def compute_ece(probs: List[float], labels: List[int], num_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) with equal-width bins in [0, 1].
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    if len(probs) != len(labels):
        raise ValueError("Lengths of probs and labels must match.")
    if not probs:
        return 0.0

    n = len(probs)
    bin_size = 1.0 / num_bins
    ece = 0.0

    for m in range(num_bins):
        bin_lower = m * bin_size
        bin_upper = (m + 1) * bin_size

        indices = [
            i
            for i, p in enumerate(probs)
            if (bin_lower <= p < bin_upper) or (m == num_bins - 1 and bin_lower <= p <= 1.0)
        ]

        if not indices:
            continue

        bin_conf = sum(probs[i] for i in indices) / len(indices)
        bin_acc = sum(labels[i] for i in indices) / len(indices)

        ece += (len(indices) / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_ace(probs: List[float], labels: List[int], num_bins: int = 10) -> float:
    """
    Computes Adaptive Calibration Error (ACE) using equal-frequency (quantile) bins.
    """
    if len(probs) != len(labels) or not probs:
        return 0.0

    paired = sorted(zip(probs, labels), key=lambda x: x[0])
    n = len(paired)
    chunk_size = math.ceil(n / num_bins)
    ace = 0.0

    for m in range(0, n, chunk_size):
        chunk = paired[m : m + chunk_size]
        if not chunk:
            continue
        bin_conf = sum(p for p, _ in chunk) / len(chunk)
        bin_acc = sum(y for _, y in chunk) / len(chunk)
        ace += (len(chunk) / n) * abs(bin_acc - bin_conf)

    return float(ace)


def compute_brier_score(probs: List[float], labels: List[int]) -> float:
    """
    Brier Score: mean squared error between probabilistic forecast and binary outcome.
    BS = (1/N) * sum_{i=1}^N (p_i - y_i)^2
    """
    if len(probs) != len(labels) or not probs:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)


def compute_reliability_diagram(probs: List[float], labels: List[int], num_bins: int = 10) -> List[Dict[str, float]]:
    """
    Generates bin data for plotting reliability diagrams.
    """
    bin_size = 1.0 / num_bins
    diagram_bins = []

    for m in range(num_bins):
        bin_lower = m * bin_size
        bin_upper = (m + 1) * bin_size
        indices = [
            i
            for i, p in enumerate(probs)
            if (bin_lower <= p < bin_upper) or (m == num_bins - 1 and bin_lower <= p <= 1.0)
        ]

        if not indices:
            diagram_bins.append({
                "bin_midpoint": (bin_lower + bin_upper) / 2.0,
                "confidence": 0.0,
                "accuracy": 0.0,
                "count": 0,
            })
            continue

        diagram_bins.append({
            "bin_midpoint": (bin_lower + bin_upper) / 2.0,
            "confidence": sum(probs[i] for i in indices) / len(indices),
            "accuracy": sum(labels[i] for i in indices) / len(indices),
            "count": len(indices),
        })

    return diagram_bins


def platt_scale(prob: float, temperature: float = 1.0) -> float:
    """
    Applies temperature scaling: p_cal = sigma(logit(p) / T).
    """
    if temperature <= 0.0:
        raise ValueError("Temperature must be strictly positive.")
    p = min(max(prob, 1e-7), 1.0 - 1e-7)
    logit = math.log(p / (1.0 - p))
    scaled_logit = logit / temperature
    return 1.0 / (1.0 + math.exp(-scaled_logit))


# Function aliases
compute_expected_calibration_error = compute_ece
compute_adaptive_calibration_error = compute_ace
