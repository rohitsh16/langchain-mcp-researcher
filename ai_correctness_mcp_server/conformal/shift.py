"""
Distribution shift testing and weighted conformal prediction.
"""
from typing import List, Tuple, Dict, Any, Optional
import math


def compute_weighted_conformal_quantile(
    scores: List[float],
    weights: List[float],
    test_weight: float,
    alpha: float,
) -> float:
    """
    Computes weighted split conformal quantile under covariate shift (Tibshirani et al. 2019).
    P_test(S_{n+1} <= q) >= 1 - alpha.
    """
    if len(scores) != len(weights):
        raise ValueError("scores and weights must have identical length.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    n = len(scores)
    if n == 0:
        raise ValueError("scores must not be empty.")

    # Sort scores with corresponding weights
    paired = sorted(zip(scores, weights), key=lambda x: x[0])
    total_weight = sum(weights) + float(test_weight)
    if total_weight <= 0:
        raise ValueError("Total weights must be strictly positive.")

    target = 1.0 - alpha
    cum_w = 0.0

    for s, w in paired:
        cum_w += float(w) / total_weight
        if cum_w >= target:
            return s

    return paired[-1][0]


def kolmogorov_smirnov_2sample(data1: List[float], data2: List[float]) -> Tuple[float, float]:
    """
    Computes 2-sample Kolmogorov-Smirnov test statistic D and asymptotic p-value
    for testing H_0: data1 and data2 come from the same continuous distribution.
    """
    n1 = len(data1)
    n2 = len(data2)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    s1 = sorted(data1)
    s2 = sorted(data2)

    # Merge unique values
    all_vals = sorted(set(s1 + s2))

    idx1 = 0
    idx2 = 0
    max_d = 0.0

    for val in all_vals:
        while idx1 < n1 and s1[idx1] <= val:
            idx1 += 1
        while idx2 < n2 and s2[idx2] <= val:
            idx2 += 1

        cdf1 = idx1 / n1
        cdf2 = idx2 / n2
        diff = abs(cdf1 - cdf2)
        if diff > max_d:
            max_d = diff

    # Asymptotic p-value approximation via Kolmogorov distribution
    en = math.sqrt((n1 * n2) / (n1 + n2))
    lambda_val = (en + 0.12 + 0.11 / en) * max_d

    # Sum Kolmogorov series
    p_val = 0.0
    for j in range(1, 101):
        term = 2.0 * ((-1) ** (j - 1)) * math.exp(-2.0 * (j ** 2) * (lambda_val ** 2))
        p_val += term
        if abs(term) < 1e-8:
            break

    p_val = max(0.0, min(1.0, p_val))
    return max_d, p_val


class DistributionShiftDetector:
    """
    Detects distribution shift between calibration and test data using 2-sample KS test.
    """

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = float(significance_level)

    def test_shift(self, calibration_scores: List[float], test_scores: List[float]) -> Dict[str, Any]:
        d_stat, p_val = kolmogorov_smirnov_2sample(calibration_scores, test_scores)
        shifted = p_val < self.significance_level
        return {
            "statistic": d_stat,
            "p_value": p_val,
            "shifted": shifted,
            "significance_level": self.significance_level,
            "n_calibration": len(calibration_scores),
            "n_test": len(test_scores),
        }
