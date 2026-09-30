"""
Split conformal prediction algorithms.
"""
from typing import List, Tuple, Optional
import math


def compute_conformal_quantile(nonconformity_scores: List[float], alpha: float) -> float:
    """
    Computes the standard finite-sample split conformal quantile:
    q_hat = Quantile({s_i}, ceil((n + 1) * (1 - alpha)) / n)

    Guarantees P(S_{n+1} <= q_hat) >= 1 - alpha under exchangeability.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("Significance level alpha must be in (0, 1).")
    n = len(nonconformity_scores)
    if n == 0:
        raise ValueError("Cannot compute conformal quantile with empty scores.")

    sorted_scores = sorted(float(s) for s in nonconformity_scores)

    # Required rank (1-indexed)
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        # In small sample sizes where (n+1)(1-alpha) > n, conservative bound is infinity (or maximum possible score)
        return float("inf")

    # 0-indexed position
    idx = min(n - 1, rank - 1)
    return sorted_scores[idx]


class SplitConformalPredictor:
    """
    Split conformal predictor for classification confidence scores or nonconformity scores.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = float(alpha)
        self.quantile: Optional[float] = None
        self.n_cal: int = 0

    def fit(self, nonconformity_scores: List[float]) -> "SplitConformalPredictor":
        """
        Fit on calibration nonconformity scores (e.g., 1 - confidence_for_true_label or verifier error).
        """
        self.quantile = compute_conformal_quantile(nonconformity_scores, self.alpha)
        self.n_cal = len(nonconformity_scores)
        return self

    def predict_inclusion(self, score: float) -> bool:
        """
        Returns True if the nonconformity score falls within the conformal prediction region.
        """
        if self.quantile is None:
            raise RuntimeError("SplitConformalPredictor must be fitted before predict_inclusion.")
        return float(score) <= self.quantile

    def get_certified_upper_bound(self) -> float:
        """Returns the certified quantile threshold."""
        if self.quantile is None:
            raise RuntimeError("Predictor is not fitted.")
        return self.quantile
