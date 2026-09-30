"""
Token-level and predictive entropy estimation.
"""
import math
from typing import List


def compute_token_entropy(logprobs: List[float]) -> float:
    """
    Computes Shannon entropy from token log-probabilities:
    H = - sum_i p_i * log(p_i)
    """
    if not logprobs:
        return 0.0

    probs = [math.exp(lp) for lp in logprobs]
    total = sum(probs)
    if total <= 0:
        return 0.0

    norm_probs = [p / total for p in probs]
    entropy = -sum(p * math.log(p) for p in norm_probs if p > 1e-12)
    return float(entropy)


def compute_predictive_entropy(probabilities: List[float]) -> float:
    """
    Computes entropy over a discrete probability distribution.
    """
    total = sum(probabilities)
    if total <= 0:
        return 0.0
    norm = [p / total for p in probabilities if p > 0]
    return float(-sum(p * math.log(p) for p in norm if p > 1e-12))
