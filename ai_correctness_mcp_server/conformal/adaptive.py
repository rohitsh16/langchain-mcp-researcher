"""
Group-conditional and adaptive conformal prediction (Mondrian conformal prediction).
"""
from typing import Dict, List, Any, Optional
from .split import compute_conformal_quantile


class MondrianConformalPredictor:
    """
    Computes group-stratified conformal prediction quantiles to guarantee coverage
    conditionally across groups: P(S <= q_g | G = g) >= 1 - alpha.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = float(alpha)
        self.group_quantiles: Dict[str, float] = {}
        self.marginal_quantile: Optional[float] = None

    def fit(self, scores: List[float], groups: List[str]) -> "MondrianConformalPredictor":
        if len(scores) != len(groups):
            raise ValueError("Scores and groups must have the same length.")
        if not scores:
            raise ValueError("Scores cannot be empty.")

        # Compute marginal quantile as fallback
        self.marginal_quantile = compute_conformal_quantile(scores, self.alpha)

        # Compute group-specific quantiles
        grouped_scores: Dict[str, List[float]] = {}
        for s, g in zip(scores, groups):
            grouped_scores.setdefault(str(g), []).append(s)

        for g, g_scores in grouped_scores.items():
            self.group_quantiles[g] = compute_conformal_quantile(g_scores, self.alpha)

        return self

    def predict_inclusion(self, score: float, group: Optional[str] = None) -> bool:
        if self.marginal_quantile is None:
            raise RuntimeError("MondrianConformalPredictor must be fitted before prediction.")

        thresh = self.group_quantiles.get(str(group), self.marginal_quantile)
        return float(score) <= thresh

    def get_group_quantile(self, group: Optional[str] = None) -> float:
        if group is not None and str(group) in self.group_quantiles:
            return self.group_quantiles[str(group)]
        if self.marginal_quantile is not None:
            return self.marginal_quantile
        raise RuntimeError("Predictor is not fitted.")
