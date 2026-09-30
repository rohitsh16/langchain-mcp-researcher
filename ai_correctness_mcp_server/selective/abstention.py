"""
Selective prediction and abstention decision rules.
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class SelectiveDecision:
    accepted: bool
    confidence: float
    threshold: float
    action: str  # "ACCEPT" or "ABSTAIN"
    metadata: Dict[str, Any]


class SelectivePredictor:
    """
    Implements confidence-based selective prediction (rejection / abstention).
    Accepts if confidence >= threshold, otherwise abstains.
    """

    def __init__(self, threshold: float = 0.8):
        self.threshold = float(threshold)

    def decide(self, confidence: float, metadata: Optional[Dict[str, Any]] = None) -> SelectiveDecision:
        conf = float(confidence)
        accepted = conf >= self.threshold
        return SelectiveDecision(
            accepted=accepted,
            confidence=conf,
            threshold=self.threshold,
            action="ACCEPT" if accepted else "ABSTAIN",
            metadata=metadata or {}
        )

    def decide_batch(self, confidences: List[float]) -> List[SelectiveDecision]:
        return [self.decide(c) for c in confidences]

    @classmethod
    def calibrate_threshold_for_coverage(cls, confidences: List[float], target_coverage: float) -> "SelectivePredictor":
        """
        Choose threshold tau such that approximately target_coverage fraction of samples are accepted.
        """
        if not (0.0 < target_coverage <= 1.0):
            raise ValueError("target_coverage must be in (0, 1].")
        if not confidences:
            return cls(threshold=0.5)

        sorted_confs = sorted(confidences, reverse=True)
        idx = min(len(sorted_confs) - 1, int(round(target_coverage * len(sorted_confs))) - 1)
        threshold = sorted_confs[max(0, idx)]
        return cls(threshold=threshold)

    @classmethod
    def calibrate_threshold_for_risk(
        cls, confidences: List[float], errors: List[float], max_risk: float
    ) -> "SelectivePredictor":
        """
        Choose smallest threshold tau such that empirical risk among accepted samples <= max_risk.
        """
        if len(confidences) != len(errors):
            raise ValueError("confidences and errors must have same length.")
        if not confidences:
            return cls(threshold=0.5)

        # Sort descending by confidence
        paired = sorted(zip(confidences, errors), key=lambda x: x[0], reverse=True)
        cum_err = 0.0
        best_threshold = 1.0

        for k, (c, e) in enumerate(paired, start=1):
            cum_err += float(e)
            risk = cum_err / k
            if risk <= max_risk:
                best_threshold = c
            else:
                # Once risk exceeds max_risk, earlier threshold was our limit
                break

        return cls(threshold=best_threshold)
