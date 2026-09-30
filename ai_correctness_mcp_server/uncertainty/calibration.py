"""
Calibration algorithms for model confidence scores.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple
import math


class Calibrator(ABC):
    """Abstract base class for probability calibrators."""

    @abstractmethod
    def fit(self, scores: List[float], labels: List[int]) -> "Calibrator":
        """Fit calibrator parameters on validation scores and binary labels."""
        pass

    @abstractmethod
    def calibrate(self, score: float) -> float:
        """Calibrate a single probability or score."""
        pass

    def calibrate_batch(self, scores: List[float]) -> List[float]:
        """Calibrate a sequence of scores."""
        return [self.calibrate(s) for s in scores]


class TemperatureScaling(Calibrator):
    """
    Temperature scaling for uncalibrated confidence/probability scores.
    Uses log-odds parameterization: calibrated_p = sigma(logit(p) / T).
    """

    def __init__(self, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError("Temperature must be strictly positive.")
        self.temperature = float(temperature)

    def fit(self, scores: List[float], labels: List[int], lr: float = 0.05, max_iter: int = 200) -> "TemperatureScaling":
        """
        Optimize temperature using NLL via 1D gradient descent / grid search.
        """
        if len(scores) != len(labels):
            raise ValueError("Scores and labels must have the same length.")
        if not scores:
            return self

        # Clip scores to avoid log(0) or division by zero
        eps = 1e-6
        clipped_scores = [max(eps, min(1.0 - eps, float(s))) for s in scores]
        logits = [math.log(s / (1.0 - s)) for s in clipped_scores]

        # Grid search over reasonable range [0.1, 10.0] followed by refinement
        best_t = 1.0
        best_nll = float("inf")

        candidate_temps = [0.1 + 0.05 * i for i in range(200)]  # 0.1 to 10.0
        for t in candidate_temps:
            nll = 0.0
            for z, y in zip(logits, labels):
                p = 1.0 / (1.0 + math.exp(-z / t))
                p = max(eps, min(1.0 - eps, p))
                nll -= y * math.log(p) + (1.0 - y) * math.log(1.0 - p)
            if nll < best_nll:
                best_nll = nll
                best_t = t

        self.temperature = best_t
        return self

    def calibrate(self, score: float) -> float:
        eps = 1e-6
        p = max(eps, min(1.0 - eps, float(score)))
        logit = math.log(p / (1.0 - p))
        scaled_logit = logit / self.temperature
        # Numerically stable sigmoid
        if scaled_logit >= 0:
            return 1.0 / (1.0 + math.exp(-scaled_logit))
        else:
            exp_val = math.exp(scaled_logit)
            return exp_val / (1.0 + exp_val)


class IsotonicCalibrator(Calibrator):
    """
    Non-parametric monotonic calibrator based on Pool Adjacent Violators Algorithm (PAVA).
    """

    def __init__(self):
        self.thresholds: List[float] = []
        self.calibrated_values: List[float] = []

    def fit(self, scores: List[float], labels: List[int]) -> "IsotonicCalibrator":
        if len(scores) != len(labels):
            raise ValueError("Scores and labels must have the same length.")
        if not scores:
            return self

        # Sort by score
        paired = sorted(zip(scores, labels), key=lambda x: x[0])
        # PAVA implementation
        # Each block is [weight, sum_labels, min_score, max_score]
        blocks: List[List[float]] = []
        for s, y in paired:
            # [weight, sum_y, score]
            blocks.append([1.0, float(y), float(s)])

            while len(blocks) >= 2:
                prev = blocks[-2]
                curr = blocks[-1]
                mean_prev = prev[1] / prev[0]
                mean_curr = curr[1] / curr[0]

                if mean_prev >= mean_curr:
                    # Violates monotonicity; pool blocks
                    pooled = [prev[0] + curr[0], prev[1] + curr[1], curr[2]]
                    blocks.pop()
                    blocks.pop()
                    blocks.append(pooled)
                else:
                    break

        self.thresholds = [b[2] for b in blocks]
        self.calibrated_values = [b[1] / b[0] for b in blocks]
        return self

    def calibrate(self, score: float) -> float:
        if not self.thresholds:
            return max(0.0, min(1.0, float(score)))

        # Monotonic piecewise constant / linear interpolation
        s = float(score)
        if s <= self.thresholds[0]:
            return self.calibrated_values[0]
        if s >= self.thresholds[-1]:
            return self.calibrated_values[-1]

        # Find interval
        for i in range(len(self.thresholds) - 1):
            if self.thresholds[i] <= s <= self.thresholds[i + 1]:
                span = self.thresholds[i + 1] - self.thresholds[i]
                if span == 0:
                    return self.calibrated_values[i]
                alpha = (s - self.thresholds[i]) / span
                return (1.0 - alpha) * self.calibrated_values[i] + alpha * self.calibrated_values[i + 1]

        return self.calibrated_values[-1]
