"""
Anytime-valid confidence sequences for sequential monitoring and optional stopping.
References:
- Howard et al. (2021) "Time-uniform, nonparametric, nonasymptotic confidence sequences"
"""
from typing import Dict, Any, List, Optional
import math


class EmpiricalBernsteinSequence:
    """
    Time-uniform confidence sequence for the mean of bounded random variables in [0, 1].
    Valid at all stopping times tau: P(forall t >= 1, mu in [L_t, U_t]) >= 1 - alpha.
    """

    def __init__(self, alpha: float = 0.05, c: float = 0.5):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        self.alpha = float(alpha)
        self.c = float(c)  # scale parameter

        self.t: int = 0
        self.sum_x: float = 0.0
        self.sum_sq_diff: float = 0.0
        self.mean: float = 0.0

    def step(self, x: float) -> Dict[str, float]:
        """
        Process a new observation x in [0, 1] and return updated anytime-valid [lower, upper] bounds.
        """
        x = max(0.0, min(1.0, float(x)))
        self.t += 1
        self.sum_x += x

        # Welford's algorithm for online variance
        old_mean = self.mean
        self.mean = self.sum_x / self.t
        self.sum_sq_diff += (x - old_mean) * (x - self.mean)

        sample_var = (self.sum_sq_diff / self.t) if self.t > 1 else 0.25

        # Howard et al. stitching boundary approximation
        # radius ~ sqrt(2 * (V_t * log(log(t) / alpha) + ...)) / t
        log_term = math.log(max(1.0, math.log(max(2.0, float(self.t)))) + 1.0) - math.log(self.alpha / 2.0)
        v_t = max(0.01, sample_var)

        # Boundary radius
        radius = math.sqrt(2.0 * v_t * log_term / self.t) + (3.0 * log_term / self.t)
        radius = min(1.0, radius)

        lower = max(0.0, self.mean - radius)
        upper = min(1.0, self.mean + radius)

        return {
            "t": self.t,
            "mean": self.mean,
            "lower": lower,
            "upper": upper,
            "radius": radius,
        }
