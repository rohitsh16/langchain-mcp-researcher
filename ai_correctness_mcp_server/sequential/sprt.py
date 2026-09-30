"""
Wald's Sequential Probability Ratio Test (SPRT) for streaming verification and early stopping.
"""
from typing import Dict, Any, Optional
import math


class WaldSPRT:
    """
    Sequential Probability Ratio Test for Bernoulli observations (0 = correct, 1 = error).
    Tests H_0: p <= p0 vs H_1: p >= p1.
    """

    def __init__(self, p0: float = 0.05, p1: float = 0.20, alpha: float = 0.05, beta: float = 0.10):
        if not (0.0 < p0 < p1 < 1.0):
            raise ValueError("Must have 0 < p0 < p1 < 1.")
        if not (0.0 < alpha < 1.0 and 0.0 < beta < 1.0):
            raise ValueError("alpha and beta must be in (0, 1).")

        self.p0 = float(p0)
        self.p1 = float(p1)
        self.alpha = float(alpha)
        self.beta = float(beta)

        # Wald boundaries
        self.upper_bound = math.log((1.0 - beta) / alpha)
        self.lower_bound = math.log(beta / (1.0 - alpha))

        # Log likelihood increments for x=1 (error) and x=0 (correct)
        self.log_llr_error = math.log(p1 / p0)
        self.log_llr_correct = math.log((1.0 - p1) / (1.0 - p0))

        self.log_llr: float = 0.0
        self.step_count: int = 0
        self.error_count: int = 0
        self.status: str = "CONTINUE"  # "ACCEPT_H0" | "ACCEPT_H1" | "CONTINUE"

    def reset(self):
        self.log_llr = 0.0
        self.step_count = 0
        self.error_count = 0
        self.status = "CONTINUE"

    def step(self, is_error: bool) -> Dict[str, Any]:
        """
        Incorporate a new observation (True if verification failed / error, False if correct).
        Returns decision status: "CONTINUE", "ACCEPT_H0" (accept low error), or "ACCEPT_H1" (reject system / high error).
        """
        if self.status != "CONTINUE":
            return self.summary()

        self.step_count += 1
        if is_error:
            self.error_count += 1
            self.log_llr += self.log_llr_error
        else:
            self.log_llr += self.log_llr_correct

        if self.log_llr >= self.upper_bound:
            self.status = "ACCEPT_H1"
        elif self.log_llr <= self.lower_bound:
            self.status = "ACCEPT_H0"
        else:
            self.status = "CONTINUE"

        return self.summary()

    def summary(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "step_count": self.step_count,
            "error_count": self.error_count,
            "log_llr": self.log_llr,
            "upper_bound": self.upper_bound,
            "lower_bound": self.lower_bound,
            "empirical_rate": (self.error_count / self.step_count) if self.step_count > 0 else 0.0,
        }
