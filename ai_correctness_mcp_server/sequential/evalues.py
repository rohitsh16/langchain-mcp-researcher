"""
E-values and test martingales for sequential hypothesis testing and verification.
"""
from typing import Dict, Any, List
import math


class EValueMartingale:
    """
    Test martingale tracking sequential evidence against a null hypothesis H_0.
    Under H_0, E[E_t | F_{t-1}] <= 1.
    Rejects H_0 at level alpha if M_t >= 1 / alpha (Ville's inequality).
    """

    def __init__(self, alpha: float = 0.05, h0_rate: float = 0.05, betting_fraction: float = 0.5):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        self.alpha = float(alpha)
        self.h0_rate = float(h0_rate)
        self.betting_fraction = float(betting_fraction)  # Kelly-style betting fraction lambda

        self.threshold = 1.0 / self.alpha
        self.martingale_value: float = 1.0
        self.t: int = 0
        self.rejected: bool = False

    def step(self, is_error: bool) -> Dict[str, Any]:
        """
        Process an observation (True = error, False = correct).
        Constructs an e-value for testing H_0: error_rate <= h0_rate.
        """
        self.t += 1
        y = 1.0 if is_error else 0.0

        # Kelly betting e-value against H0 mean mu0:
        # E_t = 1 + lambda * (Y_t - mu0)
        # We need E_t >= 0, so lambda <= 1 / (1 - mu0) and lambda <= 1 / mu0
        lam = min(self.betting_fraction, 0.9 / max(self.h0_rate, 1.0 - self.h0_rate))
        e_val = max(0.01, 1.0 + lam * (y - self.h0_rate))

        self.martingale_value *= e_val
        if self.martingale_value >= self.threshold:
            self.rejected = True

        return {
            "t": self.t,
            "e_value": e_val,
            "martingale_value": self.martingale_value,
            "threshold": self.threshold,
            "rejected_h0": self.rejected,
        }
