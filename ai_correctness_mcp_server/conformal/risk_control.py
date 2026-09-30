"""
Conformal Risk Control (CRC) algorithms for bounded loss functions.
References:
- Bates et al. (2021) "Distribution-Free Risk-Controlling Prediction Sets"
- Angelopoulos et al. (2022) "Conformal Risk Control"
"""
from typing import List, Callable, Dict, Any, Optional
import math


class ConformalRiskController:
    """
    Controls expected bounded loss E[loss(lambda)] <= alpha with distribution-free finite-sample guarantees.
    Assumes loss(lambda) is non-increasing as lambda increases (e.g., higher threshold = more stringent acceptance).
    """

    def __init__(self, target_risk: float = 0.05, max_loss: float = 1.0):
        if not (0.0 < target_risk < 1.0):
            raise ValueError("target_risk must be in (0, 1).")
        if max_loss <= 0.0:
            raise ValueError("max_loss must be positive.")
        self.target_risk = float(target_risk)
        self.max_loss = float(max_loss)
        self.best_lambda: Optional[float] = None
        self.empirical_risk: float = 0.0
        self.n_cal: int = 0

    def fit(
        self,
        candidate_lambdas: List[float],
        loss_fn: Callable[[float], List[float]],
    ) -> "ConformalRiskController":
        """
        candidate_lambdas: list of candidate parameter values, sorted ascending.
        loss_fn(lam): returns list of sample losses for all calibration instances under parameter lam.
                      Each loss must be in [0, max_loss].
        """
        sorted_lambdas = sorted(candidate_lambdas)
        if not sorted_lambdas:
            raise ValueError("candidate_lambdas must not be empty.")

        n = None
        selected_lambda = None
        selected_empirical_risk = 0.0

        for lam in sorted_lambdas:
            losses = loss_fn(lam)
            if n is None:
                n = len(losses)
                if n == 0:
                    raise ValueError("Loss function returned empty losses.")
                self.n_cal = n

            # Compute empirical mean risk
            r_hat = sum(losses) / n

            # Standard Conformal Risk Control finite-sample inflation:
            # (n / (n + 1)) * R_hat + (max_loss / (n + 1)) <= alpha
            inflated_risk = (n / (n + 1)) * r_hat + (self.max_loss / (n + 1))

            if inflated_risk <= self.target_risk:
                selected_lambda = lam
                selected_empirical_risk = r_hat
                break

        if selected_lambda is None:
            # If no lambda satisfies the bound, take the most conservative (last)
            selected_lambda = sorted_lambdas[-1]
            last_losses = loss_fn(selected_lambda)
            selected_empirical_risk = sum(last_losses) / len(last_losses)

        self.best_lambda = selected_lambda
        self.empirical_risk = selected_empirical_risk
        return self

    def get_parameter(self) -> float:
        if self.best_lambda is None:
            raise RuntimeError("ConformalRiskController has not been fitted.")
        return self.best_lambda


def hoeffding_bentkus_p_value(r_hat: float, n: int, alpha: float, b: float = 1.0) -> float:
    """
    Computes Hoeffding-Bentkus p-value for testing H_0: E[loss] > alpha vs H_1: E[loss] <= alpha.
    """
    if r_hat >= alpha:
        return 1.0

    # Hoeffding bound
    # P(R_hat <= r_hat) <= exp(-2 * n * ((alpha - r_hat) / b)^2)
    h_bound = math.exp(-2.0 * n * ((alpha - r_hat) / b) ** 2)

    # Bentkus bound (Poisson binomial bound approximation)
    if r_hat == 0:
        b_bound = (1.0 - alpha / b) ** n
    else:
        # Relative entropy / Kullback-Leibler divergence D(r_hat/b || alpha/b)
        p = r_hat / b
        q = alpha / b
        kl = p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))
        b_bound = math.e * math.exp(-n * kl)

    return min(1.0, min(h_bound, b_bound))
