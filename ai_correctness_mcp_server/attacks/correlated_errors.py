"""
Simulates correlated errors across multiple verifiers or ensemble members.
"""
from typing import List, Dict, Any
import random


def simulate_correlated_failures(
    base_accuracy: float = 0.85,
    correlation: float = 0.60,
    n_samples: int = 100,
    n_verifiers: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Simulates verification decisions across multiple verifiers where failure modes are correlated.
    Tests whether ensemble voting or naive independence assumption underestimates joint risk.
    """
    rng = random.Random(seed)
    verifier_decisions: List[List[int]] = [[] for _ in range(n_verifiers)]
    ground_truth: List[int] = [1] * n_samples  # 1 = correct claim

    for _ in range(n_samples):
        # Latent difficulty factor Z ~ Normal(0, 1)
        z = rng.gauss(0, 1)
        # If z is very low, input is hard and triggers correlated failure across verifiers
        for v in range(n_verifiers):
            noise = rng.gauss(0, 1)
            latent_score = (correlation**0.5) * z + ((1.0 - correlation) ** 0.5) * noise
            # If latent score is above threshold, verifier predicts correctly (1)
            # Calibrate threshold to base_accuracy
            threshold = -0.5 if base_accuracy > 0.8 else 0.0
            pred = 1 if latent_score > threshold else 0
            verifier_decisions[v].append(pred)

    # Compute individual accuracies and majority vote accuracy
    individual_accs = [sum(dec) / n_samples for dec in verifier_decisions]
    majority_preds = []
    for i in range(n_samples):
        votes = sum(verifier_decisions[v][i] for v in range(n_verifiers))
        majority_preds.append(1 if votes > (n_verifiers / 2) else 0)

    majority_acc = sum(majority_preds) / n_samples

    return {
        "individual_accuracies": individual_accs,
        "majority_accuracy": majority_acc,
        "correlation": correlation,
        "n_samples": n_samples,
        "correlated_failure_rate": 1.0 - majority_acc,
    }
