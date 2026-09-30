"""
Unit tests for uncertainty quantification and Kuhn semantic entropy.
"""
import pytest
import math
from ai_correctness_mcp_server.uncertainty.entropy import compute_token_entropy, compute_predictive_entropy
from ai_correctness_mcp_server.uncertainty.semantic_entropy import (
    compute_semantic_entropy,
    cluster_semantic_generations,
)
from ai_correctness_mcp_server.uncertainty.ensembles import compute_ensemble_agreement


def test_token_and_predictive_entropy():
    # Token log-probabilities for uniform distribution: log(0.5)
    logprobs = [math.log(0.5), math.log(0.5)]
    ent = compute_token_entropy(logprobs)
    assert math.isclose(ent, math.log(2), rel_tol=1e-5)

    # Deterministic outcome: logprob 0 for top, -50 for rest -> entropy ~ 0
    det_logprobs = [0.0, -50.0]
    assert math.isclose(compute_token_entropy(det_logprobs), 0.0, abs_tol=1e-5)

    # Predictive entropy across discrete probabilities
    pred_ent = compute_predictive_entropy([0.5, 0.5])
    assert math.isclose(pred_ent, math.log(2), rel_tol=1e-5)
    det_pred = compute_predictive_entropy([1.0, 0.0])
    assert math.isclose(det_pred, 0.0, abs_tol=1e-6)


def test_semantic_entropy_identical_generations():
    samples = [
        "The capital of Japan is Tokyo.",
        "Tokyo is the capital of Japan.",
        "Japan's capital city is Tokyo.",
    ]
    res = compute_semantic_entropy(samples)
    assert res["num_clusters"] == 1
    assert math.isclose(res["semantic_entropy"], 0.0, abs_tol=1e-5)


def test_semantic_entropy_divergent_generations():
    samples = [
        "The winner of the match was Team Alpha.",
        "Team Beta won the championship game.",
    ]
    res = compute_semantic_entropy(samples)
    assert res["num_clusters"] == 2
    assert res["semantic_entropy"] > 0.0


def test_ensemble_agreement():
    gens = ["Paris", "paris", "Paris, France", "London"]
    res = compute_ensemble_agreement(gens)
    assert res["majority_answer"].lower().startswith("paris")
    assert res["agreement_ratio"] >= 0.5
