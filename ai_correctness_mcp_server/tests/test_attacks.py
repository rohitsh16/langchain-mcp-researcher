"""
Unit tests for adversarial attack simulations and verifier exploitation.
"""
import pytest
from ai_correctness_mcp_server.attacks.correlated_errors import simulate_correlated_failures
from ai_correctness_mcp_server.attacks.verifier_exploitation import (
    generate_superficial_exploit,
    evaluate_verifier_susceptibility,
)
from ai_correctness_mcp_server.attacks.evidence_conflict import (
    create_conflicting_evidence_pair,
    evaluate_conflict_awareness,
)
from ai_correctness_mcp_server.attacks.distribution_shift import (
    generate_shifted_data,
    evaluate_conformal_breakdown,
)
from ai_correctness_mcp_server.conformal.split import SplitConformalPredictor
from ai_correctness_mcp_server.verification.evidence import EvidenceVerifier


def test_correlated_failures_simulation():
    res = simulate_correlated_failures(base_accuracy=0.85, correlation=0.7, n_samples=50)
    assert "individual_accuracies" in res
    assert "majority_accuracy" in res
    assert 0.0 <= res["majority_accuracy"] <= 1.0


def test_superficial_exploit_generation():
    exploit = generate_superficial_exploit("Transformers use self-attention", "non-standard hidden warp gates")
    assert "adversarial_claim" in exploit
    assert "fake_evidence" in exploit

    verifier = EvidenceVerifier()
    suscept = evaluate_verifier_susceptibility(verifier, [exploit])
    assert "susceptibility_rate" in suscept


def test_conflicting_evidence_evaluation():
    conflict = create_conflicting_evidence_pair("gravitational constant", "G = 6.674e-11", "G = 9.81")
    assert "combined_context" in conflict

    verifier = EvidenceVerifier(min_overlap_ratio=0.7)
    res = evaluate_conflict_awareness(verifier, [conflict])
    assert "blind_acceptance_rate" in res


def test_conformal_breakdown_under_shift():
    ref_scores = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] * 10
    predictor = SplitConformalPredictor(alpha=0.1).fit(ref_scores)

    # Shift scores upward (higher error)
    shifted_scores = generate_shifted_data(ref_scores, shift_type="mean_shift", shift_magnitude=-0.3)
    breakdown = evaluate_conformal_breakdown(predictor, ref_scores, shifted_scores)
    assert "coverage_gap" in breakdown
