"""
Unit tests for verification engines (Exact, Evidence, Semantic, Programmatic).
"""
import pytest
from ai_correctness_mcp_server.verification.exact import ExactVerifier
from ai_correctness_mcp_server.verification.evidence import EvidenceVerifier
from ai_correctness_mcp_server.verification.semantic import SemanticVerifier
from ai_correctness_mcp_server.verification.program import ProgrammaticVerifier


def test_exact_verifier():
    verifier = ExactVerifier(case_sensitive=False)
    res = verifier.verify("Paris", "paris")
    assert res.is_verified is True
    assert res.confidence == 1.0

    res_fail = verifier.verify("London", "Paris")
    assert res_fail.is_verified is False
    assert res_fail.confidence == 0.0

    res_no_ctx = verifier.verify("Paris", None)
    assert res_no_ctx.is_verified is False


def test_evidence_verifier():
    verifier = EvidenceVerifier(min_overlap_ratio=0.5)
    context = "The Transformer architecture was introduced in the paper Attention is All You Need in 2017."

    res_match = verifier.verify("Transformer architecture was introduced in 2017", context)
    assert res_match.is_verified is True
    assert res_match.confidence >= 0.5

    res_mismatch = verifier.verify("Convolutional neural networks were invented in 2024", context)
    assert res_mismatch.is_verified is False

    res_empty_ctx = verifier.verify("Claim with no context", "")
    assert res_empty_ctx.is_verified is False


def test_semantic_verifier_polarity():
    verifier = SemanticVerifier(threshold=0.45)
    # Contradiction in polarity
    res = verifier.verify("The earth is round", "The earth is not round")
    assert res.metadata["polarity_match"] is False
    # Verified should fail or have reduced score due to negation mismatch
    assert res.confidence < 0.5


def test_programmatic_verifier():
    verifier = ProgrammaticVerifier()
    res_correct = verifier.verify("15 + 27 = 42")
    assert res_correct.is_verified is True
    assert res_correct.confidence == 1.0

    res_incorrect = verifier.verify("2 * 8 = 17")
    assert res_incorrect.is_verified is False
    assert res_incorrect.confidence == 0.0

    res_unparseable = verifier.verify("Not a math expression")
    assert res_unparseable.is_verified is False
