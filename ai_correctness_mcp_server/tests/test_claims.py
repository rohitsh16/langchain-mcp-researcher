"""
Unit tests for atomic claim extraction and decomposition.
"""
import pytest
from ai_correctness_mcp_server.correctness.claims import extract_atomic_claims


def test_empty_and_whitespace_extraction():
    assert extract_atomic_claims("") == []
    assert extract_atomic_claims("   \n\t  ") == []


def test_basic_claim_decomposition():
    text = (
        "Albert Einstein published special relativity in 1905. "
        "He later developed the general theory of relativity."
    )
    claims = extract_atomic_claims(text)
    assert len(claims) == 2
    assert "Albert Einstein published special relativity in 1905" in [c["text"] for c in claims]
    for c in claims:
        assert c["claim_id"].startswith("claim-")
        assert c["claim_type"] == "FACTUAL_CORRECTNESS"


def test_hedging_prefix_stripping():
    text = "It is believed that the speed of light is constant, and some say that quantum mechanics is non-local."
    claims = extract_atomic_claims(text)
    assert len(claims) >= 2
    texts = [c["text"].lower() for c in claims]
    assert any("speed of light is constant" in t for t in texts)
    # Check that hedging prefix was stripped
    assert not any(t.startswith("it is believed that") for t in texts)


def test_conjunction_clause_splitting():
    text = "The model trained for 50 epochs, but the learning rate was kept constant."
    claims = extract_atomic_claims(text)
    assert len(claims) == 2
    assert any("model trained for 50 epochs" in c["text"] for c in claims)
    assert any("learning rate was kept constant" in c["text"] for c in claims)
