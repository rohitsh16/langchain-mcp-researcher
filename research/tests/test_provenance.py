"""
Unit tests for research provenance lineage graph and Rules 1-7 enforcement.
"""
import pytest
from research.ontology.schemas import (
    Claim,
    EvidenceSpan,
    EvidenceType,
    CorrectnessDimension,
    Certificate,
    CertificateDecision,
)
from research.ontology.provenance import LineageGraph, ProvenanceValidator, ProvenanceViolationError


def test_rule_1_theorems_require_spans():
    graph = LineageGraph()
    # Claim stating a mathematical theorem without source span
    claim = Claim(
        claim_id="thm-001",
        text="Theorem 1: Under exchangeability, split conformal coverage is at least 1 - alpha.",
        claim_type=CorrectnessDimension.MATHEMATICAL_VALIDITY,
        evidence_ids=[],  # Missing evidence!
    )
    graph.add_claim(claim)
    violations = ProvenanceValidator.validate(graph)
    assert len(violations) > 0
    assert any("Theorem claims must cite" in v for v in violations)


def test_rule_3_empirical_claim_requires_experiment_id():
    graph = LineageGraph()
    claim = Claim(
        claim_id="emp-001",
        text="Empirical results show model achieves 92% accuracy on GSM8K in experiment.",
        claim_type=CorrectnessDimension.FACTUAL_CORRECTNESS,
        evidence_ids=[],
    )
    graph.add_claim(claim)
    violations = ProvenanceValidator.validate(graph)
    assert any("Empirical claims must link to a valid experiment_id" in v for v in violations)


def test_rule_4_guarantee_separation():
    graph = LineageGraph()
    cert = Certificate(
        certificate_id="cert-test-1",
        output_id="out-test-1",
        correctness_type=CorrectnessDimension.FACTUAL_CORRECTNESS,
        estimated_correctness=0.95,
        certified_risk=0.05,
        confidence_level=None,  # Missing confidence level!
        assumptions=[],
        method="SPLIT_CONFORMAL",
        decision=CertificateDecision.CERTIFY,
    )
    graph.add_certificate(cert)
    violations = ProvenanceValidator.validate(graph)
    assert any("declare both confidence_level and assumptions" in v for v in violations)
