"""
Ontology package for research data models and provenance contracts.
"""
from research.ontology.schemas import (
    CorrectnessDimension,
    CertificateDecision,
    EvidenceType,
    VerificationStatus,
    ResearchQuestion,
    PaperRecord,
    EvidenceSpan,
    Claim,
    VerificationResult,
    Certificate,
)
from research.ontology.provenance import ProvenanceGraph, ProvenanceRuleValidator

__all__ = [
    "CorrectnessDimension",
    "CertificateDecision",
    "EvidenceType",
    "VerificationStatus",
    "ResearchQuestion",
    "PaperRecord",
    "EvidenceSpan",
    "Claim",
    "VerificationResult",
    "Certificate",
    "ProvenanceGraph",
    "ProvenanceRuleValidator",
]
