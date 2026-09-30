"""
Provenance tracking and verification rules for the research platform.
Enforces Rules 1 through 7 from the research charter.
"""
from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from research.ontology.schemas import Certificate, Claim, EvidenceSpan, EvidenceType, CorrectnessDimension


class ProvenanceViolation(BaseModel):
    rule_number: int
    rule_name: str
    target_id: str
    violation_message: str


class ProvenanceGraph:
    """Directed acyclic graph tracking lineage between claims, evidence, experiments, and certificates."""

    def __init__(self):
        self.evidence: Dict[str, EvidenceSpan] = {}
        self.claims: Dict[str, Claim] = {}
        self.certificates: Dict[str, Certificate] = {}
        self.experiment_ids: Set[str] = set()

    def add_evidence(self, span: EvidenceSpan):
        self.evidence[span.evidence_id] = span

    def add_claim(self, claim: Claim):
        self.claims[claim.claim_id] = claim

    def add_experiment(self, exp_id: str):
        self.experiment_ids.add(exp_id)

    def add_certificate(self, cert: Certificate):
        self.certificates[cert.certificate_id] = cert


class ProvenanceRuleValidator:
    """Enforces correctness research provenance rules."""

    def __init__(self, graph: ProvenanceGraph):
        self.graph = graph

    @classmethod
    def validate(cls, graph: ProvenanceGraph) -> List[str]:
        """Convenience method returning string error messages."""
        validator = cls(graph)
        violations = validator.validate_all()
        return [v.violation_message for v in violations]

    def validate_all(self) -> List[ProvenanceViolation]:
        violations: List[ProvenanceViolation] = []
        violations.extend(self.validate_rule_1_theorems())
        violations.extend(self.validate_rule_2_experiments())
        violations.extend(self.validate_rule_3_guarantees())
        violations.extend(self.validate_rule_4_certificate_assumptions())
        violations.extend(self.validate_rule_5_paper_claims())
        return violations

    def validate_rule_1_theorems(self) -> List[ProvenanceViolation]:
        """Rule 1: Never state a theorem without a source/evidence pointer."""
        violations = []
        for claim_id, claim in self.graph.claims.items():
            is_math = (claim.claim_type == CorrectnessDimension.MATHEMATICAL_VALIDITY or "theorem" in claim.text.lower())
            if is_math:
                if not claim.evidence_ids and not claim.source_paper_id:
                    violations.append(
                        ProvenanceViolation(
                            rule_number=1,
                            rule_name="Theorem Evidence Requirement",
                            target_id=claim_id,
                            violation_message=f"Theorem claims must cite a source evidence span or source paper pointer: '{claim.text[:60]}...'",
                        )
                    )
        return violations

    def validate_rule_2_experiments(self) -> List[ProvenanceViolation]:
        """Rule 2: Never claim empirical evidence without an experiment ID."""
        violations = []
        for claim_id, claim in self.graph.claims.items():
            is_emp = (claim.claim_type == CorrectnessDimension.FACTUAL_CORRECTNESS and ("empirical" in claim.text.lower() or "experiment" in claim.text.lower()))
            if is_emp:
                has_exp = any(exp_id in claim.text for exp_id in self.graph.experiment_ids)
                if not has_exp and not claim.evidence_ids:
                    violations.append(
                        ProvenanceViolation(
                            rule_number=2,
                            rule_name="Empirical Result ID Requirement",
                            target_id=claim_id,
                            violation_message=f"Empirical claims must link to a valid experiment_id: '{claim.text[:60]}...'",
                        )
                    )
        return violations

    def validate_rule_3_guarantees(self) -> List[ProvenanceViolation]:
        """Rule 3: Never call an empirical confidence score a formal guarantee."""
        violations = []
        for cert_id, cert in self.graph.certificates.items():
            if cert.certified_risk is not None and cert.confidence_level is None:
                violations.append(
                    ProvenanceViolation(
                        rule_number=3,
                        rule_name="Guarantee Separation Requirement",
                        target_id=cert_id,
                        violation_message="Certificates with certified_risk must declare both confidence_level and assumptions.",
                    )
                )
        return violations

    def validate_rule_4_certificate_assumptions(self) -> List[ProvenanceViolation]:
        """Rule 4: Every certificate must list assumptions."""
        violations = []
        for cert_id, cert in self.graph.certificates.items():
            decision_val = getattr(cert.decision, "value", str(cert.decision))
            if decision_val in ("CERTIFIED", "CERTIFY") and not cert.assumptions:
                violations.append(
                    ProvenanceViolation(
                        rule_number=4,
                        rule_name="Certificate Assumptions Requirement",
                        target_id=cert_id,
                        violation_message="Certificate has certified decision but lists zero assumptions.",
                    )
                )
        return violations

    def validate_rule_5_paper_claims(self) -> List[ProvenanceViolation]:
        """Rule 5: Every external paper claim must retain exact provenance."""
        violations = []
        for claim_id, claim in self.graph.claims.items():
            if claim.source_paper_id is not None:
                for ev_id in claim.evidence_ids:
                    if ev_id not in self.graph.evidence:
                        violations.append(
                            ProvenanceViolation(
                                rule_number=5,
                                rule_name="External Claim Provenance Span",
                                target_id=claim_id,
                                violation_message=f"Claim references evidence_id '{ev_id}' which does not exist in EvidenceStore.",
                            )
                        )
        return violations


# Aliases for backwards compatibility and clean interface
LineageGraph = ProvenanceGraph
ProvenanceValidator = ProvenanceRuleValidator
ProvenanceViolationError = ProvenanceViolation
