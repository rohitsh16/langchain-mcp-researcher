"""
Fluent builder for constructing and issuing certificates.
"""
from typing import List, Optional, Dict, Any
import hashlib
import uuid
from .schema import OutputCorrectnessCertificate
from .validator import CertificateValidator


class CertificateBuilder:
    """
    Constructs an OutputCorrectnessCertificate step-by-step.
    """

    def __init__(self, target_query: str, target_output: str, model_id: str = "default_model"):
        self.cert_id = f"cert-{uuid.uuid4().hex[:12]}"
        self.target_query = target_query
        self.target_output = target_output
        self.target_model = model_id
        self.dimensions: List[str] = []
        self.claims_total: int = 0
        self.claims_verified: int = 0
        self.evidence_sources: List[str] = []
        self.assumptions: List[str] = []
        self.limitations: List[str] = []
        self.verifier_names: List[str] = []
        self.estimated_correctness: float = 0.0
        self.certified_risk: Optional[float] = None
        self.confidence_level: Optional[float] = None
        self.calibration_dataset_size: Optional[int] = None
        self.calibration_ece: Optional[float] = None
        self.guarantee_type: str = "EMPIRICAL_POINT_ESTIMATE"
        self.decision: str = "INCONCLUSIVE_INSUFFICIENT_EVIDENCE"
        self.metadata: Dict[str, Any] = {}

    def with_claims_summary(self, total: int, verified: int) -> "CertificateBuilder":
        self.claims_total = int(total)
        self.claims_verified = int(verified)
        if self.claims_total > 0:
            self.estimated_correctness = self.claims_verified / self.claims_total
        return self

    def with_verifiers(self, verifiers: List[str]) -> "CertificateBuilder":
        self.verifier_names.extend(verifiers)
        return self

    def with_dimensions(self, dimensions: List[str]) -> "CertificateBuilder":
        self.dimensions.extend(dimensions)
        return self

    def with_evidence_sources(self, sources: List[str]) -> "CertificateBuilder":
        self.evidence_sources.extend(sources)
        return self

    def with_conformal_guarantee(
        self,
        certified_risk: float,
        confidence_level: float,
        calibration_size: int,
        assumptions: Optional[List[str]] = None,
        ece: Optional[float] = None,
    ) -> "CertificateBuilder":
        self.guarantee_type = "CONFORMAL_RISK_CONTROL"
        self.certified_risk = float(certified_risk)
        self.confidence_level = float(confidence_level)
        self.calibration_dataset_size = int(calibration_size)
        if ece is not None:
            self.calibration_ece = float(ece)
        if assumptions:
            self.assumptions.extend(assumptions)
        else:
            self.assumptions.append("Exchangeability between calibration and test distributions (I.I.D. assumption).")
        return self

    def with_assumptions(self, assumptions: List[str]) -> "CertificateBuilder":
        self.assumptions.extend(assumptions)
        return self

    def with_limitations(self, limitations: List[str]) -> "CertificateBuilder":
        self.limitations.extend(limitations)
        return self

    def build(self) -> OutputCorrectnessCertificate:
        # Determine decision automatically if not set
        if self.decision == "INCONCLUSIVE_INSUFFICIENT_EVIDENCE":
            if self.claims_total == 0:
                self.decision = "INCONCLUSIVE_INSUFFICIENT_EVIDENCE"
            elif self.claims_verified == self.claims_total:
                if self.guarantee_type == "CONFORMAL_RISK_CONTROL" and self.certified_risk is not None and self.certified_risk <= 0.10:
                    self.decision = "CERTIFIED"
                else:
                    self.decision = "PARTIALLY_CERTIFIED"
            elif self.claims_verified > 0:
                self.decision = "PARTIALLY_CERTIFIED"
            else:
                self.decision = "REJECTED_RISK_EXCEEDED"

        # Compute verifier fingerprint
        verifier_str = "|".join(sorted(self.verifier_names)) or "unspecified_verifier"
        fingerprint = hashlib.sha256(verifier_str.encode("utf-8")).hexdigest()[:16]

        cert = OutputCorrectnessCertificate(
            certificate_id=self.cert_id,
            target_model=self.target_model,
            target_query=self.target_query,
            target_output=self.target_output,
            decision=self.decision,
            guarantee_type=self.guarantee_type,
            estimated_correctness=round(self.estimated_correctness, 4),
            certified_risk=self.certified_risk,
            confidence_level=self.confidence_level,
            calibration_dataset_size=self.calibration_dataset_size,
            calibration_ece=self.calibration_ece,
            dimensions_verified=sorted(set(self.dimensions)),
            claims_total=self.claims_total,
            claims_verified=self.claims_verified,
            evidence_sources=self.evidence_sources,
            assumptions=sorted(set(self.assumptions)),
            limitations=sorted(set(self.limitations)),
            verifier_fingerprint=fingerprint,
            metadata=self.metadata,
        )

        CertificateValidator.enforce(cert)
        return cert
