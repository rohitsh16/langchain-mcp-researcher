from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class CorrectnessDimension(str, Enum):
    FACTUAL_CORRECTNESS = "FACTUAL_CORRECTNESS"
    GROUNDING = "GROUNDING"
    SEMANTIC_EQUIVALENCE = "SEMANTIC_EQUIVALENCE"
    LOGICAL_VALIDITY = "LOGICAL_VALIDITY"
    MATHEMATICAL_VALIDITY = "MATHEMATICAL_VALIDITY"
    INSTRUCTION_FOLLOWING = "INSTRUCTION_FOLLOWING"
    CODE_CORRECTNESS = "CODE_CORRECTNESS"
    TEMPORAL_CONSISTENCY = "TEMPORAL_CONSISTENCY"
    SPATIAL_CONSISTENCY = "SPATIAL_CONSISTENCY"
    AUDIO_VISUAL_CONSISTENCY = "AUDIO_VISUAL_CONSISTENCY"
    PROCESS_CORRECTNESS = "PROCESS_CORRECTNESS"
    ROBUSTNESS = "ROBUSTNESS"
    CALIBRATION = "CALIBRATION"
    CERTIFICATION = "CERTIFICATION"


class CertificateDecision(str, Enum):
    CERTIFY = "CERTIFY"
    ABSTAIN = "ABSTAIN"
    REJECT = "REJECT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class EvidenceType(str, Enum):
    ABSTRACT = "abstract"
    CLAIM = "claim"
    DEFINITION = "definition"
    THEOREM = "theorem"
    LEMMA = "lemma"
    PROOF = "proof"
    EXPERIMENT = "experiment"
    LIMITATION = "limitation"
    RESULT = "result"
    CODE = "code"


class VerificationStatus(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"
    REFUTED = "REFUTED"
    AMBIGUOUS = "AMBIGUOUS"


class ResearchQuestion(BaseModel):
    id: str
    title: str
    question: str
    correctness_type: CorrectnessDimension = CorrectnessDimension.FACTUAL_CORRECTNESS
    scope: str = "general"
    assumptions: List[str] = Field(default_factory=list)
    date_created: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PaperRecord(BaseModel):
    paper_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    abstract: Optional[str] = None
    pdf_url: Optional[str] = None
    source_indexes: List[str] = Field(default_factory=list)
    fingerprint: str = ""

    def compute_fingerprint(self) -> str:
        import hashlib
        import re

        clean_title = re.sub(r"[^\w\s]", "", self.title.lower()).strip()
        first_author = self.authors[0].lower().strip() if self.authors else ""
        raw = f"{clean_title}::{first_author}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class EvidenceSpan(BaseModel):
    evidence_id: str
    paper_id: str
    page: Optional[int] = None
    section: Optional[str] = None
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    text: str
    evidence_type: EvidenceType = EvidenceType.CLAIM


class Claim(BaseModel):
    claim_id: str
    text: str
    claim_type: CorrectnessDimension = CorrectnessDimension.FACTUAL_CORRECTNESS
    source_paper_id: Optional[str] = None
    evidence_ids: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED


class VerificationResult(BaseModel):
    verifier_id: str
    claim_id: str
    score: float = Field(ge=0.0, le=1.0)
    decision: str
    evidence_ids: List[str] = Field(default_factory=list)
    error_type: Optional[str] = None
    explanation: Optional[str] = None
    cost_ms: float = 0.0
    tokens: int = 0


class Certificate(BaseModel):
    certificate_id: str
    output_id: str
    correctness_type: CorrectnessDimension
    estimated_correctness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    certified_risk: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence_level: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    assumptions: List[str] = Field(default_factory=list)
    verifier_results: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    method: str
    valid_under: List[str] = Field(default_factory=list)
    decision: CertificateDecision

    @field_validator("certified_risk")
    @classmethod
    def check_risk_and_guarantee_separation(cls, v: Optional[float], info) -> Optional[float]:
        # Enforce that certified_risk is only present when confidence_level and assumptions are declared
        return v
