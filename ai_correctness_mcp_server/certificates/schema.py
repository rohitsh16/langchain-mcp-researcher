"""
Formal certificate schema adhering to CORRECTNESS_SPEC.md §3.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator


class OutputCorrectnessCertificate(BaseModel):
    certificate_id: str
    version: str = "1.0.0"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    target_model: str = "unknown"
    target_query: str
    target_output: str
    decision: str = Field(description="CERTIFIED | PARTIALLY_CERTIFIED | REJECTED_RISK_EXCEEDED | INCONCLUSIVE_INSUFFICIENT_EVIDENCE")
    guarantee_type: str = Field(description="CONFORMAL_RISK_CONTROL | SPLIT_CONFORMAL | EMPIRICAL_POINT_ESTIMATE | HEURISTIC")
    estimated_correctness: float = Field(ge=0.0, le=1.0)
    certified_risk: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence_level: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    calibration_dataset_size: Optional[int] = Field(default=None, ge=0)
    calibration_ece: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    dimensions_verified: List[str] = Field(default_factory=list)
    claims_total: int = Field(default=0, ge=0)
    claims_verified: int = Field(default=0, ge=0)
    evidence_sources: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    verifier_fingerprint: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("decision")
    @classmethod
    def validate_decision(cls, v: str) -> str:
        valid_decisions = {
            "CERTIFIED",
            "PARTIALLY_CERTIFIED",
            "REJECTED_RISK_EXCEEDED",
            "INCONCLUSIVE_INSUFFICIENT_EVIDENCE",
        }
        if v not in valid_decisions:
            raise ValueError(f"Invalid decision '{v}'. Must be one of {valid_decisions}")
        return v

    @field_validator("guarantee_type")
    @classmethod
    def validate_guarantee(cls, v: str) -> str:
        valid_types = {
            "CONFORMAL_RISK_CONTROL",
            "SPLIT_CONFORMAL",
            "EMPIRICAL_POINT_ESTIMATE",
            "HEURISTIC",
        }
        if v not in valid_types:
            raise ValueError(f"Invalid guarantee_type '{v}'. Must be one of {valid_types}")
        return v
