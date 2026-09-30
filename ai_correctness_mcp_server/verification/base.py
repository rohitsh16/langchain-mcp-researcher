"""
Base verifier interface and result data structures.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class VerificationDecision:
    verifier_id: str
    verifier_type: str
    claim: str
    is_verified: bool
    confidence: float
    evidence: Optional[str] = None
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Verifier(ABC):
    """Abstract base class for all verifiers."""

    def __init__(self, verifier_id: str, verifier_type: str):
        self.verifier_id = verifier_id
        self.verifier_type = verifier_type

    @abstractmethod
    def verify(self, claim: str, context: Optional[str] = None) -> VerificationDecision:
        """Verify a single claim against optional context/evidence."""
        pass
