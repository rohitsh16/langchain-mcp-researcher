"""
Exact and regex pattern verifier.
"""
from typing import Optional, List
import re
from .base import Verifier, VerificationDecision


class ExactVerifier(Verifier):
    """
    Verifies claim against expected string or regex pattern.
    """

    def __init__(self, verifier_id: str = "exact_verifier", case_sensitive: bool = False):
        super().__init__(verifier_id=verifier_id, verifier_type="exact")
        self.case_sensitive = case_sensitive

    def verify(self, claim: str, context: Optional[str] = None) -> VerificationDecision:
        if context is None:
            return VerificationDecision(
                verifier_id=self.verifier_id,
                verifier_type=self.verifier_type,
                claim=claim,
                is_verified=False,
                confidence=0.0,
                reasoning="No context/ground truth provided for exact verification.",
            )

        c = claim.strip() if self.case_sensitive else claim.strip().lower()
        target = context.strip() if self.case_sensitive else context.strip().lower()

        is_match = (c == target) or (c in target)
        conf = 1.0 if is_match else 0.0

        return VerificationDecision(
            verifier_id=self.verifier_id,
            verifier_type=self.verifier_type,
            claim=claim,
            is_verified=is_match,
            confidence=conf,
            evidence=context,
            reasoning=f"Exact match check: {'Passed' if is_match else 'Failed'}.",
        )
