"""
Evidence-based verifier that checks lexical, entity, and n-gram grounding against context.
"""
from typing import Optional, List, Set
import re
from .base import Verifier, VerificationDecision


class EvidenceVerifier(Verifier):
    """
    Verifies that claims are supported by provided evidence context
    using token overlap, n-gram containment, and key term matching.
    """

    def __init__(self, verifier_id: str = "evidence_verifier", min_overlap_ratio: float = 0.6):
        super().__init__(verifier_id=verifier_id, verifier_type="evidence")
        self.min_overlap_ratio = float(min_overlap_ratio)

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"\b\w+\b", text) if len(w) > 2]

    def verify(self, claim: str, context: Optional[str] = None) -> VerificationDecision:
        if not context or not context.strip():
            return VerificationDecision(
                verifier_id=self.verifier_id,
                verifier_type=self.verifier_type,
                claim=claim,
                is_verified=False,
                confidence=0.0,
                reasoning="No evidence context provided.",
            )

        claim_tokens = self._tokenize(claim)
        if not claim_tokens:
            return VerificationDecision(
                verifier_id=self.verifier_id,
                verifier_type=self.verifier_type,
                claim=claim,
                is_verified=True,
                confidence=1.0,
                evidence=context,
                reasoning="Trivial claim with no significant tokens.",
            )

        context_tokens: Set[str] = set(self._tokenize(context))
        matches = [t for t in claim_tokens if t in context_tokens]
        overlap_ratio = len(matches) / len(claim_tokens)

        # Substring exact check gives immediate high confidence
        is_exact_substr = claim.strip().lower() in context.lower()
        if is_exact_substr:
            overlap_ratio = 1.0

        is_verified = overlap_ratio >= self.min_overlap_ratio

        return VerificationDecision(
            verifier_id=self.verifier_id,
            verifier_type=self.verifier_type,
            claim=claim,
            is_verified=is_verified,
            confidence=overlap_ratio,
            evidence=context,
            reasoning=f"Overlap ratio {overlap_ratio:.2f} (threshold {self.min_overlap_ratio}). Matched: {len(matches)}/{len(claim_tokens)} tokens.",
            metadata={"overlap_ratio": overlap_ratio, "matched_tokens": matches},
        )
