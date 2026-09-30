"""
Semantic entailment and equivalence verifier.
"""
from typing import Optional, Set
import re
from .base import Verifier, VerificationDecision


class SemanticVerifier(Verifier):
    """
    Verifies semantic agreement between a claim and reference context.
    Uses token set similarity (Jaccard) with negation and polarity checks.
    """

    def __init__(self, verifier_id: str = "semantic_verifier", threshold: float = 0.5):
        super().__init__(verifier_id=verifier_id, verifier_type="semantic")
        self.threshold = float(threshold)
        self.negation_words = {"not", "never", "no", "neither", "nor", "none", "cannot", "isn't", "aren't", "wasn't"}

    def _tokens(self, text: str) -> Set[str]:
        return set(re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower()))

    def verify(self, claim: str, context: Optional[str] = None) -> VerificationDecision:
        if not context or not context.strip():
            return VerificationDecision(
                verifier_id=self.verifier_id,
                verifier_type=self.verifier_type,
                claim=claim,
                is_verified=False,
                confidence=0.0,
                reasoning="Empty context for semantic verification.",
            )

        claim_set = self._tokens(claim)
        context_set = self._tokens(context)

        if not claim_set:
            return VerificationDecision(
                verifier_id=self.verifier_id,
                verifier_type=self.verifier_type,
                claim=claim,
                is_verified=True,
                confidence=1.0,
                reasoning="Empty claim.",
            )

        intersection = claim_set.intersection(context_set)
        union = claim_set.union(context_set)
        jaccard = len(intersection) / len(union) if union else 0.0

        # Asymmetric containment: how much of the claim is covered in context
        containment = len(intersection) / len(claim_set)

        # Polarity check: does claim have negation while context doesn't, or vice-versa?
        claim_has_neg = bool(claim_set.intersection(self.negation_words))
        context_has_neg = bool(context_set.intersection(self.negation_words))
        polarity_penalty = 0.5 if (claim_has_neg != context_has_neg) else 1.0

        semantic_score = (0.4 * jaccard + 0.6 * containment) * polarity_penalty
        is_verified = semantic_score >= self.threshold

        return VerificationDecision(
            verifier_id=self.verifier_id,
            verifier_type=self.verifier_type,
            claim=claim,
            is_verified=is_verified,
            confidence=round(semantic_score, 4),
            evidence=context,
            reasoning=f"Semantic score {semantic_score:.2f} (Jaccard={jaccard:.2f}, Containment={containment:.2f}, PolarityFactor={polarity_penalty}).",
            metadata={"jaccard": jaccard, "containment": containment, "polarity_match": (claim_has_neg == context_has_neg)},
        )
