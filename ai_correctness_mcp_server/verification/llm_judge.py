"""
LLM-as-a-judge verifier with structured parsing and mock/offline fallback.
"""
from typing import Optional, Dict, Any, Callable
import json
from .base import Verifier, VerificationDecision


class LLMJudgeVerifier(Verifier):
    """
    LLM-as-a-judge verifier. Supports pluggable LLM callable or deterministic offline evaluation.
    """

    def __init__(
        self,
        verifier_id: str = "llm_judge",
        llm_caller: Optional[Callable[[str], str]] = None,
        model_name: str = "gpt-4o-mini-mock",
    ):
        super().__init__(verifier_id=verifier_id, verifier_type="llm_judge")
        self.llm_caller = llm_caller
        self.model_name = model_name

    def _build_prompt(self, claim: str, context: Optional[str]) -> str:
        return (
            f"You are an expert factual verifier.\n"
            f"Context: {context or 'None'}\n"
            f"Claim: {claim}\n"
            f"Determine if the claim is factual and supported by the context.\n"
            f"Respond with JSON: {{\"is_verified\": true/false, \"confidence\": 0.0-1.0, \"reasoning\": \"...\"}}"
        )

    def verify(self, claim: str, context: Optional[str] = None) -> VerificationDecision:
        if self.llm_caller is not None:
            prompt = self._build_prompt(claim, context)
            try:
                raw_response = self.llm_caller(prompt)
                parsed = json.loads(raw_response)
                return VerificationDecision(
                    verifier_id=self.verifier_id,
                    verifier_type=self.verifier_type,
                    claim=claim,
                    is_verified=bool(parsed.get("is_verified", False)),
                    confidence=float(parsed.get("confidence", 0.0)),
                    evidence=context,
                    reasoning=str(parsed.get("reasoning", "LLM Judge decision")),
                    metadata={"model": self.model_name},
                )
            except Exception as e:
                # Fallback on failure
                return VerificationDecision(
                    verifier_id=self.verifier_id,
                    verifier_type=self.verifier_type,
                    claim=claim,
                    is_verified=False,
                    confidence=0.0,
                    evidence=context,
                    reasoning=f"LLM Judge execution error: {str(e)}",
                )

        # Deterministic offline mock judge
        if not context:
            return VerificationDecision(
                verifier_id=self.verifier_id,
                verifier_type=self.verifier_type,
                claim=claim,
                is_verified=False,
                confidence=0.0,
                reasoning="Offline judge: no context provided.",
            )

        # Heuristic check for mock mode
        claim_lower = claim.lower().strip()
        context_lower = context.lower()
        words = [w for w in claim_lower.split() if len(w) > 3]
        matched = [w for w in words if w in context_lower]
        ratio = len(matched) / len(words) if words else 1.0
        verified = ratio >= 0.5

        return VerificationDecision(
            verifier_id=self.verifier_id,
            verifier_type=self.verifier_type,
            claim=claim,
            is_verified=verified,
            confidence=round(ratio, 3),
            evidence=context,
            reasoning=f"Offline Judge matched {len(matched)}/{len(words)} keywords in context.",
            metadata={"model": self.model_name, "mode": "offline_mock"},
        )
