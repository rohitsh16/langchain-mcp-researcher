"""
Contradictory and conflicting evidence injection attacks.
"""
from typing import Dict, Any, List


def create_conflicting_evidence_pair(
    topic: str,
    claim_a: str,
    claim_b: str,
) -> Dict[str, str]:
    """
    Creates a scenario where Document 1 states claim_a and Document 2 asserts contradicting claim_b.
    """
    doc1 = f"Source A Report: Regarding {topic}, comprehensive measurements confirm that {claim_a}."
    doc2 = f"Source B Report: In direct opposition regarding {topic}, recent replications refute this and prove that {claim_b}."
    return {
        "topic": topic,
        "document_1": doc1,
        "document_2": doc2,
        "combined_context": f"{doc1}\n{doc2}",
        "claim_a": claim_a,
        "claim_b": claim_b,
    }


def evaluate_conflict_awareness(verifier, conflict_scenarios: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Tests if verifier detects that both contradictory claims cannot be simultaneously true
    under the combined conflicting context.
    """
    double_accepts = 0
    total = len(conflict_scenarios)

    for scenario in conflict_scenarios:
        ctx = scenario["combined_context"]
        dec_a = verifier.verify(scenario["claim_a"], context=ctx)
        dec_b = verifier.verify(scenario["claim_b"], context=ctx)

        # If both are verified with high confidence despite contradiction, verifier fails conflict awareness
        if dec_a.is_verified and dec_b.is_verified:
            double_accepts += 1

    blind_acceptance_rate = double_accepts / total if total > 0 else 0.0
    return {
        "total_conflicts": total,
        "double_accepts": double_accepts,
        "blind_acceptance_rate": blind_acceptance_rate,
        "conflict_aware": blind_acceptance_rate < 0.25,
    }
