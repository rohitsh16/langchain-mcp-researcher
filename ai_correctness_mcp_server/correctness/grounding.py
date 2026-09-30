"""
Grounding verification against cited evidence spans.
Checks whether declared claims are directly substantiated by evidence passages.
"""
from typing import Any, Dict, List
from .factuality import normalize_tokens


def verify_grounding(claim: str, evidence_spans: List[str], min_overlap_ratio: float = 0.5) -> Dict[str, Any]:
    """
    Verifies that a claim is grounded in the provided evidence spans.
    """
    if not evidence_spans:
        return {
            "grounded": False,
            "score": 0.0,
            "supporting_spans": [],
            "explanation": "No evidence spans provided.",
        }

    c_tokens = set(normalize_tokens(claim))
    if not c_tokens:
        return {
            "grounded": False,
            "score": 0.0,
            "supporting_spans": [],
            "explanation": "Claim has no valid content tokens.",
        }

    best_score = 0.0
    supporting = []

    for i, span in enumerate(evidence_spans):
        s_tokens = set(normalize_tokens(span))
        if not s_tokens:
            continue
        overlap = len(c_tokens.intersection(s_tokens))
        ratio = overlap / len(c_tokens)
        if ratio > best_score:
            best_score = ratio
        if ratio >= min_overlap_ratio:
            supporting.append({"span_index": i, "overlap_ratio": round(ratio, 4)})

    grounded = best_score >= min_overlap_ratio
    return {
        "grounded": grounded,
        "score": round(best_score, 4),
        "supporting_spans": supporting,
        "explanation": f"Highest evidence overlap ratio: {best_score:.2f} (threshold: {min_overlap_ratio}).",
    }
