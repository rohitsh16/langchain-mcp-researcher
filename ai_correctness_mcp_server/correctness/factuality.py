"""
Factual correctness evaluation against reference ground truth or consensus facts.
"""
import re
from typing import Any, Dict, List, Optional


def normalize_tokens(text: str) -> List[str]:
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in clean.split() if t]


def evaluate_factuality(claim: str, reference: str, threshold: float = 0.6) -> Dict[str, Any]:
    """
    Evaluates factual alignment between a candidate claim and reference ground truth.
    Computes token precision, recall, and semantic token containment.
    """
    c_tokens = set(normalize_tokens(claim))
    r_tokens = set(normalize_tokens(reference))

    if not c_tokens or not r_tokens:
        return {
            "score": 0.0,
            "decision": "REFUTED",
            "precision": 0.0,
            "recall": 0.0,
            "explanation": "Empty claim or reference tokens.",
        }

    intersection = c_tokens.intersection(r_tokens)
    precision = len(intersection) / len(c_tokens)
    recall = len(intersection) / len(r_tokens)

    # Harmonic mean F1 score
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Strong containment bonus if majority of key claim tokens appear in reference
    score = 0.7 * precision + 0.3 * f1

    decision = "VERIFIED" if score >= threshold else "REFUTED"
    return {
        "score": round(score, 4),
        "decision": decision,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "explanation": f"Claim precision: {precision:.2f}, F1: {f1:.2f} against reference.",
    }
