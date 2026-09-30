"""
Atomic claim extraction and normalization.
Deconstructs compound prose into atomic verifiable propositions.
"""
import hashlib
import re
from typing import Any, Dict, List, Optional


def extract_atomic_claims(text: str, source_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Splits text into atomic declarative propositions.
    Removes hedging prefixes and isolates individual factual assertions.
    """
    if not text or not text.strip():
        return []

    # Clean text
    clean = re.sub(r"\s+", " ", text.strip())

    # Split by sentence boundaries
    raw_sentences = re.split(r"(?<=[.!?])\s+", clean)
    claims: List[Dict[str, Any]] = []

    # Common hedging prefixes to strip
    hedges = [
        r"^(it is believed that|some say that|i believe that|in my opinion|perhaps|possibly)\s+",
        r"^(according to (some|various) sources,)\s+",
        r"^(it should be noted that)\s+",
    ]

    for sent in raw_sentences:
        sent = sent.strip()
        if len(sent) < 5:
            continue

        # Strip hedge prefixes
        for h in hedges:
            sent = re.sub(h, "", sent, flags=re.IGNORECASE).strip()

        # Split conjunctions (and, but, however, while) if clauses are substantial
        sub_clauses = re.split(r"(?:;\s*|,\s*(?:and|but|while|whereas)\s+)", sent)
        for clause in sub_clauses:
            clause = clause.strip().rstrip(".")
            if len(clause) < 8:
                continue

            # Compute stable ID
            c_hash = hashlib.sha256(clause.lower().encode("utf-8")).hexdigest()[:8]
            claims.append({
                "claim_id": f"claim-{c_hash}",
                "text": clause,
                "claim_type": "FACTUAL_CORRECTNESS",
                "source_id": source_id,
                "verified": False,
            })

    return claims
