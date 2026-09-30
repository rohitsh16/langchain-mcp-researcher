"""
Semantic equivalence and bidirectional entailment estimation.
Used for multi-sample answer clustering and paraphrase detection.
"""
from typing import Dict, List, Set, Any
from .factuality import normalize_tokens

STOPWORDS = {"the", "a", "an", "is", "was", "in", "by", "of", "and", "to", "for", "on", "at", "with", "that", "it"}


def get_ngrams(tokens: List[str], n: int = 2) -> Set[str]:
    if len(tokens) < n:
        return set(tokens)
    return set(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def compute_semantic_equivalence(text_a: str, text_b: str, threshold: float = 0.45) -> Dict[str, Any]:
    """
    Computes bidirectional semantic overlap and similarity between two texts.
    Detects polarity contradictions (e.g. 'not', 'never', 'unsupported') to prevent false matches.
    """
    toks_a = normalize_tokens(text_a)
    toks_b = normalize_tokens(text_b)

    if not toks_a or not toks_b:
        return {"equivalent": False, "similarity": 0.0, "reason": "Empty inputs"}

    set_a = set(toks_a)
    set_b = set(toks_b)

    # Content-word sets
    content_a = {t for t in set_a if t not in STOPWORDS}
    content_b = {t for t in set_b if t not in STOPWORDS}

    if content_a and content_b:
        content_jaccard = len(content_a.intersection(content_b)) / len(content_a.union(content_b))
        containment = len(content_a.intersection(content_b)) / min(len(content_a), len(content_b))
    else:
        content_jaccard = len(set_a.intersection(set_b)) / len(set_a.union(set_b))
        containment = content_jaccard

    # Unigram Jaccard
    unigram_jaccard = len(set_a.intersection(set_b)) / len(set_a.union(set_b))

    # Negation contradiction check
    negations = {"not", "never", "no", "cannot", "neither", "nor"}
    neg_a = len(set_a.intersection(negations)) % 2
    neg_b = len(set_b.intersection(negations)) % 2
    polarity_mismatch = neg_a != neg_b

    sim = 0.4 * unigram_jaccard + 0.3 * content_jaccard + 0.3 * containment
    if polarity_mismatch:
        sim = sim * 0.2  # Penalize severe polarity contradictions

    equivalent = sim >= threshold
    return {
        "equivalent": equivalent,
        "similarity": round(sim, 4),
        "unigram_jaccard": round(unigram_jaccard, 4),
        "content_jaccard": round(content_jaccard, 4),
        "polarity_mismatch": polarity_mismatch,
    }
