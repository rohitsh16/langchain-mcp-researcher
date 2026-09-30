"""
Ensemble agreement and consistency across generation samples.
"""
from collections import Counter
from typing import Any, Dict, List
from ..correctness.factuality import normalize_tokens


def compute_ensemble_agreement(generations: List[str]) -> Dict[str, Any]:
    """
    Computes lexical and semantic agreement metrics across generation samples.
    """
    if not generations:
        return {
            "exact_agreement_ratio": 0.0,
            "mean_pairwise_jaccard": 0.0,
            "consensus_candidate": None,
            "sample_count": 0,
        }

    k = len(generations)
    if k == 1:
        return {
            "exact_agreement_ratio": 1.0,
            "mean_pairwise_jaccard": 1.0,
            "consensus_candidate": generations[0],
            "sample_count": 1,
        }

    # 1. Exact Normalized Match
    normalized = [" ".join(normalize_tokens(g)) for g in generations]
    counts = Counter(normalized)
    most_common_norm, count = counts.most_common(1)[0]
    exact_ratio = count / k

    # Pick first matching raw generation as consensus
    consensus = generations[normalized.index(most_common_norm)]

    # 2. Pairwise Jaccard
    jaccards = []
    token_sets = [set(normalize_tokens(g)) for g in generations]
    for i in range(k):
        for j in range(i + 1, k):
            u = token_sets[i].union(token_sets[j])
            inter = token_sets[i].intersection(token_sets[j])
            jaccards.append(len(inter) / len(u) if u else 1.0)

    mean_jaccard = sum(jaccards) / len(jaccards) if jaccards else 1.0

    return {
        "exact_agreement_ratio": round(float(exact_ratio), 4),
        "agreement_ratio": round(float(exact_ratio), 4),
        "mean_pairwise_jaccard": round(float(mean_jaccard), 4),
        "consensus_candidate": consensus,
        "majority_answer": consensus,
        "sample_count": k,
        "unique_variants": len(counts),
    }
