"""
Semantic entropy estimation via semantic equivalence clustering.
Implements Kuhn et al. (2023) semantic uncertainty framework.
"""
import math
from typing import Any, Dict, List, Tuple
from ..correctness.semantic import compute_semantic_equivalence


def cluster_semantic_generations(
    generations: List[str],
    similarity_threshold: float = 0.45,
) -> List[List[str]]:
    """
    Partitions generated sequences into semantic equivalence clusters.
    If generation_i is bidirectionally equivalent to generation_j, they are grouped in the same cluster.
    """
    clusters: List[List[str]] = []

    for gen in generations:
        assigned = False
        for cluster in clusters:
            # Check equivalence against representative of the cluster
            rep = cluster[0]
            sim_res = compute_semantic_equivalence(gen, rep, threshold=similarity_threshold)
            if sim_res.get("equivalent", False):
                cluster.append(gen)
                assigned = True
                break

        if not assigned:
            clusters.append([gen])

    return clusters


def compute_semantic_entropy(
    generations: List[str],
    similarity_threshold: float = 0.45,
) -> Dict[str, Any]:
    """
    Computes semantic entropy across sampled generations:
    H_sem = - sum_{c in C} p(c) * log(p(c))
    """
    if not generations:
        return {
            "semantic_entropy": 0.0,
            "normalized_entropy": 0.0,
            "num_clusters": 0,
            "clusters": [],
            "cluster_probabilities": [],
        }

    k = len(generations)
    clusters = cluster_semantic_generations(generations, similarity_threshold=similarity_threshold)
    num_clusters = len(clusters)

    probs = [len(c) / k for c in clusters]
    entropy = -sum(p * math.log(p) for p in probs if p > 0.0)

    # Normalize by log(K)
    max_entropy = math.log(k) if k > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "semantic_entropy": round(float(entropy), 4),
        "normalized_entropy": round(float(normalized_entropy), 4),
        "num_clusters": num_clusters,
        "sample_count": k,
        "cluster_probabilities": [round(p, 4) for p in probs],
        "clusters": [
            {"size": len(c), "representative": c[0], "samples": c} for c in clusters
        ],
    }
