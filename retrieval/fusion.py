"""
Reciprocal Rank Fusion (RRF) — fuses multiple ranked lists.
Each document's RRF score = sum(1 / (k + rank_i)) across all lists.
"""

from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    Merge multiple result lists using RRF.

    Args:
        ranked_lists: Each list contains dicts with at least {"text", "score"}.
                      Lists should be ordered best-first.
        k: RRF constant (60 is standard).

    Returns:
        Merged, deduplicated list sorted by RRF score (descending).
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, dict] = {}

    for results in ranked_lists:
        for rank, doc in enumerate(results, start=1):
            key = doc["text"][:120]  # dedup key (first 120 chars)
            rrf_scores[key] += 1.0 / (k + rank)
            if key not in doc_map:
                doc_map[key] = doc

    merged = []
    for key, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        doc = {**doc_map[key], "rrf_score": rrf_score}
        merged.append(doc)

    return merged
