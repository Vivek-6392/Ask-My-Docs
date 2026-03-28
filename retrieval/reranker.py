"""
CrossEncoderReranker: uses a sentence-transformers cross-encoder to rescore
candidate passages retrieved by hybrid search.
"""

from __future__ import annotations
import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info("Loading cross-encoder: %s", model_name)
        self._model = CrossEncoder(model_name, max_length=512)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Re-score `candidates` (each with key `text`) against `query`.
        Returns top_k highest-scoring candidates, sorted desc.
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs)

        scored = [
            {**c, "score": float(s)}
            for c, s in zip(candidates, scores)
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
