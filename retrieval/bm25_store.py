"""
BM25Store: sparse keyword retrieval using rank_bm25.
Persists corpus to disk so it survives restarts.
"""

from __future__ import annotations
import pickle
import logging
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_INDEX_FILE = "bm25_index.pkl"


def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


class BM25Store:
    def __init__(self, config):
        self.config = config
        self._index_path = Path(config.chroma_persist_dir) / _INDEX_FILE
        self._corpus: list[dict] = []      # {text, metadata}
        self._bm25: BM25Okapi | None = None
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        if self._index_path.exists():
            try:
                with open(self._index_path, "rb") as f:
                    data = pickle.load(f)
                self._corpus = data["corpus"]
                self._bm25 = data["bm25"]
                logger.info("BM25Store: loaded %d docs from disk", len(self._corpus))
            except Exception as exc:
                logger.warning("BM25Store: failed to load index: %s", exc)
                self._corpus = []
                self._bm25 = None

    def _save(self):
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._index_path, "wb") as f:
            pickle.dump({"corpus": self._corpus, "bm25": self._bm25}, f)

    # ── Mutation ─────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[dict]) -> None:
        self._corpus.extend(chunks)
        tokenized = [_tokenize(c["text"]) for c in self._corpus]
        self._bm25 = BM25Okapi(tokenized)
        self._save()
        logger.info("BM25Store: index rebuilt with %d docs", len(self._corpus))

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 20) -> list[dict]:
        if not self._bm25 or not self._corpus:
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Top-k by score
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = self._corpus[idx]
            meta = chunk.get("metadata", {})
            results.append({
                "text": chunk["text"],
                "metadata": meta,
                "score": float(scores[idx]),
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", "?"),
            })
        return results
