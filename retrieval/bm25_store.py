"""
BM25Store: sparse keyword retrieval using rank_bm25.
Persists corpus to disk so it survives restarts.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


class BM25Store:
    def __init__(self, config, session_id: str = "default"):
        self.config = config
        self.session_id = session_id
        clean_id = session_id.replace("-", "_")
        filename = "bm25_index.json" if session_id == "default" else f"bm25_index_{clean_id}.json"
        self._index_path = Path(config.chroma_persist_dir) / filename
        self._corpus: list[dict] = []  # {text, metadata}
        self._bm25: BM25Okapi | None = None
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _rebuild_index(self) -> None:
        if self._corpus:
            tokenized = [_tokenize(c["text"]) for c in self._corpus]
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    def _load(self) -> None:
        if self._index_path.exists():
            try:
                with open(self._index_path, "r", encoding="utf-8") as f:
                    self._corpus = json.load(f)
                self._rebuild_index()
                logger.info("BM25Store: loaded %d docs from disk", len(self._corpus))
            except Exception as exc:
                logger.warning("BM25Store: failed to load index: %s", exc)
                self._corpus = []
                self._bm25 = None

    def _save(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(self._corpus, f, ensure_ascii=False)

    # ── Mutation ─────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        self._corpus.extend(chunks)
        self._rebuild_index()
        self._save()
        logger.info("BM25Store: index rebuilt with %d docs", len(self._corpus))

    def delete_by_source(self, filename: str) -> int:
        """Remove all corpus entries whose source matches *filename*.

        Returns the number of entries removed.
        """
        before = len(self._corpus)
        self._corpus = [
            c
            for c in self._corpus
            if Path(str(c.get("metadata", {}).get("source", ""))).name != filename
        ]
        removed = before - len(self._corpus)
        if removed:
            self._rebuild_index()
            self._save()
            logger.info("BM25Store: removed %d docs for '%s'", removed, filename)
        return removed

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 20) -> list[dict]:
        if not self._bm25 or not self._corpus:
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Top-k by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = self._corpus[idx]
            meta = chunk.get("metadata", {})
            results.append(
                {
                    "text": chunk["text"],
                    "metadata": meta,
                    "score": float(scores[idx]),
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", "?"),
                }
            )
        return results
