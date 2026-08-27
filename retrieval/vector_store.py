"""
VectorStore: ChromaDB-backed dense retrieval with local HuggingFace embeddings.
No API key required — model is downloaded once and cached locally.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class VectorStore:
    COLLECTION_NAME = "ask_my_docs"

    def __init__(
        self,
        config,
        session_id: str = "default",
        collection_name: str | None = None,
    ):
        self.config = config
        self.session_id = session_id
        if collection_name:
            self.collection_name = collection_name
        elif session_id == "default":
            self.collection_name = self.COLLECTION_NAME
        else:
            clean_id = session_id.replace("-", "_")
            self.collection_name = f"session_{clean_id}"

        # Downloaded once to ~/.cache/huggingface — runs fully offline after that
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._client = chromadb.PersistentClient(
            path=config.chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def get_indexed_files(self) -> list[dict]:
        """Returns list of distinct files with chunk count."""
        if self._collection.count() == 0:
            return []
        data = self._collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for m in data.get("metadatas", []):
            if m and "source" in m:
                clean_name = Path(str(m["source"])).name
                counts[clean_name] = counts.get(clean_name, 0) + 1
        return [{"name": name, "chunks": count} for name, count in sorted(counts.items())]

    def add_chunks(self, chunks: list[dict]) -> None:
        """
        chunks: list of {text, metadata: {source, page, ...}}
        """
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks]

        embeddings = self.embeddings.embed_documents(texts)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("VectorStore: added %d chunks", len(chunks))

    def similarity_search(self, query: str, k: int = 20) -> list[dict]:
        """Returns list of {text, metadata, score}."""
        if self._collection.count() == 0:
            return []

        q_emb = self.embeddings.embed_query(query)
        results = self._collection.query(
            query_embeddings=[q_emb],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "score": 1 - dist,  # cosine similarity
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", "?"),
                }
            )
        return hits

    def delete_by_source(self, filename: str) -> int:
        """Delete all chunks whose source metadata ends with *filename*.

        Returns the number of chunks removed.
        """
        data = self._collection.get(include=["metadatas"])
        ids_to_delete = [
            doc_id
            for doc_id, meta in zip(data["ids"], data.get("metadatas", []))
            if meta and Path(str(meta.get("source", ""))).name == filename
        ]
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
            logger.info("VectorStore: deleted %d chunks for '%s'", len(ids_to_delete), filename)
        return len(ids_to_delete)

    def delete_collection(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
