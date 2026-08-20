"""
RAGPipeline: orchestrates hybrid retrieval → reranking → generation.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from retrieval.bm25_store import BM25Store
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.prompt import SYSTEM_PROMPT, build_rag_prompt
from retrieval.reranker import CrossEncoderReranker
from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(
    r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL
)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)


def clean_model_response(content: Any) -> str:
    """Return only the visible answer, excluding model reasoning blocks."""
    text = content if isinstance(content, str) else str(content)
    text = _THINK_BLOCK_RE.sub("", text)

    # An unclosed reasoning tag has no safely identifiable final answer, so
    # hide the remaining content instead of exposing the model's reasoning.
    if opening_tag := _THINK_OPEN_RE.search(text):
        text = text[:opening_tag.start()]

    return text.strip()


class RAGPipeline:
    def __init__(self, config):
        self.config = config
        self.vector_store = VectorStore(config)
        self.bm25_store = BM25Store(config)
        self.reranker = CrossEncoderReranker(config.reranker_model)
        self.llm = ChatGroq(
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            api_key=config.groq_api_key,
            max_retries=3,
            request_timeout=60.0,
        )

    def query(
        self,
        question: str,
        # None = fall back to config values; explicit int overrides (useful in tests/eval)
        top_k: int | None = None,
        vector_k: int | None = None,
        bm25_k: int | None = None,
    ) -> dict[str, Any]:
        # Resolve against config so every caller benefits from centralised tuning
        _top_k = top_k if top_k is not None else self.config.rerank_top_k
        _vector_k = vector_k if vector_k is not None else self.config.vector_top_k
        _bm25_k = bm25_k if bm25_k is not None else self.config.bm25_top_k

        t0 = time.perf_counter()

        # 1. Hybrid retrieval
        vector_results = self.vector_store.similarity_search(question, k=_vector_k)
        bm25_results = self.bm25_store.search(question, k=_bm25_k)

        logger.debug(
            "Retrieval: vector=%d hits, bm25=%d hits (vector_k=%d, bm25_k=%d)",
            len(vector_results), len(bm25_results), _vector_k, _bm25_k,
        )

        # 2. Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(
            [vector_results, bm25_results], k=60
        )

        # 3. Cross-encoder reranking
        reranked = self.reranker.rerank(question, fused, top_k=_top_k)

        # 4. Build prompt + generate
        context_str, citations = build_rag_prompt(reranked)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context_str}\n\nQuestion: {question}"),
        ]
        response = self.llm.invoke(messages)
        answer = clean_model_response(response.content)

        elapsed = (time.perf_counter() - t0) * 1000

        return {
            "answer": answer,
            "citations": citations,
            "chunks": reranked,
            "vector_hits": len(vector_results),
            "bm25_hits": len(bm25_results),
            "after_rerank": len(reranked),
            "latency_ms": round(elapsed, 1),
        }

    def add_documents(self, chunks: list[dict]) -> int:
        """Add pre-chunked documents to both stores. Returns chunk count."""
        self.vector_store.add_chunks(chunks)
        self.bm25_store.add_chunks(chunks)
        return len(chunks)
