"""
RAGPipeline: orchestrates hybrid retrieval → reranking → generation.
"""

from __future__ import annotations
import time
import logging
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from retrieval.vector_store import VectorStore
from retrieval.bm25_store import BM25Store
from retrieval.reranker import CrossEncoderReranker
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.prompt import build_rag_prompt, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


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
        )

    def query(
        self,
        question: str,
        top_k: int = 5,
        vector_k: int = 20,
        bm25_k: int = 20,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()

        # 1. Hybrid retrieval
        vector_results = self.vector_store.similarity_search(question, k=vector_k)
        bm25_results = self.bm25_store.search(question, k=bm25_k)

        # 2. Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(
            [vector_results, bm25_results], k=60
        )

        # 3. Cross-encoder reranking
        reranked = self.reranker.rerank(question, fused, top_k=top_k)

        # 4. Build prompt + generate
        context_str, citations = build_rag_prompt(reranked)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context_str}\n\nQuestion: {question}"),
        ]
        response = self.llm.invoke(messages)
        answer = response.content

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
