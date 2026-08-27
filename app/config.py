"""
Centralised configuration — supports:
1. Streamlit Secrets (production)
2. .env (local development)
3. Environment variables (CI / Docker)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

load_dotenv()


def get_env(key: str, default: str = "") -> str:
    # 1. Environment variables first (CI + local + Docker)
    value = os.getenv(key)
    if value:
        return value

    # 2. Streamlit secrets (Streamlit Cloud only)
    if STREAMLIT_AVAILABLE:
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass

    return default


@dataclass
class AppConfig:
    # ================= LLM =================
    groq_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048

    # ================= Embeddings =================
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # ================= Retrieval =================
    # Dataclass defaults deliberately match from_env() defaults so that
    # AppConfig() and AppConfig.from_env() (without env vars set) behave
    # identically — no silent config divergence.
    vector_top_k: int = 40
    bm25_top_k: int = 25
    rerank_top_k: int = 10
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ================= Storage =================
    # ⚠️  Streamlit Cloud is ephemeral — do not rely on persistence across restarts
    chroma_persist_dir: str = "./chroma_db"
    upload_dir: str = "./uploaded_docs"

    # ================= Chunking =================
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ================= Evaluation =================
    eval_dataset_path: str = "./evaluation/eval_dataset.json"
    ragas_llm_model: str = "groq/compound-mini"

    # ================= UI =================
    app_title: str = "Ask My Docs"
    max_history: int = 20

    @classmethod
    def from_env(cls) -> AppConfig:
        return cls(
            groq_api_key=get_env("GROQ_API_KEY"),
            llm_model=get_env("LLM_MODEL", "openai/gpt-oss-120b"),
            llm_temperature=float(get_env("LLM_TEMPERATURE", "0.3")),
            llm_max_tokens=int(get_env("LLM_MAX_TOKENS", "2048")),
            embedding_model=get_env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            embedding_dim=int(get_env("EMBEDDING_DIM", "384")),
            vector_top_k=int(get_env("VECTOR_TOP_K", "40")),
            bm25_top_k=int(get_env("BM25_TOP_K", "25")),
            rerank_top_k=int(get_env("RERANK_TOP_K", "10")),
            reranker_model=get_env(
                "RERANKER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            ),
            chroma_persist_dir=get_env("CHROMA_PERSIST_DIR", "./chroma_db"),
            upload_dir=get_env("UPLOAD_DIR", "./uploaded_docs"),
            chunk_size=int(get_env("CHUNK_SIZE", "512")),
            chunk_overlap=int(get_env("CHUNK_OVERLAP", "64")),
            eval_dataset_path=get_env(
                "EVAL_DATASET_PATH",
                "./evaluation/eval_dataset.json",
            ),
            ragas_llm_model=get_env("RAGAS_LLM_MODEL", "groq/compound-mini"),
            app_title=get_env("APP_TITLE", "Ask My Docs"),
            max_history=int(get_env("MAX_HISTORY", "20")),
        )

    def validate(self) -> list[str]:
        errors = []
        if not self.groq_api_key:
            errors.append("❌ GROQ_API_KEY is not set.")
        if self.chunk_size < 128:
            errors.append("❌ CHUNK_SIZE must be >= 128.")
        if self.rerank_top_k > self.vector_top_k + self.bm25_top_k:
            errors.append(
                f"❌ rerank_top_k ({self.rerank_top_k}) exceeds total candidate pool "
                f"({self.vector_top_k} + {self.bm25_top_k}). Lower rerank_top_k or raise k values."
            )
        return errors
