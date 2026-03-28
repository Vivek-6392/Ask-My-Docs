# """
# Centralised configuration — reads from environment variables / .env file.
# """

# from __future__ import annotations
# import os
# from dataclasses import dataclass
# from dotenv import load_dotenv

# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")

# @dataclass
# class AppConfig:
#     # LLM — Groq
#     groq_api_key: str = groq_api_key
#     llm_model: str = "llama-3.3-70b-versatile"
#     llm_temperature: float = 0.0
#     llm_max_tokens: int = 1024

#     # Embeddings — HuggingFace local (no API key needed)
#     embedding_model: str = "BAAI/bge-small-en-v1.5"
#     embedding_dim: int = 384

#     # Retrieval
#     vector_top_k: int = 20           # candidates from vector search
#     bm25_top_k: int = 20             # candidates from BM25
#     rerank_top_k: int = 5            # after cross-encoder reranking
#     reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#     # Storage
#     chroma_persist_dir: str = "./chroma_db"
#     upload_dir: str = "./uploaded_docs"

#     # Chunking
#     chunk_size: int = 512
#     chunk_overlap: int = 64

#     # Evaluation
#     eval_dataset_path: str = "./evaluation/eval_dataset.json"
#     ragas_llm_model: str = "llama-3.3-70b-versatile"

#     # UI
#     app_title: str = "Ask My Docs"
#     max_history: int = 20

#     @classmethod
#     def from_env(cls) -> AppConfig:
#         return cls(
#             groq_api_key=os.getenv("GROQ_API_KEY", ""),
#             llm_model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
#             llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
#             llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
#             embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
#             embedding_dim=int(os.getenv("EMBEDDING_DIM", "384")),
#             vector_top_k=int(os.getenv("VECTOR_TOP_K", "20")),
#             bm25_top_k=int(os.getenv("BM25_TOP_K", "20")),
#             rerank_top_k=int(os.getenv("RERANK_TOP_K", "5")),
#             reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
#             chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
#             upload_dir=os.getenv("UPLOAD_DIR", "./uploaded_docs"),
#             chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
#             chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "64")),
#             eval_dataset_path=os.getenv("EVAL_DATASET_PATH", "./evaluation/eval_dataset.json"),
#             ragas_llm_model=os.getenv("RAGAS_LLM_MODEL", "llama-3.3-70b-versatile"),
#             app_title=os.getenv("APP_TITLE", "Ask My Docs"),
#             max_history=int(os.getenv("MAX_HISTORY", "20")),
#         )

#     def validate(self) -> list[str]:
#         """Return list of validation errors."""
#         errors = []
#         if not self.groq_api_key:
#             errors.append("GROQ_API_KEY is not set.")
#         if self.chunk_size < 128:
#             errors.append("CHUNK_SIZE must be >= 128.")
#         return errors


"""
Centralised configuration — supports:
1. Streamlit Secrets (production)
2. .env (local development)
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Try importing streamlit (won’t break locally)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Load .env for local development
load_dotenv()


def get_env(key: str, default: str = ""):
    # ✅ 1. ALWAYS check environment variables first (CI + local + Docker)
    value = os.getenv(key)
    if value:
        return value

    # ✅ 2. Then check Streamlit secrets (only for Streamlit Cloud)
    if STREAMLIT_AVAILABLE:
        try:
            import streamlit as st
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass

    return default


@dataclass
class AppConfig:
    # ================= LLM =================
    groq_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048

    # ================= Embeddings =================
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # ================= Retrieval =================
    vector_top_k: int = 40
    bm25_top_k: int = 25
    rerank_top_k: int = 10
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ================= Storage =================
    # ⚠️ Streamlit Cloud is ephemeral → avoid relying on persistence
    chroma_persist_dir: str = "./chroma_db"
    upload_dir: str = "./uploaded_docs"

    # ================= Chunking =================
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ================= Evaluation =================
    eval_dataset_path: str = "./evaluation/eval_dataset.json"
    ragas_llm_model: str = "llama-3.3-70b-versatile"

    # ================= UI =================
    app_title: str = "Ask My Docs"
    max_history: int = 20

    @classmethod
    def from_env(cls) -> AppConfig:
        return cls(
            # 🔐 API key (FIXED)
            groq_api_key=get_env("GROQ_API_KEY"),

            # LLM
            llm_model=get_env("LLM_MODEL", "llama-3.3-70b-versatile"),
            llm_temperature=float(get_env("LLM_TEMPERATURE", "0.3")),
            llm_max_tokens=int(get_env("LLM_MAX_TOKENS", "2048")),

            # Embeddings
            embedding_model=get_env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            embedding_dim=int(get_env("EMBEDDING_DIM", "384")),

            # Retrieval (better defaults)
            vector_top_k=int(get_env("VECTOR_TOP_K", "40")),
            bm25_top_k=int(get_env("BM25_TOP_K", "25")),
            rerank_top_k=int(get_env("RERANK_TOP_K", "10")),
            reranker_model=get_env(
                "RERANKER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),

            # Storage (still defined, but don't rely on persistence)
            chroma_persist_dir=get_env("CHROMA_PERSIST_DIR", "./chroma_db"),
            upload_dir=get_env("UPLOAD_DIR", "./uploaded_docs"),

            # Chunking
            chunk_size=int(get_env("CHUNK_SIZE", "512")),
            chunk_overlap=int(get_env("CHUNK_OVERLAP", "64")),

            # Evaluation
            eval_dataset_path=get_env(
                "EVAL_DATASET_PATH",
                "./evaluation/eval_dataset.json"
            ),
            ragas_llm_model=get_env(
                "RAGAS_LLM_MODEL",
                "llama-3.3-70b-versatile"
            ),

            # UI
            app_title=get_env("APP_TITLE", "Ask My Docs"),
            max_history=int(get_env("MAX_HISTORY", "20")),
        )

    def validate(self) -> list[str]:
        """Return list of validation errors."""
        errors = []

        if not self.groq_api_key:
            errors.append("❌ GROQ_API_KEY is not set.")

        if self.chunk_size < 128:
            errors.append("❌ CHUNK_SIZE must be >= 128.")

        return errors