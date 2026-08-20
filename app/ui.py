"""
Streamlit UI components: sidebar, chat panel, upload panel.
"""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from app.config import AppConfig
from app.session import add_message, clear_history
from retrieval.ingestor import ingest_files
from retrieval.pipeline import RAGPipeline

_THINK_BLOCK_RE = re.compile(
    r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL
)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)


def _clean_assistant_message(content: str) -> str:
    """Hide reasoning from chat history, including messages saved before a reload."""
    text = _THINK_BLOCK_RE.sub("", content)
    if opening_tag := _THINK_OPEN_RE.search(text):
        text = text[:opening_tag.start()]
    return text.strip()

# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(pipeline: RAGPipeline, config: AppConfig):
    st.markdown("## 📚 Ask My Docs")
    st.caption("Hybrid RAG · BM25 + Vector · Reranking")
    st.divider()

    # Status
    doc_count = pipeline.vector_store.count()
    st.metric("Indexed Chunks", doc_count)

    st.divider()

    # Retrieval settings
    st.markdown("**Retrieval Settings**")
    config.rerank_top_k = st.slider("Final context chunks", 1, 5, config.rerank_top_k)
    config.vector_top_k = st.slider("Vector candidates", 5, 30, config.vector_top_k)
    config.bm25_top_k = st.slider("BM25 candidates", 5, 30, config.bm25_top_k)

    st.divider()

    st.session_state.retrieval_debug = st.toggle(
        "Show retrieval debug", value=st.session_state.retrieval_debug
    )

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        clear_history()
        st.rerun()

    st.divider()
    st.caption("Built with LangChain · ChromaDB · Sentence-Transformers")


# ── Chat Panel ───────────────────────────────────────────────────────────────

def render_chat(pipeline: RAGPipeline, config: AppConfig):
    st.markdown("### 💬 Chat with your documents")

    # Render history
    for msg in st.session_state.messages:
        content = msg["content"]
        if msg["role"] == "assistant":
            content = _clean_assistant_message(content)
            msg["content"] = content

        with st.chat_message(msg["role"]):
            st.markdown(content)
            if msg.get("citations"):
                _render_citations(msg["citations"])

    # Input
    if prompt := st.chat_input("Ask anything about your documents…"):
        add_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = pipeline.query(
                    question=prompt,
                    top_k=config.rerank_top_k,
                    vector_k=config.vector_top_k,
                    bm25_k=config.bm25_top_k,
                )

            answer = result["answer"]
            citations = result["citations"]

            st.markdown(answer)
            _render_citations(citations)

            if st.session_state.retrieval_debug:
                _render_debug(result)

        add_message("assistant", answer, citations)


def _render_citations(citations: list[dict]):
    if not citations:
        return
    with st.expander(f"📎 {len(citations)} source(s)", expanded=False):
        for i, c in enumerate(citations, 1):
            st.markdown(
                f"**[{i}] {c.get('source', 'Unknown')}** "
                f"— page {c.get('page', '?')} "
                f"· score {c.get('score', 0):.3f}"
            )
            st.caption(c.get("snippet", ""))
            st.divider()


def _render_debug(result: dict):
    with st.expander("🔍 Retrieval Debug", expanded=False):
        st.json({
            "vector_hits": result.get("vector_hits", 0),
            "bm25_hits": result.get("bm25_hits", 0),
            "after_rerank": result.get("after_rerank", 0),
            "latency_ms": result.get("latency_ms", 0),
        })
        st.markdown("**Reranked chunks:**")
        for chunk in result.get("chunks", []):
            st.markdown(f"- `{chunk['source']}` p{chunk.get('page','?')} — score {chunk['score']:.4f}")
            st.caption(chunk["text"][:200] + "…")


# ── Upload Panel ─────────────────────────────────────────────────────────────

def render_upload_panel(pipeline: RAGPipeline, config: AppConfig):
    st.markdown("### 📂 Upload & Index Documents")
    st.info(
        "Supported formats: **PDF, TXT, DOCX, MD**. "
        "Files are chunked, embedded, and stored in ChromaDB."
    )

    uploaded = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
    )

    if uploaded and st.button("⚡ Index Documents", use_container_width=True):
        upload_dir = Path(config.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for f in uploaded:
            dest = upload_dir / f.name
            dest.write_bytes(f.read())
            saved_paths.append(dest)

        with st.spinner(f"Ingesting {len(saved_paths)} file(s)…"):
            stats = ingest_files(saved_paths, pipeline, config)

        st.success(
            f"✅ Indexed **{stats['chunks']} chunks** from "
            f"**{stats['files']} file(s)** in {stats['elapsed_s']:.1f}s"
        )
        st.rerun()
