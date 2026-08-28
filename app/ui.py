"""
Streamlit UI components matching the reference design:
sidebar with metric boxes, document list, sliders, top navigation tabs,
custom user bubbles, inline citation pills, and source cards.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import streamlit as st

from app.config import AppConfig
from app.session import add_message, clear_history
from retrieval.ingestor import ingest_files
from retrieval.pipeline import RAGPipeline

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
_CITATION_REF_RE = re.compile(r"\[(\d+)\]")

# Match \[...\] display math — greedy‑safe with DOTALL
_DISPLAY_MATH_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
# Match \(...\) inline math
_INLINE_MATH_RE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)


def _normalize_latex(text: str) -> str:
    """Convert LLM-style LaTeX delimiters to Streamlit-compatible ones.

    Streamlit's st.markdown() understands ``$...$`` (inline) and ``$$...$$``
    (display) but NOT the ``\\(...\\)`` / ``\\[...\\]`` forms that most LLMs
    emit.  This function rewrites them so MathJax renders correctly.
    """
    # Display math first (must come before inline to avoid partial matches)
    text = _DISPLAY_MATH_RE.sub(lambda m: f"$$\n{m.group(1).strip()}\n$$", text)
    # Inline math
    text = _INLINE_MATH_RE.sub(lambda m: f"${m.group(1).strip()}$", text)
    return text


def _clean_assistant_message(content: str) -> str:
    """Hide reasoning from chat history, including messages saved before a reload."""
    text = _THINK_BLOCK_RE.sub("", content)
    if opening_tag := _THINK_OPEN_RE.search(text):
        text = text[: opening_tag.start()]
    return text.strip()


def _format_inline_citations(text: str) -> str:
    """Normalise LaTeX delimiters and replace [N] citations with bold markers.

    Returns plain markdown so that st.markdown() renders both Markdown and
    LaTeX (via MathJax) correctly.
    """
    text = _normalize_latex(text)

    def _replace_tag(match: re.Match) -> str:
        idx = match.group(1)
        return f"[{idx}]"

    return _CITATION_REF_RE.sub(_replace_tag, text)


# ── Sidebar ──────────────────────────────────────────────────────────────────


def render_sidebar(pipeline: RAGPipeline, config: AppConfig):
    # Header & Badges
    st.markdown(
        """
        <div class="brand-header">
            <div class="brand-icon">🗎</div>
            <div class="brand-title">Ask My Docs</div>
        </div>
        <div class="brand-badges">
            <span class="dark-pill-badge">Hybrid RAG</span>
            <span class="dark-pill-badge">Reranked</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Document & Chunk counts
    indexed_files = pipeline.vector_store.get_indexed_files()
    total_chunks = pipeline.vector_store.count()
    total_docs = len(indexed_files) if indexed_files else (1 if total_chunks > 0 else 0)

    st.markdown(
        f"""
        <div class="stats-container">
            <div class="stat-card">
                <div class="stat-card-label">Documents</div>
                <div class="stat-card-value">{total_docs:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-card-label">Chunks</div>
                <div class="stat-card-value">{total_chunks:,}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Your Documents
    st.markdown(
        '<div class="sidebar-section-title"><span>Your documents</span></div>',
        unsafe_allow_html=True,
    )

    if indexed_files:
        for f in indexed_files:
            col_name, col_del = st.columns([5, 1])
            with col_name:
                st.markdown(
                    f"""
                    <div class="doc-item">
                        <div class="doc-item-left">🗎 {html.escape(f["name"])}</div>
                        <span style="color: #94a3b8; font-size: 0.75rem;">{f["chunks"]} chunks</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_del:
                if st.button(
                    "🗑",
                    key=f"del_{f['name']}",
                    help=f"Delete {f['name']}",
                    use_container_width=True,
                ):
                    pipeline.vector_store.delete_by_source(f["name"])
                    pipeline.bm25_store.delete_by_source(f["name"])
                    st.rerun()
    else:
        st.caption("No documents indexed yet.")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # ── Upload Documents ──────────────────────────────────────────────────────
    if st.button("📂 Upload Documents", use_container_width=True):
        st.session_state["show_uploader"] = not st.session_state.get("show_uploader", False)

    if st.session_state.get("show_uploader", False):
        uploaded = st.file_uploader(
            "PDF, TXT, DOCX, MD — indexed automatically",
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
            label_visibility="visible",
        )

        if uploaded:
            # Deduplicate by (name, size) so reruns don't re-index the same files
            done: set = st.session_state.setdefault("indexed_upload_keys", set())
            new_files = [uf for uf in uploaded if (uf.name, uf.size) not in done]

            if new_files:
                upload_dir = Path(config.upload_dir)
                upload_dir.mkdir(parents=True, exist_ok=True)
                saved_paths = []
                for uf in new_files:
                    dest = upload_dir / uf.name
                    dest.write_bytes(uf.read())
                    saved_paths.append(dest)
                    done.add((uf.name, uf.size))

                with st.spinner(f"Indexing {len(saved_paths)} file(s)…"):
                    stats = ingest_files(saved_paths, pipeline, config)

                st.success(
                    f"✅ {stats['chunks']} chunks from {stats['files']} file(s) "
                    f"in {stats['elapsed_s']:.1f}s"
                )
                st.rerun()

    # Retrieval Settings (Now cleanly tucked into an expander)

    with st.expander("⚙️ Retrieval settings", expanded=False):
        st.session_state.rerank_top_k = st.slider(
            "Final context chunks",
            min_value=1,
            max_value=25,
            value=int(st.session_state.get("rerank_top_k", config.rerank_top_k)),
        )

        st.session_state.vector_top_k = st.slider(
            "Vector candidates",
            min_value=5,
            max_value=60,
            value=int(st.session_state.get("vector_top_k", config.vector_top_k)),
        )

        st.session_state.bm25_top_k = st.slider(
            "BM25 candidates",
            min_value=5,
            max_value=60,
            value=int(st.session_state.get("bm25_top_k", config.bm25_top_k)),
        )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        st.session_state.retrieval_debug = st.checkbox(
            "Show retrieval debug",
            value=st.session_state.get("retrieval_debug", False),
        )

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    if st.button("🗑️ Clear chat history", use_container_width=True):
        clear_history()
        st.rerun()


# ── Chat Panel ───────────────────────────────────────────────────────────────


def render_chat(pipeline: RAGPipeline, config: AppConfig):
    # Render messages
    for msg in st.session_state.messages:
        content = msg["content"]
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="user-msg-container">
                    <div class="user-msg-bubble">{html.escape(content)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            clean_content = _clean_assistant_message(content)
            formatted_md = _format_inline_citations(clean_content)
            # Wrap in a container div for styling, then render the message
            # body via st.markdown so Streamlit's MathJax/LaTeX pipeline fires.
            st.markdown(
                '<div class="assistant-msg-container">',
                unsafe_allow_html=True,
            )
            st.markdown(formatted_md)
            st.markdown("</div>", unsafe_allow_html=True)
            if msg.get("citations"):
                _render_citations(msg["citations"])

    # Base chat input
    prompt = st.chat_input("Ask anything about your documents")

    # Zero-State Welcome Screen
    if not st.session_state.messages:
        st.markdown(
            """
            <div style="text-align: center; padding: 60px 20px 40px;">
                <h2 style="font-weight: 700; color: #0f172a; margin-bottom: 8px;">Ask My Docs</h2>
                <p style="color: #64748b;">Upload documents and start asking questions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Chat execution block (triggered by input or starter chips)
    if prompt:
        # User message
        add_message("user", prompt, max_history=config.max_history)
        st.markdown(
            f"""
            <div class="user-msg-container">
                <div class="user-msg-bubble">{html.escape(prompt)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Assistant generation
        with st.spinner("Searching and reasoning…"):
            result = pipeline.query(
                question=prompt,
                top_k=st.session_state.get("rerank_top_k", config.rerank_top_k),
                vector_k=st.session_state.get("vector_top_k", config.vector_top_k),
                bm25_k=st.session_state.get("bm25_top_k", config.bm25_top_k),
            )

        answer = result["answer"]
        citations = result["citations"]

        clean_answer = _clean_assistant_message(answer)
        formatted_md = _format_inline_citations(clean_answer)

        # Use st.markdown for the body so MathJax/LaTeX renders correctly.
        st.markdown(
            '<div class="assistant-msg-container">',
            unsafe_allow_html=True,
        )
        st.markdown(formatted_md)
        st.markdown("</div>", unsafe_allow_html=True)
        _render_citations(citations)

        if st.session_state.get("retrieval_debug", False):
            _render_debug(result)

        add_message("assistant", answer, citations, max_history=config.max_history)


def _render_citations(citations: list[dict]):
    if not citations:
        return

    count = len(citations)
    with st.expander(f"🗎 {count} source{'s' if count != 1 else ''}", expanded=False):
        for i, c in enumerate(citations, 1):
            source = Path(str(c.get("source", "document"))).name
            page = c.get("page", "?")
            score = float(c.get("score", 0.0))

            # Calculate percentage match
            if 0 < score <= 1.0:
                match_pct = int(score * 100)
            elif score > 1.0:
                match_pct = min(99, int(50 + score * 8))
            else:
                match_pct = max(40, int(85 - (i - 1) * 7))

            snippet = c.get("snippet", "")
            if not snippet and "text" in c:
                snippet = c["text"][:250]

            st.markdown(
                f"""
                <div class="citation-card">
                    <div class="citation-card-top">
                        <span class="citation-card-link">{i} · {html.escape(source)}, p.{page}</span>
                        <span class="citation-card-match">{match_pct}% match</span>
                    </div>
                    <div class="citation-card-snippet">{html.escape(snippet)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_debug(result: dict):
    with st.expander("🔍 Retrieval Debug", expanded=False):
        st.json(
            {
                "vector_hits": result.get("vector_hits", 0),
                "bm25_hits": result.get("bm25_hits", 0),
                "after_rerank": result.get("after_rerank", 0),
                "latency_ms": result.get("latency_ms", 0),
            }
        )
        st.markdown("**Reranked chunks:**")
        for chunk in result.get("chunks", []):
            st.markdown(
                f"- `{chunk.get('source', '?')}` p{chunk.get('page', '?')} — score {chunk.get('score', 0):.4f}"
            )
            st.caption(chunk.get("text", "")[:200] + "…")


# ── Upload Panel ─────────────────────────────────────────────────────────────


def render_upload_panel(pipeline: RAGPipeline, config: AppConfig):
    st.markdown("### 📂 Upload & Index Documents")
    st.info(
        "Supported formats: **PDF, TXT, DOCX, MD**. "
        "Uploaded files are chunked, embedded, and added to the active session index."
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
            f"✅ Successfully indexed **{stats['chunks']} chunks** from "
            f"**{stats['files']} file(s)** in {stats['elapsed_s']:.1f}s"
        )
        st.rerun()


# ── Documents Explorer Panel ─────────────────────────────────────────────────


def render_documents_panel(pipeline: RAGPipeline, config: AppConfig):
    st.markdown("### 📑 Indexed Documents")
    st.caption("Inspect and manage documents in your current session.")

    indexed_files = pipeline.vector_store.get_indexed_files()

    if not indexed_files:
        st.info("No documents indexed yet. Use the **Upload** tab to add documents.")
        return

    st.markdown(
        f"**{len(indexed_files)} document(s)** with **{pipeline.vector_store.count():,} total chunks**"
    )

    for doc in indexed_files:
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**📄 {doc['name']}**")
                st.caption(f"{doc['chunks']} indexed chunks")
            with col2:
                st.button("View", key=f"view_{doc['name']}", use_container_width=True)
            st.divider()
