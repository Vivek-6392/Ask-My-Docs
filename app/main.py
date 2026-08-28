"""
Ask My Docs — Production RAG Application
Main Streamlit entry point
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import AppConfig
from app.session import init_session
from app.ui import render_chat, render_sidebar
from retrieval.pipeline import RAGPipeline

st.set_page_config(
    page_title="Ask My Docs",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_file = Path(__file__).parent / "styles.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_pipeline(config: AppConfig, session_id: str) -> RAGPipeline:
    # NOTE (Data eviction / lifecycle):
    # In long-lived production deployments with many ephemeral user sessions,
    # implement a background reaper/TTL eviction mechanism to periodically delete
    # old ChromaDB collections and BM25 index files for expired session IDs.
    return RAGPipeline(config, session_id=session_id)


def main():
    config = AppConfig.from_env()
    init_session(config)

    pipeline = get_pipeline(config, st.session_state.session_id)

    # Sidebar
    with st.sidebar:
        render_sidebar(pipeline, config)

    # Main area — chat only
    render_chat(pipeline, config)


if __name__ == "__main__":
    main()
