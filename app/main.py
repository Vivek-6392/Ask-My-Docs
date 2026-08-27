"""
Ask My Docs — Production RAG Application
Main Streamlit entry point
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import AppConfig
from app.session import init_session
from app.ui import render_chat, render_sidebar, render_upload_panel
from retrieval.pipeline import RAGPipeline

st.set_page_config(
    page_title="Ask My Docs",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
with open(os.path.join(os.path.dirname(__file__), "styles.css")) as f:
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

    # Layout
    col_sidebar, col_main = st.columns([1, 3])

    with col_sidebar:
        render_sidebar(pipeline, config)

    with col_main:
        tab_chat, tab_upload = st.tabs(["💬 Chat", "📂 Upload Documents"])
        with tab_chat:
            render_chat(pipeline, config)
        with tab_upload:
            render_upload_panel(pipeline, config)


if __name__ == "__main__":
    main()
