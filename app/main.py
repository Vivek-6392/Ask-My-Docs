"""
Ask My Docs — Production RAG Application
Main Streamlit entry point
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ui import render_sidebar, render_chat, render_upload_panel
from app.session import init_session
from retrieval.pipeline import RAGPipeline
from app.config import AppConfig

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
def get_pipeline(config: AppConfig) -> RAGPipeline:
    return RAGPipeline(config)


def main():
    config = AppConfig.from_env()
    init_session()

    pipeline = get_pipeline(config)

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
