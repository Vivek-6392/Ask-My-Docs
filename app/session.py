"""
Streamlit session-state helpers.
"""

import uuid

import streamlit as st


def init_session(config=None):
    defaults = {
        "messages": [],  # [{role, content, citations}]
        "doc_count": 0,
        "retrieval_debug": False,
        "selected_collection": "default",
        "session_id": str(uuid.uuid4()),
        "rerank_top_k": config.rerank_top_k if config else 10,
        "vector_top_k": config.vector_top_k if config else 40,
        "bm25_top_k": config.bm25_top_k if config else 25,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def add_message(
    role: str,
    content: str,
    citations: list | None = None,
    max_history: int = 20,
):
    st.session_state.messages.append(
        {"role": role, "content": content, "citations": citations or []}
    )
    # Trim history
    if len(st.session_state.messages) > max_history:
        st.session_state.messages = st.session_state.messages[-max_history:]


def clear_history():
    st.session_state.messages = []
