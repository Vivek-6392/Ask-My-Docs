"""
Streamlit session-state helpers.
"""

import streamlit as st


def init_session():
    defaults = {
        "messages": [],           # [{role, content, citations}]
        "doc_count": 0,
        "retrieval_debug": False,
        "selected_collection": "default",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def add_message(role: str, content: str, citations: list | None = None):
    st.session_state.messages.append(
        {"role": role, "content": content, "citations": citations or []}
    )
    # Trim history
    if len(st.session_state.messages) > 40:
        st.session_state.messages = st.session_state.messages[-40:]


def clear_history():
    st.session_state.messages = []
