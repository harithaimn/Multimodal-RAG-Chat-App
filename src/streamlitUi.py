# src/streamlitUi.py

"""
Streamlit UI for Multimodal RAG (DM Internal Tool)

Responsibilities:
- Render sidebar filters
- Manage chat session state
- Call RAGChain
- Render multimodal responses (text + images)

Single public entry:
- main()
"""

import streamlit as st
import uuid

from src.app_config import get_filter_options
from src.context_rules import (
    normalize_filters,
    validate_user_input_vs_filters
)
from src.openai_chain import RAGChain


# -------------------------------------------------
# Session Initialization
# -------------------------------------------------

def _init_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "filters" not in st.session_state:
        st.session_state.filters = {}


# -------------------------------------------------
# Sidebar Filters
# -------------------------------------------------

def _render_sidebar_filters():
    st.sidebar.header("🎯 Filters")

    filter_options = get_filter_options()
    raw_filters = {}

    for key, options in filter_options.items():
        raw_filters[key] = st.sidebar.selectbox(
            key.replace("_", " ").title(),
            options,
            index=0,
            key=f"filter_{key}"
        )

    st.session_state.filters = raw_filters
    return normalize_filters(raw_filters)


# -------------------------------------------------
# Chat History Renderer
# -------------------------------------------------

def _render_chat_history(container):
    for msg in st.session_state.chat_history:
        with container.chat_message(msg["role"]):
            st.markdown(msg["content"])


# -------------------------------------------------
# Assistant Response Renderer
# -------------------------------------------------

def _render_assistant_response(container, answer, retrieved):
    with container.chat_message("assistant"):
        st.markdown(answer)

        # Render first available image (optional)
        for r in retrieved:
            img_url = r.get("image_url")
            if img_url:
                st.image(img_url, use_container_width=True)
                break


# -------------------------------------------------
# Main Application Entry
# -------------------------------------------------

def main():
    _init_session()

    st.title("🧠 Multimodal RAG – Digital Marketing Assistant")
    st.caption("Dataset-grounded insights powered by Supermetrics + Pinecone")

    # Sidebar
    filters = _render_sidebar_filters()

    # Main chat container
    chat_container = st.container()
    _render_chat_history(chat_container)

    # User input
    user_input = st.chat_input(
        "Ask about campaign performance, creatives, or insights..."
    )

    if not user_input:
        return

    # Validation (soft warnings only)
    warnings = validate_user_input_vs_filters(user_input, filters)
    for w in warnings:
        st.warning(w)

    # Append user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    # Run RAG
    rag = RAGChain(chat_history=st.session_state.chat_history)
    answer, retrieved = rag.run(user_input, filters=filters)

    # Append assistant message
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })

    # Render response
    _render_assistant_response(chat_container, answer, retrieved)