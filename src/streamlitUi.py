# ============================================================
# src/streamlitUi.py
# ============================================================
# Streamlit-based UI for Campaign OS
# - Chat Interface
# - Sidebar Filters
# - Session Restoration & Validation
# ============================================================

import streamlit as st
from src.app_config import get_filter_options
from src.context_rules import detect_intended_industry, validate_user_input
from src.utils import (
    load_chat_history_from_s3,
    save_chat_history_to_s3,
    get_s3_key_for_session,
)
from datetime import datetime
import uuid
import json

# ============================================================
# 1. INITIALIZATION
# ============================================================

st.set_page_config(page_title="Campaign OS", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "filters" not in st.session_state:
    st.session_state.filters = {}


# ============================================================
# 2. SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🎯 Campaign Filters")

filter_options = get_filter_options()
selected_filters = {}

for key, options in filter_options.items():
    selected_filters[key] = st.sidebar.selectbox(
        f"{key}",
        options,
        index=0 if key != "Psychographics" else None,
        key=f"filter_{key}",
    )

# Save current filters in session
st.session_state.filters = selected_filters


# ============================================================
# 3. SESSION MANAGEMENT
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("💾 Session Management")

# Load saved chat history (from S3)
if st.sidebar.button("🔁 Restore Last Session"):
    with st.spinner("Restoring previous chat and filters..."):
        data = load_chat_history_from_s3(st.session_state.session_id)
        if data:
            st.session_state.chat_history = data.get("chat_history", [])
            st.session_state.filters = data.get("filters", selected_filters)
            st.success("✅ Session restored from S3.")
        else:
            st.warning("No previous session found on S3.")

# Rename session
new_session_name = st.sidebar.text_input("Rename Session", value="Campaign Chat")
st.session_state.session_name = new_session_name


# ============================================================
# 4. MAIN CHAT INTERFACE
# ============================================================

st.title("🧠 Campaign OS Assistant")
st.caption("AI-powered copywriting, audience insight, and creative ideation engine.")

chat_container = st.container()

# Display chat history
for msg in st.session_state.chat_history:
    role = msg["role"]
    with chat_container.chat_message(role):
        st.markdown(msg["content"])

# User input field with Shift+Enter multiline support
user_input = st.chat_input("Type your campaign brief or question here...")

# ============================================================
# 5. INPUT VALIDATION & CONTEXT CHECK
# ============================================================

if user_input:
    # Step 1: Detect industry intent
    intended_industry = detect_intended_industry(user_input)
    selected_industry = st.session_state.filters.get("Industry")

    # Step 2: Run validation heuristics
    validation_issues = validate_user_input(user_input, st.session_state.filters)

    if intended_industry and intended_industry != selected_industry:
        st.warning(
            f"⚠️ Detected intent for **{intended_industry}**, "
            f"but your filter is set to **{selected_industry}**."
        )

    if validation_issues:
        st.error("⚠️ Input validation failed:")
        for issue in validation_issues:
            st.markdown(f"- {issue}")

    # Step 3: Append user input to chat
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # ============================================================
    # 6. SIMULATE GENERATION / RESPONSE (placeholder for RAG chain)
    # ============================================================
    with chat_container.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Placeholder for your RAG pipeline integration
            # E.g. response = rag_chain.run(user_input, filters=selected_filters)
            response = (
                f"Generated campaign insights for {selected_industry} "
                f"({selected_filters['Campaign Objective']}) — coming soon!"
            )
            st.markdown(response)

    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # ============================================================
    # 7. SAVE SESSION (Chat + Filters) TO S3
    # ============================================================
    session_payload = {
        "chat_history": st.session_state.chat_history,
        "filters": st.session_state.filters,
        "timestamp": datetime.utcnow().isoformat(),
    }

    save_chat_history_to_s3(st.session_state.session_id, session_payload)
    st.toast("💾 Session saved to S3.")


# ============================================================
# 8. VISUAL RECOMMENDATIONS / OUTPUT DISPLAY
# ============================================================

st.markdown("---")
st.subheader("📊 Top Reference Ad Example")

col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://via.placeholder.com/300x300.png?text=Ad+Thumbnail", use_container_width=True)

with col2:
    st.metric(label="CTR", value="4.7%", delta="+0.9% MoM")
    st.metric(label="Engagement Rate", value="6.3%", delta="+1.2% MoM")
    st.markdown("**Insight:** High-performing creative featuring emotional appeal and local language use.")

st.markdown("---")
st.caption("💡 Tip: Adjust filters in the sidebar to refine campaign context or audience insights.")
