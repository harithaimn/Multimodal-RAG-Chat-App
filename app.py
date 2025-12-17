# app.py

import streamlit as st
from src import streamlitUi

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="INVOKE – Multimodal RAG (DM Internal Tool)",
    layout="wide"
)

# -------------------------------------------------
# Optional Password Gate
# -------------------------------------------------

def check_password():
    """Simple password protection using Streamlit secrets."""
    if st.session_state.get("password_correct", False):
        return True

    def password_entered():
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    st.text_input(
        "Password",
        type="password",
        on_change=password_entered,
        key="password"
    )

    if "password_correct" in st.session_state and not st.session_state.password_correct:
        st.error("Incorrect password")

    return False


# -------------------------------------------------
# Application Entry
# -------------------------------------------------

if check_password():
    streamlitUi.main()