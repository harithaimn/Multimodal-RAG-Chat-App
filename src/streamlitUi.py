# src/streamlitUi.py
"""
Streamlit UI for RAG-powered Ad Generator

Principles:
- Generation-first
- Explanation-second
- RAG is invisible to the user
"""

import streamlit as st
from openai import OpenAI

from src.vectorstore import init_vectorstore, init_embeddings, retrieve_pattern_docs
from src.openai_chain import generate_ad_with_patterns
from src.context_rules import enforce_context_rules

from dotenv import load_dotenv
import os

load_dotenv()

# =====================================================
# Clients (cached)
# =====================================================

@st.cache_resource
def load_clients():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment")

    client = OpenAI(api_key=api_key)
    index = init_vectorstore()
    embeddings = init_embeddings()
    return client, index, embeddings


# =====================================================
# Page Config
# =====================================================

def main():
    st.set_page_config(
        page_title="Ad Generator",
        layout="centered",
    )

    st.title("Ad Copy Generator")
    st.caption("Generate high-conversion ads inspired by historical performance patterns.")

    client, index, embeddings = load_clients()

    # =====================================================
    # Sidebar Controls
    # =====================================================

    st.sidebar.header("Ad Settings")

    business_type = st.sidebar.text_input(
        "Business type",
        placeholder="e.g. F&B, Burger Stall",
    )

    product = st.sidebar.text_input(
        "Product / Offer",
        placeholder="e.g. Smash burger RM6",
    )

    platform = st.sidebar.selectbox(
        "Platform",
        ["Meta Ads", "Instagram", "Facebook", "TikTok"],
    )

    language_style = st.sidebar.selectbox(
        "Language style",
        [
            "Casual Malaysian English",
            "Bahasa Melayu (Santai)",
            "English (Direct & Punchy)",
            "Mix BM + English",
        ],
    )

    mode = st.sidebar.radio(
        "Output mode",
        ["Generate Ad + Explain Why", "Generate Ad Only"],
    )

    # =====================================================
    # Main Action
    # =====================================================

    if st.button("Generate Ad", type="primary"):
        if not business_type or not product:
            st.error("Please fill in Business type and Product.")
            st.stop()

        with st.spinner("Generating ad..."):
            # -------------------------------------------------
            # 1. Retrieve pattern signals (RAG)
            # -------------------------------------------------
            query = f"{business_type} {product} ad performance"
            rag_docs = retrieve_pattern_docs(
                index=index,
                embeddings=embeddings,
                query=query,
            )

            # -------------------------------------------------
            # 2. Generate output
            # -------------------------------------------------
            raw_output = generate_ad_with_patterns(
                client=client,
                rag_docs=rag_docs,
                business_type=business_type,
                product=product,
                platform=platform,
                language_style=language_style,
            )

            # -------------------------------------------------
            # 3. Enforce context rules
            # -------------------------------------------------
            try:
                result = enforce_context_rules(
                    f"""
[AD COPY]
{raw_output['ad_copy']}

[WHY THIS WORKS (PATTERN REFERENCE)]
{raw_output['pattern_explanation']}
"""
            )
            except Exception as e:
                st.error(f"Output validation failed: {e}")
                st.stop()

        # =================================================
        # Output Rendering
        # =================================================

        st.subheader("Generated Ad Copy")
        st.text_area(
            label="",
            value=result["ad_copy"],
            height=120,
        )

        if mode == "Generate Ad + Explain Why":
            st.subheader("Why This Works")
            st.markdown(result["why"])

        # =================================================
        # Debug (optional, collapsed)
        # =================================================

        with st.expander("Debug (pattern signals)"):
            for d in rag_docs:
                st.code(d["text"])