import streamlit as st
from datetime import datetime
from src.app_config import (
    CAMPAIGN_OBJECTIVES,
    TARGET_MARKETS,
    INDUSTRIES,
    COUNTRIES
)
from src.utils import detect_intended_industry, save_chat_metadata
from src.context_rules import DROPDOWN_CONTEXTS


# ---------------------------
# SIDEBAR: Chat Session
# ---------------------------
def render_sidebar(saved_sessions: list):
    """
    Renders the sidebar UI components with chat management features, and returns user actions.
    
    Args:
        saved_sessions (list): A list of saved session dictionaries:
        {
            "session_id": str,
            "title": str,
            "last_modified": datetime
        }
    
    Returns:
        str or None: The session_id of a selected chat, or None if no selection is made.
    """
    st.title("Chat Conversations")
        
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.session_id = None # Signal to create a new session
        st.rerun()

    st.markdown("---")
    
    if not saved_sessions:
        st.write("No saved conversations yet.")
        return None
    
    st.markdown("### Past Conversations:")
    
    for session in saved_sessions:
        session_id = session['session_id']
        title = session.get("title") or f"Chat from {session['last_modified'].strftime('%Y-%m-%d %H:%M')}"
        
        # Row container
        cols = st.columns([7, 2, 1])    # [Title, Rename, Open]
        with cols[0]:
            st.markdown(f"**{title}**")
        
        with cols[1]:
            if st.button("✏️ Rename", key=f"rename_{session_id}"):
                new_title = st.text_input(
                    "Enter new title:",
                    value=title,
                    key=f"title_input_{session_id}"
                )

                if st.button("💾 Save", key=f"save_{session_id}"):
                    session["title"] = new_title.strip() or title
                    session["last_modified"] = datetime.now()
                    
                    save_chat_metadata(session)

                    st.success(f"Renamed to **{session['title']}** ✅")

                    st.rerun()
            
        with cols[2]:
            if st.button("📂 Open", key=f"open_{session_id}"):
                st.session_state.session_id = session_id
                st.rerun()

# -------------------------------------------------
# Filters: Campaign Settings & Language Options
# -------------------------------------------------
def render_filters() -> dict:
    """
    Renders the collapsible filter section and returns the selected filter values.
    
    Returns:
        dict: A dictionary containing the current values of all filters.
    """
    st.markdown("## Campaign/Ads Filters")
    filters = {}

    with st.expander("Campaign Configuration", expanded=True):
        filters['campaign_objective'] = st.selectbox(
            "Campaign Objective:",
            options=CAMPAIGN_OBJECTIVES
        )
        filters['target_market'] = st.selectbox(
            "Target Market:",
            options=TARGET_MARKETS
        )
        filters['industry'] = st.selectbox(
            "Industry:",
            options=INDUSTRIES
        )
        filters['country'] = st.selectbox(
            "Target Country:",
            options=COUNTRIES
        )
        filters['ads_language'] = st.selectbox(
            "Ads Copy Language:",
            options=("English", "Malay", "Mandarin")
        )
        
    return filters

# ---------------------------------------------------------
# Validation: Ensure user input matches selected dropdowns
# ---------------------------------------------------------
def validate_user_input(user_text: str, filters: dict):
    """
    Check for semantic mismatch between user input and dropdown selections.
    Shows Streamlit warnings if mismatch is detected.
    """

    if not user_text.strip():
        return # Nothing to validate yet
    
    text = user_text.lower()
  
    #detected_industry = detect_intended_industry(user_text)
    detected_industry = None

    for industry_name, data in DROPDOWN_CONTEXTS.get("industry", {}).items():
        keywords = [kw.lower() for kw in data.get("keywords", [])]
        if any(kw in text for kw in keywords):
            detected_industry = industry_name
            break
        
    if detected_industry and detected_industry != filters['industry']:
        st.warning(
            f"⚠️ Your text seems related to **{detected_industry}**, "
            f"but you selected **{filters['industry']}**.\n\n"
            f"💡 Did you mean to select **{detected_industry}** industry instead?"
        )
    
    # ---------------------------
    # Campaign objective mismatch (existing logic)
    # ---------------------------
    # Awareness
    if any(x in text for x in ["awareness", "reach", "exposure"]) and filters.get('campaign_objective') != "Brand Awareness":
        st.info("💡 You mentioned 'awareness' or 'reach' — consider selecting **Brand Awareness** as your objective.")

    # Lead generation
    if any(x in text for x in ["leads", "signup", "form", "conversion", "customer info"]) and filters.get('campaign_objective') != "Lead Generation":
        st.info("💡 You mentioned 'leads' or 'signups' — consider selecting **Lead Generation**.")

    # Conversions
    if any(x in text for x in ["purchase", "buy now", "checkout", "add to cart", "sales"]) and filters.get('campaign_objective') != "Conversions":
        st.info("💡 You mentioned 'purchase' or 'sales' — consider selecting **Conversions**.")

    # App installs
    if any(x in text for x in ["install", "download app", "mobile app"]) and filters.get('campaign_objective') != "App Installs":
        st.info("💡 You mentioned 'install' or 'download app' — consider selecting **App Installs**.")

    # Video views
    if any(x in text for x in ["watch", "video", "views"]) and filters.get('campaign_objective') != "Video Views":
        st.info("💡 You mentioned 'video' — consider selecting **Video Views** as your objective.")

    # Engagement
    if any(x in text for x in ["like", "comment", "share", "interaction", "engagement"]) and filters.get('campaign_objective') != "Post Engagement":
        st.info("💡 You mentioned engagement-related actions — consider selecting **Post Engagement**.")

    # Messages
    if any(x in text for x in ["dm", "message us", "whatsapp", "inbox"]) and filters.get('campaign_objective') != "Messages":
        st.info("💡 You mentioned messaging — consider selecting **Messages** objective.")

    # Traffic
    if any(x in text for x in ["link click", "website visit", "traffic"]) and filters.get('campaign_objective') != "Link Clicks":
        st.info("💡 You mentioned website or traffic — consider selecting **Link Clicks** objective.")

# -------------------------------------------------------------
# Main Chat Interface
# -------------------------------------------------------------
def render_chat_interface(chat_history):
    """
    Renders the main chat message display area.
    """
    st.title("🤖 Multimodal RAG LLM Chatbot")
    st.caption("An AI assistant for INVOKE’s Internal Marketing Ideation & Analysis")
    # st.caption("An AI assistant/ideation for the Inhouse INVOKE's Digital Marketing Team")
    
    if not chat_history or not getattr(chat_history, "messages", None):
        st.info("Start a conversation using the chat input below.")
        return
    
    for msg in chat_history.messages:
        st.chat_message(msg.type).write(msg.content)