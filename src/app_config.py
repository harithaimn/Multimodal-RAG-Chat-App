# ============================================================
# src/app_config.py
# ============================================================
# Centralized configuration for Streamlit filters
# Aligned strictly to Supermetrics → GSheet columns
# ============================================================

# -----------------------------
# Campaign-level filters
# -----------------------------

CAMPAIGN_OBJECTIVES = (
    "All",
    "Awareness",
    "Engagement",
    "Traffic",
    "Conversions",
    "Lead Generation",
)

CAMPAIGN_STATUSES = (
    "All",
    "Active",
    "Paused",
    "Deleted",
)

AD_STATUSES = (
    "All",
    "Active",
    "Paused",
    "Deleted",
)


# -----------------------------
# Output / UI preferences
# -----------------------------

ADS_RESPONSE_LANGUAGE = (
    "English",
    "Malay",
    "Mandarin",
)


# ============================================================
# Utility function (used by Streamlit UI)
# ============================================================

def get_filter_options():
    """
    Returns filter dropdown options for Streamlit UI.

    IMPORTANT:
    - Every filter here must exist in Pinecone metadata
    - No inferred / hallucinated attributes
    """
    return {
        "campaign_objective": CAMPAIGN_OBJECTIVES,
        "campaign_status": CAMPAIGN_STATUSES,
        "ad_status": AD_STATUSES,
        "language": ADS_RESPONSE_LANGUAGE,
    }
