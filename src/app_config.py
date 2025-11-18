# ============================================================
# src/app_config.py
# ============================================================
# Centralized configuration for dropdowns and static selections
# Used by Streamlit UI to populate dropdowns dynamically
# ============================================================


# CAMPAIGN_OBJECTIVES = (
#     "All",
#     "App Installs",
#     "Brand Awareness",
#     "Conversions",
#     "Event Responses",
#     "Lead Generation",
#     "Link Clicks",
#     "Messages",
#     "Page Likes",
#     "Post Engagement",
#     "Product Catalog Sales",
#     "Reach",
#     "Store Visit",
#     "Video Views"
# )

CAMPAIGN_OBJECTIVES = [
    "Awareness",
    "Engagement",
    "Traffic",
    "Conversions",
    "Lead Generation",
    "Retention",
    "Brand Loyalty",
]


TARGET_MARKETS = (
    "All",
    "T20",
    "M40",
    "B40"
)

AGE_GROUPS = (
    "All",
    "Gen Z (18-24)",
    "Millennials (25-39)",
    "Gen X (40-55)"
)

# ============================================================
# 5. PSYCHOGRAPHICS
# ============================================================
PSYCHOGRAPHICS = [
    "health-conscious",
    "budget-seeker",
    "premium-taste",
    "tech-savvy",
    "family-oriented",
    "environmentally-aware",
    "trend-follower",
    "status-driven",
]


INDUSTRIES = [
    "F&B",
    "Beauty",
    "Finance",
    "Retail",
    "Automotive",
    "Education",
    "Healthcare",
    "Technology",
    "Travel",
    "Real Estate",
    "Entertainment",
    "Fashion",
]

"""
INDUSTRIES = (
    "All",
    "Advertising & Communications",
    "Agriculture & Forestry/Wildlife",
    "Automotive",
    "Beauty & Body Care",
    "Consumer Services",
    "Education",
    "FMCG",
    "Fashion & Lifestyle",
    "Finance & Insurance",
    "Food & Beverages",
    "Healthcare & Life Sciences",
    "Information, Tech & Telecommunications",
    "Interior Design & Construction",
    "Leisure, Tourism & Travel",
    "Logistics & Transportation",
    "Manufacturing & Production",
    "Media & Entertainment",
    "NGOs",
    "Natural Resources & Energy",
    "Property & Real Estate",
)
"""

#
# AD_FORMATS = (
#     "All",
#     "Image",
#     "Video",
#     "Carousel"
# )

COUNTRIES = (
    "All",
    "MY",
    "SG"
)

ADS_RESPONSE_LANGUAGE = (
    "English",
    "Malay",
    "Mandarin"
)



# ============================================================
# 10. UTILITY FUNCTION (for Streamlit UI)
# ============================================================

def get_filter_options():
    """
    Returns a dictionary of dropdown configurations for Streamlit UI.
    Useful for dynamically building filter panels or restoring session state.
    """
    return {
        "Industry": INDUSTRIES,
        "Campaign Objective": CAMPAIGN_OBJECTIVES,
        "Target Market": TARGET_MARKETS,
        "Language": ADS_RESPONSE_LANGUAGE,
        "Psychographics": PSYCHOGRAPHICS,
        #"Platform": PLATFORMS,
        #"Ad Format": AD_FORMATS,
    }