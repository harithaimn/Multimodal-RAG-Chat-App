# ===============================================================
# src/context_rules.py
# ===============================================================
# Centralized contextual mapping for dropdown-based filtering
# Used by:
# - vectorstore.py -> metadata filters
# - streamlitUi.py -> mismatch detection
# - openai_chain.py -> contextual prompt enrichment
# ===============================================================

import re
from typing import Dict, List, Optional

DROPDOWN_CONTEXTS = {
    "industry": {
        "F&B": {
            "context": (
                "Focus on ads related to food, beverage, restaurants, and cafes. "
                "Think about appetite appeal, sensory language, and taste-driven visuals."
            ),
            "keywords": ["burger", "restaurant", "drink", "menu", "food", "coffee", "snack", "meal", "cafe"]
        },
        "Beauty": {
            "context": (
                "Focus on skincare, makeup, cosmetics, and beauty routines. "
                "Emphasize confidence, glow, transformation, and authenticity."
            ),
            "keywords": ["skincare", "lipstick", "serum", "foundation", "makeup", "beauty", "skin"]
        },
        "Finance": {
            "context": (
                "Focus on financial services, insurance, loans, or investment-related ads. "
                "Highlight trust, stability, ROI, and risk-free growth."
            ),
            "keywords": ["loan", "insurance", "investment", "bank", "savings", "financial"]
        },
        "Retail": {
            "context": (
                "Focus on promotions, sales, and e-commerce products. "
                "Emphasize urgency, limited offers, discounts, and shopping convenience."
            ),
            "keywords": ["sale", "discount", "store", "offer", "promo", "shopping", "buy"]
        },
        "Automotive": {
            "context": (
                "Focus on cars, motorcycles, EVs, and mobility solutions. "
                "Emphasize power, design, reliability, and innovation."
            ),
            "keywords": ["car", "vehicle", "automotive", "motorcycle", "EV", "ride"]
        },
        "Education": {
            "context": (
                "Focus on universities, schools, online learning, and training ads. "
                "Highlight opportunity, future growth, knowledge, and career readiness."
            ),
            "keywords": ["university", "course", "learning", "training", "education", "study"]
        },
        "Healthcare": {
            "context": (
                "Focus on medical services, wellness, supplements, and hospitals. "
                "Emphasize safety, trust, and professional care."
            ),
            "keywords": ["hospital", "doctor", "clinic", "medicine", "healthcare", "wellness"]
        },
    },

    "campaign_objective": {
        "Awareness": {
            "context": (
                "The goal is brand recognition and recall. "
                "Focus on emotional hooks, slogans, and memorable imagery."
            ),
            "keywords": ["awareness", "brand", "reach", "exposure"]
        },
        "Engagement": {
            "context": (
                "The goal is social interaction. "
                "Focus on questions, community-building, and relatable visuals."
            ),
            "keywords": ["engagement", "likes", "comments", "share"]
        },
        "Traffic": {
            "context": (
                "The goal is website or landing page visits. "
                "Focus on curiosity and clear CTAs like 'Learn More' or 'Visit Now'."
            ),
            "keywords": ["traffic", "website", "click", "visit"]
        },
        "Conversions": {
            "context": (
                "The goal is to drive purchases, signups, or actions. "
                "Focus on urgency, value, and proof — 'Buy Now', 'Get Yours Today'."
            ),
            "keywords": ["purchase", "signup", "conversion", "buy", "order"]
        },
        "Lead Generation": {
            "context": (
                "The goal is to collect customer information for follow-up. "
                "Focus on forms, incentives, and trust signals."
            ),
            "keywords": ["form", "register", "contact", "lead"]
        },
    },

    "ads_language": {
        "English": {
            "context": "Generate responses in clear, persuasive English copywriting tone.",
            "keywords": ["the", "is", "and", "you", "your"]
        },
        "Malay": {
            "context": "Generate responses fully in Bahasa Melayu. Use persuasive, local tone with cultural relevance.",
            "keywords": ["anda", "kami", "dapatkan", "promosi", "sekarang"]
        },
        "Chinese": {
            "context": "Generate responses in Simplified Chinese. Maintain concise and aspirational tone.",
            "keywords": ["购买", "优惠", "品牌", "客户"]
        }
    },

    "psychographics": {
        "health-conscious": {
            "context": (
                "Audience values wellness, organic products, and long-term health benefits. "
                "Focus on natural ingredients and trustworthiness."
            ),
            "keywords": ["organic", "healthy", "natural", "low-fat", "wellness"]
        },
        "budget-seeker": {
            "context": (
                "Audience prioritizes affordability and value for money. "
                "Focus on deals, discounts, and cost savings."
            ),
            "keywords": ["cheap", "discount", "save", "budget", "value"]
        },
        "premium-taste": {
            "context": (
                "Audience values exclusivity and premium experience. "
                "Focus on luxury, quality, and sophistication."
            ),
            "keywords": ["luxury", "exclusive", "premium", "quality", "elite"]
        },
        "tech-savvy": {
            "context": (
                "Audience embraces innovation, apps, and tech-driven convenience. "
                "Highlight cutting-edge features and modernity."
            ),
            "keywords": ["app", "AI", "smart", "digital", "tech"]
        },
    },
}


# ============================================================
# 2. INTENT DETECTION
# ============================================================

def detect_intended_industry(user_input: str) -> Optional[str]:
    """
    Detect probable industry from freeform user input using keyword matches.
    Returns an industry string if found, else None.
    """
    user_input_lower = user_input.lower()
    for industry, data in DROPDOWN_CONTEXTS["industry"].items():
        for kw in data["keywords"]:
            if kw.lower() in user_input_lower:
                return industry
    return None

# ============================================================
# 3. VALIDATION LOGIC
# ============================================================

def validate_user_input(user_input: str, selected_filters: Dict[str, str]) -> List[str]:
    """
    Validates if user input semantically aligns with chosen dropdown filters.
    Returns a list of warning messages (empty if all good).
    """
    issues = []
    text = user_input.lower()

    # Validate industry vs text
    industry = selected_filters.get("Industry")
    intended = detect_intended_industry(user_input)
    if intended and industry and intended != industry:
        issues.append(
            f"Detected industry intent '{intended}' differs from selected filter '{industry}'."
        )

    # Psychographics check
    psych = selected_filters.get("Psychographics")
    if psych and psych in DROPDOWN_CONTEXTS["psychographics"]:
        keywords = DROPDOWN_CONTEXTS["psychographics"][psych]["keywords"]
        if not any(kw.lower() in text for kw in keywords):
            issues.append(f"User input does not reflect {psych.lower()} psychographic themes.")

    # Platform hint check
    platform = selected_filters.get("Platform")
    if platform and platform in DROPDOWN_CONTEXTS["platform"]:
        platform_keywords = DROPDOWN_CONTEXTS["platform"][platform]["keywords"]
        if not any(kw.lower() in text for kw in platform_keywords):
            issues.append(f"Consider platform-specific phrasing for {platform} (e.g., {platform_keywords[0]}).")

    return issues

# ==================================================
# Helper Functions
# =================================================

def get_content(category: str, value: str) -> str:
    """ Return context string for a given dropdown selection."""
    return DROPDOWN_CONTEXTS.get(category, {}).get(value, {}).get("context", "")

def get_keywords(category: str, value: str) -> list:
    """ Return keyword list for a given dropdown selection. """
    return DROPDOWN_CONTEXTS.get(category, {}).get(value, {}).get("keywords", [])

# ============================================================
# 4. BUILD METADATA FILTERS (for Vectorstore)
# ============================================================

def build_metadata_filters(selected_filters: Dict[str, str]) -> Dict[str, str]:
    """
    Converts selected dropdowns into Pinecone-compatible metadata filters.
    Example output:
    {
        "industry": "F&B",
        "campaign_objective": "Conversions",
        "ads_language": "English",
        "psychographics": "Health-Conscious"
    }
    """
    mapping = {
        "Industry": "industry",
        "Campaign Objective": "campaign_objective",
        "Target Market": "target_market",
        "Ads Language": "ads_language",
        "Psychographics": "psychographics",
        "Platform": "platform",
    }

    filters = {}
    for key, val in selected_filters.items():
        if key in mapping and val:
            filters[mapping[key]] = val
    return filters


def detect_mismatch(user_input: str, filters: Dict[str, str]) -> list:
    """
    Detect mismatches between user text and selected dropdowns.
    Example:
        user_input: "Apply for a quick loan today"
        filters: {"industry": "Beauty"}
        => returns ["Industry mismatch: 'loan' implies Finance"]
    """
    mismatches = []
    for cat, val in filters.items():
        keywords = get_keywords(cat, val)
        # Skip if no mapping or user input is empty
        if not keywords or not user_input:
            continue
        
        # Flatten all keywords across other categories for comparison
        for other_cat, other_map in DROPDOWN_CONTEXTS.items():
            if other_cat == cat:
                continue
            for other_val, data in other_map.items():
                for kw in data["keywords"]:
                    if kw.lower() in user_input.lower() and other_val != val:
                        mismatches.append(f"{cat.capitalize()} mismatch: '{kw}' implies {other_val}")
    return mismatches



# ============================================================
# 5. ENRICH PROMPT CONTEXT (for OpenAI Chain)
# ============================================================

def enrich_prompt_context(selected_filters: Dict[str, str]) -> str:
    """
    Builds human-readable guiding context for LLM prompts.
    """
    context_parts = []
    for k, v in selected_filters.items():
        domain = k.lower().replace(" ", "_")
        if domain in DROPDOWN_CONTEXTS and v in DROPDOWN_CONTEXTS[domain]:
            ctx = DROPDOWN_CONTEXTS[domain][v]["context"]
            context_parts.append(f"{k}: {ctx}")
    return "\n".join(context_parts)