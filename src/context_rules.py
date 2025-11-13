# src/context_rules.py

DROPDOWN_CONTEXTS = {
    "industry": {
        "Food & Beverages": {
            "context": (
                "Only reference ads related to food, drinks, restaurants, or cafes. "
                "Retrieve past INVOKE F&B campaigns. Focus on taste appeal, visuals, "
                "sensory descriptions, and CTA for physical stores or delivery."
            ),
            "keywords": ["burger", "restaurant", "drink", "menu", "coffee", "stall", "food"],
        },
        "Beauty & Body Care": {
            "context": (
                "Focus on beauty, skincare, cosmetics, and wellness. Reference aesthetic visuals, "
                "emotional resonance, influencer tone, and self-care messaging."
            ),
            "keywords": ["skincare", "beauty", "spa", "cosmetic", "makeup", "glow"],
        },
        "Finance & Insurance": {
            "context": (
                "Focus on financial trust, ROI, and security. Reference ads emphasizing "
                "credibility, professionalism, and long-term benefits."
            ),
            "keywords": ["loan", "bank", "insurance", "investment", "finance"],
        },
    },
    "ads_language": {
        "English": {"context": "Generate ads in English, using professional yet conversational tone."},
        "Malay": {"context": "Generate ads in Malay (Bahasa Melayu), preserving marketing tone and flow."},
        "Mandarin": {"context": "Generate ads in Mandarin Chinese, suitable for Malaysian-Chinese audience."},
    }
}


def validate_dropdown(user_input: str, selected_values: dict):
    """
    Detects mismatches between user input and selected dropdowns (e.g. wrong industry).
    Returns a clarification message if mismatch detected.
    """
    input_lower = user_input.lower()

    # --- Industry validation ---
    detected_industry = None
    for industry, data in DROPDOWN_CONTEXTS["industry"].items():
        if any(keyword in input_lower for keyword in data.get("keywords", [])):
            detected_industry = industry
            break

    selected_industry = selected_values.get("industry")
    if detected_industry and selected_industry and detected_industry != selected_industry:
        return f"It looks like you’re describing something from **{detected_industry}**, not **{selected_industry}**. Did you mean to switch to **{detected_industry}**?"

    return None