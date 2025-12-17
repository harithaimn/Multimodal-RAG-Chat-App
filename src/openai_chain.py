"""
openai_chain.py

Generation-first, explanation-second chain.

Output format (STRICT):

[AD COPY]
<new ad>

[WHY THIS WORKS (PATTERN REFERENCE)]
- <pattern explanation>
- <pattern explanation>

RAG:
- Used for pattern reference ONLY
- Never copied verbatim
"""

from typing import List, Dict, Any
import re

from openai import OpenAI

# ============================================================
# Config
# ============================================================

MODEL_NAME = "gpt-4.1-mini"
TEMPERATURE = 0.9
MAX_TOKENS = 220

FORBIDDEN_IN_AD = [
    "dataset",
    "data shows",
    "retrieved",
    "historical ads",
    "campaign",
    "based on",
    "according to",
]

REQUIRED_SECTIONS = [
    "[AD COPY]",
    "[WHY THIS WORKS",
]

# ============================================================
# Helpers
# ============================================================

def _split_sections(text: str) -> Dict[str, str]:
    """
    Split model output into ad + explanation sections.
    """
    sections = {}

    ad_match = re.search(
        r"\[AD COPY\](.*?)(\[WHY THIS WORKS.*?\])",
        text,
        re.S | re.I,
    )

    why_match = re.search(
        r"\[WHY THIS WORKS.*?\](.*)",
        text,
        re.S | re.I,
    )

    if not ad_match or not why_match:
        raise ValueError("Output format invalid. Required sections missing.")

    sections["ad"] = ad_match.group(1).strip()
    sections["why"] = why_match.group(1).strip()

    return sections


def _validate_ad_section(ad_text: str) -> None:
    """
    Enforce ZERO dataset leakage in ad copy.
    """
    lowered = ad_text.lower()
    for phrase in FORBIDDEN_IN_AD:
        if phrase in lowered:
            raise ValueError(f"Forbidden phrase in ad copy: '{phrase}'")

    if len(ad_text.splitlines()) > 6:
        raise ValueError("Ad copy too long. Must be short-form.")


def _build_system_prompt() -> str:
    return (
        "You are a performance copywriter and growth analyst.\n\n"
        "You must follow this order strictly:\n"
        "1) Generate a NEW, deployable ad copy\n"
        "2) Explain which historical ad patterns influenced it\n\n"
        "Rules:\n"
        "- Ad copy MUST NOT reference datasets, campaigns, or examples\n"
        "- Explanation MAY reference historical patterns abstractly\n"
        "- Never copy historical ads verbatim\n"
        "- No hedging or uncertainty language\n\n"
        "Output format MUST be exactly:\n\n"
        "[AD COPY]\n"
        "<final ad text>\n\n"
        "[WHY THIS WORKS (PATTERN REFERENCE)]\n"
        "- <pattern explanation>\n"
        "- <pattern explanation>"
    )


def _build_user_prompt(
    *,
    business_type: str,
    product: str,
    platform: str,
    language_style: str,
    rag_context: str,
) -> str:
    return (
        f"Business type: {business_type}\n"
        f"Product / Offer: {product}\n"
        f"Platform: {platform}\n"
        f"Language style: {language_style}\n\n"
        "Retrieved historical ads (for pattern reference ONLY):\n"
        f"{rag_context}\n\n"
        "Generate the output now."
    )


# ============================================================
# Public API
# ============================================================

def generate_ad_with_patterns(
    *,
    client: OpenAI,
    rag_docs: List[Dict[str, Any]],
    business_type: str,
    product: str,
    platform: str = "Meta Ads",
    language_style: str = "Casual Malaysian English",
) -> Dict[str, str]:
    """
    Main generation entry point.

    Returns:
    {
        "ad_copy": "...",
        "pattern_explanation": "..."
    }
    """

    # --- Prepare RAG context (pattern signal only) ---
    rag_chunks = []
    for d in rag_docs[:5]:
        text = d.get("text", "")
        if text:
            rag_chunks.append(text[:200])

    rag_context = "\n---\n".join(rag_chunks)

    # --- Build prompts ---
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(
        business_type=business_type,
        product=product,
        platform=platform,
        language_style=language_style,
        rag_context=rag_context,
    )

    # --- Call OpenAI ---
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    raw_output = response.choices[0].message.content.strip()

    # --- Validate structure ---
    sections = _split_sections(raw_output)

    _validate_ad_section(sections["ad"])

    return {
        "ad_copy": sections["ad"],
        "pattern_explanation": sections["why"],
    }