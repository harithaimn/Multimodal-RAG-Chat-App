"""
context_rules.py

Purpose:
- Enforce behavioral rules on LLM output
- Prevent summarization, dataset worship, and leakage
- Keep generation aligned with product intent

This file is MODEL-AGNOSTIC.
It operates on OUTPUT, not PROMPTS.
"""

import re
from typing import Dict, List

# =====================================================
# Section Headers (STRICT)
# =====================================================

SECTION_AD_COPY = "[AD COPY]"
SECTION_WHY = "[WHY THIS WORKS"

REQUIRED_SECTIONS = [
    SECTION_AD_COPY,
    SECTION_WHY,
]

# =====================================================
# Forbidden Phrases
# =====================================================

FORBIDDEN_IN_AD_COPY = [
    "dataset",
    "data shows",
    "based on",
    "according to",
    "retrieved",
    "historical",
    "campaign",
    "example",
    "insight",
]

FORBIDDEN_GLOBAL = [
    "as an ai",
    "i cannot",
    "i'm unable",
    "i don't know",
]

# =====================================================
# Heuristic Limits
# =====================================================

MAX_AD_LINES = 6
MAX_AD_CHARS = 300
MIN_AD_CHARS = 20

MAX_WHY_LINES = 6

# =====================================================
# Core Validators
# =====================================================

def validate_output_structure(text: str) -> None:
    """
    Ensure required sections exist.
    """
    for section in REQUIRED_SECTIONS:
        if section.lower() not in text.lower():
            raise ValueError(f"Missing required section: {section}")


def split_sections(text: str) -> Dict[str, str]:
    """
    Extract ad copy and explanation sections.
    """
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
        raise ValueError("Unable to split output into required sections")

    return {
        "ad_copy": ad_match.group(1).strip(),
        "why": why_match.group(1).strip(),
    }


def validate_ad_copy(ad_text: str) -> None:
    """
    Enforce ad-copy-specific constraints.
    """
    lowered = ad_text.lower()

    for phrase in FORBIDDEN_IN_AD_COPY:
        if phrase in lowered:
            raise ValueError(f"Forbidden phrase in ad copy: '{phrase}'")

    if any(p in lowered for p in FORBIDDEN_GLOBAL):
        raise ValueError("System / refusal language detected in ad copy")

    lines = ad_text.splitlines()

    if len(lines) > MAX_AD_LINES:
        raise ValueError("Ad copy too long (line limit exceeded)")

    if len(ad_text) > MAX_AD_CHARS:
        raise ValueError("Ad copy too long (char limit exceeded)")

    if len(ad_text) < MIN_AD_CHARS:
        raise ValueError("Ad copy too short / low signal")


def validate_why_section(why_text: str) -> None:
    """
    Validate pattern explanation section.
    """
    lines = [l for l in why_text.splitlines() if l.strip()]

    if len(lines) > MAX_WHY_LINES:
        raise ValueError("Too many explanation bullets")

    # Encourage pattern language
    allowed_keywords = [
        "hook",
        "length",
        "emoji",
        "structure",
        "tone",
        "offer",
        "visual",
        "cta",
        "timing",
    ]

    if not any(k in why_text.lower() for k in allowed_keywords):
        raise ValueError(
            "Pattern explanation too vague (no pattern keywords found)"
        )


# =====================================================
# Public Entry Point
# =====================================================

def enforce_context_rules(output_text: str) -> Dict[str, str]:
    """
    Master enforcement function.

    Returns structured output if valid:
    {
        "ad_copy": "...",
        "why": "..."
    }
    """
    validate_output_structure(output_text)

    sections = split_sections(output_text)

    validate_ad_copy(sections["ad_copy"])
    validate_why_section(sections["why"])

    return sections