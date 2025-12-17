# src/context_rules.py
"""
Lightweight context & filter utilities for Multimodal RAG (POC).

Design principles:
- Dataset-first (no hallucinated attributes)
- Optional usage (safe to bypass)
- Pinecone-compatible filters
- Future extensibility (industry, psychographics later)
"""

from typing import Dict, Any, List, Optional


# -------------------------------------------------
# 1. Allowed filter keys (must exist in Pinecone metadata)
# -------------------------------------------------

ALLOWED_FILTER_KEYS = {
    "campaign_id",
    "campaign_name",
    "campaign_objective",
    "ad_id",
    "ad_name",
    "date",
}


# -------------------------------------------------
# 2. Normalize filters from Streamlit / UI
# -------------------------------------------------

def normalize_filters(raw_filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Cleans UI-provided filters:
    - Drops empty / 'All' values
    - Keeps only allowed metadata keys
    """
    if not raw_filters:
        return {}

    normalized = {}
    for k, v in raw_filters.items():
        if not v:
            continue
        if isinstance(v, str) and v.lower() == "all":
            continue
        if k not in ALLOWED_FILTER_KEYS:
            continue
        normalized[k] = v

    return normalized


# -------------------------------------------------
# 3. Build Pinecone metadata filter
# -------------------------------------------------

def build_pinecone_filter(filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Converts normalized filters into Pinecone-compatible filter syntax.
    """
    normalized = normalize_filters(filters)

    pinecone_filter = {}
    for k, v in normalized.items():
        pinecone_filter[k] = {"$eq": v}

    return pinecone_filter


# -------------------------------------------------
# 4. Optional: lightweight sanity checks
# -------------------------------------------------

def validate_user_input_vs_filters(
    user_input: str,
    filters: Optional[Dict[str, Any]]
) -> List[str]:
    """
    Soft warnings only.
    Never blocks execution.
    """
    warnings = []

    if not user_input or not filters:
        return warnings

    text = user_input.lower()

    # Example: objective mismatch hint
    objective = filters.get("campaign_objective")
    if objective:
        if objective.lower() not in text:
            warnings.append(
                f"User input does not explicitly reference campaign objective '{objective}'."
            )

    return warnings


# -------------------------------------------------
# 5. Derived context (dataset-only, safe)
# -------------------------------------------------

def derive_context_from_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates secondary signals strictly from dataset fields.
    Safe for POC.
    """
    context = {}

    ctr = metadata.get("ctr")
    if isinstance(ctr, (int, float)):
        if ctr >= 2.0:
            context["performance_bucket"] = "high_ctr"
        elif ctr >= 1.0:
            context["performance_bucket"] = "medium_ctr"
        else:
            context["performance_bucket"] = "low_ctr"

    return context
