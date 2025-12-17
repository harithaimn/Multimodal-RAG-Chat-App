"""
vectorstore.py

Purpose:
- Store historical ads for PATTERN retrieval
- Retrieve short, pattern-relevant snippets
- Avoid prose worship and brand leakage
"""

from typing import List, Dict, Any
import os

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

# ============================================================
# Config
# ============================================================

EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 5
MAX_CHARS_PER_DOC = 300  # hard cap to avoid prose copying

# ============================================================
# Init
# ============================================================

def init_vectorstore() -> Pinecone:
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not api_key or not index_name:
        raise RuntimeError("PINECONE_API_KEY or PINECONE_INDEX_NAME missing")

    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)


def init_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)

# ============================================================
# Normalization (VERY IMPORTANT)
# ============================================================

def normalize_ad_text(text: str) -> str:
    """
    Normalize ad text so embeddings capture STRUCTURE, not branding.
    """
    if not text:
        return ""

    text = text.strip()

    # Remove URLs
    text = text.replace("http://", "").replace("https://", "")

    # Remove hashtags
    text = " ".join(w for w in text.split() if not w.startswith("#"))

    # Hard truncate
    if len(text) > MAX_CHARS_PER_DOC:
        text = text[:MAX_CHARS_PER_DOC]

    return text


# ============================================================
# Upsert
# ============================================================

def upsert_ads(
    *,
    index,
    embeddings: OpenAIEmbeddings,
    ads: List[Dict[str, Any]],
) -> None:
    """
    Expected ad schema:
    {
        "id": str,
        "text": str,
        "platform": str,
        "objective": str,
        "language": str,
    }
    """

    vectors = []

    for ad in ads:
        raw_text = ad.get("text", "")
        clean_text = normalize_ad_text(raw_text)

        if not clean_text:
            continue

        vector = embeddings.embed_query(clean_text)

        metadata = {
            "platform": ad.get("platform", "unknown"),
            "objective": ad.get("objective", "unknown"),
            "language": ad.get("language", "unknown"),
            "length": len(clean_text),
            "has_emoji": any(ord(c) > 10000 for c in clean_text),
        }

        vectors.append(
            {
                "id": ad["id"],
                "values": vector,
                "metadata": metadata,
            }
        )

    if vectors:
        index.upsert(vectors=vectors)


# ============================================================
# Retrieval
# ============================================================

def retrieve_pattern_docs(
    *,
    index,
    embeddings: OpenAIEmbeddings,
    query: str,
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """
    Retrieve ads as PATTERN SIGNALS, not full examples.
    """

    query_embedding = embeddings.embed_query(query)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    docs = []

    for match in results.get("matches", []):
        meta = match.get("metadata", {})

        # Construct a pattern-oriented text summary
        pattern_text = (
            f"Platform: {meta.get('platform')} | "
            f"Objective: {meta.get('objective')} | "
            f"Length: {meta.get('length')} | "
            f"Emoji: {meta.get('has_emoji')}"
        )

        docs.append(
            {
                "text": pattern_text,
                "metadata": meta,
                "score": match.get("score"),
            }
        )

    return docs