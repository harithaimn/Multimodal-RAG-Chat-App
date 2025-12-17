# src/utils.py

"""
Utility helpers for Multimodal RAG POC.

Scope:
- Optional chat persistence (local / S3)
- Lightweight NLP helpers
- No LangChain message objects
- No inferred attributes
"""

import os
import json
import uuid
import boto3
import re
from typing import Dict, List, Optional
from datetime import datetime

# -------------------------------------------------
# Configuration
# -------------------------------------------------

S3_ENABLED = os.getenv("S3_ENABLED", "false").lower() == "true"
S3_BUCKET = os.getenv("AWS_S3_BUCKET_NAME")
LOCAL_SAVE_DIR = "saved_chats"

os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)

# -------------------------------------------------
# Chat Persistence (Local-first, S3 optional)
# -------------------------------------------------

def save_chat_history(
    session_id: str,
    chat_history: List[Dict],
    filters: Optional[Dict] = None,
    title: Optional[str] = None,
) -> Dict:
    """
    Save chat history to local disk.
    Optionally sync to S3 if enabled.
    """
    payload = {
        "session_id": session_id,
        "title": title or f"Session-{session_id[:8]}",
        "filters": filters or {},
        "chat_history": chat_history,
        "timestamp": datetime.utcnow().isoformat(),
    }

    filename = f"{session_id}.json"
    local_path = os.path.join(LOCAL_SAVE_DIR, filename)

    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if S3_ENABLED and S3_BUCKET:
        try:
            s3 = boto3.client("s3")
            s3.upload_file(local_path, S3_BUCKET, f"chats/{filename}")
        except Exception as e:
            print(f"⚠️ S3 upload failed: {e}")

    return payload


def load_chat_history(session_id: str) -> Optional[Dict]:
    """
    Load chat history from local disk.
    Falls back to S3 if enabled.
    """
    filename = f"{session_id}.json"
    local_path = os.path.join(LOCAL_SAVE_DIR, filename)

    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if S3_ENABLED and S3_BUCKET:
        try:
            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=S3_BUCKET, Key=f"chats/{filename}")
            return json.loads(obj["Body"].read().decode("utf-8"))
        except Exception:
            return None

    return None


def list_saved_sessions() -> List[Dict]:
    """
    List locally saved chat sessions.
    """
    sessions = []

    for fname in os.listdir(LOCAL_SAVE_DIR):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(LOCAL_SAVE_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                sessions.append({
                    "session_id": data.get("session_id"),
                    "title": data.get("title"),
                    "timestamp": data.get("timestamp"),
                })
        except Exception:
            continue

    sessions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return sessions


# -------------------------------------------------
# NLP Helpers (Optional)
# -------------------------------------------------

def extract_keywords(text: str, top_k: int = 5) -> List[str]:
    """
    Very lightweight keyword extractor.
    """
    if not text:
        return []

    text = re.sub(r"[^\w\s]", "", text.lower())
    words = text.split()
    stopwords = {
        "the", "and", "for", "with", "that", "this",
        "from", "you", "your", "our", "are", "was"
    }

    keywords = [
        w for w in words
        if w not in stopwords and len(w) > 2
    ]

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            deduped.append(w)

    return deduped[:top_k]


# -------------------------------------------------
# Session Helpers
# -------------------------------------------------

def create_new_session(
    filters: Optional[Dict] = None,
    title: Optional[str] = None
) -> str:
    """
    Create a new session ID and persist empty chat.
    """
    session_id = str(uuid.uuid4())

    save_chat_history(
        session_id=session_id,
        chat_history=[],
        filters=filters or {},
        title=title,
    )

    return session_id
