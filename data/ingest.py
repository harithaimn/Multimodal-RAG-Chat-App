# data/ingest.py
"""
PATTERN-FIRST Ingest Pipeline (Supermetrics → Pinecone)

Design principle:
- Store STRUCTURE, not prose
- Embed PATTERNS, not brand copy
- Preserve raw text ONLY as metadata

Flow:
Google Sheet (Supermetrics)
→ normalize rows
→ extract patterns (text + image)
→ embed pattern text
→ Pinecone upsert (pattern text + metadata)
"""

import os
import time
import logging
from typing import Dict, List, Any

import yaml
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
from tqdm import tqdm

from openai import OpenAI
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

# =================================================
# Environment & Config
# =================================================

load_dotenv()

CONFIG_PATH = "config/config.yaml"
GSHEET_CREDS_PATH = "config/gsheet_credentials.json"

with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GSHEET_SPREADSHEET_ID = os.getenv("GSHEET_SPREADSHEET_ID")
GSHEET_SHEET_NAME = os.getenv("GSHEET_SHEET_NAME", "DMRag-DC#3")

if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, GSHEET_SPREADSHEET_ID]):
    raise RuntimeError("Missing required environment variables")

BATCH_SIZE = CFG["workflow"].get("batch_size", 50)
EMBED_MODEL = CFG["embedding_model"]["model_name"]

CAPTION_ENABLED = CFG["workflow"].get("captioning_enabled", True)
CAPTION_MODEL = CFG["multimodal"].get("caption_model", "gpt-4o-mini")
CAPTION_TEMP = CFG["multimodal"].get("caption_temperature", 0.4)

# =================================================
# Logging
# =================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# =================================================
# Clients
# =================================================

openai_client = OpenAI(api_key=OPENAI_API_KEY)
embedding_client = OpenAIEmbeddings(model=EMBED_MODEL)

pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone_client.Index(PINECONE_INDEX_NAME)

# =================================================
# Google Sheets
# =================================================

def get_gsheet_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_file(GSHEET_CREDS_PATH, scopes=scopes)
    return gspread.authorize(creds)

def read_sheet_rows() -> List[Dict[str, Any]]:
    gc = get_gsheet_client()
    sh = gc.open_by_key(GSHEET_SPREADSHEET_ID)
    ws = sh.worksheet(GSHEET_SHEET_NAME)
    rows = ws.get_all_records()
    logging.info("Loaded %d rows from Google Sheet", len(rows))
    return rows

# =================================================
# Normalization
# =================================================

COL_MAP = {
    "Date": "date",
    "Campaign objective": "objective",
    "Ad ID": "ad_id",
    "Ad body": "ad_body",
    "Ad creative image URL": "image_url",
    "CTR (all)": "ctr",
    "Impressions": "impressions",
}

def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    ad = {v: row.get(k) for k, v in COL_MAP.items()}

    ad["ctr"] = float(ad.get("ctr") or 0)
    ad["impressions"] = int(ad.get("impressions") or 0)

    if not ad.get("ad_id"):
        ad["ad_id"] = f"fallback_{hash(ad.get('ad_body', ''))}"

    return ad

# =================================================
# Pattern Extraction
# =================================================

def bucket_ctr(ctr: float) -> str:
    if ctr >= 0.03:
        return "high"
    if ctr >= 0.015:
        return "medium"
    return "low"

def bucket_length(text: str) -> str:
    l = len(text or "")
    if l <= 80:
        return "short"
    if l <= 160:
        return "medium"
    return "long"

def has_emoji(text: str) -> bool:
    return any(ord(c) > 10000 for c in text or "")

# =================================================
# Image Pattern Tagging (Optional)
# =================================================

def tag_image_patterns(image_url: str, retries: int = 2) -> str:
    if not CAPTION_ENABLED or not image_url:
        return "no_image"

    try:
        response = openai_client.chat.completions.create(
            model=CAPTION_MODEL,
            temperature=CAPTION_TEMP,
            max_completion_tokens=40,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Describe the ad image using concise visual tags only. "
                        "Comma-separated. No sentences."
                    )
                },
                {
                    "role": "user",
                    "content": f"Image URL: {image_url}"
                }
            ]
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        logging.warning("Image tagging failed (%s)", e)
        if retries > 0:
            time.sleep(1)
            return tag_image_patterns(image_url, retries - 1)
        return "image_unknown"

# =================================================
# Pattern Text Builder (EMBED THIS)
# =================================================

def build_pattern_text(ad: Dict[str, Any], image_tags: str) -> str:
    return "\n".join([
        f"Objective: {ad.get('objective')}",
        f"CTR bucket: {bucket_ctr(ad['ctr'])}",
        f"Ad length: {bucket_length(ad.get('ad_body', ''))}",
        f"Emoji used: {has_emoji(ad.get('ad_body'))}",
        f"Image tags: {image_tags}",
    ])

# =================================================
# Ingest Runner
# =================================================

def run_ingest():
    rows = read_sheet_rows()
    ads = [normalize_row(r) for r in rows if r]

    documents = []

    for ad in tqdm(ads, desc="Extracting patterns"):
        image_tags = tag_image_patterns(ad.get("image_url"))
        pattern_text = build_pattern_text(ad, image_tags)

        metadata = {
            **ad,
            "raw_ad_body": ad.get("ad_body"),
            "image_tags": image_tags,
        }

        documents.append({
            "id": str(ad["ad_id"]),
            "text": pattern_text,
            "metadata": metadata,
        })

    logging.info("Embedding & upserting %d pattern docs", len(documents))

    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]

        texts = [d["text"] for d in batch]
        ids = [d["id"] for d in batch]
        metas = [d["metadata"] for d in batch]

        vectors = embedding_client.embed_documents(texts)
        index.upsert(zip(ids, vectors, metas))

        logging.info("Upserted batch %d–%d", i, i + len(batch))

    logging.info("✅ Pattern-first ingestion complete")

if __name__ == "__main__":
    run_ingest()