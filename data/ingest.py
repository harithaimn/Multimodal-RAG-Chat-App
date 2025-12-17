# data/ingest.py
"""
POC Ingest Pipeline (Supermetrics → Pinecone)

Flow:
Google Sheet (Supermetrics)
→ normalize rows
→ optional image captioning
→ embeddings
→ Pinecone upsert (text + metadata + image_url)

No local files.
No S3 image handling.
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

# -------------------------------------------------
# Environment & Config
# -------------------------------------------------

load_dotenv()

CONFIG_PATH = "config/config.yaml"
GSHEET_CREDS_PATH = "config/gsheet_credentials.json"

with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

# Required env vars
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GSHEET_SPREADSHEET_ID = os.getenv("GSHEET_SPREADSHEET_ID")
GSHEET_SHEET_NAME = os.getenv("GSHEET_SHEET_NAME", "DMRag-DC#3")

if not all([
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    GSHEET_SPREADSHEET_ID,
]):
    raise RuntimeError("Missing required environment variables")

# Pipeline config
BATCH_SIZE = CFG["workflow"].get("batch_size", 50)
CAPTION_ENABLED = CFG["workflow"].get("captioning_enabled", True)
CAPTION_MODEL = CFG["multimodal"].get("caption_model", "gpt-4o-mini")
CAPTION_TEMP = CFG["multimodal"].get("caption_temperature", 0.6)
EMBED_MODEL = CFG["embedding_model"]["model_name"]

# -------------------------------------------------
# Logging
# -------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# -------------------------------------------------
# Clients
# -------------------------------------------------

openai_client = OpenAI(api_key=OPENAI_API_KEY)
embedding_client = OpenAIEmbeddings(model=EMBED_MODEL)

pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = pinecone_client.list_indexes().names()
if PINECONE_INDEX_NAME not in existing_indexes:
    raise RuntimeError(
        f"Pinecone index '{PINECONE_INDEX_NAME}' not found. "
        "Create it before running ingestion."
    )

index = pinecone_client.Index(PINECONE_INDEX_NAME)

# -------------------------------------------------
# Google Sheets
# -------------------------------------------------

def get_gsheet_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_file(
        GSHEET_CREDS_PATH,
        scopes=scopes
    )
    return gspread.authorize(creds)

def read_sheet_rows() -> List[Dict[str, Any]]:
    gc = get_gsheet_client()
    sh = gc.open_by_key(GSHEET_SPREADSHEET_ID)
    ws = sh.worksheet(GSHEET_SHEET_NAME)
    rows = ws.get_all_records()
    logging.info("Loaded %d rows from Google Sheet", len(rows))
    return rows

# -------------------------------------------------
# Normalization
# -------------------------------------------------

COL_MAP = {
    "Date": "date",
    "Campaign ID": "campaign_id",
    "Campaign name": "campaign_name",
    "Campaign objective": "campaign_objective",
    "Ad ID": "ad_id",
    "Ad name": "ad_name",
    "Creative title": "creative_title",
    "Ad body": "ad_body",
    "Ad creative image URL": "image_url",
    "Impressions": "impressions",
    "Reach": "reach",
    "CTR (all)": "ctr",
    "CPC (all)": "cpc",
    "CPM (cost per 1000 impressions)": "cpm",
    "Cost": "cost",
}

INT_FIELDS = {"impressions", "reach"}
FLOAT_FIELDS = {"ctr", "cpc", "cpm", "cost"}

def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    ad = {v: row.get(k) for k, v in COL_MAP.items()}

    for k in INT_FIELDS:
        try:
            ad[k] = int(ad.get(k) or 0)
        except Exception:
            ad[k] = 0

    for k in FLOAT_FIELDS:
        try:
            ad[k] = float(ad.get(k) or 0)
        except Exception:
            ad[k] = 0.0

    if not ad.get("ad_id"):
        ad["ad_id"] = f"fallback_{hash(ad.get('ad_name', ''))}"

    return ad

# -------------------------------------------------
# Caption Generation
# -------------------------------------------------

def generate_caption(image_url: str, retries: int = 2) -> str:
    if not CAPTION_ENABLED or not image_url:
        return ""

    try:
        response = openai_client.chat.completions.create(
            model=CAPTION_MODEL,
            temperature=CAPTION_TEMP,
            max_completion_tokens=60,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a performance marketing visual analyst. "
                        "Summarize the marketing intent, emotional tone, "
                        "and creative strategy of the ad image in ONE sentence."
                    )
                },
                {
                    "role": "user",
                    "content": f"Analyze this image: {image_url}"
                }
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.warning("Caption failed (%s)", e)
        if retries > 0:
            time.sleep(1)
            return generate_caption(image_url, retries - 1)
        return ""

# -------------------------------------------------
# Document Builder
# -------------------------------------------------

def build_text(ad: Dict[str, Any]) -> str:
    return "\n".join([
        f"Campaign: {ad.get('campaign_name')}",
        f"Objective: {ad.get('campaign_objective')}",
        f"Ad Name: {ad.get('ad_name')}",
        f"Creative Title: {ad.get('creative_title')}",
        f"Ad Body: {ad.get('ad_body')}",
        f"Metrics: impressions={ad.get('impressions')}, ctr={ad.get('ctr')}"
    ])

# -------------------------------------------------
# Ingest Runner
# -------------------------------------------------

def run_ingest():
    rows = read_sheet_rows()
    ads = [normalize_row(r) for r in rows if r]

    documents = []

    for ad in tqdm(ads, desc="Preparing ads"):
        caption = generate_caption(ad["image_url"])
        text = build_text(ad)

        metadata = {
            **ad,
            "caption": caption
        }

        documents.append({
            "id": str(ad["ad_id"]),
            "text": text,
            "metadata": metadata
        })

    logging.info("Embedding & upserting %d ads", len(documents))

    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        texts = [d["text"] for d in batch]
        ids = [d["id"] for d in batch]
        metas = [d["metadata"] for d in batch]

        vectors = embedding_client.embed_documents(texts)
        index.upsert(zip(ids, vectors, metas))

        logging.info("Upserted batch %d–%d", i, i + len(batch))

    logging.info("✅ Ingestion complete")

if __name__ == "__main__":
    run_ingest()