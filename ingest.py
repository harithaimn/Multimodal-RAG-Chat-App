import os
import json
import base64
import yaml
import time
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, ValidationError
from typing import Optional

# ======================================================
# 1. LOAD ENV + CONFIG
# ======================================================
load_dotenv()
CONFIG_PATH = "config/config.yaml"

def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Missing config file at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

config = load_config()

INDEX_NAME = config["pinecone"]["index_name"]
EMBEDDING_MODEL = config["embedding_model"]["model_name"]
CAPTION_MODEL = "gpt-4o-mini"  # force gpt-4o for multimodal captioning
CAPTION_TEMP = 0.6
CAPTION_ENABLED = True
BATCH_SIZE = 50

local_data_path = "data/dataset.json"

# ======================================================
# 2. CLIENT INITIALIZATION
# ======================================================
print("Initializing OpenAI and Pinecone clients...")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# ======================================================
# 3. LOAD DATA
# ======================================================
print("\nStep 1: Loading dataset...")

def load_dataset():
    if not os.path.exists(local_data_path):
        raise FileNotFoundError("No dataset found at data/dataset.json")
    with open(local_data_path, "r") as f:
        data = json.load(f)
    print(f"✅ Loaded dataset locally ({len(data)} records)")
    return data

dataset = load_dataset()

# ======================================================
# 4. VALIDATE PINECONE INDEX
# ======================================================
print("\nStep 2: Verifying Pinecone index...")

if INDEX_NAME not in pinecone_client.list_indexes().names():
    print(f"⚠️ Index '{INDEX_NAME}' not found. Creating new index...")
    pinecone_client.create_index(
        name=INDEX_NAME,
        dimension=1536,  # for text-embedding-3-small
        metric="cosine"
    )
else:
    print(f"✅ Pinecone index '{INDEX_NAME}' found.")

index = pinecone_client.Index(INDEX_NAME)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# ======================================================
# 5. PYDANTIC MODEL SCHEMA
# ======================================================
class AdCreativeRecord(BaseModel):
    ad_id: str
    ad_name: Optional[str] = ""
    campaign_name: Optional[str] = ""
    objective: Optional[str] = ""
    spend: float = 0.0
    impressions: int = 0
    clicks: int = 0
    ctr: float = 0.0
    cpc: float = 0.0
    cpm: float = 0.0
    roas: Optional[float] = 0.0
    image_url: Optional[str] = ""
    video_url: Optional[str] = ""
    caption: Optional[str] = ""

# ======================================================
# 6. IMAGE CAPTIONING (LOCAL + URL SUPPORT)
# ======================================================

def build_image_content(image_url_or_path: str):
    """Builds OpenAI-compatible image content for remote or local files."""
    if not image_url_or_path:
        raise ValueError("Missing image reference")
    if image_url_or_path.startswith("http"):
        return {"type": "image_url", "image_url": {"url": image_url_or_path}}
    elif os.path.exists(image_url_or_path):
        with open(image_url_or_path, "rb") as f:
            base64_img = base64.b64encode(f.read()).decode("utf-8")
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
    else:
        raise FileNotFoundError(f"Invalid image path or URL: {image_url_or_path}")

def generate_caption(image_path_or_url: str, retries=2) -> str:
    """Generate a marketing psychology caption for an ad image."""
    if not CAPTION_ENABLED or not image_path_or_url:
        return ""
    try:
        image_input = build_image_content(image_path_or_url)
        response = openai_client.chat.completions.create(
            model=CAPTION_MODEL,
            temperature=CAPTION_TEMP,
            max_completion_tokens=70,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior performance marketing strategist and visual analyst. "
                        "You specialize in understanding how visual ad creatives influence emotion, trust, and conversion. "
                        "When given an image, describe it in one high-level sentence that captures its marketing psychology — "
                        "including emotion, audience targeting, brand tone, and visual strategy. "
                        "Be concise but insightful, like how a strategist summarizes an ad's creative intent for a marketing report."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this ad image.\n\n"
                                "Return ONE sentence that summarizes its marketing intent, emotional tone, and creative theme. "
                                "Do NOT describe the literal content (e.g. 'a man smiling'), but the underlying message (e.g. "
                                "'evokes trust and simplicity through minimalist design and confident expression')."
                            )
                        },
                        image_input
                    ]
                }
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        if retries > 0:
            print(f"⚠️ Caption failed, retrying ({retries}) for {image_path_or_url}: {e}")
            time.sleep(2)
            return generate_caption(image_path_or_url, retries - 1)
        print(f"❌ Failed to caption {image_path_or_url}: {e}")
        return ""

# ======================================================
# 7. INSIGHTS EXTRACTION
# ======================================================
def extract_insights(ad):
    """Normalize insight fields from raw ad data."""
    default = {"spend": 0, "impressions": 0, "clicks": 0, "ctr": 0, "cpc": 0, "cpm": 0, "purchase_roas": []}
    try:
        data = ad.get("insights", {}).get("data", [])
        if not data:
            return default
        entry = data[0]
        return {
            "spend": float(entry.get("spend", 0) or 0),
            "impressions": int(entry.get("impressions", 0) or 0),
            "clicks": int(entry.get("clicks", 0) or 0),
            "ctr": float(entry.get("ctr", 0) or 0),
            "cpc": float(entry.get("cpc", 0) or 0),
            "cpm": float(entry.get("cpm", 0) or 0),
            "purchase_roas": entry.get("purchase_roas", []),
        }
    except Exception as e:
        print(f"⚠️ Error parsing insights: {e}")
        return default

# ======================================================
# 8. DOCUMENT BUILDING
# ======================================================
print("\nStep 3: Preparing documents for embedding...")

def safe_str(value):
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)

documents = []

for ad in tqdm(dataset, desc="Processing Ads", ncols=100):
    insights = extract_insights(ad)
    campaign = ad.get("campaign", {})
    adset = ad.get("adset", {})
    creative = ad.get("creative", {})
    targeting = adset.get("targeting", {})

    img_path = creative.get("image_path") or creative.get("image_url")
    image_caption = generate_caption(img_path) if CAPTION_ENABLED else ""

    ad_text = (
        f"Ad ID: {ad.get('id', '')}\n"
        f"Ad Name: {ad.get('name', '')}\n"
        f"Status: {ad.get('status', '')}\n\n"
        f"Campaign Name: {campaign.get('name', '')}\n"
        f"Objective: {campaign.get('objective', '')}\n"
        f"Ad Set Targeting: {safe_str(targeting)}\n\n"
        f"Creative Body: {creative.get('body', '')}\n"
        f"Image: {creative.get('image_url', '')}\n"
        f"Video: {creative.get('video_url', '')}\n"
        f"Spend: {insights['spend']}, Impressions: {insights['impressions']}, "
        f"Clicks: {insights['clicks']}, CTR: {insights['ctr']}, CPC: {insights['cpc']}, CPM: {insights['cpm']}\n\n"
        f"[Marketing Psychology Summary]: {image_caption}"
    )

    try:
        meta = AdCreativeRecord(
            ad_id=ad.get("id", ""),
            ad_name=ad.get("name", ""),
            campaign_name=campaign.get("name", ""),
            objective=campaign.get("objective", ""),
            spend=insights["spend"],
            impressions=insights["impressions"],
            clicks=insights["clicks"],
            ctr=insights["ctr"],
            cpc=insights["cpc"],
            cpm=insights["cpm"],
            roas=float(insights["purchase_roas"][0].get("value")) if insights.get("purchase_roas") else 0.0,
            image_url=creative.get("image_url", ""),
            video_url=creative.get("video_url", ""),
            caption=image_caption,
        )
        documents.append({"id": meta.ad_id, "text": ad_text, "metadata": meta.model_dump()})
    except ValidationError as e:
        print(f"❌ Skipped invalid record {ad.get('id')}: {e}")

print(f"✅ Prepared {len(documents)} validated documents.")

# ======================================================
# 9. EMBEDDING + UPSERT
# ======================================================
print("\nStep 4: Generating embeddings and upserting to Pinecone...")

for i in range(0, len(documents), BATCH_SIZE):
    batch = documents[i:i + BATCH_SIZE]
    ids = [doc["id"] for doc in batch]
    texts = [doc["text"] for doc in batch]
    metadatas = [doc["metadata"] for doc in batch]

    try:
        vectors = embeddings.embed_documents(texts)
        index.upsert(vectors=zip(ids, vectors, metadatas))
        print(f"✅ Upserted batch {i // BATCH_SIZE + 1} ({len(batch)} items)")
    except Exception as e:
        print(f"⚠️ Batch {i // BATCH_SIZE + 1} failed: {e}")

print("\n✅ Ingestion complete.")
print(f"💾 Pinecone index '{INDEX_NAME}' now contains {len(documents)} vectors.")
