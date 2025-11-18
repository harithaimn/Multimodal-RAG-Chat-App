import os
import json
import base64
import yaml
import time
import boto3
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
CAPTION_MODEL = "gpt-4o-mini"
CAPTION_TEMP = 0.6
CAPTION_ENABLED = True
BATCH_SIZE = 50

local_data_path = "data/dataset.json"

# ======================================================
# 2. CLIENT INITIALIZATION
# ======================================================
print("Initializing OpenAI, Pinecone, and S3 clients...")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# AWS S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    region_name=os.getenv("AWS_REGION")
)
S3_BUCKET = os.getenv("AWS_S3_BUCKET")


# ======================================================
# 3. LOAD DATASET
# ======================================================
def load_dataset():
    if not os.path.exists(local_data_path):
        raise FileNotFoundError("No dataset found at data/dataset.json")
    with open(local_data_path, "r") as f:
        data = json.load(f)
    print(f"✅ Loaded dataset locally ({len(data)} records)")
    return data

dataset = load_dataset()


# ======================================================
# 4. VERIFY PINECONE INDEX
# ======================================================
print("\nStep 2: Verifying Pinecone index...")

if INDEX_NAME not in pinecone_client.list_indexes().names():
    print(f"⚠️ Index '{INDEX_NAME}' not found. Creating new index...")
    pinecone_client.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine"
    )
else:
    print(f"✅ Pinecone index '{INDEX_NAME}' found.")

index = pinecone_client.Index(INDEX_NAME)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)


# ======================================================
# 5. PYDANTIC MODEL
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
# 6. S3 UPLOAD HANDLER
# ======================================================
def upload_to_s3(local_path: str, ad_id: str) -> Optional[str]:
    """Uploads local image to S3 and returns public URL."""
    if not local_path or not os.path.exists(local_path):
        return None

    file_ext = os.path.splitext(local_path)[1]
    key = f"ad_creatives/{ad_id}{file_ext}"

    try:
        s3.upload_file(
            local_path,
            S3_BUCKET,
            key,
            ExtraArgs={"ACL": "public-read", "ContentType": "image/jpeg"},
        )
        url = f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"
        return url
    except Exception as e:
        print(f"❌ S3 upload failed for {local_path}: {e}")
        return None


# ======================================================
# 7. MULTIMODAL CAPTIONING
# ======================================================
def build_image_content(url):
    return {"type": "image_url", "image_url": {"url": url}}

def generate_caption(image_url: str, retries=2):
    if not image_url or not CAPTION_ENABLED:
        return ""

    try:
        response = openai_client.chat.completions.create(
            model=CAPTION_MODEL,
            temperature=CAPTION_TEMP,
            max_completion_tokens=60,
            messages=[
                {"role": "system",
                 "content": (
                    "You are a senior performance marketing strategist and visual analyst. "
                    "You specialize in understanding how visual ad creatives influence emotion, trust, and conversion. "
                    "When given an image, describe it in one high-level sentence that captures its marketing psychology — "
                    "including emotion, audience targeting, brand tone, and visual strategy. "
                    "Be concise but insightful, like how a strategist summarizes an ad's creative intent for a marketing report."
                 )},
                {"role": "user",
                 "content": [
                     {"type": "text",
                      "text": (
                                "Analyze this ad image.\n\n"
                                "Return ONE sentence that summarizes its marketing intent, emotional tone, and creative theme. "
                                "Do NOT describe the literal content (e.g. 'a man smiling'), but the underlying message (e.g. "
                                "'evokes trust and simplicity through minimalist design and confident expression')."
                            )},
                     build_image_content(image_url)
                 ]}
            ]
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        if retries > 0:
            print(f"⚠️ Caption retry ({retries}) → {image_url}")
            return generate_caption(image_url, retries - 1)
        print(f"❌ Caption failed: {e}")
        return ""


# ======================================================
# 8. INSIGHT EXTRACTION
# ======================================================
def extract_insights(ad):
    default = {"spend": 0, "impressions": 0, "clicks": 0, "ctr": 0,
               "cpc": 0, "cpm": 0, "purchase_roas": []}
    try:
        data = ad.get("insights", {}).get("data", [])
        if not data:
            return default
        entry = data[0]
        return {
            "spend": float(entry.get("spend", 0)),
            "impressions": int(entry.get("impressions", 0)),
            "clicks": int(entry.get("clicks", 0)),
            "ctr": float(entry.get("ctr", 0)),
            "cpc": float(entry.get("cpc", 0)),
            "cpm": float(entry.get("cpm", 0)),
            "purchase_roas": entry.get("purchase_roas", []),
        }
    except:
        return default


# ======================================================
# 9. BUILD DOCUMENTS
# ======================================================
print("\nStep 3: Preparing documents...")

documents = []

for ad in tqdm(dataset, desc="Processing Ads", ncols=100):
    insights = extract_insights(ad)

    campaign = ad.get("campaign", {})
    adset = ad.get("adset", {})
    creative = ad.get("creative", {})
    targeting = adset.get("targeting", {})

    local_img_path = creative.get("image_path") or ""
    url_from_json = creative.get("image_url") or None

    # 1️⃣ Upload to S3 only if needed
    if url_from_json and url_from_json.startswith("http"):
        final_image_url = url_from_json
    else:
        final_image_url = upload_to_s3(local_img_path, ad.get("id"))

    # 2️⃣ Generate caption
    caption = generate_caption(final_image_url) if final_image_url else ""

    # 3️⃣ Build ad text for embedding
    ad_text = (
        f"Ad ID: {ad.get('id')}\n"
        f"Campaign: {campaign.get('name')}\n"
        f"Objective: {campaign.get('objective')}\n"
        f"Targeting: {json.dumps(targeting, ensure_ascii=False)}\n"
        f"Creative Body: {creative.get('body')}\n"
        f"Metrics: Spend={insights['spend']}, Impressions={insights['impressions']}, "
        f"Clicks={insights['clicks']}, CTR={insights['ctr']}\n\n"
        f"[Marketing Psychology]: {caption}"
    )

    # 4️⃣ Validate + store metadata
    try:
        meta = AdCreativeRecord(
            ad_id=ad.get("id"),
            ad_name=ad.get("name"),
            campaign_name=campaign.get("name"),
            objective=campaign.get("objective"),
            spend=insights["spend"],
            impressions=insights["impressions"],
            clicks=insights["clicks"],
            ctr=insights["ctr"],
            cpc=insights["cpc"],
            cpm=insights["cpm"],
            roas=float(insights["purchase_roas"][0]["value"])
                if insights.get("purchase_roas") else 0,
            image_url=final_image_url,
            video_url=creative.get("video_url"),
            caption=caption,
        )
        documents.append({
            "id": meta.ad_id,
            "text": ad_text,
            "metadata": meta.model_dump()
        })
    except ValidationError as e:
        print(f"❌ Invalid record {ad.get('id')}: {e}")

print(f"✅ Prepared {len(documents)} documents.")


# ======================================================
# 10. UPSERT TO PINECONE
# ======================================================
print("\nStep 4: Embedding + Upserting...")

for i in range(0, len(documents), BATCH_SIZE):
    batch = documents[i:i + BATCH_SIZE]
    ids = [d["id"] for d in batch]
    texts = [d["text"] for d in batch]
    meta = [d["metadata"] for d in batch]

    try:
        vecs = embeddings.embed_documents(texts)
        index.upsert(vectors=zip(ids, vecs, meta))
        print(f"✅ Batch {i//BATCH_SIZE + 1} upserted ({len(batch)} docs)")
    except Exception as e:
        print(f"❌ Batch failed: {e}")


print("\n🎉 INGESTION COMPLETE")
print(f"📦 Total vectors in '{INDEX_NAME}': {len(documents)}")
