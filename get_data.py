import os
import json
import requests
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def safe_get(data_dict, key_path, default=None):
    """Safely get nested dictionary or list values using dot notation."""
    keys = key_path.split('.')
    val = data_dict
    try:
        for key in keys:
            if isinstance(val, list):
                key = int(key)
            val = val[key]
        return val
    except (KeyError, TypeError, IndexError, ValueError):
        return default


def determine_format_category(ad_creative):
    """Classify ad creative into Video, Carousel, or Static Image."""
    if not ad_creative:
        return "Unknown"
    if safe_get(ad_creative, 'asset_feed_spec.videos') or safe_get(ad_creative, 'object_story_spec.video_data.video_id'):
        return "Video/Reel"
    if safe_get(ad_creative, 'object_story_spec.link_data.child_attachments'):
        return "Carousel"
    if safe_get(ad_creative, 'image_url') or safe_get(ad_creative, 'asset_feed_spec.images') or safe_get(ad_creative, 'image_hash') or safe_get(ad_creative, 'thumbnail_url') or safe_get(ad_creative, 'object_story_spec.photo_data'):
        return "Static Image"
    return "Unknown"


def fetch_image_urls(hash_list, access_token, ad_account_id, api_version):
    """Query Meta API for real image URLs based on image_hashes."""
    if not hash_list:
        return {}
    print(f"Step 2: Resolving {len(hash_list)} image hashes...")
    hash_url_map = {}
    url = f"https://graph.facebook.com/{api_version}/act_{ad_account_id}/adimages"
    params = {
        'fields': 'hash,url',
        'hashes': json.dumps(list(hash_list)),
        'access_token': access_token
    }
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        for item in res.json().get('data', []):
            h, u = item.get('hash'), item.get('url')
            if h and u:
                hash_url_map[h] = u
    except Exception as e:
        print("❌ Error fetching image URLs:", e)
    return hash_url_map


def fetch_video_urls(video_ids, access_token, api_version):
    """Resolve video IDs into playable source URLs."""
    if not video_ids:
        return {}
    print(f"Step 2B: Resolving {len(video_ids)} video IDs...")
    video_url_map = {}
    for vid in video_ids:
        url = f"https://graph.facebook.com/{api_version}/{vid}"
        params = {'fields': 'source,permalink_url', 'access_token': access_token}
        try:
            res = requests.get(url, params=params)
            res.raise_for_status()
            data = res.json()
            # Prefer direct source, fallback to permalink
            video_url_map[vid] = data.get('source') or data.get('permalink_url')
        except Exception as e:
            print(f"⚠ Skipping video {vid}: {e}")
    return video_url_map


def download_images(hash_url_map, output_dir="data/images"):
    """Download static and carousel images locally."""
    os.makedirs(output_dir, exist_ok=True)
    for h, u in hash_url_map.items():
        try:
            filename = os.path.join(output_dir, f"{h}.jpg")
            if not os.path.exists(filename):
                img_data = requests.get(u, timeout=10).content
                with open(filename, "wb") as f:
                    f.write(img_data)
                print(f"✅ Downloaded {h}.jpg")
        except Exception as e:
            print(f"⚠ Failed to download {u}: {e}")



# -------------------------------------------------------------------
# Main Script
# -------------------------------------------------------------------
def get_data_script():
    load_dotenv()
    ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
    AD_ACCOUNT_ID = os.getenv("META_AD_ACCOUNT_ID")
    API_VERSION = "v24.0"

    if not all([ACCESS_TOKEN, AD_ACCOUNT_ID]):
        print("❌ Missing Meta credentials in .env file.")
        return

    os.makedirs("data", exist_ok=True)

    FIELDS = (
        "id,name,status,"
        "campaign{id,name,objective,status,start_time,stop_time},"
        "adset{id,name,optimization_goal,status,targeting{geo_locations,age_min,age_max,genders}},"
        "creative{title,body,image_hash,thumbnail_url,video_id,"
        "asset_feed_spec{videos{video_id},images{url}},"
        "object_story_spec{"
        "text_data{message},"
        "link_data{link,name,description,caption,picture,"
        "child_attachments{link,name,description,image_hash,video_id,picture}},"
        "video_data{video_id,image_url,title,call_to_action{type,value}},"
        "photo_data{image_hash,url}}},"
        "insights{spend,impressions,clicks,ctr,cpc,cpm,actions,purchase_roas}"
    )

    # -------------------------------------------------------------
    # Step 1: Fetch all ads
    # -------------------------------------------------------------
    ads, hashes, video_ids = [], set(), set()
    url = f"https://graph.facebook.com/{API_VERSION}/act_{AD_ACCOUNT_ID}/ads"
    params = {'access_token': ACCESS_TOKEN, 'fields': FIELDS, 'limit': 100}

    print("Fetching ads...")
    page = 1
    while url:
        res = requests.get(url, params=params if page == 1 else {})
        res.raise_for_status()
        data = res.json()
        page_ads = data.get("data", [])
        print(f"Fetched page {page} ({len(page_ads)} ads)")
        ads.extend(page_ads)

        for ad in page_ads:
            creative = ad.get("creative", {})
            ad["format_category"] = determine_format_category(creative)

            # Collect image hashes
            h = safe_get(creative, "image_hash")
            if h:
                hashes.add(h)
            for att in safe_get(creative, "object_story_spec.link_data.child_attachments", []):
                if att.get("image_hash"):
                    hashes.add(att["image_hash"])

            # Collect video IDs
            for v in safe_get(creative, "asset_feed_spec.videos", []):
                if "video_id" in v:
                    video_ids.add(v["video_id"])
            vid = safe_get(creative, "object_story_spec.video_data.video_id")
            if vid:
                video_ids.add(vid)

        url = data.get("paging", {}).get("next")
        page += 1

    print(f"✅ Total fetched: {len(ads)} ads across {page - 1} pages")

    # -------------------------------------------------------------
    # Step 2: Resolve Media URLs
    # -------------------------------------------------------------
    hash_url_map = fetch_image_urls(hashes, ACCESS_TOKEN, AD_ACCOUNT_ID, API_VERSION)
    download_images(hash_url_map)

    video_url_map = fetch_video_urls(video_ids, ACCESS_TOKEN, API_VERSION)

    # -------------------------------------------------------------
    # Step 3: Attach URLs into creatives
    # -------------------------------------------------------------
    for ad in ads:
        creative = ad.get("creative", {})
        top_hash = safe_get(creative, "image_hash")
        if top_hash in hash_url_map:
            creative["image_url"] = hash_url_map[top_hash]

        # Carousel attachments
        for att in safe_get(creative, "object_story_spec.link_data.child_attachments", []):
            h = att.get("image_hash")
            if h in hash_url_map:
                att["image_url"] = hash_url_map[h]

        # Video handling
        video_data = safe_get(creative, "object_story_spec.video_data")
        if video_data:
            vid = video_data.get("video_id")
            if vid and vid in video_url_map:
                video_data["video_url"] = video_url_map[vid]

    # -------------------------------------------------------------
    # Step 4: Save raw JSON
    # -------------------------------------------------------------
    json_path = "data/dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ads, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved raw dataset to {json_path}")

    # -------------------------------------------------------------
    # Step 5: Flatten to CSV
    # -------------------------------------------------------------
    records = []
    for ad in ads:
        creative = ad.get("creative", {})
        ins = safe_get(ad, "insights", {})

        spend = float(ins.get("spend", 0))
        impressions = int(ins.get("impressions", 0))
        clicks = int(ins.get("clicks", 0))
        ctr = float(ins.get("ctr", 0))
        cpc = float(ins.get("cpc", 0))
        cpm = float(ins.get("cpm", 0))

        roas_val = 0
        roas_field = ins.get("purchase_roas")
        if isinstance(roas_field, list) and len(roas_field) > 0:
            val = roas_field[0].get("value")
            if val:
                roas_val = float(val)

        records.append({
            "ad_id": ad.get("id"),
            "ad_name": ad.get("name"),
            "ad_status": ad.get("status"),
            "format_category": ad.get("format_category"),
            "creative_image_url": creative.get("image_url"),
            "thumbnail_url": creative.get("thumbnail_url"),
            "video_url": safe_get(creative, "object_story_spec.video_data.video_url"),
            "body": creative.get("body"),
            "link_url": safe_get(creative, "object_story_spec.link_data.link"),
            "spend": spend,
            "impressions": impressions,
            "clicks": clicks,
            "ctr": ctr,
            "cpc": cpc,
            "cpm": cpm,
            "roas": roas_val,
        })

    df = pd.DataFrame(records)
    csv_path = "data/meta_ads_data.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"📊 Exported structured dataset → {csv_path}")
    print(f"🕓 Run completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# -------------------------------------------------------------------
if __name__ == "__main__":
    get_data_script()
