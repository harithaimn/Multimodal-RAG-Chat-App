import os
import json
import requests
from dotenv import load_dotenv
import pandas as pd
import time
import boto3
from urllib.parse import urlparse
from tqdm import tqdm  # Add this import

# Load .env
load_dotenv()

# ============ CONFIG SETUP ============
CONFIG_PATH = "config/config.yaml"

def load_config():
    return {
        "embedding_model": {"model_name": "text-embedding-3-small"},
        "pinecone": {"index_name": "meta-ads-rag-multimodal"}
    }

config = load_config()

ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
AD_ACCOUNT_ID = os.getenv("META_AD_ACCOUNT_ID")
API_VERSION = "v24.0"

if not all([ACCESS_TOKEN, AD_ACCOUNT_ID]):
    print("❌ Missing Meta Ads credentials in .env file.")
    exit()

# AWS S3 setup
S3_BUCKET = os.getenv("AWS_S3_BUCKET")
s3 = boto3.client("s3", 
                  aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
                  aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                  region_name=os.getenv("AWS_REGION"))

# ============ HELPER FUNCTIONS ============
def safe_get(data_dict, key_path, default=None):
    """Safely get nested dictionary or list values using dot notation."""
    keys = key_path.split('.')
    val = data_dict
    try:
        for key in keys:
            if isinstance(val, list):
                try:
                    key = int(key)
                except ValueError:
                    return default
            val = val[key]
        return val
    except (KeyError, TypeError, IndexError):
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

def fetch_image_urls(hash_list, access_token, ad_account_id, api_version="v24.0", batch_size=100):
    if not hash_list:
        return {}
    print(f"Step 2A: Resolving {len(hash_list)} unique image hashes in batches of {batch_size}...")
    hash_url_map = {}
    url = f"https://graph.facebook.com/{api_version}/act_{ad_account_id}/adimages"

    for i in range(0, len(hash_list), batch_size):
        batch = list(hash_list)[i:i + batch_size]
        print(f"  - Fetching batch {i//batch_size + 1}/{(len(hash_list) - 1)//batch_size + 1} ({len(batch)} hashes)...")
        params = {
            'fields': 'hash,url',
            'hashes': json.dumps(batch),
            'access_token': access_token
        }
        try:
            res = requests.get(url, params=params, timeout=30)
            res.raise_for_status()
            data = res.json()
            if 'error' in data:
                print(f"    ⚠️ API Error in batch: {data['error']}")
                continue

            batch_count = 0
            for item in data.get('data', []):
                h, u = item.get('hash'), item.get('url')
                if h and u:
                    if h in hash_url_map:
                        print(f"    ⚠️ Warning: Hash {h} already exists in map, overwriting.")
                    hash_url_map[h] = u
                    batch_count += 1

            print(f"    - Batch {i//batch_size + 1} resolved {batch_count} unique URLs.")

        except requests.exceptions.Timeout:
            print(f"    ⚠️ Timeout fetching batch {i//batch_size + 1}. Skipping.")
        except requests.exceptions.HTTPError as e:
            print(f"    ⚠️ HTTP Error {e.response.status_code} fetching batch {i//batch_size + 1}: {e.response.text}")
        except requests.exceptions.RequestException as e:
            print(f"    ⚠️ Request Error fetching batch {i//batch_size + 1}: {e}")
        except json.JSONDecodeError:
            print(f"    ⚠️ JSON Decode Error fetching batch {i//batch_size + 1}. Response text was not JSON.")
        time.sleep(0.2)

    print(f"✅ Successfully resolved {len(hash_url_map)} unique image URLs out of {len(hash_list)} requested.")
    return hash_url_map

def fetch_video_urls_with_backoff(video_ids, access_token, api_version="v24.0", max_retries=3, base_delay=1):
    if not video_ids:
        return {}
    print(f"Step 2B: Resolving {len(video_ids)} video IDs with retry logic...")
    video_url_map = {}
    BASE_URL = f"https://graph.facebook.com/{api_version}/"
    
    for vid in video_ids:
        params = {'fields': 'source', 'access_token': access_token}
        url = f"{BASE_URL}{vid}"
        
        for attempt in range(max_retries):
            try:
                res = requests.get(url, params=params, timeout=30)
                res.raise_for_status()
                data = res.json()
                if 'source' in data:
                    video_url_map[vid] = data['source']
                    print(f"  - Resolved video {vid} on attempt {attempt+1}")
                    break
                else:
                    print(f"  - No 'source' field found for video {vid} on attempt {attempt+1}.")
                    if attempt == max_retries - 1:
                        print(f"    - Failed to resolve video {vid} after {max_retries} attempts.")
                        break
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                print(f"  - HTTP Error {status_code} for video {vid} on attempt {attempt+1}: {e.response.text}")
                if status_code == 400:
                    try:
                        error_data = e.response.json().get('error', {})
                        if error_data.get('code') == 10:
                            print(f"    - Skipping video {vid} due to permission error (#10).")
                            break
                    except json.JSONDecodeError:
                        pass
                if status_code in [500, 502, 503, 504]:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"    - Retrying video {vid} in {delay} seconds...")
                        time.sleep(delay)
                        continue
                if attempt == max_retries - 1:
                    print(f"    - Failed to resolve video {vid} after {max_retries} attempts.")
                break
            
            except requests.exceptions.RequestException as e:
                print(f"  - Request Error for video {vid} on attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"    - Retrying video {vid} in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"    - Failed to resolve video {vid} after {max_retries} attempts.")
                break

    print(f"✅ Successfully resolved {len(video_url_map)} video URLs out of {len(video_ids)}.")
    return video_url_map

def download_media(url, media_dir, ad_id, identifier, media_type="generic", prefix=""):
    if not url:
        return None

    try:
        os.makedirs(media_dir, exist_ok=True)
        parsed_url = urlparse(url)
        ext = os.path.splitext(parsed_url.path)[1]
        if not ext:
            if 'video' in url or 'mp4' in url or 'mov' in url:
                ext = '.mp4'
            else:
                ext = '.jpg'

        safe_filename = f"{prefix}{ad_id}_{identifier}{ext}"
        filepath = os.path.join(media_dir, safe_filename)

        if os.path.exists(filepath):
            print(f"  - File already exists locally: {filepath}")
            return filepath

        print(f"  - Downloading: {url[:60]}... -> {filepath}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  - Downloaded successfully: {filepath}")
        return filepath
    except Exception as e:
        print(f"⚠ Error downloading media from {url[:60]}...: {e}")
        return None

def upload_to_s3_and_get_url(local_path, ad_id, media_dir):
    """
    Uploads an image file from the local media directory to AWS S3.
    Returns the public URL of the uploaded image.
    """
    if not local_path or not os.path.exists(local_path):
        print(f"❌ Invalid image path: {local_path}")
        return None

    # Ensure directory exists
    os.makedirs(media_dir, exist_ok=True)

    # Create a unique S3 object key using ad_id or other identifiers
    file_name = os.path.basename(local_path)
    s3_key = f"ad_creatives/{ad_id}/{file_name}"  # Store in sub-directory per ad_id

    try:
        # Upload image to S3
        s3.upload_file(local_path, S3_BUCKET, s3_key, ExtraArgs={"ACL": "public-read", "ContentType": "image/jpeg"})
        # Generate public URL
        image_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"
        print(f"✅ Uploaded image to S3: {image_url}")
        return image_url
    except Exception as e:
        print(f"❌ Failed to upload {file_name} to S3: {e}")
        return None


def generate_s3_urls_for_ads(ads, media_dir="data/media"):
    """
    Processes all ads, uploads images to S3, and generates S3 URLs.
    Saves the URLs into a JSON file.
    """
    image_url_map = {}

    for ad in tqdm(ads, desc="Uploading images to S3", ncols=100):
        ad_id = ad.get('id')
        if not ad_id:
            print(f"⚠️ Ad {ad} does not have an ID, skipping...")
            continue
        
        creative = ad.get("creative", {})
        # For primary image
        primary_image_path = creative.get("local_image_path")
        if primary_image_path:
            image_url = upload_to_s3_and_get_url(primary_image_path, ad_id, media_dir)
            if image_url:
                image_url_map[ad_id] = image_url
            else:
                print(f"⚠️ Failed to upload primary image for Ad {ad_id}")

        # For carousel images (if applicable)
        children = creative.get("object_story_spec", {}).get("link_data", {}).get("child_attachments", [])
        for idx, child in enumerate(children):
            child_image_path = child.get("local_image_path")
            if child_image_path:
                image_url = upload_to_s3_and_get_url(child_image_path, ad_id, media_dir)
                if image_url:
                    image_url_map[f"{ad_id}_carousel_{idx}"] = image_url
                else:
                    print(f"⚠️ Failed to upload carousel image for Ad {ad_id}, item {idx}")

    # Save the S3 URLs to a JSON file
    s3_url_file = "data/dataset_s3_url.json"
    with open(s3_url_file, "w", encoding="utf-8") as f:
        json.dump(image_url_map, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved image URLs to {s3_url_file}")


# ============ MAIN SCRIPT ============

def get_data_script():
    media_dir = "data/media"
    os.makedirs(media_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    FIELDS = (
        "id,name,status,"
        "campaign{id,name,objective,status,daily_budget,start_time,stop_time},"
        "adset{id,name,optimization_goal,status,daily_budget}"
    )

    url = f"https://graph.facebook.com/{API_VERSION}/act_{AD_ACCOUNT_ID}/ads"
    params = {
        'access_token': ACCESS_TOKEN,
        'fields': FIELDS,
        'limit': 100
    }
    ads = []

    print("Step 1: Fetching basic ad information...")
    page = 1
    while url:
        print(f"  - Fetching basic info, page {page}...")
        try:
            res = requests.get(url, params=params if page == 1 else {}, timeout=30)
            res.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP Error {e.response.status_code} on page {page}: {e.response.text}")
            if e.response.status_code in [500, 502, 503, 504]:
                print("    - Retrying page...")
                time.sleep(1)
                continue
            else:
                print("    - Skipping page due to error.")
                break
        except requests.exceptions.RequestException as e:
            print(f"❌ Request Error on page {page}: {e}")
            print("    - Retrying page...")
            time.sleep(1)
            continue

        data = res.json()
        page_ads = data.get("data", [])
        ads.extend(page_ads)
        url = data.get("paging", {}).get("next")
        page += 1
        time.sleep(0.1)

    print(f"✅ Total basic ad info fetched: {len(ads)} ads across {page - 1} pages.")

    print("Step 1.5: Fetching creative data for each ad with retry logic...")
    all_creatives = {}

    # Fetch creative and insights (you can modify the previous part for this)
    print("Step 2: Resolving media URLs and uploading to S3...")
    generate_s3_urls_for_ads(ads)

    for idx, ad in enumerate(tqdm(ads, desc="Fetching Creatives")):
        ad_id = ad.get('id')
        if not ad_id:
            continue

        creative_fields = (
            "title,body,image_hash,thumbnail_url,image_url,"
            "asset_feed_spec{videos{video_id},images{url}},"
            "object_story_spec{"
            "text_data{message},"
            "link_data{link,name,description,caption,picture,"
            "child_attachments{link,description,image_hash,picture,name,caption,call_to_action}},"
            "video_data{video_id,image_url,title,call_to_action{type,value}},"
            "photo_data{image_hash,url}"
            "}"
        )
        ad_detail_url = f"https://graph.facebook.com/{API_VERSION}/{ad_id}"
        ad_detail_params = {
            'access_token': ACCESS_TOKEN,
            'fields': f"creative{{{creative_fields}}}"
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                ad_detail_res = requests.get(ad_detail_url, params=ad_detail_params, timeout=30)
                ad_detail_res.raise_for_status()
                ad_detail_data = ad_detail_res.json()
                fetched_creative = ad_detail_data.get('creative')
                if fetched_creative:
                    all_creatives[ad_id] = fetched_creative
                    ad['creative'] = fetched_creative
                    format_cat = determine_format_category(fetched_creative)
                    ad['format_category'] = format_cat
                    print(f"  - Fetched creative for ad {ad_id} on attempt {attempt+1}")
                else:
                    print(f"  - No creative found for ad {ad_id} on attempt {attempt+1}")
                    all_creatives[ad_id] = {}
                break
            except requests.exceptions.RequestException as e:
                print(f"    - Error fetching creative for ad {ad_id}: {e}")
                continue

        time.sleep(0.2)

    print(f"✅ Attempted to fetch creative data for {len(all_creatives)} ads out of {len(ads)}.")

    print("Step 1.75: Fetching maximum insights data for each ad with retry logic...")
    all_insights = {}
    insights_fields = "spend,impressions,clicks,ctr,cpc,cpm,actions,results,cost_per_action_type,purchase_roas"

    for idx, ad in enumerate(tqdm(ads, desc="Fetching Insights")):
        ad_id = ad.get('id')
        if not ad_id:
            continue

        insight_url_per_ad = f"https://graph.facebook.com/{API_VERSION}/{ad_id}/insights"
        insight_params_per_ad = {
            'access_token': ACCESS_TOKEN,
            'fields': insights_fields,
            'date_preset': 'maximum'
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                insight_res = requests.get(insight_url_per_ad, params=insight_params_per_ad, timeout=30)
                insight_res.raise_for_status()
                insight_data = insight_res.json()
                if 'data' in insight_data and len(insight_data['data']) > 0:
                    all_insights[ad_id] = insight_data['data'][0]
                    ad['insights'] = {'data': [all_insights[ad_id]]}
                    print(f"  - Fetched insights for ad {ad_id} on attempt {attempt+1}")
                else:
                    all_insights[ad_id] = {}
                    ad['insights'] = {'data': [{}]}
                    print(f"  - No maximum insights data found for ad {ad_id} on attempt {attempt+1}")
                break
            except requests.exceptions.RequestException as e:
                print(f"    - Error fetching insights for ad {ad_id}: {e}")
                continue

        time.sleep(0.2)

    print(f"✅ Attempted to fetch maximum insights data for {len(all_insights)} ads out of {len(ads)}.")

    print("Step 2: Collecting media hashes and IDs from fetched creatives...")
    hashes = set()
    video_ids = set()
    for ad in ads:
        creative = ad.get("creative", {})

        h = safe_get(creative, "image_hash")
        if h:
            hashes.add(h)
        else:
            print(f"  - Ad {ad.get('id')} has no primary image_hash.")

        children = safe_get(creative, "object_story_spec.link_data.child_attachments", [])
        for att in children:
            att_hash = att.get("image_hash")
            if att_hash:
                hashes.add(att_hash)

        photo_data = safe_get(creative, "object_story_spec.photo_data")
        if photo_data and 'image_hash' in photo_data:
            photo_hash = photo_data['image_hash']
            hashes.add(photo_hash)

        for v in safe_get(creative, "asset_feed_spec.videos", []):
            vid = v.get("video_id")
            if vid:
                video_ids.add(vid)

        obj_story_vid = safe_get(creative, "object_story_spec.video_data.video_id")
        if obj_story_vid:
            video_ids.add(obj_story_vid)

    print(f"  - Found {len(hashes)} unique image hashes and {len(video_ids)} unique video IDs from fetched creatives.")

    print("Step 3: Resolving collected media URLs...")
    hash_url_map = fetch_image_urls(list(hashes), ACCESS_TOKEN, AD_ACCOUNT_ID, API_VERSION)
    video_url_map = fetch_video_urls_with_backoff(list(video_ids), ACCESS_TOKEN, API_VERSION)

    print("Step 4: Injecting resolved media URLs back into ad data and downloading locally...")
    resolved_image_count = 0
    missing_image_count = 0
    resolved_video_count = 0
    missing_video_count = 0

    for ad_idx, ad in enumerate(ads):
        ad_id = ad.get("id")
        if not ad_id:
            continue

        creative = ad.get("creative", {})

        top_hash = safe_get(creative, "image_hash")
        if top_hash:
            if top_hash in hash_url_map:
                creative["image_url"] = hash_url_map[top_hash]
                local_path = download_media(creative["image_url"], media_dir, ad_id, top_hash, "image", prefix="img_")
                if local_path:
                    creative["local_image_path"] = local_path
                    resolved_image_count += 1
                else:
                    print(f"  - Failed to download primary image for ad {ad_id} (hash {top_hash})")
            else:
                print(f"  - Ad {ad_idx} ({ad_id}): Primary image hash '{top_hash}' not found in hash_url_map.")
                missing_image_count += 1

        children = safe_get(creative, "object_story_spec.link_data.child_attachments", [])
        for idx, child in enumerate(children):
            child_hash = child.get("image_hash")
            if child_hash:
                if child_hash in hash_url_map:
                    child["image_url"] = hash_url_map[child_hash]
                    local_path = download_media(child["image_url"], media_dir, ad_id, f"{child_hash}_c{idx}", "image", prefix="carousel_")
                    if local_path:
                        child["local_image_path"] = local_path
                    resolved_image_count += 1
                else:
                    print(f"  - Ad {ad_idx} ({ad_id}): Carousel item {idx} image hash '{child_hash}' not found in hash_url_map.")
                    missing_image_count += 1

        photo_data = safe_get(ad, 'creative.object_story_spec.photo_data')
        if photo_data and 'image_hash' in photo_data:
            photo_hash = photo_data['image_hash']
            if photo_hash in hash_url_map:
                photo_data['url'] = hash_url_map[photo_hash]
                local_path = download_media(photo_data['url'], media_dir, ad_id, photo_hash, "image", prefix="photo_")
                if local_path:
                    photo_data['local_image_path'] = local_path
                resolved_image_count += 1
            else:
                print(f"  - Ad {ad_idx} ({ad_id}): Photo data hash '{photo_hash}' not found in hash_url_map.")
                missing_image_count += 1

        for v_idx, v in enumerate(safe_get(creative, "asset_feed_spec.videos", [])):
            vid = v.get("video_id")
            if vid:
                if vid in video_url_map:
                    v["source_url"] = video_url_map[vid]
                    local_path = download_media(v["source_url"], media_dir, ad_id, vid, "video", prefix="video_asset_")
                    if local_path:
                        v["local_video_path"] = local_path
                    resolved_video_count += 1
                else:
                    print(f"  - Ad {ad_idx} ({ad_id}): Asset feed video ID '{vid}' not found in video_url_map.")
                    missing_video_count += 1

        video_data = safe_get(creative, "object_story_spec.video_data")
        if video_data and "video_id" in video_data:
            vid = video_data["video_id"]
            if vid in video_url_map:
                video_data["video_url"] = video_url_map[vid]
                local_path = download_media(video_data["video_url"], media_dir, ad_id, vid, "video", prefix="video_")
                if local_path:
                    video_data["local_video_path"] = local_path
                resolved_video_count += 1
            else:
                print(f"  - Ad {ad_idx} ({ad_id}): Object story video ID '{vid}' not found in video_url_map.")
                missing_video_count += 1

    print(f"  - Images: {resolved_image_count} resolved/ downloaded, {missing_image_count} missing/failed.")
    print(f"  - Videos: {resolved_video_count} resolved/ downloaded, {missing_video_count} missing/failed.")

    print("Step 4.5: Reorganizing data into Campaign -> AdSet -> Ad hierarchy...")
    hierarchical_data = {}

    for ad in ads:
        campaign_id = ad.get("campaign", {}).get("id")
        adset_id = ad.get("adset", {}).get("id")

        if not campaign_id or not adset_id:
            print(f"  - Warning: Ad {ad.get('id')} missing campaign or adset ID, skipping hierarchy assignment.")
            continue

        if campaign_id not in hierarchical_data:
            hierarchical_data[campaign_id] = {
                "id": campaign_id,
                "name": ad.get("campaign", {}).get("name", ""),
                "objective": ad.get("campaign", {}).get("objective", ""),
                "status": ad.get("campaign", {}).get("status", ""),
                "start_time": ad.get("campaign", {}).get("start_time", ""),
                "stop_time": ad.get("campaign", {}).get("stop_time", ""),
                "daily_budget": ad.get("campaign", {}).get("daily_budget", ""),
                "adsets": {}
            }

        if adset_id not in hierarchical_data[campaign_id]["adsets"]:
            hierarchical_data[campaign_id]["adsets"][adset_id] = {
                "id": adset_id,
                "name": ad.get("adset", {}).get("name", ""),
                "optimization_goal": ad.get("adset", {}).get("optimization_goal", ""),
                "status": ad.get("adset", {}).get("status", ""),
                "daily_budget": ad.get("adset", {}).get("daily_budget", ""),
                "targeting": ad.get("adset", {}).get("targeting", {}),
                "ads": []
            }

        hierarchical_data[campaign_id]["adsets"][adset_id]["ads"].append(ad)

    print(f"✅ Reorganized data into {len(hierarchical_data)} campaigns.")

    json_path = "data/dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(hierarchical_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved hierarchical dataset (Campaign -> AdSet -> Ad, with metrics and media paths) to {json_path}")

    print("Step 6: Flattening hierarchical data into CSV...")
    records = []
    for campaign_id, campaign_data in hierarchical_data.items():
        for adset_id, adset_data in campaign_data["adsets"].items():
            for ad in adset_data["ads"]:
                ins_data = ad.get("insights", {}).get("data", [])
                ins = ins_data[0] if ins_data else {}

                spend = float(ins.get("spend", 0))
                impressions = int(ins.get("impressions", 0))
                clicks = int(ins.get("clicks", 0))
                ctr_str = ins.get("ctr", "0")
                try:
                    ctr = float(ctr_str)
                except ValueError:
                    ctr = 0.0
                cpc_str = ins.get("cpc", "0")
                try:
                    cpc = float(cpc_str)
                except ValueError:
                    cpc = 0.0
                cpm_str = ins.get("cpm", "0")
                try:
                    cpm = float(cpm_str)
                except ValueError:
                    cpm = 0.0

                roas_val = 0.0
                roas_field = ins.get("purchase_roas")
                if isinstance(roas_field, list) and len(roas_field) > 0:
                    val = roas_field[0].get("value")
                    if val:
                        try:
                            roas_val = float(val)
                        except (ValueError, TypeError):
                            roas_val = 0.0
                elif isinstance(roas_field, (int, float)):
                    roas_val = float(roas_field)

                actions = ins.get("actions", [])
                conversions = sum(int(a.get("value", 0)) for a in actions if a.get("action_type") in ["offsite_conversion", "purchase", "onsite_conversion.purchase"])
                conversion_rate = (conversions / clicks * 100) if clicks > 0 else 0.0

                resolved_video_url = safe_get(ad, "creative.object_story_spec.video_data.video_url")
                local_video_path = safe_get(ad, "creative.object_story_spec.video_data.local_video_path")

                primary_local_image_path = ad.get("creative", {}).get("local_image_path")
                carousel_local_image_paths = []
                children = safe_get(ad, "creative.object_story_spec.link_data.child_attachments", [])
                for child in children:
                    child_local_path = child.get("local_image_path")
                    if child_local_path:
                        carousel_local_image_paths.append(child_local_path)

                all_local_image_paths_list = []
                if primary_local_image_path:
                    all_local_image_paths_list.append(primary_local_image_path)
                all_local_image_paths_list.extend(carousel_local_image_paths)
                combined_local_image_paths = ",".join(all_local_image_paths_list) if all_local_image_paths_list else ""

                rec = {
                    "ad_id": ad.get("id"),
                    "ad_name": ad.get("name"),
                    "ad_status": ad.get("status"),
                    "format_category": ad.get("format_category"),
                    "campaign_id": campaign_data.get("id"),
                    "campaign_name": campaign_data.get("name"),
                    "campaign_objective": campaign_data.get("objective"),
                    "adset_id": adset_data.get("id"),
                    "adset_name": adset_data.get("name"),
                    "optimization_goal": adset_data.get("optimization_goal"),
                    "targeting": adset_data.get("targeting", {}),
                    "creative_title": ad.get("creative", {}).get("title"),
                    "creative_body": ad.get("creative", {}).get("body"),
                    "creative_image_path": combined_local_image_paths,
                    "creative_video_path": local_video_path,
                    "creative_video_url": resolved_video_url,
                    "creative_thumbnail_url": ad.get("creative", {}).get("thumbnail_url"),
                    "copy_text": safe_get(ad, "creative", {}).get("object_story_spec", {}).get("text_data", {}).get("message"),
                    "link_url": safe_get(ad, "creative", {}).get("object_story_spec", {}).get("link_data", {}).get("link"),
                    "spend": spend,
                    "impressions": impressions,
                    "clicks": clicks,
                    "ctr": ctr,
                    "cpc": cpc,
                    "cpm": cpm,
                    "roas": roas_val,
                    "conversions": conversions,
                    "conversion_rate": conversion_rate,
                }
                records.append(rec)

    df = pd.DataFrame(records)
    csv_path = "data/meta_ads_data.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ Exported {len(df)} ads to {csv_path}")


if __name__ == "__main__":
    get_data_script()
