import os
import json
import uuid
import boto3
import re
#from botocore.exceptions import ClientError
#from langchain_core.messages import AIMessage, HumanMessage

from typing import Dict, List, Optional
from datetime import datetime

from src.context_rules import detect_intended_industry

# --- S3 Configuration ---
S3_ENABLED = os.getenv("S3_ENABLED", "false").lower() == "true"
S3_BUCKET = os.getenv("AWS_S3_BUCKET_NAME")
#S3_PREFIX = "chat_history/" # Use a prefix to keep chat logs organized
LOCAL_SAVE_DIR = "saved_chats/"

if not os.path_exists(LOCAL_SAVE_DIR):
    os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)

# def get_s3_client():
#     """Initializes and returns a Boto3 S3 client."""
#     return boto3.client(
#         's3',
#         aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#         aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#         region_name=os.getenv("AWS_DEFAULT_REGION")
#     )

# -----------------------------
# 2. SAVE CHAT HISTORY
# -----------------------------
def save_chat_history(
        session_id: str,
        messages: List[Dict],
        filters: Dict[str, str],
        title: Optional[str] = None, 
    ):
    """
    Saves a list of LangChain message objects to an S3 bucket as a JSON file.

    Saves chat messages + optional filters + title to S3 as JSON.

    Args:
        session_id (str): Unique session identifier
        messages (list): List of LangChain messages (AIMessage/HumanMessage)
        filters (dict, optional): Dropdown selections for this session
        title (str, optional): User-defined title for the session
    """
    data = {
        "title": title or f"Session-{session_id[:8]}",
        "filters": filters,
        "messages": [
            {
                "type": msg.type("type", "human" if i % 2 == 0 else "ai"),
                "content": msg.get("content", ""),
                "title": msg.get("title", None),
            }
            for i, msg in enumerate(messages)
        ],
        "timestamp": datetime.utcnow().isoformat(),
    }

    filename = f"{session_id}.json"
    local_path = os.path.join(LOCAL_SAVE_DIR, filename)

    # ------ Save locally ------
    with open(local_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # ------ Optionally save to S3 ------
    if S3_ENABLED:
        try:
            s3 = boto3.client("s3")
            s3.upload_file(local_path, S3_BUCKET, f"chats/{filename}")
            print(f"✅ Uploaded chat {session_id} to S3 bucket {S3_BUCKET}")
        except Exception as e:
            print(f"⚠️ S3 upload failed: {e}")
    
    return data

    # # if not S3_BUCKET_NAME:
    # #     print("Warning: AWS_S3_BUCKET_NAME not set. Chat history will not be saved.")
    # #     return

    # # s3_client = get_s3_client()
    # # s3_key = f"{S3_PREFIX}{session_id}.json"
    
    # # Serialize LangChain message objects to a JSON-compatible format
    # serializable_messages = [
    #     {"type": msg.type, "content": msg.content}
    #     for msg in messages
    # ]

    # payload = {
    #     "title": title,
    #     "filters": filters or {},
    #     "messages": serializable_messages
    # }
    
    # try:
    #     s3_client.put_object(
    #         Bucket=S3_BUCKET_NAME,
    #         Key=s3_key,
    #         Body=json.dumps(serializable_messages, indent=2),
    #         ContentType='application/json'
    #     )
    # except ClientError as e:
    #     print(f"Error saving chat history to S3: {e}")

# -----------------------------
# 3. LOAD CHAT HISTORY
# -----------------------------
def load_chat_history(session_id: str) -> Optional[Dict]:
    """
    Loads and deserializes a chat history from an S3 JSON file.
    Loads chat session from S3, returning messages + filters + title.

    Returns:
        dict: {
            "title": str,
            "filters": dict,
            "messages": list of AIMessage/HumanMessage
        }
    """
    filename = f"{session_id}.json"
    local_path = os.path.join(LOCAL_SAVE_DIR, filename)
    
    # Try loading from local first
    if os.path.exists(local_path):
        with open(local_path, "r", encoding='utf-8') as f:
            return json.load(f)
    
    # Try loading from S3 if enabled
    if S3_ENABLED:
        try:
            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=S3_BUCKET, Key=f"chats/{filename}")
            return json.loads(obj["Body"].read().decode('utf-8'))
        except Exception as e:
            print(f"⚠️ Failed to load {session_id} from S3: {e}")

    return None

    # s3_client = get_s3_client()
    # s3_key = f"{S3_PREFIX}{session_id}.json"
    
    # try:
    #     response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
    #     content = response['Body'].read().decode('utf-8')
    #     data = json.loads(content)
        
    #     # Deserialize back into LangChain message objects
    #     messages = []
    #     for msg in data.get("messages", []):
    #         if msg['type'] == 'human':
    #             messages.append(HumanMessage(content=msg['content']))
    #         elif msg['type'] == 'ai':
    #             messages.append(AIMessage(content=msg['content']))
        
    #     return {
    #         "title": data.get("title"),
    #         "filters": data.get("filters", {}),
    #         "messages": messages
    #     }
    
    # except ClientError as e:
    #     # If the file doesn't exist (e.g., a new chat), return an empty list
    #     if e.response['Error']['Code'] == 'NoSuchKey':
    #         return {"title": None, "filters": {}, "messages": []}
    #     print(f"Error loading chat history from S3: {e}")
    #     return {"title": None, "filters": {}, "messages": []}

# -----------------------------
# 4. List All Saved Sessions
# -----------------------------
def get_saved_sessions() -> List[Dict]:
    """
    Lists all saved session IDs from the S3 bucket, sorted by last modified.

    Returns: 
        list: [{"session_id": str, "title": str, "last_modified": datetime}]
    """
    sessions = []

    # ----- Local Files -----
    for fname in os.listdir(LOCAL_SAVE_DIR):
        if fname.endswith(".json"):
            fpath = os.path.join(LOCAL_SAVE_DIR, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                sessions.append(
                    {
                        "session_id": fname.replace(".json", ""),
                        "title": data.get("title", "Untitled"),
                        "filters": data.get("filters", {}),
                        "timestamp": data.get("timestamp", None),
                    }
                )
    
    # ----- S3 Files -----
    if S3_ENABLED:
        try:
            s3 = boto3.client("s3")
            response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix="chats/")
            for obj in response.get("Contents", []):
                if obj["Key"].endswith(".json"):
                    sessions.append(
                        {
                            "session_id": obj["Key"].replace("chats/", "").replace(".json", ""),
                            "title": obj["Key"].split("/")[-1],
                            "timestamp": obj["LastModified"].isoformat(),
                            "filters": {},
                        }
                    )
        except Exception as e:
            print(f"⚠️ Failed to list sessions from S3: {e}")
    
    # Sort newest first
    sessions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return sessions

    # if not S3_BUCKET_NAME:
    #     return []
    
    # s3_client = get_s3_client()
    # sessions = []

    # try:
    #     paginator = s3_client.get_paginator('list_objects_v2')
    #     pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX)
        
    #     all_objects = []
    #     for page in pages:
    #         if 'Contents' in page:
    #             all_objects.extend(page['Contents'])
        
    #     # Sort objects by last modified time, newest first
    #     sorted_objects = sorted(all_objects, key=lambda obj: obj['LastModified'], reverse=True)
        
    #     for obj in sorted_objects:
    #         key = obj['Key']
    #         # Extract session_id from the key (e.g., 'chat_history/session123.json')
    #         session_id = key.replace(S3_PREFIX, "").replace(".json", "")
    #         if session_id: # Avoid including the prefix folder itself
    #             #sessions.append({"session_id": session_id, "last_modified": obj['LastModified']})
    #             continue

    #         try:
    #             session_data = load_chat_history(session_id)
    #             title = session_data.get("title") or f"Chat from {obj['LastModified'].strftime('%Y-%m-%d %H:%M')}"
    #         except Exception:
    #             title = f"Chat from {obj['LastModified'].strftime('%Y-%m-%d %H:%M')}"
            
    #         sessions.append({
    #             "session_id": session_id,
    #             "title": title,
    #             "last_modified": obj['LastModified']
    #         })

    # except ClientError as e:
    #     print(f"Error listing saved sessions from S3: {e}")
    
    # return sessions

# ===========================================================
# 5. NLP UTILITY FUNCTIONS
# ===========================================================

def extract_keywords(user_text: str, top_k: int=5) -> List[str]:
    """
    Basic keyword extractor - could later be replaced with OpenAI embedding or regex weighting.
    """
    text = re.sub(r"[^\w\s]", "", user_text.lower())
    words = text.split()
    stopwords = {"the", "and", "for", "with", "that", "this", "from", "you", "your", "our", "are", "was"}
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return list(dict.fromkeys(keywords))[:top_k]  # Deduplicate while preserving order

def semantic_intent_summary(user_text: str) -> Dict[str, Optional[str]]:
    """
    Lightweight semantic intent detection.
    """
    industry = detect_intended_industry(user_text)
    keywords = extract_keywords(user_text)
    return {
        "detected_industry": industry,
        "keywords": keywords,
    }

# ============================================================
# 6. SESSION UTILITIES
# ============================================================
def create_new_session(filters: Dict[str, str], title: Optional[str] = None) -> str:
    """
    Creates a new chat session and returns session_id.
    """
    session_id = str(uuid.uuid4())
    data = {
        "title": title or f"Sessio-{datetime.now().strftime("%Y%m%d-%H%M")}",
        "filters": filters,
        "messages": [],
        "timestamp": datetime.utcnow().isoformat(),
    }
    filename = os.path.join(LOCAL_SAVE_DIR, f"{session_id}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return session_id