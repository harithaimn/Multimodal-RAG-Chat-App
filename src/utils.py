import os
import json
import boto3
from botocore.exceptions import ClientError
from langchain_core.messages import AIMessage, HumanMessage

# --- S3 Configuration ---
S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
S3_PREFIX = "chat_history/" # Use a prefix to keep chat logs organized

def get_s3_client():
    """Initializes and returns a Boto3 S3 client."""
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION")
    )

def save_chat_history(session_id: str, messages: list, filters: dict = None, title: str = None):
    """
    Saves a list of LangChain message objects to an S3 bucket as a JSON file.

    Saves chat messages + optional filters + title to S3 as JSON.

    Args:
        session_id (str): Unique session identifier
        messages (list): List of LangChain messages (AIMessage/HumanMessage)
        filters (dict, optional): Dropdown selections for this session
        title (str, optional): User-defined title for the session
    """
    if not S3_BUCKET_NAME:
        print("Warning: AWS_S3_BUCKET_NAME not set. Chat history will not be saved.")
        return

    s3_client = get_s3_client()
    s3_key = f"{S3_PREFIX}{session_id}.json"
    
    # Serialize LangChain message objects to a JSON-compatible format
    serializable_messages = [
        {"type": msg.type, "content": msg.content}
        for msg in messages
    ]

    payload = {
        "title": title,
        "filters": filters or {},
        "messages": serializable_messages
    }
    
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=json.dumps(serializable_messages, indent=2),
            ContentType='application/json'
        )
    except ClientError as e:
        print(f"Error saving chat history to S3: {e}")

def load_chat_history(session_id: str) -> list:
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
    if not S3_BUCKET_NAME:
        return {"title": None, "filters": {}, "messages": []}

    s3_client = get_s3_client()
    s3_key = f"{S3_PREFIX}{session_id}.json"
    
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        content = response['Body'].read().decode('utf-8')
        data = json.loads(content)
        
        # Deserialize back into LangChain message objects
        messages = []
        for msg in data.get("messages", []):
            if msg['type'] == 'human':
                messages.append(HumanMessage(content=msg['content']))
            elif msg['type'] == 'ai':
                messages.append(AIMessage(content=msg['content']))
        
        return {
            "title": data.get("title"),
            "filters": data.get("filters", {}),
            "messages": messages
        }
    
    except ClientError as e:
        # If the file doesn't exist (e.g., a new chat), return an empty list
        if e.response['Error']['Code'] == 'NoSuchKey':
            return {"title": None, "filters": {}, "messages": []}
        print(f"Error loading chat history from S3: {e}")
        return {"title": None, "filters": {}, "messages": []}

# -----------------------------
# List All Saved Sessions
# -----------------------------
def get_saved_sessions() -> list:
    """
    Lists all saved session IDs from the S3 bucket, sorted by last modified.

    Returns: 
        list: [{"session_id": str, "title": str, "last_modified": datetime}]
    """
    if not S3_BUCKET_NAME:
        return []
    
    s3_client = get_s3_client()
    sessions = []

    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX)
        
        all_objects = []
        for page in pages:
            if 'Contents' in page:
                all_objects.extend(page['Contents'])
        
        # Sort objects by last modified time, newest first
        sorted_objects = sorted(all_objects, key=lambda obj: obj['LastModified'], reverse=True)
        
        for obj in sorted_objects:
            key = obj['Key']
            # Extract session_id from the key (e.g., 'chat_history/session123.json')
            session_id = key.replace(S3_PREFIX, "").replace(".json", "")
            if session_id: # Avoid including the prefix folder itself
                #sessions.append({"session_id": session_id, "last_modified": obj['LastModified']})
                continue

            try:
                session_data = load_chat_history(session_id)
                title = session_data.get("title") or f"Chat from {obj['LastModified'].strftime('%Y-%m-%d %H:%M')}"
            except Exception:
                title = f"Chat from {obj['LastModified'].strftime('%Y-%m-%d %H:%M')}"
            
            sessions.append({
                "session_id": session_id,
                "title": title,
                "last_modified": obj['LastModified']
            })

    except ClientError as e:
        print(f"Error listing saved sessions from S3: {e}")
    
    return sessions