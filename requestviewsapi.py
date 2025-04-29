import pandas as pd
import requests
import re
import time
from tqdm import tqdm

API_KEY = "AIzaSyAVGuPz1eBeGmL4dtYwt2NkVxlP4S5qeDg"
INPUT_CSV = "tube_english_only_detected.csv"
OUTPUT_CSV = "tube_english_only_with_metadata.csv"

def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", str(url))
    return match.group(1) if match else None

def get_video_metadata(video_ids):
    url = (
        "https://www.googleapis.com/youtube/v3/videos"
        "?part=snippet,statistics,contentDetails"
        f"&id={','.join(video_ids)}"
        f"&key={API_KEY}"
    )
    response = requests.get(url)
    if response.status_code != 200:
        print("API error:", response.text)
        return {}
    items = response.json().get("items", [])
    data = {}
    for item in items:
        vid = item["id"]
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        details = item.get("contentDetails", {})
        data[vid] = {
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "publishedAt": snippet.get("publishedAt"),
            "categoryId": snippet.get("categoryId"),
            "tags": "|".join(snippet.get("tags", [])),
            "duration": details.get("duration"),
            "thumbnail": snippet.get("thumbnails", {}).get("default", {}).get("url"),
            "viewCount": stats.get("viewCount"),
            "likeCount": stats.get("likeCount"),
            "commentCount": stats.get("commentCount"),
            "channelId": snippet.get("channelId"),
            "channelTitle": snippet.get("channelTitle"),
        }
    return data

def get_channel_subscribers(channel_ids):
    url = (
        "https://www.googleapis.com/youtube/v3/channels"
        "?part=statistics"
        f"&id={','.join(channel_ids)}"
        f"&key={API_KEY}"
    )
    response = requests.get(url)
    if response.status_code != 200:
        print("API error:", response.text)
        return {}
    items = response.json().get("items", [])
    return {item["id"]: item["statistics"].get("subscriberCount") for item in items}

# Read CSV and extract video IDs
df = pd.read_csv(INPUT_CSV)
df["video_id"] = df["video_link"].apply(extract_video_id)

# Batch process video metadata (50 per API call) with progress bar
video_metadata = {}
for i in tqdm(range(0, len(df), 50), desc="Fetching video metadata"):
    batch_ids = df["video_id"].iloc[i:i+50].dropna().tolist()
    if batch_ids:
        meta = get_video_metadata(batch_ids)
        video_metadata.update(meta)
    time.sleep(0.1)  # avoid hitting rate limits

# Add metadata columns to DataFrame
for vid, meta in video_metadata.items():
    for key, value in meta.items():
        df.loc[df["video_id"] == vid, key] = value

# Get unique channel IDs and batch process subscriber counts (50 per API call) with progress bar
channel_ids = df["channelId"].dropna().unique().tolist()
channel_subs = {}
for i in tqdm(range(0, len(channel_ids), 50), desc="Fetching channel subscribers"):
    batch_ids = channel_ids[i:i+50]
    if batch_ids:
        subs = get_channel_subscribers(batch_ids)
        channel_subs.update(subs)
    time.sleep(0.1)

# Add subscriber count to DataFrame
df["channel_subscribers"] = df["channelId"].map(channel_subs)

# Save to CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved enriched data to {OUTPUT_CSV}")