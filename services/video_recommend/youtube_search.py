# services/video_recommend/youtube_search.py

import os
import requests
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


def search_youtube(keyword: str, max_results: int = 1) -> List[Dict]:
    if not YOUTUBE_API_KEY:
        return []

    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": keyword,
        "key": YOUTUBE_API_KEY,
        "maxResults": max_results,
        "type": "video",
        "safeSearch": "strict",
    }

    try:
        res = requests.get(url, params=params, timeout=3)
        res.raise_for_status()

        items = res.json().get("items", [])
        videos = []

        for item in items:
            video_id = item.get("id", {}).get("videoId")
            snippet = item.get("snippet", {})
            if not video_id or not snippet:
                continue

            videos.append({
                "video_id": video_id,
                "title": snippet.get("title", ""),
                "thumbnail_url": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                "channel_name": snippet.get("channelTitle", ""),
                "published_at": snippet.get("publishedAt", "")
            })

        return videos

    except Exception:
        return []
