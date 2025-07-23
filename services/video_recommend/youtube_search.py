# services/video_recommend/youtube_search.py

import os
import requests
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

def search_youtube(keyword: str, max_results: int = 5) -> List[Dict]:
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": keyword,
        "key": YOUTUBE_API_KEY,
        "maxResults": max_results,
        "type": "video"
    }

    res = requests.get(url, params=params)
    return [
        {
            "video_id": item["id"]["videoId"],
            "title": item["snippet"]["title"],
            "thumbnail_url": item["snippet"]["thumbnails"]["high"]["url"],
            "channel_name": item["snippet"]["channelTitle"],
            "published_at": item["snippet"]["publishedAt"]
        }
        for item in res.json().get("items", [])
    ]
