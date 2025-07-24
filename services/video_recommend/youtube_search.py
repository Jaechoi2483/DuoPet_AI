# services/video_recommend/youtube_search.py

import os
import requests
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


def search_youtube(keyword: str, max_results: int = 3) -> List[Dict]:
    if not YOUTUBE_API_KEY:
        print("❌ YouTube API 키가 설정되지 않았습니다.")
        return []

    # 1. search.list - videoId 리스트 추출
    search_url = "https://www.googleapis.com/youtube/v3/search"
    search_params = {
        "part": "snippet",
        "q": keyword,
        "key": YOUTUBE_API_KEY,
        "maxResults": max_results,
        "type": "video",
        "safeSearch": "strict"
    }

    try:
        search_res = requests.get(search_url, params=search_params, timeout=3)
        search_res.raise_for_status()
        search_items = search_res.json().get("items", [])

        video_ids = [item["id"]["videoId"] for item in search_items if "videoId" in item.get("id", {})]
        if not video_ids:
            return []

    except Exception as e:
        print(f"[❌ 검색 요청 실패] {e}")
        return []

    # 2. videos.list - 상세 정보 조회
    videos_url = "https://www.googleapis.com/youtube/v3/videos"
    videos_params = {
        "part": "snippet,contentDetails,statistics",
        "key": YOUTUBE_API_KEY,
        "id": ",".join(video_ids)
    }

    try:
        videos_res = requests.get(videos_url, params=videos_params, timeout=3)
        videos_res.raise_for_status()
        video_items = videos_res.json().get("items", [])

        results = []
        for item in video_items:
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            content = item.get("contentDetails", {})

            results.append({
                "video_id": item.get("id", ""),
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "thumbnail_url": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                "channel_name": snippet.get("channelTitle", ""),
                "duration": content.get("duration", ""),
                "view_count": int(stats.get("viewCount", 0)),
                "published_at": snippet.get("publishedAt", "")
            })

        return results

    except Exception as e:
        print(f"[❌ 상세 요청 실패] {e}")
        return []
