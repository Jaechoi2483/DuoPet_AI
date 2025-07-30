# services/video_recommend/youtube_search.py

import os
import requests
from typing import List, Dict
from dotenv import load_dotenv

# .env에서 YOUTUBE_API_KEY 불러오기
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


def search_youtube(keyword: str, max_results: int = 3) -> List[Dict]:
    '''
    주어진 키워드로 유튜브에서 영상을 검색하고, 추천에서 사용할 수 있는 상세 정보를 반환

    1단계 : search.list로 videoId 검색
    2단계: videos.list로 videoId 상세 조회 (제목, 설명, 조회수 등 포함)
    '''

    # API 키 없을 경우 종료
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

        # videoId만 추출
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
        "id": ",".join(video_ids)   # 여러 videoId를 콤마로 연결
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

            # 추천 시스템에서 사용하는 필드
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
