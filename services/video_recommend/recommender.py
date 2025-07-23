# services/video_recommend/recommender.py

from sqlalchemy.orm import Session
from services.video_recommend.models.content_entity import ContentEntity
from services.video_recommend.keyword_extractor import extract_keywords
from services.video_recommend.youtube_search import search_youtube
from typing import List, Dict


def recommend_youtube_videos_from_db_tags(
        content_id: int,
        db: Session,
        max_results: int = 5
) -> List[Dict]:
    """
    게시물 content_id를 기반으로 tags를 추출하고,
    YouTube 영상 추천 리스트를 반환합니다.
    """

    # 1. 게시물 조회
    content = db.query(ContentEntity).filter(ContentEntity.content_id == content_id).first()
    if not content:
        return []

    # 2. 태그가 없으면 추천 불가
    if not content.tags:
        return []

    # 3. KeyBERT로 키워드 추출
    keywords = extract_keywords(content.tags)
    if not keywords:
        return []

    # 4. 키워드 기반 YouTube 영상 추천
    all_results = []
    seen_ids = set()

    for keyword in keywords:
        videos = search_youtube(keyword, max_results)
        for video in videos:
            if video["video_id"] not in seen_ids:
                seen_ids.add(video["video_id"])
                all_results.append(video)

    return all_results[:max_results]
