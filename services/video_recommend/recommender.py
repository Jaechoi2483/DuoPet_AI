# services/video_recommend/recommender.py

from sqlalchemy.orm import Session
from services.video_recommend.db_models.content_entity import ContentEntity
from services.video_recommend.keyword_extractor import extract_keywords
from services.video_recommend.youtube_search import search_youtube
from typing import List, Dict


def is_pet_related(video: Dict, pet_keywords: List[str]) -> bool:
    """
    영상 제목, 설명, 채널명에 반려동물 관련 키워드 포함 여부 판단
    """
    combined = f"{video.get('title', '')} {video.get('description', '')} {video.get('channel_name', '')}".lower()
    return any(kw in combined for kw in pet_keywords)


def recommend_youtube_videos_from_db_tags(
    content_id: int,
    db: Session,
    max_results: int = 1
) -> List[Dict]:
    """
    게시물 content_id를 기반으로 태그에서 키워드를 추출하고,
    반려동물 관련 YouTube 영상을 최대 max_results만큼 추천합니다.
    """
    # 1. 게시글 조회
    content = db.query(ContentEntity).filter(ContentEntity.content_id == content_id).first()
    if not content or not content.tags:
        return []

    # 2. 키워드 추출 (최대 3개)
    keywords = extract_keywords(content.tags)[:3]
    if not keywords:
        return []

    # 3. 검색 키워드 조합
    pet_categories = ["강아지", "고양이", "반려동물", "애완동물"]
    search_keywords = list(set(f"{pet} {kw}" for pet in pet_categories for kw in keywords))
    pet_keywords = ["강아지", "고양이", "반려동물", "애완동물", "펫"]

    # 4. 필터 통과 영상 수집
    seen_ids = set()
    all_results = []

    for keyword in search_keywords:
        if len(all_results) >= max_results:
            break

        videos = search_youtube(keyword, max_results)
        if not isinstance(videos, list) or not videos:
            continue

        for video in videos:
            if not video or video["video_id"] in seen_ids:
                continue

            seen_ids.add(video["video_id"])

            if is_pet_related(video, pet_keywords):
                all_results.append(video)
                print(f"[🎯 필터 통과] 키워드: {keyword} → {video['title']}")

                if len(all_results) >= max_results:
                    return all_results

    # 5. fallback: 연관성은 낮지만 아직 max_results 미만이면 영상 보완 추천
    for keyword in search_keywords:
        if len(all_results) >= max_results:
            break

        videos = search_youtube(keyword, max_results)
        if not isinstance(videos, list) or not videos:
            continue

        for video in videos:
            if video["video_id"] in seen_ids:
                continue

            seen_ids.add(video["video_id"])
            all_results.append(video)
            print(f"[⚠️ 대체 추천] 키워드: {keyword} → {video['title']}")

            if len(all_results) >= max_results:
                break

    return all_results
