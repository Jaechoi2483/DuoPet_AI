# services/video_recommend/recommender.py

from sqlalchemy.orm import Session
from services.video_recommend.db_models.content_entity import ContentEntity
from services.video_recommend.keyword_extractor import extract_keywords
from services.video_recommend.youtube_search import search_youtube
from typing import List, Dict, Optional


# 간단한 in-memory 캐시 (운영 시에는 Redis 권장)
video_cache: Dict[str, List[Dict]] = {}

def tag_score(video: Dict, tag_keywords: List[str]) -> int:
    text = f"{video.get('title', '')} {video.get('description', '')} {video.get('channel_name', '')}".lower()
    return sum(1 for tag in tag_keywords if tag in text)

def is_highly_relevant(video: Dict, tag_keywords: List[str], required_pet: Optional[str] = None, min_score: int = 2) -> bool:
    text = f"{video.get('title', '')} {video.get('description', '')} {video.get('channel_name', '')}".lower()

    # 1. 필수 동물 키워드 필터링
    if required_pet and required_pet.lower() not in text:
        return False

    # 2. 태그 점수 기준
    return tag_score(video, tag_keywords) >= min_score

def recommend_youtube_videos_from_db_tags(
        content_id: int,
        db: Session,
        max_results: int = 3
) -> List[Dict]:
    print(f"[📥 요청 시작] content_id = {content_id}")

    # 캐시 확인
    if str(content_id) in video_cache:
        print("✅ 캐시 사용")
        return video_cache[str(content_id)][:max_results]

    content = db.query(ContentEntity).filter(ContentEntity.content_id == content_id).first()
    if not content or not content.tags:
        print("[❌ 종료] content 또는 tags 없음")
        return []

    print(f"✅ content.tags → {content.tags}")

    # 키워드 추출
    keywords = extract_keywords(content.tags, top_n=6)
    if not keywords:
        print("[❌ 종료] 키워드 추출 실패")
        return []

    print(f"🧠 추출된 keywords → {keywords}")

    # 검색 키워드 제한
    search_keywords = keywords[:6]
    tag_keywords = [tag.strip().lower() for tag in content.tags.replace(",", " ").split() if tag.strip()]

    # 태그 기준 필수 반려동물 지정
    if "고양이" in tag_keywords:
        required_pet = "고양이"
    elif "강아지" in tag_keywords:
        required_pet = "강아지"
    else:
        required_pet = None

    seen_ids = set()
    final_results = []

    for keyword in search_keywords:
        if len(final_results) >= max_results:
            break

        videos = search_youtube(keyword, max_results=3)
        print(f"🔍 '{keyword}' 검색 결과: {len(videos)}개")

        for video in videos:
            if video["video_id"] in seen_ids:
                continue
            seen_ids.add(video["video_id"])

            if is_highly_relevant(video, tag_keywords, required_pet):
                final_results.append(video)
                print(f"[🎯 정확도 통과] {video['title']} ({video['video_id']})")

                if len(final_results) >= max_results:
                    break

    # fallback (정확도 미달 영상 포함)
    if len(final_results) < max_results:
        print("⚠️ 정확도 통과 부족 → fallback 적용")
        for keyword in search_keywords:
            if len(final_results) >= max_results:
                break
            videos = search_youtube(keyword, max_results=2)
            for video in videos:
                if video["video_id"] in seen_ids:
                    continue
                seen_ids.add(video["video_id"])
                final_results.append(video)
                print(f"[⚠️ fallback 추천] {video['title']}")

                if len(final_results) >= max_results:
                    break

    # 캐시 저장
    video_cache[str(content_id)] = final_results

    print(f"🎬 최종 추천 영상 수: {len(final_results)}개")
    return final_results[:max_results]


