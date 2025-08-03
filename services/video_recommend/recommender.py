# services/video_recommend/recommender.py

from sqlalchemy.orm import Session
from services.video_recommend.db_models.board_entity import BoardEntity
from services.video_recommend.keyword_extractor import extract_keywords
from services.video_recommend.youtube_search import search_youtube
from typing import List, Dict, Optional

# 영상 추천 결과 캐시 (content_id 기준)
video_cache: Dict[str, List[Dict]] = {}

# 영상의 태그 일치 점수를 계산
# 영상의 제목,설명,채널명에 태그 키워드가 몇 개 포함되는지 계산
def tag_score(video: Dict, tag_keywords: List[str]) -> int:
    text = f"{video.get('title', '')} {video.get('description', '')} {video.get('channel_name', '')}".lower()
    return sum(1 for tag in tag_keywords if tag in text)

# 특정 태그점수와 필수 반려동물 키워드를 기반으로 영상의 관련도 평가
# 영상이 필수 반려동물 키워드를 포함하고, 태그 점수가 기준 이상인지 판단
def is_highly_relevant(video: Dict, tag_keywords: List[str], required_pet: Optional[str] = None, min_score: int = 2) -> bool:
    text = f"{video.get('title', '')} {video.get('description', '')} {video.get('channel_name', '')}".lower()
    if required_pet and required_pet.lower() not in text:
        return False
    return tag_score(video, tag_keywords) >= min_score

# 게시글의 태그를 기반으로 유튜브 영상 추천
def recommend_youtube_videos_from_db_tags(
        content_id: int,
        db: Session,
        max_results: int = 3
) -> List[Dict]:
    print(f"[요청 시작] content_id = {content_id}")

    # 캐시가 존재하면 캐시 사용
    if str(content_id) in video_cache:
        print("캐시 사용")
        return video_cache[str(content_id)][:max_results]

    # 게시글 정보 조회
    content = db.query(BoardEntity).filter(BoardEntity.content_id == content_id).first()
    if not content or not content.tags:
        print("[종료] content 또는 tags 없음")
        return []

    print(f"content.tags → {content.tags}")

    # 태그 전처리
    tag_keywords = [tag.strip().lower() for tag in content.tags.replace(",", " ").split() if tag.strip()]
    category = getattr(content, "category", "").lower()
    print(f"content.category → {category}")

    # 자유게시판 + 고양이/강아지 없음 → 태그 기반 추천
    if category == "free" and not any(pet in tag_keywords for pet in ["고양이", "강아지"]):
        keywords = tag_keywords[:6]
        required_pet = None
        print("자유게시판 - 일반 태그 기반 추천 모드")
    else:
        # 후기/질문/팁은 KeyBERT 키워드 추출 + 반려동물 필터 적용
        keywords = extract_keywords(content.tags, top_n=6)
        required_pet = None
        if "고양이" in tag_keywords:
            required_pet = "고양이"
        elif "강아지" in tag_keywords:
            required_pet = "강아지"
        print(f"고양이/강아지 필터 적용 모드 → required_pet = {required_pet}")

    if not keywords:
        print("[종료] 키워드 추출 실패")
        return []

    print(f"선택된 search_keywords → {keywords}")

    seen_ids = set()
    final_results = []

    # 정확도 기준 충족 영상 우선
    for keyword in keywords:
        if len(final_results) >= max_results:
            break

        videos = search_youtube(keyword, max_results=3)
        print(f"'{keyword}' 검색 결과: {len(videos)}개")

        for video in videos:
            if video["video_id"] in seen_ids:
                continue
            seen_ids.add(video["video_id"])

            if is_highly_relevant(video, tag_keywords, required_pet):
                final_results.append(video)
                print(f"[정확도 통과] {video['title']} ({video['video_id']})")

                if len(final_results) >= max_results:
                    break

    # fallback 추천: 정확도 기준 부족 시 단순 인기순 추가
    if len(final_results) < max_results:
        print("⚠정확도 통과 부족 → fallback 적용")
        for keyword in keywords:
            if len(final_results) >= max_results:
                break
            videos = search_youtube(keyword, max_results=2)
            for video in videos:
                if video["video_id"] in seen_ids:
                    continue
                seen_ids.add(video["video_id"])
                final_results.append(video)
                print(f"[⚠fallback 추천] {video['title']}")
                if len(final_results) >= max_results:
                    break

    # 캐시에 저장
    video_cache[str(content_id)] = final_results
    print(f"최종 추천 영상 수: {len(final_results)}개")
    return final_results[:max_results]



