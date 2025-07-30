# services/video_recommend/keyword_extractor.py

from typing import List
from keybert import KeyBERT

kw_model = KeyBERT()

def extract_keywords(text: str, top_n: int = 6) -> List[str]:
    """
    정보형 키워드 추출기 (정확도 우선)
    - 반려동물 + 증상/주의/첫날 등 정보 전달형 키워드 조합
    - 단일 태그, KeyBERT 보조 키워드 혼합
    - 중복 제거 후 top_n 개 반환
    """

    # 1. 텍스트 내 쉼표(,) 또는 공백 기준으로 태그 분리
    base_tags = [tag.strip() for tag in text.replace(",", " ").split() if tag.strip()]
    keywords = []

    # 2. 고양이, 강아지 태그 분리 (우선 처리)
    pet_tags = [tag for tag in base_tags if tag in ["고양이", "강아지"]]
    other_tags = [tag for tag in base_tags if tag not in pet_tags]

    # 3. 정보형 키워드 조합 (ex: "강아지 주의사항", "고양이 구토")
    info_suffixes = ["구토", "이유", "주의사항", "증상", "첫날", "대처법", "브이로그"]
    for pet in pet_tags:
        for word in other_tags + info_suffixes:
            keywords.append(f"{pet} {word}")

    # 4. 원본 태그도 단독 키워드로 포함
    keywords += base_tags

    # 5. KeyBERT 키워드 추출 (후순위)
    keybert_keywords = [kw for kw, _ in kw_model.extract_keywords(text, top_n=5)]
    keywords += keybert_keywords

    # 6. 중복 제거 + 순서 유지 + top_n 제한
    seen = set()
    result = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            result.append(kw)
        if len(result) >= top_n:
            break

    return result
