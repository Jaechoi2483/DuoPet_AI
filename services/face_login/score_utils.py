# services/face_login/score_utils.py

"""
신뢰도 계산 유틸리티
- cosine distance를 기반으로 신뢰도(%)를 계산하는 함수 제공
- 거리(distance)는 0에 가까울수록 동일인물일 확률이 높다고 간주함
"""

def calculate_confidence(distance: float, threshold: float = 0.3) -> float:
    """
    cosine distance → confidence (%) 변환

    예:
    - distance = 0.0  → confidence = 100%
    - distance = 0.15 → confidence = 50%
    - distance = 0.3  → confidence = 0%
    - distance > 0.3  → 음수 or 0%로 처리 가능

    Args:
        distance (float): DeepFace에서 계산된 cosine 거리
        threshold (float): 기준 임계값 (default: 0.3)

    Returns:
        float: 0.0 ~ 100.0 사이 confidence 값
    """
    if distance >= threshold:
        return 0.0
    return round((1 - (distance / threshold)) * 100, 2)
