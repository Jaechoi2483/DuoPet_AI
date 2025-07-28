# DuoPet AI 진단 서비스 사용 가이드

## 서비스 개요

DuoPet AI는 세 가지 주요 진단 서비스를 제공합니다:

1. **안구 질환 진단** - 단일 이미지로 안구 질환 검사
2. **피부 질환 진단** - 단일 이미지로 피부 상태 검사
3. **BCS (Body Condition Score) 평가** - 13개 이미지로 체형 평가

## 권장 사용 방법

### 1. 안구/피부 질환 진단 (단일 이미지)

**엔드포인트:**
- `/api/v1/health-diagnose/analyze/eye` - 안구 질환만
- `/api/v1/health-diagnose/analyze/skin` - 피부 질환만
- `/api/v1/health-diagnose/analyze` - 종합 진단 (안구+피부)

**사용 예시:**
```javascript
// 안구 질환 진단
const formData = new FormData();
formData.append('image', file);

fetch('/api/v1/health-diagnose/analyze/eye', {
  method: 'POST',
  body: formData
});

// 종합 진단 (안구+피부)
const formData = new FormData();
formData.append('images', file);
formData.append('diagnosis_types', 'eye,skin');  // BCS 제외
formData.append('pet_type', 'dog');

fetch('/api/v1/health-diagnose/analyze', {
  method: 'POST',
  body: formData
});
```

### 2. BCS 체형 평가 (다중 이미지)

**엔드포인트:** `/api/v1/health-diagnose/analyze/bcs`

**필요한 이미지 (13개):**
1. 정면 (front)
2. 후면 (back)
3. 좌측면 (left_side)
4. 우측면 (right_side)
5. 위에서 본 모습 (top)
6. 좌전방 (front_left)
7. 우전방 (front_right)
8. 좌후방 (back_left)
9. 우후방 (back_right)
10. 복부 (abdomen)
11. 가슴 (chest)
12. 척추 (spine)
13. 전체 모습 (overall)

**사용 예시:**
```javascript
const formData = new FormData();
// 13개 이미지 추가
images.forEach(image => {
  formData.append('images', image);
});
formData.append('pet_type', 'dog');
formData.append('pet_age', '5');
formData.append('pet_weight', '15');

fetch('/api/v1/health-diagnose/analyze/bcs', {
  method: 'POST',
  body: formData
});
```

### 3. 촬영 가이드 제공

**엔드포인트:** `/api/v1/health-diagnose/bcs/guide`

BCS 평가를 위한 사진 촬영 가이드를 제공합니다.

## 프론트엔드 구현 제안

### UI 분리
```
┌─────────────────────────────────────┐
│         반려동물 건강 진단           │
├─────────────────────────────────────┤
│                                     │
│  ┌─────────┐     ┌─────────┐      │
│  │ 빠른진단 │     │ 체형평가 │      │
│  └─────────┘     └─────────┘      │
│                                     │
│  빠른진단: 안구/피부 (사진 1장)      │
│  체형평가: BCS 측정 (사진 13장)      │
│                                     │
└─────────────────────────────────────┘
```

### 빠른 진단 모드
- 사용자가 한 장의 사진만 업로드
- 안구와 피부 질환을 동시에 검사
- 1-2초 내 결과 제공

### 체형 평가 모드
- 단계별 가이드 제공
- 각 단계마다 필요한 각도 설명
- 13장 모두 촬영 후 종합 평가

## 주의사항

1. **이미지 형식**: JPG, PNG 형식 지원
2. **이미지 크기**: 최대 10MB per image
3. **BCS 정확도**: 13장 모두 제공 시 가장 정확함
4. **응답 시간**: 
   - 단일 진단: 1-2초
   - BCS 평가: 3-5초

## API 응답 예시

### 안구/피부 진단 응답
```json
{
  "status": "success",
  "data": {
    "eye_health": {
      "disease": "정상",
      "confidence": 0.95
    },
    "skin_health": {
      "has_disease": false,
      "confidence": 0.88
    }
  }
}
```

### BCS 평가 응답
```json
{
  "status": "success",
  "data": {
    "bcs_category": "정상",
    "bcs_score": 5,
    "confidence": 0.92,
    "health_insights": [
      "반려동물이 이상적인 체중 범위에 있습니다"
    ],
    "recommendations": [
      "현재 식단과 운동량을 유지하세요"
    ]
  }
}
```