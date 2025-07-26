# DuoPet AI Health Diagnosis Integration Plan

## Overview

이 문서는 DuoPet AI 프로젝트의 통합 건강 진단 시스템 구현 계획을 설명합니다. 안구질환 탐지, BCS(Body Condition Score), 피부질환 진단 모델을 통합하여 종합적인 펫 건강 진단 서비스를 제공합니다.

## 현재 상태 분석

### 구현된 컴포넌트
1. **안구질환 진단 서비스** (EyeDiseaseService)
   - 모델: `best_grouped_model.keras`
   - 엔드포인트: `/analyze/eye`
   - 상태: ✅ 구현 완료

### 미구현 컴포넌트
1. **BCS 서비스**
   - 모델: `bcs_efficientnet_v1.h5`
   - 특징: 13개 이미지 입력, 3개 클래스 분류
   - 상태: ❌ 서비스 미구현

2. **피부질환 서비스**
   - Classification 모델: 개/고양이별 Binary + Multi-class
   - Segmentation 모델: 병변 영역 검출
   - 상태: ❌ 서비스 미구현

3. **통합 진단 시스템**
   - 엔드포인트: `/analyze`
   - 상태: ❌ 미구현

## 시스템 아키텍처

### 1. Model Registry Pattern
```
DuoPet_AI/
├── services/
│   ├── model_registry.py          # 모델 레지스트리
│   ├── model_adapters/           # 모델 어댑터들
│   │   ├── __init__.py
│   │   ├── base_adapter.py       # 베이스 어댑터 클래스
│   │   ├── eye_disease_adapter.py
│   │   ├── bcs_adapter.py
│   │   └── skin_disease_adapter.py
│   ├── orchestrator.py           # 통합 진단 오케스트레이터
│   ├── eye_disease_service.py    # 기존 서비스 (유지)
│   ├── bcs_service.py           # 새로 구현
│   └── skin_disease_service.py   # 새로 구현
```

### 2. 데이터 플로우
```
사용자 요청 → API Router → Orchestrator
                              ↓
                    ┌─────────┴─────────┬──────────┐
                    ↓                   ↓          ↓
            Eye Disease Service  BCS Service  Skin Disease Service
                    ↓                   ↓          ↓
                    └─────────┬─────────┴──────────┘
                              ↓
                      Result Combiner
                              ↓
                      통합 진단 결과
```

## 구현 계획

### Phase 1: 기반 구조 구축 (Priority: High)

#### 1.1 Model Registry 구현
- 모든 모델을 중앙에서 관리
- 버전 관리 및 캐싱 지원
- Framework 독립적 인터페이스

#### 1.2 Base Adapter Pattern 구현
- 모든 모델에 대한 표준화된 인터페이스
- preprocess → predict → postprocess 파이프라인

### Phase 2: 개별 서비스 구현 (Priority: High)

#### 2.1 BCS Service 구현
```python
class BCSService:
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
        
    def diagnose(self, images: List[UploadFile]) -> Dict[str, Any]:
        # 13개 이미지 처리
        # BCS 점수 계산 (1-9)
        # 상태 분류 (저체중/정상/과체중)
```

#### 2.2 Skin Disease Service 구현
```python
class SkinDiseaseService:
    def __init__(self, classification_models: Dict, segmentation_models: Dict):
        self.classification_models = classification_models
        self.segmentation_models = segmentation_models
        
    def diagnose(self, image: UploadFile, pet_type: str) -> Dict[str, Any]:
        # 1. Binary classification (정상/질환)
        # 2. Multi-class classification (질환 종류)
        # 3. Segmentation (병변 영역)
```

### Phase 3: 통합 시스템 구현 (Priority: Medium)

#### 3.1 Health Diagnosis Orchestrator
```python
class HealthDiagnosisOrchestrator:
    def __init__(self, model_registry: ModelRegistry):
        self.eye_service = EyeDiseaseService(...)
        self.bcs_service = BCSService(...)
        self.skin_service = SkinDiseaseService(...)
        
    async def diagnose(self, images: List[UploadFile], pet_type: str) -> Dict:
        # 병렬 처리
        results = await asyncio.gather(
            self._run_eye_diagnosis(images),
            self._run_bcs_analysis(images),
            self._run_skin_diagnosis(images, pet_type)
        )
        
        # 결과 통합
        return self._combine_results(*results)
```

#### 3.2 Result Combiner
- 각 진단 결과의 가중치 계산
- 종합 건강 점수 산출
- 우선순위 기반 권고사항 생성

### Phase 4: API 엔드포인트 구현 (Priority: Medium)

#### 4.1 통합 진단 엔드포인트
```python
@router.post("/analyze", response_model=StandardResponse)
async def analyze_health(
    images: List[UploadFile] = File(...),
    pet_type: str = Form(...),
    pet_age: Optional[int] = Form(None),
    orchestrator: HealthDiagnosisOrchestrator = Depends(get_orchestrator)
):
    result = await orchestrator.diagnose(images, pet_type)
    return create_success_response(data=result)
```

#### 4.2 개별 진단 엔드포인트 업데이트
- `/analyze/bcs` - BCS 전용 진단
- `/analyze/skin` - 피부질환 전용 진단

## 기술적 고려사항

### 1. 성능 최적화
- **Model Caching**: 자주 사용되는 모델은 메모리에 유지
- **Batch Processing**: 다중 이미지 처리 시 배치 처리
- **Async/Await**: 비동기 처리로 응답 시간 단축

### 2. 에러 처리
- 모델별 fallback 전략
- Graceful degradation (일부 모델 실패 시에도 나머지 결과 제공)
- 상세한 에러 로깅

### 3. 모니터링
- 모델별 성능 메트릭 추적
- 진단 정확도 피드백 수집
- API 응답 시간 모니터링

## 테스트 계획

### 1. 단위 테스트
- 각 서비스별 독립적 테스트
- 모델 어댑터 테스트
- 결과 통합 로직 테스트

### 2. 통합 테스트
- 전체 진단 플로우 테스트
- 다양한 입력 케이스 테스트
- 에러 시나리오 테스트

### 3. 성능 테스트
- 동시 요청 처리 테스트
- 메모리 사용량 모니터링
- 응답 시간 벤치마킹

## 예상 결과 형식

```json
{
  "success": true,
  "data": {
    "overall_health_score": 85,
    "eye_health": {
      "status": "healthy",
      "confidence": 0.95,
      "details": {}
    },
    "body_condition": {
      "bcs_score": 5,
      "status": "정상",
      "confidence": 0.88
    },
    "skin_health": {
      "status": "mild_dermatitis",
      "affected_area": 12.5,
      "confidence": 0.76,
      "segmentation_available": true
    },
    "recommendations": [
      "정기적인 건강 검진을 권장합니다",
      "피부 상태를 주의 깊게 관찰해주세요"
    ],
    "requires_vet_visit": false
  }
}
```

## 타임라인

1. **Week 1**: Model Registry 및 Base Adapter 구현
2. **Week 2**: BCS Service 구현
3. **Week 3**: Skin Disease Service 구현
4. **Week 4**: Orchestrator 및 통합 시스템 구현
5. **Week 5**: API 엔드포인트 구현 및 테스트
6. **Week 6**: 성능 최적화 및 문서화

## 다음 단계

1. Model Registry 구현 시작
2. BCS 모델 분석 및 서비스 설계
3. 피부질환 모델 체크포인트 로딩 방법 연구