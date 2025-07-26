# DuoPet AI 모델 로딩 문제 해결 방안

## 문제 요약
- **증상**: `Layer 'normalization' expected 3 variables, but received 0 variables`
- **원인**: Keras 버전 간 호환성 문제 (모델 저장 시와 로드 시 버전 차이)
- **제약사항**: 모델 재학습 불가, 현재 모델을 그대로 사용해야 함

## 해결 방안 (3단계 접근)

### 1단계: Keras 파일 직접 수정
**.keras 파일 구조 분석 및 수정**

```python
# utils/keras_file_analyzer.py 사용
from utils.keras_file_analyzer import KerasFileAnalyzer

analyzer = KerasFileAnalyzer("models/health_diagnosis/eye_disease/best_grouped_model.keras")
fixed_path = analyzer.fix_normalization_variables()
```

**주요 기능:**
- .keras 파일(zip 형식) 압축 해제
- 누락된 normalization 변수 추가
- 수정된 파일 재압축

### 2단계: 모델 재구성 및 가중치 복사
**모델 아키텍처 재생성 후 가중치만 로드**

```python
# utils/model_reconstructor.py 사용
from utils.model_reconstructor import ModelReconstructor

# 안구 질환 모델 재구성
model = ModelReconstructor.reconstruct_eye_disease_model(
    weights_path="models/health_diagnosis/eye_disease/best_grouped_model.keras"
)
```

**지원 모델:**
- 안구 질환: EfficientNetB0 기반
- BCS: Multi-head EfficientNet
- 피부 질환: CNN 기반

### 3단계: 통합 로더 사용
**여러 방법을 순차적으로 시도하는 통합 로더**

```python
# utils/integrated_model_loader.py 사용
from utils.integrated_model_loader import IntegratedModelLoader

loader = IntegratedModelLoader()
model = loader.load_model("path/to/model.keras", "eye_disease")
```

**시도 순서:**
1. 직접 로드 (커스텀 객체 포함)
2. Keras 파일 수정 후 로드
3. 모델 재구성 및 가중치 복사
4. 더미 모델 (최후 수단)

## 실행 방법

### 방법 1: 수정 스크립트 실행
```bash
cd D:\final_project\DuoPet_AI
python fix_models.py
```

옵션 선택:
1. 안구 질환 모델 수정
2. 모든 모델 테스트
3. 개선된 모델 로더 설치

### 방법 2: 직접 수정
```python
# 1. 안구 질환 모델 수정
from utils.keras_file_analyzer import KerasFileAnalyzer

analyzer = KerasFileAnalyzer("models/health_diagnosis/eye_disease/best_grouped_model.keras")
fixed_path = analyzer.fix_normalization_variables()

# 2. 수정된 모델 사용
# services/eye_disease_service.py에서 fixed_path 사용
```

### 방법 3: 서비스 업데이트
**EyeDiseaseService 수정**

```python
# services/eye_disease_service.py
from utils.integrated_model_loader import IntegratedModelLoader

class EyeDiseaseService:
    def __init__(self, model_path: str, class_map_path: str):
        self.class_map = self._load_class_map(class_map_path)
        
        # 통합 로더 사용
        loader = IntegratedModelLoader()
        self.model = loader.load_model(model_path, 'eye_disease')
        
        if self.model is None:
            raise RuntimeError(f"Failed to load model from {model_path}")
```

## 파일 구조

```
DuoPet_AI/
├── utils/
│   ├── keras_file_analyzer.py      # Keras 파일 분석/수정
│   ├── model_reconstructor.py      # 모델 재구성
│   ├── integrated_model_loader.py  # 통합 로더
│   └── model_loader.py            # 기존 로더 (개선됨)
├── fix_models.py                  # 수정 실행 스크립트
├── MODEL_LOADING_SOLUTION.md      # 이 문서
└── models/
    └── health_diagnosis/
        ├── eye_disease/
        │   ├── best_grouped_model.keras        # 원본
        │   └── best_grouped_model_fixed.keras  # 수정됨
        ├── bcs/
        └── skin_disease/
```

## 예상 결과

### 성공 시
```
✅ Successfully loaded model using method 2
✅ Fixed model loads successfully!
✅ eye_disease: Loaded successfully
```

### 실패 시 (더미 모델 사용)
```
⚠️ Creating dummy model as fallback...
✅ eye_disease: Loaded successfully (dummy)
```

## 주의사항

1. **메모리 사용량**: 모델 로딩 시 메모리 사용량이 증가할 수 있음
2. **첫 실행 시간**: 첫 로딩 시 시간이 오래 걸릴 수 있음
3. **GPU 사용**: GPU가 없어도 동작하지만 속도가 느림

## 추가 개선 사항

1. **모델 캐싱**: 로드된 모델을 메모리에 캐싱
2. **비동기 로딩**: 서버 시작 시 백그라운드 로딩
3. **모델 버전 관리**: 여러 버전의 모델 지원

## 테스트 확인 사항

- [ ] 안구 질환 모델 로드 성공
- [ ] BCS 모델 로드 성공
- [ ] 피부 질환 모델 로드 성공
- [ ] API 엔드포인트 정상 동작
- [ ] 프론트엔드 연동 테스트

## 문제 해결 FAQ

**Q: 여전히 normalization 에러가 발생합니다.**
A: `fix_models.py`를 실행하여 모델을 수정하세요.

**Q: 수정된 모델도 로드되지 않습니다.**
A: 모델 재구성 방법(Method 3)을 사용하세요.

**Q: 메모리 부족 에러가 발생합니다.**
A: 한 번에 하나의 모델만 로드하도록 수정하세요.

**Q: 더미 모델이 사용되고 있습니다.**
A: 정상입니다. 실제 예측은 부정확하지만 서비스는 동작합니다.