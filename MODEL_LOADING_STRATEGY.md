# DuoPet AI 모델 로딩 전략 및 개선 방안

## 현재 모델 구조 분석

### 1. 안구 질환 진단 모델
- **설정 파일명**: eye_disease_model.h5
- **실제 파일명**: best_grouped_model.keras ⚠️ **불일치**
- **모델 형식**: Keras native format (.keras)
- **문제점**: Keras 3.x normalization layer 호환성 문제

### 2. BCS (체형 평가) 모델  
- **파일명**: bcs_efficientnet_v1.h5
- **모델 형식**: HDF5 format (.h5)
- **특징**: Multi-head EfficientNet B0, 13개 이미지 입력

### 3. 피부 질환 진단 모델
- **모델 형식**: TensorFlow checkpoint files
- **구성**:
  - 분류 모델: cat_binary, dog_binary, dog_multi_136, dog_multi_456
  - 세그멘테이션 모델: cat_A2, dog_A1-A6 (U-Net 구조)

## 모델별 로딩 전략

### 1. 안구 질환 모델 (.keras)

```python
def load_eye_disease_model(model_path):
    """안구 질환 모델 전용 로더"""
    try:
        # 1차 시도: normalization fix 적용
        from utils.model_loader import load_keras_with_normalization_fix
        model = load_keras_with_normalization_fix(
            "models/health_diagnosis/eye_disease/best_grouped_model.keras",
            custom_objects={'swish': tf.nn.swish}
        )
        return model
    except Exception as e:
        logger.error(f"Eye disease model loading failed: {e}")
        # 폴백: 더미 모델 반환
        return create_dummy_eye_model()
```

### 2. BCS 모델 (.h5)

```python
def load_bcs_model(model_path):
    """BCS 모델 전용 로더"""
    try:
        # H5 파일은 legacy 로더 사용
        os.environ['TF_USE_LEGACY_KERAS'] = '1'
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'swish': tf.nn.swish},
            compile=False
        )
        # 수동 컴파일
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        logger.error(f"BCS model loading failed: {e}")
        return None
```

### 3. 피부 질환 모델 (checkpoint)

```python
def load_skin_disease_models(models_dir):
    """피부 질환 모델 로더 (체크포인트 기반)"""
    models = {}
    
    # 분류 모델 로딩
    classification_models = {
        'cat_binary': create_binary_classification_model(),
        'dog_binary': create_binary_classification_model(),
        'dog_multi_136': create_multi_classification_model(3),
        'dog_multi_456': create_multi_classification_model(3),
    }
    
    # 체크포인트에서 가중치 로드
    for model_name, model in classification_models.items():
        checkpoint_path = f"{models_dir}/classification/{model_name}/checkpoint"
        if os.path.exists(checkpoint_path):
            try:
                model.load_weights(checkpoint_path)
                models[model_name] = model
                logger.info(f"Loaded {model_name} from checkpoint")
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
    
    # 세그멘테이션 모델은 별도 처리
    # ...
    
    return models
```

## 개선 방안

### 단기 개선 (즉시 적용 가능)

1. **설정 파일 수정**
   - eye_disease/config.yaml의 model_file을 "best_grouped_model.keras"로 수정

2. **모델별 맞춤형 로더 구현**
   ```python
   # services/health_diagnosis/model_manager.py 개선
   class HealthDiagnosisModelManager:
       def __init__(self):
           self.loaders = {
               'eye_disease': self._load_eye_disease_model,
               'bcs': self._load_bcs_model,
               'skin_disease': self._load_skin_disease_models
           }
           self.models = {}
       
       def load_all_models(self):
           for model_type, loader in self.loaders.items():
               try:
                   self.models[model_type] = loader()
                   logger.info(f"Successfully loaded {model_type}")
               except Exception as e:
                   logger.error(f"Failed to load {model_type}: {e}")
                   self.models[model_type] = self._get_fallback_model(model_type)
   ```

3. **에러 처리 강화**
   - 모델 로딩 실패 시 더미 모델로 폴백
   - 각 모델 독립적으로 로딩 (하나 실패해도 다른 모델 사용 가능)

4. **로딩 최적화**
   - 모델 캐싱 구현
   - 지연 로딩 (lazy loading) 적용

### 중기 개선 (1-2주)

1. **모델 재학습/변환**
   - 모든 모델을 TensorFlow 2.x SavedModel 형식으로 통일
   - Normalization layer 문제 해결을 위한 재학습

2. **모델 버전 관리**
   - 모델 버전 트래킹 시스템 구현
   - A/B 테스트를 위한 다중 버전 지원

3. **성능 모니터링**
   - 모델별 추론 시간 측정
   - 메모리 사용량 모니터링

### 장기 개선 (1개월+)

1. **모델 서빙 아키텍처 개선**
   - TensorFlow Serving 도입 검토
   - 모델 배포 파이프라인 구축

2. **엣지 배포 최적화**
   - TensorFlow Lite 변환
   - 모델 양자화 적용

## 즉시 적용 가능한 해결책

`api/routers/health_diagnosis_router.py` 수정:

```python
# 모델 로딩 부분을 try-except로 감싸고 폴백 처리
try:
    model_manager = HealthDiagnosisModelManager()
    await model_manager.initialize()
except Exception as e:
    logger.error(f"Model initialization failed: {e}")
    # 더미 모델 매니저 사용
    model_manager = DummyModelManager()

# 각 엔드포인트에서 모델 사용 시 안전하게 처리
@router.post("/analyze/eye")
async def analyze_eye_disease(image: UploadFile = File(...)):
    if not model_manager.is_model_available('eye_disease'):
        return create_error_response(
            error_code=ErrorCode.MODEL_NOT_LOADED,
            message="Eye disease model is temporarily unavailable"
        )
    # ... 정상 처리
```

## 테스트 계획

1. **단위 테스트**
   - 각 모델 로더 개별 테스트
   - 폴백 메커니즘 테스트

2. **통합 테스트**
   - 전체 모델 로딩 프로세스
   - API 엔드포인트 테스트

3. **부하 테스트**
   - 동시 요청 처리
   - 메모리 누수 확인