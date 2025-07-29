# 피부질환 모델 CustomScaleLayer 문제 해결 가이드

## 문제 요약
- **증상**: `CustomScaleLayer`가 예상치 못한 리스트 입력을 받아 TypeError 발생
- **원인**: 모델 변환 과정에서 레이어 구조가 변경되어 입력 형태 불일치
- **영향**: dog_binary, cat_binary 모델의 v2 버전이 작동하지 않음

## 해결 방법

### 옵션 1: 깨끗한 모델로 재변환 (권장)
CustomScaleLayer 없이 표준 구조로 체크포인트를 변환합니다.

```bash
# 1. 체크포인트를 깨끗한 모델로 변환
python convert_checkpoints_clean.py

# 2. 변환된 모델 테스트
python test_fixed_models.py
```

변환 후 생성되는 파일:
- `cat_binary_model_clean.h5`
- `dog_binary_model_clean.h5`
- `dog_multi_136_model_clean.h5`

### 옵션 2: CustomScaleLayer 수정
기존 v2 모델을 유지하면서 CustomScaleLayer를 수정합니다.

```bash
# 1. CustomScaleLayer 수정 및 모델 재저장
python fix_customscalelayer.py

# 2. 수정된 모델 테스트
python test_fixed_models.py
```

### 옵션 3: 원본 모델 사용
문제가 있는 v2 모델 대신 원본 모델을 사용합니다.

## 서비스 파일 업데이트

`services/skin_disease_service.py` 파일의 모델 설정을 다음과 같이 수정:

```python
# 옵션 1: 깨끗한 모델 사용
model_configs = {
    'cat_binary': {
        'model_file': 'cat_binary_model_clean.h5',  # 변경
        'class_map': 'cat_binary_class_map.json',
        'input_size': (224, 224),
        'model_type': 'classification'
    },
    'dog_binary': {
        'model_file': 'dog_binary_model_clean.h5',  # 변경
        'class_map': 'dog_binary_class_map.json',
        'input_size': (224, 224),
        'model_type': 'classification'
    }
}

# 옵션 3: 원본 모델 사용
model_configs = {
    'cat_binary': {
        'model_file': 'cat_binary_model.h5',  # 원본 사용
        'class_map': 'cat_binary_class_map.json',
        'input_size': (224, 224),
        'model_type': 'classification'
    }
}
```

## 검증 체크리스트

1. **모델 로드 확인**
   ```python
   python -c "import tensorflow as tf; model = tf.keras.models.load_model('path/to/model.h5'); print('Success!')"
   ```

2. **예측 변동성 확인**
   - 다양한 입력에 대해 서로 다른 예측값이 나와야 함
   - 모든 입력에 대해 동일한 값이 나오면 문제 있음

3. **파일 크기 확인**
   - 정상적인 InceptionResNetV2 기반 모델: 약 220-230MB
   - 크기가 너무 작으면 가중치가 제대로 로드되지 않은 것

## 문제 해결 순서

1. **현재 상태 확인**
   ```bash
   python inspect_checkpoint_models.py
   ```

2. **깨끗한 모델로 변환**
   ```bash
   python convert_checkpoints_clean.py
   ```

3. **모델 테스트**
   ```bash
   python test_fixed_models.py
   ```

4. **서비스 파일 업데이트**
   - 작동하는 모델 파일명으로 변경

5. **통합 테스트**
   - 백엔드 서버 재시작
   - 실제 이미지로 API 테스트

## 주의사항

- CustomScaleLayer가 있는 모델은 해당 레이어 정의가 반드시 필요
- 모델 변환 시 입력/출력 shape 확인 필수
- 체크포인트 파일(.index, .data-00000-of-00001)이 모두 있어야 변환 가능

## 추가 디버깅

문제가 지속되면:
1. TensorFlow 버전 확인: `tf.__version__`
2. GPU 메모리 문제 확인
3. 체크포인트 무결성 확인