# 피부질환 모델 분석 및 통합 방안

## 모델 구조 분석

### Classification 모델
1. **Binary Classification (정상/질환 구분)**
   - `cat_binary`: 고양이 정상/질환 분류
   - `dog_binary`: 개 정상/질환 분류

2. **Multi-class Classification (질환 종류 분류)**
   - `dog_multi_136`: A1(구진/플라크), A3(비듬/각질), A6(태선화/과다색소침착)
   - `dog_multi_456`: A4(농포/여드름), A5(미란/궤양), A6(태선화/과다색소침착)

### Segmentation 모델
- `cat_A2`: 고양이 A2 병변 세그멘테이션
- `dog_A1` ~ `dog_A6`: 개 질환별 병변 영역 세그멘테이션

## 진단 플로우

```
입력 이미지
    ↓
Pet Type 확인 (개/고양이)
    ↓
Binary Classification (정상/질환)
    ↓
[질환인 경우]
    ├─ Multi-class Classification (질환 종류)
    └─ Segmentation (병변 영역)
    ↓
결과 통합
```

## 구현 전략

### 1. 모델 아키텍처 정의 문제
- TensorFlow checkpoint 파일만 있고 모델 아키텍처 정의가 없음
- 해결 방안:
  1. 모델 구조를 추정하여 재구성
  2. 또는 SavedModel 형식으로 변환된 모델 요청

### 2. Two-Stage Approach
1. **Stage 1**: Binary classification으로 질환 여부 판단
2. **Stage 2**: 질환이 감지되면 multi-class + segmentation 수행

### 3. 결과 통합
- Classification 결과와 Segmentation 결과를 결합
- 병변 영역 크기를 기반으로 심각도 평가
- 질환 종류에 따른 맞춤형 권고사항 제공

## 기술적 과제

1. **Checkpoint 로딩**
   - 모델 아키텍처를 먼저 정의해야 함
   - 일반적인 CNN 구조(ResNet, EfficientNet 등) 추정 필요

2. **다중 모델 관리**
   - 개/고양이별, 질환별 여러 모델 관리
   - 효율적인 메모리 사용을 위한 동적 로딩

3. **Segmentation 결과 처리**
   - 마스크를 시각화 가능한 형태로 변환
   - 병변 영역 크기 계산

## 임시 해결 방안

1. **Dummy Model Architecture**
   - 실제 모델 구조를 알 수 없으므로 임시 아키텍처 사용
   - 추후 정확한 모델 구조 파악 시 업데이트

2. **Selective Model Loading**
   - 필요한 모델만 선택적으로 로드
   - 메모리 효율성 향상

3. **Fallback Mechanism**
   - 모델 로딩 실패 시 대체 진단 로직
   - 에러 처리 강화