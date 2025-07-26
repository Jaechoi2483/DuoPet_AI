# DuoPet AI 서버 실행 가이드

## 필요한 환경

1. Python 3.8 이상
2. Conda 환경 (duopet)
3. TensorFlow 2.x

## 서버 실행 방법

### 1. Conda 환경 활성화
```bash
conda activate duopet
```

### 2. 프로젝트 디렉토리로 이동
```bash
cd D:\final_project\DuoPet_AI
```

### 3. 서버 실행 옵션

#### 옵션 1: uvicorn 직접 사용 (권장)
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 옵션 2: Python 모듈로 실행
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 옵션 3: 수동 환경 변수 설정 후 실행
```bash
# Windows PowerShell
$env:TF_USE_LEGACY_KERAS="1"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## 문제 해결

### TensorFlow/Keras 버전 호환성 문제
- `'Adam' object has no attribute 'build'` 에러가 발생하는 경우
- 해결책: utils/model_loader.py를 사용하여 모델 로딩

### CORS 에러
- 프론트엔드에서 API 호출 시 CORS 에러가 발생하는 경우
- FastAPI의 CORS 미들웨어가 올바르게 설정되어 있는지 확인

### 포트 충돌
- 8000 포트가 이미 사용 중인 경우
- 다른 포트 사용: `--port 8001`

## 서버 상태 확인

서버가 정상적으로 실행되면:
1. http://localhost:8000/docs - Swagger UI
2. http://localhost:8000/health - 헬스 체크
3. http://localhost:8000/health-diagnosis/status - AI 진단 서비스 상태

## 로그 확인

서버 실행 시 콘솔에서 다음과 같은 로그를 확인할 수 있습니다:
- 모델 로딩 상태
- API 요청/응답 로그
- 에러 메시지

## 개발 모드 vs 프로덕션 모드

### 개발 모드 (--reload 옵션 사용)
```bash
uvicorn api.main:app --reload
```
- 코드 변경 시 자동 재시작
- 디버깅 정보 표시

### 프로덕션 모드
```bash
uvicorn api.main:app --workers 4
```
- 멀티 워커 사용
- 성능 최적화