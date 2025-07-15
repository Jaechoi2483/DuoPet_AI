# DuoPet AI Service

DuoPet AI 서비스는 반려동물 건강 관리 플랫폼 DuoPet의 핵심 AI 기능을 제공하는 FastAPI 기반 마이크로서비스입니다.

## 🚀 주요 기능

### 1. 얼굴인식 로그인 (Face Recognition Login)
- OpenCV + DeepFace를 활용한 얼굴 인증
- 높은 정확도의 사용자 식별
- REST API: `POST /api/v1/face-login`

### 2. GPT 기반 AI 챗봇 (AI Chatbot)
- KeyBERT를 활용한 키워드 추출
- OpenAI GPT API 연동
- 반려동물 관련 전문 상담
- REST API: `POST /api/v1/chatbot`

### 3. 유튜브 영상 추천 (YouTube Recommendation)
- 키워드 기반 영상 검색
- 반려동물 관련 콘텐츠 필터링
- REST API: `POST /api/v1/video-recommend`

### 4. 이미지 기반 건강 진단 (Health Diagnosis)
- YOLOv12 + EfficientNet 활용
- 피부병, 안구 질환 등 진단
- REST API: `POST /api/v1/health-diagnose`

### 5. 영상 기반 행동 분석 (Behavior Analysis)
- YOLOv12 + MediaPipe + LSTM 활용
- 이상 행동 탐지 및 분석
- REST API: `POST /api/v1/behavior-analysis`

## 📁 프로젝트 구조

```
DuoPet_AI/
├── api/                    # FastAPI 서버 및 라우터
│   ├── main.py            # FastAPI 앱 진입점
│   └── routers/           # API 엔드포인트 라우터
├── services/              # AI 서비스 모듈
│   ├── face_login/        # 얼굴인식 서비스
│   ├── chatbot/           # 챗봇 서비스
│   ├── video_recommend/   # 영상 추천 서비스
│   ├── health_diagnosis/  # 건강 진단 서비스
│   └── behavior_analysis/ # 행동 분석 서비스
├── common/                # 공통 모듈
│   ├── config.py          # 설정 관리
│   ├── logger.py          # 로깅 설정
│   └── utils.py           # 유틸리티 함수
├── models/                # AI 모델 가중치 파일
├── config/                # 설정 파일
│   └── config.yaml        # 애플리케이션 설정
├── tests/                 # 테스트 코드
├── requirements.txt       # Python 의존성
├── Dockerfile            # Docker 설정
└── docker-compose.yml    # Docker Compose 설정
```

## 🛠️ 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd DuoPet_AI

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일을 편집하여 필요한 API 키 등을 설정
# - OPENAI_API_KEY
# - YOUTUBE_API_KEY
# - 기타 설정값
```

### 3. 서버 실행

```bash
# 개발 서버 실행
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 프로덕션 서버 실행
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Docker를 사용한 실행

```bash
# Docker 이미지 빌드
docker build -t duopet-ai .

# Docker 컨테이너 실행
docker run -p 8000:8000 --env-file .env duopet-ai

# Docker Compose 사용
docker-compose up -d
```

## 📊 API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🧪 테스트

```bash
# 단위 테스트 실행
pytest tests/

# 커버리지 포함 테스트
pytest --cov=. tests/

# 특정 테스트만 실행
pytest tests/test_face_login.py
```

## 📝 API 응답 형식

모든 API는 다음과 같은 표준 응답 형식을 따릅니다:

### 성공 응답
```json
{
  "success": true,
  "data": {
    // API별 응답 데이터
  },
  "error": null
}
```

### 에러 응답
```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "ERROR_CODE",
    "message": "에러 메시지"
  }
}
```

## 🔧 개발 가이드

### 코드 스타일
- Black을 사용한 코드 포맷팅
- Flake8을 사용한 린팅
- Type hints 사용 권장

### 커밋 메시지 규칙
- feat: 새로운 기능 추가
- fix: 버그 수정
- docs: 문서 수정
- style: 코드 포맷팅
- refactor: 코드 리팩토링
- test: 테스트 코드
- chore: 빌드 업무 수정

## 📜 라이센스

이 프로젝트는 DuoPet 팀의 소유입니다.

## 👥 개발팀

- DuoPet 4조 AI 개발팀

## 📞 문의사항

프로젝트 관련 문의사항은 이슈 트래커를 이용해주세요.