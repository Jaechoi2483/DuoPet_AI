# 🤖 DuoPet AI 개발 규칙 (RULE_AI.md)

본 문서는 DuoPet 프로젝트의 AI 서비스 개발 규칙과 실제 구현된 구조를 통합한 가이드입니다.

## 1. 📁 실제 구현된 디렉터리 구조

```
/DuoPet_AI/                   # AI 서비스 루트 디렉토리
├── api/                      # FastAPI 애플리케이션
│   ├── main.py              # FastAPI 앱 엔트리포인트
│   ├── middleware.py        # 커스텀 미들웨어 (인증, 로깅 등)
│   └── routers/             # API 라우터
│       ├── auth.py          # API 키 관리 엔드포인트
│       ├── face_login_router.py
│       ├── chatbot_router.py
│       ├── health_diagnosis_router.py
│       ├── behavior_analysis_router.py
│       └── video_recommend_router.py
├── services/                 # AI 서비스 모듈
│   ├── face_login/          # 얼굴인식 로그인
│   │   ├── models/          # 학습된 모델 가중치
│   │   ├── preprocessing/   # 전처리 (OpenCV, 얼굴 검출)
│   │   ├── inference/       # 모델 추론
│   │   ├── utils/           # 유틸리티
│   │   ├── predict.py       # 예측 메인 함수
│   │   └── train.py         # 학습 스크립트
│   ├── chatbot/             # AI 챗봇 (GPT 기반)
│   ├── health_diagnosis/    # 이미지 건강 진단
│   ├── behavior_analysis/   # 행동 분석
│   └── video_recommend/     # 영상 추천
├── common/                   # 공통 모듈
│   ├── config.py            # 환경 설정
│   ├── logger.py            # 로깅 설정
│   ├── database.py          # Oracle DB 연결
│   ├── response.py          # 표준 응답 포맷
│   ├── exceptions.py        # 커스텀 예외
│   ├── monitoring.py        # Prometheus 메트릭
│   ├── utils.py             # 공통 유틸리티
│   └── auth/                # 인증 모듈
│       ├── models.py        # API 키 모델
│       ├── service.py       # API 키 서비스
│       └── dependencies.py  # FastAPI 의존성
├── config/                   # 설정 파일
│   └── config.yaml          # 모델별 설정
├── scripts/                  # 유틸리티 스크립트
├── tests/                    # 테스트 코드
├── training/                 # 모델 학습 스크립트
├── models/                   # 모델 저장 디렉토리
├── logs/                     # 로그 파일
├── data/                     # 데이터 디렉토리
├── docs/                     # 문서
├── environment.yml          # Anaconda 환경 설정
├── requirements.txt         # pip 패키지 목록
├── Dockerfile              # Docker 이미지 빌드
└── .env.example            # 환경 변수 예시
```

## 2. 🧠 모델 및 가중치 관리

- 모델 구조는 `services/[서비스명]/models/`, 학습된 가중치는 동일 위치에 저장
- 모델 버전 구분은 `faceid_efficientnet_v1.pth` 처럼 **[기능]_[모델명]_[버전]** 형식으로 명시
- **재현성**을 위해 학습 시 사용한 `config.yaml` 파일도 함께 관리
- GPU에서 학습한 모델 공유 시 `.cpu()` 변환 or `map_location='cpu'` 사용 필수

## 3. 💡 학습 스크립트 규칙

- `services/[서비스명]/train.py` 또는 `training/` 디렉토리에 기능별 파일 분리
- `config/config.yaml` 파일을 통해 하이퍼파라미터, 경로 등 설정값 관리
- 데이터 및 모델 경로는 절대경로가 아닌, 프로젝트 루트 기준 상대경로로 지정

## 4. 🚀 API 서버 구성 (FastAPI)

### 4.1 표준 응답 JSON 구조
모든 API 응답은 `common/response.py`의 StandardResponse 형식을 따름:

```json
// 성공 응답
{
  "success": true,
  "data": {
    "result": "정상",
    "confidence": 0.95,
    "details": {}
  },
  "error": null,
  "metadata": {
    "request_id": "uuid",
    "timestamp": "2024-01-01T00:00:00Z"
  }
}

// 실패 응답
{
  "success": false,
  "data": null,
  "error": {
    "code": "FACE_NOT_DETECTED",
    "message": "얼굴을 찾을 수 없습니다.",
    "detail": {}
  },
  "metadata": {
    "request_id": "uuid",
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

### 4.2 비동기 처리
- I/O 바운드 작업(파일 읽기 등)은 `async def` 사용
- CPU 바운드 작업(모델 추론)은 FastAPI의 `run_in_threadpool` 활용

### 4.3 API 보안
- API 키 기반 인증 (`common/auth/`)
- Spring Boot JWT 토큰 검증 지원
- 스코프 기반 권한 관리

## 5. 🔗 Spring Boot 연동

### 5.1 JWT 토큰 공유
```python
# Spring Boot에서 발급한 JWT 토큰 검증
import jwt
import os

def verify_spring_jwt(token):
    """Spring Boot JWT 토큰 검증"""
    try:
        payload = jwt.decode(
            token,
            os.getenv('SPRING_JWT_SECRET'),
            algorithms=['HS256']
        )
        return payload
    except jwt.InvalidTokenError:
        return None
```

### 5.2 데이터베이스
- Oracle DB 사용 (Spring Boot와 동일)
- AI 전용 테이블 prefix: `TB_AI_`
- 연결 정보는 `.env` 파일에서 관리

## 6. 🏗️ 주요 기능별 구현 가이드

### 6.1 얼굴 인식 로그인 (`services/face_login/`)
- **기술**: OpenCV + DeepFace (VGGFace)
- **프로세스**: 얼굴 검출 → 임베딩 추출 → 유사도 비교

### 6.2 이미지 건강 진단 (`services/health_diagnosis/`)
- **기술**: YOLOv12 + EfficientNet
- **프로세스**: 병변 탐지 → 질병 분류 → 리포트 생성

### 6.3 행동 분석 (`services/behavior_analysis/`)
- **기술**: YOLOv12 + MediaPipe + LSTM
- **프로세스**: 객체 감지 → 포즈 추정 → 시퀀스 분석

### 6.4 AI 챗봇 (`services/chatbot/`)
- **기술**: KeyBERT + GPT API
- **프로세스**: 키워드 추출 → GPT 호출 → 응답 생성

### 6.5 영상 추천 (`services/video_recommend/`)
- **기술**: 추천 알고리즘 + YouTube API
- **프로세스**: 사용자 분석 → 추천 생성 → 콘텐츠 제공

## 7. 🚀 개발 환경 설정

### 7.1 Anaconda 환경
```bash
# Anaconda Prompt에서
conda env create -f environment.yml
conda activate duopet-ai
```

### 7.2 환경 변수 (.env)
```env
# Oracle DB (Spring Boot와 동일)
ORACLE_USER=duopet
ORACLE_PASSWORD=duopet
ORACLE_DSN=13.209.33.76:1521/XEPDB1

# API Keys
OPENAI_API_KEY=your_key
YOUTUBE_API_KEY=your_key

# Spring Boot 연동
SPRING_BOOT_API_URL=http://localhost:8080
SPRING_JWT_SECRET=vrDt6Hhffv9gPPEEHDBVhxY4W+gf//bxDgVljRr/+8z1ZxqEdgTmDDZ/UIquJuWQdZmJ8mz/DuzLF/pmcMFaqw==
```

### 7.3 서버 실행
```bash
# 개발 모드
python api/main.py

# 프로덕션 모드
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 8. ⚠️ 주의사항

1. **모델 파일 관리**
   - 대용량 모델 파일은 Git LFS 사용
   - 학습 결과는 반드시 문서화

2. **API 응답 시간**
   - 모델 추론은 3초 이내 목표
   - 타임아웃 설정 필수

3. **에러 처리**
   - 모든 예외는 `common/exceptions.py` 사용
   - 사용자 친화적 에러 메시지 제공

4. **로깅**
   - `common/logger.py` 사용
   - 구조화된 JSON 로그 형식

5. **테스트**
   - 각 서비스별 단위 테스트 필수
   - API 통합 테스트 작성

## 9. 📋 개발 체크리스트

- [ ] 환경 변수 설정 완료
- [ ] Oracle DB 연결 테스트
- [ ] 모델 파일 경로 확인
- [ ] API 문서 자동 생성 확인 (/docs)
- [ ] 로깅 설정 확인
- [ ] 단위 테스트 작성
- [ ] README.md 업데이트

이 규칙을 준수하여 일관성 있고 유지보수가 용이한 AI 서비스를 개발합니다.