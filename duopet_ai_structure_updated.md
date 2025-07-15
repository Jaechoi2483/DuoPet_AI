# DuoPet AI 모듈 구조 설계 문서 (Updated)

본 문서는 DuoPet 프로젝트의 실제 구현된 AI 서비스 구조를 반영한 업데이트된 문서입니다.

## 🧱 실제 구현된 폴더 구조

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
│   ├── create_admin_key.py  # 관리자 키 생성
│   └── test_api_auth.py     # API 인증 테스트
├── tests/                    # 테스트 코드
│   ├── test_auth.py
│   ├── test_monitoring.py
│   └── test_response.py
├── training/                 # 모델 학습 스크립트
├── models/                   # 모델 저장 디렉토리
├── logs/                     # 로그 파일
├── data/                     # 데이터 디렉토리
├── docs/                     # 문서
│   └── ANACONDA_SETUP.md
├── environment.yml          # Anaconda 환경 설정
├── requirements.txt         # pip 패키지 목록
├── Dockerfile              # Docker 이미지 빌드
├── docker-compose.yml      # Docker Compose 설정
└── .env.example            # 환경 변수 예시
```

## 🔄 주요 변경사항

1. **services/ 디렉토리 도입**
   - 모든 AI 서비스를 `services/` 아래에 구성
   - 더 명확한 계층 구조

2. **video_recommend 추가**
   - 영상 추천 서비스 추가
   - 원래 문서에는 없던 기능

3. **API 인증 시스템**
   - `common/auth/` 모듈로 API 키 기반 인증
   - Spring Boot JWT와 연동 가능

4. **Oracle DB 연동**
   - MongoDB 대신 Oracle DB 사용
   - Spring Boot와 동일한 DB 공유

## 🏗️ 아키텍처 특징

### 1. 모듈화된 구조
- 각 AI 서비스는 독립적인 모듈
- 공통 기능은 `common/`에서 관리
- 서비스 간 의존성 최소화

### 2. 표준화된 API
- FastAPI 기반 RESTful API
- 표준 응답 포맷 (StandardResponse)
- Prometheus 메트릭 수집

### 3. 보안 및 인증
- API 키 기반 인증
- Spring Boot JWT 토큰 검증
- 스코프 기반 권한 관리

### 4. 데이터베이스
- Oracle DB 사용 (Spring Boot와 공유)
- Redis 캐싱 (선택사항)
- SQLAlchemy ORM

## 🚀 실행 방법

### Anaconda 환경 설정
```bash
# Anaconda Prompt에서
conda env create -f environment.yml
conda activate duopet-ai
```

### 서버 실행
```bash
# 환경 변수 설정
copy .env.example .env
# .env 파일 편집

# 서버 실행
python api/main.py
```

### API 문서
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

## 📋 환경 변수

```env
# Oracle DB
ORACLE_USER=duopet
ORACLE_PASSWORD=duopet  
ORACLE_DSN=13.209.33.76:1521/XEPDB1

# API Keys
OPENAI_API_KEY=your_key
YOUTUBE_API_KEY=your_key

# Spring Boot 연동
SPRING_BOOT_API_URL=http://localhost:8080
SPRING_JWT_SECRET=same_as_spring_boot
```

## 🔗 Spring Boot 연동

1. **JWT 토큰 공유**
   - Spring Boot에서 발급한 JWT 검증
   - 동일한 Secret Key 사용

2. **API 호출**
   - Spring → Python: AI 서비스 호출
   - Python → Spring: 사용자 정보 조회

3. **데이터베이스**
   - 동일한 Oracle DB 사용
   - AI 전용 테이블 prefix: `TB_AI_`

이 구조가 실제 구현을 정확히 반영합니다.