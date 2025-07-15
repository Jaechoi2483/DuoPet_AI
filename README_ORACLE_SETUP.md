# DuoPet AI - Oracle DB 설정 가이드

## 🎯 아키텍처 개요

DuoPet AI 서비스는 Spring Boot 백엔드와 동일한 Oracle DB를 사용하며, JWT 토큰을 통해 인증을 공유합니다.

```
[React Frontend] → [Spring Boot Backend] → [Oracle DB]
                          ↓ JWT Token
                    [Python AI Service]
```

## 📋 실행 계획

### 1. 아나콘다 환경 설정
```bash
# 1. 아나콘다 설치 확인
conda --version

# 2. 프로젝트 이동
cd /mnt/d/final_project/DuoPet_AI

# 3. 환경 생성
conda env create -f environment.yml

# 4. 환경 활성화
conda activate duopet-ai
```

### 2. Oracle Instant Client 설치
```bash
# WSL/Ubuntu의 경우
sudo apt-get update
sudo apt-get install libaio1

# Oracle Instant Client Basic 21.x 다운로드
# https://www.oracle.com/database/technologies/instant-client/linux-x86-64-downloads.html

# 다운로드 후 설치
sudo mkdir -p /opt/oracle
sudo unzip instantclient-basic-linux.x64-21.x.x.x.zip -d /opt/oracle/

# 환경 변수 설정
echo 'export LD_LIBRARY_PATH=/opt/oracle/instantclient_21_x:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3. 환경 변수 설정
```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집
nano .env
```

### 4. Oracle 연결 테스트
```python
# test_oracle.py
import cx_Oracle
import os
from dotenv import load_dotenv

load_dotenv()

# Oracle 연결
connection = cx_Oracle.connect(
    user=os.getenv('ORACLE_USER'),
    password=os.getenv('ORACLE_PASSWORD'),
    dsn=os.getenv('ORACLE_DSN')
)

cursor = connection.cursor()
cursor.execute("SELECT 'Connected!' FROM DUAL")
result = cursor.fetchone()
print(f"Oracle DB: {result[0]}")

cursor.close()
connection.close()
```

### 5. JWT 토큰 검증 설정
```python
# Spring Boot와 JWT Secret 공유
import jwt
import os

def verify_spring_jwt(token):
    """Spring Boot에서 발급한 JWT 토큰 검증"""
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

## 🔧 테이블 구조 설계

### AI 서비스 전용 테이블 (예시)
```sql
-- AI 모델 버전 관리
CREATE TABLE TB_AI_MODEL_VERSION (
    MODEL_ID VARCHAR2(50) PRIMARY KEY,
    MODEL_TYPE VARCHAR2(50) NOT NULL,  -- face_recognition, behavior_analysis 등
    VERSION VARCHAR2(20) NOT NULL,
    MODEL_PATH VARCHAR2(500),
    ACCURACY NUMBER(5,2),
    CREATED_AT DATE DEFAULT SYSDATE,
    STATUS VARCHAR2(20) DEFAULT 'ACTIVE'
);

-- AI 처리 로그
CREATE TABLE TB_AI_PROCESS_LOG (
    LOG_ID NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    USER_ID VARCHAR2(50),  -- TB_USER 참조
    MODEL_TYPE VARCHAR2(50),
    REQUEST_DATA CLOB,
    RESPONSE_DATA CLOB,
    PROCESSING_TIME NUMBER,
    SUCCESS_YN CHAR(1) DEFAULT 'Y',
    CREATED_AT TIMESTAMP DEFAULT SYSTIMESTAMP
);

-- AI API 키 관리 (Python 서비스 전용)
CREATE TABLE TB_AI_API_KEY (
    KEY_ID VARCHAR2(50) PRIMARY KEY,
    KEY_HASH VARCHAR2(128) NOT NULL,
    USER_ID VARCHAR2(50),
    KEY_NAME VARCHAR2(100),
    SCOPES VARCHAR2(500),  -- JSON 형태로 저장
    STATUS VARCHAR2(20) DEFAULT 'ACTIVE',
    RATE_LIMIT NUMBER DEFAULT 100,
    CREATED_AT DATE DEFAULT SYSDATE,
    LAST_USED_AT DATE,
    EXPIRES_AT DATE
);
```

## 🚀 실행 순서

```bash
# 1. 환경 활성화
conda activate duopet-ai

# 2. Oracle 연결 테스트
python test_oracle.py

# 3. 테이블 생성 (필요시)
python scripts/create_tables.py

# 4. 서버 실행
python api/main.py

# 5. API 문서 확인
# http://localhost:8000/docs
```

## ⚙️ Spring Boot 연동

### JWT 토큰 전달 방식
```javascript
// Frontend에서 API 호출 시
const response = await fetch('http://localhost:8000/api/v1/face-login', {
    headers: {
        'Authorization': `Bearer ${springJwtToken}`,  // Spring Boot JWT
        'Content-Type': 'application/json'
    },
    // ...
});
```

### Python에서 Spring Boot API 호출
```python
import httpx
import os

async def call_spring_api(endpoint: str, jwt_token: str):
    """Spring Boot API 호출"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{os.getenv('SPRING_BOOT_API_URL')}/{endpoint}",
            headers={"Authorization": f"Bearer {jwt_token}"}
        )
        return response.json()
```

## 📝 주의사항

1. **Oracle 버전**: Spring Boot와 동일한 Oracle 버전 사용
2. **문자 인코딩**: UTF-8 설정 확인
3. **트랜잭션**: 가능한 Spring Boot에서 처리, AI는 읽기 위주
4. **보안**: JWT Secret은 절대 노출 금지

## 🔍 문제 해결

### cx_Oracle 설치 오류
```bash
# conda로 재설치
conda install -c conda-forge cx_oracle

# 또는 pip로 설치
pip install cx-Oracle
```

### Oracle 연결 실패
```python
# 상세 에러 확인
cx_Oracle.init_oracle_client(lib_dir="/opt/oracle/instantclient_21_x")
```

### JWT 검증 실패
- Spring Boot와 동일한 Secret Key 사용 확인
- 토큰 만료 시간 확인
- Algorithm (HS256) 일치 확인