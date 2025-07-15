# DuoPet AI - 아나콘다 환경 설정 가이드

## 🎯 개요
DuoPet AI 프로젝트는 아나콘다 가상환경을 사용하여 팀원 간 일관된 개발 환경을 유지합니다.

## 📋 사전 요구사항
- Anaconda 또는 Miniconda 설치
- CUDA 11.8+ (GPU 사용 시)
- Git

## 🚀 빠른 시작

### 1. 프로젝트 클론
```bash
git clone [repository-url]
cd DuoPet_AI
```

### 2. 아나콘다 환경 설정
```bash
# 실행 권한 부여
chmod +x setup_conda.sh

# 환경 설정 실행
./setup_conda.sh
```

또는 수동으로:
```bash
# 환경 생성
conda env create -f environment.yml

# 환경 활성화
conda activate duopet-ai
```

### 3. 환경 변수 설정
```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집하여 API 키 설정
nano .env  # 또는 선호하는 편집기 사용
```

## 🔧 환경 관리

### 환경 활성화/비활성화
```bash
# 활성화
conda activate duopet-ai

# 비활성화
conda deactivate
```

### 패키지 추가
```bash
# conda로 설치
conda install -c conda-forge [package-name]

# pip로 설치 (conda에 없는 경우)
pip install [package-name]

# environment.yml 업데이트
conda env export --from-history > environment.yml
```

### 환경 업데이트
```bash
# 다른 팀원이 패키지를 추가한 경우
conda env update -f environment.yml --prune
```

## 🏃‍♂️ 프로젝트 실행

### API 서버 실행
```bash
# 환경 활성화
conda activate duopet-ai

# 서버 실행
python api/main.py

# 또는 개발 모드
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 테스트 실행
```bash
# 전체 테스트
pytest

# 특정 테스트
pytest tests/test_auth.py
```

## 📊 Jupyter Notebook 사용
```bash
# Jupyter 실행
jupyter notebook

# Kernel 선택: "DuoPet AI"
```

## 🐳 Docker와 함께 사용 (선택사항)
아나콘다 환경을 Docker 이미지에 포함시킬 수 있습니다:

```dockerfile
FROM continuumio/miniconda3:latest

WORKDIR /app

# 환경 파일 복사
COPY environment.yml .

# 환경 생성
RUN conda env create -f environment.yml

# 환경 활성화를 위한 설정
SHELL ["conda", "run", "-n", "duopet-ai", "/bin/bash", "-c"]

# 앱 파일 복사
COPY . .

# 서버 실행
CMD ["conda", "run", "-n", "duopet-ai", "python", "api/main.py"]
```

## ⚠️ 주의사항

1. **환경 동기화**: 패키지를 추가/삭제한 경우 반드시 `environment.yml`을 업데이트하고 커밋
2. **Python 버전**: 팀원 모두 동일한 Python 버전(3.10) 사용
3. **GPU 설정**: GPU를 사용하는 경우 CUDA 버전 확인 필요
4. **플랫폼 차이**: Windows/Mac/Linux 간 일부 패키지가 다를 수 있음

## 🔍 문제 해결

### 환경 생성 실패
```bash
# 캐시 정리
conda clean --all

# 채널 우선순위 재설정
conda config --add channels conda-forge
conda config --set channel_priority strict
```

### GPU 인식 안됨
```python
# Python에서 확인
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### 패키지 충돌
```bash
# 환경 재생성
conda env remove -n duopet-ai
conda env create -f environment.yml
```

## 📞 지원
문제가 있으면 팀 슬랙 채널 또는 이슈 트래커에 문의하세요.