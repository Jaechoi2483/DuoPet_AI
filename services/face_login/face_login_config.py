'''
services/face_login/face_login_config.py

- config/config.yaml 파일에서 얼굴 인식 기능에 필요한 설정만 추출하는 모듈
- Pydantic 모델(FaceRecognitionConfig)을 통해 설정값을 구조화하여 제공
- 환경변수 처리(${ENV_VAR:default})를 지원하여 유연한 경로 설정 가능
'''

import os
import re
import yaml
from pathlib import Path
from pydantic import BaseModel

# config.yaml 경로 지정
CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"

# YAML 파일 로딩 함수
def load_yaml_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ${ENV_VAR:default} → 실제 값으로 변환
def resolve_env_vars(value: str) -> str:
    if not isinstance(value, str):
        return value
    pattern = r"\$\{(\w+)(?::([^\}]*))?\}"
    def replacer(match):
        env_var = match.group(1)
        default = match.group(2)
        return os.getenv(env_var, default or "")
    return re.sub(pattern, replacer, value)

# 얼굴 로그인 전용 설정 모델
class FaceRecognitionConfig(BaseModel):
    base_path: str
    threshold: float
    spring_base_url: str
    spring_api_key: str

# 설정을 읽어서 pydantic 모델로 반환
def get_face_login_config() -> FaceRecognitionConfig:
    raw = load_yaml_config(CONFIG_PATH)

    # base_path는 face_recognition 전용 경로에서 가져옴!
    base_path = resolve_env_vars(raw["models"]["face_recognition"]["base_path"])
    threshold = raw["models"]["face_recognition"]["threshold"]
    spring_base_url = resolve_env_vars(raw["external_services"]["spring_boot"]["base_url"])
    spring_api_key = resolve_env_vars(raw["external_services"]["spring_boot"].get("api_key", ""))

    return FaceRecognitionConfig(
        base_path=base_path,
        threshold=threshold,
        spring_base_url=spring_base_url,
        spring_api_key=spring_api_key
    )
