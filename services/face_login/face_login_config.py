import os
import re
import yaml
from pathlib import Path
from pydantic import BaseModel

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"

def load_yaml_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ${ENV_VAR:default} → 실제 값으로 바꿔주는 함수
def resolve_env_vars(value: str) -> str:
    if not isinstance(value, str):
        return value
    pattern = r"\$\{(\w+)(?::([^\}]*))?\}"
    def replacer(match):
        env_var = match.group(1)
        default = match.group(2)
        return os.getenv(env_var, default or "")
    return re.sub(pattern, replacer, value)

class FaceRecognitionConfig(BaseModel):
    base_path: str
    threshold: float
    spring_base_url: str
    spring_api_key: str

def get_face_login_config() -> FaceRecognitionConfig:
    raw = load_yaml_config(CONFIG_PATH)

    # ⚠여기서 resolve 해줌
    base_path = resolve_env_vars(raw["models"]["base_path"])
    threshold = raw["models"]["face_recognition"]["threshold"]
    spring_base_url = resolve_env_vars(raw["external_services"]["spring_boot"]["base_url"])
    spring_api_key = resolve_env_vars(raw["external_services"]["spring_boot"].get("api_key", ""))

    return FaceRecognitionConfig(
        base_path=base_path,
        threshold=threshold,
        spring_base_url=spring_base_url,
        spring_api_key=spring_api_key
    )
