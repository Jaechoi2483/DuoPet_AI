"""
Configuration management for DuoPet AI Service

This module handles all configuration settings using Pydantic Settings.
"""

import os
from typing import List, Optional, Dict, Any
from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml

# --- 프로젝트 경로 설정 ---
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE_PATH = os.path.join(PROJECT_ROOT_DIR, ".env")
MODEL_CONFIG_YAML_PATH = os.path.join(PROJECT_ROOT_DIR, "config", "config.yaml")


class Settings(BaseSettings):
    """
    .env 파일과 환경 변수를 통해 애플리케이션 설정을 관리합니다.
    """
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8",
        case_sensitive=False,  # 환경 변수는 보통 대소문자를 구분하지 않음
        extra='ignore'
    )

    # --- Application Info ---
    APP_NAME: str = "DuoPet AI Service"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "AI Microservices for DuoPet Platform"
    ENVIRONMENT: str = Field(default="development", description="런타임 환경 (development, staging, production)")
    DEBUG: bool = Field(default=False)

    # --- API Server ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4

    # --- Security ---
    SECRET_KEY: str = Field("duopet-ai-secret-key-2024-change-in-production", min_length=32)
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_DAYS: int = 30

    # --- API Keys ---
    OPENAI_API_KEY: Optional[str] = None
    YOUTUBE_API_KEY: Optional[str] = None
    PERPLEXITY_API_KEY: Optional[str] = None
    ADMIN_API_KEY: Optional[str] = None

    # --- Database (Oracle & MongoDB & Redis) ---
    # Oracle
    ORACLE_USER: Optional[str] = None
    ORACLE_PASSWORD: Optional[str] = None
    ORACLE_DSN: Optional[str] = None
    # 아래는 ORACLE_DSN이 없을 경우 조합용으로 사용됩니다.
    ORACLE_HOST: Optional[str] = None
    ORACLE_PORT: int = 1521
    ORACLE_SERVICE: Optional[str] = None
    DATABASE_URL: Optional[str] = None
    
    # MongoDB
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "duopet_ai"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # --- Model Configuration ---
    MODEL_PATH: str = os.path.join(PROJECT_ROOT_DIR, "db_models")
    MAX_BATCH_SIZE: int = 32
    GPU_DEVICE: str = "cuda:0"
    USE_GPU: bool = True

    # --- Logging ---
    LOG_LEVEL: str = "INFO"
    LOG_FILE_PATH: str = os.path.join(PROJECT_ROOT_DIR, "logs", "duopet_ai.log")

    # --- CORS ---
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    CORS_ALLOW_CREDENTIALS: bool = True
    ALLOWED_HOSTS: List[str] = ["*"]

    # --- File Upload ---
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_IMAGE_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "bmp"]
    ALLOWED_VIDEO_EXTENSIONS: List[str] = ["mp4", "avi", "mov", "mkv", "webm"]

    SITE_URL: str = "http://localhost:3000"
    # --- External Services ---
    SPRING_BOOT_API_URL: str = "http://localhost:8080/api"
    SPRING_BOOT_API_KEY: Optional[str] = None
    SPRING_JWT_SECRET: Optional[str] = None

    # --- Feature Flags ---
    FEATURES_FACE_LOGIN_ENABLED: bool = True
    FEATURES_CHATBOT_ENABLED: bool = True
    FEATURES_HEALTH_DIAGNOSIS_ENABLED: bool = True
    
    # --- 유효성 검사 및 자동 계산 ---
    
    @model_validator(mode='after')
    def compute_db_urls(self) -> 'Settings':
        # 1. ORACLE_DSN 계산 (기존 로직)
        if not self.ORACLE_DSN and self.ORACLE_HOST and self.ORACLE_SERVICE:
            self.ORACLE_DSN = f"{self.ORACLE_HOST}:{self.ORACLE_PORT}/{self.ORACLE_SERVICE}"

        # 👇👇👇 2. SQLAlchemy용 DATABASE_URL 계산 (새 로직)
        # DATABASE_URL이 없고, Oracle 접속 정보가 모두 있을 때 자동으로 생성
        if not self.DATABASE_URL and self.ORACLE_USER and self.ORACLE_PASSWORD and self.ORACLE_DSN:
            self.DATABASE_URL = f"oracle+oracledb://{self.ORACLE_USER}:{self.ORACLE_PASSWORD}@{self.ORACLE_DSN}"
        
        return self

    @field_validator('CORS_ORIGINS', mode='before')
    @classmethod
    def _split_str_to_list(cls, v: Any) -> List[str]:
        """쉼표로 구분된 문자열을 리스트로 변환합니다."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(',') if item.strip()]
        return v
        
    # --- 계산된 프로퍼티 ---

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        """개발 환경인지 확인하는 프로퍼티"""
        return self.ENVIRONMENT == "development"
    
    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


class ModelConfig:
    """YAML 파일에서 모델별 세부 설정을 로드합니다."""
    
    def __init__(self, config_path: str = MODEL_CONFIG_YAML_PATH):
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """YAML 설정 파일을 로드합니다."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """특정 모델의 설정을 가져옵니다."""
        return self._config.get("db_models", {}).get(model_name, {})
    
    def get_preprocessing_config(self, data_type: str) -> Dict[str, Any]:
        """전처리 설정을 가져옵니다."""
        return self._config.get("preprocessing", {}).get(data_type, {})


# --- 설정 및 유틸리티 함수 ---

@lru_cache()
def get_settings() -> Settings:
    """
    설정 객체를 반환합니다. lru_cache를 이용해 한번만 로드합니다.
    """
    return Settings()


@lru_cache()
def get_model_config() -> ModelConfig:
    """
    모델 설정 객체를 반환합니다. lru_cache를 이용해 한번만 로드합니다.
    """
    return ModelConfig()


def is_feature_enabled(feature: str) -> bool:
    """특정 기능의 활성화 여부를 확인합니다."""
    settings = get_settings()
    feature_map = {
        "face_login": settings.FEATURES_FACE_LOGIN_ENABLED,
        "chatbot": settings.FEATURES_CHATBOT_ENABLED,
        "health_diagnosis": settings.FEATURES_HEALTH_DIAGNOSIS_ENABLED,
    }
    return feature_map.get(feature.lower(), False)


def get_redis_url() -> str:
    """Redis 연결 URL을 반환합니다."""
    return get_settings().REDIS_URL


def get_mongodb_url() -> str:
    """MongoDB 연결 URL을 반환합니다."""
    return get_settings().MONGODB_URL


def get_model_path(model_filename: str) -> str:
    """모델 파일의 전체 경로를 반환합니다."""
    # 상대 경로 사용 - OS 관계없이 작동
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent  # DuoPet_AI 폴더
    model_path = BASE_DIR / "models" / model_filename
    return str(model_path)