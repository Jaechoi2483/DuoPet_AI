"""
Configuration management for DuoPet AI Service

This module handles all configuration settings using Pydantic Settings.
"""

import os
from typing import List, Optional, Dict, Any
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml

PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE_FULL_PATH = os.path.join(PROJECT_ROOT_DIR, ".env")
class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_FULL_PATH,
        env_file_encoding="utf-8",
        case_sensitive=True
    )



    # Application
    APP_NAME: str = "DuoPet AI Service"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "AI Microservices for DuoPet Platform"
    ENVIRONMENT: str = Field(default="development", description="Runtime environment")
    DEBUG: bool = Field(default=False)
    
    # API Server
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_WORKERS: int = Field(default=4)
    
    # Security
    API_SECRET_KEY: Optional[str] = Field(default="duopet-ai-secret-key-2024-change-in-production", min_length=32)
    API_KEY_SALT: Optional[str] = Field(default="duopet-ai-salt-2024")
    SECRET_KEY: Optional[str] = Field(default=None)  # JWT Secret from backend
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRATION_DAYS: int = Field(default=30)
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    YOUTUBE_API_KEY: Optional[str] = Field(default=None)
    PERPLEXITY_API_KEY: Optional[str] = Field(default=None)

    ADMIN_API_KEY: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8'
    )
    # Database
    MONGODB_URL: str = Field(default="mongodb://localhost:27017")
    MONGODB_DATABASE: str = Field(default="duopet_ai")
    REDIS_URL: str = Field(default="redis://localhost:6379")

    ORACLE_USER: Optional[str] = Field(default=None)
    ORACLE_PASSWORD: Optional[str] = Field(default=None)
    ORACLE_DSN: Optional[str] = Field(default=None)
    ORACLE_HOST: Optional[str] = Field(default=None)
    ORACLE_PORT: Optional[int] = Field(default=1521)
    ORACLE_SERVICE: Optional[str] = Field(default=None)
    DATABASE_URL: Optional[str] = Field(default=None)
    
    # Model Configuration
    MODEL_PATH: str = Field(default="/app/models")
    MAX_BATCH_SIZE: int = Field(default=32)
    GPU_DEVICE: str = Field(default="cuda:0")
    USE_GPU: bool = Field(default=True)
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FILE_PATH: str = Field(default="/app/logs/duopet_ai.log")
    LOG_MAX_SIZE: str = Field(default="100MB")
    LOG_BACKUP_COUNT: int = Field(default=5)
    
    # CORS
    CORS_ORIGINS: List[str] = Field(default=["http://localhost:3000", "http://localhost:8080"])
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    ALLOWED_HOSTS: List[str] = Field(default=["*"])
    
    # File Upload
    MAX_UPLOAD_SIZE_MB: int = Field(default=50)
    ALLOWED_IMAGE_EXTENSIONS: List[str] = Field(default=[".jpg", ".jpeg", ".png", ".bmp"])
    ALLOWED_VIDEO_EXTENSIONS: List[str] = Field(default=[".mp4", ".avi", ".mov", ".mkv"])
    
    # External Services
    SPRING_BOOT_API_URL: str = Field(default="http://localhost:8080/api")
    SPRING_BOOT_API_KEY: Optional[str] = Field(default=None)
    
    # Feature Flags
    FEATURES_FACE_LOGIN_ENABLED: bool = Field(default=True)
    FEATURES_CHATBOT_ENABLED: bool = Field(default=True)
    FEATURES_HEALTH_DIAGNOSIS_ENABLED: bool = Field(default=True)
    FEATURES_BEHAVIOR_ANALYSIS_ENABLED: bool = Field(default=True)
    FEATURES_VIDEO_RECOMMEND_ENABLED: bool = Field(default=True)

    # External Services
    SPRING_BOOT_API_URL: str = Field(default="http://localhost:8080/api")
    SPRING_BOOT_API_KEY: Optional[str] = Field(default=None)

    # 새로 추가할 부분:
    SPRING_JWT_SECRET: Optional[str] = Field(default=None)  # .env에서 값을 받으므로 Optional로 설정하거나 Field(...)로 필수로 설정

    # Feature Flags
    FEATURES_FACE_LOGIN_ENABLED: bool = Field(default=True)

    # CORS_ORIGINS는 위에 이미 정의됨
    # ALLOWED_IMAGE_EXTENSIONS는 위에 이미 정의됨
    ALLOWED_VIDEO_EXTENSIONS: List[str] = Field(default=["mp4", "avi", "mov", "webm"])

    @field_validator('CORS_ORIGINS', 'ALLOWED_IMAGE_EXTENSIONS', 'ALLOWED_VIDEO_EXTENSIONS', mode='before')
    @classmethod
    def _split_str_to_list(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [item.strip() for item in v.split(',') if item.strip()]
        return v  # 이미 리스트이거나 다른 타입이면 그대로 반환
    
    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v
    
    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.ENVIRONMENT == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.ENVIRONMENT == "development"
    
    @property
    def max_upload_size_bytes(self) -> int:
        """Get max upload size in bytes"""
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    
    def __init__(self, **kwargs):
        """Initialize settings and compute ORACLE_DSN if not provided"""
        super().__init__(**kwargs)
        if not self.ORACLE_DSN and self.ORACLE_HOST and self.ORACLE_SERVICE:
            self.ORACLE_DSN = f"{self.ORACLE_HOST}:{self.ORACLE_PORT}/{self.ORACLE_SERVICE}"


class ModelConfig:
    """Model-specific configuration loaded from YAML"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        return self._config.get("models", {}).get(model_name, {})
    
    def get_preprocessing_config(self, data_type: str) -> Dict[str, Any]:
        """Get preprocessing configuration"""
        return self._config.get("preprocessing", {}).get(data_type, {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self._config.get("api", {})
    
    def get_feature_config(self, feature: str) -> Dict[str, Any]:
        """Get feature-specific configuration"""
        return self._config.get("features", {}).get(feature, {})


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


@lru_cache()
def get_model_config() -> ModelConfig:
    """Get cached model configuration instance"""
    return ModelConfig()


# Convenience functions
def is_feature_enabled(feature: str) -> bool:
    """Check if a feature is enabled"""
    settings = get_settings()
    feature_map = {
        "face_login": settings.FEATURES_FACE_LOGIN_ENABLED,
        "chatbot": settings.FEATURES_CHATBOT_ENABLED,
        "health_diagnosis": settings.FEATURES_HEALTH_DIAGNOSIS_ENABLED,
        "behavior_analysis": settings.FEATURES_BEHAVIOR_ANALYSIS_ENABLED,
        "video_recommend": settings.FEATURES_VIDEO_RECOMMEND_ENABLED,
    }
    return feature_map.get(feature, False)


def get_redis_url() -> str:
    """Get Redis connection URL"""
    settings = get_settings()
    return settings.REDIS_URL


def get_mongodb_url() -> str:
    """Get MongoDB connection URL"""
    settings = get_settings()
    return settings.MONGODB_URL


def get_model_path(model_filename: str) -> str:
    """Get full path to a model file"""
    settings = get_settings()
    return os.path.join(settings.MODEL_PATH, model_filename)