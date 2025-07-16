"""
API Routers for DuoPet AI Service

This module exports all available routers for the FastAPI application.
"""

# Import all routers
from . import (
    # auth,  # 임시 비활성화 - Oracle DB 연동 중
    face_login_router,
    chatbot_router,
    video_recommend_router,
    health_diagnosis_router,
    behavior_analysis_router
)

# Export for easy access
__all__ = [
    # "auth",  # 임시 비활성화 - Oracle DB 연동 중
    "face_login_router",
    "chatbot_router", 
    "video_recommend_router",
    "health_diagnosis_router",
    "behavior_analysis_router"
]