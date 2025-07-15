"""
API Routers for DuoPet AI Service

This module exports all available routers for the FastAPI application.
"""

# Import all routers
from . import (
    auth,
    face_login_router,
    chatbot_router,
    video_recommend_router,
    health_diagnosis_router,
    behavior_analysis_router
)

# Export for easy access
__all__ = [
    "auth",
    "face_login_router",
    "chatbot_router", 
    "video_recommend_router",
    "health_diagnosis_router",
    "behavior_analysis_router"
]