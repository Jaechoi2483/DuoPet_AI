# --------------------------------------------------------------------------
# 파일명: api/routers/chatbot_router.py
# 설명: RAG 챗봇 서비스를 FastAPI 엔드포인트와 연동
# --------------------------------------------------------------------------
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date

# DuoPet AI 프로젝트의 공통 모듈 및 서비스 임포트
from common.response import StandardResponse, create_success_response, create_error_response, ErrorCode
from common.logger import get_logger
from services import chatbot

from services.chatbot.chatbot_db import  get_user_profile_for_chatbot

# RAG 챗봇 서비스 임포트
from services.chatbot.predict import RAGChatbot

logger = get_logger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    user_id: str = Field(..., description="User ID for context")


class SuggestedAction(BaseModel):
    name: str
    description: str
    url: str


class ChatResponseData(BaseModel):
    answer: str = Field(..., description="AI response message")
    suggested_actions: List[SuggestedAction] = Field(default=[], description="Suggested actions for the user")
    predicted_questions: List[str] = Field(default=[], description="Predicted follow-up questions")




def get_chatbot(request: Request) -> RAGChatbot:

    if not hasattr(request.app.state, 'chatbot') or request.app.state.chatbot is None:
        logger.error("Chatbot instance not found in app state. It may have failed to initialize.")
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail="챗봇 서비스를 현재 사용할 수 없습니다. 서버 초기화 중 오류가 발생했을 수 있습니다."
        )
    return request.app.state.chatbot





@router.post("/chat", response_model=StandardResponse[ChatResponseData])
async def chat(
        request_data: ChatRequest,
        chatbot: RAGChatbot = Depends(get_chatbot),
        user_profile: Dict[str, Any] = Depends(get_user_profile_for_chatbot)
):
    logger.info(f"Chat request from user: {request_data.user_id}")
    try:
        response_data = chatbot.ask(request_data.message, user_profile)
        response_model_data = ChatResponseData(**response_data)
        return create_success_response(data=response_model_data)
    except Exception as e:
        logger.error(f"Chat API 처리 중 에러 발생: {e}", exc_info=True)
        return create_error_response(
            error_code=ErrorCode.UNKNOWN_ERROR,
            message="요청을 처리하는 중 내부 오류가 발생했습니다."
        )

