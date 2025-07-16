# --------------------------------------------------------------------------
# 파일명: api/routers/chatbot_router.py
# 설명: RAG 챗봇 서비스를 FastAPI 엔드포인트와 연동
# --------------------------------------------------------------------------
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import oracledb
from datetime import date

# DuoPet AI 프로젝트의 공통 모듈 및 서비스 임포트
from common.response import StandardResponse, create_success_response, create_error_response, ErrorCode
from common.logger import get_logger
from common.database import get_oracle_connection

# RAG 챗봇 서비스 임포트
from services.chatbot.predict import RAGChatbot

logger = get_logger(__name__)
router = APIRouter()


# --- Pydantic 모델 정의 ---
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


# --- 챗봇 설정 (main.py에서 이 값들을 사용하여 객체를 생성합니다) ---
TARGET_URL = "http://localhost:3000"
SITE_FUNCTIONS = [
    {"name": "notice_board", "description": "공지사항 확인하기", "url": "/notice"},
    {"name": "free_board", "description": "자유게시판 가기", "url": "/board"},
    {"name": "health_check", "description": "반려동물 건강 진단하기", "url": "/health-check"},
    {"name": "behavior_analysis", "description": "이상행동 분석 서비스 보기", "url": "/behavior-analysis"},
    {"name": "video_recommend", "description": "추천 영상 보러가기", "url": "/recommendations"},
    {"name": "qna", "description": "qna", "url": "/qna"},
    {"name": "login", "description": "로그인", "url": "/login"}
]


# --- 의존성 주입 함수 ---
def get_chatbot(request: Request) -> RAGChatbot:
    """
    FastAPI 앱 상태(app.state)에서 초기화된 챗봇 인스턴스를 가져옵니다.
    """
    if not hasattr(request.app.state, 'chatbot') or request.app.state.chatbot is None:
        logger.error("Chatbot instance not found in app state. It may have failed to initialize.")
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail="챗봇 서비스를 현재 사용할 수 없습니다. 서버 초기화 중 오류가 발생했을 수 있습니다."
        )
    return request.app.state.chatbot


# --- 실제 DB 연동 함수 (이 함수는 그대로 유지) ---
async def get_user_profile_from_db_func(
        user_id: str,
        conn: oracledb.AsyncConnection = Depends(get_oracle_connection)  # 이 함수 내부에서 conn을 받음
) -> Dict[str, Any]:
    """
    주어진 user_id로 Oracle 데이터베이스에서 회원 정보를 조회하여 반환합니다.
    """
    user_profile = {"name": "비회원", "interests": [], "user_id": None}

    try:
        user_id_num = int(user_id)
    except ValueError:
        logger.error(f"Invalid user_id format received: '{user_id}'. Must be a number.")
        return user_profile

    try:
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                SELECT USER_ID,
                       USER_NAME,
                       NICKNAME,
                       USER_EMAIL,
                       CREATED_AT,
                       AGE,
                       GENDER,
                       PHONE,
                       ADDRESS
                FROM USERS
                WHERE USER_ID = :user_id_param
                """,
                user_id_param=user_id_num
            )
            user_row = await cursor.fetchone()

            if user_row:
                user_profile["user_id"] = user_row[0]
                user_profile["name"] = user_row[1] if user_row[1] else user_row[2]
                user_profile["nickname"] = user_row[2]
                user_profile["email"] = user_row[3]
                user_profile["member_since"] = user_row[4].strftime("%Y-%m-%d") if isinstance(user_row[4],
                                                                                              date) else None
                user_profile["age"] = user_row[5]
                user_profile["gender"] = user_row[6]
                user_profile["phone"] = user_row[7]
                user_profile["address"] = user_row[8]

                await cursor.execute(
                    """
                    SELECT PET_NAME,
                           ANIMAL_TYPE,
                           BREED,
                           AGE,
                           GENDER,
                           NEUTERED,
                           WEIGHT,
                           REGISTRATION_DATE
                    FROM PETS
                    WHERE USER_ID = :owner_id_param
                    """,
                    owner_id_param=user_id_num
                )
                pet_rows = await cursor.fetchall()

                pet_info_list = []
                for pet_row in pet_rows:
                    pet_info_list.append({
                        "name": pet_row[0],
                        "species": pet_row[1],
                        "breed": pet_row[2],
                        "age": f"{pet_row[3]}세" if pet_row[3] else None,
                        "gender": pet_row[4],
                        "neutered": pet_row[5],
                        "weight": pet_row[6],
                        "registration_date": pet_row[7].strftime("%Y-%m-%d") if isinstance(pet_row[7], date) else None
                    })
                user_profile["pet_info"] = pet_info_list
            else:
                logger.warning(f"User with ID '{user_id}' not found in DB. Returning default profile.")

    except oracledb.Error as e:
        logger.error(f"Oracle DB Error fetching user profile for '{user_id}': {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error fetching user profile for '{user_id}': {e}", exc_info=True)

    return user_profile


# --- API 엔드포인트 (수정 필요) ---
@router.post("/chat", response_model=StandardResponse[ChatResponseData])
async def chat(
        request_data: ChatRequest,
        chatbot: RAGChatbot = Depends(get_chatbot),
        conn: oracledb.AsyncConnection = Depends(get_oracle_connection)  # 💡 conn을 여기서 직접 주입받습니다.
):
    logger.info(f"Chat request from user: {request_data.user_id}")
    try:
        # 💡 주입받은 conn과 request_data.user_id를 사용하여 user_profile을 가져옵니다.
        user_profile = await get_user_profile_from_db_func(request_data.user_id, conn)

        response_data = chatbot.ask(request_data.message, user_profile)
        response_model_data = ChatResponseData(**response_data)
        return create_success_response(data=response_model_data)
    except Exception as e:
        logger.error(f"Chat API 처리 중 에러 발생: {e}", exc_info=True)
        return create_error_response(
            error_code=ErrorCode.UNKNOWN_ERROR,
            message="요청을 처리하는 중 내부 오류가 발생했습니다."
        )