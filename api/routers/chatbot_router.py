# api/routers/chatbot_router.py

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import oracledb
from datetime import date

from common.response import StandardResponse, create_success_response, create_error_response, ErrorCode
from common.logger import get_logger
from common.database import get_oracle_connection
from services.chatbot.predict import RAGChatbot

logger = get_logger(__name__)
router = APIRouter()

# --- Pydantic 모델 (변경 없음) ---
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


# --- 동기 헬퍼 함수 (변경 없음) ---
def _fetch_user_profile_sync(conn: oracledb.Connection, user_id_num: int) -> Dict[str, Any]:
    user_profile = {}
    with conn.cursor() as cursor:
        try: # 전체 DB 작업에 대한 예외 처리 추가
            # 1. USERS 테이블에서 사용자 정보 조회
            logger.info(f"DB: Attempting to fetch user info for USER_ID: {user_id_num}")
            cursor.execute(
                """
                SELECT USER_ID, USER_NAME, NICKNAME, USER_EMAIL, CREATED_AT, AGE, GENDER, PHONE, ADDRESS
                FROM USERS WHERE USER_ID = :user_id_param
                """, user_id_param=user_id_num
            )
            user_row = cursor.fetchone()
            logger.info(f"DB: Result for USERS table (USER_ID: {user_id_num}): {user_row}")

            if user_row:
                user_profile["user_id"] = user_row[0]
                user_profile["name"] = user_row[1] if user_row[1] else user_row[2]
                user_profile["nickname"] = user_row[2]
                user_profile["email"] = user_row[3]
                user_profile["member_since"] = user_row[4].strftime("%Y-%m-%d") if isinstance(user_row[4], date) else None
                user_profile["age"] = user_row[5]
                user_profile["gender"] = user_row[6]
                user_profile["phone"] = user_row[7]
                user_profile["address"] = user_row[8]

                # 2. PET 테이블에서 반려동물 정보 조회
                logger.info(f"DB: Attempting to fetch PET info for USER_ID: {user_id_num}")
                cursor.execute(
                    """
                    SELECT PET_NAME, ANIMAL_TYPE, BREED, AGE, GENDER, NEUTERED, WEIGHT, REGISTRATION_DATE
                    FROM PET WHERE USER_ID = :owner_id_param
                    """, owner_id_param=user_id_num
                )
                pet_rows = cursor.fetchall()
                logger.info(f"DB: Fetched {len(pet_rows)} pet rows for USER_ID {user_id_num}: {pet_rows}")

                pet_info_list = []
                for pet_row in pet_rows:
                    pet_info_list.append({
                        "name": pet_row[0], "species": pet_row[1], "breed": pet_row[2],
                        "age": f"{pet_row[3]}세" if pet_row[3] else None, "gender": pet_row[4],
                        "neutered": pet_row[5], "weight": pet_row[6],
                        "registration_date": pet_row[7].strftime("%Y-%m-%d") if isinstance(pet_row[7], date) else None
                    })
                user_profile["pet_info"] = pet_info_list
            else:
                logger.warning(f"DB: No user found with USER_ID: {user_id_num}. Returning empty profile.")
                # 사용자가 없는 경우 pet_info는 기본적으로 빈 리스트가 됩니다.
                user_profile["pet_info"] = []
        except Exception as e:
            logger.error(f"DB: Error during user/pet profile fetch for USER_ID {user_id_num}: {e}", exc_info=True)
            # 오류 발생 시 기본 프로필 (비회원) 반환
            user_profile = {"name": "비회원", "user_id": str(user_id_num), "pet_info": []}
    return user_profile

# 👇 1. 이 함수를 chat 함수의 "의존성 함수"로 만듭니다.
#    이제 user_id를 request_data에서 직접 받습니다.
async def get_user_profile(
        request_data: ChatRequest, # Pydantic 모델을 직접 받도록 변경
        conn: oracledb.Connection = Depends(get_oracle_connection)
) -> Dict[str, Any]:
    default_profile = {"name": "비회원", "user_id": request_data.user_id}
    try:
        user_id_num = int(request_data.user_id)
    except ValueError:
        logger.error(f"Invalid user_id format: '{request_data.user_id}'.")
        return default_profile

    try:
        user_profile = await run_in_threadpool(
            _fetch_user_profile_sync, conn=conn, user_id_num=user_id_num
        )
        if not user_profile:
            logger.warning(f"User with ID '{request_data.user_id}' not found in DB.")
            return default_profile
        return user_profile
    except Exception as e:
        logger.error(f"Error in threadpool DB fetch for '{request_data.user_id}': {e}", exc_info=True)
        return default_profile

# --- 의존성 주입 함수 (get_chatbot, 변경 없음) ---
def get_chatbot(request: Request) -> RAGChatbot:
    if not hasattr(request.app.state, 'chatbot') or request.app.state.chatbot is None:
        raise HTTPException(status_code=503, detail="챗봇 서비스를 현재 사용할 수 없습니다.")
    return request.app.state.chatbot


# --- 라우터 엔드포인트 ---
@router.post("/chat", response_model=StandardResponse[ChatResponseData])
async def chat(
        request_data: ChatRequest, # request_data는 여전히 필요합니다. (get_user_profile 의존성 해결용)
        chatbot: RAGChatbot = Depends(get_chatbot),
        # 👇 2. get_user_profile 함수의 "결과"를 user_profile 변수에 바로 주입받습니다.
        user_profile: Dict[str, Any] = Depends(get_user_profile)
):
    logger.info(f"Chat request from user: {request_data.user_id}")
    try:
        # 이제 user_profile은 이미 DB 조회가 완료된 상태로 전달됩니다.
        response_data = await chatbot.ask(request_data.message, user_profile)
        response_model_data = ChatResponseData(**response_data)
        return create_success_response(data=response_model_data)
    except Exception as e:
        logger.error(f"Chat API 처리 중 에러 발생: {e}", exc_info=True)
        return create_error_response(
            error_code=ErrorCode.UNKNOWN_ERROR,
            message="요청을 처리하는 중 내부 오류가 발생했습니다."
        )