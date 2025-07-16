from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request, Security
from fastapi.security import APIKeyHeader # 💡 APIKeyHeader 임포트
import secrets # 💡 보안을 위한 secrets 임포트

from common.response import create_success_response, StandardResponse
from common.config import get_settings, Settings # 💡 Settings 임포트

# 새로운 관리자용 라우터 생성
router = APIRouter(
    prefix="/admin",
    tags=["Admin"],
)

# ▼▼▼ 기존 admin_key_required 함수를 이 내용으로 대체하거나 새로 만듭니다 ▼▼▼
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

async def verify_admin_api_key(
    api_key: str = Security(api_key_header),
    settings: Settings = Depends(get_settings)
):
    """
    요청 헤더의 X-API-KEY가 .env 파일의 ADMIN_API_KEY와 일치하는지 확인합니다.
    """
    if not api_key or not secrets.compare_digest(api_key, settings.ADMIN_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="올바르지 않거나 누락된 관리자 API 키입니다."
        )


# 2. 챗봇 데이터 업데이트를 실행하는 API 엔드포인트
@router.post(
    "/chatbot/resync",
    summary="[관리자] 챗봇 데이터 동기화",
    description="사이트를 새로 크롤링하여 챗봇의 지식 베이스를 강제로 업데이트합니다. 백그라운드에서 실행됩니다.",
    response_model=StandardResponse,
    # ▼▼▼ 의존성(Depends)을 위에서 만든 새 함수로 교체합니다. ▼▼▼
    dependencies=[Depends(verify_admin_api_key)]
)
async def trigger_chatbot_resync(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    .env 파일의 ADMIN_API_KEY로 인증된 경우에만 챗봇 데이터 업데이트를 실행합니다.
    """
    chatbot_instance = request.app.state.chatbot
    if not chatbot_instance:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="챗봇 인스턴스가 초기화되지 않아 작업을 시작할 수 없습니다."
        )

    background_tasks.add_task(chatbot_instance.resync_data_from_site)

    return create_success_response(
        data={"message": "챗봇 데이터 업데이트 작업이 백그라운드에서 시작되었습니다."}
    )