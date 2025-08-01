import base64
import httpx
import asyncio
import json
from typing import List, Optional

from fastapi import APIRouter, Depends, Form, UploadFile, File
from common.config import get_settings, Settings
from common.logger import get_logger # 로거 임포트
from common.response import create_success_response
from common.exceptions import DuoPetException, ErrorCode

# 라우터, 로거, 설정 초기화
router = APIRouter()
logger = get_logger(__name__) # 로거 객체 사용
settings = get_settings()

# --- 프롬프트 엔지니어링: AI에게 역할을 부여하고, 원하는 답변 형식을 지정합니다 ---
SYSTEM_PROMPT_TEMPLATE = """
당신은 반려동물의 건강 상태를 이미지와 증상 설명을 기반으로 잠정적으로 분석하는 AI 어시스턴트입니다.
당신은 수의사가 아니며, 당신의 답변은 절대 전문적인 의료 진단을 대체할 수 없습니다.
사용자가 제공한 정보를 바탕으로, 다음 JSON 형식에 맞춰 답변을 생성해주세요.

{
  "potential_conditions": ["의심되는 질환명 1", "의심되는 질환명 2"],
  "severity": "mild, moderate, 또는 severe 중 하나로 평가",
  "explanation": "왜 그렇게 생각하는지에 대한 상세하고 전문적인 설명 (2-3문장)",
  "recommendations": ["보호자가 즉시 할 수 있는 권장 조치 목록"],
  "requires_vet_visit": true 또는 false (동물병원 방문이 필요한지에 대한 판단)
}

모든 답변은 한국어로 작성하고, 항상 동물병원 방문을 권장하는 내용을 포함해야 합니다.
"""


def create_user_prompt(
        diagnosis_type: str,
        animal_type: str,
        symptoms: str,
        pet_age: Optional[str] = None,
        pet_breed: Optional[str] = None,
        pet_weight: Optional[str] = None
) -> str:
    """사용자 정보를 바탕으로 AI에게 보낼 프롬프트를 생성합니다."""
    prompt = f"진단 유형: {diagnosis_type}\n"
    prompt += f"반려동물 종류: {animal_type}\n"
    if pet_age: prompt += f"나이: {pet_age}\n"
    if pet_breed: prompt += f"품종: {pet_breed}\n"
    if pet_weight: prompt += f"체중: {pet_weight}\n"
    prompt += f"보호자가 설명한 증상: {symptoms}\n\n"
    prompt += "위 정보와 첨부된 이미지를 종합적으로 분석하여, 주어진 JSON 형식에 맞춰 잠정적인 소견을 알려주세요."
    return prompt


@router.post("/analyze")
async def analyze_health_from_images(
        files: List[UploadFile] = File(...),
        diagnosis_type: str = Form(...),
        animal_type: str = Form(..., alias="pet_type"),
        symptoms: str = Form(...),
        pet_age: Optional[str] = Form(None),
        pet_breed: Optional[str] = Form(None),
        pet_weight: Optional[str] = Form(None),
        settings: Settings = Depends(get_settings),
):
    """
    이미지와 반려동물 정보를 받아 GPT-4o Vision API로 분석을 요청합니다.
    """
    logger.info(f"AI 건강 진단 요청 접수: 진단 유형='{diagnosis_type}', 반려동물 종류='{animal_type}'")
    logger.debug(f"증상: '{symptoms}', 나이: '{pet_age}', 품종: '{pet_breed}', 체중: '{pet_weight}'")

    if not settings.OPENAI_API_KEY:
        logger.error("OpenAI API 키가 설정되지 않았습니다.")
        raise DuoPetException(ErrorCode.MODEL_NOT_CONFIGURED, "OpenAI API 키가 설정되지 않았습니다.")

    # 1. 이미지 파일들을 Base64로 인코딩
    async def encode_image(file: UploadFile):
        contents = await file.read()
        return base64.b64encode(contents).decode("utf-8")

    base64_images = await asyncio.gather(*(encode_image(file) for file in files))
    logger.info(f"총 {len(base64_images)}개의 이미지 파일이 Base64로 인코딩되었습니다.")

    # 2. OpenAI API에 보낼 데이터 구성
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
    }

    image_contents = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
        for img in base64_images
    ]

    user_prompt = create_user_prompt(
        diagnosis_type=diagnosis_type,
        animal_type=animal_type,
        symptoms=symptoms,
        pet_age=pet_age,
        pet_breed=pet_breed,
        pet_weight=pet_weight
    )
    text_content = {"type": "text", "text": user_prompt}

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
            {"role": "user", "content": [text_content, *image_contents]}
        ],
        "max_tokens": 1000,
        "response_format": {"type": "json_object"}
    }
    logger.debug("OpenAI API 요청 페이로드 생성 완료.")

    # 3. OpenAI API 비동기 호출
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()

            gpt_response = response.json()
            analysis_result_str = gpt_response["choices"][0]["message"]["content"]
            analysis_result_json = json.loads(analysis_result_str)

            # 성공적인 AI 진단 결과 로깅
            logger.info("AI 진단이 성공적으로 완료되었습니다.")
            logger.info(f"진단 결과: {json.dumps(analysis_result_json, ensure_ascii=False, indent=2)}")

            return create_success_response(data={"results": analysis_result_json})

    # 4. 에러 처리
    except httpx.HTTPStatusError as e:
        error_message = f"OpenAI API 에러: {e.response.text}"
        logger.error(error_message)
        raise DuoPetException(ErrorCode.MODEL_INFERENCE_ERROR, "OpenAI API로부터 에러 응답을 받았습니다.")
    except json.JSONDecodeError as e:
        error_message = f"OpenAI 응답 JSON 파싱 오류: {e}. 응답 내용: {analysis_result_str}"
        logger.error(error_message, exc_info=True)
        raise DuoPetException(ErrorCode.MODEL_INFERENCE_ERROR, "AI 응답 형식이 올바르지 않습니다.")
    except Exception as e:
        error_message = f"AI 분석 중 예상치 못한 오류 발생: {e}"
        logger.error(error_message, exc_info=True)
        raise DuoPetException(ErrorCode.UNKNOWN_ERROR, "AI 분석 중 예상치 못한 오류가 발생했습니다.")