"""
Face Login Router

This module provides endpoints for face recognition-based authentication.
"""

from fastapi import APIRouter, UploadFile, File, Query, HTTPException, Form
from services.face_login.face_login_config import get_face_login_config
from common.logger import get_logger
from services.face_login.predict import (
    register_face_image,
    verify_face_image,
    delete_face_embedding
)

logger = get_logger(__name__)
config = get_face_login_config()
EMBEDDING_DIR = config.base_path

router = APIRouter(tags=["Face Login"])

# 얼굴 등록
@router.post("/register")
async def register_face_endpoint(
    user_id: str = Form(..., description="User ID"),
    image: UploadFile = File(..., description="Face image")
):
    try:
        return await register_face_image(user_id, image, image.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"얼굴 등록 실패: {str(e)}")

# 얼굴 인증 (user_id 없이 전체 중 유사도 매칭)
@router.post("/verify")
async def verify_face_endpoint(
    image: UploadFile = File(..., description="Face image to verify")
):
    try:
        logger.info(f"[🚀 얼굴 인증 요청] 이미지={image.filename}")
        result = await verify_face_image(image_file=image, db_dir=EMBEDDING_DIR)
        logger.info(f"[✅ 얼굴 인증 결과] {result}")
        return result
    except Exception as e:
        logger.error(f"[🔥 얼굴 인증 실패] {str(e)}")
        raise HTTPException(status_code=500, detail=f"얼굴 인증 실패: {str(e)}")

# 얼굴 삭제
@router.delete("/delete")
async def delete_face_endpoint(user_id: str = Query(..., description="User ID")):
    try:
        logger.info(f"[🗑 얼굴 삭제 요청] user_id={user_id}")
        result = delete_face_embedding(user_id)

        if result["deleted_files"]:
            logger.info(f"[✅ 삭제 완료] {len(result['deleted_files'])}개 삭제됨")
        else:
            logger.info(f"[⚠️ 삭제할 파일 없음] user_id={user_id}")

        return result

    except Exception as e:
        logger.error(f"[🔥 얼굴 삭제 실패] {str(e)}")
        raise HTTPException(status_code=500, detail=f"얼굴 삭제 실패: {str(e)}")