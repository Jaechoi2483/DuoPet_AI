"""
Face Login Router

This module provides endpoints for face recognition-based authentication.
"""

from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from services.face_login.predict import (
    register_face_image,
    verify_face_image,
    delete_face_embedding
)

router = APIRouter(tags=["Face Login"])

@router.post("/register")
async def register_face_endpoint(
    user_id: str = Query(..., description="User ID"),
    image: UploadFile = File(..., description="Face image")
):
    try:
        return await register_face_image(user_id, image, image.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"얼굴 등록 실패: {str(e)}")

@router.post("/verify")
async def verify_face_endpoint(
    user_id: str = Query(..., description="User ID"),
    image: UploadFile = File(..., description="Face image to verify")
):
    try:
        return await verify_face_image(user_id, image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"얼굴 인증 실패: {str(e)}")

@router.delete("/delete")
def delete_face_endpoint(user_id: str = Query(..., description="User ID")):
    try:
        return delete_face_embedding(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"얼굴 삭제 실패: {str(e)}")