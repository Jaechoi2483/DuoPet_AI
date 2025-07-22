'''
services/face_login/predict.py
'''

import os

import cv2
import pandas as pd
import image
import numpy as np
import requests
from fastapi import UploadFile
from deepface import DeepFace
from common.logger import get_logger

from services.face_login.image_utils import load_image_from_uploadfile
from services.face_login.embedding_utils import load_all_embeddings
from services.face_login.score_utils import calculate_confidence
from services.face_login.face_login_config import get_face_login_config

config = get_face_login_config()
threshold = config.threshold
logger = get_logger(__name__)

EMBEDDINGS_DIR = config.base_path
THRESHOLD = config.threshold
SPRING_BASE_URL = config.spring_base_url
SPRING_API_KEY = config.spring_api_key


def _check_face_registered_in_spring(user_id: str) -> bool:
    """
    Spring 서버에 얼굴 등록 여부 확인
    """
    try:
        url = f"{SPRING_BASE_URL}/users/check-face"
        headers = {"Authorization": f"Bearer {SPRING_API_KEY}"} if SPRING_API_KEY else {}
        resp = requests.get(url, params={"userId": user_id}, headers=headers, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return bool(data.get("faceRegistered"))
    except Exception as e:
        print(f"[verify_face_image] Spring check-face 호출 실패: {e}")
        return False


import io
from PIL import Image

async def register_face_image(user_id: str, image_file: UploadFile, db_filename: str):
    """
    얼굴 이미지 등록 후 임베딩 벡터와 원본 이미지 저장
    """
    logger.info(f"[📸 얼굴 등록 시작] user_id={user_id}, 파일명={image_file.filename}")

    # 1. UploadFile → PIL 이미지 변환 (DeepFace용 저장에도 필요)
    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # 2. 저장 디렉토리 생성
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    # 3. 저장 경로 및 파일명 구성
    base_filename = f"{user_id}_{db_filename}"
    img_path = os.path.join(EMBEDDINGS_DIR, base_filename)
    embedding_path = img_path + ".npy"

    # 4. 이미지 파일(.png) 저장 → DeepFace.find()에서 사용될 DB
    image.save(img_path)
    logger.info(f"[💾 이미지 저장 완료] {img_path}")

    # 5. 얼굴 임베딩 벡터 추출 (저장한 이미지 경로 기반)
    embedding = DeepFace.represent(
        img_path=img_path,
        model_name="VGG-Face",
        enforce_detection=True
    )[0]["embedding"]

    # 6. 임베딩 벡터 저장
    np.save(embedding_path, embedding)
    logger.info(f"[💾 임베딩 저장 완료] {embedding_path}")

    return {
        "message": "얼굴 등록 성공!",
        "user_id": user_id,
        "saved_image_path": img_path,
        "saved_embedding_path": embedding_path
    }

# 나중에 제거 해도 됨
def extract_user_id_from_path(path: str) -> str:
    """
    예: C:/upload_files/face_embeddings/212/face_20250722_113000.png
    → '212' 추출
    """
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 2:
        return parts[-2]  # 상위 디렉토리명이 user_id
    return "unknown"

async def verify_face_image(image_file, db_dir):
    try:
        # 1. 업로드된 이미지를 OpenCV가 읽을 수 있는 형태로 변환
        contents = await image_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        unknown_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 이미지가 정상적으로 디코딩되었는지 확인
        if unknown_img is None:
            logger.error("업로드된 이미지 파일을 디코딩할 수 없습니다.")
            return {"verified": False, "user_id": None, "error": "Invalid image file."}

        threshold = 0.35

        # 2. DB에 저장된 모든 얼굴 이미지와 하나씩 비교
        for file_name in os.listdir(db_dir):
            # 이미지 파일만 대상으로 함 (예: .jpg, .png)
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                known_img_path = os.path.join(db_dir, file_name)
                user_id = file_name.split('_')[0]

                try:
                    # DeepFace.verify 함수로 두 얼굴 이미지 비교
                    # enforce_detection=True (기본값) : 얼굴이 없으면 에러 발생
                    result = DeepFace.verify(
                        img1_path=unknown_img,
                        img2_path=known_img_path,
                        model_name="VGG-Face", # 또는 "Facenet", "ArcFace" 등
                        enforce_detection=True,
                        distance_metric="cosine"
                    )

                    distance = result["distance"]
                    confidence = calculate_confidence(distance, threshold)

                    if distance < threshold:
                        logger.info(f"인증 성공: {user_id} (거리: {distance}, 신뢰도: {confidence}%)")

                        if _check_face_registered_in_spring(user_id):
                            logger.info(f"Spring 사용자 확인됨: {user_id}")
                            return {"verified": True, "user_id": user_id, "confidence": confidence}
                        else:
                            logger.warning(f"⚠Spring DB에 얼굴 미등록: {user_id}")
                            return {"verified": False, "user_id": None, "error": "등록되지 않은 사용자입니다."}

                except ValueError as e:
                    logger.warning(f"'{known_img_path}' 또는 업로드 이미지에서 얼굴 탐지 실패: {e}")
                    continue

        logger.info("❌ 인증 실패: 일치하는 사용자를 찾지 못했습니다.")
        return {"verified": False, "user_id": None, "error": "등록된 얼굴 정보와 일치하는 사용자를 찾을 수 없습니다."
                                                             "\n다시 시도해주세요."}

    except Exception as e:
        logger.error(f"🔥 얼굴 인증 중 심각한 예외 발생: {e}")
        return {"verified": False, "user_id": None, "error": f"An unexpected error occurred: {str(e)}"}


def delete_face_embedding(user_id: str):
    """
    특정 사용자 ID에 대한 얼굴 이미지 및 임베딩 파일 삭제
    """
    deleted_files = []

    if not os.path.exists(EMBEDDINGS_DIR):
        return {"message": "저장 경로가 존재하지 않습니다.", "deleted": []}

    target_prefix = f"{user_id}_"

    for fname in os.listdir(EMBEDDINGS_DIR):
        if fname.startswith(target_prefix) and (fname.endswith(".npy") or fname.endswith(".png") or fname.endswith(".jpg")):
            path = os.path.join(EMBEDDINGS_DIR, fname)
            try:
                os.remove(path)
                deleted_files.append(fname)
            except Exception as e:
                print(f"[delete_face_embedding] 삭제 실패 - {fname}: {e}")

    return {
        "message": f"{len(deleted_files)}개 파일 삭제 완료",
        "deleted_files": deleted_files
    }
