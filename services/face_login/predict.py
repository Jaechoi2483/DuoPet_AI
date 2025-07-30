'''
services/face_login/predict.py
'''

import os

import cv2
import pandas as pd
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


import tensorflow as tf
tf.config.run_functions_eagerly(True)
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
    # 임시 파일 경로 설정
    temp_img_path = "temp_face_for_verify.png"

    try:
        # 1. 업로드된 이미지를 임시 파일로 저장 (DeepFace.find는 파일 경로를 요구)
        contents = await image_file.read()
        with open(temp_img_path, "wb") as f:
            f.write(contents)

        # 2. DeepFace.find()를 한 번만 호출하여 DB 전체에서 가장 유사한 얼굴 검색
        # find 결과는 DataFrame을 포함한 리스트. 비어있을 수 있음.
        result_dfs = DeepFace.find(
            img_path=temp_img_path,
            db_path=db_dir,
            model_name="VGG-Face",
            distance_metric="cosine",
            enforce_detection=True
        )

        # 3. 검색 결과 분석
        # find는 여러 얼굴을 찾을 수 있으므로 첫 번째 결과(가장 큰 DataFrame)를 사용
        if result_dfs and not result_dfs[0].empty:
            closest_match_df = result_dfs[0]
            # 가장 거리가 가까운 (첫 번째) 행 선택
            closest_match = closest_match_df.iloc[0]
            distance = closest_match["distance"]

            if distance < THRESHOLD:
                identity_path = closest_match["identity"]
                # 파일 경로에서 user_id 추출 (경로 구분자 통일)
                user_id = os.path.basename(identity_path).split('_')[0]

                # Spring 서버에 등록된 사용자인지 확인
                if _check_face_registered_in_spring(user_id):
                    logger.info(f"✅ 인증 성공: {user_id} (거리: {distance})")
                    return {"verified": True, "user_id": user_id}
                else:
                    logger.warning(f"⚠ Spring DB에 얼굴 미등록: {user_id}")
                    return {"verified": False, "user_id": None, "error": "등록되지 않은 사용자입니다."}

        # 4. 일치하는 얼굴이 없는 경우
        logger.info("❌ 인증 실패: 일치하는 사용자를 찾지 못했습니다.")
        return {"verified": False, "user_id": None, "error": "등록된 얼굴 정보와 일치하는 사용자를 찾을 수 없습니다."}

    except ValueError as ve:
        # DeepFace에서 얼굴을 찾지 못했을 때 발생하는 일반적인 오류
        logger.warning(f"⚠ 업로드된 이미지에서 얼굴 탐지 실패: {ve}")
        return {"verified": False, "user_id": None, "error": "이미지에서 얼굴을 찾을 수 없습니다."}
    except Exception as e:
        logger.error(f"🔥 얼굴 인증 중 심각한 예외 발생: {e}")
        return {"verified": False, "user_id": None, "error": f"An unexpected error occurred: {str(e)}"}
    finally:
        # 5. 임시 파일이 존재하면 삭제
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)


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
