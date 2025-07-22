'''
services/face_login/predict.py
'''

import os
import numpy as np
import requests
from fastapi import UploadFile
from deepface import DeepFace

from services.face_login.image_utils import load_image_from_uploadfile
from services.face_login.face_login_config import get_face_login_config

config = get_face_login_config()

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


async def register_face_image(user_id: str, image_file: UploadFile, db_filename: str):
    """
    얼굴 이미지 등록 후 임베딩 벡터 저장
    """
    img = await load_image_from_uploadfile(image_file)

    embedding = DeepFace.represent(
        img_path=img,
        model_name="VGG-Face",
        enforce_detection=True
    )[0]["embedding"]

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    save_path = os.path.join(EMBEDDINGS_DIR, f"{user_id}_{db_filename}.npy")
    np.save(save_path, embedding)

    return {
        "message": "얼굴 등록 성공!",
        "user_id": user_id,
        "saved_path": save_path
    }


async def verify_face_image(user_id: str, image_file: UploadFile):
    """
    얼굴 인증 로직
    """
    if not _check_face_registered_in_spring(user_id):
        return {"error": "얼굴 등록이 되어있지 않습니다.", "user_id": user_id}

    img = await load_image_from_uploadfile(image_file)

    current_embedding = DeepFace.represent(
        img_path=img,
        model_name="VGG-Face",
        enforce_detection=True
    )[0]["embedding"]

    target_prefix = f"{user_id}_"
    candidate_files = [
        f for f in os.listdir(EMBEDDINGS_DIR)
        if f.startswith(target_prefix) and f.endswith(".npy")
    ]

    if not candidate_files:
        return {"error": "등록된 얼굴 임베딩이 없습니다.", "user_id": user_id}

    best_score = -1
    best_file = None
    for fname in candidate_files:
        stored = np.load(os.path.join(EMBEDDINGS_DIR, fname))
        sim = np.dot(current_embedding, stored) / (
            np.linalg.norm(current_embedding) * np.linalg.norm(stored)
        )
        if sim > best_score:
            best_score = sim
            best_file = fname

    if best_score >= THRESHOLD:
        return {
            "verified_user": user_id,
            "score": round(float(best_score), 4),
            "matched_file": best_file
        }

    return {
        "error": "얼굴이 일치하지 않습니다.",
        "user_id": user_id,
        "score": round(float(best_score), 4)
    }


def delete_face_embedding(user_id: str):
    """
    특정 사용자 ID에 대한 모든 얼굴 임베딩 파일 삭제
    """
    deleted_files = []

    if not os.path.exists(EMBEDDINGS_DIR):
        return {"message": "저장 경로가 존재하지 않습니다.", "deleted": []}

    target_prefix = f"{user_id}_"

    for fname in os.listdir(EMBEDDINGS_DIR):
        if fname.startswith(target_prefix) and fname.endswith(".npy"):
            path = os.path.join(EMBEDDINGS_DIR, fname)
            try:
                os.remove(path)
                deleted_files.append(fname)
            except Exception as e:
                print(f"[delete_face_embedding] 삭제 실패 - {fname}: {e}")

    return {
        "message": f"{len(deleted_files)}개 얼굴 벡터 삭제 완료",
        "deleted_files": deleted_files
    }
