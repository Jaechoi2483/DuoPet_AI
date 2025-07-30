import os
import io
import cv2
import numpy as np
import requests
import pandas as pd
import tensorflow as tf
from PIL import Image
from fastapi import UploadFile
from deepface import DeepFace
from deepface.basemodels import VGGFace
from deepface.commons import functions

from common.logger import get_logger
from sklearn.metrics.pairwise import cosine_distances
from services.face_login.image_utils import load_image_from_uploadfile
from services.face_login.embedding_utils import load_all_embeddings
from services.face_login.score_utils import calculate_confidence
from services.face_login.face_login_config import get_face_login_config
from tensorflow.python.framework import tensor_util

# TensorFlow eager execution 설정 (2.14 대응)
tf.config.run_functions_eagerly(True)

# 환경 설정
token_config = get_face_login_config()
logger = get_logger(__name__)

EMBEDDINGS_DIR = token_config.base_path
THRESHOLD = token_config.threshold
SPRING_BASE_URL = token_config.spring_base_url
SPRING_API_KEY = token_config.spring_api_key


def _check_face_registered_in_spring(user_id: str) -> bool:
    try:
        url = f"{SPRING_BASE_URL}/users/check-face"
        headers = {"Authorization": f"Bearer {SPRING_API_KEY}"} if SPRING_API_KEY else {}
        resp = requests.get(url, params={"userId": user_id}, headers=headers, timeout=5)
        resp.raise_for_status()
        return bool(resp.json().get("faceRegistered"))
    except Exception as e:
        logger.warning(f"[verify_face_image] Spring check-face 호출 실패: {e}")
        return False


# ✅ [추가됨] VGGFace 임베딩 수동 추출 함수 (TensorFlow 2.14 대응용)
def extract_vggface_embedding(image_path: str) -> np.ndarray:
    model = VGGFace.loadModel()

    face_objs = functions.extract_faces(
        img=image_path,
        target_size=(224, 224),
        enforce_detection=True,
        detector_backend='opencv'
    )

    if not face_objs:
        raise ValueError("No face found in image")

    face_img = face_objs[0][0]
    face_input = face_img if face_img.ndim == 4 else np.expand_dims(face_img, axis=0)

    # ✅ SymbolicTensor 우회 실행
    get_embedding = tf.keras.backend.function(model.input, model.output)
    embedding = get_embedding(face_input)  # 이건 바로 실행돼서 NumPy 반환함

    return embedding[0]


async def register_face_image(user_id: str, image_file: UploadFile, db_filename: str):
    logger.info(f"[📸 얼굴 등록 시작] user_id={user_id}, 파일명={image_file.filename}")

    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    base_filename = f"{user_id}_{db_filename}"
    img_path = os.path.join(EMBEDDINGS_DIR, base_filename)
    embedding_path = img_path + ".npy"

    image.save(img_path)
    logger.info(f"[💾 이미지 저장 완료] {img_path}")

    # ✅ 임베딩 추출
    embedding = extract_vggface_embedding(img_path)

    # ✅ [여기] 임베딩 상태 확인 + 저장 시도
    logger.info(f"[디버그] embedding type: {type(embedding)}")
    logger.info(f"[디버그] embedding shape: {getattr(embedding, 'shape', 'no shape')}")

    try:
        np.save(embedding_path, embedding)
        logger.info(f"[💾 임베딩 저장 완료] {embedding_path}")
    except Exception as e:
        logger.error(f"[❌ 임베딩 저장 실패] {embedding_path} → {e}")

    return {
        "message": "얼굴 등록 성공!",
        "user_id": user_id,
        "saved_image_path": img_path,
        "saved_embedding_path": embedding_path
    }

def extract_user_id_from_path(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


async def verify_face_image(image_file, db_dir):
    try:
        contents = await image_file.read()
        temp_path = os.path.join(EMBEDDINGS_DIR, "temp_verify.png")
        with open(temp_path, "wb") as f:
            f.write(contents)

        # 업로드 이미지로부터 임베딩 추출
        try:
            unknown_embedding = extract_vggface_embedding(temp_path)
        except Exception as e:
            logger.warning(f"❌ 업로드 이미지에서 얼굴 추출 실패: {e}")
            return {"verified": False, "user_id": None, "error": "얼굴을 감지할 수 없습니다."}
        finally:
            os.remove(temp_path)

        for file in os.listdir(db_dir):
            if file.endswith(".npy") and "_" in file:
                user_id = file.split("_")[0]
                known_embedding = np.load(os.path.join(db_dir, file))

                dist = cosine_distances([unknown_embedding], [known_embedding])[0][0]
                confidence = calculate_confidence(dist, THRESHOLD)

                if dist < THRESHOLD:
                    logger.info(f"인증 성공: {user_id} (거리: {dist}, 신뢰도: {confidence}%)")
                    if _check_face_registered_in_spring(user_id):
                        return {"verified": True, "user_id": user_id, "confidence": confidence}

        return {"verified": False, "user_id": None, "error": "일치하는 사용자 없음"}

    except Exception as e:
        logger.error(f"🔥 얼굴 인증 중 예외 발생: {e}")
        return {"verified": False, "user_id": None, "error": f"예외 발생: {str(e)}"}



def delete_face_embedding(user_id: str):
    deleted_files = []
    if not os.path.exists(EMBEDDINGS_DIR):
        return {"message": "저장 경로가 존재하지 않습니다.", "deleted": []}

    target_prefix = f"{user_id}_"
    for fname in os.listdir(EMBEDDINGS_DIR):
        if fname.startswith(target_prefix) and fname.endswith((".npy", ".png", ".jpg")):
            path = os.path.join(EMBEDDINGS_DIR, fname)
            try:
                os.remove(path)
                deleted_files.append(fname)
            except Exception as e:
                logger.warning(f"[delete_face_embedding] 삭제 실패 - {fname}: {e}")

    return {
        "message": f"{len(deleted_files)}개 파일 삭제 완료",
        "deleted_files": deleted_files
    }
