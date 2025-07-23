'''
services/face_login/utils/embedding_utils.py

- 얼굴 임베딩 벡터 저장 및 불러오기 기능 제공
- 설정은 face_login_config.py에서 불러온 config.base_path를 따름
'''

import os
import numpy as np
from services.face_login.face_login_config import get_face_login_config

config = get_face_login_config()
EMBEDDING_DIR = config.base_path

def load_all_embeddings():
    """
        저장된 모든 사용자 임베딩 로드
        - 파일명 형식: {user_id}_embedding.npy
        - 반환 형식: List of (user_id, embedding)
    """
    embeddings = []
    if not os.path.exists(EMBEDDING_DIR):
        return embeddings

    for filename in os.listdir(EMBEDDING_DIR):
        if filename.endswith("_embedding.npy"):
            user_id = filename.replace("_embedding.npy", "")
            embedding_path = os.path.join(EMBEDDING_DIR, filename)
            try:
                embedding = np.load(embedding_path)
                embeddings.append((user_id, embedding))
            except Exception as e:
                print(f"[❌ 임베딩 로드 실패] {filename}: {e}")
    return embeddings

def save_user_embedding(user_id: str, embedding: np.ndarray):
    """
    사용자 임베딩 벡터 저장
    - 저장 경로: {base_path}/{user_id}_embedding.npy
    """
    if not os.path.exists(EMBEDDING_DIR):
        os.makedirs(EMBEDDING_DIR)

    save_path = os.path.join(EMBEDDING_DIR, f"{user_id}_embedding.npy")
    try:
        np.save(save_path, embedding)
        print(f"[✅ 임베딩 저장 완료] {save_path}")
    except Exception as e:
        print(f"[❌ 임베딩 저장 실패] {e}")