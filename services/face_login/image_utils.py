'''
# services/face_login/image_utils.py
# 이미지 처리 유틸
'''

from fastapi import UploadFile
import numpy as np
import cv2

async def load_image_from_uploadfile(file: UploadFile):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image