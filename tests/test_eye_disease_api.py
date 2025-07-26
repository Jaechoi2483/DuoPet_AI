
import pytest
from fastapi.testclient import TestClient
from api.main import app
import os
from PIL import Image

# 테스트 클라이언트 생성
client = TestClient(app)

# 테스트에 사용할 이미지 파일 경로
ASSETS_DIR = "D:/final_project/DuoPet_AI/tests/assets"
TEST_IMAGE_PATH = os.path.join(ASSETS_DIR, "test_image.png")

@pytest.fixture(scope="module")
def setup_test_image():
    """
    테스트용 이미지 파일을 생성하고, 테스트 완료 후 정리합니다.
    """
    # assets 디렉토리 생성
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)

    # 10x10 크기의 검은색 PNG 이미지 생성
    img = Image.new('RGB', (10, 10), color = 'black')
    img.save(TEST_IMAGE_PATH, 'PNG')

    yield TEST_IMAGE_PATH

    # 테스트 완료 후 파일 및 디렉토리 삭제
    if os.path.exists(TEST_IMAGE_PATH):
        os.remove(TEST_IMAGE_PATH)
    if os.path.exists(ASSETS_DIR) and not os.listdir(ASSETS_DIR):
        os.rmdir(ASSETS_DIR)

def test_analyze_eye_disease_success(setup_test_image):
    """안구 질환 분석 API 성공 케이스 테스트"""
    test_image_path = setup_test_image
    with open(test_image_path, "rb") as f:
        files = {"image": ("test_image.png", f, "image/png")}
        response = client.post("/api/v1/health-diagnose/analyze/eye", files=files)

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["message"] == "안구질환 분석이 완료되었습니다."
    assert "data" in data

    result = data["data"]
    assert "disease" in result
    assert "confidence" in result
    assert isinstance(result["disease"], str)
    assert isinstance(result["confidence"], float)
    assert 0 <= result["confidence"] <= 1

def test_analyze_eye_disease_invalid_format():
    """잘못된 파일 형식에 대한 에러 처리 테스트"""
    # 유효하지 않은 파일(예: 파이썬 스크립트)을 전송
    with open(__file__, "rb") as f:
        files = {"image": ("test_script.py", f, "text/x-python")}
        response = client.post("/api/v1/health-diagnose/analyze/eye", files=files)

    # 서버는 400 Bad Request를 반환해야 함
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "Invalid image format" in data["message"]

def test_health_check():
    """헬스 체크 엔드포인트 테스트"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "data" in data
    assert "status" in data["data"]
