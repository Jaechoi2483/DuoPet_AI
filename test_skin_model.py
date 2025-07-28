"""
피부질환 모델 테스트 스크립트
다양한 입력으로 모델 동작 확인
"""
import numpy as np
from services.skin_disease_service import SkinDiseaseService
from PIL import Image
import io

def test_model_with_various_inputs():
    """다양한 입력으로 모델 테스트"""
    service = SkinDiseaseService()
    
    test_cases = [
        ("완전 검은색", np.zeros((224, 224, 3), dtype=np.uint8)),
        ("완전 흰색", np.ones((224, 224, 3), dtype=np.uint8) * 255),
        ("빨간색", np.zeros((224, 224, 3), dtype=np.uint8)),
        ("랜덤 노이즈", np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)),
        ("체커보드 패턴", np.zeros((224, 224, 3), dtype=np.uint8))
    ]
    
    # 빨간색 이미지
    test_cases[2][1][:, :, 0] = 255  # R 채널만 255
    
    # 체커보드 패턴
    for i in range(0, 224, 16):
        for j in range(0, 224, 16):
            if (i//16 + j//16) % 2 == 0:
                test_cases[4][1][i:i+16, j:j+16] = 255
    
    print("=== 피부질환 모델 테스트 시작 ===\n")
    
    for name, img_array in test_cases:
        print(f"테스트: {name}")
        
        # numpy array를 파일 객체로 변환
        img = Image.fromarray(img_array)
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # 예측
        result = service.predict(img_array.astype(np.float32), "dog")
        
        if result["binary_classification"]:
            normal_prob = result["binary_classification"]["normal"]
            disease_prob = result["binary_classification"]["disease"]
            print(f"  - 정상: {normal_prob:.4f}, 질환: {disease_prob:.4f}")
            print(f"  - 판정: {result['disease_type']}, 신뢰도: {result['confidence']:.4f}")
        else:
            print(f"  - 예측 실패")
        
        print()
    
    print("=== 테스트 완료 ===")

if __name__ == "__main__":
    test_model_with_various_inputs()