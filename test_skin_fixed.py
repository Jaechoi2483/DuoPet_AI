"""
피부질환 서비스 테스트 - Graph/Eager mode 자동 처리 확인
"""
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

# 서비스 임포트
from services.skin_disease_service import get_skin_disease_service

def test_skin_disease_service():
    """피부질환 서비스 테스트"""
    
    print(f"TensorFlow {tf.__version__}")
    print(f"Eager execution: {tf.executing_eagerly()}")
    print("="*70)
    
    # 서비스 초기화
    service = get_skin_disease_service()
    
    # 테스트 이미지 생성
    test_cases = [
        ("Black image", np.zeros((224, 224, 3), dtype=np.uint8)),
        ("White image", np.ones((224, 224, 3), dtype=np.uint8) * 255),
        ("Random image", np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)),
        ("Red image", np.zeros((224, 224, 3), dtype=np.uint8) + [255, 0, 0]),
        ("Green image", np.zeros((224, 224, 3), dtype=np.uint8) + [0, 255, 0])
    ]
    
    print("\n=== Test Results ===")
    
    for test_name, image_array in test_cases:
        print(f"\n{test_name}:")
        
        # numpy array를 직접 예측
        result = service.predict(image_array.astype(np.float32), "dog")
        
        if result["binary_classification"]:
            normal_prob = result["binary_classification"]["normal"]
            disease_prob = result["binary_classification"]["disease"]
            print(f"  Normal: {normal_prob:.4f}, Disease: {disease_prob:.4f}")
            print(f"  Result: {result['disease_type']} (confidence: {result['confidence']:.4f})")
        else:
            print(f"  Error: No binary classification result")
        
        # Multi-class 결과도 출력
        if result["multi_classification"]:
            print("  Multi-class results:")
            for key, value in result["multi_classification"].items():
                print(f"    {key}: {value['class']} ({value['confidence']:.4f})")
    
    print("\n=== Analysis ===")
    print("1. Check if predictions vary between test cases")
    print("2. If all predictions are similar, the model weights might be corrupted")
    print("3. Graph/Eager mode handling should work transparently")

if __name__ == "__main__":
    test_skin_disease_service()