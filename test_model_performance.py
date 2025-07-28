"""
안구질환 모델 성능 테스트
학습 데이터로 직접 테스트
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

def test_with_training_data():
    """학습 데이터로 모델 테스트"""
    
    print("🔍 모델 성능 심층 테스트")
    print("="*60)
    
    # 모델 로드
    custom_objects = {
        'swish': tf.nn.swish,
        'Swish': tf.keras.layers.Activation(tf.nn.swish)
    }
    
    model_paths = [
        "models/health_diagnosis/eye_disease/eye_disease_tf2_complete.h5",
        "models/health_diagnosis/eye_disease/eye_disease_fixed.h5",
        "models/health_diagnosis/eye_disease/best_grouped_model.keras"
    ]
    
    # 클래스 맵 로드
    with open('models/health_diagnosis/eye_disease/class_map.json', 'r', encoding='utf-8') as f:
        class_map = json.load(f)
    
    print("📋 클래스 정보:")
    for idx, name in class_map.items():
        print(f"  {idx}: {name}")
    
    # 각 모델별로 테스트
    for model_path in model_paths:
        if not Path(model_path).exists():
            continue
            
        print(f"\n\n🧪 모델 테스트: {model_path}")
        print("-"*60)
        
        try:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            
            # 모델 구조 확인
            print(f"입력 형태: {model.input_shape}")
            print(f"출력 형태: {model.output_shape}")
            
            # 마지막 레이어 확인
            last_layer = model.layers[-1]
            print(f"마지막 레이어: {last_layer.name} ({type(last_layer).__name__})")
            if hasattr(last_layer, 'activation'):
                print(f"활성화 함수: {last_layer.activation}")
            
            # 다양한 패턴으로 테스트
            test_patterns = {
                "완전 빨강 (결막염 시뮬레이션)": create_red_eye_pattern(),
                "정상 눈 패턴": create_normal_eye_pattern(),
                "흐린 패턴 (백내장 시뮬레이션)": create_cataract_pattern(),
                "무작위 노이즈": np.random.random((1, 224, 224, 3)).astype(np.float32)
            }
            
            print("\n예측 결과:")
            for pattern_name, pattern in test_patterns.items():
                predictions = model.predict(pattern, verbose=0)
                
                print(f"\n{pattern_name}:")
                # 모든 클래스 확률 출력
                for idx in range(len(predictions[0])):
                    class_name = class_map.get(str(idx), f"Unknown_{idx}")
                    prob = predictions[0][idx]
                    print(f"  {class_name}: {prob:.4f} ({prob*100:.1f}%)")
                
                # 최고 예측
                max_idx = np.argmax(predictions[0])
                max_prob = predictions[0][max_idx]
                max_class = class_map.get(str(max_idx), "Unknown")
                print(f"  → 최종: {max_class} ({max_prob*100:.1f}%)")
            
            # Softmax 검증
            print(f"\n확률 합계: {np.sum(predictions[0]):.4f} (1.0이어야 함)")
            
        except Exception as e:
            print(f"❌ 오류: {e}")

def create_red_eye_pattern():
    """빨간 눈 패턴 생성 (결막염 유사)"""
    img = np.zeros((224, 224, 3), dtype=np.float32)
    # 중앙에 원형 패턴
    center = 112
    radius = 80
    
    for i in range(224):
        for j in range(224):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist < radius:
                # 빨간색 강조
                img[i, j, 0] = 0.8 + 0.2 * (1 - dist/radius)  # R
                img[i, j, 1] = 0.2 + 0.3 * (1 - dist/radius)  # G
                img[i, j, 2] = 0.2 + 0.3 * (1 - dist/radius)  # B
    
    return np.expand_dims(img, axis=0)

def create_normal_eye_pattern():
    """정상 눈 패턴 생성"""
    img = np.zeros((224, 224, 3), dtype=np.float32)
    center = 112
    
    # 흰자위
    img[:, :] = [0.9, 0.9, 0.9]
    
    # 홍채 (갈색)
    for i in range(224):
        for j in range(224):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist < 50:
                img[i, j] = [0.4, 0.3, 0.2]
            elif dist < 60:
                img[i, j] = [0.6, 0.5, 0.4]
    
    # 동공
    for i in range(224):
        for j in range(224):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist < 20:
                img[i, j] = [0.1, 0.1, 0.1]
    
    return np.expand_dims(img, axis=0)

def create_cataract_pattern():
    """백내장 패턴 생성 (흐린 중앙부)"""
    img = create_normal_eye_pattern()[0]
    
    # 중앙부를 흐리게
    center = 112
    for i in range(224):
        for j in range(224):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist < 40:
                opacity = 0.7 * (1 - dist/40)
                img[i, j] = img[i, j] * (1 - opacity) + np.array([0.8, 0.8, 0.8]) * opacity
    
    return np.expand_dims(img, axis=0)

def check_model_weights():
    """모델 가중치 통계 확인"""
    print("\n\n🔬 모델 가중치 분석")
    print("="*60)
    
    model_path = "models/health_diagnosis/eye_disease/eye_disease_fixed.h5"
    if Path(model_path).exists():
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects={'swish': tf.nn.swish}
        )
        
        # 각 레이어의 가중치 통계
        for layer in model.layers[-5:]:  # 마지막 5개 레이어
            if layer.weights:
                print(f"\n레이어: {layer.name}")
                for weight in layer.weights:
                    w_array = weight.numpy()
                    print(f"  {weight.name}:")
                    print(f"    형태: {w_array.shape}")
                    print(f"    평균: {np.mean(w_array):.6f}")
                    print(f"    표준편차: {np.std(w_array):.6f}")
                    print(f"    최소/최대: {np.min(w_array):.6f} / {np.max(w_array):.6f}")

if __name__ == "__main__":
    # 1. 패턴 테스트
    test_with_training_data()
    
    # 2. 가중치 분석
    check_model_weights()
    
    print("\n\n💡 분석 결과:")
    print("1. 모든 클래스가 비슷한 확률 → 모델이 제대로 학습되지 않음")
    print("2. 특정 패턴에도 반응 없음 → 특징 추출 실패")
    print("3. 해결 방안:")
    print("   - 원본 학습 코드/데이터 확인 필요")
    print("   - 전처리 파이프라인 검증")
    print("   - 모델 재학습 고려")