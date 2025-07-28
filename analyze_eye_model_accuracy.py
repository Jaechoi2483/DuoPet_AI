"""
안구질환 모델 정확도 분석 및 개선 방안 도출
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_eye_disease_model():
    """모델 구조 및 성능 분석"""
    
    print("🔍 안구질환 모델 분석 시작...")
    print("="*60)
    
    # 1. 모델 로드
    model_paths = [
        "models/health_diagnosis/eye_disease/eye_disease_tf2_complete.h5",
        "models/health_diagnosis/eye_disease/eye_disease_fixed.h5",
        "models/health_diagnosis/eye_disease/best_grouped_model.keras"
    ]
    
    model = None
    loaded_path = None
    
    for path in model_paths:
        if Path(path).exists():
            try:
                custom_objects = {
                    'swish': tf.nn.swish,
                    'Swish': tf.keras.layers.Activation(tf.nn.swish)
                }
                model = tf.keras.models.load_model(path, custom_objects=custom_objects)
                loaded_path = path
                print(f"✅ 모델 로드 성공: {path}")
                break
            except Exception as e:
                print(f"❌ 모델 로드 실패 ({path}): {e}")
    
    if not model:
        print("❌ 모델을 로드할 수 없습니다.")
        return
    
    # 2. 모델 구조 분석
    print("\n📊 모델 구조:")
    print(f"- 입력 형태: {model.input_shape}")
    print(f"- 출력 형태: {model.output_shape}")
    print(f"- 총 파라미터: {model.count_params():,}")
    print(f"- 레이어 수: {len(model.layers)}")
    
    # 3. 클래스 맵 로드
    class_map_path = "models/health_diagnosis/eye_disease/class_map.json"
    with open(class_map_path, 'r', encoding='utf-8') as f:
        class_map = json.load(f)
    
    print(f"\n📋 클래스 정보:")
    for idx, name in class_map.items():
        print(f"  - Class {idx}: {name}")
    
    # 4. 더미 이미지로 예측 테스트
    print("\n🧪 예측 테스트:")
    
    # 다양한 테스트 이미지 생성
    test_cases = [
        ("흰색 이미지", np.ones((1, 224, 224, 3), dtype=np.float32)),
        ("검은색 이미지", np.zeros((1, 224, 224, 3), dtype=np.float32)),
        ("랜덤 이미지", np.random.random((1, 224, 224, 3)).astype(np.float32)),
        ("회색 이미지", np.full((1, 224, 224, 3), 0.5, dtype=np.float32))
    ]
    
    for name, test_image in test_cases:
        predictions = model.predict(test_image, verbose=0)
        print(f"\n{name} 예측 결과:")
        
        # 상위 3개 예측 출력
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        for idx in top_indices:
            class_name = class_map.get(str(idx), "Unknown")
            prob = predictions[0][idx]
            print(f"  - {class_name}: {prob:.4f} ({prob*100:.1f}%)")
    
    # 5. 모델 출력층 분석
    print("\n🔬 출력층 분석:")
    output_layer = model.layers[-1]
    print(f"- 출력층 타입: {type(output_layer).__name__}")
    print(f"- 활성화 함수: {output_layer.activation.__name__ if hasattr(output_layer, 'activation') else 'N/A'}")
    
    # Softmax 여부 확인
    if hasattr(output_layer, 'activation') and output_layer.activation.__name__ == 'softmax':
        print("✅ Softmax 활성화 사용 중 (정상)")
    else:
        print("⚠️ Softmax 활성화가 없을 수 있음")
        
        # 마지막 몇 개 레이어 확인
        print("\n마지막 5개 레이어:")
        for i, layer in enumerate(model.layers[-5:]):
            print(f"  - {len(model.layers)-5+i}: {layer.name} ({type(layer).__name__})")
    
    # 6. 정확도 향상 방안
    print("\n💡 정확도 향상 방안:")
    print("1. 이미지 전처리 개선:")
    print("   - 정규화 방식 확인 (0-1 vs -1-1)")
    print("   - 크기 조정 방식 (resize vs crop)")
    print("   - 데이터 증강 적용")
    
    print("\n2. 모델 개선:")
    print("   - Fine-tuning with 한국 반려동물 데이터")
    print("   - 앙상블 모델 사용")
    print("   - Test Time Augmentation (TTA)")
    
    print("\n3. 추론 시 개선:")
    print("   - 여러 각도의 사진 평균")
    print("   - 신뢰도 임계값 조정")
    print("   - 상위 2-3개 예측 함께 제공")

def create_prediction_debugger():
    """예측 디버거 생성"""
    
    debugger_content = '''"""
안구질환 예측 디버거
실제 이미지로 상세 분석
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import json
from PIL import Image
import cv2

def debug_eye_prediction(image_path):
    """이미지 예측 과정 상세 디버깅"""
    
    # 모델 로드
    model = tf.keras.models.load_model(
        'models/health_diagnosis/eye_disease/eye_disease_fixed.h5',
        custom_objects={'swish': tf.nn.swish}
    )
    
    # 클래스 맵 로드
    with open('models/health_diagnosis/eye_disease/class_map.json', 'r', encoding='utf-8') as f:
        class_map = json.load(f)
    
    # 이미지 로드 및 전처리
    print(f"🖼️ 이미지 분석: {image_path}")
    img = Image.open(image_path).convert('RGB')
    print(f"원본 크기: {img.size}")
    
    # 다양한 전처리 방식 테스트
    preprocessing_methods = {
        "기본 전처리": lambda img: np.array(img.resize((224, 224))).astype(np.float32) / 255.0,
        "중앙 크롭": lambda img: center_crop_and_resize(img),
        "히스토그램 균등화": lambda img: histogram_equalize(img),
        "CLAHE 적용": lambda img: apply_clahe(img)
    }
    
    for method_name, preprocess_func in preprocessing_methods.items():
        print(f"\\n📊 {method_name}:")
        
        try:
            processed = preprocess_func(img)
            if processed.ndim == 3:
                processed = np.expand_dims(processed, axis=0)
            
            # 예측
            predictions = model.predict(processed, verbose=0)
            
            # 결과 출력
            print("예측 확률:")
            for idx, prob in enumerate(predictions[0]):
                class_name = class_map.get(str(idx), f"Unknown_{idx}")
                print(f"  {class_name}: {prob:.4f} ({prob*100:.1f}%)")
            
            # 최종 예측
            pred_idx = np.argmax(predictions[0])
            pred_class = class_map.get(str(pred_idx), "Unknown")
            confidence = predictions[0][pred_idx]
            print(f"\\n최종 진단: {pred_class} ({confidence*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ 오류: {e}")

def center_crop_and_resize(img):
    """중앙 크롭 후 리사이즈"""
    width, height = img.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    
    img_cropped = img.crop((left, top, right, bottom))
    img_resized = img_cropped.resize((224, 224))
    return np.array(img_resized).astype(np.float32) / 255.0

def histogram_equalize(img):
    """히스토그램 균등화"""
    img_array = np.array(img.resize((224, 224)))
    img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_rgb.astype(np.float32) / 255.0

def apply_clahe(img):
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용"""
    img_array = np.array(img.resize((224, 224)))
    img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    
    img_lab = cv2.merge([l_clahe, a, b])
    img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    return img_rgb.astype(np.float32) / 255.0

if __name__ == "__main__":
    # 테스트할 이미지 경로 입력
    test_image = input("테스트할 이미지 경로: ")
    if os.path.exists(test_image):
        debug_eye_prediction(test_image)
    else:
        print("❌ 이미지 파일을 찾을 수 없습니다.")
'''
    
    with open("debug_eye_prediction.py", 'w', encoding='utf-8') as f:
        f.write(debugger_content)
    
    print("\n✅ 예측 디버거 생성: debug_eye_prediction.py")

if __name__ == "__main__":
    print("🔬 안구질환 모델 정확도 분석")
    print("="*60)
    
    # 1. 모델 분석
    analyze_eye_disease_model()
    
    # 2. 디버거 생성
    create_prediction_debugger()
    
    print("\n\n📝 추가 권장사항:")
    print("1. 실제 안구 질환 이미지로 테스트")
    print("   python debug_eye_prediction.py")
    print("\n2. 모델 재학습 고려")
    print("   - 한국 반려동물 데이터셋 수집")
    print("   - Transfer Learning 적용")
    print("\n3. 앙상블 방법 적용")
    print("   - 여러 모델의 예측 평균")
    print("   - 신뢰도 기반 가중 평균")