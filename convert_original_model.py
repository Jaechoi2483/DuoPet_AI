"""
원본 안구질환 모델 변환 스크립트
클래스맵 차이와 모델 구조를 고려한 올바른 변환
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# TensorFlow 2.x 설정
tf.compat.v1.disable_v2_behavior()

def load_original_model_carefully():
    """원본 모델을 조심스럽게 로드"""
    
    print("🔧 원본 모델 변환 시작")
    print("="*80)
    
    # 경로 설정
    import platform
    if platform.system() == "Windows":
        original_model_path = r"C:\Users\ictedu1_021\Desktop\안구질환모델\best_grouped_model.keras"
        original_class_map_path = r"C:\Users\ictedu1_021\Desktop\안구질환모델\class_map.json"
    else:
        original_model_path = "/mnt/c/Users/ictedu1_021/Desktop/안구질환모델/best_grouped_model.keras"
        original_class_map_path = "/mnt/c/Users/ictedu1_021/Desktop/안구질환모델/class_map.json"
    
    # 원본 클래스맵 로드
    with open(original_class_map_path, 'r', encoding='utf-8') as f:
        original_class_map = json.load(f)
    
    print("📋 원본 클래스맵:")
    for idx, name in original_class_map.items():
        print(f"  {idx}: {name}")
    
    # 현재 사용 중인 클래스맵
    current_class_map = {
        "0": "정상",
        "1": "백내장", 
        "2": "결막염",
        "3": "각막궤양",
        "4": "기타안구질환"
    }
    
    print("\n📋 타겟 클래스맵:")
    for idx, name in current_class_map.items():
        print(f"  {idx}: {name}")
    
    # 클래스 매핑 관계 정의
    class_mapping = {
        # 원본 -> 현재
        0: 3,  # 각막 질환 -> 각막궤양
        1: 2,  # 결막 및 누관 질환 -> 결막염  
        2: 1,  # 수정체 질환 -> 백내장
        3: 4,  # 안검 질환 -> 기타안구질환
        4: 4   # 안구 내부 질환 -> 기타안구질환
    }
    
    print("\n🔄 클래스 매핑:")
    for orig_idx, new_idx in class_mapping.items():
        orig_name = original_class_map[str(orig_idx)]
        new_name = current_class_map[str(new_idx)]
        print(f"  {orig_name} -> {new_name}")
    
    # 모델 로드 시도
    print("\n📥 모델 로드 중...")
    
    try:
        # Custom objects 정의
        custom_objects = {
            'swish': tf.nn.swish,
            'Swish': tf.keras.layers.Activation(tf.nn.swish),
            'FixedDropout': tf.keras.layers.Dropout,
            'Functional': tf.keras.Model
        }
        
        # 모델 로드 (compile=False로 optimizer 문제 회피)
        with tf.keras.utils.custom_object_scope(custom_objects):
            # 먼저 모델 구조만 로드
            model = tf.keras.models.load_model(
                original_model_path,
                compile=False
            )
        
        print("✅ 모델 로드 성공!")
        
        # 모델 정보 출력
        print(f"\n📊 모델 정보:")
        print(f"  입력 형태: {model.input_shape}")
        print(f"  출력 형태: {model.output_shape}")
        print(f"  총 레이어: {len(model.layers)}")
        print(f"  총 파라미터: {model.count_params():,}")
        
        return model, class_mapping, current_class_map
        
    except Exception as e:
        print(f"\n❌ 기본 로드 실패: {e}")
        print("\n대체 방법 시도...")
        
        # 대체 방법: 가중치만 로드
        try:
            # 먼저 동일한 구조의 새 모델 생성
            # EfficientNet 기반으로 추정
            from tensorflow.keras.applications import EfficientNetB0
            
            base_model = EfficientNetB0(
                input_shape=(224, 224, 3),
                include_top=False,
                weights=None
            )
            
            # 출력층 추가
            x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
            
            model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
            
            # 가중치 로드 시도
            model.load_weights(original_model_path)
            print("✅ 가중치 로드 성공!")
            
            return model, class_mapping, current_class_map
            
        except Exception as e2:
            print(f"❌ 대체 방법도 실패: {e2}")
            return None, None, None

def create_remapped_model(original_model, class_mapping):
    """클래스를 재매핑한 새 모델 생성"""
    
    print("\n🔨 클래스 재매핑 모델 생성 중...")
    
    # 마지막 Dense 레이어 찾기
    last_dense_idx = None
    for i, layer in enumerate(original_model.layers):
        if isinstance(layer, tf.keras.layers.Dense) and layer.units == 5:
            last_dense_idx = i
            break
    
    if last_dense_idx is None:
        print("❌ 출력 레이어를 찾을 수 없습니다")
        return None
    
    # 새 모델 생성 (마지막 레이어 전까지)
    base_output = original_model.layers[last_dense_idx - 1].output
    
    # 새로운 출력층 생성 (정상 클래스 포함한 5개 클래스)
    new_output = tf.keras.layers.Dense(
        5,
        activation='softmax',
        name='eye_disease_output'
    )(base_output)
    
    # 새 모델 생성
    new_model = tf.keras.Model(
        inputs=original_model.input,
        outputs=new_output
    )
    
    # 마지막 레이어 이전까지의 가중치 복사
    for i in range(last_dense_idx):
        try:
            new_model.layers[i].set_weights(original_model.layers[i].get_weights())
        except:
            pass
    
    # 마지막 레이어 가중치 재매핑
    original_weights = original_model.layers[last_dense_idx].get_weights()
    if len(original_weights) > 0:
        orig_w = original_weights[0]  # (input_dim, 5)
        orig_b = original_weights[1] if len(original_weights) > 1 else np.zeros(5)
        
        # 새 가중치 초기화 (정상 클래스를 위한 랜덤 초기화)
        new_w = np.random.normal(0, 0.02, orig_w.shape)
        new_b = np.zeros(5)
        
        # 기존 클래스 가중치 재매핑
        for orig_idx, new_idx in class_mapping.items():
            if orig_idx < orig_w.shape[1] and new_idx < new_w.shape[1]:
                new_w[:, new_idx] = orig_w[:, orig_idx]
                new_b[new_idx] = orig_b[orig_idx]
        
        # 정상 클래스 (인덱스 0)는 약간의 positive bias
        new_b[0] = 0.1
        
        new_model.layers[-1].set_weights([new_w, new_b])
    
    print("✅ 재매핑 모델 생성 완료!")
    return new_model

def test_model(model, class_map):
    """모델 테스트"""
    print("\n🧪 모델 테스트...")
    
    # 테스트 이미지 생성
    test_patterns = {
        "빨간색 (결막염)": create_red_pattern(),
        "흐린 중앙 (백내장)": create_cloudy_center(),
        "정상 패턴": create_normal_pattern()
    }
    
    for name, pattern in test_patterns.items():
        pred = model.predict(pattern, verbose=0)
        print(f"\n{name}:")
        
        # 상위 3개 예측
        top_indices = np.argsort(pred[0])[-3:][::-1]
        for idx in top_indices:
            class_name = class_map.get(str(idx), f"Unknown_{idx}")
            prob = pred[0][idx]
            print(f"  {class_name}: {prob*100:.1f}%")

def create_red_pattern():
    img = np.zeros((224, 224, 3), dtype=np.float32)
    img[:, :, 0] = 0.9  # 강한 빨간색
    img[:, :, 1] = 0.3
    img[:, :, 2] = 0.3
    return np.expand_dims(img, axis=0)

def create_cloudy_center():
    img = np.ones((224, 224, 3), dtype=np.float32) * 0.8
    center = 112
    for i in range(224):
        for j in range(224):
            dist = np.sqrt((i-center)**2 + (j-center)**2)
            if dist < 50:
                opacity = 0.9 * (1 - dist/50)
                img[i, j] = img[i, j] * (1-opacity) + np.array([0.95, 0.95, 0.95]) * opacity
    return np.expand_dims(img, axis=0)

def create_normal_pattern():
    img = np.ones((224, 224, 3), dtype=np.float32) * 0.9
    center = 112
    # 어두운 동공
    for i in range(224):
        for j in range(224):
            dist = np.sqrt((i-center)**2 + (j-center)**2)
            if dist < 20:
                img[i, j] = [0.1, 0.1, 0.1]
            elif dist < 50:
                img[i, j] = [0.4, 0.3, 0.2]
    return np.expand_dims(img, axis=0)

def save_converted_model(model, class_map):
    """변환된 모델 저장"""
    
    output_dir = Path("models/health_diagnosis/eye_disease")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 기존 모델 백업
    existing_models = [
        "eye_disease_fixed.h5",
        "eye_disease_tf2_complete.h5",
        "best_grouped_model.keras"
    ]
    
    for model_file in existing_models:
        model_path = output_dir / model_file
        if model_path.exists():
            backup_path = model_path.with_suffix(model_path.suffix + '.backup_before_conversion')
            import shutil
            shutil.copy(model_path, backup_path)
            print(f"📦 백업: {backup_path}")
    
    # 새 모델 저장
    print("\n💾 변환된 모델 저장 중...")
    
    # H5 형식으로 저장
    h5_path = output_dir / "eye_disease_converted.h5"
    model.save(h5_path, save_format='h5')
    print(f"✅ H5 저장: {h5_path}")
    
    # Keras 형식으로도 저장
    keras_path = output_dir / "eye_disease_converted.keras"
    model.save(keras_path)
    print(f"✅ Keras 저장: {keras_path}")
    
    # 클래스맵 저장
    class_map_path = output_dir / "class_map.json"
    with open(class_map_path, 'w', encoding='utf-8') as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)
    print(f"✅ 클래스맵 저장: {class_map_path}")
    
    # 변환 정보 저장
    conversion_info = {
        "conversion_date": str(tf.timestamp().numpy()),
        "original_classes": {
            "0": "각막 질환",
            "1": "결막 및 누관 질환",
            "2": "수정체 질환",
            "3": "안검 질환",
            "4": "안구 내부 질환"
        },
        "converted_classes": class_map,
        "mapping": {
            "각막 질환 -> 각막궤양": "0 -> 3",
            "결막 및 누관 질환 -> 결막염": "1 -> 2",
            "수정체 질환 -> 백내장": "2 -> 1",
            "안검 질환 -> 기타안구질환": "3 -> 4",
            "안구 내부 질환 -> 기타안구질환": "4 -> 4",
            "(새로 추가) 정상": "-> 0"
        },
        "notes": "원본 모델에는 '정상' 클래스가 없어서 새로 추가했습니다. 정상 클래스의 가중치는 랜덤 초기화되었습니다."
    }
    
    info_path = output_dir / "conversion_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(conversion_info, f, ensure_ascii=False, indent=2)
    print(f"✅ 변환 정보 저장: {info_path}")

if __name__ == "__main__":
    # 1. 원본 모델 로드
    model, mapping, class_map = load_original_model_carefully()
    
    if model is None:
        print("\n❌ 모델 로드에 실패했습니다.")
        print("\n💡 대안:")
        print("1. 원본 모델을 TensorFlow 2.x 형식으로 다시 저장")
        print("2. 원본 학습 코드로 모델 재학습")
        print("3. 사전학습된 모델 + Transfer Learning 사용")
    else:
        # 2. 클래스 재매핑 모델 생성
        converted_model = create_remapped_model(model, mapping)
        
        if converted_model:
            # 3. 모델 테스트
            test_model(converted_model, class_map)
            
            # 4. 모델 저장
            save_converted_model(converted_model, class_map)
            
            print("\n✅ 모델 변환 완료!")
            print("\n📋 다음 단계:")
            print("1. services/eye_disease_service.py 수정")
            print("   - model_path를 'eye_disease_converted.h5'로 변경")
            print("2. 서버 재시작")
            print("3. 테스트")
            print("\n⚠️  주의: 정상 클래스는 학습되지 않았으므로 추가 학습이 필요합니다!")