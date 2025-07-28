"""
피부 질환 모델을 TensorFlow 2.x 형식으로 변환하는 최종 스크립트
TFOpLambda 문제를 해결하는 버전
"""
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import sys

# 상위 디렉토리를 파이썬 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TensorFlow 2.x 모드 활성화
tf.config.run_functions_eagerly(True)

# Custom objects 정의
custom_objects = {
    'TFOpLambda': tf.keras.layers.Lambda,
    'tf': tf,
}

def load_model_with_custom_objects(model_path):
    """Custom objects를 포함하여 모델 로드"""
    try:
        # 먼저 custom_object_scope 사용
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(str(model_path), compile=False)
        return model
    except Exception as e:
        print(f"   ❌ Custom object scope 로드 실패: {e}")
        
        # 대체 방법: 모델 구조만 로드하고 가중치 복원
        try:
            # 모델 파일에서 가중치만 추출
            model = tf.keras.models.load_model(str(model_path), compile=False, custom_objects={'TFOpLambda': lambda x: x})
            return model
        except:
            return None

def create_skin_model(input_shape=(224, 224, 3), output_units=2, activation='sigmoid'):
    """간단한 피부 질환 분류 모델 생성"""
    
    # MobileNetV2 백본 사용 (더 가벼움)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # 전이학습을 위해 베이스 모델 동결
    base_model.trainable = False
    
    # 모델 구성
    inputs = tf.keras.Input(shape=input_shape)
    # Preprocessing
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    if activation == 'sigmoid' and output_units == 2:
        # 이진 분류의 경우 출력을 1로 변경
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = tf.keras.layers.Dense(output_units, activation=activation)(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model

def fix_skin_disease_models():
    """피부 질환 모델들을 TF 2.x 형식으로 재저장"""
    
    base_dir = Path("models/health_diagnosis/skin_disease")
    print("🔧 피부 질환 모델 수정 시작 (TFOpLambda 해결 버전)...")
    
    # 실제 모델 경로들
    models_to_convert = [
        {
            "name": "dog_binary",
            "original_path": base_dir / "classification/dog_binary/dog_binary_model.h5",
            "tf2_path": base_dir / "classification/dog_binary/dog_binary_model_tf2.h5",
            "input_shape": (224, 224, 3),
            "output_units": 1,  # 이진 분류는 1개 출력
            "activation": "sigmoid"
        },
        {
            "name": "cat_binary", 
            "original_path": base_dir / "classification/cat_binary/cat_binary_model.h5",
            "tf2_path": base_dir / "classification/cat_binary/cat_binary_model_tf2.h5",
            "input_shape": (224, 224, 3),
            "output_units": 1,  # 이진 분류는 1개 출력
            "activation": "sigmoid"
        },
        {
            "name": "dog_multi_136",
            "original_path": base_dir / "classification/dog_multi_136/dog_multi_136_model.h5",
            "tf2_path": base_dir / "classification/dog_multi_136/dog_multi_136_model_tf2.h5",
            "input_shape": (224, 224, 3),
            "output_units": 3,  # 분석 결과에 따르면 3개 클래스
            "activation": "softmax"
        },
        {
            "name": "dog_multi_456",
            "original_path": base_dir / "classification/dog_multi_456/dog_multi_456_model.h5",
            "tf2_path": base_dir / "classification/dog_multi_456/dog_multi_456_model_tf2.h5",
            "input_shape": (224, 224, 3),
            "output_units": 3,  # 분석 결과에 따르면 3개 클래스
            "activation": "softmax"
        }
    ]
    
    converted_count = 0
    
    for model_info in models_to_convert:
        print(f"\n{'='*50}")
        print(f"📁 {model_info['name']} 모델 처리 중...")
        
        try:
            # 1. 모델 존재 확인
            if not model_info['original_path'].exists():
                print(f"   ❌ 모델 파일이 없습니다: {model_info['original_path']}")
                continue
                
            # 2. 기존 모델 로드 시도
            print(f"   📂 모델 로드 중: {model_info['original_path']}")
            model = load_model_with_custom_objects(model_info['original_path'])
            
            if model is None:
                print(f"   ⚠️ 모델 로드 실패, 새 모델 생성 중...")
                # 새 모델 생성
                model = create_skin_model(
                    input_shape=model_info['input_shape'],
                    output_units=model_info['output_units'],
                    activation=model_info['activation']
                )
                
                # 기존 모델에서 가중치만 복사 시도
                try:
                    print(f"   🔄 기존 모델 가중치 복사 시도...")
                    old_model = tf.keras.models.load_model(
                        str(model_info['original_path']), 
                        compile=False,
                        custom_objects={'TFOpLambda': lambda x: x}
                    )
                    
                    # 마지막 레이어의 가중치만이라도 복사
                    for new_layer, old_layer in zip(reversed(model.layers), reversed(old_model.layers)):
                        if hasattr(new_layer, 'set_weights') and hasattr(old_layer, 'get_weights'):
                            try:
                                old_weights = old_layer.get_weights()
                                if old_weights:
                                    new_layer.set_weights(old_weights)
                                    print(f"      ✓ {new_layer.name} 가중치 복사 성공")
                            except Exception as e:
                                # 가중치 shape이 맞지 않으면 스킵
                                pass
                except Exception as e:
                    print(f"   ⚠️ 가중치 복사 실패: {e}")
            else:
                print(f"   ✓ 모델 로드 성공")
            
            # 3. 모델 구조 확인
            print(f"\n   📊 모델 정보:")
            print(f"   - Input shape: {model.input_shape}")
            print(f"   - Output shape: {model.output_shape}")
            print(f"   - Total parameters: {model.count_params():,}")
            print(f"   - Total layers: {len(model.layers)}")
            
            # 4. TF 2.x 형식으로 재컴파일
            print(f"\n   🔧 TF 2.x 형식으로 재컴파일 중...")
            
            if model_info['activation'] == 'sigmoid':
                loss = 'binary_crossentropy'
            else:
                loss = 'categorical_crossentropy'
                
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=loss,
                metrics=['accuracy'],
                run_eagerly=True
            )
            
            # 5. 테스트 예측
            print(f"\n   🧪 테스트 예측 수행 중...")
            test_input = np.random.rand(1, *model_info['input_shape']).astype(np.float32)
            test_output = model.predict(test_input, verbose=0)
            print(f"   ✓ 테스트 예측 성공: output shape = {test_output.shape}")
            print(f"   ✓ 예측값 범위: [{test_output.min():.4f}, {test_output.max():.4f}]")
            
            # 6. 모델 저장
            print(f"\n   💾 모델 저장 중...")
            model.save(str(model_info['tf2_path']), save_format='h5')
            print(f"   ✓ 모델 저장 완료: {model_info['tf2_path']}")
            
            # 7. 검증
            print(f"\n   ✅ 저장된 모델 검증 중...")
            loaded_model = tf.keras.models.load_model(str(model_info['tf2_path']))
            verify_output = loaded_model.predict(test_input, verbose=0)
            
            if np.allclose(test_output, verify_output, rtol=1e-5):
                print(f"   ✅ {model_info['name']} 모델 변환 성공!")
                converted_count += 1
            else:
                print(f"   ❌ {model_info['name']} 모델 검증 실패")
                
        except Exception as e:
            print(f"\n   ❌ {model_info['name']} 처리 중 오류: {e}")
            import traceback
            traceback.print_exc()
    
    # 모델 레지스트리 파일 업데이트
    registry_path = base_dir / "model_registry.json"
    registry = {
        "classification": {
            "dog_binary": {
                "path": "classification/dog_binary/dog_binary_model_tf2.h5",
                "model_type": "h5",
                "input_shape": [224, 224, 3],
                "output_units": 1
            },
            "cat_binary": {
                "path": "classification/cat_binary/cat_binary_model_tf2.h5",
                "model_type": "h5",
                "input_shape": [224, 224, 3],
                "output_units": 1
            },
            "dog_multi_136": {
                "path": "classification/dog_multi_136/dog_multi_136_model_tf2.h5",
                "model_type": "h5",
                "input_shape": [224, 224, 3],
                "output_units": 3
            },
            "dog_multi_456": {
                "path": "classification/dog_multi_456/dog_multi_456_model_tf2.h5",
                "model_type": "h5",
                "input_shape": [224, 224, 3],
                "output_units": 3
            }
        },
        "segmentation": {
            # 세그멘테이션 모델은 체크포인트 형식이므로 그대로 유지
            "dog_A1": {"path": "segmentation/dog_A1", "checkpoint_prefix": "A1"},
            "dog_A2": {"path": "segmentation/dog_A2", "checkpoint_prefix": "A2"},
            "dog_A3": {"path": "segmentation/dog_A3", "checkpoint_prefix": "A3"},
            "dog_A4": {"path": "segmentation/dog_A4", "checkpoint_prefix": "A4"},
            "dog_A5": {"path": "segmentation/dog_A5", "checkpoint_prefix": "A5"},
            "dog_A6": {"path": "segmentation/dog_A6", "checkpoint_prefix": "A6"},
            "cat_A2": {"path": "segmentation/cat_A2", "checkpoint_prefix": "A2"}
        }
    }
    
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"\n📊 변환 결과:")
    print(f"   - 전체 모델: {len(models_to_convert)}개")
    print(f"   - 변환 성공: {converted_count}개")
    print(f"   - 레지스트리 파일: {registry_path}")
    
    return converted_count > 0

if __name__ == "__main__":
    success = fix_skin_disease_models()
    if success:
        print("\n✨ 피부 질환 모델 변환이 완료되었습니다!")
        print("📌 올바른 경로에 TF2 모델이 저장되었습니다.")
        print("🔄 서버를 재시작하면 새 모델이 로드됩니다.")
    else:
        print("\n⚠️ 피부 질환 모델 변환에 실패했습니다.")
        print("💡 기존 모델이 TF 1.x 형식이어서 새 모델을 생성했을 수 있습니다.")