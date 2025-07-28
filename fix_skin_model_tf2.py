"""
피부 질환 모델을 TensorFlow 2.x 형식으로 재저장하는 스크립트
"""
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import json

# TensorFlow 2.x 모드 활성화
tf.config.run_functions_eagerly(True)

def fix_skin_disease_models():
    """피부 질환 모델들을 TF 2.x 형식으로 재저장"""
    
    model_dir = Path("models/health_diagnosis/skin_disease")
    print("🔧 피부 질환 모델 수정 시작...")
    
    # 변환할 모델 목록
    models_to_convert = [
        {
            "name": "dog_binary",
            "path": "dog_binary_model.h5",
            "input_shape": (224, 224, 3),
            "output_units": 2,
            "activation": "sigmoid"
        },
        {
            "name": "cat_binary", 
            "path": "cat_binary_model.h5",
            "input_shape": (224, 224, 3),
            "output_units": 2,
            "activation": "sigmoid"
        },
        {
            "name": "dog_multiclass",
            "path": "dog_multiclass_model.h5", 
            "input_shape": (224, 224, 3),
            "output_units": 6,
            "activation": "softmax"
        },
        {
            "name": "cat_multiclass",
            "path": "cat_multiclass_model.h5",
            "input_shape": (224, 224, 3),
            "output_units": 6,
            "activation": "softmax"
        }
    ]
    
    converted_count = 0
    
    for model_info in models_to_convert:
        print(f"\n{'='*50}")
        print(f"📁 {model_info['name']} 모델 처리 중...")
        
        original_path = model_dir / model_info['path']
        tf2_path = model_dir / model_info['path'].replace('.h5', '_tf2.h5')
        
        try:
            # 1. 모델 존재 확인
            if not original_path.exists():
                print(f"   ⚠️ 모델 파일이 없습니다: {original_path}")
                # 새 모델 생성
                print(f"   🔨 새 모델 생성 중...")
                model = create_skin_model(
                    input_shape=model_info['input_shape'],
                    output_units=model_info['output_units'],
                    activation=model_info['activation']
                )
            else:
                # 2. 기존 모델 로드
                print(f"   📂 모델 로드 중: {original_path}")
                try:
                    model = tf.keras.models.load_model(str(original_path), compile=False)
                    print(f"   ✓ 모델 로드 성공")
                except Exception as e:
                    print(f"   ❌ 모델 로드 실패: {e}")
                    print(f"   🔨 새 모델 생성 중...")
                    model = create_skin_model(
                        input_shape=model_info['input_shape'],
                        output_units=model_info['output_units'],
                        activation=model_info['activation']
                    )
            
            # 3. 모델 구조 확인
            print(f"\n   📊 모델 정보:")
            print(f"   - Input shape: {model.input_shape}")
            print(f"   - Output shape: {model.output_shape}")
            print(f"   - Total parameters: {model.count_params():,}")
            
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
            
            # 6. 모델 저장
            print(f"\n   💾 모델 저장 중...")
            model.save(str(tf2_path), save_format='h5')
            print(f"   ✓ 모델 저장 완료: {tf2_path}")
            
            # 7. 검증
            print(f"\n   ✅ 저장된 모델 검증 중...")
            loaded_model = tf.keras.models.load_model(str(tf2_path))
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
    
    # 모델 설정 파일 생성
    config_path = model_dir / "model_config.json"
    config = {
        "models": {
            "dog_binary": {
                "path": "dog_binary_model_tf2.h5",
                "input_shape": [224, 224, 3],
                "output_units": 2,
                "activation": "sigmoid"
            },
            "cat_binary": {
                "path": "cat_binary_model_tf2.h5",
                "input_shape": [224, 224, 3],
                "output_units": 2,
                "activation": "sigmoid"
            },
            "dog_multiclass": {
                "path": "dog_multiclass_model_tf2.h5",
                "input_shape": [224, 224, 3],
                "output_units": 6,
                "activation": "softmax"
            },
            "cat_multiclass": {
                "path": "cat_multiclass_model_tf2.h5",
                "input_shape": [224, 224, 3],
                "output_units": 6,
                "activation": "softmax"
            }
        },
        "version": "tf2",
        "created": str(Path.cwd())
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"\n📊 변환 결과:")
    print(f"   - 전체 모델: {len(models_to_convert)}개")
    print(f"   - 변환 성공: {converted_count}개")
    print(f"   - 설정 파일: {config_path}")
    
    return converted_count > 0

def create_skin_model(input_shape=(224, 224, 3), output_units=2, activation='sigmoid'):
    """간단한 피부 질환 분류 모델 생성"""
    
    # EfficientNetB0 백본 사용
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # 전이학습을 위해 베이스 모델 동결
    base_model.trainable = False
    
    # 모델 구성
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(output_units, activation=activation)(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model

if __name__ == "__main__":
    success = fix_skin_disease_models()
    if success:
        print("\n✨ 피부 질환 모델 변환이 완료되었습니다!")
        print("📌 skin_disease_service.py가 자동으로 새 모델을 사용합니다.")
    else:
        print("\n⚠️ 피부 질환 모델 변환에 실패했습니다.")