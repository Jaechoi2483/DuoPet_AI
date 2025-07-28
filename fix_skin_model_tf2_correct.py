"""
피부 질환 모델을 올바른 경로에서 찾아 TensorFlow 2.x 형식으로 재저장하는 스크립트
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
    
    base_dir = Path("models/health_diagnosis/skin_disease")
    print("🔧 피부 질환 모델 수정 시작 (올바른 경로 버전)...")
    
    # 실제 모델 경로들
    models_to_convert = [
        {
            "name": "dog_binary",
            "original_path": base_dir / "classification/dog_binary/dog_binary_model.h5",
            "tf2_path": base_dir / "classification/dog_binary/dog_binary_model_tf2.h5",
            "input_shape": (224, 224, 3),
            "output_units": 2,
            "activation": "sigmoid"
        },
        {
            "name": "cat_binary", 
            "original_path": base_dir / "classification/cat_binary/cat_binary_model.h5",
            "tf2_path": base_dir / "classification/cat_binary/cat_binary_model_tf2.h5",
            "input_shape": (224, 224, 3),
            "output_units": 2,
            "activation": "sigmoid"
        },
        {
            "name": "dog_multi_136",
            "original_path": base_dir / "classification/dog_multi_136/dog_multi_136_model.h5",
            "tf2_path": base_dir / "classification/dog_multi_136/dog_multi_136_model_tf2.h5",
            "input_shape": (224, 224, 3),
            "output_units": 6,
            "activation": "softmax"
        },
        {
            "name": "dog_multi_456",
            "original_path": base_dir / "classification/dog_multi_456/dog_multi_456_model.h5",
            "tf2_path": base_dir / "classification/dog_multi_456/dog_multi_456_model_tf2.h5",
            "input_shape": (224, 224, 3),
            "output_units": 6,
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
                
            # 2. 기존 모델 로드
            print(f"   📂 모델 로드 중: {model_info['original_path']}")
            try:
                # compile=False로 로드하여 그래프 충돌 방지
                model = tf.keras.models.load_model(str(model_info['original_path']), compile=False)
                print(f"   ✓ 모델 로드 성공")
            except Exception as e:
                print(f"   ❌ 모델 로드 실패: {e}")
                continue
            
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
                "output_units": 2
            },
            "cat_binary": {
                "path": "classification/cat_binary/cat_binary_model_tf2.h5",
                "model_type": "h5",
                "input_shape": [224, 224, 3],
                "output_units": 2
            },
            "dog_multi_136": {
                "path": "classification/dog_multi_136/dog_multi_136_model_tf2.h5",
                "model_type": "h5",
                "input_shape": [224, 224, 3],
                "output_units": 6
            },
            "dog_multi_456": {
                "path": "classification/dog_multi_456/dog_multi_456_model_tf2.h5",
                "model_type": "h5",
                "input_shape": [224, 224, 3],
                "output_units": 6
            }
        },
        "segmentation": {
            # 세그멘테이션 모델은 체크포인트 형식이므로 그대로 유지
            "dog_A1": {"path": "segmentation/dog_A1", "checkpoint_prefix": "ckpt"},
            "dog_A2": {"path": "segmentation/dog_A2", "checkpoint_prefix": "ckpt"},
            "dog_A3": {"path": "segmentation/dog_A3", "checkpoint_prefix": "ckpt"},
            "dog_A4": {"path": "segmentation/dog_A4", "checkpoint_prefix": "ckpt"},
            "dog_A5": {"path": "segmentation/dog_A5", "checkpoint_prefix": "ckpt"},
            "dog_A6": {"path": "segmentation/dog_A6", "checkpoint_prefix": "ckpt"},
            "cat_A2": {"path": "segmentation/cat_A2", "checkpoint_prefix": "ckpt"}
        }
    }
    
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"\n📊 변환 결과:")
    print(f"   - 전체 모델: {len(models_to_convert)}개")
    print(f"   - 변환 성공: {converted_count}개")
    print(f"   - 레지스트리 파일: {registry_path}")
    
    # 기존에 잘못 생성된 파일들 정리
    print(f"\n🧹 잘못된 경로에 생성된 파일 정리 중...")
    wrong_files = [
        base_dir / "dog_binary_model_tf2.h5",
        base_dir / "cat_binary_model_tf2.h5",
        base_dir / "dog_multiclass_model_tf2.h5",
        base_dir / "cat_multiclass_model_tf2.h5",
        base_dir / "model_config.json"
    ]
    
    for file in wrong_files:
        if file.exists():
            file.unlink()
            print(f"   - 삭제: {file.name}")
    
    return converted_count > 0

if __name__ == "__main__":
    success = fix_skin_disease_models()
    if success:
        print("\n✨ 피부 질환 모델 변환이 완료되었습니다!")
        print("📌 올바른 경로에 TF2 모델이 저장되었습니다.")
    else:
        print("\n⚠️ 피부 질환 모델 변환에 실패했습니다.")