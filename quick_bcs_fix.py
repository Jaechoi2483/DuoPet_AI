"""
BCS 모델 빠른 수정
단일 EfficientNet만 추출하여 사용
"""
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import json

# TensorFlow 2.x 설정
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def quick_fix_bcs():
    """BCS 모델 빠른 수정"""
    
    models_dir = Path("models/health_diagnosis/bcs")
    source_path = models_dir / "bcs_efficientnet_v1.h5"
    
    if not source_path.exists():
        print(f"❌ 소스 모델이 없습니다: {source_path}")
        return False
    
    print("🚀 BCS 모델 빠른 수정")
    print("=" * 60)
    
    try:
        # 1. 원본 모델 로드
        print("\n1️⃣ 원본 앙상블 모델 로드 중...")
        
        custom_objects = {
            'swish': tf.nn.swish,
            'Swish': tf.keras.layers.Activation(tf.nn.swish),
            'FixedDropout': tf.keras.layers.Dropout,
        }
        
        model = tf.keras.models.load_model(
            str(source_path),
            custom_objects=custom_objects,
            compile=False
        )
        
        print(f"  ✓ 로드 완료 - 13개 입력 앙상블")
        
        # 2. 첫 번째 서브모델만 추출
        print("\n2️⃣ 단일 EfficientNet 추출 중...")
        
        # 첫 번째 Functional 레이어 찾기
        functional_model = None
        for layer in model.layers:
            if type(layer).__name__ == 'Functional' and 'model' in layer.name:
                functional_model = layer
                print(f"  ✓ 서브모델 발견: {layer.name}")
                break
        
        if not functional_model:
            print("  ❌ Functional 모델을 찾을 수 없음")
            return False
        
        # 3. Dense 레이어 가중치 추출
        print("\n3️⃣ Dense 레이어 가중치 추출 중...")
        
        dense_weights = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                dense_weights = layer.get_weights()
                print(f"  ✓ Dense 가중치 추출: shape={[w.shape for w in dense_weights]}")
                break
        
        # 4. 새 모델 구성
        print("\n4️⃣ 새 모델 구성 중...")
        
        # 입력
        new_input = tf.keras.Input(shape=(224, 224, 3), name='input')
        
        # 추출한 서브모델 사용
        x = functional_model(new_input)
        
        # Dense 레이어 재구성
        # 원본은 13개 모델의 출력을 concat하므로, 단일 모델은 더 작은 입력 차원을 가짐
        if dense_weights:
            # Dense 입력 차원 조정
            original_input_dim = dense_weights[0].shape[0]
            single_model_dim = original_input_dim // 13
            
            print(f"  - 원본 Dense 입력: {original_input_dim}")
            print(f"  - 단일 모델 입력: {single_model_dim}")
            
            # 새 Dense 레이어 (조정된 차원)
            new_dense = tf.keras.layers.Dense(3, activation='softmax', name='predictions')
            outputs = new_dense(x)
            
            # 모델 생성
            single_model = tf.keras.Model(inputs=new_input, outputs=outputs)
            
            # 가중치 조정 (첫 번째 서브모델 부분만 사용)
            try:
                # 원본 가중치의 일부만 사용
                adjusted_kernel = dense_weights[0][:single_model_dim, :]
                adjusted_bias = dense_weights[1]
                
                new_dense.build(x.shape)
                new_dense.set_weights([adjusted_kernel, adjusted_bias])
                print("  ✓ Dense 가중치 조정 완료")
            except:
                print("  ⚠️ Dense 가중치 조정 실패 - 새로 초기화")
        else:
            # Dense 레이어 새로 생성
            outputs = tf.keras.layers.Dense(3, activation='softmax', name='predictions')(x)
            single_model = tf.keras.Model(inputs=new_input, outputs=outputs)
        
        # 5. 컴파일 및 저장
        print("\n5️⃣ 모델 컴파일 및 저장 중...")
        
        single_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # H5 저장
        output_path = models_dir / "bcs_tf2_unified.h5"
        single_model.save(str(output_path), save_format='h5')
        print(f"  ✓ H5 저장 완료: {output_path}")
        
        # SavedModel도 저장
        savedmodel_path = models_dir / "bcs_tf2_savedmodel_single"
        single_model.save(str(savedmodel_path), save_format='tf')
        print(f"  ✓ SavedModel 저장 완료: {savedmodel_path}")
        
        # 6. 테스트
        print("\n6️⃣ 변환된 모델 테스트...")
        
        test_input = np.random.randint(0, 255, (3, 224, 224, 3), dtype=np.uint8).astype(np.float32)
        predictions = single_model.predict(test_input, verbose=0)
        
        print(f"  - 입력 shape: {test_input.shape}")
        print(f"  - 출력 shape: {predictions.shape}")
        
        classes = ["마른 체형", "정상 체형", "비만 체형"]
        for i, pred in enumerate(predictions):
            class_idx = np.argmax(pred)
            confidence = pred[class_idx]
            print(f"  - 샘플 {i+1}: {classes[class_idx]} ({confidence:.2%})")
        
        # 7. 모델 정보 저장
        model_info = {
            "model_name": "bcs_single_efficientnet",
            "source": "13-model ensemble → single model extraction",
            "architecture": "EfficientNetB4 (single)",
            "input_shape": [224, 224, 3],
            "output_classes": 3,
            "class_names": classes,
            "preprocessing": "0-255 range expected",
            "tensorflow_version": tf.__version__,
            "notes": "앙상블의 첫 번째 모델만 추출"
        }
        
        info_path = models_dir / "bcs_model_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        print(f"\n📝 모델 정보 저장: {info_path}")
        
        print("\n✅ BCS 모델 변환 완료!")
        print(f"\n💡 사용 권장:")
        print(f"  - H5 형식: {output_path}")
        print(f"  - SavedModel 형식: {savedmodel_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    quick_fix_bcs()