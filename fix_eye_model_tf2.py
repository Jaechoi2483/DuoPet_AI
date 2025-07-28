"""
안구 질환 모델을 TensorFlow 2.x 형식으로 재저장하는 스크립트
"""
import os
import tensorflow as tf
import numpy as np
from pathlib import Path

# TensorFlow 2.x 모드 활성화
tf.config.run_functions_eagerly(True)

def fix_eye_disease_model():
    """안구 질환 모델을 TF 2.x 형식으로 재저장"""
    
    model_dir = Path("models/health_diagnosis/eye_disease")
    original_model_path = model_dir / "best_grouped_model.keras"
    fixed_h5_path = model_dir / "best_grouped_model_fixed.h5"
    new_model_path = model_dir / "best_grouped_model_tf2.h5"
    
    print("🔧 안구 질환 모델 수정 시작...")
    
    try:
        # 1. 모델 로드 시도 (compile=False)
        print("1️⃣ 기존 모델 로드 중...")
        
        # H5 파일 우선 시도
        if fixed_h5_path.exists():
            model = tf.keras.models.load_model(str(fixed_h5_path), compile=False)
            print(f"   ✓ {fixed_h5_path} 로드 성공")
        elif original_model_path.exists():
            model = tf.keras.models.load_model(str(original_model_path), compile=False)
            print(f"   ✓ {original_model_path} 로드 성공")
        else:
            print("   ❌ 모델 파일을 찾을 수 없습니다.")
            return False
            
        # 2. 모델 구조 확인
        print("\n2️⃣ 모델 구조 확인:")
        print(f"   - Input shape: {model.input_shape}")
        print(f"   - Output shape: {model.output_shape}")
        print(f"   - Total layers: {len(model.layers)}")
        
        # 3. 새로운 모델 생성 (가중치 복사)
        print("\n3️⃣ TF 2.x 호환 모델 생성 중...")
        
        # 입력 레이어
        inputs = tf.keras.Input(shape=(224, 224, 3))
        
        # 모델을 함수형 API로 재구성
        x = inputs
        for layer in model.layers[1:]:  # 입력 레이어 제외
            try:
                # 레이어 가중치 가져오기
                weights = layer.get_weights()
                
                # 새 레이어 생성 및 가중치 설정
                if isinstance(layer, tf.keras.layers.Conv2D):
                    new_layer = tf.keras.layers.Conv2D(
                        filters=layer.filters,
                        kernel_size=layer.kernel_size,
                        strides=layer.strides,
                        padding=layer.padding,
                        activation=layer.activation,
                        name=layer.name + "_new"
                    )
                    x = new_layer(x)
                    new_layer.set_weights(weights)
                    
                elif isinstance(layer, tf.keras.layers.Dense):
                    new_layer = tf.keras.layers.Dense(
                        units=layer.units,
                        activation=layer.activation,
                        name=layer.name + "_new"
                    )
                    x = new_layer(x)
                    new_layer.set_weights(weights)
                    
                elif isinstance(layer, tf.keras.layers.BatchNormalization):
                    new_layer = tf.keras.layers.BatchNormalization(name=layer.name + "_new")
                    x = new_layer(x)
                    new_layer.set_weights(weights)
                    
                else:
                    # 기타 레이어는 그대로 적용
                    x = layer(x)
                    
            except Exception as e:
                print(f"   ⚠️ 레이어 {layer.name} 처리 중 오류: {e}")
                x = layer(x)
        
        # 새 모델 생성
        new_model = tf.keras.Model(inputs=inputs, outputs=x)
        
        # 4. 모델 컴파일 (TF 2.x 스타일)
        print("\n4️⃣ 모델 컴파일 중...")
        new_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            run_eagerly=True  # Eager execution 강제
        )
        
        # 5. 테스트 예측
        print("\n5️⃣ 테스트 예측 수행 중...")
        test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        test_output = new_model.predict(test_input)
        print(f"   ✓ 테스트 예측 성공: output shape = {test_output.shape}")
        
        # 6. 모델 저장
        print("\n6️⃣ 모델 저장 중...")
        new_model.save(str(new_model_path), save_format='h5')
        print(f"   ✓ 모델이 {new_model_path}에 저장되었습니다.")
        
        # 7. 검증
        print("\n7️⃣ 저장된 모델 검증 중...")
        loaded_model = tf.keras.models.load_model(str(new_model_path))
        verify_output = loaded_model.predict(test_input)
        
        if np.allclose(test_output, verify_output):
            print("   ✅ 모델 검증 성공!")
            return True
        else:
            print("   ❌ 모델 검증 실패")
            return False
            
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fix_eye_disease_model()
    if success:
        print("\n✨ 모델 수정이 완료되었습니다!")
        print("📌 eye_disease_service.py에서 'best_grouped_model_tf2.h5'를 사용하도록 수정해주세요.")
    else:
        print("\n⚠️ 모델 수정에 실패했습니다.")