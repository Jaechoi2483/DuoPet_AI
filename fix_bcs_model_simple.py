"""
BCS 모델 간단한 수정
앙상블 구조를 유지하면서 TF 2.x 호환성 확보
"""
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import json

# TensorFlow 2.x 설정
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def fix_bcs_model():
    """BCS 모델 수정"""
    
    models_dir = Path("models/health_diagnosis/bcs")
    source_path = models_dir / "bcs_efficientnet_v1.h5"
    output_path = models_dir / "bcs_tf2_unified.h5"
    
    if not source_path.exists():
        print(f"❌ 소스 모델이 없습니다: {source_path}")
        return False
    
    print("🔧 BCS 앙상블 모델 수정 시작...")
    
    try:
        # 1. 모델 로드
        print("\n📥 모델 로드 중...")
        
        # Custom objects
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
        
        print("  ✓ 모델 로드 성공")
        print(f"  - 입력 수: {len(model.inputs)}")
        print(f"  - 출력 shape: {model.output_shape}")
        
        # 2. 단일 입력 래퍼 모델 생성
        print("\n🔨 단일 입력 래퍼 생성 중...")
        
        # 단일 입력
        single_input = tf.keras.Input(shape=(224, 224, 3), name='input')
        
        # 13개 입력으로 복제
        repeated_inputs = [single_input for _ in range(13)]
        
        # 원본 모델 호출
        output = model(repeated_inputs)
        
        # 새로운 모델 생성
        wrapper_model = tf.keras.Model(inputs=single_input, outputs=output, name='bcs_wrapper')
        
        print("  ✓ 래퍼 모델 생성 완료")
        print(f"  - 래퍼 입력: {wrapper_model.input_shape}")
        print(f"  - 래퍼 출력: {wrapper_model.output_shape}")
        
        # 3. 컴파일
        print("\n🔨 모델 컴파일 중...")
        
        wrapper_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 4. 저장
        print("\n💾 저장 중...")
        wrapper_model.save(str(output_path), save_format='h5')
        print(f"  ✓ 저장 완료: {output_path}")
        
        # 5. 검증
        print("\n✅ 검증 중...")
        
        # 단일 입력 테스트
        test_input = np.random.randint(0, 255, (2, 224, 224, 3), dtype=np.uint8).astype(np.float32)
        output = wrapper_model.predict(test_input, verbose=0)
        
        print(f"  - 입력 shape: {test_input.shape}")
        print(f"  - 출력 shape: {output.shape}")
        print(f"  - 출력 예시: {output[0]}")
        print(f"  - 출력 합: {np.sum(output[0]):.4f}")
        
        # 클래스 예측
        classes = ["마른 체형", "정상 체형", "비만 체형"]
        for i, pred in enumerate(output):
            class_idx = np.argmax(pred)
            print(f"  - 샘플 {i+1}: {classes[class_idx]} ({pred[class_idx]:.2%})")
        
        # 6. 모델 정보 저장
        model_info = {
            "model_name": "bcs_model_wrapped",
            "original_structure": "13-input ensemble",
            "wrapper_structure": "single input duplicated to 13",
            "input_shape": [224, 224, 3],
            "output_classes": 3,
            "class_names": classes,
            "preprocessing": "standard (0-255 range)",
            "tensorflow_version": tf.__version__,
            "notes": "앙상블 모델을 단일 입력으로 래핑"
        }
        
        info_path = models_dir / "model_info_wrapped.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        print(f"\n📝 모델 정보 저장: {info_path}")
        
        # 대안: 원본 구조 유지 버전도 저장
        print("\n🔧 원본 구조 버전도 저장 중...")
        
        # 원본 모델 재컴파일
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        original_output_path = models_dir / "bcs_tf2_original_structure.h5"
        model.save(str(original_output_path), save_format='h5')
        print(f"  ✓ 원본 구조 저장: {original_output_path}")
        
        print("\n✅ BCS 모델 변환 완료!")
        print("\n💡 사용 권장:")
        print(f"  - 단일 입력 버전: {output_path}")
        print(f"  - 13개 입력 버전: {original_output_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    fix_bcs_model()