"""
BCS 모델 TF 2.x 변환
"""
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import json

# TensorFlow 2.x 설정
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def convert_bcs_model():
    """BCS 모델 변환"""
    
    models_dir = Path("models/health_diagnosis/bcs")
    source_path = models_dir / "bcs_efficientnet_v1.h5"
    output_path = models_dir / "bcs_tf2_unified.h5"
    
    if not source_path.exists():
        print(f"❌ 소스 모델이 없습니다: {source_path}")
        return False
    
    print("🔧 BCS 모델 변환 시작...")
    print(f"소스: {source_path}")
    print(f"대상: {output_path}")
    
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
        print(f"  - 입력 shape: {model.input_shape}")
        print(f"  - 출력 shape: {model.output_shape}")
        print(f"  - 총 레이어: {len(model.layers)}")
        
        # 2. 재컴파일
        print("\n🔨 모델 재컴파일 중...")
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 3. 저장
        print("\n💾 저장 중...")
        model.save(str(output_path), save_format='h5')
        print(f"  ✓ 저장 완료: {output_path}")
        
        # 4. 검증
        print("\n✅ 검증 중...")
        test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        
        print(f"  - 출력 shape: {output.shape}")
        print(f"  - 출력 합계: {np.sum(output[0]):.4f}")
        
        # 5. 모델 정보 저장
        model_info = {
            "model_name": "bcs_model",
            "input_shape": [224, 224, 3],
            "output_classes": 3,
            "class_names": ["마른 체형", "정상 체형", "비만 체형"],
            "preprocessing": "standard",
            "tensorflow_version": tf.__version__
        }
        
        info_path = models_dir / "model_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        print("\n✅ BCS 모델 변환 완료!")
        return True
        
    except Exception as e:
        print(f"\n❌ 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    convert_bcs_model()