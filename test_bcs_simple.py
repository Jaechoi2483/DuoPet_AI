"""
BCS 모델 간단 테스트
"""
import os
import sys
import tensorflow as tf
import numpy as np
from pathlib import Path

# TensorFlow 2.x 설정
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 경로 설정
models_base = Path("models/health_diagnosis/bcs")
sys.path.insert(0, str(models_base))

try:
    # 래퍼 클래스 import
    from bcs_ensemble_wrapper import BCSEnsembleModel
    
    print("✅ BCS 래퍼 클래스 import 성공!")
    
    # 모델 로드
    print("\n📥 BCS 모델 로드 중...")
    bcs_model = BCSEnsembleModel()
    
    # 테스트 이미지
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # 예측
    print("\n🔍 예측 테스트...")
    result = bcs_model.predict(test_image, augment=False)
    
    print(f"\n📊 예측 결과:")
    print(f"  - 체형: {result['class']}")
    print(f"  - 신뢰도: {result['confidence']:.2%}")
    print(f"  - 전체 확률: {result['probabilities']}")
    
    # 직접 모델 테스트
    print("\n📋 직접 모델 로드 테스트...")
    model_path = models_base / "bcs_tf2_ensemble.h5"
    
    if model_path.exists():
        custom_objects = {
            'swish': tf.nn.swish,
            'Swish': tf.keras.layers.Activation(tf.nn.swish)
        }
        
        model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)
        print(f"  ✓ 모델 로드 성공!")
        print(f"  - 입력 수: {len(model.inputs)}")
        print(f"  - 출력 shape: {model.output_shape}")
        
        # 13개 입력으로 테스트
        inputs_13 = [test_image.reshape(1, 224, 224, 3).astype(np.float32) for _ in range(13)]
        predictions = model.predict(inputs_13, verbose=0)
        
        classes = ['마른 체형', '정상 체형', '비만 체형']
        class_idx = np.argmax(predictions[0])
        
        print(f"\n  ✓ 직접 예측 결과: {classes[class_idx]} ({predictions[0][class_idx]:.2%})")
    
    print("\n✅ BCS 모델 테스트 완료!")
    
except ImportError as e:
    print(f"❌ Import 오류: {e}")
    print("\n💡 대안: 직접 모델 사용")
    
    # 직접 모델 사용
    model_path = models_base / "bcs_tf2_ensemble.h5"
    if model_path.exists():
        print(f"\n🔄 {model_path} 직접 로드 중...")
        
        custom_objects = {
            'swish': tf.nn.swish,
            'Swish': tf.keras.layers.Activation(tf.nn.swish)
        }
        
        model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)
        
        # 테스트
        test_input = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8).astype(np.float32)
        inputs_13 = [test_input.reshape(1, 224, 224, 3) for _ in range(13)]
        
        predictions = model.predict(inputs_13, verbose=0)
        
        classes = ['마른 체형', '정상 체형', '비만 체형']
        class_idx = np.argmax(predictions[0])
        
        print(f"\n✅ 직접 예측 성공: {classes[class_idx]} ({predictions[0][class_idx]:.2%})")

except Exception as e:
    print(f"❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()