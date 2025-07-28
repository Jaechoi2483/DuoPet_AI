"""
눈 질환 모델 간단한 수정
이미 존재하는 fixed 모델을 TF 2.x 형식으로 재저장
"""
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import json

# TensorFlow 2.x 설정
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def simple_fix():
    """간단한 모델 수정"""
    
    models_dir = Path("models/health_diagnosis/eye_disease")
    
    # 사용 가능한 모델 확인
    print("🔍 사용 가능한 모델 확인 중...")
    
    available_models = []
    for model_file in models_dir.glob("*.h5"):
        print(f"  - {model_file.name}")
        available_models.append(model_file)
    
    if not available_models:
        print("❌ H5 모델 파일이 없습니다.")
        return
    
    # eye_disease_fixed.h5 우선 사용
    fixed_model_path = models_dir / "eye_disease_fixed.h5"
    if fixed_model_path.exists():
        source_path = fixed_model_path
        print(f"\n✅ 사용할 모델: {source_path.name}")
    else:
        source_path = available_models[0]
        print(f"\n⚠️ 대체 모델 사용: {source_path.name}")
    
    output_path = models_dir / "eye_disease_tf2_simple.h5"
    
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
            metrics=[
                'accuracy',
                tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy'),
                tf.keras.metrics.AUC(name='auc', multi_label=True)
            ]
        )
        
        print("  ✓ 컴파일 완료")
        
        # 3. 저장
        print("\n💾 TF 2.x 형식으로 저장 중...")
        
        # H5 형식으로 저장
        model.save(str(output_path), save_format='h5')
        print(f"  ✓ H5 저장: {output_path}")
        
        # SavedModel 형식으로도 저장
        savedmodel_path = models_dir / "eye_disease_tf2_savedmodel"
        model.save(str(savedmodel_path), save_format='tf')
        print(f"  ✓ SavedModel 저장: {savedmodel_path}")
        
        # 4. 검증
        print("\n✅ 변환 검증 중...")
        
        # 테스트 입력
        test_input = np.random.randn(2, 224, 224, 3).astype(np.float32)
        
        # 원본 모델 예측
        original_output = model.predict(test_input, verbose=0)
        
        # 저장된 모델 다시 로드
        loaded_model = tf.keras.models.load_model(str(output_path))
        loaded_output = loaded_model.predict(test_input, verbose=0)
        
        # 출력 비교
        diff = np.mean(np.abs(original_output - loaded_output))
        print(f"  - 원본 vs 저장된 모델 차이: {diff:.8f}")
        
        if diff < 1e-5:
            print("  ✓ 변환 검증 성공!")
        else:
            print("  ⚠️ 출력 차이가 있습니다.")
        
        # 5. 모델 정보 저장
        model_info = {
            "model_name": "eye_disease_model",
            "source_file": source_path.name,
            "output_files": {
                "h5": str(output_path),
                "savedmodel": str(savedmodel_path)
            },
            "input_shape": [224, 224, 3],
            "output_classes": 5,
            "class_names": ["정상", "백내장", "녹내장", "망막질환", "각막질환"],
            "metrics": ["accuracy", "top2_accuracy", "auc"],
            "tensorflow_version": tf.__version__,
            "conversion_method": "simple_recompile_and_save"
        }
        
        info_path = models_dir / "conversion_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        print(f"\n📝 모델 정보 저장: {info_path}")
        
        # 6. 간단한 추론 테스트
        print("\n🧪 추론 테스트...")
        
        test_batch = np.random.randint(0, 255, (3, 224, 224, 3), dtype=np.uint8).astype(np.float32)
        predictions = loaded_model.predict(test_batch, verbose=0)
        
        class_names = ["정상", "백내장", "녹내장", "망막질환", "각막질환"]
        
        for i, pred in enumerate(predictions):
            predicted_class = np.argmax(pred)
            confidence = pred[predicted_class]
            print(f"\n  테스트 {i+1}:")
            print(f"    - 예측: {class_names[predicted_class]} ({confidence:.1%})")
        
        print("\n✅ 모든 작업 완료!")
        print(f"\n사용 권장 모델:")
        print(f"  - H5 형식: {output_path}")
        print(f"  - SavedModel 형식: {savedmodel_path}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_fix()