"""
안구질환 모델 간단 변환
Keras API만 사용하여 H5 재저장
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
import numpy as np
from pathlib import Path

def convert_eye_model():
    """모델을 순수 Keras로 재저장"""
    
    models_path = Path("models/health_diagnosis/eye_disease")
    
    # 원본 모델 찾기
    source_candidates = [
        models_path / "eye_disease_fixed.h5",
        models_path / "best_grouped_model.keras"
    ]
    
    source_path = None
    for candidate in source_candidates:
        if candidate.exists():
            source_path = candidate
            break
    
    if not source_path:
        print("❌ 모델 파일을 찾을 수 없습니다")
        return
    
    print(f"📥 모델 로드: {source_path}")
    
    # Keras API로 로드
    custom_objects = {'swish': keras.activations.swish}
    
    try:
        # 모델 로드
        model = keras.models.load_model(
            str(source_path),
            custom_objects=custom_objects,
            compile=False
        )
        
        # 새로운 optimizer로 컴파일
        model.compile(
            optimizer=keras.optimizers.Adam(0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 테스트
        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        output = model.predict(test_input)
        print(f"✓ 테스트 성공: {output.shape}")
        
        # 저장
        output_path = models_path / "eye_disease_keras_clean.h5"
        model.save(str(output_path), save_traces=False)
        print(f"💾 저장 완료: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        return False

if __name__ == "__main__":
    print("🔄 안구질환 모델 Keras 변환")
    print("=" * 50)
    convert_eye_model()
