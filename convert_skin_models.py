"""
피부질환 모델들 TF 2.x 변환
"""
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import json

# TensorFlow 2.x 설정
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def convert_skin_model(model_type: str):
    """개별 피부질환 모델 변환"""
    
    models_dir = Path("models/health_diagnosis/skin_disease/classification") / model_type
    
    # 소스 모델 찾기 (우선순위)
    source_candidates = [
        models_dir / f"{model_type}_model_tf2_perfect.h5",
        models_dir / f"{model_type}_model_tf2_final.h5",
        models_dir / f"{model_type}_model.h5"
    ]
    
    source_path = None
    for candidate in source_candidates:
        if candidate.exists():
            source_path = candidate
            break
    
    if not source_path:
        print(f"❌ {model_type}: 소스 모델이 없습니다")
        return False
    
    output_path = models_dir / f"{model_type}_tf2_unified.h5"
    
    print(f"\n🔧 {model_type} 모델 변환 시작...")
    print(f"  소스: {source_path.name}")
    print(f"  대상: {output_path.name}")
    
    try:
        # 1. 모델 로드
        print("  📥 로드 중...", end="")
        
        model = tf.keras.models.load_model(
            str(source_path),
            compile=False
        )
        
        print(" ✓")
        
        # 2. 재컴파일
        output_units = model.output_shape[-1]
        
        if output_units == 1:
            # Binary classification
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC()]
            )
        else:
            # Multi-class
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # 3. 저장
        model.save(str(output_path), save_format='h5')
        print(f"  ✓ 변환 완료: {output_path.name}")
        
        # 메모리 정리
        del model
        tf.keras.backend.clear_session()
        
        return True
        
    except Exception as e:
        print(f"  ❌ 실패: {e}")
        return False

def convert_all_skin_models():
    """모든 피부질환 모델 변환"""
    
    print("🚀 피부질환 모델 변환 시작")
    print("=" * 50)
    
    models_to_convert = [
        "cat_binary",
        "dog_binary", 
        "dog_multi_136",
        "dog_multi_456"
    ]
    
    success_count = 0
    
    for model_type in models_to_convert:
        if convert_skin_model(model_type):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 변환 결과: {success_count}/{len(models_to_convert)} 성공")
    
    if success_count == len(models_to_convert):
        print("✅ 모든 피부질환 모델 변환 완료!")
    else:
        print("⚠️ 일부 모델 변환 실패")

if __name__ == "__main__":
    convert_all_skin_models()