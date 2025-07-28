"""
모든 AI 모델 상태 확인
각 모델의 로드 가능 여부와 추론 테스트
"""
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import time
import json

# TensorFlow 설정
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_model_loading():
    """모든 모델 로딩 테스트"""
    
    models_dir = Path("models/health_diagnosis")
    
    # 테스트할 모델 목록
    models = {
        "👁️ 눈 질환 모델": [
            models_dir / "eye_disease" / "eye_disease_tf2_simple.h5",
            models_dir / "eye_disease" / "eye_disease_fixed.h5",
            models_dir / "eye_disease" / "best_grouped_model_fixed.h5"
        ],
        "🐕 BCS 모델": [
            models_dir / "bcs" / "bcs_tf2_unified.h5",
            models_dir / "bcs" / "bcs_efficientnet_v1.h5"
        ],
        "🐱 고양이 피부질환 (Binary)": [
            models_dir / "skin_disease" / "classification" / "cat_binary" / "cat_binary_tf2_unified.h5",
            models_dir / "skin_disease" / "classification" / "cat_binary" / "cat_binary_model_tf2_perfect.h5",
            models_dir / "skin_disease" / "classification" / "cat_binary" / "cat_binary_model.h5"
        ],
        "🐕 개 피부질환 (Binary)": [
            models_dir / "skin_disease" / "classification" / "dog_binary" / "dog_binary_tf2_unified.h5",
            models_dir / "skin_disease" / "classification" / "dog_binary" / "dog_binary_model_tf2_perfect.h5",
            models_dir / "skin_disease" / "classification" / "dog_binary" / "dog_binary_model.h5"
        ],
        "🐕 개 피부질환 (Multi-136)": [
            models_dir / "skin_disease" / "classification" / "dog_multi_136" / "dog_multi_136_tf2_unified.h5",
            models_dir / "skin_disease" / "classification" / "dog_multi_136" / "dog_multi_136_model_tf2_perfect.h5",
            models_dir / "skin_disease" / "classification" / "dog_multi_136" / "dog_multi_136_model.h5"
        ],
        "🐕 개 피부질환 (Multi-456)": [
            models_dir / "skin_disease" / "classification" / "dog_multi_456" / "dog_multi_456_tf2_unified.h5",
            models_dir / "skin_disease" / "classification" / "dog_multi_456" / "dog_multi_456_model_tf2_perfect.h5",
            models_dir / "skin_disease" / "classification" / "dog_multi_456" / "dog_multi_456_model.h5"
        ]
    }
    
    results = {}
    
    print("🚀 DuoPet AI 모델 상태 확인")
    print(f"TensorFlow 버전: {tf.__version__}")
    print("=" * 80)
    
    # Custom objects
    custom_objects = {
        'swish': tf.nn.swish,
        'Swish': tf.keras.layers.Activation(tf.nn.swish),
        'FixedDropout': tf.keras.layers.Dropout,
    }
    
    for model_name, model_paths in models.items():
        print(f"\n{model_name}")
        print("-" * 40)
        
        model_loaded = False
        working_model = None
        working_path = None
        
        for model_path in model_paths:
            if not model_path.exists():
                print(f"  ❌ {model_path.name} - 파일 없음")
                continue
            
            try:
                # 모델 로드 시도
                print(f"  🔄 {model_path.name} 로드 시도...", end="")
                
                start_time = time.time()
                model = tf.keras.models.load_model(
                    str(model_path),
                    custom_objects=custom_objects,
                    compile=False
                )
                load_time = time.time() - start_time
                
                # 간단한 추론 테스트
                test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
                output = model.predict(test_input, verbose=0)
                
                print(f" ✅ 성공! (로드: {load_time:.2f}초, 출력: {output.shape})")
                
                if not model_loaded:
                    model_loaded = True
                    working_model = model
                    working_path = model_path
                    
                    # 결과 저장
                    results[model_name] = {
                        "status": "success",
                        "working_model": model_path.name,
                        "output_shape": str(output.shape),
                        "load_time": load_time
                    }
                
                # 메모리 정리
                del model
                tf.keras.backend.clear_session()
                
            except Exception as e:
                print(f" ❌ 실패: {str(e)[:50]}...")
        
        if not model_loaded:
            results[model_name] = {
                "status": "failed",
                "error": "No working model found"
            }
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("📊 테스트 결과 요약")
    print("=" * 80)
    
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    failed_count = len(results) - success_count
    
    print(f"\n✅ 성공: {success_count}/{len(results)} 모델")
    print(f"❌ 실패: {failed_count}/{len(results)} 모델")
    
    # 권장 모델 설정
    print("\n💡 권장 모델 설정:")
    print("-" * 40)
    
    recommended_models = {}
    
    for model_name, result in results.items():
        if result["status"] == "success":
            print(f"{model_name}: {result['working_model']}")
            
            # 모델 타입 추출
            if "눈 질환" in model_name:
                recommended_models["eye_disease"] = result['working_model']
            elif "BCS" in model_name:
                recommended_models["bcs"] = result['working_model']
            elif "고양이 피부질환" in model_name:
                recommended_models["skin_cat_binary"] = result['working_model']
            elif "개 피부질환 (Binary)" in model_name:
                recommended_models["skin_dog_binary"] = result['working_model']
            elif "Multi-136" in model_name:
                recommended_models["skin_dog_multi_136"] = result['working_model']
            elif "Multi-456" in model_name:
                recommended_models["skin_dog_multi_456"] = result['working_model']
    
    # 설정 파일 저장
    config = {
        "tensorflow_version": tf.__version__,
        "test_date": str(Path().resolve()),
        "models": recommended_models,
        "test_results": results
    }
    
    config_path = Path("recommended_models_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n📝 권장 모델 설정 저장: {config_path}")
    
    # 통합 로더 업데이트 제안
    print("\n🔧 unified_model_loader.py 업데이트 제안:")
    print("-" * 40)
    print("다음 모델 경로를 사용하도록 업데이트하세요:")
    
    for key, value in recommended_models.items():
        print(f'  "{key}": "{value}"')

if __name__ == "__main__":
    test_model_loading()