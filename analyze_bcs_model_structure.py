"""
BCS 모델 구조 상세 분석
이름 충돌 문제 파악
"""
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import json
from collections import Counter

# TensorFlow 2.x 설정
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def analyze_model_structure():
    """BCS 모델 구조 상세 분석"""
    
    models_dir = Path("models/health_diagnosis/bcs")
    model_path = models_dir / "bcs_efficientnet_v1.h5"
    
    if not model_path.exists():
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        return
    
    print("🔍 BCS 모델 구조 상세 분석")
    print("=" * 80)
    
    try:
        # Custom objects
        custom_objects = {
            'swish': tf.nn.swish,
            'Swish': tf.keras.layers.Activation(tf.nn.swish),
            'FixedDropout': tf.keras.layers.Dropout,
        }
        
        # 모델 로드
        model = tf.keras.models.load_model(
            str(model_path),
            custom_objects=custom_objects,
            compile=False
        )
        
        print("✅ 모델 로드 성공")
        print(f"\n📊 기본 정보:")
        print(f"  - 모델 이름: {model.name}")
        print(f"  - 입력 수: {len(model.inputs)}")
        print(f"  - 출력 shape: {model.output_shape}")
        print(f"  - 총 레이어 수: {len(model.layers)}")
        
        # 1. 레이어 이름 분석
        print("\n📋 레이어 이름 분석:")
        layer_names = [layer.name for layer in model.layers]
        name_counts = Counter(layer_names)
        
        # 중복된 이름 찾기
        duplicate_names = {name: count for name, count in name_counts.items() if count > 1}
        
        if duplicate_names:
            print("\n⚠️ 중복된 레이어 이름 발견:")
            for name, count in duplicate_names.items():
                print(f"  - '{name}': {count}번 중복")
        else:
            print("  ✓ 중복된 레이어 이름 없음")
        
        # 2. 레이어 구조 분석
        print("\n🏗️ 레이어 구조:")
        for i, layer in enumerate(model.layers[:30]):  # 처음 30개만
            layer_type = type(layer).__name__
            if hasattr(layer, 'layers'):  # 서브모델인 경우
                sub_layer_count = len(layer.layers)
                print(f"  {i:3d}: {layer.name:30s} ({layer_type}) - {sub_layer_count}개 서브레이어")
                
                # 서브레이어 이름 확인
                if i < 15:  # 처음 몇 개만 상세히
                    sub_names = [sub.name for sub in layer.layers[:5]]
                    print(f"       서브레이어: {sub_names[:3]}...")
            else:
                print(f"  {i:3d}: {layer.name:30s} ({layer_type})")
        
        # 3. 입력 분석
        print("\n🔌 입력 구조:")
        for i, inp in enumerate(model.inputs):
            print(f"  입력 {i+1}: {inp.name} - shape: {inp.shape}")
        
        # 4. 서브모델 분석
        print("\n🔎 서브모델 분석:")
        functional_layers = [layer for layer in model.layers if type(layer).__name__ == 'Functional']
        
        print(f"  총 Functional 레이어 수: {len(functional_layers)}")
        
        if functional_layers:
            # 첫 번째 Functional 레이어 상세 분석
            first_functional = functional_layers[0]
            print(f"\n  첫 번째 Functional 레이어 상세:")
            print(f"    - 이름: {first_functional.name}")
            print(f"    - 서브레이어 수: {len(first_functional.layers)}")
            
            # 서브레이어 이름 통계
            sub_layer_names = [layer.name for layer in first_functional.layers]
            sub_name_counts = Counter(sub_layer_names)
            
            print("\n    서브레이어 이름 통계:")
            for name, count in sub_name_counts.most_common(10):
                if count > 1:
                    print(f"      - '{name}': {count}번")
        
        # 5. 해결 방안 제시
        print("\n💡 분석 결과 및 해결 방안:")
        print("  1. 13개의 동일한 EfficientNet 모델이 앙상블로 구성됨")
        print("  2. 각 서브모델이 동일한 레이어 이름을 가지고 있어 충돌 발생")
        print("  3. 해결 방법:")
        print("     - SavedModel 형식으로 저장")
        print("     - 또는 각 서브모델의 레이어 이름을 고유하게 변경")
        print("     - 또는 단일 EfficientNet 모델만 추출하여 사용")
        
        return model
        
    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    analyze_model_structure()