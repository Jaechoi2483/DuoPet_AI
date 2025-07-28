"""
BCS 모델 상세 분석
가중치 구조와 특이사항 확인
"""
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import h5py
import json

# TensorFlow 2.x 설정
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def analyze_bcs_model():
    """BCS 모델 상세 분석"""
    
    models_dir = Path("models/health_diagnosis/bcs")
    model_path = models_dir / "bcs_efficientnet_v1.h5"
    
    if not model_path.exists():
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        return
    
    print("🔍 BCS 모델 분석")
    print("=" * 60)
    
    # 1. H5 파일 구조 분석
    print("\n1️⃣ H5 파일 구조 분석:")
    try:
        with h5py.File(model_path, 'r') as f:
            print(f"  파일 크기: {model_path.stat().st_size / (1024*1024):.2f} MB")
            
            # 최상위 키들
            print(f"  최상위 구조: {list(f.keys())}")
            
            # 속성 확인
            if 'model_config' in f.attrs:
                config = json.loads(f.attrs['model_config'])
                print(f"  모델 클래스: {config.get('class_name', 'Unknown')}")
            
            print(f"  Keras 버전: {f.attrs.get('keras_version', 'Unknown')}")
            print(f"  Backend: {f.attrs.get('backend', 'Unknown')}")
            
            # 가중치 구조 탐색
            print("\n  가중치 구조:")
            weight_count = 0
            normalization_found = False
            
            def check_weights(name, obj):
                nonlocal weight_count, normalization_found
                if isinstance(obj, h5py.Dataset):
                    weight_count += 1
                    if weight_count <= 10:  # 처음 10개만 출력
                        print(f"    - {name}: shape={obj.shape}")
                    if 'normalization' in name.lower():
                        normalization_found = True
                        print(f"    ⚠️ Normalization layer 발견: {name}")
            
            if 'model_weights' in f:
                f['model_weights'].visititems(check_weights)
            
            print(f"\n  총 가중치 수: {weight_count}개")
            if normalization_found:
                print("  ⚠️ Normalization layer가 포함되어 있습니다!")
                
    except Exception as e:
        print(f"  ❌ H5 분석 실패: {e}")
    
    # 2. 모델 로드 테스트
    print("\n2️⃣ 모델 로드 테스트:")
    
    # Custom objects 정의
    custom_objects = {
        'swish': tf.nn.swish,
        'Swish': tf.keras.layers.Activation(tf.nn.swish),
        'FixedDropout': tf.keras.layers.Dropout,
    }
    
    try:
        # compile=False로 로드
        print("  - compile=False로 로드 시도...", end="")
        model = tf.keras.models.load_model(
            str(model_path),
            custom_objects=custom_objects,
            compile=False
        )
        print(" ✅ 성공!")
        
        # 모델 구조 확인
        print("\n3️⃣ 모델 구조:")
        print(f"  - 입력 shape: {model.input_shape}")
        print(f"  - 출력 shape: {model.output_shape}")
        print(f"  - 총 레이어 수: {len(model.layers)}")
        print(f"  - 총 파라미터: {model.count_params():,}")
        
        # 레이어 타입 분석
        print("\n  주요 레이어:")
        layer_types = {}
        special_layers = []
        
        for i, layer in enumerate(model.layers[:20]):  # 처음 20개 레이어
            layer_type = type(layer).__name__
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
            
            # 특수 레이어 확인
            if any(keyword in layer.name.lower() for keyword in ['normalization', 'preprocessing', 'rescaling']):
                special_layers.append((i, layer.name, layer_type))
            
            if i < 10:  # 처음 10개 상세 출력
                print(f"    {i}: {layer.name} ({layer_type})")
        
        print("\n  레이어 타입 통계:")
        for layer_type, count in sorted(layer_types.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {layer_type}: {count}개")
        
        if special_layers:
            print("\n  ⚠️ 특수 레이어 발견:")
            for idx, name, ltype in special_layers:
                print(f"    - Layer {idx}: {name} ({ltype})")
        
        # 4. 추론 테스트
        print("\n4️⃣ 추론 테스트:")
        test_input = np.random.randn(2, 224, 224, 3).astype(np.float32)
        
        # 다양한 입력 범위 테스트
        test_cases = [
            ("원본 범위", test_input),
            ("[0, 255] 범위", np.random.randint(0, 255, (2, 224, 224, 3)).astype(np.float32)),
            ("[0, 1] 범위", np.random.rand(2, 224, 224, 3).astype(np.float32)),
            ("[-1, 1] 범위", (np.random.rand(2, 224, 224, 3).astype(np.float32) - 0.5) * 2)
        ]
        
        for case_name, test_data in test_cases:
            try:
                output = model.predict(test_data, verbose=0)
                print(f"  ✅ {case_name}: 출력 shape={output.shape}, 합계={np.sum(output[0]):.4f}")
            except Exception as e:
                print(f"  ❌ {case_name}: 실패 - {e}")
        
        # 5. 가중치 분석
        print("\n5️⃣ 가중치 통계:")
        total_weights = 0
        weight_shapes = []
        
        for layer in model.layers:
            if layer.weights:
                for weight in layer.weights:
                    total_weights += weight.numpy().size
                    weight_shapes.append((layer.name, weight.name, weight.shape))
        
        print(f"  - 총 가중치 수: {total_weights:,}")
        print(f"  - 가중치 텐서 수: {len(weight_shapes)}")
        
        # 마지막 몇 개 레이어의 가중치 확인
        print("\n  마지막 레이어들의 가중치:")
        for name, weight_name, shape in weight_shapes[-10:]:
            print(f"    - {name}: {weight_name} {shape}")
        
        return model
        
    except Exception as e:
        print(f"  ❌ 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    model = analyze_bcs_model()
    
    if model:
        print("\n" + "=" * 60)
        print("💡 분석 결과:")
        print("  - BCS 모델은 EfficientNet 기반")
        print("  - Swish activation 사용")
        print("  - 3개 클래스 분류 (마른/정상/비만)")
        print("  - Normalization layer 포함 가능성 확인 필요")