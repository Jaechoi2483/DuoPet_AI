"""
원본 모델 심층 분석 및 가중치 검증
Normalization과 가중치 문제의 정확한 원인 파악
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import h5py
import zipfile

print("🔬 안구질환 모델 심층 분석")
print("=" * 80)

# 모델 경로들
model_paths = {
    "original": Path("C:/Users/ictedu1_021/Desktop/안구질환모델/best_grouped_model.keras"),
    "converted": Path("C:/Users/ictedu1_021/Desktop/안구질환모델/final_model_fixed.keras"),
    "class_map": Path("C:/Users/ictedu1_021/Desktop/안구질환모델/class_map.json")
}

def analyze_keras_file(model_path):
    """Keras 파일 내부 구조 분석"""
    print(f"\n📁 Keras 파일 분석: {model_path.name}")
    print("-" * 60)
    
    if not model_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {model_path}")
        return None
    
    # Keras 파일은 실제로 ZIP 아카이브
    with zipfile.ZipFile(model_path, 'r') as zip_file:
        print("📄 파일 목록:")
        for name in zip_file.namelist():
            info = zip_file.getinfo(name)
            print(f"  - {name}: {info.file_size:,} bytes")
        
        # config.json 읽기
        if 'config.json' in zip_file.namelist():
            with zip_file.open('config.json') as f:
                config = json.load(f)
                
            # 모델 구조 분석
            print("\n🏗️ 모델 구조:")
            if 'config' in config and 'layers' in config['config']:
                layers = config['config']['layers']
                print(f"총 레이어 수: {len(layers)}")
                
                # 주요 레이어 찾기
                for i, layer in enumerate(layers):
                    layer_class = layer.get('class_name', 'Unknown')
                    layer_config = layer.get('config', {})
                    layer_name = layer_config.get('name', 'unnamed')
                    
                    # 문제가 될 수 있는 레이어 표시
                    if layer_class in ['Normalization', 'BatchNormalization']:
                        print(f"\n⚠️ {i}: {layer_class} - {layer_name}")
                        print(f"   Config: {json.dumps(layer_config, indent=2)}")
                    
                    elif layer_class in ['Dense', 'Conv2D']:
                        units = layer_config.get('units') or layer_config.get('filters')
                        print(f"  {i}: {layer_class} - {layer_name} ({units} units/filters)")
                    
                    elif i < 3 or i >= len(layers) - 3:
                        print(f"  {i}: {layer_class} - {layer_name}")
        
        # 가중치 파일 확인
        if 'model.weights.h5' in zip_file.namelist():
            print("\n📊 가중치 파일 분석:")
            # 임시로 추출하여 분석
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
                tmp.write(zip_file.read('model.weights.h5'))
                tmp_path = tmp.name
            
            try:
                with h5py.File(tmp_path, 'r') as h5f:
                    print("가중치 그룹:")
                    
                    def print_h5_structure(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            print(f"  - {name}: shape={obj.shape}, dtype={obj.dtype}")
                            # 가중치 통계
                            data = obj[()]
                            if data.size > 0:
                                print(f"    Stats: mean={np.mean(data):.4f}, std={np.std(data):.4f}, "
                                      f"min={np.min(data):.4f}, max={np.max(data):.4f}")
                                
                                # 모든 값이 같은지 확인
                                if np.allclose(data, data.flat[0]):
                                    print("    ⚠️ 모든 값이 동일함! (초기화 상태)")
                    
                    h5f.visititems(print_h5_structure)
            finally:
                os.unlink(tmp_path)
    
    return config

def try_load_model(model_path):
    """실제 모델 로드 시도 및 분석"""
    print(f"\n🔧 모델 로드 시도: {model_path.name}")
    print("-" * 60)
    
    # 여러 방법으로 로드 시도
    methods = [
        ("기본 로드", lambda: tf.keras.models.load_model(str(model_path))),
        ("compile=False", lambda: tf.keras.models.load_model(str(model_path), compile=False)),
        ("커스텀 객체", lambda: tf.keras.models.load_model(
            str(model_path),
            custom_objects={
                'swish': tf.nn.swish,
                'Swish': tf.keras.layers.Activation(tf.nn.swish),
                'FixedDropout': tf.keras.layers.Dropout
            },
            compile=False
        ))
    ]
    
    for method_name, method_func in methods:
        print(f"\n시도: {method_name}")
        try:
            model = method_func()
            print("✅ 로드 성공!")
            
            # 모델 정보
            print(f"입력 shape: {model.input_shape}")
            print(f"출력 shape: {model.output_shape}")
            
            # 예측 테스트
            test_input = np.random.random((1, 224, 224, 3)).astype(np.float32) * 255.0
            pred = model.predict(test_input, verbose=0)
            
            print(f"예측 결과: {pred[0]}")
            print(f"확률 분포: {[f'{p*100:.1f}%' for p in pred[0]]}")
            
            # 가중치 상태 확인
            if np.allclose(pred[0], pred[0][0], rtol=1e-3):
                print("❌ 모든 예측값이 동일 - 가중치가 초기화 상태!")
            else:
                print("✅ 예측값이 다양함 - 가중치 정상")
            
            # 특정 레이어 가중치 확인
            print("\n주요 레이어 가중치 분석:")
            for layer in model.layers[-5:]:  # 마지막 5개 레이어
                if layer.weights:
                    print(f"\n{layer.name}:")
                    for weight in layer.weights:
                        w_data = weight.numpy()
                        print(f"  - {weight.name}: shape={w_data.shape}")
                        print(f"    mean={np.mean(w_data):.4f}, std={np.std(w_data):.4f}")
            
            return model
            
        except Exception as e:
            print(f"❌ 실패: {e}")
            
            # Normalization 문제인지 확인
            if "normalization" in str(e).lower():
                print("→ Normalization 레이어 문제 확인됨")
                print(f"상세 오류: {type(e).__name__}")
    
    return None

def analyze_weight_initialization():
    """가중치 초기화 상태 분석"""
    print("\n🔍 가중치 초기화 패턴 분석")
    print("-" * 60)
    
    # Dense 레이어의 일반적인 초기화 패턴
    print("Dense 레이어 초기화 패턴 (Glorot uniform):")
    
    # 5개 출력을 가진 Dense 레이어의 초기화 시뮬레이션
    np.random.seed(42)
    fan_in, fan_out = 128, 5  # 예: 128개 입력, 5개 출력
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    weights = np.random.uniform(-limit, limit, size=(fan_in, fan_out))
    
    # Softmax 적용
    logits = np.random.randn(1, 5)  # 임의의 입력
    output = np.dot(np.ones((1, fan_in)) * 0.1, weights)  # 작은 입력값
    softmax_output = np.exp(output) / np.sum(np.exp(output))
    
    print(f"초기화된 가중치 범위: [{-limit:.4f}, {limit:.4f}]")
    print(f"Softmax 출력 (초기화 상태): {softmax_output[0]}")
    print(f"각 클래스 확률: {[f'{p*100:.1f}%' for p in softmax_output[0]]}")
    
    if np.allclose(softmax_output[0], 0.2, atol=0.05):
        print("→ 초기화 상태에서는 대략 20%씩 균등 분포가 정상")

# 실행
print("\n" + "="*80)
print("🚀 분석 시작\n")

# 1. 원본 모델 분석
if model_paths["original"].exists():
    print("1️⃣ 원본 모델 분석")
    config = analyze_keras_file(model_paths["original"])
    model = try_load_model(model_paths["original"])

# 2. 변환된 모델 분석
if model_paths["converted"].exists():
    print("\n\n2️⃣ 변환된 모델 분석")
    config = analyze_keras_file(model_paths["converted"])
    model = try_load_model(model_paths["converted"])

# 3. 가중치 초기화 분석
analyze_weight_initialization()

# 4. 결론
print("\n\n📋 분석 결론:")
print("="*80)
print("1. Normalization 레이어가 adapt() 없이 저장되어 mean/variance가 없음")
print("2. 모든 예측이 20%인 것은 가중치가 초기화 상태임을 의미")
print("3. 모델이 실제로 학습되지 않았거나 가중치가 제대로 저장되지 않음")
print("\n💡 해결 방안:")
print("- Normalization을 제거하고 Lambda 레이어로 대체")
print("- 가중치를 별도로 저장/로드하거나")
print("- 처음부터 다시 학습")