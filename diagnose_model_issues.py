"""
모델 로딩 문제 진단 스크립트
TensorFlow 모델의 구조와 호환성 문제를 분석합니다.
"""

import os
import json
import tensorflow as tf
import numpy as np
from pathlib import Path
import h5py

def diagnose_keras_model(model_path):
    """Keras 모델 파일 진단"""
    print(f"\n=== Diagnosing Keras Model: {model_path} ===")
    
    try:
        # 1. 파일 정보 확인
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"File size: {file_size:.2f} MB")
        
        # 2. 모델 메타데이터 추출 시도
        try:
            with h5py.File(model_path, 'r') as f:
                print("\nH5 file structure:")
                def print_structure(name, obj):
                    if hasattr(obj, 'dtype'):
                        print(f"  {name}: {obj.shape} {obj.dtype}")
                    else:
                        print(f"  {name}/")
                f.visititems(print_structure)
                
                # 모델 설정 확인
                if 'model_config' in f.attrs:
                    config = json.loads(f.attrs['model_config'])
                    print(f"\nModel class: {config.get('class_name', 'Unknown')}")
                    print(f"Keras version: {f.attrs.get('keras_version', 'Unknown')}")
                    print(f"Backend: {f.attrs.get('backend', 'Unknown')}")
        except:
            print("Could not read as H5 file, trying as SavedModel...")
        
        # 3. 다양한 방법으로 모델 로딩 시도
        print("\n--- Attempting different loading methods ---")
        
        # 방법 1: compile=False로 로딩
        try:
            print("\n1. Loading with compile=False...")
            model = tf.keras.models.load_model(model_path, compile=False)
            print("✓ Success with compile=False")
            
            # 모델 구조 분석
            print("\nModel summary:")
            print(f"  Total layers: {len(model.layers)}")
            
            # Normalization layer 찾기
            norm_layers = [layer for layer in model.layers if 'normalization' in layer.name.lower()]
            if norm_layers:
                print(f"\nFound {len(norm_layers)} normalization layers:")
                for layer in norm_layers:
                    print(f"  - {layer.name} ({layer.__class__.__name__})")
                    if hasattr(layer, 'mean'):
                        print(f"    Mean shape: {layer.mean.shape if layer.mean is not None else 'None'}")
                    if hasattr(layer, 'variance'):
                        print(f"    Variance shape: {layer.variance.shape if layer.variance is not None else 'None'}")
            
            return model
            
        except Exception as e:
            print(f"✗ Failed with compile=False: {type(e).__name__}: {str(e)}")
        
        # 방법 2: 커스텀 객체로 로딩
        try:
            print("\n2. Loading with custom objects...")
            custom_objects = {
                'swish': tf.nn.swish,
                'Swish': tf.nn.swish,
                'silu': tf.nn.silu,
            }
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            print("✓ Success with custom objects")
            return model
        except Exception as e:
            print(f"✗ Failed with custom objects: {type(e).__name__}: {str(e)}")
        
        # 방법 3: 레거시 모드로 로딩
        try:
            print("\n3. Loading with legacy mode...")
            os.environ['TF_USE_LEGACY_KERAS'] = '1'
            model = tf.keras.models.load_model(model_path, compile=False)
            print("✓ Success with legacy mode")
            return model
        except Exception as e:
            print(f"✗ Failed with legacy mode: {type(e).__name__}: {str(e)}")
            
    except Exception as e:
        print(f"\n✗ General error: {type(e).__name__}: {str(e)}")
    
    return None


def diagnose_h5_model(model_path):
    """H5 모델 파일 진단"""
    print(f"\n=== Diagnosing H5 Model: {model_path} ===")
    
    try:
        with h5py.File(model_path, 'r') as f:
            print("H5 file contents:")
            
            # 그룹과 데이터셋 나열
            def print_attrs(name, obj):
                if isinstance(obj, h5py.Group):
                    print(f"\nGroup: {name}")
                    for key, val in obj.attrs.items():
                        if isinstance(val, bytes):
                            val = val.decode('utf-8', errors='ignore')[:100] + '...'
                        print(f"  @{key}: {val}")
                elif isinstance(obj, h5py.Dataset):
                    print(f"Dataset: {name} - Shape: {obj.shape}, Dtype: {obj.dtype}")
            
            f.visititems(print_attrs)
            
            # 모델 메타데이터
            if 'model_config' in f.attrs:
                config = json.loads(f.attrs['model_config'])
                print(f"\nModel architecture: {config.get('class_name', 'Unknown')}")
                
                # 레이어 분석
                if 'config' in config and 'layers' in config['config']:
                    layers = config['config']['layers']
                    print(f"Total layers: {len(layers)}")
                    
                    # Normalization layer 찾기
                    for i, layer in enumerate(layers):
                        if 'normalization' in layer.get('class_name', '').lower():
                            print(f"\nNormalization layer found at index {i}:")
                            print(f"  Name: {layer.get('name', 'Unknown')}")
                            print(f"  Config: {layer.get('config', {})}")
                            
        # 실제 로딩 테스트
        print("\n--- Loading test ---")
        return diagnose_keras_model(model_path)
        
    except Exception as e:
        print(f"✗ Error reading H5 file: {type(e).__name__}: {str(e)}")
        return None


def diagnose_checkpoint(checkpoint_path):
    """체크포인트 파일 진단"""
    print(f"\n=== Diagnosing Checkpoint: {checkpoint_path} ===")
    
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # 체크포인트 관련 파일 나열
    print("\nCheckpoint files:")
    for file in os.listdir(checkpoint_dir):
        if file.startswith(os.path.basename(checkpoint_path)):
            file_path = os.path.join(checkpoint_dir, file)
            size = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  {file}: {size:.2f} MB")
    
    # 체크포인트 변수 목록 확인
    try:
        print("\nCheckpoint variables:")
        reader = tf.train.load_checkpoint(checkpoint_path)
        shape_map = reader.get_variable_to_shape_map()
        dtype_map = reader.get_variable_to_dtype_map()
        
        # Normalization 관련 변수 찾기
        norm_vars = [var for var in shape_map.keys() if 'normalization' in var.lower()]
        if norm_vars:
            print(f"\nFound {len(norm_vars)} normalization-related variables:")
            for var in norm_vars[:5]:  # 처음 5개만 표시
                print(f"  {var}: shape={shape_map[var]}, dtype={dtype_map[var]}")
        else:
            print("\nNo normalization variables found in checkpoint")
            
        print(f"\nTotal variables: {len(shape_map)}")
        
    except Exception as e:
        print(f"✗ Error reading checkpoint: {type(e).__name__}: {str(e)}")
    
    return None


def main():
    """메인 진단 함수"""
    print("=== TensorFlow Model Diagnostics ===")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")
    
    # 프로젝트 루트 찾기
    project_root = Path(__file__).parent
    models_dir = project_root / "models" / "health_diagnosis"
    
    # 각 모델 진단
    models_to_check = [
        ("eye_disease/best_grouped_model.keras", diagnose_keras_model),
        ("bcs/bcs_efficientnet_v1.h5", diagnose_h5_model),
        ("skin_disease/checkpoint", diagnose_checkpoint),
    ]
    
    diagnosed_models = {}
    
    for model_path, diagnose_func in models_to_check:
        full_path = models_dir / model_path
        if full_path.exists():
            model = diagnose_func(str(full_path))
            diagnosed_models[model_path] = model
        else:
            print(f"\n✗ Model not found: {full_path}")
    
    # 진단 요약
    print("\n\n=== DIAGNOSIS SUMMARY ===")
    print(f"Models checked: {len(models_to_check)}")
    print(f"Successfully loaded: {sum(1 for m in diagnosed_models.values() if m is not None)}")
    
    print("\nRecommendations:")
    print("1. For normalization layer issues:")
    print("   - Use compile=False when loading")
    print("   - Recreate normalization layers manually")
    print("   - Or skip normalization layers during loading")
    print("\n2. For version compatibility:")
    print("   - Set TF_USE_LEGACY_KERAS=1 environment variable")
    print("   - Use tf.compat.v1 functions where needed")
    print("\n3. For checkpoint files:")
    print("   - Recreate model architecture first")
    print("   - Then load weights from checkpoint")
    
    return diagnosed_models


if __name__ == "__main__":
    diagnosed_models = main()