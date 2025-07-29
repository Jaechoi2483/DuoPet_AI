"""
간단한 체크포인트 테스트 - 실제 모델로 로드해서 확인
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from pathlib import Path

def try_load_checkpoint(model_key, checkpoint_path, output_classes):
    """특정 클래스 수로 체크포인트 로드 시도"""
    print(f"\nTrying {model_key} with {output_classes} classes...")
    
    try:
        # 모델 생성
        base_model = InceptionResNetV2(
            include_top=False, 
            weights='imagenet', 
            input_shape=(224, 224, 3),
            pooling='avg'
        )
        
        model = Sequential([
            base_model,
            Dense(2048, activation='relu'),
            Dense(output_classes, activation='sigmoid' if output_classes == 2 else 'softmax')
        ])
        
        # 체크포인트 로드 시도
        model.load_weights(str(checkpoint_path))
        print(f"  ✓ SUCCESS with {output_classes} classes!")
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "shape" in error_msg:
            print(f"  ✗ Failed: {error_msg}")
            # 에러 메시지에서 실제 클래스 수 추출 시도
            if "Received incompatible tensor with shape" in error_msg:
                import re
                match = re.search(r'shape \((\d+),\)', error_msg)
                if match:
                    actual_classes = int(match.group(1))
                    print(f"  → Actual classes: {actual_classes}")
                    return actual_classes
        else:
            print(f"  ✗ Other error: {error_msg[:100]}...")
        return False

def main():
    base_path = Path.cwd() / "models" / "health_diagnosis" / "skin_disease" / "classification"
    
    checkpoints = {
        'cat_binary': 'model-007-0.511353-0.772705-0.776322-0.768861',
        'dog_binary': 'model-004-0.437360-0.806570-0.806528-0.806891',
        'dog_multi_136': 'model-009-0.851382-0.821520'
    }
    
    print("="*80)
    print("SIMPLE CHECKPOINT TEST")
    print("="*80)
    print("Testing different class numbers by trial and error")
    
    results = {}
    
    for model_key, checkpoint_prefix in checkpoints.items():
        checkpoint_path = base_path / model_key / checkpoint_prefix
        
        if not Path(f"{checkpoint_path}.index").exists():
            print(f"\n✗ Checkpoint not found: {checkpoint_path}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Testing {model_key}")
        print(f"{'='*70}")
        
        # 가능한 클래스 수 시도
        possible_classes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 136, 456]
        
        found = False
        for num_classes in possible_classes:
            result = try_load_checkpoint(model_key, checkpoint_path, num_classes)
            if result is True:
                results[model_key] = num_classes
                found = True
                break
            elif isinstance(result, int):
                # 에러 메시지에서 실제 클래스 수를 찾은 경우
                print(f"\nRetrying with detected {result} classes...")
                if try_load_checkpoint(model_key, checkpoint_path, result) is True:
                    results[model_key] = result
                    found = True
                    break
        
        if not found:
            results[model_key] = None
    
    # 결과 요약
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    
    for model_key, classes in results.items():
        if classes:
            print(f"✓ {model_key}: {classes} classes")
        else:
            print(f"✗ {model_key}: Could not determine")
    
    # 최종 권장사항
    print("\n✅ Use these values in convert_checkpoints_corrected.py:")
    for model_key, classes in results.items():
        if classes:
            print(f"  '{model_key}': {{'output_classes': {classes}}}")

if __name__ == "__main__":
    main()