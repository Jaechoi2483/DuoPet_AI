"""
체크포인트 구조 분석 - 실제 클래스 수 확인
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from pathlib import Path

def analyze_checkpoint(checkpoint_path):
    """체크포인트의 변수 구조 분석"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {checkpoint_path}")
    print(f"{'='*80}")
    
    try:
        # 체크포인트 리더
        reader = tf.train.load_checkpoint(str(checkpoint_path))
        var_to_shape_map = reader.get_variable_to_shape_map()
        
        # 출력 레이어 찾기
        output_layers = {}
        dense_layers = {}
        
        for var_name, shape in var_to_shape_map.items():
            # Dense 레이어 찾기
            if 'dense' in var_name.lower() and 'bias' in var_name:
                layer_name = var_name.split('/')[0]
                dense_layers[layer_name] = shape[0]
            
            # 출력 레이어 후보 찾기
            if any(keyword in var_name.lower() for keyword in ['predictions', 'output', 'logits']):
                if 'bias' in var_name:
                    output_layers[var_name] = shape[0]
                elif 'kernel' in var_name and len(shape) == 2:
                    output_layers[var_name] = shape[1]
        
        # 결과 출력
        print("Dense layers found:")
        for layer, units in sorted(dense_layers.items()):
            print(f"  {layer}: {units} units")
        
        print("\nOutput layer candidates:")
        for var, classes in sorted(output_layers.items()):
            print(f"  {var}: {classes} classes")
        
        # 추정된 클래스 수
        if output_layers:
            # predictions 레이어 우선
            for var, classes in output_layers.items():
                if 'predictions' in var:
                    print(f"\n✓ Detected output classes: {classes}")
                    return classes
            
            # 그 외 출력 레이어
            classes = list(output_layers.values())[0]
            print(f"\n✓ Detected output classes: {classes}")
            return classes
        else:
            # Dense 레이어 중 가장 작은 것 (보통 마지막 레이어)
            if dense_layers:
                min_units = min(dense_layers.values())
                if min_units <= 10:  # 분류 문제의 합리적인 클래스 수
                    print(f"\n✓ Estimated output classes from dense layers: {min_units}")
                    return min_units
        
        print("\n✗ Could not determine output classes")
        return None
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def main():
    base_path = Path.cwd() / "models" / "health_diagnosis" / "skin_disease" / "classification"
    
    checkpoints = {
        'cat_binary': 'model-007-0.511353-0.772705-0.776322-0.768861',
        'dog_binary': 'model-004-0.437360-0.806570-0.806528-0.806891',
        'dog_multi_136': 'model-009-0.851382-0.821520'
    }
    
    print("="*80)
    print("CHECKPOINT STRUCTURE ANALYSIS")
    print("="*80)
    
    results = {}
    
    for model_key, checkpoint_prefix in checkpoints.items():
        checkpoint_path = base_path / model_key / checkpoint_prefix
        if Path(f"{checkpoint_path}.index").exists():
            classes = analyze_checkpoint(checkpoint_path)
            results[model_key] = classes
    
    # 결과 요약
    print(f"\n{'='*80}")
    print("SUMMARY - Detected Output Classes")
    print(f"{'='*80}")
    
    for model_key, classes in results.items():
        if classes:
            print(f"{model_key}: {classes} classes")
        else:
            print(f"{model_key}: Unknown")
    
    # 권장사항
    print("\n✅ Recommendations:")
    print("Update convert_checkpoints_clean.py with correct class numbers:")
    for model_key, classes in results.items():
        if classes:
            print(f"  '{model_key}': {{'output_classes': {classes}}}")

if __name__ == "__main__":
    main()