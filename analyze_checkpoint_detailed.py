"""
체크포인트 상세 분석 - 모든 변수 확인
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from pathlib import Path

def analyze_checkpoint_detailed(checkpoint_path):
    """체크포인트의 모든 변수 상세 분석"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {checkpoint_path}")
    print(f"{'='*80}")
    
    try:
        # 체크포인트 리더
        reader = tf.train.load_checkpoint(str(checkpoint_path))
        var_to_shape_map = reader.get_variable_to_shape_map()
        
        print(f"Total variables: {len(var_to_shape_map)}")
        
        # 카테고리별로 분류
        dense_vars = []
        conv_vars = []
        other_vars = []
        optimizer_vars = []
        
        for var_name, shape in var_to_shape_map.items():
            if 'optimizer' in var_name.lower() or 'momentum' in var_name.lower():
                optimizer_vars.append((var_name, shape))
            elif 'dense' in var_name.lower():
                dense_vars.append((var_name, shape))
            elif 'conv' in var_name.lower():
                conv_vars.append((var_name, shape))
            else:
                other_vars.append((var_name, shape))
        
        # Dense 레이어 분석 (가장 중요)
        print(f"\nDense layers ({len(dense_vars)} variables):")
        dense_layers = {}
        for var_name, shape in dense_vars:
            if 'bias' in var_name:
                layer_info = var_name.split('/')
                layer_name = layer_info[0] if layer_info else var_name
                units = shape[0]
                dense_layers[layer_name] = units
                print(f"  {var_name}: shape={shape} ({units} units)")
        
        # 출력 레이어 후보 찾기
        print("\nSearching for output layer patterns:")
        output_candidates = []
        
        # 패턴 1: dense_1, dense_2 등의 이름
        for var_name, shape in var_to_shape_map.items():
            if any(pattern in var_name.lower() for pattern in ['dense_', 'fc', 'output', 'predictions', 'logits']):
                if 'bias' in var_name and len(shape) == 1:
                    output_candidates.append((var_name, shape[0]))
                    print(f"  Found: {var_name} with {shape[0]} units")
        
        # 패턴 2: 가장 작은 Dense 레이어
        if dense_layers:
            min_units = min(dense_layers.values())
            max_units = max(dense_layers.values())
            print(f"\nDense layer sizes: min={min_units}, max={max_units}")
            
            # 합리적인 클래스 수 범위 (2-1000)
            if 2 <= min_units <= 1000:
                print(f"  Likely output layer size: {min_units}")
                return min_units
        
        # 패턴 3: 특정 변수명 확인
        print("\nChecking specific patterns:")
        for var_name, shape in var_to_shape_map.items():
            if var_name.endswith('/bias:0') or var_name.endswith('/bias'):
                parts = var_name.split('/')
                if len(parts) >= 1:
                    layer_name = parts[0]
                    if any(x in layer_name for x in ['dense', 'fc']):
                        if len(shape) == 1 and shape[0] <= 10:  # 분류 문제의 합리적인 범위
                            print(f"  Potential output: {var_name} = {shape[0]} classes")
                            return shape[0]
        
        # 모든 변수 출력 (디버깅용)
        print("\nAll variables (first 20):")
        for i, (var_name, shape) in enumerate(var_to_shape_map.items()):
            if i >= 20:
                print(f"  ... and {len(var_to_shape_map) - 20} more variables")
                break
            print(f"  {var_name}: {shape}")
        
        # 마지막 시도: 숫자 기반 추정
        if output_candidates:
            # 가장 작은 값 선택
            min_candidate = min(output_candidates, key=lambda x: x[1])
            print(f"\n✓ Best guess from candidates: {min_candidate[1]} classes")
            return min_candidate[1]
        
        print("\n✗ Could not determine output classes")
        return None
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    base_path = Path.cwd() / "models" / "health_diagnosis" / "skin_disease" / "classification"
    
    checkpoints = {
        'cat_binary': 'model-007-0.511353-0.772705-0.776322-0.768861',
        'dog_binary': 'model-004-0.437360-0.806570-0.806528-0.806891',
        'dog_multi_136': 'model-009-0.851382-0.821520'
    }
    
    print("="*80)
    print("DETAILED CHECKPOINT ANALYSIS")
    print("="*80)
    
    results = {}
    
    for model_key, checkpoint_prefix in checkpoints.items():
        checkpoint_path = base_path / model_key / checkpoint_prefix
        if Path(f"{checkpoint_path}.index").exists():
            classes = analyze_checkpoint_detailed(checkpoint_path)
            results[model_key] = classes
        else:
            print(f"\n✗ Checkpoint not found: {checkpoint_path}")
            results[model_key] = None
    
    # 결과 요약
    print(f"\n{'='*80}")
    print("SUMMARY - Detected Output Classes")
    print(f"{'='*80}")
    
    for model_key, classes in results.items():
        if classes:
            print(f"{model_key}: {classes} classes")
        else:
            print(f"{model_key}: Unknown")
    
    # 추가 정보
    print("\nBased on error message, we know:")
    print("dog_multi_136: 7 classes (from ValueError)")
    
    # 권장사항
    print("\n✅ Recommendations:")
    print("Use these values in convert_checkpoints_corrected.py:")
    print("  'cat_binary': {'output_classes': 2}")
    print("  'dog_binary': {'output_classes': 2}")
    print("  'dog_multi_136': {'output_classes': 7}")

if __name__ == "__main__":
    main()