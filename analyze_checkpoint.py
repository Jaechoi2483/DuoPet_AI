"""
체크포인트 분석 스크립트
실제 모델 구조를 파악하기 위해 체크포인트 내용 확인
"""

import tensorflow as tf
import os

print("=" * 80)
print("체크포인트 분석")
print("=" * 80)

checkpoint_paths = [
    "models/health_diagnosis/skin_disease/classification/cat_binary/model-007-0.511353-0.772705-0.776322-0.768861",
    "models/health_diagnosis/skin_disease/classification/dog_binary/model-004-0.437360-0.806570-0.806528-0.806891"
]

for ckpt_path in checkpoint_paths:
    print(f"\n분석 중: {ckpt_path}")
    print("-" * 60)
    
    try:
        # 체크포인트 읽기
        reader = tf.train.load_checkpoint(ckpt_path)
        
        # 변수 목록 확인
        var_to_shape_map = reader.get_variable_to_shape_map()
        
        print(f"총 변수 개수: {len(var_to_shape_map)}")
        print("\n주요 변수들:")
        
        # Conv2D 레이어 찾기
        conv_layers = {}
        for var_name, shape in var_to_shape_map.items():
            if 'conv2d' in var_name.lower():
                layer_name = var_name.split('/')[0]
                if layer_name not in conv_layers:
                    conv_layers[layer_name] = {}
                
                if 'kernel' in var_name:
                    conv_layers[layer_name]['kernel'] = shape
                elif 'bias' in var_name:
                    conv_layers[layer_name]['bias'] = shape
        
        # Conv 레이어 정보 출력
        for layer, info in sorted(conv_layers.items()):
            if 'bias' in info:
                print(f"  {layer}: filters={info['bias'][0]}, kernel_shape={info.get('kernel', 'unknown')}")
        
        # Dense 레이어 찾기
        print("\nDense 레이어:")
        for var_name, shape in var_to_shape_map.items():
            if 'dense' in var_name.lower() and 'bias' in var_name:
                layer_name = var_name.split('/')[0]
                print(f"  {layer}: units={shape[0]}")
        
        # 출력 레이어 찾기
        print("\n출력 레이어 추정:")
        for var_name, shape in var_to_shape_map.items():
            if 'logits' in var_name or 'output' in var_name or (var_name.startswith('dense') and 'bias' in var_name):
                if shape[0] <= 10:  # 분류 문제의 클래스 수는 보통 10 이하
                    print(f"  {var_name}: shape={shape} (가능한 출력 레이어)")
        
    except Exception as e:
        print(f"오류: {e}")

print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)