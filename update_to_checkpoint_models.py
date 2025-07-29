"""
체크포인트 변환 모델(231MB)을 사용하도록 레지스트리 업데이트
"""
import json
from pathlib import Path

def update_registry():
    """모델 레지스트리를 체크포인트 변환 모델로 업데이트"""
    
    registry_path = Path.cwd() / "models" / "health_diagnosis" / "skin_disease" / "model_registry.json"
    
    # 새로운 레지스트리 설정 - 체크포인트 변환 모델 사용
    registry = {
        "classification": {
            "dog_binary": {
                "path": "classification/dog_binary/dog_binary_model_from_checkpoint_v2.h5",
                "model_type": "h5",
                "input_shape": [224, 224, 3],
                "output_units": 2,  # 원래는 2개 출력이어야 함
                "description": "Converted from checkpoint (231MB)"
            },
            "cat_binary": {
                "path": "classification/cat_binary/cat_binary_model_from_checkpoint_v2.h5", 
                "model_type": "h5",
                "input_shape": [224, 224, 3],
                "output_units": 2,  # 원래는 2개 출력이어야 함
                "description": "Converted from checkpoint (231MB)"
            },
            "dog_multi_136": {
                "path": "classification/dog_multi_136/dog_multi_136_model_tf2_perfect.h5",
                "model_type": "h5",
                "input_shape": [224, 224, 3],
                "output_units": 3
            },
            "dog_multi_456": {
                "path": "classification/dog_multi_456/dog_multi_456_model_tf2_perfect.h5",
                "model_type": "h5",
                "input_shape": [224, 224, 3],
                "output_units": 3
            }
        },
        "segmentation": {
            "dog_A1": {"path": "segmentation/dog_A1", "checkpoint_prefix": "A1"},
            "dog_A2": {"path": "segmentation/dog_A2", "checkpoint_prefix": "A2"},
            "dog_A3": {"path": "segmentation/dog_A3", "checkpoint_prefix": "A3"},
            "dog_A4": {"path": "segmentation/dog_A4", "checkpoint_prefix": "A4"},
            "dog_A5": {"path": "segmentation/dog_A5", "checkpoint_prefix": "A5"},
            "dog_A6": {"path": "segmentation/dog_A6", "checkpoint_prefix": "A6"},
            "cat_A2": {"path": "segmentation/cat_A2", "checkpoint_prefix": "A2"}
        }
    }
    
    # 현재 레지스트리 백업
    if registry_path.exists():
        backup_path = registry_path.with_suffix('.json.backup_10mb')
        with open(registry_path, 'r', encoding='utf-8') as f:
            current = json.load(f)
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(current, f, indent=2, ensure_ascii=False)
        print(f"✓ Current registry backed up to: {backup_path.name}")
    
    # 새 레지스트리 저장
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    
    print("✓ Model registry updated to use checkpoint-converted models!")
    print("\nUpdated models:")
    for model_type, model_info in registry["classification"].items():
        if "binary" in model_type:
            print(f"  - {model_type}: {model_info['path']}")
            print(f"    Description: {model_info.get('description', 'N/A')}")
            print(f"    Output units: {model_info['output_units']}")
    
    print("\n✅ Next steps:")
    print("1. Restart the backend service")
    print("2. Test with skin disease images")
    print("\nNote: These models are 231MB (vs 10MB), indicating they have full weights")

if __name__ == "__main__":
    update_registry()