"""
원래 작동하던 모델로 레지스트리 복원
"""
import json
from pathlib import Path

def restore_registry():
    """모델 레지스트리를 원래 상태로 복원"""
    
    registry_path = Path.cwd() / "models" / "health_diagnosis" / "skin_disease" / "model_registry.json"
    
    # 백업이 있다면 복원
    backup_path = registry_path.with_suffix('.json.backup')
    if backup_path.exists():
        print(f"Restoring from backup: {backup_path}")
        with open(backup_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)
    else:
        # 백업이 없으면 원래 설정으로
        print("Creating original registry configuration")
        registry = {
            "classification": {
                "dog_binary": {
                    "path": "classification/dog_binary/dog_binary_model_tf2.h5",
                    "model_type": "h5",
                    "input_shape": [224, 224, 3],
                    "output_units": 1
                },
                "cat_binary": {
                    "path": "classification/cat_binary/cat_binary_model_tf2.h5",
                    "model_type": "h5",
                    "input_shape": [224, 224, 3],
                    "output_units": 1
                },
                "dog_multi_136": {
                    "path": "classification/dog_multi_136/dog_multi_136_model_tf2.h5",
                    "model_type": "h5",
                    "input_shape": [224, 224, 3],
                    "output_units": 3
                },
                "dog_multi_456": {
                    "path": "classification/dog_multi_456/dog_multi_456_model_tf2.h5",
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
    
    # 레지스트리 저장
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    
    print("✓ Model registry restored to original state!")
    print("\nCurrent models:")
    for model_type, model_info in registry["classification"].items():
        print(f"  - {model_type}: {model_info['path']} (output_units={model_info['output_units']})")
    
    print("\n✅ Next steps:")
    print("1. Restart the backend service")
    print("2. Test with skin disease images")
    print("\nNote: The models were already working with varying outputs (60-74%),")
    print("      just not detecting disease correctly. This is likely a threshold")
    print("      or interpretation issue, not a model weight issue.")

if __name__ == "__main__":
    restore_registry()