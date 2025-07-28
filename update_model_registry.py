"""
Model Registry 업데이트 스크립트
체크포인트에서 변환된 모델들을 레지스트리에 추가
"""
import json
from pathlib import Path

def update_model_registry():
    """모델 레지스트리를 업데이트하여 체크포인트에서 변환된 모델 추가"""
    
    # Windows 호환 경로
    registry_path = Path.cwd() / "models" / "health_diagnosis" / "skin_disease" / "model_registry.json"
    
    # 기존 레지스트리 로드
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    # 백업
    backup_path = registry_path.with_suffix('.json.backup')
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    
    # Binary 모델들 업데이트 - 2개 출력으로 변경
    if "dog_binary" in registry["classification"]:
        registry["classification"]["dog_binary"]["output_units"] = 2
        registry["classification"]["dog_binary"]["path"] = "classification/dog_binary/dog_binary_model_from_checkpoint.h5"
        registry["classification"]["dog_binary"]["note"] = "Restored from checkpoint with proper 2-output structure"
    
    if "cat_binary" in registry["classification"]:
        registry["classification"]["cat_binary"]["output_units"] = 2
        registry["classification"]["cat_binary"]["path"] = "classification/cat_binary/cat_binary_model_from_checkpoint.h5"
        registry["classification"]["cat_binary"]["note"] = "Restored from checkpoint with proper 2-output structure"
    
    # Multi-class 모델들도 업데이트
    if "dog_multi_136" in registry["classification"]:
        registry["classification"]["dog_multi_136"]["path"] = "classification/dog_multi_136/dog_multi_136_model_from_checkpoint.h5"
        registry["classification"]["dog_multi_136"]["note"] = "Restored from checkpoint"
    
    if "dog_multi_456" in registry["classification"]:
        registry["classification"]["dog_multi_456"]["path"] = "classification/dog_multi_456/dog_multi_456_model_from_checkpoint.h5"
        registry["classification"]["dog_multi_456"]["note"] = "Restored from checkpoint"
    
    # 레지스트리 저장
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    
    print("✓ Model registry updated!")
    print("\nUpdated models:")
    for model_type in ["dog_binary", "cat_binary", "dog_multi_136", "dog_multi_456"]:
        if model_type in registry["classification"]:
            info = registry["classification"][model_type]
            print(f"  - {model_type}: output_units={info['output_units']}, path={info['path']}")

if __name__ == "__main__":
    update_model_registry()