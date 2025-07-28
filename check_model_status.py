"""
모델 상태 확인 스크립트
"""
import os
from pathlib import Path
import json

def check_model_files():
    """모델 파일들 확인"""
    base_path = Path("/mnt/d/final_project/DuoPet_AI/models/health_diagnosis/skin_disease")
    
    print("=== Current Model Files Status ===\n")
    
    # 각 모델 디렉토리 확인
    model_dirs = {
        "dog_binary": base_path / "classification/dog_binary",
        "cat_binary": base_path / "classification/cat_binary",
        "dog_multi_136": base_path / "classification/dog_multi_136",
        "dog_multi_456": base_path / "classification/dog_multi_456"
    }
    
    for model_name, model_dir in model_dirs.items():
        print(f"\n{model_name}:")
        print(f"  Directory: {model_dir}")
        
        if model_dir.exists():
            # 체크포인트 파일
            checkpoint_files = list(model_dir.glob("*.index"))
            if checkpoint_files:
                print("  Checkpoint files:")
                for ckpt in checkpoint_files:
                    data_file = ckpt.with_suffix('.data-00000-of-00001')
                    if data_file.exists():
                        total_size = (ckpt.stat().st_size + data_file.stat().st_size) / (1024*1024)
                        print(f"    - {ckpt.stem}: {total_size:.2f} MB")
            
            # H5 파일
            h5_files = list(model_dir.glob("*.h5"))
            if h5_files:
                print("  H5 files:")
                for h5_file in h5_files:
                    size_mb = h5_file.stat().st_size / (1024*1024)
                    print(f"    - {h5_file.name}: {size_mb:.2f} MB")
                    
                    # from_checkpoint 파일 확인
                    if "from_checkpoint" in h5_file.name:
                        print(f"      ✓ Checkpoint-converted model found!")
        else:
            print("  ✗ Directory not found!")
    
    # 현재 레지스트리 확인
    print("\n\n=== Current Model Registry ===")
    registry_path = base_path / "model_registry.json"
    if registry_path.exists():
        with open(registry_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)
        
        for model_type, model_info in registry["classification"].items():
            print(f"\n{model_type}:")
            print(f"  Path: {model_info['path']}")
            print(f"  Output units: {model_info['output_units']}")
            
            # 실제 파일 존재 확인
            actual_path = base_path / model_info['path']
            if actual_path.exists():
                print(f"  ✓ File exists ({actual_path.stat().st_size / (1024*1024):.2f} MB)")
            else:
                print(f"  ✗ File NOT found!")
    
    print("\n\n=== Recommendations ===")
    print("1. Run the checkpoint conversion script:")
    print("   python /mnt/d/final_project/DuoPet_AI/convert_all_checkpoints.py")
    print("\n2. Test the converted models:")
    print("   python /mnt/d/final_project/DuoPet_AI/test_converted_models.py")
    print("\n3. Update the model registry:")
    print("   python /mnt/d/final_project/DuoPet_AI/update_model_registry.py")
    print("\n4. Restart the backend service to use new models")

if __name__ == "__main__":
    check_model_files()