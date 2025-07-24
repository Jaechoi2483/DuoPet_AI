#!/usr/bin/env python3
"""
Check if all required model files are in place
"""
import os
from pathlib import Path

def check_models():
    """Check for required model files"""
    base_path = Path(__file__).parent / "models"
    
    required_files = [
        # Detection models
        "behavior_analysis/detection/behavior_yolo_catdog_v1_original.pt",
        "behavior_analysis/detection/behavior_yolo_catdog_v1.pt",
        "behavior_analysis/detection/config.yaml",
        
        # Classification models
        "behavior_analysis/classification/behavior_cat_lstm_v1.pth",
        "behavior_analysis/classification/behavior_dog_lstm_v1.pth",
        "behavior_analysis/classification/config.yaml",
        
        # Custom YOLO modules
        "yolo_models/__init__.py",
        "yolo_models/experimental.py",
        "yolo_models/common.py",
        "yolo_models/yolo.py",
        
        # Custom YOLO utils
        "yolo_utils/__init__.py",
        "yolo_utils/general.py",
        "yolo_utils/torch_utils.py",
    ]
    
    print("=" * 60)
    print("Model Files Check")
    print("=" * 60)
    
    all_present = True
    
    for file_path in required_files:
        full_path = base_path / file_path
        exists = full_path.exists()
        
        status = "✅" if exists else "❌"
        print(f"{status} {file_path}")
        
        if not exists:
            all_present = False
            
    print("=" * 60)
    
    if all_present:
        print("✅ All model files are present!")
    else:
        print("❌ Some model files are missing!")
        print("\nMissing files need to be copied from:")
        print("- Original models: C:\\Users\\ictedu1_021\\Downloads\\03.AI모델\\03.AI모델")
        print("- Or created as placeholders for testing")
        
    # Check custom YOLO import
    print("\n" + "=" * 60)
    print("Testing Custom YOLO Import")
    print("=" * 60)
    
    import sys
    sys.path.insert(0, str(base_path))
    
    try:
        from yolo_models.experimental import attempt_load
        from yolo_utils.general import non_max_suppression
        from yolo_utils.torch_utils import select_device
        print("✅ Custom YOLOv5 modules imported successfully!")
    except ImportError as e:
        print(f"❌ Failed to import custom YOLOv5: {e}")

if __name__ == "__main__":
    check_models()