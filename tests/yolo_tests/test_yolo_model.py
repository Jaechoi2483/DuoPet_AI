#!/usr/bin/env python3
"""
Test script to verify YOLO model can be loaded correctly
"""
import sys
import torch

# Add current directory to path
sys.path.append('.')

try:
    # Test importing the required modules
    from yolo_models.experimental import attempt_load
    from yolo_utils.general import check_img_size
    from yolo_utils.torch_utils import select_device
    
    print("✓ Successfully imported YOLO modules")
    
    # Test loading the model
    device = select_device('')  # Use CPU by default
    print(f"✓ Selected device: {device}")
    
    # Load the catdog model
    model = attempt_load('catdog.pt', map_location=device)
    print("✓ Successfully loaded catdog.pt model")
    
    # Get model info
    stride = int(model.stride.max())
    print(f"✓ Model stride: {stride}")
    
    # Check image size
    imgsz = check_img_size(640, s=stride)
    print(f"✓ Checked image size: {imgsz}")
    
    # Get model names (classes)
    names = model.module.names if hasattr(model, 'module') else model.names
    print(f"✓ Model classes: {names}")
    
    print("\n✅ All tests passed! YOLO model is ready to use.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()