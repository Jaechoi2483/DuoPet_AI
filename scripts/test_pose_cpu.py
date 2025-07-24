#!/usr/bin/env python3
"""
Test pose estimation model with CPU-only configuration
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Show warnings but not info

import sys
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_cpu_pose_model():
    """Test SuperAnimal model with CPU-only configuration"""
    try:
        # Import CPU adapter
        from services.behavior_analysis.superanimal_adapter_cpu import SuperAnimalQuadrupedAdapterCPU
        
        print("=== Testing SuperAnimal-Quadruped Model (CPU-only) ===")
        
        # Initialize adapter
        print("1. Initializing CPU adapter...")
        adapter = SuperAnimalQuadrupedAdapterCPU()
        print("✓ CPU adapter initialized successfully")
        
        # Create test image
        print("\n2. Creating test image...")
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"✓ Test image shape: {test_image.shape}")
        
        # Test prediction
        print("\n3. Testing CPU prediction...")
        result = adapter.predict(test_image)
        
        print("✓ CPU prediction successful!")
        print(f"  - Keypoints detected: {len(result.get('keypoints', []))}")
        print(f"  - Valid keypoints: {len(result.get('valid_keypoints', []))}")
        print(f"  - Inference time: {result.get('inference_time', 'N/A')}s")
        
        if 'error' in result:
            print(f"  - Error occurred: {result['error']}")
            
    except Exception as e:
        print(f"✗ CPU test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cpu_pose_model()