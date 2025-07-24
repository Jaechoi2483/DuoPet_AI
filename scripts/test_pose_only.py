#!/usr/bin/env python3
"""
Test pose estimation model independently
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

import sys
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_pose_model():
    """Test SuperAnimal model loading and inference"""
    try:
        # Import adapter
        from services.behavior_analysis.superanimal_adapter import SuperAnimalQuadrupedAdapter
        
        print("=== Testing SuperAnimal-Quadruped Model ===")
        
        # Initialize adapter
        print("1. Initializing adapter...")
        adapter = SuperAnimalQuadrupedAdapter()
        print("✓ Adapter initialized successfully")
        
        # Create test image
        print("\n2. Creating test image...")
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"✓ Test image shape: {test_image.shape}")
        
        # Test prediction
        print("\n3. Testing prediction...")
        print("  - Calling predict method...")
        try:
            # Add timeout and more debugging
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Prediction timed out after 10 seconds")
            
            # Set timeout
            if os.name != 'nt':  # Unix/Linux only
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)
            
            print("  - About to call adapter.predict()...")
            result = adapter.predict(test_image)
            
            if os.name != 'nt':
                signal.alarm(0)  # Cancel timeout
                
            print("✓ Prediction successful!")
            print(f"  - Keypoints detected: {len(result.get('keypoints', []))}")
            print(f"  - Valid keypoints: {len(result.get('valid_keypoints', []))}")
            print(f"  - First 5 keypoint names: {result.get('keypoint_names', [])[:5]}")
        except TimeoutError as e:
            print(f"✗ Prediction timed out: {str(e)}")
        except Exception as e:
            print(f"✗ Prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pose_model()