#!/usr/bin/env python3
"""
Test the fallback pose estimation system
"""
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_fallback():
    """Test the fallback pose estimation"""
    print("=== Testing Fallback Pose Estimation ===")
    
    try:
        from services.behavior_analysis.pose_estimation_fallback import PoseEstimationFallback
        
        # Initialize fallback
        print("1. Initializing fallback estimator...")
        fallback = PoseEstimationFallback()
        print(f"✓ Initialized with {fallback.num_keypoints} keypoints")
        
        # Create test image and bbox
        print("\n2. Creating test data...")
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_bbox = [100, 100, 300, 300]  # x1, y1, x2, y2
        print(f"✓ Test image shape: {test_image.shape}")
        print(f"✓ Test bbox: {test_bbox}")
        
        # Test keypoint estimation
        print("\n3. Estimating keypoints...")
        result = fallback.estimate_keypoints_from_bbox(test_image, test_bbox)
        
        print("✓ Keypoint estimation successful!")
        print(f"  - Keypoints: {len(result['keypoints'])}")
        print(f"  - Valid keypoints: {len(result['valid_keypoints'])}")
        print(f"  - Method: {result.get('method', 'unknown')}")
        print(f"  - First 3 keypoints:")
        for i in range(min(3, len(result['keypoints']))):
            kp = result['keypoints'][i]
            conf = result['confidence_scores'][i]
            name = result['keypoint_names'][i]
            print(f"    - {name}: ({kp[0]:.1f}, {kp[1]:.1f}) conf={conf:.2f}")
        
        # Test feature extraction
        print("\n4. Extracting features...")
        features = fallback.extract_pose_features(result)
        print(f"✓ Feature extraction successful!")
        print(f"  - Feature vector shape: {features.shape}")
        print(f"  - Feature vector stats: min={features.min():.2f}, max={features.max():.2f}, mean={features.mean():.2f}")
        
        # Test with motion
        print("\n5. Testing with motion...")
        test_bbox2 = [110, 105, 310, 305]  # Slightly moved
        result2 = fallback.estimate_keypoints_from_bbox(test_image, test_bbox2)
        features2 = fallback.extract_pose_features(result2)
        
        feature_diff = np.linalg.norm(features2 - features)
        print(f"✓ Motion handling successful!")
        print(f"  - Feature difference: {feature_diff:.4f}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_fallback()
    sys.exit(0 if success else 1)