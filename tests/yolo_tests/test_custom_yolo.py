#!/usr/bin/env python3
"""
Test script to verify custom YOLO model integration with predict.py
"""
import sys
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import the predictor
from services.behavior_analysis.predict import predictor, USE_CUSTOM_YOLO

def test_yolo_detection():
    """Test YOLO detection with a dummy image"""
    print(f"Using custom YOLO: {USE_CUSTOM_YOLO}")
    
    # Create a dummy image (640x480 RGB)
    dummy_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    # Add some variation to make it more realistic
    cv2.rectangle(dummy_image, (100, 100), (300, 300), (255, 255, 255), -1)
    cv2.circle(dummy_image, (400, 200), 50, (200, 200, 200), -1)
    
    try:
        # Test detection
        print("\n🔍 Testing pet detection...")
        detections = predictor.detect_pets(dummy_image, pet_type="catdog")
        
        print(f"✅ Detection successful! Found {len(detections)} objects")
        for i, det in enumerate(detections):
            print(f"\nDetection {i+1}:")
            print(f"  - Class: {det['class']}")
            print(f"  - Confidence: {det['confidence']:.2f}")
            print(f"  - BBox: [{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, {det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]")
            
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()

def test_feature_extraction():
    """Test feature extraction from bounding box"""
    print("\n🔍 Testing feature extraction...")
    
    # Create dummy frame
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # Test bbox
    bbox = [100, 100, 300, 300]
    prev_bbox = [90, 90, 290, 290]  # Slightly moved
    
    try:
        features = predictor.extract_features_from_bbox(frame, bbox, prev_bbox)
        print(f"✅ Feature extraction successful!")
        print(f"  - Feature shape: {features.shape}")
        print(f"  - Non-zero features: {np.count_nonzero(features)}")
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()

def test_behavior_classification():
    """Test behavior classification with dummy features"""
    print("\n🔍 Testing behavior classification...")
    
    # Create dummy feature sequence (30 frames x 2048 features)
    feature_sequence = np.random.randn(30, 2048) * 0.1
    
    try:
        result = predictor.classify_behavior(feature_sequence, pet_type="dog")
        print(f"✅ Classification successful!")
        print(f"  - Behavior: {result['behavior']}")
        print(f"  - Confidence: {result['confidence']:.2f}")
        print(f"  - Is abnormal: {result['is_abnormal']}")
        
        # Show top 3 behaviors
        probs = result['all_probabilities']
        top_behaviors = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        print("\n  Top 3 predicted behaviors:")
        for behavior, prob in top_behaviors:
            print(f"    - {behavior}: {prob:.3f}")
            
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("Custom YOLO Integration Test")
    print("=" * 60)
    
    test_yolo_detection()
    test_feature_extraction()
    test_behavior_classification()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)