#!/usr/bin/env python3
"""
Test enhanced behavior analysis with fallback pose estimation
"""
import sys
import time
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_enhanced_behavior():
    """Test the enhanced behavior analysis system"""
    print("=== Testing Enhanced Behavior Analysis ===")
    
    try:
        from services.behavior_analysis.enhanced_predict import enhanced_predictor
        import numpy as np
        import cv2
        
        # Check configuration
        print("\n1. Checking configuration...")
        print(f"  - Pose estimation enabled: {enhanced_predictor.use_pose_estimation}")
        print(f"  - Using fallback: {enhanced_predictor.use_fallback}")
        print(f"  - Pose adapter loaded: {enhanced_predictor.pose_adapter is not None}")
        
        # Create test video (10 frames)
        print("\n2. Creating test video...")
        import tempfile
        import os
        temp_dir = tempfile.gettempdir()
        video_path = os.path.join(temp_dir, "test_pet_video.mp4")
        fps = 30
        width, height = 640, 480
        frames = 10
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # Generate frames with moving rectangle (simulating pet)
        for i in range(frames):
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # Draw a moving rectangle (simulating pet)
            x = 100 + i * 20
            y = 200
            cv2.rectangle(frame, (x, y), (x + 150, y + 100), (128, 128, 128), -1)
            
            # Add some texture
            cv2.circle(frame, (x + 30, y + 30), 10, (0, 0, 0), -1)  # Eye
            cv2.circle(frame, (x + 120, y + 30), 10, (0, 0, 0), -1)  # Eye
            
            out.write(frame)
        
        out.release()
        print(f"✓ Test video created: {video_path}")
        
        # Test video analysis
        print("\n3. Analyzing video with enhanced features...")
        start_time = time.time()
        
        def progress_callback(progress):
            print(f"  Progress: {progress:.1f}%", end='\r')
        
        result = enhanced_predictor.analyze_video(
            video_path,
            pet_type="dog",
            progress_callback=progress_callback,
            visualize=False
        )
        
        analysis_time = time.time() - start_time
        print(f"\n✓ Analysis completed in {analysis_time:.2f}s")
        
        # Check results
        print("\n4. Checking results...")
        print(f"  - Video duration: {result['video_duration']:.2f}s")
        print(f"  - Total frames: {result['total_frames']}")
        print(f"  - Behavior sequences: {len(result['behavior_sequences'])}")
        print(f"  - Pose estimation used: {result['pose_estimation_used']}")
        print(f"  - Pose usage percentage: {result['pose_usage_percentage']:.1f}%")
        
        if result['behavior_sequences']:
            print(f"\n  Sample behaviors detected:")
            for seq in result['behavior_sequences'][:3]:
                print(f"    - Frame {seq['frame']}: {seq['behavior']['behavior']} "
                      f"(confidence: {seq['behavior']['confidence']:.2f})")
        
        # Test single frame with pose
        print("\n5. Testing single frame pose extraction...")
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.rectangle(test_frame, (100, 150), (400, 350), (128, 128, 128), -1)
        
        # Detect pets
        detections = enhanced_predictor.detect_pets(test_frame, "catdog")
        print(f"  - Pets detected: {len(detections)}")
        
        if detections:
            bbox = detections[0]['bbox']
            features = enhanced_predictor.extract_features_with_pose(
                test_frame, bbox, detections[0]['class']
            )
            print(f"  - Feature vector shape: {features.shape}")
            print(f"  - Feature stats: min={features.min():.2f}, max={features.max():.2f}")
        
        print("\n✅ All tests completed successfully!")
        
        # Cleanup
        import os
        if os.path.exists(video_path):
            os.remove(video_path)
            
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_enhanced_behavior()
    sys.exit(0 if success else 1)