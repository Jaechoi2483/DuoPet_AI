#!/usr/bin/env python3
"""
Test enhanced behavior analysis with a real video file
"""
import sys
import time
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_with_real_video(video_path):
    """Test the enhanced behavior analysis with a real video"""
    print("=== Testing Enhanced Behavior Analysis with Real Video ===")
    
    try:
        from services.behavior_analysis.enhanced_predict import enhanced_predictor
        import os
        
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return False
            
        print(f"\n1. Using video: {video_path}")
        print(f"   File size: {os.path.getsize(video_path) / 1024 / 1024:.2f} MB")
        
        # Check configuration
        print("\n2. Checking configuration...")
        print(f"  - Pose estimation enabled: {enhanced_predictor.use_pose_estimation}")
        print(f"  - Using fallback: {enhanced_predictor.use_fallback}")
        
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
        print("\n4. Results:")
        print(f"  - Video duration: {result['video_duration']:.2f}s")
        print(f"  - Total frames: {result['total_frames']}")
        print(f"  - Behavior sequences: {len(result['behavior_sequences'])}")
        print(f"  - Pose estimation used: {result['pose_estimation_used']}")
        print(f"  - Pose usage percentage: {result['pose_usage_percentage']:.1f}%")
        
        if result['behavior_sequences']:
            print(f"\n  Behaviors detected:")
            # Count behaviors
            behavior_counts = {}
            for seq in result['behavior_sequences']:
                behavior = seq['behavior']['behavior']
                behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
            
            for behavior, count in sorted(behavior_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"    - {behavior}: {count} times")
                
            # Show first few sequences
            print(f"\n  First 5 behavior sequences:")
            for i, seq in enumerate(result['behavior_sequences'][:5]):
                print(f"    {i+1}. Frame {seq['frame']} ({seq['time']:.1f}s): "
                      f"{seq['behavior']['behavior']} "
                      f"(confidence: {seq['behavior']['confidence']:.2f})")
        
        # Check for abnormal behaviors
        if result.get('abnormal_behaviors'):
            print(f"\n  ⚠️ Abnormal behaviors detected: {len(result['abnormal_behaviors'])}")
            for ab in result['abnormal_behaviors'][:3]:
                print(f"    - {ab}")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
    else:
        # Default test video paths
        possible_paths = [
            r"D:\test_pet_video.mp4",
            r"D:\final_project\test_pet_video.mp4",
            r"D:\final_project\DuoPet_AI\test_videos\cat_video1.mp4",
            r"C:\Users\%USERNAME%\Videos\pet_video.mp4"
        ]
        
        video_file = None
        for path in possible_paths:
            expanded_path = os.path.expandvars(path)
            if os.path.exists(expanded_path):
                video_file = expanded_path
                break
        
        if not video_file:
            print("Usage: python test_with_real_video.py <video_path>")
            print("\nNo default test video found. Please provide a video path.")
            sys.exit(1)
    
    import os
    success = test_with_real_video(video_file)
    sys.exit(0 if success else 1)