#!/usr/bin/env python3
"""
Test script for enhanced behavior analysis API with pose estimation
"""

import requests
import json
import sys
import os
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000/api/v1/behavior"

def test_model_status():
    """Test model status endpoint"""
    print("\n=== Testing Model Status ===")
    response = requests.get(f"{BASE_URL}/model-status")
    
    if response.status_code == 200:
        data = response.json()['data']
        print(f"✓ Loaded models: {len(data['loaded_models'])}")
        print(f"✓ GPU available: {data['gpu_info']['available']}")
        print(f"✓ Pose estimation available: {data['pose_estimation']['available']}")
        if data['pose_estimation']['available']:
            print(f"  - Model type: {data['pose_estimation']['model_type']}")
            print(f"  - Keypoints: {data['pose_estimation']['keypoints']}")
        return True
    else:
        print(f"✗ Failed to get model status: {response.status_code}")
        return False

def test_pose_estimation(image_path):
    """Test pose estimation on single image"""
    print(f"\n=== Testing Pose Estimation ===")
    print(f"Image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"✗ Image file not found: {image_path}")
        return False
    
    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
        response = requests.post(f"{BASE_URL}/test-pose", files=files)
    
    if response.status_code == 200:
        data = response.json()['data']
        print(f"✓ Pose estimation successful")
        print(f"  - Total keypoints: {data['num_keypoints']}")
        print(f"  - Valid keypoints: {data['valid_keypoints']}")
        print(f"  - Sample keypoints:")
        for kp in data['sample_keypoints']:
            print(f"    • {kp['name']}: ({kp['position'][0]:.1f}, {kp['position'][1]:.1f}) [conf: {kp['confidence']:.3f}]")
        return True
    else:
        print(f"✗ Pose estimation failed: {response.status_code}")
        print(f"  Error: {response.json().get('message', 'Unknown error')}")
        return False

def test_video_analysis(video_path, pet_type="dog"):
    """Test video behavior analysis"""
    print(f"\n=== Testing Video Analysis ===")
    print(f"Video: {video_path}")
    print(f"Pet type: {pet_type}")
    
    if not os.path.exists(video_path):
        print(f"✗ Video file not found: {video_path}")
        return False
    
    with open(video_path, 'rb') as f:
        files = {'video': (os.path.basename(video_path), f, 'video/mp4')}
        data = {'pet_type': pet_type, 'real_time': 'true'}
        response = requests.post(f"{BASE_URL}/analyze", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()['data']
        
        # Check if it's async processing
        if 'analysis_id' in result and result.get('status') == 'processing':
            print(f"✓ Analysis started (async mode)")
            print(f"  - Analysis ID: {result['analysis_id']}")
            return result['analysis_id']
        else:
            # Direct result
            print(f"✓ Analysis completed")
            print(f"  - Video duration: {result['video_duration']:.2f}s")
            print(f"  - Behaviors detected: {len(result['behaviors'])}")
            print(f"  - Pose estimation used: {result.get('pose_estimation_used', 'Unknown')}")
            print(f"  - Pose usage: {result.get('pose_usage_percentage', 0):.1f}%")
            print(f"  - Behavior summary:")
            for behavior, count in result['behavior_summary'].items():
                print(f"    • {behavior}: {count}")
            if result['abnormal_behaviors']:
                print(f"  - Abnormal behaviors: {len(result['abnormal_behaviors'])}")
            return True
    else:
        print(f"✗ Video analysis failed: {response.status_code}")
        print(f"  Error: {response.json().get('message', 'Unknown error')}")
        return False

def check_analysis_status(analysis_id):
    """Check async analysis status"""
    print(f"\n=== Checking Analysis Status ===")
    print(f"Analysis ID: {analysis_id}")
    
    response = requests.get(f"{BASE_URL}/analysis/{analysis_id}")
    
    if response.status_code == 200:
        data = response.json()['data']
        print(f"✓ Status: {data['status']}")
        
        if data['status'] == 'processing':
            print(f"  - Progress: {data.get('progress', 0)}%")
            print(f"  - Message: {data.get('message', 'Processing...')}")
        elif data['status'] == 'completed':
            print(f"  - Video duration: {data['video_duration']:.2f}s")
            print(f"  - Behaviors detected: {len(data['behaviors'])}")
            print(f"  - Pose estimation used: {data.get('pose_estimation_used', 'Unknown')}")
            print(f"  - Behavior summary:")
            for behavior, count in data['behavior_summary'].items():
                print(f"    • {behavior}: {count}")
        
        return data
    else:
        print(f"✗ Failed to get status: {response.status_code}")
        return None

def main():
    """Main test function"""
    print("=" * 60)
    print("Enhanced Behavior Analysis API Test")
    print("=" * 60)
    
    # Test 1: Model status
    if not test_model_status():
        print("\nModel status test failed. Make sure the server is running.")
        return
    
    # Test 2: Pose estimation on image
    # Try to find a test image
    test_images = [
        "test_data/dog.jpg",
        "test_data/cat.jpg",
        "samples/dog_sample.jpg",
        "samples/cat_sample.jpg"
    ]
    
    image_tested = False
    for img_path in test_images:
        full_path = Path(__file__).parent.parent / img_path
        if full_path.exists():
            test_pose_estimation(str(full_path))
            image_tested = True
            break
    
    if not image_tested:
        print("\n⚠ No test images found for pose estimation test")
    
    # Test 3: Video analysis
    test_videos = [
        ("test_data/dog_video.mp4", "dog"),
        ("test_data/cat_video.mp4", "cat"),
        ("samples/pet_video.mp4", "dog")
    ]
    
    video_tested = False
    for video_path, pet_type in test_videos:
        full_path = Path(__file__).parent.parent / video_path
        if full_path.exists():
            result = test_video_analysis(str(full_path), pet_type)
            video_tested = True
            
            # If async, check status
            if isinstance(result, str) and result.startswith("analysis_"):
                import time
                time.sleep(2)  # Wait a bit
                check_analysis_status(result)
            break
    
    if not video_tested:
        print("\n⚠ No test videos found for behavior analysis test")
    
    print("\n" + "=" * 60)
    print("Test completed!")

if __name__ == "__main__":
    main()