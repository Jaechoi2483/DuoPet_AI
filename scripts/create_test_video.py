#!/usr/bin/env python3
"""
Create a test video for behavior analysis testing
"""
import cv2
import numpy as np
import os

def create_test_pet_video(output_path="test_pet_video.mp4", duration=5):
    """
    Create a synthetic test video with a moving rectangle simulating a pet
    
    Args:
        output_path: Output video file path
        duration: Video duration in seconds
    """
    print(f"Creating test video: {output_path}")
    
    # Video properties
    fps = 30
    width, height = 640, 480
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Error: Could not open video writer")
        return False
    
    # Generate frames
    for frame_idx in range(total_frames):
        # Create white background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Calculate pet position (moving in a circle)
        t = frame_idx / total_frames
        center_x = width // 2
        center_y = height // 2
        radius = 100
        
        # Pet position
        pet_x = int(center_x + radius * np.cos(2 * np.pi * t))
        pet_y = int(center_y + radius * np.sin(2 * np.pi * t))
        
        # Draw pet body (gray rectangle)
        pet_width = 120
        pet_height = 80
        x1 = max(0, pet_x - pet_width // 2)
        y1 = max(0, pet_y - pet_height // 2)
        x2 = min(width, pet_x + pet_width // 2)
        y2 = min(height, pet_y + pet_height // 2)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), -1)
        
        # Draw pet features
        # Head
        head_x = x1 + 20
        head_y = y1 + pet_height // 2
        cv2.circle(frame, (head_x, head_y), 25, (100, 100, 100), -1)
        
        # Eyes
        cv2.circle(frame, (head_x - 10, head_y - 10), 5, (0, 0, 0), -1)
        cv2.circle(frame, (head_x + 10, head_y - 10), 5, (0, 0, 0), -1)
        
        # Tail
        tail_x = x2 - 10
        tail_y = y1 + pet_height // 2
        cv2.ellipse(frame, (tail_x, tail_y), (30, 10), 
                    -30 + 20 * np.sin(4 * np.pi * t), 0, 180, (100, 100, 100), -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame {frame_idx}/{total_frames}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Write frame
        out.write(frame)
        
        if frame_idx % 30 == 0:
            print(f"  Progress: {frame_idx}/{total_frames} frames", end='\r')
    
    # Release video writer
    out.release()
    print(f"\n✓ Video created successfully: {output_path}")
    print(f"  Duration: {duration}s")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    
    return True

if __name__ == "__main__":
    import sys
    
    # Get output path from arguments or use default
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        # Use Windows temp directory
        import tempfile
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, "test_pet_video.mp4")
    
    # Create the video
    success = create_test_pet_video(output_path)
    
    if success:
        print(f"\nYou can now test with:")
        print(f"python scripts\\test_with_real_video.py \"{output_path}\"")
    
    sys.exit(0 if success else 1)