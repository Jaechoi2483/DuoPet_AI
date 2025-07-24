"""
Pose Estimation Fallback Module

This module provides a fallback implementation for pose estimation
when the SuperAnimal model is not available or has issues.
It generates synthetic pose features based on bounding box analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import cv2
import logging

logger = logging.getLogger(__name__)

class PoseEstimationFallback:
    """
    Fallback pose estimation that generates synthetic features
    based on bounding box and motion analysis.
    """
    
    def __init__(self):
        """Initialize the fallback pose estimator."""
        self.keypoint_names = [
            "nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left",
            "right_eye", "right_earbase", "right_earend", "right_antler_base", "right_antler_end",
            "left_eye", "left_earbase", "left_earend", "left_antler_base", "left_antler_end",
            "neck_base", "neck_end", "throat_base", "throat_end",
            "back_base", "back_end", "back_middle",
            "tail_base", "tail_end",
            "front_left_thigh", "front_left_knee", "front_left_paw",
            "front_right_thigh", "front_right_knee", "front_right_paw",
            "back_left_thigh", "back_left_knee", "back_left_paw",
            "back_right_thigh", "back_right_knee", "back_right_paw",
            "spine1", "spine2", "spine3"
        ]
        self.num_keypoints = len(self.keypoint_names)
        self.previous_bbox = None
        self.motion_history = []
        
    def estimate_keypoints_from_bbox(self, image: np.ndarray, bbox: List[float]) -> Dict[str, Any]:
        """
        Estimate keypoints based on bounding box geometry.
        
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Dictionary with estimated keypoints
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Generate synthetic keypoints based on typical quadruped anatomy
        keypoints = []
        confidence_scores = []
        
        # Head region (front 30% of bbox)
        head_x = x1 + width * 0.15
        head_y = y1 + height * 0.3
        
        # Nose
        keypoints.append((x1 + width * 0.05, cy))
        confidence_scores.append(0.8)
        
        # Eyes
        keypoints.append((head_x, head_y - height * 0.1))  # Right eye
        confidence_scores.append(0.7)
        keypoints.append((head_x, head_y + height * 0.1))  # Left eye
        confidence_scores.append(0.7)
        
        # Body center
        keypoints.append((cx, cy))  # Spine middle
        confidence_scores.append(0.9)
        
        # Tail region (back 20% of bbox)
        tail_x = x2 - width * 0.1
        keypoints.append((tail_x, cy))  # Tail base
        confidence_scores.append(0.75)
        
        # Legs (corners of bbox)
        # Front legs
        keypoints.append((x1 + width * 0.2, y2 - height * 0.1))  # Front left paw
        confidence_scores.append(0.6)
        keypoints.append((x1 + width * 0.2, y2 - height * 0.1))  # Front right paw
        confidence_scores.append(0.6)
        
        # Back legs
        keypoints.append((x2 - width * 0.2, y2 - height * 0.1))  # Back left paw
        confidence_scores.append(0.6)
        keypoints.append((x2 - width * 0.2, y2 - height * 0.1))  # Back right paw
        confidence_scores.append(0.6)
        
        # Fill remaining keypoints with interpolated positions
        while len(keypoints) < self.num_keypoints:
            # Add points along the spine
            t = len(keypoints) / self.num_keypoints
            spine_x = x1 + width * (0.2 + 0.6 * t)
            spine_y = cy + height * 0.1 * np.sin(t * np.pi)
            keypoints.append((spine_x, spine_y))
            confidence_scores.append(0.5)
        
        # Ensure we have exactly the right number of keypoints
        keypoints = keypoints[:self.num_keypoints]
        confidence_scores = confidence_scores[:self.num_keypoints]
        
        # Add motion-based adjustments
        if self.previous_bbox is not None:
            motion_vector = self._calculate_motion(bbox, self.previous_bbox)
            # Adjust keypoints based on motion
            for i in range(len(keypoints)):
                if confidence_scores[i] > 0.6:
                    x, y = keypoints[i]
                    # Add slight motion influence
                    x += motion_vector[0] * 0.1
                    y += motion_vector[1] * 0.1
                    keypoints[i] = (x, y)
        
        self.previous_bbox = bbox
        
        # Identify valid keypoints (confidence > 0.5)
        valid_keypoints = [i for i, conf in enumerate(confidence_scores) if conf > 0.5]
        
        return {
            "keypoints": keypoints,
            "keypoint_names": self.keypoint_names,
            "confidence_scores": confidence_scores,
            "valid_keypoints": valid_keypoints,
            "bbox": bbox,
            "method": "fallback"
        }
    
    def _calculate_motion(self, current_bbox: List[float], previous_bbox: List[float]) -> Tuple[float, float]:
        """Calculate motion vector between bboxes."""
        curr_cx = (current_bbox[0] + current_bbox[2]) / 2
        curr_cy = (current_bbox[1] + current_bbox[3]) / 2
        prev_cx = (previous_bbox[0] + previous_bbox[2]) / 2
        prev_cy = (previous_bbox[1] + previous_bbox[3]) / 2
        
        return (curr_cx - prev_cx, curr_cy - prev_cy)
    
    def extract_pose_features(self, keypoint_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from keypoint data.
        
        Args:
            keypoint_data: Keypoint detection result
            
        Returns:
            Feature vector
        """
        keypoints = keypoint_data["keypoints"]
        confidence_scores = keypoint_data["confidence_scores"]
        bbox = keypoint_data["bbox"]
        
        features = []
        
        # 1. Keypoint positions (normalized by bbox)
        x1, y1, x2, y2 = bbox
        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)
        
        for (x, y), conf in zip(keypoints, confidence_scores):
            # Normalize positions
            norm_x = (x - x1) / width
            norm_y = (y - y1) / height
            features.extend([norm_x, norm_y, conf])
        
        # 2. Geometric features
        # Body aspect ratio
        features.append(height / width)
        
        # Head-to-tail vector (if available)
        if len(keypoints) >= 5:
            head_x, head_y = keypoints[0]  # nose
            tail_x, tail_y = keypoints[min(23, len(keypoints)-1)]  # tail base
            body_length = np.sqrt((tail_x - head_x)**2 + (tail_y - head_y)**2)
            features.append(body_length / width)
            
            # Body angle
            body_angle = np.arctan2(tail_y - head_y, tail_x - head_x)
            features.append(np.sin(body_angle))
            features.append(np.cos(body_angle))
        else:
            features.extend([1.0, 0.0, 1.0])  # Default values
        
        # 3. Motion features
        if hasattr(self, 'motion_history') and len(self.motion_history) > 0:
            recent_motion = np.mean(self.motion_history[-5:], axis=0) if len(self.motion_history) >= 5 else [0, 0]
            features.extend(recent_motion)
        else:
            features.extend([0.0, 0.0])
        
        # 4. Confidence statistics
        valid_conf = [c for c in confidence_scores if c > 0.5]
        features.append(len(valid_conf) / len(confidence_scores))  # Ratio of valid keypoints
        features.append(np.mean(confidence_scores))  # Average confidence
        features.append(np.std(confidence_scores))   # Confidence variance
        
        # Convert to numpy array
        feature_vector = np.array(features, dtype=np.float32)
        
        # Pad or truncate to expected size (128 features)
        expected_size = 128
        if len(feature_vector) < expected_size:
            padded = np.zeros(expected_size, dtype=np.float32)
            padded[:len(feature_vector)] = feature_vector
            feature_vector = padded
        else:
            feature_vector = feature_vector[:expected_size]
        
        return feature_vector
    
    def reset(self):
        """Reset the fallback estimator state."""
        self.previous_bbox = None
        self.motion_history = []