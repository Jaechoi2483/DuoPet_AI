"""
Pose-based Feature Extractor for Behavior Analysis

This module extracts behavioral features from pose keypoints
for improved behavior classification.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)

class PoseFeatureExtractor:
    """
    Extracts behavioral features from pose keypoints.
    
    Features include:
    - Joint angles
    - Limb velocities
    - Body posture metrics
    - Movement patterns
    - Symmetry measures
    """
    
    def __init__(self, buffer_size: int = 30):
        """
        Initialize the feature extractor.
        
        Args:
            buffer_size: Number of frames to buffer for temporal features
        """
        self.buffer_size = buffer_size
        self.keypoint_buffer = deque(maxlen=buffer_size)
        
        # Define body part groups for quadrupeds
        self.body_parts = {
            'head': ['nose', 'upper_jaw', 'lower_jaw', 'right_eye', 'left_eye'],
            'ears': ['right_earbase', 'right_earend', 'left_earbase', 'left_earend'],
            'neck': ['neck_base', 'neck_end', 'throat_base', 'throat_end'],
            'spine': ['back_base', 'back_middle', 'back_end', 'spine1', 'spine2', 'spine3'],
            'tail': ['tail_base', 'tail_end'],
            'front_legs': [
                'front_left_thigh', 'front_left_knee', 'front_left_paw',
                'front_right_thigh', 'front_right_knee', 'front_right_paw'
            ],
            'back_legs': [
                'back_left_thigh', 'back_left_knee', 'back_left_paw',
                'back_right_thigh', 'back_right_knee', 'back_right_paw'
            ]
        }
        
        # Define key angles for behavior analysis
        self.key_angles = [
            # Neck angle
            ('throat_base', 'neck_base', 'back_base'),
            # Back curvature
            ('neck_base', 'back_middle', 'tail_base'),
            # Front leg angles
            ('neck_base', 'front_left_thigh', 'front_left_knee'),
            ('front_left_thigh', 'front_left_knee', 'front_left_paw'),
            ('neck_base', 'front_right_thigh', 'front_right_knee'),
            ('front_right_thigh', 'front_right_knee', 'front_right_paw'),
            # Back leg angles
            ('back_end', 'back_left_thigh', 'back_left_knee'),
            ('back_left_thigh', 'back_left_knee', 'back_left_paw'),
            ('back_end', 'back_right_thigh', 'back_right_knee'),
            ('back_right_thigh', 'back_right_knee', 'back_right_paw'),
            # Head angle
            ('upper_jaw', 'nose', 'neck_base')
        ]
        
    def extract_features(self, keypoint_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from a single frame of keypoints.
        
        Args:
            keypoint_data: Dictionary from SuperAnimalQuadrupedAdapter.predict()
            
        Returns:
            Feature vector (1D numpy array)
        """
        features = []
        
        keypoints = keypoint_data['keypoints']
        keypoint_names = keypoint_data['keypoint_names']
        confidence_scores = keypoint_data['confidence_scores']
        valid_keypoints = keypoint_data['valid_keypoints']
        
        # Create keypoint name to index mapping
        name_to_idx = {name: i for i, name in enumerate(keypoint_names)}
        
        # 1. Extract static pose features
        pose_features = self._extract_pose_features(
            keypoints, name_to_idx, valid_keypoints, confidence_scores
        )
        features.extend(pose_features)
        
        # 2. Add to buffer for temporal features
        self.keypoint_buffer.append({
            'keypoints': keypoints,
            'valid': valid_keypoints,
            'confidence': confidence_scores
        })
        
        # 3. Extract temporal features if enough frames
        if len(self.keypoint_buffer) >= 5:
            temporal_features = self._extract_temporal_features(name_to_idx)
            features.extend(temporal_features)
        else:
            # Pad with zeros if not enough frames
            features.extend([0.0] * 50)  # Approximate temporal feature size
        
        # 4. Extract symmetry features
        symmetry_features = self._extract_symmetry_features(
            keypoints, name_to_idx, valid_keypoints
        )
        features.extend(symmetry_features)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_pose_features(self, keypoints: List[Tuple[float, float]], 
                               name_to_idx: Dict[str, int],
                               valid_keypoints: List[int],
                               confidence_scores: List[float]) -> List[float]:
        """Extract static pose-based features."""
        features = []
        
        # 1. Joint angles
        for angle_def in self.key_angles:
            angle = self._calculate_angle(keypoints, name_to_idx, angle_def)
            features.append(angle if angle is not None else 0.0)
        
        # 2. Body dimensions (normalized)
        # Height estimation (nose to average of paws)
        if all(name in name_to_idx for name in ['nose', 'front_left_paw', 'back_left_paw']):
            nose_idx = name_to_idx['nose']
            if nose_idx in valid_keypoints:
                nose_y = keypoints[nose_idx][1]
                paw_heights = []
                for paw in ['front_left_paw', 'front_right_paw', 'back_left_paw', 'back_right_paw']:
                    if paw in name_to_idx and name_to_idx[paw] in valid_keypoints:
                        paw_heights.append(keypoints[name_to_idx[paw]][1])
                if paw_heights:
                    avg_paw_y = np.mean(paw_heights)
                    height = abs(nose_y - avg_paw_y)
                    features.append(height / 100.0)  # Normalize
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # 3. Body length (nose to tail base)
        if all(name in name_to_idx for name in ['nose', 'tail_base']):
            nose_idx = name_to_idx['nose']
            tail_idx = name_to_idx['tail_base']
            if nose_idx in valid_keypoints and tail_idx in valid_keypoints:
                length = np.linalg.norm(
                    np.array(keypoints[nose_idx]) - np.array(keypoints[tail_idx])
                )
                features.append(length / 100.0)  # Normalize
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # 4. Average confidence of valid keypoints
        if valid_keypoints:
            avg_confidence = np.mean([confidence_scores[i] for i in valid_keypoints])
            features.append(avg_confidence)
        else:
            features.append(0.0)
        
        # 5. Proportion of valid keypoints
        features.append(len(valid_keypoints) / len(keypoints))
        
        return features
    
    def _extract_temporal_features(self, name_to_idx: Dict[str, int]) -> List[float]:
        """Extract movement and velocity features from buffer."""
        features = []
        
        # Get recent frames
        recent_frames = list(self.keypoint_buffer)[-10:]  # Last 10 frames
        
        # 1. Movement velocities for key points
        key_points = ['nose', 'neck_base', 'back_middle', 'tail_base']
        
        for point_name in key_points:
            if point_name in name_to_idx:
                point_idx = name_to_idx[point_name]
                velocities = []
                
                for i in range(1, len(recent_frames)):
                    prev_frame = recent_frames[i-1]
                    curr_frame = recent_frames[i]
                    
                    if (point_idx in prev_frame['valid'] and 
                        point_idx in curr_frame['valid']):
                        prev_pos = np.array(prev_frame['keypoints'][point_idx])
                        curr_pos = np.array(curr_frame['keypoints'][point_idx])
                        velocity = np.linalg.norm(curr_pos - prev_pos)
                        velocities.append(velocity)
                
                if velocities:
                    features.extend([
                        np.mean(velocities),  # Average velocity
                        np.std(velocities),   # Velocity variation
                        np.max(velocities)    # Peak velocity
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0])
            else:
                features.extend([0.0, 0.0, 0.0])
        
        # 2. Overall movement (center of mass approximation)
        com_velocities = []
        for i in range(1, len(recent_frames)):
            prev_frame = recent_frames[i-1]
            curr_frame = recent_frames[i]
            
            # Calculate center of mass for valid keypoints
            prev_com = self._calculate_center_of_mass(
                prev_frame['keypoints'], prev_frame['valid']
            )
            curr_com = self._calculate_center_of_mass(
                curr_frame['keypoints'], curr_frame['valid']
            )
            
            if prev_com is not None and curr_com is not None:
                velocity = np.linalg.norm(curr_com - prev_com)
                com_velocities.append(velocity)
        
        if com_velocities:
            features.extend([
                np.mean(com_velocities),
                np.std(com_velocities),
                np.max(com_velocities)
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 3. Tail movement (if available)
        if 'tail_end' in name_to_idx:
            tail_idx = name_to_idx['tail_end']
            tail_movements = []
            
            for i in range(1, len(recent_frames)):
                if (tail_idx in recent_frames[i-1]['valid'] and 
                    tail_idx in recent_frames[i]['valid']):
                    prev_pos = recent_frames[i-1]['keypoints'][tail_idx]
                    curr_pos = recent_frames[i]['keypoints'][tail_idx]
                    movement = np.linalg.norm(
                        np.array(curr_pos) - np.array(prev_pos)
                    )
                    tail_movements.append(movement)
            
            if tail_movements:
                features.append(np.mean(tail_movements))
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        return features
    
    def _extract_symmetry_features(self, keypoints: List[Tuple[float, float]], 
                                  name_to_idx: Dict[str, int],
                                  valid_keypoints: List[int]) -> List[float]:
        """Extract symmetry features between left and right body parts."""
        features = []
        
        # Define symmetric pairs
        symmetric_pairs = [
            ('right_eye', 'left_eye'),
            ('right_earbase', 'left_earbase'),
            ('front_right_thigh', 'front_left_thigh'),
            ('front_right_knee', 'front_left_knee'),
            ('front_right_paw', 'front_left_paw'),
            ('back_right_thigh', 'back_left_thigh'),
            ('back_right_knee', 'back_left_knee'),
            ('back_right_paw', 'back_left_paw')
        ]
        
        symmetry_scores = []
        
        for right_part, left_part in symmetric_pairs:
            if (right_part in name_to_idx and left_part in name_to_idx):
                right_idx = name_to_idx[right_part]
                left_idx = name_to_idx[left_part]
                
                if right_idx in valid_keypoints and left_idx in valid_keypoints:
                    # Calculate relative positions from body center
                    if 'back_middle' in name_to_idx and name_to_idx['back_middle'] in valid_keypoints:
                        center = np.array(keypoints[name_to_idx['back_middle']])
                        right_pos = np.array(keypoints[right_idx]) - center
                        left_pos = np.array(keypoints[left_idx]) - center
                        
                        # Mirror left position
                        left_pos_mirrored = np.array([-left_pos[0], left_pos[1]])
                        
                        # Calculate symmetry score
                        symmetry = 1.0 - (np.linalg.norm(right_pos - left_pos_mirrored) / 100.0)
                        symmetry_scores.append(max(0.0, symmetry))
        
        # Average symmetry score
        if symmetry_scores:
            features.append(np.mean(symmetry_scores))
        else:
            features.append(0.0)
        
        # Variance in symmetry (indicates asymmetric movement)
        if len(symmetry_scores) > 1:
            features.append(np.std(symmetry_scores))
        else:
            features.append(0.0)
        
        return features
    
    def _calculate_angle(self, keypoints: List[Tuple[float, float]], 
                        name_to_idx: Dict[str, int],
                        angle_def: Tuple[str, str, str]) -> Optional[float]:
        """Calculate angle between three keypoints."""
        p1_name, p2_name, p3_name = angle_def
        
        if not all(name in name_to_idx for name in angle_def):
            return None
        
        p1_idx = name_to_idx[p1_name]
        p2_idx = name_to_idx[p2_name]  # Middle point
        p3_idx = name_to_idx[p3_name]
        
        # Check if all points are valid
        if not all(isinstance(keypoints[idx], (list, tuple)) and len(keypoints[idx]) == 2 
                  for idx in [p1_idx, p2_idx, p3_idx]):
            return None
        
        # Calculate vectors
        v1 = np.array(keypoints[p1_idx]) - np.array(keypoints[p2_idx])
        v2 = np.array(keypoints[p3_idx]) - np.array(keypoints[p2_idx])
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return float(np.degrees(angle))
    
    def _calculate_center_of_mass(self, keypoints: List[Tuple[float, float]], 
                                 valid_indices: List[int]) -> Optional[np.ndarray]:
        """Calculate approximate center of mass from valid keypoints."""
        if not valid_indices:
            return None
        
        valid_points = [keypoints[i] for i in valid_indices]
        return np.mean(valid_points, axis=0)
    
    def get_feature_dimension(self) -> int:
        """Get the expected dimension of the feature vector."""
        # This should match the actual feature extraction
        # Approximate calculation:
        # - Pose features: ~20
        # - Temporal features: ~50
        # - Symmetry features: ~2
        return 72  # Adjust based on actual implementation
    
    def reset_buffer(self):
        """Reset the temporal buffer."""
        self.keypoint_buffer.clear()