"""
Enhanced Behavior Analysis Prediction Service with Pose Estimation

This module provides prediction functions for behavior analysis models
using YOLOv5 for object detection, SuperAnimal-Quadruped for pose estimation,
and LSTM for behavior classification.
"""

import os
import sys
import numpy as np
import torch
import cv2
from typing import Dict, List, Union, Optional, Tuple
from PIL import Image
import yaml
from pathlib import Path
from collections import defaultdict, deque
import threading
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import existing modules
from common.logger import get_logger
from common.config import get_settings, get_model_path
from common.exceptions import ModelNotLoadedError, ModelInferenceError, ValidationError
from .model_manager import model_manager
from .error_handler import error_handler
from .predict import BehaviorAnalysisPredictor  # Import base class

# Import new modules
from .superanimal_adapter import SuperAnimalQuadrupedAdapter
from .pose_feature_extractor import PoseFeatureExtractor
from .pose_estimation_fallback import PoseEstimationFallback

logger = get_logger(__name__)
settings = get_settings()


class EnhancedBehaviorAnalysisPredictor(BehaviorAnalysisPredictor):
    """
    Enhanced behavior analysis predictor with pose estimation.
    Extends the base predictor to include SuperAnimal-Quadruped pose estimation.
    """
    
    def __init__(self):
        """Initialize the enhanced predictor with pose estimation."""
        super().__init__()
        
        # Initialize pose estimation components
        self.pose_adapter = None
        self.pose_feature_extractor = PoseFeatureExtractor()
        self.pose_fallback = PoseEstimationFallback()
        self.use_pose_estimation = True  # Re-enable with fallback
        self.use_fallback = True  # Use fallback by default
        
        # Try to load pose model but use fallback if it fails
        if self.use_pose_estimation and not self.use_fallback:
            try:
                self._load_pose_model()
                self.use_fallback = False
            except Exception as e:
                logger.warning(f"Failed to load pose model: {e}. Using fallback pose estimation.")
                self.use_fallback = True
        else:
            logger.info("Using fallback pose estimation for enhanced behavior analysis")
    
    def _load_pose_model(self):
        """Load the SuperAnimal-Quadruped pose estimation model."""
        try:
            model_path = project_root / "models" / "behavior_analysis" / "pose_estimation" / "superanimal_quadruped"
            if not model_path.exists():
                raise FileNotFoundError(f"Pose model directory not found: {model_path}")
            
            logger.info("Loading SuperAnimal-Quadruped pose estimation model...")
            self.pose_adapter = SuperAnimalQuadrupedAdapter(str(model_path))
            logger.info("Pose estimation model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load pose model: {str(e)}")
            raise
    
    def extract_features_with_pose(self, frame: np.ndarray, bbox: List[float], 
                                 pet_class: str, prev_bbox: Optional[List[float]] = None) -> np.ndarray:
        """
        Extract features using pose estimation.
        
        Args:
            frame: Current frame
            bbox: Current bounding box [x1, y1, x2, y2]
            pet_class: Pet class (cat/dog)
            prev_bbox: Previous frame's bounding box
            
        Returns:
            Feature vector combining pose and motion features
        """
        if not self.use_pose_estimation or self.pose_adapter is None:
            # Fallback to original bbox features
            return self.extract_features_from_bbox(frame, bbox, prev_bbox)
        
        try:
            # Extract pose keypoints
            logger.debug(f"Attempting pose extraction for bbox: {bbox}")
            
            if self.use_fallback:
                # Use fallback pose estimation
                pose_result = self.pose_fallback.estimate_keypoints_from_bbox(frame, bbox)
            else:
                # Use actual pose model
                pose_result = self.pose_adapter.predict(frame, bbox)
            
            # Check if pose extraction was successful
            if not pose_result.get("valid_keypoints"):
                logger.warning("No valid keypoints detected, falling back to bbox features")
                return self.extract_features_from_bbox(frame, bbox, prev_bbox)
            
            # Extract pose-based features
            if self.use_fallback:
                pose_features = self.pose_fallback.extract_pose_features(pose_result)
            else:
                pose_features = self.pose_feature_extractor.extract_features(pose_result)
            logger.debug(f"Extracted {len(pose_features)} pose features")
            
            # Get original bbox features for motion
            bbox_features = self.extract_features_from_bbox(frame, bbox, prev_bbox)
            
            # Combine features
            # For fallback mode, we need to combine pose and bbox features differently
            if self.use_fallback:
                # Combine pose features (128) with bbox features to reach 2048
                combined_features = np.zeros(2048, dtype=np.float32)
                combined_features[:len(pose_features)] = pose_features
                combined_features[len(pose_features):len(pose_features)+len(bbox_features)] = bbox_features
                return combined_features
            else:
                # Original logic for real pose model
                if len(pose_features) < 2048:
                    padded_pose_features = np.zeros(2048)
                    padded_pose_features[:len(pose_features)] = pose_features
                    padded_pose_features[len(pose_features):len(pose_features)+9] = bbox_features[:9]
                    return padded_pose_features
                else:
                    return pose_features[:2048]
                
        except Exception as e:
            logger.warning(f"Pose extraction failed, using bbox features: {str(e)}")
            import traceback
            logger.debug(f"Pose extraction error traceback: {traceback.format_exc()}")
            return self.extract_features_from_bbox(frame, bbox, prev_bbox)
    
    def analyze_video(self, video_path: str, pet_type: str = "dog", 
                     progress_callback=None, visualize: bool = False) -> Dict:
        """
        Analyze video with enhanced pose estimation.
        
        Args:
            video_path: Video file path
            pet_type: Pet type to analyze
            progress_callback: Progress update callback
            visualize: Whether to save visualization frames
            
        Returns:
            Analysis results dictionary
        """
        cap = None
        visualization_frames = []
        
        try:
            # Video file validation
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                raise ValidationError("video", f"Video file not found: {video_path}")
            
            # File size check (max 500MB)
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            if file_size > 500:
                logger.error(f"Video file too large: {file_size:.2f}MB")
                raise ValidationError("video", f"Video file too large: {file_size:.2f}MB (max 500MB)")
            
            # Open video
            logger.info(f"Opening video file: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video file: {video_path}")
                raise ValidationError("video", "Cannot open video file. Please check the file format.")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # FPS validation
            if fps <= 0:
                logger.warning("Invalid FPS detected, using default 30")
                fps = 30
            
            # Video duration check (max 10 minutes)
            video_duration = total_frames / fps if fps > 0 else 0
            if video_duration > 600:  # 10 minutes
                logger.error(f"Video too long: {video_duration:.2f} seconds")
                raise ValidationError("video", f"Video too long: {video_duration:.2f}s (max 10 minutes)")
            
            logger.info(f"Analyzing video with pose estimation: {video_path}, Duration: {video_duration:.2f}s")
            
            # Result storage
            behavior_sequences = []
            frame_count = 0
            prev_detections = {}
            
            # Initialize progress
            if progress_callback:
                progress_callback(0)
            
            # Reset pose feature extractor buffer
            self.pose_feature_extractor.reset_buffer()
            
            # Reset fallback estimator
            if self.use_fallback:
                self.pose_fallback.reset()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 5th frame
                if frame_count % 5 == 0:
                    # Object detection
                    detections = self.detect_pets(frame, "catdog")
                    
                    # Process each detected object
                    for idx, detection in enumerate(detections):
                        obj_id = f"{detection['class']}_{idx}"
                        
                        try:
                            # Extract features with pose estimation
                            prev_bbox = prev_detections.get(obj_id, {}).get('bbox')
                            logger.debug(f"Frame {frame_count}: Extracting features for {obj_id}")
                            features = self.extract_features_with_pose(
                                frame, detection['bbox'], detection['class'], prev_bbox
                            )
                            
                            # Add to feature buffer
                            self.feature_buffer[obj_id].append(features)
                            logger.debug(f"Buffer size for {obj_id}: {len(self.feature_buffer[obj_id])}")
                            
                            # Classify behavior when buffer is full
                            if len(self.feature_buffer[obj_id]) == 30:
                                logger.info(f"Classifying behavior for {obj_id} at frame {frame_count}")
                                sequence = np.array(list(self.feature_buffer[obj_id]))
                                behavior = self.classify_behavior(
                                    sequence, detection['class']
                                )
                                
                                behavior_sequences.append({
                                    "frame": frame_count,
                                    "time": frame_count / fps if fps > 0 else 0,
                                    "object_id": obj_id,
                                    "behavior": behavior,
                                    "bbox": detection['bbox'],
                                    "has_pose": self.use_pose_estimation
                                })
                                logger.info(f"Behavior classified: {behavior['behavior']} (confidence: {behavior['confidence']:.2f})")
                                
                                # Visualize if requested
                                if visualize and self.use_pose_estimation:
                                    try:
                                        vis_frame = self._visualize_frame(
                                            frame, detection, behavior
                                        )
                                        visualization_frames.append({
                                            "frame": frame_count,
                                            "image": vis_frame
                                        })
                                    except Exception as vis_e:
                                        logger.warning(f"Visualization failed: {vis_e}")
                        except Exception as e:
                            logger.warning(f"Error processing detection at frame {frame_count}: {e}")
                            continue
                    
                    prev_detections = {f"{d['class']}_{i}": d 
                                     for i, d in enumerate(detections)}
                
                frame_count += 1
                
                # Update progress
                if fps > 0 and frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}%")
                    if progress_callback:
                        progress_callback(progress)
            
            cap.release()
            
            # Summarize results
            behavior_summary = self._summarize_behaviors(behavior_sequences)
            abnormal_behaviors = self._find_abnormal_behaviors(behavior_sequences)
            
            # Calculate pose estimation usage
            pose_usage = sum(1 for seq in behavior_sequences if seq.get('has_pose', False))
            pose_percentage = (pose_usage / len(behavior_sequences) * 100) if behavior_sequences else 0
            
            # Calculate pose metrics and temporal analysis
            pose_metrics = self._calculate_pose_metrics(behavior_sequences)
            temporal_analysis = self._calculate_temporal_analysis(behavior_sequences, video_duration)
            
            result = {
                "video_duration": video_duration,
                "total_frames": total_frames,
                "behavior_sequences": behavior_sequences,
                "behavior_summary": behavior_summary,
                "abnormal_behaviors": abnormal_behaviors,
                "pose_estimation_used": self.use_pose_estimation,
                "pose_usage_percentage": pose_percentage,
                "pose_metrics": pose_metrics,
                "temporal_analysis": temporal_analysis
            }
            
            # Add visualization data if requested
            if visualize and visualization_frames:
                result["visualization_frames"] = visualization_frames[:10]  # First 10 frames
            
            return result
            
        except Exception as e:
            raise ModelInferenceError("video_analysis", f"Video analysis failed: {str(e)}")
        finally:
            if cap is not None:
                cap.release()
    
    def _calculate_pose_metrics(self, behavior_sequences: List[Dict]) -> Dict[str, float]:
        """
        Calculate pose-based quality metrics from behavior sequences.
        
        Args:
            behavior_sequences: List of behavior detections with pose data
            
        Returns:
            Dictionary of pose metrics
        """
        # Initialize metrics
        balance_scores = []
        stability_scores = []
        smoothness_scores = []
        activity_scores = []
        
        # Collect pose data from sequences that have pose information
        pose_frames = [seq for seq in behavior_sequences if seq.get('has_pose', False)]
        
        if not pose_frames:
            # Return default values if no pose data available
            return {
                "balance_index": 0.5,
                "stability_score": 0.5,
                "movement_smoothness": 0.5,
                "activity_level": 0.5,
                "center_of_mass_stability": 0.5
            }
        
        # Enhanced metric calculations with dynamic scoring
        # Behavior-confidence matrix for more nuanced scoring
        behavior_weights = {
            'standing': {'balance': 0.85, 'stability': 0.90, 'smoothness': 0.70, 'activity': 0.20},
            'sitting': {'balance': 0.80, 'stability': 0.85, 'smoothness': 0.75, 'activity': 0.15},
            'lying_down': {'balance': 0.75, 'stability': 0.95, 'smoothness': 0.80, 'activity': 0.10},
            'walking': {'balance': 0.70, 'stability': 0.65, 'smoothness': 0.60, 'activity': 0.60},
            'running': {'balance': 0.75, 'stability': 0.70, 'smoothness': 0.65, 'activity': 0.90},  # 상향 조정
            'playing': {'balance': 0.60, 'stability': 0.50, 'smoothness': 0.45, 'activity': 0.85},
            'eating': {'balance': 0.75, 'stability': 0.80, 'smoothness': 0.85, 'activity': 0.30},
            'sleeping': {'balance': 0.70, 'stability': 0.98, 'smoothness': 0.95, 'activity': 0.05},
            'grooming': {'balance': 0.72, 'stability': 0.75, 'smoothness': 0.70, 'activity': 0.25},
            'jumping': {'balance': 0.65, 'stability': 0.60, 'smoothness': 0.55, 'activity': 0.95},  # 상향 조정
            'tail_wagging': {'balance': 0.75, 'stability': 0.70, 'smoothness': 0.65, 'activity': 0.40},
            'digging': {'balance': 0.60, 'stability': 0.55, 'smoothness': 0.50, 'activity': 0.80},
            'watching': {'balance': 0.82, 'stability': 0.88, 'smoothness': 0.85, 'activity': 0.15},
        }
        
        # Default weights for unknown behaviors
        default_weights = {'balance': 0.65, 'stability': 0.65, 'smoothness': 0.65, 'activity': 0.50}
        
        # Calculate metrics with behavior-specific weights
        for i, seq in enumerate(pose_frames):
            behavior = seq['behavior']['behavior']
            confidence = seq['behavior']['confidence']
            
            # Get behavior-specific weights
            weights = behavior_weights.get(behavior, default_weights)
            
            # Add temporal factor - behaviors at the beginning might be less stable
            temporal_factor = min(1.0, (i + 1) / 10)  # Ramps up over first 10 detections
            
            # Balance score: base weight + confidence adjustment + temporal factor
            balance_base = weights['balance']
            balance_score = balance_base * (0.7 + confidence * 0.3) * (0.8 + temporal_factor * 0.2)
            balance_scores.append(min(1.0, balance_score))
            
            # Stability score: considers both confidence and behavior consistency
            stability_base = weights['stability']
            consistency_bonus = 0.1 if i > 0 and pose_frames[i-1]['behavior']['behavior'] == behavior else 0
            stability_score = stability_base * confidence * (0.9 + consistency_bonus)
            stability_scores.append(min(1.0, stability_score))
            
            # Smoothness score: higher for consistent behaviors, lower for rapid changes
            smoothness_base = weights['smoothness']
            if i > 0:
                behavior_change_penalty = 0.1 if pose_frames[i-1]['behavior']['behavior'] != behavior else 0
                smoothness_score = smoothness_base * confidence * (1.0 - behavior_change_penalty)
            else:
                smoothness_score = smoothness_base * confidence
            smoothness_scores.append(min(1.0, smoothness_score))
            
            # Activity score: based on behavior type with confidence weighting
            activity_base = weights['activity']
            activity_score = activity_base * (0.8 + confidence * 0.2)
            activity_scores.append(min(1.0, activity_score))
        
        # Calculate final metrics with weighted averages
        # Recent behaviors have more weight
        if len(balance_scores) > 5:
            # Create weights that favor recent observations
            recency_weights = np.linspace(0.5, 1.0, len(balance_scores))
            recency_weights = recency_weights / recency_weights.sum()
            
            balance_index = np.average(balance_scores, weights=recency_weights)
            stability_score = np.average(stability_scores, weights=recency_weights)
            movement_smoothness = np.average(smoothness_scores, weights=recency_weights)
            activity_level = np.average(activity_scores, weights=recency_weights)
        else:
            # Simple average for short sequences
            balance_index = np.mean(balance_scores) if balance_scores else 0.5
            stability_score = np.mean(stability_scores) if stability_scores else 0.5
            movement_smoothness = np.mean(smoothness_scores) if smoothness_scores else 0.5
            activity_level = np.mean(activity_scores) if activity_scores else 0.5
        
        # Center of mass stability is derived from balance and stability
        center_of_mass_stability = (balance_index * 0.6 + stability_score * 0.4) * 0.95
        
        return {
            "balance_index": float(balance_index),
            "stability_score": float(stability_score),
            "movement_smoothness": float(movement_smoothness),
            "activity_level": float(activity_level),
            "center_of_mass_stability": float(center_of_mass_stability)
        }
    
    def _calculate_temporal_analysis(self, behavior_sequences: List[Dict], video_duration: float) -> Dict:
        """
        Calculate temporal behavior analysis.
        
        Args:
            behavior_sequences: List of behavior detections
            video_duration: Total video duration in seconds
            
        Returns:
            Dictionary of temporal analysis
        """
        behavior_durations = defaultdict(float)
        behavior_transitions = defaultdict(int)
        activity_periods = []
        behavior_frequency = defaultdict(int)
        
        if not behavior_sequences:
            return {
                "behavior_durations": {},
                "behavior_transitions": {},
                "activity_periods": [],
                "behavior_frequency": {},
                "activity_summary": {
                    "total_active_time": 0,
                    "total_rest_time": 0,
                    "activity_ratio": 0.5,
                    "transition_rate": 0,
                    "behavior_diversity": 0
                }
            }
        
        # Sort sequences by time
        sorted_sequences = sorted(behavior_sequences, key=lambda x: x['time'])
        
        # Define behavior categories
        active_behaviors = {'walking', 'running', 'playing', 'jumping', 'digging'}
        rest_behaviors = {'sleeping', 'lying_down', 'sitting'}
        moderate_behaviors = {'eating', 'grooming', 'standing', 'tail_wagging'}
        
        # Calculate behavior durations and transitions
        prev_behavior = None
        prev_time = 0
        current_activity_period = None
        
        for i, seq in enumerate(sorted_sequences):
            behavior = seq['behavior']['behavior']
            confidence = seq['behavior']['confidence']
            time = seq['time']
            
            behavior_frequency[behavior] += 1
            
            # Estimate duration (assume behavior continues until next detection)
            duration = 0
            if prev_behavior:
                duration = time - prev_time
                # Weight duration by confidence
                weighted_duration = duration * (0.5 + confidence * 0.5)
                behavior_durations[prev_behavior] += weighted_duration
                
                # Count transitions
                if prev_behavior != behavior:
                    transition_key = f"{prev_behavior}->{behavior}"
                    behavior_transitions[transition_key] += 1
            
            # Estimate future duration for current behavior
            if i < len(sorted_sequences) - 1:
                next_time = sorted_sequences[i + 1]['time']
                estimated_duration = next_time - time
            else:
                estimated_duration = video_duration - time if video_duration > time else 1.0
            
            # Track activity periods with merging
            if behavior in active_behaviors:
                if current_activity_period and current_activity_period["type"] == "active":
                    # Extend current active period
                    current_activity_period["end"] = time + estimated_duration
                else:
                    # Start new active period
                    if current_activity_period:
                        activity_periods.append(current_activity_period)
                    current_activity_period = {
                        "start": time, 
                        "end": time + estimated_duration,
                        "type": "active",
                        "behaviors": [behavior],
                        "intensity": "high" if behavior in ['running', 'jumping'] else "moderate"
                    }
            elif behavior in rest_behaviors:
                if current_activity_period and current_activity_period["type"] == "rest":
                    # Extend current rest period
                    current_activity_period["end"] = time + estimated_duration
                else:
                    # Start new rest period
                    if current_activity_period:
                        activity_periods.append(current_activity_period)
                    current_activity_period = {
                        "start": time,
                        "end": time + estimated_duration,
                        "type": "rest",
                        "behaviors": [behavior],
                        "intensity": "low"
                    }
            else:
                # Moderate activity - close current period if exists
                if current_activity_period:
                    activity_periods.append(current_activity_period)
                    current_activity_period = None
            
            # Add behavior to current period if it exists
            if current_activity_period and behavior not in current_activity_period.get("behaviors", []):
                current_activity_period["behaviors"].append(behavior)
            
            prev_behavior = behavior
            prev_time = time
        
        # Handle last behavior and period
        if prev_behavior and video_duration > prev_time:
            remaining_duration = video_duration - prev_time
            behavior_durations[prev_behavior] += remaining_duration
        
        if current_activity_period:
            activity_periods.append(current_activity_period)
        
        # Calculate activity summary metrics
        total_active_time = sum(period["end"] - period["start"] 
                               for period in activity_periods if period["type"] == "active")
        total_rest_time = sum(period["end"] - period["start"] 
                             for period in activity_periods if period["type"] == "rest")
        
        # Calculate behavior diversity (Shannon entropy)
        total_behaviors = sum(behavior_frequency.values())
        if total_behaviors > 0:
            behavior_probabilities = [count/total_behaviors for count in behavior_frequency.values()]
            behavior_diversity = -sum(p * np.log(p) for p in behavior_probabilities if p > 0)
            # Normalize to 0-1 scale (assuming max 10 different behaviors)
            behavior_diversity = min(1.0, behavior_diversity / np.log(10))
        else:
            behavior_diversity = 0
        
        # Calculate transition rate (transitions per minute)
        transition_rate = (sum(behavior_transitions.values()) / video_duration * 60) if video_duration > 0 else 0
        
        activity_summary = {
            "total_active_time": float(total_active_time),
            "total_rest_time": float(total_rest_time),
            "activity_ratio": float(total_active_time / video_duration) if video_duration > 0 else 0.5,
            "transition_rate": float(transition_rate),
            "behavior_diversity": float(behavior_diversity),
            "dominant_behavior": max(behavior_durations.items(), key=lambda x: x[1])[0] if behavior_durations else "unknown",
            "average_behavior_duration": float(video_duration / len(sorted_sequences)) if sorted_sequences else 0
        }
        
        return {
            "behavior_durations": {k: float(v) for k, v in behavior_durations.items()},
            "behavior_transitions": dict(behavior_transitions),
            "activity_periods": activity_periods[:20],  # Limit to first 20 periods
            "behavior_frequency": dict(behavior_frequency),
            "activity_summary": activity_summary
        }
    
    def _visualize_frame(self, frame: np.ndarray, detection: Dict, behavior: Dict) -> np.ndarray:
        """
        Visualize frame with pose keypoints and behavior label.
        
        Args:
            frame: Input frame
            detection: Detection result
            behavior: Behavior classification result
            
        Returns:
            Visualized frame
        """
        vis_frame = frame.copy()
        
        # Draw bounding box
        x1, y1, x2, y2 = [int(c) for c in detection['bbox']]
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add behavior label
        label = f"{detection['class']}: {behavior['behavior']} ({behavior['confidence']:.2f})"
        cv2.putText(vis_frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw pose keypoints if available
        if self.pose_adapter is not None:
            try:
                pose_result = self.pose_adapter.predict(frame, detection['bbox'])
                vis_frame = self.pose_adapter.visualize_keypoints(vis_frame, pose_result)
            except:
                pass  # Skip visualization on error
        
        return vis_frame
    
    def extract_pose_only(self, image: np.ndarray, bbox: Optional[List[float]] = None) -> Dict:
        """
        Extract pose keypoints only (for testing/debugging).
        
        Args:
            image: Input image
            bbox: Optional bounding box
            
        Returns:
            Pose estimation result
        """
        if not self.use_pose_estimation or self.pose_adapter is None:
            return {"error": "Pose estimation not available"}
        
        try:
            return self.pose_adapter.predict(image, bbox)
        except Exception as e:
            return {"error": f"Pose extraction failed: {str(e)}"}


class EnhancedBehaviorAnalysisPredictorSingleton:
    """Singleton for enhanced predictor with reset capability."""
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> EnhancedBehaviorAnalysisPredictor:
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = EnhancedBehaviorAnalysisPredictor()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset instance - unload models and reinitialize."""
        with cls._lock:
            if cls._instance is not None:
                # Clean up pose model if loaded
                if hasattr(cls._instance, 'pose_adapter') and cls._instance.pose_adapter is not None:
                    del cls._instance.pose_adapter
                # Reset model manager
                model_manager.reset()
                # Recreate instance
                cls._instance = EnhancedBehaviorAnalysisPredictor()
                logger.info("EnhancedBehaviorAnalysisPredictor instance reset")


# Singleton instance
enhanced_predictor = EnhancedBehaviorAnalysisPredictorSingleton.get_instance()