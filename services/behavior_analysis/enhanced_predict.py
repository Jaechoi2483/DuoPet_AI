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
            # Open video
            logger.info(f"Opening video file: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video file: {video_path}")
                raise ValidationError("video", "Cannot open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps if fps > 0 else 0
            
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
                                vis_frame = self._visualize_frame(
                                    frame, detection, behavior
                                )
                                visualization_frames.append({
                                    "frame": frame_count,
                                    "image": vis_frame
                                })
                    
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
            
            result = {
                "video_duration": video_duration,
                "total_frames": total_frames,
                "behavior_sequences": behavior_sequences,
                "behavior_summary": behavior_summary,
                "abnormal_behaviors": abnormal_behaviors,
                "pose_estimation_used": self.use_pose_estimation,
                "pose_usage_percentage": pose_percentage
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