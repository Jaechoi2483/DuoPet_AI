"""
SuperAnimal-Quadruped Pose Estimation Adapter (CPU-only version for debugging)
"""

import os
import sys
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import yaml
import logging

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# TensorFlow 2.x compatibility
import tensorflow as tf
if tf.__version__.startswith('2'):
    tf = tf.compat.v1
    tf.disable_v2_behavior()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SuperAnimalQuadrupedAdapterCPU:
    """
    CPU-only adapter class for SuperAnimal-Quadruped pose estimation model
    """
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SuperAnimal-Quadruped model adapter.
        """
        if model_path is None:
            # Default path
            project_root = Path(__file__).parent.parent.parent
            model_path = project_root / "models" / "behavior_analysis" / "pose_estimation" / "superanimal_quadruped"
        
        self.model_path = Path(model_path)
        self.weights_path = self.model_path / "weights"
        self.config = config or {}
        
        # Default configuration
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
        self.input_size = (640, 480)
        self.confidence_threshold = 0.5
        
        # Initialize TensorFlow session
        self.sess = None
        self.inputs = None
        self.outputs = None
        
        # Load model
        self.load_model()
        
    def load_model(self):
        """Load the TensorFlow model with CPU-only configuration"""
        try:
            # Reset default graph
            tf.reset_default_graph()
            
            # Create CPU-only session
            config = tf.ConfigProto(
                device_count={'GPU': 0},  # Force CPU
                allow_soft_placement=True,
                log_device_placement=True  # Log device placement for debugging
            )
            
            self.sess = tf.Session(config=config)
            
            # Check which model format is available
            meta_file = self.weights_path / "snapshot-700000.meta"
            
            if not meta_file.exists():
                raise FileNotFoundError(f"Model file not found: {meta_file}")
            
            # Load from checkpoint
            logger.info("Loading model from checkpoint (CPU-only)...")
            
            # Import meta graph
            with tf.device('/cpu:0'):  # Force CPU device
                saver = tf.train.import_meta_graph(str(meta_file))
                checkpoint_path = str(self.weights_path / "snapshot-700000")
                saver.restore(self.sess, checkpoint_path)
            
            logger.info("Model restored from checkpoint")
            
            # Get graph
            graph = self.sess.graph
            
            # Find input and output tensors
            # First, let's list all placeholders and operations
            logger.info("Analyzing model graph...")
            placeholders = []
            potential_outputs = []
            
            for op in graph.get_operations():
                if op.type == 'Placeholder':
                    placeholders.append(op)
                    logger.info(f"Found placeholder: {op.name}, shape: {op.outputs[0].shape}")
                
                if 'pose' in op.name.lower() and any(x in op.name.lower() for x in ['pred', 'part', 'output']):
                    potential_outputs.append(op)
                    if len(potential_outputs) <= 5:  # Log first 5
                        logger.info(f"Found potential output: {op.name}")
            
            # Get the main input tensor
            if placeholders:
                self.inputs = placeholders[0].outputs[0]
                logger.info(f"Using input tensor: {placeholders[0].name}")
            else:
                raise ValueError("No placeholder found in model")
            
            # Try different output tensor names
            output_names = [
                "pose/part_pred/block4/BiasAdd:0",
                "pose/locref_pred/block4/BiasAdd:0",
                "pose/part_pred/block4/Conv2D:0",
                "Sigmoid:0",
                "concat_1:0"
            ]
            
            for name in output_names:
                try:
                    self.outputs = graph.get_tensor_by_name(name)
                    logger.info(f"Found output tensor: {name}")
                    break
                except:
                    continue
            
            if self.outputs is None and potential_outputs:
                # Use the last pose-related operation
                self.outputs = potential_outputs[-1].outputs[0]
                logger.info(f"Using output tensor: {potential_outputs[-1].name}")
            
            if self.outputs is None:
                raise ValueError("Could not find output tensor")
            
            logger.info(f"Model loaded successfully (CPU-only)")
            logger.info(f"Input shape: {self.inputs.shape}")
            logger.info(f"Output shape: {self.outputs.shape}")
            
            # Test the session
            logger.info("Testing session...")
            test_val = self.sess.run(tf.constant(42.0))
            logger.info(f"Session test successful: {test_val}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
            
    def preprocess_image(self, image: np.ndarray, bbox: Optional[List[float]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess image for model input
        """
        prep_info = {
            "original_shape": image.shape[:2],
            "bbox_used": bbox,
        }
        
        # Resize to model input size
        resized = cv2.resize(image, self.input_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Apply DeepLabCut normalization
        mean_pixel = np.array([123.68, 116.779, 103.939])
        normalized = rgb.astype(np.float32) - mean_pixel
        
        # Create batch of 4
        batch_size = 4
        batched = np.zeros((batch_size, self.input_size[1], self.input_size[0], 3), dtype=np.float32)
        for i in range(batch_size):
            batched[i] = normalized
        
        return batched, prep_info
        
    def predict(self, frame: np.ndarray, bbox: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Process a single frame and return the detected keypoints.
        """
        try:
            logger.info("CPU Predict called")
            
            # Preprocess image
            processed, prep_info = self.preprocess_image(frame, bbox)
            
            # Run inference
            logger.info("Running CPU inference...")
            import time
            start_time = time.time()
            
            # Run with explicit device placement
            with tf.device('/cpu:0'):
                heatmaps = self.sess.run(self.outputs, feed_dict={self.inputs: processed})
            
            inference_time = time.time() - start_time
            logger.info(f"CPU inference completed in {inference_time:.2f}s")
            
            # Return dummy result for now
            return {
                "keypoints": [(0, 0)] * self.num_keypoints,
                "keypoint_names": self.keypoint_names,
                "confidence_scores": [0.5] * self.num_keypoints,
                "valid_keypoints": list(range(self.num_keypoints)),
                "bbox": bbox,
                "inference_time": inference_time
            }
            
        except Exception as e:
            logger.error(f"CPU prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "keypoints": [(0, 0)] * self.num_keypoints,
                "keypoint_names": self.keypoint_names,
                "confidence_scores": [0.0] * self.num_keypoints,
                "valid_keypoints": [],
                "bbox": bbox,
                "error": str(e)
            }
            
    def __del__(self):
        """Cleanup TensorFlow session"""
        if self.sess is not None:
            self.sess.close()