"""
SuperAnimal-Quadruped Pose Estimation Adapter
"""

import os
import sys
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import yaml
import logging

# TensorFlow 2.x compatibility
import tensorflow as tf
if tf.__version__.startswith('2'):
    tf = tf.compat.v1
    tf.disable_v2_behavior()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SuperAnimalQuadrupedAdapter:
    """
    Adapter class for SuperAnimal-Quadruped pose estimation model
    """
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SuperAnimal-Quadruped model adapter.
        
        Args:
            model_path: Path to the model directory
            config: Configuration parameters
        """
        if model_path is None:
            # Default path
            project_root = Path(__file__).parent.parent.parent
            model_path = project_root / "models" / "behavior_analysis" / "pose_estimation" / "superanimal_quadruped"
        
        self.model_path = Path(model_path)
        self.weights_path = self.model_path / "weights"
        self.config = config or {}
        
        # Load model configuration
        self.load_config()
        
        # Initialize TensorFlow session
        self.sess = None
        self.inputs = None
        self.outputs = None
        
        # Load model
        self.load_model()
        
    def load_config(self):
        """Load model configuration"""
        config_path = self.model_path / "config" / "model_config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                model_config = yaml.safe_load(f)
                self.keypoint_names = model_config['keypoints']['names']
                self.num_keypoints = model_config['keypoints']['count']
                self.skeleton = model_config['keypoints']['skeleton']
                self.input_size = tuple(model_config.get('input_size', [640, 480]))
                self.confidence_threshold = model_config.get('confidence_threshold', 0.5)
        else:
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
            
        logger.info(f"Loaded configuration: {self.num_keypoints} keypoints")
        
    def load_model(self):
        """Load the TensorFlow model"""
        try:
            # Reset default graph
            tf.reset_default_graph()
            
            # Create session with better config
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            # Limit memory growth and disable GPU if causing issues
            config.gpu_options.per_process_gpu_memory_fraction = 0.3
            # For debugging, we can force CPU
            # config.device_count = {'GPU': 0}
            
            self.sess = tf.Session(config=config)
            
            # Check which model format is available
            pb_file = self.weights_path / "snapshot-700000.pb"
            meta_file = self.weights_path / "snapshot-700000.meta"
            
            if meta_file.exists():
                # Load from checkpoint (most common for DeepLabCut)
                logger.info("Loading model from checkpoint...")
                saver = tf.train.import_meta_graph(str(meta_file))
                checkpoint_path = str(self.weights_path / "snapshot-700000")
                saver.restore(self.sess, checkpoint_path)
                logger.info("Model restored from checkpoint")
                
                # Initialize any uninitialized variables
                uninitialized_vars = self.sess.run(tf.report_uninitialized_variables())
                if len(uninitialized_vars) > 0:
                    logger.warning(f"Found uninitialized variables: {uninitialized_vars}")
                    init = tf.variables_initializer([v for v in tf.global_variables() 
                                                   if v.name.encode() in uninitialized_vars])
                    self.sess.run(init)
                    logger.info("Initialized previously uninitialized variables")
                    
            elif pb_file.exists():
                # Load from pb file
                logger.info("Loading model from pb file...")
                with tf.gfile.GFile(str(pb_file), "rb") as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name="")
            else:
                raise FileNotFoundError(f"No model files found in {self.weights_path}")
            
            # Get input and output tensors
            graph = self.sess.graph
            
            # Try to find input/output tensors
            # Common names in DeepLabCut models
            try:
                # DeepLabCut ResNet models typically use these names
                self.inputs = graph.get_tensor_by_name("Placeholder:0")
                # For ResNet-50, the output is typically at this layer
                self.outputs = graph.get_tensor_by_name("pose/part_pred/block4/BiasAdd:0")
                logger.info("Found standard DeepLabCut tensor names")
            except:
                # Alternative names for different DeepLabCut versions
                try:
                    self.inputs = graph.get_tensor_by_name("input:0")
                    self.outputs = graph.get_tensor_by_name("output:0")
                    logger.info("Found alternative tensor names")
                except:
                    # List all operations to debug
                    logger.warning("Could not find standard tensor names, searching for alternatives...")
                    
                    # Find input placeholder
                    placeholders = []
                    for op in graph.get_operations():
                        if op.type == 'Placeholder':
                            placeholders.append(op)
                            if 'input' in op.name.lower() or len(placeholders) == 1:
                                self.inputs = op.outputs[0]
                                logger.info(f"Found input tensor: {op.name}")
                    
                    # Find output tensor (look for pose prediction layers)
                    pose_ops = []
                    for op in graph.get_operations():
                        if 'pose' in op.name.lower() and ('pred' in op.name.lower() or 'part' in op.name.lower()):
                            pose_ops.append(op)
                    
                    if pose_ops:
                        # Take the last pose prediction operation
                        self.outputs = pose_ops[-1].outputs[0]
                        logger.info(f"Found output tensor: {pose_ops[-1].name}")
                        
            if self.inputs is None or self.outputs is None:
                raise ValueError("Could not identify input/output tensors in the model")
                
            logger.info(f"Model loaded successfully")
            logger.info(f"Input shape: {self.inputs.shape}")
            logger.info(f"Output shape: {self.outputs.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
            
    def preprocess_image(self, image: np.ndarray, bbox: Optional[List[float]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess image for model input
        
        Args:
            image: Input image (BGR format from cv2)
            bbox: Optional bounding box [x1, y1, x2, y2]
            
        Returns:
            Tuple of (preprocessed image, preprocessing info)
        """
        prep_info = {
            "original_shape": image.shape[:2],
            "bbox_used": bbox,
            "crop_coords": None,
            "scale_factor": 1.0
        }
        
        # Extract ROI if bbox provided
        if bbox is not None:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            # Add padding
            pad = 20
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(image.shape[1], x2 + pad)
            y2 = min(image.shape[0], y2 + pad)
            
            prep_info["crop_coords"] = (x1, y1, x2, y2)
            image = image[y1:y2, x1:x2]
        
        # Calculate scale factor
        height, width = image.shape[:2]
        target_height, target_width = self.input_size[1], self.input_size[0]
        prep_info["scale_factor"] = min(target_width / width, target_height / height)
        
        # Resize to model input size
        resized = cv2.resize(image, self.input_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Apply DeepLabCut normalization (ImageNet mean subtraction)
        # From pose_cfg.yaml: mean_pixel: [123.68, 116.779, 103.939]
        mean_pixel = np.array([123.68, 116.779, 103.939])
        normalized = rgb.astype(np.float32) - mean_pixel
        
        # The model expects batch size 4, so we create a batch with 4 copies
        batch_size = 4
        batched = np.zeros((batch_size, self.input_size[1], self.input_size[0], 3), dtype=np.float32)
        for i in range(batch_size):
            batched[i] = normalized
        
        return batched, prep_info
        
    def predict(self, frame: np.ndarray, bbox: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Process a single frame and return the detected keypoints.
        
        Args:
            frame: Input image frame as numpy array (H,W,C)
            bbox: Optional bounding box [x1, y1, x2, y2]
            
        Returns:
            Dictionary containing keypoints and their confidence scores
        """
        try:
            logger.info("Predict called with frame shape: %s", frame.shape)
            
            # Preprocess image
            logger.info("Preprocessing image...")
            processed, prep_info = self.preprocess_image(frame, bbox)
            logger.info("Preprocessed shape: %s", processed.shape)
            
            # Run inference with timeout protection
            logger.info("Running TensorFlow inference...")
            logger.info("Input tensor shape: %s", self.inputs.shape)
            logger.info("Output tensor shape: %s", self.outputs.shape)
            logger.info("Feed dict shape: %s", processed.shape)
            
            # Try running a simple test first
            try:
                logger.info("Testing session with simple operation...")
                test_result = self.sess.run(tf.constant(1.0))
                logger.info(f"Session test successful: {test_result}")
            except Exception as e:
                logger.error(f"Session test failed: {e}")
                raise
            
            # Now run the actual inference
            import time
            start_time = time.time()
            logger.info("Starting model inference...")
            
            # For CPU inference, we might need to run it in a different way
            try:
                heatmaps = self.sess.run(self.outputs, feed_dict={self.inputs: processed})
                inference_time = time.time() - start_time
                logger.info(f"Inference completed in {inference_time:.2f}s, heatmaps shape: {heatmaps.shape}")
            except Exception as e:
                logger.error(f"Inference failed after {time.time() - start_time:.2f}s: {str(e)}")
                raise
            
            # Extract keypoints from heatmaps
            keypoints = []
            confidence_scores = []
            
            # Get dimensions
            h, w = frame.shape[:2]
            hmap_h, hmap_w = heatmaps.shape[1:3] if len(heatmaps.shape) >= 4 else (heatmaps.shape[1], heatmaps.shape[2])
            
            # Process each keypoint
            num_keypoints_in_output = heatmaps.shape[-1] if len(heatmaps.shape) >= 4 else heatmaps.shape[2]
            for i in range(self.num_keypoints):
                if i < num_keypoints_in_output:
                    if len(heatmaps.shape) >= 4:
                        heatmap = heatmaps[0, :, :, i]
                    else:
                        heatmap = heatmaps[0, :, i].reshape(hmap_h, hmap_w)
                    
                    # Find peak in heatmap
                    max_val = np.max(heatmap)
                    max_loc = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                    
                    # Convert heatmap coordinates to resized image coordinates
                    y_resized = max_loc[0] * self.input_size[1] / hmap_h
                    x_resized = max_loc[1] * self.input_size[0] / hmap_w
                    
                    # Convert to original image coordinates
                    if prep_info["crop_coords"] is not None:
                        # If cropped, adjust for crop
                        x1_crop, y1_crop = prep_info["crop_coords"][:2]
                        x = x_resized + x1_crop
                        y = y_resized + y1_crop
                    else:
                        # If not cropped, scale back to original size
                        x = x_resized * w / self.input_size[0]
                        y = y_resized * h / self.input_size[1]
                    
                    keypoints.append((float(x), float(y)))
                    confidence_scores.append(float(max_val))
                else:
                    # Missing keypoint
                    keypoints.append((0.0, 0.0))
                    confidence_scores.append(0.0)
            
            # Identify valid keypoints
            valid_keypoints = [
                i for i, conf in enumerate(confidence_scores) 
                if conf >= self.confidence_threshold
            ]
            
            return {
                "keypoints": keypoints,
                "keypoint_names": self.keypoint_names,
                "confidence_scores": confidence_scores,
                "valid_keypoints": valid_keypoints,
                "bbox": bbox
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            # Return empty result
            return {
                "keypoints": [(0, 0)] * self.num_keypoints,
                "keypoint_names": self.keypoint_names,
                "confidence_scores": [0.0] * self.num_keypoints,
                "valid_keypoints": [],
                "bbox": bbox
            }
            
    def get_skeleton_connections(self) -> List[Tuple[str, str]]:
        """Get skeleton connections for visualization"""
        return self.skeleton if hasattr(self, 'skeleton') else []
        
    def visualize_keypoints(self, image: np.ndarray, keypoint_data: Dict[str, Any]) -> np.ndarray:
        """
        Visualize keypoints on image
        
        Args:
            image: Input image
            keypoint_data: Keypoint data from predict()
            
        Returns:
            Image with keypoints drawn
        """
        vis_image = image.copy()
        keypoints = keypoint_data["keypoints"]
        valid_keypoints = keypoint_data["valid_keypoints"]
        confidence_scores = keypoint_data["confidence_scores"]
        
        # Draw keypoints
        for i in valid_keypoints:
            x, y = keypoints[i]
            conf = confidence_scores[i]
            
            # Color based on confidence
            color = (0, int(255 * conf), int(255 * (1 - conf)))
            cv2.circle(vis_image, (int(x), int(y)), 5, color, -1)
            
            # Add label
            label = f"{self.keypoint_names[i][:3]}"
            cv2.putText(vis_image, label, (int(x) + 5, int(y) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw skeleton
        skeleton = self.get_skeleton_connections()
        for connection in skeleton:
            try:
                idx1 = self.keypoint_names.index(connection[0])
                idx2 = self.keypoint_names.index(connection[1])
                
                if idx1 in valid_keypoints and idx2 in valid_keypoints:
                    pt1 = tuple(map(int, keypoints[idx1]))
                    pt2 = tuple(map(int, keypoints[idx2]))
                    cv2.line(vis_image, pt1, pt2, (0, 255, 0), 2)
            except ValueError:
                continue
                
        return vis_image
        
    def __del__(self):
        """Cleanup TensorFlow session"""
        if self.sess is not None:
            self.sess.close()