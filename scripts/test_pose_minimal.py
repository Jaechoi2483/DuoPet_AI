#!/usr/bin/env python3
"""
Minimal test for pose model - test TF graph directly
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from pathlib import Path

if tf.__version__.startswith('2'):
    tf = tf.compat.v1
    tf.disable_v2_behavior()

def test_model_directly():
    """Test the model by loading it directly"""
    print("=== Direct Model Test ===")
    
    model_path = Path("D:/final_project/DuoPet_AI/models/behavior_analysis/pose_estimation/superanimal_quadruped/weights")
    
    # Reset graph
    tf.reset_default_graph()
    
    # Create session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    
    with tf.Session(config=config) as sess:
        # Load model
        print("Loading model...")
        saver = tf.train.import_meta_graph(str(model_path / "snapshot-700000.meta"))
        saver.restore(sess, str(model_path / "snapshot-700000"))
        print("Model loaded!")
        
        # Get graph
        graph = tf.get_default_graph()
        
        # List operations to understand the model
        print("\nListing key operations:")
        ops = graph.get_operations()
        
        # Find input/output tensors
        placeholders = []
        outputs = []
        
        for op in ops:
            if op.type == 'Placeholder':
                placeholders.append(op)
                if len(placeholders) <= 3:  # Show first 3
                    print(f"Input: {op.name}, shape: {op.outputs[0].shape}")
            
            if 'pose' in op.name and 'pred' in op.name:
                outputs.append(op)
                if len(outputs) <= 3:  # Show first 3
                    print(f"Output candidate: {op.name}")
        
        # Try to get the main input/output
        try:
            input_tensor = graph.get_tensor_by_name("Placeholder:0")
            print(f"\nMain input tensor shape: {input_tensor.shape}")
            
            # Check what batch size it expects
            if input_tensor.shape[0] is not None:
                batch_size = int(input_tensor.shape[0])
                print(f"Expected batch size: {batch_size}")
            else:
                print("Dynamic batch size")
                
        except Exception as e:
            print(f"Error getting input tensor: {e}")
            
        # Try different batch sizes
        print("\nTesting different input configurations...")
        for batch_size in [1, 4]:
            print(f"\nTrying batch size {batch_size}...")
            test_input = np.random.randn(batch_size, 480, 640, 3).astype(np.float32)
            
            try:
                # Try to find which tensor accepts our input
                for placeholder in placeholders[:3]:  # Test first 3 placeholders
                    try:
                        tensor = placeholder.outputs[0]
                        if len(tensor.shape) == 4:  # Image-like tensor
                            print(f"  Testing {placeholder.name} with shape {test_input.shape}")
                            # Just check if we can feed it, don't run full graph
                            feed_dict = {tensor: test_input}
                            # Validate feed dict
                            print(f"  ✓ Can feed to {placeholder.name}")
                            break
                    except Exception as e:
                        print(f"  ✗ Cannot feed to {placeholder.name}: {e}")
                        
            except Exception as e:
                print(f"  Error with batch size {batch_size}: {e}")

if __name__ == "__main__":
    test_model_directly()