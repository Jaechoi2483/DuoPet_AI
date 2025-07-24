#!/usr/bin/env python3
"""
Simple test to check if TensorFlow can load the SuperAnimal model
"""

import os
import sys
from pathlib import Path

# Check TensorFlow availability
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("TensorFlow not available in this environment")
    sys.exit(1)

# Model path
model_path = Path(__file__).parent.parent / "models" / "behavior_analysis" / "pose_estimation" / "superanimal_quadruped" / "weights"
meta_file = model_path / "snapshot-700000.meta"

print(f"Model directory: {model_path}")
print(f"Meta file exists: {meta_file.exists()}")

if meta_file.exists():
    print("\nAttempting to load model...")
    
    # Reset and create session
    tf.reset_default_graph()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    
    with tf.Session(config=config) as sess:
        try:
            # Import meta graph
            saver = tf.train.import_meta_graph(str(meta_file))
            print("✓ Meta graph loaded successfully")
            
            # List operations
            graph = tf.get_default_graph()
            ops = graph.get_operations()
            
            print(f"\nTotal operations in graph: {len(ops)}")
            
            # Find placeholders
            placeholders = [op for op in ops if op.type == 'Placeholder']
            print(f"\nPlaceholders ({len(placeholders)}):")
            for ph in placeholders[:5]:
                print(f"  - {ph.name}: {ph.outputs[0].shape}")
            
            # Find pose-related operations
            pose_ops = [op for op in ops if 'pose' in op.name.lower()]
            print(f"\nPose-related operations ({len(pose_ops)}):")
            for op in pose_ops[-5:]:  # Last 5
                print(f"  - {op.name}: {op.type}")
            
            # Try to restore weights
            checkpoint_path = str(model_path / "snapshot-700000")
            saver.restore(sess, checkpoint_path)
            print("\n✓ Model weights restored successfully")
            
            # Try to find specific tensors
            try:
                input_tensor = graph.get_tensor_by_name("Placeholder:0")
                print(f"\nInput tensor found: {input_tensor.shape}")
            except:
                print("\nStandard input tensor not found")
            
            try:
                output_tensor = graph.get_tensor_by_name("pose/part_pred/block4/BiasAdd:0")
                print(f"Output tensor found: {output_tensor.shape}")
            except:
                print("Standard output tensor not found")
                
        except Exception as e:
            print(f"\n✗ Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
else:
    print("\n✗ Model file not found!")
    print("\nFiles in weights directory:")
    if model_path.exists():
        for f in model_path.iterdir():
            print(f"  - {f.name}")