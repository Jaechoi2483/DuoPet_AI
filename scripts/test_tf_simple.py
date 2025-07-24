#!/usr/bin/env python3
"""
Test TensorFlow setup and basic operations
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)

# Test TF 1.x compatibility mode
if tf.__version__.startswith('2'):
    tf = tf.compat.v1
    tf.disable_v2_behavior()
    print("Using TF 2.x in compatibility mode")

# Test simple operation
try:
    # Create a simple graph
    with tf.Graph().as_default():
        # Create placeholders
        x = tf.placeholder(tf.float32, shape=[None, 3])
        w = tf.Variable(tf.random_normal([3, 2]))
        b = tf.Variable(tf.zeros([2]))
        
        # Simple operation
        y = tf.matmul(x, w) + b
        
        # Create session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        
        with tf.Session(config=config) as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer())
            
            # Test data
            test_data = np.random.randn(4, 3).astype(np.float32)
            
            # Run
            print("Running simple TF operation...")
            result = sess.run(y, feed_dict={x: test_data})
            print("Success! Result shape:", result.shape)
            
except Exception as e:
    print("TensorFlow test failed:", str(e))
    import traceback
    traceback.print_exc()

print("\nTesting if GPU is available:")
print("CUDA available:", tf.test.is_gpu_available())
print("GPU devices:", tf.config.list_physical_devices('GPU') if hasattr(tf.config, 'list_physical_devices') else [])