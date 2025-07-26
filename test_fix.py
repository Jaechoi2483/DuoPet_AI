"""
간단한 테스트 스크립트 - 출력 확인용
"""

import sys
import os

print("=== Test Script Started ===")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Script path: {os.path.abspath(__file__)}")

# TensorFlow 임포트 테스트
try:
    print("\nTesting TensorFlow import...")
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print("✅ TensorFlow imported successfully!")
except Exception as e:
    print(f"❌ TensorFlow import error: {e}")

# h5py 임포트 테스트
try:
    print("\nTesting h5py import...")
    import h5py
    print(f"h5py version: {h5py.__version__}")
    print("✅ h5py imported successfully!")
except Exception as e:
    print(f"❌ h5py import error: {e}")

# 모델 파일 존재 여부 확인
print("\nChecking model files...")
model_paths = [
    "models/health_diagnosis/eye_disease/best_grouped_model.keras",
    "models/health_diagnosis/bcs/bcs_classifier.h5",
    "models/health_diagnosis/skin_disease/model.ckpt.index"
]

for path in model_paths:
    if os.path.exists(path):
        print(f"✅ Found: {path}")
    else:
        print(f"❌ Not found: {path}")

print("\n=== Test Script Completed ===")