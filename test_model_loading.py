#!/usr/bin/env python3
"""
모델 로딩 테스트 스크립트
"""
import os
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Windows 경로도 추가
if os.name == 'nt':
    windows_path = Path(r"D:\final_project\DuoPet_AI\models")
    sys.path.insert(0, str(windows_path))

print("=" * 60)
print("모델 로딩 테스트")
print("=" * 60)
print(f"Python path: {sys.path[:3]}...")

# 1. 커스텀 YOLO import 테스트
print("\n1. 커스텀 YOLO import 테스트:")
try:
    from yolo_models.experimental import attempt_load
    from yolo_utils.general import non_max_suppression
    from yolo_utils.torch_utils import select_device
    print("✅ 커스텀 YOLO 모듈 import 성공!")
    USE_CUSTOM_YOLO = True
except ImportError as e:
    print(f"❌ 커스텀 YOLO 모듈 import 실패: {e}")
    USE_CUSTOM_YOLO = False

# 2. 모델 파일 존재 확인
print("\n2. 모델 파일 확인:")
model_paths = {
    "YOLO (original)": "models/behavior_analysis/detection/behavior_yolo_catdog_v1_original.pt",
    "YOLO (v1)": "models/behavior_analysis/detection/behavior_yolo_catdog_v1.pt",
    "LSTM (dog)": "models/behavior_analysis/classification/behavior_dog_lstm_v1.pth",
    "LSTM (cat)": "models/behavior_analysis/classification/behavior_cat_lstm_v1.pth",
}

for name, path in model_paths.items():
    full_path = project_root / path
    exists = full_path.exists()
    size = full_path.stat().st_size / 1024 / 1024 if exists else 0
    print(f"{'✅' if exists else '❌'} {name}: {full_path} ({size:.1f} MB)")

# 3. LSTM 모델 구조 확인
print("\n3. LSTM 모델 구조 확인:")
import torch

lstm_path = project_root / "models/behavior_analysis/classification/behavior_dog_lstm_v1.pth"
if lstm_path.exists():
    try:
        checkpoint = torch.load(lstm_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            print(f"✅ Checkpoint keys: {list(checkpoint.keys())}")
            if 'state_dict' in checkpoint:
                print("✅ 'state_dict' 키가 있습니다. 체크포인트 형식입니다.")
        else:
            print("✅ 순수 state_dict 형식입니다.")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")

print("\n" + "=" * 60)