#!/usr/bin/env python3
"""
YOLO 모델 로딩 상세 테스트
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# models 경로 추가
models_path = project_root / "models"
sys.path.insert(0, str(models_path))

print("=" * 60)
print("YOLO 모델 로딩 상세 테스트")
print("=" * 60)
print(f"Current working directory: {os.getcwd()}")
print(f"Project root: {project_root}")
print(f"Models path: {models_path}")
print(f"Python paths: {sys.path[:3]}")

# models.yolo 해킹 테스트
print("\n1. models.yolo 해킹 테스트:")
import types
if 'models' not in sys.modules:
    sys.modules['models'] = types.ModuleType('models')
    print("✅ 'models' 모듈 추가됨")
else:
    print("✅ 'models' 모듈 이미 존재")
    
if 'models.yolo' not in sys.modules:
    sys.modules['models.yolo'] = types.ModuleType('models.yolo')
    print("✅ 'models.yolo' 모듈 추가됨")
else:
    print("✅ 'models.yolo' 모듈 이미 존재")

# 커스텀 YOLO import 테스트
print("\n2. 커스텀 YOLO import 테스트:")
try:
    from yolo_models.experimental import attempt_load
    from yolo_utils.general import non_max_suppression
    from yolo_utils.torch_utils import select_device
    print("✅ 커스텀 YOLO 모듈 import 성공!")
    USE_CUSTOM_YOLO = True
except ImportError as e:
    print(f"❌ 커스텀 YOLO 모듈 import 실패: {e}")
    USE_CUSTOM_YOLO = False

# YOLO 모델 로딩 테스트
if USE_CUSTOM_YOLO:
    print("\n3. YOLO 모델 로딩 테스트:")
    try:
        device = select_device('')
        print(f"✅ Device selected: {device}")
        
        model_path = project_root / "models/behavior_analysis/detection/behavior_yolo_catdog_v1_original.pt"
        print(f"Loading model from: {model_path}")
        
        model = attempt_load(str(model_path), map_location=device)
        print("✅ YOLO 모델 로딩 성공!")
        
        # 모델 정보
        if hasattr(model, 'names'):
            print(f"Classes: {model.names}")
        elif hasattr(model, 'module') and hasattr(model.module, 'names'):
            print(f"Classes: {model.module.names}")
            
    except Exception as e:
        print(f"❌ YOLO 모델 로딩 실패: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)