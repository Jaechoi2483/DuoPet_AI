#!/usr/bin/env python3
"""
YOLO 모델 로딩 완전 테스트 - 실제 서버 환경 시뮬레이션
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
print("YOLO 모델 로딩 완전 테스트")
print("=" * 60)

# services.behavior_analysis.predict에서 사용하는 것과 동일한 해킹 적용
import types

if 'models' not in sys.modules:
    sys.modules['models'] = types.ModuleType('models')
    print("✅ 'models' 모듈 추가됨")
    
if 'models.yolo' not in sys.modules:
    models_yolo = types.ModuleType('models.yolo')
    sys.modules['models.yolo'] = models_yolo
    print("✅ 'models.yolo' 모듈 추가됨")
    
    # Model과 Detect 클래스를 yolo_models.yolo에서 가져와서 models.yolo에 추가
    try:
        from yolo_models.yolo import Model, Detect
        setattr(models_yolo, 'Model', Model)
        setattr(models_yolo, 'Detect', Detect)
        print("✅ Model과 Detect 클래스를 models.yolo에 추가")
    except ImportError:
        # 클래스가 없으면 더미 클래스 생성
        class DummyModel:
            pass
        class DummyDetect:
            pass
        setattr(models_yolo, 'Model', DummyModel)
        setattr(models_yolo, 'Detect', DummyDetect)
        print("⚠️ 더미 Model과 Detect 클래스 사용")

# models.common도 필요할 수 있음
if 'models.common' not in sys.modules:
    models_common = types.ModuleType('models.common')
    sys.modules['models.common'] = models_common
    print("✅ 'models.common' 모듈 추가됨")
    try:
        from yolo_models.common import *
        # yolo_models.common의 모든 속성을 models.common으로 복사
        import yolo_models.common
        for attr in dir(yolo_models.common):
            if not attr.startswith('_'):
                setattr(models_common, attr, getattr(yolo_models.common, attr))
        print("✅ common 모듈 속성 복사 완료")
    except ImportError:
        print("⚠️ common 모듈 속성 복사 실패")

# 이제 커스텀 YOLO import 테스트
print("\n커스텀 YOLO import 테스트:")
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
    print("\nYOLO 모델 로딩 테스트:")
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
        
        # 모델 구조 확인
        print(f"\n모델 타입: {type(model)}")
        print(f"모델 속성: {[attr for attr in dir(model) if not attr.startswith('_')][:10]}...")
            
    except Exception as e:
        print(f"❌ YOLO 모델 로딩 실패: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)