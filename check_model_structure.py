#!/usr/bin/env python3
"""
모델 구조 확인 스크립트
"""
import torch
from pathlib import Path

def check_model(model_path, model_name):
    print(f"\n{'='*60}")
    print(f"Checking: {model_name}")
    print(f"Path: {model_path}")
    print('='*60)
    
    if not Path(model_path).exists():
        print("❌ 파일이 존재하지 않습니다.")
        return
        
    try:
        # 모델 로드
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 체크포인트 타입 확인
        if isinstance(checkpoint, dict):
            print(f"✅ Dictionary checkpoint with keys: {list(checkpoint.keys())[:5]}...")
            
            # state_dict 키가 있는 경우
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"\n📊 State dict keys (first 10):")
                for i, key in enumerate(list(state_dict.keys())[:10]):
                    print(f"  - {key}")
                print(f"  ... (total {len(state_dict)} keys)")
                
                # 모델 구조 추측
                if any('st_gcn' in key for key in state_dict.keys()):
                    print("\n🔍 모델 타입: ST-GCN (Spatial Temporal Graph Convolutional Network)")
                    print("   - 스켈레톤 기반 행동 인식 모델")
                    print("   - LSTM이 아님!")
                elif any('lstm' in key.lower() for key in state_dict.keys()):
                    print("\n🔍 모델 타입: LSTM")
                elif any('conv' in key for key in state_dict.keys()):
                    print("\n🔍 모델 타입: CNN 기반")
                else:
                    print("\n🔍 모델 타입: 알 수 없음")
                    
            # meta 정보 확인
            if 'meta' in checkpoint:
                print(f"\n📋 Meta information:")
                meta = checkpoint['meta']
                if isinstance(meta, dict):
                    for key, value in list(meta.items())[:5]:
                        print(f"  - {key}: {value}")
                        
        else:
            # 순수 state_dict인 경우
            print("✅ Pure state_dict (not a checkpoint)")
            print(f"\n📊 Model keys (first 10):")
            for i, key in enumerate(list(checkpoint.keys())[:10]):
                print(f"  - {key}")
                
    except Exception as e:
        print(f"❌ 에러 발생: {e}")

# 모든 모델 확인
base_path = Path("D:/final_project/DuoPet_AI/models/behavior_analysis")

models_to_check = [
    (base_path / "detection/behavior_yolo_catdog_v1_original.pt", "YOLO Detection Model"),
    (base_path / "classification/behavior_dog_lstm_v1.pth", "Dog Behavior Model"),
    (base_path / "classification/behavior_cat_lstm_v1.pth", "Cat Behavior Model"),
]

print("="*60)
print("DuoPet AI 모델 구조 분석")
print("="*60)

for model_path, model_name in models_to_check:
    check_model(str(model_path), model_name)

print("\n" + "="*60)
print("분석 완료!")
print("="*60)