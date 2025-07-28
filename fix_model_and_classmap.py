"""
안구질환 모델 및 클래스맵 정리
원본 모델과 클래스맵을 올바르게 사용하도록 수정
"""
import shutil
from pathlib import Path
import json

def fix_model_and_classmap():
    """모델과 클래스맵을 올바르게 설정"""
    
    print("🔧 안구질환 모델 및 클래스맵 정리")
    print("=" * 60)
    
    # 1. 올바른 클래스맵 복사
    source_classmap = Path("/mnt/c/Users/ictedu1_021/Desktop/안구질환모델/class_map.json")
    target_classmap = Path("models/health_diagnosis/eye_disease/class_map.json")
    
    # 백업
    if target_classmap.exists():
        backup_path = target_classmap.with_suffix('.json.backup_original')
        shutil.copy(target_classmap, backup_path)
        print(f"✓ 기존 클래스맵 백업: {backup_path}")
    
    # 복사
    shutil.copy(source_classmap, target_classmap)
    print(f"✓ 올바른 클래스맵 복사 완료")
    
    # 클래스맵 내용 확인
    with open(target_classmap, 'r', encoding='utf-8') as f:
        class_map = json.load(f)
    
    print("\n📋 클래스맵 내용:")
    for idx, category in class_map.items():
        print(f"  {idx}: {category}")
    
    # 2. 원본 모델 복사 (이미 있지만 확실하게)
    source_model = Path("/mnt/c/Users/ictedu1_021/Desktop/안구질환모델/best_grouped_model.keras")
    target_model = Path("models/health_diagnosis/eye_disease/best_grouped_model.keras")
    
    if source_model.exists():
        # 백업
        if target_model.exists():
            backup_model = target_model.with_suffix('.keras.backup_original')
            shutil.copy(target_model, backup_model)
            print(f"\n✓ 기존 모델 백업: {backup_model}")
        
        # 복사
        shutil.copy(source_model, target_model)
        print(f"✓ 원본 모델 복사 완료")
    
    # 3. 서비스가 대분류를 사용하는지 확인
    service_path = Path("services/eye_disease_service.py")
    with open(service_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if '"각막 질환"' in content:
        print("\n✓ 서비스는 이미 대분류를 사용하고 있습니다")
    else:
        print("\n❌ 서비스가 세부 질환을 사용하고 있습니다 - 수정 필요")
    
    print("\n✅ 완료!")
    print("\n다음 단계:")
    print("1. 서버 재시작")
    print("2. 테스트")

if __name__ == "__main__":
    fix_model_and_classmap()