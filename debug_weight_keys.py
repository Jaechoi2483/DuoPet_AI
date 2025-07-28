"""
H5 파일에서 실제 가중치 키 이름 확인
"""
import h5py
from pathlib import Path

def debug_weight_keys(h5_path):
    """H5 파일의 가중치 키 구조 확인"""
    print(f"\n🔍 {h5_path.name} 가중치 키 분석")
    print("="*60)
    
    with h5py.File(h5_path, 'r') as f:
        if 'model_weights' in f:
            all_keys = []
            
            def collect_keys(name, obj):
                if isinstance(obj, h5py.Dataset):
                    all_keys.append(name)
            
            f['model_weights'].visititems(collect_keys)
            
            # 처음 20개 키 출력
            print(f"\n처음 20개 가중치 키:")
            for i, key in enumerate(all_keys[:20]):
                print(f"  {i+1:2d}. {key}")
            
            # 마지막 10개 키 출력
            print(f"\n마지막 10개 가중치 키:")
            for i, key in enumerate(all_keys[-10:], len(all_keys)-10):
                print(f"  {i+1:2d}. {key}")
            
            print(f"\n총 {len(all_keys)}개 가중치 키")
            
            # Dense 레이어 키 찾기
            print(f"\nDense 레이어 관련 키:")
            dense_keys = [k for k in all_keys if 'dense' in k.lower()]
            for key in dense_keys:
                print(f"  - {key}")
            
            # MobileNet 레이어 구조 확인
            print(f"\nMobileNet 레이어 prefix:")
            prefixes = set()
            for key in all_keys:
                parts = key.split('/')
                if len(parts) > 0:
                    prefixes.add(parts[0])
            
            for prefix in sorted(prefixes):
                count = sum(1 for k in all_keys if k.startswith(prefix + '/'))
                print(f"  - {prefix}: {count}개 가중치")

def main():
    base_dir = Path("models/health_diagnosis/skin_disease/classification")
    
    models = [
        base_dir / "dog_binary/dog_binary_model.h5"
    ]
    
    for model_path in models:
        if model_path.exists():
            debug_weight_keys(model_path)

if __name__ == "__main__":
    main()