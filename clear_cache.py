"""Python 캐시 삭제 스크립트"""
import os
import shutil

def clear_pycache():
    """__pycache__ 디렉토리 모두 삭제"""
    count = 0
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(pycache_path)
                print(f"삭제됨: {pycache_path}")
                count += 1
            except Exception as e:
                print(f"삭제 실패: {pycache_path} - {e}")
    
    print(f"\n총 {count}개의 __pycache__ 디렉토리 삭제됨")

if __name__ == "__main__":
    clear_pycache()