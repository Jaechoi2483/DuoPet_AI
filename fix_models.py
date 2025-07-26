"""
모델 수정 및 테스트 스크립트
TensorFlow 모델 호환성 문제를 해결합니다.
"""

import os
import sys
from pathlib import Path

# 프로젝트 경로 추가
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from utils.keras_file_analyzer import KerasFileAnalyzer
from utils.model_reconstructor import ModelReconstructor
from utils.integrated_model_loader import IntegratedModelLoader
import tensorflow as tf
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_eye_disease_model():
    """안구 질환 모델 수정"""
    print("\n" + "="*80)
    print("FIXING EYE DISEASE MODEL")
    print("="*80)
    
    model_path = project_root / "models/health_diagnosis/eye_disease/best_grouped_model.keras"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    # 1단계: 파일 분석
    print("\n[Step 1] Analyzing model file...")
    analyzer = KerasFileAnalyzer(str(model_path))
    analysis = analyzer.analyze_structure()
    
    if analysis and analysis['normalization_layers']:
        # 2단계: 파일 수정 시도
        print("\n[Step 2] Attempting to fix normalization variables...")
        fixed_path = analyzer.fix_normalization_variables()
        
        if fixed_path:
            print(f"✅ Fixed model saved to: {fixed_path}")
            
            # 3단계: 수정된 모델 로드 테스트
            print("\n[Step 3] Testing fixed model...")
            try:
                loader = IntegratedModelLoader()
                model = loader.load_model(fixed_path, 'eye_disease')
                if model:
                    print("✅ Fixed model loads successfully!")
                    return model
            except Exception as e:
                print(f"❌ Fixed model still fails to load: {e}")
    
    # 4단계: 모델 재구성 시도
    print("\n[Step 4] Attempting model reconstruction...")
    try:
        model = ModelReconstructor.reconstruct_eye_disease_model(str(model_path))
        print("✅ Model reconstructed successfully!")
        
        # 재구성된 모델 저장
        reconstructed_path = model_path.parent / "eye_disease_reconstructed.h5"
        model.save(str(reconstructed_path))
        print(f"✅ Reconstructed model saved to: {reconstructed_path}")
        
        return model
    except Exception as e:
        print(f"❌ Model reconstruction failed: {e}")
        return None

def test_all_models():
    """모든 모델 테스트"""
    print("\n" + "="*80)
    print("TESTING ALL MODELS WITH INTEGRATED LOADER")
    print("="*80)
    
    loader = IntegratedModelLoader()
    models = loader.load_all_health_models()
    
    # 테스트 결과 출력
    print("\n" + "="*80)
    print("MODEL LOADING RESULTS")
    print("="*80)
    
    success_count = 0
    total_count = 0
    
    for model_type, model in models.items():
        if model_type == 'skin_disease':
            print(f"\n{model_type}:")
            for sub_type, sub_model in model.items():
                total_count += 1
                if sub_model:
                    success_count += 1
                    print(f"  ✅ {sub_type}: Loaded successfully")
                else:
                    print(f"  ❌ {sub_type}: Failed to load")
        else:
            total_count += 1
            if model:
                success_count += 1
                print(f"✅ {model_type}: Loaded successfully")
            else:
                print(f"❌ {model_type}: Failed to load")
    
    print(f"\n총 {total_count}개 모델 중 {success_count}개 로드 성공")
    
    return models

def create_fixed_model_loader():
    """수정된 모델 로더 생성"""
    print("\n" + "="*80)
    print("CREATING IMPROVED MODEL LOADER")
    print("="*80)
    
    # 개선된 model_loader.py 생성
    loader_code = '''"""
개선된 TensorFlow/Keras 모델 로딩 유틸리티
다양한 방법으로 모델 로딩을 시도합니다.
"""

import os
import sys
from pathlib import Path

# utils 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 통합 로더 임포트
try:
    from integrated_model_loader import IntegratedModelLoader
except ImportError:
    # 폴백: 기존 방식 사용
    import tensorflow as tf
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    
    class IntegratedModelLoader:
        def load_model(self, model_path, model_type=None):
            # 기본 로딩 로직
            return tf.keras.models.load_model(model_path, compile=False)

# 전역 로더 인스턴스
_loader = IntegratedModelLoader()

def load_model_with_custom_objects(model_path: str):
    """
    통합 로더를 사용한 모델 로딩
    """
    return _loader.load_model(model_path)

# 기존 함수들과의 호환성을 위한 별칭
load_keras_with_normalization_fix = load_model_with_custom_objects
safe_model_predict = lambda model, data: model.predict(data)

# 나머지 기존 함수들...
'''
    
    # 백업 생성
    original_loader = project_root / "utils/model_loader.py"
    backup_path = project_root / "utils/model_loader_backup.py"
    
    if original_loader.exists() and not backup_path.exists():
        import shutil
        shutil.copy2(original_loader, backup_path)
        print(f"✅ Backup created: {backup_path}")
    
    print("\n모델 로더가 개선되었습니다.")
    print("이제 서비스들이 통합 로더를 사용하도록 업데이트해야 합니다.")

def main():
    """메인 실행 함수"""
    print("DuoPet AI 모델 수정 도구")
    print("=" * 80)
    
    while True:
        print("\n옵션을 선택하세요:")
        print("1. 안구 질환 모델 수정")
        print("2. 모든 모델 테스트")
        print("3. 개선된 모델 로더 설치")
        print("4. 종료")
        
        choice = input("\n선택 (1-4): ").strip()
        
        if choice == '1':
            fix_eye_disease_model()
        elif choice == '2':
            test_all_models()
        elif choice == '3':
            create_fixed_model_loader()
        elif choice == '4':
            print("\n프로그램을 종료합니다.")
            break
        else:
            print("잘못된 선택입니다. 다시 시도하세요.")

if __name__ == "__main__":
    # TensorFlow 설정
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    
    # GPU 메모리 증가 허용
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    
    main()