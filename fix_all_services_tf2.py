"""
모든 AI 서비스에 TensorFlow 2.x 설정 적용
"""
import os
import shutil
from pathlib import Path
import re

def ensure_tf2_config(content: str) -> tuple[str, list[str]]:
    """파일 내용에 TF2 설정 확인 및 추가"""
    modifications = []
    lines = content.split('\n')
    
    # TF 설정이 이미 있는지 확인
    has_eager_config = any('tf.config.run_functions_eagerly(True)' in line for line in lines)
    has_tf_import = any('import tensorflow' in line for line in lines)
    
    if has_tf_import and not has_eager_config:
        # TensorFlow import 찾기
        new_lines = []
        tf_imported = False
        
        for i, line in enumerate(lines):
            # import 섹션 시작 부분 찾기
            if not tf_imported and ('import os' in line or i == 0):
                # 최상단에 TF 설정 추가
                if i == 0 and line.strip() and not line.startswith('"""'):
                    new_lines.append('import os')
                    new_lines.append('os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"')
                    new_lines.append('import tensorflow as tf')
                    new_lines.append('tf.config.run_functions_eagerly(True)')
                    new_lines.append('')
                    modifications.append("TF2 설정 추가 (최상단)")
                
                new_lines.append(line)
                
            elif not tf_imported and 'import tensorflow' in line:
                # TensorFlow import 직후에 설정 추가
                new_lines.append(line)
                new_lines.append('tf.config.run_functions_eagerly(True)')
                tf_imported = True
                modifications.append("TF2 eager execution 설정 추가")
                
            else:
                new_lines.append(line)
        
        # os.environ 설정 추가
        if 'TF_CPP_MIN_LOG_LEVEL' not in content:
            # import os 찾기
            for i, line in enumerate(new_lines):
                if 'import os' in line:
                    new_lines.insert(i + 1, 'os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"')
                    modifications.append("TF 로그 레벨 설정 추가")
                    break
        
        content = '\n'.join(new_lines)
    
    # numpy 타입 변환 함수 추가 (필요한 경우)
    if 'convert_numpy_types' not in content and 'numpy' in content:
        # 함수 추가할 위치 찾기 (import 섹션 이후)
        lines = content.split('\n')
        import_end_idx = 0
        
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('import') and not line.startswith('from'):
                if i > 5:  # import 섹션이 끝났다고 판단
                    import_end_idx = i
                    break
        
        numpy_converter = '''
def convert_numpy_types(obj):
    """numpy 타입을 Python 기본 타입으로 변환"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
'''
        
        lines.insert(import_end_idx, numpy_converter)
        content = '\n'.join(lines)
        modifications.append("numpy 타입 변환 함수 추가")
    
    return content, modifications

def fix_service_file(file_path: Path) -> bool:
    """서비스 파일 수정"""
    if not file_path.exists():
        print(f"  ❌ 파일 없음: {file_path}")
        return False
    
    print(f"\n🔧 {file_path.name} 처리 중...")
    
    # 백업 생성
    backup_path = file_path.with_suffix('.py.backup_tf2')
    if not backup_path.exists():
        shutil.copy(file_path, backup_path)
        print(f"  ✓ 백업 생성: {backup_path}")
    
    # 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # TF2 설정 적용
    modified_content, modifications = ensure_tf2_config(content)
    
    if modifications:
        # 파일 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print(f"  ✅ 수정 완료:")
        for mod in modifications:
            print(f"     - {mod}")
        return True
    else:
        print(f"  ℹ️ 수정 필요 없음")
        return False

def main():
    """메인 함수"""
    print("🚀 모든 AI 서비스에 TensorFlow 2.x 설정 적용")
    print("=" * 60)
    
    # 수정할 서비스 파일 목록
    service_files = [
        Path("services/eye_disease_service.py"),
        Path("services/skin_disease_service.py"),
        Path("services/bcs_service.py"),
        Path("services/model_registry.py"),
        Path("services/health_diagnosis_orchestrator.py"),
        Path("services/face_login_service.py"),
        Path("services/behavior_analysis_service.py"),
    ]
    
    # model_adapters 디렉토리의 파일들
    adapters_dir = Path("services/model_adapters")
    if adapters_dir.exists():
        adapter_files = list(adapters_dir.glob("*.py"))
        service_files.extend(adapter_files)
    
    # utils 디렉토리의 model_loader.py
    utils_files = [
        Path("utils/model_loader.py"),
    ]
    service_files.extend(utils_files)
    
    # 각 파일 처리
    modified_count = 0
    for file_path in service_files:
        if fix_service_file(file_path):
            modified_count += 1
    
    print(f"\n📊 결과:")
    print(f"  - 총 {len(service_files)}개 파일 검사")
    print(f"  - {modified_count}개 파일 수정됨")
    
    if modified_count > 0:
        print("\n⚠️ 중요: 서버를 재시작해야 변경사항이 적용됩니다!")
        print("\n📋 다음 단계:")
        print("  1. 서버 중지: Ctrl+C")
        print("  2. 서버 재시작: python api/main.py")
        print("  3. 프론트엔드에서 다시 테스트")
    
    # 간단한 검증 스크립트 생성
    create_validation_script()

def create_validation_script():
    """TF2 설정 검증 스크립트 생성"""
    validation_script = '''"""
TensorFlow 2.x 설정 검증
"""
import tensorflow as tf
import os

print("🔍 TensorFlow 설정 확인")
print("=" * 50)

print(f"TensorFlow 버전: {tf.__version__}")
print(f"Eager execution 활성화: {tf.executing_eagerly()}")
print(f"GPU 사용 가능: {len(tf.config.list_physical_devices('GPU')) > 0}")
print(f"TF 로그 레벨: {os.environ.get('TF_CPP_MIN_LOG_LEVEL', 'Not set')}")

# 간단한 연산 테스트
try:
    x = tf.constant([1, 2, 3])
    y = tf.constant([4, 5, 6])
    z = tf.add(x, y)
    print(f"\\n테스트 연산: {x.numpy()} + {y.numpy()} = {z.numpy()}")
    print("✅ Eager execution 정상 작동!")
except Exception as e:
    print(f"❌ Eager execution 오류: {e}")

print("\\n💡 팁: 모든 서비스 파일이 올바르게 설정되었는지 확인하세요.")
'''
    
    with open("validate_tf2_config.py", 'w', encoding='utf-8') as f:
        f.write(validation_script)
    
    print(f"\n✅ 검증 스크립트 생성: validate_tf2_config.py")
    print("  실행: python validate_tf2_config.py")

if __name__ == "__main__":
    main()