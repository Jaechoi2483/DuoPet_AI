"""
피부질환 서비스 Eager Execution 문제 해결
"""
import os
import tensorflow as tf

# TensorFlow 2.x 설정을 가장 먼저
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 설정 확인
print(f"TensorFlow 버전: {tf.__version__}")
print(f"Eager execution 활성화: {tf.executing_eagerly()}")

# 피부질환 서비스 파일 수정
from pathlib import Path

def fix_skin_disease_service():
    """피부질환 서비스 파일 수정"""
    
    service_path = Path("services/skin_disease_service.py")
    
    if not service_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {service_path}")
        return False
    
    print(f"🔧 {service_path} 수정 중...")
    
    # 파일 읽기
    with open(service_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 백업 생성
    backup_path = service_path.with_suffix('.py.backup')
    if not backup_path.exists():
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ 백업 생성: {backup_path}")
    
    # 수정할 내용
    modifications = []
    
    # 1. 파일 상단에 TensorFlow 설정 추가
    if "tf.config.run_functions_eagerly(True)" not in content:
        import_section = content.find("import tensorflow")
        if import_section != -1:
            # import tensorflow 다음 줄에 추가
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "import tensorflow" in line:
                    lines.insert(i + 1, "tf.config.run_functions_eagerly(True)")
                    lines.insert(i + 2, "")
                    break
            content = '\n'.join(lines)
            modifications.append("TensorFlow eager execution 설정 추가")
    
    # 2. numpy() 호출 부분 수정
    if ".numpy()" in content:
        # .numpy() 호출을 np.array()로 변경
        content = content.replace(".numpy()", "")
        modifications.append("numpy() 호출 제거")
    
    # 3. numpy.bool_ 타입 변환
    # predict 메서드에서 반환값 정리
    if "return {" in content:
        # 반환 전에 타입 변환 추가
        lines = content.split('\n')
        new_lines = []
        
        for line in lines:
            new_lines.append(line)
            
            # predict 메서드의 return 문 찾기
            if "def predict" in line:
                in_predict_method = True
            
            if "return {" in line and "status" in content[content.find(line):content.find(line)+500]:
                # return 문 전에 타입 변환 코드 추가
                indent = len(line) - len(line.lstrip())
                new_lines.insert(-1, " " * indent + "# numpy 타입을 Python 기본 타입으로 변환")
                new_lines.insert(-1, " " * indent + "import numpy as np")
                new_lines.insert(-1, " " * indent + "def convert_numpy_types(obj):")
                new_lines.insert(-1, " " * (indent + 4) + "if isinstance(obj, dict):")
                new_lines.insert(-1, " " * (indent + 8) + "return {k: convert_numpy_types(v) for k, v in obj.items()}")
                new_lines.insert(-1, " " * (indent + 4) + "elif isinstance(obj, list):")
                new_lines.insert(-1, " " * (indent + 8) + "return [convert_numpy_types(v) for v in obj]")
                new_lines.insert(-1, " " * (indent + 4) + "elif isinstance(obj, np.bool_):")
                new_lines.insert(-1, " " * (indent + 8) + "return bool(obj)")
                new_lines.insert(-1, " " * (indent + 4) + "elif isinstance(obj, (np.int64, np.int32)):")
                new_lines.insert(-1, " " * (indent + 8) + "return int(obj)")
                new_lines.insert(-1, " " * (indent + 4) + "elif isinstance(obj, (np.float64, np.float32)):")
                new_lines.insert(-1, " " * (indent + 8) + "return float(obj)")
                new_lines.insert(-1, " " * (indent + 4) + "elif isinstance(obj, np.ndarray):")
                new_lines.insert(-1, " " * (indent + 8) + "return obj.tolist()")
                new_lines.insert(-1, " " * (indent + 4) + "return obj")
                new_lines.insert(-1, " " * indent + "")
                modifications.append("numpy 타입 변환 함수 추가")
                break
        
        content = '\n'.join(new_lines)
        
        # return 문에서 convert_numpy_types 호출
        content = content.replace(
            "return {",
            "result = {"
        )
        content = content.replace(
            "        }",
            "        }\n        return convert_numpy_types(result)"
        )
    
    # 4. 파일 저장
    if modifications:
        with open(service_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 수정 완료:")
        for mod in modifications:
            print(f"  - {mod}")
    else:
        print("  ℹ️ 수정할 내용이 없습니다.")
    
    return True

def create_fixed_skin_service():
    """수정된 피부질환 서비스 생성"""
    
    fixed_service = '''"""
피부질환 진단 서비스 (수정 버전)
TensorFlow 2.x eager execution 활성화
"""
import os
import tensorflow as tf

# TensorFlow 설정을 가장 먼저
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from common.logger import get_logger
from services.model_registry import ModelRegistry
from services.model_adapters.skin_disease_adapter import SkinDiseaseAdapter

logger = get_logger(__name__)

def convert_numpy_types(obj):
    """numpy 타입을 Python 기본 타입으로 변환"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

class SkinDiseaseService:
    """피부질환 진단 서비스"""
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.models = {}
        self._load_models()
        
    def _load_models(self):
        """모델 로드"""
        try:
            # 모델 경로 설정
            base_path = Path(__file__).parent.parent
            models_path = base_path / "models" / "health_diagnosis" / "skin_disease" / "classification"
            
            # TF2 변환된 모델 우선 사용
            model_configs = {
                "cat_binary": [
                    models_path / "cat_binary" / "cat_binary_model_tf2_perfect.h5",
                    models_path / "cat_binary" / "cat_binary_model.h5"
                ],
                "dog_binary": [
                    models_path / "dog_binary" / "dog_binary_model_tf2_perfect.h5",
                    models_path / "dog_binary" / "dog_binary_model.h5"
                ],
                "dog_multi_136": [
                    models_path / "dog_multi_136" / "dog_multi_136_model_tf2_perfect.h5",
                    models_path / "dog_multi_136" / "dog_multi_136_model.h5"
                ],
                "dog_multi_456": [
                    models_path / "dog_multi_456" / "dog_multi_456_model_tf2_perfect.h5",
                    models_path / "dog_multi_456" / "dog_multi_456_model.h5"
                ]
            }
            
            # 각 모델 로드 시도
            for model_name, paths in model_configs.items():
                for path in paths:
                    if path.exists():
                        try:
                            logger.info(f"Loading {model_name} from {path}")
                            model = tf.keras.models.load_model(str(path), compile=False)
                            
                            # 컴파일
                            if "binary" in model_name:
                                model.compile(
                                    optimizer='adam',
                                    loss='binary_crossentropy',
                                    metrics=['accuracy']
                                )
                            else:
                                model.compile(
                                    optimizer='adam',
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy']
                                )
                            
                            self.models[model_name] = model
                            logger.info(f"Successfully loaded {model_name}")
                            break
                            
                        except Exception as e:
                            logger.warning(f"Failed to load {model_name} from {path}: {e}")
                            continue
            
            logger.info(f"Loaded {len(self.models)} skin disease models")
            
        except Exception as e:
            logger.error(f"Error loading skin disease models: {e}")
    
    def predict(self, image: np.ndarray, pet_type: str) -> Dict[str, Any]:
        """피부질환 예측"""
        
        # 이미지 전처리
        if image.shape != (224, 224, 3):
            image = tf.image.resize(image, (224, 224))
        
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # 배치 차원 추가
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # 펫 타입에 따른 모델 선택
        if pet_type.lower() == "cat":
            model_key = "cat_binary"
            multi_models = []
        else:
            model_key = "dog_binary"
            multi_models = ["dog_multi_136", "dog_multi_456"]
        
        result = {
            "status": "success",
            "pet_type": pet_type,
            "binary_classification": None,
            "multi_classification": {},
            "confidence": 0.0
        }
        
        # Binary classification
        if model_key in self.models:
            model = self.models[model_key]
            pred = model.predict(image, verbose=0)
            
            is_disease = float(pred[0][0]) > 0.5
            confidence = float(pred[0][0]) if is_disease else float(1 - pred[0][0])
            
            result["binary_classification"] = {
                "has_disease": is_disease,
                "confidence": confidence
            }
            result["confidence"] = confidence
        
        # Multi-class classification (개만)
        if pet_type.lower() == "dog" and is_disease:
            for multi_key in multi_models:
                if multi_key in self.models:
                    model = self.models[multi_key]
                    pred = model.predict(image, verbose=0)
                    
                    class_idx = int(np.argmax(pred[0]))
                    confidence = float(pred[0][class_idx])
                    
                    # 클래스 매핑
                    if "136" in multi_key:
                        classes = ["구진플라크", "무증상", "농포여드름"]
                    else:
                        classes = ["과다색소침착", "결절종괴", "미란궤양"]
                    
                    result["multi_classification"][multi_key] = {
                        "class": classes[class_idx],
                        "confidence": confidence
                    }
        
        # numpy 타입 변환
        return convert_numpy_types(result)

# 서비스 인스턴스
_service_instance = None

def get_skin_disease_service():
    """싱글톤 서비스 반환"""
    global _service_instance
    if _service_instance is None:
        _service_instance = SkinDiseaseService()
    return _service_instance
'''
    
    # 수정된 서비스 저장
    output_path = Path("services/skin_disease_service_fixed.py")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(fixed_service)
    
    print(f"\n✅ 수정된 서비스 생성: {output_path}")
    print("\n💡 사용 방법:")
    print("  1. 기존 서비스 백업: mv services/skin_disease_service.py services/skin_disease_service_original.py")
    print("  2. 수정 버전 사용: mv services/skin_disease_service_fixed.py services/skin_disease_service.py")
    print("  3. 서버 재시작")

if __name__ == "__main__":
    print("🔧 피부질환 서비스 수정")
    print("=" * 60)
    
    # 옵션 1: 기존 파일 수정
    # fix_skin_disease_service()
    
    # 옵션 2: 새 파일 생성
    create_fixed_skin_service()
    
    print("\n✅ 작업 완료!")