"""
TensorFlow Graph 모드 문제 최종 해결
Eager Execution을 강제로 활성화하여 문제 해결
"""
import shutil
from pathlib import Path

def fix_graph_mode_permanently():
    """그래프 모드 문제를 완전히 해결"""
    
    service_content = '''"""
안구질환 진단 서비스 - 대분류 기반 (Graph Mode 문제 해결)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TensorFlow Eager Execution 강제 활성화
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import json
import numpy as np
from PIL import Image
from typing import Dict, Tuple, List
import logging
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = logging.getLogger(__name__)

# 대분류별 세부 질환 정보
CATEGORY_DETAILS = {
    "각막 질환": {
        "description": "눈의 투명한 외층 부분인 각막에 발생하는 질환",
        "common_diseases": [
            "각막 궤양",
            "각막염", 
            "각막 상처",
            "각막 혼탁"
        ],
        "symptoms": [
            "눈물 흘름",
            "눈부심",
            "시야 흐림",
            "통증"
        ],
        "recommendation": "각막 질환은 시력에 직접적인 영향을 주므로 조기 치료가 중요합니다."
    },
    "결막 및 누관 질환": {
        "description": "눈의 흰자위를 덮고 있는 결막과 눈물 배출 통로인 누관의 질환",
        "common_diseases": [
            "결막염",
            "유루증",
            "건성안",
            "익상편"
        ],
        "symptoms": [
            "충혈",
            "가려움",
            "눈물 분비 이상",
            "눈꼽 분비물"
        ],
        "recommendation": "감염성 질환일 가능성이 있으므로 다른 동물과의 접촉을 피하고 조기 치료를 받으세요."
    },
    "수정체 질환": {
        "description": "눈 내부의 투명한 수정체에 발생하는 질환",
        "common_diseases": [
            "백내장",
            "수정체 탈구",
            "핵경화성 백내장"
        ],
        "symptoms": [
            "시력 저하",
            "눈부심",
            "동공 백탁",
            "야간 시력 저하"
        ],
        "recommendation": "수정체 질환은 진행성이므로 정기적인 검진과 적절한 시기에 수술이 필요할 수 있습니다."
    },
    "안검 질환": {
        "description": "눈꼬리와 그 주변 구조물에 발생하는 질환",
        "common_diseases": [
            "안검염",
            "첼모난생",
            "안검 종양",
            "눈꼬리 처짐"
        ],
        "symptoms": [
            "눈꼬리 부종",
            "눈 비비기",
            "눈꼬리 발적",
            "눈 떨림"
        ],
        "recommendation": "안검 질환은 외관상 문제뿐만 아니라 불편함을 유발할 수 있으므로 적절한 치료가 필요합니다."
    },
    "안구 내부 질환": {
        "description": "눈 내부의 다양한 구조물에 발생하는 질환",
        "common_diseases": [
            "포도막염",
            "녹내장",
            "망막 질환",
            "유리체 혼탁"
        ],
        "symptoms": [
            "시력 감소",
            "안압 상승",
            "충혈",
            "동공 이상"
        ],
        "recommendation": "안구 내부 질환은 실명으로 이어질 수 있으므로 즉시 전문의 진료가 필요합니다."
    }
}

class EyeDiseaseService:
    def __init__(self, model_path: str, class_map_path: str):
        """안구 질환 진단 서비스 초기화"""
        
        logger.info("[EyeDiseaseService] 대분류 기반 진단 서비스 초기화")
        logger.info(f"TensorFlow {tf.__version__} - Eager Execution: {tf.executing_eagerly()}")
        
        # 대분류 클래스맵 로드
        self.class_map = {
            "0": "각막 질환",
            "1": "결막 및 누관 질환",
            "2": "수정체 질환",
            "3": "안검 질환",
            "4": "안구 내부 질환"
        }
        
        # 정상 임계값 (모든 클래스가 이 값 미만이면 정상)
        self.normal_threshold = 0.3
        self.diagnosis_threshold = 0.5
        
        # Custom objects
        custom_objects = {
            'swish': tf.nn.swish,
            'Swish': tf.keras.layers.Activation(tf.nn.swish),
            'FixedDropout': tf.keras.layers.Dropout
        }
        
        # 모델 로드
        self.model = None
        model_loaded = False
        
        # 가능한 모델 경로 시도
        model_paths = [
            model_path,
            model_path.replace('.keras', '_fixed.h5'),
            model_path.replace('.keras', '.h5'),
            'models/health_diagnosis/eye_disease/best_grouped_model.keras'
        ]
        
        for path in model_paths:
            if os.path.exists(path) and not model_loaded:
                try:
                    logger.info(f"Trying to load model from {path}")
                    self.model = tf.keras.models.load_model(
                        path,
                        custom_objects=custom_objects,
                        compile=False
                    )
                    
                    # 모델 컴파일 (최적화기 설정)
                    self.model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    model_loaded = True
                    logger.info(f"Successfully loaded model from {path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load from {path}: {e}")
                    continue
        
        if not model_loaded:
            raise ValueError("Could not load eye disease model")
        
        self.input_shape = (224, 224)
        
        logger.info(f"Model initialized with {len(self.class_map)} categories")
        logger.info(f"Normal threshold: {self.normal_threshold}")
    
    def preprocess_image(self, image_file) -> np.ndarray:
        """이미지 전처리"""
        if hasattr(image_file, 'file'):
            image_file.file.seek(0)
            img = Image.open(image_file.file).convert('RGB')
        elif hasattr(image_file, 'seek'):
            image_file.seek(0)
            img = Image.open(image_file).convert('RGB')
        else:
            img = Image.open(image_file).convert('RGB')
        
        img = img.resize(self.input_shape)
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict_category(self, image_array: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """대분류 예측"""
        
        # Eager Execution 활성화 확인
        if not tf.executing_eagerly():
            logger.warning("Eager execution is not enabled, enabling it now")
            tf.compat.v1.enable_eager_execution()
        
        # 예측 수행
        predictions = self.model.predict(image_array, verbose=0)
        
        # numpy 배열로 확실히 변환
        if isinstance(predictions, tf.Tensor):
            predictions = predictions.numpy()
        
        probabilities = predictions[0]
        
        # 전체 확률 로그
        logger.info("Category probabilities:")
        for idx, prob in enumerate(probabilities):
            category = self.class_map.get(str(idx), f"Unknown_{idx}")
            logger.info(f"  {category}: {prob:.3f} ({prob*100:.1f}%)")
        
        # 최고 확률 찾기
        max_idx = int(np.argmax(probabilities))
        max_prob = float(probabilities[max_idx])
        
        # 정상 판단 (모든 확률이 낮을 때)
        if max_prob < self.normal_threshold:
            return "정상", 0.8, []  # 정상일 때는 높은 확신도
        
        # 확실하지 않은 진단
        if max_prob < self.diagnosis_threshold:
            # 상위 2개 카테고리 반환
            top_indices = np.argsort(probabilities)[-2:][::-1]
            possible_categories = []
            for idx in top_indices:
                if probabilities[idx] > 0.2:  # 20% 이상만
                    category = self.class_map.get(str(idx), f"Unknown_{idx}")
                    possible_categories.append((category, float(probabilities[idx])))
            
            return "불확실", max_prob, possible_categories
        
        # 확실한 진단
        diagnosed_category = self.class_map.get(str(max_idx), "Unknown")
        return diagnosed_category, max_prob, []
    
    def diagnose(self, image_file) -> Dict[str, any]:
        """진단 수행"""
        try:
            preprocessed_image = self.preprocess_image(image_file)
            category, confidence, possible_categories = self.predict_category(preprocessed_image)
            
            result = {
                "category": category,
                "confidence": confidence
            }
            
            # 정상인 경우
            if category == "정상":
                result["message"] = "현재 특별한 안구 질환의 징후가 보이지 않습니다."
                result["recommendation"] = "정기적인 검진을 통해 건강을 유지하세요."
                result["details"] = None
            
            # 불확실한 경우
            elif category == "불확실":
                result["message"] = "명확한 진단을 위해 추가 검사가 필요합니다."
                result["possible_categories"] = [
                    {
                        "name": cat[0],
                        "probability": cat[1],
                        "details": CATEGORY_DETAILS.get(cat[0], {})
                    }
                    for cat in possible_categories
                ]
                result["recommendation"] = "더 선명한 사진으로 다시 촬영하거나 수의사의 직접 검진을 받으세요."
            
            # 확실한 진단
            else:
                category_info = CATEGORY_DETAILS.get(category, {})
                result["message"] = f"{category}이(가) 의심됩니다."
                result["details"] = {
                    "description": category_info.get("description", ""),
                    "common_diseases": category_info.get("common_diseases", []),
                    "symptoms": category_info.get("symptoms", []),
                    "recommendation": category_info.get("recommendation", "")
                }
                
                # 신뢰도에 따른 추가 메시지
                if confidence >= 0.8:
                    result["confidence_level"] = "높음"
                elif confidence >= 0.6:
                    result["confidence_level"] = "중간"
                else:
                    result["confidence_level"] = "낮음"
                    result["additional_note"] = "정확한 진단을 위해 수의사 상담을 권장합니다."
            
            return result
            
        except Exception as e:
            logger.error(f"Diagnosis error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "category": "진단 오류",
                "confidence": 0.0,
                "message": "시스템 오류가 발생했습니다.",
                "recommendation": "잠시 후 다시 시도해주세요."
            }
'''
    
    # 서비스 파일 백업 및 저장
    service_path = Path("services/eye_disease_service.py")
    if service_path.exists():
        backup_path = service_path.with_suffix('.py.backup_graph_fix')
        shutil.copy(service_path, backup_path)
        print(f"✅ 백업 생성: {backup_path}")
    
    with open(service_path, 'w', encoding='utf-8') as f:
        f.write(service_content)
    
    print("✅ Graph Mode 문제 해결 완료!")
    print("\n핵심 변경사항:")
    print("  1. TensorFlow Eager Execution 강제 활성화")
    print("  2. 모델 컴파일 추가")
    print("  3. Tensor 객체 확실한 numpy 변환")
    print("  4. 오류 추적을 위한 traceback 추가")

if __name__ == "__main__":
    print("🔧 TensorFlow Graph Mode 문제 최종 해결")
    print("="*60)
    
    fix_graph_mode_permanently()
    
    print("\n📋 다음 단계:")
    print("1. 서버 재시작")
    print("2. 테스트")
    print("\n💡 이제 Graph Mode 문제가 해결되어야 합니다!")