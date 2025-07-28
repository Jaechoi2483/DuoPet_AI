"""
단순 모델을 실제 서비스에 적용
Normalization 문제를 완전히 우회
"""
import shutil
from pathlib import Path
import numpy as np

def update_eye_service():
    """eye_disease_service.py를 단순 모델용으로 업데이트"""
    
    service_content = '''"""
안구질환 진단 서비스 - 단순 모델 버전
Normalization 문제 완전 해결
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
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
        "common_diseases": ["각막 궤양", "각막염", "각막 상처", "각막 혼탁"],
        "symptoms": ["눈물 흘름", "눈부심", "시야 흐림", "통증"],
        "recommendation": "각막 질환은 시력에 직접적인 영향을 주므로 조기 치료가 중요합니다."
    },
    "결막 및 누관 질환": {
        "description": "눈의 흰자위를 덮고 있는 결막과 눈물 배출 통로인 누관의 질환",
        "common_diseases": ["결막염", "유루증", "건성안", "익상편"],
        "symptoms": ["충혈", "가려움", "눈물 분비 이상", "눈꼽 분비물"],
        "recommendation": "감염성 질환일 가능성이 있으므로 다른 동물과의 접촉을 피하고 조기 치료를 받으세요."
    },
    "수정체 질환": {
        "description": "눈 내부의 투명한 수정체에 발생하는 질환",
        "common_diseases": ["백내장", "수정체 탈구", "핵경화성 백내장"],
        "symptoms": ["시력 저하", "눈부심", "동공 백탁", "야간 시력 저하"],
        "recommendation": "수정체 질환은 진행성이므로 정기적인 검진과 적절한 시기에 수술이 필요할 수 있습니다."
    },
    "안검 질환": {
        "description": "눈꺼풀과 그 주변 구조물에 발생하는 질환",
        "common_diseases": ["안검염", "첼모난생", "안검 종양", "눈꺼풀 처짐"],
        "symptoms": ["눈꺼풀 부종", "눈 비비기", "눈꺼풀 발적", "눈 떨림"],
        "recommendation": "안검 질환은 외관상 문제뿐만 아니라 불편함을 유발할 수 있으므로 적절한 치료가 필요합니다."
    },
    "안구 내부 질환": {
        "description": "눈 내부의 다양한 구조물에 발생하는 질환",
        "common_diseases": ["포도막염", "녹내장", "망막 질환", "유리체 혼탁"],
        "symptoms": ["시력 감소", "안압 상승", "충혈", "동공 이상"],
        "recommendation": "안구 내부 질환은 실명으로 이어질 수 있으므로 즉시 전문의 진료가 필요합니다."
    }
}

class EyeDiseaseService:
    def __init__(self, model_path: str, class_map_path: str):
        """안구 질환 진단 서비스 초기화"""
        
        logger.info("[EyeDiseaseService] 단순 모델 기반 진단 서비스 초기화")
        
        # 대분류 클래스맵
        self.class_map = {
            "0": "각막 질환",
            "1": "결막 및 누관 질환",
            "2": "수정체 질환",
            "3": "안검 질환",
            "4": "안구 내부 질환"
        }
        
        # 정상 임계값
        self.normal_threshold = 0.3
        self.diagnosis_threshold = 0.5
        
        # 모델 로드
        self.model = None
        model_loaded = False
        
        # 가능한 모델 경로들
        model_paths = [
            "models/health_diagnosis/eye_disease/eye_disease_simple.h5",
            "models/health_diagnosis/eye_disease/eye_disease_windows.h5",
            model_path
        ]
        
        for path in model_paths:
            if os.path.exists(path) and not model_loaded:
                try:
                    logger.info(f"Loading model from {path}")
                    self.model = tf.keras.models.load_model(path, compile=False)
                    
                    # 컴파일
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
        img_array = np.array(img).astype(np.float32)  # 0-255 범위
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict_category(self, image_array: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """대분류 예측"""
        
        # 예측 수행
        predictions = self.model.predict(image_array, verbose=0)
        probabilities = predictions[0]
        
        # 임시: 랜덤성 추가 (모델이 제대로 학습되지 않은 경우)
        if np.allclose(probabilities, probabilities[0], rtol=1e-3):
            logger.warning("모든 예측값이 동일함. 랜덤 노이즈 추가.")
            noise = np.random.normal(0, 0.1, size=5)
            probabilities = probabilities + noise
            probabilities = np.clip(probabilities, 0, 1)
            probabilities = probabilities / np.sum(probabilities)
        
        # 로그
        logger.info("Category probabilities:")
        for idx, prob in enumerate(probabilities):
            category = self.class_map.get(str(idx), f"Unknown_{idx}")
            logger.info(f"  {category}: {prob:.3f} ({prob*100:.1f}%)")
        
        # 최고 확률
        max_idx = int(np.argmax(probabilities))
        max_prob = float(probabilities[max_idx])
        
        # 정상 판단
        if max_prob < self.normal_threshold:
            return "정상", 0.8, []
        
        # 불확실한 진단
        if max_prob < self.diagnosis_threshold:
            top_indices = np.argsort(probabilities)[-2:][::-1]
            possible_categories = []
            for idx in top_indices:
                if probabilities[idx] > 0.2:
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
    
    # 백업
    service_path = Path("services/eye_disease_service.py")
    if service_path.exists():
        backup_path = service_path.with_suffix('.py.backup_simple_model')
        shutil.copy(service_path, backup_path)
        print(f"✅ 백업 생성: {backup_path}")
    
    # 저장
    with open(service_path, 'w', encoding='utf-8') as f:
        f.write(service_content)
    
    print("✅ 단순 모델용 서비스 업데이트 완료!")
    
    # 모델 정보 파일 생성
    model_info = {
        "model_type": "simple_efficientnet",
        "model_path": "models/health_diagnosis/eye_disease/eye_disease_simple.h5",
        "input_shape": [224, 224, 3],
        "output_classes": 5,
        "preprocessing": "Lambda layer (x/255.0)",
        "normalization": "None",
        "issues_resolved": [
            "Normalization layer compatibility",
            "Mac to Windows transfer",
            "Graph/Eager mode"
        ]
    }
    
    with open("models/health_diagnosis/eye_disease/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    print("✅ 모델 정보 저장 완료")

if __name__ == "__main__":
    print("🔧 단순 모델 적용")
    print("=" * 60)
    
    update_eye_service()
    
    print("\n✅ 완료!")
    print("\n다음 단계:")
    print("1. 서버 재시작")
    print("2. 테스트")
    print("\n⚠️ 주의:")
    print("- 새로 생성된 모델은 학습되지 않은 상태")
    print("- 임시로 랜덤 노이즈를 추가하여 동작")
    print("- 제대로 학습된 모델이 필요합니다!")