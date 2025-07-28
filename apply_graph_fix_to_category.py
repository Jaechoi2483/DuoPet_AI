"""
Graph Mode 해결책을 대분류 서비스에 적용
fix_eye_universal.py의 핵심 해결책을 현재 서비스에 통합
"""
import shutil
from pathlib import Path

def apply_graph_fix():
    """그래프 모드 해결책 적용"""
    
    # 현재 eye_disease_service.py 읽기
    service_path = Path("services/eye_disease_service.py")
    
    # CATEGORY_DETAILS 정의 읽기
    with open(service_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # CATEGORY_DETAILS 추출
    start_idx = content.find("CATEGORY_DETAILS = {")
    end_idx = content.find("}\n\nclass EyeDiseaseService:") + 1
    category_details = content[start_idx:end_idx]
    
    # 새로운 서비스 코드
    service_content = f'''"""
안구질환 진단 서비스 - 대분류 기반 (Graph/Eager 모드 자동 처리)
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
{category_details}

class EyeDiseaseService:
    def __init__(self, model_path: str, class_map_path: str):
        """안구 질환 진단 서비스 초기화"""
        
        logger.info("[EyeDiseaseService] 대분류 기반 진단 서비스 초기화")
        
        # TF 모드 확인
        self.use_eager = tf.executing_eagerly()
        logger.info(f"[EyeDiseaseService] TensorFlow {{tf.__version__}} - Eager mode: {{self.use_eager}}")
        
        # Graph mode인 경우 Session 초기화
        if not self.use_eager:
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.session = tf.compat.v1.Session(config=config)
            logger.info("[EyeDiseaseService] Graph mode - Session created")
        else:
            self.session = None
        
        # 대분류 클래스맵
        self.class_map = {{
            "0": "각막 질환",
            "1": "결막 및 누관 질환",
            "2": "수정체 질환",
            "3": "안검 질환",
            "4": "안구 내부 질환"
        }}
        
        # 정상 임계값
        self.normal_threshold = 0.3
        self.diagnosis_threshold = 0.5
        
        # Custom objects
        custom_objects = {{
            'swish': tf.nn.swish,
            'Swish': tf.keras.layers.Activation(tf.nn.swish),
            'FixedDropout': tf.keras.layers.Dropout
        }}
        
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
                    logger.info(f"Trying to load model from {{path}}")
                    
                    # Graph mode인 경우
                    if not self.use_eager:
                        with self.session.as_default():
                            with self.session.graph.as_default():
                                self.model = tf.keras.models.load_model(
                                    path,
                                    custom_objects=custom_objects,
                                    compile=False
                                )
                                
                                # 수동 컴파일
                                self.model.compile(
                                    optimizer='adam',
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy']
                                )
                                
                                # 입력 placeholder 생성
                                self.input_placeholder = tf.compat.v1.placeholder(
                                    tf.float32, 
                                    shape=[None, 224, 224, 3],
                                    name='input_image'
                                )
                                
                                # 예측 텐서 생성
                                self.predictions_tensor = self.model(self.input_placeholder)
                                
                                # Graph 초기화
                                self.session.run(tf.compat.v1.global_variables_initializer())
                    
                    # Eager mode인 경우
                    else:
                        self.model = tf.keras.models.load_model(
                            path,
                            custom_objects=custom_objects,
                            compile=False
                        )
                        
                        self.model.compile(
                            optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy']
                        )
                    
                    model_loaded = True
                    logger.info(f"Successfully loaded model from {{path}}")
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load from {{path}}: {{e}}")
                    continue
        
        if not model_loaded:
            raise ValueError("Could not load eye disease model")
        
        self.input_shape = (224, 224)
        
        logger.info(f"Model initialized with {{len(self.class_map)}} categories")
        logger.info(f"Normal threshold: {{self.normal_threshold}}")
    
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
        """대분류 예측 - Graph/Eager mode 자동 처리"""
        
        try:
            if self.use_eager:
                # Eager mode 예측
                predictions = self.model(image_array, training=False)
                if hasattr(predictions, 'numpy'):
                    predictions_np = predictions.numpy()
                else:
                    predictions_np = predictions
            else:
                # Graph mode 예측
                predictions_np = self.session.run(
                    self.predictions_tensor,
                    feed_dict={{self.input_placeholder: image_array}}
                )
            
            probabilities = predictions_np[0]
            
        except Exception as e:
            logger.error(f"Prediction error: {{e}}")
            # Fallback: model.predict 사용
            try:
                if self.use_eager:
                    predictions_np = self.model.predict(image_array, verbose=0)
                else:
                    with self.session.as_default():
                        predictions_np = self.model.predict(image_array, verbose=0)
                
                probabilities = predictions_np[0]
                
            except Exception as e2:
                logger.error(f"Fallback prediction also failed: {{e2}}")
                # 기본값 반환
                return "진단 오류", 0.0, []
        
        # 전체 확률 로그
        logger.info("Category probabilities:")
        for idx, prob in enumerate(probabilities):
            category = self.class_map.get(str(idx), f"Unknown_{{idx}}")
            logger.info(f"  {{category}}: {{prob:.3f}} ({{prob*100:.1f}}%)")
        
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
                    category = self.class_map.get(str(idx), f"Unknown_{{idx}}")
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
            
            result = {{
                "category": category,
                "confidence": confidence
            }}
            
            # 정상인 경우
            if category == "정상":
                result["message"] = "현재 특별한 안구 질환의 징후가 보이지 않습니다."
                result["recommendation"] = "정기적인 검진을 통해 건강을 유지하세요."
                result["details"] = None
            
            # 불확실한 경우
            elif category == "불확실":
                result["message"] = "명확한 진단을 위해 추가 검사가 필요합니다."
                result["possible_categories"] = [
                    {{
                        "name": cat[0],
                        "probability": cat[1],
                        "details": CATEGORY_DETAILS.get(cat[0], {{}})
                    }}
                    for cat in possible_categories
                ]
                result["recommendation"] = "더 선명한 사진으로 다시 촬영하거나 수의사의 직접 검진을 받으세요."
            
            # 확실한 진단
            else:
                category_info = CATEGORY_DETAILS.get(category, {{}})
                result["message"] = f"{{category}}이(가) 의심됩니다."
                result["details"] = {{
                    "description": category_info.get("description", ""),
                    "common_diseases": category_info.get("common_diseases", []),
                    "symptoms": category_info.get("symptoms", []),
                    "recommendation": category_info.get("recommendation", "")
                }}
                
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
            logger.error(f"Diagnosis error: {{e}}")
            import traceback
            logger.error(traceback.format_exc())
            return {{
                "category": "진단 오류",
                "confidence": 0.0,
                "message": "시스템 오류가 발생했습니다.",
                "recommendation": "잠시 후 다시 시도해주세요."
            }}
    
    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'session') and self.session is not None:
            self.session.close()
'''
    
    # 백업
    if service_path.exists():
        backup_path = service_path.with_suffix('.py.backup_graph_fixed')
        shutil.copy(service_path, backup_path)
        print(f"✅ 백업 생성: {backup_path}")
    
    # 저장
    with open(service_path, 'w', encoding='utf-8') as f:
        f.write(service_content)
    
    print("✅ Graph Mode 해결책 적용 완료!")
    print("\n핵심 변경사항:")
    print("  1. TensorFlow 모드 자동 감지 (Graph/Eager)")
    print("  2. Graph mode: Session과 placeholder 사용")
    print("  3. Eager mode: 직접 모델 호출")
    print("  4. 두 가지 fallback 메커니즘")
    print("  5. 대분류 기반 진단 유지")

if __name__ == "__main__":
    print("🔧 Graph Mode 해결책을 대분류 서비스에 적용")
    print("="*60)
    
    apply_graph_fix()
    
    print("\n📋 다음 단계:")
    print("1. 서버 재시작")
    print("2. 테스트")
    print("\n💡 이제 정말로 Graph Mode 문제가 해결되어야 합니다!")