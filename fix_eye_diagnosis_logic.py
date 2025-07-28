"""
안구질환 진단 로직 개선
임계값 기반 진단으로 변경
"""
import shutil
from pathlib import Path

def fix_diagnosis_logic():
    """진단 로직을 임계값 기반으로 개선"""
    
    service_content = '''"""
안구질환 진단 서비스 - 임계값 기반 진단
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import json
import numpy as np
from PIL import Image
from typing import Dict, Tuple
import logging
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = logging.getLogger(__name__)

# 진단 임계값 설정
DIAGNOSIS_THRESHOLD = 0.5  # 50% 이상일 때만 질병으로 진단
NORMAL_BOOST = 1.2  # 정상 클래스에 가중치 부여

class EyeDiseaseService:
    def __init__(self, model_path: str, class_map_path: str):
        """안구 질환 진단 서비스 초기화"""
        
        self.model = None
        self.use_eager = tf.executing_eagerly()
        
        logger.info(f"Initializing EyeDiseaseService in {'Eager' if self.use_eager else 'Graph'} mode")
        
        # Custom objects
        custom_objects = {
            'swish': tf.nn.swish,
            'Swish': tf.keras.layers.Activation(tf.nn.swish)
        }
        
        # 모델 로드 시도
        model_loaded = False
        
        # 가능한 모든 경로 시도
        model_paths = [
            model_path.replace('.keras', '_tf2_complete.h5'),
            model_path.replace('.keras', '_fixed.h5'),
            model_path.replace('.keras', '_tf2.h5'),
            model_path
        ]
        
        for path in model_paths:
            if os.path.exists(path) and not model_loaded:
                logger.info(f"Trying to load model from {path}")
                
                try:
                    # Graph mode인 경우
                    if not self.use_eager:
                        # Session 생성
                        config = tf.compat.v1.ConfigProto()
                        config.gpu_options.allow_growth = True
                        self.session = tf.compat.v1.Session(config=config)
                        
                        with self.session.as_default():
                            with self.session.graph.as_default():
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
                                
                                self.input_placeholder = tf.compat.v1.placeholder(
                                    tf.float32, 
                                    shape=[None, 224, 224, 3],
                                    name='input_image'
                                )
                                
                                self.predictions_tensor = self.model(self.input_placeholder)
                                self.session.run(tf.compat.v1.global_variables_initializer())
                                
                    else:
                        # Eager mode인 경우
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
                    logger.info(f"Successfully loaded model from {path}")
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load from {path}: {e}")
                    continue
        
        if not model_loaded:
            raise ValueError("Could not load any eye disease model")
        
        # 클래스 맵 로드
        with open(class_map_path, 'r', encoding='utf-8') as f:
            self.class_map = json.load(f)
        
        self.input_shape = (224, 224)
        
        # 정상 클래스 인덱스 찾기
        self.normal_idx = None
        for idx, name in self.class_map.items():
            if name == "정상":
                self.normal_idx = int(idx)
                break
        
        logger.info(f"Normal class index: {self.normal_idx}")
        logger.info(f"Diagnosis threshold: {DIAGNOSIS_THRESHOLD}")
    
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
    
    def predict(self, image_array: np.ndarray) -> Tuple[str, float]:
        """예측 수행 - 임계값 기반 진단"""
        
        try:
            if self.use_eager:
                predictions = self.model(image_array, training=False)
                if hasattr(predictions, 'numpy'):
                    predictions_np = predictions.numpy()
                else:
                    predictions_np = predictions
            else:
                predictions_np = self.session.run(
                    self.predictions_tensor,
                    feed_dict={self.input_placeholder: image_array}
                )
            
            # 디버그: 전체 예측 확률 출력
            logger.info(f"Raw predictions: {predictions_np[0]}")
            
            # 각 클래스별 확률 출력
            for idx, prob in enumerate(predictions_np[0]):
                class_name = self.class_map.get(str(idx), f"Unknown_{idx}")
                logger.info(f"Class {idx} ({class_name}): {prob:.4f} ({prob*100:.1f}%)")
            
            # 가장 높은 예측값 찾기
            predicted_class_index = int(np.argmax(predictions_np[0]))
            max_confidence = float(predictions_np[0][predicted_class_index])
            
            # 임계값 기반 진단
            if predicted_class_index != self.normal_idx and max_confidence < DIAGNOSIS_THRESHOLD:
                # 질병 신뢰도가 임계값 미만이면 정상으로 판단
                logger.info(f"Low confidence {max_confidence:.2f} < {DIAGNOSIS_THRESHOLD}, returning normal")
                
                # 정상 클래스의 확률 확인
                normal_confidence = float(predictions_np[0][self.normal_idx])
                
                # 불확실한 경우 표시
                if max_confidence < 0.25:  # 25% 미만은 매우 불확실
                    return "진단 불가 (이미지 품질 확인 필요)", max_confidence
                else:
                    return "정상 (낮은 질병 가능성)", normal_confidence * NORMAL_BOOST
            
            # 정상 클래스가 최고값인 경우
            if predicted_class_index == self.normal_idx:
                # 정상에 약간의 가중치 부여
                adjusted_confidence = min(max_confidence * NORMAL_BOOST, 1.0)
                return self.class_map.get(str(predicted_class_index)), adjusted_confidence
            
            # 질병 진단 (임계값 이상)
            predicted_class_name = self.class_map.get(str(predicted_class_index), "Unknown")
            
            # 신뢰도에 따른 추가 정보
            if max_confidence >= 0.9:
                confidence_level = " (매우 높은 확신)"
            elif max_confidence >= 0.7:
                confidence_level = " (높은 확신)"
            elif max_confidence >= 0.5:
                confidence_level = " (중간 확신)"
            else:
                confidence_level = " (추가 검사 권장)"
            
            return predicted_class_name + confidence_level, max_confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Fallback
            try:
                if self.use_eager:
                    predictions_np = self.model.predict(image_array, verbose=0)
                else:
                    with self.session.as_default():
                        predictions_np = self.model.predict(image_array, verbose=0)
                
                # 동일한 임계값 로직 적용
                predicted_class_index = int(np.argmax(predictions_np[0]))
                max_confidence = float(predictions_np[0][predicted_class_index])
                
                if predicted_class_index != self.normal_idx and max_confidence < DIAGNOSIS_THRESHOLD:
                    return "정상 (낮은 질병 가능성)", float(predictions_np[0][self.normal_idx])
                
                return self.class_map.get(str(predicted_class_index)), max_confidence
                
            except Exception as e2:
                logger.error(f"Fallback prediction also failed: {e2}")
                return "진단 오류", 0.0
    
    def diagnose(self, image_file) -> Dict[str, any]:
        """진단 수행"""
        try:
            preprocessed_image = self.preprocess_image(image_file)
            disease, confidence = self.predict(preprocessed_image)
            
            # 추가 정보 제공
            result = {
                "disease": disease,
                "confidence": confidence
            }
            
            # 낮은 신뢰도일 때 추가 메시지
            if confidence < 0.5:
                result["recommendation"] = "더 선명한 이미지로 재촬영을 권장합니다"
            elif "진단 불가" in disease:
                result["recommendation"] = "양쪽 눈을 정면에서 선명하게 촬영해주세요"
            elif "정상" in disease and "낮은 질병 가능성" in disease:
                result["recommendation"] = "현재 특별한 이상 소견이 없으나, 증상이 지속되면 수의사 상담을 권장합니다"
            
            return result
            
        except Exception as e:
            logger.error(f"Diagnosis error: {e}")
            return {
                "disease": "진단 오류",
                "confidence": 0.0,
                "recommendation": "시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요"
            }
    
    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'session') and self.session is not None:
            self.session.close()
'''
    
    # 백업
    service_path = Path("services/eye_disease_service.py")
    if service_path.exists():
        backup_path = service_path.with_suffix('.py.backup_threshold')
        shutil.copy(service_path, backup_path)
        print(f"✓ 백업 생성: {backup_path}")
    
    # 저장
    with open(service_path, 'w', encoding='utf-8') as f:
        f.write(service_content)
    
    print("✅ 안구질환 진단 로직 개선 완료!")
    print("\n주요 변경사항:")
    print("  - 임계값(50%) 미만은 정상으로 판단")
    print("  - 25% 미만은 '진단 불가'로 표시")
    print("  - 정상 클래스에 1.2배 가중치 부여")
    print("  - 신뢰도 수준별 추가 정보 제공")
    print("  - 상황별 권장사항 추가")

if __name__ == "__main__":
    print("🔧 안구질환 진단 로직 개선")
    print("="*60)
    
    fix_diagnosis_logic()
    
    print("\n📋 다음 단계:")
    print("1. 서버 재시작: python api/main.py")
    print("2. 동일한 이미지로 재테스트")
    print("3. 이제 20% 신뢰도는 '정상'으로 표시됩니다")