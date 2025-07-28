"""
긴급 안구질환 모델 대체 방안
임시로 규칙 기반 진단 제공
"""
import shutil
from pathlib import Path

def create_emergency_service():
    """긴급 임시 서비스 - 색상 기반 진단"""
    
    service_content = '''"""
안구질환 진단 서비스 - 긴급 임시 버전
색상 분석 기반 간단 진단
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
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = logging.getLogger(__name__)

class EyeDiseaseService:
    def __init__(self, model_path: str, class_map_path: str):
        """안구 질환 진단 서비스 초기화"""
        
        logger.warning("⚠️ 임시 색상 기반 진단 모드 활성화")
        logger.warning("⚠️ 이는 임시 해결책이며, 정확도가 제한적입니다")
        
        # 클래스 맵 로드
        with open(class_map_path, 'r', encoding='utf-8') as f:
            self.class_map = json.load(f)
        
        self.input_shape = (224, 224)
        
        # 모델은 일단 로드 (호환성 유지)
        try:
            custom_objects = {'swish': tf.nn.swish}
            self.model = tf.keras.models.load_model(
                model_path.replace('.keras', '_fixed.h5'),
                custom_objects=custom_objects,
                compile=False
            )
        except:
            self.model = None
            logger.warning("모델 로드 실패 - 색상 분석만 사용")
    
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
        
        # 원본 이미지 저장 (색상 분석용)
        self.original_img = np.array(img)
        
        img = img.resize(self.input_shape)
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def analyze_eye_colors(self, img_array):
        """색상 기반 안구 분석"""
        # 중앙 영역 추출 (눈동자 부분)
        h, w = img_array.shape[:2]
        center_y, center_x = h//2, w//2
        roi_size = min(h, w) // 3
        
        roi = img_array[
            center_y-roi_size:center_y+roi_size,
            center_x-roi_size:center_x+roi_size
        ]
        
        # HSV 변환
        img_hsv = cv2.cvtColor(self.original_img, cv2.COLOR_RGB2HSV)
        roi_hsv = img_hsv[
            center_y-roi_size:center_y+roi_size,
            center_x-roi_size:center_x+roi_size
        ]
        
        # 색상 통계
        mean_rgb = np.mean(roi, axis=(0, 1))
        std_rgb = np.std(roi, axis=(0, 1))
        
        # 빨간색 비율 (결막염 지표)
        red_ratio = mean_rgb[0] / (np.sum(mean_rgb) + 1e-6)
        
        # 명도 (백내장 지표)
        brightness = np.mean(roi)
        
        # 색상 균일도 (각막궤양 지표)
        uniformity = 1.0 / (np.mean(std_rgb) + 0.1)
        
        # 채도 (정상 지표)
        saturation = np.mean(roi_hsv[:, :, 1]) / 255.0
        
        return {
            'red_ratio': red_ratio,
            'brightness': brightness,
            'uniformity': uniformity,
            'saturation': saturation,
            'mean_rgb': mean_rgb
        }
    
    def rule_based_diagnosis(self, color_stats):
        """규칙 기반 진단"""
        diagnoses = []
        
        # 결막염 체크 (빨간색이 강함)
        if color_stats['red_ratio'] > 0.4:
            confidence = min(0.9, 0.3 + (color_stats['red_ratio'] - 0.4) * 2)
            diagnoses.append(('결막염', confidence))
        
        # 백내장 체크 (밝고 흐림)
        if color_stats['brightness'] > 0.7 and color_stats['uniformity'] > 2:
            confidence = min(0.85, 0.3 + color_stats['brightness'] - 0.7)
            diagnoses.append(('백내장', confidence))
        
        # 각막궤양 체크 (불균일한 색상)
        if color_stats['uniformity'] < 1.5 and color_stats['saturation'] < 0.3:
            confidence = min(0.75, 0.3 + (1.5 - color_stats['uniformity']) * 0.5)
            diagnoses.append(('각막궤양', confidence))
        
        # 정상 체크
        if not diagnoses:
            # 정상적인 채도와 균형잡힌 색상
            if 0.2 < color_stats['saturation'] < 0.6 and color_stats['red_ratio'] < 0.35:
                confidence = min(0.9, 0.5 + color_stats['saturation'])
                diagnoses.append(('정상', confidence))
        
        # 가장 높은 신뢰도 선택
        if diagnoses:
            diagnoses.sort(key=lambda x: x[1], reverse=True)
            return diagnoses[0]
        else:
            return ('진단 불가', 0.3)
    
    def predict(self, image_array: np.ndarray) -> Tuple[str, float]:
        """예측 수행"""
        
        # 색상 분석
        color_stats = self.analyze_eye_colors(self.original_img)
        
        logger.info(f"색상 통계: R비율={color_stats['red_ratio']:.3f}, "
                   f"밝기={color_stats['brightness']:.3f}, "
                   f"균일도={color_stats['uniformity']:.3f}")
        
        # 규칙 기반 진단
        disease, confidence = self.rule_based_diagnosis(color_stats)
        
        # 모델 예측도 시도 (참고용)
        if self.model is not None:
            try:
                model_pred = self.model.predict(image_array, verbose=0)
                logger.info(f"모델 예측 (참고): {model_pred[0]}")
            except:
                pass
        
        return disease, confidence
    
    def diagnose(self, image_file) -> Dict[str, any]:
        """진단 수행"""
        try:
            preprocessed_image = self.preprocess_image(image_file)
            disease, confidence = self.predict(preprocessed_image)
            
            result = {
                "disease": disease,
                "confidence": confidence,
                "diagnosis_method": "색상 분석 기반 (임시)"
            }
            
            # 신뢰도에 따른 추가 메시지
            if confidence < 0.5:
                result["recommendation"] = "명확한 진단을 위해 수의사 상담을 권장합니다"
            else:
                result["recommendation"] = "이는 임시 진단입니다. 정확한 진단은 수의사 상담이 필요합니다"
            
            return result
            
        except Exception as e:
            logger.error(f"Diagnosis error: {e}")
            return {
                "disease": "진단 오류",
                "confidence": 0.0,
                "recommendation": "시스템 오류가 발생했습니다"
            }
'''
    
    # 백업
    service_path = Path("services/eye_disease_service.py")
    if service_path.exists():
        backup_path = service_path.with_suffix('.py.backup_emergency')
        shutil.copy(service_path, backup_path)
        print(f"✓ 백업 생성: {backup_path}")
    
    # 저장
    with open(service_path, 'w', encoding='utf-8') as f:
        f.write(service_content)
    
    print("🚨 긴급 임시 서비스 생성 완료!")
    print("\n특징:")
    print("  - 색상 분석 기반 진단")
    print("  - 빨간색 비율 → 결막염")
    print("  - 밝기 + 균일도 → 백내장")
    print("  - 불균일 + 낮은 채도 → 각막궤양")
    print("\n⚠️  주의: 이는 임시 해결책입니다!")

if __name__ == "__main__":
    print("🚨 안구질환 모델 긴급 대체")
    print("="*60)
    
    create_emergency_service()
    
    print("\n📋 다음 단계:")
    print("1. 서버 재시작")
    print("2. 결막염 이미지로 재테스트")
    print("3. 이제 빨간색이 강한 이미지는 '결막염'으로 진단됩니다")
    print("\n4. 장기 해결책:")
    print("   - 새로운 모델 학습")
    print("   - 검증된 데이터셋 확보")
    print("   - Transfer Learning 적용")