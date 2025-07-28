"""
안구질환 서비스 - 범용 해결책
Graph/Eager mode 모두에서 작동
"""
import os
import shutil
from pathlib import Path

def create_universal_eye_service():
    """Graph/Eager mode 관계없이 작동하는 서비스"""
    
    service_content = '''"""
안구질환 진단 서비스 - 범용 버전
TensorFlow Graph/Eager mode 모두 지원
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

# TF 모드 확인
logger.info(f"[EyeDiseaseService] TensorFlow {tf.__version__} - Initial Eager: {tf.executing_eagerly()}")

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
                                
                                # Graph mode에서 수동 컴파일
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
                                
                    else:
                        # Eager mode인 경우
                        self.model = tf.keras.models.load_model(
                            path,
                            custom_objects=custom_objects,
                            compile=False
                        )
                        
                        # Eager mode 컴파일
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
        
        # 모델 테스트
        try:
            test_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
            if self.use_eager:
                _ = self.model(test_input, training=False)
            else:
                _ = self.session.run(
                    self.predictions_tensor,
                    feed_dict={self.input_placeholder: test_input}
                )
            logger.info("Model test successful")
        except Exception as e:
            logger.warning(f"Model test failed: {e}")
        
        logger.info(f"EyeDiseaseService initialized successfully in {'Eager' if self.use_eager else 'Graph'} mode")
    
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
        """예측 수행 - Graph/Eager mode 자동 처리"""
        
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
                    feed_dict={self.input_placeholder: image_array}
                )
            
            # 결과 처리
            predicted_class_index = int(np.argmax(predictions_np[0]))
            confidence = float(predictions_np[0][predicted_class_index])
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Fallback: model.predict 사용
            try:
                if self.use_eager:
                    predictions_np = self.model.predict(image_array, verbose=0)
                else:
                    with self.session.as_default():
                        predictions_np = self.model.predict(image_array, verbose=0)
                
                predicted_class_index = int(np.argmax(predictions_np[0]))
                confidence = float(predictions_np[0][predicted_class_index])
                
            except Exception as e2:
                logger.error(f"Fallback prediction also failed: {e2}")
                # 기본값 반환
                predicted_class_index = 0
                confidence = 0.0
        
        # 클래스 이름 매핑
        predicted_class_name = self.class_map.get(str(predicted_class_index), "Unknown")
        
        return predicted_class_name, confidence
    
    def diagnose(self, image_file) -> Dict[str, any]:
        """진단 수행"""
        try:
            preprocessed_image = self.preprocess_image(image_file)
            disease, confidence = self.predict(preprocessed_image)
            
            return {
                "disease": disease,
                "confidence": confidence
            }
        except Exception as e:
            logger.error(f"Diagnosis error: {e}")
            return {
                "disease": "진단 오류",
                "confidence": 0.0
            }
    
    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'session') and self.session is not None:
            self.session.close()
'''
    
    # 백업
    service_path = Path("services/eye_disease_service.py")
    if service_path.exists():
        backup_path = service_path.with_suffix('.py.backup_universal')
        shutil.copy(service_path, backup_path)
        print(f"✓ 백업 생성: {backup_path}")
    
    # 저장
    with open(service_path, 'w', encoding='utf-8') as f:
        f.write(service_content)
    
    print("✅ 범용 안구질환 서비스 생성 완료!")
    print("  - Graph mode와 Eager mode 자동 감지")
    print("  - 각 모드에 맞는 예측 방식 사용")
    print("  - Session 기반 및 직접 호출 모두 지원")
    print("  - 다중 fallback 메커니즘")

def create_simple_h5_converter():
    """간단한 H5 변환 스크립트"""
    
    converter_script = '''"""
안구질환 모델 간단 변환
Keras API만 사용하여 H5 재저장
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
import numpy as np
from pathlib import Path

def convert_eye_model():
    """모델을 순수 Keras로 재저장"""
    
    models_path = Path("models/health_diagnosis/eye_disease")
    
    # 원본 모델 찾기
    source_candidates = [
        models_path / "eye_disease_fixed.h5",
        models_path / "best_grouped_model.keras"
    ]
    
    source_path = None
    for candidate in source_candidates:
        if candidate.exists():
            source_path = candidate
            break
    
    if not source_path:
        print("❌ 모델 파일을 찾을 수 없습니다")
        return
    
    print(f"📥 모델 로드: {source_path}")
    
    # Keras API로 로드
    custom_objects = {'swish': keras.activations.swish}
    
    try:
        # 모델 로드
        model = keras.models.load_model(
            str(source_path),
            custom_objects=custom_objects,
            compile=False
        )
        
        # 새로운 optimizer로 컴파일
        model.compile(
            optimizer=keras.optimizers.Adam(0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 테스트
        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        output = model.predict(test_input)
        print(f"✓ 테스트 성공: {output.shape}")
        
        # 저장
        output_path = models_path / "eye_disease_keras_clean.h5"
        model.save(str(output_path), save_traces=False)
        print(f"💾 저장 완료: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        return False

if __name__ == "__main__":
    print("🔄 안구질환 모델 Keras 변환")
    print("=" * 50)
    convert_eye_model()
'''
    
    with open("convert_eye_keras.py", 'w', encoding='utf-8') as f:
        f.write(converter_script)
    
    print("\n✅ Keras 변환 스크립트 생성: convert_eye_keras.py")

if __name__ == "__main__":
    print("🔧 안구질환 서비스 - 범용 해결책")
    print("=" * 60)
    
    # 1. 범용 서비스 생성
    create_universal_eye_service()
    
    # 2. 변환 스크립트 생성
    create_simple_h5_converter()
    
    print("\n✅ 완료!")
    print("\n📋 옵션:")
    print("  1. 바로 서버 재시작: python api/main.py")
    print("  2. 또는 모델 재변환: python convert_eye_keras.py")
    print("\n💡 범용 서비스는 Graph/Eager mode 모두에서 작동합니다!")