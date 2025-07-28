"""
안구질환 모델 완전한 TensorFlow 2.x 변환
Graph mode 충돌 해결
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.config.run_functions_eagerly(True)

import numpy as np
from pathlib import Path
import json
import shutil

print(f"TensorFlow {tf.__version__} - Eager: {tf.executing_eagerly()}")

def convert_eye_model_to_tf2():
    """안구질환 모델을 완전한 TF2 형식으로 변환"""
    
    models_path = Path("models/health_diagnosis/eye_disease")
    
    # 1. 원본 모델 로드
    print("\n🔄 안구질환 모델 TF2 변환 시작...")
    
    # Custom objects
    custom_objects = {
        'swish': tf.nn.swish,
        'Swish': tf.keras.layers.Activation(tf.nn.swish)
    }
    
    # 가능한 모델 파일들
    model_candidates = [
        models_path / "eye_disease_fixed.h5",
        models_path / "best_grouped_model.keras"
    ]
    
    model = None
    source_path = None
    
    for candidate in model_candidates:
        if candidate.exists():
            try:
                print(f"  📥 모델 로드 시도: {candidate}")
                model = tf.keras.models.load_model(
                    str(candidate),
                    custom_objects=custom_objects,
                    compile=False
                )
                source_path = candidate
                print(f"  ✓ 모델 로드 성공!")
                break
            except Exception as e:
                print(f"  ❌ 로드 실패: {e}")
    
    if model is None:
        print("❌ 모델을 찾을 수 없습니다")
        return False
    
    # 2. 모델 구조 확인
    print(f"\n📊 모델 구조:")
    print(f"  - 입력: {model.input_shape}")
    print(f"  - 출력: {model.output_shape}")
    print(f"  - 레이어 수: {len(model.layers)}")
    
    # 3. 완전한 TF2 모델로 재구성
    print("\n🔧 TF2 모델로 재구성...")
    
    # 새로운 모델 생성 (함수형 API 사용)
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    # 모델을 함수처럼 호출
    outputs = model(inputs, training=False)
    
    # 새 모델 생성
    tf2_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # 가중치 복사
    tf2_model.set_weights(model.get_weights())
    
    # 컴파일
    tf2_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 4. 테스트
    print("\n🧪 변환된 모델 테스트...")
    test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
    
    try:
        # __call__ 메서드 사용 (TF2 스타일)
        predictions = tf2_model(test_input, training=False)
        print(f"  ✓ 직접 호출 성공: {predictions.shape}")
        
        # predict 메서드도 테스트
        predictions2 = tf2_model.predict(test_input, verbose=0)
        print(f"  ✓ predict 메서드 성공: {predictions2.shape}")
        
    except Exception as e:
        print(f"  ❌ 테스트 실패: {e}")
        return False
    
    # 5. 저장
    output_path = models_path / "eye_disease_tf2_complete.h5"
    
    # 백업
    if output_path.exists():
        backup_path = output_path.with_suffix('.h5.bak')
        shutil.copy(output_path, backup_path)
    
    # SavedModel 형식으로도 저장
    savedmodel_path = models_path / "eye_disease_tf2_savedmodel"
    
    print(f"\n💾 모델 저장...")
    tf2_model.save(str(output_path), save_format='h5')
    print(f"  ✓ H5 형식: {output_path}")
    
    tf2_model.save(str(savedmodel_path), save_format='tf')
    print(f"  ✓ SavedModel 형식: {savedmodel_path}")
    
    print("\n✅ 변환 완료!")
    return True

def create_tf2_service():
    """TF2 호환 서비스 생성"""
    
    service_content = '''"""
안구질환 진단 서비스 - TensorFlow 2.x 완전 호환 버전
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.config.run_functions_eagerly(True)

import json
import numpy as np
from PIL import Image
from typing import Dict, Tuple
import logging
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class EyeDiseaseService:
    def __init__(self, model_path: str, class_map_path: str):
        """안구 질환 진단 서비스 초기화"""
        
        # Custom objects
        custom_objects = {
            'swish': tf.nn.swish,
            'Swish': tf.keras.layers.Activation(tf.nn.swish)
        }
        
        # 모델 로드 우선순위
        model_loaded = False
        
        # 1. TF2 완전 변환 모델
        tf2_complete_path = model_path.replace('.keras', '_tf2_complete.h5')
        if os.path.exists(tf2_complete_path):
            try:
                logger.info(f"Loading TF2 complete model from {tf2_complete_path}")
                self.model = tf.keras.models.load_model(
                    tf2_complete_path,
                    custom_objects=custom_objects
                )
                model_loaded = True
                logger.info("Successfully loaded TF2 complete model")
            except Exception as e:
                logger.warning(f"Failed to load TF2 complete model: {e}")
        
        # 2. 기존 fixed 모델
        if not model_loaded:
            fixed_path = model_path.replace('.keras', '_fixed.h5')
            if os.path.exists(fixed_path):
                try:
                    logger.info(f"Loading fixed model from {fixed_path}")
                    self.model = tf.keras.models.load_model(
                        fixed_path,
                        custom_objects=custom_objects,
                        compile=False
                    )
                    self.model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    model_loaded = True
                    logger.info("Successfully loaded fixed model")
                except Exception as e:
                    logger.warning(f"Failed to load fixed model: {e}")
        
        if not model_loaded:
            raise ValueError("Could not load any eye disease model")
        
        # 클래스 맵 로드
        with open(class_map_path, 'r', encoding='utf-8') as f:
            self.class_map = json.load(f)
        
        self.input_shape = (224, 224)
        logger.info(f"EyeDiseaseService initialized")
    
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
    
    @tf.function
    def predict_tf2(self, image_array):
        """TF2 스타일 예측 (graph mode에서도 작동)"""
        return self.model(image_array, training=False)
    
    def predict(self, image_array: np.ndarray) -> Tuple[str, float]:
        """예측 수행"""
        # TF2 스타일로 예측
        try:
            # 직접 호출 사용
            predictions = self.predict_tf2(image_array)
            
            # numpy로 변환
            if hasattr(predictions, 'numpy'):
                predictions_np = predictions.numpy()
            else:
                predictions_np = predictions
                
        except Exception as e:
            logger.warning(f"TF2 predict failed, using legacy predict: {e}")
            # 대체 방법
            predictions_np = self.model.predict(image_array, verbose=0)
        
        predicted_class_index = int(np.argmax(predictions_np[0]))
        confidence = float(np.max(predictions_np[0]))
        
        predicted_class_name = self.class_map.get(str(predicted_class_index), "Unknown")
        
        return predicted_class_name, confidence
    
    def diagnose(self, image_file) -> Dict[str, any]:
        """진단 수행"""
        preprocessed_image = self.preprocess_image(image_file)
        disease, confidence = self.predict(preprocessed_image)
        
        return {
            "disease": disease,
            "confidence": confidence
        }
'''
    
    # 서비스 업데이트
    service_path = Path("services/eye_disease_service_tf2.py")
    with open(service_path, 'w', encoding='utf-8') as f:
        f.write(service_content)
    
    print(f"\n✅ TF2 서비스 생성: {service_path}")
    
    # 기존 서비스 교체 가이드
    print("\n📋 서비스 교체 방법:")
    print("  1. 백업: cp services/eye_disease_service.py services/eye_disease_service.original.py")
    print("  2. 교체: cp services/eye_disease_service_tf2.py services/eye_disease_service.py")
    print("  3. 서버 재시작")

def test_converted_model():
    """변환된 모델 테스트"""
    print("\n🧪 변환된 모델 종합 테스트...")
    
    test_code = '''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.config.run_functions_eagerly(True)

import numpy as np
from pathlib import Path

print(f"Test - TF {tf.__version__} - Eager: {tf.executing_eagerly()}")

# 모델 로드
model_path = Path("models/health_diagnosis/eye_disease/eye_disease_tf2_complete.h5")

if model_path.exists():
    print(f"\\n모델 로드: {model_path}")
    
    custom_objects = {'swish': tf.nn.swish}
    model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)
    
    # 테스트 입력
    test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
    
    # 1. 직접 호출
    print("\\n1. 직접 호출 테스트...")
    try:
        output1 = model(test_input, training=False)
        print(f"  ✓ 성공: {output1.shape}")
    except Exception as e:
        print(f"  ❌ 실패: {e}")
    
    # 2. predict 메서드
    print("\\n2. predict 메서드 테스트...")
    try:
        output2 = model.predict(test_input, verbose=0)
        print(f"  ✓ 성공: {output2.shape}")
    except Exception as e:
        print(f"  ❌ 실패: {e}")
    
    # 3. tf.function 래핑
    print("\\n3. tf.function 테스트...")
    @tf.function
    def predict_fn(x):
        return model(x, training=False)
    
    try:
        output3 = predict_fn(test_input)
        print(f"  ✓ 성공: {output3.shape}")
    except Exception as e:
        print(f"  ❌ 실패: {e}")
    
else:
    print(f"❌ 모델 파일 없음: {model_path}")
'''
    
    with open("test_tf2_complete.py", 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    print(f"  ✓ 테스트 스크립트 생성: test_tf2_complete.py")

if __name__ == "__main__":
    print("🚀 안구질환 모델 완전한 TF2 변환")
    print("=" * 60)
    
    # 1. 모델 변환
    if convert_eye_model_to_tf2():
        # 2. 서비스 생성
        create_tf2_service()
        
        # 3. 테스트 스크립트
        test_converted_model()
        
        print("\n✅ 모든 작업 완료!")
        print("\n📋 다음 단계:")
        print("  1. 테스트: python test_tf2_complete.py")
        print("  2. 서비스 교체 후 서버 재시작")
    else:
        print("\n❌ 변환 실패")