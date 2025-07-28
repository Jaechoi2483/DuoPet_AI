"""
BCS 앙상블 모델 래퍼
13개 입력을 유지하면서 사용하기 쉽게 만들기
"""
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import gc

# TensorFlow 2.x 설정
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_ensemble_wrapper():
    """앙상블 구조를 유지하는 래퍼 생성"""
    
    models_dir = Path("models/health_diagnosis/bcs")
    source_path = models_dir / "bcs_efficientnet_v1.h5"
    
    print("🚀 BCS 앙상블 모델 래퍼 생성")
    print("=" * 60)
    
    try:
        # 1. 원본 모델 로드
        print("\n1️⃣ 원본 앙상블 모델 로드 중...")
        print("  ⏳ 큰 모델이므로 시간이 걸릴 수 있습니다...")
        
        custom_objects = {
            'swish': tf.nn.swish,
            'Swish': tf.keras.layers.Activation(tf.nn.swish),
            'FixedDropout': tf.keras.layers.Dropout,
        }
        
        # 메모리 정리
        gc.collect()
        tf.keras.backend.clear_session()
        
        model = tf.keras.models.load_model(
            str(source_path),
            custom_objects=custom_objects,
            compile=False
        )
        
        print(f"  ✓ 로드 완료 - 13개 입력 앙상블")
        print(f"  - 모델 크기: ~885MB")
        print(f"  - 총 파라미터: {model.count_params():,}")
        
        # 2. 컴파일만 다시
        print("\n2️⃣ 모델 재컴파일 중...")
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("  ✓ 컴파일 완료")
        
        # 3. 원본 모델을 그대로 저장 시도
        print("\n3️⃣ 재저장 시도...")
        
        # 옵션 1: 원본 경로에 덮어쓰기 (백업 권장)
        backup_path = models_dir / "bcs_efficientnet_v1_backup.h5"
        if not backup_path.exists():
            print(f"  📦 백업 생성: {backup_path}")
            import shutil
            shutil.copy2(source_path, backup_path)
        
        # 옵션 2: 새로운 이름으로 저장
        output_path = models_dir / "bcs_tf2_ensemble.h5"
        
        try:
            model.save(str(output_path), save_format='h5')
            print(f"  ✓ H5 저장 성공: {output_path}")
        except:
            print("  ⚠️ H5 저장 실패 - SavedModel 시도...")
            
            # SavedModel로 저장
            savedmodel_path = models_dir / "bcs_tf2_ensemble_savedmodel"
            model.save(str(savedmodel_path), save_format='tf')
            print(f"  ✓ SavedModel 저장: {savedmodel_path}")
        
        # 4. 사용하기 쉬운 래퍼 클래스 생성
        print("\n4️⃣ 래퍼 클래스 코드 생성...")
        
        wrapper_code = '''"""
BCS 앙상블 모델 래퍼
13개 입력을 자동으로 처리
"""
import tensorflow as tf
import numpy as np
from pathlib import Path

class BCSEnsembleModel:
    """BCS 앙상블 모델 래퍼"""
    
    def __init__(self, model_path=None):
        self.models_dir = Path("models/health_diagnosis/bcs")
        
        if model_path is None:
            # 기본 경로들 시도
            candidates = [
                self.models_dir / "bcs_tf2_ensemble.h5",
                self.models_dir / "bcs_tf2_ensemble_savedmodel",
                self.models_dir / "bcs_efficientnet_v1.h5"
            ]
            
            for path in candidates:
                if path.exists():
                    model_path = path
                    break
        
        if model_path is None or not Path(model_path).exists():
            raise FileNotFoundError("BCS 모델을 찾을 수 없습니다")
        
        print(f"BCS 모델 로드 중: {model_path}")
        
        custom_objects = {
            'swish': tf.nn.swish,
            'Swish': tf.keras.layers.Activation(tf.nn.swish),
        }
        
        self.model = tf.keras.models.load_model(
            str(model_path),
            custom_objects=custom_objects,
            compile=False
        )
        
        # 컴파일
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.class_names = ["마른 체형", "정상 체형", "비만 체형"]
    
    def preprocess_image(self, image):
        """이미지 전처리"""
        if isinstance(image, str):
            # 파일 경로인 경우
            image = tf.keras.preprocessing.image.load_img(
                image, target_size=(224, 224)
            )
            image = tf.keras.preprocessing.image.img_to_array(image)
        
        # 배치 차원 추가
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # float32 변환
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        return image
    
    def predict(self, image, augment=True):
        """
        예측 수행
        
        Args:
            image: 입력 이미지 (파일 경로 또는 numpy array)
            augment: True면 13개 augmentation 사용, False면 동일 이미지 13개 복사
        
        Returns:
            예측 결과 딕셔너리
        """
        # 전처리
        image = self.preprocess_image(image)
        
        if augment:
            # 13가지 다른 augmentation 적용
            augmented_images = self._augment_image(image)
        else:
            # 동일한 이미지 13개 복사
            augmented_images = [image for _ in range(13)]
        
        # 예측
        predictions = self.model.predict(augmented_images, verbose=0)
        
        # 결과 정리
        class_idx = np.argmax(predictions)
        confidence = predictions[0][class_idx]
        
        return {
            "class": self.class_names[class_idx],
            "class_index": int(class_idx),
            "confidence": float(confidence),
            "probabilities": {
                name: float(prob) 
                for name, prob in zip(self.class_names, predictions[0])
            }
        }
    
    def _augment_image(self, image):
        """13가지 augmentation 생성"""
        augmented = []
        
        # 원본
        augmented.append(image)
        
        # 좌우 반전
        augmented.append(tf.image.flip_left_right(image))
        
        # 회전 (-15, -10, -5, 5, 10, 15도)
        for angle in [-15, -10, -5, 5, 10, 15]:
            rotated = tf.keras.preprocessing.image.apply_affine_transform(
                image[0], theta=angle
            )
            augmented.append(np.expand_dims(rotated, axis=0))
        
        # 밝기 조정 (-0.1, 0.1)
        augmented.append(tf.image.adjust_brightness(image, -0.1))
        augmented.append(tf.image.adjust_brightness(image, 0.1))
        
        # 대비 조정 (0.9, 1.1)
        augmented.append(tf.image.adjust_contrast(image, 0.9))
        augmented.append(tf.image.adjust_contrast(image, 1.1))
        
        # 13개가 안 되면 원본으로 채우기
        while len(augmented) < 13:
            augmented.append(image)
        
        return augmented[:13]

# 사용 예시
if __name__ == "__main__":
    # 모델 로드
    bcs_model = BCSEnsembleModel()
    
    # 예측 (실제 이미지 경로 사용)
    # result = bcs_model.predict("path/to/dog/image.jpg")
    
    # 또는 numpy array
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = bcs_model.predict(dummy_image)
    
    print(f"예측 결과: {result['class']} (신뢰도: {result['confidence']:.2%})")
'''
        
        wrapper_path = models_dir / "bcs_ensemble_wrapper.py"
        with open(wrapper_path, 'w', encoding='utf-8') as f:
            f.write(wrapper_code)
        
        print(f"  ✓ 래퍼 클래스 생성: {wrapper_path}")
        
        # 5. 간단한 테스트
        print("\n5️⃣ 모델 테스트...")
        
        # 13개 동일 입력으로 테스트
        test_input = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8).astype(np.float32)
        test_inputs_13 = [test_input for _ in range(13)]
        
        predictions = model.predict(test_inputs_13, verbose=0)
        
        print(f"  - 출력 shape: {predictions.shape}")
        print(f"  - 예측 확률: {predictions[0]}")
        
        classes = ["마른 체형", "정상 체형", "비만 체형"]
        class_idx = np.argmax(predictions[0])
        print(f"  - 예측 결과: {classes[class_idx]} ({predictions[0][class_idx]:.2%})")
        
        # 6. 사용 가이드
        print("\n" + "=" * 60)
        print("✅ BCS 앙상블 모델 준비 완료!")
        print("\n💡 사용 방법:")
        print("  1. 직접 사용 (13개 입력 필요):")
        print("     ```python")
        print("     model = tf.keras.models.load_model('models/health_diagnosis/bcs/bcs_tf2_ensemble.h5')")
        print("     inputs_13 = [image for _ in range(13)]  # 같은 이미지 13개")
        print("     predictions = model.predict(inputs_13)")
        print("     ```")
        print("\n  2. 래퍼 클래스 사용 (권장):")
        print("     ```python")
        print("     from models.health_diagnosis.bcs.bcs_ensemble_wrapper import BCSEnsembleModel")
        print("     bcs_model = BCSEnsembleModel()")
        print("     result = bcs_model.predict('image.jpg')")
        print("     ```")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    create_ensemble_wrapper()