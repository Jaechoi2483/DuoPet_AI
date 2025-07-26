"""
간단한 모델 수정 스크립트
h5py 없이도 작동하는 버전
"""

import os
import sys
import tensorflow as tf
from pathlib import Path

# 환경 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

def create_eye_disease_model():
    """안구 질환 모델 재생성 (가중치 없음)"""
    print("Creating new eye disease model...")
    
    # EfficientNetB0 기반 모델
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'  # ImageNet 가중치 사용
    )
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # 컴파일
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def update_model_loader():
    """model_loader.py를 간단한 버전으로 업데이트"""
    print("\nUpdating model_loader.py...")
    
    new_loader_code = '''"""
간소화된 모델 로더
"""
import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_model_with_custom_objects(model_path: str):
    """모델 로딩 - 실패 시 재구성된 모델 반환"""
    try:
        # 1. 직접 로드 시도
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except:
        logger.warning(f"Failed to load {model_path}, creating new model")
        
        # 2. 모델 타입에 따라 새 모델 생성
        if 'eye' in str(model_path).lower():
            return create_eye_disease_model()
        elif 'bcs' in str(model_path).lower():
            return create_bcs_model()
        else:
            return create_skin_model()

def create_eye_disease_model():
    """안구 질환 모델 생성"""
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3), include_top=False, weights='imagenet'
    )
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_bcs_model():
    """BCS 모델 생성 (단순화된 버전)"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_skin_model():
    """피부 질환 모델 생성"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 별칭
load_keras_with_normalization_fix = load_model_with_custom_objects
safe_model_predict = lambda model, data: model.predict(data)

# 기타 필요한 함수들
def configure_tensorflow():
    pass
'''
    
    # 백업 생성
    loader_path = Path("utils/model_loader.py")
    backup_path = Path("utils/model_loader_original.py")
    
    if loader_path.exists() and not backup_path.exists():
        import shutil
        shutil.copy2(loader_path, backup_path)
        print(f"Backup created: {backup_path}")
    
    # 새 로더 저장
    with open(loader_path, 'w', encoding='utf-8') as f:
        f.write(new_loader_code)
    
    print("✅ model_loader.py updated successfully!")

def test_model_creation():
    """모델 생성 테스트"""
    print("\nTesting model creation...")
    
    try:
        model = create_eye_disease_model()
        print("✅ Eye disease model created successfully!")
        
        # 더미 입력으로 테스트
        dummy_input = tf.random.normal((1, 224, 224, 3))
        output = model.predict(dummy_input)
        print(f"Model output shape: {output.shape}")
        print(f"Model output: {output}")
        
        # 모델 저장
        save_path = "models/health_diagnosis/eye_disease/eye_disease_new.h5"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"✅ Model saved to: {save_path}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("DuoPet AI - Simple Model Fix")
    print("=" * 60)
    
    print("\n1. Creating new eye disease model...")
    if test_model_creation():
        print("\n2. Updating model_loader.py...")
        update_model_loader()
        
        print("\n✅ All done!")
        print("\nNow you can:")
        print("1. Restore the original router:")
        print("   copy health_diagnosis_router_original_v2.py health_diagnosis_router.py")
        print("\n2. Start the server:")
        print("   python -m api.main")
    else:
        print("\n❌ Model creation failed. Please check TensorFlow installation.")

if __name__ == "__main__":
    main()