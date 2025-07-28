"""
모든 피부질환 체크포인트를 올바른 H5 모델로 변환
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.optimizers import Adam
import json
from pathlib import Path

# 설정 파일 로드
base_path = Path("/mnt/d/final_project/DuoPet_AI/models/health_diagnosis/skin_disease")

# config.yaml 내용을 직접 정의 (yaml 모듈 없이)
config = {
    'models': {
        'classification': {
            'cat_binary': {
                'checkpoint_prefix': 'model-007-0.511353-0.772705-0.776322-0.768861',
                'output_classes': 2,
                'model_type': 'checkpoint'
            },
            'dog_binary': {
                'checkpoint_prefix': 'model-004-0.437360-0.806570-0.806528-0.806891',
                'output_classes': 2,
                'model_type': 'checkpoint'
            },
            'dog_multi_136': {
                'checkpoint_prefix': 'model-009-0.851382-0.821520',
                'output_classes': 3,
                'model_type': 'checkpoint'
            },
            'dog_multi_456': {
                'checkpoint_prefix': 'model-005-0.881675-0.851780',
                'output_classes': 3,
                'model_type': 'checkpoint'
            }
        }
    }
}

def create_unet_model():
    """원본 학습 코드와 동일한 U-Net 구조 생성"""
    input_size = (224, 224, 3)
    inputs = Input(input_size, name="INPUT")
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    # Bottleneck
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    # Decoder
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    
    # Output
    conv10 = Conv2D(2, 1, activation='sigmoid')(conv9)
    
    model = Model(inputs=inputs, outputs=conv10)
    return model

def create_classification_model(output_classes):
    """원본 학습 코드와 동일한 구조로 분류 모델 생성"""
    network = InceptionResNetV2(
        include_top=False, 
        weights='imagenet', 
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    model = Sequential()
    model.add(network)
    model.add(Dense(2048, activation='relu'))
    
    # Binary는 2개 출력 + sigmoid, Multi는 N개 출력 + softmax
    if output_classes == 2:
        model.add(Dense(output_classes, activation='sigmoid'))
    else:
        model.add(Dense(output_classes, activation='softmax'))
    
    return model

def convert_checkpoint_to_h5(model_key, model_info):
    """체크포인트를 H5로 변환"""
    model_type = model_info.get('model_type')
    if model_type != 'checkpoint':
        print(f"Skipping {model_key} - not a checkpoint model")
        return
        
    checkpoint_prefix = model_info['checkpoint_prefix']
    output_classes = model_info['output_classes']
    
    # 경로 설정
    if 'segmentation' in config:
        # Segmentation 모델 처리를 위한 config 업데이트 필요
        pass
    
    # Classification 모델은 classification 폴더에
    model_dir = base_path / "classification" / model_key
        
    checkpoint_path = model_dir / checkpoint_prefix
    
    # H5 저장 경로 (원본 백업)
    h5_path = model_dir / f"{model_key}_model_from_checkpoint.h5"
    
    print(f"\n{'='*50}")
    print(f"Converting: {model_key}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output classes: {output_classes}")
    
    try:
        # 모델 생성
        model = create_classification_model(output_classes)
            
        print(f"Model structure - Input: {model.input_shape}, Output: {model.output_shape}")
        
        # 체크포인트 로드
        model.load_weights(str(checkpoint_path))
        print("✓ Checkpoint loaded successfully!")
        
        # 모델 저장
        model.save(str(h5_path))
        print(f"✓ Saved to: {h5_path}")
        
        # 간단한 검증
        import numpy as np
        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        pred = model.predict(test_input, verbose=0)
        print(f"Test prediction shape: {pred.shape}")
        print(f"Test prediction: {pred[0]}")
        
        # 예측값 범위 확인
        if output_classes == 2:
            print(f"Binary output - Normal: {pred[0][0]:.4f}, Disease: {pred[0][1]:.4f}")
        
    except Exception as e:
        print(f"✗ Error converting {model_key}: {e}")
        import traceback
        traceback.print_exc()

def main():
    """모든 classification 모델 변환"""
    print("Starting checkpoint conversion...")
    
    # Classification 모델들만 변환
    classification_models = {
        k: v for k, v in config['models']['classification'].items()
    }
    
    for model_key, model_info in classification_models.items():
        convert_checkpoint_to_h5(model_key, model_info)
    
    print(f"\n{'='*50}")
    print("Conversion complete!")
    
    # 변환 결과 요약
    print("\n=== Conversion Summary ===")
    for model_key in classification_models:
        model_dir = base_path / "classification" / model_key
        h5_files = list(model_dir.glob("*.h5"))
        print(f"\n{model_key}:")
        for h5_file in h5_files:
            size_mb = h5_file.stat().st_size / (1024 * 1024)
            print(f"  - {h5_file.name}: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()