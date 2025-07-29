"""
간단한 체크포인트 변환 - CustomScaleLayer 없이
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionResNetV2
import numpy as np
from pathlib import Path
import json

def create_simple_model(output_classes):
    """InceptionResNetV2 기반 간단한 모델 생성"""
    # Input
    inputs = Input(shape=(224, 224, 3))
    
    # InceptionResNetV2 백본 (imagenet 가중치)
    base_model = InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )
    
    # Feature extraction
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu', name='dense_features')(x)
    
    # Output layer
    if output_classes == 2:
        outputs = Dense(output_classes, activation='sigmoid', name='predictions')(x)
    else:
        outputs = Dense(output_classes, activation='softmax', name='predictions')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def convert_checkpoint(model_key, checkpoint_info, base_path):
    """체크포인트를 간단한 모델로 변환"""
    print(f"\n{'='*70}")
    print(f"Converting {model_key}")
    print(f"{'='*70}")
    
    model_dir = base_path / "classification" / model_key
    checkpoint_path = model_dir / checkpoint_info['checkpoint_prefix']
    
    # 체크포인트 파일 확인
    if not Path(f"{checkpoint_path}.index").exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        # 1. 간단한 모델 생성
        output_classes = checkpoint_info['output_classes']
        print(f"Creating simple model with {output_classes} output classes...")
        model = create_simple_model(output_classes)
        print(f"Model structure: Input {model.input_shape} -> Output {model.output_shape}")
        
        # 2. 체크포인트에서 가중치 로드
        print(f"Loading weights from checkpoint...")
        # 가중치만 로드 (optimizer state 무시)
        checkpoint = tf.train.Checkpoint(model=model)
        status = checkpoint.restore(str(checkpoint_path))
        status.expect_partial()
        print("✓ Weights loaded successfully")
        
        # 3. 빠른 테스트
        print("Quick test...")
        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        pred = model.predict(test_input, verbose=0)
        print(f"  Test output shape: {pred.shape}")
        print(f"  Test prediction: {pred[0][:5]}...")
        
        # 4. 모델 저장
        save_path = model_dir / f"{model_key}_model_simple.h5"
        print(f"\nSaving to: {save_path.name}")
        model.save(str(save_path))
        
        # 파일 크기 확인
        size_mb = save_path.stat().st_size / (1024 * 1024)
        print(f"✓ Saved successfully ({size_mb:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    base_path = Path.cwd() / "models" / "health_diagnosis" / "skin_disease"
    
    checkpoints = {
        'cat_binary': {
            'checkpoint_prefix': 'model-007-0.511353-0.772705-0.776322-0.768861',
            'output_classes': 2
        },
        'dog_binary': {
            'checkpoint_prefix': 'model-004-0.437360-0.806570-0.806528-0.806891',
            'output_classes': 2
        },
        'dog_multi_136': {
            'checkpoint_prefix': 'model-009-0.851382-0.821520',
            'output_classes': 7
        }
    }
    
    print("="*70)
    print("SIMPLE CHECKPOINT CONVERTER")
    print("="*70)
    print("Converting without CustomScaleLayer")
    
    success_count = 0
    for model_key, checkpoint_info in checkpoints.items():
        if convert_checkpoint(model_key, checkpoint_info, base_path):
            success_count += 1
    
    # 요약
    print(f"\n{'='*70}")
    print("CONVERSION SUMMARY")
    print(f"{'='*70}")
    print(f"Successfully converted: {success_count}/{len(checkpoints)} models")
    
    if success_count > 0:
        print("\n✅ Next steps:")
        print("1. Test with: python test_fixed_skin_models.py")
        print("2. Update skin_disease_service.py to use _simple.h5 models")
        print("\nThese models don't have CustomScaleLayer!")

if __name__ == "__main__":
    main()