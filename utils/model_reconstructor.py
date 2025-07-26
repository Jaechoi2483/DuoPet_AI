"""
모델 재구성 도구
모델 아키텍처를 재생성하고 가중치만 복사하는 방법으로 호환성 문제 해결
"""

import tensorflow as tf
import numpy as np
import json
import h5py
from pathlib import Path
import zipfile
import tempfile
from typing import Dict, Any, Optional

class ModelReconstructor:
    """모델 재구성을 통한 호환성 문제 해결"""
    
    @staticmethod
    def reconstruct_eye_disease_model(weights_path: Optional[str] = None) -> tf.keras.Model:
        """안구 질환 모델 재구성"""
        print("Reconstructing eye disease model architecture...")
        
        # EfficientNetB0 기반 모델 구조
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'  # ImageNet 사전학습 가중치 사용
        )
        
        # 모델 구성
        inputs = tf.keras.Input(shape=(224, 224, 3))
        
        # 전처리 레이어 (Normalization 대신 Lambda 사용)
        x = tf.keras.layers.Lambda(
            lambda x: tf.keras.applications.efficientnet.preprocess_input(x)
        )(inputs)
        
        # Base model
        x = base_model(x, training=False)
        
        # Top layers
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Output layer (5 classes)
        outputs = tf.keras.layers.Dense(5, activation='softmax', name='predictions')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # 가중치 로드 시도
        if weights_path:
            try:
                print(f"Loading weights from: {weights_path}")
                ModelReconstructor._load_weights_from_keras_file(model, weights_path)
            except Exception as e:
                print(f"Warning: Could not load all weights: {e}")
        
        # 컴파일
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model reconstruction complete!")
        return model
    
    @staticmethod
    def reconstruct_bcs_model(weights_path: Optional[str] = None) -> tf.keras.Model:
        """BCS 모델 재구성 (Multi-head EfficientNet)"""
        print("Reconstructing BCS model architecture...")
        
        # 13개 입력을 위한 multi-head 구조
        inputs = []
        features = []
        
        # 공유 base model
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # 13개 뷰에 대한 입력 생성
        for i in range(13):
            inp = tf.keras.Input(shape=(224, 224, 3), name=f'input_{i}')
            inputs.append(inp)
            
            # 전처리
            x = tf.keras.layers.Lambda(
                lambda x: tf.keras.applications.efficientnet.preprocess_input(x)
            )(inp)
            
            # Feature extraction
            feat = base_model(x, training=False)
            feat = tf.keras.layers.GlobalAveragePooling2D()(feat)
            features.append(feat)
        
        # 모든 특징 연결
        combined = tf.keras.layers.Concatenate()(features)
        
        # Dense layers
        x = tf.keras.layers.BatchNormalization()(combined)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Output (3 classes: 저체중, 정상, 과체중)
        outputs = tf.keras.layers.Dense(3, activation='softmax', name='bcs_output')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # 가중치 로드
        if weights_path:
            try:
                if weights_path.endswith('.h5'):
                    # H5 파일에서 직접 로드
                    model.load_weights(weights_path, by_name=True, skip_mismatch=True)
                else:
                    ModelReconstructor._load_weights_from_keras_file(model, weights_path)
            except Exception as e:
                print(f"Warning: Could not load all weights: {e}")
        
        # 컴파일
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("BCS model reconstruction complete!")
        return model
    
    @staticmethod
    def reconstruct_skin_disease_model(model_type: str, num_classes: int = 2) -> tf.keras.Model:
        """피부 질환 모델 재구성"""
        print(f"Reconstructing skin disease model: {model_type}")
        
        # 기본 CNN 구조
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            
            # 전처리
            tf.keras.layers.Lambda(lambda x: x / 255.0),
            
            # Convolutional blocks
            tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            
            tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            
            tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            
            tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            
            # Dense layers
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            
            # Output
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # 컴파일
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    @staticmethod
    def _load_weights_from_keras_file(model: tf.keras.Model, keras_path: str):
        """Keras 파일에서 가중치만 추출하여 로드"""
        print("Extracting weights from .keras file...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # .keras 파일 압축 해제
            with zipfile.ZipFile(keras_path, 'r') as zip_ref:
                zip_ref.extractall(temp_path)
            
            # variables.h5에서 가중치 로드
            variables_path = temp_path / "variables.h5"
            if variables_path.exists():
                # 가중치 매핑 시도
                with h5py.File(variables_path, 'r') as f:
                    # 레이어별로 가중치 매핑
                    loaded_count = 0
                    for layer in model.layers:
                        layer_name = layer.name
                        
                        # 가능한 가중치 이름들 시도
                        possible_names = [
                            layer_name,
                            f"{layer_name}/kernel:0",
                            f"{layer_name}/bias:0",
                            f"{layer_name}/gamma:0",
                            f"{layer_name}/beta:0",
                            f"{layer_name}/moving_mean:0",
                            f"{layer_name}/moving_variance:0"
                        ]
                        
                        weights_to_load = []
                        for weight_name in possible_names:
                            if weight_name in f:
                                weights_to_load.append(f[weight_name][...])
                        
                        if weights_to_load and hasattr(layer, 'set_weights'):
                            try:
                                # 현재 레이어의 가중치 shape과 일치하는지 확인
                                current_weights = layer.get_weights()
                                if len(weights_to_load) == len(current_weights):
                                    # Shape 확인
                                    shapes_match = all(
                                        w1.shape == w2.shape 
                                        for w1, w2 in zip(weights_to_load, current_weights)
                                    )
                                    if shapes_match:
                                        layer.set_weights(weights_to_load)
                                        loaded_count += 1
                            except Exception as e:
                                print(f"  Warning: Could not load weights for {layer_name}: {e}")
                    
                    print(f"Successfully loaded weights for {loaded_count} layers")
    
    @staticmethod
    def create_weight_compatible_model(original_model_path: str, model_type: str) -> Optional[tf.keras.Model]:
        """원본 모델과 호환되는 새 모델 생성"""
        print(f"\n=== Creating weight-compatible model for {model_type} ===")
        
        if model_type == "eye_disease":
            return ModelReconstructor.reconstruct_eye_disease_model(original_model_path)
        elif model_type == "bcs":
            return ModelReconstructor.reconstruct_bcs_model(original_model_path)
        elif model_type.startswith("skin_"):
            # 피부 질환 모델 타입에 따라 클래스 수 결정
            if "binary" in model_type:
                num_classes = 2
            elif "136" in model_type or "456" in model_type:
                num_classes = 3
            else:
                num_classes = 2  # 기본값
            
            model = ModelReconstructor.reconstruct_skin_disease_model(model_type, num_classes)
            
            # 체크포인트에서 가중치 로드
            if original_model_path:
                try:
                    model.load_weights(original_model_path)
                    print("Checkpoint weights loaded successfully")
                except Exception as e:
                    print(f"Warning: Could not load checkpoint weights: {e}")
            
            return model
        else:
            print(f"Unknown model type: {model_type}")
            return None


def test_reconstruction():
    """재구성 테스트"""
    print("Testing model reconstruction...")
    
    # 1. 안구 질환 모델 테스트
    print("\n" + "="*60)
    print("Testing Eye Disease Model Reconstruction")
    print("="*60)
    
    eye_model = ModelReconstructor.reconstruct_eye_disease_model()
    print(f"Model created: {eye_model}")
    print(f"Input shape: {eye_model.input_shape}")
    print(f"Output shape: {eye_model.output_shape}")
    
    # 더미 입력으로 테스트
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    output = eye_model.predict(dummy_input)
    print(f"Test prediction shape: {output.shape}")
    print(f"Test prediction: {output}")
    
    # 2. BCS 모델 테스트
    print("\n" + "="*60)
    print("Testing BCS Model Reconstruction")
    print("="*60)
    
    bcs_model = ModelReconstructor.reconstruct_bcs_model()
    print(f"Model created: {bcs_model}")
    print(f"Number of inputs: {len(bcs_model.inputs)}")
    print(f"Output shape: {bcs_model.output_shape}")
    
    # 3. 피부 질환 모델 테스트
    print("\n" + "="*60)
    print("Testing Skin Disease Model Reconstruction")
    print("="*60)
    
    skin_model = ModelReconstructor.reconstruct_skin_disease_model("skin_binary", 2)
    print(f"Model created: {skin_model}")
    print(f"Model summary:")
    skin_model.summary()


if __name__ == "__main__":
    test_reconstruction()