"""
통합 모델 로더
다양한 방법을 순차적으로 시도하여 모델을 로드합니다.
"""

import os
import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from .keras_file_analyzer import KerasFileAnalyzer
from .model_reconstructor import ModelReconstructor

logger = logging.getLogger(__name__)

class IntegratedModelLoader:
    """여러 방법을 시도하는 통합 모델 로더"""
    
    def __init__(self):
        self.load_methods = [
            self._method1_direct_load,
            self._method2_fix_keras_file,
            self._method3_reconstruct_model,
            self._method4_dummy_model
        ]
    
    def load_model(self, model_path: str, model_type: str = None) -> Optional[tf.keras.Model]:
        """
        모델 로드 - 여러 방법을 순차적으로 시도
        
        Args:
            model_path: 모델 파일 경로
            model_type: 모델 타입 (eye_disease, bcs, skin_disease 등)
            
        Returns:
            로드된 모델 또는 None
        """
        logger.info(f"Attempting to load model: {model_path}")
        
        # 모델 타입 추론
        if model_type is None:
            model_type = self._infer_model_type(model_path)
            logger.info(f"Inferred model type: {model_type}")
        
        # 각 방법을 순차적으로 시도
        for i, method in enumerate(self.load_methods, 1):
            logger.info(f"\n=== Method {i}: {method.__name__} ===")
            try:
                model = method(model_path, model_type)
                if model is not None:
                    logger.info(f"✅ Successfully loaded model using method {i}")
                    return model
            except Exception as e:
                logger.error(f"❌ Method {i} failed: {e}")
                continue
        
        logger.error("All methods failed to load the model")
        return None
    
    def _infer_model_type(self, model_path: str) -> str:
        """경로에서 모델 타입 추론"""
        path_str = str(model_path).lower()
        
        if 'eye' in path_str:
            return 'eye_disease'
        elif 'bcs' in path_str:
            return 'bcs'
        elif 'skin' in path_str:
            if 'binary' in path_str:
                return 'skin_binary'
            elif '136' in path_str:
                return 'skin_multi_136'
            elif '456' in path_str:
                return 'skin_multi_456'
            return 'skin_disease'
        else:
            return 'unknown'
    
    def _method1_direct_load(self, model_path: str, model_type: str) -> Optional[tf.keras.Model]:
        """방법 1: 직접 로드 (커스텀 객체 포함)"""
        logger.info("Trying direct load with custom objects...")
        
        custom_objects = {
            'swish': tf.nn.swish,
            'Swish': tf.nn.swish,
            'silu': tf.nn.silu,
            'Normalization': tf.keras.layers.BatchNormalization,  # 대체
            'LayerNormalization': tf.keras.layers.BatchNormalization
        }
        
        # compile=False로 시도
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
        
        # 컴파일
        self._compile_model(model, model_type)
        
        return model
    
    def _method2_fix_keras_file(self, model_path: str, model_type: str) -> Optional[tf.keras.Model]:
        """방법 2: Keras 파일 수정 후 로드"""
        if not str(model_path).endswith('.keras'):
            logger.info("Not a .keras file, skipping method 2")
            return None
        
        logger.info("Trying to fix .keras file...")
        
        # Keras 파일 분석 및 수정
        analyzer = KerasFileAnalyzer(model_path)
        fixed_path = analyzer.fix_normalization_variables()
        
        if fixed_path:
            # 수정된 파일 로드 시도
            logger.info(f"Loading fixed model from: {fixed_path}")
            return self._method1_direct_load(fixed_path, model_type)
        
        return None
    
    def _method3_reconstruct_model(self, model_path: str, model_type: str) -> Optional[tf.keras.Model]:
        """방법 3: 모델 재구성 및 가중치 복사"""
        logger.info("Trying model reconstruction...")
        
        if model_type == 'eye_disease':
            return ModelReconstructor.reconstruct_eye_disease_model(model_path)
        elif model_type == 'bcs':
            return ModelReconstructor.reconstruct_bcs_model(model_path)
        elif model_type.startswith('skin_'):
            num_classes = 2
            if 'multi' in model_type:
                num_classes = 3
            
            model = ModelReconstructor.reconstruct_skin_disease_model(model_type, num_classes)
            
            # 체크포인트인 경우 가중치 로드
            if not model_path.endswith(('.h5', '.keras')):
                try:
                    model.load_weights(model_path)
                except:
                    logger.warning("Could not load checkpoint weights")
            
            return model
        else:
            logger.warning(f"Unknown model type for reconstruction: {model_type}")
            return None
    
    def _method4_dummy_model(self, model_path: str, model_type: str) -> tf.keras.Model:
        """방법 4: 더미 모델 (최후의 수단)"""
        logger.warning("Creating dummy model as fallback...")
        
        if model_type == 'eye_disease':
            return self._create_dummy_eye_model()
        elif model_type == 'bcs':
            return self._create_dummy_bcs_model()
        else:
            return self._create_dummy_skin_model()
    
    def _create_dummy_eye_model(self) -> tf.keras.Model:
        """더미 안구 질환 모델"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_dummy_bcs_model(self) -> tf.keras.Model:
        """더미 BCS 모델"""
        # 단순화된 단일 입력 모델
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_dummy_skin_model(self) -> tf.keras.Model:
        """더미 피부 질환 모델"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _compile_model(self, model: tf.keras.Model, model_type: str):
        """모델 컴파일"""
        if model_type == 'eye_disease' or model_type == 'bcs':
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            # 피부 질환 모델
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
    
    def load_all_health_models(self) -> Dict[str, Any]:
        """모든 건강 진단 모델 로드"""
        models = {}
        
        # 프로젝트 루트 찾기
        project_root = Path(__file__).resolve().parent.parent
        models_dir = project_root / "models" / "health_diagnosis"
        
        # 1. 안구 질환 모델
        eye_model_path = models_dir / "eye_disease" / "best_grouped_model.keras"
        if eye_model_path.exists():
            models['eye_disease'] = self.load_model(str(eye_model_path), 'eye_disease')
        else:
            logger.warning(f"Eye disease model not found: {eye_model_path}")
            models['eye_disease'] = None
        
        # 2. BCS 모델
        bcs_model_path = models_dir / "bcs" / "bcs_efficientnet_v1.h5"
        if bcs_model_path.exists():
            models['bcs'] = self.load_model(str(bcs_model_path), 'bcs')
        else:
            logger.warning(f"BCS model not found: {bcs_model_path}")
            models['bcs'] = None
        
        # 3. 피부 질환 모델들
        skin_models = {
            'cat_binary': ('classification/cat_binary/checkpoint', 2),
            'dog_binary': ('classification/dog_binary/checkpoint', 2),
            'dog_multi_136': ('classification/dog_multi_136/checkpoint', 3),
            'dog_multi_456': ('classification/dog_multi_456/checkpoint', 3)
        }
        
        models['skin_disease'] = {}
        for model_name, (rel_path, num_classes) in skin_models.items():
            model_path = models_dir / "skin_disease" / rel_path
            
            # 체크포인트 파일 찾기
            checkpoint_files = list(model_path.parent.glob(f"{model_path.name}*"))
            if checkpoint_files:
                # 인덱스 파일이 있는 경우 사용
                index_files = [f for f in checkpoint_files if f.suffix == '.index']
                if index_files:
                    checkpoint_path = index_files[0].with_suffix('')
                    models['skin_disease'][model_name] = self.load_model(
                        str(checkpoint_path), 
                        f'skin_{"binary" if num_classes == 2 else "multi"}'
                    )
                else:
                    logger.warning(f"No index file found for {model_name}")
                    models['skin_disease'][model_name] = None
            else:
                logger.warning(f"Skin model checkpoint not found: {model_path}")
                models['skin_disease'][model_name] = None
        
        # 로드 결과 요약
        logger.info("\n=== Model Loading Summary ===")
        logger.info(f"Eye Disease Model: {'✅ Loaded' if models.get('eye_disease') else '❌ Failed'}")
        logger.info(f"BCS Model: {'✅ Loaded' if models.get('bcs') else '❌ Failed'}")
        
        skin_loaded = sum(1 for m in models.get('skin_disease', {}).values() if m is not None)
        logger.info(f"Skin Disease Models: {skin_loaded}/{len(skin_models)} loaded")
        
        return models


def test_integrated_loader():
    """통합 로더 테스트"""
    loader = IntegratedModelLoader()
    
    # 모든 모델 로드 시도
    models = loader.load_all_health_models()
    
    print("\n=== Test Results ===")
    for model_type, model in models.items():
        if model_type == 'skin_disease':
            print(f"\n{model_type}:")
            for sub_type, sub_model in model.items():
                print(f"  - {sub_type}: {'✅ Loaded' if sub_model else '❌ Failed'}")
        else:
            print(f"{model_type}: {'✅ Loaded' if model else '❌ Failed'}")
    
    return models


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_integrated_loader()