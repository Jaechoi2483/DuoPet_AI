"""
통합 모델 로더
TensorFlow 2.x 호환성을 보장하고 그래프 충돌을 방지하는 통합 모델 로딩 시스템
"""
import os
import gc
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager
import numpy as np
import tensorflow as tf
from datetime import datetime

# TensorFlow 설정
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ModelContainer:
    """개별 모델을 위한 컨테이너"""
    
    def __init__(self, model_name: str, model_path: Path, model_type: str):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.graph = None
        self.session = None
        self.loaded = False
        self.last_used = None
        self.load_count = 0
        self.error_count = 0
        
    def load(self, custom_objects: Dict = None):
        """모델 로드 (격리된 그래프 사용)"""
        try:
            # 새로운 그래프 생성
            self.graph = tf.Graph()
            
            with self.graph.as_default():
                # 모델 로드
                self.model = tf.keras.models.load_model(
                    str(self.model_path),
                    custom_objects=custom_objects or {},
                    compile=False
                )
                
                # 모델 컴파일
                self._compile_model()
                
            self.loaded = True
            self.load_count += 1
            self.last_used = datetime.now()
            
            return True
            
        except Exception as e:
            self.error_count += 1
            print(f"❌ 모델 로드 실패 ({self.model_name}): {e}")
            return False
    
    def _compile_model(self):
        """모델 타입에 따른 컴파일"""
        if self.model_type == "eye_disease":
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        elif self.model_type == "bcs":
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        elif "binary" in self.model_type:
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC()]
            )
        else:  # multi-class
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """예측 수행"""
        if not self.loaded or self.model is None:
            raise RuntimeError(f"모델이 로드되지 않았습니다: {self.model_name}")
        
        with self.graph.as_default():
            predictions = self.model.predict(input_data, verbose=0)
        
        self.last_used = datetime.now()
        return predictions
    
    def unload(self):
        """모델 언로드 및 메모리 정리"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.graph is not None:
            del self.graph
            self.graph = None
        
        # 가비지 컬렉션
        gc.collect()
        tf.keras.backend.clear_session()
        
        self.loaded = False


class UnifiedModelLoader:
    """통합 모델 로더"""
    
    def __init__(self, models_dir: str = None):
        self.models_dir = Path(models_dir or "models/health_diagnosis")
        self.model_containers: Dict[str, ModelContainer] = {}
        self.model_configs = self._load_model_configs()
        self.custom_objects = {
            'swish': tf.nn.swish,
            'Swish': tf.nn.swish,
            'silu': tf.nn.silu,
        }
        self._lock = threading.Lock()
        self.max_loaded_models = 3  # 동시에 로드할 최대 모델 수
        
    def _load_model_configs(self) -> Dict[str, Dict]:
        """모델 설정 로드"""
        configs = {
            "eye_disease": {
                "path": "eye_disease/eye_disease_tf2_complete.h5",
                "fallback_paths": [
                    "eye_disease/eye_disease_fixed.h5",
                    "eye_disease/best_grouped_model_fixed.h5"
                ],
                "type": "eye_disease",
                "input_shape": (224, 224, 3),
                "output_classes": 5,
                "preprocessing": "efficientnet"
            },
            "bcs": {
                "path": "bcs/bcs_tf2_unified.h5",
                "fallback_paths": [
                    "bcs/bcs_efficientnet_v1.h5"
                ],
                "type": "bcs",
                "input_shape": (224, 224, 3),
                "output_classes": 3,
                "preprocessing": "standard"
            },
            "skin_cat_binary": {
                "path": "skin_disease/classification/cat_binary/cat_binary_tf2_unified.h5",
                "fallback_paths": [
                    "skin_disease/classification/cat_binary/cat_binary_model_tf2_perfect.h5",
                    "skin_disease/classification/cat_binary/cat_binary_model.h5"
                ],
                "type": "skin_binary",
                "input_shape": (224, 224, 3),
                "output_classes": 1,
                "preprocessing": "mobilenet"
            },
            "skin_dog_binary": {
                "path": "skin_disease/classification/dog_binary/dog_binary_tf2_unified.h5",
                "fallback_paths": [
                    "skin_disease/classification/dog_binary/dog_binary_model_tf2_perfect.h5",
                    "skin_disease/classification/dog_binary/dog_binary_model.h5"
                ],
                "type": "skin_binary",
                "input_shape": (224, 224, 3),
                "output_classes": 1,
                "preprocessing": "mobilenet"
            },
            "skin_dog_multi_136": {
                "path": "skin_disease/classification/dog_multi_136/dog_multi_136_tf2_unified.h5",
                "fallback_paths": [
                    "skin_disease/classification/dog_multi_136/dog_multi_136_model_tf2_perfect.h5",
                    "skin_disease/classification/dog_multi_136/dog_multi_136_model.h5"
                ],
                "type": "skin_multi",
                "input_shape": (224, 224, 3),
                "output_classes": 3,
                "preprocessing": "mobilenet"
            },
            "skin_dog_multi_456": {
                "path": "skin_disease/classification/dog_multi_456/dog_multi_456_tf2_unified.h5",
                "fallback_paths": [
                    "skin_disease/classification/dog_multi_456/dog_multi_456_model_tf2_perfect.h5",
                    "skin_disease/classification/dog_multi_456/dog_multi_456_model.h5"
                ],
                "type": "skin_multi",
                "input_shape": (224, 224, 3),
                "output_classes": 3,
                "preprocessing": "mobilenet"
            }
        }
        
        return configs
    
    def _find_available_model_path(self, config: Dict) -> Optional[Path]:
        """사용 가능한 모델 경로 찾기"""
        # 메인 경로 확인
        main_path = self.models_dir / config["path"]
        if main_path.exists():
            return main_path
        
        # 폴백 경로들 확인
        for fallback in config.get("fallback_paths", []):
            fallback_path = self.models_dir / fallback
            if fallback_path.exists():
                print(f"  ⚠️ 폴백 경로 사용: {fallback}")
                return fallback_path
        
        return None
    
    def _manage_memory(self):
        """메모리 관리 - 오래된 모델 언로드"""
        with self._lock:
            loaded_models = [
                (name, container) 
                for name, container in self.model_containers.items() 
                if container.loaded
            ]
            
            # 로드된 모델이 최대치를 초과하면
            if len(loaded_models) >= self.max_loaded_models:
                # 가장 오래 사용하지 않은 모델 찾기
                loaded_models.sort(key=lambda x: x[1].last_used or datetime.min)
                
                # 가장 오래된 모델 언로드
                oldest_name, oldest_container = loaded_models[0]
                print(f"  🗑️ 메모리 관리: {oldest_name} 언로드")
                oldest_container.unload()
    
    @contextmanager
    def get_model(self, model_name: str):
        """모델 컨텍스트 매니저"""
        try:
            # 모델 로드 확인
            if model_name not in self.model_containers:
                if not self._load_model(model_name):
                    raise RuntimeError(f"모델 로드 실패: {model_name}")
            
            container = self.model_containers[model_name]
            
            # 모델이 언로드되었다면 다시 로드
            if not container.loaded:
                if not container.load(self.custom_objects):
                    raise RuntimeError(f"모델 재로드 실패: {model_name}")
            
            yield container
            
        except Exception as e:
            print(f"❌ 모델 사용 중 오류 ({model_name}): {e}")
            raise
    
    def _load_model(self, model_name: str) -> bool:
        """모델 로드"""
        with self._lock:
            if model_name not in self.model_configs:
                print(f"❌ 알 수 없는 모델: {model_name}")
                return False
            
            config = self.model_configs[model_name]
            
            # 사용 가능한 모델 경로 찾기
            model_path = self._find_available_model_path(config)
            if not model_path:
                print(f"❌ 모델 파일을 찾을 수 없음: {model_name}")
                return False
            
            # 메모리 관리
            self._manage_memory()
            
            # 모델 컨테이너 생성
            container = ModelContainer(
                model_name=model_name,
                model_path=model_path,
                model_type=config["type"]
            )
            
            # 모델 로드
            if container.load(self.custom_objects):
                self.model_containers[model_name] = container
                print(f"✅ 모델 로드 성공: {model_name}")
                return True
            
            return False
    
    def predict(self, model_name: str, input_data: np.ndarray) -> Dict[str, Any]:
        """예측 수행"""
        try:
            with self.get_model(model_name) as container:
                # 전처리
                processed_input = self._preprocess_input(
                    input_data, 
                    self.model_configs[model_name]["preprocessing"]
                )
                
                # 예측
                predictions = container.predict(processed_input)
                
                # 후처리
                result = self._postprocess_output(
                    predictions,
                    model_name,
                    self.model_configs[model_name]
                )
                
                return {
                    "status": "success",
                    "model": model_name,
                    "predictions": result,
                    "raw_output": predictions.tolist()
                }
                
        except Exception as e:
            return {
                "status": "error",
                "model": model_name,
                "error": str(e)
            }
    
    def _preprocess_input(self, input_data: np.ndarray, preprocessing: str) -> np.ndarray:
        """입력 전처리"""
        if preprocessing == "efficientnet":
            return tf.keras.applications.efficientnet.preprocess_input(input_data)
        elif preprocessing == "mobilenet":
            return tf.keras.applications.mobilenet_v2.preprocess_input(input_data)
        else:  # standard
            return input_data / 255.0
    
    def _postprocess_output(self, predictions: np.ndarray, model_name: str, config: Dict) -> Dict:
        """출력 후처리"""
        if "binary" in model_name:
            # Binary classification
            prob = float(predictions[0][0])
            return {
                "class": "positive" if prob > 0.5 else "negative",
                "confidence": prob if prob > 0.5 else 1 - prob,
                "probabilities": {"negative": 1 - prob, "positive": prob}
            }
        else:
            # Multi-class classification
            class_idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][class_idx])
            
            # 클래스 이름 매핑
            class_names = self._get_class_names(model_name)
            
            return {
                "class": class_names[class_idx] if class_names else str(class_idx),
                "class_index": class_idx,
                "confidence": confidence,
                "probabilities": {
                    class_names[i] if class_names else str(i): float(prob)
                    for i, prob in enumerate(predictions[0])
                }
            }
    
    def _get_class_names(self, model_name: str) -> Optional[list]:
        """모델별 클래스 이름 반환"""
        class_mappings = {
            "eye_disease": ["정상", "백내장", "녹내장", "망막질환", "각막질환"],
            "bcs": ["마른 체형", "정상 체형", "비만 체형"],
            "skin_dog_multi_136": ["구진플라크", "무증상", "농포여드름"],
            "skin_dog_multi_456": ["과다색소침착", "결절종괴", "미란궤양"]
        }
        
        return class_mappings.get(model_name)
    
    def get_status(self) -> Dict:
        """로더 상태 반환"""
        status = {
            "loaded_models": [],
            "available_models": list(self.model_configs.keys()),
            "memory_usage": {},
            "statistics": {
                "total_loads": 0,
                "total_errors": 0
            }
        }
        
        for name, container in self.model_containers.items():
            if container.loaded:
                status["loaded_models"].append({
                    "name": name,
                    "type": container.model_type,
                    "last_used": container.last_used.isoformat() if container.last_used else None,
                    "load_count": container.load_count,
                    "error_count": container.error_count
                })
            
            status["statistics"]["total_loads"] += container.load_count
            status["statistics"]["total_errors"] += container.error_count
        
        return status
    
    def cleanup(self):
        """모든 모델 언로드 및 정리"""
        print("🧹 모든 모델 언로드 중...")
        
        with self._lock:
            for name, container in self.model_containers.items():
                if container.loaded:
                    container.unload()
                    print(f"  ✓ {name} 언로드 완료")
            
            self.model_containers.clear()
        
        # 전체 메모리 정리
        gc.collect()
        tf.keras.backend.clear_session()
        
        print("✅ 정리 완료")


# 싱글톤 인스턴스
_model_loader_instance = None

def get_model_loader() -> UnifiedModelLoader:
    """싱글톤 모델 로더 반환"""
    global _model_loader_instance
    if _model_loader_instance is None:
        _model_loader_instance = UnifiedModelLoader()
    return _model_loader_instance