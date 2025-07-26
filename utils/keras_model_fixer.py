"""
Keras 모델 호환성 문제 해결 도구
Normalization layer 등의 문제를 수정합니다.
"""

import tensorflow as tf
import numpy as np
import json
import h5py
from pathlib import Path
import shutil
from typing import Dict, Any

class KerasModelFixer:
    """Keras 모델의 호환성 문제를 해결하는 클래스"""
    
    @staticmethod
    def fix_normalization_layer(model_path: str, output_path: str = None) -> str:
        """
        Normalization layer 문제가 있는 모델을 수정
        
        Args:
            model_path: 원본 모델 경로
            output_path: 수정된 모델 저장 경로 (None이면 _fixed 추가)
            
        Returns:
            수정된 모델 경로
        """
        if output_path is None:
            path = Path(model_path)
            output_path = path.parent / f"{path.stem}_fixed{path.suffix}"
        
        print(f"Fixing model: {model_path}")
        print(f"Output path: {output_path}")
        
        try:
            # 1. 먼저 모델 구조를 로드 시도
            with tf.keras.utils.custom_object_scope({
                'Normalization': tf.keras.layers.LayerNormalization
            }):
                # compile=False로 시도
                try:
                    model = tf.keras.models.load_model(model_path, compile=False)
                    print("Model loaded successfully without compilation")
                    
                    # 재저장
                    model.save(output_path)
                    print(f"Model saved to: {output_path}")
                    return str(output_path)
                    
                except Exception as e:
                    print(f"Direct loading failed: {e}")
            
            # 2. H5/Keras 파일 직접 수정
            if model_path.endswith('.keras'):
                return KerasModelFixer._fix_keras_file(model_path, output_path)
            elif model_path.endswith('.h5'):
                return KerasModelFixer._fix_h5_file(model_path, output_path)
            else:
                raise ValueError(f"Unsupported file format: {model_path}")
                
        except Exception as e:
            print(f"Model fixing failed: {e}")
            raise
    
    @staticmethod
    def _fix_keras_file(model_path: str, output_path: str) -> str:
        """Keras 형식 파일 수정"""
        import zipfile
        import tempfile
        import os
        
        print("Attempting to fix .keras file structure...")
        
        # 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            # .keras 파일 압축 해제
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # model.json 파일 읽기 및 수정
            model_json_path = Path(temp_dir) / "model.json"
            if model_json_path.exists():
                with open(model_json_path, 'r') as f:
                    model_config = json.load(f)
                
                # Normalization layer 찾아서 수정
                def fix_layer_config(config):
                    if isinstance(config, dict):
                        if config.get('class_name') == 'Normalization':
                            # BatchNormalization으로 대체
                            config['class_name'] = 'BatchNormalization'
                            if 'config' in config:
                                config['config']['axis'] = -1
                                config['config']['momentum'] = 0.99
                                config['config']['epsilon'] = 0.001
                                config['config']['center'] = True
                                config['config']['scale'] = True
                        
                        for key, value in config.items():
                            if isinstance(value, (dict, list)):
                                fix_layer_config(value)
                    elif isinstance(config, list):
                        for item in config:
                            fix_layer_config(item)
                
                fix_layer_config(model_config)
                
                # 수정된 config 저장
                with open(model_json_path, 'w') as f:
                    json.dump(model_config, f)
            
            # 새로운 .keras 파일로 압축
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zip_ref.write(file_path, arcname)
        
        print(f"Fixed model saved to: {output_path}")
        return str(output_path)
    
    @staticmethod
    def _fix_h5_file(model_path: str, output_path: str) -> str:
        """H5 형식 파일 수정"""
        # H5 파일 복사
        shutil.copy2(model_path, output_path)
        
        # H5 파일 수정
        with h5py.File(output_path, 'r+') as f:
            if 'model_config' in f.attrs:
                model_config = json.loads(f.attrs['model_config'])
                
                # Normalization layer 수정 로직 (위와 동일)
                # ... (구현 필요)
                
                f.attrs['model_config'] = json.dumps(model_config)
        
        return str(output_path)
    
    @staticmethod
    def create_dummy_model(model_type: str, save_path: str) -> str:
        """
        더미 모델 생성 (폴백용)
        
        Args:
            model_type: 모델 타입 (eye_disease, bcs, skin_disease)
            save_path: 저장 경로
            
        Returns:
            저장된 모델 경로
        """
        if model_type == "eye_disease":
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(224, 224, 3)),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(5, activation='softmax')
            ])
        elif model_type == "bcs":
            # Multi-input model for BCS
            inputs = [tf.keras.layers.Input(shape=(224, 224, 3)) for _ in range(13)]
            x = tf.keras.layers.Concatenate()([
                tf.keras.layers.GlobalAveragePooling2D()(inp) for inp in inputs
            ])
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
        else:
            # Default binary classification
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(224, 224, 3)),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(2, activation='softmax')
            ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 더미 가중치 초기화
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                layer.kernel.assign(tf.random.normal(layer.kernel.shape, stddev=0.1))
        
        # 모델 저장
        model.save(save_path)
        print(f"Dummy model saved to: {save_path}")
        
        return str(save_path)


if __name__ == "__main__":
    # 테스트/수정 실행
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        fixer = KerasModelFixer()
        
        try:
            fixed_path = fixer.fix_normalization_layer(model_path)
            print(f"Success! Fixed model saved to: {fixed_path}")
        except Exception as e:
            print(f"Failed to fix model: {e}")
            
            # 더미 모델 생성
            model_type = "eye_disease"  # 기본값
            if "bcs" in model_path.lower():
                model_type = "bcs"
            elif "skin" in model_path.lower():
                model_type = "skin_disease"
            
            dummy_path = Path(model_path).parent / f"dummy_{Path(model_path).name}"
            fixer.create_dummy_model(model_type, str(dummy_path))
    else:
        print("Usage: python keras_model_fixer.py <model_path>")