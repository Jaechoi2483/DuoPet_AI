"""
통합 TensorFlow 2.x 모델 변환기
모든 DuoPet AI 모델을 TF 2.x 호환 형식으로 변환
"""
import os
import json
import tensorflow as tf
import numpy as np
from pathlib import Path
import h5py
from typing import Dict, Any, Optional, Tuple
import shutil

# TensorFlow 2.x 설정
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class UnifiedModelConverter:
    """통합 모델 변환기"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.models_dir = self.project_root / "models" / "health_diagnosis"
        self.backup_dir = self.project_root / "models" / "backup"
        self.conversion_log = []
        
    def backup_original_models(self):
        """원본 모델 백업"""
        print("📦 원본 모델 백업 중...")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 백업할 모델 목록
        models_to_backup = [
            "eye_disease/best_grouped_model.keras",
            "eye_disease/best_grouped_model_fixed.h5",
            "eye_disease/eye_disease_fixed.h5",
            "bcs/bcs_efficientnet_v1.h5",
            "skin_disease/classification/cat_binary/cat_binary_model.h5",
            "skin_disease/classification/dog_binary/dog_binary_model.h5",
            "skin_disease/classification/dog_multi_136/dog_multi_136_model.h5",
            "skin_disease/classification/dog_multi_456/dog_multi_456_model.h5"
        ]
        
        for model_path in models_to_backup:
            src = self.models_dir / model_path
            if src.exists():
                dst = self.backup_dir / model_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(src, dst)
                    print(f"  ✓ 백업: {model_path}")
    
    def analyze_model_structure(self, model_path: Path) -> Dict[str, Any]:
        """모델 구조 분석"""
        info = {
            "path": str(model_path),
            "exists": model_path.exists(),
            "size_mb": 0,
            "format": "unknown",
            "layers": [],
            "input_shape": None,
            "output_shape": None,
            "total_params": 0,
            "issues": []
        }
        
        if not model_path.exists():
            info["issues"].append("File not found")
            return info
        
        info["size_mb"] = model_path.stat().st_size / (1024 * 1024)
        
        try:
            # H5 파일 분석
            with h5py.File(model_path, 'r') as f:
                info["format"] = "h5"
                
                # 구조 확인
                if 'model_config' in f.attrs:
                    config = json.loads(f.attrs['model_config'])
                    info["keras_version"] = f.attrs.get('keras_version', 'Unknown').decode() if hasattr(f.attrs.get('keras_version', 'Unknown'), 'decode') else str(f.attrs.get('keras_version', 'Unknown'))
                    
                # 가중치 구조 확인
                if 'model_weights' in f:
                    def count_weights(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            info["total_params"] += obj.size
                    f['model_weights'].visititems(count_weights)
                    
        except Exception as e:
            info["issues"].append(f"H5 analysis error: {str(e)}")
        
        # 모델 로딩 시도
        try:
            model = tf.keras.models.load_model(str(model_path), compile=False)
            info["input_shape"] = model.input_shape
            info["output_shape"] = model.output_shape
            info["layers"] = [(l.name, l.__class__.__name__) for l in model.layers[:10]]
            
            # Normalization layer 확인
            norm_layers = [l for l in model.layers if 'normalization' in l.name.lower()]
            if norm_layers:
                info["issues"].append(f"Contains {len(norm_layers)} normalization layers")
                
        except Exception as e:
            info["issues"].append(f"Model loading error: {str(e)}")
            
        return info
    
    def convert_eye_disease_model(self) -> bool:
        """눈 질환 모델 변환"""
        print("\n👁️ 눈 질환 모델 변환 시작...")
        
        # 이미 변환된 모델 사용
        src_path = self.models_dir / "eye_disease" / "eye_disease_fixed.h5"
        dst_path = self.models_dir / "eye_disease" / "eye_disease_tf2_unified.h5"
        
        if not src_path.exists():
            print("  ❌ 소스 모델이 없습니다.")
            return False
            
        try:
            # 모델 로드
            model = tf.keras.models.load_model(str(src_path), compile=False)
            
            # 재컴파일 (TF 2.x 옵티마이저 사용)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # 저장
            model.save(str(dst_path), save_format='h5')
            
            # 검증
            test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
            output = model.predict(test_input)
            
            print(f"  ✓ 변환 완료: {dst_path.name}")
            print(f"  ✓ 출력 shape: {output.shape}")
            
            self.conversion_log.append({
                "model": "eye_disease",
                "status": "success",
                "output_path": str(dst_path)
            })
            
            return True
            
        except Exception as e:
            print(f"  ❌ 변환 실패: {e}")
            self.conversion_log.append({
                "model": "eye_disease",
                "status": "failed",
                "error": str(e)
            })
            return False
    
    def convert_bcs_model(self) -> bool:
        """BCS 모델 변환"""
        print("\n🐕 BCS 모델 변환 시작...")
        
        src_path = self.models_dir / "bcs" / "bcs_efficientnet_v1.h5"
        dst_path = self.models_dir / "bcs" / "bcs_tf2_unified.h5"
        
        if not src_path.exists():
            print("  ❌ 소스 모델이 없습니다.")
            return False
            
        try:
            # Custom objects 정의
            custom_objects = {
                'swish': tf.nn.swish,
                'Swish': tf.keras.layers.Activation(tf.nn.swish)
            }
            
            # 모델 로드
            model = tf.keras.models.load_model(
                str(src_path), 
                custom_objects=custom_objects,
                compile=False
            )
            
            # 재컴파일
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # 저장
            model.save(str(dst_path), save_format='h5')
            
            # 검증
            test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
            output = model.predict(test_input)
            
            print(f"  ✓ 변환 완료: {dst_path.name}")
            print(f"  ✓ 출력 shape: {output.shape}")
            
            self.conversion_log.append({
                "model": "bcs",
                "status": "success",
                "output_path": str(dst_path)
            })
            
            return True
            
        except Exception as e:
            print(f"  ❌ 변환 실패: {e}")
            self.conversion_log.append({
                "model": "bcs",
                "status": "failed",
                "error": str(e)
            })
            return False
    
    def convert_skin_disease_model(self, model_type: str) -> bool:
        """피부 질환 모델 변환"""
        print(f"\n🐾 피부 질환 모델 변환: {model_type}")
        
        # 이미 변환된 perfect 버전 사용
        src_path = self.models_dir / "skin_disease" / "classification" / model_type / f"{model_type}_model_tf2_perfect.h5"
        dst_path = self.models_dir / "skin_disease" / "classification" / model_type / f"{model_type}_tf2_unified.h5"
        
        if not src_path.exists():
            # 원본 모델로 시도
            src_path = self.models_dir / "skin_disease" / "classification" / model_type / f"{model_type}_model.h5"
            
        if not src_path.exists():
            print(f"  ❌ 소스 모델이 없습니다: {src_path}")
            return False
            
        try:
            # 모델 로드
            model = tf.keras.models.load_model(str(src_path), compile=False)
            
            # 출력 레이어에 따른 loss 함수 결정
            output_shape = model.output_shape[-1]
            if output_shape == 1:
                loss = 'binary_crossentropy'
                metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
            else:
                loss = 'categorical_crossentropy'
                metrics = ['accuracy']
            
            # 재컴파일
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss=loss,
                metrics=metrics
            )
            
            # 저장
            model.save(str(dst_path), save_format='h5')
            
            # 검증
            test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
            output = model.predict(test_input)
            
            print(f"  ✓ 변환 완료: {dst_path.name}")
            print(f"  ✓ 출력 shape: {output.shape}")
            
            self.conversion_log.append({
                "model": f"skin_{model_type}",
                "status": "success",
                "output_path": str(dst_path)
            })
            
            return True
            
        except Exception as e:
            print(f"  ❌ 변환 실패: {e}")
            self.conversion_log.append({
                "model": f"skin_{model_type}",
                "status": "failed",
                "error": str(e)
            })
            return False
    
    def run_full_conversion(self):
        """전체 모델 변환 실행"""
        print("🚀 통합 TensorFlow 2.x 모델 변환 시작")
        print("=" * 60)
        
        # 1. 백업
        self.backup_original_models()
        
        # 2. 현재 모델 상태 분석
        print("\n📊 현재 모델 상태 분석...")
        models_to_analyze = [
            "eye_disease/eye_disease_fixed.h5",
            "bcs/bcs_efficientnet_v1.h5",
            "skin_disease/classification/cat_binary/cat_binary_model.h5",
            "skin_disease/classification/dog_binary/dog_binary_model.h5",
            "skin_disease/classification/dog_multi_136/dog_multi_136_model.h5",
            "skin_disease/classification/dog_multi_456/dog_multi_456_model.h5"
        ]
        
        for model_path in models_to_analyze:
            full_path = self.models_dir / model_path
            info = self.analyze_model_structure(full_path)
            print(f"\n  {model_path}:")
            print(f"    - Size: {info['size_mb']:.1f} MB")
            print(f"    - Format: {info['format']}")
            if info['issues']:
                print(f"    - Issues: {', '.join(info['issues'])}")
        
        # 3. 모델별 변환
        print("\n🔧 모델 변환 시작...")
        
        # 눈 질환 모델
        self.convert_eye_disease_model()
        
        # BCS 모델
        self.convert_bcs_model()
        
        # 피부 질환 모델들
        skin_models = ["cat_binary", "dog_binary", "dog_multi_136", "dog_multi_456"]
        for model_type in skin_models:
            self.convert_skin_disease_model(model_type)
        
        # 4. 변환 결과 요약
        print("\n📋 변환 결과 요약")
        print("=" * 60)
        
        success_count = sum(1 for log in self.conversion_log if log['status'] == 'success')
        failed_count = sum(1 for log in self.conversion_log if log['status'] == 'failed')
        
        print(f"✅ 성공: {success_count}개")
        print(f"❌ 실패: {failed_count}개")
        
        # 변환 로그 저장
        log_path = self.project_root / "conversion_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.conversion_log, f, indent=2, ensure_ascii=False)
        
        print(f"\n📝 변환 로그 저장: {log_path}")
        
        return success_count > 0

if __name__ == "__main__":
    converter = UnifiedModelConverter()
    converter.run_full_conversion()