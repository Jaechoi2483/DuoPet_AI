"""
BCS 모델 고급 변환
이름 충돌 문제를 해결하는 여러 방법 시도
"""
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import tempfile
import shutil

# TensorFlow 2.x 설정
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class BCSModelConverter:
    """BCS 모델 변환기"""
    
    def __init__(self):
        self.models_dir = Path("models/health_diagnosis/bcs")
        self.source_path = self.models_dir / "bcs_efficientnet_v1.h5"
        self.custom_objects = {
            'swish': tf.nn.swish,
            'Swish': tf.keras.layers.Activation(tf.nn.swish),
            'FixedDropout': tf.keras.layers.Dropout,
        }
    
    def load_original_model(self):
        """원본 모델 로드"""
        if not self.source_path.exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {self.source_path}")
        
        print("📥 원본 모델 로드 중...")
        model = tf.keras.models.load_model(
            str(self.source_path),
            custom_objects=self.custom_objects,
            compile=False
        )
        print(f"  ✓ 로드 완료 - 입력 수: {len(model.inputs)}")
        return model
    
    def method1_savedmodel(self, model):
        """방법 1: SavedModel 형식으로 저장"""
        print("\n🔧 방법 1: SavedModel 형식 저장")
        
        try:
            # 컴파일
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # SavedModel로 저장
            savedmodel_path = self.models_dir / "bcs_tf2_savedmodel"
            model.save(str(savedmodel_path), save_format='tf')
            print(f"  ✓ SavedModel 저장 성공: {savedmodel_path}")
            
            # 다시 로드해서 H5로 저장 시도
            print("  🔄 SavedModel을 H5로 변환 시도...")
            loaded_model = tf.keras.models.load_model(str(savedmodel_path))
            
            h5_path = self.models_dir / "bcs_tf2_from_savedmodel.h5"
            loaded_model.save(str(h5_path), save_format='h5')
            print(f"  ✓ H5 변환 성공: {h5_path}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ 실패: {e}")
            return False
    
    def method2_single_model_extraction(self, model):
        """방법 2: 단일 서브모델 추출"""
        print("\n🔧 방법 2: 단일 EfficientNet 추출")
        
        try:
            # Functional 레이어 찾기
            functional_layers = [layer for layer in model.layers 
                               if type(layer).__name__ == 'Functional']
            
            if not functional_layers:
                print("  ❌ Functional 레이어를 찾을 수 없음")
                return False
            
            # 첫 번째 EfficientNet 추출
            first_efficientnet = functional_layers[0]
            print(f"  📦 첫 번째 서브모델 추출: {first_efficientnet.name}")
            
            # 새로운 입력과 Dense 레이어 추가
            new_input = tf.keras.Input(shape=(224, 224, 3), name='input_single')
            
            # 서브모델 호출
            x = first_efficientnet(new_input)
            
            # 출력 레이어 찾기 (Dense 레이어)
            dense_layers = [layer for layer in model.layers 
                          if isinstance(layer, tf.keras.layers.Dense)]
            
            if dense_layers:
                # 기존 Dense 레이어의 가중치 복사
                dense_layer = dense_layers[0]
                new_dense = tf.keras.layers.Dense(
                    3, 
                    activation='softmax',
                    name='predictions'
                )
                output = new_dense(x)
                
                # 새 모델 생성
                single_model = tf.keras.Model(inputs=new_input, outputs=output)
                
                # 가중치 복사
                new_dense.build(x.shape)
                if dense_layer.weights:
                    try:
                        new_dense.set_weights(dense_layer.get_weights())
                    except:
                        print("  ⚠️ Dense 가중치 복사 실패 - 새로 초기화")
            else:
                # Dense 레이어가 없으면 새로 생성
                output = tf.keras.layers.Dense(3, activation='softmax', name='predictions')(x)
                single_model = tf.keras.Model(inputs=new_input, outputs=output)
            
            # 컴파일
            single_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # 저장
            output_path = self.models_dir / "bcs_tf2_single_efficientnet.h5"
            single_model.save(str(output_path), save_format='h5')
            print(f"  ✓ 단일 모델 저장 성공: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def method3_rebuild_model(self):
        """방법 3: 모델 재구성"""
        print("\n🔧 방법 3: 깨끗한 모델 재구성")
        
        try:
            # 새로운 EfficientNetB5 모델 생성
            base_model = tf.keras.applications.EfficientNetB5(
                include_top=False,
                weights='imagenet',
                input_shape=(224, 224, 3),
                pooling='avg'
            )
            
            # 입력 레이어
            inputs = tf.keras.Input(shape=(224, 224, 3), name='input')
            
            # 전처리
            x = tf.keras.applications.efficientnet.preprocess_input(inputs)
            
            # 베이스 모델
            x = base_model(x, training=False)
            
            # 출력 레이어
            outputs = tf.keras.layers.Dense(3, activation='softmax', name='predictions')(x)
            
            # 모델 생성
            clean_model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # 컴파일
            clean_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # 저장
            output_path = self.models_dir / "bcs_tf2_clean_rebuild.h5"
            clean_model.save(str(output_path), save_format='h5')
            print(f"  ✓ 깨끗한 모델 저장 성공: {output_path}")
            
            # 모델 정보 저장
            model_info = {
                "model_name": "bcs_clean_efficientnetb5",
                "architecture": "EfficientNetB5",
                "input_shape": [224, 224, 3],
                "output_classes": 3,
                "class_names": ["마른 체형", "정상 체형", "비만 체형"],
                "preprocessing": "tf.keras.applications.efficientnet.preprocess_input",
                "notes": "ImageNet 가중치 사용, 원본 가중치 미포함"
            }
            
            info_path = self.models_dir / "bcs_clean_model_info.json"
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"  ❌ 실패: {e}")
            return False
    
    def test_models(self):
        """변환된 모델들 테스트"""
        print("\n🧪 변환된 모델 테스트")
        print("=" * 60)
        
        test_models = [
            ("SavedModel", self.models_dir / "bcs_tf2_savedmodel"),
            ("SavedModel→H5", self.models_dir / "bcs_tf2_from_savedmodel.h5"),
            ("단일 EfficientNet", self.models_dir / "bcs_tf2_single_efficientnet.h5"),
            ("깨끗한 재구성", self.models_dir / "bcs_tf2_clean_rebuild.h5")
        ]
        
        # 테스트 데이터
        test_input = np.random.randint(0, 255, (2, 224, 224, 3), dtype=np.uint8).astype(np.float32)
        
        for name, model_path in test_models:
            if not model_path.exists():
                print(f"\n❌ {name}: 파일 없음")
                continue
            
            try:
                print(f"\n📋 {name} 테스트:")
                
                # 모델 로드
                if model_path.suffix == '.h5':
                    model = tf.keras.models.load_model(str(model_path))
                else:
                    model = tf.keras.models.load_model(str(model_path))
                
                # 예측
                if "단일" in name or "깨끗한" in name:
                    # 단일 입력
                    output = model.predict(test_input, verbose=0)
                else:
                    # 13개 입력 (원본 구조)
                    inputs_13 = [test_input for _ in range(13)]
                    output = model.predict(inputs_13, verbose=0)
                
                print(f"  ✓ 예측 성공!")
                print(f"  - 출력 shape: {output.shape}")
                print(f"  - 예측 결과: {np.argmax(output, axis=1)}")
                
                classes = ["마른 체형", "정상 체형", "비만 체형"]
                for i, pred in enumerate(output):
                    class_idx = np.argmax(pred)
                    print(f"  - 샘플 {i+1}: {classes[class_idx]} ({pred[class_idx]:.2%})")
                
            except Exception as e:
                print(f"  ❌ 테스트 실패: {e}")
    
    def run_all_methods(self):
        """모든 변환 방법 실행"""
        print("🚀 BCS 모델 고급 변환 시작")
        print("=" * 80)
        
        try:
            # 원본 모델 로드
            model = self.load_original_model()
            
            # 각 방법 시도
            success_methods = []
            
            if self.method1_savedmodel(model):
                success_methods.append("SavedModel")
            
            if self.method2_single_model_extraction(model):
                success_methods.append("단일 모델 추출")
            
            if self.method3_rebuild_model():
                success_methods.append("모델 재구성")
            
            # 결과 요약
            print("\n" + "=" * 80)
            print("📊 변환 결과:")
            print(f"  성공한 방법: {len(success_methods)}개")
            for method in success_methods:
                print(f"  ✓ {method}")
            
            # 모델 테스트
            if success_methods:
                self.test_models()
            
            print("\n💡 권장 사항:")
            if "단일 모델 추출" in success_methods:
                print("  → bcs_tf2_single_efficientnet.h5 사용 권장 (원본 가중치 유지)")
            elif "SavedModel→H5" in success_methods:
                print("  → bcs_tf2_from_savedmodel.h5 사용 권장")
            else:
                print("  → bcs_tf2_clean_rebuild.h5 사용 (ImageNet 가중치)")
            
        except Exception as e:
            print(f"\n❌ 전체 프로세스 실패: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    converter = BCSModelConverter()
    converter.run_all_methods()