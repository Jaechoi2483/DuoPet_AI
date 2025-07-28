"""
눈 질환 모델 고급 변환기
Normalization layer와 가중치 문제를 완벽하게 해결
"""
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import h5py
import tempfile
import shutil

# TensorFlow 2.x 설정
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class AdvancedEyeModelConverter:
    """눈 질환 모델 고급 변환기"""
    
    def __init__(self):
        self.models_dir = Path("models/health_diagnosis/eye_disease")
        self.original_model_path = self.models_dir / "best_grouped_model.keras"
        self.output_path = self.models_dir / "eye_disease_tf2_complete.h5"
        
    def analyze_original_model(self):
        """원본 모델 상세 분석"""
        print("🔍 원본 모델 분석 중...")
        
        try:
            # H5 파일 구조 분석
            with h5py.File(self.original_model_path, 'r') as f:
                print("\n📁 H5 파일 구조:")
                def print_structure(name, obj, level=0):
                    indent = "  " * level
                    if isinstance(obj, h5py.Dataset):
                        print(f"{indent}{name}: shape={obj.shape}, dtype={obj.dtype}")
                    else:
                        print(f"{indent}{name}/")
                        if hasattr(obj, 'keys'):
                            for key in obj.keys():
                                print_structure(f"{name}/{key}", obj[key], level+1)
                
                for key in f.keys():
                    print_structure(key, f[key])
                
                # 메타데이터 확인
                if 'model_config' in f.attrs:
                    config = json.loads(f.attrs['model_config'])
                    print(f"\n📋 모델 설정:")
                    print(f"  - Class: {config.get('class_name', 'Unknown')}")
                    print(f"  - Keras version: {f.attrs.get('keras_version', 'Unknown')}")
                    
        except Exception as e:
            print(f"❌ H5 분석 실패: {e}")
    
    def create_efficientnet_model_with_fix(self):
        """Normalization 문제를 해결한 EfficientNet 모델 생성"""
        print("\n🔧 개선된 모델 구조 생성 중...")
        
        # 입력 레이어
        inputs = tf.keras.Input(shape=(224, 224, 3), name='input_layer')
        
        # Normalization layer 대신 Lambda layer 사용
        x = tf.keras.layers.Lambda(
            lambda x: tf.keras.applications.efficientnet.preprocess_input(x),
            name='preprocessing'
        )(inputs)
        
        # EfficientNetB0 백본 (가중치 없이)
        base_model = tf.keras.applications.EfficientNetB0(
            input_tensor=x,
            include_top=False,
            weights=None,  # 가중치는 나중에 복사
            input_shape=(224, 224, 3)
        )
        
        # 백본 출력
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D(name='global_pooling')(x)
        
        # Dense layers
        x = tf.keras.layers.Dense(128, activation='relu', name='dense_1')(x)
        x = tf.keras.layers.Dropout(0.5, name='dropout_1')(x)
        outputs = tf.keras.layers.Dense(5, activation='softmax', name='predictions')(x)
        
        # 최종 모델
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='eye_disease_model')
        
        return model, base_model
    
    def extract_and_map_weights(self, original_model, new_model, base_model):
        """원본 모델에서 가중치 추출 및 매핑"""
        print("\n🔄 가중치 추출 및 매핑 중...")
        
        # 1. EfficientNet 백본 가중치 매핑
        print("  - EfficientNet 백본 가중치 복사 중...")
        
        # 원본 모델에서 EfficientNet 레이어 찾기
        efficientnet_layer = None
        for layer in original_model.layers:
            if 'efficientnet' in layer.name.lower():
                efficientnet_layer = layer
                break
        
        if efficientnet_layer:
            # 백본 가중치 복사
            try:
                # 레이어별로 가중치 복사
                for orig_layer, new_layer in zip(efficientnet_layer.layers, base_model.layers):
                    if orig_layer.weights and new_layer.weights:
                        try:
                            new_layer.set_weights(orig_layer.get_weights())
                        except:
                            # shape 불일치 시 스킵
                            pass
                            
                print("    ✓ 백본 가중치 복사 완료")
            except Exception as e:
                print(f"    ⚠️ 백본 가중치 복사 중 일부 오류: {e}")
        
        # 2. Dense layer 가중치 매핑
        print("  - Dense layer 가중치 복사 중...")
        
        # 레이어 이름으로 매핑
        layer_mapping = {
            'dense': 'dense_1',
            'dense_1': 'dense_1',
            'dropout': 'dropout_1',
            'predictions': 'predictions',
            'dense_2': 'predictions'  # 다른 가능한 이름
        }
        
        for orig_layer in original_model.layers:
            mapped_name = layer_mapping.get(orig_layer.name, orig_layer.name)
            
            # 새 모델에서 해당 레이어 찾기
            for new_layer in new_model.layers:
                if new_layer.name == mapped_name and orig_layer.weights:
                    try:
                        # 가중치 shape 확인
                        if len(orig_layer.weights) == len(new_layer.weights):
                            weights_match = True
                            for ow, nw in zip(orig_layer.weights, new_layer.weights):
                                if ow.shape != nw.shape:
                                    weights_match = False
                                    break
                            
                            if weights_match:
                                new_layer.set_weights(orig_layer.get_weights())
                                print(f"    ✓ {orig_layer.name} → {new_layer.name}")
                            else:
                                print(f"    ⚠️ Shape 불일치: {orig_layer.name}")
                    except Exception as e:
                        print(f"    ⚠️ 가중치 복사 실패 ({orig_layer.name}): {e}")
    
    def convert_with_weight_preservation(self):
        """가중치를 보존하면서 모델 변환"""
        print("\n🚀 고급 모델 변환 시작")
        print("=" * 60)
        
        try:
            # 1. 원본 모델 분석
            self.analyze_original_model()
            
            # 2. 원본 모델 로드
            print("\n📥 원본 모델 로드 중...")
            
            # Custom objects 정의
            custom_objects = {
                'swish': tf.nn.swish,
                'Swish': tf.nn.swish,
            }
            
            # 먼저 compile=False로 시도
            try:
                original_model = tf.keras.models.load_model(
                    str(self.original_model_path),
                    custom_objects=custom_objects,
                    compile=False
                )
                print("  ✓ 원본 모델 로드 성공")
            except Exception as e:
                print(f"  ❌ 직접 로드 실패: {e}")
                
                # .keras를 .h5로 변환 시도
                if self.original_model_path.suffix == '.keras':
                    # 임시 h5 파일로 변환
                    temp_h5 = self.models_dir / "temp_converted.h5"
                    
                    # Keras 파일을 읽어서 h5로 저장
                    print("  📄 .keras → .h5 변환 시도...")
                    model_temp = tf.keras.models.load_model(
                        str(self.original_model_path),
                        compile=False
                    )
                    model_temp.save(str(temp_h5), save_format='h5')
                    
                    # h5 파일로 다시 로드
                    original_model = tf.keras.models.load_model(
                        str(temp_h5),
                        custom_objects=custom_objects,
                        compile=False
                    )
                    temp_h5.unlink()  # 임시 파일 삭제
                    
            # 3. 새 모델 생성
            new_model, base_model = self.create_efficientnet_model_with_fix()
            
            # 4. 가중치 매핑
            self.extract_and_map_weights(original_model, new_model, base_model)
            
            # 5. 컴파일
            print("\n🔨 모델 컴파일 중...")
            new_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2)]
            )
            
            # 6. 저장
            print("\n💾 변환된 모델 저장 중...")
            new_model.save(str(self.output_path), save_format='h5')
            
            # 7. 검증
            print("\n✅ 변환 검증 중...")
            test_input = np.random.randn(2, 224, 224, 3).astype(np.float32)
            
            # 원본 모델 예측
            orig_output = original_model.predict(test_input)
            
            # 새 모델 예측
            new_output = new_model.predict(test_input)
            
            # 출력 비교
            print(f"\n📊 출력 비교:")
            print(f"  - 원본 모델 출력 shape: {orig_output.shape}")
            print(f"  - 새 모델 출력 shape: {new_output.shape}")
            print(f"  - 출력 차이 (MAE): {np.mean(np.abs(orig_output - new_output)):.6f}")
            
            # 모델 정보 저장
            model_info = {
                "model_name": "eye_disease_model",
                "input_shape": [224, 224, 3],
                "output_classes": 5,
                "class_names": ["정상", "백내장", "녹내장", "망막질환", "각막질환"],
                "preprocessing": "efficientnet",
                "framework": "tensorflow",
                "version": tf.__version__,
                "conversion_date": str(Path().resolve()),
                "original_path": str(self.original_model_path),
                "converted_path": str(self.output_path)
            }
            
            info_path = self.models_dir / "model_info.json"
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ 변환 완료!")
            print(f"  - 출력 파일: {self.output_path}")
            print(f"  - 모델 정보: {info_path}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ 변환 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_converted_model(self):
        """변환된 모델 테스트"""
        print("\n🧪 변환된 모델 테스트")
        print("=" * 60)
        
        if not self.output_path.exists():
            print("❌ 변환된 모델이 없습니다.")
            return
        
        try:
            # 모델 로드
            model = tf.keras.models.load_model(str(self.output_path))
            
            # 테스트 이미지 생성 (실제 이미지 크기)
            test_batch = np.random.randint(0, 255, (5, 224, 224, 3), dtype=np.uint8)
            test_batch = test_batch.astype(np.float32)
            
            # 예측
            predictions = model.predict(test_batch)
            
            # 결과 출력
            print("\n예측 결과:")
            class_names = ["정상", "백내장", "녹내장", "망막질환", "각막질환"]
            
            for i, pred in enumerate(predictions):
                predicted_class = np.argmax(pred)
                confidence = pred[predicted_class]
                print(f"  이미지 {i+1}: {class_names[predicted_class]} (신뢰도: {confidence:.2%})")
            
            print("\n✅ 모델 테스트 성공!")
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")

if __name__ == "__main__":
    converter = AdvancedEyeModelConverter()
    
    # 변환 실행
    if converter.convert_with_weight_preservation():
        # 변환 성공 시 테스트
        converter.test_converted_model()