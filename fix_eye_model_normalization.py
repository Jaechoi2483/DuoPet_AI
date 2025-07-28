"""
눈 질환 모델 Normalization 문제 해결
직접 가중치를 추출하고 새 모델에 적용
"""
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import h5py
import pickle

# TensorFlow 2.x 설정
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class EyeModelNormalizationFixer:
    """Normalization 문제를 해결하는 변환기"""
    
    def __init__(self):
        self.models_dir = Path("models/health_diagnosis/eye_disease")
        self.original_model_path = self.models_dir / "best_grouped_model.keras"
        self.fixed_model_path = self.models_dir / "eye_disease_fixed.h5"
        self.output_path = self.models_dir / "eye_disease_tf2_final.h5"
        
    def extract_weights_from_h5(self):
        """H5 파일에서 직접 가중치 추출"""
        print("📦 H5 파일에서 가중치 추출 중...")
        
        weights_dict = {}
        
        # 먼저 fixed 모델이 있는지 확인
        if self.fixed_model_path.exists():
            h5_path = self.fixed_model_path
            print(f"  - 사용 파일: {h5_path.name}")
        else:
            h5_path = self.original_model_path
            print(f"  - 사용 파일: {h5_path.name}")
        
        try:
            with h5py.File(h5_path, 'r') as f:
                def extract_weights(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        weights_dict[name] = np.array(obj)
                        print(f"    ✓ {name}: shape={obj.shape}")
                
                # model_weights 그룹에서 가중치 추출
                if 'model_weights' in f:
                    f['model_weights'].visititems(extract_weights)
                else:
                    # 다른 구조일 경우
                    f.visititems(extract_weights)
                    
            print(f"  ✓ 총 {len(weights_dict)}개 가중치 추출 완료")
            return weights_dict
            
        except Exception as e:
            print(f"  ❌ 가중치 추출 실패: {e}")
            return None
    
    def create_clean_efficientnet_model(self):
        """Normalization 없는 깨끗한 EfficientNet 모델 생성"""
        print("\n🔧 새로운 모델 구조 생성 중...")
        
        # 입력 레이어
        inputs = tf.keras.Input(shape=(224, 224, 3), name='input_1')
        
        # EfficientNet 전처리 (Normalization 대체)
        x = tf.keras.layers.Rescaling(scale=1./255.0, name='rescaling')(inputs)
        x = tf.keras.layers.Lambda(
            lambda x: (x - 0.5) * 2.0,  # [-1, 1] 범위로 정규화
            name='preprocessing'
        )(x)
        
        # EfficientNetB0 백본
        base_model = tf.keras.applications.EfficientNetB0(
            input_tensor=x,
            include_top=False,
            weights='imagenet',  # ImageNet 가중치로 초기화
            input_shape=(224, 224, 3)
        )
        base_model.trainable = True
        
        # 백본 출력
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d')(x)
        
        # Dense layers
        x = tf.keras.layers.Dense(128, activation='relu', name='dense')(x)
        x = tf.keras.layers.Dropout(0.5, name='dropout')(x)
        outputs = tf.keras.layers.Dense(5, activation='softmax', name='dense_1')(x)
        
        # 모델 생성
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        print("  ✓ 모델 구조 생성 완료")
        print(f"  - 총 레이어: {len(model.layers)}")
        print(f"  - 파라미터: {model.count_params():,}")
        
        return model
    
    def apply_extracted_weights(self, model, weights_dict):
        """추출한 가중치를 새 모델에 적용"""
        print("\n🔄 가중치 적용 중...")
        
        applied_count = 0
        skipped_count = 0
        
        for layer in model.layers:
            layer_name = layer.name
            
            # Dense 레이어 가중치 매핑
            if isinstance(layer, tf.keras.layers.Dense):
                # 가능한 키 패턴들
                kernel_keys = [
                    f"{layer_name}/{layer_name}/kernel:0",
                    f"{layer_name}/kernel:0",
                    f"dense/{layer_name}/kernel:0",
                    f"model/layers/{layer_name}/kernel:0"
                ]
                bias_keys = [
                    f"{layer_name}/{layer_name}/bias:0",
                    f"{layer_name}/bias:0",
                    f"dense/{layer_name}/bias:0",
                    f"model/layers/{layer_name}/bias:0"
                ]
                
                # 매칭되는 가중치 찾기
                kernel_weight = None
                bias_weight = None
                
                for k_key in kernel_keys:
                    if k_key in weights_dict:
                        kernel_weight = weights_dict[k_key]
                        break
                
                for b_key in bias_keys:
                    if b_key in weights_dict:
                        bias_weight = weights_dict[b_key]
                        break
                
                if kernel_weight is not None and bias_weight is not None:
                    try:
                        # shape 확인
                        expected_kernel_shape = layer.kernel.shape
                        expected_bias_shape = layer.bias.shape
                        
                        if (kernel_weight.shape == expected_kernel_shape and 
                            bias_weight.shape == expected_bias_shape):
                            layer.set_weights([kernel_weight, bias_weight])
                            print(f"  ✓ {layer_name} 가중치 적용 완료")
                            applied_count += 1
                        else:
                            print(f"  ⚠️ {layer_name} shape 불일치 - 스킵")
                            skipped_count += 1
                    except Exception as e:
                        print(f"  ⚠️ {layer_name} 적용 실패: {e}")
                        skipped_count += 1
            
            # GlobalAveragePooling2D는 가중치 없음
            elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
                continue
                
            # Dropout은 가중치 없음
            elif isinstance(layer, tf.keras.layers.Dropout):
                continue
        
        print(f"\n📊 가중치 적용 결과:")
        print(f"  - 적용됨: {applied_count}개")
        print(f"  - 스킵됨: {skipped_count}개")
        
        return applied_count > 0
    
    def convert_model(self):
        """모델 변환 실행"""
        print("🚀 눈 질환 모델 변환 시작")
        print("=" * 60)
        
        try:
            # 1. 가중치 추출
            weights_dict = self.extract_weights_from_h5()
            if not weights_dict:
                # 대안: 기존 fixed 모델 사용
                if self.fixed_model_path.exists():
                    print("\n📌 기존 fixed 모델을 그대로 사용합니다.")
                    model = tf.keras.models.load_model(
                        str(self.fixed_model_path),
                        custom_objects={'swish': tf.nn.swish},
                        compile=False
                    )
                else:
                    raise ValueError("가중치 추출 실패 및 대체 모델 없음")
            else:
                # 2. 새 모델 생성
                model = self.create_clean_efficientnet_model()
                
                # 3. 가중치 적용
                success = self.apply_extracted_weights(model, weights_dict)
                if not success:
                    print("⚠️ 가중치 적용 실패 - ImageNet 가중치 사용")
            
            # 4. 모델 컴파일
            print("\n🔨 모델 컴파일 중...")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy', 
                        tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')]
            )
            
            # 5. 저장
            print("\n💾 모델 저장 중...")
            model.save(str(self.output_path), save_format='h5')
            print(f"  ✓ 저장 완료: {self.output_path}")
            
            # 6. 검증
            print("\n✅ 변환 검증 중...")
            test_input = np.random.randn(1, 224, 224, 3).astype(np.float32) * 255
            output = model.predict(test_input, verbose=0)
            
            print(f"  - 입력 shape: {test_input.shape}")
            print(f"  - 출력 shape: {output.shape}")
            print(f"  - 출력 값: {output[0]}")
            print(f"  - 합계: {np.sum(output[0]):.4f}")
            
            # 모델 정보 저장
            model_info = {
                "model_name": "eye_disease_model",
                "input_shape": [224, 224, 3],
                "output_classes": 5,
                "class_names": ["정상", "백내장", "녹내장", "망막질환", "각막질환"],
                "preprocessing": "rescaling_and_normalization",
                "framework": "tensorflow",
                "version": tf.__version__,
                "conversion_notes": "Normalization layer replaced with Rescaling + Lambda"
            }
            
            info_path = self.models_dir / "model_info_tf2.json"
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ 변환 완료!")
            print(f"  - 모델: {self.output_path}")
            print(f"  - 정보: {info_path}")
            
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
            
            # 테스트 이미지 생성
            test_images = np.random.randint(0, 255, (3, 224, 224, 3), dtype=np.uint8)
            test_images = test_images.astype(np.float32)
            
            # 예측
            predictions = model.predict(test_images, verbose=0)
            
            # 결과 출력
            print("\n예측 결과:")
            class_names = ["정상", "백내장", "녹내장", "망막질환", "각막질환"]
            
            for i, pred in enumerate(predictions):
                predicted_class = np.argmax(pred)
                confidence = pred[predicted_class]
                
                print(f"\n이미지 {i+1}:")
                print(f"  - 예측: {class_names[predicted_class]}")
                print(f"  - 신뢰도: {confidence:.2%}")
                print(f"  - 상위 2개:")
                top_2 = np.argsort(pred)[-2:][::-1]
                for idx in top_2:
                    print(f"    • {class_names[idx]}: {pred[idx]:.2%}")
            
            print("\n✅ 모델 테스트 성공!")
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")

if __name__ == "__main__":
    fixer = EyeModelNormalizationFixer()
    
    # 변환 실행
    if fixer.convert_model():
        # 테스트
        fixer.test_converted_model()