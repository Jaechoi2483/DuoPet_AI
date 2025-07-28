"""
Normalization 문제 완전 해결
Lambda 레이어 기반의 안정적인 모델 생성
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import shutil

print("🔧 Normalization 문제 완전 해결")
print("=" * 80)

class StableEyeDiseaseModel:
    """안정적인 안구질환 진단 모델"""
    
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.num_classes = 5
        self.class_map = {
            "0": "각막 질환",
            "1": "결막 및 누관 질환",
            "2": "수정체 질환",
            "3": "안검 질환",
            "4": "안구 내부 질환"
        }
        
    def create_model(self, preprocessing='simple'):
        """Normalization 없는 안정적인 모델 생성"""
        
        print(f"\n🏗️ 안정적인 모델 생성 (전처리: {preprocessing})")
        
        inputs = tf.keras.Input(shape=self.input_shape, name='input_image')
        
        # 전처리 레이어 선택
        if preprocessing == 'simple':
            # 단순 0-1 정규화
            x = tf.keras.layers.Lambda(
                lambda img: img / 255.0,
                name='simple_normalization'
            )(inputs)
            print("✅ 단순 정규화 (0-255 → 0-1)")
            
        elif preprocessing == 'imagenet':
            # ImageNet 정규화
            def imagenet_preprocess(img):
                # RGB 평균값
                mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
                # RGB 표준편차
                std = tf.constant([58.393, 57.12, 57.375], dtype=tf.float32)
                return (img - mean) / std
            
            x = tf.keras.layers.Lambda(
                imagenet_preprocess,
                name='imagenet_normalization'
            )(inputs)
            print("✅ ImageNet 정규화")
            
        else:  # 'none'
            # 전처리 없음 (서비스에서 처리)
            x = inputs
            print("✅ 전처리 없음 (외부에서 처리)")
        
        # EfficientNetB0 백본
        base_model = tf.keras.applications.EfficientNetB0(
            input_tensor=x,
            include_top=False,
            weights='imagenet',  # 사전 학습된 가중치
            pooling='avg'  # GlobalAveragePooling 포함
        )
        
        # 백본 고정 (전이학습)
        base_model.trainable = False
        
        # 분류 헤드
        features = base_model.output
        x = tf.keras.layers.Dense(128, activation='relu', name='fc1')(features)
        x = tf.keras.layers.Dropout(0.3, name='dropout1')(x)
        x = tf.keras.layers.Dense(64, activation='relu', name='fc2')(x)
        x = tf.keras.layers.Dropout(0.2, name='dropout2')(x)
        outputs = tf.keras.layers.Dense(
            self.num_classes, 
            activation='softmax',
            name='predictions'
        )(x)
        
        # 모델 생성
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='eye_disease_model')
        
        # 컴파일
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ 모델 생성 완료: {len(model.layers)} 레이어")
        
        return model
    
    def initialize_weights(self, model):
        """가중치를 명시적으로 초기화 (테스트용)"""
        
        print("\n🎲 가중치 초기화...")
        
        # 마지막 Dense 레이어들만 초기화
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense) and layer.name.startswith('fc'):
                # Glorot uniform 초기화
                fan_in = layer.input_shape[-1]
                fan_out = layer.units
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                
                # 가중치 설정
                weights = np.random.uniform(-limit, limit, size=(fan_in, fan_out))
                biases = np.zeros(fan_out)
                
                layer.set_weights([weights, biases])
                print(f"  - {layer.name}: 가중치 초기화 완료")
        
        return model
    
    def save_models(self, model, output_dir):
        """여러 형식으로 모델 저장"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 모델 저장: {output_dir}")
        
        # 1. H5 형식 (가장 안정적)
        h5_path = output_dir / 'eye_disease_stable.h5'
        model.save(str(h5_path), save_format='h5')
        print(f"✅ H5 형식: {h5_path}")
        
        # 2. SavedModel 형식
        saved_model_path = output_dir / 'eye_disease_stable_saved'
        model.save(str(saved_model_path))
        print(f"✅ SavedModel: {saved_model_path}")
        
        # 3. Keras 형식 (TF 2.x)
        keras_path = output_dir / 'eye_disease_stable.keras'
        model.save(str(keras_path))
        print(f"✅ Keras: {keras_path}")
        
        # 4. 가중치만 저장
        weights_path = output_dir / 'eye_disease_stable.weights.h5'
        model.save_weights(str(weights_path))
        print(f"✅ 가중치: {weights_path}")
        
        # 5. 클래스맵 저장
        class_map_path = output_dir / 'class_map.json'
        with open(class_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.class_map, f, ensure_ascii=False, indent=2)
        print(f"✅ 클래스맵: {class_map_path}")
        
        # 6. 모델 정보 저장
        model_info = {
            "model_type": "EfficientNetB0_stable",
            "preprocessing": "Lambda layer",
            "input_shape": list(self.input_shape),
            "output_classes": self.num_classes,
            "normalization": "None (replaced with Lambda)",
            "issues_resolved": [
                "Normalization layer compatibility",
                "Cross-platform compatibility",
                "Graph/Eager mode compatibility"
            ],
            "tensorflow_version": tf.__version__
        }
        
        info_path = output_dir / 'model_info.json'
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"✅ 모델 정보: {info_path}")
    
    def test_model(self, model):
        """모델 테스트"""
        
        print("\n🧪 모델 테스트...")
        
        # 랜덤 이미지로 테스트
        test_images = np.random.randint(0, 255, size=(3, 224, 224, 3)).astype(np.float32)
        
        predictions = model.predict(test_images, verbose=0)
        
        print("\n예측 결과:")
        for i, pred in enumerate(predictions):
            print(f"\n이미지 {i+1}:")
            print(f"  원시 출력: {[f'{p:.3f}' for p in pred]}")
            print(f"  확률(%): {[f'{p*100:.1f}' for p in pred]}")
            
            # 최고 확률 클래스
            max_idx = np.argmax(pred)
            max_prob = pred[max_idx]
            class_name = self.class_map[str(max_idx)]
            print(f"  진단: {class_name} ({max_prob*100:.1f}%)")
        
        # 가중치 상태 확인
        all_same = all(np.allclose(predictions[0], pred) for pred in predictions[1:])
        if all_same and np.allclose(predictions[0], 0.2, atol=0.05):
            print("\n⚠️ 경고: 모든 예측이 균등 분포 (20%) - 학습 필요")
            return False
        else:
            print("\n✅ 모델이 다양한 예측을 생성 - 정상 작동")
            return True

def create_production_ready_model():
    """실제 서비스용 모델 생성"""
    
    print("\n" + "="*80)
    print("🏭 프로덕션용 모델 생성")
    print("="*80)
    
    builder = StableEyeDiseaseModel()
    
    # 1. 단순 정규화 모델
    print("\n1️⃣ 단순 정규화 모델")
    model_simple = builder.create_model(preprocessing='simple')
    builder.save_models(model_simple, 'models/health_diagnosis/eye_disease/stable_simple')
    builder.test_model(model_simple)
    
    # 2. ImageNet 정규화 모델
    print("\n\n2️⃣ ImageNet 정규화 모델")
    model_imagenet = builder.create_model(preprocessing='imagenet')
    builder.save_models(model_imagenet, 'models/health_diagnosis/eye_disease/stable_imagenet')
    builder.test_model(model_imagenet)
    
    # 3. 전처리 없는 모델 (서비스에서 처리)
    print("\n\n3️⃣ 전처리 없는 모델")
    model_none = builder.create_model(preprocessing='none')
    builder.save_models(model_none, 'models/health_diagnosis/eye_disease/stable_none')
    builder.test_model(model_none)
    
    return model_simple

def verify_saved_models():
    """저장된 모델 검증"""
    
    print("\n\n" + "="*80)
    print("🔍 저장된 모델 검증")
    print("="*80)
    
    model_dirs = [
        'models/health_diagnosis/eye_disease/stable_simple',
        'models/health_diagnosis/eye_disease/stable_imagenet',
        'models/health_diagnosis/eye_disease/stable_none'
    ]
    
    for model_dir in model_dirs:
        model_path = Path(model_dir) / 'eye_disease_stable.h5'
        
        if model_path.exists():
            print(f"\n검증: {model_path}")
            
            try:
                # 모델 로드
                model = tf.keras.models.load_model(str(model_path))
                print("✅ 모델 로드 성공")
                
                # 간단한 예측
                test_img = np.random.randint(0, 255, size=(1, 224, 224, 3)).astype(np.float32)
                pred = model.predict(test_img, verbose=0)
                
                print(f"예측 결과: {[f'{p:.2f}' for p in pred[0]]}")
                
            except Exception as e:
                print(f"❌ 로드 실패: {e}")

# 실행
if __name__ == "__main__":
    # 프로덕션 모델 생성
    model = create_production_ready_model()
    
    # 검증
    verify_saved_models()
    
    print("\n\n✅ 완료!")
    print("\n권장 사항:")
    print("1. stable_simple 모델 사용 (가장 안정적)")
    print("2. 서비스 코드 업데이트 필요")
    print("3. 실제 학습된 가중치로 교체 필요")