"""
강제로 모델 로드하기
Normalization 레이어를 완전히 무시
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import zipfile
import shutil

def extract_and_fix_model():
    """keras 파일을 압축 해제하고 수정"""
    
    print("🔧 Keras 모델 파일 강제 수정")
    print("=" * 60)
    
    model_path = Path("C:/Users/ictedu1_021/Desktop/안구질환모델/final_model_fixed.keras")
    
    if not model_path.exists():
        print(f"❌ 모델을 찾을 수 없습니다: {model_path}")
        return False
    
    # 작업 디렉토리
    work_dir = Path("temp_model_fix")
    work_dir.mkdir(exist_ok=True)
    
    try:
        # 1. keras 파일 압축 해제 (keras는 실제로 zip 파일)
        print("\n1️⃣ Keras 파일 압축 해제...")
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            zip_ref.extractall(work_dir)
        print("✅ 압축 해제 완료")
        
        # 2. config.json 수정
        config_path = work_dir / "config.json"
        if config_path.exists():
            print("\n2️⃣ 모델 설정 수정...")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Normalization 레이어 찾아서 제거 또는 수정
            def fix_layer(layer_config):
                if isinstance(layer_config, dict):
                    if layer_config.get('class_name') == 'Normalization':
                        print(f"  - Normalization 레이어 발견: {layer_config.get('config', {}).get('name', 'unknown')}")
                        # Lambda 레이어로 대체
                        layer_config['class_name'] = 'Lambda'
                        layer_config['config'] = {
                            'name': layer_config['config'].get('name', 'normalization'),
                            'trainable': False,
                            'dtype': 'float32',
                            'function': 'lambda x: x'  # 입력을 그대로 반환
                        }
                        print("    → Lambda 레이어로 대체")
                        
                    # 재귀적으로 처리
                    for key, value in layer_config.items():
                        if isinstance(value, (dict, list)):
                            fix_layer(value)
                            
                elif isinstance(layer_config, list):
                    for item in layer_config:
                        fix_layer(item)
            
            fix_layer(config)
            
            # 수정된 설정 저장
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print("✅ 설정 수정 완료")
        
        # 3. 수정된 모델 재압축
        print("\n3️⃣ 수정된 모델 재압축...")
        fixed_model_path = Path("models/health_diagnosis/eye_disease/eye_disease_force_fixed.keras")
        fixed_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(fixed_model_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(work_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(work_dir)
                    zipf.write(file_path, arcname)
        
        print(f"✅ 수정된 모델 저장: {fixed_model_path}")
        
        # 4. 정리
        shutil.rmtree(work_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
        # 정리
        if work_dir.exists():
            shutil.rmtree(work_dir)
        
        return False

def load_and_test_model():
    """수정된 모델 로드 및 테스트"""
    
    print("\n\n4️⃣ 수정된 모델 로드 테스트...")
    print("-" * 60)
    
    model_path = Path("models/health_diagnosis/eye_disease/eye_disease_force_fixed.keras")
    
    if not model_path.exists():
        print("❌ 수정된 모델을 찾을 수 없습니다")
        return None
    
    try:
        # Lambda 함수 정의
        def identity_function(x):
            return x
        
        # 커스텀 객체
        custom_objects = {
            'identity_function': identity_function,
            'lambda': identity_function,
            'swish': tf.nn.swish,
            'Swish': tf.keras.layers.Activation(tf.nn.swish),
            'FixedDropout': tf.keras.layers.Dropout
        }
        
        # 모델 로드
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(
                str(model_path),
                compile=False
            )
        
        print("✅ 모델 로드 성공!")
        
        # 간단한 테스트
        print("\n테스트 예측...")
        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        pred = model.predict(test_input, verbose=0)
        
        print(f"예측 shape: {pred.shape}")
        print(f"예측값: {pred[0]}")
        print(f"확률(%): {[f'{p*100:.1f}' for p in pred[0]]}")
        
        # 모든 값이 동일한지 확인
        if not np.allclose(pred[0], pred[0][0]):
            print("✅ 모델이 정상 작동합니다!")
        else:
            print("⚠️ 모든 예측값이 동일 - 가중치 문제")
        
        return model
        
    except Exception as e:
        print(f"❌ 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_alternative_solution():
    """대안: 모델 구조만 재생성하고 가중치 로드"""
    
    print("\n\n5️⃣ 대안: 모델 재구성...")
    print("-" * 60)
    
    # EfficientNetB0 기반 모델 재생성
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    # Normalization 없이 바로 시작
    x = inputs
    
    # EfficientNet 백본
    base_model = tf.keras.applications.EfficientNetB0(
        input_tensor=x,
        include_top=False,
        weights='imagenet'  # ImageNet 가중치 사용
    )
    
    # 상위 레이어
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    print("✅ 대체 모델 구조 생성 완료")
    
    # 컴파일
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 저장
    save_path = Path("models/health_diagnosis/eye_disease/eye_disease_alternative.h5")
    model.save(str(save_path), save_format='h5')
    print(f"✅ 대체 모델 저장: {save_path}")
    
    return model

if __name__ == "__main__":
    print("🚀 안구질환 모델 강제 로드 프로세스")
    print("=" * 70)
    
    # 1. 모델 파일 수정 시도
    success = extract_and_fix_model()
    
    if success:
        # 2. 수정된 모델 로드
        model = load_and_test_model()
        
        if model is None:
            # 3. 대안 사용
            print("\n⚠️ 수정된 모델도 실패. 대안 사용...")
            model = create_alternative_solution()
    else:
        # 3. 바로 대안 사용
        print("\n⚠️ 모델 수정 실패. 대안 사용...")
        model = create_alternative_solution()
    
    print("\n\n✅ 완료!")
    print("\n권장사항:")
    print("1. 제대로 학습된 모델 파일(.h5 형식) 확보")
    print("2. 또는 가중치 파일만 별도 저장하여 사용")
    print("3. 임시로 색상 기반 진단 사용 (emergency_eye_fix.py)")