"""
원본 안구질환 모델 로드 문제 해결
normalization 레이어 문제 우회
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import json
from pathlib import Path
import h5py
import warnings
warnings.filterwarnings('ignore')

def extract_weights_from_keras_file():
    """Keras 파일에서 직접 가중치 추출"""
    
    print("🔧 원본 모델 가중치 추출")
    print("="*80)
    
    # 경로 설정
    import platform
    if platform.system() == "Windows":
        original_model_path = r"C:\Users\ictedu1_021\Desktop\안구질환모델\best_grouped_model.keras"
        original_class_map_path = r"C:\Users\ictedu1_021\Desktop\안구질환모델\class_map.json"
    else:
        original_model_path = "/mnt/c/Users/ictedu1_021/Desktop/안구질환모델/best_grouped_model.keras"
        original_class_map_path = "/mnt/c/Users/ictedu1_021/Desktop/안구질환모델/class_map.json"
    
    # 클래스맵 로드
    with open(original_class_map_path, 'r', encoding='utf-8') as f:
        original_class_map = json.load(f)
    
    print("\n📋 원본 대분류 클래스맵:")
    for idx, name in original_class_map.items():
        print(f"  {idx}: {name}")
    
    # 현재 사용할 세부 클래스맵
    target_class_map = {
        "0": "정상",
        "1": "백내장",
        "2": "결막염", 
        "3": "각막궤양",
        "4": "기타안구질환"
    }
    
    print("\n📋 타겟 세부 클래스맵:")
    for idx, name in target_class_map.items():
        print(f"  {idx}: {name}")
    
    # H5 파일로 직접 읽기 시도
    print("\n📂 Keras 파일 구조 분석...")
    
    try:
        # 먼저 파일 구조 확인
        with h5py.File(original_model_path, 'r') as f:
            print("\n파일의 최상위 키:")
            for key in f.keys():
                print(f"  - {key}")
            
            # 모델 구조 정보 확인
            if 'model_config' in f.attrs:
                import json
                config = json.loads(f.attrs['model_config'].decode('utf-8'))
                print("\n모델 아키텍처 발견!")
                
                # 입출력 정보
                if 'config' in config:
                    if 'layers' in config['config']:
                        layers = config['config']['layers']
                        print(f"총 레이어 수: {len(layers)}")
                        
                        # 출력층 찾기
                        for layer in layers:
                            if layer.get('class_name') == 'Dense':
                                layer_config = layer.get('config', {})
                                if layer_config.get('units') == 5:
                                    print(f"\n출력층 발견: {layer_config.get('name')}")
                                    print(f"  - Units: {layer_config.get('units')}")
                                    print(f"  - Activation: {layer_config.get('activation')}")
        
        # 모델 로드 시도 - normalization 레이어 스킵
        print("\n🔄 수정된 방법으로 모델 로드 시도...")
        
        # Custom object 정의
        def custom_normalization(*args, **kwargs):
            # Normalization 레이어를 BatchNormalization으로 대체
            return tf.keras.layers.BatchNormalization(*args, **kwargs)
        
        custom_objects = {
            'Normalization': custom_normalization,
            'swish': tf.nn.swish,
            'Swish': tf.keras.layers.Activation(tf.nn.swish),
            'FixedDropout': tf.keras.layers.Dropout
        }
        
        # 모델 로드
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(
                original_model_path,
                compile=False
            )
        
        print("✅ 모델 로드 성공!")
        
        # 모델 정보
        print(f"\n📊 모델 정보:")
        print(f"  입력 형태: {model.input_shape}")
        print(f"  출력 형태: {model.output_shape}")
        print(f"  총 레이어: {len(model.layers)}")
        print(f"  총 파라미터: {model.count_params():,}")
        
        # 마지막 Dense 레이어 찾기
        output_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Dense) and layer.units == 5:
                output_layer = layer
                break
        
        if output_layer:
            print(f"\n🎯 출력층 분석: {output_layer.name}")
            weights = output_layer.get_weights()
            if len(weights) >= 2:
                w, b = weights[0], weights[1]
                print(f"  가중치 형태: {w.shape}")
                print(f"  편향 형태: {b.shape}")
                print(f"\n  각 클래스별 편향값:")
                for i, bias in enumerate(b):
                    print(f"    클래스 {i} ({original_class_map.get(str(i))}): {bias:.6f}")
                
                # 편향이 모두 0인지 확인
                if np.all(np.abs(b) < 1e-6):
                    print("\n⚠️ 경고: 모든 편향값이 0입니다!")
                    print("→ 모델이 제대로 학습되지 않았을 가능성이 있습니다.")
                else:
                    print("\n✅ 편향값이 정상적으로 학습되었습니다.")
        
        # 간단한 테스트
        print("\n🧪 간단한 테스트...")
        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        prediction = model.predict(test_input, verbose=0)
        
        print("\n예측 결과:")
        for i, prob in enumerate(prediction[0]):
            class_name = original_class_map.get(str(i), f"Unknown_{i}")
            print(f"  {class_name}: {prob*100:.2f}%")
        
        return model, original_class_map, target_class_map
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("\n상세 오류 정보:")
        import traceback
        traceback.print_exc()
        
        return None, None, None

def create_fine_tuned_model(base_model, original_classes, target_classes):
    """대분류 모델을 세부 분류용으로 변환"""
    
    print("\n🔨 세부 분류 모델 생성...")
    
    # 기존 모델의 특징 추출 부분만 사용
    # 마지막 Dense 레이어 제거
    base_output = None
    for i in range(len(base_model.layers) - 1, -1, -1):
        layer = base_model.layers[i]
        if not isinstance(layer, tf.keras.layers.Dense):
            base_output = layer.output
            break
    
    if base_output is None:
        print("❌ 특징 추출 레이어를 찾을 수 없습니다.")
        return None
    
    # 새로운 분류 헤드 추가
    x = tf.keras.layers.Dense(256, activation='relu', name='new_dense_1')(base_output)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu', name='new_dense_2')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(5, activation='softmax', name='eye_disease_output')(x)
    
    # 새 모델 생성
    new_model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    
    # 기존 레이어는 동결 (선택적)
    for layer in base_model.layers[:-1]:
        layer.trainable = False
    
    print("✅ 세부 분류 모델 생성 완료!")
    print(f"  총 레이어: {len(new_model.layers)}")
    print(f"  학습 가능한 파라미터: {new_model.count_params():,}")
    
    return new_model

def save_model_safely(model, target_class_map):
    """모델을 안전하게 저장"""
    
    output_dir = Path("models/health_diagnosis/eye_disease")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 기존 파일 백업
    import shutil
    for file in output_dir.glob("*.h5"):
        backup = file.with_suffix(file.suffix + '.backup_before_fix')
        shutil.copy(file, backup)
        print(f"📦 백업: {backup}")
    
    # 새 모델 저장
    print("\n💾 모델 저장 중...")
    
    # H5 형식
    h5_path = output_dir / "eye_disease_from_original.h5"
    model.save(h5_path, save_format='h5')
    print(f"✅ H5 저장: {h5_path}")
    
    # Keras 형식
    keras_path = output_dir / "eye_disease_from_original.keras" 
    try:
        model.save(keras_path)
        print(f"✅ Keras 저장: {keras_path}")
    except:
        print("⚠️ Keras 형식 저장 실패 (H5는 성공)")
    
    # 클래스맵 저장
    class_map_path = output_dir / "class_map.json"
    with open(class_map_path, 'w', encoding='utf-8') as f:
        json.dump(target_class_map, f, ensure_ascii=False, indent=2)
    print(f"✅ 클래스맵 저장: {class_map_path}")
    
    # 변환 정보 저장
    info = {
        "note": "원본 모델은 대분류(5개)였으나, 세부 질환 분류용으로 변환",
        "original_classes": {
            "0": "각막 질환",
            "1": "결막 및 누관 질환", 
            "2": "수정체 질환",
            "3": "안검 질환",
            "4": "안구 내부 질환"
        },
        "target_classes": target_class_map,
        "important": "이 모델은 추가 학습이 필요합니다! 특히 '정상' 클래스 데이터로 fine-tuning 필수",
        "recommendation": "세부 질환별 데이터셋으로 transfer learning 수행 권장"
    }
    
    info_path = output_dir / "model_conversion_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"✅ 변환 정보 저장: {info_path}")

if __name__ == "__main__":
    # 1. 원본 모델 로드
    model, orig_classes, target_classes = extract_weights_from_keras_file()
    
    if model is not None:
        # 2. 세부 분류용 모델 생성
        new_model = create_fine_tuned_model(model, orig_classes, target_classes)
        
        if new_model:
            # 3. 모델 저장
            save_model_safely(new_model, target_classes)
            
            print("\n✅ 완료!")
            print("\n📋 중요 사항:")
            print("1. 원본은 대분류 모델이었으므로 세부 질환 분류를 위해서는 추가 학습 필수")
            print("2. 특히 '정상' 클래스는 원본에 없었으므로 반드시 학습 필요")
            print("3. services/eye_disease_service.py에서 'eye_disease_from_original.h5' 사용")
            print("4. 임시로라도 동작하게 하려면 긴급 색상 기반 진단 사용 권장")
    else:
        print("\n💡 대안 제안:")
        print("1. 사전학습된 EfficientNet/ResNet + 새로운 분류 헤드")
        print("2. 정상/비정상 이진 분류부터 시작")
        print("3. 각 질환별 One-vs-Rest 분류기 앙상블")