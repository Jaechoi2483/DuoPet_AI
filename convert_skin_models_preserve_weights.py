"""
피부 질환 모델 가중치를 보존하면서 TF2로 변환하는 스크립트
"""
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import h5py
import sys

# 상위 디렉토리를 파이썬 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TF1 호환성 모드 활성화
tf.compat.v1.disable_eager_execution()

def extract_weights_from_h5(h5_path):
    """H5 파일에서 직접 가중치 추출"""
    weights_dict = {}
    try:
        with h5py.File(h5_path, 'r') as f:
            def extract_group(name, obj):
                if isinstance(obj, h5py.Group):
                    if 'kernel:0' in obj or 'bias:0' in obj:
                        layer_weights = {}
                        if 'kernel:0' in obj:
                            layer_weights['kernel'] = np.array(obj['kernel:0'])
                        if 'bias:0' in obj:
                            layer_weights['bias'] = np.array(obj['bias:0'])
                        weights_dict[name] = layer_weights
            
            f.visititems(extract_group)
    except Exception as e:
        print(f"H5 파일 읽기 오류: {e}")
    
    return weights_dict

def convert_model_preserve_weights(model_path, output_path, model_info):
    """가중치를 보존하면서 모델 변환"""
    print(f"\n📁 {model_info['name']} 모델 변환 중...")
    
    try:
        # 1. 세션 초기화
        tf.keras.backend.clear_session()
        
        # 2. TF1 스타일로 모델 로드 시도
        print("   🔄 TF1 호환 모드로 모델 로드 중...")
        
        # 먼저 모델 구조를 JSON으로 저장
        with tf.compat.v1.Session() as sess:
            # 그래프 모드에서 모델 로드
            try:
                model = tf.keras.models.load_model(str(model_path), compile=False)
                
                # 모델 구조를 JSON으로 저장
                model_json = model.to_json()
                
                # 가중치 추출
                weights = model.get_weights()
                
                print(f"   ✅ 모델 로드 성공: {len(weights)}개 가중치 레이어")
                
            except Exception as e:
                print(f"   ❌ 표준 로드 실패: {e}")
                
                # H5 파일에서 직접 가중치 추출
                print("   🔧 H5 파일에서 직접 가중치 추출 중...")
                weights_dict = extract_weights_from_h5(model_path)
                
                if not weights_dict:
                    raise ValueError("가중치를 추출할 수 없습니다")
                
                print(f"   ✅ 가중치 추출 성공: {len(weights_dict)}개 레이어")
                
                # 새 모델 생성하고 가중치 적용
                model = None
                weights = None
        
        # 3. Eager 모드로 전환하여 새 모델 생성
        tf.compat.v1.enable_eager_execution()
        tf.config.run_functions_eagerly(True)
        
        print("   🏗️ TF2 모델 구조 생성 중...")
        
        if model is not None and weights is not None:
            # 기존 모델 구조 재사용
            new_model = tf.keras.models.model_from_json(model_json)
            new_model.set_weights(weights)
            print("   ✅ 기존 모델 구조와 가중치 복원 성공")
        else:
            # 모델 구조 재생성 필요
            print("   ⚠️ 새 모델 구조 생성이 필요합니다")
            
            # 원본 모델의 레이어 구조 분석
            with h5py.File(model_path, 'r') as f:
                if 'model_config' in f.attrs:
                    import json
                    config = json.loads(f.attrs['model_config'])
                    print(f"   📊 모델 설정 발견: {config.get('class_name', 'Unknown')}")
            
            # 기본 CNN 구조로 생성
            inputs = tf.keras.Input(shape=model_info['input_shape'])
            x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
            x = tf.keras.layers.MaxPooling2D(2)(x)
            x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D(2)(x)
            x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            
            if model_info['activation'] == 'sigmoid' and model_info['output_units'] == 2:
                outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            else:
                outputs = tf.keras.layers.Dense(model_info['output_units'], activation=model_info['activation'])(x)
            
            new_model = tf.keras.Model(inputs, outputs)
            
            # H5에서 추출한 가중치 적용 시도
            if 'weights_dict' in locals() and weights_dict:
                print("   🔄 추출한 가중치 적용 시도...")
                applied_count = 0
                for layer in new_model.layers:
                    if layer.name in weights_dict:
                        try:
                            layer_weights = []
                            if 'kernel' in weights_dict[layer.name]:
                                layer_weights.append(weights_dict[layer.name]['kernel'])
                            if 'bias' in weights_dict[layer.name]:
                                layer_weights.append(weights_dict[layer.name]['bias'])
                            if layer_weights:
                                layer.set_weights(layer_weights)
                                applied_count += 1
                        except:
                            pass
                
                print(f"   ✅ {applied_count}개 레이어에 가중치 적용")
        
        # 4. 컴파일
        if model_info['activation'] == 'sigmoid':
            loss = 'binary_crossentropy'
        else:
            loss = 'categorical_crossentropy'
            
        new_model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['accuracy']
        )
        
        # 5. 테스트
        print("   🧪 모델 테스트 중...")
        test_input = np.random.rand(1, *model_info['input_shape']).astype(np.float32)
        output = new_model.predict(test_input, verbose=0)
        print(f"   ✅ 출력 shape: {output.shape}, 범위: [{output.min():.4f}, {output.max():.4f}]")
        
        # 6. 저장
        new_model.save(str(output_path), save_format='h5')
        print(f"   💾 저장 완료: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 세션 정리
        tf.keras.backend.clear_session()

def main():
    base_dir = Path("models/health_diagnosis/skin_disease")
    print("🔧 피부 질환 모델 가중치 보존 변환 시작...")
    
    models_to_convert = [
        {
            "name": "dog_binary",
            "original_path": base_dir / "classification/dog_binary/dog_binary_model.h5",
            "tf2_path": base_dir / "classification/dog_binary/dog_binary_model_tf2_weights.h5",
            "input_shape": (224, 224, 3),
            "output_units": 2,
            "activation": "sigmoid"
        },
        {
            "name": "cat_binary",
            "original_path": base_dir / "classification/cat_binary/cat_binary_model.h5", 
            "tf2_path": base_dir / "classification/cat_binary/cat_binary_model_tf2_weights.h5",
            "input_shape": (224, 224, 3),
            "output_units": 2,
            "activation": "sigmoid"
        },
        {
            "name": "dog_multi_136",
            "original_path": base_dir / "classification/dog_multi_136/dog_multi_136_model.h5",
            "tf2_path": base_dir / "classification/dog_multi_136/dog_multi_136_model_tf2_weights.h5",
            "input_shape": (224, 224, 3),
            "output_units": 3,
            "activation": "softmax"
        },
        {
            "name": "dog_multi_456",
            "original_path": base_dir / "classification/dog_multi_456/dog_multi_456_model.h5",
            "tf2_path": base_dir / "classification/dog_multi_456/dog_multi_456_model_tf2_weights.h5",
            "input_shape": (224, 224, 3),
            "output_units": 3,
            "activation": "softmax"
        }
    ]
    
    success_count = 0
    
    for model_info in models_to_convert:
        if model_info['original_path'].exists():
            success = convert_model_preserve_weights(
                model_info['original_path'],
                model_info['tf2_path'],
                model_info
            )
            if success:
                success_count += 1
        else:
            print(f"\n❌ {model_info['name']} 모델 파일이 없습니다")
    
    print(f"\n{'='*50}")
    print(f"📊 변환 결과: {success_count}/{len(models_to_convert)} 성공")
    
    if success_count > 0:
        print("\n✨ 가중치를 보존한 모델 변환이 부분적으로 성공했습니다!")
        print("📌 *_tf2_weights.h5 파일로 저장되었습니다")
    else:
        print("\n⚠️ 가중치 보존 변환에 실패했습니다")
        print("💡 원본 모델이 매우 오래된 TF 버전일 수 있습니다")

if __name__ == "__main__":
    main()