"""
피부 질환 모델의 가중치를 정확히 복원하는 최종 버전
"""
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import h5py

# TensorFlow 2.x eager 모드
tf.config.run_functions_eagerly(True)

def create_exact_model_structure(model_name, input_shape=(224, 224, 3)):
    """원본과 정확히 동일한 구조의 모델 생성"""
    
    # Input layers
    inputs = tf.keras.Input(shape=input_shape, name='input')
    
    # Preprocessing layers
    x = tf.keras.layers.Lambda(lambda x: x / 127.5, name='tf_math_truediv')(inputs)
    x = tf.keras.layers.Lambda(lambda x: x - 1.0, name='tf_math_subtract')(x)
    
    # MobileNetV2 백본
    mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=None,  # 가중치는 나중에 복원
        input_tensor=x
    )
    mobilenet._name = 'mobilenetv2_1.00_224'
    
    # 백본 출력
    x = mobilenet.output
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d')(x)
    
    # Dense 레이어 - 모델별로 다른 이름 사용
    if 'binary' in model_name:
        if 'dog' in model_name:
            x = tf.keras.layers.Dense(128, activation='relu', name='dense_2')(x)
            x = tf.keras.layers.Dropout(0.5, name='dropout_1')(x)
            outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='dense_3')(x)
        else:  # cat_binary
            x = tf.keras.layers.Dense(128, activation='relu', name='dense')(x)
            x = tf.keras.layers.Dropout(0.5, name='dropout')(x)
            outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='dense_1')(x)
    else:  # multi class
        if '136' in model_name:
            x = tf.keras.layers.Dense(128, activation='relu', name='dense_4')(x)
            x = tf.keras.layers.Dropout(0.5, name='dropout_2')(x)
            outputs = tf.keras.layers.Dense(3, activation='softmax', name='dense_5')(x)
        else:  # 456
            x = tf.keras.layers.Dense(128, activation='relu', name='dense_6')(x)
            x = tf.keras.layers.Dropout(0.5, name='dropout_3')(x)
            outputs = tf.keras.layers.Dense(3, activation='softmax', name='dense_7')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='model')
    return model

def load_weights_from_h5(h5_path):
    """H5 파일에서 모든 가중치 추출"""
    weights_dict = {}
    
    with h5py.File(h5_path, 'r') as f:
        if 'model_weights' in f:
            def extract_weights(name, obj):
                if isinstance(obj, h5py.Dataset):
                    weights_dict[name] = np.array(obj)
            
            f['model_weights'].visititems(extract_weights)
    
    return weights_dict

def apply_weights_to_model(model, weights_dict, model_name):
    """추출한 가중치를 모델에 정확히 적용"""
    applied_count = 0
    
    # Dense 레이어 가중치 적용
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer_name = layer.name
            # 중첩된 경로 처리
            kernel_key = f"{layer_name}/{layer_name}/kernel:0"
            bias_key = f"{layer_name}/{layer_name}/bias:0"
            
            if kernel_key in weights_dict and bias_key in weights_dict:
                try:
                    layer.set_weights([weights_dict[kernel_key], weights_dict[bias_key]])
                    applied_count += 1
                    print(f"  ✓ {layer_name}: Dense 가중치 적용")
                except Exception as e:
                    print(f"  ❌ {layer_name}: 가중치 적용 실패 - {e}")
    
    # MobileNet 가중치 적용
    mobilenet_layer = None
    for layer in model.layers:
        if layer.name == 'mobilenetv2_1.00_224':
            mobilenet_layer = layer
            break
    
    if mobilenet_layer:
        print(f"  🔄 MobileNet 가중치 적용 중...")
        mobilenet_applied = 0
        
        for sub_layer in mobilenet_layer.layers:
            layer_name = sub_layer.name
            weights_to_set = []
            
            # 각 레이어 타입별로 가중치 키 구성
            if isinstance(sub_layer, tf.keras.layers.Conv2D):
                kernel_key = f"mobilenetv2_1.00_224/{layer_name}/kernel:0"
                if kernel_key in weights_dict:
                    weights_to_set.append(weights_dict[kernel_key])
                    
            elif isinstance(sub_layer, tf.keras.layers.DepthwiseConv2D):
                kernel_key = f"mobilenetv2_1.00_224/{layer_name}/depthwise_kernel:0"
                if kernel_key in weights_dict:
                    weights_to_set.append(weights_dict[kernel_key])
                    
            elif isinstance(sub_layer, tf.keras.layers.BatchNormalization):
                gamma_key = f"mobilenetv2_1.00_224/{layer_name}/gamma:0"
                beta_key = f"mobilenetv2_1.00_224/{layer_name}/beta:0"
                mean_key = f"mobilenetv2_1.00_224/{layer_name}/moving_mean:0"
                var_key = f"mobilenetv2_1.00_224/{layer_name}/moving_variance:0"
                
                if all(key in weights_dict for key in [gamma_key, beta_key, mean_key, var_key]):
                    weights_to_set = [
                        weights_dict[gamma_key],
                        weights_dict[beta_key],
                        weights_dict[mean_key],
                        weights_dict[var_key]
                    ]
            
            if weights_to_set:
                try:
                    sub_layer.set_weights(weights_to_set)
                    mobilenet_applied += 1
                except Exception as e:
                    pass  # 일부 레이어는 가중치가 없을 수 있음
        
        print(f"  ✓ MobileNet: {mobilenet_applied}개 서브레이어에 가중치 적용")
        applied_count += mobilenet_applied
    
    return applied_count

def convert_single_model(model_info):
    """단일 모델 변환 - 정확한 가중치 복원"""
    print(f"\n{'='*60}")
    print(f"📁 {model_info['name']} 모델 변환 중...")
    
    try:
        # 1. 원본 모델에서 가중치 추출
        print(f"  📂 가중치 추출 중: {model_info['original_path']}")
        weights_dict = load_weights_from_h5(model_info['original_path'])
        print(f"  ✅ {len(weights_dict)}개 가중치 텐서 추출 완료")
        
        # 2. 정확한 구조의 새 모델 생성
        print(f"  🏗️ 원본과 동일한 구조의 모델 생성 중...")
        new_model = create_exact_model_structure(model_info['name'])
        print(f"  ✅ 모델 생성 완료")
        
        # 3. 가중치 적용
        print(f"  🔄 가중치 복원 중...")
        applied_count = apply_weights_to_model(new_model, weights_dict, model_info['name'])
        print(f"  ✅ 총 {applied_count}개 레이어/서브레이어에 가중치 복원 완료")
        
        # 4. 컴파일
        if 'binary' in model_info['name']:
            loss = 'binary_crossentropy'
        else:
            loss = 'categorical_crossentropy'
        
        new_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=['accuracy']
        )
        
        # 5. 테스트
        print(f"  🧪 모델 테스트 중...")
        test_input = np.random.rand(1, 224, 224, 3).astype(np.float32) * 255  # 0-255 범위
        new_output = new_model.predict(test_input, verbose=0)
        
        print(f"  📊 출력 shape: {new_output.shape}")
        print(f"  📊 출력 범위: [{new_output.min():.4f}, {new_output.max():.4f}]")
        
        # 출력이 제대로 나오는지 확인 (랜덤 초기화가 아닌지)
        if 'binary' in model_info['name']:
            if abs(new_output[0,0] - 0.5) < 0.01:
                print(f"  ⚠️ 경고: 출력이 0.5에 가까움 - 가중치가 제대로 로드되지 않았을 수 있음")
        else:
            if np.allclose(new_output[0], 1.0/new_output.shape[1], rtol=0.01):
                print(f"  ⚠️ 경고: 출력이 균등분포 - 가중치가 제대로 로드되지 않았을 수 있음")
        
        # 6. 저장
        output_path = str(model_info['original_path']).replace('.h5', '_tf2_final.h5')
        new_model.save(output_path, save_format='h5')
        print(f"  💾 저장 완료: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    base_dir = Path("models/health_diagnosis/skin_disease")
    print("🔧 피부 질환 모델 최종 가중치 복원")
    
    models_to_convert = [
        {
            "name": "dog_binary",
            "original_path": base_dir / "classification/dog_binary/dog_binary_model.h5"
        },
        {
            "name": "cat_binary", 
            "original_path": base_dir / "classification/cat_binary/cat_binary_model.h5"
        },
        {
            "name": "dog_multi_136",
            "original_path": base_dir / "classification/dog_multi_136/dog_multi_136_model.h5"
        },
        {
            "name": "dog_multi_456",
            "original_path": base_dir / "classification/dog_multi_456/dog_multi_456_model.h5"
        }
    ]
    
    success_count = 0
    
    for model_info in models_to_convert:
        if model_info['original_path'].exists():
            if convert_single_model(model_info):
                success_count += 1
        else:
            print(f"\n❌ {model_info['name']} 모델 파일이 없습니다")
    
    print(f"\n{'='*60}")
    print(f"📊 변환 결과: {success_count}/{len(models_to_convert)} 성공")
    
    if success_count > 0:
        print("\n✨ 가중치를 복원한 TF2 모델 생성 완료!")
        print("📌 *_tf2_final.h5 파일로 저장되었습니다")
        print("\n⚠️ 중요: 출력값을 확인하여 가중치가 제대로 로드되었는지 확인하세요!")
        print("   - binary 모델: 0.5와 다른 값이 나와야 함")
        print("   - multi 모델: 균등분포(0.333...)가 아닌 값이 나와야 함")

if __name__ == "__main__":
    main()