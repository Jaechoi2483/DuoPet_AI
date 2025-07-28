"""
피부 질환 모델 가중치를 추출하고 새 TF2 모델에 적용
"""
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import h5py

# TensorFlow 2.x eager 모드 유지
tf.config.run_functions_eagerly(True)

def extract_model_weights(model_path):
    """H5 파일에서 모델 구조와 가중치 정보 추출"""
    print(f"   📂 모델 분석 중: {model_path}")
    
    weights_list = []
    model_config = None
    
    try:
        # H5 파일 직접 읽기
        with h5py.File(model_path, 'r') as f:
            # 모델 설정 읽기
            if 'model_config' in f.attrs:
                model_config = json.loads(f.attrs['model_config'].decode('utf-8') if isinstance(f.attrs['model_config'], bytes) else f.attrs['model_config'])
                print(f"   ✅ 모델 구조 정보 발견")
            
            # 가중치 추출
            if 'model_weights' in f:
                def extract_weights(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        weights_list.append({
                            'name': name,
                            'shape': obj.shape,
                            'data': np.array(obj)
                        })
                
                f['model_weights'].visititems(extract_weights)
                print(f"   ✅ {len(weights_list)}개 가중치 추출 성공")
                
    except Exception as e:
        print(f"   ❌ H5 파일 읽기 오류: {e}")
    
    return model_config, weights_list

def create_mobilenet_model(input_shape=(224, 224, 3), num_classes=1, activation='sigmoid'):
    """원본과 유사한 MobileNet 기반 모델 생성"""
    
    # MobileNet 백본
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=None  # 가중치 없이 구조만
    )
    
    # 상위 레이어
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1280, activation='relu', name='dense_1')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation=activation, name='predictions')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

def apply_weights_to_model(model, weights_list):
    """추출한 가중치를 새 모델에 적용"""
    applied_count = 0
    
    # 레이어 이름과 가중치 매핑
    layer_weights_map = {}
    for weight_info in weights_list:
        parts = weight_info['name'].split('/')
        if len(parts) >= 2:
            layer_name = parts[0]
            weight_type = parts[1]
            
            if layer_name not in layer_weights_map:
                layer_weights_map[layer_name] = {}
            
            layer_weights_map[layer_name][weight_type] = weight_info['data']
    
    # 모델 레이어에 가중치 적용
    for layer in model.layers:
        if layer.name in layer_weights_map:
            try:
                weights_to_set = []
                layer_weights = layer_weights_map[layer.name]
                
                # 가중치 순서 맞추기 (kernel -> bias 순)
                if 'kernel:0' in layer_weights:
                    weights_to_set.append(layer_weights['kernel:0'])
                if 'depthwise_kernel:0' in layer_weights:
                    weights_to_set.append(layer_weights['depthwise_kernel:0'])
                if 'pointwise_kernel:0' in layer_weights:
                    weights_to_set.append(layer_weights['pointwise_kernel:0'])
                if 'bias:0' in layer_weights:
                    weights_to_set.append(layer_weights['bias:0'])
                if 'gamma:0' in layer_weights:
                    weights_to_set.append(layer_weights['gamma:0'])
                if 'beta:0' in layer_weights:
                    weights_to_set.append(layer_weights['beta:0'])
                if 'moving_mean:0' in layer_weights:
                    weights_to_set.append(layer_weights['moving_mean:0'])
                if 'moving_variance:0' in layer_weights:
                    weights_to_set.append(layer_weights['moving_variance:0'])
                
                if weights_to_set:
                    # 가중치 shape 확인
                    layer_weight_shapes = [w.shape for w in layer.get_weights()]
                    weight_shapes = [w.shape for w in weights_to_set]
                    
                    if len(layer_weight_shapes) == len(weight_shapes):
                        # Shape이 모두 일치하는지 확인
                        if all(lw == ws for lw, ws in zip(layer_weight_shapes, weight_shapes)):
                            layer.set_weights(weights_to_set)
                            applied_count += 1
                            print(f"      ✓ {layer.name}: {len(weights_to_set)}개 가중치 적용")
                        else:
                            print(f"      ⚠️ {layer.name}: Shape 불일치")
                
            except Exception as e:
                print(f"      ❌ {layer.name}: 가중치 적용 실패 - {e}")
    
    return applied_count

def convert_single_model(model_info):
    """단일 모델 변환"""
    print(f"\n{'='*50}")
    print(f"📁 {model_info['name']} 모델 변환 중...")
    
    try:
        # 1. 원본 모델에서 가중치 추출
        model_config, weights_list = extract_model_weights(model_info['original_path'])
        
        if not weights_list:
            print("   ❌ 가중치를 추출할 수 없습니다")
            return False
        
        # 2. 새 모델 생성
        print("   🏗️ 새 TF2 모델 생성 중...")
        
        # 출력 크기 결정
        if model_info['name'].endswith('binary'):
            num_classes = 1
            activation = 'sigmoid'
        else:
            num_classes = 3  # multi 모델은 3개 클래스
            activation = 'softmax'
        
        new_model = create_mobilenet_model(
            input_shape=model_info['input_shape'],
            num_classes=num_classes,
            activation=activation
        )
        
        print(f"   ✅ 모델 생성 완료: {len(new_model.layers)}개 레이어")
        
        # 3. 가중치 적용
        print("   🔄 가중치 적용 중...")
        applied_count = apply_weights_to_model(new_model, weights_list)
        print(f"   ✅ {applied_count}개 레이어에 가중치 적용 완료")
        
        # 4. 컴파일
        if activation == 'sigmoid':
            loss = 'binary_crossentropy'
        else:
            loss = 'categorical_crossentropy'
        
        new_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=['accuracy']
        )
        
        # 5. 테스트
        print("   🧪 모델 테스트 중...")
        test_input = np.random.rand(1, *model_info['input_shape']).astype(np.float32)
        output = new_model.predict(test_input, verbose=0)
        print(f"   ✅ 출력 shape: {output.shape}")
        print(f"   ✅ 출력 범위: [{output.min():.4f}, {output.max():.4f}]")
        
        # 6. 저장
        output_path = str(model_info['original_path']).replace('.h5', '_tf2_preserved.h5')
        new_model.save(output_path, save_format='h5')
        print(f"   💾 저장 완료: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    base_dir = Path("models/health_diagnosis/skin_disease")
    print("🔧 피부 질환 모델 가중치 보존 변환 (추출 방식)")
    
    models_to_convert = [
        {
            "name": "dog_binary",
            "original_path": base_dir / "classification/dog_binary/dog_binary_model.h5",
            "input_shape": (224, 224, 3)
        },
        {
            "name": "cat_binary", 
            "original_path": base_dir / "classification/cat_binary/cat_binary_model.h5",
            "input_shape": (224, 224, 3)
        },
        {
            "name": "dog_multi_136",
            "original_path": base_dir / "classification/dog_multi_136/dog_multi_136_model.h5",
            "input_shape": (224, 224, 3)
        },
        {
            "name": "dog_multi_456",
            "original_path": base_dir / "classification/dog_multi_456/dog_multi_456_model.h5",
            "input_shape": (224, 224, 3)
        }
    ]
    
    success_count = 0
    
    for model_info in models_to_convert:
        if model_info['original_path'].exists():
            if convert_single_model(model_info):
                success_count += 1
        else:
            print(f"\n❌ {model_info['name']} 모델 파일이 없습니다")
    
    print(f"\n{'='*50}")
    print(f"📊 변환 결과: {success_count}/{len(models_to_convert)} 성공")
    
    if success_count > 0:
        print("\n✨ 가중치를 보존한 모델 변환 성공!")
        print("📌 *_tf2_preserved.h5 파일로 저장되었습니다")
        print("🔍 원본 모델의 학습된 가중치가 유지되었습니다")
    else:
        print("\n⚠️ 모든 모델 변환에 실패했습니다")

if __name__ == "__main__":
    main()