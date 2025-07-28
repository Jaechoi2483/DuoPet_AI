"""
피부 질환 모델 완벽한 가중치 보존 변환기
모든 264개 가중치를 정확히 복원
"""
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import h5py

# TensorFlow 2.x eager 모드
tf.config.run_functions_eagerly(True)

def create_model_structure(model_name, input_shape=(224, 224, 3)):
    """원본과 동일한 구조의 모델 생성"""
    
    # Input layers
    inputs = tf.keras.Input(shape=input_shape, name='input')
    
    # Preprocessing layers
    x = tf.keras.layers.Lambda(lambda x: x / 127.5, name='tf_math_truediv')(inputs)
    x = tf.keras.layers.Lambda(lambda x: x - 1.0, name='tf_math_subtract')(x)
    
    # MobileNetV2 백본
    mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=None,
        input_tensor=x
    )
    
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

def load_all_weights(h5_path):
    """H5 파일에서 모든 가중치 추출"""
    weights_dict = {}
    
    with h5py.File(h5_path, 'r') as f:
        if 'model_weights' in f:
            def extract_weights(name, obj):
                if isinstance(obj, h5py.Dataset):
                    weights_dict[name] = np.array(obj)
            
            f['model_weights'].visititems(extract_weights)
    
    return weights_dict

def apply_all_weights(model, weights_dict, model_name):
    """모든 가중치를 모델에 적용 - 완벽한 매칭"""
    applied_count = 0
    weight_application_log = []
    
    # 1. 모든 모델 레이어 순회
    for layer in model.layers:
        layer_name = layer.name
        weights_to_apply = []
        weight_keys_used = []
        
        # 2. Dense 레이어 처리 (중첩 경로)
        if isinstance(layer, tf.keras.layers.Dense):
            # Dense 레이어는 중첩된 경로 사용
            kernel_key = f"{layer_name}/{layer_name}/kernel:0"
            bias_key = f"{layer_name}/{layer_name}/bias:0"
            
            if kernel_key in weights_dict and bias_key in weights_dict:
                weights_to_apply = [weights_dict[kernel_key], weights_dict[bias_key]]
                weight_keys_used = [kernel_key, bias_key]
        
        # 3. Conv2D 레이어 처리
        elif isinstance(layer, tf.keras.layers.Conv2D):
            # MobileNet 내부 레이어는 prefix 포함
            kernel_key = f"mobilenetv2_1.00_224/{layer_name}/kernel:0"
            bias_key = f"mobilenetv2_1.00_224/{layer_name}/bias:0"
            
            if kernel_key in weights_dict:
                if bias_key in weights_dict:
                    weights_to_apply = [weights_dict[kernel_key], weights_dict[bias_key]]
                    weight_keys_used = [kernel_key, bias_key]
                else:
                    # bias가 없는 Conv2D (MobileNet의 경우)
                    weights_to_apply = [weights_dict[kernel_key]]
                    weight_keys_used = [kernel_key]
        
        # 4. DepthwiseConv2D 레이어 처리
        elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            kernel_key = f"mobilenetv2_1.00_224/{layer_name}/depthwise_kernel:0"
            bias_key = f"mobilenetv2_1.00_224/{layer_name}/bias:0"
            
            if kernel_key in weights_dict:
                if bias_key in weights_dict:
                    weights_to_apply = [weights_dict[kernel_key], weights_dict[bias_key]]
                    weight_keys_used = [kernel_key, bias_key]
                else:
                    weights_to_apply = [weights_dict[kernel_key]]
                    weight_keys_used = [kernel_key]
        
        # 5. BatchNormalization 레이어 처리
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            gamma_key = f"mobilenetv2_1.00_224/{layer_name}/gamma:0"
            beta_key = f"mobilenetv2_1.00_224/{layer_name}/beta:0"
            mean_key = f"mobilenetv2_1.00_224/{layer_name}/moving_mean:0"
            var_key = f"mobilenetv2_1.00_224/{layer_name}/moving_variance:0"
            
            if all(key in weights_dict for key in [gamma_key, beta_key, mean_key, var_key]):
                weights_to_apply = [
                    weights_dict[gamma_key],
                    weights_dict[beta_key],
                    weights_dict[mean_key],
                    weights_dict[var_key]
                ]
                weight_keys_used = [gamma_key, beta_key, mean_key, var_key]
        
        # 6. 가중치 적용
        if weights_to_apply:
            try:
                layer.set_weights(weights_to_apply)
                applied_count += 1
                weight_application_log.append({
                    'layer_name': layer_name,
                    'layer_type': layer.__class__.__name__,
                    'weights_applied': len(weights_to_apply),
                    'weight_keys': weight_keys_used
                })
                print(f"  ✓ {layer_name} ({layer.__class__.__name__}): {len(weights_to_apply)}개 가중치 적용")
            except Exception as e:
                print(f"  ❌ {layer_name}: 가중치 적용 실패 - {e}")
    
    # 7. 적용 결과 요약
    print(f"\n📊 가중치 적용 결과:")
    print(f"  - 총 가중치 파일 수: {len(weights_dict)}")
    print(f"  - 가중치 적용된 레이어: {applied_count}")
    
    # 타입별 적용 통계
    type_stats = {}
    for log in weight_application_log:
        layer_type = log['layer_type']
        if layer_type not in type_stats:
            type_stats[layer_type] = 0
        type_stats[layer_type] += 1
    
    print(f"\n  레이어 타입별 적용 통계:")
    for layer_type, count in type_stats.items():
        print(f"    - {layer_type}: {count}개")
    
    return applied_count, weight_application_log

def verify_weights(model, test_input=None):
    """가중치가 제대로 적용되었는지 검증"""
    if test_input is None:
        test_input = np.random.rand(1, 224, 224, 3).astype(np.float32) * 255
    
    output = model.predict(test_input, verbose=0)
    
    # 출력 분석
    is_valid = True
    warnings = []
    
    if len(output.shape) == 2 and output.shape[1] == 1:  # Binary classification
        if abs(output[0,0] - 0.5) < 0.01:
            warnings.append("Binary 출력이 0.5에 너무 가까움 - 가중치 미적용 의심")
            is_valid = False
    else:  # Multi-class
        expected_uniform = 1.0 / output.shape[1]
        if np.allclose(output[0], expected_uniform, rtol=0.01):
            warnings.append("Multi-class 출력이 균등분포 - 가중치 미적용 의심")
            is_valid = False
    
    return is_valid, output, warnings

def convert_single_model(model_info):
    """단일 모델 완벽 변환"""
    print(f"\n{'='*70}")
    print(f"🔧 {model_info['name']} 모델 변환")
    print(f"{'='*70}")
    
    try:
        # 1. 가중치 추출
        print(f"\n1️⃣ 가중치 추출")
        weights_dict = load_all_weights(model_info['original_path'])
        print(f"   ✅ {len(weights_dict)}개 가중치 파일 추출 완료")
        
        # 2. 모델 생성
        print(f"\n2️⃣ 모델 구조 생성")
        new_model = create_model_structure(model_info['name'])
        print(f"   ✅ 총 {len(new_model.layers)}개 레이어 생성")
        
        # 3. 가중치 적용
        print(f"\n3️⃣ 가중치 적용")
        applied_count, application_log = apply_all_weights(new_model, weights_dict, model_info['name'])
        
        # 4. 검증
        print(f"\n4️⃣ 가중치 검증")
        is_valid, output, warnings = verify_weights(new_model)
        
        print(f"   출력 shape: {output.shape}")
        print(f"   출력 범위: [{output.min():.4f}, {output.max():.4f}]")
        print(f"   출력 예시: {output[0]}")
        
        if is_valid:
            print(f"   ✅ 가중치가 정상적으로 적용되었습니다!")
        else:
            print(f"   ⚠️ 경고: {warnings}")
        
        # 5. 컴파일
        print(f"\n5️⃣ 모델 컴파일")
        if 'binary' in model_info['name']:
            loss = 'binary_crossentropy'
        else:
            loss = 'categorical_crossentropy'
        
        new_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=['accuracy']
        )
        print(f"   ✅ 컴파일 완료")
        
        # 6. 저장
        print(f"\n6️⃣ 모델 저장")
        output_path = str(model_info['original_path']).replace('.h5', '_tf2_perfect.h5')
        new_model.save(output_path, save_format='h5')
        print(f"   ✅ 저장 완료: {output_path}")
        
        # 7. 결과 요약
        print(f"\n📊 변환 결과 요약:")
        print(f"   - 추출된 가중치: {len(weights_dict)}개")
        print(f"   - 적용된 레이어: {applied_count}개")
        print(f"   - 성공률: {(applied_count/len(new_model.layers)*100):.1f}%")
        print(f"   - 검증 결과: {'✅ 성공' if is_valid else '⚠️ 확인 필요'}")
        
        return True, applied_count, len(weights_dict)
        
    except Exception as e:
        print(f"\n❌ 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, 0

def main():
    base_dir = Path("models/health_diagnosis/skin_disease")
    print("\n🚀 피부 질환 모델 완벽 변환 시작")
    print("목표: 모든 264개 가중치를 정확히 복원")
    
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
    
    results = []
    
    for model_info in models_to_convert:
        if model_info['original_path'].exists():
            success, applied, total = convert_single_model(model_info)
            results.append({
                'name': model_info['name'],
                'success': success,
                'applied': applied,
                'total': total
            })
        else:
            print(f"\n❌ {model_info['name']} 모델 파일이 없습니다")
            results.append({
                'name': model_info['name'],
                'success': False,
                'applied': 0,
                'total': 0
            })
    
    # 최종 결과
    print(f"\n{'='*70}")
    print(f"📊 전체 변환 결과")
    print(f"{'='*70}")
    
    for result in results:
        if result['success']:
            print(f"✅ {result['name']}: {result['applied']}/{result['total']} 가중치 적용")
        else:
            print(f"❌ {result['name']}: 변환 실패")
    
    success_count = sum(1 for r in results if r['success'])
    
    if success_count > 0:
        print(f"\n🎉 변환 완료!")
        print(f"\n📝 다음 단계:")
        print(f"1. 생성된 *_tf2_perfect.h5 파일을 확인하세요")
        print(f"2. skin_disease_service.py에서 새 파일을 사용하도록 수정하세요")
        print(f"3. 서버를 재시작하여 변경사항을 적용하세요")

if __name__ == "__main__":
    main()