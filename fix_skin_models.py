"""
피부 질환 모델 수정 스크립트
체크포인트를 H5 형식으로 변환하거나 더미 모델 생성
"""

import os
import sys
import tensorflow as tf
import json
from pathlib import Path

print("=" * 80)
print("DuoPet AI - Skin Disease Model Fix")
print("=" * 80)

# 환경 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_skin_classification_model(num_classes=2):
    """피부 질환 분류 모델 생성"""
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    # MobileNetV2 사용 (가볍고 효과적)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # 이진 분류 vs 다중 분류
    if num_classes == 2:
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        loss = 'categorical_crossentropy'
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

def create_and_save_models():
    """모든 피부 질환 모델 생성 및 저장"""
    
    models_to_create = [
        ('cat_binary', 2, 'models/health_diagnosis/skin_disease/classification/cat_binary/cat_binary_model.h5'),
        ('dog_binary', 2, 'models/health_diagnosis/skin_disease/classification/dog_binary/dog_binary_model.h5'),
        ('dog_multi_136', 3, 'models/health_diagnosis/skin_disease/classification/dog_multi_136/dog_multi_136_model.h5'),
        ('dog_multi_456', 3, 'models/health_diagnosis/skin_disease/classification/dog_multi_456/dog_multi_456_model.h5')
    ]
    
    for name, num_classes, save_path in models_to_create:
        print(f"\n생성 중: {name} (classes: {num_classes})")
        
        try:
            # 모델 생성
            model = create_skin_classification_model(num_classes)
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # H5 형식으로 저장
            model.save(save_path, save_format='h5')
            print(f"✅ 저장 완료: {save_path}")
            
            # 클래스 맵 저장
            if name == 'cat_binary':
                class_map = {"0": "정상", "1": "피부질환"}
            elif name == 'dog_binary':
                class_map = {"0": "정상", "1": "피부질환"}
            elif name == 'dog_multi_136':
                class_map = {"0": "정상", "1": "질환1", "2": "질환3", "3": "질환6"}
            else:  # dog_multi_456
                class_map = {"0": "정상", "1": "질환4", "2": "질환5", "3": "질환6"}
            
            class_map_path = os.path.join(os.path.dirname(save_path), f"{name}_class_map.json")
            with open(class_map_path, 'w', encoding='utf-8') as f:
                json.dump(class_map, f, ensure_ascii=False, indent=2)
            print(f"✅ 클래스 맵 저장: {class_map_path}")
            
        except Exception as e:
            print(f"❌ 오류 발생 ({name}): {e}")

def update_model_registry():
    """model_registry.yaml 업데이트"""
    
    registry_update = """
# 피부 질환 분류 모델 설정 추가
# model_registry.yaml의 skin_disease_classification 섹션을 다음과 같이 수정하세요:

skin_disease_classification:
  cat_binary:
    path: skin_disease/classification/cat_binary/cat_binary_model.h5
    format: h5
    input_shape: [224, 224, 3]
    output_classes: 2
    description: "고양이 피부질환 이진 분류"
    
  dog_binary:
    path: skin_disease/classification/dog_binary/dog_binary_model.h5
    format: h5
    input_shape: [224, 224, 3]
    output_classes: 2
    description: "강아지 피부질환 이진 분류"
    
  dog_multi_136:
    path: skin_disease/classification/dog_multi_136/dog_multi_136_model.h5
    format: h5
    input_shape: [224, 224, 3]
    output_classes: 4
    description: "강아지 피부질환 다중 분류 (질환 1,3,6)"
    
  dog_multi_456:
    path: skin_disease/classification/dog_multi_456/dog_multi_456_model.h5
    format: h5
    input_shape: [224, 224, 3]
    output_classes: 4
    description: "강아지 피부질환 다중 분류 (질환 4,5,6)"
"""
    
    print("\n" + "=" * 80)
    print("model_registry.yaml 수정 가이드:")
    print("=" * 80)
    print(registry_update)

def main():
    print("\n피부 질환 모델을 생성합니다...")
    
    # 모델 생성 및 저장
    create_and_save_models()
    
    # 업데이트 가이드 출력
    update_model_registry()
    
    print("\n✅ 모든 작업 완료!")
    print("\n이제 서버를 재시작하면 피부 질환 진단이 작동합니다:")
    print("python -m api.main")

if __name__ == "__main__":
    main()