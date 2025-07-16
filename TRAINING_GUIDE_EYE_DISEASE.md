# 🐾 DuoPet AI 반려동물 안구질환 모델 학습 가이드

## 📋 목차
1. [개요](#개요)
2. [사전 준비사항](#사전-준비사항)
3. [Step 1: 데이터 준비 및 업로드](#step-1-데이터-준비-및-업로드)
4. [Step 2: Google Colab 환경 설정](#step-2-google-colab-환경-설정)
5. [Step 3: 데이터 준비 및 전처리](#step-3-데이터-준비-및-전처리)
6. [Step 4: 모델 학습 코드 작성](#step-4-모델-학습-코드-작성)
7. [Step 5: 모델 학습 실행](#step-5-모델-학습-실행)
8. [Step 6: 모델 평가 및 저장](#step-6-모델-평가-및-저장)
9. [Step 7: DuoPet AI 프로젝트에 통합](#step-7-duopet-ai-프로젝트에-통합)
10. [문제 해결 가이드](#문제-해결-가이드)

---

## 개요

이 가이드는 **E:\DATA\153.반려동물 안구질환 데이터**를 사용하여 Google Colab에서 안구질환 진단 모델을 학습하는 과정을 설명합니다. 

### 🎯 목표
- TensorFlow 2.14.0을 사용한 안구질환 분류 모델 학습
- 기존 피부질환 모델과 동일한 형식으로 모델 저장
- DuoPet AI 서비스에 즉시 통합 가능한 모델 생성

### 📊 모델 사양
- **입력 크기**: 224x224x3 (RGB 이미지)
- **정규화**: ImageNet 표준 정규화
- **모델 형식**: TensorFlow SavedModel 또는 Keras .h5
- **예상 질환 분류**: 정상, 백내장, 결막염, 각막궤양 등

---

## 사전 준비사항

### 필요한 것들:
- ✅ Google 계정
- ✅ Google Drive 저장 공간 (최소 10GB)
- ✅ E:\DATA\153.반려동물 안구질환 데이터
- ✅ 안정적인 인터넷 연결

---

## Step 1: 데이터 준비 및 업로드

### 1.1 데이터 압축
```powershell
# PowerShell 또는 Windows 탐색기에서
# E:\DATA\153.반려동물 안구질환 데이터 폴더를 우클릭
# "압축(ZIP) 폴더로 보내기" 선택
# 파일명: eye_disease_dataset.zip
```

### 1.2 Google Drive 폴더 구조 생성
```
Google Drive/
└── DuoPet_AI_Training/
    ├── datasets/
    │   └── eye_disease_dataset.zip
    ├── models/
    └── logs/
```

### 1.3 파일 업로드
1. [Google Drive](https://drive.google.com) 접속
2. `DuoPet_AI_Training/datasets/` 폴더에 `eye_disease_dataset.zip` 업로드
3. 업로드 완료 확인 (파일 크기와 대조)

---

## Step 2: Google Colab 환경 설정

### 2.1 새 노트북 생성
1. [Google Colab](https://colab.research.google.com) 접속
2. "새 노트북" 클릭
3. 노트북 이름을 `DuoPet_Eye_Disease_Training.ipynb`로 변경

### 2.2 GPU 런타임 설정
```python
# 첫 번째 셀: GPU 설정 확인
# 메뉴: 런타임 > 런타임 유형 변경 > 하드웨어 가속기: GPU (T4 추천)

!nvidia-smi
```

### 2.3 Google Drive 마운트
```python
from google.colab import drive
drive.mount('/content/drive')

# 마운트 확인
!ls /content/drive/MyDrive/DuoPet_AI_Training/datasets/
```

---

## Step 3: 데이터 준비 및 전처리

### 3.1 필요한 라이브러리 설치
```python
# DuoPet AI와 동일한 환경 구성
!pip install tensorflow==2.14.0
!pip install opencv-python==4.8.1.78
!pip install pandas numpy matplotlib seaborn
!pip install scikit-learn
!pip install albumentations

# 설치 확인
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
```

### 3.2 데이터 압축 해제
```python
import zipfile
import os

# 압축 해제
zip_path = "/content/drive/MyDrive/DuoPet_AI_Training/datasets/eye_disease_dataset.zip"
extract_path = "/content/eye_disease_data"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# 데이터 구조 확인
!find /content/eye_disease_data -type d -name "*" | head -20
```

### 3.3 데이터 탐색 및 이해
```python
import os
import pandas as pd
from pathlib import Path

# 데이터 구조 파악
data_root = Path("/content/eye_disease_data")

# 클래스별 이미지 수 확인
class_counts = {}
for class_dir in data_root.iterdir():
    if class_dir.is_dir():
        image_count = len(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))
        class_counts[class_dir.name] = image_count

print("클래스별 이미지 수:")
for class_name, count in sorted(class_counts.items()):
    print(f"  {class_name}: {count}개")
```

---

## Step 4: 모델 학습 코드 작성

### 4.1 데이터 전처리 함수
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# DuoPet AI와 동일한 전처리 설정
IMG_SIZE = 224
BATCH_SIZE = 32

# ImageNet 정규화 값
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def preprocess_input(x):
    """DuoPet AI와 동일한 전처리 적용"""
    x = x / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x

# 데이터 증강 설정 (학습용)
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.15,
    validation_split=0.2
)

# 검증/테스트용 (증강 없음)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)
```

### 4.2 데이터 로더 생성
```python
# 학습 데이터 로더
train_generator = train_datagen.flow_from_directory(
    data_root,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# 검증 데이터 로더
validation_generator = val_datagen.flow_from_directory(
    data_root,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# 클래스 정보 저장
class_indices = train_generator.class_indices
num_classes = len(class_indices)
print(f"총 클래스 수: {num_classes}")
print(f"클래스 매핑: {class_indices}")

# 클래스 이름 저장 (나중에 사용)
import json
class_names = {v: k for k, v in class_indices.items()}
with open('/content/class_names.json', 'w', encoding='utf-8') as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)
```

### 4.3 모델 아키텍처 정의
```python
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

def create_eye_disease_model(num_classes):
    """
    DuoPet AI의 다른 모델들과 일관된 아키텍처 사용
    EfficientNetB0 기반 전이학습 모델
    """
    # 사전학습된 EfficientNetB0 (ImageNet weights)
    base_model = EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Fine-tuning을 위해 상위 레이어만 학습 가능하게 설정
    base_model.trainable = True
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # 커스텀 분류 헤드
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    return model

# 모델 생성
model = create_eye_disease_model(num_classes)
model.summary()
```

### 4.4 학습 설정
```python
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

# 옵티마이저 설정
optimizer = Adam(learning_rate=0.0001)

# 모델 컴파일
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# 콜백 설정
checkpoint_path = "/content/drive/MyDrive/DuoPet_AI_Training/models/eye_disease_best_model.h5"
callbacks = [
    ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]
```

---

## Step 5: 모델 학습 실행

### 5.1 학습 시작
```python
# 에포크 수 설정
EPOCHS = 50

# 학습 실행
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

print("학습 완료!")
```

### 5.2 학습 과정 시각화
```python
import matplotlib.pyplot as plt

def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 정확도 그래프
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # 손실 그래프
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/DuoPet_AI_Training/training_history.png')
    plt.show()

plot_training_history(history)
```

---

## Step 6: 모델 평가 및 저장

### 6.1 모델 평가
```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 최종 모델 로드
best_model = tf.keras.models.load_model(checkpoint_path)

# 테스트 데이터로 평가
test_generator = val_datagen.flow_from_directory(
    data_root,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# 예측 수행
predictions = best_model.predict(test_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# 분류 리포트
print("\n=== 분류 성능 리포트 ===")
print(classification_report(y_true, y_pred, 
                          target_names=list(class_names.values())))

# 혼동 행렬
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(class_names.values()),
            yticklabels=list(class_names.values()))
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/DuoPet_AI_Training/confusion_matrix.png')
plt.show()
```

### 6.2 다양한 형식으로 모델 저장
```python
# 1. Keras H5 형식 (이미 저장됨)
print(f"H5 모델 저장 위치: {checkpoint_path}")

# 2. TensorFlow SavedModel 형식 (권장)
savedmodel_path = "/content/drive/MyDrive/DuoPet_AI_Training/models/eye_disease_savedmodel"
best_model.save(savedmodel_path)
print(f"SavedModel 저장 위치: {savedmodel_path}")

# 3. 모델 설정 파일 생성
config = {
    "model_type": "eye_disease_classification",
    "architecture": "EfficientNetB0",
    "input_shape": [224, 224, 3],
    "num_classes": num_classes,
    "class_names": class_names,
    "preprocessing": {
        "normalization": "imagenet",
        "mean": IMAGENET_MEAN.tolist(),
        "std": IMAGENET_STD.tolist()
    },
    "training_info": {
        "dataset": "153.반려동물 안구질환 데이터",
        "epochs": EPOCHS,
        "best_val_accuracy": float(max(history.history['val_accuracy'])),
        "final_val_loss": float(history.history['val_loss'][-1])
    }
}

config_path = "/content/drive/MyDrive/DuoPet_AI_Training/models/eye_disease_config.json"
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
print(f"설정 파일 저장 위치: {config_path}")
```

### 6.3 모델 테스트 함수
```python
def test_model_inference(model_path, test_image_path):
    """모델 추론 테스트"""
    import cv2
    
    # 모델 로드
    model = tf.keras.models.load_model(model_path)
    
    # 이미지 로드 및 전처리
    img = cv2.imread(test_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    
    # 예측
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    print(f"예측 클래스: {class_names[predicted_class]}")
    print(f"신뢰도: {confidence:.2%}")
    
    return predicted_class, confidence

# 테스트 실행 (예시)
# test_model_inference(checkpoint_path, "/path/to/test/image.jpg")
```

---

## Step 7: DuoPet AI 프로젝트에 통합

### 7.1 모델 파일 다운로드
1. Google Drive에서 다음 파일들을 다운로드:
   - `eye_disease_best_model.h5`
   - `eye_disease_savedmodel/` (폴더 전체)
   - `eye_disease_config.json`
   - `class_names.json`

2. 로컬 저장 위치:
   ```
   D:\final_project\DuoPet_AI\models\health_diagnosis\eye_disease\
   ├── eye_disease_model.h5
   ├── eye_disease_savedmodel/
   ├── config.json
   └── class_names.json
   ```

### 7.2 DuoPet AI 코드 수정
```python
# services/health_diagnosis/predict.py에 추가할 코드

class EyeDiseasePredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        
        # 클래스 이름 로드
        with open(os.path.join(os.path.dirname(model_path), 'class_names.json'), 'r', encoding='utf-8') as f:
            self.class_names = json.load(f)
        
        # 설정 로드
        with open(os.path.join(os.path.dirname(model_path), 'config.json'), 'r', encoding='utf-8') as f:
            self.config = json.load(f)
    
    def predict(self, image_path):
        """안구질환 예측"""
        # 이미지 전처리
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = self._preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        
        # 예측
        predictions = self.model.predict(img)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        return {
            "disease_type": self.class_names[str(predicted_class)],
            "confidence": confidence,
            "all_predictions": {
                self.class_names[str(i)]: float(predictions[0][i]) 
                for i in range(len(predictions[0]))
            }
        }
    
    def _preprocess_input(self, x):
        """ImageNet 정규화"""
        x = x / 255.0
        x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return x
```

---

## 문제 해결 가이드

### 🔧 일반적인 문제와 해결방법

#### 1. GPU 메모리 부족
```python
# 배치 크기 줄이기
BATCH_SIZE = 16  # 32에서 16으로 감소

# 또는 mixed precision 사용
from tensorflow.keras.mixed_precision import Policy
policy = Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

#### 2. 학습이 수렴하지 않음
```python
# Learning rate 조정
optimizer = Adam(learning_rate=0.00001)  # 더 작은 값으로

# 또는 더 많은 데이터 증강
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,  # 증가
    width_shift_range=0.3,  # 증가
    height_shift_range=0.3,  # 증가
    horizontal_flip=True,
    vertical_flip=True,  # 추가
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],  # 추가
    validation_split=0.2
)
```

#### 3. 과적합 문제
```python
# Dropout 증가
x = layers.Dropout(0.6)(x)  # 0.5에서 0.6으로

# L2 정규화 추가
x = layers.Dense(256, activation='relu', 
                 kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
```

#### 4. 클래스 불균형
```python
# 클래스 가중치 계산
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))

# 학습 시 적용
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    class_weight=class_weight_dict,  # 추가
    callbacks=callbacks
)
```

---

## 📝 체크리스트

학습을 시작하기 전에 확인하세요:

- [ ] Google Colab GPU 런타임 설정 완료
- [ ] Google Drive 마운트 성공
- [ ] 데이터셋 압축 파일 업로드 완료
- [ ] TensorFlow 2.14.0 설치 확인
- [ ] 데이터 구조 및 클래스 확인
- [ ] 모델 아키텍처 정의 완료
- [ ] 학습 파라미터 설정 (epochs, batch_size 등)
- [ ] 저장 경로 설정 완료

---

## 🎉 축하합니다!

이제 DuoPet AI 안구질환 진단 모델 학습을 완료했습니다. 학습된 모델은 기존 피부질환 모델과 동일한 형식으로 저장되어 DuoPet AI 서비스에 즉시 통합할 수 있습니다.

### 다음 단계:
1. 모델 성능 검증
2. API 엔드포인트 구현
3. 프론트엔드 연동
4. 실제 서비스 배포

질문이나 문제가 있으면 언제든지 문의해 주세요! 🐶🐱