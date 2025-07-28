#!/usr/bin/env python3
"""
DuoPet AI 반려동물 안구질환 모델 학습 스크립트 (Windows 버전)
- 14개의 세분화된 질병을 5개의 상위 카테고리로 그룹화하여 분류
- Windows CPU 환경에 최적화
- TensorFlow 2.x 호환, .h5 형식 저장
"""

# 기본 라이브러리 import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import unicodedata
import json
import random
import platform
from pathlib import Path
from datetime import datetime
import time

# 데이터 처리 관련
import numpy as np
import pandas as pd
from tqdm import tqdm

# TensorFlow/Keras import
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

# scikit-learn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# 시각화 관련
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')  # Windows GUI 없이 실행

# 플랫폼별 한글 폰트 설정
def set_korean_font():
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    print(f"✅ 폰트 설정 완료: {plt.rcParams['font.family']}")

set_korean_font()

# 경로 설정
BASE_DIR = Path("/mnt/d/final_project/DuoPet_AI")
DATASET_PATH = Path("E:/DATA/153.반려동물 안구질환 데이터/eye_disease_dataset")
TRAIN_DIR = DATASET_PATH / "Training"
VAL_DIR = DATASET_PATH / "Validation"
MODEL_DIR = BASE_DIR / "models" / "health_diagnosis" / "eye_disease"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 시드 설정
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 하이퍼파라미터
IMG_SIZE = 224
BATCH_SIZE = 32  # Windows CPU에서도 32 가능
EPOCHS = 30  # 충분한 학습을 위해 30으로 증가
LEARNING_RATE = 5e-5

# 질병 그룹화 매핑 사전
DISEASE_GROUP_MAP = {
    # 수정체 질환
    '백내장': '수정체 질환',
    '핵경화': '수정체 질환',
    # 각막 질환
    '각막궤양': '각막 질환',
    '각막부골편': '각막 질환',
    '궤양성각막질환': '각막 질환',
    '비궤양성각막질환': '각막 질환',
    '색소침착성각막염': '각막 질환',
    '비궤양성각막염': '각막 질환',
    # 안검(눈꺼풀) 질환
    '안검염': '안검 질환',
    '안검종양': '안검 질환',
    '안검내반증': '안검 질환',
    # 결막/누관 질환
    '결막염': '결막 및 누관 질환',
    '유루증': '결막 및 누관 질환',
    # 안구 내부 질환
    '유리체변성': '안구 내부 질환'
}

VALID_DISEASES = list(DISEASE_GROUP_MAP.keys())

# 대분류 클래스 매핑
CLASS_MAP = {
    "각막 질환": 0,
    "결막 및 누관 질환": 1,
    "수정체 질환": 2,
    "안검 질환": 3,
    "안구 내부 질환": 4
}

class DataProcessor:
    """데이터 수집 및 전처리 클래스"""
    def __init__(self, train_dir, val_dir):
        self.train_dir = train_dir
        self.val_dir = val_dir

    def collect_image_paths_and_labels(self, root_dir):
        """재귀적으로 이미지 파일과 레이블 수집"""
        image_data = []
        print(f"📁 {root_dir} 디렉토리 검색 중...")
        
        # 재귀적으로 모든 jpg 파일 찾기
        image_files = list(Path(root_dir).rglob("*.jpg"))
        print(f"  총 {len(image_files)}개의 JPG 파일 발견")
        
        # 진행 표시를 위한 tqdm 사용
        for image_path in tqdm(image_files, desc=f"  {root_dir.name} 데이터 처리 중"):
            # 대응하는 JSON 파일 경로
            json_path = image_path.with_suffix('.json')
            
            # 경로에서 질병 이름 추출 (역순으로 검색)
            disease_name = None
            for part in reversed(image_path.parts):
                normalized_part = unicodedata.normalize('NFC', part)
                if normalized_part in VALID_DISEASES:
                    disease_name = normalized_part
                    break
            
            if json_path.exists() and disease_name:
                # 질병 이름을 대분류로 매핑
                grouped_label = DISEASE_GROUP_MAP.get(disease_name)
                if grouped_label:
                    image_data.append({
                        'path': str(image_path),
                        'label': grouped_label,
                        'original_disease': disease_name
                    })
        
        print(f"  유효한 데이터 {len(image_data)}개 수집 완료")
        return pd.DataFrame(image_data)

    def prepare_data(self):
        """학습 및 검증 데이터 준비"""
        train_df = self.collect_image_paths_and_labels(self.train_dir)
        val_df = self.collect_image_paths_and_labels(self.val_dir)
        
        return train_df, val_df

class ModelPipeline:
    """모델 학습 파이프라인"""
    def __init__(self, train_df, val_df, class_map, epochs):
        self.train_df = train_df
        self.val_df = val_df
        self.class_map = class_map
        self.num_classes = len(class_map)
        self.epochs = epochs

    def preprocess_input_custom(self, image):
        """커스텀 전처리 함수 - Lambda layer로 구현"""
        # EfficientNet 전처리를 직접 구현
        return image / 255.0

    def get_data_generators(self):
        """데이터 제너레이터 생성"""
        # 데이터 증강 설정
        train_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocess_input_custom,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocess_input_custom
        )

        # 클래스 인덱스 역매핑
        class_indices = {v: k for k, v in self.class_map.items()}
        
        # 제너레이터 생성
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=self.train_df,
            x_col='path',
            y_col='label',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            classes=list(self.class_map.keys()),
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_dataframe(
            dataframe=self.val_df,
            x_col='path',
            y_col='label',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            classes=list(self.class_map.keys()),
            shuffle=False
        )
        
        return train_generator, val_generator

    def build_network(self):
        """EfficientNetB0 기반 모델 구축"""
        print("\n🏗️ 모델 구축 중...")
        
        # 입력 레이어
        input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        
        # EfficientNetB0 백본
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_tensor=input_tensor
        )
        
        # Fine-tuning 설정: 마지막 40개 레이어만 학습
        base_model.trainable = True
        for layer in base_model.layers[:-40]:
            layer.trainable = False
        
        # 분류 헤드
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = Dropout(0.5)(x)
        output = Dense(self.num_classes, activation='softmax', name='disease_output')(x)
        
        # 모델 생성
        model = Model(inputs=input_tensor, outputs=output)
        model.summary()
        
        return model

    def train(self):
        """모델 학습"""
        train_gen, val_gen = self.get_data_generators()
        model = self.build_network()
        
        # 옵티마이저 설정
        optimizer = Adam(learning_rate=LEARNING_RATE)
        
        # 저장 경로 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_folder = MODEL_DIR / f"eye_disease_windows_{timestamp}"
        save_folder.mkdir(exist_ok=True)
        
        # 모델 체크포인트 경로
        best_model_path = str(save_folder / 'best_model.h5')
        
        # 콜백 설정
        callbacks = [
            ModelCheckpoint(
                best_model_path,
                verbose=1,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # 모델 컴파일
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall'),
                     tf.keras.metrics.AUC(name='auc')]
        )
        
        # 클래스 가중치 계산
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_gen.classes),
            y=train_gen.classes
        )
        class_weight_dict = dict(enumerate(class_weights))
        print("\n⚖️ 클래스 가중치 적용:")
        for idx, weight in class_weight_dict.items():
            class_name = list(self.class_map.keys())[idx]
            print(f"  {class_name}: {weight:.2f}")
        
        # 학습 시작
        print("\n🚀 학습 시작...")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.epochs,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # 결과 저장
        self.save_training_results(history, save_folder)
        
        return load_model(best_model_path), save_folder

    def save_training_results(self, history, save_folder):
        """학습 결과 저장 및 시각화"""
        # 학습 기록 저장
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(save_folder / 'training_history.csv', index=False)
        
        # 학습 곡선 그리기
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Train')
        axes[0, 1].plot(history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Train')
        axes[1, 0].plot(history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Train')
        axes[1, 1].plot(history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_folder / 'training_curves.png', dpi=300)
        plt.close()

def evaluate_model(model, val_generator, class_map, save_folder):
    """모델 평가 및 결과 저장"""
    print("\n📊 모델 평가 중...")
    
    # 예측 수행
    predictions = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes
    
    # 클래스 이름 목록
    class_names = list(class_map.keys())
    
    # 분류 리포트
    print("\n=== 질병 그룹 분류 성능 리포트 ===")
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)
    
    # 분류 리포트 저장
    with open(save_folder / 'classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Confusion Matrix 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix - 질병 그룹 분류')
    plt.xlabel('예측된 클래스')
    plt.ylabel('실제 클래스')
    plt.tight_layout()
    plt.savefig(save_folder / 'confusion_matrix.png', dpi=300)
    plt.close()
    
    # 성능 메트릭 계산
    accuracy = np.sum(y_pred == y_true) / len(y_true)
    print(f"\n전체 정확도: {accuracy:.4f}")

def save_model_files(model, save_folder, class_map):
    """최종 모델 파일들 저장"""
    print("\n💾 모델 파일 저장 중...")
    
    # 1. H5 형식으로 저장
    h5_path = MODEL_DIR / 'eye_disease_model.h5'
    model.save(str(h5_path), save_format='h5')
    print(f"✅ H5 모델 저장: {h5_path}")
    
    # 2. SavedModel 형식으로도 저장
    saved_model_path = MODEL_DIR / 'eye_disease_saved_model'
    model.save(str(saved_model_path))
    print(f"✅ SavedModel 저장: {saved_model_path}")
    
    # 3. 클래스 매핑 저장 (JSON 형식)
    class_map_for_json = {str(v): k for k, v in class_map.items()}
    with open(MODEL_DIR / 'class_map.json', 'w', encoding='utf-8') as f:
        json.dump(class_map_for_json, f, ensure_ascii=False, indent=2)
    print("✅ 클래스 맵 저장")
    
    # 4. 학습 정보 저장
    training_info = {
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_architecture": "EfficientNetB0",
        "input_shape": [IMG_SIZE, IMG_SIZE, 3],
        "num_classes": len(class_map),
        "preprocessing": "Division by 255.0",
        "framework": "TensorFlow " + tf.__version__,
        "platform": "Windows CPU",
        "disease_groups": list(class_map.keys()),
        "training_parameters": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "loss": "categorical_crossentropy"
        }
    }
    
    with open(MODEL_DIR / 'model_info.json', 'w', encoding='utf-8') as f:
        json.dump(training_info, f, ensure_ascii=False, indent=2)
    print("✅ 모델 정보 저장")

def main():
    """메인 실행 함수"""
    print("🐾 DuoPet AI 안구질환 모델 학습 시작 (Windows 버전)")
    print("=" * 70)
    
    # 데이터셋 확인
    if not DATASET_PATH.exists():
        print(f"❌ 데이터셋 경로를 찾을 수 없습니다: {DATASET_PATH}")
        return
    
    # 데이터 처리
    processor = DataProcessor(TRAIN_DIR, VAL_DIR)
    train_df, val_df = processor.prepare_data()
    
    if train_df.empty or val_df.empty:
        print("🚨 오류: 학습 또는 검증 데이터를 찾을 수 없습니다.")
        return
    
    # 데이터 분포 출력
    print(f"\n학습 데이터: {len(train_df)}개")
    print(f"검증 데이터: {len(val_df)}개")
    print(f"질병 그룹 수: {len(CLASS_MAP)}")
    print(f"그룹 목록: {list(CLASS_MAP.keys())}")
    
    print("\n📊 그룹별 데이터 분포 (학습 데이터):")
    print(train_df['label'].value_counts())
    
    print("\n📊 그룹별 데이터 분포 (검증 데이터):")
    print(val_df['label'].value_counts())
    
    # 모델 학습
    pipeline = ModelPipeline(train_df, val_df, CLASS_MAP, EPOCHS)
    model, save_folder = pipeline.train()
    
    # 모델 평가
    _, val_gen = pipeline.get_data_generators()
    evaluate_model(model, val_gen, CLASS_MAP, save_folder)
    
    # 최종 모델 저장
    save_model_files(model, save_folder, CLASS_MAP)
    
    print("\n✅ 학습 완료!")
    print(f"모델 저장 위치: {MODEL_DIR}")
    print(f"상세 결과 저장 위치: {save_folder}")
    
    # 메모리 정리
    K.clear_session()

if __name__ == "__main__":
    # GPU 비활성화 (CPU만 사용)
    tf.config.set_visible_devices([], 'GPU')
    
    # CPU 스레드 최적화
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    
    main()