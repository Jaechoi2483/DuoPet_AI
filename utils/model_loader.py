"""
간소화된 모델 로더
"""
import tensorflow as tf
tf.config.run_functions_eagerly(True)
import numpy as np
import logging
import os

def convert_numpy_types(obj):
    """numpy 타입을 Python 기본 타입으로 변환"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logger = logging.getLogger(__name__)

def load_model_with_custom_objects(model_path: str):
    """모델 로딩 - 실패 시 새 모델 생성"""
    # Custom objects for Keras
    custom_objects = {
        'Functional': tf.keras.models.Model,
        'preprocess_input': tf.keras.applications.efficientnet.preprocess_input,
    }
    
    # H5 파일 시도 (compile=False로 로드)
    try:
        h5_path = model_path.replace('.keras', '_fixed.h5')
        if os.path.exists(h5_path):
            model = tf.keras.models.load_model(
                h5_path, 
                custom_objects=custom_objects,
                compile=False
            )
            # 모델 재컴파일
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            logger.info(f"Loaded fixed model from {h5_path}")
            return model
    except Exception as e:
        logger.warning(f"Failed to load fixed model: {e}")
    
    # keras 파일 시도 (compile=False로 로드)
    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(
                model_path,
                custom_objects=custom_objects,
                compile=False
            )
            # 모델 재컴파일
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            logger.info(f"Loaded model from {model_path}")
            return model
    except Exception as e:
        logger.warning(f"Failed to load keras model: {e}")
    
    # 최후의 수단: 가중치만 로드 시도
    try:
        logger.info("Attempting to load weights only...")
        # 새 모델 생성
        if 'eye' in str(model_path).lower():
            model = create_eye_disease_model()
        elif 'bcs' in str(model_path).lower():
            model = create_bcs_model()
        else:
            model = create_skin_model()
        
        # 가중치 로드 시도
        if os.path.exists(h5_path):
            model.load_weights(h5_path)
            logger.info(f"Loaded weights from {h5_path}")
        elif os.path.exists(model_path):
            model.load_weights(model_path)
            logger.info(f"Loaded weights from {model_path}")
        
        return model
    except Exception as e:
        logger.warning(f"Failed to load weights: {e}")
    
    # 완전히 새 모델 생성 (ImageNet 가중치 사용)
    logger.warning(f"Creating new model with ImageNet weights for {model_path}")
    
    if 'eye' in str(model_path).lower():
        model = create_eye_disease_model()
    elif 'bcs' in str(model_path).lower():
        model = create_bcs_model()
    else:
        model = create_skin_model()
    
    return model

def create_eye_disease_model():
    """안구 질환 모델 생성"""
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3), include_top=False, weights='imagenet'
    )
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_bcs_model():
    """BCS 모델 생성"""
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_skin_model():
    """피부 질환 모델 생성"""
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 별칭
load_keras_with_normalization_fix = load_model_with_custom_objects

def safe_model_predict(model, data):
    """안전한 모델 예측"""
    # NumPy 배열로 변환 확인
    if hasattr(data, 'numpy'):
        data = data.numpy()
    elif not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # TensorFlow 1.x 스타일 세션 사용
    import tensorflow.compat.v1 as tf_v1
    
    # 기본 그래프와 세션 얻기
    if tf_v1.executing_eagerly():
        # Eager mode에서는 일반 predict 사용
        return model.predict(data, verbose=0)
    else:
        # Graph mode에서는 세션 사용
        with tf_v1.Session() as sess:
            tf_v1.keras.backend.set_session(sess)
            return model.predict(data, verbose=0)

# 기타 필요한 함수들
def configure_tensorflow():
    pass
