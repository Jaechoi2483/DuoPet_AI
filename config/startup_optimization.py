"""
서버 시작 속도 최적화 설정
"""

import os
import tensorflow as tf

def optimize_tensorflow():
    """TensorFlow 최적화 설정"""
    
    # GPU 메모리 증가 비활성화
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU 설정 오류: {e}")
    
    # CPU 스레드 수 제한 (시작 속도 향상)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    
    # TensorFlow 로그 레벨 설정
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR만 표시
    
    # JIT 컴파일 비활성화 (시작 속도 향상)
    tf.config.optimizer.set_jit(False)

def check_optimization_env():
    """최적화 환경 변수 확인"""
    optimizations = {
        'LAZY_LOAD_MODELS': os.getenv('LAZY_LOAD_MODELS', 'false'),
        'SKIP_BCS_MODEL': os.getenv('SKIP_BCS_MODEL', 'false'),
        'SKIP_RAG_CHATBOT': os.getenv('SKIP_RAG_CHATBOT', 'false'),
        'SKIP_UNUSED_MODELS': os.getenv('SKIP_UNUSED_MODELS', 'false'),
    }
    
    print("=== 최적화 설정 ===")
    for key, value in optimizations.items():
        print(f"{key}: {value}")
    print("==================")
    
    return optimizations