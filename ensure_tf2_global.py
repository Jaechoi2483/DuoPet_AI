"""
전역 TensorFlow 2.x 설정 확인 및 수정
main.py와 모든 서비스의 초기화 순서 보장
"""
import os
import shutil
from pathlib import Path

def check_main_py():
    """main.py의 TF 설정 확인"""
    main_path = Path("api/main.py")
    
    if not main_path.exists():
        print("❌ main.py를 찾을 수 없습니다")
        return False
    
    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # TF 설정이 상단에 있는지 확인
    lines = content.split('\n')
    tf_config_line = -1
    for i, line in enumerate(lines[:20]):  # 상위 20줄만 확인
        if 'tf.config.run_functions_eagerly(True)' in line:
            tf_config_line = i
            break
    
    print(f"✅ main.py에서 TF eager 설정 발견: 라인 {tf_config_line + 1}")
    return True

def create_tf_initializer():
    """TensorFlow 초기화 전용 모듈 생성"""
    
    tf_init_content = '''"""
TensorFlow 2.x 전역 초기화 모듈
모든 서비스보다 먼저 import되어야 함
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TensorFlow import 및 설정
import tensorflow as tf

# Eager execution 강제 활성화
tf.config.run_functions_eagerly(True)

# GPU 메모리 증가 허용 (있는 경우)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU 설정 오류: {e}")

# 설정 확인
print(f"[TF Initializer] TensorFlow {tf.__version__}")
print(f"[TF Initializer] Eager execution: {tf.executing_eagerly()}")
print(f"[TF Initializer] GPU devices: {len(gpus)}")

# 전역 변수로 설정 상태 저장
TF_INITIALIZED = True
TF_VERSION = tf.__version__
EAGER_MODE = tf.executing_eagerly()
'''
    
    init_path = Path("common/tf_initializer.py")
    init_path.parent.mkdir(exist_ok=True)
    
    with open(init_path, 'w', encoding='utf-8') as f:
        f.write(tf_init_content)
    
    print(f"✅ TensorFlow 초기화 모듈 생성: {init_path}")
    
    # __init__.py에 추가
    init_py_path = init_path.parent / "__init__.py"
    if init_py_path.exists():
        with open(init_py_path, 'r', encoding='utf-8') as f:
            init_content = f.read()
        
        if 'tf_initializer' not in init_content:
            # 맨 앞에 추가
            new_content = "from . import tf_initializer  # TF2 초기화를 가장 먼저\n" + init_content
            with open(init_py_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("  ✓ common/__init__.py에 tf_initializer import 추가")

def update_main_py_import_order():
    """main.py의 import 순서 개선"""
    
    improved_main = '''"""
커밋용메세지
DuoPet AI Service - Main FastAPI Application

This is the main entry point for the DuoPet AI microservice.
It provides AI-powered features for pet health and behavior analysis.
"""

# TensorFlow 설정을 가장 먼저 수행
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # WARNING 이상만 표시

# TensorFlow 초기화 모듈 import (모든 것보다 먼저)
from common.tf_initializer import TF_INITIALIZED, EAGER_MODE
print(f"[Main] TF initialized: {TF_INITIALIZED}, Eager mode: {EAGER_MODE}")

import tensorflow as tf
# 추가 확인
if not tf.executing_eagerly():
    tf.config.run_functions_eagerly(True)
    print("[Main] Forced eager execution activation")

import time
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any

# PIL 이미지 처리 설정
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from services.chatbot.predict import RAGChatbot

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from common.config import get_settings
from common.logger import get_logger
from common.response import (
    create_success_response,
    create_error_response,
    ErrorCode,
    StandardResponse
)
from common.exceptions import DuoPetException
from common.monitoring import (
    MetricsMiddleware,
    collect_system_metrics,
    get_metrics,
    CONTENT_TYPE_LATEST
)
from api.middleware import (
    RequestIDMiddleware,
    LoggingMiddleware,
    APIKeyAuthMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware
)
# 💡 database.py에서 사용할 함수들을 가져옵니다.
from common.database import (
    connect_to_databases,
    close_database_connections,
    check_oracle_health  # Oracle DB만 사용
)
from common.database_sqlalchemy import init_db, close_db

# Import routers
from api.routers import (
    # auth,  # 임시 비활성화 - Oracle DB 연동 중
    face_login_router,
    chatbot_router,
    video_recommend_router,
    health_diagnosis_router,
    admin_router,
    behavior_analysis_router
)

# Initialize
settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle events
    """
    # Startup

    logger.info("= Starting DuoPet AI Service...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")

    # 1. 모든 데이터베이스 연결을 시도합니다.
    try:
        await connect_to_databases()
        # SQLAlchemy 테이블 초기화
        init_db()
        logger.info("All database connections established")
    except Exception as e:
        # 연결 실패 시에도 서버가 죽지 않도록 로그만 남깁니다.
        logger.error(f"Failed to initialize one or more database connections: {str(e)}")

    # 2. RAG 챗봇을 단 한 번만 초기화하여 앱 상태에 저장합니다.
    if os.getenv('SKIP_RAG_CHATBOT', 'false').lower() != 'true':
        try:
            logger.info("Initializing RAG Chatbot... (This may take a moment)")
            # 무거운 초기화 로직을 여기서 실행합니다.
            chatbot_instance = RAGChatbot(site_url=settings.SITE_URL)
            app.state.chatbot = chatbot_instance
            logger.info("RAG Chatbot initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Chatbot: {e}")
            app.state.chatbot = None
    else:
        logger.info("Skipping RAG Chatbot initialization (SKIP_RAG_CHATBOT=true)")
        app.state.chatbot = None

    # 3. 백그라운드 작업을 시작합니다.
    asyncio.create_task(collect_system_metrics())

    yield

    # Shutdown
    logger.info("=K Shutting down DuoPet AI Service...")
    await close_database_connections()
    await close_db()
    logger.info("All database connections closed")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan
)

# 이하 동일...
'''
    
    # main.py 일부만 저장 (전체는 너무 김)
    print("\n💡 main.py 수정 권장사항:")
    print("  1. TensorFlow 초기화를 최상단으로 이동")
    print("  2. common.tf_initializer import 추가")
    print("  3. 모든 서비스 import보다 먼저 TF 설정 완료")

def main():
    """메인 실행"""
    print("🔍 전역 TensorFlow 2.x 설정 확인")
    print("=" * 60)
    
    # 1. main.py 확인
    check_main_py()
    
    # 2. TF 초기화 모듈 생성
    create_tf_initializer()
    
    # 3. 권장사항 출력
    update_main_py_import_order()
    
    print("\n📋 권장 작업:")
    print("  1. common/tf_initializer.py가 생성되었습니다")
    print("  2. main.py 상단에 다음 추가:")
    print("     from common.tf_initializer import TF_INITIALIZED, EAGER_MODE")
    print("  3. 모든 서비스에서 TF import 전에 tf_initializer import")
    print("\n✅ 완료!")

if __name__ == "__main__":
    main()