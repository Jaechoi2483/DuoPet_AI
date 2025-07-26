@echo off
echo ========================================
echo DuoPet AI - Ultra Fast Start Mode
echo ========================================

REM 콘다 환경 활성화
call conda activate duopet

REM 최적화 환경 변수 설정
set LAZY_LOAD_MODELS=true
set SKIP_BCS_MODEL=true
set SKIP_RAG_CHATBOT=true
set SKIP_UNUSED_MODELS=true
set TF_CPP_MIN_LOG_LEVEL=3

echo.
echo Starting server in ULTRA FAST mode...
echo (Some features will load on first use)
echo.

REM uvicorn 직접 실행 (reload 완전 비활성화)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level warning

pause