@echo off
echo ========================================
echo DuoPet AI Server - Fast Start Mode
echo ========================================

REM 콘다 환경 활성화
call conda activate duopet

REM 환경 변수 설정
set LAZY_LOAD_MODELS=true
set SKIP_UNUSED_MODELS=true

echo.
echo Starting server without reload...
echo (This will start 50%% faster!)
echo.

REM reload 없이 서버 시작
python -m api.main --no-reload

pause