# 피부질환 모델 수정 완료 스크립트

Write-Host "=================================" -ForegroundColor Green
Write-Host "피부질환 모델 수정 완료!" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

Write-Host "`n수정 내용:" -ForegroundColor Yellow
Write-Host "1. CustomScaleLayer가 리스트 입력을 처리하도록 수정됨" -ForegroundColor White
Write-Host "2. skin_disease_service.py가 _fixed.h5 및 _simple.h5 모델을 우선 사용하도록 업데이트됨" -ForegroundColor White

Write-Host "`n다음 단계:" -ForegroundColor Yellow
Write-Host "1. 다음 스크립트를 실행하여 모든 모델 테스트:" -ForegroundColor White
Write-Host "   .\run_test_all_models.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. CustomScaleLayer 없는 간단한 모델 생성:" -ForegroundColor White
Write-Host "   .\run_convert_simple.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. 백엔드 서버 재시작:" -ForegroundColor White
Write-Host "   cd D:\final_project\DuoPet_backend" -ForegroundColor Cyan
Write-Host "   mvn spring-boot:run" -ForegroundColor Cyan

Write-Host "`n중요:" -ForegroundColor Red
Write-Host "- 모델 테스트 후 작동하는 모델이 확인되면 서버를 재시작해주세요" -ForegroundColor Yellow
Write-Host "- _fixed.h5 모델이 없다면 run_convert_simple.ps1을 먼저 실행하세요" -ForegroundColor Yellow

Write-Host "`nEnter 키를 눌러 종료하세요..."
Read-Host