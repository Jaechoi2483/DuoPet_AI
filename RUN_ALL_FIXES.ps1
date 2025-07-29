# 피부질환 모델 전체 수정 작업 실행 스크립트

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  DuoPet AI Skin Disease Model Fix" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

$originalPath = Get-Location

try {
    Set-Location -Path "D:\final_project\DuoPet_AI"
    
    # Step 1: Test all existing models
    Write-Host "`n[Step 1] Testing all existing H5 model files" -ForegroundColor Yellow
    Write-Host "Testing about 32 model files..." -ForegroundColor Gray
    python test_all_models.py
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error occurred during model testing" -ForegroundColor Red
    }
    
    # User confirmation
    Write-Host "`nPlease check the results above. Press Enter to continue..." -ForegroundColor Yellow
    Read-Host
    
    # Step 2: Create simple models without CustomScaleLayer
    Write-Host "`n[Step 2] Creating simple models without CustomScaleLayer" -ForegroundColor Yellow
    Write-Host "Generating _simple.h5 models from checkpoints..." -ForegroundColor Gray
    python convert_checkpoint_simple.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Simple model creation completed!" -ForegroundColor Green
    } else {
        Write-Host "Error occurred during model creation" -ForegroundColor Red
    }
    
    # Step 3: Test generated models
    Write-Host "`n[Step 3] Testing fixed models" -ForegroundColor Yellow
    python test_fixed_skin_models.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nAll tests completed!" -ForegroundColor Green
    }
    
    # Result summary
    Write-Host "`n================================================" -ForegroundColor Green
    Write-Host "  Task Completed - Result Summary" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Green
    
    Write-Host "`nCompleted tasks:" -ForegroundColor Green
    Write-Host "  1. Fixed CustomScaleLayer in skin_disease_service.py" -ForegroundColor White
    Write-Host "  2. Updated model load priority (_fixed, _simple first)" -ForegroundColor White
    Write-Host "  3. Tested all H5 model files" -ForegroundColor White
    Write-Host "  4. Created simple models without CustomScaleLayer (if needed)" -ForegroundColor White
    
    Write-Host "`nNext steps:" -ForegroundColor Yellow
    Write-Host "  1. Check models marked as 'WORKING' in test results" -ForegroundColor White
    Write-Host "  2. Restart backend server:" -ForegroundColor White
    Write-Host "     cd D:\final_project\DuoPet_backend" -ForegroundColor Cyan
    Write-Host "     mvn spring-boot:run" -ForegroundColor Cyan
    Write-Host "  3. Test skin disease diagnosis in frontend" -ForegroundColor White
    
    Write-Host "`nImportant notes:" -ForegroundColor Red
    Write-Host "  - dog_multi_136 actually has 7 classes (not 136)" -ForegroundColor Yellow
    Write-Host "  - cat_binary_model.h5 uses sigmoid output (0.5 threshold)" -ForegroundColor Yellow
    Write-Host "  - Do not use models with low variation (std < 0.001)" -ForegroundColor Yellow
    
} catch {
    Write-Host "Error occurred: $_" -ForegroundColor Red
} finally {
    Set-Location -Path $originalPath
}

Write-Host "`nPress Enter to exit..."
Read-Host