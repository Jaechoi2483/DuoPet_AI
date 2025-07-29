# Test All Skin Disease Models Script

Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Testing All Skin Disease Models" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Save current directory
$originalPath = Get-Location

try {
    # Navigate to DuoPet_AI folder
    Set-Location -Path "D:\final_project\DuoPet_AI"
    
    Write-Host "`n1. Testing all H5 model files..." -ForegroundColor Yellow
    python test_all_models.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nTest completed!" -ForegroundColor Green
        Write-Host "Check the results above and update skin_disease_service.py with working models." -ForegroundColor Yellow
    } else {
        Write-Host "`nError occurred during testing" -ForegroundColor Red
    }
    
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
} finally {
    # Return to original directory
    Set-Location -Path $originalPath
}

Write-Host "`nPress Enter to exit..."
Read-Host