# Convert to Simple Models Without CustomScaleLayer

Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Simple Model Conversion Start" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Save current directory
$originalPath = Get-Location

try {
    # Navigate to DuoPet_AI folder
    Set-Location -Path "D:\final_project\DuoPet_AI"
    
    Write-Host "`n1. Converting to simple models without CustomScaleLayer..." -ForegroundColor Yellow
    python convert_checkpoint_simple.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nConversion completed!" -ForegroundColor Green
        
        Write-Host "`n2. Testing converted models..." -ForegroundColor Yellow
        python test_fixed_skin_models.py
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`nAll tasks completed!" -ForegroundColor Green
        }
    } else {
        Write-Host "`nError occurred during conversion" -ForegroundColor Red
    }
    
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
} finally {
    # Return to original directory
    Set-Location -Path $originalPath
}

Write-Host "`nPress Enter to exit..."
Read-Host