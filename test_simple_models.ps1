# Test Simple Models Script

Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Testing Simple Models" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

$originalPath = Get-Location

try {
    Set-Location -Path "D:\final_project\DuoPet_AI"
    
    Write-Host "`nTesting newly created _simple.h5 models..." -ForegroundColor Yellow
    
    # Python script to test simple models
    $pythonScript = @"
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from pathlib import Path

def test_simple_model(model_path):
    try:
        model = tf.keras.models.load_model(str(model_path), compile=False)
        
        # Test with different inputs
        test_inputs = [
            np.zeros((1, 224, 224, 3), dtype=np.float32),
            np.ones((1, 224, 224, 3), dtype=np.float32),
            np.random.random((1, 224, 224, 3)).astype(np.float32)
        ]
        
        predictions = []
        for test_input in test_inputs:
            pred = model.predict(test_input, verbose=0)
            predictions.append(pred[0])
        
        # Calculate variation
        predictions = np.array(predictions)
        std_dev = np.std(predictions, axis=0)
        mean_std = np.mean(std_dev)
        
        print(f"Model: {model_path.name}")
        print(f"  Output shape: {model.output_shape}")
        print(f"  Variation: {mean_std:.4f}")
        print(f"  Status: {'WORKING' if mean_std > 0.001 else 'NO VARIATION'}")
        print(f"  Sample prediction: {predictions[2][:5]}...")
        print()
        
    except Exception as e:
        print(f"Error testing {model_path.name}: {e}")

# Test simple models
base_path = Path.cwd() / "models" / "health_diagnosis" / "skin_disease" / "classification"

simple_models = [
    base_path / "cat_binary" / "cat_binary_model_simple.h5",
    base_path / "dog_binary" / "dog_binary_model_simple.h5",
    base_path / "dog_multi_136" / "dog_multi_136_model_simple.h5"
]

print("Testing Simple Models (without CustomScaleLayer)")
print("=" * 60)

for model_path in simple_models:
    if model_path.exists():
        test_simple_model(model_path)
    else:
        print(f"Model not found: {model_path.name}")
"@

    # Save and run the Python script
    $pythonScript | Out-File -FilePath "test_simple_models_temp.py" -Encoding UTF8
    python test_simple_models_temp.py
    
    # Cleanup
    Remove-Item "test_simple_models_temp.py" -ErrorAction SilentlyContinue
    
    Write-Host "`nSimple models test completed!" -ForegroundColor Green
    Write-Host "If these models show good variation, update skin_disease_service.py to use them." -ForegroundColor Yellow
    
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
} finally {
    Set-Location -Path $originalPath
}

Write-Host "`nPress Enter to exit..."
Read-Host