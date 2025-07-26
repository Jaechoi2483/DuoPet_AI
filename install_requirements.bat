@echo off
echo Installing required packages for model fixing...
echo.

REM Activate conda environment
call conda activate duopet

REM Install h5py
echo Installing h5py...
pip install h5py

REM Install other potentially missing packages
echo Installing additional packages...
pip install zipfile36
pip install tensorflow

echo.
echo Installation complete!
echo You can now run: python fix_models.py
pause