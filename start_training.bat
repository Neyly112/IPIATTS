@echo off
REM Quick start script cho Windows

echo ===============================================================================
echo       MATCHA-TTS TRAINING VOI PROSODY - QUICK START
echo ===============================================================================
echo.

REM Buoc 1: Kiem tra du lieu
echo [BUOC 1/4] Kiem tra du lieu...
python scripts\check_data.py --filelist data\99-audio-text-file-list\audio_text_train_filelist.txt.cleaned
if errorlevel 1 (
    echo.
    echo ❌ Du lieu khong hop le! Vui long kiem tra lai.
    pause
    exit /b 1
)
echo.

REM Buoc 2: Tinh data statistics (neu chua co)
echo [BUOC 2/4] Tinh toan data statistics...
if not exist data_stats.json (
    python matcha\utils\generate_data_statistics.py --filelist data\99-audio-text-file-list\audio_text_train_filelist.txt.cleaned --output data_stats.json
) else (
    echo Da ton tai data_stats.json, bo qua buoc nay.
)
echo.

REM Buoc 3: Hien thi cau hinh
echo [BUOC 3/4] Cau hinh hien tai:
echo   - PhoBERT: vinai/phobert-base
echo   - Batch size: 16 (co the giam xuong 8/4 neu thieu RAM)
echo   - Output: outputs/matcha_prosody
echo.

REM Buoc 4: Bat dau training
echo [BUOC 4/4] Bat dau training...
echo.
echo Luu y: Co the mo TensorBoard trong terminal khac:
echo   tensorboard --logdir outputs/matcha_prosody/logs
echo.
echo ===============================================================================
echo.

python train_matcha_prosody.py

echo.
echo ===============================================================================
echo Training hoan tat hoac bi gian doan.
echo.
pause
