@echo off
REM ========================================
REM FULL AUTOMATED PIPELINE - SETUP TO TRAINING
REM ========================================
setlocal enabledelayedexpansion

REM Store temp files on D: to avoid C: running out of space
set "TMP=D:\tmp"
set "TEMP=D:\tmp"
if not exist "D:\tmp" mkdir "D:\tmp"

echo.
echo ========================================
echo MATCHA-TTS FULL AUTOMATED PIPELINE
echo ========================================
echo.

REM ========================================
REM STEP 0: CREATE & SETUP VIRTUAL ENVIRONMENT
REM ========================================
if not exist "venv\Scripts\activate.bat" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created!
)

REM Activate virtual environment
call venv\Scripts\activate.bat

echo.
echo ========================================
echo CHECKING SYSTEM REQUIREMENTS
echo ========================================

REM Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>nul
if errorlevel 1 (
    echo [INFO] PyTorch not installed yet, will install with CUDA support...
    set NEED_INSTALL=1
) else (
    echo [INFO] PyTorch already installed
    set NEED_INSTALL=0
)

REM Check if all dependencies installed
python -c "import lightning, transformers, librosa, phonemizer" 2>nul
if errorlevel 1 set NEED_INSTALL=1

if "%NEED_INSTALL%"=="1" (
    echo.
    echo ========================================
    echo INSTALLING DEPENDENCIES
    echo ========================================
    echo This will take 10-20 minutes...
    echo.
    
    REM Install PyTorch first (CUDA 11.8)
    echo [1/5] Installing PyTorch with CUDA 11.8...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    if errorlevel 1 (
        echo [ERROR] PyTorch installation failed!
        exit /b 1
    )
    
    REM Verify CUDA
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'✓ CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"
    if errorlevel 1 (
        echo [WARNING] CUDA not available, training will be VERY slow!
        echo Continue anyway? Press Ctrl+C to cancel, or
        pause
    )
    
    REM Install main dependencies
    echo [2/5] Installing main dependencies from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Main dependencies installation failed!
        exit /b 1
    )
    
    REM Install additional critical packages
    echo [3/5] Installing audio/ML packages...
    pip install matplotlib scipy librosa soundfile tensorboard einops
    pip install wget praat-parselmouth transformers conformer
    pip install cython setuptools wheel
    
    REM Build monotonic_align (fallback to Python if build fails)
    echo [4/5] Building monotonic_align...
    if exist "matcha\utils\monotonic_align\core.py" (
        echo [INFO] Using Python fallback for monotonic_align (already created^)
    ) else (
        echo [INFO] Attempting to build Cython extension...
        cd matcha\utils\monotonic_align
        python setup.py build_ext --inplace 2>nul
        if errorlevel 1 (
            echo [WARNING] Cython build failed (need Visual C++ Build Tools^)
            echo [INFO] Using Python fallback instead (slower but works^)
            cd ..\..\..
        ) else (
            echo [SUCCESS] Cython extension built!
            cd ..\..\..
        )
    )
    
    REM Verify phonemizer + eSpeak-NG
    echo [5/5] Checking eSpeak-NG (required for phonemizer^)...
    python -c "from phonemizer.backend import EspeakBackend; EspeakBackend('vi'); print('✓ eSpeak-NG OK')" 2>nul
    if errorlevel 1 (
        echo.
        echo [ERROR] eSpeak-NG not found!
        echo.
        echo Please install eSpeak-NG manually:
        echo   1. Download: https://github.com/espeak-ng/espeak-ng/releases
        echo   2. Install espeak-ng-X64.msi
        echo   3. Add to PATH: C:\Program Files\eSpeak NG\
        echo   4. Run this script again
        echo.
        pause
        exit /b 1
    )
    
    echo.
    echo [SUCCESS] All dependencies installed and verified!
    echo.
)

echo.
echo ========================================
echo STEP 1: REMOVE SILENCE (VAD)
echo ========================================
echo Input:  data/raw/*.mp3
echo Output: data/vad/*.wav
echo.
python scripts\remove_silence.py
if errorlevel 1 (
    echo [ERROR] Step 1 failed!
    exit /b 1
)
echo [SUCCESS] Step 1 completed!
echo.

echo ========================================
echo STEP 2: TRANSCRIBE ^& CUT INTO SENTENCES
echo ========================================
echo Input:  data/vad/*.wav
echo Output: data/subs/*.wav + _all.txt
echo.
echo WARNING: This step will take a long time (30-60 min)
python scripts\transcribe_cut.py
if errorlevel 1 (
    echo [ERROR] Step 2 failed!
    exit /b 1
)
echo [SUCCESS] Step 2 completed!
echo.

echo ========================================
echo STEP 3: PREPARE FOR CLEANING
echo ========================================
echo Copying transcription file...
copy data\99-audio-text-file-list\_all.txt data\99-audio-text-file-list\_all_corrected.txt >nul
if errorlevel 1 (
    echo [ERROR] Step 3 failed!
    exit /b 1
)
echo [SUCCESS] Step 3 completed!
echo.
echo NOTE: If you want to run spelling correction with PhoGPT:
echo   python scripts\correct_spelling_mistakes.py
echo   (Requires GPU with 8GB+ VRAM and 'pip install triton')
echo.

echo ========================================
echo STEP 4: NORMALIZE ^& ADD IPA PHONEMES
echo ========================================
echo Input:  _all_corrected.txt
echo Output: _all_normal_ipa.txt
echo (This script also embeds IPA so later splits will include phonemes.)
echo.
python scripts\cleaner.py
if errorlevel 1 (
    echo [ERROR] Step 4 failed!
    exit /b 1
)
echo [SUCCESS] Step 4 completed!
echo.

echo ========================================
echo STEP 5: SPLIT TRAIN/VAL/TEST (INCLUDES IPA)
echo ========================================
echo Input:  _all_normal_ipa.txt
echo Output: audio_text_*.txt and .txt.cleaned with audio|text|ipa
echo.
python scripts\split.py
if errorlevel 1 (
    echo [ERROR] Step 5 failed!
    exit /b 1
)
echo [SUCCESS] Step 5 completed!
echo.

echo ========================================
echo STEP 6: VALIDATE DATA
echo ========================================
python scripts\check_data.py --filelist data\99-audio-text-file-list\audio_text_train.txt.cleaned
python scripts\check_data.py --filelist data\99-audio-text-file-list\audio_text_val.txt.cleaned
python scripts\check_data.py --filelist data\99-audio-text-file-list\audio_text_test.txt.cleaned
echo.

echo ========================================
echo STEP 6.5: GENERATE DATA STATISTICS
echo ========================================
echo Calculating dataset statistics (mean/std for mel normalization^)...
python matcha\utils\generate_data_statistics.py --filelist data\99-audio-text-file-list\audio_text_train.txt.cleaned
if errorlevel 1 (
    echo [WARNING] Statistics generation failed - will use default values
) else (
    echo [SUCCESS] Statistics saved!
)
echo.

echo ========================================
echo DATA PIPELINE COMPLETED!
echo ========================================
echo.

echo ========================================
echo STEP 7: TRAIN MODEL
echo ========================================
echo Training Matcha-TTS with Vietnamese Prosody...
echo This will take several hours/days depending on GPU.
echo.
python train_matcha_prosody.py
if errorlevel 1 (
    echo [ERROR] Training failed!
    exit /b 1
)
echo [SUCCESS] Training completed!
echo.

echo ========================================
echo STEP 8: TEST CHECKPOINT
echo ========================================
echo Testing trained model checkpoint...
echo.
python test_checkpoint.py
if errorlevel 1 (
    echo [ERROR] Testing failed!
    exit /b 1
)
echo [SUCCESS] Testing completed!
echo.

echo ========================================
echo ALL STEPS COMPLETED SUCCESSFULLY!
echo ========================================
echo.
echo Pipeline finished:
echo   1. Data processing ✓
echo   2. Model training ✓
echo   3. Checkpoint testing ✓
echo.
echo Check logs and checkpoints in output directories.
