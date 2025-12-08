@echo off
REM ========================================
REM PIPELINE ĐẦY ĐỦ - XỬ LÝ DỮ LIỆU TỪ ĐẦU ĐẾN CUỐI
REM ========================================

REM Store temp files on D: to avoid C: running out of space
set "TMP=D:\tmp"
set "TEMP=D:\tmp"
if not exist "D:\tmp" mkdir "D:\tmp"

echo.
echo ========================================
echo MATCHA-TTS DATA PROCESSING PIPELINE
echo ========================================
echo.

REM Kiểm tra virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run: python -m venv venv
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

echo.
echo ========================================
echo STEP 1: REMOVE SILENCE (VAD)
echo ========================================
echo Input:  data/raw/*.mp3
echo Output: data/vad/*.wav
echo.
pause
python scripts\remove_silence.py
if errorlevel 1 (
    echo [ERROR] Step 1 failed!
    pause
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
pause
python scripts\transcribe_cut.py
if errorlevel 1 (
    echo [ERROR] Step 2 failed!
    pause
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
    pause
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
pause
python scripts\cleaner.py
if errorlevel 1 (
    echo [ERROR] Step 4 failed!
    pause
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
pause
python scripts\split.py
if errorlevel 1 (
    echo [ERROR] Step 5 failed!
    pause
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
echo PIPELINE COMPLETED SUCCESSFULLY!
echo ========================================
echo.
echo You can now run training:
echo   python train_matcha_prosody.py
echo.
pause
