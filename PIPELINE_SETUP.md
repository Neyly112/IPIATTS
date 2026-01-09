# 📋 PIPELINE XỬ LÝ DỮ LIỆU TỪ ĐẦU ĐẾN CUỐI

## 🎯 TỔNG QUAN PIPELINE

```
data/raw/*.mp3 (audio gốc dài)
    ↓
[1] remove_silence.py → data/vad/*.wav (loại bỏ silence, giữ speech)
    ↓
[2] transcribe_cut.py → data/subs/*.wav (cắt thành từng câu + transcribe)
                     → data/99-audio-text-file-list/_all.txt
    ↓
[3] correct_spelling_mistakes.py → _all_corrected.txt (sửa lỗi chính tả - OPTIONAL)
    ↓
[4] cleaner.py → _all_normal_ipa.txt (chuẩn hóa + IPA phonemization)
    ↓
[5] split.py → train/val/test splits
    ↓
[6] generate_data_statistics.py → data_stats.json (mean/std cho mel normalization)
    ↓
[7] train_matcha_prosody.py → TRAINING MODEL
    ↓
[8] test_checkpoint.py → TEST & GENERATE AUDIO SAMPLES
    ↓
✅ DONE! Audio samples in outputs/test_samples/
```

## 🚀 CHẠY TỰ ĐỘNG (KHUYẾN NGHỊ)

**Cách nhanh nhất - chỉ 1 lệnh:**
```cmd
run_full_pipeline.bat
```

Script này sẽ **TỰ ĐỘNG**:
✅ Tạo virtual environment (nếu chưa có)
✅ Cài đặt PyTorch với CUDA 11.8
✅ Cài đặt TẤT CẢ dependencies (requirements.txt + additional packages)
✅ Build monotonic_align (hoặc dùng Python fallback nếu không có C++ compiler)
✅ Kiểm tra eSpeak-NG (bắt buộc cho phonemizer)
✅ Chạy toàn bộ pipeline xử lý dữ liệu (bước 1-6)
✅ Generate data statistics
✅ **Train model** (có thể tắt nếu chỉ muốn xử lý data)
✅ **Test checkpoint** (có thể tắt nếu chỉ muốn xử lý data)

**Hoàn toàn không cần nhấn nút gì - để qua đêm được!**

---

## 📋 CHẠY TỪNG BƯỚC THỦ CÔNG (Nếu cần kiểm soát chi tiết)

```
TextToSpeech/
├── data/
│   ├── raw/                    # ← BƯỚC 0: Đặt file audio gốc ở đây
│   │   ├── voice1.mp3
│   │   ├── voice2.mp3
│   │   └── ...
│   │
│   ├── vad/                    # ← BƯỚC 1: Sau remove_silence
│   │   ├── voice1.wav
│   │   ├── voice2.wav
│   │   └── ...
│   │
│   ├── subs/                   # ← BƯỚC 2: Sau transcribe_cut
│   │   ├── voice1_0001.wav
│   │   ├── voice1_0002.wav
│   │   ├── voice2_0001.wav
│   │   └── ...
│   │
│   └── 99-audio-text-file-list/
│       ├── _all.txt                           # BƯỚC 2 output
│       ├── _all_corrected.txt                 # BƯỚC 3 output
│       ├── _all_normal_ipa.txt                # BƯỚC 4 output
│       ├── audio_text_train.txt               # BƯỚC 5 output
│       ├── audio_text_train.txt.cleaned       # (với IPA)
│       ├── audio_text_val.txt
│       ├── audio_text_val.txt.cleaned
│       ├── audio_text_test.txt
│       └── audio_text_test.txt.cleaned
│
└── scripts/
    ├── remove_silence.py          # BƯỚC 1
    ├── transcribe_cut.py          # BƯỚC 2
    ├── correct_spelling_mistakes.py  # BƯỚC 3
    ├── cleaner.py                 # BƯỚC 4
    └── split.py                   # BƯỚC 5
```

---

## 🚀 HƯỚNG DẪN CHẠY TỪNG BƯỚC

### BƯỚC 0: Chuẩn bị file audio gốc

**Yêu cầu:**
- Đặt file audio (.mp3 hoặc .wav) vào thư mục `data/raw/`
- Tên file: `voice1.mp3`, `voice2.mp3`, `voice3.mp3`, ...
- Chất lượng: Bất kỳ sample rate nào (script sẽ tự chuyển đổi)

**Kiểm tra:**
```cmd
dir data\raw
```

Kết quả mong đợi:
```
voice1.mp3
voice2.mp3
voice3.mp3
...
```

---

### BƯỚC 1: Loại bỏ khoảng lặng (VAD - Voice Activity Detection)

**Mục đích:**
- Loại bỏ các đoạn im lặng (silence) trong audio gốc
- Chỉ giữ lại phần có giọng nói (speech)
- Chuyển đổi sang định dạng WAV, mono, 16kHz

**Chạy script:**
```cmd
python scripts\remove_silence.py
```

**Output:**
- Thư mục: `data/vad/`
- Format: WAV, mono, 16kHz
- Tên file: `voice1.wav`, `voice2.wav`, ...

**Ví dụ:**
```
Input:  data/raw/voice1.mp3 (5 phút, có nhiều khoảng lặng)
Output: data/vad/voice1.wav (3 phút, chỉ còn giọng nói)
```

**Kiểm tra:**
```cmd
dir data\vad
```

---

### BƯỚC 2: Transcribe và cắt thành từng câu

**Mục đích:**
- Dùng Whisper AI để tự động transcribe (chuyển giọng nói → text)
- Cắt file audio dài thành từng câu ngắn (sentence-level)
- Tạo file filelist `_all.txt` với format `audio_path|transcription`

**Chạy script:**
```cmd
python scripts\transcribe_cut.py
```

**Output:**
- Thư mục: `data/subs/`
- File audio: `voice1_0001.wav`, `voice1_0002.wav`, ... (mỗi file = 1 câu)
- File text: `data/99-audio-text-file-list/_all.txt`

**Format `_all.txt`:**
```
voice1_0001.wav|Xin chào, tôi là trợ lý ảo.
voice1_0002.wav|Hôm nay thời tiết đẹp quá.
voice2_0001.wav|Tôi yêu trí tuệ nhân tạo.
```

**Lưu ý:**
- Script tự động lọc bỏ:
  - Câu quá ngắn (< 2 từ)
  - Câu trùng lặp
  - Câu hallucination của Whisper
- Whisper chạy trên CPU với quantization (tiết kiệm RAM)

**Kiểm tra:**
```cmd
dir data\subs
type data\99-audio-text-file-list\_all.txt
```

---

### BƯỚC 3: Sửa lỗi chính tả (Optional - cần GPU mạnh)

**Mục đích:**
- Dùng PhoGPT (LLM tiếng Việt) để sửa lỗi chính tả trong transcription
- Whisper đôi khi transcribe sai → cần sửa

**Chạy script:**
```cmd
python scripts\correct_spelling_mistakes.py
```

**Yêu cầu:**
- GPU với 8GB+ VRAM (hoặc dùng quantization 4-bit)
- Hoặc skip bước này nếu transcription của Whisper đã tốt

**Output:**
- File: `data/99-audio-text-file-list/_all_corrected.txt`

**Format:**
```
voice1_0001.wav|Xin chào, tôi là trợ lý ảo.
voice1_0002.wav|Hôm nay thời tiết đẹp quá.
```

**Lưu ý:**
- Nếu skip bước này, đổi tên `_all.txt` → `_all_corrected.txt`:
  ```cmd
  copy data\99-audio-text-file-list\_all.txt data\99-audio-text-file-list\_all_corrected.txt
  ```

---

### BƯỚC 4: Chuẩn hóa text và thêm IPA phonemes

**Mục đích:**
- Chuẩn hóa text tiếng Việt (normalize)
- Chuyển đổi số → chữ (20 → hai mươi)
- Thêm IPA phonemes (phiên âm quốc tế)

**Chạy script:**
```cmd
python scripts\cleaner.py
```

**Output:**
- File: `data/99-audio-text-file-list/_all_normal_ipa.txt`

**Format:**
```
voice1_0001.wav|Xin chào, tôi là trợ lý ảo.|s i n   ch a o   t o i   l a   t r aw   l i   ao
voice1_0002.wav|Hôm nay thời tiết đẹp quá.|h aw m   n a j   t oi   t i ɛ t   d ɛ p   k w a
```

**Kiểm tra:**
```cmd
type data\99-audio-text-file-list\_all_normal_ipa.txt
```

---

### BƯỚC 5: Chia tập train/val/test

**Mục đích:**
- Chia dữ liệu thành 3 tập:
  - Train: 85% (dùng để training)
  - Validation: 10% (dùng để kiểm tra overfitting)
  - Test: 5% (dùng để đánh giá cuối cùng)

**Chạy script:**
```cmd
python scripts\split.py
```

**Output:**
```
data/99-audio-text-file-list/
├── audio_text_train.txt         (85% - text version)
├── audio_text_train.txt.cleaned (85% - IPA version)
├── audio_text_val.txt           (10% - text version)
├── audio_text_val.txt.cleaned   (10% - IPA version)
├── audio_text_test.txt          (5% - text version)
└── audio_text_test.txt.cleaned  (5% - IPA version)
```

**Format file `.txt`:**
```
voice1_0001.wav|Xin chào, tôi là trợ lý ảo.
voice1_0002.wav|Hôm nay thời tiết đẹp quá.
```

**Format file `.txt.cleaned` (dùng cho training):**
```
voice1_0001.wav|s i n   ch a o   t o i   l a   t r aw   l i   ao
voice1_0002.wav|h aw m   n a j   t oi   t i ɛ t   d ɛ p   k w a
```

**Kiểm tra:**
```cmd
type data\99-audio-text-file-list\audio_text_train.txt.cleaned
```

---

### BƯỚC 6: Generate data statistics (QUAN TRỌNG!)

**Mục đích:**
- Tính toán mean/std của mel-spectrogram từ toàn bộ training data
- Cần thiết cho mel normalization trong quá trình training
- Giúp model hội tụ nhanh và ổn định hơn

**Chạy script:**
```cmd
python matcha\utils\generate_data_statistics.py --filelist data\99-audio-text-file-list\audio_text_train.txt.cleaned
```

**Output:**
- File: `data_stats.json` (chứa mean/std values)

**Lưu ý:**
- Script `run_full_pipeline.bat` đã tự động chạy bước này
- Nếu skip, model sẽ dùng default values (có thể kém hơn)

---

## ✅ CHECKLIST ĐẦY ĐỦ

### Trước khi bắt đầu
- [ ] Python 3.8-3.11 đã cài
- [ ] Đã cài dependencies: `pip install -r requirements.txt`
- [ ] Đã cài eSpeak-NG
- [ ] File audio gốc trong `data/raw/`

### Chạy pipeline
- [ ] BƯỚC 1: `python scripts\remove_silence.py` → `data/vad/*.wav` OK
- [ ] BƯỚC 2: `python scripts\transcribe_cut.py` → `data/subs/*.wav` + `_all.txt` OK
- [ ] BƯỚC 3 (Optional): `python scripts\correct_spelling_mistakes.py` → `_all_corrected.txt` OK
  - Hoặc: `copy _all.txt _all_corrected.txt`
- [ ] BƯỚC 4: `python scripts\cleaner.py` → `_all_normal_ipa.txt` OK
- [ ] BƯỚC 5: `python scripts\split.py` → train/val/test splits OK
- [ ] BƯỚC 6: `python matcha\utils\generate_data_statistics.py` → `data_stats.json` OK

### Kiểm tra kết quả
- [ ] File `audio_text_train.txt.cleaned` tồn tại
- [ ] File `audio_text_val.txt.cleaned` tồn tại
- [ ] File `audio_text_test.txt.cleaned` tồn tại
- [ ] File `data_stats.json` tồn tại (mean/std values)
- [ ] Format: `audio_path|ipa_phonemes`
- [ ] Đường dẫn audio đúng (ví dụ: `voice1_0001.wav` nằm trong `data/subs/`)

---

## 🔧 TROUBLESHOOTING

### 1. BƯỚC 1: Lỗi "No module named 'torch'"
```cmd
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. BƯỚC 2: Lỗi "No module named 'whisper'"
```cmd
pip install openai-whisper
```

### 3. BƯỚC 2: Lỗi "Microsoft Visual C++ required" (monotonic_align)
**KHÔNG CẦN LO!** Script đã tự động tạo Python fallback.
- File `matcha/utils/monotonic_align/core.py` là pure Python version
- Chạy bình thường, chỉ chậm hơn Cython version một chút
- Không cần cài Visual C++ Build Tools

### 4. BƯỚC 3: GPU Out of Memory
Sửa trong `correct_spelling_mistakes.py`:
```python
QUANTIZATION = "4bit"  # Thay vì "float16"
```

Hoặc skip bước này:
```cmd
copy data\99-audio-text-file-list\_all.txt data\99-audio-text-file-list\_all_corrected.txt
```

### 5. BƯỚC 4: Lỗi "espeak-ng not found"
- Windows: Cài eSpeak-NG từ https://github.com/espeak-ng/espeak-ng/releases
- Script `run_full_pipeline.bat` sẽ tự động kiểm tra và thông báo
- Sửa trong `cleaner.py`:
  ```python
  from phonemizer.backend.espeak.wrapper import EspeakWrapper
  EspeakWrapper.set_library(r"C:\Program Files\eSpeak NG\libespeak-ng.dll")
  ```

### 6. BƯỚC 5: Lỗi "File not found: _all_corrected.txt"
Đảm bảo đã chạy BƯỚC 3 hoặc copy file:
```cmd
copy data\99-audio-text-file-list\_all.txt data\99-audio-text-file-list\_all_corrected.txt
```

### 7. Script `run_full_pipeline.bat` tự động dừng
- Kiểm tra log để xem bước nào lỗi
- Script sẽ hiển thị `[ERROR] Step X failed!` và dừng lại
- Sửa lỗi theo hướng dẫn, sau đó chạy lại từ bước đó

---

## 📊 KIỂM TRA CHẤT LƯỢNG DỮ LIỆU

Sau khi hoàn thành pipeline, kiểm tra:

```cmd
python scripts\check_data.py --filelist data\99-audio-text-file-list\audio_text_train.txt.cleaned
```

**Kết quả mong đợi:**
```
✓ Tổng số dòng: 8000
✓ Dòng hợp lệ: 8000
✓ File audio tồn tại: 8000

================================================================================
✅ FILELIST HỢP LỆ - Sẵn sàng để training!
```

---

## 🎯 SAU KHI HOÀN THÀNH PIPELINE

Bạn đã có:
- ✅ File audio đã xử lý (sentence-level, 22.05kHz, mono)
- ✅ Transcription với IPA phonemes
- ✅ Chia tập train/val/test

**Bước tiếp theo:**
1. Cập nhật config trong `train_matcha_prosody.py`:
   ```python
   CONFIG = {
       "train_filelist": "data/99-audio-text-file-list/audio_text_train.txt.cleaned",
       "val_filelist": "data/99-audio-text-file-list/audio_text_val.txt.cleaned",
       ...
   }
   ```

2. Chạy training:
   ```cmd
   python train_matcha_prosody.py
   ```

---

**CHÚC BẠN THÀNH CÔNG! 🎤✨**
