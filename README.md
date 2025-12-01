# 🍵 MATCHA-TTS VỚI PROSODY ANALYSIS (PHOBERT)

Hướng dẫn đầy đủ từ A-Z: Cài đặt môi trường → Chuẩn bị dữ liệu → Training → Sử dụng model

---

## 📋 MỤC LỤC

1. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
2. [Cài đặt môi trường](#cài-đặt-môi-trường)
3. [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
4. [Training model](#training-model)
5. [Sử dụng checkpoint](#sử-dụng-checkpoint)
6. [Troubleshooting](#troubleshooting)

---

## ⚙️ YÊU CẦU HỆ THỐNG

### Phần cứng tối thiểu
- **CPU**: 4 cores trở lên
- **RAM**: 16GB
- **GPU**: NVIDIA GPU với CUDA (khuyến nghị)
  - GTX 1660 (6GB VRAM) - Tối thiểu
  - RTX 3060 (12GB VRAM) - Khuyến nghị
  - RTX 4090 (24GB VRAM) - Tối ưu
- **Ổ cứng**: 50GB trống

### Phần mềm
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8 - 3.11 (khuyến nghị 3.11)
- **CUDA**: 11.8 hoặc 12.1 (nếu dùng GPU)
- **Git**: Để clone repository

---

## 🚀 CÀI ĐẶT MÔI TRƯỜNG

### BƯỚC 1: Clone repository

```bash
git clone https://github.com/Neyly112/IPIATTS.git
cd IPIATTS
git checkout ket_hop_LLM_prosody_PhoBert
```

### BƯỚC 2: Tạo Python virtual environment

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### BƯỚC 3: Cài đặt PyTorch (với CUDA)

**Kiểm tra CUDA version:**
```cmd
nvidia-smi
```

**Cài PyTorch (CUDA 11.8):**
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Cài PyTorch (CUDA 12.1):**
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Cài PyTorch (CPU only - không khuyến nghị):**
```cmd
pip install torch torchvision torchaudio
```

**Kiểm tra PyTorch:**
```cmd
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

Kết quả mong đợi:
```
PyTorch: 2.x.x+cu118
CUDA available: True
CUDA version: 11.8
```

### BƯỚC 4: Cài đặt TẤT CẢ dependencies (1 lệnh duy nhất!)

```cmd
pip install -r requirements.txt
```

**File `requirements.txt` đã bao gồm TẤT CẢ:**
- ✅ PyTorch (chú ý: cài PyTorch riêng ở BƯỚC 3)
- ✅ Lightning (training framework)
- ✅ Transformers (PhoBERT)
- ✅ Phonemizer (text → IPA)
- ✅ Underthesea (Vietnamese NLP)
- ✅ Librosa, SoundFile (audio processing)
- ✅ TensorBoard (monitoring)
- ✅ Tất cả dependencies khác

**Nếu gặp lỗi khi cài hàng loạt**, cài từng nhóm quan trọng:

```cmd
# Nhóm 1: Framework chính
pip install lightning==2.0.0
pip install transformers>=4.30.0

# Nhóm 2: Audio processing
pip install librosa soundfile praat-parselmouth

# Nhóm 3: Text & Phonemizer
pip install phonemizer underthesea num2words

# Nhóm 4: Utilities
pip install tensorboard einops tqdm numpy pandas matplotlib
```

### BƯỚC 5: Cài đặt eSpeak-NG (Phonemizer engine)

**⚠ LƯU Ý:** eSpeak-NG KHÔNG thể cài qua pip, phải cài riêng!

**Windows:**

1. Download từ: <https://github.com/espeak-ng/espeak-ng/releases>
2. Chọn file `espeak-ng-X64.msi` (64-bit)
3. Cài đặt vào `C:\Program Files\eSpeak NG\`
4. Kiểm tra:

   ```cmd
   "C:\Program Files\eSpeak NG\espeak-ng.exe" --version
   ```

**Linux (Ubuntu/Debian):**

```bash
sudo apt-get install espeak-ng
```

**macOS:**

```bash
brew install espeak-ng
```

**Kiểm tra phonemizer:**

```cmd
python -c "from phonemizer.backend import EspeakBackend; print('Phonemizer OK')"
```

### BƯỚC 6: Kiểm tra cài đặt hoàn chỉnh

```cmd
python -c "import torch; import lightning; import transformers; from phonemizer.backend import EspeakBackend; print('✅ Tất cả dependencies đã sẵn sàng!')"
```

---

## 📂 CHUẨN BỊ DỮ LIỆU

### ⚡ OPTION 1: Chạy Pipeline Tự Động (Khuyến nghị)

**Nếu bạn có file audio gốc chưa xử lý:**

1. **Đặt file audio vào thư mục:**
   ```cmd
   data\raw\voice1.mp3
   data\raw\voice2.mp3
   ...
   ```

2. **Chạy pipeline tự động:**
   ```cmd
   run_full_pipeline.bat
   ```

   Hoặc chạy từng bước thủ công:
   ```cmd
   python scripts\remove_silence.py          # Bước 1: Loại bỏ silence
   python scripts\transcribe_cut.py          # Bước 2: Transcribe + cắt câu
   python scripts\process_cleaner.py         # Bước 3: Chuẩn hóa + IPA
   python scripts\split.py                   # Bước 4: Chia train/val/test
   ```

3. **Xem hướng dẫn chi tiết:**
   - Đọc file: [PIPELINE_SETUP.md](PIPELINE_SETUP.md)

---

### 📝 OPTION 2: Sử dụng Dữ Liệu Có Sẵn

**Nếu bạn đã có file audio + transcription:**

#### BƯỚC 1: Cấu trúc thư mục

```
TextToSpeech/
├── data/
│   ├── subs/                # File audio đã cắt (sentence-level)
│   │   ├── voice1_0001.wav
│   │   ├── voice1_0002.wav
│   │   └── ...
│   └── 99-audio-text-file-list/
│       └── audio_text_train.txt.cleaned    # Filelist với IPA
```

#### BƯỚC 2: Format file filelist

File `.cleaned` cần có format: `audio_path|ipa_phonemes`

**Ví dụ:**
```
voice1_0001.wav|s i n   ch a o   t o i   l a   t r aw   l i   ao
voice1_0002.wav|h aw m   n a j   t oi   t i ɛ t   d ɛ p   k w a
```

**Lưu ý:**
- Đường dẫn audio: tương đối từ thư mục `data/subs/`
- Mỗi dòng: `audio_filename|ipa_phonemes`
- Encoding: UTF-8

#### BƯỚC 3: Kiểm tra dữ liệu

```cmd
python scripts\check_data.py --filelist data\99-audio-text-file-list\audio_text_train.txt.cleaned
```

**Kết quả mong đợi:**
```
✅ FILELIST HỢP LỆ - Sẵn sàng để training!
```

#### BƯỚC 4: Kiểm tra n_vocab

```cmd
python -c "from matcha.text.symbols import symbols; print(f'n_vocab = {len(symbols)}')"
```

Cập nhật giá trị vào `train_matcha_prosody.py`:
```python
CONFIG = {
    "n_vocab": 256,  # Thay bằng giá trị vừa kiểm tra
    ...
}
```

---

## 🏋️ TRAINING MODEL

### BƯỚC 1: Cập nhật config

Mở file `train_matcha_prosody.py` và kiểm tra:

```python
CONFIG = {
    # Đường dẫn dữ liệu (ĐÃ CÓ PHONEMES)
    "train_filelist": "data/99-audio-text-file-list/audio_text_train_filelist_with_phonemes.txt",
    "val_filelist": "data/99-audio-text-file-list/audio_text_val_filelist_with_phonemes.txt",
    
    # Model settings
    "n_vocab": 256,        # Số phonemes
    "n_spks": 1,           # 1 = single speaker, >1 = multi-speaker
    "batch_size": 16,      # Giảm xuống 8/4 nếu OOM
    "learning_rate": 1e-4,
    "max_epochs": 1000,    # Số epochs training
    
    # Prosody với PhoBERT
    "llm_model_name": "vinai/phobert-base",
    "prosody_dim": 256,
    
    # Output
    "output_dir": "outputs/matcha_prosody",
    
    # GPU/CPU
    "accelerator": "gpu",  # "gpu" hoặc "cpu"
}
```

**Tùy chỉnh:**
- `batch_size`: Giảm nếu GPU thiếu VRAM
- `max_epochs`: 1000 cho training đầy đủ, 10-100 để test
- `n_spks`: Số speaker trong dataset

### BƯỚC 2: Test nhanh (khuyến nghị lần đầu)

Sửa trong `train_matcha_prosody.py`:
```python
CONFIG = {
    ...
    "max_epochs": 10,  # Test 10 epochs trước
    "batch_size": 8,   # Batch size nhỏ để an toàn
}
```

Chạy:
```cmd
python train_matcha_prosody.py
```

**Thời gian test:** ~30 phút - 1 giờ

**Kiểm tra:**
- Không có lỗi → OK, tiếp tục training đầy đủ
- Có lỗi → Xem [Troubleshooting](#troubleshooting)

### BƯỚC 3: Training đầy đủ

Sửa lại config:
```python
CONFIG = {
    ...
    "max_epochs": 1000,
    "batch_size": 16,  # hoặc 8 tùy GPU
}
```

**Chạy training:**
```cmd
python train_matcha_prosody.py
```

**Hoặc dùng quick start:**
```cmd
start_training.bat
```

**Lưu ý:**
- Đừng tắt máy trong quá trình training
- Tắt chế độ Sleep/Hibernate
- Thời gian: 1-7 ngày tùy GPU và dataset

### BƯỚC 4: Theo dõi training với TensorBoard

Mở terminal mới và chạy:
```cmd
tensorboard --logdir outputs/matcha_prosody/logs
```

Truy cập: http://localhost:6006

**Metrics cần xem:**
- `train_loss` và `val_loss` - Phải giảm dần
- `dur_loss` - Duration prediction loss
- `prior_loss` - Encoder quality
- `diff_loss` - Decoder/CFM quality
- `learning_rate` - Learning rate schedule

### BƯỚC 5: Resume training (nếu bị gián đoạn)

Sửa config:
```python
CONFIG = {
    ...
    "resume_from_checkpoint": "outputs/matcha_prosody/checkpoints/last.ckpt",
}
```

Chạy lại:
```cmd
python train_matcha_prosody.py
```

### BƯỚC 6: Checkpoints

**Vị trí:**
```
outputs/matcha_prosody/
├── checkpoints/
│   ├── matcha-prosody-epoch=050-val_loss=0.234.ckpt
│   ├── matcha-prosody-epoch=100-val_loss=0.189.ckpt
│   ├── matcha-prosody-epoch=150-val_loss=0.145.ckpt  ← Best model
│   └── last.ckpt                                      ← Latest
└── logs/
    └── tensorboard_logs/
```

**Loại checkpoint:**
- `matcha-prosody-epoch=XXX-val_loss=Y.YYY.ckpt` - Top 3 best models
- `last.ckpt` - Checkpoint mới nhất (để resume)

**Early Stopping:**
Model tự động dừng nếu `val_loss` không giảm sau 50 epochs.

---

## 🎤 SỬ DỤNG CHECKPOINT

### BƯỚC 1: Load model từ checkpoint

```python
from matcha.models.matcha_tts import MatchaTTS
import torch

# Load model (prosody tự động được bật)
model = MatchaTTS.load_from_checkpoint(
    "outputs/matcha_prosody/checkpoints/matcha-prosody-epoch=150-val_loss=0.145.ckpt"
)
model.eval()

# Chuyển sang GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("✅ Model loaded!")
```

### BƯỚC 2: Chuẩn bị text input

```python
from matcha.text import text_to_sequence
from matcha.utils.utils import intersperse

# Text tiếng Việt
text = "xin chào, hôm nay tôi học về trí tuệ nhân tạo"

# Convert text → phoneme IDs
x = torch.tensor(
    intersperse(text_to_sequence(text, ["basic_cleaners_phothong"])[0], 0)
)[None].to(device)

x_lengths = torch.tensor([x.shape[-1]], device=device)

print(f"Input shape: {x.shape}")
```

### BƯỚC 3: Synthesize mel-spectrogram

```python
with torch.no_grad():
    output = model.synthesise(
        x,
        x_lengths,
        n_timesteps=10,      # Số ODE steps (10-50, càng nhiều càng chất lượng)
        temperature=0.667,    # Sampling temperature
        length_scale=1.0,     # Speaking rate (>1 = chậm, <1 = nhanh)
    )

mel = output["mel"]           # Mel-spectrogram
mel_lengths = output["mel_lengths"]
rtf = output["rtf"]           # Real-time factor

print(f"Mel shape: {mel.shape}")
print(f"RTF: {rtf:.4f}")
```

### BƯỚC 4: Convert mel → audio (với HiFi-GAN vocoder)

```python
from matcha.cli import load_vocoder, to_waveform
import soundfile as sf

# Load vocoder
vocoder, denoiser = load_vocoder(
    "hifigan_univ_v1",
    "path/to/vocoder/g_02500000",
    device
)

# Convert mel → waveform
audio = to_waveform(mel, vocoder, denoiser)

# Lưu file
sf.write("output.wav", audio.cpu().numpy(), 22050, "PCM_24")
print("✅ Đã lưu: output.wav")
```

### BƯỚC 5: Script hoàn chỉnh

```python
"""
Synthesis script - Sử dụng Matcha-TTS với Prosody
"""
import torch
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import text_to_sequence
from matcha.utils.utils import intersperse
from matcha.cli import load_vocoder, to_waveform
import soundfile as sf

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load model
model = MatchaTTS.load_from_checkpoint(
    "outputs/matcha_prosody/checkpoints/last.ckpt"
).to(device).eval()

# 3. Load vocoder
vocoder, denoiser = load_vocoder(
    "hifigan_univ_v1",
    "path/to/vocoder/g_02500000",
    device
)

# 4. Synthesize
text = "xin chào, đây là giọng nói tiếng việt với prosody tự nhiên"

x = torch.tensor(
    intersperse(text_to_sequence(text, ["basic_cleaners_phothong"])[0], 0)
)[None].to(device)
x_lengths = torch.tensor([x.shape[-1]], device=device)

with torch.no_grad():
    output = model.synthesise(x, x_lengths, n_timesteps=10)
    audio = to_waveform(output["mel"], vocoder, denoiser)

# 5. Save
sf.write("output.wav", audio.cpu().numpy(), 22050, "PCM_24")
print(f"✅ Saved: output.wav (RTF: {output['rtf']:.4f})")
```

---

## 🔧 TROUBLESHOOTING

### 1. CUDA Out of Memory

**Lỗi:**
```
RuntimeError: CUDA out of memory
```

**Giải pháp:**
```python
# Trong train_matcha_prosody.py
CONFIG = {
    "batch_size": 8,  # Giảm từ 16 → 8
    # hoặc
    "batch_size": 4,  # Giảm xuống 4
}
```

Hoặc thêm gradient accumulation:
```python
trainer = pl.Trainer(
    ...
    accumulate_grad_batches=4,
)
```

### 2. PhoBERT download thất bại

**Lỗi:**
```
Cannot download vinai/phobert-base
```

**Giải pháp:**
```cmd
python -c "from transformers import AutoModel; AutoModel.from_pretrained('vinai/phobert-base', cache_dir='./models')"
```

Download thủ công (~1GB), sau đó sửa config:
```python
"llm_model_name": "./models/vinai--phobert-base",
```

### 3. eSpeak-NG không tìm thấy

**Lỗi:**
```
espeak-ng not found
```

**Giải pháp (Windows):**
```python
# Trong scripts/add_phonemes.py
from phonemizer.backend.espeak.wrapper import EspeakWrapper
EspeakWrapper.set_library(r"C:\Program Files\eSpeak NG\libespeak-ng.dll")
```

### 4. DataModule lỗi

**Lỗi:**
```
TextMelDataModule not compatible
```

**Giải pháp:**
Kiểm tra file `matcha/data/text_mel_datamodule.py` có tương thích không:
```cmd
python -c "from matcha.data.text_mel_datamodule import TextMelDataModule; print('OK')"
```

Nếu lỗi, cần implement custom DataModule dựa trên `matcha/utils/data/ljspeech.py`.

### 5. Loss không giảm

**Nguyên nhân:**
- Data statistics sai
- Learning rate quá cao/thấp
- Filelist format sai

**Giải pháp:**
1. Kiểm tra filelist:
   ```cmd
   python scripts\check_data.py --filelist data/.../audio_text_train_filelist_with_phonemes.txt
   ```

2. Giảm learning rate:
   ```python
   "learning_rate": 5e-5,  # Thay vì 1e-4
   ```

3. Kiểm tra TensorBoard để debug

### 6. File audio không tìm thấy

**Lỗi:**
```
FileNotFoundError: audio file not found
```

**Giải pháp:**
- Kiểm tra đường dẫn trong filelist phải đúng
- Đường dẫn có thể là tương đối từ root project
- Ví dụ: `data/vad/voice5_0207.wav` (không phải `D:\BAOTRAN\...`)

---

## 📊 KIỂM TRA CHẤT LƯỢNG

### 1. Real-Time Factor (RTF)

```python
rtf = output["rtf"]
print(f"RTF: {rtf:.4f}")
```

- RTF < 0.05: Rất nhanh (real-time)
- RTF = 0.1-0.3: Chấp nhận được
- RTF > 1.0: Chậm

### 2. Mel Cepstral Distortion (MCD)

So sánh mel-spectrogram sinh ra với ground truth.

### 3. Listening Test (MOS Score)

Đánh giá chủ quan (1-5 điểm):
- 5 = Excellent
- 4 = Good
- 3 = Fair
- 2 = Poor
- 1 = Bad

### 4. Prosody Quality

Kiểm tra:
- ✅ Intonation (ngữ điệu)
- ✅ Stress (trọng âm)
- ✅ Rhythm (nhịp điệu)
- ✅ Naturalness (tự nhiên)

---

## 📚 TÀI LIỆU THAM KHẢO

- **Matcha-TTS Paper**: [arXiv:2309.03199](https://arxiv.org/abs/2309.03199)
- **PhoBERT**: [vinai/phobert-base](https://huggingface.co/vinai/phobert-base)
- **Matcha-TTS GitHub**: [shivammehta25/Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)

---

## 🎯 CHECKLIST ĐẦY ĐỦ

### Cài đặt môi trường
- [ ] Python 3.8-3.11 installed
- [ ] CUDA installed (nếu dùng GPU)
- [ ] PyTorch with CUDA installed
- [ ] Lightning installed
- [ ] Transformers installed
- [ ] Phonemizer + eSpeak-NG installed
- [ ] Kiểm tra: `python -c "import torch; print(torch.cuda.is_available())"`

### Chuẩn bị dữ liệu
- [ ] File audio (.wav) 22050Hz
- [ ] Filelist format: `audio|text`
- [ ] Fix filelist (nếu bị ngắt dòng)
- [ ] Thêm phonemes: `audio|text|phonemes`
- [ ] Kiểm tra với `check_data.py`
- [ ] Kiểm tra n_vocab

### Training
- [ ] Cập nhật config trong `train_matcha_prosody.py`
- [ ] Test với 10 epochs trước
- [ ] Training đầy đủ 1000 epochs
- [ ] Theo dõi với TensorBoard
- [ ] Checkpoints được lưu

### Sử dụng
- [ ] Load checkpoint thành công
- [ ] Synthesize text → mel
- [ ] Convert mel → audio với vocoder
- [ ] Kiểm tra chất lượng audio

---

**CHÚC BẠN THÀNH CÔNG! 🍵🎤**

*Matcha-TTS + PhoBERT Prosody Analysis for Vietnamese TTS*
