# ✅ HOÀN TẤT - CÁC BƯỚC TIẾP THEO

## 🎉 Đã hoàn thành

### Code đã được đơn giản hóa
- ✅ Xóa tất cả code baseline cũ (không có prosody)
- ✅ Prosody luôn được bật với PhoBERT
- ✅ Chỉ còn 1 phiên bản duy nhất
- ✅ CLI và UI đã được đơn giản hóa

### File mới
- ✅ `train_matcha_prosody.py` - Script training sẵn sàng
- ✅ `HUONG_DAN_TRAINING.md` - Hướng dẫn chi tiết
- ✅ `README_PROSODY.md` - Tổng quan
- ✅ `start_training.bat` - Quick start
- ✅ `scripts/check_data.py` - Kiểm tra dữ liệu

---

## 📝 CÁC BƯỚC TRAINING (TÓM TẮT)

### BƯỚC 1: Kiểm tra dữ liệu
```cmd
python scripts\check_data.py --filelist data\99-audio-text-file-list\audio_text_train_filelist.txt.cleaned
```

✅ **Nếu thành công**: Thấy "✅ FILELIST HỢP LỆ"

❌ **Nếu lỗi**: Sửa file filelist theo format:
```
audio_path|text|phonemes
```

---

### BƯỚC 2: Tính Data Statistics

**Cách 1: Dùng script có sẵn** (nếu DataModule tương thích)
```cmd
python matcha\utils\generate_data_statistics.py --filelist data\99-audio-text-file-list\audio_text_train_filelist.txt.cleaned --output data_stats.json
```

**Cách 2: Tính thủ công** (nếu cách 1 lỗi)

Bạn cần tính mean và std của mel-spectrogram. Có thể:
1. Dùng script riêng để load audio và tính mel
2. Hoặc để giá trị mặc định: `mel_mean=0.0, mel_std=1.0` (training sẽ chậm hơn)

**Sau đó cập nhật vào `train_matcha_prosody.py`:**
```python
DATA_STATISTICS = {
    "mel_mean": -5.123,  # Thay bằng giá trị tính được
    "mel_std": 2.456,    # Thay bằng giá trị tính được
}
```

---

### BƯỚC 3: Kiểm tra/Implement DataModule

File `matcha/data/text_mel_datamodule.py` cần tương thích với dữ liệu của bạn.

**Kiểm tra:**
```cmd
python -c "from matcha.data.text_mel_datamodule import TextMelDataModule; print('OK')"
```

✅ **Nếu OK**: Tiếp tục bước 4

❌ **Nếu lỗi**: Cần implement hoặc sửa DataModule
- Xem `matcha/utils/data/ljspeech.py` để tham khảo
- Hoặc tạo custom DataModule cho format của bạn

---

### BƯỚC 4: Chỉnh sửa config (nếu cần)

Mở `train_matcha_prosody.py` và kiểm tra:

```python
CONFIG = {
    # Đường dẫn (ĐÃ ĐÚNG)
    "train_filelist": "data/99-audio-text-file-list/audio_text_train_filelist.txt.cleaned",
    "val_filelist": "data/99-audio-text-file-list/audio_text_val_filelist.txt.cleaned",
    
    # Có thể cần chỉnh
    "n_vocab": 256,        # Kiểm tra matcha/text/symbols.py
    "n_spks": 1,           # 1 = single speaker
    "batch_size": 16,      # Giảm xuống 8/4 nếu thiếu VRAM
    
    # Giữ nguyên
    "llm_model_name": "vinai/phobert-base",
    "prosody_dim": 256,
    "learning_rate": 1e-4,
    "max_epochs": 1000,
}
```

---

### BƯỚC 5: Bắt đầu Training!

**Cách 1: Quick Start (Windows)**
```cmd
start_training.bat
```

**Cách 2: Chạy trực tiếp**
```cmd
python train_matcha_prosody.py
```

**Mở TensorBoard** (terminal riêng)
```cmd
tensorboard --logdir outputs/matcha_prosody/logs
```
Truy cập: http://localhost:6006

---

## 📊 Theo dõi Training

### Metrics cần xem
- `train_loss` và `val_loss` giảm dần
- `dur_loss` (duration prediction)
- `prior_loss` (encoder quality)
- `diff_loss` (decoder/CFM quality)

### Checkpoints
Được lưu tại: `outputs/matcha_prosody/checkpoints/`
- `matcha-prosody-epoch=XXX-val_loss=Y.YYY.ckpt` - Best models
- `last.ckpt` - Checkpoint mới nhất

---

## 🧪 Test Model

```python
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import text_to_sequence
from matcha.utils.utils import intersperse
import torch

# Load model
model = MatchaTTS.load_from_checkpoint(
    "outputs/matcha_prosody/checkpoints/last.ckpt"
)
model.eval()

# Chuẩn bị text
text = "xin chào, hôm nay tôi học về trí tuệ nhân tạo"
x = torch.tensor(
    intersperse(text_to_sequence(text, ["basic_cleaners_phothong"])[0], 0)
)[None]
x_lengths = torch.tensor([x.shape[-1]])

# Synthesize (prosody tự động chạy)
with torch.no_grad():
    output = model.synthesise(
        x, x_lengths,
        n_timesteps=10,
        temperature=0.667,
        length_scale=1.0,
    )

print(f"Mel spectrogram shape: {output['mel'].shape}")
print(f"RTF (Real-time factor): {output['rtf']:.4f}")
print(f"Mel lengths: {output['mel_lengths']}")

# Để nghe được âm thanh, cần vocoder (HiFi-GAN)
# Xem matcha/cli.py để tham khảo cách load vocoder
```

---

## 🔧 Troubleshooting

### Out of Memory (CUDA OOM)
```python
# Trong train_matcha_prosody.py
CONFIG = {
    "batch_size": 8,  # Giảm từ 16 → 8
    # hoặc
    "batch_size": 4,  # Giảm xuống 4
}
```

### PhoBERT download lỗi
```cmd
python -c "from transformers import AutoModel; AutoModel.from_pretrained('vinai/phobert-base')"
```
PhoBERT sẽ được download (~1GB) lần đầu tiên.

### DataModule không tương thích
Bạn cần implement custom DataModule hoặc sửa lại `text_mel_datamodule.py` cho phù hợp với format dữ liệu của bạn.

### Loss không giảm
1. Kiểm tra `DATA_STATISTICS` đã đúng chưa
2. Thử giảm learning rate: `1e-4` → `5e-5`
3. Kiểm tra filelist format

---

## 📚 Đọc thêm

- **`README_PROSODY.md`**: Tổng quan về code
- **`HUONG_DAN_TRAINING.md`**: Hướng dẫn chi tiết đầy đủ
- **`CHANGELOG.md`**: Những thay đổi trong code

---

## 🎯 Checklist

Trước khi training, đảm bảo:

- [ ] Đã chạy `scripts\check_data.py` thành công
- [ ] Đã tính `DATA_STATISTICS` (hoặc để mặc định)
- [ ] `TextMelDataModule` hoạt động (hoặc đã implement custom)
- [ ] Đã kiểm tra GPU: `torch.cuda.is_available()`
- [ ] Đã cài đặt: `pip install -r requirements_prosody.txt`
- [ ] Đã đọc `HUONG_DAN_TRAINING.md`

---

## ✨ Kết quả mong đợi

Sau khi training hoàn tất:
- **Giọng nói tự nhiên** với prosody (ngữ điệu) phù hợp
- **Real-time factor (RTF)**: ~0.01-0.05 (tùy GPU)
- **Checkpoint files** trong `outputs/matcha_prosody/checkpoints/`
- **TensorBoard logs** trong `outputs/matcha_prosody/logs/`

---

**CHÚC BẠN TRAINING THÀNH CÔNG! 🍵🎤**

*Matcha-TTS + PhoBERT Prosody Analysis*
