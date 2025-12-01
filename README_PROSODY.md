# 🍵 Matcha-TTS với Prosody Analysis (PhoBERT)

Phiên bản Matcha-TTS được tích hợp **LLM Prosody Analysis** sử dụng PhoBERT để tạo giọng nói tiếng Việt tự nhiên hơn.

---

## ✨ Tính năng

- ✅ **Prosody Analysis với PhoBERT**: Phân tích ngữ điệu từ văn bản tiếng Việt
- ✅ **Attention-based Fusion**: Kết hợp prosody với text features
- ✅ **Frozen LLM + Adapters**: Training hiệu quả, tiết kiệm bộ nhớ
- ✅ **Backward Compatible**: Code đơn giản, không cần flag use_prosody

---

## 📁 Cấu trúc thư mục

```
IPIATTS/
├── matcha/                              # Code chính
│   ├── models/
│   │   ├── matcha_tts.py               # Model chính (ĐÃ CÓ PROSODY)
│   │   └── components/
│   │       ├── prosody_analyzer.py      # PhoBERT prosody
│   │       └── prosody_fusion.py        # Attention fusion
│   ├── data/
│   │   └── text_mel_datamodule.py      # DataLoader
│   └── utils/
│       └── generate_data_statistics.py  # Tính mel stats
│
├── data/                                # Dữ liệu của bạn
│   ├── 99-audio-text-file-list/
│   │   ├── audio_text_train_filelist.txt.cleaned
│   │   ├── audio_text_val_filelist.txt.cleaned
│   │   └── audio_text_test_filelist.txt.cleaned
│   ├── vad/                             # Audio files
│   └── vad1/                            # Audio files
│
├── train_matcha_prosody.py              # Script training chính ⭐
├── start_training.bat                   # Quick start cho Windows
├── HUONG_DAN_TRAINING.md               # Hướng dẫn chi tiết ⭐
└── requirements_prosody.txt             # Dependencies
```

---

## 🚀 Quick Start (3 bước)

### Bước 1: Cài đặt dependencies

```cmd
pip install -r requirements_prosody.txt
```

### Bước 2: Kiểm tra dữ liệu

```cmd
python scripts\check_data.py --filelist data\99-audio-text-file-list\audio_text_train_filelist.txt.cleaned
```

### Bước 3: Bắt đầu training

**Cách 1: Dùng script tự động (Windows)**
```cmd
start_training.bat
```

**Cách 2: Chạy trực tiếp**
```cmd
python train_matcha_prosody.py
```

---

## 📖 Hướng dẫn chi tiết

Xem file **`HUONG_DAN_TRAINING.md`** để biết:

- Cách chuẩn bị dữ liệu
- Tính data statistics
- Chỉnh sửa cấu hình
- Implement DataModule
- Theo dõi training với TensorBoard
- Troubleshooting

---

## ⚙️ Cấu hình trong `train_matcha_prosody.py`

```python
CONFIG = {
    # Dữ liệu (ĐÃ ĐÚNG)
    "train_filelist": "data/99-audio-text-file-list/audio_text_train_filelist.txt.cleaned",
    "val_filelist": "data/99-audio-text-file-list/audio_text_val_filelist.txt.cleaned",
    
    # Model
    "n_vocab": 256,              # Số phonemes
    "n_spks": 1,                 # Single speaker
    "batch_size": 16,            # Giảm xuống 8/4 nếu OOM
    "learning_rate": 1e-4,
    "max_epochs": 1000,
    
    # Prosody (PhoBERT)
    "llm_model_name": "vinai/phobert-base",
    "prosody_dim": 256,
    
    # Output
    "output_dir": "outputs/matcha_prosody",
}
```

**Chỉ cần chỉnh:**
- `batch_size`: Tùy GPU (16 → 8 → 4 nếu thiếu VRAM)
- `n_vocab`: Kiểm tra `matcha/text/symbols.py`
- `n_spks`: 1 cho single speaker, >1 cho multi-speaker

---

## 📊 Theo dõi Training

Mở terminal mới và chạy:

```cmd
tensorboard --logdir outputs/matcha_prosody/logs
```

Sau đó truy cập: http://localhost:6006

**Metrics cần theo dõi:**
- `train_loss` và `val_loss` giảm dần
- `dur_loss`, `prior_loss`, `diff_loss`
- Learning rate schedule

---

## 🧪 Test Model

```python
from matcha.models.matcha_tts import MatchaTTS
import torch

# Load checkpoint
model = MatchaTTS.load_from_checkpoint(
    "outputs/matcha_prosody/checkpoints/last.ckpt"
)
model.eval()

# Synthesize (prosody tự động được bật)
from matcha.text import text_to_sequence
from matcha.utils.utils import intersperse

text = "xin chào, đây là giọng nói tiếng việt"
x = torch.tensor(
    intersperse(text_to_sequence(text, ["basic_cleaners_phothong"])[0], 0)
)[None]
x_lengths = torch.tensor([x.shape[-1]])

with torch.no_grad():
    output = model.synthesise(x, x_lengths, n_timesteps=10)

print(f"Mel shape: {output['mel'].shape}")
print(f"RTF: {output['rtf']:.4f}")  # Real-time factor
```

---

## 📋 Checklist trước khi Training

- [ ] Đã cài đặt dependencies: `pip install -r requirements_prosody.txt`
- [ ] Dữ liệu đã đúng format: `audio_path|text|phonemes`
- [ ] Kiểm tra filelist: `python scripts\check_data.py --filelist ...`
- [ ] Tính data statistics: `python matcha\utils\generate_data_statistics.py ...`
- [ ] Cập nhật `DATA_STATISTICS` trong `train_matcha_prosody.py`
- [ ] Kiểm tra GPU: `torch.cuda.is_available()`
- [ ] DataModule đã được implement hoặc tương thích

---

## 💡 Tips

1. **Thiếu VRAM?** Giảm `batch_size` hoặc `prosody_dim`
2. **Training chậm?** Tăng `num_workers` trong DataLoader
3. **Overfitting?** Tăng dropout, thêm data augmentation
4. **Loss không giảm?** Kiểm tra learning rate và data statistics

---

## 🔧 Troubleshooting

| Vấn đề | Giải pháp |
|--------|-----------|
| CUDA Out of Memory | Giảm `batch_size` xuống 8 hoặc 4 |
| PhoBERT download fail | Chạy: `python -c "from transformers import AutoModel; AutoModel.from_pretrained('vinai/phobert-base')"` |
| Loss NaN | Giảm learning rate hoặc kiểm tra data statistics |
| File not found | Kiểm tra đường dẫn trong filelist |

Xem thêm trong **`HUONG_DAN_TRAINING.md`**

---

## 📚 File quan trọng

1. **`train_matcha_prosody.py`** - Script training chính
2. **`HUONG_DAN_TRAINING.md`** - Hướng dẫn từng bước chi tiết
3. **`start_training.bat`** - Quick start script
4. **`matcha/models/matcha_tts.py`** - Model đã tích hợp prosody
5. **`matcha/models/components/prosody_analyzer.py`** - PhoBERT analyzer

---

## 🎯 Kết quả mong đợi

Sau khi training thành công:

- **Checkpoint**: `outputs/matcha_prosody/checkpoints/`
- **Logs**: `outputs/matcha_prosody/logs/`
- **Giọng nói**: Tự nhiên hơn với prosody phù hợp
- **RTF**: ~0.01-0.05 (tùy GPU)

---

## 📞 Hỗ trợ

Nếu gặp vấn đề:
1. Đọc kỹ `HUONG_DAN_TRAINING.md`
2. Kiểm tra TensorBoard logs
3. Debug với `batch_size=1`
4. Kiểm tra format dữ liệu

---

**Chúc bạn training thành công! 🍵🎤**

*Matcha-TTS + PhoBERT Prosody Analysis for Vietnamese TTS*
