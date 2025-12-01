# 🍵 HƯỚNG DẪN TRAINING MATCHA-TTS VỚI PROSODY

## Tổng quan
Model này sử dụng **PhoBERT** để phân tích prosody (ngữ điệu) từ văn bản tiếng Việt, giúp tạo giọng nói tự nhiên hơn.

---

## BƯỚC 1: CHUẨN BỊ MÔI TRƯỜNG

### 1.1. Cài đặt dependencies

```cmd
pip install -r requirements_prosody.txt
```

### 1.2. Kiểm tra GPU (khuyến nghị)

```cmd
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Yêu cầu:**
- VRAM: Tối thiểu 8GB (khuyến nghị 12GB+)
- Nếu thiếu VRAM: Giảm `batch_size` xuống 8 hoặc 4 trong `train_matcha_prosody.py`

---

## BƯỚC 2: CHUẨN BỊ DỮ LIỆU

### 2.1. Cấu trúc thư mục hiện tại

```
data/
├── 99-audio-text-file-list/
│   ├── audio_text_train_filelist.txt.cleaned  ✓ (đã có)
│   ├── audio_text_val_filelist.txt.cleaned    ✓ (đã có)
│   └── audio_text_test_filelist.txt.cleaned   ✓ (đã có)
├── vad/        # Audio files
└── vad1/       # Audio files
```

### 2.2. Format file filelist

File `.cleaned` cần có format:
```
audio_path|transcription|phonemes
```

**Ví dụ:**
```
data/vad1/audio_001.wav|xin chào tất cả mọi người|s i n ch a o t a t k a m o i ng ư ơ i
data/vad1/audio_002.wav|hôm nay thời tiết đẹp|h o m n ai th ơ i t i e t d e p
```

### 2.3. Kiểm tra dữ liệu

```cmd
python scripts/check_data.py --filelist data/99-audio-text-file-list/audio_text_train_filelist.txt.cleaned
```

**Kiểm tra thủ công:**
1. Đảm bảo tất cả file audio tồn tại
2. Phonemes phải được phân tách bằng khoảng trắng
3. Không có dòng trống hoặc lỗi format

---

## BƯỚC 3: TÍNH TOÁN DATA STATISTICS

Data statistics (mel_mean, mel_std) cần thiết để normalize mel-spectrogram.

```cmd
python matcha/utils/generate_data_statistics.py --filelist data/99-audio-text-file-list/audio_text_train_filelist.txt.cleaned --output data_stats.json
```

**Sau đó update vào `train_matcha_prosody.py`:**

```python
DATA_STATISTICS = {
    "mel_mean": -5.123,     # Thay bằng giá trị từ data_stats.json
    "mel_std": 2.456,       # Thay bằng giá trị từ data_stats.json
}
```

---

## BƯỚC 4: CHỈNH SỬA CẤU HÌNH TRAINING

Mở file `train_matcha_prosody.py` và chỉnh sửa:

### 4.1. Đường dẫn dữ liệu (đã đúng)

```python
CONFIG = {
    "train_filelist": "data/99-audio-text-file-list/audio_text_train_filelist.txt.cleaned",
    "val_filelist": "data/99-audio-text-file-list/audio_text_val_filelist.txt.cleaned",
    ...
}
```

### 4.2. Model settings

```python
CONFIG = {
    ...
    "n_vocab": 256,        # Kiểm tra matcha/text/symbols.py
    "n_spks": 1,           # 1 speaker (thay đổi nếu multi-speaker)
    "batch_size": 16,      # Giảm xuống 8/4 nếu out of memory
    "learning_rate": 1e-4,
    "max_epochs": 1000,    # Có thể train lâu hơn nếu cần
    ...
}
```

### 4.3. PhoBERT settings (giữ nguyên)

```python
CONFIG = {
    ...
    "llm_model_name": "vinai/phobert-base",  # PhoBERT cho tiếng Việt
    "prosody_dim": 256,
    ...
}
```

---

## BƯỚC 5: IMPLEMENT DATAMODULE

File `matcha/data/text_mel_datamodule.py` cần được implement hoặc sửa lại cho phù hợp.

**Nếu file này chưa có hoặc không tương thích:**

1. Kiểm tra file hiện tại:
```cmd
python -c "from matcha.data.text_mel_datamodule import TextMelDataModule; print('OK')"
```

2. Nếu lỗi, bạn cần:
   - Sao chép từ Matcha-TTS gốc
   - Hoặc implement custom DataModule cho dữ liệu của bạn

**Reference:** Xem `matcha/utils/data/ljspeech.py` để tham khảo cách implement

---

## BƯỚC 6: BẮT ĐẦU TRAINING

### 6.1. Training từ đầu

```cmd
python train_matcha_prosody.py
```

### 6.2. Theo dõi training với TensorBoard

Mở terminal mới và chạy:

```cmd
tensorboard --logdir outputs/matcha_prosody/logs
```

Sau đó mở trình duyệt: http://localhost:6006

### 6.3. Resume từ checkpoint (nếu bị gián đoạn)

Chỉnh sửa trong `train_matcha_prosody.py`:

```python
CONFIG = {
    ...
    "resume_from_checkpoint": "outputs/matcha_prosody/checkpoints/last.ckpt",
}
```

Sau đó chạy lại:

```cmd
python train_matcha_prosody.py
```

---

## BƯỚC 7: KIỂM TRA KẾT QUẢ

### 7.1. Các file checkpoint

Training sẽ tạo ra các file:

```
outputs/matcha_prosody/
├── checkpoints/
│   ├── matcha-prosody-epoch=050-val_loss=0.234.ckpt  # Best model
│   ├── matcha-prosody-epoch=100-val_loss=0.189.ckpt
│   └── last.ckpt                                      # Latest checkpoint
└── logs/
    └── tensorboard_logs/
```

### 7.2. Test synthesis

```python
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import text_to_sequence
from matcha.utils.utils import intersperse
import torch

# Load model
model = MatchaTTS.load_from_checkpoint(
    "outputs/matcha_prosody/checkpoints/matcha-prosody-epoch=100-val_loss=0.189.ckpt"
)
model.eval()

# Prepare text
text = "xin chào, đây là giọng đọc tiếng việt"
x = torch.tensor(
    intersperse(text_to_sequence(text, ["basic_cleaners_phothong"])[0], 0)
)[None]
x_lengths = torch.tensor([x.shape[-1]])

# Synthesize
with torch.no_grad():
    output = model.synthesise(
        x, x_lengths,
        n_timesteps=10,
        temperature=0.667,
        length_scale=1.0,
    )

print(f"Generated mel shape: {output['mel'].shape}")
print(f"RTF: {output['rtf']:.4f}")
```

---

## TROUBLESHOOTING

### Lỗi: Out of Memory (CUDA OOM)

**Giải pháp:**
1. Giảm `batch_size` xuống 8 hoặc 4
2. Giảm `prosody_dim` xuống 128
3. Sử dụng gradient accumulation:
   ```python
   trainer = pl.Trainer(
       ...
       accumulate_grad_batches=4,  # Thêm dòng này
   )
   ```

### Lỗi: PhoBERT download failed

**Giải pháp:**
1. Download thủ công:
   ```cmd
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('vinai/phobert-base')"
   ```

2. Hoặc dùng Simple Prosody (không cần PhoBERT):
   - Xem file `matcha/models/components/prosody_analyzer.py`
   - Thay `LLMProsodyAnalyzer` bằng `SimpleProsodyAnalyzer`

### Loss không giảm

**Kiểm tra:**
1. Data statistics đã được tính đúng chưa?
2. Filelist format có đúng không?
3. Learning rate có quá cao/thấp?
4. Thử giảm learning rate: `1e-4` → `5e-5`

### Validation loss tăng (overfitting)

**Giải pháp:**
1. Tăng dropout: `0.1` → `0.2`
2. Thêm data augmentation
3. Tăng kích thước validation set
4. Early stopping đã được enable sẵn (patience=50)

---

## THÔNG TIN BỔ SUNG

### Thời gian training

- **Dataset nhỏ** (<10 giờ audio): ~1-2 ngày
- **Dataset trung bình** (10-50 giờ): ~3-5 ngày
- **Dataset lớn** (>50 giờ): ~1 tuần

*Tùy thuộc vào GPU và cấu hình*

### Yêu cầu phần cứng

| Cấu hình    | GPU          | VRAM | Batch Size | Training Time |
|-------------|--------------|------|------------|---------------|
| Tối thiểu   | GTX 1660     | 6GB  | 4          | Rất chậm      |
| Khuyến nghị | RTX 3060     | 12GB | 16         | Trung bình    |
| Tối ưu      | RTX 4090     | 24GB | 32+        | Nhanh         |

### Kiểm tra chất lượng

1. **MOS Score**: Đánh giá chủ quan (1-5)
2. **RTF**: Real-time factor (càng thấp càng nhanh)
3. **Mel Cepstral Distortion**: So sánh với ground truth
4. **Nghe trực tiếp**: Kiểm tra naturalness, prosody, pronunciation

---

## HỖ TRỢ

Nếu gặp vấn đề:
1. Kiểm tra TensorBoard logs
2. Xem file `outputs/matcha_prosody/logs/`
3. Debug với batch_size=1 để tìm lỗi
4. Kiểm tra format dữ liệu cẩn thận

---

**Chúc bạn training thành công! 🍵**
