# 🚀 HƯỚNG DẪN NHANH - MATCHA-TTS

## Bạn đang ở đâu?

### ❓ Tôi mới bắt đầu, chưa có gì cả
→ Đọc: [README.md](README.md) → Phần "Cài đặt môi trường"

### 📁 Tôi có file audio gốc chưa xử lý (voice1.mp3, voice2.mp3, ...)
→ Đọc: [PIPELINE_SETUP.md](PIPELINE_SETUP.md)
→ Chạy: `run_full_pipeline.bat`

### ✅ Tôi đã có file filelist với IPA phonemes
→ Đọc: [README.md](README.md) → Phần "Training model"
→ Chạy: `python train_matcha_prosody.py`

---

## 📋 Pipeline Đầy Đủ (Từ Audio Gốc → Model)

```
1. Đặt audio gốc vào: data/raw/
                ↓
2. Chạy: run_full_pipeline.bat
   - Remove silence (VAD)
   - Transcribe với Whisper
   - Cắt thành từng câu
   - Chuẩn hóa + IPA phonemes
   - Chia train/val/test
                ↓
3. Kiểm tra: python scripts\check_data.py
                ↓
4. Training: python train_matcha_prosody.py
```

---

## 🔧 Troubleshooting Nhanh

### Lỗi: "ConnectionResetError" khi cài PyTorch
→ Xem README.md → Phần "Cài đặt PyTorch"
→ Thử: `pip install --timeout=100 torch`
→ Hoặc: Tải thủ công từ pytorch.org

### Lỗi: "File audio không tồn tại"
→ Bạn chưa chạy pipeline xử lý dữ liệu
→ Đọc: [PIPELINE_SETUP.md](PIPELINE_SETUP.md)

### Lỗi: "espeak-ng not found"
→ Cài eSpeak-NG từ: https://github.com/espeak-ng/espeak-ng/releases
→ Xem README.md → Phần "Cài đặt eSpeak-NG"
→ Script `run_full_pipeline.bat` sẽ tự động kiểm tra và thông báo

### Lỗi: "Microsoft Visual C++ required" (monotonic_align)
→ KHÔNG CẦN CÀI! Script đã tự động dùng Python fallback
→ Chạy bình thường, chỉ chậm hơn một chút

### Lỗi: "CUDA Out of Memory"
→ Giảm batch_size trong train_matcha_prosody.py
→ `"batch_size": 8` → `"batch_size": 4`

---

## 📚 Tài Liệu Chi Tiết

| File | Nội dung |
|------|----------|
| [README.md](README.md) | Hướng dẫn đầy đủ từ A-Z |
| [PIPELINE_SETUP.md](PIPELINE_SETUP.md) | Pipeline xử lý dữ liệu từ audio gốc |
| `run_full_pipeline.bat` | Script tự động chạy toàn bộ pipeline |

---

## ⚡ Quick Start (Nếu bạn biết mình đang làm gì)

### Cách 1: Tự động 100% (Khuyến nghị)
```cmd
# Chỉ cần 1 lệnh - script tự làm tất cả!
run_full_pipeline.bat
```
Script sẽ tự động:
- Tạo venv
- Cài PyTorch + tất cả dependencies
- Xử lý dữ liệu (VAD → transcribe → normalize → split)
- Generate data statistics
- Train model
- Test checkpoint

### Cách 2: Thủ công (Nếu muốn kiểm soát)
```cmd
# 1. Setup environment
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install matplotlib scipy librosa transformers conformer

# 2. Process data
run_full_pipeline.bat

# 3. Train
python train_matcha_prosody.py
```

---

**Cần giúp đỡ? → Đọc README.md phần "Troubleshooting"**
