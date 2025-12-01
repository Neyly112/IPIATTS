# 🎤 TRANSCRIBE_CUT.PY - HƯỚNG DẪN SỬ DỤNG

## Mục đích
Script này sử dụng **OpenAI Whisper** để:
1. Tự động transcribe (chuyển giọng nói → text) file audio tiếng Việt
2. Cắt file audio dài thành từng câu ngắn (sentence-level segments)
3. Lọc bỏ các đoạn không hợp lệ (hallucination, duplicate, quá ngắn)
4. Tạo file filelist cho training TTS

---

## Cấu hình

Mở file `transcribe_cut.py` và chỉnh sửa:

```python
# CONFIG
WHISPER_MODEL = "small"  # Chọn model Whisper
MIN_DURATION_SEC = 0.4   # Bỏ qua đoạn audio < 0.4s
MIN_WORDS = 2            # Bỏ qua câu < 2 từ
```

### Chọn Whisper Model

| Model | Kích thước | VRAM | Tốc độ | Độ chính xác |
|-------|-----------|------|--------|--------------|
| `tiny` | ~40MB | ~1GB | Rất nhanh | Thấp |
| `base` | ~75MB | ~1GB | Nhanh | Trung bình |
| `small` | ~240MB | ~2GB | Vừa | Tốt ✅ (khuyến nghị) |
| `medium` | ~770MB | ~5GB | Chậm | Rất tốt |
| `large` | ~1.5GB | ~10GB | Rất chậm | Xuất sắc |

**Khuyến nghị:**
- **CPU**: Dùng `small` hoặc `base`
- **GPU (6-8GB VRAM)**: Dùng `small` hoặc `medium`
- **GPU (12GB+ VRAM)**: Dùng `medium` hoặc `large`

---

## Chạy Script

### Bước 1: Đảm bảo đã có dữ liệu VAD

```cmd
dir data\vad
```

Kết quả mong đợi: `voice1.wav`, `voice2.wav`, ...

### Bước 2: Chạy transcription

```cmd
cd scripts
python transcribe_cut.py
```

### Bước 3: Theo dõi tiến trình

Script sẽ hiển thị:
```
[1/20] Processing: voice1.wav
Transcribing voice1.wav ...
  skip voice1_0001.wav         ⇐ too_short_audio (0.25s): Xin chào
  skip voice1_0002.wav         ⇐ duplicate: Hôm nay thời tiết đẹp
  ✓ Kept: 45, Skipped: 3

[2/20] Processing: voice2.wav
...
```

---

## Output

### 1. File audio segments

```
data/subs/
├── voice1_0001.wav
├── voice1_0002.wav
├── voice2_0001.wav
└── ...
```

- **Format**: WAV, mono, 16kHz (sau đó resample 22.05kHz)
- **Mỗi file**: 1 câu (2-10 giây)

### 2. Transcription file

```
data/99-audio-text-file-list/_all.txt
```

**Format:**
```
voice1_0001.wav|Xin chào, tôi là trợ lý ảo.
voice1_0002.wav|Hôm nay thời tiết đẹp quá.
voice2_0001.wav|Tôi yêu học trí tuệ nhân tạo.
```

---

## Cơ chế lọc (Filtering)

Script tự động bỏ qua các đoạn:

1. **Hallucination** - Whisper tự tạo nội dung không có trong audio
   ```
   ❌ "hãy subscribe cho kênh ghiền mì gõ..."
   ```

2. **Duplicate** - Câu giống y hệt câu trước
   ```
   ❌ "Xin chào" (lần 2)
   ```

3. **Too short text** - Câu < 2 từ
   ```
   ❌ "Ừ"
   ❌ "Ok"
   ```

4. **Too short audio** - Đoạn audio < 0.4 giây
   ```
   ❌ 0.25s audio
   ```

---

## Tối ưu hóa

### GPU Out of Memory

Giảm model size:
```python
WHISPER_MODEL = "base"  # Thay vì "small"
```

### Transcription sai nhiều

Tăng model size:
```python
WHISPER_MODEL = "medium"  # Hoặc "large"
```

### Quá nhiều đoạn bị bỏ qua

Giảm ngưỡng lọc:
```python
MIN_DURATION_SEC = 0.2  # Thay vì 0.4
MIN_WORDS = 1           # Thay vì 2
```

---

## Troubleshooting

### Lỗi: "No module named 'whisper'"
```cmd
pip install openai-whisper
```

### Lỗi: "CUDA out of memory"
```python
# Trong transcribe_cut.py, sửa dòng:
model = whisper.load_model(WHISPER_MODEL, device="cpu")  # Force CPU
```

### Transcription toàn chữ Trung Quốc/Anh
- Whisper đã detect sai ngôn ngữ
- Kiểm tra file audio có phải tiếng Việt không
- Thử tăng kích thước model

---

## Thời gian ước tính

Với CPU (Intel i7) và model `small`:
- **1 phút audio** → ~2-3 phút xử lý
- **10 phút audio** → ~20-30 phút
- **1 giờ audio** → ~2-3 giờ

Với GPU (RTX 3060) và model `small`:
- **1 phút audio** → ~30 giây
- **10 phút audio** → ~5 phút
- **1 giờ audio** → ~30 phút

---

## Bước tiếp theo

Sau khi chạy xong script này, tiếp tục với:

```cmd
python scripts\process_cleaner.py  # Chuẩn hóa + IPA
python scripts\split.py            # Chia train/val/test
```

Hoặc chạy toàn bộ pipeline:
```cmd
run_full_pipeline.bat
```
