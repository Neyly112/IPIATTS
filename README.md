**Matcha-TTS — Hướng Dẫn Chuẩn Bị Data, Huấn Luyện, Demo & Đánh Giá**

Giới thiệu ngắn: hướng dẫn này mô tả các bước chuẩn bị dữ liệu để train `MatchaTTS`, lệnh để huấn luyện trên terminal, cách chạy demo web với checkpoint đã train, và cách dùng các script `draw.py`, `draw_mel.py`, `gen_batch.py` và notebook `score.ipynb` để đánh giá.

**Yêu cầu**: cài dependencies:

```bash
python -m pip install -r requirements.txt
```

**1. Chuẩn bị dữ liệu (Data prep)**
- **Audio**: đặt file WAV mono, sampling rate = 22050 Hz, lưu theo tên duy nhất (ví dụ `utt001.wav`).
- **Transcripts / filelist**: tạo file text dạng `filename.wav|transcript` mỗi dòng một mẫu. Thư mục mẫu có sẵn: `data/99-audio-text-file-list/`.
- **Cấu trúc đề xuất**:
  - `data/raw/` — audio thô (gốc)
  - `data/subs/` — audio tham chiếu (nếu cần copy)
  - `data/99-audio-text-file-list/audio_text_train_filelist.txt` — danh sách train
  - `data/99-audio-text-file-list/audio_text_val_filelist.txt` — danh sách val
  - `data/99-audio-text-file-list/audio_text_test_filelist.txt` — danh sách test
- **Làm sạch văn bản**: chạy các script trong `scripts/` nếu cần (ví dụ `scripts/cleaner.py`, `scripts/correct_spelling_mistakes.py`, `scripts/phonemize.py`). Mục tiêu là text đã chuẩn (loại bỏ ký tự lạ, chuẩn hóa chữ hoa/thường, chấm câu nếu cần).
- **Filelist format**: mỗi dòng: `relative/path/to/audio.wav|text to synthesize` (không có BOM, mã hóa UTF-8).

Gợi ý lệnh kiểm tra nhanh (Windows PowerShell / cmd):

```bash
# kiểm tra sampling rate và channel
python - <<'PY'
import soundfile as sf, sys
f='data/raw/utt001.wav'
info=sf.info(f)
print(info)
PY
```

**2. Tiền xử lý (nếu cần)**
- Nếu repo có bước preprocess riêng (ví dụ tạo mel, alignment), chạy script preprocess tương ứng. Nếu không, training có thể đọc filelist trực tiếp và model sẽ xử lý online.

**3. Huấn luyện Matcha-TTS**
- Kiểm tra file `matcha/train.py` hoặc `matcha/cli.py` để biết options cụ thể. Thông thường khởi chạy training bằng Python:

```bash
# ví dụ chung (cập nhật path checkpoint/output theo repo của bạn)
python matcha/train.py \
  --config configs/matcha_vi.yaml \
  --data-root data/ \
  --train-filelist data/99-audio-text-file-list/audio_text_train_filelist.txt \
  --val-filelist data/99-audio-text-file-list/audio_text_val_filelist.txt \
  --gpus 1 \
  --batch-size 8 \
  --max-epochs 600
```

- Nếu repo sử dụng Lightning CLI hoặc hydra, lệnh có thể tương tự:

```bash
# hydra style
python -m matcha.train --config-name=your_config
```

- Kết quả: checkpoints sẽ lưu vào `logs/matcha_vi/checkpoints/` (ví dụ `last.ckpt` hoặc `checkpoint_epoch599.ckpt`).

**4. Sinh batch / inference hàng loạt**
- Dùng `gen_batch.py` để tạo audio từ filelist và (nếu cần) copy audio gốc vào thư mục ref. Cập nhật các đường dẫn trong file trước khi chạy (biến `CHECKPOINT_PATH`, `TEST_FILE_PATH`, `SOURCE_WAV_FOLDER`, `OUTPUT_FOLDER`).

Chạy:

```bash
python gen_batch.py
```

Sau khi chạy, kết quả sẽ nằm trong `OUTPUT_FOLDER/gen` (audio máy) và `OUTPUT_FOLDER/ref` (audio gốc). File `OUTPUT_FOLDER/text.txt` chứa danh sách filename|text.

**5. Chạy demo Web (local)**
- Sửa đường dẫn `MATCHA_CHECKPOINT` và `VOCODER_URL` trong `run_web.py` cho đúng checkpoint và vocoder đã tải.
- Ví dụ chạy:

```bash
python run_web.py
```

- Giao diện sẽ khởi chạy Gradio (mở browser). Tham số: tốc độ, số steps, temperature.

**6. Đánh giá & trực quan (draw, draw_mel, score)**
- `draw.py`: gom và vẽ các scalar đã ghi trong TensorBoard (ví dụ loss theo epoch). Cấu hình `log_dir` ở đầu file, sau đó chạy file để hiển thị biểu đồ.

```bash
python draw.py
```

- `draw_mel.py`: vẽ mel-spectrogram so sánh giữa `ref` và `gen`. Cấu hình `FOLDER_REF` và `FOLDER_GEN` ở đầu file, sau đó chạy:

```bash
python draw_mel.py
```

- `score.ipynb`: notebook dùng Whisper + UTMOS + F0 comparison để chấm điểm MOS / WER / CER / F0. Mở `score.ipynb` trong Jupyter / Colab và cập nhật đường dẫn `GEN_AUDIO_FOLDER`, `REF_AUDIO_FOLDER`, `TEST_FILE_LIST` ở đầu notebook rồi chạy các cell.

Lưu ý: `score.ipynb` có nhiều dependency (Whisper, torchaudio, fastdtw...). Cài trước các package trong `requirements.txt`.

**7. Kiến nghị & troubleshooting**
- Kiểm tra sampling rate, mono/stereo, và độ dài tối thiểu (một số hàm trích F0 cần >=512 samples sau trim).
- Nếu gặp lỗi khi load checkpoint do PyTorch 2.x security, repo có các patch trong `gen_batch.py`/`run_web.py` (ví dụ `torch.load = partial(torch.load, weights_only=False)` hoặc patch `torch.serialization.add_safe_globals`). Giữ những patch đó nếu cần.
- Vocoder: repo thường tự tải vocoder vào thư mục user data; nếu không, tải tay và cập nhật đường dẫn trong `run_web.py` hoặc `gen_batch.py`.

**8. Một ví dụ ngắn các lệnh terminal (Windows)**

```powershell
# 1) Cài dependencies
python -m pip install -r requirements.txt

# 2) Chuẩn bị filelist (ví dụ copy filelist mẫu, chỉnh sửa)
notepad data\99-audio-text-file-list\audio_text_train_filelist.txt

# 3) Huấn luyện (ví dụ)
python matcha/train.py --config configs/matcha_vi.yaml --data-root data --train-filelist data\99-audio-text-file-list\audio_text_train_filelist.txt

# 4) Sinh batch để kiểm tra chất lượng (chỉnh CHECKPOINT_PATH trong gen_batch.py)
python gen_batch.py

# 5) Chạy demo web (chỉnh MATCHA_CHECKPOINT trong run_web.py)
python run_web.py

# 6) Vẽ biểu đồ tensorboard scalar
python draw.py

# 7) Vẽ mel spectrogram so sánh
python draw_mel.py

# 8) Mở notebook đánh giá
jupyter notebook score.ipynb
```

**Tệp tham chiếu trong repo**
- [run_web.py](run_web.py) — demo Gradio
- [gen_batch.py](gen_batch.py) — sinh hàng loạt + copy ref
- [draw.py](draw.py) — vẽ biểu đồ tensorboard
- [draw_mel.py](draw_mel.py) — vẽ mel spectrogram so sánh
- [score.ipynb](score.ipynb) — notebook chấm điểm (MOS/WER/CER/F0)
- `checkpoints/` — chứa các checkpoint đã train (ví dụ `last.ckpt`, `checkpoint_epoch599.ckpt`).

Nếu bạn muốn, tôi có thể:
- Tùy chỉnh README theo config training thực tế trong `matcha/` (nếu bạn muốn, tôi sẽ mở `matcha/train.py` và trích lệnh chính xác). 
- Chạy kiểm tra nhỏ (ví dụ kiểm tra một file WAV) trên môi trường hiện tại.

---
Tạo bởi hướng dẫn nhanh từ repo hiện tại. Nếu muốn, tôi sẽ cập nhật lệnh huấn luyện chính xác sau khi xem file `matcha/train.py` hoặc `matcha/cli.py`.
