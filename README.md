# 🇻🇳 Dự án huấn luyện mô hình Text-to-Speech tiếng Việt sử dụng Matcha-TTS
TTS dùng https://github.com/shivammehta25/Matcha-TTS (model with 18.2 millions parameters)
## 1. Cài đặt môi trường
- Yêu cầu: Python 3.10, cài đặt **espeak-ng**
- Chỉnh sửa `Matcha-TTS/requirements.txt`:
  - Thêm: `underthesea`, `num2words`
  - Xóa: `torchvision`, `piper_phonemize` (để bypass hạn chế Python 3.10)
- Cài đặt thư viện:
pip install -e . --find-links=https://download.pytorch.org/whl/torch_stable.html

> Lưu ý: cần MSVC để build Monotonic Alignment Search

---

## 2. Chuẩn bị dữ liệu
- **Audio**: File `.wav`, Mono, Sample rate **22050 Hz**, được xử lý trước (normalize, trim silence, v.v.)
- **Filelist**: Định dạng `filename.wav|nội dung văn bản` (UTF-8)

## 3. Chỉnh sửa mã nguồn
- Copy các file:
- `../scripts/cleaner.py` → `Matcha-TTS/matcha/text/cleaners.py`
- `../scripts/symbols.py` → `Matcha-TTS/matcha/text/symbols.py`
- `../data/matcha_exp_vi.yaml` → `Matcha-TTS/configs/experiment/matcha_vi.yaml`  
  (chỉnh `max_epochs` trong file này)
- `../data/matcha_data_vi.yaml` → `Matcha-TTS/configs/data/matcha_vi.yaml`  
  (chỉnh `num_workers` ≤ số threads CPU)
- Edit `Matcha-TTS/matcha/cli.py`: đổi `english_cleaners2` thành `basic_cleaners_phothong`

---
## 4. Sinh thống kê dữ liệu
- Chạy:
python matcha/utils/generate_data_statistics.py -i matcha_vi.yaml

- Lấy 2 giá trị `mel_mean` và `mel_std`, sau đó chỉnh lại trong file `matcha_vi.yaml` (thường đã đúng sẵn)

---

## 5. Huấn luyện (Training)
- Train mới:
python matcha/train.py experiment=matcha_vi
- Logs và checkpoints sẽ lưu tại:  
`Matcha-TTS/logs/matcha_vi` (đã cấu hình riêng, không phải mặc định)

- Tiếp tục train (Resume):

python matcha/train.py experiment=matcha_ngngngan ckpt_path="logs/matcha_vi/checkpoints/checkpoint_epoch__.ckpt"

> Lưu ý: phải **rename file checkpoint** để bỏ ký tự `=` trong tên

---

## 6. Sử dụng & Chạy thử (Inference)
- **Cập nhật CHECKPOINT_PATH** trong các script trước khi chạy
- **Sinh audio hàng loạt (Batch Inference)**:  
python gen_batch.py
- **Chạy Web Demo (Gradio)**:  
python run_web.py

---

## 7. Đánh giá chất lượng (Evaluation)
- **Vẽ biểu đồ Loss (Tensorboard)**:  
python draw.py
- **So sánh Mel Spectrogram (Ref vs Gen)**:  
python draw_mel.py
- **Chấm điểm tự động (MOS, WER, CER)**:  
- Sử dụng notebook `score.ipynb`  
- Cần cài thêm `whisper`, `torchaudio`

