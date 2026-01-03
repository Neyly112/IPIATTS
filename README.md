Dự án huấn luyện mô hình Text-to-Speech tiếng Việt sử dụng Matcha-TTS.

1. Cài đặt môi trường
Yêu cầu: Python 3.10, cài đặt espeak-ng.

Bash: pip install -r requirements.txt

2. Chuẩn bị dữ liệu
Audio: File .wav, Mono, Sample rate 22050 Hz.
Filelist: Định dạng filename.wav|nội dung văn bản (UTF-8).
Cấu trúc thư mục khuyến nghị:
Plaintext
data/
├── raw/              # Chứa file wav gốc (vd: utt001.wav)
└── filelists/        # Chứa file txt (audio_text_train_filelist.txt, audio_text_val_filelist.txt)

3. Huấn luyện (Training)
Cấu hình nằm trong configs/experiment/matcha_vi.yaml.

Train mới:
Bash: python matcha/train.py experiment=matcha_vi

Tiếp tục train (Resume):
Bash: python matcha/train.py experiment=matcha_vi ckpt_path="logs/matcha_vi/checkpoints/checkpoint_epoch___.ckpt" (vd: checkpoint_epoch=539.ckpt xóa dấu "=")

4. Sử dụng & Chạy thử (Inference)
Lưu ý: Cần cập nhật đường dẫn CHECKPOINT_PATH trong các file script trước khi chạy.
Sinh audio hàng loạt (Batch Inference): Đọc file list và sinh audio vào thư mục output.
Bash: python gen_batch.py

Chạy Web Demo (Gradio): Giao diện web để nhập text và nghe thử trực tiếp.
Bash: python run_web.py

5. Đánh giá chất lượng (Evaluation)
Vẽ biểu đồ Loss (Tensorboard):

Bash: python draw.py
So sánh Mel Spectrogram (Ref vs Gen):
Bash: python draw_mel.py
Chấm điểm tự động (MOS, WER, CER): Sử dụng notebook score.ipynb. Cần cài thêm whisper, torchaudio.