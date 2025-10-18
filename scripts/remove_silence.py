import os
import sys
import time
import torch
from tqdm import tqdm

# Bảo đảm import được các module cùng thư mục
sys.path.insert(0, os.path.dirname(__file__))

# ---- Chỉ import đúng các biến có trong _constants.py của bạn ----
from _constants import (
    RAW_DATA_PATH, VAD_DATA_PATH, SEGMENTS_DIR,
    LIST_VID
)
from _utils import (
    ensure_dir, load_audio_mono_16k, save_waveform_mono,
    cut_by_timestamps, concat_segments
)

# ==== Cấu hình nội bộ cho script (không phụ thuộc _constants.py) ====
TARGET_SR = 16000               # Silero VAD cần 16 kHz mono
SAVE_SPEECH_ONLY_FILE = True    # True: gộp tất cả đoạn có lời nói thành 1 file
SAVE_EACH_SEGMENT = False       # True: xuất từng đoạn rời vào SEGMENTS_DIR
# ====================================================================

def load_silero_vad():
    """
    Tải model Silero VAD từ torch.hub.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", onnx=False)
    (get_speech_timestamps, _, read_audio, _, _) = utils  # read_audio không dùng vì ta đã có torchaudio
    model = model.to(device)
    model.eval()
    return model, get_speech_timestamps, device

def run_vad_on_file(in_path: str, out_base: str, model, get_speech_timestamps, device: str):
    """
    Chạy VAD trên 1 file:
    - Đọc file -> mono 16k
    - Tìm timestamps có lời nói
    - (Tuỳ chọn) xuất từng đoạn
    - (Tuỳ chọn) gộp tất cả đoạn -> 1 file duy nhất
    """
    # 1) Load audio
    wav = load_audio_mono_16k(in_path, target_sr=TARGET_SR)  # [T], float32
    if wav.numel() == 0:
        print(f"[WARN] Empty waveform: {in_path}")
        return

    # 2) Silero cần input 1D float ở sr=16k, trên device phù hợp
    wav_dev = wav.to(device)

    # 3) Lấy timestamps (đơn vị sample)
    timestamps = get_speech_timestamps(wav_dev, model, sampling_rate=TARGET_SR)
    # Chuyển về CPU để cắt
    wav = wav.cpu()

    # 4) Cắt các đoạn speech
    segments = cut_by_timestamps(wav, TARGET_SR, timestamps)

    # 5) Xuất file
    # 5a) Xuất từng đoạn
    if SAVE_EACH_SEGMENT and segments:
        for idx, seg in enumerate(segments, 1):
            seg_out = os.path.join(SEGMENTS_DIR, f"{out_base}_seg{idx:04d}.wav")
            save_waveform_mono(seg_out, seg, sr=TARGET_SR)

    # 5b) Xuất file gộp (speech-only)
    if SAVE_SPEECH_ONLY_FILE:
        merged = concat_segments(segments)
        out_path = os.path.join(VAD_DATA_PATH, f"{out_base}.wav")  # giữ tên gốc, đuôi wav
        save_waveform_mono(out_path, merged, sr=TARGET_SR)

def main():
    t0 = time.time()

    # Chuẩn bị thư mục
    ensure_dir(VAD_DATA_PATH)
    if SAVE_EACH_SEGMENT:
        ensure_dir(SEGMENTS_DIR)

    # Dùng LIST_VID từ _constants.py (đã quét sẵn data/raw)
    files = LIST_VID
    if not files:
        print(f"[INFO] Không tìm thấy audio trong {RAW_DATA_PATH}")
        return

    # Load Silero VAD
    print("[INFO] Loading Silero VAD...")
    model, get_speech_timestamps, device = load_silero_vad()
    print(f"[INFO] Model loaded on {device}.")

    # Xử lý từng file
    print(f"[INFO] Found {len(files)} file(s) in {RAW_DATA_PATH}")
    for fname in tqdm(files, ncols=80, desc="VAD"):
        in_path = os.path.join(RAW_DATA_PATH, fname)
        base, _ = os.path.splitext(fname)
        try:
            run_vad_on_file(in_path, base, model, get_speech_timestamps, device)
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

    print(f"[DONE] Elapsed: {time.time() - t0:.2f}s")
    print(f"       Output: {VAD_DATA_PATH}")

if __name__ == "__main__":
    main()
