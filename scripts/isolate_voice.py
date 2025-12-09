"""Separate vocal (voice) with Demucs – Optimized for Local PC (Windows/Mac/Linux)"""

import os
import torch
import torchaudio
from tqdm import tqdm
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import save_audio
import gc # Garbage collector để dọn RAM

# === 1. Cấu hình Đường dẫn (Dựa theo cấu trúc folder của bạn) ===
# Sử dụng os.getcwd() để lấy thư mục hiện tại, giúp chạy được trên mọi hệ điều hành
BASE_DIR = os.getcwd() 
RAW_VAD_PATH = os.path.join(BASE_DIR, "data", "vad_add")    # Input
VOICE_OUT_PATH = os.path.join(BASE_DIR, "data", "vad_voice") # Output

os.makedirs(VOICE_OUT_PATH, exist_ok=True)

# === 2. Cấu hình Thiết bị (Tự động nhận diện) ===
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available(): # Hỗ trợ Mac M1/M2/M3
        return "mps"
    else:
        return "cpu"

device = get_device()
print(f"[INFO] Running on device: {device.upper()}")

# Load model
# "htdemucs" nhanh và nhẹ hơn, phù hợp máy cá nhân. Nếu máy yếu quá có thể dùng "htdemucs_ft"
MODEL = get_model("htdemucs").to(device)

# === 3. Tham số xử lý ===
# LƯU Ý QUAN TRỌNG:
# Máy cá nhân (GPU 4GB-8GB VRAM) nên để 10-30s. Nếu CPU thì để bao nhiêu cũng được nhưng chậm.
CHUNK_SEC = 30       
TARGET_SR = 44100
VOCAL_INDEX = MODEL.sources.index("vocals")

def isolate_voice(infile: str, outfile: str):
    """
    Tách giọng khỏi nhạc nền.
    """
    try:
        wav, sr = torchaudio.load(infile)
    except Exception as e:
        print(f"[SKIP] Cannot load {infile}: {e}")
        return

    # Resample nếu cần
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    
    # Ép stereo (2 kênh) nếu file là mono
    if wav.size(0) == 1:
        wav = wav.repeat(2, 1)

    # Chuẩn bị chunk
    total_len = wav.size(1)
    chunk_len = int(CHUNK_SEC * TARGET_SR)
    
    # Nếu file ngắn hơn chunk, xử lý 1 lần luôn
    if total_len <= chunk_len:
        chunks = [(0, total_len)]
    else:
        chunks = [(i, min(i + chunk_len, total_len)) for i in range(0, total_len, chunk_len)]

    pieces = []
    
    # Xử lý từng đoạn
    for start, end in chunks:
        part = wav[:, start:end]
        
        # Normalize nhẹ (Optional - giữ lại từ code gốc của bạn)
        # Lưu ý: Nếu âm thanh quá nhỏ, đoạn này có thể gây nhiễu, nhưng với VAD data thì thường ổn.
        if part.std() > 0:
            part = (part - part.mean()) / (part.std() + 1e-8)
        
        part = part.to(device) # Chuyển sang GPU/MPS

        with torch.no_grad(): # Thay thế inference_mode bằng no_grad để tương thích tốt hơn
            # split=True giúp tiết kiệm VRAM bằng cách xử lý từng đoạn nhỏ bên trong Demucs
            out = apply_model(MODEL, part[None], device=device, split=True, progress=False)
            vocal = out.squeeze()[VOCAL_INDEX].cpu() # Đưa kết quả về CPU ngay

        pieces.append(vocal)
        
        # Dọn dẹp VRAM ngay lập tức
        del part, out
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    # Ghép lại và lưu
    if pieces:
        vocal_all = torch.cat(pieces, dim=1)
        save_audio(vocal_all, outfile, samplerate=TARGET_SR)
        del vocal_all
    
    del wav, pieces
    gc.collect() # Dọn RAM hệ thống

def main():
    if not os.path.exists(RAW_VAD_PATH):
        print(f"[ERROR] Input directory not found: {RAW_VAD_PATH}")
        print("Please check your 'data' folder structure.")
        return

    wav_files = [f for f in os.listdir(RAW_VAD_PATH) if f.lower().endswith((".wav", ".mp3", ".flac"))]
    print(f"[INFO] Found {len(wav_files)} files in {RAW_VAD_PATH}")

    # ncols=80 để thanh process bar gọn gàng trên terminal
    for fn in tqdm(wav_files, desc="Processing", ncols=100):
        infile = os.path.join(RAW_VAD_PATH, fn)
        outfile = os.path.join(VOICE_OUT_PATH, fn)
        
        # Bỏ qua nếu file output đã tồn tại (để chạy tiếp nếu bị ngắt)
        if os.path.exists(outfile):
            continue

        try:
            isolate_voice(infile, outfile)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[OOM ERROR] File {fn} caused Out of Memory. Try lowering CHUNK_SEC.")
                if device == "cuda": torch.cuda.empty_cache()
            else:
                print(f"\n[ERROR] {fn}: {e}")
        except Exception as e:
            print(f"\n[ERROR] {fn}: {e}")

    print(f"[DONE] Files saved to: {VOICE_OUT_PATH}")

if __name__ == "__main__":
    main()