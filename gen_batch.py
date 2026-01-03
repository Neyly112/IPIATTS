import os
import datetime as dt
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm
from pathlib import Path
from functools import partial
import omegaconf
import random
import shutil  # <--- THƯ VIỆN ĐỂ COPY FILE

# --- Import các module của Matcha ---
from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import intersperse, get_user_data_dir, assert_model_downloaded

# ==========================================
# CẤU HÌNH NGƯỜI DÙNG (SỬA LẠI ĐƯỜNG DẪN)
# ==========================================
CHECKPOINT_PATH = "logs\\matcha_vi\\checkpoints\\checkpoint_epoch599_new_voice.ckpt"
TEST_FILE_PATH = "data\\99-audio-text-file-list\\audio_text_test_filelist.txt"

# [QUAN TRỌNG] Đường dẫn thư mục chứa file WAV GỐC để copy sang
SOURCE_WAV_FOLDER = "data\\subs"

OUTPUT_FOLDER = "ket_qua_test_full_pack_1"  # Folder tổng
NUM_SAMPLES = 50                          # Số lượng mẫu
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCODER_NAME = "hifigan_univ_v1"

# URL Vocoder
VOCODER_URLS = {
    "hifigan_T2_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1",
    "hifigan_univ_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/g_02500000",
}

# ==========================================
# XỬ LÝ AN TOÀN CHO TORCH
# ==========================================
torch.load = partial(torch.load, weights_only=False)
torch.serialization.add_safe_globals([
    omegaconf.dictconfig.DictConfig, 
    omegaconf.listconfig.ListConfig
])

# ==========================================
# CÁC HÀM HỖ TRỢ
# ==========================================

def load_hifigan(checkpoint_path, device):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)["generator"])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan

def load_vocoder(vocoder_name, device):
    save_dir = get_user_data_dir()
    vocoder_path = save_dir / f"{vocoder_name}"
    assert_model_downloaded(vocoder_path, VOCODER_URLS[vocoder_name])
    print(f"[!] Loading Vocoder: {vocoder_name} from {vocoder_path}")
    vocoder = load_hifigan(vocoder_path, device)
    denoiser = Denoiser(vocoder, mode="zeros")
    return vocoder, denoiser

def load_matcha(checkpoint_path, device):
    print(f"[!] Loading Matcha Model form: {checkpoint_path}")
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    _ = model.eval()
    return model

def process_text(text: str, device: torch.device):
    x = torch.tensor(
        intersperse(text_to_sequence(text, ["basic_cleaners_phothong"])[0], 0),
        dtype=torch.long,
        device=device,
    )[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    return {"x": x, "x_lengths": x_lengths}

def to_waveform(mel, vocoder, denoiser=None, denoiser_strength=0.00025):
    audio = vocoder(mel).clamp(-1, 1)
    if denoiser is not None:
        audio = denoiser(audio.squeeze(), strength=denoiser_strength).cpu().squeeze()
    return audio.cpu().squeeze()

# ==========================================
# HÀM MAIN
# ==========================================

def main():
    print(f"--- Bắt đầu quy trình tạo Audio & Copy dữ liệu ---")
    print(f"Device: {DEVICE}")
    
    # 1. Tạo cấu trúc thư mục output
    # gen: chứa file máy tạo
    # ref: chứa file gốc
    folder_gen = os.path.join(OUTPUT_FOLDER, "gen")
    folder_ref = os.path.join(OUTPUT_FOLDER, "ref")
    
    os.makedirs(folder_gen, exist_ok=True)
    os.makedirs(folder_ref, exist_ok=True)
    
    # File text để lưu lại các câu đã chọn
    out_text_path = os.path.join(OUTPUT_FOLDER, "text.txt")

    # 2. Load Models
    try:
        model = load_matcha(CHECKPOINT_PATH, DEVICE)
        vocoder, denoiser = load_vocoder(VOCODER_NAME, DEVICE)
    except Exception as e:
        print(f"Lỗi khi load model: {e}")
        return

    # 3. Đọc và chọn ngẫu nhiên file test
    if not os.path.exists(TEST_FILE_PATH):
        print(f"Lỗi: Không tìm thấy file test tại {TEST_FILE_PATH}")
        return

    print("Đang đọc file list...")
    with open(TEST_FILE_PATH, 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f if line.strip()]

    total_lines = len(all_lines)
    print(f"Tổng số dòng trong file: {total_lines}")

    # --- LOGIC RANDOM ---
    if NUM_SAMPLES > 0 and NUM_SAMPLES < total_lines:
        print(f"-> Đang chọn ngẫu nhiên {NUM_SAMPLES} mẫu...")
        selected_lines = random.sample(all_lines, NUM_SAMPLES)
    else:
        print(f"-> Chạy hết toàn bộ.")
        selected_lines = all_lines
    # --------------------

    # Mở file text.txt để ghi dần
    with open(out_text_path, "w", encoding="utf-8") as f_out:
        
        # 4. Vòng lặp chạy Inference
        for line in tqdm(selected_lines, desc="Processing"):
            parts = line.split('|')
            if len(parts) < 2: continue
                
            filename = parts[0].strip()
            text = parts[1].strip()

            if not filename.endswith(".wav"): filename += ".wav"

            # --- A. TẠO AUDIO MÁY (GEN) ---
            try:
                text_processed = process_text(text, DEVICE)
                with torch.inference_mode():
                    output = model.synthesise(
                        text_processed["x"], text_processed["x_lengths"],
                        n_timesteps=30, temperature=0.667, spks=None, length_scale=1.0
                    )
                    waveform = to_waveform(output["mel"], vocoder, denoiser)

                # Lưu vào folder 'gen'
                path_gen = os.path.join(folder_gen, filename)
                sf.write(path_gen, waveform.numpy(), 22050, "PCM_24")
            except Exception as e:
                print(f"Lỗi tạo audio {filename}: {e}")
                continue # Nếu tạo lỗi thì bỏ qua, không copy ref luôn

            # --- B. COPY AUDIO GỐC (REF) ---
            try:
                # Giả sử file list chỉ chứa tên file (file.wav), ta nối với đường dẫn gốc
                src_path = os.path.join(SOURCE_WAV_FOLDER, filename)
                dst_path = os.path.join(folder_ref, filename)
                
                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_path)
                else:
                    print(f"\n[Cảnh báo] Không tìm thấy file gốc: {src_path}")
            except Exception as e:
                print(f"\nLỗi copy file gốc {filename}: {e}")

            # --- C. LƯU TEXT ---
            # Ghi dòng này vào file text.txt
            f_out.write(f"{filename}|{text}\n")

    print("=" * 50)
    print(f"HOÀN THÀNH!")
    print(f"1. Audio máy: {folder_gen}")
    print(f"2. Audio gốc: {folder_ref}")
    print(f"3. Text list: {out_text_path}")
    print("=" * 50)

if __name__ == "__main__":
    main()