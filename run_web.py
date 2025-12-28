import os
import sys
import torch
import numpy as np
import gradio as gr
from pathlib import Path

# --- CHỐNG CRASH DO MATPLOTLIB ---
import matplotlib
matplotlib.use('Agg') 

# --- CẤU HÌNH (SỬA ĐƯỜNG DẪN CỦA BẠN VÀO ĐÂY) ---
MATCHA_CHECKPOINT = r"D:\Bai Tap\DACNTT\Matcha-TTS_Right\logs\matcha_vi\checkpoints\checkpoint_epoch379_new.ckpt"
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCODER_URL = "D:\Bai Tap\DACNTT\Matcha-TTS_Right\logs\matcha_vi\vocoder\hifigan_univ_v1.pt"

# ==============================================================================
# 🛑 FIX LỖI BẢO MẬT PYTORCH 2.6
# ==============================================================================
import omegaconf
import typing
import builtins 

try:
    torch.serialization.add_safe_globals([
        omegaconf.dictconfig.DictConfig, 
        omegaconf.listconfig.ListConfig,
        omegaconf.base.ContainerMetadata,
        typing.Any,
        np.dtype,
        builtins.dict, builtins.list, builtins.set, builtins.tuple
    ])
except AttributeError: pass

_org_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False 
    return _org_load(*args, **kwargs)
torch.load = patched_load
# ==============================================================================

print("🔥 BẮT ĐẦU CHƯƠNG TRÌNH...", flush=True)

# --- IMPORT ---
try:
    from matcha.hifigan.config import v1
    from matcha.hifigan.env import AttrDict
    from matcha.hifigan.models import Generator as HiFiGAN
    from matcha.models.matcha_tts import MatchaTTS
    from matcha.text import text_to_sequence
    from matcha.utils.utils import get_user_data_dir, intersperse
except ImportError as e:
    print(f"❌ LỖI IMPORT: {e}")
    input("🔴 Bấm Enter để thoát...")
    sys.exit(1)

# --- CÁC HÀM HỖ TRỢ ---
def process_text(text: str, device: torch.device):
    x = torch.tensor(
        intersperse(text_to_sequence(text, ["basic_cleaners_phothong"])[0], 0),
        dtype=torch.long,
        device=device,
    )[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    return {"x": x, "x_lengths": x_lengths}

def load_vocoder(device):
    save_dir = get_user_data_dir()
    save_dir.mkdir(exist_ok=True, parents=True)
    vocoder_path = save_dir / "g_02500000"
    
    if not vocoder_path.exists():
        print(f"⬇️ Đang tải Vocoder từ Internet...")
        try:
            torch.hub.download_url_to_file(VOCODER_URL, str(vocoder_path))
        except Exception as e:
            print(f"❌ Lỗi tải file: {e}")
            sys.exit(1)

    try:
        h = AttrDict(v1)
        hifigan = HiFiGAN(h).to(device)
        state_dict = torch.load(vocoder_path, map_location=device)
        hifigan.load_state_dict(state_dict["generator"])
        hifigan.eval()
        hifigan.remove_weight_norm()
        return hifigan
    except Exception as e:
        print(f"❌ Lỗi Vocoder: {e}")
        sys.exit(1)

# --- KHỞI TẠO ---
print("\n⏳ Đang khởi tạo Model...", flush=True)
if not os.path.exists(MATCHA_CHECKPOINT):
    print(f"❌ Sai đường dẫn checkpoint: {MATCHA_CHECKPOINT}")
    sys.exit(1)

try:
    model = MatchaTTS.load_from_checkpoint(MATCHA_CHECKPOINT, map_location=DEVICE)
    model.eval().to(DEVICE)
    vocoder = load_vocoder(DEVICE)
    print("✅ Model & Vocoder OK.")
except Exception as e:
    print(f"\n❌ LỖI LOAD MODEL: {e}")
    sys.exit(1)

# --- WEB UI ---
print("\n🚀 Đang khởi động Web Server...", flush=True)

@torch.inference_mode()
def run_tts(text, speed_user, steps, temp):
    if not text.strip(): return None
    print(f"🗣️ Input: {text[:20]}... | Speed: {speed_user}x")
    
    try:
        # --- [SỬA ĐỔI QUAN TRỌNG] ---
        # Nghịch đảo giá trị để đúng logic người dùng
        # Người dùng chọn 1.5 (Nhanh) -> Model nhận 1/1.5 = 0.66 (Ngắn)
        actual_length_scale = 1.0 / speed_user
        
        processed = process_text(text, DEVICE)
        out_matcha = model.synthesise(
            processed["x"], 
            processed["x_lengths"],
            n_timesteps=int(steps),
            temperature=temp,
            length_scale=actual_length_scale, # Truyền giá trị đã nghịch đảo
            spks=None
        )
        mel = out_matcha["mel"]
        audio = vocoder(mel).clamp(-1, 1).cpu().squeeze().numpy()
        
        max_amp = np.max(np.abs(audio))
        if max_amp > 0.01:
            audio = audio / max_amp * 0.9
        return (22050, audio)
        
    except Exception as e:
        print(f"❌ LỖI KHI ĐỌC: {e}")
        return None

with gr.Blocks(title="Matcha-TTS Local") as demo:
    gr.Markdown("## 🍵 Matcha-TTS Tiếng Việt")
    with gr.Row():
        inp = gr.Textbox(label="Văn bản", value="Xin chào Việt Nam.", lines=2)
        btn = gr.Button("🔊 Đọc", variant="primary")
    with gr.Row():
        # Slider chỉnh lại nhãn cho dễ hiểu
        sld_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Tốc độ đọc (1.0 = Chuẩn, >1 = Nhanh)")
        sld_steps = gr.Slider(10, 50, value=30, step=10, label="Chất lượng (Steps)")
        sld_temp = gr.Slider(0.1, 1.0, value=0.667, label="Độ biểu cảm (Temp)")
    out = gr.Audio(label="Audio", type="numpy", autoplay=True)
    btn.click(run_tts, [inp, sld_speed, sld_steps, sld_temp], [out])

try:
    demo.launch(inbrowser=True)
except Exception as e:
    print(f"❌ Lỗi khởi động Web: {e}")
    input("🔴 Bấm Enter để kết thúc...")