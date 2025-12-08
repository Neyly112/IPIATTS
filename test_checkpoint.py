from pathlib import Path

import argparse
import soundfile as sf
from matcha.cli import load_vocoder, to_waveform
from matcha.utils.utils import intersperse
from matcha.text import text_to_sequence
from matcha.models.matcha_tts import MatchaTTS
import torch

# Load model (prosody tự động được bật)
CHECKPOINT_ROOT = Path("outputs/matcha_prosody/checkpoints")
TARGET_CHECKPOINT = CHECKPOINT_ROOT / \
    "matcha-prosody-epoch=001-loss" / "val_epoch=3.954.ckpt"
VOCODER_CHECKPOINT = Path(
     "matcha/hifigan/checkpoints/g_02500000"
)


def resolve_checkpoint(preferred: Path) -> Path:
    if preferred.exists():
        return preferred
    last_ckpt = CHECKPOINT_ROOT / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt
    available = sorted(p for p in CHECKPOINT_ROOT.rglob("*.ckpt"))
    raise FileNotFoundError(
        f"no checkpoint found under {CHECKPOINT_ROOT}; tried {preferred} and {last_ckpt}, "
        f"available: {[p.name for p in available]}"
    )


checkpoint_path = resolve_checkpoint(TARGET_CHECKPOINT)


def load_checkpoint(path: Path) -> MatchaTTS:
    with torch.serialization.safe_globals([argparse.Namespace]):
        return MatchaTTS.load_from_checkpoint(path, weights_only=False)


model = load_checkpoint(checkpoint_path)
model.eval()

# Chuyển sang GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("✅ Model loaded!")


# BƯỚC 2: Chuẩn bị text input


# Text tiếng Việt
text = "xin chào, hôm nay tôi học về trí tuệ nhân tạo"

# Convert text → phoneme IDs
x = torch.tensor(
    intersperse(text_to_sequence(text, ["basic_cleaners_phothong"])[0], 0)
)[None].to(device)

x_lengths = torch.tensor([x.shape[-1]], device=device)

print(f"Input shape: {x.shape}")


# BƯỚC 3: Synthesize mel-spectrogram

with torch.no_grad():
    output = model.synthesise(
        x,
        x_lengths,
        n_timesteps=10,      # Số ODE steps (10-50, càng nhiều càng chất lượng)
        temperature=0.667,    # Sampling temperature
        length_scale=1.0,     # Speaking rate (>1 = chậm, <1 = nhanh)
    )

mel = output["mel"]           # Mel-spectrogram
mel_lengths = output["mel_lengths"]
rtf = output["rtf"]           # Real-time factor

print(f"Mel shape: {mel.shape}")
print(f"RTF: {rtf:.4f}")


# BƯỚC 4: Convert mel → audio (với HiFi-GAN vocoder)


# Load vocoder
vocoder, denoiser = load_vocoder(
    "hifigan_univ_v1",
    VOCODER_CHECKPOINT,
    device
)

# Convert mel → waveform
audio = to_waveform(mel, vocoder, denoiser)

# Lưu file
sf.write("output.wav", audio.cpu().numpy(), 22050, "PCM_24")
print("✅ Đã lưu: output.wav")


# BƯỚC 5: Script hoàn chỉnh


"""
Synthesis script - Sử dụng Matcha-TTS với Prosody
"""

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load model
model = load_checkpoint(checkpoint_path).to(device).eval()

# 3. Load vocoder
vocoder, denoiser = load_vocoder(
    "hifigan_univ_v1",
    VOCODER_CHECKPOINT,
    device
)

# 4. Synthesize
text = "xin chào, đây là giọng nói tiếng việt với prosody tự nhiên"

x = torch.tensor(
    intersperse(text_to_sequence(text, ["basic_cleaners_phothong"])[0], 0)
)[None].to(device)
x_lengths = torch.tensor([x.shape[-1]], device=device)

with torch.no_grad():
    output = model.synthesise(x, x_lengths, n_timesteps=10)
    audio = to_waveform(output["mel"], vocoder, denoiser)

# 5. Save
sf.write("output2.wav", audio.cpu().numpy(), 22050, "PCM_24")
print(f"✅ Saved: output.wav (RTF: {output['rtf']:.4f})")
