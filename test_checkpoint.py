"""
Test Checkpoint Script - Kiểm tra model đã train
Tự động load checkpoint mới nhất và tạo audio mẫu
"""
from pathlib import Path
import argparse
import soundfile as sf
from matcha.cli import load_vocoder, to_waveform
from matcha.utils.utils import intersperse
from matcha.text import text_to_sequence
from matcha.models.matcha_tts import MatchaTTS
import torch


# Configuration
CHECKPOINT_ROOT = Path("outputs/matcha_prosody/checkpoints")
VOCODER_CHECKPOINT = Path("matcha/hifigan/checkpoints/g_02500000")
OUTPUT_DIR = Path("outputs/test_samples")

# Test sentences
TEST_SENTENCES = [
    "xin chào, hôm nay tôi học về trí tuệ nhân tạo",
    "đây là giọng nói tiếng việt với prosody tự nhiên",
    "chúng tôi đang kiểm tra mô hình text to speech",
]


def resolve_checkpoint(checkpoint_root: Path) -> Path:
    """Tự động tìm checkpoint tốt nhất hoặc mới nhất"""
    # Tìm checkpoint có val_loss thấp nhất
    checkpoints = sorted(
        checkpoint_root.glob("matcha-prosody-epoch=*-val_loss=*.ckpt"),
        key=lambda p: float(p.stem.split("val_loss=")[-1])
    )
    if checkpoints:
        print(f"[INFO] Found best checkpoint: {checkpoints[0].name}")
        return checkpoints[0]
    
    # Fallback: last.ckpt
    last_ckpt = checkpoint_root / "last.ckpt"
    if last_ckpt.exists():
        print(f"[INFO] Using last checkpoint: {last_ckpt.name}")
        return last_ckpt
    
    # Không tìm thấy gì
    available = list(checkpoint_root.rglob("*.ckpt"))
    raise FileNotFoundError(
        f"No checkpoint found in {checkpoint_root}\n"
        f"Available: {[p.name for p in available]}"
    )


def load_checkpoint(path: Path, device: torch.device) -> MatchaTTS:
    """Load model từ checkpoint"""
    print(f"[LOADING] Checkpoint: {path}")
    with torch.serialization.safe_globals([argparse.Namespace]):
        model = MatchaTTS.load_from_checkpoint(path, weights_only=False)
    model.eval()
    model = model.to(device)
    return model


def synthesize_text(model: MatchaTTS, vocoder, denoiser, text: str, device: torch.device):
    """Synthesize audio từ text"""
    # Convert text → phoneme IDs
    x = torch.tensor(
        intersperse(text_to_sequence(text, ["basic_cleaners_phothong"])[0], 0)
    )[None].to(device)
    x_lengths = torch.tensor([x.shape[-1]], device=device)
    
    # Synthesize mel
    with torch.no_grad():
        output = model.synthesise(
            x, x_lengths,
            n_timesteps=10,
            temperature=0.667,
            length_scale=1.0,
        )
    
    # Convert mel → audio
    audio = to_waveform(output["mel"], vocoder, denoiser)
    
    return audio.cpu().numpy(), output["rtf"]


def main():
    print("=" * 80)
    print("MATCHA-TTS CHECKPOINT TESTING")
    print("=" * 80)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using: {device}")
    
    # Load checkpoint
    checkpoint_path = resolve_checkpoint(CHECKPOINT_ROOT)
    model = load_checkpoint(checkpoint_path, device)
    print("✅ Model loaded successfully!")
    
    # Load vocoder
    print(f"[LOADING] Vocoder: {VOCODER_CHECKPOINT}")
    vocoder, denoiser = load_vocoder("hifigan_univ_v1", VOCODER_CHECKPOINT, device)
    print("✅ Vocoder loaded successfully!")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Synthesize test sentences
    print("\n" + "=" * 80)
    print("GENERATING TEST SAMPLES")
    print("=" * 80)
    
    for i, text in enumerate(TEST_SENTENCES, 1):
        print(f"\n[{i}/{len(TEST_SENTENCES)}] Text: {text}")
        
        audio, rtf = synthesize_text(model, vocoder, denoiser, text, device)
        
        # Save audio
        output_path = OUTPUT_DIR / f"sample_{i:02d}.wav"
        sf.write(output_path, audio, 22050, "PCM_24")
        
        print(f"  ✅ Saved: {output_path} (RTF: {rtf:.4f})")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETED!")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print(f"Generated {len(TEST_SENTENCES)} audio samples")


if __name__ == "__main__":
    main()
