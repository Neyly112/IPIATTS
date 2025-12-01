
import os
import torch
import torchaudio
import torchaudio.functional as AF
import soundfile as sf
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_audio_mono_16k(filepath: str, target_sr: int = 16000) -> torch.Tensor:
    try:
        wav, sr = torchaudio.load(filepath)
    except Exception as e:
        print(f"[ torchaudio.load failed: {e}] -> trying soundfile for {filepath}")
        data, sr = sf.read(filepath, always_2d=True)
        wav = torch.from_numpy(data.T)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = AF.resample(wav, sr, target_sr)
    wav = wav.squeeze(0).contiguous()
    if wav.dtype != torch.float32:
        wav = wav.to(torch.float32)
    return wav.cpu()

def save_waveform_mono(path: str, wav_1d: torch.Tensor, sr: int = 16000):
    ensure_dir(os.path.dirname(path))
    torchaudio.save(path, wav_1d.unsqueeze(0), sr)

def cut_by_timestamps(wav_1d: torch.Tensor, sr: int, timestamps: list) -> list:
    clips = []
    n = wav_1d.numel()
    for seg in timestamps:
        s = max(0, min(int(seg["start"]), n))
        e = max(0, min(int(seg["end"]), n))
        if e > s:
            clips.append(wav_1d[s:e].clone())
    return clips

def concat_segments(segments: list) -> torch.Tensor:
    if not segments:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(segments, dim=0)

def resample_to_22050(input_dir: str, output_dir: str):
    """
    Resample all audio files in input_dir to 22.05kHz mono and save to output_dir.
    This is the standard sample rate for TTS (e.g., LJSpeech, Matcha-TTS).
    """
    import glob
    from tqdm import tqdm
    
    TARGET_SR = 22050
    ensure_dir(output_dir)
    
    # Find all wav files
    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    
    if not wav_files:
        print(f"[WARN] No .wav files found in {input_dir}")
        return
    
    print(f"Resampling {len(wav_files)} files to {TARGET_SR}Hz...")
    
    for wav_path in tqdm(wav_files, desc="Resampling", ncols=80):
        try:
            # Load audio
            wav, sr = torchaudio.load(wav_path)
            
            # Convert to mono if stereo
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if sr != TARGET_SR:
                wav = AF.resample(wav, sr, TARGET_SR)
            
            # Save
            filename = os.path.basename(wav_path)
            output_path = os.path.join(output_dir, filename)
            torchaudio.save(output_path, wav, TARGET_SR)
            
        except Exception as e:
            print(f"[ERROR] Failed to resample {wav_path}: {e}")
    
    print(f"✓ Resampled files saved to: {output_dir}")




# -*- coding: utf-8 -*-
"""
Các hàm tiện ích xử lý audio:
- load_audio(): đọc file wav/mp3 và trả về waveform + metadata
- cut_audio_timestamp(): cắt đoạn theo thời gian
- cut_audio_timestamp_vits2(): cắt và chuẩn hoá xuống mono 22.05 kHz (chuẩn dataset TTS)
"""

import torch
from torchaudio import load as _load_file, info as _read_info, save as _save_file
from torchaudio.functional import resample as _resample

# -----------------------------
# 1. Load audio
# -----------------------------
def load_audio(filepath: str) -> dict:
    """
    Trả về dict:
        {
            "waveform": Tensor [channels, length],
            "sample_rate": int,
            "bits_per_sample": int,
            "encoding": str
        }
    """
    waveform, sample_rate = _load_file(filepath)
    metadata = _read_info(filepath)
    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
        "bits_per_sample": metadata.bits_per_sample or 16,
        "encoding": metadata.encoding or "PCM_S"
    }

# -----------------------------
# 2. Cắt đoạn audio theo thời gian (giữ nguyên định dạng)
# -----------------------------
def cut_audio_timestamp(audio_file: dict, outfile: str, start: float, end: float) -> None:
    """
    Cắt đoạn từ 'start' → 'end' (tính bằng giây).
    Lưu file cùng định dạng như gốc.
    """
    sr = audio_file["sample_rate"]
    start_frame = int(start * sr)
    end_frame = int(end * sr)
    cut_waveform = audio_file["waveform"][:, start_frame:end_frame]

    _save_file(
        outfile, cut_waveform,
        sample_rate=sr,
        bits_per_sample=audio_file["bits_per_sample"],
        encoding=audio_file["encoding"]
    )

# -----------------------------
# 3. Cắt + Chuẩn hoá theo chuẩn TTS (mono, 22.05kHz)
# -----------------------------
_LJspeech_rate = 22050  # chuẩn dataset LJSpeech

def cut_audio_timestamp_vits2(audio_file: dict, outfile: str, start: float, end: float) -> None:
    """
    Cắt audio, chuyển sang mono, resample xuống 22.05 kHz
    Dùng cho bước tạo dữ liệu huấn luyện TTS (VITS2 / Matcha)
    """
    sr = audio_file["sample_rate"]
    start_frame = int(start * sr)
    end_frame   = int(end * sr)
    cut_waveform = audio_file["waveform"][:, start_frame:end_frame]

    # chuyển sang mono (trung bình 2 kênh)
    mono_wav = cut_waveform.mean(dim=0, keepdim=True)

    # resample 48k → 22.05kHz
    down_wav = _resample(mono_wav, orig_freq=sr, new_freq=_LJspeech_rate)

    _save_file(
        outfile, down_wav,
        sample_rate=_LJspeech_rate,
        bits_per_sample=audio_file["bits_per_sample"],
        encoding=audio_file["encoding"]
    )
