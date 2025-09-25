
import os
import torch
import torchaudio
import torchaudio.functional as AF

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_audio_mono_16k(filepath: str, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(filepath)  # [C, T]
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
