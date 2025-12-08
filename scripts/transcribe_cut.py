"""Transcribe audio and cut it into sentence-level segments using Whisper."""

import os
import torch
import whisper  # pip install openai-whisper

from _constants import LIST_VID, LIST_VID_VAD, VAD_DATA_PATH, SUBS_DATA_PATH, AUDIO_TEXT_FILE_LIST_PATH, FIELD_SEP
from _utils import save_waveform_mono
import torchaudio
import torchaudio.functional as AF

# ==============================================================
# CONFIG
# ==============================================================
WHISPER_MODEL = "small"  # "tiny", "base", "small", "medium", "large" (large cần GPU mạnh)
HALLUCINATIONS_TEXT = "hãy subscribe cho kênh ghiền mì gõ để không bỏ lỡ những video hấp dẫn"
MIN_DURATION_SEC = 0.4   # Bỏ qua đoạn audio < 0.4s
MIN_WORDS = 2            # Bỏ qua câu < 2 từ

# ==============================================================
# LOAD WHISPER MODEL (AUTO DETECT GPU/CPU)
# ==============================================================
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper '{WHISPER_MODEL}' model on {device}...")
    
    model = whisper.load_model(WHISPER_MODEL, device="cpu")  # Load to CPU first
    
    # Quantize to save RAM (optional but recommended for CPU)
    try:
        model.encoder = torch.quantization.quantize_dynamic(
            model.encoder, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
        )
        model.decoder = torch.quantization.quantize_dynamic(
            model.decoder, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
        )
        print("✓ Model quantized (8-bit)")
    except Exception as e:
        print(f"[WARN] Quantization failed: {e}")
    
    # Move to target device
    model = model.to(device)
    print(f"✓ Model ready on {device}")
    return model

MODEL = load_whisper_model()


# ==============================================================
# FUNCTIONS
# ==============================================================

@torch.inference_mode()
def transcribe(infile: str) -> list[dict]:
    """
    Run Whisper transcription with Vietnamese optimization.
    Handles garbage output by retrying without context.
    """
    print(f"Transcribing {os.path.basename(infile)} ...")
    
    # Load audio using torchaudio (không cần FFmpeg)
    waveform, sample_rate = torchaudio.load(infile)
    if waveform.shape[0] > 1:  # Convert to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != 16000:  # Resample to 16kHz
        waveform = AF.resample(waveform, sample_rate, 16000)
    audio_np = waveform.squeeze().numpy()
    
    # Initial transcription with prompt
    result = MODEL.transcribe(
        audio_np,  # Pass numpy array thay vì file path
        verbose=False, 
        language="vi",
        initial_prompt="Chính tả tiếng Việt, giọng kể chuyện."
    )["segments"]
    
    # Clear GPU cache if using CUDA
    device = next(MODEL.parameters()).device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Check if all segments are hallucinations
    if all(seg["text"].strip().lower() == HALLUCINATIONS_TEXT for seg in result):
        print("  → Garbage detected, retrying without context...")
        result = MODEL.transcribe(
            audio_np,  # Dùng audio_np thay vì infile
            verbose=False, 
            language="vi",
            condition_on_previous_text=False
        )["segments"]
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    return result


def should_keep_segment(txt: str, start: float, end: float, prev_txt_lower: str) -> tuple[bool, str]:
    """
    Determine if a segment should be kept.
    Returns: (keep: bool, reason: str)
    """
    txt_stripped = txt.strip()
    txt_lower = txt_stripped.lower()
    
    # Filter hallucinations
    if txt_lower == HALLUCINATIONS_TEXT:
        return (False, "hallucinated")
    
    # Filter duplicates
    if txt_lower == prev_txt_lower:
        return (False, "duplicate")
    
    # Filter too short text
    if " " not in txt_stripped or len(txt_stripped.split()) < MIN_WORDS:
        return (False, "too_short_text")
    
    # Filter too short audio
    duration = end - start
    if duration < MIN_DURATION_SEC:
        return (False, f"too_short_audio ({duration:.2f}s)")
    
    return (True, "")


def cut_audio_and_save_text(infile: str, res_trans: list[dict], file_id: str, outdir: str, text_file_buffer):
    """Cut audio into sentence clips and record transcripts with filtering."""
    # Load audio và chuyển sang mono 22.05kHz ngay từ đầu
    TARGET_SR = 22050  # Sample rate chuẩn cho TTS
    
    wav, sr = torchaudio.load(infile)
    
    # Convert to mono
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    
    # Resample to 22.05kHz nếu cần
    if sr != TARGET_SR:
        wav = AF.resample(wav, sr, TARGET_SR)
    
    audio = wav.squeeze(0)  # [T]
    sr = TARGET_SR
    prev_txt_lower = ""
    kept_count = 0
    skipped_count = 0

    for chunk in res_trans:
        txt = chunk["text"].strip()
        if not txt:
            continue

        start = float(chunk["start"])
        end = float(chunk["end"])
        
        # Check if should keep this segment
        should_keep, reason = should_keep_segment(txt, start, end, prev_txt_lower)
        
        outfile = f"{file_id}_{chunk['id']:04d}.wav"
        
        if not should_keep:
            print(f"  skip {outfile:>25} ⇐ {reason}: {txt[:50]}...")
            skipped_count += 1
            continue
        
        # Extract audio segment
        start_samp = int(start * sr)
        end_samp = int(end * sr)
        clip = audio[start_samp:end_samp]

        # Save segment
        out_path = os.path.join(outdir, outfile)
        save_waveform_mono(out_path, clip, sr)

        # Write metadata line
        text_file_buffer.write(outfile + FIELD_SEP + txt + "\n")
        prev_txt_lower = txt.lower()
        kept_count += 1

    print(f"  ✓ Kept: {kept_count}, Skipped: {skipped_count}")



# ==============================================================
# MAIN
# ==============================================================
def main():
    print("=" * 80)
    print("WHISPER TRANSCRIPTION & AUDIO SEGMENTATION")
    print("=" * 80)
    print(f"Model: {WHISPER_MODEL}")
    print(f"Input:  {VAD_DATA_PATH}")
    print(f"Output: {SUBS_DATA_PATH}")
    print(f"Filters: MIN_DURATION={MIN_DURATION_SEC}s, MIN_WORDS={MIN_WORDS}")
    print("=" * 80)
    print()
    
    os.makedirs(SUBS_DATA_PATH, exist_ok=True)
    os.makedirs(AUDIO_TEXT_FILE_LIST_PATH, exist_ok=True)

    transcription_file = os.path.join(AUDIO_TEXT_FILE_LIST_PATH, "_all.txt")
    
    # Initialize empty file
    with open(transcription_file, "w", encoding="utf-8") as f:
        f.write("")

    # Get list of audio files
    files_to_process = LIST_VID_VAD
    if not files_to_process:
        print(f"[WARN] No audio files found in {VAD_DATA_PATH}")
        return
    
    print(f"Found {len(files_to_process)} file(s) to process")
    print()

    # Process each file
    for idx, vid_id in enumerate(files_to_process, 1):
        infile = os.path.join(VAD_DATA_PATH, vid_id)
        
        if not os.path.exists(infile):
            print(f"[{idx}/{len(files_to_process)}] {vid_id} - NOT FOUND, skipping")
            continue

        print(f"[{idx}/{len(files_to_process)}] Processing: {vid_id}")
        
        # Transcribe
        res_trans = transcribe(infile)
        
        # Cut and save
        with open(transcription_file, "a", encoding="utf-8") as f:
            cut_audio_and_save_text(infile, res_trans, vid_id, SUBS_DATA_PATH, f)
        
        print()

    print("=" * 80)
    print("✅ TRANSCRIPTION COMPLETED!")
    print("=" * 80)
    print(f"Transcription file: {transcription_file}")
    print(f"Audio segments:     {SUBS_DATA_PATH} (22.05kHz mono)")
    print()
    print("=" * 80)
    print("Next step:")
    print("  python scripts\\process_cleaner.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
