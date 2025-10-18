"""Transcribe audio and cut it into sentence-level segments using Whisper."""

import os
import torch
import whisper  # pip install openai-whisper

from _constants import LIST_VID, LIST_VID_VAD, VAD_DATA_PATH, SUBS_DATA_PATH, AUDIO_TEXT_FILE_LIST_PATH, FIELD_SEP
from _utils import load_audio_mono_16k, save_waveform_mono, resample_to_22050

# ==============================================================
# LOAD WHISPER MODEL ON CPU (QUANTIZED)
# ==============================================================
print("Loading Whisper model on CPU (quantized)...")
MODEL = whisper.load_model("small", device="cpu")

# Quantize model to save RAM (optional, safe for CPU)
MODEL.encoder = torch.quantization.quantize_dynamic(
    MODEL.encoder, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
)
MODEL.decoder = torch.quantization.quantize_dynamic(
    MODEL.decoder, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
)

HALLUCINATIONS_TEXT = "hãy subscribe cho kênh ghiền mì gõ để không bỏ lỡ những video hấp dẫn"


# ==============================================================
# FUNCTIONS
# ==============================================================

@torch.inference_mode()
def transcribe(infile: str) -> list[dict]:
    """Run Whisper transcription and handle garbage output."""
    print(f"Transcribing {os.path.basename(infile)} ...")
    result = MODEL.transcribe(infile, verbose=False, language="vi")["segments"]
    if all(seg["text"].strip().lower() == HALLUCINATIONS_TEXT for seg in result):
        print("Garbage transcription detected — retrying without previous context...")
        result = MODEL.transcribe(infile, verbose=False, language="vi", condition_on_previous_text=False)["segments"]
    return result


def cut_audio_and_save_text(infile: str, res_trans: list[dict], file_id: str, outdir: str, text_file_buffer):
    """Cut audio into sentence clips and record transcripts."""
    # Load mono 16k audio
    audio = load_audio_mono_16k(infile)
    sr = 16000
    prev_txt = ""

    for chunk in res_trans:
        txt = chunk["text"].strip()
        if not txt:
            continue

        start = float(chunk["start"])
        end = float(chunk["end"])
        start_samp = int(start * sr)
        end_samp = int(end * sr)

        clip = audio[start_samp:end_samp]

        # skip empty or duplicate or bad text
        tmp = txt.lower()
        outfile = f"{file_id}_{chunk['id']:04d}.wav"
        if tmp == HALLUCINATIONS_TEXT or tmp == prev_txt:
            print("skip", outfile, "⇐ hallucinated:", txt)
            continue
        elif len(txt.split()) < 2:
            print("skip", outfile, "⇐ too short:", txt)
            continue

        # Save segment
        out_path = os.path.join(outdir, outfile)
        save_waveform_mono(out_path, clip, sr)

        # Write metadata line
        text_file_buffer.write(outfile + FIELD_SEP + txt + "\n")
        prev_txt = tmp


# ==============================================================
# MAIN
# ==============================================================
def main():
    os.makedirs(SUBS_DATA_PATH, exist_ok=True)
    os.makedirs(AUDIO_TEXT_FILE_LIST_PATH, exist_ok=True)

    transcription_file = os.path.join(AUDIO_TEXT_FILE_LIST_PATH, "_all.txt")
    with open(transcription_file, "w", encoding="utf-8") as f:
        f.write("")

    for vid_id in LIST_VID_VAD:
        infile = os.path.join(VAD_DATA_PATH, vid_id)
        print("DEBUG:", infile)

        if not os.path.exists(infile):
            print(f"{vid_id} not found")
            continue

        print(f"{vid_id} -> transcribing and cutting...")
        res_trans = transcribe(infile)

        with open(transcription_file, "a", encoding="utf-8") as f:
            cut_audio_and_save_text(infile, res_trans, vid_id, SUBS_DATA_PATH, f)

    print("Done! Check:", transcription_file)

    resample_to_22050(SUBS_DATA_PATH, "data/data_22k_matchaTTS")
    
    print("Resample 22.05kHz for Matcha: done")

if __name__ == "__main__":
    main()
