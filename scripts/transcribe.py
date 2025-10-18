"""Transcribe audio (vi) rồi cắt thành các segment 22.05 kHz mono cho TTS"""

import os, os.path
import torch
import whisper  # pip install openai-whisper

from _constants import VAD_DATA_PATH, SUBS_DATA_PATH, AUDIO_TEXT_FILE_LIST_PATH, FIELD_SEP
from _utils import load_audio, cut_audio_timestamp_vits2

# ===================== Whisper model (ưu tiên GPU nếu có) =====================
def _load_whisper() -> whisper.Whisper:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("large", device="cpu")  # load trước trên CPU
    # cố gắng lượng tử hóa encoder/decoder (nếu máy không đủ RAM có thể bỏ qua bước này)
    try:
        model.encoder = torch.quantization.quantize_dynamic(
            model.encoder, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
        )
        model.decoder = torch.quantization.quantize_dynamic(
            model.decoder, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
        )
    except Exception as e:
        print("[WARN] quantize_dynamic thất bại, dùng nguyên bản:", e)
    # chuyển sang thiết bị chạy
    model.to(device)
    print(f"[INFO] Whisper loaded on {device}")
    return model

MODEL = _load_whisper()

# ===================== Cấu hình & Heuristics lọc =====================
HALLUCINATIONS_TEXT = "hãy subscribe cho kênh ghiền mì gõ để không bỏ lỡ những video hấp dẫn"
MIN_SEC = 0.40         # segment ngắn hơn 0.4s thì bỏ
MIN_WORDS = 2          # câu < 2 từ thì bỏ

@torch.inference_mode()
def transcribe_vi(infile: str) -> list[dict]:
    """
    Trả về danh sách segments dạng Whisper (start, end, id, text).
    Có initial_prompt để giảm 'ảo'.
    Nếu toàn rác -> chạy lại với condition_on_previous_text=False.
    """
    params = dict(verbose=False, language="vi",
                  initial_prompt="Chính tả tiếng Việt, giọng kể chuyện.")
    res = MODEL.transcribe(infile, **params)["segments"]
    dev = next(MODEL.parameters()).device
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # nếu toàn bộ đều y hệt câu 'ảo' thì thử lại
    all_garbage = all(ch["text"].strip().lower() == HALLUCINATIONS_TEXT for ch in res) and len(res) > 0
    if all_garbage:
        print("        [NOTE] Garbage transcript → retry without previous text conditioning")
        params.pop("initial_prompt", None)
        res = MODEL.transcribe(infile, **params, condition_on_previous_text=False)["segments"]
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return res

def should_keep_chunk(txt: str, start: float, end: float, prev_txt_norm: str) -> tuple[bool, str]:
    """Quyết định có giữ đoạn text này không; trả về (keep, reason)."""
    t = txt.strip()
    t_norm = t.lower()
    if t_norm == HALLUCINATIONS_TEXT:
        return (False, "hallucinated")
    if t_norm == prev_txt_norm:
        return (False, "duplicate")
    if " " not in t or len(t.split()) < MIN_WORDS:
        return (False, "too_short_text")
    if (end - start) < MIN_SEC:
        return (False, "too_short_audio")
    return (True, "")

def process_one_file(infile: str, text_file_handle) -> None:
    """Transcribe + Cut 22.05kHz mono + ghi metadata."""
    vid_name = os.path.basename(infile)          # ví dụ 'voice1.wav'
    base = os.path.splitext(vid_name)[0]         # 'voice1'
    outdir = SUBS_DATA_PATH
    os.makedirs(outdir, exist_ok=True)

    print(f"[RUN] {vid_name} → transcribe")
    segments = transcribe_vi(infile)

    # nạp audio gốc (dual 48k) -> cut bằng hàm đã nội suy về 22.05k & mono
    audio_file = load_audio(infile)

    prev_txt_norm = ""
    kept, skipped = 0, 0

    for seg in segments:
        txt = seg["text"].strip()
        start, end = float(seg["start"]), float(seg["end"])
        keep, reason = should_keep_chunk(txt, start, end, prev_txt_norm)

        outfile = f"{base}_{seg['id']:04d}.wav"
        outpath = os.path.join(outdir, outfile)

        # luôn cắt audio (để ước lượng tổn thất dữ liệu sau) rồi decide lưu text
        cut_audio_timestamp_vits2(audio_file, outpath, start, end)  # -> mono 22.05 kHz

        if keep:
            text_file_handle.write(outfile + FIELD_SEP + txt + "\n")
            prev_txt_norm = txt.lower()
            kept += 1
        else:
            skipped += 1
            print(f"        skip {outfile:>20} ⇐ {reason}: {txt}")

    print(f"[DONE] {vid_name}: kept={kept}, skipped={skipped}")

def main():
    os.makedirs(SUBS_DATA_PATH, exist_ok=True)
    os.makedirs(AUDIO_TEXT_FILE_LIST_PATH, exist_ok=True)

    # Khởi tạo/ghi đè file tổng hợp
    transcription_file = os.path.join(AUDIO_TEXT_FILE_LIST_PATH, "_all.txt")
    with open(transcription_file, "w", encoding="utf-8") as f:
        f.write("")

    # Duyệt toàn bộ .wav sau VAD trong data/vad/
    files = sorted(
        fn for fn in os.listdir(VAD_DATA_PATH)
        if fn.lower().endswith(".wav")
    )
    if not files:
        print(f"[WARN] Không thấy file .wav nào trong {VAD_DATA_PATH}")
        return

    for fn in files:
        infile = os.path.join(VAD_DATA_PATH, fn)
        with open(transcription_file, "a", encoding="utf-8") as f:
            print(f"[INFO] Processing {fn}")
            process_one_file(infile, f)

    print(f"[OK] Ghi xong: {transcription_file}")
    print(f"     Segments đã lưu ở: {SUBS_DATA_PATH}")

if __name__ == "__main__":
    main()
