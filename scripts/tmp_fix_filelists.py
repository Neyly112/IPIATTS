from pathlib import Path

root = Path(".")
master_file = root / "data/99-audio-text-file-list/_all_normal_ipa.txt"
target_files = [
    root / "data/99-audio-text-file-list/audio_text_train.txt.cleaned",
    root / "data/99-audio-text-file-list/audio_text_val.txt.cleaned",
]

master = {}
for line in master_file.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    parts = line.split("|")
    if len(parts) != 3:
        raise SystemExit(f"invalid master line: {line}")
    audio, text, phoneme = parts
    master[audio] = (text, phoneme)

for target_file in target_files:
    lines = []
    missing = []
    for line in target_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        audio = line.split("|", 1)[0]
        if audio not in master:
            missing.append(audio)
            continue
        text, phoneme = master[audio]
        lines.append(f"{audio}|{text}|{phoneme}")
    if missing:
        raise SystemExit(f"missing entries for {target_file.name}: {missing}")
    target_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"updated {target_file.name} with {len(lines)} entries")
