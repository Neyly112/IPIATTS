import os.path

RAW_DATA_PATH      = os.path.join("data", "raw")
VAD_DATA_PATH      = os.path.join("data", "vad")
MERGED_DATA_PATH   = os.path.join("data", "merged")
SUBS_DATA_PATH     = os.path.join("data", "subs")
SEGMENTS_DIR       = os.path.join(VAD_DATA_PATH, "segments")
VAD_DATA_PATH_TEST = os.path.join("data", "vad1")
AUDIO_TEXT_FILE_LIST_PATH = os.path.join("data", "99-audio-text-file-list")
FIELD_SEP          = "|"

# Tự động lấy danh sách file audio trong RAW_DATA_PATH
LIST_VID = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith((".wav", ".mp3"))]
LIST_VID_VAD = [f for f in os.listdir(VAD_DATA_PATH) if f.endswith((".wav", ".mp3"))]
