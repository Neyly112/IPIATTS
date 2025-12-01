import os
import os.path

# Get project root directory (parent of scripts folder)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# Define paths relative to project root
RAW_DATA_PATH      = os.path.join(_PROJECT_ROOT, "data", "raw")
VAD_DATA_PATH      = os.path.join(_PROJECT_ROOT, "data", "vad")
VAD_DATA_PATH_ADD    = os.path.join(_PROJECT_ROOT, "data", "vad_add")
MERGED_DATA_PATH   = os.path.join(_PROJECT_ROOT, "data", "merged")
SUBS_DATA_PATH     = os.path.join(_PROJECT_ROOT, "data", "subs")
SUBS_DATA_PATH_ADD = os.path.join(_PROJECT_ROOT, "data", "subs_add")
SUBS_DATA_PATH_ADD_CON = os.path.join(_PROJECT_ROOT, "data", "subs_add_con")

SEGMENTS_DIR       = os.path.join(VAD_DATA_PATH, "segments")
VAD_DATA_PATH_TEST = os.path.join(_PROJECT_ROOT, "data", "vad1")
AUDIO_TEXT_FILE_LIST_PATH = os.path.join(_PROJECT_ROOT, "data", "99-audio-text-file-list")
FIELD_SEP          = "|"

# Tự động lấy danh sách file audio trong RAW_DATA_PATH
LIST_VID = []
LIST_VID_VAD = []

if os.path.exists(RAW_DATA_PATH):
    LIST_VID = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith((".wav", ".mp3"))]

if os.path.exists(VAD_DATA_PATH):
    LIST_VID_VAD = [f for f in os.listdir(VAD_DATA_PATH) if f.endswith((".wav", ".mp3"))]
