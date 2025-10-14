"""split 3 sets: train-val-test"""

import math
import pandas as pd
from pathlib import Path

from _constants import AUDIO_TEXT_FILE_LIST_PATH, FIELD_SEP

# ================== PATHS ==================
BASE = Path(AUDIO_TEXT_FILE_LIST_PATH)  # vd: data/99-audio-text-file-list
TRANSCRIPTION_FILE = BASE / "_all_normal_ipa.txt"

print("Reading from:", TRANSCRIPTION_FILE.resolve())
if not TRANSCRIPTION_FILE.exists():
    raise FileNotFoundError(f"Không tìm thấy file: {TRANSCRIPTION_FILE.resolve()}")

# ================== LOAD ==================
DATA = pd.read_csv(
    TRANSCRIPTION_FILE,
    sep=FIELD_SEP,
    names=["audio", "text", "ipa"],
    encoding="utf-8"
)

# ================== CONFIG ==================
RANDOM_STATE = 42
TRAIN_SIZE = 0.85
VAL_SIZE   = 0.10
TEST_SIZE  = 0.05

if not math.isclose(TRAIN_SIZE + VAL_SIZE + TEST_SIZE, 1.0, rel_tol=1e-9, abs_tol=1e-9):
    raise ValueError("Train/Val/Test phải cộng = 1.")

# Shuffle 1 lần (tái lập được)
DATA = DATA.sample(frac=1.0, random_state=RANDOM_STATE, ignore_index=True)

# ================== SPLIT ==================
N = len(DATA)
n_train = int(round(N * TRAIN_SIZE))
n_val   = int(round(N * VAL_SIZE))
n_test  = N - n_train - n_val

# Giữ cho mỗi tập có ít nhất 1 mẫu nếu N đủ lớn
if N >= 3:
    if n_train == 0: n_train = 1
    if n_val == 0:   n_val   = 1
    if n_train + n_val >= N:
        n_val = max(1, N - n_train - 1)
    n_test = N - n_train - n_val

i0, i1, i2 = 0, n_train, n_train + n_val
train_df = DATA.iloc[i0:i1, :]
val_df   = DATA.iloc[i1:i2, :]
test_df  = DATA.iloc[i2:,   :]

# ================== OUTPUT ==================
BASE.mkdir(parents=True, exist_ok=True)
TRAIN_SET_FILE = BASE / "audio_text_train.txt"
VAL_SET_FILE   = BASE / "audio_text_val.txt"
TEST_SET_FILE  = BASE / "audio_text_test.txt"  # sửa typo tên file

def save_csv(df: pd.DataFrame, filename: Path) -> None:
    # Bản text (audio|text)
    df.to_csv(
        filename,
        columns=["audio", "text"],
        sep=FIELD_SEP,
        index=False,
        header=False,
        encoding="utf-8"
    )
    # Bản ipa (audio|ipa)
    df.to_csv(
        str(filename) + ".cleaned",
        columns=["audio", "ipa"],
        sep=FIELD_SEP,
        index=False,
        header=False,
        encoding="utf-8"
    )

save_csv(train_df, TRAIN_SET_FILE)
save_csv(val_df,   VAL_SET_FILE)
save_csv(test_df,  TEST_SET_FILE)

print(f"Done. N={N} → train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
print("Wrote to:")
print("  ", TRAIN_SET_FILE.resolve())
print("  ", VAL_SET_FILE.resolve())
print("  ", TEST_SET_FILE.resolve())
