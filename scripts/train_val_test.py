"""split 3 sets: train-val-test keeping IPA in the same file"""

import os
import pandas as pd
from _constants import AUDIO_TEXT_FILE_LIST_PATH, FIELD_SEP

# 1. Đọc dữ liệu
TRANSCRIPTION_FILE = os.path.join(AUDIO_TEXT_FILE_LIST_PATH, "_all_normal_ipa.txt")
# Đảm bảo đọc đủ 3 cột
DATA = pd.read_csv(TRANSCRIPTION_FILE, sep=FIELD_SEP, names=["audio", "text", "ipa"])

# 2. Trộn dữ liệu (Shuffle)
RANDOM_STATE = 42
# Chỉ cần sample 1 lần là đủ để xáo trộn toàn bộ dataset
DATA = DATA.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# 3. Định nghĩa file đầu ra
TRAIN_SET_FILE = os.path.join(AUDIO_TEXT_FILE_LIST_PATH, "audio_text_train_filelist.txt")
VAL_SET_FILE   = os.path.join(AUDIO_TEXT_FILE_LIST_PATH, "audio_text_val_filelist.txt")
TEST_SET_FILE  = os.path.join(AUDIO_TEXT_FILE_LIST_PATH, "audio_text_test_filelist.txt")

# 4. Tính toán tỷ lệ chia
N = len(DATA) 
TRAIN_SIZE = .85  # 85%
VAL_SIZE   = .1   # 10%
TEST_SIZE  = .05  # 5%
assert abs((TRAIN_SIZE + VAL_SIZE + TEST_SIZE) - 1.0) < 1e-9, "Tổng tỷ lệ phải bằng 1."

idx_train = int(N * TRAIN_SIZE)
idx_val   = int(N * (TRAIN_SIZE + VAL_SIZE))

# 5. Hàm lưu file (Đã sửa)
def save_csv(df: pd.DataFrame, filename: str) -> None:
    """
    Lưu file bao gồm cả cột IPA, không tách ra file riêng.
    Format: audio|text|ipa
    """
    print(f"Đang lưu {filename} với {len(df)} dòng...")
    # Lưu cả 3 cột vào file chính
    df.to_csv(
        filename, 
        columns=["audio", "text", "ipa"], 
        sep=FIELD_SEP, 
        index=False, 
        header=False
    )

# 6. Thực hiện chia và lưu
save_csv(DATA.iloc[:idx_train, :],      TRAIN_SET_FILE)
save_csv(DATA.iloc[idx_train:idx_val, :], VAL_SET_FILE)
save_csv(DATA.iloc[idx_val:, :],         TEST_SET_FILE)

print("✅ Đã chia tập dữ liệu xong! (Bao gồm IPA trong cùng 1 file)")