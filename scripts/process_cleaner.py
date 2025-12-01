# -*- coding: utf-8 -*-
"""
Process cleaner - Chuẩn hóa và thêm IPA phonemes
"""

import os
import sys
from tqdm import tqdm
import pandas as pd

# Import cleaner function
from cleaner import basic_cleaners_phothong
from _constants import AUDIO_TEXT_FILE_LIST_PATH, FIELD_SEP

# Input/Output files
INPUT_FILE = os.path.join(AUDIO_TEXT_FILE_LIST_PATH, "_all_corrected.txt")
OUTPUT_FILE = os.path.join(AUDIO_TEXT_FILE_LIST_PATH, "_all_normal_ipa.txt")

def main():
    print("=" * 80)
    print("CHUẨN HÓA VÀ THÊM IPA PHONEMES")
    print("=" * 80)
    print()
    
    # Kiểm tra file input
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] File không tồn tại: {INPUT_FILE}")
        print()
        print("Hãy chạy các bước trước:")
        print("  1. python scripts\\transcribe_cut.py")
        print("  2. python scripts\\correct_spelling_mistakes.py")
        print("     (hoặc copy _all.txt → _all_corrected.txt)")
        sys.exit(1)
    
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print()
    
    # Load data
    print("Loading data...")
    try:
        data = pd.read_csv(
            INPUT_FILE,
            sep=FIELD_SEP,
            names=["audio", "text"],
            encoding="utf-8"
        )
    except Exception as e:
        print(f"[ERROR] Không thể đọc file: {e}")
        sys.exit(1)
    
    print(f"✓ Loaded {len(data)} rows")
    print()
    
    # Process
    print("Processing (chuẩn hóa + phonemization)...")
    print("⚠ Bước này có thể mất 5-30 phút tùy số lượng dữ liệu")
    print()
    
    ipa_list = []
    failed_count = 0
    
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Processing", ncols=80):
        try:
            ipa = basic_cleaners_phothong(row["text"])
            ipa_list.append(ipa)
        except Exception as e:
            print(f"\n[WARN] Failed at line {idx+1}: {row['text'][:50]}... → {e}")
            ipa_list.append("")  # Empty IPA for failed cases
            failed_count += 1
    
    data["ipa"] = ipa_list
    
    # Save
    print()
    print("Saving output...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    data.to_csv(
        OUTPUT_FILE,
        columns=["audio", "text", "ipa"],
        sep=FIELD_SEP,
        index=False,
        header=False,
        encoding="utf-8"
    )
    
    print()
    print("=" * 80)
    print("✅ HOÀN THÀNH!")
    print("=" * 80)
    print(f"Total rows:    {len(data)}")
    print(f"Success:       {len(data) - failed_count}")
    print(f"Failed:        {failed_count}")
    print()
    print(f"Output saved:  {OUTPUT_FILE}")
    print()
    print("Next step:")
    print("  python scripts\\split.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
