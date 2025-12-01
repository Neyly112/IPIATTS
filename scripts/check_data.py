"""
Script kiểm tra filelist và dữ liệu
"""

import argparse
import sys
from pathlib import Path


def check_filelist(filelist_path):
    """Kiểm tra format và tính hợp lệ của filelist"""
    
    print(f"Đang kiểm tra: {filelist_path}")
    print("=" * 80)
    
    if not Path(filelist_path).exists():
        print(f"❌ File không tồn tại: {filelist_path}")
        return False
    
    errors = []
    warnings = []
    total_lines = 0
    missing_audio = []
    
    with open(filelist_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            line = line.strip()
            
            if not line:
                warnings.append(f"Dòng {line_num}: Dòng trống")
                continue
            
            parts = line.split('|')
            
            if len(parts) != 3:
                errors.append(f"Dòng {line_num}: Không đúng format (cần 3 phần phân cách bởi |)")
                continue
            
            audio_path, text, phonemes = parts
            
            # Kiểm tra audio file
            if not Path(audio_path).exists():
                missing_audio.append(f"Dòng {line_num}: {audio_path}")
            
            # Kiểm tra text
            if not text.strip():
                errors.append(f"Dòng {line_num}: Text rỗng")
            
            # Kiểm tra phonemes
            if not phonemes.strip():
                errors.append(f"Dòng {line_num}: Phonemes rỗng")
            
            phoneme_list = phonemes.split()
            if len(phoneme_list) == 0:
                errors.append(f"Dòng {line_num}: Phonemes không được phân tách bằng khoảng trắng")
    
    # In kết quả
    print(f"✓ Tổng số dòng: {total_lines}")
    print(f"✓ Dòng hợp lệ: {total_lines - len(errors)}")
    
    if warnings:
        print(f"\n⚠ Cảnh báo ({len(warnings)}):")
        for w in warnings[:5]:  # Chỉ hiện 5 cái đầu
            print(f"  {w}")
        if len(warnings) > 5:
            print(f"  ... và {len(warnings) - 5} cảnh báo khác")
    
    if missing_audio:
        print(f"\n❌ File audio không tồn tại ({len(missing_audio)}):")
        for m in missing_audio[:5]:
            print(f"  {m}")
        if len(missing_audio) > 5:
            print(f"  ... và {len(missing_audio) - 5} file khác")
    
    if errors:
        print(f"\n❌ Lỗi format ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... và {len(errors) - 10} lỗi khác")
        return False
    
    print("\n" + "=" * 80)
    if missing_audio:
        print("⚠ CÓ FILE AUDIO KHÔNG TỒN TẠI - Cần kiểm tra lại đường dẫn")
        return False
    else:
        print("✅ FILELIST HỢP LỆ - Sẵn sàng để training!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra filelist")
    parser.add_argument(
        "--filelist",
        type=str,
        required=True,
        help="Đường dẫn đến file filelist cần kiểm tra"
    )
    
    args = parser.parse_args()
    
    success = check_filelist(args.filelist)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
