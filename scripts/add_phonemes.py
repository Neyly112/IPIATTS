"""
Script thêm phonemes vào filelist
Chuyển đổi: audio|text → audio|text|phonemes
"""

import os
import sys
import argparse
from tqdm import tqdm
from phonemizer.backend import EspeakBackend

# Setup eSpeak cho Windows
if os.name == "nt":
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(r"C:\Program Files\eSpeak NG\libespeak-ng.dll")
    except:
        print("⚠ Không tìm thấy eSpeak NG. Vui lòng cài đặt:")
        print("  https://github.com/espeak-ng/espeak-ng/releases")
        sys.exit(1)

def add_phonemes(input_file, output_file):
    """
    Đọc file audio|text
    Tạo file audio|text|phonemes
    """
    
    print(f"Đang xử lý: {input_file}")
    print("=" * 80)
    
    # Khởi tạo phonemizer
    try:
        backend = EspeakBackend(
            "vi",  # Vietnamese
            preserve_punctuation=False,
            language_switch="remove-flags",
            with_stress=False,
        )
        print("✓ eSpeak backend khởi tạo thành công")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo eSpeak: {e}")
        print("Vui lòng cài đặt eSpeak NG:")
        print("  https://github.com/espeak-ng/espeak-ng/releases")
        sys.exit(1)
    
    # Đọc file input
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"✓ Đọc {len(lines)} dòng từ file")
    print("✓ Đang phonemize...")
    
    # Process từng dòng
    output_lines = []
    errors = 0
    
    for i, line in enumerate(tqdm(lines), 1):
        parts = line.split('|')
        
        if len(parts) == 1:
            # Chỉ có audio file
            print(f"\n⚠ Dòng {i}: Thiếu text - {line}")
            errors += 1
            continue
        elif len(parts) == 2:
            # audio|text - cần thêm phonemes
            audio_path, text = parts
        elif len(parts) == 3:
            # audio|text|phonemes - đã có phonemes rồi
            output_lines.append(line)
            continue
        else:
            print(f"\n⚠ Dòng {i}: Format không hợp lệ - {line}")
            errors += 1
            continue
        
        # Phonemize text
        try:
            phonemes = backend.phonemize([text], strip=True)[0]
            # Chuyển từ IPA sang space-separated
            # Ví dụ: "siŋ tʃaːu" → "s i ŋ tʃ a ː u"
            phonemes_spaced = " ".join(list(phonemes.replace(" ", "  ")))
            
            output_line = f"{audio_path}|{text}|{phonemes_spaced}"
            output_lines.append(output_line)
        except Exception as e:
            print(f"\n❌ Dòng {i}: Lỗi phonemize - {e}")
            errors += 1
    
    # Ghi file output
    print(f"\n✓ Đang ghi file output: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print("=" * 80)
    print(f"✅ HOÀN TẤT!")
    print(f"   - Tổng dòng xử lý: {len(lines)}")
    print(f"   - Dòng thành công: {len(output_lines)}")
    print(f"   - Lỗi: {errors}")
    print(f"   - File output: {output_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Thêm phonemes vào filelist")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="File input (audio|text)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="File output (audio|text|phonemes)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ File không tồn tại: {args.input}")
        sys.exit(1)
    
    add_phonemes(args.input, args.output)


if __name__ == "__main__":
    main()
