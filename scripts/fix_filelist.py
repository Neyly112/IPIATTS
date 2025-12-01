"""
Script sửa lỗi filelist - gộp các dòng text bị ngắt
"""

import argparse
import re


def fix_filelist(input_file, output_file):
    """
    Sửa lỗi text bị ngắt nhiều dòng
    Format đúng: audio.wav|text content
    """
    
    print(f"Đang sửa file: {input_file}")
    print("=" * 80)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    current_line = ""
    line_num = 0
    fixes = 0
    
    for line in lines:
        line = line.rstrip('\n')
        line_num += 1
        
        # Kiểm tra xem dòng này có bắt đầu bằng tên file audio không
        # Pattern: voice*.wav hoặc audio*.wav
        if re.match(r'^[a-zA-Z0-9_]+\.wav\|', line):
            # Đây là dòng mới bắt đầu bằng audio file
            if current_line:
                # Lưu dòng trước đó
                fixed_lines.append(current_line)
            current_line = line
        else:
            # Dòng này là phần tiếp theo của text bị ngắt
            if current_line:
                # Gộp vào dòng hiện tại
                current_line += " " + line
                fixes += 1
                print(f"  Fix dòng {line_num}: Gộp text bị ngắt")
            else:
                # Dòng trống hoặc lỗi - bỏ qua
                print(f"  ⚠ Bỏ qua dòng {line_num}: {line[:50]}...")
    
    # Thêm dòng cuối cùng
    if current_line:
        fixed_lines.append(current_line)
    
    # Ghi file output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines))
    
    print("=" * 80)
    print(f"✅ HOÀN TẤT!")
    print(f"   - Tổng dòng input: {len(lines)}")
    print(f"   - Tổng dòng output: {len(fixed_lines)}")
    print(f"   - Số lỗi đã sửa: {fixes}")
    print(f"   - File output: {output_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Sửa lỗi filelist bị ngắt dòng")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="File input cần sửa"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="File output sau khi sửa"
    )
    
    args = parser.parse_args()
    fix_filelist(args.input, args.output)


if __name__ == "__main__":
    main()
