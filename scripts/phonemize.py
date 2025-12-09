import os
import re
import pandas as pd
from underthesea import text_normalize
from phonemizer.backend import EspeakBackend
from num2words import num2words 
import tqdm

from _constants import AUDIO_TEXT_FILE_LIST_PATH, FIELD_SEP

# 1. Cấu hình ESpeak
if os.name == "nt" and not EspeakBackend.is_available():
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    # Đảm bảo đường dẫn này đúng trên máy bạn
    EspeakWrapper.set_library(r"C:\Program Files\eSpeak NG\libespeak-ng.dll")

# Khởi tạo backend (Giữ nguyên cấu hình tốt của bạn)
ESPEAK = EspeakBackend("vi", preserve_punctuation=True, language_switch="remove-flags", with_stress=True, tie=True)

# 2. Đọc dữ liệu
TRANSCRIPTION_FILE = os.path.join(AUDIO_TEXT_FILE_LIST_PATH, "_all_corrected.txt")
# Lưu ý: Thêm header=None nếu file không có tiêu đề, hoặc names như bạn làm là OK
RAW_DATA = pd.read_csv(TRANSCRIPTION_FILE, sep=FIELD_SEP, names=["audio", "text"])

# 3. Regex xử lý số
NAM_20xx = re.compile(r"^20\d{2}$")
CHU_SO = re.compile(r"^\d+\.?\d+$")  

def special_normalize(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt
    """
    if not isinstance(text, str): return "" # Phòng hổi text bị NaN
    
    txt = text_normalize(text).replace("-", " ")
    res = []
    for word in txt.split():
        if word == "%":
            res.append("phần trăm")
        elif NAM_20xx.match(word) is not None:
            # 2025 -> hai nghìn không trăm hai mươi lăm
            try:
                num = "hai nghìn không trăm " + num2words(int(word[-2:]), lang="vi")
                res.append(num)
            except: res.append(word)
        elif CHU_SO.match(word.strip('.,;!?')) is not None:
            clean_num = word.strip('.,;!?')
            try:
                num = num2words(float(clean_num), lang="vi")
                res.append(num)
            except: res.append(word)
        elif word in ".,;!?":
            if res:
                res[-1] += word
        else:
            res.append(word)
    return " ".join(res).strip()

def get_spaced_ipa(text):
    """
    Hàm chuyển Text -> IPA có khoảng cách
    """
    if not text or str(text) == 'nan': return ""
    
    try:
        # --- SỬA LỖI TẠI ĐÂY ---
        # 1. Đưa 'text' vào trong ngoặc vuông [] để thành List
        # 2. phonemize trả về List, nên phải lấy phần tử đầu tiên [0]
        ipa_list = ESPEAK.phonemize([text], strip=True)
        
        if not ipa_list: return ""
        
        raw_ipa = ipa_list[0]
        # -----------------------

        # Tách từng ký tự ra bằng khoảng trắng
        # list(raw_ipa) -> ['s', 'i', 'n', ...]
        spaced_ipa = " ".join(list(raw_ipa))
        
        # Xử lý dọn dẹp khoảng trắng kép thừa (nếu có)
        spaced_ipa = re.sub(r'\s+', ' ', spaced_ipa)
        
        return spaced_ipa
    except Exception as e:
        print(f"Lỗi IPA dòng: {text} | {e}")
        return ""

# --- MAIN PROCESS ---

print("Đang chuẩn hóa Text...")
RAW_DATA["text"] = RAW_DATA["text"].map(special_normalize)

print("Đang tạo IPA (có space)...")
# Dùng progress_apply nếu có tqdm, không thì dùng map/apply thường
tqdm.tqdm.pandas()
RAW_DATA["ipa"] = RAW_DATA["text"].progress_apply(get_spaced_ipa)

# Sửa đường dẫn Audio
RAW_DATA["audio"] = RAW_DATA["audio"].apply(lambda x: f"..\\data\\data_matchaTTS\\{x}")

# Lưu file
SAVE_FILE = os.path.join(AUDIO_TEXT_FILE_LIST_PATH, "_all_normal_ipa.txt")
# Lưu 3 cột: audio|text|ipa
RAW_DATA.to_csv(SAVE_FILE, sep=FIELD_SEP, index=False, header=False)

print(f"✅ Đã xong! File lưu tại: {SAVE_FILE}")
print("Ví dụ 5 dòng đầu:")
print(RAW_DATA.head())