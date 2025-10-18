import os
import re
import pandas as pd
from underthesea import text_normalize
from phonemizer.backend import EspeakBackend
from num2words import num2words  

from _constants import AUDIO_TEXT_FILE_LIST_PATH, FIELD_SEP

if os.name == "nt" and not EspeakBackend.is_available():
	from phonemizer.backend.espeak.wrapper import EspeakWrapper
	EspeakWrapper.set_library(r"C:\Program Files\eSpeak NG\libespeak-ng.dll")
ESPEAK = EspeakBackend("vi", preserve_punctuation=True, language_switch="remove-flags", with_stress=True, tie=True)

TRANSCRIPTION_FILE = os.path.join(AUDIO_TEXT_FILE_LIST_PATH, "_all_corrected.txt")
RAW_DATA = pd.read_csv(TRANSCRIPTION_FILE, sep=FIELD_SEP, names=["audio", "text"])


NAM_20xx = re.compile(r"^20\d{2}$")
CHU_SO = re.compile(r"^\d+\.?\d+$")  
def special_normalize(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt ở mức cơ bản, không thiên về vùng miền.
    Dùng cho giọng đọc trung lập (phổ thông).
    """
    txt = text_normalize(text).replace("-", " ")
    res = []
    for word in txt.split():
        if word == "%":
            res.append("phần trăm")
        elif NAM_20xx.match(word) is not None:
            # 2025 -> hai nghìn không trăm hai mươi lăm
            num = "hai nghìn không trăm " + num2words(int(word[-2:]), lang="vi")
            res.append(num)
        elif CHU_SO.match(word.strip('.,;!?')) is not None:
            clean_num = word.strip('.,;!?')
            num = num2words(float(clean_num), lang="vi")
            res.append(num)

        elif word in ".,;!?":
            if res:
                res[-1] += word
        else:
            res.append(word)
    return " ".join(res).strip()


RAW_DATA["text"] = RAW_DATA["text"].map(special_normalize)
RAW_DATA["ipa"] = ESPEAK.phonemize(RAW_DATA["text"], strip=True)  

RAW_DATA["audio"] = RAW_DATA["audio"].radd("..\\data\\data_matchaTTS\\")

SAVE_FILE = os.path.join(AUDIO_TEXT_FILE_LIST_PATH, "_all_normal_ipa.txt")
RAW_DATA.to_csv(SAVE_FILE, sep=FIELD_SEP, index=False, header=False)