# -*- coding: utf-8 -*-
"""
Vietnamese cleaner - giọng phổ thông (chuẩn miền Bắc)
Dùng để thay thế matcha/text/cleaners.py trong Matcha-TTS
"""

import platform
import re
from underthesea import text_normalize
from phonemizer.backend import EspeakBackend
from num2words import num2words

# --- eSpeak setup (Linux: Kaggle, Colab, Ubuntu) ---
if platform.system().lower() == "windows" and not EspeakBackend.is_available():
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.set_library(r"C:\Program Files\eSpeak NG\libespeak-ng.dll")

# Espeak: Vietnamese 
_ESPEAK = EspeakBackend(
    "vi",
    preserve_punctuation=True,
    language_switch="remove-flags",
    with_stress=True,
    tie=True
)

_RE_YEAR_20xx = re.compile(r"^20\d{2}$")
_RE_NUMBER = re.compile(r"^\d+$")


def basic_cleaners_phothong(text: str) -> str:
    """
    Chuẩn hoá và phiên âm tiếng Việt phổ thông.
    Dành cho huấn luyện Matcha-TTS tiếng Việt.
    """

    txt = text_normalize(text).replace("-", " ")

    tokens = []
    for word in txt.split():
        if word == "%":
            tokens.extend(["phần", "trăm"])
        elif word == "&":
            tokens.append("và")
        elif _RE_YEAR_20xx.match(word):   
            num = num2words(int(word), lang="vi").replace("ngàn", "nghìn")
            tokens.extend(num.split())
        elif _RE_NUMBER.match(word):  
            num = num2words(int(word), lang="vi").replace("ngàn", "nghìn")
            tokens.extend(num.split())
        elif word in ".,;!?":
            if tokens:
                tokens[-1] += word
        else:
            tokens.append(word)


    ipa_list = _ESPEAK.phonemize(tokens, strip=True)
    return " ".join(ipa_list).strip()
