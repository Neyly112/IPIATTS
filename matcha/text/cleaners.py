# -*- coding: utf-8 -*-
"""
Vietnamese cleaner - giọng phổ thông (chuẩn miền Bắc)
Dùng để thay thế matcha/text/cleaners.py trong Matcha-TTS
"""

import os
import platform
import re
from pathlib import Path
from underthesea import text_normalize
from phonemizer.backend import EspeakBackend
from num2words import num2words

# --- eSpeak setup (attempt best-effort detection on Windows) ---


def _load_espeak_backend():
    """Try to initialise eSpeak; return backend or None."""
    candidates = []

    env_base = os.environ.get("ESPEAKNGPATH")
    if env_base:
        candidates.append(Path(env_base))

    candidates.extend([
        Path(r"C:\Program Files\eSpeak NG"),
        Path(r"C:\Program Files (x86)\eSpeak NG"),
        Path(r"C:\Program Files\eSpeak"),
        Path(r"C:\Program Files (x86)\eSpeak"),
    ])

    dll_names = [
        "libespeak-ng.dll",
        "bin/libespeak-ng.dll",
        "espeak-ng.dll",
        "bin/espeak-ng.dll",
        # Legacy/older names
        "espeak.dll",
        "bin/espeak.dll",
        "libespeak.dll",
        "bin/libespeak.dll",
    ]

    for base in candidates:
        for rel in dll_names:
            dll_path = (base / rel).resolve()
            if dll_path.exists():
                try:
                    from phonemizer.backend.espeak.wrapper import EspeakWrapper

                    EspeakWrapper.set_library(str(dll_path))
                    backend = EspeakBackend(
                        "vi",
                        preserve_punctuation=True,
                        language_switch="remove-flags",
                        with_stress=True,
                        tie=True,
                    )
                    return backend
                except Exception:
                    continue

    # Last attempt: default discovery
    try:
        return EspeakBackend(
            "vi",
            preserve_punctuation=True,
            language_switch="remove-flags",
            with_stress=True,
            tie=True,
        )
    except Exception as e:
        print(f"[WARNING] eSpeak not available: {e}")
        print("[WARNING] Tried paths: " + ", ".join(str(p)
              for p in candidates))
        return None


_ESPEAK = _load_espeak_backend()

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

    # If espeak is available, phonemize; otherwise return as-is
    if _ESPEAK is not None:
        ipa_list = _ESPEAK.phonemize(tokens, strip=True)
        return " ".join(ipa_list).strip()
    else:
        # Fallback: return the tokens as-is (less reliable, may affect symbols mapping)
        return " ".join(tokens).strip()
