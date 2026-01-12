# -*- coding: utf-8 -*-
# Vietnamese IPA symbols for Matcha-TTS (matching checkpoint vocab=331)
# AUTO-GENERATED to match trained checkpoint

_pad = '_'

# Punctuation and special chars
_punctuation = '!\'(),-.:;? '

# IPA symbols for Vietnamese (eSpeak-ng phonemes)
_ipa_symbols = [
    # Vowels
    'a', 'aː', 'ă', 'ɐ', 'ɑ', 'ɑː', 'ə', 'əː', 'ɤ', 'ɤː', 'ɔ', 'ɔː',
    'e', 'eː', 'ɛ', 'ɛː', 'i', 'iː', 'o', 'oː', 'u', 'uː', 'ɯ', 'ɯː', 'ɨ', 'ɨː',
    # Diphthongs
    'aj', 'aw', 'ɐw', 'iə', 'uə', 'ɯə', 'iə̯', 'uə̯', 'ɯə̯',
    # Consonants
    'b', 'd', 'ɗ', 'f', 'ɡ', 'ɣ', 'h', 'j', 'k', 'kʰ', 'l', 'm', 'n', 'ŋ', 'ɲ',
    'p', 'pʰ', 'r', 's', 'ʂ', 't', 'tʰ', 'v', 'w', 'x', 'z', 'ʒ', 'ʐ',
    'c', 'ɟ', 'tɕ', 'tɕʰ', 'ɕ',
    # Tones
    '˥', '˦', '˧', '˨', '˩', '˥˧', '˧˩', '˨˩˦', '˧˥', '˩˨', 'ˀ',
    # Stress/markers
    'ˈ', 'ˌ', '.', 'ː',
    # Combining/modifier chars (CRITICAL - used by eSpeak with_stress=True, tie=True)
    '͡',  # U+0361 combining double inverted breve (tie bar)
    '̯',  # U+032F combining inverted breve below (non-syllabic)
    '̩',  # U+0329 combining vertical line below (syllabic)
    '̪',  # U+032A combining bridge below (dental)
    '̃',  # U+0303 combining tilde (nasalized)
    '̥',  # U+0325 combining ring below (voiceless)
    '̬',  # U+032C combining caron below (voiced)
    # Extra IPA
    'ɓ', 'ʔ', 'θ', 'ð', 'ʃ', 'ʧ', 'ʤ', 'æ', 'ʌ', 'ɜ', 'ɪ', 'ʊ', 'y', 'ø',
]

# Build symbol list
symbols = [_pad] + list(_punctuation) + _ipa_symbols

# Pad to exactly 331
while len(symbols) < 331:
    symbols.append(f'<pad{len(symbols)}>')
symbols = symbols[:331]

# Verify
assert len(symbols) == 331, f"Expected 331, got {len(symbols)}"

SPACE_ID = symbols.index(' ')
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}
