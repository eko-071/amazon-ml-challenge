"""Text normalization + script-to-Latin transliteration (item 2.1).

Deterministic, rule-based script conversion — not an external lookup or
data augmentation (fair-play safe). Latin/LatinExt text passes through
untouched; each of the 8 Indic scripts in the data is romanized via its
own ``indic_transliteration`` scheme before the Latin-only
abbreviation regexes in ``normalize_text`` run.
"""

import re

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Unicode block -> transliteration source scheme, covering every non-Latin
# script the audit found in S2/S3 (train+test). First non-ASCII char wins;
# mixed-script strings keep their Latin spans intact (verified).
_SCHEMES_BY_BLOCK = (
    ((0x0900, 0x097F), sanscript.DEVANAGARI),
    ((0x0980, 0x09FF), sanscript.BENGALI),
    ((0x0B80, 0x0BFF), sanscript.TAMIL),
    ((0x0C00, 0x0C7F), sanscript.TELUGU),
    ((0x0C80, 0x0CFF), sanscript.KANNADA),
    ((0x0D00, 0x0D7F), sanscript.MALAYALAM),
    ((0x0A80, 0x0AFF), sanscript.GUJARATI),
    ((0x0A00, 0x0A7F), sanscript.GURMUKHI),
)

_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")


def _detect_scheme(text):
    for ch in text:
        o = ord(ch)
        if o < 0x80:
            continue
        for (lo, hi), scheme in _SCHEMES_BY_BLOCK:
            if lo <= o <= hi:
                return scheme
        return None  # non-ASCII, non-Indic (e.g. LatinExt): nothing to do
    return None


def transliterate_to_latin(text):
    if not isinstance(text, str) or not text:
        return ""
    if not _NON_ASCII_RE.search(text):
        return text  # fast path: 70-80% of rows are pure ASCII
    scheme = _detect_scheme(text)
    if scheme is None:
        return text
    try:
        return transliterate(text, scheme, sanscript.ITRANS)
    except Exception:
        return text


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'\b(corporation|corp)\b', 'corp', t)
    t = re.sub(r'\b(limited|ltd)\b', 'ltd', t)
    t = re.sub(r'\b(private|pvt)\b', 'pvt', t)
    t = re.sub(r'\b(company|co)\b', 'co', t)
    t = re.sub(r'\b(road|rd)\b', 'rd', t)
    t = re.sub(r'\b(street|st)\b', 'st', t)
    return re.sub(r'\s+', ' ', t).strip()


def full_normalize(text):
    return normalize_text(transliterate_to_latin(text))
