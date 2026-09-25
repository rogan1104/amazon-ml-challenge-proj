"""Lightweight Indic-script → Latin transliteration (stdlib only, no external APIs)."""
from __future__ import annotations

import re
import unicodedata

from analysis.normalize import WS_RE, _safe_str, basic_normalize, normalize_name

# Devanagari (U+0900–U+097F) — covers Hindi/Marathi/Sanskrit business names in train.
_DEVANAGARI = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "ळ": "l", "क्ष": "ksh", "त्र": "tr", "ज्ञ": "gy",
    "ा": "a", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    "ं": "n", "ः": "h", "्": "",
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}

# Gujarati (U+0A80–U+0AFF)
_GUJARATI = {
    "અ": "a", "આ": "aa", "ઇ": "i", "ઈ": "ee", "ઉ": "u", "ઊ": "oo",
    "એ": "e", "ઐ": "ai", "ઓ": "o", "ઔ": "au",
    "ક": "k", "ખ": "kh", "ગ": "g", "ઘ": "gh", "ઙ": "ng",
    "ચ": "ch", "છ": "chh", "જ": "j", "ઝ": "jh", "ઞ": "ny",
    "ટ": "t", "ઠ": "th", "ડ": "d", "ઢ": "dh", "ણ": "n",
    "ત": "t", "થ": "th", "દ": "d", "ધ": "dh", "ન": "n",
    "પ": "p", "ફ": "ph", "બ": "b", "ભ": "bh", "મ": "m",
    "ય": "y", "ર": "r", "લ": "l", "વ": "v", "શ": "sh", "સ": "s", "હ": "h",
    "ા": "a", "િ": "i", "ી": "ee", "ુ": "u", "ૂ": "oo",
    "ે": "e", "ૈ": "ai", "ો": "o", "ૌ": "au", "ં": "n", "ઃ": "h", "્": "",
}

# Tamil (U+0B80–U+0BFF) — subset for common business-name syllables
_TAMIL = {
    "அ": "a", "ஆ": "aa", "இ": "i", "ஈ": "ee", "உ": "u", "ஊ": "oo",
    "எ": "e", "ஏ": "ee", "ஐ": "ai", "ஒ": "o", "ஓ": "oo", "ஔ": "au",
    "க": "k", "ங": "ng", "ச": "s", "ஜ": "j", "ஞ": "ny",
    "ட": "t", "ண": "n", "த": "th", "ந": "n", "ப": "p", "ம": "m",
    "ய": "y", "ர": "r", "ல": "l", "வ": "v", "ழ": "zh", "ள": "l", "ற": "r", "ன": "n",
    "ா": "aa", "ி": "i", "ீ": "ee", "ு": "u", "ூ": "oo",
    "ெ": "e", "ே": "ee", "ை": "ai", "ொ": "o", "ோ": "oo", "ௌ": "au",
    "ம்": "m", "ன்": "n", "ர்": "r", "ல்": "l", "ள்": "l", "ட்": "t", "க்": "k",
}

_SCRIPT_MAPS = [_DEVANAGARI, _GUJARATI, _TAMIL]
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")


def has_indic_script(text: str) -> bool:
    text = _safe_str(text)
    return bool(_NON_ASCII_RE.search(text))


def transliterate_char(ch: str) -> str:
    for mapping in _SCRIPT_MAPS:
        if ch in mapping:
            return mapping[ch]
    decomposed = unicodedata.normalize("NFKD", ch)
    base = decomposed[0] if decomposed else ch
    if ord(base) < 128:
        return base.lower()
    return ""


def transliterate_to_latin(text: str) -> str:
    """Map Indic scripts to approximate Latin; leave ASCII as-is."""
    text = _safe_str(text)
    if not text:
        return ""
    out = []
    for ch in text:
        if ord(ch) < 128:
            out.append(ch.lower())
        else:
            out.append(transliterate_char(ch))
    result = "".join(out)
    result = WS_RE.sub(" ", result).strip()
    return result


def transliterated_name_key(name: str) -> str:
    """Normalized retrieval key from transliterated business name."""
    latin = transliterate_to_latin(name)
    if not latin:
        return ""
    return normalize_name(latin)


def transliterated_ngrams(name: str, n: int = 3) -> set[str]:
    from analysis.normalize import char_ngrams

    latin = transliterate_to_latin(name)
    return char_ngrams(latin, n=n)
