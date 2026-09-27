"""Lightweight Indic-script → Latin transliteration (stdlib only, zero external APIs).

Covers Devanagari (Hindi/Marathi/Sanskrit), Gujarati, and Tamil scripts commonly
present in regional business names, normalizing them into phonetic Latin equivalents.
"""
from __future__ import annotations

import re
import unicodedata

# Devanagari (U+0900–U+097F)
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

# Tamil (U+0B80–U+0BFF)
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
_WS_RE = re.compile(r"\s+")


def has_indic_script(text: object) -> bool:
    """Return True if text contains non-ASCII characters."""
    if text is None:
        return False
    s = str(text)
    return bool(_NON_ASCII_RE.search(s))


def transliterate_char(ch: str) -> str:
    """Map a single character to Latin phonetics."""
    for mapping in _SCRIPT_MAPS:
        if ch in mapping:
            return mapping[ch]
    decomposed = unicodedata.normalize("NFKD", ch)
    base = decomposed[0] if decomposed else ch
    if ord(base) < 128:
        return base.lower()
    return ""


def transliterate_to_latin(text: object) -> str:
    """Map Indic scripts to approximate Latin characters; leave ASCII unchanged."""
    if text is None:
        return ""
    s = str(text)
    if not s or s.lower() == "nan":
        return ""
    out = []
    for ch in s:
        if ord(ch) < 128:
            out.append(ch.lower())
        else:
            out.append(transliterate_char(ch))
    result = "".join(out)
    return _WS_RE.sub(" ", result).strip()
