"""Blocking-key extraction for retrieval channels (reuses analysis.normalize)."""
from __future__ import annotations

import re

from analysis.normalize import (
    DIGIT_RE,
    POSTAL_RE,
    WS_RE,
    _safe_str,
    basic_normalize,
    char_ngrams,
    extract_digits,
    extract_postal,
    normalize_address,
    normalize_name,
)
from analysis.retrieval.transliterate import (
    has_indic_script,
    transliterated_name_key,
    transliterated_ngrams,
)

HOUSE_NUM_RE = re.compile(r"^\s*(\d+[a-zA-Z]?)\b")


def sql_style_norm_name(name: str) -> str:
    """Match baseline E name normalization (ASCII punctuation strip, keep &)."""
    name = _safe_str(name).lower().strip()
    name = re.sub(r"[^\w\s&]", " ", name)
    return WS_RE.sub(" ", name).strip()


def sql_style_norm_addr(addr: str) -> str:
    addr = _safe_str(addr).lower().strip()
    addr = re.sub(r"[^\w\s]", " ", addr)
    return WS_RE.sub(" ", addr).strip()


def name_prefix_key(name: str, n: int = 5) -> str:
    base = re.sub(r"[^\w\s]", " ", _safe_str(name).lower())
    base = WS_RE.sub(" ", base).strip()
    return base[:n]


def baseline_keys(name: str, addr: str, country: str, prefix_len: int = 5, min_prefix: int = 3) -> list[str]:
    """Keys for reference baseline E channel."""
    keys: list[str] = []
    nn = sql_style_norm_name(name)
    na = sql_style_norm_addr(addr)
    c = _safe_str(country).strip()
    if nn:
        keys.append(f"bn:{nn}")
    if na:
        keys.append(f"ba:{na}")
    pfx = name_prefix_key(name, prefix_len)
    if len(pfx) >= min_prefix and c:
        keys.append(f"cp:{c}|{pfx}")
    return keys


def char_ngram_keys(name: str, n: int = 3) -> set[str]:
    nn = normalize_name(name).replace(" ", "")
    grams = char_ngrams(nn, n=n) if nn else set()
    return {f"ng:{g}" for g in grams if len(g) == n}


def address_numeric_keys(addr: str, country: str, include_postal: bool = True, include_house: bool = True,
                         min_digit_len: int = 3) -> set[str]:
    addr = _safe_str(addr)
    c = _safe_str(country).strip()
    keys: set[str] = set()

    if include_postal:
        for p in extract_postal(addr):
            if c:
                keys.add(f"postal:{c}|{p}")
            keys.add(f"postal:{p}")

    if include_house:
        m = HOUSE_NUM_RE.search(addr)
        if m:
            hn = m.group(1).lower()
            if c:
                keys.add(f"house:{c}|{hn}")
            keys.add(f"house:{hn}")

    for d in extract_digits(addr):
        if len(d) >= min_digit_len:
            if c:
                keys.add(f"num:{c}|{d}")
            keys.add(f"num:{d}")

    return keys


def transliteration_keys(name: str, n: int = 3) -> set[str]:
    keys: set[str] = set()
    tkey = transliterated_name_key(name)
    if len(tkey) >= 4:
        keys.add(f"tl:{tkey}")
    if has_indic_script(name):
        for g in transliterated_ngrams(name, n=n):
            if g:
                keys.add(f"tng:{g}")
    return keys
