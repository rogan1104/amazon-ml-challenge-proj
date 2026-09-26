"""Unicode-safe, language-agnostic text normalization for business records."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

LEGAL_SUFFIX_MAP = {
    "corp": "corporation",
    "inc": "incorporated",
    "pvt": "private",
    "ltd": "limited",
}
_DROP_TOKENS = frozenset({"www", "http", "https", "com", "net", "org"})
_NON_ALNUM = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_UNDERSCORE = re.compile(r"_+")
_WHITESPACE = re.compile(r"\s+")
_LEGAL_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(LEGAL_SUFFIX_MAP, key=len, reverse=True)) + r")\b",
    flags=re.UNICODE,
)


def _expand_legal_token(match: re.Match) -> str:
    return LEGAL_SUFFIX_MAP[match.group(0)]


def normalize_text(value: object) -> str:
    """Lowercase, strip punctuation, expand legal abbreviations, and trim."""
    if value is None:
        return ""
    text = str(value)
    if not text or text.lower() == "nan":
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("&", " and ").replace("+", " and ")
    text = _NON_ALNUM.sub(" ", text)
    text = _UNDERSCORE.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    if not text:
        return ""
    text = _LEGAL_PATTERN.sub(_expand_legal_token, text)
    return " ".join(tok for tok in text.split(" ") if tok and tok not in _DROP_TOKENS)


def tokenize(value: object) -> List[str]:
    """Normalize and split on whitespace."""
    text = normalize_text(value)
    return text.split() if text else []


def record_text(business_name: object, business_address: object) -> str:
    """Return the shared normalized name/address document."""
    name = normalize_text(business_name)
    address = normalize_text(business_address)
    if name and address:
        return f"{name} {address}"
    return name or address


def normalize_series(values: Iterable[object]) -> List[str]:
    """Normalize an iterable of values."""
    return [normalize_text(v) for v in values]
