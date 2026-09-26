"""Text normalization for business names and addresses.

The same function is applied to both fields. It is language- and country-agnostic:
legal-suffix expansion only touches Latin abbreviation tokens; other scripts
(Hindi, Tamil, French accents, etc.) pass through after lowercasing.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

from transliterate import has_indic_script, transliterate_to_latin

# Whole-token legal / entity-type abbreviations. Applied after punctuation is
# stripped, so "Ltd." and "LTD" both become the token "ltd" first.
# Longer keys are expanded first so "pvt ltd" is handled token-by-token.
LEGAL_SUFFIX_MAP = {
    "corp": "corporation",
    "inc": "incorporated",
    "pvt": "private",
    "ltd": "limited",
}

# Tokens that are almost always website leftovers in this dataset, not identity.
_DROP_TOKENS = frozenset({"www", "http", "https", "com", "net", "org"})

# Keep letters from any script, marks (accents), digits, and spaces.
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
    """Lowercase, strip punctuation, expand legal suffixes, collapse whitespace.

    ``&`` (and ``+``, which is used the same way in this data) become ``and``
    *before* punctuation is removed. Returns a single space-joined string.
    Empty / missing inputs become ``""``.
    """
    if value is None:
        return ""
    text = str(value)
    if not text or text.lower() == "nan":
        return ""

    # NFKC folds compatibility characters (fullwidth punctuation, etc.) without
    # requiring any external transliteration tables.
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("&", " and ").replace("+", " and ")
    text = _NON_ALNUM.sub(" ", text)
    text = _UNDERSCORE.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    if not text:
        return ""

    text = _LEGAL_PATTERN.sub(_expand_legal_token, text)
    tokens = [tok for tok in text.split(" ") if tok and tok not in _DROP_TOKENS]
    return " ".join(tokens)


def tokenize(value: object) -> List[str]:
    """Normalize then split on whitespace. Empty input -> empty list."""
    text = normalize_text(value)
    return text.split() if text else []


def record_text(business_name: object, business_address: object) -> str:
    """Pooled name+address string used as the TF-IDF document.

    If Indic text is present, transliterated Latin is appended so cosine similarity
    can score cross-script pairs against Latin reference names.
    """
    name = normalize_text(business_name)
    address = normalize_text(business_address)
    parts: List[str] = []
    if name:
        parts.append(name)
    if has_indic_script(business_name):
        t_latin = normalize_text(transliterate_to_latin(business_name))
        if t_latin and t_latin != name:
            parts.append(t_latin)
    if address:
        parts.append(address)
    return " ".join(parts)


def normalize_series(values: Iterable[object]) -> List[str]:
    """Helper for applying :func:`normalize_text` to a column."""
    return [normalize_text(v) for v in values]

