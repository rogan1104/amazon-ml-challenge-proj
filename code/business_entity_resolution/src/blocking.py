"""Blocking pipeline for Business Entity Resolution.

Generates bucket blocking keys from business records (country, name, address)
to build candidate pairs without full O(N^2) comparison.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any, Iterable, Sequence

# Punctuation and whitespace patterns
PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
WS_RE = re.compile(r"\s+")

# Existing digit pattern: 3+ consecutive digits
DIGIT_3PLUS_RE = re.compile(r"\d{3,}")

# Loosened structured numbering pattern on RAW address text:
# Captures house numbers with hyphens/slashes/dots like "16-11-23/37/A", "12-13-415", "4/50", "23-B", "3-6-275/1"
STRUCTURED_NUM_RAW_RE = re.compile(r"\b\d{1,5}(?:[-/][0-9a-zA-Z]+)+\b")

# Fallback pattern for normalized address text:
# Detects 2 or more adjacent 2+ digit numbers that were separated by hyphens/slashes before normalization
STRUCTURED_NUM_NORM_RE = re.compile(r"\b\d{2,}(?:\s+\d{2,})+\b")

# Generic address stopwords that appear frequently in addresses and could cause bucket explosion
# if allowed to generate unconstrained address-token blocking keys.
ADDRESS_STOPWORDS = {
    "road", "street", "near", "opp", "opposite", "behind", "beside",
    "floor", "flat", "house", "plot", "building", "cross", "main",
    "lane", "avenue", "dist", "district", "post", "state", "india",
    "area", "nagar", "colony", "block", "sector", "layout", "circle",
    "bhavan", "complex", "tower", "phase", "stage", "dept", "ward",
    "village", "mandal", "taluk", "town", "city", "line", "room",
}


def normalize_text(text: str | None) -> str:
    """Standard normalization: lowercase, NFKD unicode normalization, strip punctuation, collapse whitespace."""
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text).lower().strip()
    text = PUNCT_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def blocking_keys(
    country: str | None,
    name_n: str | None,
    addr_n: str | None,
    addr_raw: str | None = None,
    max_name_tokens: int = 3,
    max_addr_tokens: int = 3,
    max_digit_keys: int = 5,
) -> list[tuple[str, str, str]]:
    """Generate candidate bucket blocking keys for a record.

    NOTE ON `addr_raw`:
        Standard normalization converts punctuation like hyphens and slashes into spaces
        (e.g., 'H.No.16-11-23/37/A' -> 'h no 16 11 23 37 a'). This destroys the structural
        delimiters that make 2-digit numbers identifiable as house-number components rather
        than generic numbers (like floor, ward, or sector numbers).
        Passing `addr_raw` allows extracting structured numbering before punctuation is stripped.
        If `addr_raw` is None, a fallback pattern on `addr_n` detects adjacent 2-digit runs.

    Args:
        country: Country code (e.g., 'IN', 'US').
        name_n: Normalized business name.
        addr_n: Normalized business address.
        addr_raw: Optional unnormalized address string containing original punctuation.
        max_name_tokens: Maximum name token keys to generate per record.
        max_addr_tokens: Maximum address token keys to generate per record.
        max_digit_keys: Maximum digit/house-number keys to generate per record.

    Returns:
        List of unique blocking key tuples, e.g. (country, 'nm', 'drea'),
        (country, 'ad', 'osar'), (country, 'dg', '500036'), (country, 'hn', '16-11-23/37/a').
    """
    c = (country or "").strip().upper()
    keys: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def add_key(k_type: str, k_val: str) -> None:
        key = (c, k_type, k_val)
        if key not in seen:
            seen.add(key)
            keys.append(key)

    # 1. Existing Name-Token Keys: (country, 'nm', tok[:4])
    # Generates keys for the first few significant name tokens of length >= 4.
    if name_n:
        name_tokens = [tok for tok in name_n.split() if len(tok) >= 4 and not tok.isdigit()]
        for tok in name_tokens[:max_name_tokens]:
            add_key("nm", tok[:4])

    # 2. NEW Address-Token Key (Requirement 1): (country, 'ad', tok[:4])
    # Catches cases where names differ completely (transliteration, Telugu/Indic script vs English, typos)
    # but addresses are near-identical.
    if addr_n:
        addr_tokens = [
            tok for tok in addr_n.split()
            if len(tok) >= 4 and tok not in ADDRESS_STOPWORDS and not tok.isdigit()
        ]
        # Cap how many address-token keys get generated per record to prevent bucket explosion
        for tok in addr_tokens[:max_addr_tokens]:
            add_key("ad", tok[:4])

    # 3. Digit Keys & Structured House Numbering (Requirement 2):
    # (a) Existing 3+ consecutive digit runs: r"\d{3,}"
    # Checked on addr_n as well as addr_raw.
    digit_sources = [addr_n or ""]
    if addr_raw:
        digit_sources.append(addr_raw)

    for src in digit_sources:
        for d in DIGIT_3PLUS_RE.findall(src):
            add_key("dg", d)

    # (b) Loosened 2-digit runs for structured house-numbering:
    # Captures 2-digit components (r"\d{2,}") ONLY when surrounded/joined by
    # punctuation like hyphens/slashes/dots in the structured address.
    structured_count = 0

    if addr_raw:
        # Extract directly from raw text using original punctuation delimiters
        raw_matches = STRUCTURED_NUM_RAW_RE.findall(addr_raw)
        for m in raw_matches:
            if structured_count >= max_digit_keys:
                break
            # Composite house-number key (e.g. ('IN', 'hn', '16-11-23/37/a'))
            norm_m = m.lower().strip()
            add_key("hn", norm_m)
            structured_count += 1

            # Extract individual 2-digit components within this structured house number
            parts = re.findall(r"\d{2,}", m)
            for part in parts:
                if structured_count >= max_digit_keys:
                    break
                add_key("dg", part)
                structured_count += 1
    elif addr_n:
        # Fallback when addr_raw is not provided:
        # Detect adjacent 2-digit token sequences created from punctuation stripping
        norm_seqs = STRUCTURED_NUM_NORM_RE.findall(addr_n)
        for seq in norm_seqs:
            if structured_count >= max_digit_keys:
                break
            parts = seq.split()
            # Composite key joined by hyphen
            add_key("hn", "-".join(parts))
            structured_count += 1
            for p in parts:
                if structured_count >= max_digit_keys:
                    break
                add_key("dg", p)
                structured_count += 1

    return keys


def build_blocking_buckets(
    records: Sequence[dict[str, Any]],
    use_raw_address: bool = True,
) -> dict[tuple[str, str, str], list[str]]:
    """Build inverted index mapping blocking key -> list of record entity IDs."""
    buckets: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for rec in records:
        entity_id = rec.get("entity_id") or rec.get("id")
        if not entity_id:
            continue
        country = rec.get("country", "")
        name_n = rec.get("name_n") or normalize_text(rec.get("business_name", ""))
        addr_n = rec.get("addr_n") or normalize_text(rec.get("business_address", ""))
        addr_raw = rec.get("business_address") if use_raw_address else None

        keys = blocking_keys(country, name_n, addr_n, addr_raw=addr_raw)
        for k in keys:
            buckets[k].append(str(entity_id))
    return buckets


def query_blocking_candidates(
    record: dict[str, Any],
    buckets: dict[tuple[str, str, str], list[str]],
    max_bucket_size: int = 5000,
    max_candidates: int = 500,
    use_raw_address: bool = True,
) -> set[str]:
    """Retrieve candidate entity IDs for a query record from blocking buckets.

    Args:
        record: Query record dict.
        buckets: Pre-built blocking buckets index.
        max_bucket_size: Skip buckets larger than this to prevent explosive fanout.
        max_candidates: Maximum candidates to retrieve for this record.
        use_raw_address: Whether to pass raw business_address to blocking_keys.

    Returns:
        Set of candidate entity IDs.
    """
    country = record.get("country", "")
    name_n = record.get("name_n") or normalize_text(record.get("business_name", ""))
    addr_n = record.get("addr_n") or normalize_text(record.get("business_address", ""))
    addr_raw = record.get("business_address") if use_raw_address else None

    keys = blocking_keys(country, name_n, addr_n, addr_raw=addr_raw)
    candidates: set[str] = set()

    for k in keys:
        postings = buckets.get(k)
        if not postings:
            continue
        # Stop-key filtering: skip huge buckets that cause billions of false collisions
        if len(postings) > max_bucket_size:
            continue
        for eid in postings:
            candidates.add(eid)
            if len(candidates) >= max_candidates:
                return candidates

    return candidates
