"""Text normalization utilities for entity resolution forensics."""
import re
import unicodedata

# Legal suffix patterns
LEGAL_SUFFIXES = [
    (r"\bcorp\b\.?", "corporation"),
    (r"\bcorporation\b", "corporation"),
    (r"\binc\b\.?", "incorporated"),
    (r"\bincorporated\b", "incorporated"),
    (r"\bltd\b\.?", "limited"),
    (r"\blimited\b", "limited"),
    (r"\bllc\b\.?", "llc"),
    (r"\bllp\b\.?", "llp"),
    (r"\blp\b\.?", "lp"),
    (r"\bco\b\.?", "company"),
    (r"\bcompany\b", "company"),
    (r"\bpvt\b\.?", "private"),
    (r"\bprivate\b", "private"),
    (r"\bplc\b\.?", "plc"),
    (r"\bgmbh\b", "gmbh"),
    (r"\bsa\b\.?", "sa"),
    (r"\bsrl\b", "srl"),
    (r"\bpty\b", "pty"),
]

# Address abbreviation patterns
ADDR_ABBREVS = [
    (r"\brd\b\.?", "road"),
    (r"\broad\b", "road"),
    (r"\bst\b\.?", "street"),
    (r"\bstreet\b", "street"),
    (r"\bave\b\.?", "avenue"),
    (r"\bavenue\b", "avenue"),
    (r"\bblvd\b\.?", "boulevard"),
    (r"\bboulevard\b", "boulevard"),
    (r"\bdr\b\.?", "drive"),
    (r"\bdrive\b", "drive"),
    (r"\bln\b\.?", "lane"),
    (r"\blane\b", "lane"),
    (r"\bct\b\.?", "court"),
    (r"\bcourt\b", "court"),
    (r"\bapt\b\.?", "apartment"),
    (r"\bapartment\b", "apartment"),
    (r"\bste\b\.?", "suite"),
    (r"\bsuite\b", "suite"),
    (r"\bfl\b\.?", "floor"),
    (r"\bfloor\b", "floor"),
    (r"\bno\b\.?", "number"),
    (r"\bnum\b\.?", "number"),
]

PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
WS_RE = re.compile(r"\s+")
DIGIT_RE = re.compile(r"\d+")
POSTAL_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b|\b\d{6}\b|\b\d{5}\b")


def _safe_str(text) -> str:
    if text is None or (isinstance(text, float) and text != text):
        return ""
    return str(text) if not isinstance(text, str) else text


def basic_normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    text = _safe_str(text)
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.lower().strip()
    text = WS_RE.sub(" ", text)
    return text


def normalize_name(text: str) -> str:
    """Normalize business name for blocking/comparison."""
    text = basic_normalize(text)
    text = text.replace("&", " and ")
    text = PUNCT_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def normalize_address(text: str) -> str:
    """Normalize address for blocking/comparison."""
    text = basic_normalize(text)
    text = PUNCT_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def extract_tokens(text: str) -> set:
    """Extract word tokens from normalized text."""
    norm = basic_normalize(text)
    return set(norm.split()) if norm else set()


def extract_digits(text: str) -> set:
    """Extract numeric tokens."""
    text = _safe_str(text)
    if not text:
        return set()
    return set(DIGIT_RE.findall(text))


def extract_postal(text: str) -> set:
    """Extract postal/PIN codes."""
    text = _safe_str(text)
    if not text:
        return set()
    return set(POSTAL_RE.findall(text))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def char_ngrams(text: str, n: int = 3) -> set:
    text = basic_normalize(text).replace(" ", "")
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def name_prefix(text: str, n: int = 5) -> str:
    return normalize_name(text)[:n]


def has_pattern(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, _safe_str(text), re.IGNORECASE))


def detect_name_transformations(name1: str, name2: str) -> list[str]:
    """Detect which transformation patterns appear between a matched pair."""
    patterns = []
    n1, n2 = _safe_str(name1), _safe_str(name2)
    bn1, bn2 = basic_normalize(n1), basic_normalize(n2)

    if n1 == n2:
        patterns.append("exact")
        return patterns

    if bn1 == bn2:
        patterns.append("case_only")

    nn1, nn2 = normalize_name(n1), normalize_name(n2)
    if nn1 == nn2 and bn1 != bn2:
        patterns.append("punctuation_only")
    elif nn1 == nn2:
        patterns.append("normalized_exact")

    if "&" in n1 or "&" in n2 or " and " in bn1 or " and " in bn2:
        alt1 = bn1.replace(" and ", " & ")
        alt2 = bn2.replace(" and ", " & ")
        if alt1 == bn2 or alt2 == bn1 or bn1.replace("&", "and") == bn2.replace("&", "and"):
            patterns.append("ampersand_and")

    for pat, canonical in LEGAL_SUFFIXES:
        if has_pattern(n1, pat) or has_pattern(n2, pat):
            patterns.append(f"suffix_{canonical}")

    t1, t2 = set(nn1.split()), set(nn2.split())
    if t1 == t2 and nn1 != nn2:
        patterns.append("word_order")
    elif t1 & t2 and len(t1.symmetric_difference(t2)) <= 2:
        if len(t1) != len(t2):
            patterns.append("missing_or_extra_words")
        else:
            patterns.append("minor_token_diff")

    if len(nn1) > 3 and len(nn2) > 3:
        from rapidfuzz.distance import Levenshtein
        ratio = Levenshtein.normalized_similarity(nn1, nn2)
        if 0.7 <= ratio < 1.0:
            patterns.append("typo_or_substitution")

    return patterns if patterns else ["other_variation"]


def detect_address_transformations(addr1: str, addr2: str) -> list[str]:
    """Detect address transformation patterns between matched pair."""
    patterns = []
    a1, a2 = _safe_str(addr1), _safe_str(addr2)
    ba1, ba2 = basic_normalize(a1), basic_normalize(a2)

    if a1 == a2:
        patterns.append("exact")
        return patterns

    if ba1 == ba2:
        patterns.append("case_only")

    na1, na2 = normalize_address(a1), normalize_address(a2)
    if na1 == na2:
        patterns.append("normalized_exact")

    for pat, canonical in ADDR_ABBREVS:
        if has_pattern(a1, pat) or has_pattern(a2, pat):
            patterns.append(f"abbrev_{canonical}")

    d1, d2 = extract_digits(a1), extract_digits(a2)
    if d1 != d2:
        if not d1 or not d2:
            patterns.append("missing_numbers")
        else:
            patterns.append("numeric_diff")

    p1, p2 = extract_postal(a1), extract_postal(a2)
    if p1 != p2:
        if not p1 or not p2:
            patterns.append("missing_postal")
        else:
            patterns.append("postal_diff")

    t1, t2 = set(na1.split()), set(na2.split())
    if t1 == t2 and na1 != na2:
        patterns.append("component_reorder")

    return patterns if patterns else ["other_variation"]
