"""TF-IDF character n-gram blocking with cosine NearestNeighbors.

NOTE: Address-token (``ad``) keys and structured house-number digit keys are
included. Structured digits are extracted from the raw address because
``normalize_text`` destroys hyphen/slash/dot structure.

Pipeline
--------
1. Normalize name and address (see ``normalize.py``) and concatenate them
   into one document per record.
2. Fit ``TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4))`` on text
   pooled from every source (IDF is not source-specific).
3. Retrieve top-k Source 2/3 neighbors for each Source 1 record with sklearn
   ``NearestNeighbors(metric='cosine')``.

Why extra blocking keys?
    Brute-force cosine against ~10M S2/S3 rows is not feasible. We first put
    records into *open-set* buckets and run NearestNeighbors *inside* each
    bucket, then merge by cosine distance to a global top-k. Buckets never
    hard-code country names: whatever strings appear (US, India, France, ...)
    become partition labels automatically.

Bucket keys (union; a true match only needs to share ONE key):
    * ``(country, 'np', first-4 of a content name token)``
    * ``(country, 'ad', first-4 of a content address token)``  - locality even when names diverge
    * ``(country, 'dg', a 3+ digit token from the *normalized* address)``  - house / PIN
    * ``(country, 'dg', concatenated adjacent 2+ digit runs from the *raw* address)``
      when those runs sit next to ``-`` / ``/`` / ``.`` (Indian ``16-11-23/37/A`` style).
      Bare 2-digit tokens are never emitted alone — that would explode country buckets.
    * fallback ``(country, 'fb', first 3 chars of the name)`` if nothing else

``normalize_text()`` turns hyphens/slashes/dots into spaces, so structured house
numbers are only visible on the original address. ``blocking_keys`` therefore
takes an optional ``addr_raw``; ``prepare_corpus`` keeps that column and
``_build_buckets`` forwards it. 3-arg callers still work (structured keys skipped).
"""

from __future__ import annotations

import gc
import re
import time
from array import array
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors

from normalize import normalize_text, record_text
from transliterate import has_indic_script, transliterate_to_latin

# Tokens that are legal-form / glue words - they make useless name prefixes.
_SKIP_NAME_TOKENS = frozenset(
    {
        "private",
        "limited",
        "incorporated",
        "corporation",
        "llc",
        "llp",
        "plc",
        "company",
        "and",
        "the",
        "of",
        "dba",
        "services",
        "service",
        "group",
        "holdings",
        "holding",
    }
)
_DIGIT_TOKEN = re.compile(r"\d{3,}")
_POSTAL_TOKEN = re.compile(r"\b\d{5,6}\b")
# 2+ digit runs in RAW addresses; kept only when they touch house-number punctuation.
_STRUCTURED_DIGIT_RUN = re.compile(r"\d{2,}")
_HOUSE_PUNCT = frozenset("-./")
_SKIP_ADDR_TOKENS = frozenset(
    {
        "road",
        "street",
        "avenue",
        "lane",
        "drive",
        "plot",
        "house",
        "floor",
        "block",
        "sector",
        "phase",
        "colony",
        "nagar",
        "layout",
        "cross",
        "main",
        "area",
        "village",
        "district",
        "state",
        "india",
        "near",
        "behind",
        "opposite",
        "beside",
        "apartment",
        "building",
        "complex",
        "tower",
        "society",
        "number",
        "post",
        "office",
        "shop",
        "market",
        "south",
        "north",
        "east",
        "west",
        "city",
        "town",
        "station",
        "door",
        "flat",
        "room",
        "gate",
        "park",
    }
)
_STOP_NGRAMS = frozenset(
    {
        "the",
        "and",
        "inc",
        "ltd",
        "pvt",
        "cor",
        "ion",
        "ent",
        "ing",
        "com",
        "llc",
        "plc",
        "ser",
        "gro",
        "hol",
        "lim",
        "ite",
        "ted",
        "ati",
    }
)
_MAX_ADDR_TOKEN_KEYS = 4
_MAX_STRUCTURED_DIGIT_PAIRS = 2
_MAX_NAME_NGRAM_KEYS = 3
MAX_BUCKET_SIZE = 50_000
BlockKey = Tuple[str, str, str]


def _raw_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if not text or text.lower() == "nan":
        return ""
    return text


def _raw_address(value: object) -> str:
    return _raw_text(value)


def _structured_digit_runs(addr_raw: str) -> List[str]:
    """2+ digit runs adjacent to ``-`` / ``/`` / ``.`` in the unnormalized address.

    Must not run on ``addr_n``: ``normalize_text`` replaces that punctuation with
    spaces, so ``16-11-23/37/A`` becomes isolated 2-digit tokens with no structure.
    """
    if not addr_raw:
        return []
    runs: List[str] = []
    for match in _STRUCTURED_DIGIT_RUN.finditer(addr_raw):
        start, end = match.span()
        left = addr_raw[start - 1] if start else ""
        right = addr_raw[end] if end < len(addr_raw) else ""
        if left in _HOUSE_PUNCT or right in _HOUSE_PUNCT:
            runs.append(match.group(0))
    return runs


def prepare_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Attach normalized fields used for keys + the TF-IDF document.

    If Indic text is detected, transliteration is appended so TF-IDF can score
    cross-script matches in the same embedding space.
    """
    name_raw = [_raw_text(v) for v in df["business_name"]]
    addr_raw = [_raw_text(v) for v in df["business_address"]]
    name_n = [normalize_text(v) for v in name_raw]
    addr_n = [normalize_text(v) for v in addr_raw]

    docs: List[str] = []
    for nr, nn, an in zip(name_raw, name_n, addr_n):
        if has_indic_script(nr):
            t_latin = normalize_text(transliterate_to_latin(nr))
            doc = f"{nn} {t_latin} {an}".strip()
        else:
            doc = f"{nn} {an}".strip()
        docs.append(doc)

    return pd.DataFrame(
        {
            "entity_id": df["entity_id"].to_numpy(),
            "country": df["country"].to_numpy(),
            "name_raw": name_raw,
            "addr_raw": addr_raw,
            "name_n": name_n,
            "addr_n": addr_n,
            "text": docs,
        }
    )


def blocking_keys(
    country: str,
    name_n: str,
    addr_n: str,
    addr_raw: str = "",
    name_raw: str = "",
) -> List[BlockKey]:
    """Return open-set multi-channel bucket keys for one record.

    Channels:
    - np: Name token prefix (first 4 chars of content tokens >= 4 chars)
    - sn: Short name tokens (length 2-3, e.g. BMW, KFC, TCS, IBM)
    - tl: Transliterated name prefix tokens (for Indic script names)
    - ng: Character 3-grams on name content tokens
    - pin: 5 or 6 digit postal / PIN codes
    - dg: 3+ digit runs and structured house numbers
    - ad: Address content token prefixes
    - fb: Fallback
    """
    keys: List[BlockKey] = []
    seen = set()
    c = country.strip() if country else ""

    # 1. Standard name token prefixes (length >= 4)
    tokens = [t for t in name_n.split() if len(t) >= 4 and t not in _SKIP_NAME_TOKENS]
    for tok in tokens[:4]:
        key = (c, "np", tok[:4])
        if key not in seen:
            seen.add(key)
            keys.append(key)

    # 2. Short name handling: whole short names (<= 5 chars) or 2-3 char tokens (BMW, KFC, TCS)
    if 2 <= len(name_n) <= 5 and not name_n.isdigit():
        key = (c, "sn", name_n.replace(" ", ""))
        if key not in seen:
            seen.add(key)
            keys.append(key)
    short_tokens = [
        t for t in name_n.split()
        if 2 <= len(t) <= 3 and not t.isdigit() and t not in _SKIP_NAME_TOKENS
    ]
    for tok in short_tokens[:3]:
        key = (c, "sn", tok)
        if key not in seen:
            seen.add(key)
            keys.append(key)

    # 3. Indic transliteration channel
    if name_raw and has_indic_script(name_raw):
        t_latin = transliterate_to_latin(name_raw)
        t_latin_n = normalize_text(t_latin)
        t_tokens = [t for t in t_latin_n.split() if len(t) >= 4 and t not in _SKIP_NAME_TOKENS]
        for tok in t_tokens[:4]:
            # Emit both as 'np' so it directly aligns with S1 English tokens, and 'tl'
            for prefix_kind in ("np", "tl"):
                key = (c, prefix_kind, tok[:4])
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        for tok in [t for t in t_latin_n.split() if 2 <= len(t) <= 3 and not t.isdigit() and t not in _SKIP_NAME_TOKENS][:2]:
            key = (c, "sn", tok)
            if key not in seen:
                seen.add(key)
                keys.append(key)

    # 4. Character 3-gram channel on name tokens (handles typos / prefix variations)
    ngram_count = 0
    for tok in tokens[:2]:
        if len(tok) >= 4 and ngram_count < _MAX_NAME_NGRAM_KEYS:
            for i in range(len(tok) - 2):
                gram = tok[i : i + 3]
                if gram not in _STOP_NGRAMS:
                    key = (c, "ng", gram)
                    if key not in seen:
                        seen.add(key)
                        keys.append(key)
                        ngram_count += 1
                        if ngram_count >= _MAX_NAME_NGRAM_KEYS:
                            break

    # 5. Address locality tokens
    addr_tokens = [
        t
        for t in addr_n.split()
        if len(t) >= 4 and not t.isdigit() and t not in _SKIP_ADDR_TOKENS
    ]
    for tok in addr_tokens[:_MAX_ADDR_TOKEN_KEYS]:
        key = (c, "ad", tok[:4])
        if key not in seen:
            seen.add(key)
            keys.append(key)

    # 6. Postal / PIN code channel (6 digits in India, 5 digits in US/France)
    for pin in _POSTAL_TOKEN.findall(addr_n):
        key = (c, "pin", pin)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    if addr_raw:
        for pin in _POSTAL_TOKEN.findall(addr_raw):
            key = (c, "pin", pin)
            if key not in seen:
                seen.add(key)
                keys.append(key)

    # 7. Address 3+ digit runs
    for digits in _DIGIT_TOKEN.findall(addr_n)[:3]:
        key = (c, "dg", digits)
        if key not in seen:
            seen.add(key)
            keys.append(key)

    # 8. Structured house numbers from raw address
    runs = _structured_digit_runs(addr_raw)
    pair_count = 0
    for left, right in zip(runs, runs[1:]):
        if pair_count >= _MAX_STRUCTURED_DIGIT_PAIRS:
            break
        key = (c, "dg", left + right)
        if key not in seen:
            seen.add(key)
            keys.append(key)
        pair_count += 1
    for digits in runs:
        if len(digits) < 3:
            continue
        key = (c, "dg", digits)
        if key not in seen:
            seen.add(key)
            keys.append(key)

    # 9. Fallback key if nothing else matched
    if not keys:
        keys.append((c, "fb", name_n[:3] if name_n else "_"))
    return keys


def _pooled_sample(texts: Sequence[pd.Series], n_per_source: int, seed: int) -> List[str]:
    """Stratified sample across sources so IDF sees every source's noise."""
    rng = np.random.RandomState(seed)
    pooled: List[str] = []
    for series in texts:
        values = series.tolist()
        if not values:
            continue
        if len(values) <= n_per_source:
            pooled.extend(values)
        else:
            idx = rng.choice(len(values), size=n_per_source, replace=False)
            pooled.extend(values[i] for i in idx)
    return pooled


def fit_tfidf(
    source_texts: Sequence[pd.Series],
    ngram_range: tuple[int, int] = (3, 4),
    min_df: int = 3,
    max_df: float = 0.7,
    max_features: int = 80_000,
    sample_per_source: int = 250_000,
    seed: int = 42,
) -> TfidfVectorizer:
    """Fit char_wb TF-IDF on pooled name+address text from all sources.

    Fitting on every row would materialize a ~12M-row count matrix just to
    throw it away. A large *pooled* sample from each source learns the same
    n-gram vocabulary / IDF and keeps peak RAM workable. ``transform`` later
    still scores every record.
    """
    sample = _pooled_sample(source_texts, sample_per_source, seed)
    # min_df cannot exceed the sample size (debug --limit runs).
    min_df_eff = min(min_df, max(1, len(sample) // 50))
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=min_df_eff,
        max_df=max_df,
        max_features=max_features,
        lowercase=False,  # already normalized
        norm="l2",
        sublinear_tf=True,
        dtype=np.float32,
    )
    t0 = time.time()
    vectorizer.fit(sample)
    n_feat = len(vectorizer.get_feature_names_out())
    print(
        f"[blocking] TF-IDF fit on {len(sample):,} pooled docs "
        f"-> {n_feat:,} char {ngram_range} grams ({time.time() - t0:.1f}s)"
    )
    return vectorizer


def transform_texts(
    vectorizer: TfidfVectorizer,
    texts: Sequence[str],
    chunk_size: int = 100_000,
) -> csr_matrix:
    """Transform in chunks so we never densify the full corpus."""
    n_feat = len(vectorizer.get_feature_names_out())
    if not len(texts):
        return csr_matrix((0, n_feat), dtype=np.float32)
    blocks = []
    n = len(texts)
    for start in range(0, n, chunk_size):
        chunk = list(texts[start : start + chunk_size])
        blocks.append(vectorizer.transform(chunk))
    matrix = vstack(blocks, format="csr")
    matrix.sort_indices()
    return matrix


def _build_buckets(
    countries: Sequence[str],
    names: Sequence[str],
    addrs: Sequence[str],
    addrs_raw: Sequence[str] | None = None,
    names_raw: Sequence[str] | None = None,
) -> Dict[BlockKey, List[int]]:
    buckets: Dict[BlockKey, List[int]] = defaultdict(list)
    addr_raw_iter: Sequence[str] = addrs_raw if addrs_raw is not None else [""] * len(names)
    name_raw_iter: Sequence[str] = names_raw if names_raw is not None else [""] * len(names)
    for i, (country, name, addr, a_raw, n_raw) in enumerate(
        zip(countries, names, addrs, addr_raw_iter, name_raw_iter)
    ):
        for key in blocking_keys(country, name, addr, a_raw, n_raw):
            buckets[key].append(i)
    return buckets


def reduce_candidates(
    s1_ids: Sequence[str],
    acc_idx: Sequence[array],
    acc_dist: Sequence[array],
    k: int,
    cand_ids: np.ndarray,
) -> Dict[str, List[str]]:
    """Per S1: unique candidate row-indexes with the smallest cosine distance."""
    results: Dict[str, List[str]] = {}
    for eid, idx_buf, dist_buf in zip(s1_ids, acc_idx, acc_dist):
        if not idx_buf:
            results[str(eid)] = []
            continue
        idxs = np.frombuffer(idx_buf, dtype=np.int32)
        dists = np.frombuffer(dist_buf, dtype=np.float32)
        order = np.argsort(dists, kind="mergesort")
        out: List[str] = []
        seen = set()
        for j in order:
            ci = int(idxs[j])
            if ci in seen:
                continue
            seen.add(ci)
            out.append(str(cand_ids[ci]))
            if len(out) >= k:
                break
        results[str(eid)] = out
    return results


def _query_bucket(
    x_s1: csr_matrix,
    x_cand: csr_matrix,
    s1_pos: np.ndarray,
    cand_pos: np.ndarray,
    cand_ids: np.ndarray,
    k: int,
    n_jobs: int,
    query_batch_size: int,
) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (s1_row, neighbor_ids, cosine_distances) for one bucket."""
    n_cand = len(cand_pos)
    n_s1 = len(s1_pos)
    n_neighbors = int(min(k, n_cand))
    x_index = x_cand[cand_pos]

    # Small buckets: pairwise cosine is cheaper than fitting NN.
    if n_cand <= 800 or n_s1 * n_cand <= 200_000:
        for start in range(0, n_s1, query_batch_size):
            batch = s1_pos[start : start + query_batch_size]
            dists = cosine_distances(x_s1[batch], x_index)
            dists = np.nan_to_num(dists, nan=1.0, posinf=1.0)
            part = np.argpartition(dists, n_neighbors - 1, axis=1)[:, :n_neighbors]
            for row, drow, idx in zip(batch, dists, part):
                order = idx[np.argsort(drow[idx])]
                yield row, cand_pos[order], drow[order]
        return

    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute",
        n_jobs=n_jobs if n_cand >= 5_000 else 1,
    )
    nn.fit(x_index)
    for start in range(0, n_s1, query_batch_size):
        batch = s1_pos[start : start + query_batch_size]
        dists, neigh = nn.kneighbors(x_s1[batch], n_neighbors=n_neighbors)
        dists = np.nan_to_num(dists, nan=1.0, posinf=1.0)
        for row, drow, idx in zip(batch, dists, neigh):
            yield row, cand_pos[idx], drow


def retrieve_candidates(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    k: int = 50,
    query_batch_size: int = 256,
    n_jobs: int = -1,
    vectorizer_kwargs: Mapping | None = None,
    max_bucket_size: int = MAX_BUCKET_SIZE,
) -> Dict[str, List[str]]:
    """Return ``{s1_entity_id: [s2/s3 ids...]}`` with at most ``k`` neighbors."""
    t_all = time.time()
    print("[blocking] normalizing name+address and building multi-channel documents ...")
    s1 = prepare_corpus(s1)
    s2 = prepare_corpus(s2)
    s3 = prepare_corpus(s3)

    vectorizer = fit_tfidf(
        [s1["text"], s2["text"], s3["text"]],
        **(vectorizer_kwargs or {}),
    )

    print("[blocking] transforming S2+S3 ...")
    candidates = pd.concat([s2, s3], ignore_index=True)
    t0 = time.time()
    x_cand = transform_texts(vectorizer, candidates["text"].tolist())
    print(f"[blocking] S2+S3 matrix {x_cand.shape} in {time.time() - t0:.1f}s")

    print("[blocking] transforming S1 ...")
    t0 = time.time()
    x_s1 = transform_texts(vectorizer, s1["text"].tolist())
    print(f"[blocking] S1 matrix {x_s1.shape} in {time.time() - t0:.1f}s")

    cand_ids = candidates["entity_id"].to_numpy()
    s1_ids = s1["entity_id"].to_numpy()

    print("[blocking] building multi-channel inverted index buckets ...")
    t0 = time.time()
    cand_buckets = _build_buckets(
        candidates["country"],
        candidates["name_n"],
        candidates["addr_n"],
        candidates["addr_raw"],
        candidates["name_raw"],
    )
    s1_buckets = _build_buckets(
        s1["country"],
        s1["name_n"],
        s1["addr_n"],
        s1["addr_raw"],
        s1["name_raw"],
    )
    print(
        f"[blocking] {len(s1_buckets):,} S1 keys, {len(cand_buckets):,} S2/S3 keys "
        f"({time.time() - t0:.1f}s)"
    )

    # Free raw strings; matrices + ids + buckets are enough from here.
    del s1, s2, s3, candidates
    gc.collect()

    n_s1 = len(s1_ids)
    acc_idx = [array("i") for _ in range(n_s1)]
    acc_dist = [array("f") for _ in range(n_s1)]

    shared_keys = [key for key in s1_buckets if key in cand_buckets]
    print(f"[blocking] querying {len(shared_keys):,} overlapping buckets (k={k}, max_bucket={max_bucket_size:,}) ...")
    t0 = time.time()
    skipped_large = 0
    for bi, key in enumerate(shared_keys, start=1):
        cand_pos = np.asarray(cand_buckets[key], dtype=np.int64)
        if len(cand_pos) > max_bucket_size:
            skipped_large += 1
            continue
        s1_pos = np.asarray(s1_buckets[key], dtype=np.int64)
        for row, pos, dists in _query_bucket(
            x_s1,
            x_cand,
            s1_pos,
            cand_pos,
            cand_ids,
            k=k,
            n_jobs=n_jobs,
            query_batch_size=query_batch_size,
        ):
            acc_idx[row].frombytes(np.asarray(pos, dtype=np.int32).tobytes())
            acc_dist[row].frombytes(np.asarray(dists, dtype=np.float32).tobytes())

        if bi % 2000 == 0 or bi == len(shared_keys):
            elapsed = time.time() - t0
            print(
                f"[blocking]   buckets {bi:,}/{len(shared_keys):,} "
                f"({elapsed:.1f}s)"
            )

    if skipped_large:
        print(f"[blocking] skipped {skipped_large:,} oversized stop-buckets (> {max_bucket_size:,})")

    del cand_buckets, s1_buckets
    gc.collect()
    print("[blocking] reducing to global top-k ...")
    results = reduce_candidates(s1_ids, acc_idx, acc_dist, k, cand_ids)
    empty = sum(1 for v in results.values() if not v)
    print(
        f"[blocking] done in {time.time() - t_all:.1f}s "
        f"({empty:,} S1 with 0 candidates)"
    )
    return results
