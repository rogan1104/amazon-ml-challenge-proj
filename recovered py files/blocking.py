"""TF-IDF character n-gram blocking with open-set bucket retrieval."""
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

_SKIP_NAME_TOKENS = frozenset({"private", "limited", "incorporated", "corporation", "llc", "llp", "plc", "company", "and", "the", "of", "dba", "services", "service", "group", "holdings", "holding"})
_DIGIT_TOKEN = re.compile(r"\d{3,}")
BlockKey = Tuple[str, str, str]


def prepare_corpus(df: pd.DataFrame) -> pd.DataFrame:
    name_n = [normalize_text(v) for v in df["business_name"]]
    addr_n = [normalize_text(v) for v in df["business_address"]]
    text = [record_text(n, a) for n, a in zip(df["business_name"], df["business_address"])]
    return pd.DataFrame({"entity_id": df["entity_id"].to_numpy(), "country": df["country"].to_numpy(), "name_n": name_n, "addr_n": addr_n, "text": text})


def blocking_keys(country: str, name_n: str, addr_n: str) -> List[BlockKey]:
    keys: List[BlockKey] = []
    tokens = [t for t in name_n.split() if len(t) >= 4 and t not in _SKIP_NAME_TOKENS]
    seen = set()
    for token in tokens[:4]:
        key = (country, "np", token[:4])
        if key not in seen:
            seen.add(key)
            keys.append(key)
    for digits in _DIGIT_TOKEN.findall(addr_n)[:3]:
        key = (country, "dg", digits)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    if not keys:
        keys.append((country, "fb", name_n[:3] if name_n else "_"))
    return keys


def _pooled_sample(texts: Sequence[pd.Series], n_per_source: int, seed: int) -> List[str]:
    rng = np.random.RandomState(seed)
    pooled: List[str] = []
    for series in texts:
        values = series.tolist()
        if len(values) <= n_per_source:
            pooled.extend(values)
        else:
            idx = rng.choice(len(values), size=n_per_source, replace=False)
            pooled.extend(values[i] for i in idx)
    return pooled


def fit_tfidf(source_texts: Sequence[pd.Series], ngram_range: tuple[int, int] = (3, 4), min_df: int = 3, max_df: float = 0.7, max_features: int = 80_000, sample_per_source: int = 250_000, seed: int = 42) -> TfidfVectorizer:
    sample = _pooled_sample(source_texts, sample_per_source, seed)
    min_df_eff = min(min_df, max(1, len(sample) // 50))
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, min_df=min_df_eff, max_df=max_df, max_features=max_features, lowercase=False, norm="l2", sublinear_tf=True, dtype=np.float32)
    t0 = time.time()
    vectorizer.fit(sample)
    print(f"[blocking] TF-IDF fit on {len(sample):,} pooled docs -> {len(vectorizer.get_feature_names_out()):,} char {ngram_range} grams ({time.time()-t0:.1f}s)")
    return vectorizer


def transform_texts(vectorizer: TfidfVectorizer, texts: Sequence[str], chunk_size: int = 100_000) -> csr_matrix:
    n_feat = len(vectorizer.get_feature_names_out())
    if not len(texts):
        return csr_matrix((0, n_feat), dtype=np.float32)
    blocks = [vectorizer.transform(list(texts[start:start+chunk_size])) for start in range(0, len(texts), chunk_size)]
    matrix = vstack(blocks, format="csr")
    matrix.sort_indices()
    return matrix


def _build_buckets(countries: Sequence[str], names: Sequence[str], addrs: Sequence[str]) -> Dict[BlockKey, List[int]]:
    buckets: Dict[BlockKey, List[int]] = defaultdict(list)
    for i, (country, name, addr) in enumerate(zip(countries, names, addrs)):
        for key in blocking_keys(country, name, addr):
            buckets[key].append(i)
    return buckets


def reduce_candidates(s1_ids: Sequence[str], acc_idx: Sequence[array], acc_dist: Sequence[array], k: int, cand_ids: np.ndarray) -> Dict[str, List[str]]:
    results: Dict[str, List[str]] = {}
    for eid, idx_buf, dist_buf in zip(s1_ids, acc_idx, acc_dist):
        if not idx_buf:
            results[str(eid)] = []
            continue
        idxs = np.frombuffer(idx_buf, dtype=np.int32)
        dists = np.frombuffer(dist_buf, dtype=np.float32)
        order = np.argsort(dists, kind="mergesort")
        out, seen = [], set()
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


def _query_bucket(x_s1: csr_matrix, x_cand: csr_matrix, s1_pos: np.ndarray, cand_pos: np.ndarray, cand_ids: np.ndarray, k: int, n_jobs: int, query_batch_size: int) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    n_cand, n_s1 = len(cand_pos), len(s1_pos)
    n_neighbors = int(min(k, n_cand))
    x_index = x_cand[cand_pos]
    if n_cand <= 800 or n_s1 * n_cand <= 200_000:
        for start in range(0, n_s1, query_batch_size):
            batch = s1_pos[start:start+query_batch_size]
            dists = cosine_distances(x_s1[batch], x_index)
            dists = np.nan_to_num(dists, nan=1.0, posinf=1.0)
            part = np.argpartition(dists, n_neighbors-1, axis=1)[:, :n_neighbors]
            for row, drow, idx in zip(batch, dists, part):
                order = idx[np.argsort(drow[idx])]
                yield row, cand_pos[order], drow[order]
        return
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute", n_jobs=n_jobs if n_cand >= 5_000 else 1)
    nn.fit(x_index)
    for start in range(0, n_s1, query_batch_size):
        batch = s1_pos[start:start+query_batch_size]
        dists, neigh = nn.kneighbors(x_s1[batch], n_neighbors=n_neighbors)
        dists = np.nan_to_num(dists, nan=1.0, posinf=1.0)
        for row, drow, idx in zip(batch, dists, neigh):
            yield row, cand_pos[idx], drow


def retrieve_candidates(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, k: int = 50, query_batch_size: int = 256, n_jobs: int = -1, vectorizer_kwargs: Mapping | None = None) -> Dict[str, List[str]]:
    t_all = time.time()
    print("[blocking] normalizing name+address ...")
    s1, s2, s3 = prepare_corpus(s1), prepare_corpus(s2), prepare_corpus(s3)
    vectorizer = fit_tfidf([s1["text"], s2["text"], s3["text"]], **(vectorizer_kwargs or {}))
    print("[blocking] transforming S2+S3 ...")
    candidates = pd.concat([s2, s3], ignore_index=True)
    t0 = time.time()
    x_cand = transform_texts(vectorizer, candidates["text"].tolist())
    print(f"[blocking] S2+S3 matrix {x_cand.shape} in {time.time()-t0:.1f}s")
    print("[blocking] transforming S1 ...")
    t0 = time.time()
    x_s1 = transform_texts(vectorizer, s1["text"].tolist())
    print(f"[blocking] S1 matrix {x_s1.shape} in {time.time()-t0:.1f}s")
    cand_ids, s1_ids = candidates["entity_id"].to_numpy(), s1["entity_id"].to_numpy()
    print("[blocking] building bucket inverted index ...")
    t0 = time.time()
    cand_buckets = _build_buckets(candidates["country"], candidates["name_n"], candidates["addr_n"])
    s1_buckets = _build_buckets(s1["country"], s1["name_n"], s1["addr_n"])
    print(f"[blocking] {len(s1_buckets):,} S1 keys, {len(cand_buckets):,} S2/S3 keys ({time.time()-t0:.1f}s)")
    del s1, s2, s3, candidates
    gc.collect()
    acc_idx, acc_dist = [array("i") for _ in range(len(s1_ids))], [array("f") for _ in range(len(s1_ids))]
    shared_keys = [key for key in s1_buckets if key in cand_buckets]
    print(f"[blocking] querying {len(shared_keys):,} overlapping buckets (k={k}) ...")
    t0 = time.time()
    for bi, key in enumerate(shared_keys, start=1):
        s1_pos = np.asarray(s1_buckets[key], dtype=np.int64)
        cand_pos = np.asarray(cand_buckets[key], dtype=np.int64)
        for row, pos, dists in _query_bucket(x_s1, x_cand, s1_pos, cand_pos, cand_ids, k, n_jobs, query_batch_size):
            acc_idx[row].frombytes(np.asarray(pos, dtype=np.int32).tobytes())
            acc_dist[row].frombytes(np.asarray(dists, dtype=np.float32).tobytes())
        if bi % 2000 == 0 or bi == len(shared_keys):
            print(f"[blocking]   buckets {bi:,}/{len(shared_keys):,} ({time.time()-t0:.1f}s)")
    del cand_buckets, s1_buckets
    gc.collect()
    print("[blocking] reducing to global top-k ...")
    results = reduce_candidates(s1_ids, acc_idx, acc_dist, k, cand_ids)
    empty = sum(1 for values in results.values() if not values)
    print(f"[blocking] done in {time.time()-t_all:.1f}s ({empty:,} S1 with 0 candidates)")
    return results
