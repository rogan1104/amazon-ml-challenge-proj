"""Build labeled pair features from stage-1 candidates, in output batches."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Sequence

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from data_io import iter_candidate_rows, load_ground_truth, load_source, parse_id_list
from normalize import normalize_text, record_text

PAIR_COLUMNS = [
    "source1_entity_id", "candidate_entity_id", "label",
    "name_levenshtein_ratio", "name_token_jaccard", "name_address_tfidf_cosine",
    "address_token_jaccard", "country_match", "name_length_diff",
]


def load_truth_map(path: str | Path) -> Dict[str, set[str]]:
    truth = load_ground_truth(path)
    return {str(s1): set(parse_id_list(raw)) for s1, raw in zip(truth["source1_entity_id"], truth["matched_entity_ids"])}


def load_record_tables(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    s1 = load_source(data_dir / "train_source1.tsv")
    s2 = load_source(data_dir / "train_source2.tsv")
    s3 = load_source(data_dir / "train_source3.tsv")
    candidates = pd.concat([s2, s3], ignore_index=True)
    if s1["entity_id"].duplicated().any() or candidates["entity_id"].duplicated().any():
        raise ValueError("Entity IDs must be unique within their source tables")
    return s1, candidates


def pair_batches(candidate_path: str | Path, rows_per_batch: int = 20_000) -> Iterator[pd.DataFrame]:
    """Stream candidate lists and explode them into bounded pair dataframes."""
    rows: list[tuple[str, str]] = []
    for s1_id, candidate_ids in iter_candidate_rows(candidate_path):
        rows.extend((str(s1_id), str(candidate_id)) for candidate_id in candidate_ids)
        while len(rows) >= rows_per_batch:
            yield pd.DataFrame(rows[:rows_per_batch], columns=PAIR_COLUMNS[:2])
            del rows[:rows_per_batch]
    if rows:
        yield pd.DataFrame(rows, columns=PAIR_COLUMNS[:2])


def fit_shared_vectorizer(s1: pd.DataFrame, candidates: pd.DataFrame, sample_size: int = 500_000, random_state: int = 42) -> TfidfVectorizer:
    """Fit one shared char n-gram TF-IDF vectorizer on a pooled record sample."""
    rng = np.random.RandomState(random_state)
    parts = []
    for frame, quota in ((s1, sample_size // 2), (candidates, sample_size - sample_size // 2)):
        take = min(len(frame), quota)
        indices = rng.choice(len(frame), take, replace=False) if take < len(frame) else np.arange(len(frame))
        names = frame["business_name"].to_numpy()
        addresses = frame["business_address"].to_numpy()
        parts.extend(record_text(names[int(i)], addresses[int(i)]) for i in indices)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), max_features=100_000, dtype=np.float32, sublinear_tf=True, norm="l2")
    vectorizer.fit(parts)
    return vectorizer


def transform_source_once(vectorizer: TfidfVectorizer, frame: pd.DataFrame, chunk_size: int = 50_000) -> csr_matrix:
    """Transform each unique source record exactly once, in bounded text chunks."""
    names = frame["business_name"].to_numpy()
    addresses = frame["business_address"].to_numpy()
    blocks = []
    for start in range(0, len(frame), chunk_size):
        stop = min(start + chunk_size, len(frame))
        docs = [record_text(names[i], addresses[i]) for i in range(start, stop)]
        blocks.append(vectorizer.transform(docs))
    if not blocks:
        return csr_matrix((0, len(vectorizer.vocabulary_)), dtype=np.float32)
    return vstack(blocks, format="csr")


def _batched_token_jaccard(left_docs: Sequence[str], right_docs: Sequence[str]) -> np.ndarray:
    """Compute pair-aligned token Jaccard with sparse binary count matrices."""
    n = len(left_docs)
    if n == 0:
        return np.empty(0, dtype=np.float32)
    vectorizer = CountVectorizer(binary=True, tokenizer=str.split, preprocessor=None, token_pattern=None, lowercase=False, dtype=np.int32)
    try:
        both = vectorizer.fit_transform(list(left_docs) + list(right_docs)).tocsr()
    except ValueError as exc:  # Empty vocabulary when every input is blank.
        if "empty vocabulary" not in str(exc):
            raise
        return np.zeros(n, dtype=np.float32)
    left, right = both[:n], both[n:]
    intersection = np.asarray(left.multiply(right).sum(axis=1)).ravel()
    union = np.asarray(left.sum(axis=1)).ravel() + np.asarray(right.sum(axis=1)).ravel() - intersection
    return np.divide(intersection, union, out=np.zeros(n, dtype=np.float32), where=union != 0).astype(np.float32, copy=False)


def make_feature_batch(
    pairs: pd.DataFrame,
    s1: pd.DataFrame,
    candidates: pd.DataFrame,
    truth: Mapping[str, set[str]],
    vectorizer: TfidfVectorizer,
    x_s1: csr_matrix,
    x_candidates: csr_matrix,
    s1_index: pd.Index,
    candidate_index: pd.Index,
) -> pd.DataFrame:
    """Join records, then compute all similarities for this batch."""
    left = pairs.merge(s1, left_on="source1_entity_id", right_on="entity_id", how="left", validate="many_to_one")
    left = left.rename(columns={"business_name": "name_left", "business_address": "address_left", "country": "country_left"}).drop(columns="entity_id")
    joined = left.merge(candidates, left_on="candidate_entity_id", right_on="entity_id", how="left", validate="many_to_one", suffixes=("", "_right"))
    if joined["business_name"].isna().any():
        raise ValueError(f"{int(joined['business_name'].isna().sum())} candidate IDs missing from S2/S3")

    name_l = [normalize_text(v) for v in joined["name_left"]]
    name_r = [normalize_text(v) for v in joined["business_name"]]
    addr_l = [normalize_text(v) for v in joined["address_left"]]
    addr_r = [normalize_text(v) for v in joined["business_address"]]

    # RapidFuzz's cpdist compares aligned pairs in native code; it avoids one
    # Python function call per pair while keeping memory bounded to this batch.
    name_ratio = process.cpdist(name_l, name_r, scorer=fuzz.ratio, processor=None, workers=-1, dtype=np.float32) / 100.0
    name_jaccard = _batched_token_jaccard(name_l, name_r)
    address_jaccard = _batched_token_jaccard(addr_l, addr_r)

    # The full-source CSR matrices were transformed once before pair iteration.
    # Here only gather the rows needed by this pair batch and multiply them.
    li = s1_index.get_indexer(joined["source1_entity_id"])
    ri = candidate_index.get_indexer(joined["candidate_entity_id"])
    if (li < 0).any() or (ri < 0).any():
        raise ValueError("Pair IDs could not be mapped to their cached TF-IDF rows")
    row_products = x_s1[li].multiply(x_candidates[ri])
    cosine = np.asarray(row_products.sum(axis=1)).ravel().astype(np.float32, copy=False)

    labels = np.fromiter(
        (int(str(candidate_id) in truth.get(str(s1_id), set())) for s1_id, candidate_id in zip(joined["source1_entity_id"], joined["candidate_entity_id"])),
        dtype=np.int8,
        count=len(joined),
    )
    return pd.DataFrame({
        "source1_entity_id": joined["source1_entity_id"].astype(str),
        "candidate_entity_id": joined["candidate_entity_id"].astype(str),
        "label": labels,
        "name_levenshtein_ratio": np.asarray(name_ratio, dtype=np.float32),
        "name_token_jaccard": name_jaccard,
        "name_address_tfidf_cosine": cosine,
        "address_token_jaccard": address_jaccard,
        "country_match": (joined["country_left"].to_numpy() == joined["country"].to_numpy()).astype(np.int8),
        "name_length_diff": np.abs(np.asarray([len(v) for v in name_l]) - np.asarray([len(v) for v in name_r])).astype(np.int32),
    }, columns=PAIR_COLUMNS)


def build_features(data_dir: str | Path, candidate_path: str | Path, output_path: str | Path, rows_per_batch: int = 20_000, fit_sample_size: int = 500_000) -> None:
    """Write the labeled training feature table incrementally to Parquet."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet output requires a locally installed PyArrow engine") from exc
    s1, candidates = load_record_tables(data_dir)
    truth = load_truth_map(Path(data_dir) / "train_ground_truth.tsv")
    print(f"[features] loaded {len(s1):,} S1 and {len(candidates):,} S2/S3 records")
    vectorizer = fit_shared_vectorizer(s1, candidates, fit_sample_size)
    print(f"[features] transforming each source record once (vocabulary={len(vectorizer.vocabulary_):,})")
    x_s1 = transform_source_once(vectorizer, s1)
    x_candidates = transform_source_once(vectorizer, candidates)
    s1_index = pd.Index(s1["entity_id"])
    candidate_index = pd.Index(candidates["entity_id"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    total = 0
    try:
        for pairs in pair_batches(candidate_path, rows_per_batch):
            features = make_feature_batch(pairs, s1, candidates, truth, vectorizer, x_s1, x_candidates, s1_index, candidate_index)
            table = pa.Table.from_pandas(features, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
            total += len(features)
            if total % (rows_per_batch * 10) < len(features):
                print(f"[features] wrote {total:,} pairs")
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("Candidate file contained no pairs")
    print(f"[features] wrote {total:,} rows to {output_path}")


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    project_root = root / "student_resource" if (root / "student_resource" / "dataset").is_dir() else root
    parser = argparse.ArgumentParser(description="Build labeled stage-2 pair features")
    parser.add_argument("--data-dir", type=Path, default=project_root / "dataset" / "train")
    parser.add_argument("--candidates", type=Path, default=project_root / "output" / "candidate_pairs.tsv")
    parser.add_argument("--output", type=Path, default=project_root / "output" / "train_features.parquet")
    parser.add_argument("--batch-rows", type=int, default=20_000)
    parser.add_argument("--fit-sample-size", type=int, default=500_000)
    args = parser.parse_args()
    build_features(args.data_dir, args.candidates, args.output, args.batch_rows, args.fit_sample_size)


if __name__ == "__main__":
    main()
