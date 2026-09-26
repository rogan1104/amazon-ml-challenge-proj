"""
Build labeled S1–candidate pairs for pair scoring (train/val only).

Primary negatives come from retrieved candidates not in ground truth — not random sampling.
Positives are every GT match (including matches missing from the candidate file).
"""
from __future__ import annotations

import csv
import hashlib
import io
import unittest
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet, Any

from analysis.retrieval.keys import name_prefix_key, sql_style_norm_name

try:
    from rapidfuzz.fuzz import ratio as _fuzz_ratio
except ImportError:  # pragma: no cover
    _fuzz_ratio = None

NAME_SIMILARITY_FLAG_THRESHOLD = 85  # rapidfuzz ratio 0–100


def candidate_source(entity_id: str) -> str:
    """Return ``S2`` or ``S3`` from entity id prefix."""
    eid = entity_id.strip()
    if eid.startswith("S2-"):
        return "S2"
    if eid.startswith("S3-"):
        return "S3"
    raise ValueError(f"Not an S2/S3 entity id: {entity_id!r}")


def match_count_bucket(n_matches: int) -> str:
    """GT match-count bucket for stratified splits (0, 1, 2, 3, 4+)."""
    if n_matches <= 0:
        return "0"
    if n_matches == 1:
        return "1"
    if n_matches == 2:
        return "2"
    if n_matches == 3:
        return "3"
    return "4+"


def normalize_country_label(country: str) -> str:
    c = (country or "").strip().upper()
    if c in ("INDIA", "IN"):
        return "IN"
    return c or "UNKNOWN"


@dataclass(frozen=True)
class EntityRecord:
    """Minimal fields for hard-negative flags (S1 or S2/S3)."""

    business_name: str = ""
    country: str = ""


def _clean_id_set(ids: AbstractSet[str] | None) -> set[str]:
    if not ids:
        return set()
    return {x.strip() for x in ids if x and str(x).strip()}


def cp_key(country: str, business_name: str, prefix_len: int = 5, min_prefix: int = 3) -> str | None:
    c = (country or "").strip()
    if not c:
        return None
    pfx = name_prefix_key(business_name, prefix_len)
    if len(pfx) < min_prefix:
        return None
    return f"cp:{c}|{pfx}"


def load_gt_by_s1(gt_path: Path | str) -> dict[str, set[str]]:
    """Stream ground-truth TSV into s1_id -> matched id set."""
    out: dict[str, set[str]] = {}
    path = Path(gt_path)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1 = row["source1_entity_id"].strip()
            raw = (row.get("matched_entity_ids") or "").strip()
            if not raw:
                out[s1] = set()
                continue
            out[s1] = {mid.strip() for mid in raw.split(",") if mid.strip()}
    return out


def stream_candidate_pairs(
    candidate_pairs_path: Path | str,
) -> Iterator[tuple[str, set[str]]]:
    """Yield (s1_id, candidate_ids) row by row from candidate_pairs.tsv."""
    path = Path(candidate_pairs_path)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1 = row["source1_entity_id"].strip()
            raw = (row.get("candidate_entity_ids") or "").strip()
            cands = {x.strip() for x in raw.split(",") if x.strip()} if raw else set()
            yield s1, cands


def _cheap_name_similarity_flag(name_a: str, name_b: str) -> bool:
    if _fuzz_ratio is None:
        return False
    na = sql_style_norm_name(name_a)
    nb = sql_style_norm_name(name_b)
    if not na or not nb:
        return False
    return _fuzz_ratio(na, nb) >= NAME_SIMILARITY_FLAG_THRESHOLD


def hard_negative_flags(
    s1: EntityRecord,
    candidate: EntityRecord,
) -> dict[str, bool]:
    """Retrieval-style evidence flags (no full feature vector)."""
    s1_country = normalize_country_label(s1.country)
    cand_country = normalize_country_label(candidate.country)
    s1_nn = sql_style_norm_name(s1.business_name)
    cand_nn = sql_style_norm_name(candidate.business_name)
    s1_cp = cp_key(s1.country, s1.business_name)
    cand_cp = cp_key(candidate.country, candidate.business_name)

    return {
        "hard_same_country": bool(s1_country != "UNKNOWN" and s1_country == cand_country),
        "hard_same_norm_name": bool(s1_nn and s1_nn == cand_nn),
        "hard_same_cp_key": bool(s1_cp and s1_cp == cand_cp),
        "hard_high_name_similarity": _cheap_name_similarity_flag(
            s1.business_name, candidate.business_name
        ),
    }


def _pair_row(
    s1_id: str,
    candidate_id: str,
    label: int,
    *,
    in_retrieval: bool,
    s1_records: Mapping[str, EntityRecord],
    candidate_records: Mapping[str, EntityRecord],
    pair_evidence: Mapping[tuple[str, str], Mapping[str, Any]] | None,
) -> dict[str, Any]:
    src = candidate_source(candidate_id)
    row: dict[str, Any] = {
        "s1_id": s1_id,
        "candidate_id": candidate_id,
        "candidate_source": src,
        "label": label,
        "in_retrieval": in_retrieval,
    }

    s1_rec = s1_records.get(s1_id, EntityRecord())
    cand_rec = candidate_records.get(candidate_id, EntityRecord())

    if label == 0:
        row.update(hard_negative_flags(s1_rec, cand_rec))
    else:
        row.update(
            {
                "hard_same_country": False,
                "hard_same_norm_name": False,
                "hard_same_cp_key": False,
                "hard_high_name_similarity": False,
            }
        )

    evidence: dict[str, Any] = {}
    if pair_evidence:
        evidence = dict(pair_evidence.get((s1_id, candidate_id), {}))
    row["retrieval_evidence"] = evidence
    return row


def build_labeled_pairs(
    gt_path: Path | str,
    candidate_pairs_path: Path | str,
    *,
    s1_records: Mapping[str, EntityRecord] | None = None,
    candidate_records: Mapping[str, EntityRecord] | None = None,
    pair_evidence: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    gt_by_s1: Mapping[str, AbstractSet[str]] | None = None,
    batch_size: int = 4096,
) -> Iterator[list[dict[str, Any]]]:
    """
    Yield batches of labeled pair dicts.

    Pass 1: positives from GT (all true matches).
    Pass 2: negatives from candidate_pairs (retrieved but not in GT).

    Does not materialize the full pair list; only GT and optional record maps are held.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    gt = dict(gt_by_s1) if gt_by_s1 is not None else load_gt_by_s1(gt_path)
    s1_recs = s1_records or {}
    cand_recs = candidate_records or {}

    batch: list[dict[str, Any]] = []

    def flush() -> Iterator[list[dict[str, Any]]]:
        nonlocal batch
        if batch:
            out = batch
            batch = []
            yield out

    # Pass 1: stream candidates — emit negatives; record which GT positives were retrieved.
    retrieved_positives: set[tuple[str, str]] = set()
    for s1_id, cands in stream_candidate_pairs(candidate_pairs_path):
        gt_set = _clean_id_set(gt.get(s1_id))
        for cid in sorted(cands):
            if cid in gt_set:
                retrieved_positives.add((s1_id, cid))
                continue
            batch.append(
                _pair_row(
                    s1_id,
                    cid,
                    0,
                    in_retrieval=True,
                    s1_records=s1_recs,
                    candidate_records=cand_recs,
                    pair_evidence=pair_evidence,
                )
            )
            if len(batch) >= batch_size:
                yield from flush()

    for chunk in flush():
        yield chunk

    # Pass 2: positives from GT (includes matches absent from candidate file).
    batch = []
    for s1_id, matches in gt.items():
        for mid in sorted(_clean_id_set(matches)):
            batch.append(
                _pair_row(
                    s1_id,
                    mid,
                    1,
                    in_retrieval=(s1_id, mid) in retrieved_positives,
                    s1_records=s1_recs,
                    candidate_records=cand_recs,
                    pair_evidence=pair_evidence,
                )
            )
            if len(batch) >= batch_size:
                yield from flush()

    yield from flush()


def split_by_s1(
    gt_by_s1: Mapping[str, AbstractSet[str]],
    country_by_s1: Mapping[str, str],
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[frozenset[str], frozenset[str]]:
    """
    Deterministic S1-level train/validation split.

    Stratifies by (country, GT match-count bucket). No S1 appears in both splits.
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")

    strata: dict[tuple[str, str], list[str]] = {}
    for s1_id in gt_by_s1.keys():
        country = normalize_country_label(country_by_s1.get(s1_id, ""))
        bucket = match_count_bucket(len(_clean_id_set(gt_by_s1.get(s1_id))))
        key = (country, bucket)
        strata.setdefault(key, []).append(s1_id)

    train: set[str] = set()
    val: set[str] = set()
    threshold = int(round(val_fraction * 10_000))

    for _key, ids in strata.items():
        for s1_id in sorted(ids):
            digest = hashlib.md5(f"{seed}:{s1_id}".encode("utf-8")).hexdigest()
            bucket_val = int(digest[: 8], 16) % 10_000
            if bucket_val < threshold:
                val.add(s1_id)
            else:
                train.add(s1_id)

    overlap = train & val
    if overlap:
        raise RuntimeError(f"split leakage: {len(overlap)} S1 in both splits")

    return frozenset(train), frozenset(val)


class TestPairDatasetBuilder(unittest.TestCase):
    def _write_tsv(self, path: Path, header: list[str], rows: list[list[str]]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(header)
            w.writerows(rows)

    def test_labels_and_sources(self) -> None:
        gt = io.StringIO(
            "source1_entity_id\tmatched_entity_ids\n"
            "S1-a\tS2-pos,S3-pos\n"
            "S1-b\t\n"
        )
        cand = io.StringIO(
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1-a\tS2-pos,S2-neg,S3-extra\n"
            "S1-b\tS3-only\n"
        )

        gt_path = Path(self._tmp_gt)
        cand_path = Path(self._tmp_cand)
        gt_path.write_text(gt.getvalue(), encoding="utf-8")
        cand_path.write_text(cand.getvalue(), encoding="utf-8")

        pairs = []
        for batch in build_labeled_pairs(gt_path, cand_path, batch_size=100):
            pairs.extend(batch)

        by_key = {(p["s1_id"], p["candidate_id"]): p for p in pairs}

        self.assertEqual(by_key[("S1-a", "S2-pos")]["label"], 1)
        self.assertEqual(by_key[("S1-a", "S2-pos")]["candidate_source"], "S2")
        self.assertTrue(by_key[("S1-a", "S2-pos")]["in_retrieval"])

        self.assertEqual(by_key[("S1-a", "S3-pos")]["label"], 1)
        self.assertFalse(by_key[("S1-a", "S3-pos")]["in_retrieval"])

        self.assertEqual(by_key[("S1-a", "S2-neg")]["label"], 0)
        self.assertEqual(by_key[("S1-a", "S3-extra")]["label"], 0)

        self.assertEqual(by_key[("S1-b", "S3-only")]["label"], 0)
        self.assertEqual(len([p for p in pairs if p["label"] == 1]), 2)
        self.assertEqual(len([p for p in pairs if p["label"] == 0]), 3)

    def test_hard_flags_on_negative(self) -> None:
        gt_path = Path(self._tmp_gt)
        cand_path = Path(self._tmp_cand)
        gt_path.write_text(
            "source1_entity_id\tmatched_entity_ids\nS1-x\tS2-good\n",
            encoding="utf-8",
        )
        cand_path.write_text(
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1-x\tS2-good,S2-hard\n",
            encoding="utf-8",
        )
        s1_recs = {"S1-x": EntityRecord(business_name="Acme Corp", country="US")}
        cand_recs = {
            "S2-hard": EntityRecord(business_name="Acme Corp", country="US"),
        }
        negs = []
        for batch in build_labeled_pairs(
            gt_path,
            cand_path,
            s1_records=s1_recs,
            candidate_records=cand_recs,
        ):
            negs.extend(p for p in batch if p["label"] == 0)
        self.assertEqual(len(negs), 1)
        self.assertTrue(negs[0]["hard_same_norm_name"])
        self.assertTrue(negs[0]["hard_same_country"])
        self.assertTrue(negs[0]["hard_same_cp_key"])

    def test_split_no_leakage(self) -> None:
        gt_by_s1 = {
            f"S1-{i}": set() if i % 5 == 0 else {f"S2-{i}"}
            for i in range(50)
        }
        country_by_s1 = {f"S1-{i}": "US" if i % 2 == 0 else "India" for i in range(50)}
        train, val = split_by_s1(gt_by_s1, country_by_s1, val_fraction=0.2, seed=7)
        self.assertFalse(train & val)
        self.assertEqual(len(train) + len(val), 50)
        train2, val2 = split_by_s1(gt_by_s1, country_by_s1, val_fraction=0.2, seed=7)
        self.assertEqual(train, train2)
        self.assertEqual(val, val2)

    def setUp(self) -> None:
        import tempfile

        self._td = tempfile.TemporaryDirectory()
        self._tmp_gt = str(Path(self._td.name) / "gt.tsv")
        self._tmp_cand = str(Path(self._td.name) / "cand.tsv")

    def tearDown(self) -> None:
        self._td.cleanup()


if __name__ == "__main__":
    unittest.main()
