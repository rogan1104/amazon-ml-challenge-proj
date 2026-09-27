"""Evaluate blocking-key pair recall on the full train ground truth.

This is the bucket-stage recall ceiling: a true pair is counted as recovered
if the two records share at least one blocking key. It does not run the
full 12.5M-row TF-IDF k-NN (hours / high RAM).
"""
from __future__ import annotations

import csv
import sys
import time
from collections import Counter
from pathlib import Path

import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from blocking import blocking_keys
from normalize import normalize_text

DEFAULT_S1 = Path(r"c:\Users\appal\Downloads\train_source1.tsv")
DEFAULT_S2 = Path(
    r"c:\Users\appal\Downloads\Telegram Desktop\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source2.tsv"
)
DEFAULT_S3 = Path(
    r"c:\Users\appal\Downloads\Telegram Desktop\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source3.tsv"
)
DEFAULT_GT = Path(
    r"c:\Users\appal\Downloads\Telegram Desktop\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_ground_truth.tsv"
)


def parse_ids(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for part in raw.split(","):
        eid = part.strip()
        if eid and eid not in seen:
            seen.add(eid)
            out.append(eid)
    return out


def load_needed_records(path: Path, wanted: set[str]) -> dict[str, tuple[str, str, str]]:
    recs: dict[str, tuple[str, str, str]] = {}
    if not wanted or not path.is_file():
        return recs
    remaining = set(wanted)
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            eid = row["entity_id"]
            if eid in remaining:
                recs[eid] = (row["country"], row["business_name"], row["business_address"])
                remaining.remove(eid)
                if not remaining:
                    break
    return recs


def keys_for(rec: tuple[str, str, str]):
    country, name, addr = rec
    name_n = normalize_text(name)
    addr_n = normalize_text(addr)
    return blocking_keys(country.strip(), name_n, addr_n, addr, name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multi-channel blocking recall on train ground truth")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing train_source{1,2,3}.tsv and train_ground_truth.tsv")
    parser.add_argument("--s1", type=Path, default=None)
    parser.add_argument("--s2", type=Path, default=None)
    parser.add_argument("--s3", type=Path, default=None)
    parser.add_argument("--gt", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate first N GT rows (0=all)")
    args = parser.parse_args()

    if args.data_dir:
        p_s1 = args.s1 or args.data_dir / "train_source1.tsv"
        p_s2 = args.s2 or args.data_dir / "train_source2.tsv"
        p_s3 = args.s3 or args.data_dir / "train_source3.tsv"
        p_gt = args.gt or args.data_dir / "train_ground_truth.tsv"
    else:
        root_data = Path(__file__).resolve().parents[1] / "dataset" / "train"
        p_s1 = args.s1 or (root_data / "train_source1.tsv" if (root_data / "train_source1.tsv").is_file() else DEFAULT_S1)
        p_s2 = args.s2 or (root_data / "train_source2.tsv" if (root_data / "train_source2.tsv").is_file() else DEFAULT_S2)
        p_s3 = args.s3 or (root_data / "train_source3.tsv" if (root_data / "train_source3.tsv").is_file() else DEFAULT_S3)
        p_gt = args.gt or (root_data / "train_ground_truth.tsv" if (root_data / "train_ground_truth.tsv").is_file() else DEFAULT_GT)

    t0 = time.time()
    print(f"[eval] reading ground truth from {p_gt} ...", flush=True)
    truth: dict[str, list[str]] = {}
    wanted_s2: set[str] = set()
    wanted_s3: set[str] = set()
    wanted_s1: set[str] = set()
    with p_gt.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        id_col = "source1_entity_id"
        match_col = "matched_entity_ids"
        count = 0
        for row in reader:
            s1 = row[id_col]
            ids = parse_ids(row[match_col])
            truth[s1] = ids
            wanted_s1.add(s1)
            for eid in ids:
                if eid.startswith("S2-"):
                    wanted_s2.add(eid)
                elif eid.startswith("S3-"):
                    wanted_s3.add(eid)
                else:
                    wanted_s2.add(eid)
                    wanted_s3.add(eid)
            count += 1
            if args.limit and count >= args.limit:
                break

    print(
        f"[eval] GT S1={len(truth):,}  needed S2={len(wanted_s2):,}  needed S3={len(wanted_s3):,}",
        flush=True,
    )
    print("[eval] loading source records for true-match IDs ...", flush=True)
    s1 = load_needed_records(p_s1, wanted_s1)
    s2 = load_needed_records(p_s2, wanted_s2)
    s3 = load_needed_records(p_s3, wanted_s3)
    right = {**s2, **s3}
    print(
        f"[eval] loaded S1={len(s1):,} S2={len(s2):,} S3={len(s3):,}  ({time.time()-t0:.1f}s)",
        flush=True,
    )

    print("[eval] scoring blocking-key overlap on true pairs ...", flush=True)
    total_true = 0
    recovered = 0
    missing_rec = 0
    entities_with = 0
    entities_full = 0
    hit_by_kind = Counter()
    miss_no_shared = 0

    s1_key_cache: dict[str, set] = {}
    right_key_cache: dict[str, set] = {}

    for i, (s1_id, true_ids) in enumerate(truth.items(), start=1):
        if not true_ids:
            continue
        entities_with += 1
        if s1_id not in s1:
            missing_rec += len(true_ids)
            total_true += len(true_ids)
            continue
        if s1_id not in s1_key_cache:
            s1_key_cache[s1_id] = set(keys_for(s1[s1_id]))
        left_keys = s1_key_cache[s1_id]
        hits = 0
        for cid in true_ids:
            total_true += 1
            rec = right.get(cid)
            if rec is None:
                missing_rec += 1
                continue
            if cid not in right_key_cache:
                right_key_cache[cid] = set(keys_for(rec))
            shared = left_keys & right_key_cache[cid]
            if shared:
                recovered += 1
                hits += 1
                kinds = {k[1] for k in shared}
                for kind in kinds:
                    hit_by_kind[kind] += 1
            else:
                miss_no_shared += 1
        if hits == len(true_ids):
            entities_full += 1
        if i % 200_000 == 0:
            print(f"[eval]   {i:,}/{len(truth):,} S1 scored", flush=True)

    pair_rec = recovered / total_true if total_true else 1.0
    entity_rec = entities_full / entities_with if entities_with else 1.0
    elapsed = time.time() - t0
    print()
    print("=== Train blocking-key recall (bucket overlap) ===")
    print(f"true_pairs              {total_true:,}")
    print(f"recovered_pairs         {recovered:,}")
    print(f"pair_recall             {pair_rec:.4f}  ({pair_rec*100:.2f}%)")
    print(f"entity_recall           {entity_rec:.4f}  ({entities_full:,}/{entities_with:,} fully covered)")
    print(f"missed (no shared key)  {miss_no_shared:,}")
    print(f"missing source records  {missing_rec:,}")
    print("pairs sharing key type (a pair can count in multiple types):")
    for kind, n in hit_by_kind.most_common():
        print(f"  {kind}: {n:,}  ({n/total_true:.2%} of true pairs)")
    print(f"elapsed_s               {elapsed:.1f}")
    print()
    print(
        "Note: this is blocking recall, not matcher F0.5. "
        "A true match that shares a key can still be dropped later by top-k TF-IDF."
    )


if __name__ == "__main__":
    main()
