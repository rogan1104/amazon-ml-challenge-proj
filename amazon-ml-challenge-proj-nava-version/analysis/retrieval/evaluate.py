"""Optional train-set recall evaluation against ground truth (run on PC only)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from analysis.config import TRAIN


def evaluate_recall(candidate_tsv: Path, gt_path: Path | None = None) -> dict:
    """
    Compute true-match pair recall from a candidate_pairs TSV vs train GT.
    Reads GT and candidates in streaming fashion; does not materialize all pairs.
    """
    gt_path = gt_path or TRAIN["GT"]

    # Load candidates: s1_id -> set(candidate_ids)
    candidates: dict[str, set[str]] = {}
    with open(candidate_tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1 = row["source1_entity_id"].strip()
            raw = row.get("candidate_entity_ids", "").strip()
            candidates[s1] = set(raw.split(",")) if raw else set()

    total_true = 0
    recalled = 0
    s2_true = s2_rec = s3_true = s3_rec = 0

    with open(gt_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1 = row["source1_entity_id"].strip()
            raw = row.get("matched_entity_ids", "").strip()
            if not raw:
                continue
            cands = candidates.get(s1, set())
            for mid in raw.split(","):
                mid = mid.strip()
                if not mid:
                    continue
                total_true += 1
                if mid in cands:
                    recalled += 1
                if mid.startswith("S2-"):
                    s2_true += 1
                    if mid in cands:
                        s2_rec += 1
                elif mid.startswith("S3-"):
                    s3_true += 1
                    if mid in cands:
                        s3_rec += 1

    return {
        "total_true_pairs": total_true,
        "recalled_pairs": recalled,
        "recall": round(recalled / total_true, 6) if total_true else 0,
        "s2_recall": round(s2_rec / s2_true, 6) if s2_true else 0,
        "s3_recall": round(s3_rec / s3_true, 6) if s3_true else 0,
        "s2_true_pairs": s2_true,
        "s3_true_pairs": s3_true,
    }


def merge_eval_into_metrics(metrics_path: Path, eval_result: dict) -> None:
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    data["evaluation"] = eval_result
    metrics_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
