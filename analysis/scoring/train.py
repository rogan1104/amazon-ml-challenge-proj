"""
Train LightGBM pair scorer on retrieved candidates + GT positives.

Usage (PC, after candidate_pairs.tsv exists):
  python -m analysis.scoring.train \\
    --gt dataset/train/train_ground_truth.tsv \\
    --candidates reports/retrieval/candidate_pairs_train_all.tsv \\
    --s1 dataset/train/train_source1.tsv \\
    --s2 dataset/train/train_source2.tsv \\
    --s3 dataset/train/train_source3.tsv \\
    --output-dir reports/scoring/lgbm_baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from analysis.scoring.evaluate import (
    DEFAULT_THRESHOLDS,
    evaluate_threshold_sweep,
    score_pairs_with_model,
)
from analysis.scoring.features import ScoringEntity
from analysis.scoring.model import (
    DEFAULT_LGBM_PARAMS,
    EntityStore,
    MissingLightGBMDependency,
    PairLGBMModel,
    feature_matrix_from_pairs,
    is_retrieved_scoring_pair,
    is_training_pair,
    require_lightgbm,
    train_lightgbm_classifier,
)
from analysis.scoring.pairs import (
    build_labeled_pairs,
    load_gt_by_s1,
    split_by_s1,
    stream_candidate_pairs,
)


def load_country_by_s1(s1_path: Path | str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(s1_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row.get("entity_id", "").strip()
            if eid:
                out[eid] = (row.get("country") or "").strip()
    return out


def load_entity_store(
    *,
    s1_path: Path | str | None = None,
    s2_path: Path | str | None = None,
    s3_path: Path | str | None = None,
    entity_ids: set[str] | None = None,
) -> EntityStore:
    """Stream source TSVs into EntityStore (optionally filter to entity_ids)."""
    store = EntityStore()

    def _ingest(path: Path | str) -> None:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                eid = row.get("entity_id", "").strip()
                if not eid:
                    continue
                if entity_ids is not None and eid not in entity_ids:
                    continue
                store.entities[eid] = ScoringEntity(
                    business_name=row.get("business_name", "") or "",
                    business_address=row.get("business_address", "") or "",
                    country=row.get("country", "") or "",
                )

    if s1_path:
        _ingest(s1_path)
    if s2_path:
        _ingest(s2_path)
    if s3_path:
        _ingest(s3_path)
    return store


def candidates_by_s1_for_ids(
    candidate_pairs_path: Path | str,
    s1_ids: set[str],
) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {s1: set() for s1 in s1_ids}
    for s1, cands in stream_candidate_pairs(candidate_pairs_path):
        if s1 in s1_ids:
            out[s1] = set(cands)
    return out


def iter_filtered_pair_batches(
    gt_path: Path | str,
    candidate_pairs_path: Path | str,
    *,
    s1_ids: set[str] | None = None,
    training: bool = False,
    scoring_retrieved_only: bool = False,
    batch_size: int = 4096,
    gt_by_s1: dict[str, set[str]] | None = None,
) -> Iterator[list[dict[str, Any]]]:
    gt = gt_by_s1 if gt_by_s1 is not None else load_gt_by_s1(gt_path)
    for batch in build_labeled_pairs(
        gt_path,
        candidate_pairs_path,
        gt_by_s1=gt,
        batch_size=batch_size,
    ):
        filtered: list[dict[str, Any]] = []
        for p in batch:
            if s1_ids is not None and p["s1_id"] not in s1_ids:
                continue
            if training and not is_training_pair(p):
                continue
            if scoring_retrieved_only and not is_retrieved_scoring_pair(p):
                continue
            filtered.append(p)
        if filtered:
            yield filtered


def collect_pairs(
    gt_path: Path | str,
    candidate_pairs_path: Path | str,
    *,
    s1_ids: set[str] | None = None,
    training: bool = False,
    scoring_retrieved_only: bool = False,
    max_pairs: int | None = None,
    gt_by_s1: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for batch in iter_filtered_pair_batches(
        gt_path,
        candidate_pairs_path,
        s1_ids=s1_ids,
        training=training,
        scoring_retrieved_only=scoring_retrieved_only,
        gt_by_s1=gt_by_s1,
    ):
        for p in batch:
            out.append(p)
            if max_pairs is not None and len(out) >= max_pairs:
                return out
    return out


def run_training_pipeline(
    *,
    gt_path: Path,
    candidate_pairs_path: Path,
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    output_dir: Path,
    val_fraction: float = 0.2,
    split_seed: int = 42,
    max_train_pairs: int | None = None,
    max_val_pairs: int | None = None,
    lgbm_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    require_lightgbm()

    gt_by_s1 = load_gt_by_s1(gt_path)
    country_by_s1 = load_country_by_s1(s1_path)
    train_s1, val_s1 = split_by_s1(
        gt_by_s1, country_by_s1, val_fraction=val_fraction, seed=split_seed
    )

    train_pairs = collect_pairs(
        gt_path,
        candidate_pairs_path,
        s1_ids=set(train_s1),
        training=True,
        max_pairs=max_train_pairs,
        gt_by_s1=gt_by_s1,
    )
    val_scoring_pairs = collect_pairs(
        gt_path,
        candidate_pairs_path,
        s1_ids=set(val_s1),
        scoring_retrieved_only=True,
        max_pairs=max_val_pairs,
        gt_by_s1=gt_by_s1,
    )

    entity_ids: set[str] = set()
    for p in train_pairs + val_scoring_pairs:
        entity_ids.add(p["s1_id"])
        entity_ids.add(p["candidate_id"])

    store = load_entity_store(
        s1_path=s1_path,
        s2_path=s2_path,
        s3_path=s3_path,
        entity_ids=entity_ids,
    )

    X_train, y_train = feature_matrix_from_pairs(train_pairs, store)
    fitted_params = {**DEFAULT_LGBM_PARAMS, **(lgbm_params or {})}
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    if n_pos > 0:
        fitted_params["scale_pos_weight"] = n_neg / n_pos

    model = train_lightgbm_classifier(
        X_train, y_train, params=lgbm_params, scale_pos_weight=fitted_params.get("scale_pos_weight")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "pair_lgbm.txt"
    model.save(model_path)

    scored_val = score_pairs_with_model(val_scoring_pairs, store, model)
    val_gt = {s1: gt_by_s1.get(s1, set()) for s1 in val_s1}
    sweep = evaluate_threshold_sweep(
        scored_val,
        val_gt,
        sorted(val_s1),
        country_by_s1=country_by_s1,
        thresholds=DEFAULT_THRESHOLDS,
    )

    report = {
        "model_path": str(model_path),
        "lgbm_params": fitted_params,
        "feature_importance": model.feature_importance(),
        "split": {
            "train_s1": len(train_s1),
            "val_s1": len(val_s1),
            "train_pairs": len(train_pairs),
            "val_scoring_pairs": len(val_scoring_pairs),
        },
        "validation": sweep,
    }
    report_path = output_dir / "train_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def _format_threshold_table(threshold_results: list[dict[str, Any]]) -> str:
    lines = [
        "threshold  macro_f05  macro_P  macro_R  retrieved_pos_cov  n_pred_matches",
    ]
    for row in threshold_results:
        lines.append(
            f"{row['threshold']:>9.2f}  "
            f"{row['macro_f05']:>9.4f}  "
            f"{row['macro_precision']:>7.4f}  "
            f"{row['macro_recall']:>7.4f}  "
            f"{row['retrieved_positive_coverage']:>17.4f}  "
            f"{row['n_predicted_match_pairs']:>14d}"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Train LightGBM pair scoring model")
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--s1", type=Path, required=True)
    p.add_argument("--s2", type=Path, required=True)
    p.add_argument("--s3", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("reports/scoring/lgbm_baseline"))
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--max-train-pairs", type=int, default=None)
    p.add_argument("--max-val-pairs", type=int, default=None)
    args = p.parse_args()

    try:
        report = run_training_pipeline(
            gt_path=args.gt,
            candidate_pairs_path=args.candidates,
            s1_path=args.s1,
            s2_path=args.s2,
            s3_path=args.s3,
            output_dir=args.output_dir,
            val_fraction=args.val_fraction,
            split_seed=args.split_seed,
            max_train_pairs=args.max_train_pairs,
            max_val_pairs=args.max_val_pairs,
        )
    except MissingLightGBMDependency as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc

    print(json.dumps(report["lgbm_params"], indent=2))
    print("\nFeature importance (gain, normalized):")
    for name, val in sorted(
        report["feature_importance"].items(), key=lambda x: -x[1]
    ):
        print(f"  {name}: {val}")
    print("\nThreshold sweep (validation):")
    print(_format_threshold_table(report["validation"]["threshold_results"]))
    unret = report["validation"]["unretrieved_gt_positives"]
    print(
        f"\nUnretrieved GT positives (val): {unret['unretrieved_gt_positive_pairs']} "
        f"/ {unret['gt_positive_pairs']}"
    )
    print(f"\nWrote {report['report_path']}")


class TestTrainPipelineSmoke(unittest.TestCase):
    def test_end_to_end_synthetic(self) -> None:
        try:
            require_lightgbm()
        except MissingLightGBMDependency:
            self.skipTest("LightGBM not available")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gt_path = root / "gt.tsv"
            cand_path = root / "cand.tsv"
            s1_path = root / "s1.tsv"
            s2_path = root / "s2.tsv"
            s3_path = root / "s3.tsv"

            gt_path.write_text(
                "source1_entity_id\tmatched_entity_ids\n"
                "S1-a\tS2-good\n"
                "S1-b\t\n"
                "S1-c\tS3-good\n"
                "S1-d\tS2-x\n",
                encoding="utf-8",
            )
            cand_path.write_text(
                "source1_entity_id\tcandidate_entity_ids\n"
                "S1-a\tS2-good,S2-bad\n"
                "S1-b\tS3-spurious\n"
                "S1-c\tS3-good,S3-bad\n"
                "S1-d\tS2-other\n",  # S2-x not retrieved (unretrieved positive)
                encoding="utf-8",
            )
            s1_path.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S1-a\tAcme Corp\t1 Main St\tUS\n"
                "S1-b\tLonely\t2 Oak\tUS\n"
                "S1-c\tBeta LLC\t3 Pine\tIndia\n"
                "S1-d\tDelta\t4 Elm\tUS\n",
                encoding="utf-8",
            )
            s2_path.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S2-good\tAcme Corp\t1 Main St\tUS\n"
                "S2-bad\tOther Co\t9 Road\tUS\n"
                "S2-other\tWrong\t8 Lane\tUS\n"
                "S2-x\tDelta\t4 Elm\tUS\n",
                encoding="utf-8",
            )
            s3_path.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S3-good\tBeta LLC\t3 Pine\tIndia\n"
                "S3-bad\tNoise Inc\t0 Zero\tIndia\n"
                "S3-spurious\tFake\t1 Fake\tUS\n",
                encoding="utf-8",
            )

            out = root / "model_out"
            report = run_training_pipeline(
                gt_path=gt_path,
                candidate_pairs_path=cand_path,
                s1_path=s1_path,
                s2_path=s2_path,
                s3_path=s3_path,
                output_dir=out,
                val_fraction=0.25,
                split_seed=99,
                lgbm_params={"n_estimators": 30, "min_child_samples": 1},
            )

            self.assertTrue((out / "pair_lgbm.txt").is_file())
            self.assertIn("feature_importance", report)
            self.assertIn("threshold_results", report["validation"])
            all_positives = collect_pairs(
                gt_path, cand_path, gt_by_s1=load_gt_by_s1(gt_path)
            )
            unretrieved_pos = [
                p for p in all_positives if p["label"] == 1 and not p["in_retrieval"]
            ]
            self.assertEqual(len(unretrieved_pos), 1)
            self.assertEqual(unretrieved_pos[0]["s1_id"], "S1-d")

            train_pairs = collect_pairs(
                gt_path, cand_path, training=True, gt_by_s1=load_gt_by_s1(gt_path)
            )
            unretrieved_in_train = [
                p for p in train_pairs if p["label"] == 1 and not p["in_retrieval"]
            ]
            self.assertEqual(len(unretrieved_in_train), 0)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        unittest.main()
    else:
        main()
