"""
Baseline E key-family recall / candidate-cost decomposition (analysis only).

Deterministic sample: first N S1 rows in train_source1.tsv file order.

Usage (PC):
  python -m analysis.retrieval.analyze_baseline_key_families --sample 10000
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Literal

from analysis.config import REPORTS, TRAIN
from analysis.retrieval.build import _index_path
from analysis.retrieval.config import ChannelName, RetrievalConfig
from analysis.retrieval.index import InvertedIndex
from analysis.retrieval.keys import baseline_keys

Target = Literal["S2", "S3"]

CONDITIONS: dict[str, frozenset[str]] = {
    "bn": frozenset({"bn"}),
    "ba": frozenset({"ba"}),
    "cp": frozenset({"cp"}),
    "bn_ba": frozenset({"bn", "ba"}),
    "bn_cp": frozenset({"bn", "cp"}),
    "ba_cp": frozenset({"ba", "cp"}),
    "bn_ba_cp": frozenset({"bn", "ba", "cp"}),
}

_FAMILY_PREFIX = {"bn": "bn:", "ba": "ba:", "cp": "cp:"}


def _filter_keys(keys: list[str], families: frozenset[str]) -> list[str]:
    prefixes = tuple(_FAMILY_PREFIX[f] for f in families)
    return [k for k in keys if k.startswith(prefixes)]


def load_first_n_s1(path: Path, n: int) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
            if len(rows) >= n:
                break
    return rows


def extract_baseline_keys(row: dict, prefix_len: int = 5, min_prefix: int = 3) -> list[str]:
    return baseline_keys(
        row.get("business_name", ""),
        row.get("business_address", ""),
        row.get("country", ""),
        prefix_len=prefix_len,
        min_prefix=min_prefix,
    )


def load_gt_for_sample(sample_ids: set[str], gt_path: Path) -> dict[str, dict]:
    """s1_id -> {S2: set, S3: set, all: set}"""
    gt: dict[str, dict] = {
        s1: {"S2": set(), "S3": set(), "all": set()} for s1 in sample_ids
    }
    with open(gt_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1 = row["source1_entity_id"].strip()
            if s1 not in sample_ids:
                continue
            raw = row.get("matched_entity_ids", "").strip()
            if not raw:
                continue
            for mid in raw.split(","):
                mid = mid.strip()
                if not mid:
                    continue
                gt[s1]["all"].add(mid)
                if mid.startswith("S2-"):
                    gt[s1]["S2"].add(mid)
                elif mid.startswith("S3-"):
                    gt[s1]["S3"].add(mid)
    return gt


def _percentile(sorted_vals: list[int], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def retrieve_condition(
    index: InvertedIndex,
    queries: list[tuple[str, list[str]]],
    target: Target,
) -> dict[str, set[str]]:
    return index.lookup_keys_exact_batch(
        ChannelName.BASELINE, target, queries  # type: ignore[arg-type]
    )


def evaluate_target(
    cands_by_s1: dict[str, set[str]],
    gt: dict[str, dict],
    sample_ids: list[str],
    target: Target,
) -> dict:
    true_pairs = 0
    recalled_pairs = 0
    cand_counts: list[int] = []
    tp_pairs = 0
    total_cand_pairs = 0

    for s1 in sample_ids:
        true_set = gt[s1][target]
        cand_set = cands_by_s1.get(s1, set())
        n_true = len(true_set)
        n_hit = len(true_set & cand_set)
        n_cand = len(cand_set)

        true_pairs += n_true
        recalled_pairs += n_hit
        tp_pairs += n_hit
        total_cand_pairs += n_cand
        cand_counts.append(n_cand)

    cand_counts_sorted = sorted(cand_counts)
    missed = true_pairs - recalled_pairs

    return {
        "pair_recall": round(recalled_pairs / true_pairs, 6) if true_pairs else 0.0,
        "gt_positive_pairs": true_pairs,
        "gt_positive_recalled": recalled_pairs,
        "gt_positive_missed": missed,
        "avg_candidates_per_s1": round(statistics.mean(cand_counts), 4) if cand_counts else 0,
        "median_candidates_per_s1": statistics.median(cand_counts) if cand_counts else 0,
        "p95_candidates_per_s1": round(_percentile(cand_counts_sorted, 95), 2),
        "p99_candidates_per_s1": round(_percentile(cand_counts_sorted, 99), 2),
        "max_candidates_per_s1": max(cand_counts) if cand_counts else 0,
        "total_candidate_pairs": total_cand_pairs,
        "pair_precision": round(tp_pairs / total_cand_pairs, 6) if total_cand_pairs else 0.0,
    }


def evaluate_subset(
    cands_by_s1: dict[str, set[str]],
    gt: dict[str, dict],
    s1_ids: list[str],
    target: Target,
) -> dict:
    return evaluate_target(cands_by_s1, gt, s1_ids, target)


def run_analysis(sample: int, index_dir: Path, output_dir: Path) -> dict:
    cfg = RetrievalConfig(index_dir=index_dir)
    index_path = _index_path(cfg)
    s1_path = TRAIN["S1"]
    gt_path = TRAIN["GT"]

    if not index_path.is_file():
        raise FileNotFoundError(f"Index missing: {index_path}")
    if not s1_path.is_file():
        raise FileNotFoundError(f"S1 missing: {s1_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"GT missing: {gt_path}")

    t_load = time.perf_counter()
    rows = load_first_n_s1(s1_path, sample)
    sample_ids = [r["entity_id"].strip() for r in rows if r.get("entity_id", "").strip()]
    sample_id_set = set(sample_ids)
    gt = load_gt_for_sample(sample_id_set, gt_path)

    s1_country: dict[str, str] = {
        r["entity_id"].strip(): (r.get("country") or "").strip().upper()
        for r in rows
        if r.get("entity_id", "").strip()
    }
    singleton_ids = [s1 for s1 in sample_ids if len(gt[s1]["all"]) == 0]
    nonsingleton_ids = [s1 for s1 in sample_ids if len(gt[s1]["all"]) > 0]
    us_ids = [s1 for s1 in sample_ids if s1_country.get(s1) == "US"]
    in_ids = [s1 for s1 in sample_ids if s1_country.get(s1) == "IN"]

    precomputed: list[tuple[str, list[str]]] = []
    for row in rows:
        s1 = row.get("entity_id", "").strip()
        if not s1:
            continue
        precomputed.append((s1, extract_baseline_keys(row)))

    load_sec = round(time.perf_counter() - t_load, 3)

    report: dict = {
        "sample_method": f"first_{sample}_s1_rows_in_train_source1_tsv_file_order",
        "sample_s1_count": len(sample_ids),
        "load_sec": load_sec,
        "index_path": str(index_path),
        "conditions": {},
        "summary_table": [],
        "interpretation_questions": {},
    }

    with InvertedIndex(index_path, read_only=True) as index:
        for cond_name, families in CONDITIONS.items():
            print(f"[key-family] condition={cond_name} ...", flush=True)
            t0 = time.perf_counter()
            queries = [(s1, _filter_keys(keys, families)) for s1, keys in precomputed]

            by_target: dict[str, dict] = {}
            cands_s2: dict[str, set[str]] = {}
            cands_s3: dict[str, set[str]] = {}

            for target in ("S2", "S3"):
                cands = retrieve_condition(index, queries, target)  # type: ignore[arg-type]
                if target == "S2":
                    cands_s2 = cands
                else:
                    cands_s3 = cands
                by_target[target] = evaluate_target(cands, gt, sample_ids, target)  # type: ignore[arg-type]

            runtime_sec = round(time.perf_counter() - t0, 3)

            combined_counts = [
                len(cands_s2.get(s1, set()) | cands_s3.get(s1, set()))
                for s1 in sample_ids
            ]
            combined_sorted = sorted(combined_counts)

            cond_report = {
                "runtime_sec": runtime_sec,
                "S2": by_target["S2"],
                "S3": by_target["S3"],
                "combined_candidates_per_s1": {
                    "avg": round(statistics.mean(combined_counts), 4) if combined_counts else 0,
                    "median": statistics.median(combined_counts) if combined_counts else 0,
                    "p95": round(_percentile(combined_sorted, 95), 2),
                    "p99": round(_percentile(combined_sorted, 99), 2),
                    "max": max(combined_counts) if combined_counts else 0,
                    "total": sum(combined_counts),
                },
                "recall_by_country": {
                    "US": {
                        "S2": evaluate_subset(cands_s2, gt, us_ids, "S2"),
                        "S3": evaluate_subset(cands_s3, gt, us_ids, "S3"),
                    },
                    "IN": {
                        "S2": evaluate_subset(cands_s2, gt, in_ids, "S2"),
                        "S3": evaluate_subset(cands_s3, gt, in_ids, "S3"),
                    },
                },
                "recall_by_singleton": {
                    "singleton": {
                        "S2": evaluate_subset(cands_s2, gt, singleton_ids, "S2"),
                        "S3": evaluate_subset(cands_s3, gt, singleton_ids, "S3"),
                    },
                    "non_singleton": {
                        "S2": evaluate_subset(cands_s2, gt, nonsingleton_ids, "S2"),
                        "S3": evaluate_subset(cands_s3, gt, nonsingleton_ids, "S3"),
                    },
                },
            }
            report["conditions"][cond_name] = cond_report
            report["summary_table"].append({
                "condition": cond_name,
                "S2_recall": by_target["S2"]["pair_recall"],
                "S3_recall": by_target["S3"]["pair_recall"],
                "avg_candidates_combined": cond_report["combined_candidates_per_s1"]["avg"],
                "p95_candidates_combined": cond_report["combined_candidates_per_s1"]["p95"],
                "runtime_sec": runtime_sec,
            })
            print(
                f"  S2 recall={by_target['S2']['pair_recall']:.4f} "
                f"S3={by_target['S3']['pair_recall']:.4f} "
                f"avg_cand={cond_report['combined_candidates_per_s1']['avg']:.1f} "
                f"runtime={runtime_sec}s",
                flush=True,
            )

    # Interpretation helpers A–E
    c = report["conditions"]
    bn_ba = c["bn_ba"]
    cp_only = c["cp"]
    full = c["bn_ba_cp"]
    report["interpretation_questions"] = {
        "A_cp_recall_alone": {
            "S2": cp_only["S2"]["pair_recall"],
            "S3": cp_only["S3"]["pair_recall"],
        },
        "B_bn_or_ba_recall": {
            "S2": bn_ba["S2"]["pair_recall"],
            "S3": bn_ba["S3"]["pair_recall"],
        },
        "C_incremental_recall_adding_cp_to_bn_ba": {
            "S2": round(full["S2"]["pair_recall"] - bn_ba["S2"]["pair_recall"], 6),
            "S3": round(full["S3"]["pair_recall"] - bn_ba["S3"]["pair_recall"], 6),
        },
        "D_full_vs_cached_baseline_E_note": (
            "Compare bn_ba_cp S2/S3 recall to cached phase10 ~37.5% per-target; "
            "sample recall will differ from full train."
        ),
        "E_cp_cost_share": {
            "cp_only_avg_candidates": c["cp"]["combined_candidates_per_s1"]["avg"],
            "full_avg_candidates": full["combined_candidates_per_s1"]["avg"],
            "bn_ba_avg_candidates": bn_ba["combined_candidates_per_s1"]["avg"],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"baseline_key_families_{sample}.json"
    md_path = output_dir / f"baseline_key_families_{sample}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_format_markdown(report), encoding="utf-8")
    report["output_json"] = str(json_path)
    report["output_md"] = str(md_path)
    return report


def _format_markdown(report: dict) -> str:
    lines = [
        "# Baseline key-family decomposition",
        "",
        f"**Sample:** {report['sample_method']} (n={report['sample_s1_count']})",
        "",
        "## Summary",
        "",
        "| condition | S2 recall | S3 recall | avg candidates | p95 candidates | runtime (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary_table"]:
        lines.append(
            f"| {row['condition']} | {row['S2_recall']:.4f} | {row['S3_recall']:.4f} | "
            f"{row['avg_candidates_combined']:.2f} | {row['p95_candidates_combined']:.1f} | "
            f"{row['runtime_sec']:.2f} |"
        )
    lines.extend(["", "## Interpretation (A–E)", ""])
    iq = report["interpretation_questions"]
    lines.append(f"- **A (cp alone):** S2={iq['A_cp_recall_alone']['S2']}, S3={iq['A_cp_recall_alone']['S3']}")
    lines.append(f"- **B (bn OR ba):** S2={iq['B_bn_or_ba_recall']['S2']}, S3={iq['B_bn_or_ba_recall']['S3']}")
    lines.append(
        f"- **C (incremental cp on bn+ba):** S2=+{iq['C_incremental_recall_adding_cp_to_bn_ba']['S2']}, "
        f"S3=+{iq['C_incremental_recall_adding_cp_to_bn_ba']['S3']}"
    )
    lines.append(f"- **D:** {iq['D_full_vs_cached_baseline_E_note']}")
    lines.append(f"- **E (cost):** {json.dumps(iq['E_cp_cost_share'])}")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Baseline key-family recall/cost analysis")
    p.add_argument("--sample", type=int, default=10_000)
    p.add_argument("--index-dir", default="reports/cache/retrieval_index")
    p.add_argument("--output-dir", default=str(REPORTS / "retrieval"))
    args = p.parse_args()

    report = run_analysis(
        sample=args.sample,
        index_dir=Path(args.index_dir),
        output_dir=Path(args.output_dir),
    )
    print(f"\nWrote {report['output_json']}")
    print(f"Wrote {report['output_md']}")
    print("\nSummary table:")
    for row in report["summary_table"]:
        print(row)


if __name__ == "__main__":
    main()
