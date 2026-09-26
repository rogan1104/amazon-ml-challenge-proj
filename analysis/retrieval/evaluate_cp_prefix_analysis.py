"""
Evaluate CP prefix lengths 3–8 using the analysis-only cp prefix index.

Deterministic sample: first 500 S1 rows in train_source1.tsv file order.

Usage (PC, after build_cp_prefix_analysis_index):
  python -m analysis.retrieval.evaluate_cp_prefix_analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Literal

from analysis.config import CACHE, REPORTS, TRAIN
from analysis.retrieval.keys import _safe_str, name_prefix_key

Target = Literal["S2", "S3"]

PREFIX_LENGTHS = (3, 4, 5, 6, 7, 8)
MIN_PREFIX_LEN = 3
SAMPLE_DEFAULT = 500
DEFAULT_DB = CACHE / "cp_prefix_analysis" / "index_cp_prefixes.sqlite"


def load_first_n_s1(path: Path, n: int) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
            if len(rows) >= n:
                break
    return rows


def cp_keys_for_row(row: dict) -> dict[int, list[str]]:
    """Return cp{n}: keys per prefix length for one S1 row."""
    c = _safe_str(row.get("country", "")).strip()
    name = row.get("business_name", "")
    by_len: dict[int, list[str]] = {}
    if not c:
        return by_len
    for n in PREFIX_LENGTHS:
        pfx = name_prefix_key(name, n)
        if len(pfx) >= MIN_PREFIX_LEN:
            by_len[n] = [f"cp{n}:{c}|{pfx}"]
    return by_len


def load_gt_for_sample(sample_ids: set[str], gt_path: Path) -> dict[str, dict]:
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


def lookup_keys_batch(
    conn: sqlite3.Connection,
    target: Target,
    queries: list[tuple[str, list[str]]],
) -> dict[str, set[str]]:
    """Key-centric batch lookup against analysis postings table."""
    out: dict[str, set[str]] = {qid: set() for qid, _ in queries}
    flat: list[tuple[str, str]] = []
    for qid, keys in queries:
        for k in keys:
            flat.append((qid, k))
    if not flat:
        return out

    conn.execute("DROP TABLE IF EXISTS batch_keys")
    conn.execute("CREATE TEMP TABLE batch_keys (qid TEXT NOT NULL, key TEXT NOT NULL)")
    conn.executemany("INSERT INTO batch_keys (qid, key) VALUES (?, ?)", flat)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_batch_keys_key ON batch_keys (key)"
    )

    cur = conn.execute(
        """
        SELECT b.qid, p.entity_id
        FROM batch_keys b
        INNER JOIN postings p ON p.target = ? AND p.key = b.key
        """,
        (target,),
    )
    for qid, eid in cur:
        out[qid].add(eid)
    conn.execute("DROP TABLE IF EXISTS batch_keys")
    return out


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

    s = sorted(cand_counts)
    missed = true_pairs - recalled_pairs

    return {
        "pair_recall": round(recalled_pairs / true_pairs, 6) if true_pairs else 0.0,
        "gt_positive_pairs": true_pairs,
        "gt_positive_pairs_recalled": recalled_pairs,
        "gt_positive_pairs_missed": missed,
        "avg_candidates_per_s1": round(statistics.mean(cand_counts), 4) if cand_counts else 0,
        "median_candidates_per_s1": statistics.median(cand_counts) if cand_counts else 0,
        "p95_candidates_per_s1": round(_percentile(s, 95), 2),
        "p99_candidates_per_s1": round(_percentile(s, 99), 2),
        "max_candidates_per_s1": max(cand_counts) if cand_counts else 0,
        "total_candidate_pairs": total_cand_pairs,
        "pair_precision": round(tp_pairs / total_cand_pairs, 6) if total_cand_pairs else 0.0,
    }


def subset_pair_recall(
    cands_by_s1: dict[str, set[str]],
    gt: dict[str, dict],
    s1_ids: list[str],
    target: Target,
) -> float:
    if not s1_ids:
        return 0.0
    return evaluate_target(cands_by_s1, gt, s1_ids, target)["pair_recall"]


def run_evaluation(
    sample: int,
    db_path: Path,
    output_dir: Path,
) -> dict:
    s1_path = TRAIN["S1"]
    gt_path = TRAIN["GT"]

    if not db_path.is_file():
        raise FileNotFoundError(
            f"Analysis index missing: {db_path}. "
            "Run: python -m analysis.retrieval.build_cp_prefix_analysis_index"
        )
    if not s1_path.is_file():
        raise FileNotFoundError(s1_path)
    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)

    rows = load_first_n_s1(s1_path, sample)
    sample_ids = [r["entity_id"].strip() for r in rows if r.get("entity_id", "").strip()]
    sample_id_set = set(sample_ids)
    gt = load_gt_for_sample(sample_id_set, gt_path)

    s1_country = {
        r["entity_id"].strip(): (r.get("country") or "").strip().upper()
        for r in rows
        if r.get("entity_id", "").strip()
    }
    us_ids = [s for s in sample_ids if s1_country.get(s) == "US"]
    in_ids = [s for s in sample_ids if s1_country.get(s) == "IN"]
    singleton_ids = [s for s in sample_ids if len(gt[s]["all"]) == 0]
    nonsingleton_ids = [s for s in sample_ids if len(gt[s]["all"]) > 0]

    keys_by_len: dict[int, list[tuple[str, list[str]]]] = {}
    for n in PREFIX_LENGTHS:
        queries: list[tuple[str, list[str]]] = []
        for r in rows:
            eid = r.get("entity_id", "").strip()
            if not eid:
                continue
            by_len = cp_keys_for_row(r)
            queries.append((eid, by_len.get(n, [])))
        keys_by_len[n] = queries

    report: dict = {
        "sample_method": f"first_{sample}_s1_rows_in_train_source1_tsv_file_order",
        "sample_s1_count": len(sample_ids),
        "analysis_index_path": str(db_path),
        "by_prefix_len": {},
        "summary_table": [],
    }

    conn = sqlite3.connect(db_path)
    try:
        for n in PREFIX_LENGTHS:
            queries = keys_by_len[n]
            plen_report: dict = {"prefix_len": n, "targets": {}}

            for target in ("S2", "S3"):
                print(f"[cp-prefix eval] len={n} target={target} ...", flush=True)
                t0 = time.perf_counter()
                cands = lookup_keys_batch(conn, target, queries)  # type: ignore[arg-type]
                runtime_sec = round(time.perf_counter() - t0, 3)

                metrics = evaluate_target(cands, gt, sample_ids, target)  # type: ignore[arg-type]
                plen_report["targets"][target] = {
                    **metrics,
                    "recall_US": subset_pair_recall(cands, gt, us_ids, target),  # type: ignore[arg-type]
                    "recall_IN": subset_pair_recall(cands, gt, in_ids, target),  # type: ignore[arg-type]
                    "recall_singleton": subset_pair_recall(
                        cands, gt, singleton_ids, target  # type: ignore[arg-type]
                    ),
                    "recall_non_singleton": subset_pair_recall(
                        cands, gt, nonsingleton_ids, target  # type: ignore[arg-type]
                    ),
                    "runtime_sec": runtime_sec,
                }
                print(
                    f"  recall={metrics['pair_recall']:.4f} "
                    f"precision={metrics['pair_precision']:.4f} "
                    f"avg_cand={metrics['avg_candidates_per_s1']:.1f} "
                    f"runtime={runtime_sec}s",
                    flush=True,
                )

            s2 = plen_report["targets"]["S2"]
            s3 = plen_report["targets"]["S3"]
            report["summary_table"].append({
                "prefix_len": n,
                "S2_recall": s2["pair_recall"],
                "S3_recall": s3["pair_recall"],
                "S2_precision": s2["pair_precision"],
                "S3_precision": s3["pair_precision"],
                "avg_candidates_S2": s2["avg_candidates_per_s1"],
                "avg_candidates_S3": s3["avg_candidates_per_s1"],
                "p95_candidates_S2": s2["p95_candidates_per_s1"],
                "p95_candidates_S3": s3["p95_candidates_per_s1"],
                "runtime_sec_S2": s2["runtime_sec"],
                "runtime_sec_S3": s3["runtime_sec"],
            })
            report["by_prefix_len"][str(n)] = plen_report
    finally:
        conn.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"cp_prefix_analysis_eval_{sample}.json"
    md_path = output_dir / f"cp_prefix_analysis_eval_{sample}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_format_md(report), encoding="utf-8")
    report["output_json"] = str(json_path)
    report["output_md"] = str(md_path)
    _print_comparison_table(report)
    return report


def _format_md(report: dict) -> str:
    lines = [
        "# CP prefix analysis evaluation",
        "",
        f"**Sample:** {report['sample_method']} (n={report['sample_s1_count']})",
        f"**Index:** `{report['analysis_index_path']}`",
        "",
        "## Summary",
        "",
        "| prefix | S2 recall | S3 recall | S2 prec | S3 prec | avg cand S2 | p95 S2 | rt S2 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary_table"]:
        lines.append(
            f"| {row['prefix_len']} | {row['S2_recall']:.4f} | {row['S3_recall']:.4f} | "
            f"{row['S2_precision']:.4f} | {row['S3_precision']:.4f} | "
            f"{row['avg_candidates_S2']:.2f} | {row['p95_candidates_S2']:.1f} | "
            f"{row['runtime_sec_S2']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def _print_comparison_table(report: dict) -> None:
    print("\n=== CP prefix comparison (500 S1) ===")
    hdr = (
        f"{'plen':>4} {'S2_rec':>8} {'S3_rec':>8} {'S2_prec':>8} {'S3_prec':>8} "
        f"{'avgS2':>8} {'avgS3':>8} {'p95S2':>8} {'p95S3':>8} {'rtS2':>7} {'rtS3':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for row in report["summary_table"]:
        print(
            f"{row['prefix_len']:>4} "
            f"{row['S2_recall']:>8.4f} {row['S3_recall']:>8.4f} "
            f"{row['S2_precision']:>8.4f} {row['S3_precision']:>8.4f} "
            f"{row['avg_candidates_S2']:>8.1f} {row['avg_candidates_S3']:>8.1f} "
            f"{row['p95_candidates_S2']:>8.1f} {row['p95_candidates_S3']:>8.1f} "
            f"{row['runtime_sec_S2']:>7.2f} {row['runtime_sec_S3']:>7.2f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate CP prefix analysis index")
    p.add_argument("--sample", type=int, default=SAMPLE_DEFAULT)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--output-dir", type=Path, default=REPORTS / "retrieval")
    args = p.parse_args()

    report = run_evaluation(sample=args.sample, db_path=args.db, output_dir=args.output_dir)
    print(f"\nWrote {report['output_json']}")
    print(f"Wrote {report['output_md']}")


if __name__ == "__main__":
    main()
