"""
CP prefix-length selectivity analysis (analysis only).

Deterministic sample: first 500 S1 rows in train_source1.tsv file order.

The on-disk baseline index is built with cp prefix_len=5 (country|first 5 name chars).
Other prefix lengths use the same key *format* but those exact keys were not indexed
on S2/S3 unless they coincidentally match a 5-char key — results for prefix_len != 5
are reported with index_compatibility flags (no silent approximation).

Usage (PC):
  python -m analysis.retrieval.analyze_cp_prefix_selectivity
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Literal

from analysis.config import REPORTS, TRAIN
from analysis.retrieval.build import _index_path
from analysis.retrieval.config import ChannelName, RetrievalConfig
from analysis.retrieval.index import InvertedIndex
from analysis.retrieval.keys import baseline_keys

Target = Literal["S2", "S3"]

# Must match BaselineConfig / index build (phase10 baseline E).
INDEX_BUILD_CP_PREFIX_LEN = 5

PREFIX_LENGTHS = (3, 4, 5, 6, 7, 8)

SAMPLE_DEFAULT = 500


def load_first_n_s1(path: Path, n: int) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
            if len(rows) >= n:
                break
    return rows


def cp_keys_for_row(row: dict, prefix_len: int, min_prefix: int = 3) -> list[str]:
    keys = baseline_keys(
        row.get("business_name", ""),
        row.get("business_address", ""),
        row.get("country", ""),
        prefix_len=prefix_len,
        min_prefix=min_prefix,
    )
    return [k for k in keys if k.startswith("cp:")]


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


def _candidate_distribution(cands: dict[str, set[str]], sample_ids: list[str]) -> dict:
    counts = [len(cands.get(s1, set())) for s1 in sample_ids]
    s = sorted(counts)
    return {
        "avg_candidates_per_s1": round(statistics.mean(counts), 4) if counts else 0,
        "median_candidates_per_s1": statistics.median(counts) if counts else 0,
        "p95_candidates_per_s1": round(_percentile(s, 95), 2),
        "p99_candidates_per_s1": round(_percentile(s, 99), 2),
        "max_candidates_per_s1": max(counts) if counts else 0,
    }


def _recall_metrics(
    cands: dict[str, set[str]],
    gt: dict[str, dict],
    sample_ids: list[str],
    target: Target,
) -> dict:
    true_pairs = 0
    recalled = 0
    for s1 in sample_ids:
        true_set = gt[s1][target]
        true_pairs += len(true_set)
        recalled += len(true_set & cands.get(s1, set()))
    missed = true_pairs - recalled
    return {
        "gt_pair_recall": round(recalled / true_pairs, 6) if true_pairs else 0.0,
        "gt_positive_pairs": true_pairs,
        "gt_positive_pairs_recalled": recalled,
        "gt_positive_pairs_missed": missed,
    }


def _subset_recall(
    cands: dict[str, set[str]],
    gt: dict[str, dict],
    s1_ids: list[str],
    target: Target,
) -> float:
    if not s1_ids:
        return 0.0
    m = _recall_metrics(cands, gt, s1_ids, target)
    return m["gt_pair_recall"]


def _keys_in_key_df(
    index: InvertedIndex,
    target: Target,
    keys: set[str],
) -> tuple[int, int]:
    """Return (keys_present_in_index, total_keys)."""
    if not keys:
        return 0, 0
    ch = ChannelName.BASELINE.value
    found = 0
    conn = index._conn
    batch = list(keys)
    chunk = 200
    for i in range(0, len(batch), chunk):
        part = batch[i : i + chunk]
        ph = ",".join("?" * len(part))
        sql = f"""
            SELECT COUNT(*) FROM key_df
            WHERE channel=? AND target=? AND key IN ({ph})
        """
        row = conn.execute(sql, [ch, target, *part]).fetchone()
        found += row[0] if row else 0
    return found, len(keys)


def run_analysis(sample: int, index_dir: Path, output_dir: Path) -> dict:
    cfg = RetrievalConfig(index_dir=index_dir)
    index_path = _index_path(cfg)
    s1_path = TRAIN["S1"]
    gt_path = TRAIN["GT"]

    if not index_path.is_file():
        raise FileNotFoundError(index_path)
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

    keys_by_prefix: dict[int, list[tuple[str, list[str]]]] = {}
    for plen in PREFIX_LENGTHS:
        keys_by_prefix[plen] = [
            (r["entity_id"].strip(), cp_keys_for_row(r, plen))
            for r in rows
            if r.get("entity_id", "").strip()
        ]

    report: dict = {
        "sample_method": f"first_{sample}_s1_rows_in_train_source1_tsv_file_order",
        "sample_s1_count": len(sample_ids),
        "index_path": str(index_path),
        "index_build_cp_prefix_len": INDEX_BUILD_CP_PREFIX_LEN,
        "index_limitation_note": (
            "S2/S3 baseline postings in index_train.sqlite were indexed with "
            f"cp prefix_len={INDEX_BUILD_CP_PREFIX_LEN} only. "
            "Lookups for other prefix lengths are exact-key retrievals against that "
            "index; they do not simulate re-indexing S2/S3 at shorter/longer prefixes."
        ),
        "by_prefix_len": {},
        "summary_table": [],
    }

    with InvertedIndex(index_path, read_only=True) as index:
        index.enable_fanout_profiling()

        for plen in PREFIX_LENGTHS:
            queries = keys_by_prefix[plen]
            unique_cp_keys = {k for _, ks in queries for k in ks}
            plen_report: dict = {
                "prefix_len": plen,
                "index_compatible_with_current_index": plen == INDEX_BUILD_CP_PREFIX_LEN,
                "targets": {},
            }

            for target in ("S2", "S3"):
                print(f"[cp-prefix] len={plen} target={target} ...", flush=True)
                keys_in_df, keys_total = _keys_in_key_df(
                    index, target, unique_cp_keys  # type: ignore[arg-type]
                )

                index.reset_fanout_profiling()
                t0 = time.perf_counter()
                cands = index.lookup_keys_exact_batch(
                    ChannelName.BASELINE, target, queries  # type: ignore[arg-type]
                )
                sqlite_sec = round(time.perf_counter() - t0, 3)
                prof = index.get_fanout_profile()
                posting_rows = prof.posting_rows_returned if prof else 0

                dist = _candidate_distribution(cands, sample_ids)
                recall = _recall_metrics(cands, gt, sample_ids, target)  # type: ignore[arg-type]

                plen_report["targets"][target] = {
                    "unique_cp_keys_queried": len(unique_cp_keys),
                    "cp_keys_present_in_key_df": keys_in_df,
                    "cp_keys_missing_from_key_df": keys_total - keys_in_df,
                    "total_posting_rows_returned": posting_rows,
                    **dist,
                    **recall,
                    "recall_US": _subset_recall(cands, gt, us_ids, target),  # type: ignore[arg-type]
                    "recall_IN": _subset_recall(cands, gt, in_ids, target),  # type: ignore[arg-type]
                    "recall_singleton": _subset_recall(
                        cands, gt, singleton_ids, target,  # type: ignore[arg-type]
                    ),
                    "recall_non_singleton": _subset_recall(
                        cands, gt, nonsingleton_ids, target,  # type: ignore[arg-type]
                    ),
                    "sqlite_retrieval_sec": sqlite_sec,
                }
                print(
                    f"  recall={recall['gt_pair_recall']:.4f} "
                    f"avg_cand={dist['avg_candidates_per_s1']:.1f} "
                    f"posting_rows={posting_rows:,} "
                    f"keys_in_df={keys_in_df}/{keys_total} "
                    f"runtime={sqlite_sec}s",
                    flush=True,
                )

            s2 = plen_report["targets"]["S2"]
            s3 = plen_report["targets"]["S3"]
            report["summary_table"].append({
                "prefix_len": plen,
                "index_compatible": plen == INDEX_BUILD_CP_PREFIX_LEN,
                "S2_recall": s2["gt_pair_recall"],
                "S3_recall": s3["gt_pair_recall"],
                "avg_candidates_S2": s2["avg_candidates_per_s1"],
                "avg_candidates_S3": s3["avg_candidates_per_s1"],
                "p95_candidates_S2": s2["p95_candidates_per_s1"],
                "p95_candidates_S3": s3["p95_candidates_per_s1"],
                "sqlite_sec_S2": s2["sqlite_retrieval_sec"],
                "sqlite_sec_S3": s3["sqlite_retrieval_sec"],
            })
            report["by_prefix_len"][str(plen)] = plen_report

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"cp_prefix_selectivity_{sample}.json"
    md_path = output_dir / f"cp_prefix_selectivity_{sample}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_format_md(report), encoding="utf-8")
    report["output_json"] = str(json_path)
    report["output_md"] = str(md_path)
    return report


def _format_md(report: dict) -> str:
    lines = [
        "# CP prefix selectivity",
        "",
        f"**Sample:** {report['sample_method']} (n={report['sample_s1_count']})",
        "",
        report["index_limitation_note"],
        "",
        "## Summary",
        "",
        "| prefix | compatible | S2 recall | S3 recall | avg cand S2 | p95 S2 | runtime S2 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary_table"]:
        compat = "yes" if row["index_compatible"] else "**no**"
        lines.append(
            f"| {row['prefix_len']} | {compat} | {row['S2_recall']:.4f} | "
            f"{row['S3_recall']:.4f} | {row['avg_candidates_S2']:.2f} | "
            f"{row['p95_candidates_S2']:.1f} | {row['sqlite_sec_S2']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="CP prefix-length selectivity (analysis only)")
    p.add_argument("--sample", type=int, default=SAMPLE_DEFAULT)
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


if __name__ == "__main__":
    main()
