#!/usr/bin/env python3
"""
CLI for high-recall candidate retrieval experiment.

Modes (independently benchmarkable):
  baseline         — reference baseline E keys (norm name / addr / country+prefix5)
  char_ngram       — character n-gram name retrieval
  address_numeric  — house number / postal / numeric token retrieval
  transliteration  — Indic-script → Latin name retrieval
  all              — union of all four channels

Does NOT modify analysis/phase10_blocking.py (reference simulator only).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from analysis.retrieval.build import build_indexes
from analysis.retrieval.channels import active_channels
from analysis.retrieval.config import MODES, RetrievalConfig
from analysis.retrieval.evaluate import evaluate_recall, merge_eval_into_metrics
from analysis.retrieval.benchmark import run_benchmark
from analysis.retrieval.runner import write_candidate_pairs


def cmd_build_index(args: argparse.Namespace) -> None:
    cfg = RetrievalConfig(split=args.split, mode=args.mode, index_dir=Path(args.index_dir))
    print(f"Building index: split={cfg.split} mode={cfg.mode} channels={[c.name.value for c in active_channels(cfg)]}")
    build_indexes(cfg)


def cmd_run(args: argparse.Namespace) -> None:
    cfg = RetrievalConfig(
        split=args.split,
        mode=args.mode,
        index_dir=Path(args.index_dir),
        output_dir=Path(args.output_dir),
    )
    out = args.output or str(cfg.output_tsv())
    print(f"Retrieval run: mode={cfg.mode} split={cfg.split} -> {out}")
    summary = write_candidate_pairs(
        cfg,
        output_path=Path(out),
        max_s1_rows=args.limit,
        progress_every=args.progress_every,
    )
    print(f"  S1={summary['total_s1']:,} avg_cand={summary['avg_candidates_per_s1']} "
          f"zero_s1={summary['s1_zero_candidate_pct']}% elapsed={summary['elapsed_sec']}s")

    if args.evaluate and cfg.split == "train":
        ev = evaluate_recall(Path(out))
        merge_eval_into_metrics(cfg.metrics_json(), ev)
        print(f"  TRAIN RECALL: {ev['recall']*100:.2f}% "
              f"(S2={ev['s2_recall']*100:.2f}% S3={ev['s3_recall']*100:.2f}%)")


def cmd_benchmark(args: argparse.Namespace) -> None:
    cfg = RetrievalConfig(
        split=args.split,
        mode=args.mode,
        index_dir=Path(args.index_dir),
        output_dir=Path(args.output_dir),
    )
    print(f"Benchmark retrieval: mode={cfg.mode} split={cfg.split} limit={args.limit} S1 rows")
    report = run_benchmark(
        cfg,
        limit=args.limit,
        compare_row_loop=not args.no_compare_row,
        verify=not args.no_verify,
    )
    batched = report["batched"]
    print(
        f"  batched: {batched['rows']:,} rows in {batched['total_sec']:.2f}s "
        f"({report['throughput_s1_per_sec']:,.0f} S1/s, sqlite={batched['sqlite_sec']:.2f}s)"
    )
    if "sqlite_speedup_vs_row_loop" in report:
        print(f"  sqlite speedup vs per-row loop (same sample): {report['sqlite_speedup_vs_row_loop']}x")
    if report.get("semantics_check", {}).get("ok"):
        print("  semantics check: batched baseline == per-row on sample")
    print(f"  estimated full train retrieval: ~{report['estimated_full_train_hours']:.2f} hours")
    print(f"  report: {report['benchmark_json']}")


def cmd_full(args: argparse.Namespace) -> None:
    """build-index + run (+ optional evaluate) in one shot."""
    cmd_build_index(args)
    cmd_run(args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="High-recall candidate retrieval experiment for Entity Resolution",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--mode",
            choices=list(MODES.keys()),
            required=True,
            help="Retrieval mode: baseline | char_ngram | address_numeric | transliteration | all",
        )
        p.add_argument("--split", choices=["train", "test"], default="train")
        p.add_argument(
            "--index-dir",
            default="reports/cache/retrieval_index",
            help="Directory for SQLite inverted indexes",
        )
        p.add_argument(
            "--output-dir",
            default="reports/retrieval",
            help="Directory for candidate_pairs.tsv and metrics JSON",
        )

    p_build = sub.add_parser("build-index", help="Build inverted indexes from S2/S3 TSV")
    add_common(p_build)
    p_build.set_defaults(func=cmd_build_index)

    p_run = sub.add_parser("run", help="Retrieve candidates for S1 using built index")
    add_common(p_run)
    p_run.add_argument("--output", default=None, help="Output candidate_pairs.tsv path")
    p_run.add_argument(
        "--evaluate",
        action="store_true",
        help="On train split, compute true-match recall vs GT",
    )
    p_run.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N S1 rows (benchmark / smoke test)",
    )
    p_run.add_argument(
        "--progress-every",
        type=int,
        default=25_000,
        help="Log progress to stderr every N S1 rows (0 to disable)",
    )
    p_run.set_defaults(func=cmd_run)

    p_bench = sub.add_parser(
        "benchmark",
        help="Profile retrieval on a small S1 sample (uses existing index)",
    )
    add_common(p_bench)
    p_bench.add_argument("--limit", type=int, default=10_000)
    p_bench.add_argument(
        "--no-compare-row",
        action="store_true",
        help="Skip slow per-row SQLite comparison (baseline mode)",
    )
    p_bench.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip batched vs per-row semantics check",
    )
    p_bench.set_defaults(func=cmd_benchmark)

    p_full = sub.add_parser("full", help="build-index then run (convenience)")
    add_common(p_full)
    p_full.add_argument("--output", default=None)
    p_full.add_argument("--evaluate", action="store_true")
    p_full.set_defaults(func=cmd_full)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
