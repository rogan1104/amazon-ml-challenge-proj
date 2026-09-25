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
    summary = write_candidate_pairs(cfg, output_path=Path(out))
    print(f"  S1={summary['total_s1']:,} avg_cand={summary['avg_candidates_per_s1']} "
          f"zero_s1={summary['s1_zero_candidate_pct']}% elapsed={summary['elapsed_sec']}s")

    if args.evaluate and cfg.split == "train":
        ev = evaluate_recall(Path(out))
        merge_eval_into_metrics(cfg.metrics_json(), ev)
        print(f"  TRAIN RECALL: {ev['recall']*100:.2f}% "
              f"(S2={ev['s2_recall']*100:.2f}% S3={ev['s3_recall']*100:.2f}%)")


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
    p_run.set_defaults(func=cmd_run)

    p_full = sub.add_parser("full", help="build-index then run (convenience)")
    add_common(p_full)
    p_full.add_argument("--output", default=None)
    p_full.add_argument("--evaluate", action="store_true")
    p_full.set_defaults(func=cmd_full)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
