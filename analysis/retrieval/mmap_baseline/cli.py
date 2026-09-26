#!/usr/bin/env python3
"""CLI for mmap CSR baseline index build and SQLite equivalence checks."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from analysis.config import TEST, TRAIN
from analysis.retrieval.build import _index_path
from analysis.retrieval.channels import BaselineChannel
from analysis.retrieval.config import BaselineConfig, RetrievalConfig
from analysis.retrieval.index import InvertedIndex
from analysis.retrieval.mmap_baseline.build import build_baseline_mmap_index
from analysis.retrieval.mmap_baseline.retrieve import MmapBaselineRetriever
from analysis.retrieval.mmap_baseline.schema import read_manifest


def _split_paths(split: str) -> dict[str, Path]:
    base = TRAIN if split == "train" else TEST
    return {"S1": base["S1"], "S2": base["S2"], "S3": base["S3"]}


def cmd_build(args: argparse.Namespace) -> None:
    paths = _split_paths(args.split)
    cfg = BaselineConfig(
        name_prefix_len=args.name_prefix_len,
        min_prefix_len=args.min_prefix_len,
        max_candidates_per_s1=args.max_candidates,
    )
    build_baseline_mmap_index(
        Path(args.output),
        split=args.split,
        paths={"S2": paths["S2"], "S3": paths["S3"]},
        cfg=cfg,
        batch_size=args.batch_size,
        entity_limit=args.entity_limit,
        rebuild=args.rebuild,
    )


def _load_first_n_s1(path: Path, n: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("entity_id", "").strip():
                rows.append(row)
            if len(rows) >= n:
                break
    return rows


def verify_mmap_vs_sqlite(
    index_root: Path,
    sqlite_path: Path,
    s1_rows: list[dict],
    cfg: BaselineConfig,
) -> dict:
    channel = BaselineChannel(cfg)
    mismatches: list[str] = []
    comparisons = 0

    with InvertedIndex(sqlite_path, read_only=True) as sqlite_index:
        with MmapBaselineRetriever(index_root, cfg) as mmap_ret:
            for target in ("S2", "S3"):
                sqlite_hits = channel.retrieve_batch(sqlite_index, target, s1_rows)  # type: ignore[arg-type]
                mmap_hits = mmap_ret.retrieve_batch(target, s1_rows)
                for row in s1_rows:
                    s1_id = row["entity_id"].strip()
                    comparisons += 1
                    a = sqlite_hits.get(s1_id, set())
                    b = mmap_hits.get(s1_id, set())
                    if a != b:
                        mismatches.append(
                            f"{s1_id}/{target}: sqlite={len(a)} mmap={len(b)} "
                            f"only_sqlite={len(a - b)} only_mmap={len(b - a)}"
                        )

    return {
        "sample_s1_rows": len(s1_rows),
        "comparisons": comparisons,
        "mismatches": len(mismatches),
        "ok": len(mismatches) == 0,
        "mismatch_examples": mismatches[:10],
        "sqlite_index": str(sqlite_path),
        "mmap_index": str(index_root),
    }


def cmd_verify(args: argparse.Namespace) -> None:
    index_root = Path(args.index)
    manifest = read_manifest(index_root)
    if manifest is None or not manifest.complete:
        raise SystemExit(f"Incomplete or missing mmap index: {index_root}")

    cfg = BaselineConfig(
        name_prefix_len=manifest.baseline_config.get("name_prefix_len", 5),
        min_prefix_len=manifest.baseline_config.get("min_prefix_len", 3),
        max_candidates_per_s1=manifest.baseline_config.get("max_candidates_per_s1", 10_000),
        cp_max_df=manifest.baseline_config.get("cp_max_df"),
    )

    sqlite_path = Path(args.sqlite) if args.sqlite else _index_path(
        RetrievalConfig(split=args.split, mode="baseline"),
    )
    if not sqlite_path.is_file():
        raise SystemExit(
            f"SQLite index not found: {sqlite_path}\n"
            "Build the retrieval SQLite index first, or pass --sqlite PATH."
        )

    paths = _split_paths(args.split)
    s1_rows = _load_first_n_s1(paths["S1"], args.sample)
    if not s1_rows:
        raise SystemExit(f"No S1 rows read from {paths['S1']}")

    report = verify_mmap_vs_sqlite(index_root, sqlite_path, s1_rows, cfg)
    print(f"verify: sample={report['sample_s1_rows']} comparisons={report['comparisons']} "
          f"mismatches={report['mismatches']} ok={report['ok']}")
    if report["mismatch_examples"]:
        for line in report["mismatch_examples"]:
            print(f"  {line}")
    if not report["ok"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mmap CSR baseline index (experimental)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build mmap index from S2/S3 TSV")
    p_build.add_argument("--split", choices=("train", "test"), default="train")
    p_build.add_argument(
        "--output",
        default="reports/cache/baseline_mmap/train",
        help="Index root directory",
    )
    p_build.add_argument("--batch-size", type=int, default=50_000)
    p_build.add_argument(
        "--entity-limit",
        type=int,
        default=None,
        help="Index only first N entities per target (smoke tests)",
    )
    p_build.add_argument("--rebuild", action="store_true")
    p_build.add_argument("--name-prefix-len", type=int, default=5)
    p_build.add_argument("--min-prefix-len", type=int, default=3)
    p_build.add_argument("--max-candidates", type=int, default=10_000)
    p_build.set_defaults(func=cmd_build)

    p_verify = sub.add_parser("verify", help="Compare mmap vs SQLite on N S1 rows")
    p_verify.add_argument("--split", choices=("train", "test"), default="train")
    p_verify.add_argument(
        "--index",
        default="reports/cache/baseline_mmap/train",
        help="Mmap index root",
    )
    p_verify.add_argument("--sqlite", default=None, help="Path to index_{split}.sqlite")
    p_verify.add_argument("--sample", type=int, default=50)
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
