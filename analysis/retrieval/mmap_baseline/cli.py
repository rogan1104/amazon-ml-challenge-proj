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
from analysis.retrieval.config import BaselineConfig, ChannelName, RetrievalConfig
from analysis.retrieval.index import InvertedIndex, apply_candidate_cap
from analysis.retrieval.mmap_baseline.build import build_baseline_mmap_index
from analysis.retrieval.mmap_baseline.retrieve import MmapBaselineRetriever
from analysis.retrieval.mmap_baseline.schema import (
    BaselineMmapManifest,
    load_mmap_entity_universe,
    read_manifest,
)


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


def _sqlite_baseline_batch(
    channel: BaselineChannel,
    sqlite_index: InvertedIndex,
    target: str,
    s1_rows: list[dict],
    *,
    restrict_to_entities: set[str] | None,
) -> dict[str, set[str]]:
    """
    Baseline candidate sets from SQLite.

    Full-index mode: same as BaselineChannel.retrieve_batch (cap on full postings).
    Partial mode: restrict postings to ``restrict_to_entities``, then apply cap
    (equivalent to a SQLite index built only on the mmap entity subset).
    """
    queries: list[tuple[str, list[str]]] = []
    for row in s1_rows:
        s1_id = row.get("entity_id", "").strip()
        if not s1_id:
            continue
        queries.append((s1_id, list(channel.extract_query_keys(row))))

    if not queries:
        return {}

    cap = channel.cfg.max_candidates_per_s1
    if restrict_to_entities is None:
        return channel.retrieve_batch(sqlite_index, target, s1_rows)  # type: ignore[arg-type]

    raw = sqlite_index.lookup_keys_exact_batch(
        ChannelName.BASELINE, target, queries,  # type: ignore[arg-type]
    )
    return {
        qid: apply_candidate_cap(
            {eid for eid in hits if eid in restrict_to_entities},
            cap,
        )
        for qid, hits in raw.items()
    }


def verify_mmap_vs_sqlite(
    index_root: Path,
    sqlite_path: Path,
    s1_rows: list[dict],
    cfg: BaselineConfig,
    *,
    partial_build: bool = False,
    mmap_entity_universe: dict[str, set[str]] | None = None,
) -> dict:
    channel = BaselineChannel(cfg)
    mismatches: list[str] = []
    comparisons = 0
    universe = mmap_entity_universe or (
        load_mmap_entity_universe(index_root) if partial_build else None
    )

    with InvertedIndex(sqlite_path, read_only=True) as sqlite_index:
        with MmapBaselineRetriever(index_root, cfg) as mmap_ret:
            for target in ("S2", "S3"):
                allowed = universe.get(target) if universe else None
                sqlite_hits = _sqlite_baseline_batch(
                    channel,
                    sqlite_index,
                    target,
                    s1_rows,
                    restrict_to_entities=allowed if partial_build else None,
                )
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
        "partial_build": partial_build,
        "mmap_entity_counts": (
            {t: len(universe[t]) for t in ("S2", "S3")} if universe else None
        ),
    }


def cmd_verify(args: argparse.Namespace) -> None:
    index_root = Path(args.index)
    manifest = read_manifest(index_root)
    if manifest is None or not manifest.complete:
        raise SystemExit(f"Incomplete or missing mmap index: {index_root}")

    cfg = BaselineMmapManifest.baseline_config_from_manifest(manifest.baseline_config)

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

    partial = manifest.is_partial_build()
    if getattr(args, "full_index", False):
        partial = False
    elif getattr(args, "partial", False):
        partial = True

    report = verify_mmap_vs_sqlite(
        index_root,
        sqlite_path,
        s1_rows,
        cfg,
        partial_build=partial,
    )
    mode = "partial" if report["partial_build"] else "full"
    print(
        f"verify ({mode}): sample={report['sample_s1_rows']} "
        f"comparisons={report['comparisons']} mismatches={report['mismatches']} ok={report['ok']}"
    )
    if report.get("mmap_entity_counts"):
        print(f"  mmap entity universe: {report['mmap_entity_counts']}")
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
    p_verify.add_argument(
        "--partial",
        action="store_true",
        help="Restrict SQLite hits to entity_ids in the mmap index (default when manifest entity_limit is set)",
    )
    p_verify.add_argument(
        "--full-index",
        action="store_true",
        help="Compare against full SQLite index without entity restriction",
    )
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
