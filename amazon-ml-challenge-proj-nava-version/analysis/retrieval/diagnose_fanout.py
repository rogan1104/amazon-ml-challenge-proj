"""
Profile key-centric fan-out vs SQLite fetch on a tiny S1 sample.

Usage (PC, ~20 rows, seconds only):
  python -m analysis.retrieval.diagnose_fanout --sample 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from analysis.retrieval.benchmark import _collect_s1_rows
from analysis.retrieval.build import _index_path
from analysis.retrieval.channels import BaselineChannel
from analysis.retrieval.config import RetrievalConfig
from analysis.retrieval.index import InvertedIndex


def run_diagnosis(cfg: RetrievalConfig, sample: int, index_path: Path) -> dict:
    channel = BaselineChannel(cfg.baseline)
    rows = _collect_s1_rows(cfg, sample)
    queries_n = len([r for r in rows if r.get("entity_id", "").strip()])

    report: dict = {
        "sample_s1_rows": queries_n,
        "index_path": str(index_path),
        "benchmark_note": (
            "benchmark.py 'sqlite_sec' times the entire retrieve_batch() call, "
            "including Python fan-out in _fanout_keycentric_postings — not SQL alone."
        ),
        "targets": {},
    }

    with InvertedIndex(index_path, read_only=True) as index:
        index.enable_fanout_profiling()
        for target in ("S2", "S3"):
            print(f"[fanout] baseline retrieve_batch target={target} ...", flush=True)
            index.reset_fanout_profiling()
            hits = channel.retrieve_batch(index, target, rows)  # type: ignore[arg-type]
            prof = index.get_fanout_profile()
            assert prof is not None
            d = prof.to_dict()
            d["total_profiled_sec"] = round(
                d["temp_table_sec"] + d["sql_fetch_sec"] + d["python_fanout_sec"],
                4,
            )
            d["sql_plus_fetch_pct"] = round(
                100 * d["sql_fetch_sec"] / max(d["total_profiled_sec"], 1e-9),
                1,
            )
            d["python_fanout_pct"] = round(
                100 * d["python_fanout_sec"] / max(d["total_profiled_sec"], 1e-9),
                1,
            )
            d["sample_s1_with_candidates"] = sum(1 for s in hits.values() if s)
            report["targets"][target] = d
            print(
                f"[fanout] target={target} "
                f"fetch={d['sql_fetch_sec']:.3f}s fanout={d['python_fanout_sec']:.3f}s "
                f"posting_rows={d['posting_rows_returned']:,} "
                f"fanout_ops={d['fanout_add_operations']:,} "
                f"cp_fanout_ops={d['cp_fanout_add_operations']:,}",
                flush=True,
            )

    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Profile key-centric fan-out (tiny sample)")
    p.add_argument("--split", default="train")
    p.add_argument("--sample", type=int, default=20)
    p.add_argument("--index-dir", default="reports/cache/retrieval_index")
    args = p.parse_args()

    cfg = RetrievalConfig(split=args.split, index_dir=Path(args.index_dir))
    index_path = _index_path(cfg)
    if not index_path.is_file():
        print(f"Index not found: {index_path}", file=sys.stderr)
        raise SystemExit(2)
    if not cfg.paths()["S1"].is_file():
        print(f"S1 not found: {cfg.paths()['S1']}", file=sys.stderr)
        raise SystemExit(2)

    report = run_diagnosis(cfg, args.sample, index_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
