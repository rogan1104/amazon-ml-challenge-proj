"""Time mmap baseline retrieval on the first N S1 rows (no TSV output)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

from analysis.retrieval.build import _stream_tsv
from analysis.retrieval.config import BaselineConfig, RetrievalConfig, Split
from analysis.retrieval.mmap_baseline.retrieve import MmapBaselineRetriever
from analysis.retrieval.mmap_baseline.schema import BaselineMmapManifest, read_manifest


def _stream_limited_s1(
    s1_path: Path,
    *,
    limit: int,
    batch_size: int,
) -> Iterator[list[dict]]:
    remaining = limit
    for batch in _stream_tsv(s1_path, batch_size):
        if remaining <= 0:
            break
        rows = [r for r in batch if r.get("entity_id", "").strip()]
        if len(rows) > remaining:
            rows = rows[:remaining]
        if rows:
            yield rows
        remaining -= len(rows)


def run_mmap_retrieval_benchmark(
    index_root: Path,
    *,
    split: Split = "train",
    limit: int = 500,
    batch_size: int = 50_000,
    cfg: BaselineConfig | None = None,
    s1_path: Path | None = None,
) -> dict:
    """
    Retrieve baseline candidates for the first ``limit`` S1 rows via mmap only.

    Uses ``baseline_keys`` and ``max_candidates_per_s1`` from ``cfg`` (manifest default).
    Does not write candidate_pairs.tsv or touch SQLite.
    """
    index_root = Path(index_root)
    manifest = read_manifest(index_root)
    if manifest is None or not manifest.complete:
        raise FileNotFoundError(f"Incomplete or missing mmap index: {index_root}")

    cfg = cfg or BaselineMmapManifest.baseline_config_from_manifest(manifest.baseline_config)
    rcfg = RetrievalConfig(split=split, batch_size=batch_size)
    s1_path = s1_path or rcfg.paths()["S1"]
    if not s1_path.is_file():
        raise FileNotFoundError(s1_path)

    total_s1 = 0
    total_s2_pairs = 0
    total_s3_pairs = 0
    total_combined_pairs = 0
    max_combined = 0
    zero_combined = 0

    t0 = time.perf_counter()
    with MmapBaselineRetriever(index_root, cfg) as retriever:
        for batch in _stream_limited_s1(s1_path, limit=limit, batch_size=batch_size):
            batch_candidates: dict[str, set[str]] = {
                r["entity_id"].strip(): set() for r in batch
            }
            s2_hits = retriever.retrieve_batch("S2", batch)
            s3_hits = retriever.retrieve_batch("S3", batch)

            for s1_id in batch_candidates:
                s2 = s2_hits.get(s1_id, set())
                s3 = s3_hits.get(s1_id, set())
                combined = s2 | s3
                batch_candidates[s1_id] = combined

                total_s1 += 1
                total_s2_pairs += len(s2)
                total_s3_pairs += len(s3)
                n_combined = len(combined)
                total_combined_pairs += n_combined
                max_combined = max(max_combined, n_combined)
                if n_combined == 0:
                    zero_combined += 1

    wall_sec = time.perf_counter() - t0
    rows_per_sec = total_s1 / wall_sec if wall_sec > 0 else 0.0

    report = {
        "engine": "mmap_baseline",
        "split": split,
        "limit_s1": limit,
        "s1_rows_processed": total_s1,
        "wall_sec": round(wall_sec, 4),
        "rows_per_sec": round(rows_per_sec, 2),
        "total_s2_candidate_pairs": total_s2_pairs,
        "total_s3_candidate_pairs": total_s3_pairs,
        "total_combined_candidate_pairs": total_combined_pairs,
        "avg_s2_candidates_per_s1": round(total_s2_pairs / total_s1, 2) if total_s1 else 0,
        "avg_s3_candidates_per_s1": round(total_s3_pairs / total_s1, 2) if total_s1 else 0,
        "avg_combined_candidates_per_s1": round(total_combined_pairs / total_s1, 2) if total_s1 else 0,
        "max_combined_candidates_per_s1": max_combined,
        "s1_zero_combined_candidates": zero_combined,
        "index_path": str(index_root),
        "s1_path": str(s1_path),
        "batch_size": batch_size,
        "baseline_config": BaselineMmapManifest.baseline_config_dict(cfg),
    }
    return report


def write_benchmark_report(report: dict, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    limit = report.get("limit_s1", report.get("s1_rows_processed", 0))
    split = report.get("split", "train")
    out_path = output_dir / f"benchmark_mmap_{split}_{limit}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["benchmark_json"] = str(out_path)
    return out_path
