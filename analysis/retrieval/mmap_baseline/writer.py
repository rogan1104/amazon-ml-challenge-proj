"""Production mmap baseline candidate_pairs.tsv writer (streams S1; no full RAM load)."""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Iterator

from analysis.config import REPORTS
from analysis.retrieval.build import _stream_tsv
from analysis.retrieval.config import RetrievalConfig, Split
from analysis.retrieval.mmap_baseline.retrieve import MmapBaselineRetriever
from analysis.retrieval.mmap_baseline.schema import (
    BaselineMmapManifest,
    read_manifest,
)


def default_candidate_pairs_path(split: Split) -> Path:
    return REPORTS / "retrieval" / f"candidate_pairs_{split}_mmap.tsv"


def default_metrics_path(split: Split) -> Path:
    return REPORTS / "retrieval" / f"retrieval_metrics_{split}_mmap.json"


def _stream_s1_batches(
    cfg: RetrievalConfig,
    max_s1_rows: int | None = None,
    *,
    s1_path: Path | None = None,
) -> Iterator[list[dict]]:
    source = s1_path or cfg.paths()["S1"]
    if max_s1_rows is None:
        yield from _stream_tsv(source, cfg.batch_size)
        return
    remaining = max_s1_rows
    for batch in _stream_tsv(source, cfg.batch_size):
        if remaining <= 0:
            break
        if len(batch) > remaining:
            batch = batch[:remaining]
        yield batch
        remaining -= len(batch)


def write_candidate_pairs_mmap(
    index_root: Path,
    output_path: Path,
    *,
    split: Split = "train",
    batch_size: int = 50_000,
    max_s1_rows: int | None = None,
    progress_every: int = 25_000,
    metrics_path: Path | None = None,
    s1_path: Path | None = None,
) -> dict:
    """
    Union S2+S3 mmap baseline hits per S1 row; write challenge candidate_pairs.tsv.

    Same schema and semantics as ``analysis.retrieval.runner.write_candidate_pairs``
    for baseline-only mode: sorted comma-joined IDs, empty field if no candidates.
    """
    index_root = Path(index_root)
    output_path = Path(output_path)
    manifest = read_manifest(index_root)
    if manifest is None or not manifest.complete:
        raise FileNotFoundError(
            f"Incomplete or missing mmap index: {index_root}. Run build first.",
        )

    cfg = BaselineMmapManifest.baseline_config_from_manifest(manifest.baseline_config)
    rcfg = RetrievalConfig(split=split, batch_size=batch_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    total_s1 = 0
    zero_cand = 0
    total_cand_pairs = 0
    max_cand = 0

    with MmapBaselineRetriever(index_root, cfg) as retriever:
        with output_path.open("w", encoding="utf-8", newline="") as out:
            writer = csv.writer(out, delimiter="\t", lineterminator="\n")
            writer.writerow(["source1_entity_id", "candidate_entity_ids"])

            for batch in _stream_s1_batches(
                rcfg, max_s1_rows=max_s1_rows, s1_path=s1_path,
            ):
                rows = [r for r in batch if r.get("entity_id", "").strip()]
                if not rows:
                    continue

                batch_candidates: dict[str, set[str]] = {
                    r["entity_id"].strip(): set() for r in rows
                }
                for target in ("S2", "S3"):
                    hits_by_s1 = retriever.retrieve_batch(target, rows)
                    for s1_id, hits in hits_by_s1.items():
                        batch_candidates.setdefault(s1_id, set()).update(hits)

                for row in rows:
                    s1_id = row["entity_id"].strip()
                    cands = batch_candidates.get(s1_id, set())
                    total_s1 += 1
                    n = len(cands)
                    total_cand_pairs += n
                    max_cand = max(max_cand, n)
                    if n == 0:
                        zero_cand += 1
                    if progress_every > 0 and total_s1 % progress_every == 0:
                        elapsed = time.perf_counter() - t0
                        rate = total_s1 / elapsed if elapsed > 0 else 0.0
                        print(
                            f"  mmap retrieval: {total_s1:,} S1 ({rate:,.0f}/s, {elapsed:.1f}s)",
                            file=sys.stderr,
                            flush=True,
                        )
                    writer.writerow([s1_id, ",".join(sorted(cands))])

    elapsed = round(time.perf_counter() - t0, 2)
    summary = {
        "engine": "mmap_baseline",
        "mode": "baseline",
        "split": split,
        "channels": ["baseline"],
        "total_s1": total_s1,
        "s1_zero_candidates": zero_cand,
        "s1_zero_candidate_pct": round(100 * zero_cand / total_s1, 4) if total_s1 else 0,
        "total_candidate_pairs": total_cand_pairs,
        "avg_candidates_per_s1": round(total_cand_pairs / total_s1, 2) if total_s1 else 0,
        "max_candidates_per_s1": max_cand,
        "output_tsv": str(output_path),
        "index_path": str(index_root),
        "elapsed_sec": elapsed,
        "max_s1_rows": max_s1_rows,
    }

    mpath = metrics_path or default_metrics_path(split)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["metrics_json"] = str(mpath)
    return summary
