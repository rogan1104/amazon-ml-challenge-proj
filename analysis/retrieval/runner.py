"""Run candidate retrieval: union channels, separate S2/S3, deduplicate."""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Callable, Iterator

from analysis.retrieval.build import _index_path, _stream_tsv
from analysis.retrieval.channels import RetrievalChannel, active_channels
from analysis.retrieval.config import RetrievalConfig, Target


def _stream_s1_batches(
    cfg: RetrievalConfig,
    max_s1_rows: int | None = None,
) -> Iterator[list[dict]]:
    """Stream S1 TSV in batches, optionally capped for benchmarks."""
    if max_s1_rows is None:
        yield from _stream_tsv(cfg.paths()["S1"], cfg.batch_size)
        return

    remaining = max_s1_rows
    for batch in _stream_tsv(cfg.paths()["S1"], cfg.batch_size):
        if remaining <= 0:
            break
        if len(batch) > remaining:
            batch = batch[:remaining]
        yield batch
        remaining -= len(batch)


def retrieve_for_s1(
    index_path: Path,
    cfg: RetrievalConfig,
    channels: list[RetrievalChannel] | None = None,
    *,
    max_s1_rows: int | None = None,
    progress_every: int = 25_000,
    on_progress: Callable[[int, float], None] | None = None,
) -> Iterator[tuple[str, set[str]]]:
    """
    Yield (s1_entity_id, deduplicated candidate entity_ids) for each S1 row.
    S2 and S3 candidates are retrieved separately then unioned.
    """
    from analysis.retrieval.index import InvertedIndex

    channels = channels or active_channels(cfg)
    t0 = time.perf_counter()
    processed = 0

    with InvertedIndex(index_path, read_only=True) as index:
        for batch in _stream_s1_batches(cfg, max_s1_rows=max_s1_rows):
            rows = [r for r in batch if r.get("entity_id", "").strip()]
            if not rows:
                continue

            batch_candidates: dict[str, set[str]] = {
                r["entity_id"].strip(): set() for r in rows
            }

            for target in ("S2", "S3"):
                for channel in channels:
                    hits_by_s1 = channel.retrieve_batch(index, target, rows)  # type: ignore[arg-type]
                    for s1_id, hits in hits_by_s1.items():
                        batch_candidates.setdefault(s1_id, set()).update(hits)

            for row in rows:
                s1_id = row["entity_id"].strip()
                processed += 1
                if on_progress and progress_every > 0 and processed % progress_every == 0:
                    on_progress(processed, time.perf_counter() - t0)
                yield s1_id, batch_candidates.get(s1_id, set())


def write_candidate_pairs(
    cfg: RetrievalConfig,
    index_path: Path | None = None,
    output_path: Path | None = None,
    channels: list[RetrievalChannel] | None = None,
    *,
    max_s1_rows: int | None = None,
    progress_every: int = 25_000,
) -> dict:
    """Write candidate_pairs.tsv and return summary stats."""
    index_path = index_path or _index_path(cfg)
    output_path = output_path or cfg.output_tsv()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found: {index_path}. Run build-index first."
        )

    channels = channels or active_channels(cfg)
    t0 = time.perf_counter()
    total_s1 = 0
    zero_cand = 0
    total_cand_pairs = 0
    max_cand = 0

    def _progress(n: int, elapsed: float) -> None:
        rate = n / elapsed if elapsed > 0 else 0.0
        print(
            f"  retrieval progress: {n:,} S1 rows ({rate:,.0f} rows/s, {elapsed:.1f}s elapsed)",
            file=sys.stderr,
            flush=True,
        )

    with open(output_path, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for s1_id, cands in retrieve_for_s1(
            index_path,
            cfg,
            channels,
            max_s1_rows=max_s1_rows,
            progress_every=progress_every,
            on_progress=_progress,
        ):
            total_s1 += 1
            n = len(cands)
            total_cand_pairs += n
            max_cand = max(max_cand, n)
            if n == 0:
                zero_cand += 1
            writer.writerow([s1_id, ",".join(sorted(cands))])

    elapsed = round(time.perf_counter() - t0, 2)
    summary = {
        "mode": cfg.mode,
        "split": cfg.split,
        "channels": [c.name.value for c in channels],
        "total_s1": total_s1,
        "s1_zero_candidates": zero_cand,
        "s1_zero_candidate_pct": round(100 * zero_cand / total_s1, 4) if total_s1 else 0,
        "total_candidate_pairs": total_cand_pairs,
        "avg_candidates_per_s1": round(total_cand_pairs / total_s1, 2) if total_s1 else 0,
        "max_candidates_per_s1": max_cand,
        "output_tsv": str(output_path),
        "index_path": str(index_path),
        "elapsed_sec": elapsed,
        "max_s1_rows": max_s1_rows,
    }

    metrics_path = cfg.metrics_json()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
