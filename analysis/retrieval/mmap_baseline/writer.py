"""Stream candidate_pairs.tsv from mmap baseline retrieval."""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Callable, Iterator

from analysis.retrieval.build import _stream_tsv
from analysis.retrieval.config import RetrievalConfig
from analysis.retrieval.mmap_baseline.retrieve import MmapBaselineRetriever


def _stream_s1_batches(
    cfg: RetrievalConfig,
    max_s1_rows: int | None = None,
) -> Iterator[list[dict]]:
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


def write_candidate_pairs_mmap(
    index_root: Path,
    cfg: RetrievalConfig,
    output_path: Path,
    *,
    max_s1_rows: int | None = None,
    progress_every: int = 25_000,
) -> dict:
    """Union S2+S3 mmap baseline hits per S1 row; write challenge-format TSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    total_s1 = 0
    zero_cand = 0
    total_cand_pairs = 0
    max_cand = 0

    with MmapBaselineRetriever(index_root, cfg.baseline) as retriever:
        with output_path.open("w", encoding="utf-8", newline="") as out:
            writer = csv.writer(out, delimiter="\t", lineterminator="\n")
            writer.writerow(["source1_entity_id", "candidate_entity_ids"])

            for batch in _stream_s1_batches(cfg, max_s1_rows=max_s1_rows):
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
                            f"  mmap retrieval: {total_s1:,} S1 ({rate:,.0f}/s)",
                            file=sys.stderr,
                            flush=True,
                        )
                    writer.writerow([s1_id, ",".join(sorted(cands))])

    elapsed = round(time.perf_counter() - t0, 2)
    return {
        "engine": "mmap_baseline",
        "total_s1": total_s1,
        "s1_zero_candidates": zero_cand,
        "total_candidate_pairs": total_cand_pairs,
        "avg_candidates_per_s1": round(total_cand_pairs / total_s1, 2) if total_s1 else 0,
        "max_candidates_per_s1": max_cand,
        "output_tsv": str(output_path),
        "elapsed_sec": elapsed,
        "max_s1_rows": max_s1_rows,
    }
