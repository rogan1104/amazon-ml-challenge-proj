"""Run candidate retrieval: union channels, separate S2/S3, deduplicate."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Iterator

from analysis.retrieval.build import _index_path, _stream_tsv
from analysis.retrieval.channels import RetrievalChannel, active_channels
from analysis.retrieval.config import RetrievalConfig, Target


def retrieve_for_s1(
    index_path: Path,
    cfg: RetrievalConfig,
    channels: list[RetrievalChannel] | None = None,
) -> Iterator[tuple[str, set[str]]]:
    """
    Yield (s1_entity_id, deduplicated candidate entity_ids) for each S1 row.
    S2 and S3 candidates are retrieved separately then unioned.
    """
    from analysis.retrieval.index import InvertedIndex

    channels = channels or active_channels(cfg)
    s1_path = cfg.paths()["S1"]

    with InvertedIndex(index_path) as index:
        for batch in _stream_tsv(s1_path, cfg.batch_size):
            for row in batch:
                s1_id = row.get("entity_id", "").strip()
                if not s1_id:
                    continue

                candidates: set[str] = set()
                for target in ("S2", "S3"):
                    for channel in channels:
                        hits = channel.retrieve(index, target, row)  # type: ignore[arg-type]
                        candidates |= hits

                yield s1_id, candidates


def write_candidate_pairs(
    cfg: RetrievalConfig,
    index_path: Path | None = None,
    output_path: Path | None = None,
    channels: list[RetrievalChannel] | None = None,
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
    t0 = time.time()
    total_s1 = 0
    zero_cand = 0
    total_cand_pairs = 0
    max_cand = 0

    with open(output_path, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for s1_id, cands in retrieve_for_s1(index_path, cfg, channels):
            total_s1 += 1
            n = len(cands)
            total_cand_pairs += n
            max_cand = max(max_cand, n)
            if n == 0:
                zero_cand += 1
            writer.writerow([s1_id, ",".join(sorted(cands))])

    elapsed = round(time.time() - t0, 2)
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
    }

    metrics_path = cfg.metrics_json()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
