"""Build disk-backed inverted indexes from TSV sources (streaming, chunked)."""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Iterator

from analysis.retrieval.channels import RetrievalChannel, active_channels
from analysis.retrieval.config import ChannelName, RetrievalConfig, Target
from analysis.retrieval.index import InvertedIndex


def _stream_tsv(path: Path, batch_size: int) -> Iterator[list[dict]]:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        batch: list[dict] = []
        for row in reader:
            batch.append(row)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def _index_path(cfg: RetrievalConfig) -> Path:
    return cfg.index_dir / f"index_{cfg.split}.sqlite"


def build_indexes(cfg: RetrievalConfig, channels: list[RetrievalChannel] | None = None) -> Path:
    """
    Stream S2 and S3 TSV files; build inverted indexes per channel per target.
    Does not read S1 (query-only at retrieval time).
    """
    channels = channels or active_channels(cfg)
    paths = cfg.paths()
    db_path = _index_path(cfg)
    t0 = time.time()

    with InvertedIndex(db_path) as index:
        for target in ("S2", "S3"):
            tpath = paths[target]
            for channel in channels:
                index.clear_channel(channel.name, target)  # type: ignore[arg-type]

                total_postings = 0
                for batch in _stream_tsv(tpath, cfg.batch_size):
                    rows: list[tuple[str, str]] = []
                    for row in batch:
                        eid = row.get("entity_id", "").strip()
                        if not eid:
                            continue
                        for key in channel.extract_index_keys(row):
                            rows.append((key, eid))
                    total_postings += index.add_postings_batch(
                        channel.name, target, rows  # type: ignore[arg-type]
                    )

                index.finalize_df(channel.name, target)  # type: ignore[arg-type]
                st = index.stats(channel.name, target)  # type: ignore[arg-type]
                print(
                    f"  [{channel.name.value}/{target}] "
                    f"postings={st['postings']:,} keys={st['keys']:,} entities={st['entities']:,}"
                )

        index.set_meta("split", cfg.split)
        index.set_meta("build_elapsed_sec", str(round(time.time() - t0, 2)))
        index.set_meta(
            "channels",
            ",".join(c.name.value for c in channels),
        )

    print(f"Index built: {db_path} ({time.time()-t0:.1f}s)")
    return db_path
