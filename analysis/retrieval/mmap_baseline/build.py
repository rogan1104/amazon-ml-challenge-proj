"""Build mmap CSR baseline indexes by streaming S2/S3 TSV (same keys as SQLite path)."""
from __future__ import annotations

import csv
import heapq
import shutil
import time
from pathlib import Path
from typing import Iterator, Literal

import numpy as np

from analysis.retrieval.config import BaselineConfig, Split
from analysis.retrieval.keys import baseline_keys
from analysis.retrieval.mmap_baseline.schema import (
    SCHEMA_VERSION,
    FAMILIES,
    BaselineMmapManifest,
    FamilyStats,
    TargetStats,
    family_dir,
    manifest_path,
    read_manifest,
    target_dir,
    write_manifest,
)

Target = Literal["S2", "S3"]
_SORT_CHUNK_LINES = 400_000


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


def _append_posting_lines(
    handle,
    rowid: int,
    keys: list[str],
) -> None:
    for key in keys:
        handle.write(f"{key}\t{rowid}\n")


def _sort_postings_file(raw_path: Path, sorted_path: Path) -> None:
    """External sort of key\\trowid lines (UTF-8 keys, stable for equal keys)."""
    run_paths: list[Path] = []
    chunk: list[str] = []

    with raw_path.open("r", encoding="utf-8") as inp:
        for line in inp:
            chunk.append(line)
            if len(chunk) >= _SORT_CHUNK_LINES:
                chunk.sort()
                run = sorted_path.with_suffix(f".run{len(run_paths)}.tmp")
                run.write_text("".join(chunk), encoding="utf-8")
                run_paths.append(run)
                chunk = []

    if chunk:
        chunk.sort()
        if not run_paths:
            sorted_path.write_text("".join(chunk), encoding="utf-8")
            return
        run = sorted_path.with_suffix(f".run{len(run_paths)}.tmp")
        run.write_text("".join(chunk), encoding="utf-8")
        run_paths.append(run)

    if not run_paths:
        sorted_path.write_text("", encoding="utf-8")
        return

    with sorted_path.open("w", encoding="utf-8") as out:
        handles = [p.open("r", encoding="utf-8") for p in run_paths]
        try:
            heap: list[tuple[str, int]] = []
            for i, h in enumerate(handles):
                line = h.readline()
                if line:
                    heapq.heappush(heap, (line, i))
            while heap:
                line, idx = heapq.heappop(heap)
                out.write(line)
                nxt = handles[idx].readline()
                if nxt:
                    heapq.heappush(heap, (nxt, idx))
        finally:
            for h in handles:
                h.close()
            for p in run_paths:
                p.unlink(missing_ok=True)


def _write_csr_from_sorted(sorted_path: Path, family_out: Path) -> FamilyStats:
    family_out.mkdir(parents=True, exist_ok=True)
    keys_blob = bytearray()
    keys_off: list[int] = [0]
    posts_eid: list[int] = []
    posts_off: list[int] = [0]

    prev_key: str | None = None
    key_count = 0
    posting_count = 0

    def flush_key(key: str, rowids: list[int]) -> None:
        nonlocal key_count, posting_count
        key_count += 1
        kb = key.encode("utf-8")
        keys_blob.extend(kb)
        keys_off.append(len(keys_blob))
        rowids.sort()
        posts_eid.extend(rowids)
        posts_off.append(len(posts_eid))
        posting_count += len(rowids)

    current_key: str | None = None
    current_rows: list[int] = []

    with sorted_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            key, _, rid_s = line.partition("\t")
            rid = int(rid_s)
            if current_key is None:
                current_key = key
                current_rows = [rid]
            elif key == current_key:
                current_rows.append(rid)
            else:
                flush_key(current_key, current_rows)
                current_key = key
                current_rows = [rid]

    if current_key is not None:
        flush_key(current_key, current_rows)

    # Empty family
    if key_count == 0:
        (family_out / "keys.str").write_bytes(b"")
        np.zeros(1, dtype=np.uint64).tofile(family_out / "keys.idx")
        np.zeros(1, dtype=np.uint64).tofile(family_out / "posts.idx")
        (family_out / "posts.eid").write_bytes(b"")
        return FamilyStats(key_count=0, posting_count=0)

    (family_out / "keys.str").write_bytes(bytes(keys_blob))
    np.asarray(keys_off, dtype=np.uint64).tofile(family_out / "keys.idx")
    np.asarray(posts_off, dtype=np.uint64).tofile(family_out / "posts.idx")
    np.asarray(posts_eid, dtype=np.uint32).tofile(family_out / "posts.eid")

    return FamilyStats(key_count=key_count, posting_count=posting_count)


def build_target(
    tsv_path: Path,
    index_root: Path,
    target: Target,
    cfg: BaselineConfig,
    *,
    batch_size: int = 50_000,
    entity_limit: int | None = None,
) -> TargetStats:
    """Stream one S2 or S3 source file; write entity_ids + per-family CSR."""
    tdir = target_dir(index_root, target)
    tdir.mkdir(parents=True, exist_ok=True)
    entity_path = tdir / "entity_ids.txt"

    tmp_dir = tdir / ".build_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    raw_handles = {
        fam: (tmp_dir / f"{fam}.postings.raw").open("w", encoding="utf-8")
        for fam in FAMILIES
    }

    entity_count = 0
    try:
        with entity_path.open("w", encoding="utf-8", newline="\n") as ent_out:
            for batch in _stream_tsv(tsv_path, batch_size):
                for row in batch:
                    if entity_limit is not None and entity_count >= entity_limit:
                        break
                    eid = row.get("entity_id", "").strip()
                    if not eid:
                        continue
                    keys = baseline_keys(
                        row.get("business_name", ""),
                        row.get("business_address", ""),
                        row.get("country", ""),
                        prefix_len=cfg.name_prefix_len,
                        min_prefix=cfg.min_prefix_len,
                    )
                    ent_out.write(f"{eid}\n")
                    for fam in FAMILIES:
                        fam_keys = [k for k in keys if k.startswith(f"{fam}:")]
                        if fam_keys:
                            _append_posting_lines(raw_handles[fam], entity_count, fam_keys)
                    entity_count += 1
                if entity_limit is not None and entity_count >= entity_limit:
                    break
    finally:
        for h in raw_handles.values():
            h.close()

    family_stats: dict[str, FamilyStats] = {}
    for fam in FAMILIES:
        raw_path = tmp_dir / f"{fam}.postings.raw"
        sorted_path = tmp_dir / f"{fam}.postings.sorted"
        if not raw_path.is_file() or raw_path.stat().st_size == 0:
            sorted_path.write_text("", encoding="utf-8")
            family_stats[fam] = _write_csr_from_sorted(
                sorted_path, family_dir(index_root, target, fam),
            )
            raw_path.unlink(missing_ok=True)
            continue
        _sort_postings_file(raw_path, sorted_path)
        family_stats[fam] = _write_csr_from_sorted(sorted_path, family_dir(index_root, target, fam))
        raw_path.unlink(missing_ok=True)
        sorted_path.unlink(missing_ok=True)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    return TargetStats(entity_count=entity_count, families=family_stats)


def build_baseline_mmap_index(
    index_root: Path,
    *,
    split: Split = "train",
    paths: dict[str, Path],
    cfg: BaselineConfig | None = None,
    batch_size: int = 50_000,
    entity_limit: int | None = None,
    rebuild: bool = False,
) -> BaselineMmapManifest:
    """
    Build S2 and S3 mmap indexes under index_root.

    Idempotent: skips if manifest.complete and config matches (unless rebuild=True).
    """
    cfg = cfg or BaselineConfig()
    index_root = Path(index_root)

    existing = read_manifest(index_root)
    if (
        not rebuild
        and existing is not None
        and existing.complete
        and existing.schema_version == SCHEMA_VERSION
        and BaselineMmapManifest.matches_config(existing, cfg)
        and existing.split == split
        and existing.entity_limit == entity_limit
    ):
        return existing

    manifest = BaselineMmapManifest(
        schema_version=SCHEMA_VERSION,
        split=split,
        baseline_config=BaselineMmapManifest.baseline_config_dict(cfg),
        targets={},
        complete=False,
        entity_limit=entity_limit,
    )
    write_manifest(index_root, manifest)

    t0 = time.perf_counter()
    for target in ("S2", "S3"):
        print(f"[mmap_baseline] building {target} from {paths[target]} ...")
        stats = build_target(
            paths[target],
            index_root,
            target,  # type: ignore[arg-type]
            cfg,
            batch_size=batch_size,
            entity_limit=entity_limit,
        )
        manifest.targets[target] = stats
        write_manifest(index_root, manifest)
        print(
            f"[mmap_baseline] {target}: entities={stats.entity_count:,} "
            f"bn_keys={stats.families.get('bn', FamilyStats()).key_count:,} "
            f"ba_keys={stats.families.get('ba', FamilyStats()).key_count:,} "
            f"cp_keys={stats.families.get('cp', FamilyStats()).key_count:,}"
        )

    manifest.complete = True
    write_manifest(index_root, manifest)
    print(f"[mmap_baseline] complete in {time.perf_counter() - t0:.1f}s -> {index_root}")
    return manifest
