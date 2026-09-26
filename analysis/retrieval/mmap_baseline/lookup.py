"""Mmap CSR exact key lookup (no full in-memory inverted dict)."""
from __future__ import annotations

import bisect
import mmap
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from analysis.retrieval.mmap_baseline.schema import (
    FAMILIES,
    KEYS_IDX,
    KEYS_STR,
    POSTS_EID,
    POSTS_IDX,
    ENTITY_IDS,
    family_dir,
    filter_keys_by_family,
    target_dir,
)

class CSRFamilyIndex:
    """One family (bn/ba/cp): sorted keys -> contiguous uint32 entity rowids."""

    def __init__(self, family_path: Path):
        self.path = Path(family_path)
        self._keys_str_path = self.path / KEYS_STR
        self._keys_idx_path = self.path / KEYS_IDX
        self._posts_eid_path = self.path / POSTS_EID
        self._posts_idx_path = self.path / POSTS_IDX

        for p in (self._keys_str_path, self._keys_idx_path, self._posts_idx_path, self._posts_eid_path):
            if not p.is_file():
                raise FileNotFoundError(f"Missing CSR file: {p}")

        raw_idx = self._keys_idx_path.read_bytes()
        n_off = len(raw_idx) // 8
        self._keys_off = np.frombuffer(raw_idx, dtype=np.uint64, count=n_off)
        self.n_keys = max(len(self._keys_off) - 1, 0)

        raw_posts_idx = self._posts_idx_path.read_bytes()
        self._posts_off = np.frombuffer(raw_posts_idx, dtype=np.uint64)

        self._keys_str_fh = None
        self._keys_str_mm = None
        self._posts_eid_fh = None
        self._posts_eid_mm = None
        self._key_bytes: list[bytes] = []

        if self.n_keys > 0 and self._keys_str_path.stat().st_size > 0:
            self._keys_str_fh = self._keys_str_path.open("rb")
            self._keys_str_mm = mmap.mmap(
                self._keys_str_fh.fileno(), 0, access=mmap.ACCESS_READ,
            )
            for i in range(self.n_keys):
                start = int(self._keys_off[i])
                end = int(self._keys_off[i + 1])
                self._key_bytes.append(self._keys_str_mm[start:end])

        if self._posts_eid_path.stat().st_size > 0:
            self._posts_eid_fh = self._posts_eid_path.open("rb")
            self._posts_eid_mm = mmap.mmap(
                self._posts_eid_fh.fileno(), 0, access=mmap.ACCESS_READ,
            )

        self.posting_count = int(self._posts_off[-1]) if len(self._posts_off) else 0
        self.key_count = self.n_keys

    def close(self) -> None:
        if self._keys_str_mm is not None:
            self._keys_str_mm.close()
        if self._posts_eid_mm is not None:
            self._posts_eid_mm.close()
        if self._keys_str_fh is not None:
            self._keys_str_fh.close()
        if self._posts_eid_fh is not None:
            self._posts_eid_fh.close()

    def __enter__(self) -> CSRFamilyIndex:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _key_at(self, i: int) -> bytes:
        return self._key_bytes[i]

    def find_key_index(self, key: str) -> int | None:
        key_b = key.encode("utf-8")
        idx = bisect.bisect_left(self._key_bytes, key_b)
        if idx < self.n_keys and self._key_bytes[idx] == key_b:
            return idx
        return None

    def postings_rowids_array(self, key: str) -> np.ndarray:
        ki = self.find_key_index(key)
        if ki is None or self._posts_eid_mm is None:
            return np.empty(0, dtype=np.uint32)
        start = int(self._posts_off[ki]) * 4
        end = int(self._posts_off[ki + 1]) * 4
        mv = memoryview(self._posts_eid_mm)[start:end]
        return np.frombuffer(mv, dtype=np.uint32).copy()

    def lookup_keys_to_rowids(self, keys: Sequence[str]) -> set[int]:
        found: set[int] = set()
        for key in {k for k in keys if k}:
            arr = self.postings_rowids_array(key)
            if arr.size:
                found.update(int(x) for x in arr)
        return found


class TargetMmapIndex:
    """S2 or S3 partition: entity table + bn/ba/cp CSR indexes."""

    def __init__(self, index_root: Path, target: str):
        self.target = target
        tdir = target_dir(index_root, target)  # type: ignore[arg-type]
        epath = tdir / ENTITY_IDS
        if not epath.is_file():
            raise FileNotFoundError(epath)
        self.entity_ids: list[str] = epath.read_text(encoding="utf-8").splitlines()
        self._families: dict[str, CSRFamilyIndex] = {}
        for fam in FAMILIES:
            fpath = family_dir(index_root, target, fam)  # type: ignore[arg-type]
            if fpath.is_dir():
                self._families[fam] = CSRFamilyIndex(fpath)

    def close(self) -> None:
        for idx in self._families.values():
            idx.close()

    def __enter__(self) -> TargetMmapIndex:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def rowids_to_entity_ids(self, rowids: set[int]) -> set[str]:
        out: set[str] = set()
        n = len(self.entity_ids)
        for rid in rowids:
            if 0 <= rid < n:
                out.add(self.entity_ids[rid])
        return out

    def lookup_exact_batch(
        self,
        queries: Sequence[tuple[str, Sequence[str]]],
    ) -> dict[str, set[str]]:
        """
        Match SQLite key-centric fan-out: union entity_ids per query_id across keys.
        """
        if not queries:
            return {}

        out: dict[str, set[str]] = {qid: set() for qid, _ in queries}

        for fam, fam_index in self._families.items():
            key_to_qids: dict[str, set[str]] = defaultdict(set)
            for qid, keys in queries:
                for key in filter_keys_by_family(keys, fam):
                    key_to_qids[key].add(qid)

            for key, qids in key_to_qids.items():
                arr = fam_index.postings_rowids_array(key)
                if arr.size == 0:
                    continue
                entity_ids = self.entity_ids
                for rid in arr:
                    eid = entity_ids[int(rid)]
                    for qid in qids:
                        out[qid].add(eid)

        return out


class BaselineMmapStore:
    """Both targets for one split."""

    def __init__(self, index_root: Path):
        self.index_root = Path(index_root)
        self.s2 = TargetMmapIndex(self.index_root, "S2")
        self.s3 = TargetMmapIndex(self.index_root, "S3")

    def close(self) -> None:
        self.s2.close()
        self.s3.close()

    def __enter__(self) -> BaselineMmapStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def target_index(self, target: str) -> TargetMmapIndex:
        if target == "S2":
            return self.s2
        if target == "S3":
            return self.s3
        raise ValueError(f"Unknown target {target!r}")
