"""Baseline retrieval against mmap CSR indexes (same caps as BaselineChannel)."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from analysis.retrieval.config import BaselineConfig
from analysis.retrieval.index import apply_candidate_cap
from analysis.retrieval.keys import baseline_keys
from analysis.retrieval.mmap_baseline.lookup import BaselineMmapStore, TargetMmapIndex


class MmapBaselineRetriever:
    """Drop-in semantic mirror of BaselineChannel batch retrieval (mmap backend)."""

    def __init__(self, index_root: Path, cfg: BaselineConfig | None = None):
        self.index_root = Path(index_root)
        self.cfg = cfg or BaselineConfig()
        self._store: BaselineMmapStore | None = None

    def open(self) -> None:
        if self._store is None:
            self._store = BaselineMmapStore(self.index_root)

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

    def __enter__(self) -> MmapBaselineRetriever:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def extract_query_keys(self, row: dict) -> list[str]:
        return baseline_keys(
            row.get("business_name", ""),
            row.get("business_address", ""),
            row.get("country", ""),
            prefix_len=self.cfg.name_prefix_len,
            min_prefix=self.cfg.min_prefix_len,
        )

    def _target(self, target: str) -> TargetMmapIndex:
        if self._store is None:
            raise RuntimeError("Call open() first")
        return self._store.target_index(target)

    def retrieve_batch(
        self,
        target: str,
        query_rows: Sequence[dict],
    ) -> dict[str, set[str]]:
        queries: list[tuple[str, list[str]]] = []
        for row in query_rows:
            s1_id = row.get("entity_id", "").strip()
            if not s1_id:
                continue
            queries.append((s1_id, self.extract_query_keys(row)))

        if not queries:
            return {}

        # cp_max_df filtering is SQLite-specific (key_df join); mmap path matches
        # unlimited CP behavior unless a sidecar is added later.
        raw = self._target(target).lookup_exact_batch(queries)
        cap = self.cfg.max_candidates_per_s1
        return {qid: apply_candidate_cap(hits, cap) for qid, hits in raw.items()}
