"""Memory-mapped CSR baseline inverted index (parallel to SQLite; same keys)."""

from analysis.retrieval.mmap_baseline.schema import SCHEMA_VERSION, BaselineMmapManifest

__all__ = ["SCHEMA_VERSION", "BaselineMmapManifest"]
