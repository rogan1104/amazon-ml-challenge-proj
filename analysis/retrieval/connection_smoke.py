"""Minimal read-only index + TEMP batch_query connection smoke test."""
from __future__ import annotations

import sys
from pathlib import Path

from analysis.retrieval.config import CACHE, ChannelName
from analysis.retrieval.index import InvertedIndex, readonly_sqlite_uri


def default_index_path() -> Path:
    return CACHE / "retrieval_index" / "index_train.sqlite"


def run_smoke(index_path: Path | None = None) -> int:
    index_path = index_path or default_index_path()
    if not index_path.is_file():
        print(f"SKIP: index not found: {index_path}", file=sys.stderr)
        return 2

    uri = readonly_sqlite_uri(index_path)
    print(f"readonly URI: {uri}")

    with InvertedIndex(index_path, read_only=True) as index:
        postings = index._postings
        n = index._conn.execute(f"SELECT COUNT(*) FROM {postings}").fetchone()[0]
        print(f"postings rows: {n:,}")

        index._ensure_batch_table()
        batch = index.lookup_keys_exact_batch(
            ChannelName.BASELINE,
            "S2",
            [("smoke-s1", ["__nonexistent_key__"])],
        )
        assert batch.get("smoke-s1") == set(), batch

    print("connection smoke: OK")
    return 0


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_index_path()
    raise SystemExit(run_smoke(path))


if __name__ == "__main__":
    main()
