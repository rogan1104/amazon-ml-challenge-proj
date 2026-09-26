"""
Build a separate SQLite index for CP prefix-length analysis (cp3–cp8).

Analysis only — does not modify production index_train.sqlite or retrieval modules.

Usage (PC):
  python -m analysis.retrieval.build_cp_prefix_analysis_index
  python -m analysis.retrieval.build_cp_prefix_analysis_index --rebuild
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import time
from pathlib import Path

from analysis.config import CACHE, TRAIN
from analysis.retrieval.keys import _safe_str, name_prefix_key

PREFIX_LENGTHS = (3, 4, 5, 6, 7, 8)
MIN_PREFIX_LEN = 3
SCHEMA_VERSION = "1"
META_COMPLETE = "build_complete"

DEFAULT_DB = CACHE / "cp_prefix_analysis" / "index_cp_prefixes.sqlite"
DEFAULT_BATCH = 50_000


def _stream_tsv(path: Path, batch_size: int):
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


def cp_keys_for_entity(country: str, name: str) -> list[str]:
    """One posting key per prefix length (cp3: … cp8:), same norm as production cp."""
    c = _safe_str(country).strip()
    if not c:
        return []
    out: list[str] = []
    for n in PREFIX_LENGTHS:
        pfx = name_prefix_key(name, n)
        if len(pfx) >= MIN_PREFIX_LEN:
            out.append(f"cp{n}:{c}|{pfx}")
    return out


def _meta_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
    return row[0] if row else None


def _meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
        (key, value),
    )


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS postings (
            target TEXT NOT NULL,
            key TEXT NOT NULL,
            entity_id TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_postings_target_key
            ON postings (target, key);
        """
    )


def _is_complete(conn: sqlite3.Connection) -> bool:
    if _meta_get(conn, "schema_version") != SCHEMA_VERSION:
        return False
    return _meta_get(conn, META_COMPLETE) == "1"


def _print_stats(conn: sqlite3.Connection, elapsed_sec: float, db_path: Path) -> None:
    print("\n--- CP prefix analysis index stats ---")
    for target in ("S2", "S3"):
        row = conn.execute(
            "SELECT COUNT(DISTINCT entity_id) FROM postings WHERE target=?",
            (target,),
        ).fetchone()
        n_ent = row[0] if row else 0
        print(f"  rows indexed ({target} entities): {n_ent:,}")

    for n in PREFIX_LENGTHS:
        prefix = f"cp{n}:"
        for target in ("S2", "S3"):
            postings = conn.execute(
                """
                SELECT COUNT(*), COUNT(DISTINCT key)
                FROM postings
                WHERE target=? AND key LIKE ?
                """,
                (target, prefix + "%"),
            ).fetchone()
            post_n, uniq = postings or (0, 0)
            print(
                f"  {prefix} target={target} postings={post_n:,} unique_keys={uniq:,}"
            )

    size_mb = db_path.stat().st_size / (1024 * 1024) if db_path.is_file() else 0.0
    print(f"  build_time_sec: {elapsed_sec:.2f}")
    print(f"  db_size_mb: {size_mb:.2f}")
    print(f"  db_path: {db_path}")


def build_index(
    db_path: Path,
    batch_size: int = DEFAULT_BATCH,
    rebuild: bool = False,
) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.is_file() and not rebuild:
        conn = sqlite3.connect(db_path)
        try:
            _create_schema(conn)
            if _is_complete(conn):
                print(f"Analysis index already complete: {db_path}")
                print("Use --rebuild to replace it.")
                elapsed = float(_meta_get(conn, "build_elapsed_sec") or 0)
                _print_stats(conn, elapsed, db_path)
                return db_path
        finally:
            conn.close()

    if db_path.is_file():
        db_path.unlink()

    t0 = time.time()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _create_schema(conn)
        conn.execute("DELETE FROM postings")
        conn.execute("DELETE FROM meta")

        rows_per_target: dict[str, int] = {"S2": 0, "S3": 0}

        for target, tpath in (("S2", TRAIN["S2"]), ("S3", TRAIN["S3"])):
            if not tpath.is_file():
                raise FileNotFoundError(tpath)
            print(f"Indexing {target} from {tpath} ...", flush=True)
            for batch in _stream_tsv(tpath, batch_size):
                buffer: list[tuple[str, str, str]] = []
                for row in batch:
                    eid = row.get("entity_id", "").strip()
                    if not eid:
                        continue
                    rows_per_target[target] += 1
                    for key in cp_keys_for_entity(
                        row.get("country", ""),
                        row.get("business_name", ""),
                    ):
                        buffer.append((target, key, eid))
                if buffer:
                    conn.executemany(
                        "INSERT INTO postings (target, key, entity_id) VALUES (?, ?, ?)",
                        buffer,
                    )
            conn.commit()
            print(f"  {target} source rows: {rows_per_target[target]:,}", flush=True)

        elapsed = time.time() - t0
        _meta_set(conn, "schema_version", SCHEMA_VERSION)
        _meta_set(conn, META_COMPLETE, "1")
        _meta_set(conn, "build_elapsed_sec", str(round(elapsed, 2)))
        _meta_set(conn, "rows_S2", str(rows_per_target["S2"]))
        _meta_set(conn, "rows_S3", str(rows_per_target["S3"]))
        conn.commit()
        _print_stats(conn, elapsed, db_path)
    finally:
        conn.close()

    return db_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build CP prefix analysis index (cp3–cp8, train S2/S3 only)"
    )
    p.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="Output SQLite path (never touches retrieval_index/)",
    )
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and rebuild even if a completed index exists",
    )
    args = p.parse_args()
    build_index(args.db, batch_size=args.batch_size, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
