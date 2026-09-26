"""Disk-backed inverted index for O(1) key lookups (no S1×S2 cross product)."""
from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from analysis.retrieval.config import ChannelName, Target

# Reused SQL strings (stable object identity helps SQLite statement cache).
_SQL_EXACT_IN = """
    SELECT DISTINCT entity_id FROM postings
    WHERE channel=? AND target=? AND key IN ({placeholders})
"""
_SQL_SCORED_IN = """
    SELECT entity_id FROM postings
    WHERE channel=? AND target=? AND key IN ({placeholders})
"""
_SQL_SCORED_MAXDF = """
    SELECT p.entity_id
    FROM postings p
    JOIN key_df d ON p.channel=d.channel AND p.target=d.target AND p.key=d.key
    WHERE p.channel=? AND p.target=? AND p.key IN ({placeholders})
      AND d.df <= ?
"""
_SQL_BATCH_EXACT = """
    SELECT b.qid, p.entity_id
    FROM batch_query b
    INNER JOIN postings p
        ON p.channel = ? AND p.target = ? AND p.key = b.key
"""
_SQL_BATCH_SCORED = """
    SELECT b.qid, p.entity_id
    FROM batch_query b
    INNER JOIN postings p
        ON p.channel = ? AND p.target = ? AND p.key = b.key
"""
_SQL_BATCH_SCORED_MAXDF = """
    SELECT b.qid, p.entity_id
    FROM batch_query b
    INNER JOIN postings p
        ON p.channel = ? AND p.target = ? AND p.key = b.key
    INNER JOIN key_df d
        ON d.channel = p.channel AND d.target = p.target AND d.key = p.key
    WHERE d.df <= ?
"""


def open_index_connection(db_path: Path, *, read_only: bool) -> sqlite3.Connection:
    """
    Open the on-disk index.

    read_only=True is retrieval mode: only SELECT on index tables, but the
    connection must allow TEMP tables (batch_query). SQLite mode=ro forbids
    TEMP DDL, so we use a normal file connection and never write index tables.
    """
    resolved = db_path.resolve()
    if read_only and not resolved.is_file():
        raise FileNotFoundError(resolved)

    # Plain path string — reliable on Windows (no ATTACH / URI edge cases).
    conn = sqlite3.connect(str(resolved))

    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=268435456")
    conn.execute("PRAGMA cache_size=-256000")

    if not read_only:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

    return conn


class InvertedIndex:
    """
    SQLite inverted index: (channel, target, key) -> entity_id postings.

    Stored on disk so indexes for millions of rows fit in bounded RAM during build.
    """

    def __init__(self, db_path: Path, *, read_only: bool = False):
        self.db_path = Path(db_path)
        self._read_only = read_only
        if not read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = open_index_connection(self.db_path, read_only=read_only)
        if not read_only:
            self._create_schema()
        self._batch_table_ready = False

    def _assert_writable(self) -> None:
        if self._read_only:
            raise RuntimeError("Index is open read_only for retrieval; cannot modify index tables.")

    def _create_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS postings (
                channel TEXT NOT NULL,
                target  TEXT NOT NULL,
                key     TEXT NOT NULL,
                entity_id TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_postings_lookup
                ON postings (channel, target, key);
            CREATE INDEX IF NOT EXISTS idx_postings_entity
                ON postings (channel, target, entity_id);

            CREATE TABLE IF NOT EXISTS key_df (
                channel TEXT NOT NULL,
                target  TEXT NOT NULL,
                key     TEXT NOT NULL,
                df      INTEGER NOT NULL,
                PRIMARY KEY (channel, target, key)
            );

            CREATE TABLE IF NOT EXISTS meta (
                k TEXT PRIMARY KEY,
                v TEXT NOT NULL
            );
        """)
        self._conn.commit()

    def _ensure_batch_table(self) -> None:
        if self._batch_table_ready:
            return
        self._conn.executescript("""
            CREATE TEMP TABLE IF NOT EXISTS batch_query (
                qid TEXT NOT NULL,
                key TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_batch_query_key ON batch_query (key);
        """)
        self._batch_table_ready = True

    def clear_channel(self, channel: ChannelName, target: Target) -> None:
        self._assert_writable()
        ch, tg = channel.value, target
        self._conn.execute("DELETE FROM postings WHERE channel=? AND target=?", (ch, tg))
        self._conn.execute("DELETE FROM key_df WHERE channel=? AND target=?", (ch, tg))
        self._conn.commit()

    def add_postings_batch(
        self,
        channel: ChannelName,
        target: Target,
        rows: Iterable[tuple[str, str]],
    ) -> int:
        """Insert (key, entity_id) pairs. Returns rows inserted."""
        self._assert_writable()
        ch, tg = channel.value, target
        data = [(ch, tg, k, eid) for k, eid in rows if k and eid]
        if not data:
            return 0
        self._conn.executemany(
            "INSERT INTO postings (channel, target, key, entity_id) VALUES (?,?,?,?)",
            data,
        )
        self._conn.commit()
        return len(data)

    def finalize_df(self, channel: ChannelName, target: Target) -> None:
        """Compute document frequency per key (for stop-key filtering)."""
        self._assert_writable()
        ch, tg = channel.value, target
        self._conn.execute("DELETE FROM key_df WHERE channel=? AND target=?", (ch, tg))
        self._conn.execute("""
            INSERT INTO key_df (channel, target, key, df)
            SELECT channel, target, key, COUNT(DISTINCT entity_id)
            FROM postings
            WHERE channel=? AND target=?
            GROUP BY channel, target, key
        """, (ch, tg))
        self._conn.commit()

    def iter_entity_keys(
        self,
        channel: ChannelName,
        target: Target,
        entity_id: str,
    ) -> list[str]:
        ch, tg = channel.value, target
        cur = self._conn.execute(
            "SELECT key FROM postings WHERE channel=? AND target=? AND entity_id=?",
            (ch, tg, entity_id),
        )
        return [r[0] for r in cur.fetchall()]

    def lookup_keys(
        self,
        channel: ChannelName,
        target: Target,
        keys: Iterable[str],
        max_df: int | None = None,
    ) -> Counter:
        """
        Return entity_id -> hit count for all keys (accumulated overlap score).
        Skips keys with df > max_df when max_df is set.
        """
        ch, tg = channel.value, target
        key_list = list({k for k in keys if k})
        if not key_list:
            return Counter()

        scores: Counter = Counter()
        chunk = 500
        for i in range(0, len(key_list), chunk):
            batch = key_list[i : i + chunk]
            placeholders = ",".join("?" * len(batch))
            params: list = [ch, tg, *batch]

            if max_df is not None:
                sql = _SQL_SCORED_MAXDF.format(placeholders=placeholders)
                params.append(max_df)
            else:
                sql = _SQL_SCORED_IN.format(placeholders=placeholders)

            for (eid,) in self._conn.execute(sql, params):
                scores[eid] += 1
        return scores

    def lookup_keys_exact(
        self,
        channel: ChannelName,
        target: Target,
        keys: Iterable[str],
    ) -> set[str]:
        """Union of entity_ids for exact key hits (baseline channel)."""
        ch, tg = channel.value, target
        key_list = list({k for k in keys if k})
        if not key_list:
            return set()

        found: set[str] = set()
        chunk = 500
        for i in range(0, len(key_list), chunk):
            batch = key_list[i : i + chunk]
            placeholders = ",".join("?" * len(batch))
            params = [ch, tg, *batch]
            sql = _SQL_EXACT_IN.format(placeholders=placeholders)
            for (eid,) in self._conn.execute(sql, params):
                found.add(eid)
        return found

    def lookup_keys_exact_batch(
        self,
        channel: ChannelName,
        target: Target,
        queries: Sequence[tuple[str, Sequence[str]]],
    ) -> dict[str, set[str]]:
        """
        Batch exact lookup: many S1 rows in one SQLite join per sub-batch.

        queries: (query_id, keys) — typically S1 entity_id and blocking keys.
        """
        if not queries:
            return {}

        ch, tg = channel.value, target
        self._ensure_batch_table()
        out: dict[str, set[str]] = {qid: set() for qid, _ in queries}

        sub_batch = 2500
        for start in range(0, len(queries), sub_batch):
            chunk = queries[start : start + sub_batch]
            pairs: list[tuple[str, str]] = []
            for qid, keys in chunk:
                for k in {k for k in keys if k}:
                    pairs.append((qid, k))
            if not pairs:
                continue

            with self._conn:
                self._conn.execute("DELETE FROM batch_query")
                self._conn.executemany(
                    "INSERT INTO batch_query (qid, key) VALUES (?, ?)",
                    pairs,
                )
                for qid, eid in self._conn.execute(_SQL_BATCH_EXACT, (ch, tg)):
                    out[qid].add(eid)

        return out

    def lookup_keys_batch(
        self,
        channel: ChannelName,
        target: Target,
        queries: Sequence[tuple[str, Sequence[str]]],
        max_df: int | None = None,
    ) -> dict[str, Counter]:
        """Batch scored lookup (overlap counts) for many S1 rows at once."""
        if not queries:
            return {}

        ch, tg = channel.value, target
        self._ensure_batch_table()
        out: dict[str, Counter] = {qid: Counter() for qid, _ in queries}

        sub_batch = 1500
        for start in range(0, len(queries), sub_batch):
            chunk = queries[start : start + sub_batch]
            pairs: list[tuple[str, str]] = []
            for qid, keys in chunk:
                for k in {k for k in keys if k}:
                    pairs.append((qid, k))
            if not pairs:
                continue

            with self._conn:
                self._conn.execute("DELETE FROM batch_query")
                self._conn.executemany(
                    "INSERT INTO batch_query (qid, key) VALUES (?, ?)",
                    pairs,
                )
                if max_df is not None:
                    sql = _SQL_BATCH_SCORED_MAXDF
                    params = (ch, tg, max_df)
                else:
                    sql = _SQL_BATCH_SCORED
                    params = (ch, tg)
                for qid, eid in self._conn.execute(sql, params):
                    out[qid][eid] += 1

        return out

    def set_meta(self, key: str, value: str) -> None:
        self._assert_writable()
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
        return row[0] if row else None

    def stats(self, channel: ChannelName, target: Target) -> dict:
        ch, tg = channel.value, target
        keys = self._conn.execute(
            "SELECT COUNT(*) FROM key_df WHERE channel=? AND target=?",
            (ch, tg),
        ).fetchone()[0]
        postings = self._conn.execute(
            "SELECT COUNT(*) FROM postings WHERE channel=? AND target=?",
            (ch, tg),
        ).fetchone()[0]
        entities = self._conn.execute(
            "SELECT COUNT(DISTINCT entity_id) FROM postings WHERE channel=? AND target=?",
            (ch, tg),
        ).fetchone()[0]
        return {"keys": keys, "postings": postings, "entities": entities}

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "InvertedIndex":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def select_top_candidates(
    scores: Counter,
    min_overlap: int = 1,
    max_candidates: int | None = None,
) -> set[str]:
    """Filter by minimum overlap and optional per-S1 cap (highest scores first)."""
    if not scores:
        return set()
    filtered = {eid for eid, s in scores.items() if s >= min_overlap}
    if max_candidates is None or len(filtered) <= max_candidates:
        return filtered
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:max_candidates]
    return {eid for eid, _ in ranked}


def apply_candidate_cap(hits: set[str], max_candidates: int | None) -> set[str]:
    """Match BaselineChannel cap semantics (insertion order from SQL iteration)."""
    if max_candidates is None or len(hits) <= max_candidates:
        return hits
    return set(list(hits)[:max_candidates])
