"""Disk-backed inverted index for O(1) key lookups (no S1×S2 cross product)."""
from __future__ import annotations

import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
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
# Legacy qid×key join (reference for equivalence tests only).
_SQL_BATCH_EXACT_QID_JOIN = """
    SELECT b.qid, p.entity_id
    FROM batch_query b
    INNER JOIN postings p
        ON p.channel = ? AND p.target = ? AND p.key = b.key
"""
# Key-centric: scan each unique key's postings once, fan out to qids in Python.
_SQL_KEYCENTRIC_POSTINGS = """
    SELECT u.key, p.entity_id
    FROM batch_unique_key u
    INNER JOIN postings p
        ON p.channel = ? AND p.target = ? AND p.key = u.key
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


def _blocking_key_family(key: str) -> str:
    if key.startswith("bn:"):
        return "bn"
    if key.startswith("ba:"):
        return "ba"
    if key.startswith("cp:"):
        return "cp"
    return "other"


@dataclass
class FanoutProfileStats:
    """Optional instrumentation for key-centric batch fan-out (default off)."""

    temp_table_sec: float = 0.0
    sql_fetch_sec: float = 0.0
    python_fanout_sec: float = 0.0
    unique_keys_queried: int = 0
    posting_rows_returned: int = 0
    fanout_add_operations: int = 0
    max_qids_sharing_one_key: int = 0
    fanout_ops_bn: int = 0
    fanout_ops_ba: int = 0
    fanout_ops_cp: int = 0
    fanout_ops_other: int = 0
    cp_unique_keys: int = 0
    cp_posting_rows: int = 0
    cp_qids_associated: int = 0
    cp_fanout_add_operations: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


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
        self._fanout_profile: FanoutProfileStats | None = None

    def enable_fanout_profiling(self) -> None:
        """Turn on timing/stats inside _fanout_keycentric_postings (no semantic change)."""
        self._fanout_profile = FanoutProfileStats()

    def reset_fanout_profiling(self) -> None:
        if self._fanout_profile is not None:
            self._fanout_profile = FanoutProfileStats()

    def get_fanout_profile(self) -> FanoutProfileStats | None:
        return self._fanout_profile

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

            CREATE TEMP TABLE IF NOT EXISTS batch_unique_key (
                key TEXT NOT NULL PRIMARY KEY
            );
        """)
        self._batch_table_ready = True

    def _fanout_keycentric_postings(
        self,
        channel: ChannelName,
        target: Target,
        key_to_qids: dict[str, set[str]],
        out: dict[str, set[str]],
    ) -> None:
        """Fetch postings once per unique key; fan entity_ids to all sharing qids."""
        if not key_to_qids:
            return

        ch, tg = channel.value, target
        unique_keys = list(key_to_qids.keys())
        key_chunk = 2000

        prof = self._fanout_profile
        for i in range(0, len(unique_keys), key_chunk):
            keys_slice = unique_keys[i : i + key_chunk]
            slice_map = {k: key_to_qids[k] for k in keys_slice}
            if prof is not None:
                if slice_map:
                    prof.max_qids_sharing_one_key = max(
                        prof.max_qids_sharing_one_key,
                        max(len(qs) for qs in slice_map.values()),
                    )
                cp_keys = [k for k in slice_map if k.startswith("cp:")]
                prof.cp_unique_keys += len(cp_keys)
                prof.cp_qids_associated += sum(len(slice_map[k]) for k in cp_keys)
                prof.unique_keys_queried += len(keys_slice)

                t_temp0 = time.perf_counter()
                with self._conn:
                    self._conn.execute("DELETE FROM batch_unique_key")
                    self._conn.executemany(
                        "INSERT OR IGNORE INTO batch_unique_key (key) VALUES (?)",
                        [(k,) for k in keys_slice],
                    )
                prof.temp_table_sec += time.perf_counter() - t_temp0

                t_fetch0 = time.perf_counter()
                rows = list(
                    self._conn.execute(_SQL_KEYCENTRIC_POSTINGS, (ch, tg)),
                )
                prof.sql_fetch_sec += time.perf_counter() - t_fetch0
                prof.posting_rows_returned += len(rows)

                t_fan0 = time.perf_counter()
                for key, eid in rows:
                    qids = slice_map.get(key, ())
                    n_q = len(qids)
                    prof.fanout_add_operations += n_q
                    fam = _blocking_key_family(key)
                    if fam == "bn":
                        prof.fanout_ops_bn += n_q
                    elif fam == "ba":
                        prof.fanout_ops_ba += n_q
                    elif fam == "cp":
                        prof.fanout_ops_cp += n_q
                        prof.cp_posting_rows += 1
                        prof.cp_fanout_add_operations += n_q
                    else:
                        prof.fanout_ops_other += n_q
                    for qid in qids:
                        out[qid].add(eid)
                prof.python_fanout_sec += time.perf_counter() - t_fan0
            else:
                with self._conn:
                    self._conn.execute("DELETE FROM batch_unique_key")
                    self._conn.executemany(
                        "INSERT OR IGNORE INTO batch_unique_key (key) VALUES (?)",
                        [(k,) for k in keys_slice],
                    )
                    for key, eid in self._conn.execute(
                        _SQL_KEYCENTRIC_POSTINGS, (ch, tg),
                    ):
                        for qid in slice_map.get(key, ()):
                            out[qid].add(eid)

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
        Batch exact lookup: key-centric — one postings scan per unique blocking key.

        queries: (query_id, keys) — typically S1 entity_id and blocking keys.
        """
        if not queries:
            return {}

        self._ensure_batch_table()
        out: dict[str, set[str]] = {qid: set() for qid, _ in queries}

        sub_batch = 2500
        for start in range(0, len(queries), sub_batch):
            chunk = queries[start : start + sub_batch]
            key_to_qids: dict[str, set[str]] = defaultdict(set)
            for qid, keys in chunk:
                for k in {k for k in keys if k}:
                    key_to_qids[k].add(qid)
            if not key_to_qids:
                continue

            self._fanout_keycentric_postings(channel, target, key_to_qids, out)

        return out

    def lookup_keys_exact_batch_qid_join(
        self,
        channel: ChannelName,
        target: Target,
        queries: Sequence[tuple[str, Sequence[str]]],
    ) -> dict[str, set[str]]:
        """Legacy qid×key join (equivalence tests only)."""
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
                for qid, eid in self._conn.execute(
                    _SQL_BATCH_EXACT_QID_JOIN, (ch, tg),
                ):
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
