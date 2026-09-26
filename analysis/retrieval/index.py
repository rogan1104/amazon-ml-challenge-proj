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


class InvertedIndex:
    """
    SQLite inverted index: (channel, target, key) -> entity_id postings.

    Stored on disk so indexes for millions of rows fit in bounded RAM during build.

    Read-only retrieval uses an in-memory connection with the on-disk index ATTACHed
    read-only, so TEMP batch_query tables never write to index_train.sqlite.
    """

    _ATTACH_ALIAS = "idx"

    def __init__(self, db_path: Path, *, read_only: bool = False):
        self.db_path = Path(db_path)
        self._read_only = read_only
        if read_only and self.db_path.exists():
            self._conn = sqlite3.connect(":memory:")
            disk_uri = f"file:{self.db_path.resolve().as_posix()}?mode=ro"
            self._conn.execute(
                f"ATTACH DATABASE '{disk_uri}' AS {self._ATTACH_ALIAS}",
            )
            pfx = f"{self._ATTACH_ALIAS}."
            self._postings = f"{pfx}postings"
            self._key_df = f"{pfx}key_df"
            self._meta = f"{pfx}meta"
            self._conn.execute("PRAGMA temp_store=MEMORY")
            self._conn.execute(f"PRAGMA {self._ATTACH_ALIAS}.mmap_size=268435456")
            self._conn.execute(f"PRAGMA {self._ATTACH_ALIAS}.cache_size=-256000")
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._postings = "postings"
            self._key_df = "key_df"
            self._meta = "meta"
            self._conn.execute("PRAGMA temp_store=MEMORY")
            self._conn.execute("PRAGMA mmap_size=268435456")
            self._conn.execute("PRAGMA cache_size=-256000")
            self._create_schema()

        self._batch_table_ready = False

    def _sql(self, template: str) -> str:
        """Qualify index tables (postings/key_df/meta) for attached read-only DB."""
        return (
            template.replace(" postings ", f" {self._postings} ")
            .replace(" postings\n", f" {self._postings}\n")
            .replace("FROM postings", f"FROM {self._postings}")
            .replace("JOIN postings", f"JOIN {self._postings}")
            .replace(" key_df ", f" {self._key_df} ")
            .replace("JOIN key_df", f"JOIN {self._key_df}")
            .replace("FROM meta", f"FROM {self._meta}")
        )

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
            self._sql(
                "SELECT key FROM postings WHERE channel=? AND target=? AND entity_id=?",
            ),
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
                sql = self._sql(_SQL_SCORED_MAXDF.format(placeholders=placeholders))
                params.append(max_df)
            else:
                sql = self._sql(_SQL_SCORED_IN.format(placeholders=placeholders))

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
            sql = self._sql(_SQL_EXACT_IN.format(placeholders=placeholders))
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
                for qid, eid in self._conn.execute(
                    self._sql(_SQL_BATCH_EXACT), (ch, tg),
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
                    sql = self._sql(_SQL_BATCH_SCORED_MAXDF)
                    params = (ch, tg, max_df)
                else:
                    sql = self._sql(_SQL_BATCH_SCORED)
                    params = (ch, tg)
                for qid, eid in self._conn.execute(sql, params):
                    out[qid][eid] += 1

        return out

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute(
            self._sql("SELECT v FROM meta WHERE k=?"),
            (key,),
        ).fetchone()
        return row[0] if row else None

    def stats(self, channel: ChannelName, target: Target) -> dict:
        ch, tg = channel.value, target
        keys = self._conn.execute(
            self._sql("SELECT COUNT(*) FROM key_df WHERE channel=? AND target=?"),
            (ch, tg),
        ).fetchone()[0]
        postings = self._conn.execute(
            self._sql("SELECT COUNT(*) FROM postings WHERE channel=? AND target=?"),
            (ch, tg),
        ).fetchone()[0]
        entities = self._conn.execute(
            self._sql(
                "SELECT COUNT(DISTINCT entity_id) FROM postings WHERE channel=? AND target=?",
            ),
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
