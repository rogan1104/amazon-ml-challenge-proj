"""Disk-backed inverted index for O(1) key lookups (no S1×S2 cross product)."""
from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from analysis.retrieval.config import ChannelName, Target


class InvertedIndex:
    """
    SQLite inverted index: (channel, target, key) -> entity_id postings.

    Stored on disk so indexes for millions of rows fit in bounded RAM during build.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._create_schema()

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
                sql = f"""
                    SELECT p.entity_id
                    FROM postings p
                    JOIN key_df d ON p.channel=d.channel AND p.target=d.target AND p.key=d.key
                    WHERE p.channel=? AND p.target=? AND p.key IN ({placeholders})
                      AND d.df <= ?
                """
                params.append(max_df)
            else:
                sql = f"""
                    SELECT entity_id FROM postings
                    WHERE channel=? AND target=? AND key IN ({placeholders})
                """
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
            sql = f"""
                SELECT DISTINCT entity_id FROM postings
                WHERE channel=? AND target=? AND key IN ({placeholders})
            """
            for (eid,) in self._conn.execute(sql, params):
                found.add(eid)
        return found

    def set_meta(self, key: str, value: str) -> None:
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
