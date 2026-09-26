"""
Verify key-centric batch exact lookup matches legacy qid-join and per-row baseline.

Run only (tiny sample):
  python -m analysis.retrieval.test_keycentric_equivalence
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

from analysis.retrieval.benchmark import _collect_s1_rows
from analysis.retrieval.build import _index_path
from analysis.retrieval.channels import BaselineChannel
from analysis.retrieval.config import ChannelName, RetrievalConfig
from analysis.retrieval.index import InvertedIndex


def _reference_per_row(
    index: InvertedIndex,
    channel: BaselineChannel,
    target: str,
    queries: list[tuple[str, list[str]]],
) -> dict[str, set[str]]:
    """Same as BaselineChannel.retrieve() per query (single lookup_keys_exact per S1)."""
    out: dict[str, set[str]] = {}
    for qid, keys in queries:
        key_list = list({k for k in keys if k})
        hits = index.lookup_keys_exact(channel.name, target, key_list)  # type: ignore[arg-type]
        out[qid] = hits
    return out


def _build_synthetic_index(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE postings (
            channel TEXT NOT NULL, target TEXT NOT NULL,
            key TEXT NOT NULL, entity_id TEXT NOT NULL
        );
        CREATE INDEX idx_postings_lookup ON postings (channel, target, key);
        CREATE TABLE key_df (
            channel TEXT NOT NULL, target TEXT NOT NULL,
            key TEXT NOT NULL, df INTEGER NOT NULL,
            PRIMARY KEY (channel, target, key)
        );
    """)
    rows = [
        ("baseline", "S2", "bn:alpha shop", "S2-a1"),
        ("baseline", "S2", "bn:alpha shop", "S2-a2"),
        ("baseline", "S2", "ba:1 main st", "S2-addr1"),
        ("baseline", "S2", "cp:US|pizza", "S2-p0"),
        ("baseline", "S2", "cp:US|pizza", "S2-p1"),
        ("baseline", "S2", "cp:US|pizza", "S2-p2"),
        ("baseline", "S3", "cp:US|pizza", "S3-p0"),
        ("baseline", "S3", "bn:beta llc", "S3-b1"),
    ]
    conn.executemany("INSERT INTO postings VALUES (?,?,?,?)", rows)
    conn.commit()
    conn.close()


def _synthetic_queries(n: int = 40) -> list[tuple[str, list[str]]]:
    queries: list[tuple[str, list[str]]] = []
    for i in range(n):
        qid = f"S1-{i:04d}"
        keys: list[str] = []
        if i % 3 != 2:
            keys.append("cp:US|pizza")
        if i % 5 == 0:
            keys.append("bn:alpha shop")
        if i % 7 == 0:
            keys.append("ba:1 main st")
        if i % 11 == 0:
            keys.append("bn:beta llc")
        if not keys:
            keys.append(f"bn:unique-{i}")
        queries.append((qid, keys))
    return queries


def run_equivalence(index_path: Path, queries: list[tuple[str, list[str]]]) -> dict:
    channel = BaselineChannel(RetrievalConfig().baseline)

    results = {"queries": len(queries), "targets": {}}
    with InvertedIndex(index_path, read_only=True) as index:
        for target in ("S2", "S3"):
            new = index.lookup_keys_exact_batch(
                ChannelName.BASELINE, target, queries  # type: ignore[arg-type]
            )
            old = index.lookup_keys_exact_batch_qid_join(
                ChannelName.BASELINE, target, queries  # type: ignore[arg-type]
            )
            ref = _reference_per_row(index, channel, target, queries)
            results["targets"][target] = {
                "keycentric_vs_qid_join": new == old,
                "keycentric_vs_per_row_keys": new == ref,
                "mismatch_qid_join": [q for q in new if new[q] != old.get(q, set())][:5],
                "mismatch_per_row": [q for q in new if new[q] != ref.get(q, set())][:5],
            }
    results["ok"] = all(
        t["keycentric_vs_qid_join"] and t["keycentric_vs_per_row_keys"]
        for t in results["targets"].values()
    )
    return results


def main() -> None:
    cfg = RetrievalConfig()
    real_index = _index_path(cfg)
    real_s1 = cfg.paths()["S1"]

    if real_index.is_file() and real_s1.is_file():
        print(f"Using real index + train S1 (sample=50): {real_index}")
        channel = BaselineChannel(cfg.baseline)
        rows = _collect_s1_rows(cfg, 50)
        queries = [
            (r["entity_id"].strip(), list(channel.extract_query_keys(r)))
            for r in rows
            if r.get("entity_id", "").strip()
        ]
        index_path = real_index
    else:
        print("Using synthetic index (real index/S1 not found)")
        td = Path(tempfile.mkdtemp())
        index_path = td / "synthetic.sqlite"
        _build_synthetic_index(index_path)
        queries = _synthetic_queries(40)

    report = run_equivalence(index_path, queries)
    print(report)
    if not report["ok"]:
        print("EQUIVALENCE FAILED", file=sys.stderr)
        raise SystemExit(1)
    print("equivalence: OK")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
