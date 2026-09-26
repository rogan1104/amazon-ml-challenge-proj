"""
Bounded regression: key-centric batch lookup vs tiny reference (seconds only).

Compares lookup_keys_exact_batch() to a reference built from ONE bounded
key IN (...) query over the sample's unique keys — no qid×key join, no per-S1
lookup_keys_exact loop on the full index.

Run:
  python -m analysis.retrieval.test_keycentric_equivalence
"""
from __future__ import annotations

import sys
from pathlib import Path

from analysis.retrieval.build import _index_path
from analysis.retrieval.config import ChannelName, RetrievalConfig
from analysis.retrieval.index import InvertedIndex
from analysis.retrieval.keys import baseline_keys

# Exactly 5 deterministic S1 probes (fixed strings → stable bn/ba/cp keys).
_DETERMINISTIC_S1: list[tuple[str, str, str, str]] = [
    ("S1-EQ-01", "Zzquiv Probe Alpha One", "9000 Zzquiv Regression Lane", "US"),
    ("S1-EQ-02", "Zzquiv Probe Alpha One", "9001 Other Zzquiv Street", "US"),
    ("S1-EQ-03", "Mmmquiv Unique Beta Two", "42 Mmmquiv Court", "IN"),
    ("S1-EQ-04", "Yyyquiv Gamma Three", "7 Yyyquiv Road", "FR"),
    ("S1-EQ-05", "Yyyquiv Gamma Three", "8 Yyyquiv Road", "FR"),
]


def deterministic_queries() -> list[tuple[str, list[str]]]:
    cfg = RetrievalConfig().baseline
    out: list[tuple[str, list[str]]] = []
    for qid, name, addr, country in _DETERMINISTIC_S1:
        keys = baseline_keys(
            name,
            addr,
            country,
            prefix_len=cfg.name_prefix_len,
            min_prefix=cfg.min_prefix_len,
        )
        out.append((qid, list(keys)))
    return out


def _bounded_reference(
    index: InvertedIndex,
    channel: str,
    target: str,
    queries: list[tuple[str, list[str]]],
) -> dict[str, set[str]]:
    """
    Reference candidate sets: union entity_ids per key using a single bounded
    IN lookup over unique keys in the sample (at most ~15 keys for 5 S1 rows).
    """
    unique_keys = sorted({k for _, keys in queries for k in keys if k})
    key_to_eids: dict[str, set[str]] = {k: set() for k in unique_keys}

    if unique_keys:
        chunk = 32
        conn = index._conn
        for i in range(0, len(unique_keys), chunk):
            batch = unique_keys[i : i + chunk]
            placeholders = ",".join("?" * len(batch))
            sql = (
                f"SELECT key, entity_id FROM postings "
                f"WHERE channel=? AND target=? AND key IN ({placeholders})"
            )
            params: list = [channel, target, *batch]
            for key, eid in conn.execute(sql, params):
                key_to_eids[key].add(eid)

    ref: dict[str, set[str]] = {}
    for qid, keys in queries:
        acc: set[str] = set()
        for k in keys:
            if k:
                acc |= key_to_eids.get(k, set())
        ref[qid] = acc
    return ref


def run_equivalence(index_path: Path, queries: list[tuple[str, list[str]]]) -> dict:
    channel_name = ChannelName.BASELINE.value
    results: dict = {"queries": len(queries), "index": str(index_path), "targets": {}}

    with InvertedIndex(index_path, read_only=True) as index:
        for target in ("S2", "S3"):
            print(
                f"[equiv] comparing key-centric vs bounded reference: "
                f"channel={channel_name} target={target} ...",
                flush=True,
            )
            got = index.lookup_keys_exact_batch(
                ChannelName.BASELINE, target, queries  # type: ignore[arg-type]
            )
            ref = _bounded_reference(index, channel_name, target, queries)
            mismatches = [qid for qid, keys in queries if got.get(qid, set()) != ref.get(qid, set())]
            results["targets"][target] = {
                "ok": not mismatches,
                "mismatches": mismatches[:5],
                "sample_counts": {
                    qid: len(got.get(qid, set())) for qid, _ in queries[:3]
                },
            }
            print(f"[equiv] target={target} ok={not mismatches}", flush=True)

    results["ok"] = all(t["ok"] for t in results["targets"].values())
    return results


def main() -> None:
    index_path = _index_path(RetrievalConfig())
    if not index_path.is_file():
        print(f"Index not found: {index_path}", file=sys.stderr)
        raise SystemExit(2)

    queries = deterministic_queries()
    print(f"[equiv] index={index_path} queries={len(queries)}", flush=True)
    for qid, keys in queries:
        print(f"  {qid}: {len(keys)} keys", flush=True)

    report = run_equivalence(index_path, queries)
    print(report)
    if not report["ok"]:
        print("EQUIVALENCE FAILED", file=sys.stderr)
        raise SystemExit(1)
    print("equivalence: OK")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
