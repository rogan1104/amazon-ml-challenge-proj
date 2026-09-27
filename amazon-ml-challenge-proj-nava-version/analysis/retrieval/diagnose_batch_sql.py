"""
Diagnose lookup_keys_exact_batch SQL cost (tiny S1 sample only).

Usage (PC, existing index — do NOT use for full benchmark):
  python -m analysis.retrieval.diagnose_batch_sql --sample 50
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from analysis.retrieval.build import _index_path
from analysis.retrieval.channels import BaselineChannel
from analysis.retrieval.config import RetrievalConfig
from analysis.retrieval.index import InvertedIndex, _SQL_BATCH_EXACT_QID_JOIN_QID_JOIN
from analysis.retrieval.keys import baseline_keys
from analysis.retrieval.benchmark import _collect_s1_rows


def _key_family(key: str) -> str:
    if key.startswith("bn:"):
        return "bn"
    if key.startswith("ba:"):
        return "ba"
    if key.startswith("cp:"):
        return "cp"
    return "other"


def diagnose(cfg: RetrievalConfig, sample: int, index_path: Path) -> dict:
    channel = BaselineChannel(cfg.baseline)
    rows = _collect_s1_rows(cfg, sample)
    queries = [(r["entity_id"].strip(), list(channel.extract_query_keys(r))) for r in rows]
    queries = [(qid, ks) for qid, ks in queries if qid]

    pairs: list[tuple[str, str]] = []
    fam_counts: Counter = Counter()
    unique_keys: set[str] = set()
    for qid, keys in queries:
        for k in {k for k in keys if k}:
            pairs.append((qid, k))
            fam_counts[_key_family(k)] += 1
            unique_keys.add(k)

    ch = channel.name.value
    report: dict = {
        "sample_s1_rows": len(queries),
        "batch_query_rows": len(pairs),
        "unique_keys_in_batch": len(unique_keys),
        "keys_by_family": dict(fam_counts),
        "sub_batch_size": 2500,
        "join_sql": " ".join(_SQL_BATCH_EXACT_QID_JOIN.split()),
    }

    with InvertedIndex(index_path, read_only=True) as index:
        conn = index._conn
        for target in ("S2", "S3"):
            tg = target
            target_report: dict = {}

            # key_df stats for keys in sample
            dfs: list[int] = []
            for k in unique_keys:
                row = conn.execute(
                    """
                    SELECT df FROM key_df
                    WHERE channel=? AND target=? AND key=?
                    """,
                    (ch, tg, k),
                ).fetchone()
                if row:
                    dfs.append(row[0])
            if dfs:
                dfs.sort()
                target_report["key_df_in_sample"] = {
                    "keys_found": len(dfs),
                    "df_min": dfs[0],
                    "df_median": dfs[len(dfs) // 2],
                    "df_p95": dfs[int(len(dfs) * 0.95)],
                    "df_max": dfs[-1],
                    "df_sum": sum(dfs),
                }

            conn.execute("DELETE FROM batch_query")
            index._ensure_batch_table()
            conn.executemany(
                "INSERT INTO batch_query (qid, key) VALUES (?, ?)",
                pairs,
            )

            plan = conn.execute(
                f"EXPLAIN QUERY PLAN { _SQL_BATCH_EXACT_QID_JOIN }",
                (ch, tg),
            ).fetchall()
            target_report["explain_query_plan"] = [list(r) for r in plan]

            t0 = time.perf_counter()
            join_rows = 0
            qid_counts: Counter = Counter()
            for qid, _eid in conn.execute(_SQL_BATCH_EXACT_QID_JOIN, (ch, tg)):
                join_rows += 1
                qid_counts[qid] += 1
            join_sec = time.perf_counter() - t0

            target_report["join_result_rows"] = join_rows
            target_report["join_wall_sec"] = round(join_sec, 4)
            target_report["avg_join_rows_per_s1"] = round(
                join_rows / max(len(queries), 1), 2,
            )
            if qid_counts:
                vals = sorted(qid_counts.values())
                target_report["join_rows_per_s1_p95"] = vals[int(len(vals) * 0.95)]

            # Per family: restrict batch_query to one prefix, measure join output
            per_fam: dict = {}
            for fam in ("bn", "ba", "cp"):
                fam_pairs = [(q, k) for q, k in pairs if k.startswith(f"{fam}:")]
                if not fam_pairs:
                    continue
                conn.execute("DELETE FROM batch_query")
                conn.executemany(
                    "INSERT INTO batch_query (qid, key) VALUES (?, ?)",
                    fam_pairs,
                )
                t0 = time.perf_counter()
                n = 0
                for _ in conn.execute(_SQL_BATCH_EXACT_QID_JOIN, (ch, tg)):
                    n += 1
                per_fam[fam] = {
                    "batch_query_rows": len(fam_pairs),
                    "join_result_rows": n,
                    "join_sec": round(time.perf_counter() - t0, 4),
                }
            target_report["per_key_family"] = per_fam

            # Duplicate-key fan-out: how many qids share the same blocking key?
            key_to_qids: dict[str, set[str]] = defaultdict(set)
            for qid, k in pairs:
                key_to_qids[k].add(qid)
            shared = sorted(
                ((k, len(qs), _key_family(k)) for k, qs in key_to_qids.items()),
                key=lambda x: -x[1],
            )[:10]
            target_report["top_shared_keys"] = [
                {"key": k[:80], "family": fam, "qids_sharing": n}
                for k, n, fam in shared
            ]

            report[f"target_{target}"] = target_report

    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose batch exact SQL join (tiny sample)")
    p.add_argument("--split", default="train")
    p.add_argument("--sample", type=int, default=50)
    p.add_argument("--index-dir", default="reports/cache/retrieval_index")
    args = p.parse_args()

    cfg = RetrievalConfig(split=args.split, index_dir=Path(args.index_dir))
    index_path = _index_path(cfg)
    if not index_path.is_file():
        print(f"Index not found: {index_path}", file=sys.stderr)
        raise SystemExit(2)
    if not cfg.paths()["S1"].is_file():
        print(f"S1 TSV not found: {cfg.paths()['S1']}", file=sys.stderr)
        raise SystemExit(2)

    out = diagnose(cfg, args.sample, index_path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
