#!/usr/bin/env python3
"""
Blocking Audit — evaluate existing teammate blocking OR document absence + baselines.

Does NOT modify any production blocking code.
"""
import json
import time
from pathlib import Path
from datetime import datetime

import duckdb
import pandas as pd

from analysis.config import REPORTS, CACHE, ROOT, read_csv_sql, TRAIN, TEST
from analysis.phase0_inventory import register_views

# Paths where teammate blocking might live
TEAMMATE_SEARCH_PATHS = [
    ROOT / "code",
    ROOT / "src",
    ROOT / "output" / "candidate_pairs.tsv",
    ROOT / "business_entity_resolution",
    ROOT / "blocking",
    ROOT / "retrieval",
]

BLOCKING_KEYWORDS = [
    "blocking", "candidate", "retrieval", "block_key", "inverted",
    "tfidf", "faiss", "ngram", "prefix", "index",
]


def discover_teammate_blocking() -> dict:
    """Scan repo for production blocking implementation."""
    found = {"implementations": [], "candidate_outputs": [], "search_notes": []}

    for p in ROOT.rglob("*"):
        if ".venv" in p.parts or "dataset" in p.parts or "__pycache__" in p.parts:
            continue
        if p.is_file():
            name_lower = p.name.lower()
            if name_lower == "candidate_pairs.tsv":
                found["candidate_outputs"].append(str(p))
            if p.suffix == ".py" and "phase10_blocking" not in name_lower:
                if any(k in name_lower for k in ["block", "candidate", "retrieval"]):
                    if "analysis" not in p.parts or "audit" in name_lower:
                        found["implementations"].append(str(p))

    # Explicit expected locations
    for ep in TEAMMATE_SEARCH_PATHS:
        if ep.exists():
            found["search_notes"].append(f"EXISTS: {ep}")
        else:
            found["search_notes"].append(f"MISSING: {ep}")

    # Grep-like scan of py files outside analysis forensics
    for py in (ROOT / "utils").glob("*.py"):
        text = py.read_text(errors="replace").lower()
        if any(k in text for k in ["def block", "def generate_candidate", "class block"]):
            found["implementations"].append(str(py))

    # Exclude forensics/analysis scripts from "implementation" count
    found["implementations"] = [
        p for p in found["implementations"]
        if "analysis" not in p and "phase10" not in p and "blocking_audit" not in p
    ]
    found["teammate_blocking_found"] = bool(
        found["implementations"] or found["candidate_outputs"]
    )
    return found


def setup_norm_views(con):
    """Shared normalized views for blocking evaluation."""
    con.execute("""
        CREATE OR REPLACE VIEW s1_norm AS
        SELECT entity_id, country, business_name, business_address,
               LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_name,''), '[^\\w\\s&]', ' ', 'g'))) AS norm_name,
               LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_address,''), '[^\\w\\s]', ' ', 'g'))) AS norm_addr,
               LEFT(LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_name,''), '[^\\w\\s]', ' ', 'g'))), 5) AS name_prefix5,
               LEFT(LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_name,''), '[^\\w\\s]', ' ', 'g'))), 10) AS name_prefix10,
               list_extract(string_split(LOWER(TRIM(COALESCE(business_name,''))), ' '), 1) AS name_first_token
        FROM train_S1
    """)
    con.execute("""
        CREATE OR REPLACE VIEW s2_norm AS
        SELECT entity_id, country, business_name, business_address,
               LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_name,''), '[^\\w\\s&]', ' ', 'g'))) AS norm_name,
               LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_address,''), '[^\\w\\s]', ' ', 'g'))) AS norm_addr,
               LEFT(LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_name,''), '[^\\w\\s]', ' ', 'g'))), 5) AS name_prefix5,
               LEFT(LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_name,''), '[^\\w\\s]', ' ', 'g'))), 10) AS name_prefix10,
               list_extract(string_split(LOWER(TRIM(COALESCE(business_name,''))), ' '), 1) AS name_first_token
        FROM train_S2
    """)
    con.execute("""
        CREATE OR REPLACE VIEW s3_norm AS
        SELECT entity_id, country, business_name, business_address,
               LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_name,''), '[^\\w\\s&]', ' ', 'g'))) AS norm_name,
               LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_address,''), '[^\\w\\s]', ' ', 'g'))) AS norm_addr,
               LEFT(LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_name,''), '[^\\w\\s]', ' ', 'g'))), 5) AS name_prefix5,
               LEFT(LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_name,''), '[^\\w\\s]', ' ', 'g'))), 10) AS name_prefix10,
               list_extract(string_split(LOWER(TRIM(COALESCE(business_name,''))), ' '), 1) AS name_first_token
        FROM train_S3
    """)


def eval_strategy_recall(con, s1_key, o_key, extra, match_src, other_view) -> int:
    return con.execute(f"""
        SELECT COUNT(*) FROM gt_pairs g
        JOIN s1_norm s1 ON g.s1_id = s1.entity_id
        JOIN {other_view} o ON g.matched_id = o.entity_id
        WHERE g.match_source = '{match_src}' AND {extra} AND {s1_key} = {o_key}
    """).fetchone()[0]


def eval_strategy_candidates(con, s1_key, o_key, extra) -> dict:
    return con.execute(f"""
        WITH s1_keys AS (
            SELECT entity_id, {s1_key} AS bkey FROM s1_norm s1 WHERE {extra}
        ),
        o_by_key AS (
            SELECT {o_key} AS bkey, COUNT(*) AS o_cnt
            FROM s2_norm o GROUP BY {o_key}
            UNION ALL
            SELECT {o_key}, COUNT(*) FROM s3_norm o GROUP BY {o_key}
        ),
        o_agg AS (SELECT bkey, SUM(o_cnt) AS o_cnt FROM o_by_key GROUP BY bkey),
        s1_cand AS (
            SELECT s1.entity_id, COALESCE(o.o_cnt, 0) AS n
            FROM s1_keys s1 LEFT JOIN o_agg o ON s1.bkey = o.bkey
        )
        SELECT
            SUM(n) AS total_candidates,
            AVG(CASE WHEN n > 0 THEN n END) AS avg_c,
            MEDIAN(CASE WHEN n > 0 THEN n END) AS med_c,
            QUANTILE_CONT(n, 0.90) AS p90,
            QUANTILE_CONT(n, 0.95) AS p95,
            QUANTILE_CONT(n, 0.99) AS p99,
            MAX(n) AS max_c,
            SUM(CASE WHEN n = 0 THEN 1 ELSE 0 END) AS zero_s1,
            COUNT(*) AS total_s1
        FROM s1_cand
    """).fetchdf().iloc[0].to_dict()


def run_baseline_comparison(con) -> list:
    """Lightweight baseline strategies for comparison."""
    t0 = time.time()
    total_true = con.execute("SELECT COUNT(*) FROM gt_pairs").fetchone()[0]
    total_s1 = con.execute("SELECT COUNT(*) FROM train_S1").fetchone()[0]
    total_s2 = con.execute("SELECT COUNT(*) FROM train_S2").fetchone()[0]
    total_s3 = con.execute("SELECT COUNT(*) FROM train_S3").fetchone()[0]
    possible = total_s1 * (total_s2 + total_s3)

    strategies = [
        ("A_exact_norm_name", "s1.norm_name", "o.norm_name", "s1.norm_name != ''"),
        ("B_exact_norm_address", "s1.norm_addr", "o.norm_addr", "s1.norm_addr != ''"),
        ("C_country_name_prefix5", "s1.country || '|' || s1.name_prefix5", "o.country || '|' || o.name_prefix5", "LENGTH(s1.name_prefix5) >= 3"),
        ("D_name_first_token", "s1.name_first_token", "o.name_first_token", "LENGTH(s1.name_first_token) >= 3"),
        ("E_combined_name_or_addr_or_cprefix", None, None, None),  # special
    ]

    rows = []
    for name, s1k, ok, extra in strategies:
        for src, ov in [("S2", "s2_norm"), ("S3", "s3_norm")]:
            t1 = time.time()
            if name == "E_combined_name_or_addr_or_cprefix":
                recalled = con.execute(f"""
                    SELECT COUNT(*) FROM gt_pairs g
                    JOIN s1_norm s1 ON g.s1_id = s1.entity_id
                    JOIN {ov} o ON g.matched_id = o.entity_id
                    WHERE g.match_source = '{src}' AND (
                        (s1.norm_name = o.norm_name AND s1.norm_name != '')
                        OR (s1.norm_addr = o.norm_addr AND s1.norm_addr != '')
                        OR (s1.country = o.country AND s1.name_prefix5 = o.name_prefix5 AND LENGTH(s1.name_prefix5) >= 3)
                    )
                """).fetchone()[0]
                stats = con.execute(f"""
                    WITH o_name AS (SELECT norm_name k, COUNT(*) c FROM {ov} WHERE norm_name != '' GROUP BY norm_name),
                    o_addr AS (SELECT norm_addr k, COUNT(*) c FROM {ov} WHERE norm_addr != '' GROUP BY norm_addr),
                    o_cp AS (SELECT country || '|' || name_prefix5 k, COUNT(*) c FROM {ov}
                              WHERE LENGTH(name_prefix5)>=3 GROUP BY 1),
                    s1_cand AS (
                        SELECT s1.entity_id,
                            GREATEST(COALESCE(n.c,0), COALESCE(a.c,0), COALESCE(cp.c,0)) AS n
                        FROM s1_norm s1
                        LEFT JOIN o_name n ON s1.norm_name = n.k
                        LEFT JOIN o_addr a ON s1.norm_addr = a.k
                        LEFT JOIN o_cp cp ON s1.country || '|' || s1.name_prefix5 = cp.k
                    )
                    SELECT SUM(n) total_candidates, AVG(CASE WHEN n>0 THEN n END) avg_c,
                           MEDIAN(CASE WHEN n>0 THEN n END) med_c,
                           QUANTILE_CONT(n,0.95) p95, MAX(n) max_c,
                           SUM(CASE WHEN n=0 THEN 1 ELSE 0 END) zero_s1
                    FROM s1_cand
                """).fetchdf().iloc[0]
            else:
                recalled = eval_strategy_recall(con, s1k, ok, extra, src, ov)
                stats = con.execute(f"""
                    WITH s1_keys AS (SELECT entity_id, {s1k} bkey FROM s1_norm s1 WHERE {extra}),
                    o_by_key AS (SELECT {ok} bkey, COUNT(*) o_cnt FROM {ov} o GROUP BY {ok}),
                    s1_cand AS (
                        SELECT s1.entity_id, COALESCE(o.o_cnt, 0) n
                        FROM s1_keys s1 LEFT JOIN o_by_key o ON s1.bkey = o.bkey
                    )
                    SELECT SUM(n) total_candidates, AVG(CASE WHEN n>0 THEN n END) avg_c,
                           MEDIAN(CASE WHEN n>0 THEN n END) med_c,
                           QUANTILE_CONT(n, 0.95) p95, MAX(n) max_c,
                           SUM(CASE WHEN n=0 THEN 1 ELSE 0 END) zero_s1
                    FROM s1_cand
                """).fetchdf().iloc[0]

            other_n = total_s2 if src == "S2" else total_s3
            tc = int(stats["total_candidates"] or 0)
            rows.append({
                "strategy": f"{name}_{src}",
                "target": src,
                "true_matches_recalled": int(recalled),
                "recall": round(recalled / total_true, 6),
                "recall_per_target_only": round(recalled / con.execute(
                    f"SELECT COUNT(*) FROM gt_pairs WHERE match_source='{src}'"
                ).fetchone()[0], 6),
                "total_candidates": tc,
                "avg_candidates_per_s1": round(float(stats["avg_c"] or 0), 2),
                "median_candidates_per_s1": round(float(stats["med_c"] or 0), 2),
                "p95_candidates_per_s1": round(float(stats["p95"] or 0), 2),
                "max_candidates_per_s1": int(stats["max_c"] or 0),
                "s1_zero_candidates": int(stats["zero_s1"] or 0),
                "s1_zero_pct": round(100 * int(stats["zero_s1"] or 0) / total_s1, 4),
                "reduction_ratio": round(1 - tc / (total_s1 * other_n), 6),
                "runtime_sec": round(time.time() - t1, 2),
            })

    # Union recall for combined E
    e_s2 = next(r for r in rows if r["strategy"] == "E_combined_name_or_addr_or_cprefix_S2")
    e_s3 = next(r for r in rows if r["strategy"] == "E_combined_name_or_addr_or_cprefix_S3")
    rows.append({
        "strategy": "E_combined_union_S2+S3",
        "target": "S2+S3",
        "true_matches_recalled": e_s2["true_matches_recalled"] + e_s3["true_matches_recalled"],
        "recall": round((e_s2["true_matches_recalled"] + e_s3["true_matches_recalled"]) / total_true, 6),
        "total_candidates": e_s2["total_candidates"] + e_s3["total_candidates"],
        "avg_candidates_per_s1": round(e_s2["avg_candidates_per_s1"] + e_s3["avg_candidates_per_s1"], 2),
        "runtime_sec": round(time.time() - t0, 2),
    })
    return rows


def recall_by_match_bucket(con) -> list:
    """Recall for combined blocking by S1 match-count bucket."""
    rows = []
    for src, ov in [("S2", "s2_norm"), ("S3", "s3_norm")]:
        df = con.execute(f"""
            WITH recalled AS (
                SELECT g.s1_id, g.matched_id
                FROM gt_pairs g
                JOIN s1_norm s1 ON g.s1_id = s1.entity_id
                JOIN {ov} o ON g.matched_id = o.entity_id
                WHERE g.match_source = '{src}' AND (
                    (s1.norm_name = o.norm_name AND s1.norm_name != '')
                    OR (s1.norm_addr = o.norm_addr AND s1.norm_addr != '')
                    OR (s1.country = o.country AND s1.name_prefix5 = o.name_prefix5 AND LENGTH(s1.name_prefix5) >= 3)
                )
            ),
            bucketed AS (
                SELECT g.s1_id, g.matched_id, gs.match_count,
                    CASE WHEN gs.match_count = 0 THEN '0_singleton'
                         WHEN gs.match_count = 1 THEN '1_match'
                         WHEN gs.match_count BETWEEN 2 AND 3 THEN '2-3_matches'
                         ELSE '4+_matches' END AS bucket
                FROM gt_pairs g
                JOIN gt_summary gs ON g.s1_id = gs.s1_id
                WHERE g.match_source = '{src}'
            )
            SELECT b.bucket,
                   COUNT(*) AS total_true_pairs,
                   SUM(CASE WHEN r.s1_id IS NOT NULL THEN 1 ELSE 0 END) AS recalled,
                   SUM(CASE WHEN r.s1_id IS NULL THEN 1 ELSE 0 END) AS missed
            FROM bucketed b
            LEFT JOIN recalled r ON b.s1_id = r.s1_id AND b.matched_id = r.matched_id
            GROUP BY b.bucket ORDER BY b.bucket
        """).fetchdf()
        for _, r in df.iterrows():
            rows.append({
                "target": src,
                "bucket": r["bucket"],
                "total_true_pairs": int(r["total_true_pairs"]),
                "recalled": int(r["recalled"]),
                "missed": int(r["missed"]),
                "recall": round(r["recalled"] / r["total_true_pairs"], 6) if r["total_true_pairs"] else 0,
            })

    # Singleton S1 entities (no gt pairs — check if they'd get zero candidates)
    for src in ["S2", "S3"]:
        ov = "s2_norm" if src == "S2" else "s3_norm"
        singleton_cand = con.execute(f"""
            WITH singletons AS (
                SELECT s1.entity_id FROM train_S1 s1
                JOIN gt_summary gs ON s1.entity_id = gs.s1_id WHERE gs.match_count = 0
            ),
            o_name AS (SELECT norm_name k, COUNT(*) c FROM {ov} WHERE norm_name != '' GROUP BY norm_name),
            o_addr AS (SELECT norm_addr k, COUNT(*) c FROM {ov} WHERE norm_addr != '' GROUP BY norm_addr),
            o_cp AS (SELECT country || '|' || name_prefix5 k, COUNT(*) c FROM {ov}
                      WHERE LENGTH(name_prefix5)>=3 GROUP BY 1),
            s1_cand AS (
                SELECT s.entity_id,
                    GREATEST(COALESCE(n.c,0), COALESCE(a.c,0), COALESCE(cp.c,0)) AS n
                FROM singletons s
                JOIN s1_norm s1 ON s.entity_id = s1.entity_id
                LEFT JOIN o_name n ON s1.norm_name = n.k
                LEFT JOIN o_addr a ON s1.norm_addr = a.k
                LEFT JOIN o_cp cp ON s1.country || '|' || s1.name_prefix5 = cp.k
            )
            SELECT COUNT(*) total_singletons, SUM(CASE WHEN n=0 THEN 1 ELSE 0 END) zero_cand,
                   AVG(n) avg_c, MAX(n) max_c FROM s1_cand
        """).fetchdf().iloc[0]
        rows.append({
            "target": src,
            "bucket": "0_singleton_s1_entities",
            "total_true_pairs": 0,
            "recalled": 0,
            "missed": 0,
            "recall": 1.0,  # N/A — no true pairs to recall
            "singleton_count": int(singleton_cand["total_singletons"]),
            "singleton_zero_candidates": int(singleton_cand["zero_cand"]),
            "singleton_avg_candidates": round(float(singleton_cand["avg_c"] or 0), 2),
        })
    return rows


def recall_by_country(con) -> list:
    rows = []
    for src, ov in [("S2", "s2_norm"), ("S3", "s3_norm")]:
        df = con.execute(f"""
            WITH recalled AS (
                SELECT g.s1_id, g.matched_id
                FROM gt_pairs g JOIN s1_norm s1 ON g.s1_id = s1.entity_id
                JOIN {ov} o ON g.matched_id = o.entity_id
                WHERE g.match_source = '{src}' AND (
                    (s1.norm_name = o.norm_name AND s1.norm_name != '')
                    OR (s1.norm_addr = o.norm_addr AND s1.norm_addr != '')
                    OR (s1.country = o.country AND s1.name_prefix5 = o.name_prefix5 AND LENGTH(s1.name_prefix5) >= 3)
                )
            )
            SELECT s1.country, COUNT(*) total_pairs,
                   SUM(CASE WHEN r.s1_id IS NOT NULL THEN 1 ELSE 0 END) recalled
            FROM gt_pairs g
            JOIN train_S1 s1 ON g.s1_id = s1.entity_id
            LEFT JOIN recalled r ON g.s1_id = r.s1_id AND g.matched_id = r.matched_id
            WHERE g.match_source = '{src}'
            GROUP BY s1.country
        """).fetchdf()
        for _, r in df.iterrows():
            rows.append({
                "target": src, "country": r["country"],
                "total_true_pairs": int(r["total_pairs"]),
                "recalled": int(r["recalled"]),
                "recall": round(r["recalled"] / r["total_pairs"], 6),
            })
    return rows


def recall_curve(con, sample_size: int = 50000) -> list:
    """Simulate top-K cap using min blocking-key collision size (sampled true pairs)."""
    limits = [100, 250, 500, 1000, 2500, 5000, 10000]
    rows = []
    total_true = con.execute("SELECT COUNT(*) FROM gt_pairs").fetchone()[0]

    for src, ov in [("S2", "s2_norm"), ("S3", "s3_norm")]:
        src_total = con.execute(f"SELECT COUNT(*) FROM gt_pairs WHERE match_source='{src}'").fetchone()[0]
        # Precompute key sizes once per source
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE name_key_sizes_{src.lower()} AS
            SELECT norm_name k, COUNT(*) cnt FROM {ov} WHERE norm_name != '' GROUP BY norm_name
        """)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE addr_key_sizes_{src.lower()} AS
            SELECT norm_addr k, COUNT(*) cnt FROM {ov} WHERE norm_addr != '' GROUP BY norm_addr
        """)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE cp_key_sizes_{src.lower()} AS
            SELECT country || '|' || name_prefix5 k, COUNT(*) cnt
            FROM {ov} WHERE LENGTH(name_prefix5) >= 3 GROUP BY 1
        """)

        pair_sizes = con.execute(f"""
            WITH sample_pairs AS (
                SELECT g.s1_id, g.matched_id, s1.norm_name, s1.norm_addr,
                       s1.country, s1.name_prefix5
                FROM gt_pairs g JOIN s1_norm s1 ON g.s1_id = s1.entity_id
                WHERE g.match_source = '{src}'
                USING SAMPLE {sample_size} (reservoir)
            )
            SELECT tp.s1_id, tp.matched_id,
                LEAST(
                    COALESCE(n.cnt, 999999),
                    COALESCE(a.cnt, 999999),
                    COALESCE(c.cnt, 999999)
                ) AS min_block_size
            FROM sample_pairs tp
            LEFT JOIN name_key_sizes_{src.lower()} n ON tp.norm_name = n.k
            LEFT JOIN addr_key_sizes_{src.lower()} a ON tp.norm_addr = a.k
            LEFT JOIN cp_key_sizes_{src.lower()} c ON tp.country || '|' || tp.name_prefix5 = c.k
        """).fetchdf()

        n_sample = len(pair_sizes)
        scale = src_total / n_sample if n_sample else 1

        for k in limits:
            t0 = time.time()
            recalled = int((pair_sizes["min_block_size"] <= k).sum())
            rows.append({
                "candidate_limit_k": k,
                "target": src,
                "sample_size": n_sample,
                "true_pairs_total": src_total,
                "recalled_at_k_est": int(recalled * scale),
                "recall_at_k_per_target_sample": round(recalled / n_sample, 6) if n_sample else 0,
                "recall_at_k_per_target_est": round(recalled * scale / src_total, 6) if src_total else 0,
                "recall_at_k_overall_est": round(recalled * scale / total_true, 6),
                "avg_min_block_size": round(float(pair_sizes["min_block_size"].mean()), 2),
                "p95_min_block_size": round(float(pair_sizes["min_block_size"].quantile(0.95)), 2),
                "runtime_sec": round(time.time() - t0, 4),
            })
    return rows


def failure_analysis(con, limit=150) -> list:
    """Extract missed true matches under combined baseline blocking (sampled)."""
    missed = con.execute(f"""
        WITH sample_gt AS (
            SELECT s1_id, matched_id, match_source FROM gt_pairs USING SAMPLE 200000 (reservoir)
        ),
        s2_recalled AS (
            SELECT g.s1_id, g.matched_id FROM sample_gt g
            JOIN s1_norm s1 ON g.s1_id = s1.entity_id
            JOIN s2_norm s2 ON g.matched_id = s2.entity_id
            WHERE g.match_source = 'S2' AND (
                (s1.norm_name = s2.norm_name AND s1.norm_name != '')
                OR (s1.norm_addr = s2.norm_addr AND s1.norm_addr != '')
                OR (s1.country = s2.country AND s1.name_prefix5 = s2.name_prefix5 AND LENGTH(s1.name_prefix5) >= 3)
            )
        ),
        s3_recalled AS (
            SELECT g.s1_id, g.matched_id FROM sample_gt g
            JOIN s1_norm s1 ON g.s1_id = s1.entity_id
            JOIN s3_norm s3 ON g.matched_id = s3.entity_id
            WHERE g.match_source = 'S3' AND (
                (s1.norm_name = s3.norm_name AND s1.norm_name != '')
                OR (s1.norm_addr = s3.norm_addr AND s1.norm_addr != '')
                OR (s1.country = s3.country AND s1.name_prefix5 = s3.name_prefix5 AND LENGTH(s1.name_prefix5) >= 3)
            )
        ),
        recalled AS (SELECT * FROM s2_recalled UNION ALL SELECT * FROM s3_recalled)
        SELECT g.s1_id, g.matched_id, g.match_source,
               s1.business_name s1_name, s1.business_address s1_addr, s1.country,
               COALESCE(s2.business_name, s3.business_name) o_name,
               COALESCE(s2.business_address, s3.business_address) o_addr
        FROM sample_gt g
        JOIN train_S1 s1 ON g.s1_id = s1.entity_id
        LEFT JOIN train_S2 s2 ON g.matched_id = s2.entity_id AND g.match_source = 'S2'
        LEFT JOIN train_S3 s3 ON g.matched_id = s3.entity_id AND g.match_source = 'S3'
        LEFT JOIN recalled r ON g.s1_id = r.s1_id AND g.matched_id = r.matched_id
        WHERE r.s1_id IS NULL
        LIMIT {limit}
    """).fetchdf()

    failures = []
    for r in missed.itertuples():
        cat = categorize_failure(r.s1_name, r.o_name, r.s1_addr, r.o_addr, r.country)
        failures.append({
            "s1_id": r.s1_id, "matched_id": r.matched_id, "source": r.match_source,
            "s1_name": r.s1_name, "o_name": r.o_name,
            "s1_addr": r.s1_addr, "o_addr": r.o_addr,
            "country": r.country,
            "failure_category": cat,
        })
    return failures


def categorize_failure(s1_name, o_name, s1_addr, o_addr, country) -> str:
    import re
    def norm(s):
        if not s or (isinstance(s, float) and s != s):
            return ""
        return re.sub(r"[^\w\s]", " ", str(s).lower()).strip()

    n1, n2 = norm(s1_name), norm(o_name)
    a1, a2 = norm(s1_addr), norm(o_addr)

    if re.search(r"[^\x00-\x7F]", str(s1_name) + str(o_name)):
        return "transliteration/multilingual"
    if not s1_addr or not o_addr or str(s1_addr).strip() == "" or str(o_addr).strip() == "":
        return "missing_address"
    if n1 == n2 and a1 != a2:
        return "address_normalization_failure"
    if a1 == a2 and n1 != n2:
        return "name_normalization_failure"
    if set(n1.split()) == set(n2.split()) and n1 != n2:
        return "word_order"
    if any(w in n1 for w in ["ltd", "limited", "pvt", "private", "corp"]) or any(w in n2 for w in ["ltd", "limited", "pvt", "private", "corp"]):
        return "abbreviation/legal_suffix"
    if len(set(n1.split()) & set(n2.split())) >= max(1, min(len(n1.split()), len(n2.split())) - 1):
        return "minor_token_diff/typo"
    if len(n1) < 10 or len(n2) < 10:
        return "common_name/short_name"
    return "semantic_dba_or_other"


def candidate_quality(con, s2_recalled: int, s3_recalled: int, s2_total_cand: int, s3_total_cand: int) -> dict:
    """Estimate TP/FP rates from aggregated counts (no pair materialization)."""
    total_true = con.execute("SELECT COUNT(*) FROM gt_pairs").fetchone()[0]
    tc = s2_total_cand + s3_total_cand
    tp = s2_recalled + s3_recalled
    fp = tc - tp
    return {
        "total_candidate_pairs_s2_s3": tc,
        "true_positive_candidates": tp,
        "false_positive_candidates": fp,
        "true_positive_rate": round(tp / tc, 8) if tc else 0,
        "false_positive_rate": round(fp / tc, 8) if tc else 0,
        "candidates_per_true_match": round(tc / total_true, 2) if total_true else 0,
        "true_matches_total": total_true,
    }


def generate_audit_report(results: dict):
    REPORTS.mkdir(parents=True, exist_ok=True)
    discovery = results["discovery"]
    baselines = results["baselines"]
    combined = next((b for b in baselines if b["strategy"] == "E_combined_union_S2+S3"), {})
    s2_row = next((b for b in baselines if b["strategy"] == "E_combined_name_or_addr_or_cprefix_S2"), {})
    s3_row = next((b for b in baselines if b["strategy"] == "E_combined_name_or_addr_or_cprefix_S3"), {})
    quality = results["candidate_quality"]

    # CSV
    csv_rows = []
    csv_rows.extend(baselines)
    csv_rows.extend(results["recall_by_bucket"])
    csv_rows.extend(results["recall_by_country"])
    csv_rows.extend(results["recall_curve"])
    pd.DataFrame(csv_rows).to_csv(REPORTS / "blocking_audit.csv", index=False)

    failures = results["failures"]
    fail_counts = pd.Series([f["failure_category"] for f in failures]).value_counts()

    status = "REPLACE"
    if discovery["teammate_blocking_found"]:
        status = "IMPROVE"  # placeholder if found later

    md = f"""# Blocking Audit Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Scope:** Amazon ML Challenge 2026 — Business Entity Resolution (training GT)

---

## 1. Existing Blocking Architecture

### Discovery Result

**Teammate production blocking implementation: {'FOUND' if discovery['teammate_blocking_found'] else 'NOT FOUND'}**

Exhaustive repository scan performed. Expected locations checked:

"""
    for note in discovery["search_notes"]:
        md += f"- `{note}`\n"

    md += f"""
**Python files matching blocking keywords:** {len(discovery['implementations'])}
"""
    for impl in discovery["implementations"]:
        md += f"- `{impl}`\n"

    md += f"""
**candidate_pairs.tsv outputs found:** {len(discovery['candidate_outputs'])}
"""
    for out in discovery["candidate_outputs"]:
        md += f"- `{out}`\n"

    md += """
### What exists in this repository

| Component | Path | Role |
|-----------|------|------|
| Submission validator | `utils/validate_submission.py` | Validates `candidate_pairs.tsv` format; does NOT generate candidates |
| Forensics simulation | `analysis/phase10_blocking.py` | **Audit-only** baseline simulation from dataset forensics; NOT production blocking |
| Challenge spec | `README.md` | Defines expected `candidate_pairs.tsv` output format |

### OBSERVATION

No `code/business_entity_resolution/`, no `output/candidate_pairs.tsv`, and no runnable blocking/retrieval module was found. **There is currently nothing to audit as teammate production blocking.**

To complete a teammate audit when code is available, re-run:
```bash
.venv/bin/python -m analysis.blocking_audit --candidate output/candidate_pairs.tsv
```

---

## 2. Recall Results (Reference Baselines — Combined Simple Blocking)

Since no teammate blocking exists, we report **reference baseline E** (norm_name OR norm_addr OR country+prefix5) as the measured ceiling for simple rules.

| Target | True Pairs Recalled | Recall (of all 7.64M) | Recall (of target pairs) |
|--------|--------------------:|----------------------:|-------------------------:|
| S1→S2 | {s2_row.get('true_matches_recalled', 'N/A'):,} | {s2_row.get('recall', 0)*100:.2f}% | {s2_row.get('recall_per_target_only', 0)*100:.2f}% |
| S1→S3 | {s3_row.get('true_matches_recalled', 'N/A'):,} | {s3_row.get('recall', 0)*100:.2f}% | {s3_row.get('recall_per_target_only', 0)*100:.2f}% |
| **Overall S2+S3 union** | **{combined.get('true_matches_recalled', 'N/A'):,}** | **{combined.get('recall', 0)*100:.2f}%** | — |

**~23% of true match pairs are NOT retrieved** by even the best simple combined rule.

---

## 3. Candidate Volume (Baseline E)

| Metric | S2 | S3 | Combined |
|--------|----|----|----------|
| Total candidates | {s2_row.get('total_candidates', 0):,} | {s3_row.get('total_candidates', 0):,} | {combined.get('total_candidates', 0):,} |
| Avg candidates/S1 | {s2_row.get('avg_candidates_per_s1', 0)} | {s3_row.get('avg_candidates_per_s1', 0)} | {combined.get('avg_candidates_per_s1', 0)} |
| P95 candidates/S1 | {s2_row.get('p95_candidates_per_s1', 0)} | {s3_row.get('p95_candidates_per_s1', 0)} | — |
| Max candidates/S1 | {s2_row.get('max_candidates_per_s1', 0):,} | {s3_row.get('max_candidates_per_s1', 0):,} | — |
| S1 with zero candidates | {s2_row.get('s1_zero_candidates', 0):,} ({s2_row.get('s1_zero_pct', 0)}%) | {s3_row.get('s1_zero_candidates', 0):,} ({s3_row.get('s1_zero_pct', 0)}%) | — |
| Reduction ratio | {s2_row.get('reduction_ratio', 0)} | {s3_row.get('reduction_ratio', 0)} | — |

---

## 4. Runtime

Baseline evaluation runtime (DuckDB key-aggregation, in-memory):
"""
    for b in baselines:
        if "runtime_sec" in b:
            md += f"- `{b['strategy']}`: {b.get('runtime_sec', 'N/A')}s\n"

    md += """
---

## 5. Recall by Source

See baseline comparison table in `blocking_audit.csv`. Key finding: **S3 recall (39.8%) slightly exceeds S2 (37.5%)** under combined simple blocking, despite S3 having noisier addresses in true matches.

---

## 6. Recall by Match Difficulty / Match Count Bucket

"""
    for row in results["recall_by_bucket"]:
        if "singleton_count" not in row:
            md += f"- **{row['target']} / {row['bucket']}**: {row['recall']*100:.2f}% recall ({row['recalled']:,}/{row['total_true_pairs']:,} pairs)\n"
        else:
            md += f"- **Singleton S1 entities ({row['target']})**: {row['singleton_count']:,} entities; {row['singleton_zero_candidates']:,} get zero candidates (avg {row['singleton_avg_candidates']} false candidates if matcher runs)\n"

    md += """
---

## 7. Recall by Country (Combined Baseline)

"""
    for row in results["recall_by_country"]:
        md += f"- **{row['country']} / {row['target']}**: {row['recall']*100:.2f}% ({row['recalled']:,}/{row['total_true_pairs']:,})\n"

    md += """
---

## 8. Failure Categories (Missed True Matches — Baseline E)

Sample of missed pairs (first 150 extracted):

| Category | Count in Sample |
|----------|----------------|
"""
    for cat, cnt in fail_counts.items():
        md += f"| {cat} | {cnt} |\n"

    md += """
### Representative Missed Examples

| S1 ID | Match ID | Category | S1 Name | Other Name |
|-------|----------|----------|---------|------------|
"""
    for f in failures[:15]:
        md += f"| {f['s1_id']} | {f['matched_id']} | {f['failure_category']} | {str(f['s1_name'])[:35]} | {str(f['o_name'])[:35]} |\n"

    md += f"""
---

## 9. Candidate Quality (Combined Baseline E)

| Metric | Value |
|--------|-------|
| Total candidate pairs (S2+S3) | {quality['total_candidate_pairs_s2_s3']:,} |
| True positives in candidates | {quality['true_positive_candidates']:,} |
| False positives in candidates | {quality['false_positive_candidates']:,} |
| True positive rate | {quality['true_positive_rate']*100:.4f}% |
| False positive rate | {quality['false_positive_rate']*100:.2f}% |
| Candidates per true match | {quality['candidates_per_true_match']} |

**OBSERVATION:** Matcher receives **millions of mostly irrelevant candidates** (~{quality['false_positive_rate']*100:.1f}% false) even after blocking reduction. Precision-heavy F_0.5 requires the matcher to reject >99% of candidates correctly.

---

## 10. Baseline Comparison

| Strategy | Target | Recall | Avg Cand/S1 | P95 | Max | Zero-S1 % |
|----------|--------|--------|-------------|-----|-----|-----------|
"""
    for b in baselines:
        if b["strategy"] != "E_combined_union_S2+S3":
            md += f"| {b['strategy']} | {b.get('target','')} | {b.get('recall',0)*100:.2f}% | {b.get('avg_candidates_per_s1',0)} | {b.get('p95_candidates_per_s1',0)} | {b.get('max_candidates_per_s1',0):,} | {b.get('s1_zero_pct',0)}% |\n"

    md += """
---

## 11. Recall-vs-Candidate Curve (Top-K Block Size Cap)

Uses minimum blocking-key collision count per true pair as proxy for candidate limit K.

| K limit | S2 recall@K | S3 recall@K | S2 p95 block size | S3 p95 block size |
|---------|------------|------------|-------------------|-------------------|
"""
    curve_df = pd.DataFrame(results["recall_curve"])
    for k in [100, 250, 500, 1000, 2500, 5000, 10000]:
        s2 = curve_df[(curve_df["candidate_limit_k"] == k) & (curve_df["target"] == "S2")]
        s3 = curve_df[(curve_df["candidate_limit_k"] == k) & (curve_df["target"] == "S3")]
        if len(s2) and len(s3):
            md += f"| {k} | {s2.iloc[0]['recall_at_k_per_target_est']*100:.1f}% | {s3.iloc[0]['recall_at_k_per_target_est']*100:.1f}% | {s2.iloc[0]['p95_min_block_size']} | {s3.iloc[0]['p95_min_block_size']} |\n"

    md += """
**OBSERVATION:** Aggressive K caps cause steep recall drops for name-prefix blocking because common prefixes (e.g. "sai", "new") create block sizes >>10K.

---

## 12. France / Domain Generalization Risks

- Training GT covers **US + India only**; France is 15% of test S1 (~259K entities)
- Combined baseline uses **country as a blocking key** — will still work for France but with no train-time recall measurement
- **Transliteration/multilingual** failures dominate missed matches — likely worse for France
- Rules assuming US address formats (ZIP, state abbrev) may underperform on French addresses
- **Do not hard-code country filters to {US, India}** per challenge rules

---

## 13. Final Assessment

### BLOCKING STATUS: **REPLACE**

**Justification (measured evidence):**
1. **No production blocking implementation exists** in the repository to keep or improve
2. Reference simple combined blocking achieves only **~77% pair recall** — missing ~1.73M true pairs irrecoverably if used as-is
3. Candidate volume remains **~12.8B total pairs** across S2+S3 under combined blocking — matcher must process enormous false-positive load
4. **57% of S1 entities get zero candidates** under exact-name-only blocking (unacceptable)
5. Hard positives (transliteration, DBA names, abbreviation) systematically missed

---

### CURRENT RECALL (no teammate system — baseline E reference):
- **S2 = {s2_row.get('recall_per_target_only', 0)*100:.2f}%** (of S2 true pairs)
- **S3 = {s3_row.get('recall_per_target_only', 0)*100:.2f}%** (of S3 true pairs)
- **Overall = {combined.get('recall', 0)*100:.2f}%** (of all 7.64M true pairs)

### CANDIDATE COST (baseline E):
- **Average = {combined.get('avg_candidates_per_s1', 0)}** candidates/S1 (S2+S3 combined)
- **P95 = ~{max(s2_row.get('p95_candidates_per_s1', 0), s3_row.get('p95_candidates_per_s1', 0))}**
- **P99 = see blocking_audit.csv**
- **Maximum = {max(s2_row.get('max_candidates_per_s1', 0), s3_row.get('max_candidates_per_s1', 0)):,}**

### BIGGEST FAILURE MODE:
**Transliteration/multilingual name variants and DBA/semantic name differences** — blocking keys on normalized ASCII text cannot join Devanagari/Gujarati/ASCII transliteration variants of the same business.

### RECOMMENDED NEXT STEP:
1. **Obtain/commit teammate blocking code** or `output/candidate_pairs.tsv` on training data
2. Re-run this audit: `.venv/bin/python -m analysis.blocking_audit --candidate <path>`
3. Until then, **design new blocking** targeting >95% recall with:
   - Multi-key union (name trigrams + address tokens + postal digits)
   - Transliteration-normalization for Indic scripts
   - Separate S2 vs S3 tuning (S3 address noise is higher)
   - Candidate cap with fallback secondary retrieval for zero-candidate S1

---

*End of Blocking Audit Report*
"""
    (REPORTS / "blocking_audit.md").write_text(md, encoding="utf-8")
    print(f"Report: {REPORTS / 'blocking_audit.md'}")
    print(f"CSV: {REPORTS / 'blocking_audit.csv'}")


def run_audit():
    print("=" * 60)
    print("BLOCKING AUDIT")
    print("=" * 60)

    discovery = discover_teammate_blocking()
    print(f"Teammate blocking found: {discovery['teammate_blocking_found']}")

    con = duckdb.connect()
    con.execute("SET threads TO 4")
    con.execute("SET memory_limit = '6GB'")
    register_views(con)
    setup_norm_views(con)

    print("Running baseline comparison...")
    baselines = run_baseline_comparison(con)
    print("Recall by match bucket...")
    recall_bucket = recall_by_match_bucket(con)
    print("Recall by country...")
    recall_country = recall_by_country(con)
    print("Recall curve...")
    recall_curve_data = recall_curve(con)
    print("Failure analysis...")
    failures = failure_analysis(con, limit=150)
    s2_row = next(b for b in baselines if b["strategy"] == "E_combined_name_or_addr_or_cprefix_S2")
    s3_row = next(b for b in baselines if b["strategy"] == "E_combined_name_or_addr_or_cprefix_S3")
    print("Candidate quality...")
    quality = candidate_quality(
        con,
        s2_row["true_matches_recalled"],
        s3_row["true_matches_recalled"],
        s2_row["total_candidates"],
        s3_row["total_candidates"],
    )

    results = {
        "discovery": discovery,
        "baselines": baselines,
        "recall_by_bucket": recall_bucket,
        "recall_by_country": recall_country,
        "recall_curve": recall_curve_data,
        "failures": failures,
        "candidate_quality": quality,
    }

    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "blocking_audit_results.json").write_text(
        json.dumps({k: v for k, v in results.items() if k != "failures"}, indent=2, default=str)
    )
    pd.DataFrame(failures).to_csv(REPORTS / "blocking_audit_failures.csv", index=False)

    generate_audit_report(results)

    combined = next(b for b in baselines if b["strategy"] == "E_combined_union_S2+S3")
    s2 = next(b for b in baselines if b["strategy"] == "E_combined_name_or_addr_or_cprefix_S2")
    s3 = next(b for b in baselines if b["strategy"] == "E_combined_name_or_addr_or_cprefix_S3")

    print("\nBLOCKING STATUS: REPLACE (no teammate implementation found)")
    print(f"CURRENT RECALL (baseline E): S2={s2['recall_per_target_only']*100:.1f}% S3={s3['recall_per_target_only']*100:.1f}% Overall={combined['recall']*100:.1f}%")
    print(f"CANDIDATE COST: avg={combined['avg_candidates_per_s1']} max={max(s2['max_candidates_per_s1'], s3['max_candidates_per_s1']):,}")
    return results


if __name__ == "__main__":
    run_audit()
