#!/usr/bin/env python3
"""Complete report from cached phases + lightweight phases 11-13."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
from analysis.config import CACHE
from analysis.phase0_inventory import register_views
from analysis.phase11_12_13 import investigate_leakage, recommend_validation
from analysis.generate_report import generate_report
from analysis.run_forensics import print_summary
from analysis.normalize import normalize_name, normalize_address, _safe_str
from rapidfuzz.distance import Levenshtein

CACHE.mkdir(parents=True, exist_ok=True)


def load_all_cached():
    all_results = {}
    for p in sorted(CACHE.glob("phase*.json")):
        if p.name.startswith("_"):
            continue
        data = json.loads(p.read_text())
        all_results.update(data)
    return all_results


def quick_hard_cases(con, top_n=30):
    """Fast hard-case discovery on small sample."""
    pairs = con.execute("""
        WITH sp AS (
            SELECT s1_id, matched_id, match_source FROM gt_pairs
            USING SAMPLE 5000 (reservoir)
        )
        SELECT p.s1_id, p.matched_id, p.match_source,
               s1.business_name AS s1_name, s1.business_address AS s1_addr,
               COALESCE(s2.business_name, s3.business_name) AS o_name,
               COALESCE(s2.business_address, s3.business_address) AS o_addr
        FROM sp p
        JOIN train_S1 s1 ON p.s1_id = s1.entity_id
        LEFT JOIN train_S2 s2 ON p.matched_id = s2.entity_id AND p.match_source = 'S2'
        LEFT JOIN train_S3 s3 ON p.matched_id = s3.entity_id AND p.match_source = 'S3'
    """).fetchdf()

    hard_pos = []
    for r in pairs.itertuples():
        ns = Levenshtein.normalized_similarity(normalize_name(r.s1_name), normalize_name(r.o_name))
        ass = Levenshtein.normalized_similarity(normalize_address(r.s1_addr), normalize_address(r.o_addr))
        comb = (ns + ass) / 2
        hard_pos.append({
            "s1_id": r.s1_id, "matched_id": r.matched_id, "source": r.match_source,
            "s1_name": _safe_str(r.s1_name), "o_name": _safe_str(r.o_name),
            "s1_addr": _safe_str(r.s1_addr), "o_addr": _safe_str(r.o_addr),
            "name_sim": round(ns, 4), "addr_sim": round(ass, 4),
            "combined_sim": round(comb, 4), "difficulty_score": round(1 - comb, 4),
        })
    hard_pos.sort(key=lambda x: x["difficulty_score"], reverse=True)

    # Hard negatives: same name in S1/S2 sample, not in gt (limit 500)
    negs = con.execute("""
        WITH s1 AS (SELECT entity_id, business_name, business_address FROM train_S1 USING SAMPLE 500 (reservoir)),
             s2 AS (SELECT entity_id, business_name, business_address FROM train_S2 USING SAMPLE 1000 (reservoir)),
             pairs AS (
                 SELECT s1.entity_id s1_id, s2.entity_id s2_id,
                        s1.business_name s1_name, s2.business_name s2_name,
                        s1.business_address s1_addr, s2.business_address s2_addr
                 FROM s1 JOIN s2 ON LOWER(TRIM(s1.business_name)) = LOWER(TRIM(s2.business_name))
                 LIMIT 200
             )
        SELECT p.* FROM pairs p
        LEFT JOIN gt_pairs g ON p.s1_id = g.s1_id AND p.s2_id = g.matched_id
        WHERE g.s1_id IS NULL
    """).fetchdf()

    hard_neg = []
    for r in negs.itertuples():
        ns = Levenshtein.normalized_similarity(normalize_name(r.s1_name), normalize_name(r.s2_name))
        ass = Levenshtein.normalized_similarity(normalize_address(r.s1_addr), normalize_address(r.s2_addr))
        comb = (ns + ass) / 2
        hard_neg.append({
            "s1_id": r.s1_id, "s2_id": r.s2_id,
            "s1_name": _safe_str(r.s1_name), "s2_name": _safe_str(r.s2_name),
            "s1_addr": _safe_str(r.s1_addr), "s2_addr": _safe_str(r.s2_addr),
            "name_sim": round(ns, 4), "addr_sim": round(ass, 4),
            "danger_score": round(comb, 4),
        })
    hard_neg.sort(key=lambda x: x["danger_score"], reverse=True)
    return {"hardest_positives": hard_pos[:top_n], "most_dangerous_negatives": hard_neg[:top_n]}


if __name__ == "__main__":
    print("Loading cached results...")
    all_results = load_all_cached()
    print(f"  Loaded {len(all_results)} result sections from cache")

    con = duckdb.connect()
    con.execute("SET threads TO 4")
    register_views(con)

    if "hard_cases" not in all_results:
        print("Phase 11 (quick hard cases)...")
        t = time.time()
        all_results["hard_cases"] = quick_hard_cases(con)
        (CACHE / "phase11_hard_cases.json").write_text(
            json.dumps({"hard_cases": all_results["hard_cases"]}, indent=2))
        print(f"  Done in {time.time()-t:.1f}s")

    if "leakage" not in all_results:
        print("Phase 12 (leakage)...")
        t = time.time()
        all_results["leakage"] = investigate_leakage(con)
        (CACHE / "phase12_leakage.json").write_text(
            json.dumps({"leakage": all_results["leakage"]}, indent=2))
        print(f"  Done in {time.time()-t:.1f}s")

    if "validation" not in all_results:
        print("Phase 13 (validation)...")
        t = time.time()
        all_results["validation"] = recommend_validation(con)
        (CACHE / "phase13_validation.json").write_text(
            json.dumps({"validation": all_results["validation"]}, indent=2))
        print(f"  Done in {time.time()-t:.1f}s")

    print("Generating report...")
    generate_report(all_results)
    print_summary(all_results)
