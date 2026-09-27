#!/usr/bin/env python3
"""Fast blocking audit using cached forensics + targeted DuckDB queries."""
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.config import REPORTS, CACHE, ROOT
from analysis.phase0_inventory import register_views
from analysis.blocking_audit import (
    setup_norm_views, recall_by_country, recall_curve,
    categorize_failure, generate_audit_report,
)

TOTAL_TRUE = 7638365
TOTAL_S1 = 2206821
TOTAL_S2 = 5034616
TOTAL_S3 = 5285603


def discover():
    impl = list((ROOT / "code").glob("**/*")) if (ROOT / "code").exists() else []
    cand = list(ROOT.glob("**/candidate_pairs.tsv"))
    cand = [c for c in cand if ".venv" not in str(c)]
    return {
        "teammate_blocking_found": bool(impl or cand),
        "implementations": [str(p) for p in impl],
        "candidate_outputs": [str(c) for c in cand],
        "search_notes": [
            f"{'EXISTS' if (ROOT/'code').exists() else 'MISSING'}: {ROOT/'code'}",
            f"{'EXISTS' if (ROOT/'output').exists() else 'MISSING'}: {ROOT/'output'/'candidate_pairs.tsv'}",
            f"{'EXISTS' if (ROOT/'src').exists() else 'MISSING'}: {ROOT/'src'}",
            "NOTE: analysis/phase10_blocking.py is forensics simulation, NOT production blocking",
            "NOTE: utils/validate_submission.py validates format only",
        ],
    }


def baselines_from_cache():
    data = json.loads((CACHE / "phase10_blocking.json").read_text())["blocking"]["strategies"]
    rows = []
    for s in data:
        name = s["strategy"].replace(" -> ", "_").replace(" ", "_")
        src = "S2" if "S2" in s["strategy"] else "S3"
        tgt_total = 3693619 if src == "S2" else 3944746
        rows.append({
            "strategy": name,
            "target": src,
            "true_matches_recalled": s["true_matches_recalled"],
            "recall": s["true_match_recall"],
            "recall_per_target_only": round(s["true_matches_recalled"] / tgt_total, 6),
            "total_candidates": s["total_candidate_pairs"],
            "avg_candidates_per_s1": s.get("avg_candidates_per_s1", 0),
            "median_candidates_per_s1": s.get("median_candidates_per_s1", 0),
            "p95_candidates_per_s1": s.get("p95_candidates_per_s1", 0),
            "max_candidates_per_s1": s.get("max_candidates_for_one_s1", 0),
            "s1_zero_candidates": s.get("s1_with_zero_candidates", 0),
            "s1_zero_pct": s.get("s1_zero_candidate_pct", 0),
            "reduction_ratio": s.get("reduction_ratio", 0),
        })
    for r in rows:
        if "combined" in r["strategy"] and "S2" in r["strategy"]:
            r["strategy"] = "E_combined_name_or_addr_or_cprefix_S2"
        elif "combined" in r["strategy"] and "S3" in r["strategy"]:
            r["strategy"] = "E_combined_name_or_addr_or_cprefix_S3"
        elif r["strategy"].startswith("exact_norm_name"):
            r["strategy"] = "A_exact_norm_name_" + r["target"]
        elif r["strategy"].startswith("exact_norm_address"):
            r["strategy"] = "B_exact_norm_address_" + r["target"]
        elif "country + name_prefix5" in r["strategy"]:
            r["strategy"] = "C_country_name_prefix5_" + r["target"]
        elif r["strategy"].startswith("name_first_token"):
            r["strategy"] = "D_name_first_token_" + r["target"]

    s2c = next(r for r in rows if r["strategy"] == "E_combined_name_or_addr_or_cprefix_S2")
    s3c = next(r for r in rows if r["strategy"] == "E_combined_name_or_addr_or_cprefix_S3")
    rows.append({
        "strategy": "E_combined_union_S2+S3",
        "target": "S2+S3",
        "true_matches_recalled": s2c["true_matches_recalled"] + s3c["true_matches_recalled"],
        "recall": round((s2c["true_matches_recalled"] + s3c["true_matches_recalled"]) / TOTAL_TRUE, 6),
        "total_candidates": s2c["total_candidates"] + s3c["total_candidates"],
        "avg_candidates_per_s1": round(s2c["avg_candidates_per_s1"] + s3c["avg_candidates_per_s1"], 2),
    })
    return rows


def recall_buckets_fast(con):
    rows = []
    for src, ov in [("S2", "s2_norm"), ("S3", "s3_norm")]:
        df = con.execute(f"""
            WITH recalled AS (
                SELECT g.s1_id, g.matched_id FROM gt_pairs g
                JOIN s1_norm s1 ON g.s1_id = s1.entity_id JOIN {ov} o ON g.matched_id = o.entity_id
                WHERE g.match_source='{src}' AND (
                    (s1.norm_name=o.norm_name AND s1.norm_name!='')
                    OR (s1.norm_addr=o.norm_addr AND s1.norm_addr!='')
                    OR (s1.country=o.country AND s1.name_prefix5=o.name_prefix5 AND LENGTH(s1.name_prefix5)>=3))
            )
            SELECT CASE WHEN gs.match_count=0 THEN '0_singleton' WHEN gs.match_count=1 THEN '1_match'
                        WHEN gs.match_count BETWEEN 2 AND 3 THEN '2-3_matches' ELSE '4+_matches' END bucket,
                   COUNT(*) total, SUM(CASE WHEN r.s1_id IS NOT NULL THEN 1 ELSE 0 END) recalled
            FROM gt_pairs g JOIN gt_summary gs ON g.s1_id=gs.s1_id
            LEFT JOIN recalled r ON g.s1_id=r.s1_id AND g.matched_id=r.matched_id
            WHERE g.match_source='{src}' GROUP BY 1 ORDER BY 1
        """).fetchdf()
        for _, r in df.iterrows():
            rows.append({"target": src, "bucket": r["bucket"], "total_true_pairs": int(r["total"]),
                         "recalled": int(r["recalled"]), "missed": int(r["total"]-r["recalled"]),
                         "recall": round(r["recalled"]/r["total"], 6)})
    return rows


def failures_from_hard_cases():
    hc = json.loads((CACHE / "phase11_hard_cases.json").read_text())["hard_cases"]
    positives = hc.get("hardest_positives", [])
    failures = []
    for p in positives:
        failures.append({
            **p,
            "failure_category": categorize_failure(
                p.get("s1_name"), p.get("o_name"), p.get("s1_addr"), p.get("o_addr"), p.get("country", "")),
        })
    return failures[:150]


def main():
    print("Fast blocking audit...")
    discovery = discover()
    discovery["teammate_blocking_found"] = bool(discovery["candidate_outputs"])  # only real outputs count

    baselines = baselines_from_cache()

    con = duckdb.connect()
    con.execute("SET threads TO 4")
    register_views(con)
    setup_norm_views(con)

    print("Recall buckets...")
    buckets = recall_buckets_fast(con)
    print("Recall by country...")
    countries = recall_by_country(con)
    print("Recall curve (sampled)...")
    curve = recall_curve(con, sample_size=30000)
    failures = failures_from_hard_cases()

    s2 = next(b for b in baselines if b["strategy"] == "E_combined_name_or_addr_or_cprefix_S2")
    s3 = next(b for b in baselines if b["strategy"] == "E_combined_name_or_addr_or_cprefix_S3")
    quality = {
        "total_candidate_pairs_s2_s3": s2["total_candidates"] + s3["total_candidates"],
        "true_positive_candidates": s2["true_matches_recalled"] + s3["true_matches_recalled"],
        "false_positive_candidates": s2["total_candidates"] + s3["total_candidates"] - s2["true_matches_recalled"] - s3["true_matches_recalled"],
        "true_positive_rate": round((s2["true_matches_recalled"]+s3["true_matches_recalled"])/(s2["total_candidates"]+s3["total_candidates"]), 8),
        "false_positive_rate": round(1-(s2["true_matches_recalled"]+s3["true_matches_recalled"])/(s2["total_candidates"]+s3["total_candidates"]), 8),
        "candidates_per_true_match": round((s2["total_candidates"]+s3["total_candidates"])/TOTAL_TRUE, 2),
        "true_matches_total": TOTAL_TRUE,
    }
    quality["false_positive_rate"] = 1 - quality["true_positive_rate"]

    results = {"discovery": discovery, "baselines": baselines, "recall_by_bucket": buckets,
               "recall_by_country": countries, "recall_curve": curve, "failures": failures,
               "candidate_quality": quality}
    generate_audit_report(results)

    comb = next(b for b in baselines if b["strategy"] == "E_combined_union_S2+S3")
    print(f"\nBLOCKING STATUS: REPLACE")
    print(f"S2={s2['recall_per_target_only']*100:.1f}% S3={s3['recall_per_target_only']*100:.1f}% Overall={comb['recall']*100:.1f}%")


if __name__ == "__main__":
    main()
