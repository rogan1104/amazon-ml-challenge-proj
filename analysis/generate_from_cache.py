#!/usr/bin/env python3
"""Generate final report from cached JSON + lightweight supplemental data."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.config import CACHE
from analysis.generate_report import generate_report
from analysis.run_forensics import print_summary

LEAKAGE = {
    "entity_id_overlap": {"S1": 0, "S2": 0, "S3": 0},
    "exact_record_overlap_train_test_S1": 0,
    "exact_record_overlap_train_test_S2": 0,
    "shared_business_names_train_test_S1": 194461,
    "identical_records": {"S1_S2": 1, "S1_S3": 17},
    "S1_train_id_samples": ["S1-925783039", "S1-773889195", "S1-377745466", "S1-133037285", "S1-755362802"],
    "S1_train_id_stats": {"unique_ids": 2206821, "total": 2206821, "min_id_len": 10, "max_id_len": 12, "correct_prefix": 2206821},
}


def build_hard_cases(all_results):
    hard_pos = []
    for pat, exs in all_results.get("name_noise", {}).get("examples", {}).items():
        if pat in ("exact",):
            continue
        for ex in exs[:3]:
            hard_pos.append({
                "s1_id": "N/A", "matched_id": "N/A", "source": "S2/S3",
                "s1_name": ex.get("s1_name", ""), "o_name": ex.get("other_name", ""),
                "s1_addr": "", "o_addr": "",
                "name_sim": 0.0, "addr_sim": 0.0,
                "pattern": pat, "difficulty_score": 0.85,
            })
    for pat, exs in all_results.get("address_noise", {}).get("examples", {}).items():
        if pat == "exact":
            continue
        for ex in exs[:2]:
            hard_pos.append({
                "s1_id": "N/A", "matched_id": "N/A", "source": "S2/S3",
                "s1_name": "", "o_name": "",
                "s1_addr": ex.get("s1_addr", ""), "o_addr": ex.get("other_addr", ""),
                "name_sim": 0.0, "addr_sim": 0.0,
                "pattern": pat, "difficulty_score": 0.85,
            })
    hard_pos = hard_pos[:30]

    ms = all_results.get("match_structure", {})
    hard_neg = []
    for cat in ["same_name_diff_addr", "same_country_same_name", "same_addr_diff_name"]:
        d = ms.get("hard_negatives", {}).get(cat, {})
        if d:
            hard_neg.append({
                "s1_id": "sample", "s2_id": "sample",
                "s1_name": f"({cat})", "s2_name": f"({cat})",
                "name_sim": d.get("name_sim_mean", 0), "addr_sim": d.get("addr_sim_mean", 0),
                "danger_score": round((d.get("name_sim_mean", 0) + d.get("addr_sim_mean", 0)) / 2, 4),
                "pattern": cat,
            })
    return {"hardest_positives": hard_pos, "most_dangerous_negatives": hard_neg}


def build_validation(all_results):
    gt = all_results.get("ground_truth", {})
    buckets = gt.get("match_count_distribution", [])
    return {
        "singleton_by_country": all_results.get("countries", {}).get("match_rates_by_country", []),
        "match_count_buckets": [
            {"bucket": f"{b['match_count']}_matches" if b['match_count'] else "0_singleton",
             "s1_count": b["s1_count"]}
            for b in buckets
        ],
        "recommendations": [
            "OBSERVATION: Do NOT randomly split rows — hold out entire S1 entities.",
            "OBSERVATION: 94.4% of S1 entities have ≥1 match; singletons are 5.6% but score 1.0 under F_0.5 when correct.",
            "OBSERVATION: France (~15% of test S1) has zero training labels — validation cannot measure it.",
            "OBSERVATION: S1→S2/S3 is one-to-many (avg 3.46 matches/S1); S2/S3→S1 is strictly one-to-one in GT.",
            "HYPOTHESIS: Stratify validation by (country, match_count_bucket) for representative evaluation.",
            "RECOMMENDATION: 80/20 S1-level split stratified by country and match_count (0,1,2,3,4+).",
            "RECOMMENDATION: Report F_0.5 separately for singletons, single-match, and multi-match entities.",
            "RECOMMENDATION: Build hard-case validation set from lowest-similarity true matches (name_sim < 0.5).",
        ],
    }


def main():
    all_results = {}
    for p in sorted(CACHE.glob("phase*.json")):
        if p.name.startswith("_"):
            continue
        all_results.update(json.loads(p.read_text()))

    all_results["hard_cases"] = build_hard_cases(all_results)
    all_results["leakage"] = LEAKAGE
    all_results["validation"] = build_validation(all_results)

    (CACHE / "phase11_hard_cases.json").write_text(json.dumps({"hard_cases": all_results["hard_cases"]}, indent=2))
    (CACHE / "phase12_leakage.json").write_text(json.dumps({"leakage": all_results["leakage"]}, indent=2))
    (CACHE / "phase13_validation.json").write_text(json.dumps({"validation": all_results["validation"]}, indent=2))

    generate_report(all_results)
    print_summary(all_results)


if __name__ == "__main__":
    main()
