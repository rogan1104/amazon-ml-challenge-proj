#!/usr/bin/env python3
"""
Dataset Forensics Pipeline — Amazon ML Challenge 2026
Business Entity Resolution

Memory-efficient analysis using DuckDB. Restartable via cache.
"""
import json
import sys
import time
import duckdb
from pathlib import Path

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.config import REPORTS, CACHE
from analysis.phase0_inventory import run_inventory, register_views
from analysis.phase1_profiling import run_all_profiling
from analysis.phase2_comparison import compare_sources
from analysis.phase3_ground_truth import analyze_ground_truth
from analysis.phase4_match_structure import analyze_match_structure
from analysis.phase5_6_noise import analyze_name_noise, analyze_address_noise
from analysis.phase7_country import analyze_countries
from analysis.phase8_cross_source import analyze_cross_source
from analysis.phase9_signals import analyze_signals
from analysis.phase10_blocking import simulate_blocking
from analysis.phase11_12_13 import find_hard_cases, investigate_leakage, recommend_validation
from analysis.generate_report import generate_report


PHASES = [
    ("phase0_inventory", lambda con, r: r.update({"inventory": run_inventory(con)})),
    ("phase0_views", lambda con, r: register_views(con)),
    ("phase1_profiling", lambda con, r: r.update({"profiling": run_all_profiling(con)})),
    ("phase2_comparison", lambda con, r: r.update({"comparison": compare_sources(con)})),
    ("phase3_ground_truth", lambda con, r: r.update({"ground_truth": analyze_ground_truth(con)})),
    ("phase4_match_structure", lambda con, r: r.update({"match_structure": analyze_match_structure(con)})),
    ("phase5_name_noise", lambda con, r: r.update({"name_noise": analyze_name_noise(con)})),
    ("phase6_address_noise", lambda con, r: r.update({"address_noise": analyze_address_noise(con)})),
    ("phase7_country", lambda con, r: r.update({"countries": analyze_countries(con)})),
    ("phase8_cross_source", lambda con, r: r.update({"cross_source": analyze_cross_source(con)})),
    ("phase9_signals", lambda con, r: r.update({"signals": analyze_signals(con)})),
    ("phase10_blocking", lambda con, r: r.update({"blocking": simulate_blocking(con)})),
    ("phase11_hard_cases", lambda con, r: r.update({"hard_cases": find_hard_cases(con)})),
    ("phase12_leakage", lambda con, r: r.update({"leakage": investigate_leakage(con)})),
    ("phase13_validation", lambda con, r: r.update({"validation": recommend_validation(con)})),
]


def load_cache(phase_name: str):
    path = CACHE / f"{phase_name}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_cache(phase_name: str, data):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{phase_name}.json"

    def default(o):
        if hasattr(o, "item"):
            return o.item()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    path.write_text(json.dumps(data, indent=2, default=default))


def run_pipeline(skip_cached: bool = True):
    REPORTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")
    con.execute("SET threads TO 4")
    con.execute("SET memory_limit = '6GB'")
    CACHE.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{CACHE / 'duckdb_tmp'}'")

    # Views must exist every run (in-memory DB); not restorable from JSON cache
    print("[SETUP] Registering DuckDB views...")
    register_views(con)

    all_results = {}
    total_start = time.time()
    timings = {}

    for phase_name, phase_fn in PHASES:
        if phase_name == "phase0_views":
            continue  # handled above
        if skip_cached:
            cached = load_cache(phase_name)
            if cached is not None:
                print(f"[CACHE HIT] {phase_name}")
                all_results.update(cached)
                continue

        print(f"[RUNNING] {phase_name}...")
        t0 = time.time()
        try:
            phase_result = {}
            phase_fn(con, phase_result)
            elapsed = round(time.time() - t0, 2)
            timings[phase_name] = elapsed
            print(f"  ✓ {phase_name} completed in {elapsed}s")

            # Merge phase_result into all_results and cache
            cache_data = {k: v for k, v in phase_result.items()}
            save_cache(phase_name, cache_data)
            all_results.update(phase_result)
        except Exception as e:
            print(f"  ✗ {phase_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise  # Stop pipeline on failure

    all_results["timings"] = timings
    all_results["total_elapsed_sec"] = round(time.time() - total_start, 2)

    print("\n[GENERATING REPORT]")
    generate_report(all_results)

    print("\n[TERMINAL SUMMARY]")
    print_summary(all_results)

    save_cache("_full_results", all_results)
    return all_results


def print_summary(r: dict):
    inv = r.get("inventory", {})
    gt = r.get("ground_truth", {})
    buckets = gt.get("buckets", {})
    prof = r.get("profiling", {})

    print("\n" + "=" * 60)
    print("DATASET SCALE")
    print("=" * 60)
    for key in ["train_S1", "train_S2", "train_S3", "test_S1", "test_S2", "test_S3"]:
        if key in inv:
            print(f"  {key}: {inv[key]['row_count']:,} rows ({inv[key]['file_size_mb']} MB)")

    print("\n" + "=" * 60)
    print("GROUND TRUTH")
    print("=" * 60)
    print(f"  Singleton %: {buckets.get('singleton_pct', 'N/A')}")
    print(f"  1-match %: {buckets.get('one_match_pct', 'N/A')}")
    print(f"  Multi-match %: {buckets.get('multi_match_pct', 'N/A')}")
    print(f"  Avg matches/S1: {buckets.get('avg_matches', 'N/A'):.2f}" if buckets.get('avg_matches') else "  Avg matches/S1: N/A")
    print(f"  Max matches/S1: {buckets.get('max_matches', 'N/A')}")
    print(f"  S2 matches: {buckets.get('total_s2_matches', 'N/A'):,}" if buckets.get('total_s2_matches') else "")
    print(f"  S3 matches: {buckets.get('total_s3_matches', 'N/A'):,}" if buckets.get('total_s3_matches') else "")

    print("\n" + "=" * 60)
    print("DATA QUALITY")
    print("=" * 60)
    for label, s in prof.items():
        print(f"  {label}: missing name={s.get('missing_business_name_pct',0):.2f}%, "
              f"missing addr={s.get('missing_business_address_pct',0):.2f}%, "
              f"dup rows={s.get('exact_dup_row_pct',0):.2f}%")

    print("\n" + "=" * 60)
    print("NOISE (top patterns)")
    print("=" * 60)
    if "name_noise" in r:
        print("  Name:")
        for pat, info in list(r["name_noise"].get("pattern_frequencies", {}).items())[:5]:
            print(f"    {pat}: {info['pct']}%")
    if "address_noise" in r:
        print("  Address:")
        for pat, info in list(r["address_noise"].get("pattern_frequencies", {}).items())[:5]:
            print(f"    {pat}: {info['pct']}%")

    print("\n" + "=" * 60)
    print("COUNTRY")
    print("=" * 60)
    c = r.get("countries", {})
    print(f"  Train: {', '.join(c.get('train_countries', []))}")
    print(f"  Test-only: {', '.join(c.get('test_only_countries', []))}")

    print("\n" + "=" * 60)
    print("BLOCKING (top strategies by recall)")
    print("=" * 60)
    if "blocking" in r:
        for s in sorted(r["blocking"]["strategies"], key=lambda x: -x.get("true_match_recall", 0))[:5]:
            print(f"  {s['strategy']}: recall={s.get('true_match_recall',0):.4f}, "
                  f"avg_cand={s.get('avg_candidates_per_s1',0)}, zero_s1={s.get('s1_zero_candidate_pct',0):.1f}%")

    print("\n" + "=" * 60)
    print("HARD CASES")
    print("=" * 60)
    if "hard_cases" in r:
        hp = r["hard_cases"].get("hardest_positives", [])
        hn = r["hard_cases"].get("most_dangerous_negatives", [])
        if hp:
            print(f"  Hardest positive: name_sim={hp[0]['name_sim']}, addr_sim={hp[0]['addr_sim']}")
            print(f"    {hp[0]['s1_name'][:50]} ↔ {hp[0]['o_name'][:50]}")
        if hn:
            print(f"  Most dangerous negative: name_sim={hn[0]['name_sim']}, addr_sim={hn[0]['addr_sim']}")
            print(f"    {hn[0]['s1_name'][:50]} ↔ {hn[0]['s2_name'][:50]}")

    print("\n" + "=" * 60)
    print("WHAT WE NOW KNOW")
    print("=" * 60)
    print("  • Dataset is ~26M rows, tab-separated, UTF-8, good quality")
    print("  • One-to-many matching dominates; singletons are significant")
    print("  • US + India in train; France test-only")
    print("  • Name/address noise patterns confirmed with measured frequencies")
    print("  • Exact name blocking high precision but insufficient recall alone")
    print("  • Combined blocking needed; country is strong signal")
    print("  • No entity ID overlap between train/test splits")
    print("  • F_0.5 requires precision-heavy, singleton-aware approach")

    print("\n" + "=" * 60)
    print("WHAT WE STILL DON'T KNOW")
    print("=" * 60)
    print("  • France entity resolution patterns (no train labels)")
    print("  • Optimal blocking recall/cost tradeoff for full pipeline")
    print("  • Public vs private leaderboard test split")
    print("  • Whether S2/S3 need source-specific matching logic")
    print("  • Best normalization rules for cross-country generalization")

    print("\n" + "=" * 60)
    print("RECOMMENDED NEXT EXPERIMENT")
    print("=" * 60)
    print("  Build stratified S1-level validation split (by country + match_count)")
    print("  → Prototype combined blocking (name OR addr OR country+prefix)")
    print("  → Measure validation F_0.5 with rule-based baseline matcher")
    print("  → Profile France test records for format patterns")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true", help="Ignore cache and rerun all phases")
    args = parser.parse_args()
    run_pipeline(skip_cached=not args.no_cache)
