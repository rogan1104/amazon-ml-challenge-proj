"""Generate the Dataset Intelligence Report from analysis results."""
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from analysis.config import REPORTS


def generate_report(all_results: dict):
    """Write markdown report and CSV summaries."""
    REPORTS.mkdir(parents=True, exist_ok=True)

    # CSV exports
    if "profiling" in all_results:
        rows = []
        for label, stats in all_results["profiling"].items():
            row = {k: v for k, v in stats.items() if k != "country_distribution"}
            rows.append(row)
        pd.DataFrame(rows).to_csv(REPORTS / "source_statistics.csv", index=False)

    if "countries" in all_results:
        c = all_results["countries"]
        country_rows = []
        for key, dist in c.items():
            if isinstance(dist, list) and dist and isinstance(dist[0], dict) and "country" in dist[0]:
                for d in dist:
                    country_rows.append({"source": key, **d})
        if country_rows:
            pd.DataFrame(country_rows).to_csv(REPORTS / "country_statistics.csv", index=False)

    if "ground_truth" in all_results:
        gt = all_results["ground_truth"]
        pd.DataFrame(gt.get("match_count_distribution", [])).to_csv(
            REPORTS / "match_multiplicity.csv", index=False)

    if "blocking" in all_results:
        pd.DataFrame(all_results["blocking"]["strategies"]).to_csv(
            REPORTS / "blocking_experiments.csv", index=False)

    if "signals" in all_results:
        pd.DataFrame(all_results["signals"]["signals"]).to_csv(
            REPORTS / "signal_analysis.csv", index=False)

    if "hard_cases" in all_results:
        hc = all_results["hard_cases"]
        rows = []
        for r in hc.get("hardest_positives", []):
            rows.append({**r, "case_type": "hard_positive"})
        for r in hc.get("most_dangerous_negatives", []):
            rows.append({**r, "case_type": "hard_negative"})
        if rows:
            pd.DataFrame(rows).to_csv(REPORTS / "hard_cases.csv", index=False)

    # Markdown report
    md = _build_markdown(all_results)
    (REPORTS / "dataset_intelligence_report.md").write_text(md, encoding="utf-8")
    print(f"Report written to {REPORTS / 'dataset_intelligence_report.md'}")


def _build_markdown(r: dict) -> str:
    lines = []
    w = lines.append

    w("# Dataset Intelligence Report")
    w(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w("**Challenge:** Amazon ML Challenge 2026 — Business Entity Resolution")
    w("\n---\n")

    # 1. Executive Summary
    w("## 1. Executive Summary")
    inv = r.get("inventory", {})
    gt = r.get("ground_truth", {})
    buckets = gt.get("buckets", {})
    w(f"- **Total dataset scale:** ~26M rows across 7 TSV files (~1.9 GB)")
    s1_ent = gt.get('total_s1_entities', 'N/A')
    w(f"- **Training S1 entities:** {s1_ent:,}" if isinstance(s1_ent, int) else f"- **Training S1 entities:** {s1_ent}")
    w(f"- **Singleton rate:** {buckets.get('singleton_pct', 'N/A')}%")
    w(f"- **Multi-match rate:** {buckets.get('multi_match_pct', 'N/A')}%")
    w(f"- **Average matches per S1:** {buckets.get('avg_matches', 'N/A'):.2f}" if buckets.get('avg_matches') else "- **Average matches per S1:** N/A")
    w(f"- **Countries in train:** {', '.join(r.get('countries', {}).get('train_countries', []))}")
    w(f"- **Test-only countries:** {', '.join(r.get('countries', {}).get('test_only_countries', []))}")
    w(f"- **Matching is predominantly one-to-many** (S1 → multiple S2/S3)")
    w(f"- **Key noise:** name abbreviations, address format variations, punctuation/case differences")
    w(f"- **Critical constraint:** F_0.5 metric penalizes false merges; singletons matter")

    # 2. Dataset Scale
    w("\n## 2. Dataset Scale\n")
    w("| File | Rows | Size (MB) | Columns |")
    w("|------|------|-----------|---------|")
    total_rows = 0
    total_mb = 0
    for label, info in inv.items():
        if label.startswith("_"):
            continue
        w(f"| {info['label']} | {info['row_count']:,} | {info['file_size_mb']} | {info['column_count']} |")
        total_rows += info["row_count"]
        total_mb += info["file_size_mb"]
    w(f"| **TOTAL** | **{total_rows:,}** | **{total_mb:.1f}** | |")

    # 3. Schema & Data Quality
    w("\n## 3. Schema & Data Quality\n")
    w("### Schema")
    w("- **Source files:** `entity_id`, `business_name`, `business_address`, `country`")
    w("- **Ground truth:** `source1_entity_id`, `matched_entity_ids` (comma-separated S2/S3 IDs)")
    w("- **Delimiter:** Tab-separated (verified)")
    w("- **Encoding:** UTF-8")
    w("\n### Data Quality Summary")
    if "profiling" in r:
        w("| Source | Rows | Missing Name % | Missing Addr % | Missing Country % | Dup ID % | Exact Dup Row % |")
        w("|--------|------|----------------|----------------|-------------------|----------|-----------------|")
        for label, s in r["profiling"].items():
            w(f"| {label} | {s['total_rows']:,} | {s['missing_business_name_pct']:.2f} | "
              f"{s['missing_business_address_pct']:.2f} | {s['missing_country_pct']:.2f} | "
              f"{s['duplicate_entity_id_pct']:.4f} | {s['exact_dup_row_pct']:.2f} |")

    # 4. Source Comparison
    w("\n## 4. Source Comparison\n")
    if "comparison" in r:
        for split, rows in r["comparison"].items():
            w(f"### {split.upper()}")
            w("| Source | Avg Name Len | Avg Addr Len | Name Missing % | Legal Suffix % | Non-ASCII Name % | Punct Name % |")
            w("|--------|-------------|-------------|----------------|----------------|-----------------|--------------|")
            for row in rows:
                w(f"| {row['source']} | {row['avg_name_len']:.1f} | {row['avg_addr_len']:.1f} | "
                  f"{row['name_missing_pct']:.2f} | {row['legal_suffix_pct']:.2f} | "
                  f"{row['non_ascii_name_pct']:.2f} | {row['name_punct_pct']:.2f} |")

    # 5. Ground Truth Structure
    w("\n## 5. Ground Truth Structure\n")
    if gt:
        w(f"- Total S1 entities: **{gt['total_s1_entities']:,}**")
        w(f"- Total true match pairs: **{gt.get('total_true_match_pairs', 'N/A'):,}**")
        b = gt.get("buckets", {})
        w(f"\n### Match Count Distribution")
        w("| Matches | S1 Count | % |")
        w("|---------|----------|---|")
        for d in gt.get("match_count_distribution", [])[:20]:
            pct = 100 * d["s1_count"] / gt["total_s1_entities"]
            w(f"| {d['match_count']} | {d['s1_count']:,} | {pct:.2f} |")
        if len(gt.get("match_count_distribution", [])) > 20:
            w("| ... | (see match_multiplicity.csv) | |")

        w(f"\n### Summary Buckets")
        w(f"- Zero matches (singletons): {b.get('zero_matches', 0):,} ({b.get('singleton_pct', 0)}%)")
        w(f"- Exactly 1 match: {b.get('one_match', 0):,} ({b.get('one_match_pct', 0)}%)")
        w(f"- 2 matches: {b.get('two_matches', 0):,}")
        w(f"- 3 matches: {b.get('three_matches', 0):,}")
        w(f"- 4+ matches: {b.get('four_plus_matches', 0):,}")
        w(f"- Max matches for one S1: {b.get('max_matches', 0)}")
        w(f"- S1 with both S2 and S3 matches: {b.get('s1_with_both_sources', 0):,}")
        w(f"- Total S2 matches in GT: {b.get('total_s2_matches', 0):,}")
        w(f"- Total S3 matches in GT: {b.get('total_s3_matches', 0):,}")

    # 6. Match Multiplicity
    w("\n## 6. Match Multiplicity\n")
    if "cross_source" in r:
        cs = r["cross_source"]
        mts = cs.get("matching_type_summary", {})
        w(f"- S2 entities in GT: {mts.get('S2_entities_in_gt', 'N/A'):,}")
        w(f"- S2 one-to-one: {mts.get('S2_one_to_one_pct', 'N/A')}%")
        w(f"- S2 many-to-one: {mts.get('S2_many_to_one_count', 'N/A'):,} entities matched to multiple S1")
        w(f"- S3 entities in GT: {mts.get('S3_entities_in_gt', 'N/A'):,}")
        w(f"- S3 one-to-one: {mts.get('S3_one_to_one_pct', 'N/A')}%")
        w(f"- S3 many-to-one: {mts.get('S3_many_to_one_count', 'N/A'):,}")
        w(f"\n**OBSERVATION:** Problem behaves as **one-to-many** (S1 → S2/S3) with some **many-to-one** on the S2/S3 side.")

    # 7. Name Noise
    w("\n## 7. Name Noise Analysis\n")
    if "name_noise" in r:
        freq = r["name_noise"].get("pattern_frequencies", {})
        w("| Pattern | Count | % of Pattern Hits |")
        w("|---------|-------|-------------------|")
        for pat, info in list(freq.items())[:20]:
            w(f"| {pat} | {info['count']:,} | {info['pct']}% |")
        w("\n### Representative Examples")
        examples = r["name_noise"].get("examples", {})
        for pat in list(examples.keys())[:5]:
            w(f"\n**{pat}:**")
            for ex in examples[pat][:3]:
                w(f"- S1: `{ex['s1_name']}` ↔ Other: `{ex['other_name']}`")

    # 8. Address Noise
    w("\n## 8. Address Noise Analysis\n")
    if "address_noise" in r:
        freq = r["address_noise"].get("pattern_frequencies", {})
        w("| Pattern | Count | % |")
        w("|---------|-------|---|")
        for pat, info in list(freq.items())[:20]:
            w(f"| {pat} | {info['count']:,} | {info['pct']}% |")
        w("\n### Representative Examples")
        examples = r["address_noise"].get("examples", {})
        for pat in list(examples.keys())[:5]:
            w(f"\n**{pat}:**")
            for ex in examples[pat][:3]:
                w(f"- S1: `{ex['s1_addr']}` ↔ Other: `{ex['other_addr']}`")

    # 9. Country Analysis
    w("\n## 9. Country Analysis\n")
    if "countries" in r:
        c = r["countries"]
        w(f"- **Train countries:** {', '.join(c.get('train_countries', []))}")
        w(f"- **Test countries:** {', '.join(c.get('test_countries', []))}")
        w(f"- **Test-only:** {', '.join(c.get('test_only_countries', []))}")
        w(f"\n### France Test Set")
        for src, info in c.get("france_test", {}).items():
            w(f"- Test {src}: {info.get('cnt', 0):,} France records ({info.get('pct_of_source', 0)}%)")
        w(f"\n### Match Rates by Country (Train)")
        w("| Country | S1 Count | Singletons | With Matches | Avg Matches | Singleton % |")
        w("|---------|----------|------------|--------------|-------------|-------------|")
        for row in c.get("match_rates_by_country", []):
            w(f"| {row['country']} | {row['s1_count']:,} | {row['singletons']:,} | "
              f"{row['with_matches']:,} | {row['avg_matches']:.2f} | {row['singleton_pct']:.2f} |")

    # 10. Cross-Source Relationships
    w("\n## 10. Cross-Source Record Relationships\n")
    if "cross_source" in r:
        cs = r["cross_source"]
        w(f"- S2 exact duplicate groups: {cs.get('S2_exact_duplicate_groups', {})}")
        w(f"- S3 exact duplicate groups: {cs.get('S3_exact_duplicate_groups', {})}")
        w(f"- S2 duplicate name groups: {cs.get('S2_duplicate_name_groups', {})}")
        w(f"- S3 duplicate name groups: {cs.get('S3_duplicate_name_groups', {})}")

    # 11. Signal/Feature Analysis
    w("\n## 11. Signal/Feature Analysis\n")
    if "signals" in r:
        w("| Signal | Condition Pairs | True Matches | P(match\\|condition) |")
        w("|--------|----------------|--------------|---------------------|")
        for s in r["signals"]["signals"]:
            w(f"| {s['signal']} | {s['condition_pairs']:,} | {s['true_matches']:,} | {s['p_match_given_condition']:.6f} |")

    # 12. Blocking Experiments
    w("\n## 12. Blocking Experiments\n")
    if "blocking" in r:
        w("| Strategy | Candidates | Avg/S1 | Recall | Zero-Cand S1 % | Max Cand |")
        w("|----------|-----------|--------|--------|--------------|----------|")
        for s in sorted(r["blocking"]["strategies"], key=lambda x: -x.get("true_match_recall", 0)):
            w(f"| {s['strategy']} | {s.get('total_candidate_pairs', 0):,} | "
              f"{s.get('avg_candidates_per_s1', 0)} | {s.get('true_match_recall', 0):.4f} | "
              f"{s.get('s1_zero_candidate_pct', 0):.2f} | {s.get('max_candidates_for_one_s1', 0):,} |")

    # 13. Hardest Positive Matches
    w("\n## 13. Hardest Positive Matches\n")
    if "hard_cases" in r:
        w("| S1 ID | Match ID | Name Sim | Addr Sim | S1 Name | Other Name |")
        w("|-------|----------|----------|----------|---------|------------|")
        for h in r["hard_cases"].get("hardest_positives", [])[:15]:
            w(f"| {h['s1_id']} | {h['matched_id']} | {h['name_sim']} | {h['addr_sim']} | "
              f"{h['s1_name'][:40]} | {h['o_name'][:40]} |")

    # 14. Hardest Negative Matches
    w("\n## 14. Hardest Negative Matches\n")
    if "hard_cases" in r:
        w("| S1 ID | S2 ID | Name Sim | Addr Sim | S1 Name | S2 Name |")
        w("|-------|-------|----------|----------|---------|---------|")
        for h in r["hard_cases"].get("most_dangerous_negatives", [])[:15]:
            w(f"| {h['s1_id']} | {h['s2_id']} | {h['name_sim']} | {h['addr_sim']} | "
              f"{h['s1_name'][:40]} | {h['s2_name'][:40]} |")

    # 15. Potential Leakage/Artifacts
    w("\n## 15. Potential Leakage/Artifacts\n")
    if "leakage" in r:
        leak = r["leakage"]
        w(f"- Entity ID overlap train/test: S1={leak['entity_id_overlap']['S1']}, "
          f"S2={leak['entity_id_overlap']['S2']}, S3={leak['entity_id_overlap']['S3']}")
        w(f"- Exact record overlap train/test S1: {leak.get('exact_record_overlap_train_test_S1', 'N/A'):,}")
        w(f"- Exact record overlap train/test S2: {leak.get('exact_record_overlap_train_test_S2', 'N/A'):,}")
        w(f"- Shared business names train/test S1: {leak.get('shared_business_names_train_test_S1', 'N/A'):,}")
        w(f"- Identical records S1-S2 in train: {leak.get('identical_records', {}).get('S1_S2', 'N/A'):,}")
        w(f"- Identical records S1-S3 in train: {leak.get('identical_records', {}).get('S1_S3', 'N/A'):,}")

    # 16. Validation Strategy
    w("\n## 16. Validation Strategy Recommendation\n")
    if "validation" in r:
        for rec in r["validation"].get("recommendations", []):
            w(f"- {rec}")

    # 17. Key Findings
    w("\n## 17. Key Findings\n")
    w("1. **Scale:** ~2.2M S1 train entities, ~5M S2, ~5.3M S3; test is ~78% of train size.")
    w("2. **Singletons are significant** — correctly predicting no-match is worth full credit.")
    w("3. **One-to-many matching** is the dominant pattern; some S2/S3 entities match multiple S1.")
    w("4. **Name and address noise is real and measurable** — not just theoretical.")
    w("5. **Country is a strong blocking signal** but France is test-only.")
    w("6. **Exact normalized name blocking** has high precision but misses many true matches.")
    w("7. **Combined blocking strategies** needed for acceptable recall.")
    w("8. **F_0.5 rewards precision** — conservative matching may outperform aggressive recall.")

    # 18. Engineering Implications
    w("\n## 18. Engineering Implications\n")
    w("- Blocking must achieve >95% recall or true matches are lost forever.")
    w("- Matching model must distinguish same-name/different-business pairs.")
    w("- Singleton detection is a first-class problem, not an edge case.")
    w("- Country-aware normalization needed but cannot rely on train-only countries.")
    w("- Candidate_pairs.tsv must contain all final matches (pipeline audit requirement).")
    w("- Memory-efficient processing essential for 5M+ record sources.")

    # 19. Open Questions
    w("\n## 19. Open Questions\n")
    w("- How well do train-derived patterns transfer to France test entities?")
    w("- What is the optimal blocking recall vs. candidate explosion tradeoff?")
    w("- Are S2 and S3 noise profiles sufficiently different to warrant source-specific models?")
    w("- How should many-to-one S2/S3 matches be handled in prediction?")
    w("- What is the public/private leaderboard split ratio?")

    # 20. Recommended Next Experiments
    w("\n## 20. Recommended Next Experiments\n")
    w("1. Prototype combined blocking pipeline and measure end-to-end recall on validation split.")
    w("2. Build stratified S1-level validation split and establish baseline F_0.5.")
    w("3. Feature ablation: name-only vs address-only vs combined similarity on validation.")
    w("4. Investigate France test records for address/name format patterns.")
    w("5. Test learned vs rule-based normalization on hardest positive cases.")
    w("6. Measure false-merge rate on hard negative pairs with candidate blocking strategies.")

    w("\n---\n*End of Dataset Intelligence Report*")
    return "\n".join(lines)
