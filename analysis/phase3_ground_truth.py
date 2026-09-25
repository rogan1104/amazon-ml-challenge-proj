"""Phase 3: Ground Truth Forensics."""
import duckdb


def analyze_ground_truth(con: duckdb.DuckDBPyConnection) -> dict:
    """Deep analysis of train_ground_truth.tsv."""
    gt = {}

    # Basic counts
    gt["total_s1_entities"] = con.execute("SELECT COUNT(*) FROM gt_summary").fetchone()[0]

    # Match count distribution
    dist_df = con.execute("""
        SELECT match_count, COUNT(*) AS s1_count
        FROM gt_summary
        GROUP BY match_count
        ORDER BY match_count
    """).fetchdf()
    gt["match_count_distribution"] = dist_df.to_dict("records")

    buckets = con.execute("""
        SELECT
            SUM(CASE WHEN match_count = 0 THEN 1 ELSE 0 END) AS zero_matches,
            SUM(CASE WHEN match_count = 1 THEN 1 ELSE 0 END) AS one_match,
            SUM(CASE WHEN match_count = 2 THEN 1 ELSE 0 END) AS two_matches,
            SUM(CASE WHEN match_count = 3 THEN 1 ELSE 0 END) AS three_matches,
            SUM(CASE WHEN match_count >= 4 THEN 1 ELSE 0 END) AS four_plus_matches,
            AVG(match_count) AS avg_matches,
            MAX(match_count) AS max_matches,
            MEDIAN(match_count) AS median_matches,
            SUM(s2_match_count) AS total_s2_matches,
            SUM(s3_match_count) AS total_s3_matches,
            SUM(CASE WHEN s2_match_count > 0 AND s3_match_count > 0 THEN 1 ELSE 0 END) AS s1_with_both_sources,
            SUM(CASE WHEN s2_match_count > 0 AND s3_match_count = 0 THEN 1 ELSE 0 END) AS s1_s2_only,
            SUM(CASE WHEN s2_match_count = 0 AND s3_match_count > 0 THEN 1 ELSE 0 END) AS s1_s3_only
        FROM gt_summary
    """).fetchdf().iloc[0].to_dict()

    total = gt["total_s1_entities"]
    buckets["singleton_pct"] = round(100 * buckets["zero_matches"] / total, 4)
    buckets["one_match_pct"] = round(100 * buckets["one_match"] / total, 4)
    buckets["multi_match_pct"] = round(100 * (total - buckets["zero_matches"]) / total, 4)
    gt["buckets"] = buckets

    # S2/S3 match count distributions per S1
    s2_dist = con.execute("""
        SELECT s2_match_count, COUNT(*) AS cnt
        FROM gt_summary GROUP BY s2_match_count ORDER BY s2_match_count
    """).fetchdf().to_dict("records")
    s3_dist = con.execute("""
        SELECT s3_match_count, COUNT(*) AS cnt
        FROM gt_summary GROUP BY s3_match_count ORDER BY s3_match_count
    """).fetchdf().to_dict("records")
    gt["s2_matches_per_s1_distribution"] = s2_dist
    gt["s3_matches_per_s1_distribution"] = s3_dist

    # S2 entity appearing for multiple S1
    s2_multi = con.execute("""
        SELECT matched_id, COUNT(DISTINCT s1_id) AS s1_count
        FROM gt_pairs WHERE match_source = 'S2'
        GROUP BY matched_id HAVING COUNT(DISTINCT s1_id) > 1
    """).fetchdf()
    gt["s2_entities_matched_to_multiple_s1"] = len(s2_multi)
    gt["s2_max_s1_per_entity"] = int(s2_multi["s1_count"].max()) if len(s2_multi) > 0 else 0
    gt["s2_multi_s1_examples"] = s2_multi.head(10).to_dict("records")

    s3_multi = con.execute("""
        SELECT matched_id, COUNT(DISTINCT s1_id) AS s1_count
        FROM gt_pairs WHERE match_source = 'S3'
        GROUP BY matched_id HAVING COUNT(DISTINCT s1_id) > 1
    """).fetchdf()
    gt["s3_entities_matched_to_multiple_s1"] = len(s3_multi)
    gt["s3_max_s1_per_entity"] = int(s3_multi["s1_count"].max()) if len(s3_multi) > 0 else 0
    gt["s3_multi_s1_examples"] = s3_multi.head(10).to_dict("records")

    # Duplicate IDs inside matched_entity_ids
    dup_in_list = con.execute("""
        WITH exploded AS (
            SELECT source1_entity_id, TRIM(x) AS mid
            FROM train_gt, UNNEST(string_split(matched_entity_ids, ',')) AS t(x)
            WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != ''
        ),
        dups AS (
            SELECT source1_entity_id, mid, COUNT(*) AS cnt
            FROM exploded GROUP BY source1_entity_id, mid HAVING COUNT(*) > 1
        )
        SELECT COUNT(*) AS dup_pairs, COUNT(DISTINCT source1_entity_id) AS affected_s1
        FROM dups
    """).fetchdf().iloc[0].to_dict()
    gt["duplicate_ids_in_lists"] = dup_in_list

    # Malformed records
    malformed = con.execute("""
        SELECT
            SUM(CASE WHEN source1_entity_id IS NULL OR TRIM(source1_entity_id)='' THEN 1 ELSE 0 END) AS missing_s1,
            SUM(CASE WHEN source1_entity_id NOT LIKE 'S1-%' THEN 1 ELSE 0 END) AS bad_s1_prefix,
            SUM(CASE WHEN matched_entity_ids LIKE '%S1-%' THEN 1 ELSE 0 END) AS s1_in_matches,
            SUM(CASE WHEN matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
                      AND matched_entity_ids NOT LIKE '%S2-%'
                      AND matched_entity_ids NOT LIKE '%S3-%' THEN 1 ELSE 0 END) AS invalid_match_ids
        FROM train_gt
    """).fetchdf().iloc[0].to_dict()
    gt["malformed_records"] = malformed

    # GT row count vs S1 count
    s1_count = con.execute("SELECT COUNT(*) FROM train_S1").fetchone()[0]
    gt_count = con.execute("SELECT COUNT(*) FROM train_gt").fetchone()[0]
    gt["s1_vs_gt_row_count"] = {"s1_rows": s1_count, "gt_rows": gt_count, "match": s1_count == gt_count}

    # Total true match pairs
    gt["total_true_match_pairs"] = con.execute("SELECT COUNT(*) FROM gt_pairs").fetchone()[0]

    return gt
