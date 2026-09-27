"""Phase 8: Cross-Source Record Relationships."""
import duckdb


def analyze_cross_source(con: duckdb.DuckDBPyConnection) -> dict:
    """Multiplicity and duplicate analysis across sources."""
    result = {}

    for src, view in [("S2", "train_S2"), ("S3", "train_S3")]:
        # Exact duplicate businesses within source
        exact_dups = con.execute(f"""
            SELECT COUNT(*) AS dup_groups, SUM(cnt) AS dup_rows
            FROM (
                SELECT business_name, business_address, country, COUNT(*) AS cnt
                FROM {view}
                GROUP BY business_name, business_address, country
                HAVING COUNT(*) > 1
            )
        """).fetchdf().iloc[0].to_dict()
        result[f"{src}_exact_duplicate_groups"] = exact_dups

        # Near duplicate names
        name_dups = con.execute(f"""
            SELECT COUNT(*) AS dup_name_groups, SUM(cnt) AS dup_rows
            FROM (
                SELECT LOWER(TRIM(business_name)) AS norm_name, COUNT(*) AS cnt
                FROM {view} WHERE business_name IS NOT NULL AND TRIM(business_name) != ''
                GROUP BY LOWER(TRIM(business_name)) HAVING COUNT(*) > 1
            )
        """).fetchdf().iloc[0].to_dict()
        result[f"{src}_duplicate_name_groups"] = name_dups

    # Multiplicity distributions
    result["s1_to_s2_multiplicity"] = con.execute("""
        SELECT s2_match_count AS matches, COUNT(*) AS s1_count
        FROM gt_summary GROUP BY s2_match_count ORDER BY s2_match_count
    """).fetchdf().to_dict("records")

    result["s1_to_s3_multiplicity"] = con.execute("""
        SELECT s3_match_count AS matches, COUNT(*) AS s1_count
        FROM gt_summary GROUP BY s3_match_count ORDER BY s3_match_count
    """).fetchdf().to_dict("records")

    result["s2_to_s1_multiplicity"] = con.execute("""
        SELECT s1_count, COUNT(*) AS s2_entity_count
        FROM (
            SELECT matched_id, COUNT(DISTINCT s1_id) AS s1_count
            FROM gt_pairs WHERE match_source = 'S2'
            GROUP BY matched_id
        ) t
        GROUP BY s1_count ORDER BY s1_count
    """).fetchdf().to_dict("records")

    result["s3_to_s1_multiplicity"] = con.execute("""
        SELECT s1_count, COUNT(*) AS s3_entity_count
        FROM (
            SELECT matched_id, COUNT(DISTINCT s1_id) AS s1_count
            FROM gt_pairs WHERE match_source = 'S3'
            GROUP BY matched_id
        ) t
        GROUP BY s1_count ORDER BY s1_count
    """).fetchdf().to_dict("records")

    # Matching type summary
    s2_one_to_one = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT matched_id FROM gt_pairs WHERE match_source='S2'
            GROUP BY matched_id HAVING COUNT(DISTINCT s1_id) = 1
        )
    """).fetchone()[0]
    s2_total_matched = con.execute("""
        SELECT COUNT(DISTINCT matched_id) FROM gt_pairs WHERE match_source='S2'
    """).fetchone()[0]
    s3_one_to_one = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT matched_id FROM gt_pairs WHERE match_source='S3'
            GROUP BY matched_id HAVING COUNT(DISTINCT s1_id) = 1
        )
    """).fetchone()[0]
    s3_total_matched = con.execute("""
        SELECT COUNT(DISTINCT matched_id) FROM gt_pairs WHERE match_source='S3'
    """).fetchone()[0]

    result["matching_type_summary"] = {
        "S2_entities_in_gt": s2_total_matched,
        "S2_one_to_one_pct": round(100 * s2_one_to_one / s2_total_matched, 2) if s2_total_matched else 0,
        "S2_many_to_one_count": s2_total_matched - s2_one_to_one,
        "S3_entities_in_gt": s3_total_matched,
        "S3_one_to_one_pct": round(100 * s3_one_to_one / s3_total_matched, 2) if s3_total_matched else 0,
        "S3_many_to_one_count": s3_total_matched - s3_one_to_one,
    }

    return result
