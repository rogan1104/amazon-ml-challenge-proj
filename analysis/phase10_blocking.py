"""Phase 10: Blocking Strategy Simulation (key-aggregation, memory-efficient)."""
import duckdb
import time
from analysis.config import CACHE


def simulate_blocking(con: duckdb.DuckDBPyConnection) -> dict:
    """Simulate blocking via key collision stats + exact recall on gt_pairs."""
    t0 = time.time()
    results = []

    total_s1 = con.execute("SELECT COUNT(*) FROM train_S1").fetchone()[0]
    total_s2 = con.execute("SELECT COUNT(*) FROM train_S2").fetchone()[0]
    total_s3 = con.execute("SELECT COUNT(*) FROM train_S3").fetchone()[0]
    total_true = con.execute("SELECT COUNT(*) FROM gt_pairs").fetchone()[0]

    con.execute("""
        CREATE OR REPLACE VIEW s1_norm AS
        SELECT entity_id, country,
               LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_name,''), '[^\\w\\s&]', ' ', 'g'))) AS norm_name,
               LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_address,''), '[^\\w\\s]', ' ', 'g'))) AS norm_addr,
               LEFT(LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_name,''), '[^\\w\\s]', ' ', 'g'))), 5) AS name_prefix5,
               LEFT(LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_name,''), '[^\\w\\s]', ' ', 'g'))), 10) AS name_prefix10,
               list_extract(string_split(LOWER(TRIM(COALESCE(business_name,''))), ' '), 1) AS name_first_token
        FROM train_S1
    """)
    con.execute("""
        CREATE OR REPLACE VIEW s2_norm AS
        SELECT entity_id, country,
               LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_name,''), '[^\\w\\s&]', ' ', 'g'))) AS norm_name,
               LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_address,''), '[^\\w\\s]', ' ', 'g'))) AS norm_addr,
               LEFT(LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_name,''), '[^\\w\\s]', ' ', 'g'))), 5) AS name_prefix5,
               LEFT(LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_name,''), '[^\\w\\s]', ' ', 'g'))), 10) AS name_prefix10,
               list_extract(string_split(LOWER(TRIM(COALESCE(business_name,''))), ' '), 1) AS name_first_token
        FROM train_S2
    """)
    con.execute("""
        CREATE OR REPLACE VIEW s3_norm AS
        SELECT entity_id, country,
               LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_name,''), '[^\\w\\s&]', ' ', 'g'))) AS norm_name,
               LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_address,''), '[^\\w\\s]', ' ', 'g'))) AS norm_addr,
               LEFT(LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_name,''), '[^\\w\\s]', ' ', 'g'))), 5) AS name_prefix5,
               LEFT(LOWER(TRIM(REGEXP_REPLACE(COALESCE(business_name,''), '[^\\w\\s]', ' ', 'g'))), 10) AS name_prefix10,
               list_extract(string_split(LOWER(TRIM(COALESCE(business_name,''))), ' '), 1) AS name_first_token
        FROM train_S3
    """)

    # (strategy_name, s1_key_expr, o_key_expr, join_extra, match_src, other_view, other_count)
    strategies = [
        ("exact_norm_name", "s1.norm_name", "o.norm_name", "s1.norm_name != ''", "S2", "s2_norm", total_s2),
        ("exact_norm_name", "s1.norm_name", "o.norm_name", "s1.norm_name != ''", "S3", "s3_norm", total_s3),
        ("name_prefix_5", "s1.name_prefix5", "o.name_prefix5", "LENGTH(s1.name_prefix5) >= 3", "S2", "s2_norm", total_s2),
        ("name_prefix_5", "s1.name_prefix5", "o.name_prefix5", "LENGTH(s1.name_prefix5) >= 3", "S3", "s3_norm", total_s3),
        ("name_prefix_10", "s1.name_prefix10", "o.name_prefix10", "LENGTH(s1.name_prefix10) >= 5", "S2", "s2_norm", total_s2),
        ("name_prefix_10", "s1.name_prefix10", "o.name_prefix10", "LENGTH(s1.name_prefix10) >= 5", "S3", "s3_norm", total_s3),
        ("name_first_token", "s1.name_first_token", "o.name_first_token", "LENGTH(s1.name_first_token) >= 3", "S2", "s2_norm", total_s2),
        ("name_first_token", "s1.name_first_token", "o.name_first_token", "LENGTH(s1.name_first_token) >= 3", "S3", "s3_norm", total_s3),
        ("country + name_prefix5", "s1.country || '|' || s1.name_prefix5", "o.country || '|' || o.name_prefix5",
         "LENGTH(s1.name_prefix5) >= 3", "S2", "s2_norm", total_s2),
        ("country + name_prefix5", "s1.country || '|' || s1.name_prefix5", "o.country || '|' || o.name_prefix5",
         "LENGTH(s1.name_prefix5) >= 3", "S3", "s3_norm", total_s3),
        ("country + norm_name", "s1.country || '|' || s1.norm_name", "o.country || '|' || o.norm_name",
         "s1.norm_name != ''", "S2", "s2_norm", total_s2),
        ("country + norm_name", "s1.country || '|' || s1.norm_name", "o.country || '|' || o.norm_name",
         "s1.norm_name != ''", "S3", "s3_norm", total_s3),
        ("exact_norm_address", "s1.norm_addr", "o.norm_addr", "s1.norm_addr != ''", "S2", "s2_norm", total_s2),
        ("exact_norm_address", "s1.norm_addr", "o.norm_addr", "s1.norm_addr != ''", "S3", "s3_norm", total_s3),
        ("country + norm_addr", "s1.country || '|' || s1.norm_addr", "o.country || '|' || o.norm_addr",
         "s1.norm_addr != ''", "S2", "s2_norm", total_s2),
        ("country + norm_addr", "s1.country || '|' || s1.norm_addr", "o.country || '|' || o.norm_addr",
         "s1.norm_addr != ''", "S3", "s3_norm", total_s3),
    ]

    for strat_name, s1_key, o_key, extra, match_src, other_view, other_count in strategies:
        full_name = f"{strat_name} -> {match_src}"

        # Recall: join gt_pairs with blocking condition only
        recalled = con.execute(f"""
            SELECT COUNT(*) FROM gt_pairs g
            JOIN s1_norm s1 ON g.s1_id = s1.entity_id
            JOIN {other_view} o ON g.matched_id = o.entity_id
            WHERE g.match_source = '{match_src}' AND {extra} AND {s1_key} = {o_key}
        """).fetchone()[0]

        # Candidate stats via key collision aggregation
        stats = con.execute(f"""
            WITH s1_keys AS (
                SELECT entity_id, {s1_key} AS bkey FROM s1_norm s1 WHERE {extra}
            ),
            o_keys AS (
                SELECT entity_id, {o_key} AS bkey FROM {other_view} o
            ),
            o_by_key AS (
                SELECT bkey, COUNT(*) AS o_cnt FROM o_keys GROUP BY bkey
            ),
            s1_cand AS (
                SELECT s1.entity_id, COALESCE(o.o_cnt, 0) AS n
                FROM s1_keys s1
                LEFT JOIN o_by_key o ON s1.bkey = o.bkey
            ),
            key_cross AS (
                SELECT SUM(s1_cnt * o_cnt) AS total_pairs
                FROM (
                    SELECT s1.bkey, COUNT(*) AS s1_cnt FROM s1_keys s1 GROUP BY s1.bkey
                ) s1
                JOIN o_by_key o ON s1.bkey = o.bkey
            )
            SELECT
                (SELECT total_pairs FROM key_cross) AS total_candidates,
                (SELECT AVG(n) FROM s1_cand WHERE n > 0) AS avg_candidates,
                (SELECT MEDIAN(n) FROM s1_cand WHERE n > 0) AS median_candidates,
                (SELECT MAX(n) FROM s1_cand) AS max_candidates,
                (SELECT QUANTILE_CONT(n, 0.95) FROM s1_cand WHERE n > 0) AS p95_candidates,
                (SELECT COUNT(*) FROM s1_cand WHERE n = 0) AS s1_zero
        """).fetchdf().iloc[0]

        total_candidates = int(stats["total_candidates"] or 0)
        s1_zero = int(stats["s1_zero"] or 0)

        results.append({
            "strategy": full_name,
            "total_candidate_pairs": total_candidates,
            "avg_candidates_per_s1": round(float(stats["avg_candidates"] or 0), 2),
            "median_candidates_per_s1": round(float(stats["median_candidates"] or 0), 2),
            "s1_with_zero_candidates": s1_zero,
            "s1_zero_candidate_pct": round(100 * s1_zero / total_s1, 4),
            "true_matches_recalled": int(recalled),
            "total_true_matches": total_true,
            "true_match_recall": round(recalled / total_true, 6) if total_true else 0,
            "reduction_ratio": round(1 - total_candidates / (total_s1 * other_count), 6) if total_s1 * other_count else 0,
            "max_candidates_for_one_s1": int(stats["max_candidates"] or 0),
            "p95_candidates_per_s1": round(float(stats["p95_candidates"] or 0), 2),
        })

    # Combined strategy: recall via OR conditions on gt; candidate estimate via union of keys
    for match_src, other_view, other_count in [("S2", "s2_norm", total_s2), ("S3", "s3_norm", total_s3)]:
        recalled = con.execute(f"""
            SELECT COUNT(*) FROM gt_pairs g
            JOIN s1_norm s1 ON g.s1_id = s1.entity_id
            JOIN {other_view} o ON g.matched_id = o.entity_id
            WHERE g.match_source = '{match_src}' AND (
                (s1.norm_name = o.norm_name AND s1.norm_name != '')
                OR (s1.norm_addr = o.norm_addr AND s1.norm_addr != '')
                OR (s1.country = o.country AND s1.name_prefix5 = o.name_prefix5 AND LENGTH(s1.name_prefix5) >= 3)
            )
        """).fetchone()[0]

        # Estimate combined candidates: max per S1 across individual key collision counts
        stats = con.execute(f"""
            WITH s1 AS (SELECT * FROM s1_norm),
            o AS (SELECT * FROM {other_view}),
            o_name AS (SELECT norm_name AS k, COUNT(*) c FROM o WHERE norm_name != '' GROUP BY norm_name),
            o_addr AS (SELECT norm_addr AS k, COUNT(*) c FROM o WHERE norm_addr != '' GROUP BY norm_addr),
            o_cp AS (SELECT country || '|' || name_prefix5 AS k, COUNT(*) c FROM o
                      WHERE LENGTH(name_prefix5) >= 3 GROUP BY country || '|' || name_prefix5),
            s1_cand AS (
                SELECT s1.entity_id,
                    GREATEST(
                        COALESCE((SELECT c FROM o_name WHERE k = s1.norm_name), 0),
                        COALESCE((SELECT c FROM o_addr WHERE k = s1.norm_addr), 0),
                        COALESCE((SELECT c FROM o_cp WHERE k = s1.country || '|' || s1.name_prefix5), 0)
                    ) AS n
                FROM s1
            )
            SELECT SUM(n) AS total_candidates, AVG(n) AS avg_c, MAX(n) AS max_c,
                   SUM(CASE WHEN n = 0 THEN 1 ELSE 0 END) AS s1_zero
            FROM s1_cand
        """).fetchdf().iloc[0]

        total_candidates = int(stats["total_candidates"] or 0)
        s1_zero = int(stats["s1_zero"] or 0)

        results.append({
            "strategy": f"combined (name OR addr OR country+prefix5) -> {match_src}",
            "total_candidate_pairs": total_candidates,
            "avg_candidates_per_s1": round(float(stats["avg_c"] or 0), 2),
            "s1_with_zero_candidates": s1_zero,
            "s1_zero_candidate_pct": round(100 * s1_zero / total_s1, 4),
            "true_matches_recalled": int(recalled),
            "total_true_matches": total_true,
            "true_match_recall": round(recalled / total_true, 6) if total_true else 0,
            "reduction_ratio": round(1 - total_candidates / (total_s1 * other_count), 6),
            "max_candidates_for_one_s1": int(stats["max_c"] or 0),
        })

    return {
        "strategies": results,
        "total_true_pairs": total_true,
        "total_possible_pairs": total_s1 * (total_s2 + total_s3),
        "_elapsed_sec": round(time.time() - t0, 2),
    }
