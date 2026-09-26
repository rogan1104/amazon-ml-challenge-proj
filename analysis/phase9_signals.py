"""Phase 9: Information Content / Signal Analysis."""
import duckdb
from analysis.normalize import normalize_name, normalize_address, extract_postal


def analyze_signals(con: duckdb.DuckDBPyConnection) -> dict:
    """Estimate discriminative power of fields using training data."""
    signals = []

    # Build all possible S1-S2 candidate pairs features at aggregate level using SQL
    # P(match | exact normalized name) - using name-level join
    norm_name_match = con.execute("""
        WITH s1_names AS (
            SELECT entity_id, LOWER(TRIM(REGEXP_REPLACE(business_name, '[^\\w\\s]', ' ', 'g'))) AS norm_name
            FROM train_S1 WHERE business_name IS NOT NULL AND TRIM(business_name) != ''
        ),
        s2_names AS (
            SELECT entity_id, LOWER(TRIM(REGEXP_REPLACE(business_name, '[^\\w\\s]', ' ', 'g'))) AS norm_name
            FROM train_S2 WHERE business_name IS NOT NULL AND TRIM(business_name) != ''
        ),
        name_pairs AS (
            SELECT s1.entity_id AS s1_id, s2.entity_id AS s2_id, s1.norm_name
            FROM s1_names s1
            JOIN s2_names s2 ON s1.norm_name = s2.norm_name AND s1.norm_name != ''
        ),
        gt_set AS (
            SELECT s1_id, matched_id FROM gt_pairs WHERE match_source = 'S2'
        )
        SELECT
            COUNT(*) AS total_same_norm_name_pairs,
            SUM(CASE WHEN g.s1_id IS NOT NULL THEN 1 ELSE 0 END) AS true_matches,
            SUM(CASE WHEN g.s1_id IS NULL THEN 1 ELSE 0 END) AS false_pairs
        FROM name_pairs np
        LEFT JOIN gt_set g ON np.s1_id = g.s1_id AND np.s2_id = g.matched_id
    """).fetchdf().iloc[0]
    total_n = norm_name_match["total_same_norm_name_pairs"]
    if total_n > 0:
        signals.append({
            "signal": "exact_normalized_name (S1-S2)",
            "condition_pairs": int(total_n),
            "true_matches": int(norm_name_match["true_matches"]),
            "false_pairs": int(norm_name_match["false_pairs"]),
            "p_match_given_condition": round(norm_name_match["true_matches"] / total_n, 6),
        })

    # P(match | exact normalized address) S1-S2
    norm_addr_match = con.execute("""
        WITH s1_addr AS (
            SELECT entity_id, LOWER(TRIM(REGEXP_REPLACE(business_address, '[^\\w\\s]', ' ', 'g'))) AS norm_addr
            FROM train_S1 WHERE business_address IS NOT NULL AND TRIM(business_address) != ''
        ),
        s2_addr AS (
            SELECT entity_id, LOWER(TRIM(REGEXP_REPLACE(business_address, '[^\\w\\s]', ' ', 'g'))) AS norm_addr
            FROM train_S2 WHERE business_address IS NOT NULL AND TRIM(business_address) != ''
        ),
        addr_pairs AS (
            SELECT s1.entity_id AS s1_id, s2.entity_id AS s2_id
            FROM s1_addr s1 JOIN s2_addr s2 ON s1.norm_addr = s2.norm_addr AND s1.norm_addr != ''
        ),
        gt_set AS (SELECT s1_id, matched_id FROM gt_pairs WHERE match_source = 'S2')
        SELECT COUNT(*) AS total, SUM(CASE WHEN g.s1_id IS NOT NULL THEN 1 ELSE 0 END) AS true_m
        FROM addr_pairs np LEFT JOIN gt_set g ON np.s1_id = g.s1_id AND np.s2_id = g.matched_id
    """).fetchdf().iloc[0]
    if norm_addr_match["total"] > 0:
        signals.append({
            "signal": "exact_normalized_address (S1-S2)",
            "condition_pairs": int(norm_addr_match["total"]),
            "true_matches": int(norm_addr_match["true_m"]),
            "false_pairs": int(norm_addr_match["total"] - norm_addr_match["true_m"]),
            "p_match_given_condition": round(norm_addr_match["true_m"] / norm_addr_match["total"], 6),
        })

    # P(match | same country + exact name)
    country_name = con.execute("""
        WITH s1 AS (
            SELECT entity_id, country, LOWER(TRIM(business_name)) AS norm_name
            FROM train_S1 WHERE business_name IS NOT NULL AND country IS NOT NULL
        ),
        s2 AS (
            SELECT entity_id, country, LOWER(TRIM(business_name)) AS norm_name
            FROM train_S2 WHERE business_name IS NOT NULL AND country IS NOT NULL
        ),
        pairs AS (
            SELECT s1.entity_id AS s1_id, s2.entity_id AS s2_id
            FROM s1 JOIN s2 ON s1.country = s2.country AND s1.norm_name = s2.norm_name AND s1.norm_name != ''
        ),
        gt_set AS (SELECT s1_id, matched_id FROM gt_pairs WHERE match_source = 'S2')
        SELECT COUNT(*) AS total, SUM(CASE WHEN g.s1_id IS NOT NULL THEN 1 ELSE 0 END) AS true_m
        FROM pairs p LEFT JOIN gt_set g ON p.s1_id = g.s1_id AND p.s2_id = g.matched_id
    """).fetchdf().iloc[0]
    if country_name["total"] > 0:
        signals.append({
            "signal": "same_country + exact_name (S1-S2)",
            "condition_pairs": int(country_name["total"]),
            "true_matches": int(country_name["true_m"]),
            "false_pairs": int(country_name["total"] - country_name["true_m"]),
            "p_match_given_condition": round(country_name["true_m"] / country_name["total"], 6),
        })

    # P(match | same country) - estimated via per-country counts (avoid O(n^2) cross join)
    country_baseline = con.execute("""
        WITH s1c AS (SELECT country, COUNT(*) AS n FROM train_S1 WHERE country IS NOT NULL GROUP BY country),
             s2c AS (SELECT country, COUNT(*) AS n FROM train_S2 WHERE country IS NOT NULL GROUP BY country),
             pairs AS (
                 SELECT s1c.country, s1c.n * s2c.n AS possible_pairs
                 FROM s1c JOIN s2c ON s1c.country = s2c.country
             ),
             true_m AS (SELECT COUNT(*) AS n FROM gt_pairs WHERE match_source = 'S2')
        SELECT SUM(possible_pairs) AS total_possible,
               (SELECT n FROM true_m) AS true_matches
        FROM pairs
    """).fetchdf().iloc[0]
    if country_baseline["total_possible"] > 0:
        signals.append({
            "signal": "same_country (S1-S2) baseline (estimated)",
            "condition_pairs": int(country_baseline["total_possible"]),
            "true_matches": int(country_baseline["true_matches"]),
            "false_pairs": int(country_baseline["total_possible"] - country_baseline["true_matches"]),
            "p_match_given_condition": round(country_baseline["true_matches"] / country_baseline["total_possible"], 10),
        })

    # Exact business_name match
    exact_name = con.execute("""
        WITH pairs AS (
            SELECT s1.entity_id AS s1_id, s2.entity_id AS s2_id
            FROM train_S1 s1 JOIN train_S2 s2 ON s1.business_name = s2.business_name
            WHERE s1.business_name IS NOT NULL AND TRIM(s1.business_name) != ''
        ),
        gt_set AS (SELECT s1_id, matched_id FROM gt_pairs WHERE match_source = 'S2')
        SELECT COUNT(*) AS total, SUM(CASE WHEN g.s1_id IS NOT NULL THEN 1 ELSE 0 END) AS true_m
        FROM pairs p LEFT JOIN gt_set g ON p.s1_id = g.s1_id AND p.s2_id = g.matched_id
    """).fetchdf().iloc[0]
    if exact_name["total"] > 0:
        signals.append({
            "signal": "exact_raw_name (S1-S2)",
            "condition_pairs": int(exact_name["total"]),
            "true_matches": int(exact_name["true_m"]),
            "false_pairs": int(exact_name["total"] - exact_name["true_m"]),
            "p_match_given_condition": round(exact_name["true_m"] / exact_name["total"], 6),
        })

    # Exact business_address match
    exact_addr = con.execute("""
        WITH pairs AS (
            SELECT s1.entity_id AS s1_id, s2.entity_id AS s2_id
            FROM train_S1 s1 JOIN train_S2 s2 ON s1.business_address = s2.business_address
            WHERE s1.business_address IS NOT NULL AND TRIM(s1.business_address) != ''
        ),
        gt_set AS (SELECT s1_id, matched_id FROM gt_pairs WHERE match_source = 'S2')
        SELECT COUNT(*) AS total, SUM(CASE WHEN g.s1_id IS NOT NULL THEN 1 ELSE 0 END) AS true_m
        FROM pairs p LEFT JOIN gt_set g ON p.s1_id = g.s1_id AND p.s2_id = g.matched_id
    """).fetchdf().iloc[0]
    if exact_addr["total"] > 0:
        signals.append({
            "signal": "exact_raw_address (S1-S2)",
            "condition_pairs": int(exact_addr["total"]),
            "true_matches": int(exact_addr["true_m"]),
            "false_pairs": int(exact_addr["total"] - exact_addr["true_m"]),
            "p_match_given_condition": round(exact_addr["true_m"] / exact_addr["total"], 6),
        })

    # Same for S1-S3 (key signals only)
    for signal_name, join_cond in [
        ("exact_normalized_name (S1-S3)", "s1.norm_name = s3.norm_name"),
        ("exact_normalized_address (S1-S3)", "s1.norm_addr = s3.norm_addr"),
        ("same_country + exact_name (S1-S3)", "s1.country = s3.country AND s1.norm_name = s3.norm_name"),
    ]:
        if "norm_name" in join_cond:
            q = f"""
                WITH s1 AS (SELECT entity_id, country,
                    LOWER(TRIM(REGEXP_REPLACE(business_name, '[^\\w\\s]', ' ', 'g'))) AS norm_name,
                    '' AS norm_addr FROM train_S1 WHERE business_name IS NOT NULL),
                s3 AS (SELECT entity_id, country,
                    LOWER(TRIM(REGEXP_REPLACE(business_name, '[^\\w\\s]', ' ', 'g'))) AS norm_name,
                    '' AS norm_addr FROM train_S3 WHERE business_name IS NOT NULL),
                pairs AS (SELECT s1.entity_id AS s1_id, s3.entity_id AS s3_id FROM s1 JOIN s3 ON {join_cond} AND s1.norm_name != ''),
                gt_set AS (SELECT s1_id, matched_id FROM gt_pairs WHERE match_source = 'S3')
                SELECT COUNT(*) AS total, SUM(CASE WHEN g.s1_id IS NOT NULL THEN 1 ELSE 0 END) AS true_m
                FROM pairs p LEFT JOIN gt_set g ON p.s1_id = g.s1_id AND p.s3_id = g.matched_id
            """
        elif "norm_addr" in join_cond:
            q = f"""
                WITH s1 AS (SELECT entity_id,
                    LOWER(TRIM(REGEXP_REPLACE(business_address, '[^\\w\\s]', ' ', 'g'))) AS norm_addr FROM train_S1 WHERE business_address IS NOT NULL),
                s3 AS (SELECT entity_id,
                    LOWER(TRIM(REGEXP_REPLACE(business_address, '[^\\w\\s]', ' ', 'g'))) AS norm_addr FROM train_S3 WHERE business_address IS NOT NULL),
                pairs AS (SELECT s1.entity_id AS s1_id, s3.entity_id AS s3_id FROM s1 JOIN s3 ON {join_cond} AND s1.norm_addr != ''),
                gt_set AS (SELECT s1_id, matched_id FROM gt_pairs WHERE match_source = 'S3')
                SELECT COUNT(*) AS total, SUM(CASE WHEN g.s1_id IS NOT NULL THEN 1 ELSE 0 END) AS true_m
                FROM pairs p LEFT JOIN gt_set g ON p.s1_id = g.s1_id AND p.s3_id = g.matched_id
            """
        else:
            q = f"""
                WITH s1 AS (SELECT entity_id, country, LOWER(TRIM(business_name)) AS norm_name FROM train_S1 WHERE business_name IS NOT NULL AND country IS NOT NULL),
                s3 AS (SELECT entity_id, country, LOWER(TRIM(business_name)) AS norm_name FROM train_S3 WHERE business_name IS NOT NULL AND country IS NOT NULL),
                pairs AS (SELECT s1.entity_id AS s1_id, s3.entity_id AS s3_id FROM s1 JOIN s3 ON {join_cond} AND s1.norm_name != ''),
                gt_set AS (SELECT s1_id, matched_id FROM gt_pairs WHERE match_source = 'S3')
                SELECT COUNT(*) AS total, SUM(CASE WHEN g.s1_id IS NOT NULL THEN 1 ELSE 0 END) AS true_m
                FROM pairs p LEFT JOIN gt_set g ON p.s1_id = g.s1_id AND p.s3_id = g.matched_id
            """
        row = con.execute(q).fetchdf().iloc[0]
        if row["total"] > 0:
            signals.append({
                "signal": signal_name,
                "condition_pairs": int(row["total"]),
                "true_matches": int(row["true_m"]),
                "false_pairs": int(row["total"] - row["true_m"]),
                "p_match_given_condition": round(row["true_m"] / row["total"], 6),
            })

    return {"signals": signals}
