"""Phases 11-13: Hard Cases, Leakage, Validation Strategy."""
import duckdb
from rapidfuzz.distance import Levenshtein
from analysis.normalize import normalize_name, normalize_address, _safe_str


def find_hard_cases(con: duckdb.DuckDBPyConnection, top_n: int = 50) -> dict:
    """Find hardest true matches and dangerous false candidates."""
    pairs = con.execute("""
        WITH sample_pairs AS (
            SELECT s1_id, matched_id, match_source
            FROM gt_pairs USING SAMPLE 20000 (reservoir)
        )
        SELECT p.s1_id, p.matched_id, p.match_source,
               s1.business_name AS s1_name, s1.business_address AS s1_addr, s1.country AS s1_country,
               COALESCE(s2.business_name, s3.business_name) AS o_name,
               COALESCE(s2.business_address, s3.business_address) AS o_addr,
               COALESCE(s2.country, s3.country) AS o_country
        FROM sample_pairs p
        JOIN train_S1 s1 ON p.s1_id = s1.entity_id
        LEFT JOIN train_S2 s2 ON p.matched_id = s2.entity_id AND p.match_source = 'S2'
        LEFT JOIN train_S3 s3 ON p.matched_id = s3.entity_id AND p.match_source = 'S3'
    """).fetchdf()

    hard_positives = []
    for r in pairs.itertuples():
        nn1 = normalize_name(r.s1_name)
        nn2 = normalize_name(r.o_name)
        na1 = normalize_address(r.s1_addr)
        na2 = normalize_address(r.o_addr)
        name_sim = Levenshtein.normalized_similarity(nn1, nn2) if nn1 or nn2 else 0
        addr_sim = Levenshtein.normalized_similarity(na1, na2) if na1 or na2 else 0
        combined = (name_sim + addr_sim) / 2
        hard_positives.append({
            "s1_id": r.s1_id, "matched_id": r.matched_id, "source": r.match_source,
            "s1_name": _safe_str(r.s1_name), "o_name": _safe_str(r.o_name),
            "s1_addr": _safe_str(r.s1_addr), "o_addr": _safe_str(r.o_addr),
            "name_sim": round(name_sim, 4), "addr_sim": round(addr_sim, 4),
            "combined_sim": round(combined, 4),
            "difficulty_score": round(1 - combined, 4),
        })

    hard_positives.sort(key=lambda x: x["difficulty_score"], reverse=True)
    hardest = hard_positives[:top_n]

    # Dangerous false candidates via exact name match not in GT
    # Filter to non-matches via SQL anti-join on the sample
    false_only = con.execute("""
        WITH pairs AS (
            SELECT s1.entity_id AS s1_id, s2.entity_id AS s2_id,
                   s1.business_name AS s1_name, s2.business_name AS s2_name,
                   s1.business_address AS s1_addr, s2.business_address AS s2_addr
            FROM (SELECT * FROM train_S1 USING SAMPLE 2000 (reservoir)) s1
            JOIN (SELECT * FROM train_S2 USING SAMPLE 5000 (reservoir)) s2
              ON LOWER(TRIM(s1.business_name)) = LOWER(TRIM(s2.business_name))
        )
        SELECT p.* FROM pairs p
        LEFT JOIN gt_pairs g ON p.s1_id = g.s1_id AND p.s2_id = g.matched_id
        WHERE g.s1_id IS NULL
        LIMIT 2000
    """).fetchdf()

    hard_negatives = []
    for r in false_only.itertuples():
        nn1, nn2 = normalize_name(r.s1_name), normalize_name(r.s2_name)
        na1, na2 = normalize_address(r.s1_addr), normalize_address(r.s2_addr)
        name_sim = Levenshtein.normalized_similarity(nn1, nn2)
        addr_sim = Levenshtein.normalized_similarity(na1, na2)
        combined = (name_sim + addr_sim) / 2
        hard_negatives.append({
            "s1_id": r.s1_id, "s2_id": r.s2_id,
            "s1_name": _safe_str(r.s1_name), "s2_name": _safe_str(r.s2_name),
            "s1_addr": _safe_str(r.s1_addr), "s2_addr": _safe_str(r.s2_addr),
            "name_sim": round(name_sim, 4), "addr_sim": round(addr_sim, 4),
            "combined_sim": round(combined, 4),
            "danger_score": round(combined, 4),
        })

    hard_negatives.sort(key=lambda x: x["danger_score"], reverse=True)
    return {"hardest_positives": hardest, "most_dangerous_negatives": hard_negatives[:top_n]}


def investigate_leakage(con: duckdb.DuckDBPyConnection) -> dict:
    """Investigate potential leakage and artifacts."""
    leak = {}

    for label, view in [("S1_train", "train_S1"), ("S2_train", "train_S2"), ("S1_test", "test_S1")]:
        sample_ids = con.execute(f"SELECT entity_id FROM {view} LIMIT 10").fetchdf()["entity_id"].tolist()
        leak[f"{label}_id_samples"] = sample_ids

    overlap_s1 = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT entity_id FROM train_S1 INTERSECT SELECT entity_id FROM test_S1
        )
    """).fetchone()[0]
    overlap_s2 = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT entity_id FROM train_S2 INTERSECT SELECT entity_id FROM test_S2
        )
    """).fetchone()[0]
    overlap_s3 = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT entity_id FROM train_S3 INTERSECT SELECT entity_id FROM test_S3
        )
    """).fetchone()[0]
    leak["entity_id_overlap"] = {"S1": overlap_s1, "S2": overlap_s2, "S3": overlap_s3}

    leak["exact_record_overlap_train_test_S1"] = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT business_name, business_address, country FROM train_S1
            INTERSECT
            SELECT business_name, business_address, country FROM test_S1
        )
    """).fetchone()[0]

    leak["exact_record_overlap_train_test_S2"] = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT business_name, business_address, country FROM train_S2
            INTERSECT
            SELECT business_name, business_address, country FROM test_S2
        )
    """).fetchone()[0]

    leak["shared_business_names_train_test_S1"] = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT LOWER(TRIM(business_name)) FROM train_S1 WHERE business_name IS NOT NULL
            INTERSECT
            SELECT DISTINCT LOWER(TRIM(business_name)) FROM test_S1 WHERE business_name IS NOT NULL
        )
    """).fetchone()[0]

    # ID prefix pattern stats (avoid numeric cast failures)
    for label, view, prefix in [("S1_train", "train_S1", "S1-"), ("S2_train", "train_S2", "S2-"),
                                 ("S1_test", "test_S1", "S1-")]:
        stats = con.execute(f"""
            SELECT COUNT(DISTINCT entity_id) AS unique_ids, COUNT(*) AS total,
                   MIN(LENGTH(entity_id)) AS min_id_len, MAX(LENGTH(entity_id)) AS max_id_len,
                   SUM(CASE WHEN entity_id LIKE '{prefix}%' THEN 1 ELSE 0 END) AS correct_prefix
            FROM {view}
        """).fetchdf().iloc[0].to_dict()
        leak[f"{label}_id_stats"] = stats

    leak["identical_records"] = {
        "S1_S2": con.execute("""
            SELECT COUNT(*) FROM (
                SELECT business_name, business_address, country FROM train_S1
                INTERSECT SELECT business_name, business_address, country FROM train_S2
            )
        """).fetchone()[0],
        "S1_S3": con.execute("""
            SELECT COUNT(*) FROM (
                SELECT business_name, business_address, country FROM train_S1
                INTERSECT SELECT business_name, business_address, country FROM train_S3
            )
        """).fetchone()[0],
    }

    return leak


def recommend_validation(con: duckdb.DuckDBPyConnection) -> dict:
    """Recommend validation split strategy based on data structure."""
    rec = {}

    singleton_by_country = con.execute("""
        SELECT s1.country, COUNT(*) AS cnt,
               SUM(CASE WHEN g.match_count=0 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS singleton_pct
        FROM train_S1 s1 JOIN gt_summary g ON s1.entity_id = g.s1_id
        GROUP BY s1.country
    """).fetchdf().to_dict("records")
    rec["singleton_by_country"] = singleton_by_country

    rec["recommendations"] = [
        "OBSERVATION: Do NOT randomly split rows — split by S1 entity groups to prevent leakage.",
        "OBSERVATION: Singletons are significant; macro F0.5 includes them at full weight.",
        "OBSERVATION: France appears only in test — validation cannot measure France from train labels.",
        "HYPOTHESIS: Stratify validation by (country, match_count_bucket) for representative evaluation.",
        "HYPOTHESIS: Hold out entire S1 entities, not individual match pairs.",
        "RECOMMENDATION: 80/20 S1-level split stratified by country and match_count (0,1,2,3,4+).",
        "RECOMMENDATION: Report F_0.5 separately by country and singleton vs non-singleton.",
        "RECOMMENDATION: Build hard-case validation subset from lowest-similarity true matches.",
    ]

    rec["match_count_buckets"] = con.execute("""
        SELECT
            CASE WHEN match_count = 0 THEN '0_singleton'
                 WHEN match_count = 1 THEN '1_match'
                 WHEN match_count = 2 THEN '2_matches'
                 WHEN match_count = 3 THEN '3_matches'
                 ELSE '4+_matches' END AS bucket,
            COUNT(*) AS s1_count
        FROM gt_summary GROUP BY bucket ORDER BY bucket
    """).fetchdf().to_dict("records")

    return rec
