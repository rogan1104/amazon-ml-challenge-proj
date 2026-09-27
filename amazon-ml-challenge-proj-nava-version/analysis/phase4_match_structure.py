"""Phase 4: Match Structure Analysis."""
import duckdb
import json
import time
from pathlib import Path
from collections import defaultdict
from rapidfuzz.distance import Levenshtein
from analysis.config import CACHE
from analysis.normalize import (
    normalize_name, normalize_address, extract_tokens, extract_digits,
    extract_postal, jaccard, char_ngrams, basic_normalize
)


def _str(v):
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return str(v) if not isinstance(v, str) else v


def _pair_features(s1_name, s1_addr, s1_country, s2_name, s2_addr, s2_country):
    s1_name, s1_addr, s1_country = _str(s1_name), _str(s1_addr), _str(s1_country)
    s2_name, s2_addr, s2_country = _str(s2_name), _str(s2_addr), _str(s2_country)
    nn1, nn2 = normalize_name(s1_name), normalize_name(s2_name)
    na1, na2 = normalize_address(s1_addr), normalize_address(s2_addr)
    t1, t2 = extract_tokens(s1_name), extract_tokens(s2_name)
    ta1, ta2 = extract_tokens(s1_addr), extract_tokens(s2_addr)
    d1, d2 = extract_digits(s1_addr), extract_digits(s2_addr)
    p1, p2 = extract_postal(s1_addr), extract_postal(s2_addr)
    ng1, ng2 = char_ngrams(s1_name), char_ngrams(s2_name)
    nag1, nag2 = char_ngrams(s1_addr), char_ngrams(s2_addr)

    return {
        "exact_name": s1_name == s2_name,
        "exact_address": s1_addr == s2_addr,
        "exact_country": s1_country == s2_country,
        "norm_name_eq": nn1 == nn2,
        "norm_addr_eq": na1 == na2,
        "name_sim": Levenshtein.normalized_similarity(nn1, nn2) if nn1 or nn2 else 0,
        "addr_sim": Levenshtein.normalized_similarity(na1, na2) if na1 or na2 else 0,
        "name_token_jaccard": jaccard(t1, t2),
        "addr_token_jaccard": jaccard(ta1, ta2),
        "name_trigram_jaccard": jaccard(ng1, ng2),
        "addr_trigram_jaccard": jaccard(nag1, nag2),
        "digit_jaccard": jaccard(d1, d2),
        "postal_overlap": len(p1 & p2) > 0 if p1 and p2 else False,
    }


def analyze_match_structure(con: duckdb.DuckDBPyConnection, sample_negatives: int = 50000) -> dict:
    """Analyze true matches vs negatives. Uses sampling for negatives."""
    t0 = time.time()
    cache_path = CACHE / "match_pairs_sample.parquet"
    results = {}

    # Build true match pairs view (S1-S2 and S1-S3)
    con.execute("""
        CREATE OR REPLACE VIEW true_pairs_s2 AS
        SELECT p.s1_id, p.matched_id AS s2_id,
               s1.business_name AS s1_name, s1.business_address AS s1_addr, s1.country AS s1_country,
               s2.business_name AS s2_name, s2.business_address AS s2_addr, s2.country AS s2_country
        FROM gt_pairs p
        JOIN train_S1 s1 ON p.s1_id = s1.entity_id
        JOIN train_S2 s2 ON p.matched_id = s2.entity_id
        WHERE p.match_source = 'S2'
    """)
    con.execute("""
        CREATE OR REPLACE VIEW true_pairs_s3 AS
        SELECT p.s1_id, p.matched_id AS s3_id,
               s1.business_name AS s1_name, s1.business_address AS s1_addr, s1.country AS s1_country,
               s3.business_name AS s3_name, s3.business_address AS s3_addr, s3.country AS s3_country
        FROM gt_pairs p
        JOIN train_S1 s1 ON p.s1_id = s1.entity_id
        JOIN train_S3 s3 ON p.matched_id = s3.entity_id
        WHERE p.match_source = 'S3'
    """)

    true_s2_count = con.execute("SELECT COUNT(*) FROM true_pairs_s2").fetchone()[0]
    true_s3_count = con.execute("SELECT COUNT(*) FROM true_pairs_s3").fetchone()[0]
    results["true_pair_counts"] = {"S1_S2": true_s2_count, "S1_S3": true_s3_count}

    # Sample true pairs for detailed feature computation (max 200k each for speed)
    max_sample = 200_000
    for label, view, name_col, addr_col, country_col in [
        ("S1_S2", "true_pairs_s2", "s2_name", "s2_addr", "s2_country"),
        ("S1_S3", "true_pairs_s3", "s3_name", "s3_addr", "s3_country"),
    ]:
        df = con.execute(f"""
            SELECT s1_id, s1_name, s1_addr, s1_country,
                   {name_col} AS o_name, {addr_col} AS o_addr, {country_col} AS o_country
            FROM {view}
            USING SAMPLE {min(max_sample, true_s2_count if label=='S1_S2' else true_s3_count)}
            (reservoir)
        """).fetchdf()

        features = [_pair_features(r.s1_name, r.s1_addr, r.s1_country, r.o_name, r.o_addr, r.o_country)
                    for r in df.itertuples()]
        results[f"true_{label}"] = _summarize_features(features, label)

    # Random negatives: random S1 x random S2/S3 (not in GT)
    random_neg_s2 = con.execute(f"""
        WITH random_s1 AS (
            SELECT entity_id, business_name, business_address, country
            FROM train_S1 USING SAMPLE {sample_negatives // 2} (reservoir)
        ),
        random_s2 AS (
            SELECT entity_id, business_name, business_address, country
            FROM train_S2 USING SAMPLE {sample_negatives // 2} (reservoir)
        ),
        pairs AS (
            SELECT r1.entity_id AS s1_id, r2.entity_id AS s2_id,
                   r1.business_name AS s1_name, r1.business_address AS s1_addr, r1.country AS s1_country,
                   r2.business_name AS s2_name, r2.business_address AS s2_addr, r2.country AS s2_country
            FROM random_s1 r1 CROSS JOIN random_s2 r2
            LIMIT {sample_negatives}
        )
        SELECT * FROM pairs
    """).fetchdf()

    neg_features = [_pair_features(r.s1_name, r.s1_addr, r.s1_country, r.s2_name, r.s2_addr, r.s2_country)
                    for r in random_neg_s2.itertuples()]
    results["random_negatives_S1_S2"] = _summarize_features(neg_features, "random_neg")

    # Hard negatives
    hard_negs = _build_hard_negatives(con, limit=30000)
    results["hard_negatives"] = hard_negs
    results["_elapsed_sec"] = round(time.time() - t0, 2)
    return results


def _summarize_features(features: list, label: str) -> dict:
    if not features:
        return {"label": label, "count": 0}

    n = len(features)
    bool_keys = ["exact_name", "exact_address", "exact_country", "norm_name_eq", "norm_addr_eq", "postal_overlap"]
    float_keys = ["name_sim", "addr_sim", "name_token_jaccard", "addr_token_jaccard",
                  "name_trigram_jaccard", "addr_trigram_jaccard", "digit_jaccard"]

    summary = {"label": label, "count": n}
    for k in bool_keys:
        summary[f"{k}_pct"] = round(100 * sum(f[k] for f in features) / n, 2)
    for k in float_keys:
        vals = sorted(f[k] for f in features)
        summary[f"{k}_mean"] = round(sum(vals) / n, 4)
        summary[f"{k}_median"] = round(vals[n // 2], 4)
        summary[f"{k}_p25"] = round(vals[n // 4], 4)
        summary[f"{k}_p75"] = round(vals[3 * n // 4], 4)

    return summary


def _build_hard_negatives(con, limit=30000) -> dict:
    """Build hard negative categories."""
    categories = {}

    # Same normalized name, different address (not matches)
    same_name = con.execute(f"""
        WITH name_groups AS (
            SELECT LOWER(TRIM(business_name)) AS norm_name, entity_id, business_name, business_address, country,
                   CASE WHEN entity_id LIKE 'S1-%' THEN 'S1'
                        WHEN entity_id LIKE 'S2-%' THEN 'S2' ELSE 'S3' END AS src
            FROM (
                SELECT entity_id, business_name, business_address, country FROM train_S1
                UNION ALL SELECT entity_id, business_name, business_address, country FROM train_S2
                UNION ALL SELECT entity_id, business_name, business_address, country FROM train_S3
            )
            WHERE business_name IS NOT NULL AND TRIM(business_name) != ''
        ),
        multi AS (
            SELECT norm_name FROM name_groups GROUP BY norm_name HAVING COUNT(DISTINCT entity_id) >= 2 LIMIT 5000
        )
        SELECT ng.norm_name, ng.entity_id, ng.business_name, ng.business_address, ng.country, ng.src
        FROM name_groups ng JOIN multi m ON ng.norm_name = m.norm_name
        LIMIT 10000
    """).fetchdf()

    # Sample pairs from same name groups that are NOT in GT
    gt_pairs_set = set(
        (r[0], r[1]) for r in con.execute("SELECT s1_id, matched_id FROM gt_pairs").fetchall()
    )
    same_name_neg = []
    by_name = defaultdict(list)
    for r in same_name.itertuples():
        by_name[r.norm_name].append(r)
    for norm_name, records in by_name.items():
        for i, r1 in enumerate(records):
            for r2 in records[i + 1:]:
                if r1.src == "S1" and r2.src in ("S2", "S3"):
                    if (r1.entity_id, r2.entity_id) not in gt_pairs_set:
                        feat = _pair_features(r1.business_name, r1.business_address, r1.country,
                                              r2.business_name, r2.business_address, r2.country)
                        same_name_neg.append(feat)
                if len(same_name_neg) >= limit // 3:
                    break
            if len(same_name_neg) >= limit // 3:
                break
        if len(same_name_neg) >= limit // 3:
            break
    categories["same_name_diff_addr"] = _summarize_features(same_name_neg, "same_name_diff_addr")

    # Same country + high name similarity but not match (sample via SQL)
    sim_neg = con.execute(f"""
        WITH s1_sample AS (
            SELECT entity_id, business_name, business_address, country
            FROM train_S1 USING SAMPLE 5000 (reservoir)
        ),
        s2_sample AS (
            SELECT entity_id, business_name, business_address, country
            FROM train_S2 USING SAMPLE 20000 (reservoir)
        )
        SELECT s1.entity_id AS s1_id, s2.entity_id AS s2_id,
               s1.business_name AS s1_name, s1.business_address AS s1_addr, s1.country AS s1_country,
               s2.business_name AS s2_name, s2.business_address AS s2_addr, s2.country AS s2_country
        FROM s1_sample s1
        JOIN s2_sample s2 ON s1.country = s2.country
        WHERE LOWER(s1.business_name) = LOWER(s2.business_name)
          AND s1.business_address != s2.business_address
        LIMIT 5000
    """).fetchdf()

    sim_neg_filtered = []
    for r in sim_neg.itertuples():
        if (r.s1_id, r.s2_id) not in gt_pairs_set:
            sim_neg_filtered.append(_pair_features(r.s1_name, r.s1_addr, r.s1_country,
                                                    r.s2_name, r.s2_addr, r.s2_country))
    categories["same_country_same_name"] = _summarize_features(sim_neg_filtered, "same_country_same_name")

    # Same address different name
    same_addr = con.execute("""
        WITH s1_sample AS (
            SELECT entity_id, business_name, business_address, country
            FROM train_S1 USING SAMPLE 5000 (reservoir)
        ),
        s2_sample AS (
            SELECT entity_id, business_name, business_address, country
            FROM train_S2 USING SAMPLE 20000 (reservoir)
        )
        SELECT s1.entity_id AS s1_id, s2.entity_id AS s2_id,
               s1.business_name AS s1_name, s1.business_address AS s1_addr, s1.country AS s1_country,
               s2.business_name AS s2_name, s2.business_address AS s2_addr, s2.country AS s2_country
        FROM s1_sample s1
        JOIN s2_sample s2 ON LOWER(TRIM(s1.business_address)) = LOWER(TRIM(s2.business_address))
        WHERE LOWER(s1.business_name) != LOWER(s2.business_name)
        LIMIT 5000
    """).fetchdf()

    same_addr_filtered = []
    for r in same_addr.itertuples():
        if (r.s1_id, r.s2_id) not in gt_pairs_set:
            same_addr_filtered.append(_pair_features(r.s1_name, r.s1_addr, r.s1_country,
                                                      r.s2_name, r.s2_addr, r.s2_country))
    categories["same_addr_diff_name"] = _summarize_features(same_addr_filtered, "same_addr_diff_name")

    return categories
