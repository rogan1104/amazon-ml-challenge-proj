"""Phases 5 & 6: Name and Address Noise Analysis."""
import duckdb
from collections import Counter
from analysis.normalize import detect_name_transformations, detect_address_transformations


def analyze_name_noise(con: duckdb.DuckDBPyConnection, max_pairs: int = 100_000) -> dict:
    """Measure actual name transformation patterns in true matches."""
    pairs_s2 = con.execute(f"""
        SELECT s1.business_name AS s1_name, s2.business_name AS s2_name
        FROM gt_pairs p
        JOIN train_S1 s1 ON p.s1_id = s1.entity_id
        JOIN train_S2 s2 ON p.matched_id = s2.entity_id
        WHERE p.match_source = 'S2'
        USING SAMPLE {max_pairs // 2} (reservoir)
    """).fetchdf()

    pairs_s3 = con.execute(f"""
        SELECT s1.business_name AS s1_name, s3.business_name AS s3_name
        FROM gt_pairs p
        JOIN train_S1 s1 ON p.s1_id = s1.entity_id
        JOIN train_S3 s3 ON p.matched_id = s3.entity_id
        WHERE p.match_source = 'S3'
        USING SAMPLE {max_pairs // 2} (reservoir)
    """).fetchdf()

    pattern_counter = Counter()
    examples = {}

    for df, col in [(pairs_s2, "s2_name"), (pairs_s3, "s3_name")]:
        for r in df.itertuples():
            patterns = detect_name_transformations(r.s1_name, getattr(r, col))
            for p in patterns:
                pattern_counter[p] += 1
                if p not in examples:
                    examples[p] = []
                if len(examples[p]) < 15:
                    examples[p].append({
                        "s1_name": r.s1_name,
                        "other_name": getattr(r, col),
                    })

    total = sum(pattern_counter.values())
    freq = {k: {"count": v, "pct": round(100 * v / total, 2)} for k, v in pattern_counter.most_common()}
    return {"total_pattern_hits": total, "pattern_frequencies": freq, "examples": examples}


def analyze_address_noise(con: duckdb.DuckDBPyConnection, max_pairs: int = 100_000) -> dict:
    """Measure actual address transformation patterns in true matches."""
    pairs_s2 = con.execute(f"""
        SELECT s1.business_address AS s1_addr, s2.business_address AS s2_addr
        FROM gt_pairs p
        JOIN train_S1 s1 ON p.s1_id = s1.entity_id
        JOIN train_S2 s2 ON p.matched_id = s2.entity_id
        WHERE p.match_source = 'S2'
        USING SAMPLE {max_pairs // 2} (reservoir)
    """).fetchdf()

    pairs_s3 = con.execute(f"""
        SELECT s1.business_address AS s1_addr, s3.business_address AS s3_addr
        FROM gt_pairs p
        JOIN train_S1 s1 ON p.s1_id = s1.entity_id
        JOIN train_S3 s3 ON p.matched_id = s3.entity_id
        WHERE p.match_source = 'S3'
        USING SAMPLE {max_pairs // 2} (reservoir)
    """).fetchdf()

    pattern_counter = Counter()
    examples = {}

    for df, col in [(pairs_s2, "s2_addr"), (pairs_s3, "s3_addr")]:
        for r in df.itertuples():
            patterns = detect_address_transformations(r.s1_addr, getattr(r, col))
            for p in patterns:
                pattern_counter[p] += 1
                if p not in examples:
                    examples[p] = []
                if len(examples[p]) < 15:
                    examples[p].append({
                        "s1_addr": r.s1_addr,
                        "other_addr": getattr(r, col),
                    })

    total = sum(pattern_counter.values())
    freq = {k: {"count": v, "pct": round(100 * v / total, 2)} for k, v in pattern_counter.most_common()}
    return {"total_pattern_hits": total, "pattern_frequencies": freq, "examples": examples}
