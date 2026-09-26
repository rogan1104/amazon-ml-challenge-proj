"""Phase 1: Basic Data Profiling per source."""
import duckdb


def profile_source(con: duckdb.DuckDBPyConnection, view_name: str, label: str) -> dict:
    """Compute comprehensive profiling stats for one source view."""
    stats = con.execute(f"""
        WITH base AS (
            SELECT
                entity_id,
                business_name,
                business_address,
                country,
                LENGTH(business_name) AS name_len,
                LENGTH(business_address) AS addr_len,
                business_name IS NULL OR TRIM(business_name) = '' AS name_missing,
                business_address IS NULL OR TRIM(business_address) = '' AS addr_missing,
                country IS NULL OR TRIM(country) = '' AS country_missing
            FROM {view_name}
        ),
        dup_ids AS (
            SELECT entity_id, COUNT(*) AS cnt
            FROM base GROUP BY entity_id HAVING COUNT(*) > 1
        ),
        dup_rows AS (
            SELECT business_name, business_address, country, COUNT(*) AS cnt
            FROM base
            GROUP BY business_name, business_address, country
            HAVING COUNT(*) > 1
        ),
        dup_names AS (
            SELECT business_name, COUNT(*) AS cnt
            FROM base WHERE NOT name_missing
            GROUP BY business_name HAVING COUNT(*) > 1
        ),
        dup_addrs AS (
            SELECT business_address, COUNT(*) AS cnt
            FROM base WHERE NOT addr_missing
            GROUP BY business_address HAVING COUNT(*) > 1
        ),
        country_dist AS (
            SELECT country, COUNT(*) AS cnt
            FROM base GROUP BY country ORDER BY cnt DESC
        )
        SELECT
            (SELECT COUNT(*) FROM base) AS total_rows,
            (SELECT COUNT(DISTINCT entity_id) FROM base) AS unique_entity_ids,
            (SELECT COALESCE(SUM(cnt - 1), 0) FROM dup_ids) AS duplicate_entity_id_count,
            (SELECT COUNT(*) FROM base WHERE entity_id IS NULL OR TRIM(entity_id) = '') AS missing_entity_id,
            (SELECT COUNT(*) FROM base WHERE name_missing) AS missing_business_name,
            (SELECT COUNT(*) FROM base WHERE name_missing) AS empty_business_name,
            (SELECT COUNT(*) FROM base WHERE addr_missing) AS missing_business_address,
            (SELECT COUNT(*) FROM base WHERE addr_missing) AS empty_business_address,
            (SELECT COUNT(*) FROM base WHERE country_missing) AS missing_country,
            (SELECT COUNT(DISTINCT country) FROM base WHERE NOT country_missing) AS unique_countries,
            (SELECT COUNT(DISTINCT business_name) FROM base WHERE NOT name_missing) AS unique_names,
            (SELECT COUNT(DISTINCT business_address) FROM base WHERE NOT addr_missing) AS unique_addresses,
            (SELECT COALESCE(SUM(cnt), 0) FROM dup_rows) AS exact_dup_row_count,
            (SELECT COALESCE(SUM(cnt), 0) FROM dup_names) AS dup_name_row_count,
            (SELECT COALESCE(SUM(cnt), 0) FROM dup_addrs) AS dup_addr_row_count,
            (SELECT MIN(name_len) FROM base WHERE NOT name_missing) AS name_len_min,
            (SELECT MAX(name_len) FROM base WHERE NOT name_missing) AS name_len_max,
            (SELECT AVG(name_len) FROM base WHERE NOT name_missing) AS name_len_mean,
            (SELECT MEDIAN(name_len) FROM base WHERE NOT name_missing) AS name_len_median,
            (SELECT QUANTILE_CONT(name_len, 0.25) FROM base WHERE NOT name_missing) AS name_len_p25,
            (SELECT QUANTILE_CONT(name_len, 0.75) FROM base WHERE NOT name_missing) AS name_len_p75,
            (SELECT MIN(addr_len) FROM base WHERE NOT addr_missing) AS addr_len_min,
            (SELECT MAX(addr_len) FROM base WHERE NOT addr_missing) AS addr_len_max,
            (SELECT AVG(addr_len) FROM base WHERE NOT addr_missing) AS addr_len_mean,
            (SELECT MEDIAN(addr_len) FROM base WHERE NOT addr_missing) AS addr_len_median,
            (SELECT QUANTILE_CONT(addr_len, 0.25) FROM base WHERE NOT addr_missing) AS addr_len_p25,
            (SELECT QUANTILE_CONT(addr_len, 0.75) FROM base WHERE NOT addr_missing) AS addr_len_p75
    """).fetchdf().iloc[0].to_dict()

    total = stats["total_rows"]
    stats["label"] = label
    stats["duplicate_entity_id_pct"] = round(100 * stats["duplicate_entity_id_count"] / total, 4) if total else 0
    stats["exact_dup_row_pct"] = round(100 * stats["exact_dup_row_count"] / total, 4) if total else 0
    stats["dup_name_pct"] = round(100 * stats["dup_name_row_count"] / total, 4) if total else 0
    stats["dup_addr_pct"] = round(100 * stats["dup_addr_row_count"] / total, 4) if total else 0
    stats["missing_entity_id_pct"] = round(100 * stats["missing_entity_id"] / total, 4) if total else 0
    stats["missing_business_name_pct"] = round(100 * stats["missing_business_name"] / total, 4) if total else 0
    stats["missing_business_address_pct"] = round(100 * stats["missing_business_address"] / total, 4) if total else 0
    stats["missing_country_pct"] = round(100 * stats["missing_country"] / total, 4) if total else 0

    country_df = con.execute(f"""
        SELECT country, COUNT(*) AS cnt
        FROM {view_name}
        WHERE country IS NOT NULL AND TRIM(country) != ''
        GROUP BY country ORDER BY cnt DESC
    """).fetchdf()
    stats["country_distribution"] = country_df.to_dict("records")

    return stats


def run_all_profiling(con: duckdb.DuckDBPyConnection) -> dict:
    views = [
        ("train_S1", "S1 train"),
        ("train_S2", "S2 train"),
        ("train_S3", "S3 train"),
        ("test_S1", "S1 test"),
        ("test_S2", "S2 test"),
        ("test_S3", "S3 test"),
    ]
    return {label: profile_source(con, view, label) for view, label in views}
