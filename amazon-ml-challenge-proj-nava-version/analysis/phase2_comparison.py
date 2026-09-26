"""Phase 2: Source Comparison."""
import duckdb


def compare_sources(con: duckdb.DuckDBPyConnection) -> dict:
    """Compare S1/S2/S3 characteristics across train and test."""
    comparison = {}

    for split in ["train", "test"]:
        rows = []
        for src in ["S1", "S2", "S3"]:
            view = f"{split}_{src}"
            row = con.execute(f"""
                SELECT
                    '{split}' AS split,
                    '{src}' AS source,
                    COUNT(*) AS total_rows,
                    AVG(LENGTH(business_name)) AS avg_name_len,
                    MEDIAN(LENGTH(business_name)) AS med_name_len,
                    AVG(LENGTH(business_address)) AS avg_addr_len,
                    MEDIAN(LENGTH(business_address)) AS med_addr_len,
                    SUM(CASE WHEN business_name IS NULL OR TRIM(business_name)='' THEN 1 ELSE 0 END)*100.0/COUNT(*) AS name_missing_pct,
                    SUM(CASE WHEN business_address IS NULL OR TRIM(business_address)='' THEN 1 ELSE 0 END)*100.0/COUNT(*) AS addr_missing_pct,
                    SUM(CASE WHEN country IS NULL OR TRIM(country)='' THEN 1 ELSE 0 END)*100.0/COUNT(*) AS country_missing_pct,
                    COUNT(DISTINCT business_name)*100.0/COUNT(*) AS name_uniqueness_pct,
                    COUNT(DISTINCT business_address)*100.0/COUNT(*) AS addr_uniqueness_pct,
                    SUM(CASE WHEN regexp_matches(business_name, '[[:punct:]]') THEN 1 ELSE 0 END)*100.0/COUNT(*) AS name_punct_pct,
                    SUM(CASE WHEN regexp_matches(business_address, '[[:punct:]]') THEN 1 ELSE 0 END)*100.0/COUNT(*) AS addr_punct_pct,
                    SUM(CASE WHEN regexp_matches(business_name, '[0-9]') THEN 1 ELSE 0 END)*100.0/COUNT(*) AS name_numeric_pct,
                    SUM(CASE WHEN regexp_matches(business_address, '[0-9]') THEN 1 ELSE 0 END)*100.0/COUNT(*) AS addr_numeric_pct,
                    SUM(CASE WHEN business_name = UPPER(business_name) AND LENGTH(business_name)>2 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS name_all_upper_pct,
                    SUM(CASE WHEN regexp_matches(business_name, '(?i)(corp|inc|ltd|llc|pvt|co\\.)') THEN 1 ELSE 0 END)*100.0/COUNT(*) AS legal_suffix_pct,
                    SUM(CASE WHEN regexp_matches(business_name, '(?i)(rd|st|ave|blvd|dr|ln)') THEN 1 ELSE 0 END)*100.0/COUNT(*) AS name_addr_abbrev_pct,
                    SUM(CASE WHEN regexp_matches(business_address, '(?i)(road|street|avenue|boulevard|drive|lane|apt|suite)') THEN 1 ELSE 0 END)*100.0/COUNT(*) AS addr_full_word_pct,
                    SUM(CASE WHEN regexp_matches(business_address, '(?i)(^|[[:space:]])(rd|st|ave|blvd|dr|ln|apt|ste)([[:space:]]|$|\\.)') THEN 1 ELSE 0 END)*100.0/COUNT(*) AS addr_abbrev_pct,
                    SUM(CASE WHEN regexp_matches(business_name, '[^[:ascii:]]') THEN 1 ELSE 0 END)*100.0/COUNT(*) AS non_ascii_name_pct,
                    SUM(CASE WHEN regexp_matches(business_address, '[^[:ascii:]]') THEN 1 ELSE 0 END)*100.0/COUNT(*) AS non_ascii_addr_pct
                FROM {view}
            """).fetchdf().iloc[0].to_dict()
            rows.append(row)
        comparison[split] = rows

    return comparison
