"""Phase 7: Country Analysis."""
import duckdb


def analyze_countries(con: duckdb.DuckDBPyConnection) -> dict:
    """Country distribution and match behavior."""
    result = {}

    for split in ["train", "test"]:
        for src in ["S1", "S2", "S3"]:
            key = f"{split}_{src}"
            df = con.execute(f"""
                SELECT country, COUNT(*) AS cnt,
                       COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS pct
                FROM {key}
                WHERE country IS NOT NULL AND TRIM(country) != ''
                GROUP BY country ORDER BY cnt DESC
            """).fetchdf()
            result[key] = df.to_dict("records")

    # Train-only vs test-only countries
    train_countries = set()
    test_countries = set()
    for src in ["S1", "S2", "S3"]:
        for c in con.execute(f"SELECT DISTINCT country FROM train_{src} WHERE country IS NOT NULL").fetchall():
            train_countries.add(c[0])
        for c in con.execute(f"SELECT DISTINCT country FROM test_{src} WHERE country IS NOT NULL").fetchall():
            test_countries.add(c[0])

    result["train_countries"] = sorted(train_countries)
    result["test_countries"] = sorted(test_countries)
    result["test_only_countries"] = sorted(test_countries - train_countries)
    result["train_only_countries"] = sorted(train_countries - test_countries)

    # France test set investigation
    result["france_test"] = {}
    for src in ["S1", "S2", "S3"]:
        fr = con.execute(f"""
            SELECT COUNT(*) AS cnt,
                   AVG(LENGTH(business_name)) AS avg_name_len,
                   AVG(LENGTH(business_address)) AS avg_addr_len
            FROM test_{src} WHERE country = 'France'
        """).fetchdf().iloc[0].to_dict()
        total = con.execute(f"SELECT COUNT(*) FROM test_{src}").fetchone()[0]
        fr["pct_of_source"] = round(100 * fr["cnt"] / total, 4) if total else 0
        result["france_test"][src] = fr

    # Match rates by country (train S1)
    result["match_rates_by_country"] = con.execute("""
        SELECT s1.country,
               COUNT(*) AS s1_count,
               SUM(CASE WHEN g.match_count = 0 THEN 1 ELSE 0 END) AS singletons,
               SUM(CASE WHEN g.match_count > 0 THEN 1 ELSE 0 END) AS with_matches,
               AVG(g.match_count) AS avg_matches,
               SUM(CASE WHEN g.match_count = 0 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS singleton_pct
        FROM train_S1 s1
        JOIN gt_summary g ON s1.entity_id = g.s1_id
        WHERE s1.country IS NOT NULL
        GROUP BY s1.country
        ORDER BY s1_count DESC
    """).fetchdf().to_dict("records")

    # Source-specific country distributions in train
    result["source_country_crosstab"] = con.execute("""
        SELECT country,
               SUM(CASE WHEN src='S1' THEN cnt ELSE 0 END) AS train_S1,
               SUM(CASE WHEN src='S2' THEN cnt ELSE 0 END) AS train_S2,
               SUM(CASE WHEN src='S3' THEN cnt ELSE 0 END) AS train_S3
        FROM (
            SELECT country, 'S1' AS src, COUNT(*) AS cnt FROM train_S1 GROUP BY country
            UNION ALL SELECT country, 'S2', COUNT(*) FROM train_S2 GROUP BY country
            UNION ALL SELECT country, 'S3', COUNT(*) FROM train_S3 GROUP BY country
        ) t
        GROUP BY country ORDER BY train_S1 DESC
    """).fetchdf().to_dict("records")

    return result
