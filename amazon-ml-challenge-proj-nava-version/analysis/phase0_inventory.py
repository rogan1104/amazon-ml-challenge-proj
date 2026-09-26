"""Phase 0: Environment & Dataset Inventory."""
import os
import time
import duckdb
from pathlib import Path
from analysis.config import SOURCE_FILES, TRAIN, GT_COLS, SOURCE_COLS, read_csv_sql, REPORTS, CACHE


def run_inventory(con: duckdb.DuckDBPyConnection) -> dict:
    """Scan all TSV files and return inventory metadata."""
    t0 = time.time()
    inventory = {}

    for label, path in SOURCE_FILES.items():
        info = _inspect_file(con, label, path, is_gt=False)
        inventory[label] = info

    gt_info = _inspect_file(con, "train_GT", TRAIN["GT"], is_gt=True)
    inventory["train_GT"] = gt_info

    inventory["_elapsed_sec"] = round(time.time() - t0, 2)
    return inventory


def _inspect_file(con, label, path: Path, is_gt: bool) -> dict:
    cols = GT_COLS if is_gt else SOURCE_COLS
    rel = path.relative_to(path.parents[2]) if path.is_absolute() else path

    file_size = os.path.getsize(path)
    csv = read_csv_sql(path)
    row_count = con.execute(f"SELECT COUNT(*) FROM {csv}").fetchone()[0]

    sample = con.execute(f"SELECT * FROM {csv} LIMIT 5").fetchdf()

    # Delimiter check: count tabs in first data line
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.readline()  # header
        first_data = f.readline()
    tab_count = first_data.count("\t")
    expected_tabs = len(cols) - 1

    # Malformed row check: lines with wrong tab count (sample first 100k data lines)
    malformed = 0
    checked = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            checked += 1
            if line.count("\t") != expected_tabs:
                malformed += 1
            if checked >= 100_000:
                break

    # Encoding check
    encoding = "utf-8"
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, _ in enumerate(f):
                if i > 10000:
                    break
    except UnicodeDecodeError:
        encoding = "mixed/unknown"

    # Approx memory: file_size * ~2 for in-memory string representation
    approx_mem_mb = round(file_size * 2.5 / (1024 * 1024), 1)

    dtypes = {c: str(sample[c].dtype) for c in sample.columns}

    return {
        "label": label,
        "path": str(path),
        "relative_path": str(rel),
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
        "row_count": row_count,
        "column_count": len(cols),
        "columns": cols,
        "dtypes_inferred": dtypes,
        "delimiter_tabs_per_row": tab_count,
        "delimiter_correct": tab_count == expected_tabs,
        "encoding": encoding,
        "malformed_rows_sampled": malformed,
        "lines_sampled_for_malformed": checked,
        "approx_memory_mb": approx_mem_mb,
    }


def register_views(con: duckdb.DuckDBPyConnection):
    """Register DuckDB views for all source files."""
    for label, path in SOURCE_FILES.items():
        cols = SOURCE_COLS
        view = label.replace("-", "_")
        con.execute(f"""
            CREATE OR REPLACE VIEW {view} AS
            SELECT * FROM {read_csv_sql(path)}
        """)

    con.execute(f"""
        CREATE OR REPLACE VIEW train_gt AS
        SELECT * FROM {read_csv_sql(TRAIN["GT"])}
    """)

    # Parsed ground truth: explode matched_entity_ids
    con.execute("""
        CREATE OR REPLACE VIEW gt_pairs AS
        SELECT
            g.source1_entity_id AS s1_id,
            TRIM(m.id) AS matched_id,
            CASE WHEN TRIM(m.id) LIKE 'S2-%' THEN 'S2'
                 WHEN TRIM(m.id) LIKE 'S3-%' THEN 'S3'
                 ELSE 'OTHER' END AS match_source
        FROM train_gt g,
        LATERAL (
            SELECT UNNEST(
                CASE WHEN g.matched_entity_ids IS NULL OR TRIM(g.matched_entity_ids) = ''
                     THEN []::VARCHAR[]
                     ELSE string_split(g.matched_entity_ids, ',')
                END
            ) AS id
        ) m
        WHERE TRIM(m.id) != ''
    """)

    con.execute("""
        CREATE OR REPLACE VIEW gt_summary AS
        SELECT
            source1_entity_id AS s1_id,
            matched_entity_ids,
            CASE WHEN matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = ''
                 THEN 0
                 ELSE len(string_split(matched_entity_ids, ','))
            END AS match_count,
            CASE WHEN matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = ''
                 THEN 0
                 ELSE len(list_filter(string_split(matched_entity_ids, ','), x -> TRIM(x) LIKE 'S2-%'))
            END AS s2_match_count,
            CASE WHEN matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = ''
                 THEN 0
                 ELSE len(list_filter(string_split(matched_entity_ids, ','), x -> TRIM(x) LIKE 'S3-%'))
            END AS s3_match_count
        FROM train_gt
    """)
