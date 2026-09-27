"""Configuration for dataset forensics pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
REPORTS = ROOT / "reports"
CACHE = REPORTS / "cache"

TRAIN = {
    "S1": DATASET / "train" / "train_source1.tsv",
    "S2": DATASET / "train" / "train_source2.tsv",
    "S3": DATASET / "train" / "train_source3.tsv",
    "GT": DATASET / "train" / "train_ground_truth.tsv",
}
TEST = {
    "S1": DATASET / "test" / "test_source1.tsv",
    "S2": DATASET / "test" / "test_source2.tsv",
    "S3": DATASET / "test" / "test_source3.tsv",
}

SOURCE_FILES = {
    "train_S1": TRAIN["S1"],
    "train_S2": TRAIN["S2"],
    "train_S3": TRAIN["S3"],
    "test_S1": TEST["S1"],
    "test_S2": TEST["S2"],
    "test_S3": TEST["S3"],
}

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS = ["source1_entity_id", "matched_entity_ids"]

# DuckDB read options (auto-detect columns from header)
def read_csv_sql(path: Path) -> str:
    return f"read_csv('{path}', header=true, delim='\\t', quote='', escape='', nullstr='')"
