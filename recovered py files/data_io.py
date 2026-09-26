"""Load source TSVs and read/write candidate pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def load_source(path: str | Path) -> pd.DataFrame:
    """Read a source TSV, preserving empty strings."""
    df = pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8", keep_default_na=False, usecols=SOURCE_COLUMNS)
    missing = [c for c in SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    df["country"] = df["country"].str.strip()
    return df


def load_sources(data_dir: str | Path, split: str = "train") -> Dict[str, pd.DataFrame]:
    """Load the three sources for the requested split."""
    data_dir = Path(data_dir)
    prefix = "train" if split == "train" else "test"
    return {
        "s1": load_source(data_dir / f"{prefix}_source1.tsv"),
        "s2": load_source(data_dir / f"{prefix}_source2.tsv"),
        "s3": load_source(data_dir / f"{prefix}_source3.tsv"),
    }


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    """Read train ground truth TSV."""
    return pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8", keep_default_na=False)


def parse_id_list(raw: object) -> List[str]:
    """Parse a comma-separated ID list, dropping blanks and duplicates."""
    text = "" if raw is None else str(raw).strip()
    if not text:
        return []
    seen = set()
    out = []
    for part in text.split(","):
        eid = part.strip()
        if eid and eid not in seen:
            seen.add(eid)
            out.append(eid)
    return out


def write_candidate_pairs(path: str | Path, source1_ids: Sequence[str], candidates: Mapping[str, Sequence[str]]) -> None:
    """Write one row per S1 entity, with comma-separated candidate IDs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in source1_ids:
            ids = list(dict.fromkeys(candidates.get(s1_id, ())))
            fh.write(f"{s1_id}\t{','.join(ids)}\n")


def iter_candidate_rows(path: str | Path) -> Iterable[tuple[str, List[str]]]:
    """Stream candidate TSV rows without quoting assumptions."""
    with Path(path).open(encoding="utf-8") as fh:
        if not fh.readline():
            return
        for line in fh:
            s1, sep, rest = line.rstrip("\n").partition("\t")
            if sep:
                yield s1, parse_id_list(rest)
