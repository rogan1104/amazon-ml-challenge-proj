"""Load source TSVs and write candidate_pairs.tsv (tab-separated, no quoting)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def load_source(path: str | Path) -> pd.DataFrame:
    """Read a source file. Empty fields stay empty strings, not NaN."""
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        encoding="utf-8",
        keep_default_na=False,
        usecols=SOURCE_COLUMNS,
    )
    missing = [c for c in SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    # Country is an open string label - just strip whitespace, never map values.
    df["country"] = df["country"].str.strip()
    return df


def load_sources(data_dir: str | Path, split: str = "train") -> Dict[str, pd.DataFrame]:
    """Load source1/2/3 from ``dataset/{split}/``.

    ``split`` is ``train`` or ``test`` so the same code runs on either folder.
    """
    data_dir = Path(data_dir)
    prefix = "train" if split == "train" else "test"
    return {
        "s1": load_source(data_dir / f"{prefix}_source1.tsv"),
        "s2": load_source(data_dir / f"{prefix}_source2.tsv"),
        "s3": load_source(data_dir / f"{prefix}_source3.tsv"),
    }


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    """Train labels. Empty ``matched_entity_ids`` -> empty string."""
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        encoding="utf-8",
        keep_default_na=False,
    )
    return df


def parse_id_list(raw: object) -> List[str]:
    text = "" if raw is None else str(raw).strip()
    if not text:
        return []
    # Preserve order, drop duplicates (validator forbids intra-list dupes).
    seen = set()
    out: List[str] = []
    for part in text.split(","):
        eid = part.strip()
        if eid and eid not in seen:
            seen.add(eid)
            out.append(eid)
    return out


def write_candidate_pairs(
    path: str | Path,
    source1_ids: Sequence[str],
    candidates: Mapping[str, Sequence[str]],
) -> None:
    """Write one row per S1 entity. ID lists are comma-joined, never quoted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in source1_ids:
            ids = candidates.get(s1_id, ())
            # Defensive unique while keeping retrieval order.
            uniq = list(dict.fromkeys(ids))
            fh.write(f"{s1_id}\t{','.join(uniq)}\n")


def iter_candidate_rows(path: str | Path) -> Iterable[tuple[str, List[str]]]:
    """Stream ``candidate_pairs.tsv`` without quoting assumptions."""
    with Path(path).open(encoding="utf-8") as fh:
        header = fh.readline()
        if not header:
            return
        for line in fh:
            s1, sep, rest = line.rstrip("\n").partition("\t")
            if not sep:
                continue
            yield s1, parse_id_list(rest)
