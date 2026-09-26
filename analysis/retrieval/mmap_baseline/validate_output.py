"""Validate candidate_pairs.tsv for the scoring / feature pipeline."""
from __future__ import annotations

import csv
from pathlib import Path

from analysis.scoring.pairs import candidate_source


def validate_candidate_pairs_tsv(
    path: Path | str,
    *,
    expected_s1_ids: list[str] | None = None,
) -> dict:
    """
    Check challenge-format candidate TSV (same contract as SQLite runner output).

    Raises ValueError on schema or ID violations.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    seen_s1: set[str] = set()
    total_rows = 0
    total_cands = 0

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", "candidate_entity_ids"]:
            raise ValueError(f"Bad header: {reader.fieldnames!r}")

        for row in reader:
            s1_id = (row.get("source1_entity_id") or "").strip()
            if not s1_id:
                raise ValueError("Empty source1_entity_id row")
            if s1_id in seen_s1:
                raise ValueError(f"Duplicate source1_entity_id row: {s1_id!r}")
            seen_s1.add(s1_id)
            total_rows += 1

            raw = (row.get("candidate_entity_ids") or "").strip()
            if not raw:
                continue
            parts = [x.strip() for x in raw.split(",") if x.strip()]
            if len(parts) != len(set(parts)):
                raise ValueError(f"Duplicate candidate ids in row {s1_id!r}")
            for cid in parts:
                candidate_source(cid)
            total_cands += len(parts)

    if expected_s1_ids is not None:
        missing = [s for s in expected_s1_ids if s not in seen_s1]
        extra = seen_s1 - set(expected_s1_ids)
        if missing:
            raise ValueError(f"Missing S1 ids in output ({len(missing)}): {missing[:5]}...")
        if extra:
            raise ValueError(f"Unexpected extra S1 ids ({len(extra)}): {list(extra)[:5]}...")

    return {
        "path": str(path),
        "s1_rows": total_rows,
        "total_candidate_ids": total_cands,
        "ok": True,
    }
