"""Blocking-quality metrics against train_ground_truth.tsv (optional)."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

from data_io import parse_id_list


def pair_recall(
    candidates: Mapping[str, Sequence[str]],
    truth: Mapping[str, Sequence[str]],
) -> Dict[str, float]:
    """Candidate-generation recall: fraction of true S2/S3 IDs retrieved.

    This is the recall *ceiling* for the matching stage - a true match that
    never appears here cannot be recovered later.
    """
    total_true = 0
    recovered = 0
    entities_with_matches = 0
    entities_fully_covered = 0

    for s1_id, true_ids in truth.items():
        if not true_ids:
            continue
        entities_with_matches += 1
        cand = set(candidates.get(s1_id, ()))
        total_true += len(true_ids)
        hit = sum(1 for t in true_ids if t in cand)
        recovered += hit
        if hit == len(true_ids):
            entities_fully_covered += 1

    pair_rec = recovered / total_true if total_true else 1.0
    entity_rec = (
        entities_fully_covered / entities_with_matches if entities_with_matches else 1.0
    )
    return {
        "true_pairs": float(total_true),
        "recovered_pairs": float(recovered),
        "pair_recall": pair_rec,
        "entities_with_matches": float(entities_with_matches),
        "entities_fully_covered": float(entities_fully_covered),
        "entity_recall": entity_rec,
    }


def truth_from_frame(source1_ids: Sequence[str], matched_raw: Sequence[object]) -> Dict[str, List[str]]:
    return {
        s1: parse_id_list(raw) for s1, raw in zip(source1_ids, matched_raw)
    }
