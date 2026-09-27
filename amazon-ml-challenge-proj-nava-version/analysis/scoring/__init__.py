"""Pair-level training data for entity resolution scoring."""

from analysis.scoring.pairs import (
    EntityRecord,
    build_labeled_pairs,
    candidate_source,
    load_gt_by_s1,
    match_count_bucket,
    split_by_s1,
    stream_candidate_pairs,
)

__all__ = [
    "EntityRecord",
    "build_labeled_pairs",
    "candidate_source",
    "load_gt_by_s1",
    "match_count_bucket",
    "split_by_s1",
    "stream_candidate_pairs",
]
