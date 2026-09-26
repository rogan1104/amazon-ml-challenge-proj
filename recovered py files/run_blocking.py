"""CLI for stage 1 blocking.

Example (from ``code/business_entity_resolution``)::

    python src/run_blocking.py --split train --k 50

Reads ``dataset/{train,test}/*_source{1,2,3}.tsv`` and writes
``output/candidate_pairs.tsv``. No network calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python src/run_blocking.py`` without installing a package.
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from blocking import retrieve_candidates
from data_io import load_ground_truth, load_sources, write_candidate_pairs
from metrics import pair_recall, truth_from_frame


def _repo_root() -> Path:
    """``student_resource/`` - parents: src/ -> business_entity_resolution/ -> code/ -> root."""
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    p = argparse.ArgumentParser(description="Stage 1: TF-IDF cosine blocking")
    p.add_argument(
        "--data-dir",
        default=None,
        help="Folder containing *_source1/2/3.tsv (default: dataset/<split>)",
    )
    p.add_argument("--split", choices=("train", "test"), default="train")
    p.add_argument(
        "--output",
        default=None,
        help="Path for candidate_pairs.tsv (default: <root>/output/candidate_pairs.tsv)",
    )
    p.add_argument("--k", type=int, default=50, help="Neighbors per S1 entity")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Debug: use only the first N rows of each source (0 = all)",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--sample-per-source", type=int, default=250_000)
    p.add_argument(
        "--evaluate",
        action="store_true",
        help="After writing, score pair-recall vs train_ground_truth.tsv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = _repo_root()
    data_dir = Path(args.data_dir) if args.data_dir else root / "dataset" / args.split
    output = Path(args.output) if args.output else root / "output" / "candidate_pairs.tsv"

    sources = load_sources(data_dir, split=args.split)
    if args.limit:
        for key in sources:
            sources[key] = sources[key].head(args.limit).copy()
        print(f"[run] limit={args.limit} rows per source")

    print(
        f"[run] S1={len(sources['s1']):,}  S2={len(sources['s2']):,}  "
        f"S3={len(sources['s3']):,}  k={args.k}"
    )

    candidates = retrieve_candidates(
        sources["s1"],
        sources["s2"],
        sources["s3"],
        k=args.k,
        query_batch_size=args.batch_size,
        vectorizer_kwargs={"sample_per_source": args.sample_per_source},
    )

    s1_ids = sources["s1"]["entity_id"].tolist()
    write_candidate_pairs(output, s1_ids, candidates)
    print(f"[run] wrote {output} ({len(s1_ids):,} rows)")

    if args.evaluate:
        gt_path = data_dir / "train_ground_truth.tsv"
        if not gt_path.is_file():
            print(f"[run] --evaluate skipped: {gt_path} not found")
            return
        gt = load_ground_truth(gt_path)
        truth = truth_from_frame(gt["source1_entity_id"], gt["matched_entity_ids"])
        stats = pair_recall(candidates, truth)
        print(
            "[eval] pair_recall={pair_recall:.4f} "
            "({recovered_pairs:.0f}/{true_pairs:.0f})  "
            "entity_recall={entity_recall:.4f} "
            "({entities_fully_covered:.0f}/{entities_with_matches:.0f})".format(**stats)
        )


if __name__ == "__main__":
    main()
