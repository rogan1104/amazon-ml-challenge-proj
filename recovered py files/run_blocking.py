"""CLI entrypoint for stage-1 candidate blocking."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from blocking import retrieve_candidates
from data_io import load_ground_truth, load_sources, write_candidate_pairs
from metrics import pair_recall, truth_from_frame


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Stage 1: TF-IDF cosine blocking")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--output", default=None)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-per-source", type=int, default=250_000)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    project_root = root / "student_resource" if (root / "student_resource" / "dataset").is_dir() else root
    args.data_dir = Path(args.data_dir) if args.data_dir else project_root / "dataset" / args.split
    args.output = Path(args.output) if args.output else project_root / "output" / "candidate_pairs.tsv"
    return args


def main() -> None:
    args = parse_args()
    sources = load_sources(args.data_dir, split=args.split)
    if args.limit:
        for key in sources:
            sources[key] = sources[key].head(args.limit).copy()
        print(f"[run] limit={args.limit} rows per source")
    print(f"[run] S1={len(sources['s1']):,}  S2={len(sources['s2']):,}  S3={len(sources['s3']):,}  k={args.k}")
    candidates = retrieve_candidates(sources["s1"], sources["s2"], sources["s3"], k=args.k, query_batch_size=args.batch_size, vectorizer_kwargs={"sample_per_source": args.sample_per_source})
    s1_ids = sources["s1"]["entity_id"].tolist()
    write_candidate_pairs(args.output, s1_ids, candidates)
    print(f"[run] wrote {args.output} ({len(s1_ids):,} rows)")
    if args.evaluate:
        gt_path = Path(args.data_dir) / "train_ground_truth.tsv"
        if not gt_path.is_file():
            print(f"[run] --evaluate skipped: {gt_path} not found")
            return
        gt = load_ground_truth(gt_path)
        truth = truth_from_frame(gt["source1_entity_id"], gt["matched_entity_ids"])
        stats = pair_recall(candidates, truth)
        print("[eval] pair_recall={pair_recall:.4f} ({recovered_pairs:.0f}/{true_pairs:.0f})  entity_recall={entity_recall:.4f} ({entities_fully_covered:.0f}/{entities_with_matches:.0f})".format(**stats))


if __name__ == "__main__":
    main()
