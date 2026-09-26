"""Integration: mmap production writer → scoring pipeline candidate TSV contract."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from analysis.retrieval.config import BaselineConfig, ChannelName
from analysis.retrieval.index import InvertedIndex
from analysis.retrieval.keys import baseline_keys
from analysis.retrieval.mmap_baseline.build import build_baseline_mmap_index
from analysis.retrieval.mmap_baseline.cli import verify_mmap_vs_sqlite
from analysis.retrieval.mmap_baseline.validate_output import validate_candidate_pairs_tsv
from analysis.retrieval.mmap_baseline.writer import write_candidate_pairs_mmap
from analysis.scoring.pairs import stream_candidate_pairs


def _write_tsv(path: Path, rows: list[dict]) -> None:
    cols = ["entity_id", "business_name", "business_address", "country"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)


class TestMmapProductionIntegration(unittest.TestCase):
    def test_smoke_100_s1_candidate_tsv_for_scoring_pipeline(self) -> None:
        cfg = BaselineConfig()
        s2_rows = [
            {
                "entity_id": f"S2-{i:04d}",
                "business_name": f"Prod Entity {i} LLC",
                "business_address": f"{i} Main St",
                "country": "US",
            }
            for i in range(20)
        ]
        s3_rows = [
            {
                "entity_id": f"S3-{i:04d}",
                "business_name": f"Alt Entity {i} Inc",
                "business_address": f"{i} Oak Ave",
                "country": "US",
            }
            for i in range(15)
        ]
        s1_rows = [
            {
                "entity_id": f"S1-{i:05d}",
                "business_name": f"Prod Entity {i % 20} LLC",
                "business_address": f"{i % 20} Main St",
                "country": "US",
            }
            for i in range(100)
        ]
        expected_s1 = [r["entity_id"] for r in s1_rows]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            s1_path = root / "train_source1.tsv"
            s2_path = root / "train_source2.tsv"
            s3_path = root / "train_source3.tsv"
            _write_tsv(s1_path, s1_rows)
            _write_tsv(s2_path, s2_rows)
            _write_tsv(s3_path, s3_rows)

            idx_root = root / "mmap_index"
            build_baseline_mmap_index(
                idx_root,
                split="train",
                paths={"S2": s2_path, "S3": s3_path},
                cfg=cfg,
            )

            out_tsv = root / "candidate_pairs_smoke.tsv"
            summary = write_candidate_pairs_mmap(
                idx_root,
                out_tsv,
                split="train",
                batch_size=25,
                max_s1_rows=100,
                s1_path=s1_path,
            )
            self.assertEqual(summary["total_s1"], 100)

            validate_candidate_pairs_tsv(out_tsv, expected_s1_ids=expected_s1)

            loaded = list(stream_candidate_pairs(out_tsv))
            self.assertEqual(len(loaded), 100)
            for s1_id, cands in loaded:
                self.assertTrue(s1_id.startswith("S1-"))
                for cid in cands:
                    self.assertTrue(cid.startswith("S2-") or cid.startswith("S3-"))

            # Small SQLite reference on same corpus (cheap; not full train retrieval).
            sqlite_path = root / "ref.sqlite"
            with InvertedIndex(sqlite_path) as index:
                for target, rows in (("S2", s2_rows), ("S3", s3_rows)):
                    batch: list[tuple[str, str]] = []
                    for row in rows:
                        eid = row["entity_id"]
                        for key in baseline_keys(
                            row["business_name"],
                            row["business_address"],
                            row["country"],
                            prefix_len=cfg.name_prefix_len,
                            min_prefix=cfg.min_prefix_len,
                        ):
                            batch.append((key, eid))
                    index.add_postings_batch(ChannelName.BASELINE, target, batch)  # type: ignore[arg-type]
                    index.finalize_df(ChannelName.BASELINE, target)  # type: ignore[arg-type]

            sample_s1 = s1_rows[:10]
            report = verify_mmap_vs_sqlite(
                idx_root, sqlite_path, sample_s1, cfg, partial_build=False,
            )
            self.assertTrue(report["ok"], msg=report.get("mismatch_examples"))


if __name__ == "__main__":
    unittest.main()
