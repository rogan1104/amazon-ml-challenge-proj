"""Tests: synthetic build + lookup; optional SQLite cross-check in temp dir."""
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
from analysis.retrieval.mmap_baseline.lookup import BaselineMmapStore
from analysis.retrieval.mmap_baseline.retrieve import MmapBaselineRetriever
from analysis.retrieval.mmap_baseline.schema import BaselineMmapManifest
def _write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["entity_id", "business_name", "business_address", "country"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)


class TestMmapBaselineSynthetic(unittest.TestCase):
    def test_baseline_config_dict_without_cp_max_df_on_main(self) -> None:
        """Regression: main BaselineConfig has no cp_max_df; manifest must still serialize."""
        cfg = BaselineConfig()
        self.assertFalse(hasattr(cfg, "cp_max_df"))
        blob = BaselineMmapManifest.baseline_config_dict(cfg)
        self.assertIsNone(blob["cp_max_df"])
        self.assertEqual(
            BaselineMmapManifest.normalize_baseline_config(blob),
            BaselineMmapManifest.normalize_baseline_config(
                {"name_prefix_len": 5, "min_prefix_len": 3, "max_candidates_per_s1": 10_000},
            ),
        )
        roundtrip = BaselineMmapManifest.baseline_config_from_manifest(blob)
        self.assertEqual(roundtrip.name_prefix_len, cfg.name_prefix_len)
        self.assertEqual(roundtrip.max_candidates_per_s1, cfg.max_candidates_per_s1)

        manifest = BaselineMmapManifest(
            schema_version=1,
            split="train",
            baseline_config=blob,
            targets={},
            complete=False,
        )
        self.assertTrue(BaselineMmapManifest.matches_config(manifest, cfg))

    def test_keygen_build_lookup_exact_sets(self) -> None:
        cfg = BaselineConfig(name_prefix_len=5, min_prefix_len=3)
        s2_rows = [
            {
                "entity_id": "S2-A",
                "business_name": "Acme Widgets LLC",
                "business_address": "100 Main Street",
                "country": "US",
            },
            {
                "entity_id": "S2-B",
                "business_name": "Acme Supplies Inc",
                "business_address": "200 Oak Avenue",
                "country": "US",
            },
            {
                "entity_id": "S2-C",
                "business_name": "Beta Only Corp",
                "business_address": "100 Main Street",
                "country": "US",
            },
        ]
        s3_rows = [
            {
                "entity_id": "S3-X",
                "business_name": "Gamma Shop",
                "business_address": "7 Gamma Road",
                "country": "FR",
            },
        ]
        s1_probe = {
            "entity_id": "S1-Q1",
            "business_name": "Acme Widgets LLC",
            "business_address": "100 Main Street",
            "country": "US",
        }
        expected_keys = baseline_keys(
            s1_probe["business_name"],
            s1_probe["business_address"],
            s1_probe["country"],
            prefix_len=cfg.name_prefix_len,
            min_prefix=cfg.min_prefix_len,
        )
        self.assertTrue(any(k.startswith("bn:") for k in expected_keys))
        self.assertTrue(any(k.startswith("ba:") for k in expected_keys))
        self.assertTrue(any(k.startswith("cp:") for k in expected_keys))

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            s2_path = root / "s2.tsv"
            s3_path = root / "s3.tsv"
            _write_tsv(s2_path, s2_rows)
            _write_tsv(s3_path, s3_rows)
            idx_root = root / "mmap_index"
            build_baseline_mmap_index(
                idx_root,
                split="train",
                paths={"S2": s2_path, "S3": s3_path},
                cfg=cfg,
                batch_size=10,
            )

            with MmapBaselineRetriever(idx_root, cfg) as ret:
                s2_hits = ret.retrieve_batch("S2", [s1_probe])
                s3_hits = ret.retrieve_batch("S3", [s1_probe])

            # bn + ba + cp on S2 should match A and C (shared norm addr), plus prefix bucket peers.
            self.assertIn("S2-A", s2_hits["S1-Q1"])
            self.assertIn("S2-C", s2_hits["S1-Q1"])
            self.assertIn("S2-B", s2_hits["S1-Q1"])  # shared cp:US|acme prefix with Acme*
            self.assertNotIn("S3-X", s2_hits["S1-Q1"])
            self.assertEqual(s3_hits["S1-Q1"], set())

            # Direct family lookup: ba key hits A and C only.
            with BaselineMmapStore(idx_root) as store:
                ba_key = next(k for k in expected_keys if k.startswith("ba:"))
                arr = store.s2._families["ba"].postings_rowids_array(ba_key)
                self.assertGreater(arr.size, 0)

    def test_mmap_matches_sqlite_on_synthetic(self) -> None:
        cfg = BaselineConfig()
        s2_rows = [
            {
                "entity_id": "S2-1",
                "business_name": "Hello World Co",
                "business_address": "1 First Ave",
                "country": "US",
            },
            {
                "entity_id": "S2-2",
                "business_name": "Hello Again",
                "business_address": "2 Second Ave",
                "country": "US",
            },
        ]
        s3_rows: list[dict] = []
        s1_rows = [
            {
                "entity_id": "S1-1",
                "business_name": "Hello World Co",
                "business_address": "1 First Ave",
                "country": "US",
            },
        ]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            s2_path = root / "s2.tsv"
            s3_path = root / "s3.tsv"
            _write_tsv(s2_path, s2_rows)
            _write_tsv(s3_path, s3_rows)
            idx_root = root / "mmap"
            build_baseline_mmap_index(
                idx_root,
                split="train",
                paths={"S2": s2_path, "S3": s3_path},
                cfg=cfg,
            )

            sqlite_path = root / "test.sqlite"
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

            report = verify_mmap_vs_sqlite(idx_root, sqlite_path, s1_rows, cfg)
            self.assertTrue(report["ok"], msg=report.get("mismatch_examples"))


if __name__ == "__main__":
    unittest.main()
