"""Regression tests for candidate TSV validation."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from analysis.retrieval.mmap_baseline.validate_output import validate_candidate_pairs_tsv

_DEFAULT_LIMIT = 128 * 1024


class TestValidateOutput(unittest.TestCase):
    def test_accepts_candidate_field_larger_than_default_csv_limit(self) -> None:
        """Regression: wide candidate_entity_ids must not hit _csv.Error 131072."""
        big_field = ",".join(f"S2-{i:010d}" for i in range(20_000))
        self.assertGreater(len(big_field), _DEFAULT_LIMIT)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "wide.tsv"
            with path.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, delimiter="\t", lineterminator="\n")
                w.writerow(["source1_entity_id", "candidate_entity_ids"])
                w.writerow(["S1-WIDE", big_field])

            report = validate_candidate_pairs_tsv(path, expected_s1_ids=["S1-WIDE"])
            self.assertTrue(report["ok"])
            self.assertEqual(report["s1_rows"], 1)
            self.assertEqual(report["total_candidate_ids"], 20_000)


if __name__ == "__main__":
    unittest.main()
