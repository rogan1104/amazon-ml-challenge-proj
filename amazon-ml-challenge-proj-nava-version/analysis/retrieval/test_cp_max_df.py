"""Tests for baseline CP key_df filtering (cp_max_df)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from analysis.retrieval.config import ChannelName
from analysis.retrieval.index import InvertedIndex


def _seed_baseline_index(index: InvertedIndex) -> None:
    ch = ChannelName.BASELINE
    target = "S2"
    rows = [
        ("bn:exact_name", "E-BN"),
        ("ba:exact_addr", "E-BA"),
        ("cp:US|highf", "E-CP-HIGH"),
        ("cp:US|lowfx", "E-CP-LOW"),
    ]
    index.add_postings_batch(ch, target, rows)  # type: ignore[arg-type]
    index.finalize_df(ch, target)  # type: ignore[arg-type]
    conn = index._conn
    conn.execute(
        """
        UPDATE key_df SET df = ?
        WHERE channel = ? AND target = ? AND key = ?
        """,
        (5000, ch.value, target, "cp:US|highf"),
    )
    conn.execute(
        """
        UPDATE key_df SET df = ?
        WHERE channel = ? AND target = ? AND key = ?
        """,
        (100, ch.value, target, "cp:US|lowfx"),
    )
    conn.commit()


class TestCpMaxDf(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.db_path = Path(self._td.name) / "test.sqlite"
        self.index = InvertedIndex(self.db_path)
        _seed_baseline_index(self.index)
        self.index.close()
        self.index = InvertedIndex(self.db_path, read_only=True)
        self.queries = [
            (
                "Q1",
                [
                    "bn:exact_name",
                    "ba:exact_addr",
                    "cp:US|highf",
                    "cp:US|lowfx",
                ],
            ),
        ]

    def tearDown(self) -> None:
        self.index.close()
        self._td.cleanup()

    def test_none_matches_exact_batch(self) -> None:
        plain = self.index.lookup_keys_exact_batch(
            ChannelName.BASELINE, "S2", self.queries,
        )
        unfiltered_via_high_cap = self.index.lookup_keys_exact_batch_with_cp_max_df(
            ChannelName.BASELINE, "S2", self.queries, cp_max_df=999_999,
        )
        self.assertEqual(plain, unfiltered_via_high_cap)
        self.assertEqual(
            plain["Q1"],
            {"E-BN", "E-BA", "E-CP-HIGH", "E-CP-LOW"},
        )

        filtered = self.index.lookup_keys_exact_batch_with_cp_max_df(
            ChannelName.BASELINE, "S2", self.queries, cp_max_df=500,
        )
        self.assertEqual(filtered["Q1"], {"E-BN", "E-BA", "E-CP-LOW"})

    def test_bn_ba_unaffected(self) -> None:
        out = self.index.lookup_keys_exact_batch_with_cp_max_df(
            ChannelName.BASELINE,
            "S2",
            [("Q1", ["bn:exact_name", "ba:exact_addr"])],
            cp_max_df=1,
        )
        self.assertEqual(out["Q1"], {"E-BN", "E-BA"})

    def test_cp_above_threshold_excluded(self) -> None:
        out = self.index.lookup_keys_exact_batch_with_cp_max_df(
            ChannelName.BASELINE,
            "S2",
            [("Q1", ["cp:US|highf", "cp:US|lowfx"])],
            cp_max_df=500,
        )
        self.assertEqual(out["Q1"], {"E-CP-LOW"})

    def test_single_row_lookup_consistent(self) -> None:
        keys = ["bn:exact_name", "cp:US|highf", "cp:US|lowfx"]
        batch = self.index.lookup_keys_exact_batch_with_cp_max_df(
            ChannelName.BASELINE, "S2", [("Q1", keys)], cp_max_df=500,
        )
        single = self.index.lookup_keys_exact_with_cp_max_df(
            ChannelName.BASELINE, "S2", keys, cp_max_df=500,
        )
        self.assertEqual(batch["Q1"], single)

    def test_output_format(self) -> None:
        out = self.index.lookup_keys_exact_batch_with_cp_max_df(
            ChannelName.BASELINE, "S2", self.queries, cp_max_df=500,
        )
        self.assertIsInstance(out, dict)
        self.assertIn("Q1", out)
        self.assertIsInstance(out["Q1"], set)


if __name__ == "__main__":
    unittest.main()
