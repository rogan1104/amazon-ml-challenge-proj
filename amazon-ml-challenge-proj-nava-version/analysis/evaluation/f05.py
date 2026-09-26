"""
Macro F_0.5 evaluation for ML Challenge 2026 Business Entity Resolution.

Matches the challenge README:
  F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
  Macro-average F_0.5 over every Source 1 entity in the evaluation universe.

Singleton ground truth (no true matches): F_0.5 = 1.0 if prediction is empty,
0.0 if any match is predicted.
"""
from __future__ import annotations

import unittest
from collections.abc import Mapping, Sequence
from typing import AbstractSet

# README β = 0.5 → (1 + β²) = 1.25, β² = 0.25
_F05_NUM = 1.25
_F05_DEN_COEF = 0.25


def _clean_id_set(ids: AbstractSet[str] | None) -> set[str]:
    if not ids:
        return set()
    return {x.strip() for x in ids if x and str(x).strip()}


def f05_from_precision_recall(precision: float, recall: float) -> float:
    """F_0.5 from precision and recall (both in [0, 1])."""
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    denom = _F05_DEN_COEF * precision + recall
    if denom <= 0.0:
        return 0.0
    return (_F05_NUM * precision * recall) / denom


def per_s1_prf(
    ground_truth: AbstractSet[str],
    predicted: AbstractSet[str],
) -> tuple[float, float, float]:
    """
    Per-S1 precision, recall, and F_0.5.

    ground_truth / predicted are sets of S2-/S3- entity ids (empty GT = singleton).
    """
    gt = _clean_id_set(ground_truth)
    pred = _clean_id_set(predicted)

    if not gt:
        if not pred:
            return 1.0, 1.0, 1.0
        return 0.0, 0.0, 0.0

    if not pred:
        return 0.0, 0.0, 0.0

    tp = len(gt & pred)
    precision = tp / len(pred)
    recall = tp / len(gt)
    f05 = f05_from_precision_recall(precision, recall)
    return precision, recall, f05


def macro_f05(per_entity_f05: Sequence[float]) -> float:
    """Unweighted mean of per-entity F_0.5 scores (0.0 if empty sequence)."""
    if not per_entity_f05:
        return 0.0
    return sum(per_entity_f05) / len(per_entity_f05)


def _match_count_bucket(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if n == 3:
        return "3"
    return "4+"


def _macro_for_ids(
    s1_ids: Sequence[str],
    f05_by_s1: Mapping[str, float],
) -> float:
    if not s1_ids:
        return 0.0
    return macro_f05([f05_by_s1[s1] for s1 in s1_ids])


def evaluate_f05(
    gt_by_s1: Mapping[str, AbstractSet[str]],
    pred_by_s1: Mapping[str, AbstractSet[str]],
    *,
    s1_universe: Sequence[str] | None = None,
    country_by_s1: Mapping[str, str] | None = None,
) -> dict:
    """
    Compute macro F_0.5 and breakdowns.

    Parameters
    ----------
    gt_by_s1
        Ground-truth matched entity ids per source1_entity_id.
    pred_by_s1
        Predicted matched entity ids per source1_entity_id.
    s1_universe
        Every S1 entity to score. Missing keys in pred_by_s1 count as empty prediction.
        Defaults to sorted keys of gt_by_s1.
    country_by_s1
        Optional map S1 -> country label (e.g. US, India) for regional breakdowns.
    """
    if s1_universe is None:
        universe = sorted(gt_by_s1.keys())
    else:
        universe = list(s1_universe)

    gt_clean: dict[str, set[str]] = {
        s1: _clean_id_set(gt_by_s1.get(s1)) for s1 in universe
    }
    pred_clean: dict[str, set[str]] = {
        s1: _clean_id_set(pred_by_s1.get(s1)) for s1 in universe
    }

    per_s1: dict[str, dict] = {}
    f05_by_s1: dict[str, float] = {}

    for s1 in universe:
        p, r, f = per_s1_prf(gt_clean[s1], pred_clean[s1])
        per_s1[s1] = {
            "precision": round(p, 6),
            "recall": round(r, 6),
            "f05": round(f, 6),
            "gt_size": len(gt_clean[s1]),
            "pred_size": len(pred_clean[s1]),
            "tp": len(gt_clean[s1] & pred_clean[s1]),
        }
        f05_by_s1[s1] = f

    overall = macro_f05([f05_by_s1[s1] for s1 in universe])

    singleton_ids = [s1 for s1 in universe if len(gt_clean[s1]) == 0]
    matched_ids = [s1 for s1 in universe if len(gt_clean[s1]) > 0]

    breakdown: dict = {
        "overall_macro_f05": round(overall, 6),
        "n_s1": len(universe),
        "by_entity_type": {
            "singleton": {
                "macro_f05": round(_macro_for_ids(singleton_ids, f05_by_s1), 6),
                "n_s1": len(singleton_ids),
            },
            "matched": {
                "macro_f05": round(_macro_for_ids(matched_ids, f05_by_s1), 6),
                "n_s1": len(matched_ids),
            },
        },
        "by_match_count_bucket": {},
    }

    for bucket in ("0", "1", "2", "3", "4+"):
        ids = [s1 for s1 in universe if _match_count_bucket(len(gt_clean[s1])) == bucket]
        breakdown["by_match_count_bucket"][bucket] = {
            "macro_f05": round(_macro_for_ids(ids, f05_by_s1), 6),
            "n_s1": len(ids),
        }

    if country_by_s1 is not None:
        by_country: dict[str, dict] = {}
        for s1 in universe:
            raw = (country_by_s1.get(s1) or "").strip()
            label = raw.upper() if raw else "UNKNOWN"
            if label == "INDIA":
                label = "IN"
            if label not in by_country:
                by_country[label] = []
            by_country[label].append(s1)
        breakdown["by_country"] = {
            country: {
                "macro_f05": round(_macro_for_ids(ids, f05_by_s1), 6),
                "n_s1": len(ids),
            }
            for country, ids in sorted(by_country.items())
        }
    else:
        breakdown["by_country"] = {}

    return {
        "macro_f05": round(overall, 6),
        "per_s1": per_s1,
        "breakdown": breakdown,
    }


class TestF05Metric(unittest.TestCase):
    """Hand-checkable examples (README + edge cases)."""

    def test_readme_example(self) -> None:
        gt = {"S2-00047", "S3-00812"}
        pred = {"S2-00047", "S2-00193", "S3-00812"}
        p, r, f = per_s1_prf(gt, pred)
        self.assertAlmostEqual(p, 2 / 3, places=6)
        self.assertAlmostEqual(r, 1.0, places=6)
        self.assertAlmostEqual(f, 5 / 7, places=6)  # README ~0.714

    def test_singleton_correct(self) -> None:
        p, r, f = per_s1_prf(set(), set())
        self.assertEqual((p, r, f), (1.0, 1.0, 1.0))

    def test_singleton_false_positive(self) -> None:
        p, r, f = per_s1_prf(set(), {"S2-1"})
        self.assertEqual((p, r, f), (0.0, 0.0, 0.0))

    def test_missed_all_non_singleton(self) -> None:
        p, r, f = per_s1_prf({"S2-a"}, set())
        self.assertEqual((p, r, f), (0.0, 0.0, 0.0))

    def test_perfect_non_singleton(self) -> None:
        gt = {"S2-a", "S3-b"}
        p, r, f = per_s1_prf(gt, set(gt))
        self.assertEqual((p, r, f), (1.0, 1.0, 1.0))

    def test_extra_false_positives_only(self) -> None:
        gt = {"S2-a"}
        pred = {"S2-a", "S2-b", "S3-c"}
        p, r, f = per_s1_prf(gt, pred)
        self.assertAlmostEqual(p, 1 / 3, places=6)
        self.assertAlmostEqual(r, 1.0, places=6)
        expected_f = f05_from_precision_recall(1 / 3, 1.0)
        self.assertAlmostEqual(f, expected_f, places=6)

    def test_missing_pred_treated_as_empty(self) -> None:
        gt_by_s1 = {"S1-a": {"S2-1"}, "S1-b": set()}
        pred_by_s1: dict[str, set[str]] = {}  # S1-a missing
        report = evaluate_f05(gt_by_s1, pred_by_s1)
        self.assertEqual(report["per_s1"]["S1-a"]["f05"], 0.0)
        self.assertEqual(report["per_s1"]["S1-b"]["f05"], 1.0)
        # macro = (0 + 1) / 2
        self.assertAlmostEqual(report["macro_f05"], 0.5, places=6)

    def test_macro_and_breakdowns(self) -> None:
        gt_by_s1 = {
            "S1-us-s": set(),
            "S1-us-m": {"S2-1"},
            "S1-in-m": {"S2-2", "S3-3"},
        }
        pred_by_s1 = {
            "S1-us-s": set(),
            "S1-us-m": {"S2-1"},
            "S1-in-m": {"S2-2"},  # partial recall
        }
        country = {
            "S1-us-s": "US",
            "S1-us-m": "US",
            "S1-in-m": "India",
        }
        report = evaluate_f05(
            gt_by_s1,
            pred_by_s1,
            country_by_s1=country,
        )
        self.assertAlmostEqual(report["per_s1"]["S1-in-m"]["recall"], 0.5, places=6)
        self.assertEqual(report["breakdown"]["by_entity_type"]["singleton"]["n_s1"], 1)
        self.assertEqual(report["breakdown"]["by_entity_type"]["matched"]["n_s1"], 2)
        self.assertEqual(report["breakdown"]["by_match_count_bucket"]["0"]["n_s1"], 1)
        self.assertEqual(report["breakdown"]["by_match_count_bucket"]["2"]["n_s1"], 1)
        self.assertIn("US", report["breakdown"]["by_country"])
        self.assertIn("IN", report["breakdown"]["by_country"])


if __name__ == "__main__":
    unittest.main()
