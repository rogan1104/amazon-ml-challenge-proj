"""Validation metrics for pair scoring (macro F_0.5 + threshold sweep)."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from analysis.evaluation.f05 import evaluate_f05, macro_f05

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def predictions_from_probas(
    scored_pairs: Sequence[dict[str, Any]],
    threshold: float,
    s1_universe: Sequence[str],
) -> dict[str, set[str]]:
    """Build pred_by_s1 from retrieved pair probabilities."""
    pred: dict[str, set[str]] = {s1: set() for s1 in s1_universe}
    for row in scored_pairs:
        if float(row["proba"]) >= threshold:
            pred.setdefault(row["s1_id"], set()).add(row["candidate_id"])
    return pred


def _macro_avg(values: Sequence[float]) -> float:
    return macro_f05(list(values)) if values else 0.0


def unretrieved_gt_positive_stats(
    gt_by_s1: Mapping[str, set[str]],
    candidates_by_s1: Mapping[str, set[str]],
    s1_universe: Sequence[str],
) -> dict[str, int]:
    """Count GT positives not present in retrieval candidates."""
    total_gt = 0
    unretrieved = 0
    for s1 in s1_universe:
        gt_set = gt_by_s1.get(s1, set())
        cands = candidates_by_s1.get(s1, set())
        total_gt += len(gt_set)
        unretrieved += len(gt_set - cands)
    retrieved = total_gt - unretrieved
    return {
        "gt_positive_pairs": total_gt,
        "retrieved_gt_positive_pairs": retrieved,
        "unretrieved_gt_positive_pairs": unretrieved,
    }


def retrieved_positive_coverage(
    scored_pairs: Sequence[dict[str, Any]],
    threshold: float,
) -> float:
    """Fraction of retrieved GT positives predicted at threshold."""
    pos = [p for p in scored_pairs if int(p["label"]) == 1 and p.get("in_retrieval")]
    if not pos:
        return 0.0
    hit = sum(1 for p in pos if float(p["proba"]) >= threshold)
    return hit / len(pos)


def evaluate_threshold(
    scored_pairs: Sequence[dict[str, Any]],
    gt_by_s1: Mapping[str, set[str]],
    s1_universe: Sequence[str],
    country_by_s1: Mapping[str, str] | None,
    threshold: float,
) -> dict[str, Any]:
    pred_by_s1 = predictions_from_probas(scored_pairs, threshold, s1_universe)
    f05_report = evaluate_f05(
        gt_by_s1,
        pred_by_s1,
        s1_universe=s1_universe,
        country_by_s1=country_by_s1,
    )

    per_s1 = f05_report["per_s1"]
    macro_p = _macro_avg([per_s1[s]["precision"] for s in s1_universe])
    macro_r = _macro_avg([per_s1[s]["recall"] for s in s1_universe])

    n_predicted_matches = sum(len(pred_by_s1.get(s1, set())) for s1 in s1_universe)

    return {
        "threshold": threshold,
        "macro_f05": f05_report["macro_f05"],
        "macro_precision": round(macro_p, 6),
        "macro_recall": round(macro_r, 6),
        "singleton_macro_f05": f05_report["breakdown"]["by_entity_type"]["singleton"]["macro_f05"],
        "non_singleton_macro_f05": f05_report["breakdown"]["by_entity_type"]["matched"]["macro_f05"],
        "by_match_count_bucket": f05_report["breakdown"]["by_match_count_bucket"],
        "by_country": f05_report["breakdown"].get("by_country", {}),
        "retrieved_positive_coverage": round(
            retrieved_positive_coverage(scored_pairs, threshold), 6
        ),
        "n_candidate_pairs_scored": len(scored_pairs),
        "n_predicted_match_pairs": n_predicted_matches,
        "f05_detail": f05_report,
    }


def evaluate_threshold_sweep(
    scored_pairs: Sequence[dict[str, Any]],
    gt_by_s1: Mapping[str, set[str]],
    s1_universe: Sequence[str],
    country_by_s1: Mapping[str, str] | None = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    candidates_by_s1: dict[str, set[str]] = defaultdict(set)
    for row in scored_pairs:
        candidates_by_s1[row["s1_id"]].add(row["candidate_id"])
    unret = unretrieved_gt_positive_stats(gt_by_s1, candidates_by_s1, s1_universe)
    rows = [
        {
            k: v
            for k, v in evaluate_threshold(
                scored_pairs,
                gt_by_s1,
                s1_universe,
                country_by_s1,
                t,
            ).items()
            if k != "f05_detail"
        }
        for t in thresholds
    ]
    return {
        "unretrieved_gt_positives": unret,
        "threshold_results": rows,
    }


def score_pairs_with_model(
    pairs: Sequence[dict[str, Any]],
    store,
    model,
) -> list[dict[str, Any]]:
    from analysis.scoring.model import feature_matrix_from_pairs

    X, _ = feature_matrix_from_pairs(pairs, store)
    probas = model.predict_proba(X)
    out: list[dict[str, Any]] = []
    for pair, proba in zip(pairs, probas, strict=True):
        row = dict(pair)
        row["proba"] = float(proba)
        out.append(row)
    return out
