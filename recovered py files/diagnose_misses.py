"""Read-only diagnostics for train ground-truth pairs missed by blocking."""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Dict, Mapping, Sequence

import pandas as pd

from data_io import iter_candidate_rows, load_ground_truth, load_source, parse_id_list
from normalize import normalize_text, tokenize

_DIGITS = re.compile(r"\d{3,}")


def load_truth(path: str | Path) -> Dict[str, list[str]]:
    frame = load_ground_truth(path)
    return {str(s1): parse_id_list(raw) for s1, raw in zip(frame["source1_entity_id"], frame["matched_entity_ids"])}


def compare_coverage(truth: Mapping[str, Sequence[str]], candidate_path: str | Path) -> tuple[set[str], list[tuple[str, str]]]:
    missing_by_s1: dict[str, list[str]] = {}
    for s1_id, candidate_ids in iter_candidate_rows(candidate_path):
        true_ids = truth.get(str(s1_id), ())
        absent = [candidate_id for candidate_id in true_ids if candidate_id not in set(candidate_ids)]
        if absent:
            missing_by_s1[str(s1_id)] = absent
    full = {s1_id for s1_id, ids in truth.items() if ids and s1_id not in missing_by_s1}
    pairs = [(s1_id, candidate_id) for s1_id, ids in missing_by_s1.items() for candidate_id in ids]
    return full, pairs


def detect_script(text: object) -> str:
    scripts = {"Latin" if "LATIN" in unicodedata.name(char, "") else "non-Latin" for char in str(text or "") if char.isalpha()}
    if not scripts:
        return "no_letters"
    return "mixed" if len(scripts) > 1 else next(iter(scripts))


def records_by_id(frame: pd.DataFrame, wanted: set[str]) -> dict[str, tuple[str, str, str]]:
    selected = frame.loc[frame.entity_id.isin(wanted), ["entity_id", "business_name", "business_address", "country"]]
    return {str(row.entity_id): (row.business_name, row.business_address, row.country) for row in selected.itertuples(index=False)}


def build_details(missed_pairs: Sequence[tuple[str, str]], data_dir: Path) -> pd.DataFrame:
    s1_ids = {s1 for s1, _ in missed_pairs}
    s2_ids = {cid for _, cid in missed_pairs if cid.startswith("S2-")}
    s3_ids = {cid for _, cid in missed_pairs if cid.startswith("S3-")}
    left = records_by_id(load_source(data_dir / "train_source1.tsv"), s1_ids)
    s2 = records_by_id(load_source(data_dir / "train_source2.tsv"), s2_ids)
    s3 = records_by_id(load_source(data_dir / "train_source3.tsv"), s3_ids)
    right = {**s2, **s3}
    rows = []
    for s1_id, candidate_id in missed_pairs:
        if s1_id not in left or candidate_id not in right:
            raise ValueError(f"Source record missing for pair {s1_id}, {candidate_id}")
        name1, addr1, country1 = left[s1_id]
        name2, addr2, country2 = right[candidate_id]
        n1, n2 = normalize_text(name1), normalize_text(name2)
        t1, t2 = set(tokenize(name1)), set(tokenize(name2))
        d1, d2 = bool(_DIGITS.search(normalize_text(addr1))), bool(_DIGITS.search(normalize_text(addr2)))
        script1, script2 = detect_script(n1), detect_script(n2)
        rows.append({
            "source1_entity_id": s1_id, "missed_candidate_entity_id": candidate_id,
            "source1_business_name": name1, "missed_candidate_business_name": name2,
            "source1_business_address": addr1, "missed_candidate_business_address": addr2,
            "source1_country": country1, "missed_candidate_country": country2,
            "source1_normalized_name_length": len(n1), "missed_candidate_normalized_name_length": len(n2),
            "source1_name_script": script1, "missed_candidate_name_script": script2,
            "any_name_token_in_common": bool(t1 & t2), "country_match": country1 == country2,
            "source1_address_has_3plus_digit_token": d1, "missed_candidate_address_has_3plus_digit_token": d2,
            "either_address_missing_digit_token": not (d1 and d2),
            "either_name_very_short_lt10": len(n1) < 10 or len(n2) < 10,
            "either_name_non_latin": script1 in {"non-Latin", "mixed"} or script2 in {"non-Latin", "mixed"},
        })
    return pd.DataFrame(rows)


def print_summary(truth: Mapping[str, Sequence[str]], fully_covered: set[str], detail: pd.DataFrame) -> None:
    matched = {s1: ids for s1, ids in truth.items() if ids}
    missed_entities = set(detail.source1_entity_id) if not detail.empty else set()
    print(f"Matched S1 entities: {len(matched):,}")
    print(f"Fully covered: {len(fully_covered):,} ({len(fully_covered)/len(matched):.2%})" if matched else "Fully covered: 0")
    print(f"Missed at least one true match: {len(missed_entities):,} ({len(missed_entities)/len(matched):.2%})" if matched else "Missed at least one true match: 0")
    print(f"Missed true pairs: {len(detail):,}")
    if detail.empty:
        return
    signals = {
        "Either normalized name shorter than 10 characters": detail.either_name_very_short_lt10,
        "No normalized name token in common": ~detail.any_name_token_in_common,
        "At least one name uses non-Latin/mixed script": detail.either_name_non_latin,
        "At least one address lacks a 3+ digit token": detail.either_address_missing_digit_token,
    }
    for title, flags in signals.items():
        print(f"{title}: {flags.mean():.2%} ({int(flags.sum()):,}/{len(detail):,})")
    stats = detail.groupby("source1_country", dropna=False).agg(
        missed_pairs=("source1_entity_id", "size"),
        short=("either_name_very_short_lt10", "mean"),
        zero_token=("any_name_token_in_common", lambda values: (~values).mean()),
        non_latin=("either_name_non_latin", "mean"),
        missing_digits=("either_address_missing_digit_token", "mean"),
    ).sort_values("missed_pairs", ascending=False)
    print("By Source 1 country:")
    for country, row in stats.iterrows():
        print(f"  {country or '(empty)'}: {int(row.missed_pairs):,} pairs; short={row.short:.1%}, zero-token={row.zero_token:.1%}, non-Latin={row.non_latin:.1%}, missing-address-digits={row.missing_digits:.1%}")


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    project_root = root / "student_resource" if (root / "student_resource" / "dataset").is_dir() else root
    parser = argparse.ArgumentParser(description="Diagnose missed train blocking matches")
    parser.add_argument("--data-dir", type=Path, default=project_root / "dataset" / "train")
    parser.add_argument("--candidates", type=Path, default=project_root / "output" / "candidate_pairs.tsv")
    parser.add_argument("--output", type=Path, default=project_root / "output" / "blocking_misses_detail.csv")
    args = parser.parse_args()
    truth = load_truth(args.data_dir / "train_ground_truth.tsv")
    full, missed = compare_coverage(truth, args.candidates)
    details = build_details(missed, args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.output, index=False, encoding="utf-8")
    print_summary(truth, full, details)
    print(f"Detail CSV: {args.output} ({len(details):,} rows)")


if __name__ == "__main__":
    main()
