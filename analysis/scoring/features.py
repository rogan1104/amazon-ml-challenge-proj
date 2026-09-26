"""
Numeric pair features for gradient-boosted matchers (LightGBM / XGBoost).

Uses analysis.normalize for text/token/n-gram logic and analysis.retrieval.keys
for baseline-aligned name/address normalization (CP / exact flags).
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any, Mapping

from analysis.normalize import (
    char_ngrams,
    extract_digits,
    extract_tokens,
    jaccard,
    normalize_address,
    normalize_name,
)
from analysis.retrieval.keys import sql_style_norm_addr, sql_style_norm_name
from analysis.scoring.pairs import cp_key, normalize_country_label
from rapidfuzz.fuzz import ratio, token_set_ratio, token_sort_ratio, WRatio

CHAR_NGRAM_N = 3


# Stable column order for training pipelines.
FEATURE_NAMES: tuple[str, ...] = (
    # Name
    "name_exact_sql_norm",
    "name_fuzz_ratio",
    "name_fuzz_wratio",
    "name_fuzz_token_set_ratio",
    "name_fuzz_token_sort_ratio",
    "name_token_jaccard",
    "name_char_ngram_jaccard",
    "name_len_diff",
    "name_len_ratio",
    # Address
    "addr_exact_sql_norm",
    "addr_fuzz_ratio",
    "addr_fuzz_token_set_ratio",
    "addr_token_jaccard",
    "addr_numeric_overlap_count",
    "addr_numeric_jaccard",
    "addr_len_diff",
    "addr_len_ratio",
    # Context
    "ctx_country_match",
    "ctx_same_cp_key",
    "ctx_same_sql_norm_name",
    "ctx_same_sql_norm_addr",
    "ctx_candidate_is_s2",
    "ctx_candidate_is_s3",
    "ctx_retrieval_evidence_count",
)


@dataclass(frozen=True)
class ScoringEntity:
    """Side payload for feature computation (S1 or S2/S3 candidate)."""

    business_name: str = ""
    business_address: str = ""
    country: str = ""


@dataclass
class EntityFeatureSide:
    """Precomputed fields for one entity (reuse across many pair rows)."""

    norm_name: str
    sql_norm_name: str
    norm_addr: str
    sql_norm_addr: str
    name_tokens: frozenset[str]
    addr_tokens: frozenset[str]
    name_char_ngrams: frozenset[str]
    addr_digits: frozenset[str]
    country_norm: str
    cp_key: str | None

    @classmethod
    def build(cls, entity: ScoringEntity) -> EntityFeatureSide:
        nn = normalize_name(entity.business_name)
        na = normalize_address(entity.business_address)
        snn = sql_style_norm_name(entity.business_name)
        sna = sql_style_norm_addr(entity.business_address)
        name_compact = nn.replace(" ", "")
        return cls(
            norm_name=nn,
            sql_norm_name=snn,
            norm_addr=na,
            sql_norm_addr=sna,
            name_tokens=frozenset(extract_tokens(entity.business_name)),
            addr_tokens=frozenset(extract_tokens(entity.business_address)),
            name_char_ngrams=frozenset(
                char_ngrams(name_compact, n=CHAR_NGRAM_N) if name_compact else set()
            ),
            addr_digits=frozenset(extract_digits(entity.business_address)),
            country_norm=normalize_country_label(entity.country),
            cp_key=cp_key(entity.country, entity.business_name),
        )


def _len_diff_ratio(a: str, b: str) -> tuple[float, float]:
    la, lb = len(a), len(b)
    diff = float(abs(la - lb))
    mx = max(la, lb)
    ratio = (min(la, lb) / mx) if mx > 0 else 1.0
    return diff, ratio


def _fuzz01(a: str, b: str, fn) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return fn(a, b) / 100.0


def _retrieval_evidence_count(evidence: Mapping[str, Any] | None) -> float:
    if not evidence:
        return 0.0
    n = 0
    for v in evidence.values():
        if v is True:
            n += 1
        elif v is False or v is None:
            continue
        elif isinstance(v, (int, float)) and v != 0:
            n += 1
        elif isinstance(v, str) and v.strip():
            n += 1
    return float(n)


def pair_features(
    s1: EntityFeatureSide,
    candidate: EntityFeatureSide,
    *,
    candidate_source: str,
    retrieval_evidence: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """
    Fixed numeric feature dict for one S1 ↔ candidate pair.

    All values are floats in [0, 1] except overlap counts and length diffs (non-negative).
    """
    name_len_diff, name_len_ratio = _len_diff_ratio(s1.norm_name, candidate.norm_name)
    addr_len_diff, addr_len_ratio = _len_diff_ratio(s1.norm_addr, candidate.norm_addr)
    num_overlap = len(s1.addr_digits & candidate.addr_digits)

    src = candidate_source.strip().upper()
    is_s2 = 1.0 if src == "S2" else 0.0
    is_s3 = 1.0 if src == "S3" else 0.0

    feats: dict[str, float] = {
        "name_exact_sql_norm": 1.0 if s1.sql_norm_name and s1.sql_norm_name == candidate.sql_norm_name else 0.0,
        "name_fuzz_ratio": _fuzz01(s1.norm_name, candidate.norm_name, ratio),
        "name_fuzz_wratio": _fuzz01(s1.norm_name, candidate.norm_name, WRatio),
        "name_fuzz_token_set_ratio": _fuzz01(s1.norm_name, candidate.norm_name, token_set_ratio),
        "name_fuzz_token_sort_ratio": _fuzz01(s1.norm_name, candidate.norm_name, token_sort_ratio),
        "name_token_jaccard": jaccard(set(s1.name_tokens), set(candidate.name_tokens)),
        "name_char_ngram_jaccard": jaccard(
            set(s1.name_char_ngrams), set(candidate.name_char_ngrams)
        ),
        "name_len_diff": name_len_diff,
        "name_len_ratio": name_len_ratio,
        "addr_exact_sql_norm": 1.0 if s1.sql_norm_addr and s1.sql_norm_addr == candidate.sql_norm_addr else 0.0,
        "addr_fuzz_ratio": _fuzz01(s1.norm_addr, candidate.norm_addr, ratio),
        "addr_fuzz_token_set_ratio": _fuzz01(s1.norm_addr, candidate.norm_addr, token_set_ratio),
        "addr_token_jaccard": jaccard(set(s1.addr_tokens), set(candidate.addr_tokens)),
        "addr_numeric_overlap_count": float(num_overlap),
        "addr_numeric_jaccard": jaccard(set(s1.addr_digits), set(candidate.addr_digits)),
        "addr_len_diff": addr_len_diff,
        "addr_len_ratio": addr_len_ratio,
        "ctx_country_match": 1.0
        if s1.country_norm != "UNKNOWN"
        and s1.country_norm == candidate.country_norm
        else 0.0,
        "ctx_same_cp_key": 1.0
        if s1.cp_key and s1.cp_key == candidate.cp_key
        else 0.0,
        "ctx_same_sql_norm_name": 1.0
        if s1.sql_norm_name and s1.sql_norm_name == candidate.sql_norm_name
        else 0.0,
        "ctx_same_sql_norm_addr": 1.0
        if s1.sql_norm_addr and s1.sql_norm_addr == candidate.sql_norm_addr
        else 0.0,
        "ctx_candidate_is_s2": is_s2,
        "ctx_candidate_is_s3": is_s3,
        "ctx_retrieval_evidence_count": _retrieval_evidence_count(retrieval_evidence),
    }
    return {k: float(feats[k]) for k in FEATURE_NAMES}


def pair_features_from_entities(
    s1_entity: ScoringEntity,
    candidate_entity: ScoringEntity,
    *,
    candidate_source: str,
    retrieval_evidence: Mapping[str, Any] | None = None,
    s1_cache: EntityFeatureSide | None = None,
    candidate_cache: EntityFeatureSide | None = None,
) -> dict[str, float]:
    """Convenience wrapper with optional pre-built sides."""
    s1_side = s1_cache or EntityFeatureSide.build(s1_entity)
    cand_side = candidate_cache or EntityFeatureSide.build(candidate_entity)
    return pair_features(
        s1_side,
        cand_side,
        candidate_source=candidate_source,
        retrieval_evidence=retrieval_evidence,
    )


class TestPairFeatures(unittest.TestCase):
    def test_exact_match(self) -> None:
        ent = ScoringEntity(
            business_name="Acme Corp & Co.",
            business_address="123 Main St, Boston",
            country="US",
        )
        feats = pair_features_from_entities(ent, ent, candidate_source="S2")
        self.assertEqual(feats["name_exact_sql_norm"], 1.0)
        self.assertEqual(feats["addr_exact_sql_norm"], 1.0)
        self.assertGreaterEqual(feats["name_fuzz_ratio"], 0.99)
        self.assertEqual(feats["ctx_country_match"], 1.0)
        self.assertEqual(feats["ctx_candidate_is_s2"], 1.0)
        self.assertEqual(feats["ctx_candidate_is_s3"], 0.0)

    def test_clear_non_match(self) -> None:
        s1 = ScoringEntity("Alpha LLC", "1 First Ave", "US")
        cand = ScoringEntity("Totally Different GmbH", "99 Other Road", "IN")
        feats = pair_features_from_entities(s1, cand, candidate_source="S3")
        self.assertEqual(feats["name_exact_sql_norm"], 0.0)
        self.assertLess(feats["name_fuzz_ratio"], 0.5)
        self.assertEqual(feats["ctx_country_match"], 0.0)
        self.assertEqual(feats["ctx_candidate_is_s3"], 1.0)

    def test_typo_noisy_name(self) -> None:
        s1 = ScoringEntity("Invictus & Co", "addr", "US")
        cand = ScoringEntity("Invictus and Co", "addr", "US")
        feats = pair_features_from_entities(s1, cand, candidate_source="S2")
        self.assertGreater(feats["name_fuzz_ratio"], 0.7)
        self.assertGreaterEqual(feats["name_token_jaccard"], 0.5)

    def test_address_number_difference(self) -> None:
        s1 = ScoringEntity("Shop", "123 Main Street", "US")
        cand = ScoringEntity("Shop", "124 Main Street", "US")
        feats = pair_features_from_entities(s1, cand, candidate_source="S2")
        self.assertEqual(feats["addr_numeric_overlap_count"], 0.0)
        self.assertGreater(feats["addr_fuzz_ratio"], 0.8)

    def test_empty_name_address(self) -> None:
        s1 = ScoringEntity("", "", "US")
        cand = ScoringEntity("", "", "US")
        feats = pair_features_from_entities(s1, cand, candidate_source="S2")
        self.assertEqual(feats["name_fuzz_ratio"], 1.0)
        self.assertEqual(feats["addr_fuzz_ratio"], 1.0)
        self.assertEqual(feats["name_len_diff"], 0.0)

    def test_source_one_hot(self) -> None:
        ent = ScoringEntity("X", "Y", "US")
        s2 = pair_features_from_entities(ent, ent, candidate_source="S2")
        s3 = pair_features_from_entities(ent, ent, candidate_source="S3")
        self.assertEqual((s2["ctx_candidate_is_s2"], s2["ctx_candidate_is_s3"]), (1.0, 0.0))
        self.assertEqual((s3["ctx_candidate_is_s2"], s3["ctx_candidate_is_s3"]), (0.0, 1.0))

    def test_feature_names_stable(self) -> None:
        ent = ScoringEntity("A", "B", "US")
        feats = pair_features_from_entities(ent, ent, candidate_source="S2")
        self.assertEqual(list(feats.keys()), list(FEATURE_NAMES))


if __name__ == "__main__":
    unittest.main()
