"""Retrieval channel implementations (each independently switchable)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import Iterable

from analysis.retrieval.config import (
    AddressNumericConfig,
    BaselineConfig,
    ChannelName,
    CharNgramConfig,
    RetrievalConfig,
    Target,
    TransliterationConfig,
)
from analysis.retrieval.index import InvertedIndex, select_top_candidates
from analysis.retrieval.keys import (
    address_numeric_keys,
    baseline_keys,
    char_ngram_keys,
    transliteration_keys,
)


class RetrievalChannel(ABC):
    name: ChannelName

    @abstractmethod
    def extract_index_keys(self, row: dict) -> Iterable[str]:
        """Keys to index for a source record (S2 or S3)."""

    @abstractmethod
    def extract_query_keys(self, row: dict) -> Iterable[str]:
        """Keys to query for an S1 record."""

    @abstractmethod
    def retrieve(
        self,
        index: InvertedIndex,
        target: Target,
        query_row: dict,
    ) -> set[str]:
        """Return candidate entity_ids from target source for one S1 row."""


class BaselineChannel(RetrievalChannel):
    """Reference baseline E keys (norm name, norm addr, country+prefix5)."""

    name = ChannelName.BASELINE

    def __init__(self, cfg: BaselineConfig):
        self.cfg = cfg

    def extract_index_keys(self, row: dict) -> Iterable[str]:
        return baseline_keys(
            row.get("business_name", ""),
            row.get("business_address", ""),
            row.get("country", ""),
            prefix_len=self.cfg.name_prefix_len,
            min_prefix=self.cfg.min_prefix_len,
        )

    extract_query_keys = extract_index_keys

    def retrieve(self, index: InvertedIndex, target: Target, query_row: dict) -> set[str]:
        keys = list(self.extract_query_keys(query_row))
        hits = index.lookup_keys_exact(self.name, target, keys)
        if len(hits) > self.cfg.max_candidates_per_s1:
            return set(list(hits)[: self.cfg.max_candidates_per_s1])
        return hits


class CharNgramChannel(RetrievalChannel):
    """Character n-gram overlap retrieval on normalized business names."""

    name = ChannelName.CHAR_NGRAM

    def __init__(self, cfg: CharNgramConfig):
        self.cfg = cfg

    def extract_index_keys(self, row: dict) -> Iterable[str]:
        return char_ngram_keys(row.get("business_name", ""), n=self.cfg.n)

    extract_query_keys = extract_index_keys

    def retrieve(self, index: InvertedIndex, target: Target, query_row: dict) -> set[str]:
        keys = list(self.extract_query_keys(query_row))
        scores = index.lookup_keys(
            self.name, target, keys, max_df=self.cfg.max_df,
        )
        return select_top_candidates(
            scores,
            min_overlap=self.cfg.min_overlap,
            max_candidates=self.cfg.max_candidates_per_s1,
        )


class AddressNumericChannel(RetrievalChannel):
    """House number, postal/PIN/ZIP, and significant numeric token retrieval."""

    name = ChannelName.ADDRESS_NUMERIC

    def __init__(self, cfg: AddressNumericConfig):
        self.cfg = cfg

    def extract_index_keys(self, row: dict) -> Iterable[str]:
        return address_numeric_keys(
            row.get("business_address", ""),
            row.get("country", ""),
            include_postal=self.cfg.include_postal,
            include_house=self.cfg.include_house_number,
            min_digit_len=self.cfg.min_digit_token_len,
        )

    extract_query_keys = extract_index_keys

    def retrieve(self, index: InvertedIndex, target: Target, query_row: dict) -> set[str]:
        keys = list(self.extract_query_keys(query_row))
        scores = index.lookup_keys(
            self.name, target, keys, max_df=self.cfg.max_df,
        )
        return select_top_candidates(
            scores,
            min_overlap=1,
            max_candidates=self.cfg.max_candidates_per_s1,
        )


class TransliterationChannel(RetrievalChannel):
    """Indic-script → Latin transliteration keys (+ optional translit n-grams)."""

    name = ChannelName.TRANSLITERATION

    def __init__(self, cfg: TransliterationConfig):
        self.cfg = cfg

    def extract_index_keys(self, row: dict) -> Iterable[str]:
        keys = set(transliteration_keys(row.get("business_name", ""), n=self.cfg.ngram_n))
        return keys

    extract_query_keys = extract_index_keys

    def retrieve(self, index: InvertedIndex, target: Target, query_row: dict) -> set[str]:
        keys = list(self.extract_query_keys(query_row))
        if not keys:
            return set()

        exact = {k for k in keys if k.startswith("tl:")}
        ngrams = {k for k in keys if k.startswith("tng:")}

        found = index.lookup_keys_exact(self.name, target, exact)

        if self.cfg.use_ngram_fallback and ngrams:
            scores = index.lookup_keys(self.name, target, ngrams)
            found |= select_top_candidates(
                scores,
                min_overlap=self.cfg.min_ngram_overlap,
                max_candidates=self.cfg.max_candidates_per_s1,
            )

        if len(found) > self.cfg.max_candidates_per_s1:
            return set(list(found)[: self.cfg.max_candidates_per_s1])
        return found


def build_channel_registry(cfg: RetrievalConfig) -> dict[ChannelName, RetrievalChannel]:
    return {
        ChannelName.BASELINE: BaselineChannel(cfg.baseline),
        ChannelName.CHAR_NGRAM: CharNgramChannel(cfg.char_ngram),
        ChannelName.ADDRESS_NUMERIC: AddressNumericChannel(cfg.address_numeric),
        ChannelName.TRANSLITERATION: TransliterationChannel(cfg.transliteration),
    }


def active_channels(cfg: RetrievalConfig) -> list[RetrievalChannel]:
    registry = build_channel_registry(cfg)
    return [registry[ch] for ch in cfg.active_channels()]
