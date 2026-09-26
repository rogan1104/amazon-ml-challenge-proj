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
from analysis.retrieval.index import InvertedIndex, apply_candidate_cap, select_top_candidates
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

    def retrieve_batch(
        self,
        index: InvertedIndex,
        target: Target,
        query_rows: list[dict],
    ) -> dict[str, set[str]]:
        """Default: per-row retrieve (channels override with batched SQLite)."""
        out: dict[str, set[str]] = {}
        for row in query_rows:
            s1_id = row.get("entity_id", "").strip()
            if not s1_id:
                continue
            out[s1_id] = self.retrieve(index, target, row)
        return out


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
        if self.cfg.cp_max_df is not None:
            hits = index.lookup_keys_exact_with_cp_max_df(
                self.name, target, keys, self.cfg.cp_max_df,
            )
        else:
            hits = index.lookup_keys_exact(self.name, target, keys)
        return apply_candidate_cap(hits, self.cfg.max_candidates_per_s1)

    def retrieve_batch(
        self,
        index: InvertedIndex,
        target: Target,
        query_rows: list[dict],
    ) -> dict[str, set[str]]:
        queries: list[tuple[str, list[str]]] = []
        for row in query_rows:
            s1_id = row.get("entity_id", "").strip()
            if not s1_id:
                continue
            queries.append((s1_id, list(self.extract_query_keys(row))))

        if self.cfg.cp_max_df is not None:
            raw = index.lookup_keys_exact_batch_with_cp_max_df(
                self.name, target, queries, self.cfg.cp_max_df,
            )
        else:
            raw = index.lookup_keys_exact_batch(self.name, target, queries)
        cap = self.cfg.max_candidates_per_s1
        return {qid: apply_candidate_cap(hits, cap) for qid, hits in raw.items()}


class CharNgramChannel(RetrievalChannel):
    """Character n-gram overlap retrieval on normalized business names."""

    name = ChannelName.CHAR_NGRAM

    def __init__(self, cfg: CharNgramConfig):
        self.cfg = cfg

    def extract_index_keys(self, row: dict) -> Iterable[str]:
        return char_ngram_keys(row.get("business_name", ""), n=self.cfg.n)

    extract_query_keys = extract_index_keys

    def _select(self, scores: Counter) -> set[str]:
        return select_top_candidates(
            scores,
            min_overlap=self.cfg.min_overlap,
            max_candidates=self.cfg.max_candidates_per_s1,
        )

    def retrieve(self, index: InvertedIndex, target: Target, query_row: dict) -> set[str]:
        keys = list(self.extract_query_keys(query_row))
        scores = index.lookup_keys(
            self.name, target, keys, max_df=self.cfg.max_df,
        )
        return self._select(scores)

    def retrieve_batch(
        self,
        index: InvertedIndex,
        target: Target,
        query_rows: list[dict],
    ) -> dict[str, set[str]]:
        queries: list[tuple[str, list[str]]] = []
        for row in query_rows:
            s1_id = row.get("entity_id", "").strip()
            if not s1_id:
                continue
            queries.append((s1_id, list(self.extract_query_keys(row))))

        scored = index.lookup_keys_batch(
            self.name, target, queries, max_df=self.cfg.max_df,
        )
        return {qid: self._select(scores) for qid, scores in scored.items()}


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

    def _select(self, scores: Counter) -> set[str]:
        return select_top_candidates(
            scores,
            min_overlap=1,
            max_candidates=self.cfg.max_candidates_per_s1,
        )

    def retrieve(self, index: InvertedIndex, target: Target, query_row: dict) -> set[str]:
        keys = list(self.extract_query_keys(query_row))
        scores = index.lookup_keys(
            self.name, target, keys, max_df=self.cfg.max_df,
        )
        return self._select(scores)

    def retrieve_batch(
        self,
        index: InvertedIndex,
        target: Target,
        query_rows: list[dict],
    ) -> dict[str, set[str]]:
        queries: list[tuple[str, list[str]]] = []
        for row in query_rows:
            s1_id = row.get("entity_id", "").strip()
            if not s1_id:
                continue
            queries.append((s1_id, list(self.extract_query_keys(row))))

        scored = index.lookup_keys_batch(
            self.name, target, queries, max_df=self.cfg.max_df,
        )
        return {qid: self._select(scores) for qid, scores in scored.items()}


class TransliterationChannel(RetrievalChannel):
    """Indic-script → Latin transliteration keys (+ optional translit n-grams)."""

    name = ChannelName.TRANSLITERATION

    def __init__(self, cfg: TransliterationConfig):
        self.cfg = cfg

    def extract_index_keys(self, row: dict) -> Iterable[str]:
        keys = set(transliteration_keys(row.get("business_name", ""), n=self.cfg.ngram_n))
        return keys

    extract_query_keys = extract_index_keys

    def _merge_hits(self, exact: set[str], ngram_scores: Counter) -> set[str]:
        found = set(exact)
        if self.cfg.use_ngram_fallback and ngram_scores:
            found |= select_top_candidates(
                ngram_scores,
                min_overlap=self.cfg.min_ngram_overlap,
                max_candidates=self.cfg.max_candidates_per_s1,
            )
        return apply_candidate_cap(found, self.cfg.max_candidates_per_s1)

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

        return apply_candidate_cap(found, self.cfg.max_candidates_per_s1)

    def retrieve_batch(
        self,
        index: InvertedIndex,
        target: Target,
        query_rows: list[dict],
    ) -> dict[str, set[str]]:
        exact_queries: list[tuple[str, list[str]]] = []
        ngram_queries: list[tuple[str, list[str]]] = []
        empty: dict[str, set[str]] = {}

        for row in query_rows:
            s1_id = row.get("entity_id", "").strip()
            if not s1_id:
                continue
            keys = list(self.extract_query_keys(row))
            if not keys:
                empty[s1_id] = set()
                continue
            exact_keys = [k for k in keys if k.startswith("tl:")]
            ngram_keys = [k for k in keys if k.startswith("tng:")]
            if exact_keys:
                exact_queries.append((s1_id, exact_keys))
            if ngram_keys:
                ngram_queries.append((s1_id, ngram_keys))

        exact_hits = index.lookup_keys_exact_batch(self.name, target, exact_queries)
        ngram_scores = (
            index.lookup_keys_batch(self.name, target, ngram_queries)
            if self.cfg.use_ngram_fallback
            else {}
        )

        out: dict[str, set[str]] = dict(empty)
        all_ids = {qid for qid, _ in exact_queries} | {qid for qid, _ in ngram_queries}
        for s1_id in all_ids:
            out[s1_id] = self._merge_hits(
                exact_hits.get(s1_id, set()),
                ngram_scores.get(s1_id, Counter()),
            )
        return out


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
