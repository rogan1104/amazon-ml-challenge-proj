"""Configuration for the retrieval experiment."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

from analysis.config import CACHE, REPORTS, ROOT, TEST, TRAIN

Split = Literal["train", "test"]
Target = Literal["S2", "S3"]


class ChannelName(str, Enum):
    BASELINE = "baseline"
    CHAR_NGRAM = "char_ngram"
    ADDRESS_NUMERIC = "address_numeric"
    TRANSLITERATION = "transliteration"


MODES: dict[str, list[ChannelName]] = {
    "baseline": [ChannelName.BASELINE],
    "char_ngram": [ChannelName.CHAR_NGRAM],
    "address_numeric": [ChannelName.ADDRESS_NUMERIC],
    "transliteration": [ChannelName.TRANSLITERATION],
    "all": list(ChannelName),
}


@dataclass
class CharNgramConfig:
    n: int = 3
    min_overlap: int = 2
    max_df: int = 50_000
    max_candidates_per_s1: int = 5_000


@dataclass
class AddressNumericConfig:
    min_digit_token_len: int = 3
    max_df: int = 20_000
    include_postal: bool = True
    include_house_number: bool = True
    max_candidates_per_s1: int = 3_000


@dataclass
class TransliterationConfig:
    min_key_len: int = 4
    use_ngram_fallback: bool = True
    ngram_n: int = 3
    min_ngram_overlap: int = 2
    max_candidates_per_s1: int = 3_000


@dataclass
class BaselineConfig:
    """Mirrors reference baseline E keys (read-only spec; does not import phase10)."""
    name_prefix_len: int = 5
    min_prefix_len: int = 3
    max_candidates_per_s1: int = 10_000
    cp_max_df: int | None = None


@dataclass
class RetrievalConfig:
    split: Split = "train"
    mode: str = "all"
    index_dir: Path = field(default_factory=lambda: CACHE / "retrieval_index")
    output_dir: Path = field(default_factory=lambda: REPORTS / "retrieval")
    batch_size: int = 50_000
    char_ngram: CharNgramConfig = field(default_factory=CharNgramConfig)
    address_numeric: AddressNumericConfig = field(default_factory=AddressNumericConfig)
    transliteration: TransliterationConfig = field(default_factory=TransliterationConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)

    def paths(self) -> dict[str, Path]:
        base = TRAIN if self.split == "train" else TEST
        return {"S1": base["S1"], "S2": base["S2"], "S3": base["S3"]}

    def active_channels(self) -> list[ChannelName]:
        if self.mode not in MODES:
            raise ValueError(f"Unknown mode {self.mode!r}. Choose from: {list(MODES)}")
        return MODES[self.mode]

    def output_tsv(self) -> Path:
        ch = self.mode.replace("+", "_")
        return self.output_dir / f"candidate_pairs_{self.split}_{ch}.tsv"

    def metrics_json(self) -> Path:
        return self.output_dir / f"retrieval_metrics_{self.split}_{self.mode}.json"
