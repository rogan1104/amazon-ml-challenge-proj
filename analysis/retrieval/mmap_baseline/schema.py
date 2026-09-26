"""On-disk layout and manifest for mmap CSR baseline indexes."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from analysis.retrieval.config import BaselineConfig

SCHEMA_VERSION = 1
FAMILIES = ("bn", "ba", "cp")
Target = Literal["S2", "S3"]
SPLIT_LITERAL = Literal["train", "test"]

# CSR file names under each target/family/
KEYS_IDX = "keys.idx"
KEYS_STR = "keys.str"
POSTS_IDX = "posts.idx"
POSTS_EID = "posts.eid"
ENTITY_IDS = "entity_ids.txt"


@dataclass
class FamilyStats:
    key_count: int = 0
    posting_count: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"key_count": self.key_count, "posting_count": self.posting_count}


@dataclass
class TargetStats:
    entity_count: int = 0
    families: dict[str, FamilyStats] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_count": self.entity_count,
            "families": {k: v.to_dict() for k, v in self.families.items()},
        }


@dataclass
class BaselineMmapManifest:
    schema_version: int
    split: str
    baseline_config: dict[str, Any]
    targets: dict[str, TargetStats]
    complete: bool = False

    @staticmethod
    def baseline_config_dict(cfg: BaselineConfig) -> dict[str, Any]:
        return {
            "name_prefix_len": cfg.name_prefix_len,
            "min_prefix_len": cfg.min_prefix_len,
            "max_candidates_per_s1": cfg.max_candidates_per_s1,
            "cp_max_df": cfg.cp_max_df,
        }

    @classmethod
    def matches_config(cls, manifest: BaselineMmapManifest, cfg: BaselineConfig) -> bool:
        return manifest.baseline_config == cls.baseline_config_dict(cfg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split": self.split,
            "baseline_config": self.baseline_config,
            "targets": {t: ts.to_dict() for t, ts in self.targets.items()},
            "complete": self.complete,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineMmapManifest:
        targets: dict[str, TargetStats] = {}
        for tname, tdata in data.get("targets", {}).items():
            fam: dict[str, FamilyStats] = {}
            for fname, fdata in tdata.get("families", {}).items():
                fam[fname] = FamilyStats(
                    key_count=int(fdata.get("key_count", 0)),
                    posting_count=int(fdata.get("posting_count", 0)),
                )
            targets[tname] = TargetStats(
                entity_count=int(tdata.get("entity_count", 0)),
                families=fam,
            )
        return cls(
            schema_version=int(data["schema_version"]),
            split=str(data["split"]),
            baseline_config=dict(data["baseline_config"]),
            targets=targets,
            complete=bool(data.get("complete", False)),
        )


def manifest_path(index_root: Path) -> Path:
    return index_root / "manifest.json"


def target_dir(index_root: Path, target: Target) -> Path:
    return index_root / target


def family_dir(index_root: Path, target: Target, family: str) -> Path:
    return target_dir(index_root, target) / family


def write_manifest(index_root: Path, manifest: BaselineMmapManifest) -> None:
    index_root.mkdir(parents=True, exist_ok=True)
    manifest_path(index_root).write_text(
        json.dumps(manifest.to_dict(), indent=2),
        encoding="utf-8",
    )


def read_manifest(index_root: Path) -> BaselineMmapManifest | None:
    path = manifest_path(index_root)
    if not path.is_file():
        return None
    return BaselineMmapManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def family_from_key(key: str) -> str | None:
    if key.startswith("bn:"):
        return "bn"
    if key.startswith("ba:"):
        return "ba"
    if key.startswith("cp:"):
        return "cp"
    return None


def filter_keys_by_family(keys: Iterable[str], family: str) -> list[str]:
    prefix = f"{family}:"
    return [k for k in keys if k.startswith(prefix)]
