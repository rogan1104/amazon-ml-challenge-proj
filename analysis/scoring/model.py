"""LightGBM pair scorer (binary match classifier)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analysis.scoring.features import FEATURE_NAMES, EntityFeatureSide, ScoringEntity, pair_features

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "n_estimators": 200,
}


class MissingLightGBMDependency(RuntimeError):
    """Raised when LightGBM is not installed or cannot load native libraries."""


def require_lightgbm():
    try:
        import lightgbm as lgb  # noqa: PLC0415
    except ImportError as exc:
        raise MissingLightGBMDependency(
            "LightGBM is not installed. Install with: pip install lightgbm"
        ) from exc
    except OSError as exc:
        raise MissingLightGBMDependency(
            "LightGBM is installed but failed to load (often missing libomp on macOS). "
            "Install OpenMP (e.g. brew install libomp) or use a Linux training environment."
        ) from exc
    return lgb


def is_training_pair(pair: dict[str, Any]) -> bool:
    """
    Training rows: retrieved negatives and retrieved GT positives only.

    Unretrieved GT positives are excluded from training.
    """
    if int(pair["label"]) == 0:
        return True
    return bool(pair.get("in_retrieval"))


def is_retrieved_scoring_pair(pair: dict[str, Any]) -> bool:
    """Validation/inference rows limited to retrieved candidate pairs."""
    return bool(pair.get("in_retrieval"))


@dataclass
class EntityStore:
    """entity_id -> ScoringEntity (+ optional feature cache)."""

    entities: dict[str, ScoringEntity] = field(default_factory=dict)
    _cache: dict[str, EntityFeatureSide] = field(default_factory=dict, repr=False)

    def get_side(self, entity_id: str) -> EntityFeatureSide:
        if entity_id not in self.entities:
            self.entities[entity_id] = ScoringEntity()
        if entity_id not in self._cache:
            self._cache[entity_id] = EntityFeatureSide.build(self.entities[entity_id])
        return self._cache[entity_id]

    def preload_side(self, entity_id: str) -> None:
        self.get_side(entity_id)


def feature_vector_for_pair(
    pair: dict[str, Any],
    store: EntityStore,
) -> list[float]:
    s1_side = store.get_side(pair["s1_id"])
    cand_side = store.get_side(pair["candidate_id"])
    feats = pair_features(
        s1_side,
        cand_side,
        candidate_source=str(pair["candidate_source"]),
        retrieval_evidence=pair.get("retrieval_evidence"),
    )
    return [feats[name] for name in FEATURE_NAMES]


def feature_matrix_from_pairs(
    pairs: Sequence[dict[str, Any]],
    store: EntityStore,
) -> tuple[np.ndarray, np.ndarray]:
    if not pairs:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty((0,), dtype=np.int8)
    rows = [feature_vector_for_pair(p, store) for p in pairs]
    y = np.array([int(p["label"]) for p in pairs], dtype=np.int8)
    return np.asarray(rows, dtype=np.float32), y


@dataclass
class PairLGBMModel:
    """Wrapper around a trained LightGBM booster."""

    booster: Any
    params: dict[str, Any]
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        lgb = require_lightgbm()
        if X.size == 0:
            return np.empty((0,), dtype=np.float64)
        return self.booster.predict(X)

    def feature_importance(self) -> dict[str, float]:
        imp = self.booster.feature_importance(importance_type="gain")
        total = float(imp.sum()) or 1.0
        return {
            name: round(float(v) / total, 6)
            for name, v in zip(self.feature_names, imp, strict=True)
        }

    def save(self, model_path: Path | str) -> None:
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))
        meta = {
            "feature_names": list(self.feature_names),
            "params": self.params,
            "feature_importance_gain": self.feature_importance(),
        }
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_path: Path | str) -> PairLGBMModel:
        lgb = require_lightgbm()
        path = Path(model_path)
        booster = lgb.Booster(model_file=str(path))
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        params = DEFAULT_LGBM_PARAMS.copy()
        names: tuple[str, ...] = FEATURE_NAMES
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            params = meta.get("params", params)
            fn = meta.get("feature_names")
            if fn:
                names = tuple(fn)
        return cls(booster=booster, params=params, feature_names=names)


def train_lightgbm_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    params: dict[str, Any] | None = None,
    scale_pos_weight: float | None = None,
) -> PairLGBMModel:
    lgb = require_lightgbm()
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y length mismatch")
    if X.shape[0] == 0:
        raise ValueError("Cannot train on zero rows")

    p = {**DEFAULT_LGBM_PARAMS, **(params or {})}
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if scale_pos_weight is None and n_pos > 0:
        scale_pos_weight = n_neg / n_pos
    if scale_pos_weight is not None:
        p["scale_pos_weight"] = scale_pos_weight

    n_estimators = int(p.pop("n_estimators", 200))
    train_set = lgb.Dataset(X, label=y, feature_name=list(FEATURE_NAMES))
    booster = lgb.train(p, train_set, num_boost_round=n_estimators)
    return PairLGBMModel(booster=booster, params={**p, "n_estimators": n_estimators})
