import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import joblib
from sklearn import set_config

from src.config import (
    BEST_PARAMS_PATH,
    CORE_MODEL_PATH,
    PIPELINE_PATH,
    RAW_FEATURES_PATH,
    RAW_SURVIVORS_PATH,
)


set_config(transform_output="pandas")


METADATA_PATH = os.path.join(os.path.dirname(PIPELINE_PATH), "model_metadata.json")


@dataclass(frozen=True)
class ModelAssets:
    full_pipeline: Any
    core_model: Any
    raw_feature_names: list[str]
    raw_survivors: list[str]
    best_params: dict[str, Any] | None
    metadata: dict[str, Any] | None


def _load_json_if_exists(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_model_assets() -> ModelAssets:
    full_pipeline = joblib.load(PIPELINE_PATH)
    core_model = joblib.load(CORE_MODEL_PATH)
    raw_feature_names = joblib.load(RAW_FEATURES_PATH)
    raw_survivors = joblib.load(RAW_SURVIVORS_PATH)

    if hasattr(full_pipeline, "set_output"):
        full_pipeline.set_output(transform="pandas")

    return ModelAssets(
        full_pipeline=full_pipeline,
        core_model=core_model,
        raw_feature_names=list(raw_feature_names),
        raw_survivors=list(raw_survivors),
        best_params=_load_json_if_exists(BEST_PARAMS_PATH),
        metadata=_load_json_if_exists(METADATA_PATH),
    )
