from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_intelligence.review_chassis.hashing import stable_hash
from football_intelligence.review_chassis.models import GENERIC_UI_CONFIG_SCHEMA_VERSION_V1, ReviewUIConfig


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def load_ui_config(path: Path) -> ReviewUIConfig:
    return ReviewUIConfig.model_validate(read_json(path))


def ui_config_hash(ui_config: ReviewUIConfig) -> str:
    payload = ui_config.model_dump(mode="json")
    if payload.get("schema_version") == GENERIC_UI_CONFIG_SCHEMA_VERSION_V1:
        payload.pop("comparison_panels", None)
        payload.pop("decision_to_output_mapping", None)
    return stable_hash(payload)
