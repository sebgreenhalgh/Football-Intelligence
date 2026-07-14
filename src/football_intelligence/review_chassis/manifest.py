from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_intelligence.review_chassis.hashing import stable_hash
from football_intelligence.review_chassis.models import GENERIC_MANIFEST_SCHEMA_VERSION_V1, GenericReviewManifest


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def load_manifest(path: Path) -> GenericReviewManifest:
    return GenericReviewManifest.model_validate(read_json(path))


def manifest_hash(manifest: GenericReviewManifest) -> str:
    payload = manifest.model_dump(mode="json")
    payload["manifest_hash"] = ""
    if payload.get("schema_version") == GENERIC_MANIFEST_SCHEMA_VERSION_V1:
        v2_asset_keys = {
            "visibility_policy",
            "reveal_group_id",
            "reveal_button_label",
            "reveal_requires_existing_decision",
            "record_reveal_event",
            "visible_after_decision_values",
            "visible_after_completion",
        }
        for case in payload.get("cases", []):
            for asset in case.get("evidence_assets", []):
                for key in v2_asset_keys:
                    asset.pop(key, None)
    return stable_hash(payload)
