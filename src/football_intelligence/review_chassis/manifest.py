from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_intelligence.review_chassis.hashing import stable_hash
from football_intelligence.review_chassis.models import GenericReviewManifest


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
    return stable_hash(payload)
