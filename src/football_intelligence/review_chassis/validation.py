from __future__ import annotations

from pathlib import Path
from typing import Any

from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.config import load_ui_config, ui_config_hash
from football_intelligence.review_chassis.hashing import sha256_file
from football_intelligence.review_chassis.manifest import load_manifest, manifest_hash
from football_intelligence.review_chassis.server import STATIC_ROOT


def _asset_path(evidence_root: Path, case_id: str, relative_path: str) -> Path:
    return (evidence_root / case_id / relative_path).resolve()


def validate_review_chassis_package(
    *,
    manifest_path: Path,
    ui_config_path: Path,
    evidence_root: Path,
    decisions_root: Path | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    ui_config = load_ui_config(ui_config_path)
    missing_assets = []
    hash_mismatches = []
    mp4_assets = []
    gif_count = 0
    image_sequence_count = 0
    for case in manifest.cases:
        if set(case.allowed_decisions) - {option.value for option in ui_config.decisions}:
            hash_mismatches.append(f"{case.case_id}:decision_not_in_ui_config")
        for asset in case.evidence_assets:
            path = _asset_path(evidence_root, case.case_id, asset.relative_path)
            if asset.media_type == "image/gif":
                gif_count += 1
            if asset.asset_type == "image_sequence":
                image_sequence_count += 1
            if asset.media_type.startswith("video/") or asset.relative_path.lower().endswith(".mp4"):
                mp4_assets.append(f"{case.case_id}:{asset.relative_path}")
            if not path.exists() or not path.is_file():
                missing_assets.append(str(path))
            elif sha256_file(path) != asset.sha256:
                hash_mismatches.append(f"{case.case_id}:{asset.relative_path}")
    index_text = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    app_text = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    css_exists = (STATIC_ROOT / "styles.css").exists()
    video_element_present = "<video" in index_text.lower() or 'createelement("video"' in app_text.lower()
    state_ready = True
    if decisions_root is not None:
        state_ready = (decisions_root / "review_decisions.json").exists() and (
            decisions_root / "review_decision_events.jsonl"
        ).exists()
    passed = (
        bool(manifest.cases)
        and not missing_assets
        and not hash_mismatches
        and not mp4_assets
        and not video_element_present
        and css_exists
        and state_ready
    )
    return {
        "schema_version": "football_intelligence.review_chassis.validation.v1",
        "passed": passed,
        "manifest_schema_version": manifest.schema_version,
        "ui_config_schema_version": ui_config.schema_version,
        "manifest_hash": manifest_hash(manifest),
        "ui_config_hash": ui_config_hash(ui_config),
        "review_case_count": len(manifest.cases),
        "missing_asset_count": len(missing_assets),
        "hash_mismatch_count": len(hash_mismatches),
        "mp4_asset_count": len(mp4_assets),
        "gif_asset_count": gif_count,
        "image_sequence_asset_count": image_sequence_count,
        "video_element_present": video_element_present,
        "canonical_chassis_source_paths": [
            str(STATIC_ROOT / "index.html"),
            str(STATIC_ROOT / "app.js"),
            str(STATIC_ROOT / "styles.css"),
        ],
        "decisions_state_ready": state_ready,
        **safety_payload(),
    }
