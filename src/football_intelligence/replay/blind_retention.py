from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.core.fingerprints import sha256_file

PROTECTED_CATEGORIES = {
    "source_video_manifest",
    "selection_artifact",
    "extraction_recipe",
    "extraction_manifest",
    "canonical_source_frame",
    "control_extraction_inventory",
    "frozen_configuration",
    "input_closure",
    "pipeline_run_manifest",
    "review_candidate_rows",
    "review_evidence",
    "blank_decision_template",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def retained_file(path: Path, category: str) -> dict[str, Any]:
    if category not in PROTECTED_CATEGORIES:
        raise ValueError(f"unknown retention category: {category}")
    return {
        "path": str(path),
        "category": category,
        "byte_size": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        "may_be_classified_expendable": False,
    }


def build_retention_manifest(stage_root: Path) -> dict[str, Any]:
    retention_root = stage_root / "retention"
    entries = [
        retained_file(stage_root / "source/source_video_manifest.json", "source_video_manifest"),
        retained_file(stage_root / "selection/blind_window_selection.json", "selection_artifact"),
        retained_file(stage_root / "selection/blind_window_selection_seal.json", "selection_artifact"),
        retained_file(stage_root / "frames/extraction_a/frame_manifest.json", "extraction_manifest"),
        retained_file(stage_root / "frames/extraction_b/frame_manifest.json", "control_extraction_inventory"),
        retained_file(stage_root / "pipeline/frozen_configuration_manifest.json", "frozen_configuration"),
        retained_file(stage_root / "pipeline/input_closure.json", "input_closure"),
        retained_file(stage_root / "review/blind_review_candidate_rows.json", "review_candidate_rows"),
        retained_file(stage_root / "review/blind_review_decision_template.json", "blank_decision_template"),
    ]
    manifest = {
        "schema_version": "m5.blind_window.retention_manifest.v1",
        "created_at": utc_now(),
        "protected_categories": sorted(PROTECTED_CATEGORIES),
        "entries": entries,
        "cleanup_classification_rule": (
            "All listed entries and descendants of canonical/control frame roots are preserve-only."
        ),
    }
    write_json(retention_root / "blind_window_retention_manifest.json", manifest)
    write_json(
        retention_root / "blind_window_preservation_inventory.json",
        {
            "schema_version": "m5.blind_window.preservation_inventory.v1",
            "created_at": utc_now(),
            "preserve_count": len(entries),
            "entries": entries,
        },
    )
    (retention_root / "DO_NOT_DELETE.md").write_text(
        "# Do Not Delete M5.3 Blind Window Evidence\n\n"
        "Selection seals, source manifests, canonical frames, control frames, run manifests, review candidates, "
        "review evidence, and blank decision templates are retained evidence and must not be classified as "
        "expendable.\n",
        encoding="utf-8",
    )
    return manifest


def cleanup_classification_for_retained_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    protected_markers = [
        "05_blind_second_window/source/",
        "05_blind_second_window/selection/",
        "05_blind_second_window/frames/extraction_a/",
        "05_blind_second_window/frames/extraction_b/frame_manifest.json",
        "05_blind_second_window/pipeline/",
        "05_blind_second_window/review/blind_review_candidate_rows.json",
        "05_blind_second_window/review/evidence/",
        "05_blind_second_window/review/blind_review_decision_template.json",
    ]
    return "preserve" if any(marker in normalized for marker in protected_markers) else "manual_review_required"
