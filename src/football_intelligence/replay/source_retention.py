from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.core.fingerprints import semantic_hash, sha256_file
from football_intelligence.replay.blind_window_selection import probe_video_metadata


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_source_retention_artifacts(
    *,
    stage_root: Path,
    source_video: Path,
    selection: dict[str, Any],
    canonical_manifest: Path | None = None,
    control_manifest: Path | None = None,
    repo_commit: str | None = None,
    dirty_state: bool | None = None,
) -> dict[str, Any]:
    metadata = probe_video_metadata(source_video)
    source_sha = sha256_file(source_video)
    source_root = stage_root / "source"
    canonical_hash = read_json(canonical_manifest)["ordered_frame_inventory_hash"] if canonical_manifest else None
    control_hash = read_json(control_manifest)["ordered_frame_inventory_hash"] if control_manifest else None
    manifest = {
        "schema_version": "m5.blind_window.source_video_manifest.v1",
        "created_at": utc_now(),
        "source_video_uri": str(source_video),
        "source_video_sha256": source_sha,
        "byte_size": source_video.stat().st_size,
        "video_codec_metadata": metadata.as_dict(),
        "dimensions": {"width": metadata.width, "height": metadata.height},
        "fps": metadata.fps,
        "frame_count": metadata.frame_count,
        "duration_seconds": metadata.duration_seconds,
        "selected_source_interval": {
            "start_seconds": selection["selected_start_seconds"],
            "end_seconds": selection["selected_end_seconds"],
            "duration_seconds": selection["duration_seconds"],
        },
    }
    manifest["manifest_hash"] = semantic_hash(manifest)
    write_json(source_root / "source_video_manifest.json", manifest)

    contract = {
        "schema_version": "m5.blind_window.artifact_retention_contract.v1",
        "created_at": utc_now(),
        "source_video_uri": str(source_video),
        "source_video_sha256": source_sha,
        "byte_size": source_video.stat().st_size,
        "video_codec_metadata": metadata.as_dict(),
        "dimensions": {"width": metadata.width, "height": metadata.height},
        "fps": metadata.fps,
        "frame_count": metadata.frame_count,
        "duration_seconds": metadata.duration_seconds,
        "selected_source_interval": manifest["selected_source_interval"],
        "extraction_recipe": {
            "output_fps": 10,
            "output_width": 2730,
            "output_height": 720,
            "duration_seconds": 60,
            "jpeg_quality": 95,
            "source_frame_offset": "round(k * 25 / 10)",
            "interpolation": "OpenCV INTER_AREA",
            "format": "JPEG",
            "no_crop": True,
            "no_padding": True,
            "no_boxes": True,
            "no_annotations": True,
            "no_overlays": True,
        },
        "extraction_software_versions": {"python": platform.python_version()},
        "canonical_frame_set_uri": str(canonical_manifest.parent) if canonical_manifest else None,
        "control_frame_set_uri": str(control_manifest.parent) if control_manifest else None,
        "canonical_frame_inventory_hash": canonical_hash,
        "control_frame_inventory_hash": control_hash,
        "extraction_command_or_entry_point": "fi-pipeline blind-window extract",
        "git_commit": repo_commit,
        "dirty_state": dirty_state,
        "backup_status": "local_primary_only_backup_not_confirmed",
        "deletion_prohibition": True,
        "retention_reason": (
            "Source media and canonical frames are the reproducibility root for M5.3 and may not be deleted "
            "merely because downstream overlays exist."
        ),
        "downstream_artifact_references": [
            "selection/blind_window_selection_seal.json",
            "frames/extraction_a/frame_manifest.json",
            "frames/extraction_b/frame_manifest.json",
            "validation/frame_extraction_repeatability.json",
            "review/blind_review_candidate_rows.json",
        ],
    }
    contract["manifest_hash"] = semantic_hash(contract)
    write_json(source_root / "artifact_retention_contract.json", contract)
    (source_root / "DO_NOT_DELETE_SOURCE_EVIDENCE.md").write_text(
        "# Do Not Delete Source Evidence\n\n"
        "The source panorama video, sealed selection artifacts, canonical 600-frame extraction, and control "
        "extraction are retained evidence. They must not be deleted merely because downstream overlays, reports, "
        "or review packs exist.\n",
        encoding="utf-8",
    )
    with (source_root / "source_access_ledger.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                {
                    "opened_at": utc_now(),
                    "source_video_uri": str(source_video),
                    "source_video_sha256": source_sha,
                    "purpose": "source_retention_contract",
                },
                sort_keys=True,
            )
            + "\n"
        )
    return contract
