from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_intelligence.replay.source_access import SourceAccessLedger
from football_intelligence.step2_visual_continuity.sparse_handoff_package import (
    M4_MAX_OVERLAY_FRAMES_PER_PATHLET,
    overlay_selection,
)
from football_intelligence.step2_visual_continuity.topology_qa import sample_frame_sequences


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def required_overlay_frames(pathlets: list[dict[str, Any]]) -> dict[int, set[str]]:
    required: dict[int, set[str]] = {}
    for pathlet in overlay_selection(pathlets):
        member_frames = [_safe_int(frame) for frame in pathlet.get("member_frame_sequences", [])]
        frames = sample_frame_sequences(
            _safe_int(pathlet.get("min_frame_sequence")),
            _safe_int(pathlet.get("max_frame_sequence")),
            member_frames,
            max_frames=M4_MAX_OVERLAY_FRAMES_PER_PATHLET,
        )
        for frame in frames:
            required.setdefault(frame, set()).add(str(pathlet.get("m4_handoff_pathlet_id", "")))
    return required


def _image_dimensions(path: Path, fallback: dict[str, Any]) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.width, image.height
    except Exception:
        width = fallback.get("width")
        height = fallback.get("height")
        return (int(width) if isinstance(width, int) else None, int(height) if isinstance(height, int) else None)


def build_frame_lookup(
    *,
    frame_manifest: dict[str, Any],
    pathlets: list[dict[str, Any]],
    artifact_root: Path,
    output_path: Path,
    ledger: SourceAccessLedger,
) -> tuple[dict[int, str], dict[str, Any]]:
    manifest_frames = frame_manifest.get("frames", [])
    if not isinstance(manifest_frames, list):
        raise ValueError("frame manifest must contain a frames list")
    by_sequence = {
        _safe_int(row.get("frame_sequence")): row
        for row in manifest_frames
        if isinstance(row, dict) and _safe_int(row.get("frame_sequence")) >= 0
    }
    required = required_overlay_frames(pathlets)
    records: list[dict[str, Any]] = []
    lookup: dict[int, str] = {}
    missing: list[int] = []
    for frame_sequence in sorted(required):
        frame_record = by_sequence.get(frame_sequence)
        if frame_record is None:
            missing.append(frame_sequence)
            records.append(
                {
                    "frame_sequence": frame_sequence,
                    "root_relative_source_frame_uri": "",
                    "byte_hash": None,
                    "width": None,
                    "height": None,
                    "readable": False,
                    "reason_required": sorted(required[frame_sequence]),
                }
            )
            continue
        raw_path = Path(str(frame_record.get("frame_file", "")))
        source_path = raw_path if raw_path.is_absolute() else artifact_root / raw_path
        if not source_path.exists():
            missing.append(frame_sequence)
            readable = False
            byte_hash = None
            width = frame_record.get("width")
            height = frame_record.get("height")
        else:
            ledger_record = ledger.record_binary_read(
                source_path,
                purpose=f"source frame for M4 overlay rendering frame_sequence={frame_sequence}",
                allowed_input_id="frame_root.stage3c_hq_short",
            )
            readable = True
            byte_hash = ledger_record["byte_hash"]
            width, height = _image_dimensions(source_path, frame_record)
            lookup[frame_sequence] = str(source_path.resolve())
        relative_uri = (
            source_path.resolve().relative_to(artifact_root.resolve()).as_posix()
            if source_path.resolve().is_relative_to(artifact_root.resolve())
            else source_path.as_posix()
        )
        records.append(
            {
                "frame_sequence": frame_sequence,
                "root_relative_source_frame_uri": relative_uri,
                "byte_hash": byte_hash,
                "width": width,
                "height": height,
                "readable": readable,
                "reason_required": sorted(required[frame_sequence]),
            }
        )
    payload = {
        "schema_version": "m5.true_replay.frame_lookup.v1",
        "frame_manifest_run_id": frame_manifest.get("run_id", ""),
        "required_frame_count": len(required),
        "resolved_frame_count": len(lookup),
        "missing_frame_sequences": missing,
        "records": records,
        "passed": not missing and len(lookup) == len(required),
    }
    write_json(output_path, payload)
    if not payload["passed"]:
        raise FileNotFoundError(f"missing required M4 source frames: {missing[:20]}")
    return lookup, payload
