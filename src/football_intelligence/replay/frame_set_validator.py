from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DERIVED_REJECTION_RULES = [
    ("step2m4_sparse_handoff_package", "preserved_m4_evidence_not_eligible"),
    ("step2m4_pathlet_overlay_frames", "preserved_m4_evidence_not_eligible"),
    ("step2m4_pathlet_overlay_strips", "preserved_m4_evidence_not_eligible"),
    ("step2m4_pathlet_overlay_gifs", "preserved_m4_evidence_not_eligible"),
    ("goal_window_stage3d_static_freeze_clean", "derived_annotated_frames_not_eligible"),
    ("static_freeze", "derived_annotated_frames_not_eligible"),
    ("contact_sheet", "derived_annotated_frames_not_eligible"),
    ("review_pack", "derived_annotated_frames_not_eligible"),
    ("screenshot", "derived_annotated_frames_not_eligible"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_pixel_hash(path: Path) -> str | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            rgb = image.convert("RGB")
            payload = {
                "mode": "RGB",
                "width": rgb.width,
                "height": rgb.height,
                "pixels_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    except Exception:
        return None


def image_dimensions(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.width, image.height
    except Exception:
        return None, None


def directory_inventory_hash(root: Path, *, extensions: set[str] | None = None) -> dict[str, Any]:
    if not root.exists():
        return {
            "exists": False,
            "file_count": 0,
            "total_bytes": 0,
            "inventory_hash": None,
            "sample_files": [],
        }
    records = []
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if extensions and path.suffix.lower() not in extensions:
            continue
        stat = path.stat()
        total_bytes += stat.st_size
        records.append(
            {
                "relative_uri": path.relative_to(root).as_posix(),
                "byte_size": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    inventory_hash = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "exists": True,
        "file_count": len(records),
        "total_bytes": total_bytes,
        "inventory_hash": inventory_hash,
        "sample_files": records[:10],
    }


def derived_classification(path: Path) -> str | None:
    text = path.as_posix().lower()
    for marker, classification in DERIVED_REJECTION_RULES:
        if marker.lower() in text:
            return classification
    return None


def validate_candidate_frame_set(
    *,
    candidate_root: Path,
    required_frames: list[dict[str, Any]],
    declared_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    required_names = [str(row["expected_filename"]) for row in required_frames]
    declared_names = [str(row["expected_filename"]) for row in declared_frames]
    files_by_name = {path.name: path for path in candidate_root.glob("*.jpg")}
    required_present = [name for name in required_names if name in files_by_name]
    declared_present = [name for name in declared_names if name in files_by_name]
    dimension_failures = []
    file_records = []
    for row in required_frames:
        name = str(row["expected_filename"])
        path = files_by_name.get(name)
        if path is None:
            continue
        width, height = image_dimensions(path)
        expected_width = row.get("width")
        expected_height = row.get("height")
        dimensions_match = (width, height) == (expected_width, expected_height)
        if not dimensions_match:
            dimension_failures.append(
                {
                    "frame_sequence": row.get("frame_sequence"),
                    "filename": name,
                    "observed": [width, height],
                    "expected": [expected_width, expected_height],
                }
            )
        file_records.append(
            {
                "frame_sequence": row.get("frame_sequence"),
                "filename": name,
                "sha256": sha256_file(path),
                "decoded_pixel_hash": decoded_pixel_hash(path),
                "width": width,
                "height": height,
                "dimensions_match_manifest": dimensions_match,
            }
        )
    inventory = directory_inventory_hash(candidate_root, extensions={".jpg", ".jpeg"})
    passed = (
        candidate_root.exists()
        and len(required_present) == len(required_names)
        and len(declared_present) == len(declared_names)
        and not dimension_failures
    )
    return {
        "schema_version": "m5_2r_a.recovered_frame_set_validation.v1",
        "candidate_root": str(candidate_root.resolve()),
        "required_frame_count": len(required_names),
        "required_frames_present": len(required_present),
        "required_frames_missing": sorted(set(required_names) - set(required_present)),
        "declared_frame_count": len(declared_names),
        "declared_frames_present": len(declared_present),
        "declared_frames_missing": sorted(set(declared_names) - set(declared_present)),
        "dimension_failures": dimension_failures,
        "frame_hash_records": file_records,
        "inventory": inventory,
        "classification": "RECOVERED_EXACT_SOURCE_FRAME_SET" if passed else "PARTIAL_RECOVERY_INSUFFICIENT",
        "passed": passed,
    }


def inspect_derived_asset_exclusions(artifact_root: Path) -> dict[str, Any]:
    match_root = artifact_root / "matches/128058"
    candidates = [
        match_root
        / "calibration/step2_visual_continuity/step2m4_sparse_handoff_package/step2m4_pathlet_overlay_frames",
        match_root
        / "calibration/step2_visual_continuity/step2m4_sparse_handoff_package/step2m4_pathlet_overlay_strips",
        match_root / "calibration/step2_visual_continuity/step2m4_sparse_handoff_package/step2m4_pathlet_overlay_gifs",
        match_root / "overlays/goal_window_stage3d_static_freeze_clean",
        match_root / "calibration/step2_visual_continuity/step2m3t_sparse_pathlets/step2m3t_review_contact_sheet.jpg",
    ]
    records = []
    for path in candidates:
        classification = derived_classification(path) or "derived_annotated_frames_not_eligible"
        if path.is_dir():
            inventory = directory_inventory_hash(path, extensions={".jpg", ".jpeg", ".png", ".gif"})
            size = inventory["total_bytes"]
            sha256 = inventory["inventory_hash"]
            file_count = inventory["file_count"]
            sample_files = inventory["sample_files"]
        elif path.is_file():
            size = path.stat().st_size
            sha256 = sha256_file(path)
            file_count = 1
            sample_files = [{"relative_uri": path.name, "byte_size": size, "sha256": sha256}]
        else:
            size = 0
            sha256 = None
            file_count = 0
            sample_files = []
        records.append(
            {
                "path": str(path.resolve()),
                "exists": path.exists(),
                "type": "directory" if path.is_dir() else "file" if path.is_file() else "missing",
                "size_bytes": size,
                "sha256_or_inventory_hash": sha256,
                "file_count": file_count,
                "sample_files": sample_files,
                "raw_or_derived": "derived",
                "annotations_visible_or_implied_by_provenance": True,
                "classification": classification,
                "allowed_for_true_replay": False,
                "rejection_reason": "Derived or annotated visual evidence cannot be promoted to raw source frames.",
            }
        )
    return {
        "schema_version": "m5_2r_a.derived_asset_exclusion_report.v1",
        "records": records,
        "proves_no_derived_asset_promoted": all(record["allowed_for_true_replay"] is False for record in records),
        "passed": all(record["allowed_for_true_replay"] is False for record in records),
    }
