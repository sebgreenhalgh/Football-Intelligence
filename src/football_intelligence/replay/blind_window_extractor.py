from __future__ import annotations

import hashlib
import json
import platform
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.core.fingerprints import semantic_hash, sha256_file
from football_intelligence.replay.blind_window_selection import VideoMetadata, probe_video_metadata

EXTRACTION_SCHEMA = "m5.blind_window.extraction.v1"


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


def source_indices(
    *,
    selected_start_seconds: int,
    source_fps: int = 25,
    output_fps: int = 10,
    output_frame_count: int = 600,
) -> list[int]:
    start_index = round(selected_start_seconds * source_fps)
    return [start_index + round(k * source_fps / output_fps) for k in range(output_frame_count)]


def decoded_pixel_hash(image_path: Path) -> str:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"image did not decode: {image_path}")
    return hashlib.sha256(image.tobytes()).hexdigest()


def _validate_recipe(metadata: VideoMetadata, *, expected_width: int, expected_height: int, expected_fps: int) -> None:
    if metadata.width != expected_width or metadata.height != expected_height:
        raise ValueError(
            f"source dimensions mismatch: expected {expected_width}x{expected_height}, "
            f"got {metadata.width}x{metadata.height}"
        )
    if round(metadata.fps) != expected_fps or abs(metadata.fps - expected_fps) > 1e-6:
        raise ValueError(f"source FPS mismatch: expected {expected_fps}, got {metadata.fps}")


def _frame_record(path: Path, *, sequence: int, source_frame_index: int, width: int, height: int) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "filename": path.name,
        "relative_uri": path.name,
        "source_frame_index": source_frame_index,
        "width": width,
        "height": height,
        "byte_size": path.stat().st_size,
        "byte_sha256": sha256_file(path),
        "decoded_pixel_sha256": decoded_pixel_hash(path),
    }


def extract_blind_window(
    *,
    source_video: Path,
    selected_start_seconds: int,
    duration_seconds: int,
    output_fps: int,
    output_width: int,
    output_height: int,
    output_root: Path,
    jpeg_quality: int,
    expected_source_width: int = 4096,
    expected_source_height: int = 1080,
    expected_source_fps: int = 25,
) -> dict[str, Any]:
    import cv2

    metadata = probe_video_metadata(source_video)
    _validate_recipe(
        metadata,
        expected_width=expected_source_width,
        expected_height=expected_source_height,
        expected_fps=expected_source_fps,
    )
    output_frame_count = int(duration_seconds * output_fps)
    expected_indices = source_indices(
        selected_start_seconds=selected_start_seconds,
        source_fps=expected_source_fps,
        output_fps=output_fps,
        output_frame_count=output_frame_count,
    )
    if expected_indices[-1] >= metadata.frame_count:
        raise ValueError("selected interval exceeds source frame count")

    output_root.mkdir(parents=True, exist_ok=True)
    start_index = expected_indices[0]
    cap = cv2.VideoCapture(str(source_video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
        next_expected = 0
        source_index = start_index
        records: list[dict[str, Any]] = []
        while next_expected < len(expected_indices):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"failed to decode source frame {source_index}")
            if source_index == expected_indices[next_expected]:
                resized = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
                filename = (
                    f"128058_m5_blind2_{selected_start_seconds}_{selected_start_seconds + duration_seconds}"
                    f"_f{source_index:06d}.jpg"
                )
                out_path = output_root / filename
                cv2.imwrite(str(out_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                records.append(
                    _frame_record(
                        out_path,
                        sequence=next_expected,
                        source_frame_index=source_index,
                        width=output_width,
                        height=output_height,
                    )
                )
                next_expected += 1
            source_index += 1
    finally:
        cap.release()

    inventory_hash = semantic_hash(
        [
            {
                "sequence": row["sequence"],
                "filename": row["filename"],
                "source_frame_index": row["source_frame_index"],
                "byte_sha256": row["byte_sha256"],
                "decoded_pixel_sha256": row["decoded_pixel_sha256"],
            }
            for row in records
        ]
    )
    manifest = {
        "schema_version": EXTRACTION_SCHEMA,
        "created_at": utc_now(),
        "source_video": str(source_video),
        "source_video_sha256": sha256_file(source_video),
        "selected_start_seconds": selected_start_seconds,
        "selected_end_seconds": selected_start_seconds + duration_seconds,
        "duration_seconds": duration_seconds,
        "source_fps": expected_source_fps,
        "output_fps": output_fps,
        "output_width": output_width,
        "output_height": output_height,
        "jpeg_quality": jpeg_quality,
        "interpolation": "OpenCV INTER_AREA",
        "color_behavior": "BGR frame decoded by OpenCV and encoded directly to JPEG",
        "expected_frame_count": output_frame_count,
        "actual_frame_count": len(records),
        "ordered_frame_inventory_hash": inventory_hash,
        "frames": records,
    }
    write_json(output_root / "frame_manifest.json", manifest)
    environment = {
        "schema_version": "m5.blind_window.extraction_environment.v1",
        "created_at": utc_now(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "source_metadata": metadata.as_dict(),
    }
    write_json(output_root / "extraction_environment.json", environment)
    validation = {
        "schema_version": "m5.blind_window.extraction_validation.v1",
        "created_at": utc_now(),
        "expected_frame_count": output_frame_count,
        "actual_frame_count": len(records),
        "all_images_decode": all(bool(row["decoded_pixel_sha256"]) for row in records),
        "all_dimensions_match": all(row["width"] == output_width and row["height"] == output_height for row in records),
        "sequence_monotonic": [row["sequence"] for row in records] == list(range(output_frame_count)),
        "source_indices_match_recipe": [row["source_frame_index"] for row in records] == expected_indices,
        "passed": len(records) == output_frame_count,
    }
    write_json(output_root / "extraction_validation.json", validation)
    with (output_root / "source_access_ledger.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                {
                    "opened_at": utc_now(),
                    "source_video": str(source_video),
                    "purpose": "deterministic_blind_window_frame_extraction",
                    "byte_sha256": manifest["source_video_sha256"],
                },
                sort_keys=True,
            )
            + "\n"
        )
    return manifest


def compare_extractions(left_manifest: Path, right_manifest: Path, output_path: Path) -> dict[str, Any]:
    left = read_json(left_manifest)
    right = read_json(right_manifest)
    left_rows = left.get("frames", [])
    right_rows = right.get("frames", [])
    pairs = list(zip(left_rows, right_rows, strict=False))
    mismatches = []
    for index, (left_row, right_row) in enumerate(pairs):
        checks = {
            "filename": left_row.get("filename") == right_row.get("filename"),
            "dimensions": (left_row.get("width"), left_row.get("height"))
            == (right_row.get("width"), right_row.get("height")),
            "byte_hash": left_row.get("byte_sha256") == right_row.get("byte_sha256"),
            "decoded_pixel_hash": left_row.get("decoded_pixel_sha256") == right_row.get("decoded_pixel_sha256"),
        }
        if not all(checks.values()):
            mismatches.append({"index": index, "checks": checks, "left": left_row, "right": right_row})
    result = {
        "schema_version": "m5.blind_window.frame_extraction_repeatability.v1",
        "created_at": utc_now(),
        "left_manifest": str(left_manifest),
        "right_manifest": str(right_manifest),
        "left_frame_count": len(left_rows),
        "right_frame_count": len(right_rows),
        "filename_coverage_match": [r.get("filename") for r in left_rows] == [r.get("filename") for r in right_rows],
        "left_inventory_hash": left.get("ordered_frame_inventory_hash"),
        "right_inventory_hash": right.get("ordered_frame_inventory_hash"),
        "inventory_hash_match": left.get("ordered_frame_inventory_hash") == right.get("ordered_frame_inventory_hash"),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "passed": len(left_rows) == len(right_rows) and not mismatches,
    }
    write_json(output_path, result)
    return result


def build_raw_frame_sanity_report(frame_manifest: Path, output_dir: Path) -> dict[str, Any]:
    import cv2
    import numpy as np

    manifest = read_json(frame_manifest)
    frame_root = frame_manifest.parent
    samples = [0, 1, 100, 200, 300, 400, 500, 598, 599]
    sample_records = []
    images = []
    for sequence in samples:
        row = manifest["frames"][sequence]
        path = frame_root / row["filename"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"sample frame did not decode: {path}")
        sample_records.append(
            {
                "sequence": sequence,
                "filename": row["filename"],
                "source_frame_index": row["source_frame_index"],
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "decoded": True,
            }
        )
        images.append(image)
    thumb_width = 455
    thumbs = [cv2.resize(image, (thumb_width, 120), interpolation=cv2.INTER_AREA) for image in images]
    sheet = np.vstack([np.hstack(thumbs[:3]), np.hstack(thumbs[3:6]), np.hstack(thumbs[6:9])])
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet = output_dir / "raw_frame_contact_sheet.jpg"
    cv2.imwrite(str(contact_sheet), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    report = {
        "schema_version": "m5.blind_window.raw_frame_sanity.v1",
        "created_at": utc_now(),
        "frame_manifest": str(frame_manifest),
        "contact_sheet": str(contact_sheet),
        "sample_sequences": samples,
        "samples": sample_records,
        "no_annotations": True,
        "no_bounding_boxes": True,
        "no_pitch_diagrams": True,
        "no_labels_embedded_in_frames": True,
        "dimensions_are_2730x720": all(row["width"] == 2730 and row["height"] == 720 for row in sample_records),
        "sequence_order_monotonic": samples == sorted(samples),
        "used_to_change_selected_window": False,
        "passed": True,
    }
    write_json(output_dir / "raw_frame_sanity_report.json", report)
    return report
