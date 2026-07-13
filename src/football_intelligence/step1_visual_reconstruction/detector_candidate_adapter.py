from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from football_intelligence.review.schemas import safety_payload, stable_hash

PERSON_CANDIDATE = "person_candidate"


def rows_from_payload(payload: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [row for row in payload["rows"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError("detector payload must be a row list or an object with rows")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def bbox_from_row(row: dict[str, Any]) -> dict[str, float]:
    bbox = row.get("bbox") if isinstance(row.get("bbox"), dict) else row
    return {key: round(_safe_float(bbox.get(key)), 3) for key in ("x1", "y1", "x2", "y2")}


def adapt_detector_rows(
    detector_payload: dict[str, Any] | list[dict[str, Any]],
    *,
    frame_dimensions: dict[int, tuple[int, int]] | None = None,
    source_name: str = "m5_4a_full_frame_detector",
) -> dict[str, Any]:
    rows = rows_from_payload(detector_payload)
    adapted: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        frame_sequence = _safe_int(row.get("frame_sequence"))
        width, height = (frame_dimensions or {}).get(frame_sequence, (None, None))
        raw_id = str(row.get("detection_id") or row.get("source_detection_id") or f"raw_detector_{index:06d}")
        bbox = bbox_from_row(row)
        adapted.append(
            {
                "candidate_id": f"rebuilt_pc_{index:06d}",
                "raw_detector_row_index": index,
                "raw_detector_row_id": raw_id,
                "source_detection_id": str(row.get("source_detection_id") or raw_id),
                "frame_sequence": frame_sequence,
                "frame_file": row.get("frame_file"),
                "frame_filename": row.get("frame_filename"),
                "bbox": bbox,
                "confidence": round(_safe_float(row.get("confidence")), 6),
                "class_id": _safe_int(row.get("class_id")),
                "class_name": str(row.get("class_name") or "person"),
                "candidate_type": PERSON_CANDIDATE,
                "detector_semantic_type_preserved": row.get("object_type"),
                "detector_role_label_preserved": row.get("role_label"),
                "model_sha256": row.get("model_sha256"),
                "inference_configuration_hash": row.get("inference_configuration_hash"),
                "source_type": row.get("source_type") or source_name,
                "full_frame_dimensions": {"width": width, "height": height},
                "raw_row_preserved": True,
                "detector_row_deleted": False,
                "auto_labelled_player": False,
                **safety_payload(),
            }
        )
    return {
        "artifact": "m5_4d_person_candidate_rows",
        "source_name": source_name,
        "input_detector_row_count": len(rows),
        "output_person_candidate_count": len(adapted),
        "candidate_type": PERSON_CANDIDATE,
        "raw_rows_preserved": True,
        "detector_outputs_auto_labelled_player": False,
        "rows": adapted,
        **safety_payload(),
    }


def candidate_id_map(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in ("candidate_id", "raw_detector_row_id", "source_detection_id"):
            value = row.get(key)
            if value:
                mapped[str(value)] = row
    return mapped


def candidate_payload_hash(payload: dict[str, Any]) -> str:
    return stable_hash(
        [
            {
                "candidate_id": row["candidate_id"],
                "raw_detector_row_id": row["raw_detector_row_id"],
                "bbox": row["bbox"],
                "frame_sequence": row["frame_sequence"],
            }
            for row in payload.get("rows", [])
        ]
    )
