# ruff: noqa: E501

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    bbox_area,
    bbox_from_item,
    bbox_iou,
    governance_stamp,
    round_float,
    safe_float,
    source_footpoint,
    validate_payload,
    visual_stamp,
)


STAGE = "STEP1_VISUAL_PERSON_RECONSTRUCTION"
ARTIFACT = "step1_person_candidates"

SOURCE_CANDIDATE_TYPES = {
    "player": "player_candidate_source",
    "official": "official_candidate_source",
    "referee": "referee_candidate_source",
    "staff": "staff_context_candidate_source",
    "unknown": "unknown_candidate_source",
}

SOURCE_PRIORITY = {
    "referee_candidate_source": 0,
    "official_candidate_source": 1,
    "player_candidate_source": 2,
    "staff_context_candidate_source": 3,
    "unknown_candidate_source": 4,
    "person_candidate": 5,
    "unknown_person_candidate": 6,
}

LOW_BBOX_QUALITY_THRESHOLD = 0.45


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def candidate_type_for_source(source_name: str, detection: dict[str, Any]) -> str:
    role = str(detection.get("role_label", "")).lower()
    reason = str(detection.get("classification_reason", "")).lower()
    if "false_positive" in role or "false_positive" in reason:
        return "false_positive_candidate"
    if source_name == "unknown" and "outside_playing_area_roi" in reason:
        return "unknown_candidate_source"
    return SOURCE_CANDIDATE_TYPES.get(source_name, "unknown_person_candidate")


def crop_quality_from_detection(detection: dict[str, Any]) -> float | None:
    if detection.get("crop_quality") is not None:
        return round_float(detection.get("crop_quality"), digits=4)
    evidence = detection.get("team_colour_evidence")
    if isinstance(evidence, dict) and evidence.get("crop_quality") is not None:
        return round_float(evidence.get("crop_quality"), digits=4)
    return None


def bbox_quality(
    bbox: dict[str, Any] | None,
    detection: dict[str, Any],
    frame_meta: dict[str, Any],
) -> tuple[float, str, list[str]]:
    if not bbox:
        return 0.0, "missing_bbox", ["missing_bbox"]

    width = max(0.0, safe_float(bbox["x2"]) - safe_float(bbox["x1"]))
    height = max(0.0, safe_float(bbox["y2"]) - safe_float(bbox["y1"]))
    area = bbox_area(bbox)
    confidence = safe_float(detection.get("confidence"))
    frame_width = safe_float(frame_meta.get("width"), 0.0)
    frame_height = safe_float(frame_meta.get("height"), 0.0)
    reasons: list[str] = []

    score = 0.55 + min(0.35, max(0.0, confidence) * 0.35)
    if width < 8.0 or height < 18.0 or area < 180.0:
        score -= 0.35
        reasons.append("tiny_bbox")
    elif width < 14.0 or height < 30.0 or area < 360.0:
        score -= 0.18
        reasons.append("small_bbox")

    edge_touch = (
        safe_float(bbox["x1"]) <= 1.0
        or safe_float(bbox["y1"]) <= 1.0
        or (frame_width > 0 and safe_float(bbox["x2"]) >= frame_width - 1.0)
        or (frame_height > 0 and safe_float(bbox["y2"]) >= frame_height - 1.0)
    )
    if edge_touch:
        score -= 0.22
        reasons.append("bbox_truncated_at_frame_edge")

    if confidence < 0.12:
        score -= 0.20
        reasons.append("low_bbox_confidence")

    if not reasons:
        reasons.append("bbox_plausible")
    return round(max(0.0, min(1.0, score)), 4), "+".join(reasons), reasons


def roi_status_for_detection(source_name: str, detection: dict[str, Any]) -> str:
    reason = str(detection.get("classification_reason", "")).lower()
    if detection.get("recovery_status") == "restored_by_near_side_recovery":
        return "near_side_recovery_visual_context"
    if "outside_playing_area_roi" in reason:
        return "outside_playing_roi"
    if source_name == "staff" or detection.get("exclusion_zone"):
        return "staff_or_hard_exclusion_context"
    if source_name in {"official", "referee"} or detection.get("official_zone"):
        return "official_referee_context"
    return "inside_or_unverified_visual_roi"


def frame_meta_for_detection(
    detection: dict[str, Any],
    frame: dict[str, Any],
    manifest_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    frame_id = str(detection.get("frame_id") or frame.get("frame_id") or "")
    meta = dict(manifest_by_id.get(frame_id, {}))
    for key, value in frame.items():
        if key != "detections":
            meta[key] = value
    if "frame_file" not in meta and detection.get("frame_file"):
        meta["frame_file"] = detection.get("frame_file")
    return meta


def source_row(
    detection: dict[str, Any],
    *,
    source_name: str,
    frame: dict[str, Any],
    frame_meta: dict[str, Any],
    source_path: str,
    row_index: int,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    bbox = bbox_from_item(detection)
    if not bbox:
        warnings.append("missing_bbox")
        bbox_payload = {"x1": None, "y1": None, "x2": None, "y2": None}
    else:
        bbox_payload = {key: round_float(bbox[key]) for key in ["x1", "y1", "x2", "y2"]}

    source_detection_id = str(detection.get("source_detection_id") or "").strip()
    if not source_detection_id:
        source_detection_id = str(detection.get("detection_id") or f"missing_source_detection_id_{row_index}")
        warnings.append("missing_source_detection_id")

    quality_score, quality_reason, quality_reasons = bbox_quality(bbox, detection, frame_meta)
    if quality_score < LOW_BBOX_QUALITY_THRESHOLD:
        warnings.append("low_bbox_quality")
    roi_status = roi_status_for_detection(source_name, detection)
    if roi_status == "outside_playing_roi":
        warnings.append("candidate_outside_playing_roi")

    candidate_type = candidate_type_for_source(source_name, detection)
    row = visual_stamp(
        {
            "frame_id": str(detection.get("frame_id") or frame.get("frame_id") or ""),
            "frame_sequence": int(safe_float(detection.get("frame_sequence", frame.get("frame_sequence", -1)), -1)),
            "timestamp_seconds": round_float(detection.get("timestamp_seconds", frame.get("timestamp_seconds", 0.0))),
            "frame_file": str(detection.get("frame_file") or frame_meta.get("frame_file") or ""),
            "detection_id": str(detection.get("detection_id") or ""),
            "source_detection_id": source_detection_id,
            "bbox": bbox_payload,
            "footpoint": source_footpoint(detection, bbox),
            "candidate_type": candidate_type,
            "bbox_confidence": round_float(detection.get("confidence"), digits=6),
            "bbox_quality_score": quality_score,
            "bbox_quality_reason": quality_reason,
            "crop_quality": crop_quality_from_detection(detection),
            "roi_status": roi_status,
            "duplicate_group_id": "unassigned",
            "duplicate_action": "unassigned",
            "state": "unknown",
            "confidence": 0.0,
            "reason": "step1a_inventory_state_not_assigned",
            "original_detection_id": str(detection.get("detection_id") or ""),
            "original_role_source": source_name,
            "source_role_labels": [str(detection.get("role_label") or source_name)],
            "source_candidate_types": [candidate_type],
            "source_object_types": [str(detection.get("object_type") or "")],
            "source_model_stages": [str(detection.get("model_stage") or "")],
            "classification_reasons": [str(detection.get("classification_reason") or "")],
            "source_path": source_path,
            "bbox_quality_reasons": quality_reasons,
            "qa_warnings": warnings,
        }
    )
    return row, warnings


def flatten_source_payload(
    payload: dict[str, Any],
    *,
    source_name: str,
    source_path: str,
    manifest_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    row_index = 0
    for frame in payload.get("frames", []):
        for detection in frame.get("detections", []):
            frame_meta = frame_meta_for_detection(detection, frame, manifest_by_id)
            row, row_warnings = source_row(
                detection,
                source_name=source_name,
                frame=frame,
                frame_meta=frame_meta,
                source_path=source_path,
                row_index=row_index,
            )
            rows.append(row)
            warnings.extend(f"{row['frame_id']}:{row['original_detection_id']}:{warning}" for warning in row_warnings)
            row_index += 1
    return rows, warnings


def visual_duplicate(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not left.get("bbox") or not right.get("bbox"):
        return False
    iou = bbox_iou(left.get("bbox"), right.get("bbox"))
    if iou >= 0.98:
        return True
    if left.get("source_detection_id") and left.get("source_detection_id") == right.get("source_detection_id"):
        lx = left.get("footpoint", {})
        rx = right.get("footpoint", {})
        if lx.get("x") is not None and rx.get("x") is not None:
            gap = ((safe_float(lx["x"]) - safe_float(rx["x"])) ** 2 + (safe_float(lx["y"]) - safe_float(rx["y"])) ** 2) ** 0.5
            return gap <= 2.0 and bbox_iou(left.get("bbox"), right.get("bbox")) >= 0.90
    return False


def merge_source_rows(rows: list[dict[str, Any]], *, duplicate_group_id: str, action: str) -> dict[str, Any]:
    best = sorted(
        rows,
        key=lambda row: (
            SOURCE_PRIORITY.get(str(row.get("candidate_type")), 99),
            -safe_float(row.get("bbox_quality_score")),
            -safe_float(row.get("bbox_confidence")),
        ),
    )[0]
    merged = dict(best)
    source_types = sorted({value for row in rows for value in row.get("source_candidate_types", [])})
    source_roles = sorted({value for row in rows for value in row.get("source_role_labels", [])})
    original_ids = sorted({str(row.get("original_detection_id", "")) for row in rows if row.get("original_detection_id")})
    source_object_types = sorted({value for row in rows for value in row.get("source_object_types", [])})
    source_model_stages = sorted({value for row in rows for value in row.get("source_model_stages", [])})
    classification_reasons = sorted({value for row in rows for value in row.get("classification_reasons", []) if value})
    qa_warnings = sorted({value for row in rows for value in row.get("qa_warnings", [])})

    if len(source_types) > 1:
        merged["candidate_type"] = "person_candidate"
        qa_warnings.append("candidate_source_disagreement")

    merged["original_detection_ids"] = original_ids
    merged["source_role_labels"] = source_roles
    merged["source_candidate_types"] = source_types
    merged["source_object_types"] = source_object_types
    merged["source_model_stages"] = source_model_stages
    merged["classification_reasons"] = classification_reasons
    merged["duplicate_group_id"] = duplicate_group_id
    merged["duplicate_action"] = action
    merged["qa_warnings"] = sorted(set(qa_warnings))
    return merged


def split_visual_duplicate_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for row in rows:
        for group in groups:
            if any(visual_duplicate(row, other) for other in group):
                group.append(row)
                break
        else:
            groups.append([row])
    return groups


def merge_duplicate_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["frame_sequence"]), str(row["source_detection_id"]))].append(row)

    merged_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for (frame_sequence, source_detection_id), group_rows in sorted(grouped.items()):
        duplicate_group_id = f"frame{frame_sequence:06d}_{source_detection_id[-24:]}"
        if len(group_rows) == 1:
            merged_rows.append(merge_source_rows(group_rows, duplicate_group_id=duplicate_group_id, action="unique"))
            continue

        visual_groups = split_visual_duplicate_groups(group_rows)
        if len(visual_groups) > 1:
            warnings.append(f"duplicate_source_detection_id_conflict:{frame_sequence}:{source_detection_id}")
            for visual_group in visual_groups:
                row = merge_source_rows(
                    visual_group,
                    duplicate_group_id=duplicate_group_id,
                    action="duplicate_source_detection_id_conflict_kept_separate",
                )
                row["qa_warnings"] = sorted(set(row.get("qa_warnings", []) + ["duplicate_source_detection_id_conflict"]))
                merged_rows.append(row)
            continue

        row = merge_source_rows(
            group_rows,
            duplicate_group_id=duplicate_group_id,
            action="merged_exact_source_detection_id",
        )
        if len(set(row.get("source_candidate_types", []))) > 1:
            warnings.append(f"candidate_source_disagreement:{frame_sequence}:{source_detection_id}")
        merged_rows.append(row)

    merged_rows.sort(key=lambda row: (int(row["frame_sequence"]), safe_float(row["bbox"].get("y1")), safe_float(row["bbox"].get("x1"))))
    for index, row in enumerate(merged_rows):
        row["detection_id"] = f"step1_person_candidate_{int(row['frame_sequence']):06d}_{index:05d}"
    return merged_rows, warnings


def frame_records(manifest_frames: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_frame[str(row["frame_id"])].append(row)
    records = []
    for frame in sorted(manifest_frames, key=lambda item: int(safe_float(item.get("frame_sequence"), -1))):
        frame_id = str(frame.get("frame_id", ""))
        frame_rows = rows_by_frame.get(frame_id, [])
        records.append(
            {
                "frame_id": frame_id,
                "frame_sequence": int(safe_float(frame.get("frame_sequence"), -1)),
                "timestamp_seconds": round_float(frame.get("timestamp_seconds")),
                "frame_file": str(frame.get("frame_file", "")),
                "candidate_count": len(frame_rows),
                "candidate_detection_ids": [row["detection_id"] for row in frame_rows],
            }
        )
    return records


def count_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(str(row["candidate_type"]) for row in rows)
    by_state = Counter(str(row["state"]) for row in rows)
    warning_counts = Counter(warning for row in rows for warning in row.get("qa_warnings", []))
    duplicate_actions = Counter(str(row["duplicate_action"]) for row in rows)
    return {
        "total_rows": len(rows),
        "candidate_counts_by_type": dict(sorted(by_type.items())),
        "state_counts": dict(sorted(by_state.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "duplicate_actions": dict(sorted(duplicate_actions.items())),
    }


def build_candidate_inventory_payload(
    *,
    manifest_frames: list[dict[str, Any]],
    source_payloads: dict[str, dict[str, Any]],
    source_paths: dict[str, str],
) -> dict[str, Any]:
    manifest_by_id = {str(frame.get("frame_id")): frame for frame in manifest_frames}
    source_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for source_name, payload in source_payloads.items():
        rows, source_warnings = flatten_source_payload(
            payload,
            source_name=source_name,
            source_path=source_paths.get(source_name, ""),
            manifest_by_id=manifest_by_id,
        )
        source_rows.extend(rows)
        warnings.extend(source_warnings)

    merged_rows, duplicate_warnings = merge_duplicate_rows(source_rows)
    warnings.extend(duplicate_warnings)
    for row in merged_rows:
        for warning in row.get("qa_warnings", []):
            if warning in {"candidate_source_disagreement", "low_bbox_quality", "candidate_outside_playing_roi"}:
                warnings.append(f"{row['frame_id']}:{row['source_detection_id']}:{warning}")

    payload = governance_stamp(
        {
            "stage": STAGE,
            "artifact": ARTIFACT,
            "created_at": utc_iso(),
            "source_paths": source_paths,
            "frame_count": len(manifest_frames),
            "rows": merged_rows,
            "frames": frame_records(manifest_frames, merged_rows),
            "summary": count_summary(merged_rows),
            "warnings": sorted(set(warnings))[:500],
            "warning_sample_limit": 500,
            "no_metrics_calculated": True,
            "identity_tracking_performed": False,
            "player_slots_assigned": False,
            "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
            "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
            "production_ready": PRODUCTION_READY,
        }
    )
    validate_payload(payload, artifact=ARTIFACT)
    return payload


def per_frame_type_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    frame_ids: dict[int, str] = {}
    for row in rows:
        seq = int(row["frame_sequence"])
        frame_ids[seq] = str(row["frame_id"])
        counts[seq][str(row["candidate_type"])] += 1
    out = []
    for seq in sorted(counts):
        total = sum(counts[seq].values())
        out.append({"frame_sequence": seq, "frame_id": frame_ids.get(seq, ""), "total": total, **dict(sorted(counts[seq].items()))})
    return out


def candidate_inventory_report_markdown(payload: dict[str, Any]) -> str:
    rows = list(payload.get("rows", []))
    summary = payload.get("summary", {})
    warning_counts = Counter(warning for row in rows for warning in row.get("qa_warnings", []))
    by_frame = sorted(per_frame_type_counts(rows), key=lambda item: (-int(item["total"]), int(item["frame_sequence"])))[:30]
    type_counts = summary.get("candidate_counts_by_type", {})
    lines = [
        "# Step1.A Person Candidate Inventory Report",
        "",
        f"- Warning: `{VISUAL_ONLY_WARNING}`.",
        "- production_ready false",
        "- project_wide_defaults_changed false",
        "- stage3d_registries_changed false",
        "- no metrics calculated",
        "- no identity tracking performed",
        "- no player slots assigned",
        "",
        "## Candidate Counts By Type",
        "",
        "| candidate_type | count |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in sorted(type_counts.items()))
    lines.extend(
        [
            "",
            "## Highest Count Frames By Candidate Type",
            "",
            "| frame_sequence | frame_id | total | type_counts |",
            "|---:|---|---:|---|",
        ]
    )
    for item in by_frame:
        type_text = ", ".join(f"{key}={value}" for key, value in item.items() if key not in {"frame_sequence", "frame_id", "total"})
        lines.append(f"| {item['frame_sequence']} | {item['frame_id']} | {item['total']} | {type_text} |")
    lines.extend(
        [
            "",
            "## Warning Counts",
            "",
            "| warning | count |",
            "|---|---:|",
        ]
    )
    required_warning_names = [
        "missing_bbox",
        "missing_source_detection_id",
        "duplicate_source_detection_id_conflict",
        "low_bbox_quality",
        "candidate_outside_playing_roi",
        "candidate_source_disagreement",
    ]
    for name in required_warning_names:
        lines.append(f"| {name} | {warning_counts.get(name, 0)} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Official, referee, staff, and unknown source rows are retained as person reconstruction context.",
            "- Exact duplicate visual detections are merged by source_detection_id while preserving source role labels.",
            "- Candidate source disagreements are retained as `person_candidate` rows with source type lists.",
        ]
    )
    return "\n".join(lines) + "\n"
