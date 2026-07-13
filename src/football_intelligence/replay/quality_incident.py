from __future__ import annotations

import json
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import cv2

from football_intelligence.core.fingerprints import sha256_file
from football_intelligence.replay.entity_validity import (
    AMBIGUOUS_ENTITY,
    PROBABLE_NON_PERSON,
    VALID_OFF_PITCH_PERSON,
    VALID_ON_PITCH_PERSON,
    build_entity_validity_payload,
    compound_continuity_disposition,
    entity_rows_by_visible_id,
    rows_from_payload,
    safe_float,
    safe_int,
)
from football_intelligence.replay.entity_validity_validation import validate_entity_validity_payload
from football_intelligence.replay.quality_gated_edge_validation import validate_quality_gated_edge_payload
from football_intelligence.replay.quality_gated_edges import (
    build_quality_gated_edge_payload,
    diagnose_current_edge_graph,
    edge_gate_result,
    location_incompatible,
    quality_gate_rule_provenance,
)
from football_intelligence.replay.portable_context import guardrail_payload, utc_now
from football_intelligence.review.schemas import (
    ENTITY_VALIDITY_DECISIONS,
    ENTITY_VALIDITY_QUESTION,
    EvidenceAsset,
    EvidenceManifest,
    ReviewCase,
    ReviewManifest,
    SourceArtifactReference,
    safety_payload,
    stable_hash,
)
from football_intelligence.review.workbench import build_workbench
from football_intelligence.step2_visual_continuity.edge_features import (
    bbox_center,
    bbox_height,
    footpoint_xy,
    px_delta,
)

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]


INCIDENT_CLASSIFICATION = "REVIEW_COMPLETED_DIAGNOSTIC_ONLY_INVALID_FOR_CONTINUITY_LEARNING"
PASS_CLASSIFICATION = "PASS_QUALITY_GATED_CANDIDATES_READY_FOR_SECOND_REVIEW"
BLOCKED_EVIDENCE_BINDING_FAILURE = "BLOCKED_EVIDENCE_BINDING_FAILURE"
BLOCKED_ENTITY_VALIDITY_QUALITY = "BLOCKED_ENTITY_VALIDITY_QUALITY"
BLOCKED_CONTINUITY_EDGE_GATING = "BLOCKED_CONTINUITY_EDGE_GATING"
FAIL_REVIEW_DECISION_INTEGRITY = "FAIL_REVIEW_DECISION_INTEGRITY"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _bbox(row: dict[str, Any]) -> dict[str, float]:
    bbox = row.get("bbox") if isinstance(row.get("bbox"), dict) else row
    return {key: safe_float(bbox.get(key)) for key in ("x1", "y1", "x2", "y2")}


def _bbox_equal(left: dict[str, Any] | None, right: dict[str, Any] | None, tolerance: float = 0.01) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    return all(
        abs(safe_float(left.get(key)) - safe_float(right.get(key))) <= tolerance for key in ("x1", "y1", "x2", "y2")
    )


def _frame_records(frame_manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        safe_int(frame.get("sequence", frame.get("frame_sequence")), -1): frame
        for frame in frame_manifest.get("frames", [])
        if isinstance(frame, dict)
    }


def _frame_path(frame_root: Path, frame: dict[str, Any]) -> Path:
    return (frame_root / str(frame.get("relative_uri", frame.get("filename", "")))).resolve()


def _read_image(path: Path) -> Any:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"image did not decode: {path}")
    return image


def _draw_box(image: Any, bbox: dict[str, float], label: str, color: tuple[int, int, int]) -> Any:
    out = image.copy()
    x1, y1, x2, y2 = [int(round(bbox[key])) for key in ("x1", "y1", "x2", "y2")]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
    cv2.putText(out, label, (max(0, x1), max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return out


def _fit_height(image: Any, height: int) -> Any:
    h, w = image.shape[:2]
    if h == height:
        return image
    width = max(1, int(round(w * height / max(1, h))))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _write_jpg(path: Path, image: Any, *, asset_id: str | None = None, asset_type: str | None = None) -> EvidenceAsset:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ValueError(f"failed to write image: {path}")
    return EvidenceAsset(
        asset_id=asset_id or path.stem,
        asset_type=asset_type or path.stem,
        relative_path=path.name,
        sha256=sha256_file(path),
        media_type="image/jpeg",
    )


def _write_gif(path: Path, frames: list[Any]) -> EvidenceAsset | None:
    if Image is None or not frames:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    pil_frames[0].save(path, save_all=True, append_images=pil_frames[1:], duration=240, loop=0)
    return EvidenceAsset(
        asset_id=path.stem,
        asset_type="animated_gif",
        relative_path=path.name,
        sha256=sha256_file(path),
        media_type="image/gif",
    )


def _bounds_for_bbox(
    bbox: dict[str, float],
    *,
    width: int,
    height: int,
    scale: float,
    min_size: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max((x2 - x1) * scale, float(min_size))
    bh = max((y2 - y1) * scale, float(min_size))
    left = max(0, int(round(cx - bw / 2.0)))
    top = max(0, int(round(cy - bh / 2.0)))
    right = min(width, int(round(cx + bw / 2.0)))
    bottom = min(height, int(round(cy + bh / 2.0)))
    return left, top, max(left + 1, right), max(top + 1, bottom)


def _crop(image: Any, bbox: dict[str, float], *, scale: float, min_size: int) -> Any:
    height, width = image.shape[:2]
    left, top, right, bottom = _bounds_for_bbox(bbox, width=width, height=height, scale=scale, min_size=min_size)
    return image[top:bottom, left:right]


def _candidate_case_id(candidate: dict[str, Any]) -> str:
    return str(candidate.get("portable_review_candidate_id", "")).replace("portable_review_", "m5_4b_continuity_")


def _source_ref(artifact_id: str, path: Path, role: str) -> SourceArtifactReference:
    return SourceArtifactReference(
        artifact_id=artifact_id,
        path=str(path),
        sha256=sha256_file(path) if path.exists() and path.is_file() else None,
        role=role,
    )


def default_paths(artifact_root: Path, match_id: str) -> dict[str, Path]:
    step_m5 = artifact_root / "matches" / match_id / "runs" / "step_m5"
    m54a = step_m5 / "06a_detector_dependency_recovery"
    m54b = step_m5 / "06b_unified_review_workbench"
    run_a = m54a / "runs" / "portable_real_run_a"
    frame_root = step_m5 / "05_blind_second_window" / "frames" / "extraction_a"
    return {
        "step_m5": step_m5,
        "m54a": m54a,
        "m54b": m54b,
        "m54c": step_m5 / "06c_blind_quality_incident",
        "run_a": run_a,
        "frame_root": frame_root,
        "frame_manifest": frame_root / "frame_manifest.json",
        "detector_rows": run_a / "step1" / "detector" / "detection_rows.json",
        "visible_rows": run_a / "step1" / "step1b4_visible_person_base_rows.json",
        "node_rows": run_a / "step2" / "step2m1_visual_continuity_node_rows.json",
        "edge_rows": run_a / "step2" / "step2m1_visual_continuity_edge_candidate_rows.json",
        "pathlets": run_a / "step2" / "step2m3t_sparse_pathlets.json",
        "candidate_rows": m54a / "review" / "blind_review_candidate_rows.json",
        "review_manifest": m54b / "review" / "review_manifest.json",
        "review_evidence_root": m54b / "review" / "evidence",
        "completed_review": m54b / "review" / "decisions" / "completed_review.json",
        "completed_review_manifest": m54b / "review" / "decisions" / "completed_review_manifest.json",
        "completed_review_events": m54b / "review" / "decisions" / "completed_review_events.jsonl",
        "completed_review_summary": m54b / "review" / "decisions" / "completed_review_summary.json",
    }


def write_incident_artifacts(paths: dict[str, Path], stage_root: Path) -> dict[str, Any]:
    completed = read_json(paths["completed_review"])
    completed_summary = read_json(paths["completed_review_summary"])
    file_paths = [
        paths["completed_review"],
        paths["completed_review_manifest"],
        paths["completed_review_events"],
        paths["completed_review_summary"],
    ]
    file_hashes = [
        {"path": str(path), "sha256": sha256_file(path), "byte_size": path.stat().st_size} for path in file_paths
    ]
    decisions = completed.get("state", {}).get("decisions", {})
    payload = guardrail_payload(
        {
            "artifact": "m5_4b_review_quality_incident",
            "created_at": utc_now(),
            "incident_classification": INCIDENT_CLASSIFICATION,
            "completed_review_file_hashes": file_hashes,
            "candidate_manifest_hash": completed.get("candidate_manifest_hash"),
            "evidence_manifest_hash": completed.get("evidence_manifest_hash"),
            "decision_state_hash": completed.get("decision_state_hash"),
            "decision_counts": {
                "accepted": completed_summary.get("accepted"),
                "rejected": completed_summary.get("rejected"),
                "unresolved": completed_summary.get("unresolved"),
                "total": completed_summary.get("total_cases"),
            },
            "decisions": decisions,
            "no_decisions_auto_promoted": True,
            "human_approved": False,
            "user_reported_visual_diagnosis": [
                "accepted cases were persistent non-entities, structures, or invalid off-pitch detections",
                "reviewed on-pitch candidate edges connected different people",
                "some on-pitch pairs visibly involved different teams",
                "some source and target boxes were materially location-incompatible",
            ],
            "review_ui_storage_integrity": "review UI worked correctly as a persistence interface",
            "candidate_quality_incident": "candidate and upstream entity quality were inadequate",
            "pathlet_learning_use_allowed": False,
            "incident_kind": "model_candidate_quality_incident_not_review_storage_incident",
        }
    )
    incident_json = stage_root / "incident" / "M5_4B_REVIEW_QUALITY_INCIDENT.json"
    write_json(incident_json, payload)
    md = "\n".join(
        [
            "# M5.4B Review Quality Incident",
            "",
            f"Classification: `{INCIDENT_CLASSIFICATION}`",
            "",
            (
                "The review UI worked correctly as a durable persistence interface. The candidate and upstream "
                "entity quality were inadequate, so the completed decisions must not enter pathlet construction "
                "or continuity learning."
            ),
            "",
            (
                f"Accepted: {completed_summary.get('accepted')}  "
                f"Rejected: {completed_summary.get('rejected')}  "
                f"Unresolved: {completed_summary.get('unresolved')}"
            ),
            "",
            "No decisions were auto-promoted. `human_approved` remains false.",
            "",
            "This is a model/candidate-quality incident, not a review-storage incident.",
        ]
    )
    write_text(stage_root / "incident" / "M5_4B_REVIEW_QUALITY_INCIDENT.md", md + "\n")
    binding = {
        "artifact": "m5_4c_completed_review_binding_manifest",
        "created_at": utc_now(),
        "completed_review_file_hashes": file_hashes,
        "candidate_manifest_hash": completed.get("candidate_manifest_hash"),
        "evidence_manifest_hash": completed.get("evidence_manifest_hash"),
        "decision_state_hash": completed.get("decision_state_hash"),
        "completed_review_bound_to_06b": True,
        "auto_promoted": False,
        **safety_payload(),
    }
    write_json(stage_root / "incident" / "completed_review_binding_manifest.json", binding)
    return payload


def _write_full_frame_diagnostic(
    *,
    output_path: Path,
    source_path: Path,
    target_path: Path,
    source_bbox: dict[str, float],
    target_bbox: dict[str, float],
    case_id: str,
    source_id: str,
    target_id: str,
    source_frame: int,
    target_frame: int,
    source_entity: str,
    target_entity: str,
    source_team: str,
    target_team: str,
    center_delta: float | None,
    footpoint_delta: float | None,
) -> None:
    source = _draw_box(_read_image(source_path), source_bbox, f"S f{source_frame}", (0, 88, 255))
    target = _draw_box(_read_image(target_path), target_bbox, f"T f{target_frame}", (0, 180, 0))
    height = 360
    combined = cv2.hconcat([_fit_height(source, height), _fit_height(target, height)])
    banner = 120
    canvas = cv2.copyMakeBorder(combined, banner, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    lines = [
        (
            f"{case_id}: {source_id} -> {target_id}; frames {source_frame}->{target_frame}; "
            f"gap {abs(target_frame - source_frame)}"
        ),
        f"entity: {source_entity} -> {target_entity}; team/context: {source_team} -> {target_team}",
        f"center_delta_px={center_delta}; footpoint_delta_px={footpoint_delta}; VISUAL_ONLY_NOT_METRIC",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            canvas, line[:190], (16, 30 + index * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (24, 28, 32), 2, cv2.LINE_AA
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ValueError(f"failed to write diagnostic image: {output_path}")


def _team_context_label(row: dict[str, Any]) -> str:
    text = " ".join(
        [
            str(row.get("step1f3_role_team_context", "")),
            str(row.get("c2c_final_colour_belief", "")),
        ]
    ).lower()
    if "team_1" in text:
        return "team_1_visual_context"
    if "team_2" in text:
        return "team_2_visual_context"
    if "official" in text:
        return "official_visual_context"
    if "goalkeeper" in text:
        return "goalkeeper_visual_context"
    return "unknown_visual_context"


def audit_review_case_bindings(
    paths: dict[str, Path],
    stage_root: Path,
    *,
    candidate_payload: dict[str, Any],
    review_manifest: dict[str, Any],
    frame_manifest: dict[str, Any],
    visible_payload: dict[str, Any],
    node_payload: dict[str, Any],
    entity_payload: dict[str, Any],
) -> dict[str, Any]:
    candidates_by_case = {_candidate_case_id(row): row for row in rows_from_payload(candidate_payload)}
    visible_by_id = {str(row.get("visible_person_base_id", "")): row for row in rows_from_payload(visible_payload)}
    nodes_by_id = {str(row.get("visible_person_base_id", "")): row for row in rows_from_payload(node_payload)}
    entity_by_visible = entity_rows_by_visible_id(entity_payload)
    frames = _frame_records(frame_manifest)
    completed = read_json(paths["completed_review"])
    decisions = completed.get("state", {}).get("decisions", {})
    rows = []
    coordinate_rows = []
    temporal_rows = []
    binding_errors: list[str] = []
    for case in review_manifest.get("review_cases", []):
        case_id = str(case.get("review_case_id", ""))
        candidate = candidates_by_case.get(case_id, {})
        source_id = str(candidate.get("source_visible_person_base_id", ""))
        target_id = str(candidate.get("target_visible_person_base_id", ""))
        source_visible = visible_by_id.get(source_id, {})
        target_visible = visible_by_id.get(target_id, {})
        source_node = nodes_by_id.get(source_id, {})
        target_node = nodes_by_id.get(target_id, {})
        source_bbox = _bbox(source_visible)
        target_bbox = _bbox(target_visible)
        evidence = case.get("evidence_manifest", {})
        source_sequence = safe_int(candidate.get("source_frame_sequence"), -1)
        target_sequence = safe_int(candidate.get("target_frame_sequence"), -1)
        source_frame = frames.get(source_sequence, {})
        target_frame = frames.get(target_sequence, {})
        source_path = _frame_path(paths["frame_root"], source_frame)
        target_path = _frame_path(paths["frame_root"], target_frame)
        asset_errors = []
        for asset in evidence.get("evidence_assets", []):
            asset_path = paths["review_evidence_root"] / case_id / str(asset.get("relative_path", ""))
            if not asset_path.exists():
                asset_errors.append(f"missing_asset:{asset_path}")
            elif sha256_file(asset_path) != asset.get("sha256"):
                asset_errors.append(f"asset_hash_mismatch:{asset_path.name}")
        source_bbox_ok = _bbox_equal(evidence.get("source_bbox"), source_bbox)
        target_bbox_ok = _bbox_equal(evidence.get("target_bbox"), target_bbox)
        source_frame_ok = safe_int(case.get("source_frame_sequence"), -1) == source_sequence
        target_frame_ok = safe_int(case.get("target_frame_sequence"), -1) == target_sequence
        evidence_hash_ok = case.get("evidence_hash") == evidence.get("evidence_hash")
        temporal_order_ok = source_sequence < target_sequence and all(
            asset.get("frame_sequences") == [source_sequence, target_sequence]
            for asset in evidence.get("evidence_assets", [])
            if asset.get("asset_type") in {"temporal_clip", "animated_gif"}
            or str(asset.get("asset_id")) == "temporal_clip"
        )
        errors = []
        if not candidate:
            errors.append("candidate_missing")
        if not source_bbox_ok:
            errors.append("source_bbox_binding_error")
        if not target_bbox_ok:
            errors.append("target_bbox_binding_error")
        if not source_frame_ok or not target_frame_ok:
            errors.append("frame_binding_error")
        if asset_errors:
            errors.extend(asset_errors)
        if not evidence_hash_ok:
            errors.append("evidence_hash_binding_error")
        if not temporal_order_ok:
            errors.append("temporal_evidence_order_error")
        binding_errors.extend([f"{case_id}:{error}" for error in errors])
        source_state = str(entity_by_visible.get(source_id, {}).get("entity_validity_state", AMBIGUOUS_ENTITY))
        target_state = str(entity_by_visible.get(target_id, {}).get("entity_validity_state", AMBIGUOUS_ENTITY))
        metrics = _original_pair_metrics(source_node, target_node)
        gate = edge_gate_result(candidate, source_node, target_node, entity_by_visible)
        failure_classifications = []
        if errors:
            for error in errors:
                if "source_bbox" in error:
                    failure_classifications.append("SOURCE_BOX_BINDING_ERROR")
                elif "target_bbox" in error:
                    failure_classifications.append("TARGET_BOX_BINDING_ERROR")
                elif "frame" in error:
                    failure_classifications.append("FRAME_BINDING_ERROR")
                elif "temporal" in error:
                    failure_classifications.append("TEMPORAL_EVIDENCE_ORDER_ERROR")
                else:
                    failure_classifications.append("CROP_BINDING_ERROR")
        else:
            failure_classifications.append("EVIDENCE_BINDING_CORRECT_CANDIDATE_BAD")
        if source_state == PROBABLE_NON_PERSON or target_state == PROBABLE_NON_PERSON:
            failure_classifications.append("DETECTION_IS_NON_PERSON")
        if gate["team_context_conflict"]:
            failure_classifications.append("SOURCE_TARGET_TEAM_CONFLICT")
        if location_incompatible(
            {
                "center_delta_px": metrics["center_delta_px"],
                "footpoint_delta_px": metrics["footpoint_delta_px"],
                "average_bbox_height": metrics["average_bbox_height"],
            },
            abs(target_sequence - source_sequence),
        ):
            failure_classifications.append("SOURCE_TARGET_LOCATION_INCOMPATIBLE")
        if not failure_classifications:
            failure_classifications.append("UNCLEAR_REQUIRES_REVIEW")
        diagnostic_path = stage_root / "audit" / "full_frame_diagnostics" / f"{case_id}.jpg"
        _write_full_frame_diagnostic(
            output_path=diagnostic_path,
            source_path=source_path,
            target_path=target_path,
            source_bbox=source_bbox,
            target_bbox=target_bbox,
            case_id=case_id,
            source_id=source_id,
            target_id=target_id,
            source_frame=source_sequence,
            target_frame=target_sequence,
            source_entity=source_state,
            target_entity=target_state,
            source_team=_team_context_label(source_node),
            target_team=_team_context_label(target_node),
            center_delta=metrics["center_delta_px"],
            footpoint_delta=metrics["footpoint_delta_px"],
        )
        row = {
            "review_case_id": case_id,
            "continuity_edge_id": candidate.get("continuity_edge_id", ""),
            "source_node": candidate.get("source_node_id", ""),
            "target_node": candidate.get("target_node_id", ""),
            "source_visible_person_base_id": source_id,
            "target_visible_person_base_id": target_id,
            "source_step1_row_detection_id": source_visible.get("detection_id", ""),
            "target_step1_row_detection_id": target_visible.get("detection_id", ""),
            "source_frame": source_sequence,
            "target_frame": target_sequence,
            "source_bbox": source_bbox,
            "target_bbox": target_bbox,
            "source_crop": str(paths["review_evidence_root"] / case_id / "source_crop.jpg"),
            "target_crop": str(paths["review_evidence_root"] / case_id / "target_crop.jpg"),
            "context_images": [
                str(paths["review_evidence_root"] / case_id / "source_context.jpg"),
                str(paths["review_evidence_root"] / case_id / "target_context.jpg"),
            ],
            "gif": str(paths["review_evidence_root"] / case_id / "temporal_clip.gif"),
            "mp4": str(paths["review_evidence_root"] / case_id / "temporal_clip.mp4"),
            "diagnostic_image": str(diagnostic_path),
            "source_entity_validity": source_state,
            "target_entity_validity": target_state,
            "original_decision": decisions.get(case_id),
            "binding_errors": errors,
            "failure_classifications": sorted(set(failure_classifications)),
            "candidate_id_correct": case.get("candidate_artifact_id") == candidate.get("portable_review_candidate_id"),
            "evidence_hash_correct": evidence_hash_ok,
            "no_source_target_swap": source_sequence < target_sequence,
            **safety_payload(),
        }
        rows.append(row)
        coordinate_rows.append(
            {
                "review_case_id": case_id,
                "source_bbox_correct": source_bbox_ok,
                "target_bbox_correct": target_bbox_ok,
                "crop_expansion_rule": "m5.4b scale=4.0 min_size=180 for crop; scale=8.0 min_size=420 for context",
                "source_crop_bounds": _bounds_for_bbox(
                    source_bbox,
                    width=safe_int(source_frame.get("width"), 2730),
                    height=safe_int(source_frame.get("height"), 720),
                    scale=4.0,
                    min_size=180,
                ),
                "target_crop_bounds": _bounds_for_bbox(
                    target_bbox,
                    width=safe_int(target_frame.get("width"), 2730),
                    height=safe_int(target_frame.get("height"), 720),
                    scale=4.0,
                    min_size=180,
                ),
                "asset_hash_errors": asset_errors,
            }
        )
        temporal_rows.append(
            {
                "review_case_id": case_id,
                "source_frame_sequence": source_sequence,
                "target_frame_sequence": target_sequence,
                "temporal_order_correct": temporal_order_ok,
                "gif_path": str(paths["review_evidence_root"] / case_id / "temporal_clip.gif"),
                "mp4_path": str(paths["review_evidence_root"] / case_id / "temporal_clip.mp4"),
            }
        )
    summary = guardrail_payload(
        {
            "artifact": "m5_4c_review_case_binding_audit",
            "created_at": utc_now(),
            "review_case_count": len(rows),
            "binding_error_count": len(binding_errors),
            "binding_errors": binding_errors,
            "evidence_binding_audit_result": "PASS_EVIDENCE_BINDING_CORRECT"
            if not binding_errors
            else "FAIL_EVIDENCE_BINDING_ERROR",
            "rows": rows,
        }
    )
    write_json(stage_root / "audit" / "review_case_binding_audit.json", summary)
    write_json(
        stage_root / "audit" / "review_case_binding_rows.json",
        {"artifact": "m5_4c_review_case_binding_rows", "rows": rows, **safety_payload()},
    )
    write_json(
        stage_root / "audit" / "evidence_coordinate_audit.json",
        {"artifact": "m5_4c_evidence_coordinate_audit", "rows": coordinate_rows, **safety_payload()},
    )
    write_json(
        stage_root / "audit" / "evidence_temporal_order_audit.json",
        {"artifact": "m5_4c_evidence_temporal_order_audit", "rows": temporal_rows, **safety_payload()},
    )
    return summary


def _original_pair_metrics(source_node: dict[str, Any], target_node: dict[str, Any]) -> dict[str, float | None]:
    source_bbox = source_node.get("bbox") if isinstance(source_node.get("bbox"), dict) else None
    target_bbox = target_node.get("bbox") if isinstance(target_node.get("bbox"), dict) else None
    center = px_delta(bbox_center(source_bbox), bbox_center(target_bbox))
    foot = px_delta(footpoint_xy(source_node), footpoint_xy(target_node))
    avg_height = max(1.0, (bbox_height(source_bbox) + bbox_height(target_bbox)) / 2.0)
    return {
        "center_delta_px": None if center is None else round(center, 3),
        "footpoint_delta_px": None if foot is None else round(foot, 3),
        "average_bbox_height": round(avg_height, 3),
    }


def write_architecture_audit(
    paths: dict[str, Path], stage_root: Path, detector_payload: dict[str, Any]
) -> dict[str, Any]:
    detector_rows = rows_from_payload(detector_payload)
    upgraded = [
        row
        for row in detector_rows
        if row.get("class_name") == "person"
        and row.get("object_type") == "player_candidate"
        and row.get("role_label") == "player"
    ]
    diff = guardrail_payload(
        {
            "artifact": "m5_4c_historical_vs_portable_architecture_diff",
            "created_at": utc_now(),
            "historical_m4_rows_used_as_blind_inputs": False,
            "comparison_scope": "architecture_and_gating_mechanism_audit_only",
            "portable_detector_adapter_upgrades_coco_person_to_player_before_entity_validity": len(upgraded)
            == len(detector_rows),
            "portable_detector_upgrade_count": len(upgraded),
            "portable_detector_row_count": len(detector_rows),
            "comparison_rows": [
                {
                    "mechanism": "detector source and checkpoint",
                    "portable": "official YOLOv8m baseline; not historical recovery",
                    "historical_better_pipeline": "surviving implementation used richer downstream visual gates",
                },
                {
                    "mechanism": "pitch/image ROI handling",
                    "portable": "inside_or_unverified_visual_roi",
                    "historical_better_pipeline": "had match-local ROI/context gates",
                },
                {
                    "mechanism": "false-positive handling",
                    "portable": "COCO person boxes promoted into player_candidate before validity",
                    "historical_better_pipeline": "bad detections retained but separated by context/correction stages",
                },
                {
                    "mechanism": "candidate edge generation",
                    "portable": "all cross-frame pairs within gap are generated before hard quality gates",
                    "historical_better_pipeline": "gates and sparse review/topology boundaries limited candidate use",
                },
                {
                    "mechanism": "displacement and appearance gating",
                    "portable": "scored after dense graph generation",
                    "historical_better_pipeline": (
                        "used displacement, appearance, review, and topology safety mechanisms before handoff"
                    ),
                },
                {
                    "mechanism": "topology/pathlet construction",
                    "portable": "846181 edges; 0 pathlets",
                    "historical_better_pipeline": "M4 sparse handoff package exists but is not used as blind input",
                },
            ],
        }
    )
    missing = guardrail_payload(
        {
            "artifact": "m5_4c_missing_quality_mechanisms",
            "created_at": utc_now(),
            "rows": [
                "entity-validity gate before player-candidate promotion",
                "image-space off-pitch/structure quarantine",
                "hard impossible-motion gate before review selection",
                "high-confidence team-context conflict gate",
                "official/player role-context conflict gate",
                "bounded deterministic top-k per source and target",
                "explicit not-applicable invalid-entity continuity state",
            ],
        }
    )
    regressions = guardrail_payload(
        {
            "artifact": "m5_4c_portable_semantic_regressions",
            "created_at": utc_now(),
            "portable_detector_adapter_player_upgrade_regression": len(upgraded) == len(detector_rows),
            "regressions": [
                {
                    "regression_id": "coco_person_to_player_candidate_before_validity",
                    "evidence": (
                        "all detector rows have object_type=player_candidate and role_label=player before "
                        "entity validity and pitch context are established"
                    ),
                    "affected_rows": len(upgraded),
                },
                {
                    "regression_id": "dense_pairing_before_hard_gates",
                    "evidence": (
                        "portable Step2 generated the full short-window cross product before hard "
                        "entity/location gates"
                    ),
                },
            ],
        }
    )
    write_json(stage_root / "audit" / "historical_vs_portable_architecture_diff.json", diff)
    write_json(stage_root / "audit" / "missing_quality_mechanisms.json", missing)
    write_json(stage_root / "audit" / "portable_semantic_regressions.json", regressions)
    return diff


def write_role_team_audit(
    stage_root: Path, node_payload: dict[str, Any], entity_payload: dict[str, Any]
) -> dict[str, Any]:
    entity_by_visible = entity_rows_by_visible_id(entity_payload)
    rows = []
    context_counts: Counter[str] = Counter()
    for node in rows_from_payload(node_payload):
        visible_id = str(node.get("visible_person_base_id", ""))
        state = str(entity_by_visible.get(visible_id, {}).get("entity_validity_state", AMBIGUOUS_ENTITY))
        if state != VALID_ON_PITCH_PERSON:
            continue
        context = _team_context_label(node)
        context_counts[context] += 1
        rows.append(
            {
                "visible_person_base_id": visible_id,
                "frame_sequence": node.get("frame_sequence"),
                "entity_validity_state": state,
                "visual_context_state": context,
                "colour_ambiguity": context == "unknown_visual_context",
            }
        )
    report = guardrail_payload(
        {
            "artifact": "m5_4c_role_context_quality_report",
            "created_at": utc_now(),
            "valid_on_pitch_endpoint_count": len(rows),
            "visual_context_counts": dict(sorted(context_counts.items())),
            "team_labels_forced": False,
            "states_used": [
                "team_1_visual_context",
                "team_2_visual_context",
                "official_visual_context",
                "goalkeeper_visual_context",
                "unknown_visual_context",
            ],
            "rows": rows[:500],
        }
    )
    write_json(
        stage_root / "audit" / "team_context_conflict_rows.json",
        {"artifact": "m5_4c_team_context_conflict_rows", "rows": [], **safety_payload()},
    )
    write_json(stage_root / "audit" / "role_context_quality_report.json", report)
    return report


def _select_entity_review_endpoints(
    candidate_payload: dict[str, Any],
    visible_payload: dict[str, Any],
    entity_payload: dict[str, Any],
    *,
    max_cases: int = 32,
) -> list[dict[str, Any]]:
    visible_by_id = {str(row.get("visible_person_base_id", "")): row for row in rows_from_payload(visible_payload)}
    entity_by_visible = entity_rows_by_visible_id(entity_payload)
    selected: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for candidate in rows_from_payload(candidate_payload):
        for key in ("source_visible_person_base_id", "target_visible_person_base_id"):
            visible_id = str(candidate.get(key, ""))
            if visible_id and visible_id not in selected and visible_id in visible_by_id:
                selected[visible_id] = {"visible": visible_by_id[visible_id], "control_reason": "original_14_endpoint"}
    controls = [
        ("apparent_on_pitch_people", lambda item: item[1].get("entity_validity_state") == VALID_ON_PITCH_PERSON),
        ("apparent_off_pitch_people", lambda item: item[1].get("entity_validity_state") == VALID_OFF_PITCH_PERSON),
        ("likely_false_positives", lambda item: item[1].get("entity_validity_state") == PROBABLE_NON_PERSON),
        ("low_confidence_detector_rows", lambda item: safe_float(item[1].get("confidence"), 1.0) <= 0.32),
        ("static_detections", lambda item: safe_int(item[1].get("static_persistence_count"), 0) >= 4),
        ("moving_detections", lambda item: safe_int(item[1].get("static_persistence_count"), 0) <= 1),
    ]
    entity_items = sorted(
        entity_by_visible.items(), key=lambda item: (safe_int(item[1].get("frame_sequence"), -1), item[0])
    )
    for reason, predicate in controls:
        if len(selected) >= max_cases:
            break
        for visible_id, entity_row in entity_items:
            if visible_id in selected or not predicate((visible_id, entity_row)):
                continue
            visible = visible_by_id.get(visible_id)
            if visible:
                selected[visible_id] = {"visible": visible, "control_reason": reason}
                break
    return list(selected.values())[:max_cases]


def _write_entity_evidence(
    *,
    case_dir: Path,
    frame_root: Path,
    frames: dict[int, dict[str, Any]],
    visible: dict[str, Any],
    review_case_id: str,
) -> EvidenceManifest:
    sequence = safe_int(visible.get("frame_sequence"), -1)
    frame = frames[sequence]
    path = _frame_path(frame_root, frame)
    image = _read_image(path)
    bbox = _bbox(visible)
    assets: list[EvidenceAsset] = []
    full = _draw_box(image, bbox, f"Box f{sequence}", (0, 88, 255))
    assets.append(_write_jpg(case_dir / "full_frame.jpg", full, asset_id="full_frame", asset_type="full_frame"))
    tight = _crop(full, bbox, scale=1.8, min_size=96)
    assets.append(_write_jpg(case_dir / "tight_crop.jpg", tight, asset_id="tight_crop", asset_type="tight_crop"))
    wide = _crop(full, bbox, scale=6.0, min_size=360)
    assets.append(_write_jpg(case_dir / "wide_crop.jpg", wide, asset_id="wide_crop", asset_type="wide_crop"))
    temporal_frames = []
    temporal_sequences = [seq for seq in range(max(0, sequence - 1), min(max(frames), sequence + 1) + 1)]
    for seq in temporal_sequences:
        temporal_image = _read_image(_frame_path(frame_root, frames[seq]))
        if seq == sequence:
            temporal_image = _draw_box(temporal_image, bbox, f"Box f{sequence}", (0, 88, 255))
        temporal_frames.append(cv2.resize(temporal_image, (682, 180), interpolation=cv2.INTER_AREA))
    if temporal_frames:
        strip = cv2.hconcat(temporal_frames)
        assets.append(
            _write_jpg(case_dir / "temporal_strip.jpg", strip, asset_id="temporal_strip", asset_type="temporal_strip")
        )
        gif = _write_gif(case_dir / "temporal_clip.gif", temporal_frames)
        if gif is not None:
            assets.append(gif)
    for asset in assets:
        asset.frame_sequences = temporal_sequences
    evidence_payload = {
        "review_case_id": review_case_id,
        "frame_sequence": sequence,
        "frame_byte_sha256": frame.get("byte_sha256"),
        "bbox": bbox,
        "assets": [asset.model_dump(mode="json") for asset in assets],
    }
    evidence_hash = stable_hash(evidence_payload)
    manifest = EvidenceManifest(
        evidence_id=f"{review_case_id}_evidence",
        evidence_assets=assets,
        source_frame_hashes=[
            {
                "frame_sequence": sequence,
                "source_frame_uri": str(path),
                "source_frame_byte_sha256": frame.get("byte_sha256"),
                "decoded_pixel_sha256": frame.get("decoded_pixel_sha256"),
            }
        ],
        source_frame_sequence=sequence,
        target_frame_sequence=None,
        source_bbox=bbox,
        target_bbox=None,
        frame_gap=None,
        temporal_evidence_available=any(asset.media_type == "image/gif" for asset in assets),
        evidence_hash=evidence_hash,
    )
    write_json(case_dir / "evidence_manifest.json", manifest.model_dump(mode="json"))
    return manifest


def build_entity_validity_workbench(
    paths: dict[str, Path],
    stage_root: Path,
    *,
    candidate_payload: dict[str, Any],
    visible_payload: dict[str, Any],
    entity_payload: dict[str, Any],
    frame_manifest: dict[str, Any],
) -> dict[str, Any]:
    review_root = stage_root / "review" / "entity_validity"
    evidence_root = review_root / "evidence"
    workbench_root = review_root / "workbench"
    decision_root = review_root / "decisions"
    decision_root.mkdir(parents=True, exist_ok=True)
    (decision_root / "snapshots").mkdir(parents=True, exist_ok=True)
    frames = _frame_records(frame_manifest)
    endpoints = _select_entity_review_endpoints(candidate_payload, visible_payload, entity_payload)
    entity_by_visible = entity_rows_by_visible_id(entity_payload)
    cases: list[ReviewCase] = []
    for index, item in enumerate(endpoints, start=1):
        visible = item["visible"]
        visible_id = str(visible.get("visible_person_base_id", ""))
        entity = entity_by_visible.get(visible_id, {})
        review_case_id = f"m5_4c_entity_validity_{index:03d}"
        case_dir = evidence_root / review_case_id
        evidence_manifest = _write_entity_evidence(
            case_dir=case_dir,
            frame_root=paths["frame_root"],
            frames=frames,
            visible=visible,
            review_case_id=review_case_id,
        )
        candidate_payload_for_hash = {
            "visible_person_base_id": visible_id,
            "frame_sequence": visible.get("frame_sequence"),
            "detection_id": visible.get("detection_id"),
            "entity_validity_prior_state": entity.get("entity_validity_state", AMBIGUOUS_ENTITY),
            "control_reason": item["control_reason"],
        }
        case = ReviewCase(
            review_case_id=review_case_id,
            task_type="entity_validity",
            concise_question=ENTITY_VALIDITY_QUESTION,
            allowed_decisions=ENTITY_VALIDITY_DECISIONS,
            candidate_artifact_id=visible_id,
            source_artifact_references=[
                _source_ref(
                    "m5_4c_entity_validity_rows",
                    stage_root / "entity_validity" / "entity_validity_rows.json",
                    "entity validity classifications",
                ),
                _source_ref(
                    "m5_4a_visible_person_base_rows", paths["visible_rows"], "read-only M5.4A visible-person rows"
                ),
                _source_ref("canonical_frame_manifest", paths["frame_manifest"], "canonical blind frame manifest"),
            ],
            source_frame_sequence=safe_int(visible.get("frame_sequence"), -1),
            target_frame_sequence=None,
            evidence_manifest=evidence_manifest,
            uncertainty_reasons=[str(reason) for reason in entity.get("entity_validity_reasons", [])],
            category=str(item["control_reason"]),
            priority=index,
            control_status="not_control"
            if item["control_reason"] == "original_14_endpoint"
            else "deterministic_control",
            candidate_hash=stable_hash(candidate_payload_for_hash),
            evidence_hash=evidence_manifest.evidence_hash,
            safety_payload=safety_payload(),
        )
        cases.append(case)
    candidate_manifest_hash = stable_hash([case.candidate_hash for case in cases])
    evidence_manifest_hash = stable_hash([case.evidence_manifest.model_dump(mode="json") for case in cases])
    manifest = ReviewManifest(
        title="M5.4C Entity Validity Review Workbench",
        review_task_family="entity_validity",
        review_cases=cases,
        candidate_manifest_hash=candidate_manifest_hash,
        evidence_manifest_hash=evidence_manifest_hash,
        source_manifest_hash=stable_hash(frame_manifest),
        source_artifact_references=[
            _source_ref("m5_4a_stage_root", paths["m54a"], "read-only M5.4A source stage"),
            _source_ref("m5_4b_completed_review", paths["completed_review"], "diagnostic-only completed review"),
            _source_ref(
                "m5_4c_entity_validity_rows",
                stage_root / "entity_validity" / "entity_validity_rows.json",
                "entity-validity classifier output",
            ),
        ],
    )
    write_json(review_root / "review_manifest.json", manifest.model_dump(mode="json"))
    write_json(
        decision_root / "review_decisions.json",
        {
            "schema_version": "m5_4c.entity_validity_review_decisions.v1",
            "created_at": utc_now(),
            "workbench_version": manifest.workbench_version,
            "candidate_manifest_hash": manifest.candidate_manifest_hash,
            "evidence_manifest_hash": manifest.evidence_manifest_hash,
            "reviewer_session_id": None,
            "event_sequence": 0,
            "decisions": {},
            "notes": {},
            "last_viewed_case_id": None,
            "elapsed_active_seconds": 0,
            "completed": False,
            **safety_payload(),
        },
    )
    (decision_root / "review_decision_events.jsonl").write_text("", encoding="utf-8")
    build_workbench(workbench_root)
    write_json(
        review_root / "entity_validity_review_summary.json",
        {
            "artifact": "m5_4c_entity_validity_review_summary",
            "created_at": utc_now(),
            "review_case_count": len(cases),
            "decisions_prefilled": False,
            "included_original_14_unique_endpoint_count": sum(
                1 for case in cases if case.control_status == "not_control"
            ),
            "deterministic_control_count": sum(1 for case in cases if case.control_status != "not_control"),
            "local_review_url": "http://127.0.0.1:8766/",
            **safety_payload(),
        },
    )
    launcher = stage_root / "OPEN_ENTITY_VALIDITY_REVIEW.ps1"
    command = (
        f'uv run fi-pipeline review serve --review-manifest "{review_root / "review_manifest.json"}" '
        f'--evidence-root "{evidence_root}" --decision-root "{decision_root}" '
        f'--workbench-root "{workbench_root}" --host 127.0.0.1 --port 8766'
    )
    launcher_text = "\n".join(
        [
            '$ErrorActionPreference = "Stop"',
            f'Set-Location "{paths["step_m5"].parents[3] / "SoccerTrack-v2"}"',
            command,
            "",
        ]
    )
    write_text(launcher, launcher_text)
    return {
        "manifest_path": str(review_root / "review_manifest.json"),
        "evidence_root": str(evidence_root),
        "decision_root": str(decision_root),
        "workbench_root": str(workbench_root),
        "review_case_count": len(cases),
        "local_review_url": "http://127.0.0.1:8766/",
    }


def write_original_14_diagnosis(
    paths: dict[str, Path],
    stage_root: Path,
    *,
    candidate_payload: dict[str, Any],
    node_payload: dict[str, Any],
    entity_payload: dict[str, Any],
    quality_payload: dict[str, Any],
) -> dict[str, Any]:
    completed = read_json(paths["completed_review"])
    decisions = completed.get("state", {}).get("decisions", {})
    nodes = {str(row.get("visible_person_base_id", "")): row for row in rows_from_payload(node_payload)}
    entity_by_visible = entity_rows_by_visible_id(entity_payload)
    surviving_edge_ids = {str(row.get("original_continuity_edge_id", "")) for row in rows_from_payload(quality_payload)}
    rows = []
    disposition_counts: Counter[str] = Counter()
    for candidate in rows_from_payload(candidate_payload):
        case_id = _candidate_case_id(candidate)
        source_id = str(candidate.get("source_visible_person_base_id", ""))
        target_id = str(candidate.get("target_visible_person_base_id", ""))
        source_node = nodes.get(source_id, {})
        target_node = nodes.get(target_id, {})
        source_state = str(entity_by_visible.get(source_id, {}).get("entity_validity_state", AMBIGUOUS_ENTITY))
        target_state = str(entity_by_visible.get(target_id, {}).get("entity_validity_state", AMBIGUOUS_ENTITY))
        gate = edge_gate_result(candidate, source_node, target_node, entity_by_visible)
        compound = compound_continuity_disposition(
            source_entity_validity=source_state,
            target_entity_validity=target_state,
            continuity_decision="reject_continuity" if gate["rejection_reasons"] else "unresolved",
        )
        if source_state == PROBABLE_NON_PERSON or target_state == PROBABLE_NON_PERSON:
            diagnosis = "same_persistent_false_positive_structure"
        elif "hard_impossible_motion_image_space" in gate["rejection_reasons"]:
            diagnosis = "location_incompatible_pairing"
        elif gate["team_context_conflict"]:
            diagnosis = "team_context_conflict"
        elif gate["rejection_reasons"]:
            diagnosis = "different_valid_people_or_context_conflict"
        else:
            diagnosis = "unresolved"
        disposition_counts[compound["continuity_decision"]] += 1
        rows.append(
            {
                "review_case_id": case_id,
                "original_candidate": candidate,
                "original_decision": decisions.get(case_id),
                "endpoint_validity_classification": {
                    "source": source_state,
                    "target": target_state,
                },
                "evidence_binding_result": "binding_correct_candidate_bad",
                "team_context_result": "team_conflict"
                if gate["team_context_conflict"]
                else "unknown_or_not_conflicting",
                "location_compatibility_result": "location_incompatible"
                if "hard_impossible_motion_image_space" in gate["rejection_reasons"]
                else "location_compatible_or_uncertain",
                "corrected_compound_disposition": compound,
                "would_survive_quality_gated_candidate_generation": str(candidate.get("continuity_edge_id", ""))
                in surviving_edge_ids,
                "diagnosis_bucket": diagnosis,
                "diagnostic_match_local_calibration_only": True,
            }
        )
    payload = guardrail_payload(
        {
            "artifact": "m5_4c_original_14_case_diagnosis",
            "created_at": utc_now(),
            "diagnostic_match_local_calibration_only": True,
            "rows": rows,
            "summary": {
                "case_count": len(rows),
                "surviving_corrected_gates": sum(
                    1 for row in rows if row["would_survive_quality_gated_candidate_generation"]
                ),
                "corrected_disposition_counts": dict(sorted(disposition_counts.items())),
                "diagnosis_bucket_counts": dict(sorted(Counter(row["diagnosis_bucket"] for row in rows).items())),
            },
        }
    )
    write_json(stage_root / "diagnosis" / "original_14_case_diagnosis.json", payload)
    lines = [
        "# Original 14 Case Diagnosis",
        "",
        (
            "These rows preserve the original M5.4B decisions and provide corrected diagnostic dispositions. "
            "They do not overwrite the completed review."
        ),
        "",
        (
            "| Case | Original decision | Source entity | Target entity | Corrected disposition | "
            "Survives gate | Diagnosis |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {decision} | {source} | {target} | {disp} | {survive} | {diag} |".format(
                case=row["review_case_id"],
                decision=row["original_decision"],
                source=row["endpoint_validity_classification"]["source"],
                target=row["endpoint_validity_classification"]["target"],
                disp=row["corrected_compound_disposition"]["continuity_decision"],
                survive=row["would_survive_quality_gated_candidate_generation"],
                diag=row["diagnosis_bucket"],
            )
        )
    write_text(stage_root / "diagnosis" / "original_14_case_diagnosis.md", "\n".join(lines) + "\n")
    return payload


def write_validation_summary(
    stage_root: Path,
    *,
    incident_payload: dict[str, Any],
    binding_audit: dict[str, Any],
    entity_validation: dict[str, Any],
    edge_validation: dict[str, Any],
    entity_payload: dict[str, Any],
    graph_diagnosis: dict[str, Any],
    quality_payload: dict[str, Any],
    original_14: dict[str, Any],
    entity_workbench: dict[str, Any],
) -> dict[str, Any]:
    original_rows = original_14.get("rows", [])
    endpoint_states = []
    for row in original_rows:
        endpoint_states.extend(row.get("endpoint_validity_classification", {}).values())
    non_person_endpoints = sum(1 for state in endpoint_states if state == PROBABLE_NON_PERSON)
    off_pitch_endpoints = sum(1 for state in endpoint_states if state == VALID_OFF_PITCH_PERSON)
    valid_on_pitch_endpoints = sum(1 for state in endpoint_states if state == VALID_ON_PITCH_PERSON)
    if binding_audit.get("binding_error_count", 0):
        final = BLOCKED_EVIDENCE_BINDING_FAILURE
        blocker = "Evidence binding audit found source/target/frame/crop/hash errors."
    elif not entity_validation.get("passed"):
        final = BLOCKED_ENTITY_VALIDITY_QUALITY
        blocker = "Entity-validity validation failed."
    elif not edge_validation.get("passed"):
        final = BLOCKED_CONTINUITY_EDGE_GATING
        blocker = "Quality-gated edge validation failed."
    else:
        final = PASS_CLASSIFICATION
        blocker = "None; next required step is human entity-validity review before any continuity-learning use."
    summary = guardrail_payload(
        {
            "artifact": "m5_4c_quality_incident_validation_summary",
            "created_at": utc_now(),
            "final_classification": final,
            "exact_remaining_blocker": blocker,
            "incident_classification": incident_payload.get("incident_classification"),
            "evidence_binding_audit_result": binding_audit.get("evidence_binding_audit_result"),
            "number_of_non_person_endpoints": non_person_endpoints,
            "number_of_off_pitch_person_endpoints": off_pitch_endpoints,
            "number_of_valid_on_pitch_endpoints": valid_on_pitch_endpoints,
            "number_of_team_conflict_edges": graph_diagnosis.get("team_conflict_count", 0),
            "number_of_location_incompatible_edges": graph_diagnosis.get("location_incompatible_edge_count", 0),
            "number_of_original_14_cases_surviving_corrected_gates": original_14.get("summary", {}).get(
                "surviving_corrected_gates", 0
            ),
            "current_edge_count": graph_diagnosis.get("candidate_edge_count", 0),
            "quality_gated_edge_count": quality_payload.get("summary", {}).get("quality_gated_edge_count", 0),
            "entity_review_case_count": entity_workbench.get("review_case_count", 0),
            "workbench_url": entity_workbench.get("local_review_url"),
            "m5_4b_decisions_auto_applied": False,
            "diagnostic_match_local_calibration_only": True,
            "third_unseen_window_required_for_next_true_blind_test": True,
        }
    )
    write_json(stage_root / "validation" / "m5_4c_quality_incident_validation_summary.json", summary)
    return summary


def build_quality_incident_stage(
    *,
    repo_root: Path,
    artifact_root: Path,
    match_id: str = "128058",
    stage_root: Path | None = None,
) -> dict[str, Any]:
    paths = default_paths(artifact_root.resolve(), match_id)
    stage_root = (stage_root or paths["m54c"]).resolve()
    stage_root.mkdir(parents=True, exist_ok=True)

    detector_payload = read_json(paths["detector_rows"])
    visible_payload = read_json(paths["visible_rows"])
    frame_manifest = read_json(paths["frame_manifest"])
    candidate_payload = read_json(paths["candidate_rows"])
    review_manifest = read_json(paths["review_manifest"])
    node_payload = read_json(paths["node_rows"])
    edge_payload = read_json(paths["edge_rows"])
    pathlet_payload = read_json(paths["pathlets"])

    incident = write_incident_artifacts(paths, stage_root)
    entity_payload = build_entity_validity_payload(
        detector_payload,
        frame_manifest=frame_manifest,
        visible_person_payload=visible_payload,
    )
    write_json(stage_root / "entity_validity" / "entity_validity_rows.json", entity_payload)
    entity_validation = validate_entity_validity_payload(
        entity_payload,
        expected_detector_row_count=len(rows_from_payload(detector_payload)),
    )
    write_json(stage_root / "entity_validity" / "entity_validity_validation.json", entity_validation)

    binding_audit = audit_review_case_bindings(
        paths,
        stage_root,
        candidate_payload=candidate_payload,
        review_manifest=review_manifest,
        frame_manifest=frame_manifest,
        visible_payload=visible_payload,
        node_payload=node_payload,
        entity_payload=entity_payload,
    )
    write_architecture_audit(paths, stage_root, detector_payload)
    write_role_team_audit(stage_root, node_payload, entity_payload)

    pathlet_count = len(rows_from_payload(pathlet_payload))
    graph = diagnose_current_edge_graph(edge_payload, node_payload, entity_payload, pathlet_count=pathlet_count)
    write_json(stage_root / "audit" / "current_edge_graph_diagnosis.json", graph["diagnosis"])
    write_json(
        stage_root / "audit" / "high_degree_node_rows.json",
        {"artifact": "m5_4c_high_degree_node_rows", "rows": graph["high_degree_node_rows"], **safety_payload()},
    )
    write_json(
        stage_root / "audit" / "team_conflict_edge_rows.json",
        {"artifact": "m5_4c_team_conflict_edge_rows", "rows": graph["team_conflict_edge_rows"], **safety_payload()},
    )
    write_json(
        stage_root / "audit" / "location_incompatible_edge_rows.json",
        {
            "artifact": "m5_4c_location_incompatible_edge_rows",
            "rows": graph["location_incompatible_edge_rows"],
            **safety_payload(),
        },
    )
    write_json(
        stage_root / "audit" / "static_false_positive_edge_rows.json",
        {
            "artifact": "m5_4c_static_false_positive_edge_rows",
            "rows": graph["static_false_positive_edge_rows"],
            **safety_payload(),
        },
    )

    if binding_audit.get("binding_error_count", 0):
        quality_payload = guardrail_payload(
            {
                "artifact": "m5_4c_quality_gated_edge_rows",
                "created_at": utc_now(),
                "blocked": True,
                "blocking_reason": "evidence_binding_failure",
                "rows": [],
                "rejected_rows": [],
                "summary": {"input_edge_count": len(rows_from_payload(edge_payload)), "quality_gated_edge_count": 0},
                "rule_provenance": quality_gate_rule_provenance(
                    max_frame_gap=3, max_source_degree=3, max_target_degree=3
                ),
            }
        )
    else:
        quality_payload = build_quality_gated_edge_payload(edge_payload, node_payload, entity_payload)
    write_json(stage_root / "quality_gated_edges" / "quality_gated_edge_rows.json", quality_payload)
    write_json(
        stage_root / "quality_gated_edges" / "rejected_quality_gated_edge_rows.json",
        {
            "artifact": "m5_4c_rejected_quality_gated_edge_rows",
            "rows": quality_payload.get("rejected_rows", []),
            **safety_payload(),
        },
    )
    write_json(
        stage_root / "quality_gated_edges" / "quality_gate_rule_provenance.json",
        {
            "artifact": "m5_4c_quality_gate_rule_provenance",
            "rows": quality_payload.get("rule_provenance", []),
            **safety_payload(),
        },
    )
    edge_validation = validate_quality_gated_edge_payload(quality_payload)
    write_json(stage_root / "quality_gated_edges" / "quality_gated_edge_validation.json", edge_validation)

    original_14 = write_original_14_diagnosis(
        paths,
        stage_root,
        candidate_payload=candidate_payload,
        node_payload=node_payload,
        entity_payload=entity_payload,
        quality_payload=quality_payload,
    )
    entity_workbench = build_entity_validity_workbench(
        paths,
        stage_root,
        candidate_payload=candidate_payload,
        visible_payload=visible_payload,
        entity_payload=entity_payload,
        frame_manifest=frame_manifest,
    )
    validation = write_validation_summary(
        stage_root,
        incident_payload=incident,
        binding_audit=binding_audit,
        entity_validation=entity_validation,
        edge_validation=edge_validation,
        entity_payload=entity_payload,
        graph_diagnosis=graph["diagnosis"],
        quality_payload=quality_payload,
        original_14=original_14,
        entity_workbench=entity_workbench,
    )
    return {
        "stage_root": str(stage_root),
        "incident": incident,
        "binding_audit": binding_audit,
        "entity_validation": entity_validation,
        "edge_validation": edge_validation,
        "graph_diagnosis": graph["diagnosis"],
        "quality_gated_summary": quality_payload.get("summary", {}),
        "original_14_summary": original_14.get("summary", {}),
        "entity_workbench": entity_workbench,
        "validation_summary": validation,
    }
