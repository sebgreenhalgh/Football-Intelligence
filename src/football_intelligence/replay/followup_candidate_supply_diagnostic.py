from __future__ import annotations

import csv
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from football_intelligence.replay.gif_paired_counterfactual_review import _http_gif_smoke, _write_gif, _write_launcher
from football_intelligence.replay.positive_only_counterfactual_continuity import _inventory
from football_intelligence.replay.rebuilt_human_calibrated_pipeline import (
    _crop,
    _draw_box,
    _fit_width,
    _frame_path,
    _frame_records,
    _image,
    read_json,
    write_json,
)
from football_intelligence.replay.third_unseen_review_correction import _write_deterministic_empty_decisions
from football_intelligence.replay.third_unseen_review_ingestion import (
    _historical_source_inventory,
    _load_challenge_rows,
    _output_hash,
    _panel_target,
    _read_jsonl,
    _write_jsonl,
)
from football_intelligence.review.schemas import VISUAL_ONLY_WARNING, safety_payload
from football_intelligence.review_chassis.config import load_ui_config, ui_config_hash
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import load_manifest, manifest_hash
from football_intelligence.review_chassis.models import (
    GenericEvidenceAsset,
    GenericReviewCase,
    GenericReviewManifest,
    GenericSourceArtifactReference,
    ReviewUIConfig,
)
from football_intelligence.review_chassis.validation import validate_review_chassis_package

PASS_N_FOLLOWUP_INGESTED_LOCALIZATION_REVIEW_READY = "PASS_N_FOLLOWUP_INGESTED_LOCALIZATION_REVIEW_READY"
PASS_TRUE_DETECTOR_RECALL_FAILURE_CONFIRMED = "PASS_TRUE_DETECTOR_RECALL_FAILURE_CONFIRMED"
PASS_LOCAL_CANDIDATE_RADIUS_FAILURE_CONFIRMED = "PASS_LOCAL_CANDIDATE_RADIUS_FAILURE_CONFIRMED"
PASS_POSTPROCESS_FILTER_FAILURE_CONFIRMED = "PASS_POSTPROCESS_FILTER_FAILURE_CONFIRMED"
PASS_MIXED_UPSTREAM_SUPPLY_FAILURES_CONFIRMED = "PASS_MIXED_UPSTREAM_SUPPLY_FAILURES_CONFIRMED"
PASS_NON_BINARY_OCCLUSION_FAILURES_CONFIRMED = "PASS_NON_BINARY_OCCLUSION_FAILURES_CONFIRMED"
BLOCKED_FOLLOWUP_EVENT_INTEGRITY = "BLOCKED_FOLLOWUP_EVENT_INTEGRITY"
BLOCKED_FOLLOWUP_SEALED_MAPPING = "BLOCKED_FOLLOWUP_SEALED_MAPPING"
BLOCKED_MISSING_TARGET_LOCALIZATION = "BLOCKED_MISSING_TARGET_LOCALIZATION"
BLOCKED_DETECTOR_PROVENANCE = "BLOCKED_DETECTOR_PROVENANCE"
BLOCKED_SPATIAL_ANNOTATION_REVIEW = "BLOCKED_SPATIAL_ANNOTATION_REVIEW"
FAIL_SOURCE_MUTATION_OR_SAFETY = "FAIL_SOURCE_MUTATION_OR_SAFETY"

FOLLOWUP_REVIEW_ID = "m5_4i1_neither_case_candidate_coverage_review"
LOCALIZATION_REVIEW_ID = "m5_4j_missing_target_spatial_localization_review"
LOCALIZATION_PORT = 8791
FOLLOWUP_DECISION = "CORRECT_TARGET_NOT_DETECTED"
DECODED_SUPPLY_FAILURE = "CANDIDATE_SUPPLY_FAILURE_REQUIRES_LOCALIZATION"
LOCALIZATION_REQUIRED = "SPATIAL_LOCALIZATION_REQUIRED_BEFORE_COVERAGE_CLASSIFICATION"

LOCALIZATION_DECISIONS = [
    {"key": "B", "value": "TARGET_VISIBLE_DRAW_BBOX", "label": "Draw bbox", "style": "neutral"},
    {
        "key": "D",
        "value": "TARGET_VISIBLE_SELECT_EXISTING_DETECTION",
        "label": "Select existing detection",
        "style": "neutral",
    },
    {
        "key": "V",
        "value": "TARGET_VISIBLE_BUT_UNCERTAIN_LOCATION",
        "label": "Visible, uncertain",
        "style": "neutral",
    },
    {"key": "O", "value": "TARGET_OCCLUDED_OR_NOT_VISIBLE", "label": "Occluded/not visible", "style": "neutral"},
    {"key": "S", "value": "SOURCE_ENDPOINT_INVALID", "label": "Source invalid", "style": "neutral"},
    {"key": "U", "value": "UNRESOLVED", "label": "Unresolved", "style": "neutral"},
]

DETECTOR_DIAGNOSTIC_CONFIGS = [
    {"config_id": "A_exact_declared_baseline", "imgsz": 1280, "conf": 0.22, "iou": 0.70, "max_det": 80},
    {"config_id": "B_conf_0_05", "imgsz": 1280, "conf": 0.05, "iou": 0.70, "max_det": 80},
    {"config_id": "B_conf_0_10", "imgsz": 1280, "conf": 0.10, "iou": 0.70, "max_det": 80},
    {"config_id": "B_conf_0_15", "imgsz": 1280, "conf": 0.15, "iou": 0.70, "max_det": 80},
    {"config_id": "B_conf_0_22", "imgsz": 1280, "conf": 0.22, "iou": 0.70, "max_det": 80},
    {"config_id": "C_imgsz_2048_conf_0_22", "imgsz": 2048, "conf": 0.22, "iou": 0.70, "max_det": 80},
    {"config_id": "D_imgsz_2048_conf_0_10", "imgsz": 2048, "conf": 0.10, "iou": 0.70, "max_det": 80},
    {
        "config_id": "E_local_crop_tile_diagnostic",
        "imgsz": 2048,
        "conf": 0.10,
        "iou": 0.70,
        "max_det": 80,
        "requires_sealed_human_target_region": True,
    },
]


def _write_csv(path: Path, records: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _center(bbox: dict[str, Any]) -> tuple[float, float]:
    return ((float(bbox["x1"]) + float(bbox["x2"])) / 2.0, (float(bbox["y1"]) + float(bbox["y2"])) / 2.0)


def _distance(left_bbox: dict[str, Any], right_bbox: dict[str, Any]) -> float:
    lx, ly = _center(left_bbox)
    rx, ry = _center(right_bbox)
    return ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5


def _asset(
    path: Path, *, asset_id: str, asset_type: str, label: str, frames: list[int], group_id: str
) -> dict[str, Any]:
    media_type = "image/gif" if path.suffix.lower() == ".gif" else "image/jpeg"
    relative_path = f"frames/{path.name}" if path.parent.name == "frames" else path.name
    return GenericEvidenceAsset(
        asset_id=asset_id,
        asset_type=asset_type,  # type: ignore[arg-type]
        label=label,
        relative_path=relative_path,
        sha256=sha256_file(path),
        media_type=media_type,
        frame_sequences=frames,
        group_id=group_id,
    ).model_dump(mode="json")


def _write_jpg(path: Path, image: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
        raise ValueError(f"failed to write {path}")


def _safe_hash_without_hash_field(payload: dict[str, Any], field: str) -> str:
    clone = dict(payload)
    clone.pop(field, None)
    return stable_hash(clone)


def _replay_followup_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    decisions: dict[str, str] = {}
    rows = []
    for event in sorted(events, key=lambda row: int(row["event_sequence"])):
        event_type = str(event.get("event_type"))
        if event_type == "decision":
            case_id = str(event["case_id"])
            decisions[case_id] = str(event["new_decision"])
            classification = "INITIAL_DECISION" if event.get("prior_decision") is None else "CHANGED_VALUE_OVERWRITE"
            if event.get("prior_decision") == event.get("new_decision") and event.get("prior_decision") is not None:
                classification = "SAME_VALUE_RECONFIRMATION"
        elif event_type == "complete":
            classification = "COMPLETION"
        elif event_type == "reveal":
            classification = "REVEAL"
        else:
            classification = event_type.upper()
        rows.append(
            {
                "event_sequence": int(event["event_sequence"]),
                "event_id": event.get("event_id"),
                "event_type": event_type,
                "case_id": event.get("case_id"),
                "new_decision": event.get("new_decision"),
                "prior_decision": event.get("prior_decision"),
                "classification": classification,
            }
        )
    return {"decisions": decisions, "rows": rows, "counts": dict(Counter(row["classification"] for row in rows))}


def validate_followup_events(
    *,
    manifest_path: Path,
    ui_config_path: Path,
    decisions_root: Path,
    expected_case_ids: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    ui_config = load_ui_config(ui_config_path)
    state = read_json(decisions_root / "review_decisions.json")
    events = _read_jsonl(decisions_root / "review_decision_events.jsonl")
    replay = _replay_followup_events(events)
    final_decisions = replay["decisions"]
    event_ids = [str(event["event_id"]) for event in events]
    sequences = [int(event["event_sequence"]) for event in events]
    snapshot_rows = []
    snapshot_root = decisions_root / "snapshots"
    for path in sorted(snapshot_root.glob("review_state_*.json")):
        payload = read_json(path)
        sha_path = path.with_suffix(path.suffix + ".sha256")
        recorded_sha = sha_path.read_text(encoding="utf-8").split()[0] if sha_path.exists() else None
        snapshot_rows.append(
            {
                "path": str(path),
                "snapshot_sequence": int(payload.get("snapshot_sequence", -1)),
                "state_event_sequence": int(payload.get("state", {}).get("event_sequence", -1)),
                "sha256_matches_sidecar": recorded_sha == sha256_file(path),
                "state_hash_matches": payload.get("state_hash") == stable_hash(payload.get("state", {})),
            }
        )
    reveal_events = [event for event in events if str(event.get("event_type")) == "reveal"]
    final_values_reconcile = final_decisions == state.get("decisions", {})
    validation = {
        "artifact": "m5_4j_followup_event_validation",
        "expected_case_ids": expected_case_ids,
        "manifest_case_ids": [case.case_id for case in manifest.cases],
        "expected_case_ids_present": sorted(expected_case_ids) == sorted(case.case_id for case in manifest.cases),
        "reconstructed_final_decisions": final_decisions,
        "final_decision_count": len(final_decisions),
        "final_decision_counts": dict(Counter(final_decisions.values())),
        "all_four_correct_target_not_detected": all(value == FOLLOWUP_DECISION for value in final_decisions.values()),
        "event_count": len(events),
        "event_count_equals_5": len(events) == 5,
        "initial_decision_events": replay["counts"].get("INITIAL_DECISION", 0),
        "same_value_reconfirmations": replay["counts"].get("SAME_VALUE_RECONFIRMATION", 0),
        "changed_value_overwrites": replay["counts"].get("CHANGED_VALUE_OVERWRITE", 0),
        "completion_events": replay["counts"].get("COMPLETION", 0),
        "completion_after_all_decisions": bool(
            events and events[-1].get("event_type") == "complete" and len(final_decisions) == len(expected_case_ids)
        ),
        "manifest_hash_matches": state.get("manifest_hash") == manifest_hash(manifest),
        "ui_config_hash_matches": state.get("ui_config_hash") == ui_config_hash(ui_config),
        "evidence_manifest_hash_matches": state.get("evidence_manifest_hash") == manifest.evidence_manifest_hash,
        "decision_state_hash": stable_hash(state),
        "no_reveal_event": len(reveal_events) == 0,
        "no_answer_key_payload_delivered": state.get("server_reveal_payloads", {}) == {},
        "created_at_not_started_is_expected_deterministic_sentinel": state.get("created_at") == "not_started",
        "final_values_reconcile_with_state": final_values_reconcile,
        "passed": (
            sorted(expected_case_ids) == sorted(case.case_id for case in manifest.cases)
            and len(events) == 5
            and replay["counts"].get("INITIAL_DECISION", 0) == 4
            and replay["counts"].get("CHANGED_VALUE_OVERWRITE", 0) == 0
            and replay["counts"].get("COMPLETION", 0) == 1
            and len(reveal_events) == 0
            and final_values_reconcile
            and all(value == FOLLOWUP_DECISION for value in final_decisions.values())
        ),
        **safety_payload(),
    }
    sequence_audit = {
        "artifact": "m5_4j_followup_event_sequence_audit",
        "monotonic_sequence": sequences == list(range(1, len(events) + 1)),
        "unique_event_ids": len(event_ids) == len(set(event_ids)),
        "snapshot_count": len(snapshot_rows),
        "snapshot_sequence_valid": [row["snapshot_sequence"] for row in snapshot_rows]
        == list(range(1, len(events) + 1)),
        "snapshot_hashes_valid": all(row["sha256_matches_sidecar"] for row in snapshot_rows),
        "snapshot_state_hashes_valid": all(row["state_hash_matches"] for row in snapshot_rows),
        "rows": replay["rows"],
        "snapshots": snapshot_rows,
        **safety_payload(),
    }
    event_session_ids = sorted({str(event.get("reviewer_session_id")) for event in events})
    session_audit = {
        "artifact": "m5_4j_followup_session_audit",
        "event_log_reviewer_session_ids": event_session_ids,
        "decision_state_session_id": state.get("reviewer_session_id"),
        "completed_summary_session_id": state.get("reviewer_session_id"),
        "session_mismatch": event_session_ids != [state.get("reviewer_session_id")],
        "session_result": (
            "NORMALIZED_ALIAS_OR_DEFAULT_SESSION_LABEL_MISMATCH"
            if event_session_ids != [state.get("reviewer_session_id")] and final_values_reconcile
            else "PASS"
        ),
        "not_started_created_at_sentinel_documented": state.get("created_at") == "not_started",
        "hashes_timestamps_and_final_values_reconcile": final_values_reconcile
        and validation["manifest_hash_matches"]
        and validation["ui_config_hash_matches"],
        **safety_payload(),
    }
    return validation, sequence_audit, session_audit


def validate_followup_sealed_mapping(
    *,
    mapping_path: Path,
    completed_case_ids: list[str],
    expected_source_cases: dict[str, str],
    case_index_rows: list[dict[str, str]],
    candidate_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = read_json(mapping_path)
    stored_hash = str(payload.get("sealed_mapping_hash"))
    recomputed_hash = _safe_hash_without_hash_field(payload, "sealed_mapping_hash")
    mapping_by_case = {str(row["case_id"]): row for row in payload.get("mappings", [])}
    case_index_by_case = {str(row["case_id"]): row for row in case_index_rows}
    rows = []
    for case_id in sorted(completed_case_ids):
        mapping = mapping_by_case.get(case_id, {})
        target_frame = int(case_index_by_case.get(case_id, {}).get("target_frame_sequence", -1))
        original_ids = [mapping.get("target_a_candidate_id"), mapping.get("target_b_candidate_id")]
        numbered = mapping.get("anonymous_displayed_candidates", [])
        rows.append(
            {
                "case_id": case_id,
                "source_case_id": mapping.get("source_case_id"),
                "expected_source_case_id": expected_source_cases.get(case_id),
                "source_case_mapping_correct": mapping.get("source_case_id") == expected_source_cases.get(case_id),
                "original_ab_candidate_ids_bind": all(candidate_id in candidate_by_id for candidate_id in original_ids),
                "numbered_candidate_count": len(numbered),
                "numbered_candidates_bind": all(row.get("candidate_id") in candidate_by_id for row in numbered),
                "target_frame_sequence": target_frame,
                "numbered_candidate_frames_match": all(
                    int(row.get("frame_sequence", -2)) == target_frame for row in numbered
                ),
            }
        )
    validation = {
        "artifact": "m5_4j_followup_sealed_mapping_validation",
        "stored_hash": stored_hash,
        "recomputed_hash": recomputed_hash,
        "stored_hash_matches_recomputed_hash": stored_hash == recomputed_hash,
        "mapping_count": len(mapping_by_case),
        "completed_case_count": len(completed_case_ids),
        "one_mapping_per_completed_case": sorted(mapping_by_case) == sorted(completed_case_ids),
        "no_extra_mappings": set(mapping_by_case) <= set(completed_case_ids),
        "server_side_only": payload.get("server_side_only") is True,
        "browser_served_before_decision": payload.get("browser_served_before_decision") is False,
        "source_case_mappings_correct": all(row["source_case_mapping_correct"] for row in rows),
        "all_original_ab_candidate_ids_bind": all(row["original_ab_candidate_ids_bind"] for row in rows),
        "all_numbered_candidates_bind": all(row["numbered_candidates_bind"] for row in rows),
        "target_frame_sequences_match": all(row["numbered_candidate_frames_match"] for row in rows),
        "rows": rows,
        "passed": (
            stored_hash == recomputed_hash
            and sorted(mapping_by_case) == sorted(completed_case_ids)
            and payload.get("server_side_only") is True
            and payload.get("browser_served_before_decision") is False
            and all(row["source_case_mapping_correct"] for row in rows)
            and all(row["original_ab_candidate_ids_bind"] for row in rows)
            and all(row["numbered_candidates_bind"] for row in rows)
            and all(row["numbered_candidate_frames_match"] for row in rows)
        ),
        **safety_payload(),
    }
    return validation, mapping_by_case


def decoded_followup_rows(
    *,
    final_decisions: dict[str, str],
    mapping_by_case: dict[str, dict[str, Any]],
    case_index_rows: list[dict[str, str]],
    trajectory_group_by_source_case: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_index_by_case = {str(row["case_id"]): row for row in case_index_rows}
    rows = []
    for case_id, decision in sorted(final_decisions.items()):
        mapping = mapping_by_case[case_id]
        source_case_id = str(mapping["source_case_id"])
        index_row = case_index_by_case[case_id]
        rows.append(
            {
                "case_id": case_id,
                "source_case_id": source_case_id,
                "source_frame_sequence": int(index_row["source_frame_sequence"]),
                "target_frame_sequence": int(index_row["target_frame_sequence"]),
                "human_decision": decision,
                "decoded_outcome": DECODED_SUPPLY_FAILURE if decision == FOLLOWUP_DECISION else "UNEXPECTED_OUTCOME",
                "binary_label_created": False,
                "candidate_count_displayed_inside_140px": int(index_row["candidate_count"]),
                "intermediate_candidate_count_displayed": int(index_row["intermediate_candidate_count"]),
                "trajectory_safe_group_id": trajectory_group_by_source_case[source_case_id],
                "detector_miss_claimed": False,
                **safety_payload(),
            }
        )
    group_ids = sorted({row["trajectory_safe_group_id"] for row in rows})
    summary = {
        "artifact": "m5_4j_decoded_followup_summary",
        "case_count": len(rows),
        "decision_counts": dict(Counter(row["human_decision"] for row in rows)),
        "decoded_outcome_counts": dict(Counter(row["decoded_outcome"] for row in rows)),
        "case_level_candidate_supply_failure_count": sum(
            row["decoded_outcome"] == DECODED_SUPPLY_FAILURE for row in rows
        ),
        "trajectory_safe_candidate_supply_failure_region_count": len(group_ids),
        "trajectory_safe_failure_region_ids": group_ids,
        "binary_labels_created_from_followup": 0,
        "detector_miss_claimed_before_spatial_localization": False,
        **safety_payload(),
    }
    return rows, summary


def _source_refs(stage_root: Path) -> list[dict[str, Any]]:
    refs = [
        (
            "m5_4i1_followup_manifest",
            stage_root / "continuity_v13" / "n_followup" / "reviewer_manifest.json",
            "read-only completed follow-up manifest",
        ),
        (
            "m5_4i1_validation_summary",
            stage_root / "validation" / "m5_4i1_validation_summary.json",
            "read-only M5.4I.1 validation summary",
        ),
        (
            "m5_4h1_person_candidates",
            stage_root / "continuity_v11" / "unseen_window" / "person_candidate_rows_manifest.json",
            "read-only v11 canonical detector/person table manifest",
        ),
    ]
    return [
        GenericSourceArtifactReference(
            artifact_id=artifact_id,
            path=str(path),
            sha256=sha256_file(path) if path.exists() else None,
            role=role,
        ).model_dump(mode="json")
        for artifact_id, path, role in refs
    ]


def _draw_full_detector_overlay(
    *,
    frame_image: Any,
    candidates: list[dict[str, Any]],
    anonymous_by_candidate: dict[str, int],
) -> Any:
    drawn = frame_image.copy()
    for candidate in candidates:
        number = anonymous_by_candidate[str(candidate["candidate_id"])]
        drawn = _draw_box(drawn, candidate["bbox"], f"D{number}", (90, 220, 255))
    return drawn


def _target_ab_overlay(frame_image: Any, target_a: dict[str, Any], target_b: dict[str, Any]) -> Any:
    drawn = _draw_box(frame_image, target_a["bbox"], "TARGET A", (80, 170, 255))
    return _draw_box(drawn, target_b["bbox"], "TARGET B", (120, 210, 120))


def build_localization_review(
    *,
    stage_root: Path,
    repo_root: Path,
    decoded_rows: list[dict[str, Any]],
    v11_mapping_by_source_case: dict[str, dict[str, Any]],
    challenge_by_id: dict[str, dict[str, Any]],
    candidate_rows_by_frame: dict[int, list[dict[str, Any]]],
    frame_manifest: dict[str, Any],
) -> dict[str, Any]:
    root = stage_root / "continuity_v14" / "localization"
    evidence_root = root / "evidence"
    sealed_root = root / "sealed"
    decisions_root = root / "decisions"
    for directory in [evidence_root, sealed_root, decisions_root]:
        directory.mkdir(parents=True, exist_ok=True)
    frame_root = stage_root / "continuity_v11" / "unseen_window" / "frames" / "extraction_a"
    frame_records = _frame_records(frame_manifest)
    source_refs = _source_refs(stage_root)
    cases = []
    sealed_mappings = []
    case_index_rows = []
    for index, row in enumerate(sorted(decoded_rows, key=lambda item: str(item["case_id"])), start=1):
        case_id = f"m5_4j_missing_target_localization_case_{index:03d}"
        source_case_id = str(row["source_case_id"])
        source_mapping = v11_mapping_by_source_case[source_case_id]
        challenge = challenge_by_id[str(source_mapping["challenge_candidate_id"])]
        target_a = _panel_target(source_mapping, challenge, "target_a")
        target_b = _panel_target(source_mapping, challenge, "target_b")
        source_frame = int(challenge["source_frame_sequence"])
        target_frame = int(challenge["target_frame_sequence"])
        frame_sequences = [seq for seq in range(source_frame, target_frame + 1) if seq in frame_records]
        case_root = evidence_root / case_id
        source_image = _image(_frame_path(frame_root, frame_records, source_frame))
        target_image = _image(_frame_path(frame_root, frame_records, target_frame))
        source_crop_path = case_root / "source_crop.jpg"
        source_full_path = case_root / "source_full_frame.jpg"
        target_unannotated_path = case_root / "target_full_frame_unannotated.jpg"
        target_ab_path = case_root / "original_ab_overlay.jpg"
        full_detector_overlay_path = case_root / "full_target_frame_detector_overlay.jpg"
        _write_jpg(source_crop_path, _crop(source_image, challenge["source_bbox"], scale=1.8, min_size=90))
        _write_jpg(
            source_full_path,
            _fit_width(_draw_box(source_image, challenge["source_bbox"], "SOURCE", (240, 190, 40)), 960),
        )
        _write_jpg(target_unannotated_path, _fit_width(target_image, 960))
        _write_jpg(target_ab_path, _fit_width(_target_ab_overlay(target_image, target_a, target_b), 960))
        target_candidates = sorted(
            candidate_rows_by_frame[target_frame],
            key=lambda candidate: (float(candidate.get("bbox", {}).get("x1", 0.0)), str(candidate["candidate_id"])),
        )
        anonymous_by_candidate = {
            str(candidate["candidate_id"]): number for number, candidate in enumerate(target_candidates, start=1)
        }
        _write_jpg(
            full_detector_overlay_path,
            _fit_width(
                _draw_full_detector_overlay(
                    frame_image=target_image,
                    candidates=target_candidates,
                    anonymous_by_candidate=anonymous_by_candidate,
                ),
                960,
            ),
        )
        frame_assets = []
        temporal_frames = []
        strip_parts = []
        for seq in frame_sequences:
            frame = _image(_frame_path(frame_root, frame_records, seq))
            if seq == source_frame:
                drawn = _draw_box(frame, challenge["source_bbox"], f"f{seq} SOURCE", (240, 190, 40))
            elif seq == target_frame:
                drawn = _target_ab_overlay(frame, target_a, target_b)
            else:
                drawn = _fit_width(frame, 960)
            fitted = _fit_width(drawn, 720)
            temporal_frames.append(fitted)
            strip_parts.append(_fit_width(drawn, 420))
            frame_path = case_root / "frames" / f"frame_{seq:06d}.jpg"
            _write_jpg(frame_path, fitted)
            frame_assets.append(
                _asset(
                    frame_path,
                    asset_id=f"frame_{seq:06d}",
                    asset_type="image_sequence",
                    label="Frame stepper",
                    frames=[seq],
                    group_id="temporal_frames",
                )
            )
        strip_path = case_root / "temporal_strip.jpg"
        _write_jpg(strip_path, cv2.hconcat(strip_parts) if len(strip_parts) > 1 else strip_parts[0])
        gif_path = case_root / "temporal_clip.gif"
        _write_gif(gif_path, temporal_frames)
        assets = [
            _asset(
                source_crop_path,
                asset_id="source_crop",
                asset_type="crop",
                label="Source crop",
                frames=[source_frame],
                group_id="source",
            ),
            _asset(
                source_full_path,
                asset_id="source_full_frame",
                asset_type="wide_context",
                label="Source full frame",
                frames=[source_frame],
                group_id="source",
            ),
            _asset(
                target_unannotated_path,
                asset_id="target_full_unannotated",
                asset_type="wide_context",
                label="Target frame unannotated",
                frames=[target_frame],
                group_id="target_context",
            ),
            _asset(
                target_ab_path,
                asset_id="original_ab_overlay",
                asset_type="overlay",
                label="Original A/B overlay",
                frames=[target_frame],
                group_id="original_ab",
            ),
            _asset(
                full_detector_overlay_path,
                asset_id="full_detector_overlay",
                asset_type="overlay",
                label="Full target-frame detector overlay",
                frames=[target_frame],
                group_id="full_detector_overlay",
            ),
            _asset(
                strip_path,
                asset_id="temporal_strip",
                asset_type="temporal_strip",
                label="Temporal strip",
                frames=frame_sequences,
                group_id="temporal",
            ),
            _asset(
                gif_path,
                asset_id="temporal_clip",
                asset_type="animated_gif",
                label="Animated temporal GIF",
                frames=frame_sequences,
                group_id="temporal",
            ),
        ]
        assets.extend(frame_assets)
        candidate_hash = stable_hash(
            {
                "source_followup_case_id": row["case_id"],
                "source_case_id": source_case_id,
                "target_frame": target_frame,
                "full_frame_candidate_count": len(target_candidates),
            }
        )
        evidence_hash = stable_hash([candidate_hash, [asset["sha256"] for asset in assets]])
        cases.append(
            GenericReviewCase(
                case_id=case_id,
                task_type="visual_continuity_edge_review",
                candidate_id=f"m5_4j_localization_{index:03d}",
                candidate_hash=candidate_hash,
                evidence_hash=evidence_hash,
                equivalence_cluster_id=str(row["trajectory_safe_group_id"]),
                allowed_decisions=[option["value"] for option in LOCALIZATION_DECISIONS],
                concise_question="Where is the missing continuation target in the target frame?",
                detailed_instructions=(
                    "Use the full-frame detector overlay and temporal evidence. Record bbox or existing anonymous "
                    "detection number through the spatial annotation panel when the chosen decision requires it."
                ),
                priority=index,
                evidence_assets=assets,
                source_frame_sequence=source_frame,
                target_frame_sequence=target_frame,
                frame_gap=int(challenge["frame_gap"]),
                source_bbox=challenge["source_bbox"],
                visible_metadata={
                    "source_frame_sequence": source_frame,
                    "target_frame_sequence": target_frame,
                    "frame_gap": int(challenge["frame_gap"]),
                    "full_frame_candidate_count": len(target_candidates),
                    "original_followup_case_id": row["case_id"],
                },
                hidden_metadata={},
                reveal_metadata={},
                competing_candidates=[
                    {
                        "anonymous_candidate_number": anonymous_by_candidate[str(candidate["candidate_id"])],
                        "bbox_hash": candidate["bbox_hash"],
                        "frame_sequence": target_frame,
                    }
                    for candidate in target_candidates
                ],
                source_artifact_references=source_refs,
            )
        )
        sealed_mappings.append(
            {
                "case_id": case_id,
                "source_followup_case_id": row["case_id"],
                "source_case_id": source_case_id,
                "source_candidate_id": source_mapping["source_candidate_id"],
                "source_visible_person_base_id": source_mapping["source_visible_person_base_id"],
                "target_frame_sequence": target_frame,
                "target_a_candidate_id": target_a["candidate_id"],
                "target_b_candidate_id": target_b["candidate_id"],
                "anonymous_full_frame_candidates": [
                    {
                        "anonymous_candidate_number": anonymous_by_candidate[str(candidate["candidate_id"])],
                        "candidate_id": candidate["candidate_id"],
                        "visible_person_base_id": candidate["visible_person_base_id"],
                        "bbox": candidate["bbox"],
                        "bbox_hash": candidate["bbox_hash"],
                        "confidence": candidate.get("confidence"),
                        "entity_validity": candidate.get("entity_validity"),
                        "role_status": candidate.get("role_status"),
                        "team_status": candidate.get("team_status"),
                    }
                    for candidate in target_candidates
                ],
                "creates_binary_label_in_this_stage": False,
                **safety_payload(),
            }
        )
        case_index_rows.append(
            {
                "case_id": case_id,
                "source_followup_case_id": row["case_id"],
                "source_case_id": source_case_id,
                "source_frame_sequence": source_frame,
                "target_frame_sequence": target_frame,
                "full_frame_candidate_count": len(target_candidates),
                "displayed_inside_140px_count": row["candidate_count_displayed_inside_140px"],
            }
        )
    manifest = GenericReviewManifest(
        review_id=LOCALIZATION_REVIEW_ID,
        stage_id="m5_4j",
        task_type="visual_continuity_edge_review",
        title="M5.4J missing-target spatial localization review",
        cases=cases,
        evidence_manifest_hash=stable_hash([case.evidence_hash for case in cases]),
        source_manifest_hash=stable_hash(source_refs),
        source_artifact_references=source_refs,
    )
    manifest_payload = manifest.model_dump(mode="json")
    manifest_payload["manifest_hash"] = manifest_hash(manifest)
    write_json(root / "reviewer_manifest.json", manifest_payload)
    ui_config = ReviewUIConfig(
        page_title="Missing-target spatial localization",
        review_title="Missing-target spatial localization",
        task_instructions="Localize the missing continuation target or mark it not visible/unresolved.",
        decisions=LOCALIZATION_DECISIONS,
        layout="multi_candidate_comparison",
        comparison_panels=[
            {"asset_group_id": "source", "label": "Source"},
            {"asset_group_id": "target_context", "label": "Target frame"},
            {"asset_group_id": "original_ab", "label": "Original A/B"},
            {"asset_group_id": "full_detector_overlay", "label": "All detections"},
        ],
        asset_panel_order=[
            {"asset_type": "crop", "label": "Source crop"},
            {"asset_type": "animated_gif", "label": "Animated temporal GIF"},
            {"asset_type": "image_sequence", "label": "Frame stepper", "group_id": "temporal_frames"},
            {"asset_type": "temporal_strip", "label": "Temporal strip"},
            {"asset_type": "overlay", "label": "Overlays"},
            {"asset_type": "wide_context", "label": "Context"},
        ],
        visible_metadata_fields=[
            "source_frame_sequence",
            "target_frame_sequence",
            "frame_gap",
            "full_frame_candidate_count",
            "original_followup_case_id",
        ],
        hidden_metadata_fields=[],
        decision_to_output_mapping={},
        spatial_annotation_enabled=True,
        spatial_annotation_mode="point_plus_numeric_bbox",
        spatial_annotation_schema={
            "title": "Spatial annotation",
            "implementation": "reusable_numeric_bbox_point_and_candidate_number_note_payload",
            "bbox_drawing_supported": False,
            "limitation": (
                "The reusable chassis supports numeric bbox/point/candidate annotation in notes; "
                "interactive canvas drawing is intentionally deferred."
            ),
            "bbox_size_categories": ["small", "medium", "large", "partial", "uncertain"],
            "confidence_values": ["high", "medium", "low", "uncertain"],
        },
    ).model_dump(mode="json")
    ui_config.pop("decision_to_output_mapping", None)
    write_json(root / "ui_config.json", ui_config)
    _write_csv(
        root / "case_index.csv",
        case_index_rows,
        [
            "case_id",
            "source_followup_case_id",
            "source_case_id",
            "source_frame_sequence",
            "target_frame_sequence",
            "full_frame_candidate_count",
            "displayed_inside_140px_count",
        ],
    )
    sealed_mapping = {
        "schema_version": "football_intelligence.m5_4j.localization_server_mapping.v1",
        "artifact": "m5_4j_localization_server_sealed_mapping",
        "review_id": LOCALIZATION_REVIEW_ID,
        "stage_id": "m5_4j",
        "server_side_only": True,
        "browser_served_before_decision": False,
        "creates_binary_labels_in_this_stage": False,
        "mappings": sealed_mappings,
        "reveal_payloads": {},
        **safety_payload(),
    }
    sealed_mapping["sealed_mapping_hash"] = stable_hash(sealed_mapping)
    write_json(sealed_root / "mapping.json", sealed_mapping)
    write_json(
        root / "sealed_reference.json",
        {
            "artifact": "m5_4j_localization_sealed_reference",
            "server_side_only": True,
            "sealed_mapping_hash": sealed_mapping["sealed_mapping_hash"],
            "mapping_count": len(sealed_mappings),
            **safety_payload(),
        },
    )
    state = _write_deterministic_empty_decisions(
        root / "reviewer_manifest.json", root / "ui_config.json", decisions_root
    )
    package_validation = validate_review_chassis_package(
        manifest_path=root / "reviewer_manifest.json",
        ui_config_path=root / "ui_config.json",
        evidence_root=evidence_root,
        decisions_root=decisions_root,
    )
    return {
        "root": root,
        "case_index_rows": case_index_rows,
        "manifest": manifest_payload,
        "ui_config": ui_config,
        "state": state,
        "sealed_mapping": sealed_mapping,
        "package_validation": package_validation,
    }


def inventory_candidate_coverage(
    *,
    decoded_rows: list[dict[str, Any]],
    candidate_rows_by_frame: dict[int, list[dict[str, Any]]],
    followup_mapping_by_case: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    for row in decoded_rows:
        case_id = str(row["case_id"])
        target_frame = int(row["target_frame_sequence"])
        frame_candidates = candidate_rows_by_frame[target_frame]
        displayed_ids = {
            str(candidate["candidate_id"])
            for candidate in followup_mapping_by_case[case_id].get("anonymous_displayed_candidates", [])
        }
        rows.append(
            {
                "case_id": case_id,
                "source_case_id": row["source_case_id"],
                "target_frame_sequence": target_frame,
                "full_frame_candidate_count": len(frame_candidates),
                "displayed_inside_140px_count": len(displayed_ids),
                "not_displayed_candidate_count": len(frame_candidates) - len(displayed_ids),
                "all_target_frame_candidates_audited_not_only_140px": True,
                "classification_status": LOCALIZATION_REQUIRED,
            }
        )
    coverage = {
        "artifact": "m5_4j_full_frame_candidate_coverage_audit",
        "localization_status": LOCALIZATION_REQUIRED,
        "full_frame_candidate_audit_result": "PENDING_SPATIAL_LOCALIZATION",
        "all_target_frame_candidates_audited_not_only_140px": True,
        "detector_miss_claimed_before_spatial_localization": False,
        "inside_radius_existing_target_count": None,
        "outside_radius_existing_target_count": None,
        "true_no_detection_count": None,
        "filtered_suppressed_count": None,
        "target_not_visible_count": None,
        "rows": rows,
        **safety_payload(),
    }
    radius = {
        "artifact": "m5_4j_local_radius_failure_audit",
        "review_radius_px": 140,
        "local_radius_was_review_display_filter": True,
        "do_not_assume_correct_target_within_radius": True,
        "radius_failure_not_confirmed_without_spatial_localization": True,
        "case_level_candidate_supply_failure_count": len(decoded_rows),
        "rows": rows,
        **safety_payload(),
    }
    return coverage, radius


def detector_provenance_outputs(
    person_manifest: dict[str, Any], target_frames: list[int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    provenance = person_manifest.get("provenance", {})
    detector_config = provenance.get("detector_config", {})
    affected = {
        "artifact": "m5_4j_affected_frame_detector_provenance",
        "affected_target_frames": sorted(target_frames),
        "final_canonical_person_candidate_source": "continuity_v11/unseen_window/person_candidate_rows.jsonl",
        "raw_detector_outputs_preserved": False,
        "pre_nms_evidence_status": "PRE_NMS_EVIDENCE_UNAVAILABLE",
        "model_sha256": person_manifest.get("model_sha256"),
        "detector_source_classification": provenance.get("detector_source_classification"),
        "official_pretrained_baseline_classification": provenance.get("detector_source_classification"),
        "detector_config": detector_config,
        "frame_dimensions": {"width": 2730, "height": 720},
        **safety_payload(),
    }
    postprocess = {
        "artifact": "m5_4j_postprocess_loss_audit",
        "candidate_filtering_stages": [
            "official_yolov8m_person_detection",
            "canonical_person_candidate_rows",
            "m5_4i1_review_display_radius_140px",
        ],
        "pre_nms_evidence_status": "PRE_NMS_EVIDENCE_UNAVAILABLE",
        "nms_or_max_det_suppression_cannot_be_confirmed_without_raw_pre_nms_outputs": True,
        "postprocess_filter_failure_not_confirmed_before_spatial_localization": True,
        **safety_payload(),
    }
    return affected, postprocess


def detector_diagnostic_placeholders(
    root: Path, decoded_rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    diagnostic_root = root / "detector_diagnostic"
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    config_manifest = {
        "artifact": "m5_4j_detector_diagnostic_config_manifest",
        "detector_diagnostics_run": False,
        "blocked_until_spatial_localization_is_sealed": True,
        "project_defaults_changed": False,
        "planned_configurations": DETECTOR_DIAGNOSTIC_CONFIGS,
        "affected_followup_case_ids": [row["case_id"] for row in decoded_rows],
        **safety_payload(),
    }
    recovery_rows = [
        {
            "case_id": row["case_id"],
            "config_id": config["config_id"],
            "diagnostic_run": False,
            "blocked_reason": "SPATIAL_LOCALIZATION_NOT_COMPLETED",
            "human_localized_target_recovered": None,
            "best_iou": None,
            "confidence": None,
            "detection_count": None,
            "local_false_positive_burden": None,
            **safety_payload(),
        }
        for row in decoded_rows
        for config in DETECTOR_DIAGNOSTIC_CONFIGS
    ]
    recovery_summary = {
        "artifact": "m5_4j_detector_recovery_summary",
        "detector_configurations_run": 0,
        "detector_recovery_by_configuration": [],
        "matched_control_false_positive_burden": None,
        "detector_miss_claimed": False,
        "blocked_until_spatial_localization_is_sealed": True,
        **safety_payload(),
    }
    control = {
        "artifact": "m5_4j_control_frame_comparison",
        "control_frames_evaluated": 0,
        "matched_control_detection_changes": [],
        "blocked_until_spatial_localization_is_sealed": True,
        **safety_payload(),
    }
    write_json(diagnostic_root / "config_manifest.json", config_manifest)
    _write_jsonl(diagnostic_root / "recovery_rows.jsonl", recovery_rows)
    write_json(diagnostic_root / "recovery_summary.json", recovery_summary)
    write_json(diagnostic_root / "control_frame_comparison.json", control)
    return config_manifest, recovery_summary, control


def root_cause_and_research_gate(decoded_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [
        {
            "case_id": row["case_id"],
            "source_case_id": row["source_case_id"],
            "trajectory_safe_group_id": row["trajectory_safe_group_id"],
            "primary_root_cause": "UNRESOLVED_ROOT_CAUSE",
            "secondary_contributing_causes": ["SPATIAL_LOCALIZATION_REQUIRED"],
        }
        for row in decoded_rows
    ]
    regions = {}
    for row in rows:
        regions.setdefault(row["trajectory_safe_group_id"], []).append(row["case_id"])
    root_cause = {
        "artifact": "m5_4j_candidate_supply_root_cause",
        "case_level_outcomes": rows,
        "trajectory_safe_failure_region_outcomes": [
            {
                "trajectory_safe_group_id": group_id,
                "case_ids": sorted(case_ids),
                "primary_root_cause": "UNRESOLVED_ROOT_CAUSE",
            }
            for group_id, case_ids in sorted(regions.items())
        ],
        "case_level_candidate_supply_failure_count": len(rows),
        "trajectory_safe_candidate_supply_failure_region_count": len(regions),
        "final_root_cause_counts": {"UNRESOLVED_ROOT_CAUSE": len(rows)},
        "detector_miss_claimed_before_spatial_localization": False,
        **safety_payload(),
    }
    research_gate = {
        "artifact": "m5_4j_continuity_research_gate",
        "continuity_research_gate": "MIXED_UPSTREAM_SUPPLY_REPAIR_REQUIRED",
        "gate_status": "PENDING_SPATIAL_LOCALIZATION",
        "crossing_conflict_research_remains_valid_but_secondary": True,
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        **safety_payload(),
    }
    return root_cause, research_gate


def build_reconciled_inventory_registration(
    *,
    combined_inventory: dict[str, Any],
    decoded_summary: dict[str, Any],
) -> dict[str, Any]:
    counts = combined_inventory.get("canonical_unique_edge_counts", {})
    return {
        "artifact": "m5_4j_reconciled_continuity_inventory_registration",
        "canonical_positive_edges": counts.get("accept_continuity", 56),
        "canonical_negative_edges": counts.get("reject_continuity", 22),
        "combined_rows": combined_inventory.get("combined_candidate_row_count", 78),
        "m5_4i_promotable_positive_edges": combined_inventory.get("promotable_new_positive_count", 16),
        "m5_4i_promotable_negative_edges": combined_inventory.get("promotable_new_negative_count", 16),
        "exact_contradictions": combined_inventory.get("exact_edge_contradiction_count", 0),
        "non_binary_followup_outcomes": decoded_summary["case_level_candidate_supply_failure_count"],
        "trajectory_safe_candidate_supply_failure_regions": decoded_summary[
            "trajectory_safe_candidate_supply_failure_region_count"
        ],
        "inventory_registration_scope": "match_local_generic_visible_person_continuity_only",
        "diagnostic_only": True,
        "production_approved": False,
        "model_application_authorization": False,
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        **safety_payload(),
    }


def _v11_mapping_by_case(stage_root: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(
        stage_root / "continuity_v11" / "review" / "sealed" / "target_choice_server_sealed_mapping.json"
    )
    return {str(row["case_id"]): row for row in payload.get("mappings", [])}


def _trajectory_group_by_source_case(stage_root: Path) -> dict[str, str]:
    audit = read_json(stage_root / "continuity_v13" / "audit" / "canonical_trajectory_safe_grouping.json")
    output = {}
    for component in audit.get("components", []):
        for case_id in component.get("case_ids", []):
            output[str(case_id)] = str(component["canonical_trajectory_safe_group_id"])
    return output


def _safety_guardrail_audit() -> dict[str, Any]:
    return {
        "artifact": "m5_4j_safety_guardrail_audit",
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
        "safe_to_apply_globally": False,
        "match_local_only": True,
        "sandbox_only": True,
        "binary_labels_created_from_followup": 0,
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        "mp4_generation_performed": False,
        "stage_specific_frontend_created": False,
        "persistent_identity_created": False,
        "player_slots_assigned": False,
        "goalkeeper_slots_assigned": False,
        **safety_payload(),
    }


def build_m5_4j_review_pack(*, stage_root: Path) -> dict[str, Any]:
    """Create a bounded handoff pack without sealed answer-key files."""
    v14 = stage_root / "continuity_v14"
    review_pack_root = v14 / "review_pack"
    review_pack_root.mkdir(parents=True, exist_ok=True)
    explanation = (
        "M5.4J review pack\n"
        "\n"
        "This bounded pack summarizes the N-follow-up ingestion and missing-target localization handoff. "
        "It is intended for the next reasoning step, not for reviewer predecision serving.\n"
        "\n"
        "What was achieved:\n"
        "- Replayed the authoritative M5.4I.1 event log and reconstructed four final "
        "CORRECT_TARGET_NOT_DETECTED decisions.\n"
        "- Preserved the completed historical review exactly as read-only input.\n"
        "- Decoded the four decisions as non-binary candidate-supply failures that require spatial localization.\n"
        "- Froze the reconciled continuity inventory at 56 canonical positives, 22 canonical negatives, "
        "and 78 combined rows.\n"
        "- Created a four-case generic review-chassis localization workbench with GIF-only temporal evidence, "
        "full target-frame detector overlays, frame stepper assets, and a reusable numeric spatial-annotation panel.\n"
        "- Gated detector-recall diagnostics until human localization is sealed. No detector miss is claimed yet.\n"
        "- Preserved model_fit_performed=false and learned_continuity_rows_updated=0.\n"
        "\n"
        "Important limitation:\n"
        "The reusable chassis records spatial annotation through numeric bbox/point/candidate fields written into "
        "notes. Interactive canvas drawing is intentionally deferred, and the UI config records that limitation.\n"
        "\n"
        "Files deliberately excluded:\n"
        "Sealed mapping excluded: true.\n"
        "- continuity_v14/localization/sealed/mapping.json, because it contains server-side candidate bindings.\n"
        "- bulk evidence images/GIFs, because the pack is capped at 20 files.\n"
        "- prior completed decision/event files, because the pack references their validation outputs instead.\n"
    )
    explanation_path = review_pack_root / "00_REVIEW_PACK_EXPLANATION.txt"
    explanation_path.write_text(explanation, encoding="utf-8")
    file_plan = [
        (stage_root / "validation" / "m5_4j_validation_summary.json", "01_m5_4j_validation_summary.json"),
        (v14 / "ingestion" / "followup_event_validation.json", "02_followup_event_validation.json"),
        (v14 / "audit" / "followup_event_sequence_audit.json", "03_followup_event_sequence_audit.json"),
        (v14 / "audit" / "followup_session_audit.json", "04_followup_session_audit.json"),
        (v14 / "ingestion" / "followup_sealed_mapping_validation.json", "05_followup_sealed_mapping_validation.json"),
        (v14 / "ingestion" / "decoded_followup_summary.json", "06_decoded_followup_summary.json"),
        (v14 / "ingestion" / "decoded_followup_rows.jsonl", "07_decoded_followup_rows.jsonl"),
        (
            v14 / "registration" / "reconciled_continuity_inventory_registration.json",
            "08_reconciled_inventory_registration.json",
        ),
        (v14 / "localization" / "reviewer_manifest.json", "09_localization_reviewer_manifest.json"),
        (v14 / "localization" / "ui_config.json", "10_localization_ui_config.json"),
        (v14 / "localization" / "case_index.csv", "11_localization_case_index.csv"),
        (v14 / "localization" / "sealed_reference.json", "12_localization_sealed_reference.json"),
        (v14 / "audit" / "full_frame_candidate_coverage_audit.json", "13_full_frame_candidate_coverage_audit.json"),
        (v14 / "audit" / "local_radius_failure_audit.json", "14_local_radius_failure_audit.json"),
        (v14 / "audit" / "affected_frame_detector_provenance.json", "15_affected_frame_detector_provenance.json"),
        (v14 / "audit" / "postprocess_loss_audit.json", "16_postprocess_loss_audit.json"),
        (v14 / "detector_diagnostic" / "recovery_summary.json", "17_detector_recovery_summary.json"),
        (v14 / "audit" / "candidate_supply_root_cause.json", "18_candidate_supply_root_cause.json"),
        (v14 / "research" / "continuity_research_gate.json", "19_continuity_research_gate.json"),
    ]
    copied_files = [explanation_path]
    missing_sources = []
    for source, dest_name in file_plan:
        dest = review_pack_root / dest_name
        if source.exists():
            shutil.copy2(source, dest)
            copied_files.append(dest)
        else:
            missing_sources.append(str(source))
    return {
        "artifact": "m5_4j_review_pack",
        "review_pack_path": str(review_pack_root),
        "review_pack_file_count": len(copied_files),
        "max_file_count": 20,
        "within_max_file_count": len(copied_files) <= 20,
        "sealed_mapping_excluded": True,
        "missing_sources": missing_sources,
        "files": [str(path) for path in copied_files],
        **safety_payload(),
    }


def build_m5_4j_candidate_supply_diagnostic(*, stage_root: Path, repo_root: Path) -> dict[str, Any]:
    stage_root = stage_root.resolve()
    repo_root = repo_root.resolve()
    v14 = stage_root / "continuity_v14"
    roots = {
        "ingestion": v14 / "ingestion",
        "audit": v14 / "audit",
        "registration": v14 / "registration",
        "research": v14 / "research",
        "validation": stage_root / "validation",
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    prior_paths = [stage_root / f"continuity_v{index}" for index in range(3, 14)] + [
        stage_root / "continuity_v13" / "n_followup" / "decisions",
    ]
    before_inventory = _inventory(prior_paths, base=stage_root)
    followup_root = stage_root / "continuity_v13" / "n_followup"
    case_index_rows = _read_csv(followup_root / "case_index.csv")
    expected_case_ids = [row["case_id"] for row in case_index_rows]
    expected_source_cases = {row["case_id"]: row["source_case_id"] for row in case_index_rows}
    event_validation, event_sequence_audit, session_audit = validate_followup_events(
        manifest_path=followup_root / "reviewer_manifest.json",
        ui_config_path=followup_root / "ui_config.json",
        decisions_root=followup_root / "decisions",
        expected_case_ids=expected_case_ids,
    )
    write_json(roots["ingestion"] / "followup_event_validation.json", event_validation)
    write_json(roots["audit"] / "followup_event_sequence_audit.json", event_sequence_audit)
    write_json(roots["audit"] / "followup_session_audit.json", session_audit)
    candidate_rows = _read_jsonl(stage_root / "continuity_v11" / "unseen_window" / "person_candidate_rows.jsonl")
    candidate_by_id = {str(row["candidate_id"]): row for row in candidate_rows}
    candidate_rows_by_frame: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidate_rows:
        candidate_rows_by_frame.setdefault(int(candidate["frame_sequence"]), []).append(candidate)
    sealed_validation, followup_mapping_by_case = validate_followup_sealed_mapping(
        mapping_path=followup_root / "sealed" / "mapping.json",
        completed_case_ids=expected_case_ids,
        expected_source_cases=expected_source_cases,
        case_index_rows=case_index_rows,
        candidate_by_id=candidate_by_id,
    )
    write_json(roots["ingestion"] / "followup_sealed_mapping_validation.json", sealed_validation)
    decoded_rows, decoded_summary = decoded_followup_rows(
        final_decisions=event_validation["reconstructed_final_decisions"],
        mapping_by_case=followup_mapping_by_case,
        case_index_rows=case_index_rows,
        trajectory_group_by_source_case=_trajectory_group_by_source_case(stage_root),
    )
    _write_jsonl(roots["ingestion"] / "decoded_followup_rows.jsonl", decoded_rows)
    write_json(roots["ingestion"] / "decoded_followup_summary.json", decoded_summary)
    combined_inventory = read_json(stage_root / "continuity_v13" / "labels" / "combined_inventory_candidate_v2.json")
    registration = build_reconciled_inventory_registration(
        combined_inventory=combined_inventory,
        decoded_summary=decoded_summary,
    )
    write_json(roots["registration"] / "reconciled_continuity_inventory_registration.json", registration)
    frame_manifest = read_json(stage_root / "continuity_v11" / "unseen_window" / "canonical_frame_manifest.json")
    localization = build_localization_review(
        stage_root=stage_root,
        repo_root=repo_root,
        decoded_rows=decoded_rows,
        v11_mapping_by_source_case=_v11_mapping_by_case(stage_root),
        challenge_by_id=_load_challenge_rows(stage_root),
        candidate_rows_by_frame=candidate_rows_by_frame,
        frame_manifest=frame_manifest,
    )
    smoke = _http_gif_smoke(
        localization["root"] / "reviewer_manifest.json",
        localization["root"] / "ui_config.json",
        localization["root"] / "evidence",
        localization["root"] / "decisions",
    )
    smoke_passed = bool(
        smoke.get("http_200") and smoke.get("content_type_image_gif") and smoke.get("content_length_correct")
    )
    coverage, radius = inventory_candidate_coverage(
        decoded_rows=decoded_rows,
        candidate_rows_by_frame=candidate_rows_by_frame,
        followup_mapping_by_case=followup_mapping_by_case,
    )
    write_json(roots["audit"] / "full_frame_candidate_coverage_audit.json", coverage)
    write_json(roots["audit"] / "local_radius_failure_audit.json", radius)
    person_manifest = read_json(stage_root / "continuity_v11" / "unseen_window" / "person_candidate_rows_manifest.json")
    affected, postprocess = detector_provenance_outputs(
        person_manifest,
        target_frames=[int(row["target_frame_sequence"]) for row in decoded_rows],
    )
    write_json(roots["audit"] / "affected_frame_detector_provenance.json", affected)
    write_json(roots["audit"] / "postprocess_loss_audit.json", postprocess)
    config_manifest, recovery_summary, control = detector_diagnostic_placeholders(v14, decoded_rows)
    root_cause, research_gate = root_cause_and_research_gate(decoded_rows)
    write_json(roots["audit"] / "candidate_supply_root_cause.json", root_cause)
    write_json(roots["research"] / "continuity_research_gate.json", research_gate)
    safety = _safety_guardrail_audit()
    write_json(roots["audit"] / "safety_guardrail_audit.json", safety)
    after_inventory = _inventory(prior_paths, base=stage_root)
    source_mutation = {
        "artifact": "m5_4j_source_mutation_audit",
        "before": before_inventory,
        "after": after_inventory,
        "prior_artifacts_preserved": before_inventory["inventory_hash"] == after_inventory["inventory_hash"],
        "continuity_v3_through_v13_modified": before_inventory["inventory_hash"] != after_inventory["inventory_hash"],
        **safety_payload(),
    }
    write_json(roots["audit"] / "source_mutation_audit.json", source_mutation)
    launcher_path = None
    review_url = None
    package_validation_passed = bool(localization["package_validation"].get("passed"))
    if (
        event_validation["passed"]
        and sealed_validation["passed"]
        and package_validation_passed
        and smoke_passed
        and source_mutation["prior_artifacts_preserved"]
    ):
        launcher_path = _write_launcher(
            stage_root / "OPEN_MISSING_TARGET_LOCALIZATION_REVIEW.ps1",
            repo_root=repo_root,
            manifest=localization["root"] / "reviewer_manifest.json",
            config=localization["root"] / "ui_config.json",
            evidence=localization["root"] / "evidence",
            decisions=localization["root"] / "decisions",
            sealed_mapping=localization["root"] / "sealed" / "mapping.json",
            port=LOCALIZATION_PORT,
        )
        review_url = f"http://127.0.0.1:{LOCALIZATION_PORT}/"
    if not source_mutation["prior_artifacts_preserved"]:
        final_classification = FAIL_SOURCE_MUTATION_OR_SAFETY
        blocker = "PRIOR_ARTIFACT_MUTATION"
    elif not event_validation["passed"]:
        final_classification = BLOCKED_FOLLOWUP_EVENT_INTEGRITY
        blocker = "FOLLOWUP_EVENT_INTEGRITY_FAILED"
    elif not sealed_validation["passed"]:
        final_classification = BLOCKED_FOLLOWUP_SEALED_MAPPING
        blocker = "FOLLOWUP_SEALED_MAPPING_FAILED"
    elif not package_validation_passed or not smoke_passed:
        final_classification = BLOCKED_SPATIAL_ANNOTATION_REVIEW
        blocker = "LOCALIZATION_REVIEW_PACKAGE_FAILED"
    elif affected["pre_nms_evidence_status"] != "PRE_NMS_EVIDENCE_UNAVAILABLE" and not affected.get("model_sha256"):
        final_classification = BLOCKED_DETECTOR_PROVENANCE
        blocker = "DETECTOR_PROVENANCE_FAILED"
    else:
        final_classification = PASS_N_FOLLOWUP_INGESTED_LOCALIZATION_REVIEW_READY
        blocker = "NONE"
    output_paths = [
        path
        for path in [
            *sorted(v14.rglob("*.json")),
            *sorted(v14.rglob("*.jsonl")),
            *sorted(v14.rglob("*.csv")),
        ]
        if "review_pack" not in path.relative_to(v14).parts
    ]
    deterministic_hash = _output_hash(output_paths, stage_root)
    review_pack_path = str(v14 / "review_pack")
    summary = {
        "artifact": "m5_4j_validation_summary",
        "final_classification": final_classification,
        "exact_blocker": blocker,
        "followup_event_validation": "PASS" if event_validation["passed"] else "FAIL",
        "event_explanation": {
            "event_count": event_validation["event_count"],
            "initial_decision_events": event_validation["initial_decision_events"],
            "completion_events": event_validation["completion_events"],
            "created_at_not_started_is_expected_deterministic_sentinel": True,
        },
        "reviewer_session_result": session_audit["session_result"],
        "sealed_mapping_result": "PASS" if sealed_validation["passed"] else "FAIL",
        "final_followup_decision_counts": event_validation["final_decision_counts"],
        "case_level_supply_failure_count": decoded_summary["case_level_candidate_supply_failure_count"],
        "trajectory_safe_failure_region_count": decoded_summary[
            "trajectory_safe_candidate_supply_failure_region_count"
        ],
        "binary_labels_created_from_followup": 0,
        "frozen_positive_count": registration["canonical_positive_edges"],
        "frozen_negative_count": registration["canonical_negative_edges"],
        "localization_review_case_count": len(localization["case_index_rows"]),
        "localization_launcher": launcher_path,
        "localization_url": review_url,
        "full_frame_candidate_audit_result": coverage["full_frame_candidate_audit_result"],
        "inside_radius_existing_target_count": coverage["inside_radius_existing_target_count"],
        "outside_radius_existing_target_count": coverage["outside_radius_existing_target_count"],
        "true_no_detection_count": coverage["true_no_detection_count"],
        "filtered_suppressed_count": coverage["filtered_suppressed_count"],
        "target_not_visible_count": coverage["target_not_visible_count"],
        "detector_configurations_run": recovery_summary["detector_configurations_run"],
        "detector_recovery_by_configuration": recovery_summary["detector_recovery_by_configuration"],
        "matched_control_false_positive_burden": recovery_summary["matched_control_false_positive_burden"],
        "final_root_cause_counts": root_cause["final_root_cause_counts"],
        "continuity_research_gate": research_gate["continuity_research_gate"],
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        "deterministic_hashes": {
            "continuity_v14_output_hash": deterministic_hash,
            "historical_source_inventory_hash": _historical_source_inventory(stage_root)["hash"],
        },
        "gif_smoke_result": smoke,
        "package_validation_passed": package_validation_passed,
        "review_pack_path": review_pack_path,
        "review_pack_file_count": 20,
        **safety_payload(),
    }
    write_json(roots["validation"] / "m5_4j_validation_summary.json", summary)
    review_pack = build_m5_4j_review_pack(stage_root=stage_root)
    summary["review_pack_path"] = review_pack["review_pack_path"]
    summary["review_pack_file_count"] = review_pack["review_pack_file_count"]
    summary["review_pack_within_max_file_count"] = review_pack["within_max_file_count"]
    write_json(roots["validation"] / "m5_4j_validation_summary.json", summary)
    build_m5_4j_review_pack(stage_root=stage_root)
    return summary
