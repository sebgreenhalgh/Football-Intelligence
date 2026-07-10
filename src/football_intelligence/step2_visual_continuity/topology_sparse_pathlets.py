# ruff: noqa: E501

from __future__ import annotations

import json
import mimetypes
import shutil
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

from football_intelligence.paths import CLIP_ID, MATCH_ID, ensure_dir
from football_intelligence.step2_visual_continuity.io import (
    STEP2M1_NODE_ROWS_PATH,
    STEP2M1_OUTPUT_DIR,
    STEP2M2_OUTPUT_DIR,
    STEP2M3_ACCEPTED_EDGE_SUMMARY_PATH,
    STEP2M3_ACCEPTED_EDGES_JSONL_GZ_PATH,
    STEP2M3_FREEZE_CANDIDATE_MANIFEST_PATH,
    STEP2M3_OUTPUT_DIR,
    STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_ROWS_JSONL_GZ_PATH,
    STEP2M3R_GROUP_TOPOLOGY_AUDIT_ROWS_PATH,
    STEP2M3R_OUTPUT_DIR,
    STEP2M3R_REVIEW_PROGRESS_SUMMARY_PATH,
    STEP2M3R_REVIEWED_TOPOLOGY_DECISIONS_PATH,
    STEP2M3S_FREEZE_CANDIDATE_MANIFEST_PATH,
    STEP2M3S_HANDOFF_MANIFEST_PATH,
    STEP2M3S_HANDOFF_SAFE_EDGES_JSONL_GZ_PATH,
    STEP2M3S_OUTPUT_DIR,
    STEP2M3T_EDGE_BURST_ANIMATIONS_DIR,
    STEP2M3T_EDGE_BURST_STRIPS_DIR,
    STEP2M3T_FREEZE_CANDIDATE_MANIFEST_PATH,
    STEP2M3T_HANDOFF_MANIFEST_PATH,
    STEP2M3T_ISSUE_REGISTER_PATH,
    STEP2M3T_OUTPUT_DIR,
    STEP2M3T_PATHLET_ANIMATIONS_DIR,
    STEP2M3T_PATHLET_STRIPS_DIR,
    STEP2M3T_REVIEW_CANDIDATE_ROWS_PATH,
    STEP2M3T_REVIEW_CONTACT_SHEET_PATH,
    STEP2M3T_REVIEW_DECISION_SUMMARY_PATH,
    STEP2M3T_REVIEW_PACK_DIR,
    STEP2M3T_REVIEW_PACK_MANIFEST_PATH,
    STEP2M3T_REVIEW_PROGRESS_SUMMARY_PATH,
    STEP2M3T_REVIEW_UI_HTML_PATH,
    STEP2M3T_REVIEWED_SPARSE_PATHLET_DECISIONS_PATH,
    STEP2M3T_SAFETY_GUARDRAIL_AUDIT_PATH,
    STEP2M3T_SELECTED_SPARSE_EDGE_SAMPLE_PATH,
    STEP2M3T_SELECTED_SPARSE_EDGE_SUMMARY_PATH,
    STEP2M3T_SELECTED_SPARSE_EDGES_JSONL_GZ_PATH,
    STEP2M3T_SPARSE_CANDIDATE_EDGE_SAMPLE_PATH,
    STEP2M3T_SPARSE_CANDIDATE_EDGE_SUMMARY_PATH,
    STEP2M3T_SPARSE_CANDIDATE_EDGES_JSONL_GZ_PATH,
    STEP2M3T_SPARSE_PATHLET_SAMPLE_PATH,
    STEP2M3T_SPARSE_PATHLET_SUMMARY_PATH,
    STEP2M3T_SPARSE_PATHLETS_PATH,
    STEP2M3T_TOPOLOGY_QUARANTINE_SUMMARY_PATH,
    STEP2M3T_TOPOLOGY_QUARANTINED_EDGES_JSONL_GZ_PATH,
    STEP2M3T_TOPOLOGY_QUARANTINED_PATHLETS_PATH,
    STEP2M3T_VALIDATION_SUMMARY_PATH,
    STEP2_VISUAL_CONTINUITY_DIR,
    read_json,
    read_jsonl_gz_rows,
    write_json,
    write_jsonl_gz,
    write_text,
)
from football_intelligence.step2_visual_continuity.schema import (
    DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_FRAMES,
    DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_SECONDS,
    NO_AUTO_PROMOTION,
    PRODUCTION_READY,
    VISUAL_ONLY_WARNING,
    assert_no_forbidden_keys,
    forbidden_keys_present,
    guardrail_stamp,
    rows_from_payload,
    safe_float,
    safe_int,
    utc_iso,
    visual_stamp,
)
from football_intelligence.step2_visual_continuity.topology_qa import (
    cv2,
    decision_rows_from_payload,
    draw_scaled_box,
    horizontal_image_strip,
    node_lookup_from_payload,
    np,
    safe_asset_stem,
    sample_frame_sequences,
    tile_from_frame,
    write_animation_gif,
    write_image,
    write_json_atomic,
)
from football_intelligence.step2_visual_continuity.topology_safe_handoff_subset import (
    M3S_ACCEPT_DECISION,
    decision_indexes,
    load_m3r_reviewed_decisions,
)


M3T_ACCEPT_DECISION = "accept_sparse_pathlet_for_visual_handoff"
M3T_REJECT_DECISION = "reject_or_quarantine_sparse_pathlet"
M3T_UNSURE_DECISION = "unsure_needs_later_review"
M3T_CURRENT_REVIEW_VERSION = "step2m3t_sparse_pathlets_review_v1"
M3T_CURRENT_VISUAL_EVIDENCE_VERSION = "step2m3t_visual_evidence_v1_animation"
M3T_MAX_PATHLET_SPAN_FRAMES = DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_FRAMES
M3T_MAX_PATHLET_SPAN_SECONDS = DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_SECONDS
M3T_REVIEW_TARGET_CARDS = 40
M3T_REVIEW_HARD_MAX_CARDS = 60
M3T_MIN_SELECTION_SCORE = 0.25


def m3t_guardrail_fields() -> dict[str, Any]:
    return {
        "match_local_only": True,
        "safe_to_apply_globally": False,
        "requires_future_match_validation": True,
        "production_ready": PRODUCTION_READY,
        "no_auto_promotion": NO_AUTO_PROMOTION,
        "human_approved": False,
        "no_identity_tracking_performed": True,
        "no_player_slots_assigned": True,
        "no_goalkeeper_slots_assigned": True,
        "no_expected_22_role_states": True,
        "no_exact_count_forcing": True,
        "no_metric_event_tactical_or_physical_performance_analysis": True,
        "official_referee_exclusion_performed": False,
        "bad_detection_rows_deleted": False,
    }


def step2m3t_output_paths() -> dict[str, Path]:
    return {
        "step2m3t_output_dir": STEP2M3T_OUTPUT_DIR,
        "sparse_candidate_edges": STEP2M3T_SPARSE_CANDIDATE_EDGES_JSONL_GZ_PATH,
        "sparse_candidate_edge_summary": STEP2M3T_SPARSE_CANDIDATE_EDGE_SUMMARY_PATH,
        "sparse_candidate_edge_sample": STEP2M3T_SPARSE_CANDIDATE_EDGE_SAMPLE_PATH,
        "selected_sparse_edges": STEP2M3T_SELECTED_SPARSE_EDGES_JSONL_GZ_PATH,
        "selected_sparse_edge_summary": STEP2M3T_SELECTED_SPARSE_EDGE_SUMMARY_PATH,
        "selected_sparse_edge_sample": STEP2M3T_SELECTED_SPARSE_EDGE_SAMPLE_PATH,
        "sparse_pathlets": STEP2M3T_SPARSE_PATHLETS_PATH,
        "sparse_pathlet_summary": STEP2M3T_SPARSE_PATHLET_SUMMARY_PATH,
        "sparse_pathlet_sample": STEP2M3T_SPARSE_PATHLET_SAMPLE_PATH,
        "topology_quarantined_edges": STEP2M3T_TOPOLOGY_QUARANTINED_EDGES_JSONL_GZ_PATH,
        "topology_quarantined_pathlets": STEP2M3T_TOPOLOGY_QUARANTINED_PATHLETS_PATH,
        "topology_quarantine_summary": STEP2M3T_TOPOLOGY_QUARANTINE_SUMMARY_PATH,
        "review_candidates": STEP2M3T_REVIEW_CANDIDATE_ROWS_PATH,
        "review_ui": STEP2M3T_REVIEW_UI_HTML_PATH,
        "review_contact_sheet": STEP2M3T_REVIEW_CONTACT_SHEET_PATH,
        "review_progress": STEP2M3T_REVIEW_PROGRESS_SUMMARY_PATH,
        "review_decision": STEP2M3T_REVIEW_DECISION_SUMMARY_PATH,
        "reviewed_sparse_pathlet_decisions": STEP2M3T_REVIEWED_SPARSE_PATHLET_DECISIONS_PATH,
        "handoff_manifest": STEP2M3T_HANDOFF_MANIFEST_PATH,
        "validation_summary": STEP2M3T_VALIDATION_SUMMARY_PATH,
        "safety_guardrail_audit": STEP2M3T_SAFETY_GUARDRAIL_AUDIT_PATH,
        "issue_register": STEP2M3T_ISSUE_REGISTER_PATH,
        "freeze_candidate_manifest": STEP2M3T_FREEZE_CANDIDATE_MANIFEST_PATH,
        "review_pack_manifest": STEP2M3T_REVIEW_PACK_MANIFEST_PATH,
    }


def assert_m3t_output_path_isolation() -> None:
    m3t_root = STEP2M3T_OUTPUT_DIR.resolve()
    blocked_roots = [
        STEP2M1_OUTPUT_DIR.resolve(),
        STEP2M2_OUTPUT_DIR.resolve(),
        STEP2M3_OUTPUT_DIR.resolve(),
        STEP2M3R_OUTPUT_DIR.resolve(),
        STEP2M3S_OUTPUT_DIR.resolve(),
    ]
    for path in step2m3t_output_paths().values():
        resolved = path.resolve()
        if resolved != m3t_root and m3t_root not in resolved.parents:
            raise ValueError(f"Step2.M3T output path is outside the M3T root: {resolved}")
        if any(resolved == root or root in resolved.parents for root in blocked_roots):
            raise ValueError(f"Step2.M3T output path points inside an earlier Step2 folder: {resolved}")


def m3t_rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(STEP2M3T_OUTPUT_DIR.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def m3t_abs_asset_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else STEP2M3T_OUTPUT_DIR / path


M3T_REVIEW_APPROVAL_FLAGS = [
    "approve_any_identity_tracking",
    "approve_any_player_slot_use",
    "approve_any_goalkeeper_slot_use",
    "approve_any_metric_use",
    "approve_event_or_tactical_analysis",
    "approve_exact_22_or_exact_two_goalkeeper_forcing",
    "approve_official_referee_exclusion",
    "approve_bad_detection_deletion",
    "approve_production_promotion",
]


def read_m3t_reviewed_decisions() -> dict[str, Any]:
    if not STEP2M3T_REVIEWED_SPARSE_PATHLET_DECISIONS_PATH.exists():
        return guardrail_stamp(
            {
                "artifact": "step2m3t_reviewed_sparse_pathlet_decisions",
                "created_at": utc_iso(),
                "source_match_id": MATCH_ID,
                "source_clip_id": CLIP_ID,
                "current_review_version": M3T_CURRENT_REVIEW_VERSION,
                "current_visual_evidence_version": M3T_CURRENT_VISUAL_EVIDENCE_VERSION,
                "reviewed_decision_rows": 0,
                "rows": [],
                **m3t_guardrail_fields(),
            }
        )
    raw_payload = json.loads(STEP2M3T_REVIEWED_SPARSE_PATHLET_DECISIONS_PATH.read_text(encoding="utf-8"))
    decision_rows = decision_rows_from_payload(raw_payload)
    normalized = dict(raw_payload) if isinstance(raw_payload, dict) else {}
    normalized.pop("decisions", None)
    normalized.update(
        {
            "artifact": "step2m3t_reviewed_sparse_pathlet_decisions",
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "current_review_version": normalized.get("current_review_version", M3T_CURRENT_REVIEW_VERSION),
            "current_visual_evidence_version": normalized.get("current_visual_evidence_version", M3T_CURRENT_VISUAL_EVIDENCE_VERSION),
            "reviewed_decision_rows": len(decision_rows),
            "rows": decision_rows,
            **m3t_guardrail_fields(),
        }
    )
    return guardrail_stamp(normalized)


def m3t_candidate_by_review_id(review_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("step2m3t_review_candidate_id", "")): row
        for row in rows_from_payload(review_payload)
        if row.get("step2m3t_review_candidate_id")
    }


def normalize_m3t_decision(decision: str) -> str:
    if decision == "unsure":
        return M3T_UNSURE_DECISION
    return str(decision)


def m3t_review_decision_row(candidate: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    decision = normalize_m3t_decision(str(payload.get("human_review_decision", "")))
    if decision not in {M3T_ACCEPT_DECISION, M3T_REJECT_DECISION, M3T_UNSURE_DECISION}:
        raise ValueError(f"Step2.M3T sparse pathlet review decision is not allowed: {decision}")
    row = {
        "step2m3t_review_candidate_id": str(candidate.get("step2m3t_review_candidate_id", "")),
        "review_subject_type": str(candidate.get("review_subject_type", "")),
        "step2m3t_review_category": str(candidate.get("step2m3t_review_category", "")),
        "pathlet_id": str(candidate.get("pathlet_id", "")),
        "continuity_edge_id": str(candidate.get("continuity_edge_id", "")),
        "accepted_continuity_edge_ids": list(candidate.get("accepted_continuity_edge_ids", []) or []),
        "min_frame_sequence": safe_int(candidate.get("min_frame_sequence", candidate.get("source_frame_sequence")), -1),
        "max_frame_sequence": safe_int(candidate.get("max_frame_sequence", candidate.get("target_frame_sequence")), -1),
        "evidence_type": str(candidate.get("evidence_type", "")),
        "evidence_animation_gif_path": str(candidate.get("evidence_animation_gif_path", "")),
        "evidence_static_strip_path": str(candidate.get("evidence_static_strip_path", "")),
        "sampled_frame_sequences": list(candidate.get("sampled_frame_sequences", []) or []),
        "human_review_decision": decision,
        "reviewer_name": str(payload.get("reviewer_name", "")),
        "notes": str(payload.get("notes", "")),
        "reviewed_at": utc_iso(),
        "human_confirmed": True,
        "current_review_version": M3T_CURRENT_REVIEW_VERSION,
        "review_decisions_collected_with_review_version": M3T_CURRENT_REVIEW_VERSION,
        "current_visual_evidence_version": M3T_CURRENT_VISUAL_EVIDENCE_VERSION,
        "review_decisions_collected_with_visual_evidence_version": M3T_CURRENT_VISUAL_EVIDENCE_VERSION,
        "review_decisions_visual_evidence_version_matches_current": True,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "match_local_only": True,
        "safe_to_apply_globally": False,
        "requires_future_match_validation": True,
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
        "approve_any_identity_tracking": False,
        "approve_any_player_slot_use": False,
        "approve_any_goalkeeper_slot_use": False,
        "approve_any_metric_use": False,
        "approve_event_or_tactical_analysis": False,
        "approve_exact_22_or_exact_two_goalkeeper_forcing": False,
        "approve_official_referee_exclusion": False,
        "approve_bad_detection_deletion": False,
        "approve_production_promotion": False,
        **m3t_guardrail_fields(),
    }
    visual_stamp(row)
    assert_no_forbidden_keys(row)
    return row


def write_m3t_reviewed_decisions_payload(decision_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(decision_rows, key=lambda row: str(row.get("step2m3t_review_candidate_id", "")))
    payload = guardrail_stamp(
        {
            "artifact": "step2m3t_reviewed_sparse_pathlet_decisions",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "current_review_version": M3T_CURRENT_REVIEW_VERSION,
            "current_visual_evidence_version": M3T_CURRENT_VISUAL_EVIDENCE_VERSION,
            "reviewed_decision_rows": len(ordered),
            "rows": ordered,
            **m3t_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(payload)
    write_json_atomic(STEP2M3T_REVIEWED_SPARSE_PATHLET_DECISIONS_PATH, payload)
    return payload


def validate_m3t_reviewed_rows(review_payload: dict[str, Any], reviewed_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = m3t_candidate_by_review_id(review_payload)
    usable_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in decision_rows_from_payload(reviewed_payload):
        candidate_id = str(row.get("step2m3t_review_candidate_id", ""))
        if not candidate_id or candidate_id not in candidates:
            errors.append({"issue_code": "unknown_step2m3t_review_candidate", "candidate_id": candidate_id})
            continue
        if candidate_id in seen:
            errors.append({"issue_code": "duplicate_step2m3t_review_candidate", "candidate_id": candidate_id})
            continue
        seen.add(candidate_id)
        row_errors: list[dict[str, Any]] = []
        decision = str(row.get("human_review_decision", ""))
        if decision not in {M3T_ACCEPT_DECISION, M3T_REJECT_DECISION, M3T_UNSURE_DECISION}:
            row_errors.append({"issue_code": "invalid_step2m3t_review_decision", "candidate_id": candidate_id, "decision": decision})
        review_version = str(row.get("review_decisions_collected_with_review_version", row.get("current_review_version", "")))
        if review_version != M3T_CURRENT_REVIEW_VERSION:
            row_errors.append({"issue_code": "step2m3t_review_version_mismatch", "candidate_id": candidate_id, "observed": review_version, "expected": M3T_CURRENT_REVIEW_VERSION})
        evidence_version = str(row.get("review_decisions_collected_with_visual_evidence_version", row.get("current_visual_evidence_version", "")))
        if evidence_version != M3T_CURRENT_VISUAL_EVIDENCE_VERSION:
            row_errors.append({"issue_code": "step2m3t_visual_evidence_version_mismatch", "candidate_id": candidate_id, "observed": evidence_version, "expected": M3T_CURRENT_VISUAL_EVIDENCE_VERSION})
        if row.get("review_decisions_visual_evidence_version_matches_current") is not True:
            row_errors.append({"issue_code": "step2m3t_visual_evidence_match_flag_false", "candidate_id": candidate_id})
        if row.get("visual_only_warning") != VISUAL_ONLY_WARNING or row.get("do_not_use_for_metrics") is not True:
            row_errors.append({"issue_code": "step2m3t_visual_only_guardrail_missing", "candidate_id": candidate_id})
        if row.get("production_ready") is not False:
            row_errors.append({"issue_code": "step2m3t_production_ready_true", "candidate_id": candidate_id})
        if row.get("no_auto_promotion") is not True:
            row_errors.append({"issue_code": "step2m3t_no_auto_promotion_false", "candidate_id": candidate_id})
        if row.get("human_approved") is not False:
            row_errors.append({"issue_code": "step2m3t_human_approved_true_by_default", "candidate_id": candidate_id})
        enabled_approval_flags = [flag for flag in M3T_REVIEW_APPROVAL_FLAGS if row.get(flag) is not False]
        if enabled_approval_flags:
            row_errors.append({"issue_code": "step2m3t_forbidden_review_approval_flag_enabled", "candidate_id": candidate_id, "flags": enabled_approval_flags})
        forbidden = forbidden_keys_present(row)
        if forbidden:
            row_errors.append({"issue_code": "step2m3t_forbidden_keys_present_in_review_decision", "candidate_id": candidate_id, "forbidden_keys_present": forbidden})
        if row_errors:
            errors.extend(row_errors)
            continue
        usable_rows.append(row)
    return usable_rows, errors


def m3t_review_progress_payload(review_payload: dict[str, Any], reviewed_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    review_rows = rows_from_payload(review_payload)
    reviewed_payload = reviewed_payload or {"rows": []}
    raw_reviewed_rows = decision_rows_from_payload(reviewed_payload)
    usable_rows, validation_errors = validate_m3t_reviewed_rows(review_payload, reviewed_payload)
    decision_counts = Counter(str(row.get("human_review_decision", "")) for row in usable_rows)
    category_counts = Counter(str(row.get("step2m3t_review_category", "")) for row in review_rows)
    candidates = m3t_candidate_by_review_id(review_payload)
    reviewed_by_category = Counter(
        str(candidates.get(str(row.get("step2m3t_review_candidate_id", "")), {}).get("step2m3t_review_category", ""))
        for row in usable_rows
    )
    accepted_by_category = Counter(
        str(candidates.get(str(row.get("step2m3t_review_candidate_id", "")), {}).get("step2m3t_review_category", ""))
        for row in usable_rows
        if row.get("human_review_decision") == M3T_ACCEPT_DECISION
    )
    rejected_by_category = Counter(
        str(candidates.get(str(row.get("step2m3t_review_candidate_id", "")), {}).get("step2m3t_review_category", ""))
        for row in usable_rows
        if row.get("human_review_decision") == M3T_REJECT_DECISION
    )
    unsure_by_category = Counter(
        str(candidates.get(str(row.get("step2m3t_review_candidate_id", "")), {}).get("step2m3t_review_category", ""))
        for row in usable_rows
        if row.get("human_review_decision") == M3T_UNSURE_DECISION
    )
    category_progress = {
        category: {
            "total": category_counts.get(category, 0),
            "reviewed": reviewed_by_category.get(category, 0),
            "accepted": accepted_by_category.get(category, 0),
            "rejected": rejected_by_category.get(category, 0),
            "unsure": unsure_by_category.get(category, 0),
        }
        for category in sorted(category_counts)
    }
    collected_versions = {
        str(row.get("review_decisions_collected_with_review_version", row.get("current_review_version", "")))
        for row in raw_reviewed_rows
        if row.get("review_decisions_collected_with_review_version") or row.get("current_review_version")
    }
    collected_evidence_versions = {
        str(row.get("review_decisions_collected_with_visual_evidence_version", row.get("current_visual_evidence_version", "")))
        for row in raw_reviewed_rows
        if row.get("review_decisions_collected_with_visual_evidence_version") or row.get("current_visual_evidence_version")
    }
    review_version_matches = not raw_reviewed_rows or collected_versions == {M3T_CURRENT_REVIEW_VERSION}
    evidence_version_matches = not raw_reviewed_rows or collected_evidence_versions == {M3T_CURRENT_VISUAL_EVIDENCE_VERSION}
    evidence_available = sum(1 for row in review_rows if row.get("evidence_available") is True)
    animation_available = sum(1 for row in review_rows if row.get("evidence_animation_gif_path"))
    sparse_pathlet_review_completed = len(review_rows) > 0 and len(usable_rows) == len(review_rows) and not validation_errors
    forbidden = sorted(set(forbidden_keys_present(review_payload)) | set(forbidden_keys_present(reviewed_payload)))
    progress = guardrail_stamp(
        {
            "artifact": "step2m3t_review_progress_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "total_review_candidates": len(review_rows),
            "reviewed_candidates": len(usable_rows),
            "persisted_reviewed_decision_rows": len(raw_reviewed_rows),
            "persisted_review_decision_file_exists": STEP2M3T_REVIEWED_SPARSE_PATHLET_DECISIONS_PATH.exists(),
            "persisted_decision_row_count_matches_reviewed_candidates": len(raw_reviewed_rows) == len(usable_rows),
            "accepted_count": decision_counts.get(M3T_ACCEPT_DECISION, 0),
            "rejected_count": decision_counts.get(M3T_REJECT_DECISION, 0),
            "unsure_count": decision_counts.get(M3T_UNSURE_DECISION, 0),
            "sparse_pathlet_review_completed": sparse_pathlet_review_completed,
            "review_category_counts": dict(sorted(category_counts.items())),
            "review_category_progress": category_progress,
            "visual_evidence_available_count": evidence_available,
            "visual_evidence_missing_count": len(review_rows) - evidence_available,
            "animation_evidence_available_count": animation_available,
            "animation_evidence_missing_count": len(review_rows) - animation_available,
            "current_review_version": M3T_CURRENT_REVIEW_VERSION,
            "review_decisions_collected_with_review_version": sorted(collected_versions),
            "review_decisions_version_matches_current": review_version_matches,
            "current_visual_evidence_version": M3T_CURRENT_VISUAL_EVIDENCE_VERSION,
            "review_decisions_collected_with_visual_evidence_version": sorted(collected_evidence_versions),
            "review_decisions_visual_evidence_version_matches_current": evidence_version_matches,
            "validation_errors": validation_errors,
            "forbidden_keys_present": forbidden,
            **m3t_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(progress)
    return progress


def m3t_review_decision_summary_payload(review_payload: dict[str, Any], reviewed_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    reviewed_payload = reviewed_payload or {"rows": []}
    progress = m3t_review_progress_payload(review_payload, reviewed_payload)
    usable_rows, _errors = validate_m3t_reviewed_rows(review_payload, reviewed_payload)
    decision_summary = guardrail_stamp(
        {
            "artifact": "step2m3t_review_decision_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "total_review_candidates": progress["total_review_candidates"],
            "reviewed_candidates": progress["reviewed_candidates"],
            "persisted_reviewed_decision_rows": progress["persisted_reviewed_decision_rows"],
            "accepted_count": progress["accepted_count"],
            "rejected_count": progress["rejected_count"],
            "unsure_count": progress["unsure_count"],
            "human_review_decision_counts": dict(sorted(Counter(str(row.get("human_review_decision", "")) for row in usable_rows).items())),
            "sparse_pathlet_review_completed": progress["sparse_pathlet_review_completed"],
            "review_category_progress": progress["review_category_progress"],
            "visual_evidence_available_count": progress["visual_evidence_available_count"],
            "visual_evidence_missing_count": progress["visual_evidence_missing_count"],
            "animation_evidence_available_count": progress["animation_evidence_available_count"],
            "animation_evidence_missing_count": progress["animation_evidence_missing_count"],
            "current_review_version": M3T_CURRENT_REVIEW_VERSION,
            "review_decisions_collected_with_review_version": progress["review_decisions_collected_with_review_version"],
            "review_decisions_version_matches_current": progress["review_decisions_version_matches_current"],
            "current_visual_evidence_version": M3T_CURRENT_VISUAL_EVIDENCE_VERSION,
            "review_decisions_collected_with_visual_evidence_version": progress["review_decisions_collected_with_visual_evidence_version"],
            "review_decisions_visual_evidence_version_matches_current": progress["review_decisions_visual_evidence_version_matches_current"],
            "validation_errors": progress["validation_errors"],
            "forbidden_keys_present": progress["forbidden_keys_present"],
            **m3t_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(decision_summary)
    return decision_summary


def refresh_m3t_review_summaries(
    review_payload: dict[str, Any],
    reviewed_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    progress = m3t_review_progress_payload(review_payload, reviewed_payload)
    decision_summary = m3t_review_decision_summary_payload(review_payload, reviewed_payload)
    write_json_atomic(STEP2M3T_REVIEW_PROGRESS_SUMMARY_PATH, progress)
    write_json_atomic(STEP2M3T_REVIEW_DECISION_SUMMARY_PATH, decision_summary)
    return progress, decision_summary


def save_m3t_review_decision(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    review_payload = read_json(STEP2M3T_REVIEW_CANDIDATE_ROWS_PATH)
    candidates = m3t_candidate_by_review_id(review_payload)
    candidate_id = str(payload.get("step2m3t_review_candidate_id", ""))
    candidate = candidates.get(candidate_id)
    if not candidate:
        raise ValueError("unknown_step2m3t_review_candidate")
    decision = m3t_review_decision_row(candidate, payload)
    reviewed_payload = read_m3t_reviewed_decisions()
    by_id = {
        str(row.get("step2m3t_review_candidate_id", "")): row
        for row in decision_rows_from_payload(reviewed_payload)
        if row.get("step2m3t_review_candidate_id")
    }
    by_id[str(decision["step2m3t_review_candidate_id"])] = decision
    updated_payload = write_m3t_reviewed_decisions_payload(list(by_id.values()))
    progress, _decision_summary = refresh_m3t_review_summaries(review_payload, updated_payload)
    return decision, updated_payload, progress


def review_progress_payload(review_payload: dict[str, Any], reviewed_payload: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    reviewed_payload = reviewed_payload or read_m3t_reviewed_decisions()
    return m3t_review_progress_payload(review_payload, reviewed_payload), m3t_review_decision_summary_payload(review_payload, reviewed_payload)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def edge_id(row: dict[str, Any]) -> str:
    return str(row.get("continuity_edge_id", ""))


def group_topology_flags(group_audit: dict[str, Any], edge_audit: dict[str, Any]) -> dict[str, bool]:
    branch_merge = (
        edge_audit.get("branch_merge_point_visual_only") is True
        or group_audit.get("has_branching") is True
        or group_audit.get("has_merging") is True
    )
    duplicate = edge_audit.get("duplicate_frame_group_segment_visual_only") is True or safe_int(group_audit.get("frames_with_multiple_members_count"), 0) > 0
    role_mix = edge_audit.get("role_context_mixed_group_visual_only") is True or group_audit.get("has_role_context_mixing") is True
    return {
        "branch_merge_history": branch_merge,
        "duplicate_frame_member_history": duplicate,
        "role_context_mixing_history": role_mix,
        "topology_group_high_risk": edge_audit.get("topology_group_high_risk") is True or group_audit.get("high_topology_risk") is True,
    }


def m3r_decision_maps(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    indexes = decision_indexes(decisions)
    return {
        "accepted_edges": set(indexes["accepted_edges"]),
        "rejected_edges": set(indexes["rejected_edges"]),
        "unsure_edges": set(indexes["unsure_edges"]),
        "accepted_groups": set(indexes["accepted_groups"]),
        "rejected_groups": set(indexes["rejected_groups"]),
        "unsure_groups": set(indexes["unsure_groups"]),
        "accepted_edge_rows": indexes["accepted_edges"],
        "rejected_edge_rows": indexes["rejected_edges"],
        "unsure_edge_rows": indexes["unsure_edges"],
        "accepted_group_rows": indexes["accepted_groups"],
        "rejected_group_rows": indexes["rejected_groups"],
        "unsure_group_rows": indexes["unsure_groups"],
    }


def score_candidate_edge(
    edge: dict[str, Any],
    *,
    edge_audit: dict[str, Any],
    group_audit: dict[str, Any],
    decision_maps: dict[str, Any],
    m3s_seed_edge_ids: set[str],
) -> dict[str, Any]:
    continuity_edge_id = edge_id(edge)
    group_id = str(edge_audit.get("visual_continuity_group_id", ""))
    bucket = str(edge.get("source_review_bucket", edge_audit.get("source_review_bucket", "")))
    base_score = safe_float(edge.get("adapted_edge_score_sandbox", edge.get("original_edge_score_sandbox", 0.0)))
    flags = group_topology_flags(group_audit, edge_audit)
    rewards: list[str] = []
    penalties: list[str] = []
    score = base_score
    if continuity_edge_id in m3s_seed_edge_ids:
        score += 0.55
        rewards.append("m3s_handoff_safe_seed")
    if continuity_edge_id in decision_maps["accepted_edges"]:
        score += 0.45
        rewards.append("m3r_accepted_edge_level_review")
    if str(edge.get("human_review_decision", "")) in {"accept_short_window_visual_continuity_edge", M3S_ACCEPT_DECISION}:
        score += 0.2
        rewards.append("human_accepted_visual_continuity")
    if bucket in {"safe_auto_accept_candidate", "safe_auto_accept_audit"}:
        score += 0.08
        rewards.append("safe_auto_accept_support")
    if flags["branch_merge_history"]:
        score -= 0.18
        penalties.append("branch_merge_topology_history")
    if flags["duplicate_frame_member_history"]:
        score -= 0.15
        penalties.append("duplicate_frame_member_conflict_history")
    if flags["role_context_mixing_history"]:
        score -= 0.14
        penalties.append("role_context_mixing_history")
    if flags["topology_group_high_risk"]:
        score -= 0.08
        penalties.append("high_topology_risk_group_history")
    if group_id in decision_maps["rejected_groups"]:
        score -= 0.2
        penalties.append("rejected_m3r_group_membership")
    if bucket in {"high_uncertainty_low_margin", "merged_or_ambiguous", "role_state_mismatch", "bad_detection_proxy_adjacent"}:
        score -= 0.12
        penalties.append(f"risky_source_bucket_{bucket}")
    direct_rejected = continuity_edge_id in decision_maps["rejected_edges"]
    direct_unsure = continuity_edge_id in decision_maps["unsure_edges"]
    span_cap_violation = safe_int(edge.get("frame_gap"), 0) > M3T_MAX_PATHLET_SPAN_FRAMES
    eligible = not direct_rejected and not direct_unsure and not span_cap_violation and clamp01(score) >= M3T_MIN_SELECTION_SCORE
    forced_quarantine_reasons: list[str] = []
    if direct_rejected:
        forced_quarantine_reasons.append("rejected_by_m3r")
    if direct_unsure:
        forced_quarantine_reasons.append("unsure_by_m3r")
    if span_cap_violation:
        forced_quarantine_reasons.append("span_cap_violation")
    if not eligible and not forced_quarantine_reasons:
        forced_quarantine_reasons.append("low_sparse_selection_score")
    row = dict(edge)
    row.update(
        {
            "visual_continuity_group_id": group_id,
            "m3t_sparse_candidate_score": round(clamp01(score), 4),
            "m3t_sparse_candidate_eligible": eligible,
            "m3t_sparse_selection_rewards": rewards,
            "m3t_sparse_selection_penalties": penalties,
            "m3s_handoff_seed_edge": continuity_edge_id in m3s_seed_edge_ids,
            "m3r_edge_level_accepted": continuity_edge_id in decision_maps["accepted_edges"],
            "m3r_edge_level_rejected": direct_rejected,
            "m3r_edge_level_unsure": direct_unsure,
            "m3r_group_rejected_membership": group_id in decision_maps["rejected_groups"],
            "m3r_group_unsure_membership": group_id in decision_maps["unsure_groups"],
            **flags,
            "forced_quarantine_reasons": forced_quarantine_reasons,
            "short_window_visual_continuity_only": True,
            **m3t_guardrail_fields(),
        }
    )
    visual_stamp(row)
    assert_no_forbidden_keys(row)
    return row


def build_sparse_candidate_edges(
    accepted_edges: list[dict[str, Any]],
    edge_audit_rows: list[dict[str, Any]],
    group_audit_rows: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    m3s_seed_edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    edge_audit_by_id = {edge_id(row): row for row in edge_audit_rows}
    group_audit_by_id = {str(row.get("visual_continuity_group_id", "")): row for row in group_audit_rows}
    maps = m3r_decision_maps(decisions)
    seed_ids = {edge_id(row) for row in m3s_seed_edges}
    rows: list[dict[str, Any]] = []
    for edge in accepted_edges:
        audit = edge_audit_by_id.get(edge_id(edge), {})
        group_audit = group_audit_by_id.get(str(audit.get("visual_continuity_group_id", "")), {})
        rows.append(
            score_candidate_edge(
                edge,
                edge_audit=audit,
                group_audit=group_audit,
                decision_maps=maps,
                m3s_seed_edge_ids=seed_ids,
            )
        )
    return rows


def sparse_select_edges(candidate_edges: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    used_out: set[str] = set()
    used_in: set[str] = set()
    used_out_transition: set[tuple[int, int, str]] = set()
    used_in_transition: set[tuple[int, int, str]] = set()
    ordered = sorted(
        candidate_edges,
        key=lambda row: (
            row.get("m3s_handoff_seed_edge") is True,
            row.get("m3r_edge_level_accepted") is True,
            safe_float(row.get("m3t_sparse_candidate_score")),
            -safe_int(row.get("frame_gap"), 0),
            edge_id(row),
        ),
        reverse=True,
    )
    for row in ordered:
        source = str(row.get("source_visible_person_base_id", ""))
        target = str(row.get("target_visible_person_base_id", ""))
        source_frame = safe_int(row.get("source_frame_sequence"), -1)
        target_frame = safe_int(row.get("target_frame_sequence"), -1)
        transition = (source_frame, target_frame)
        reasons = list(row.get("forced_quarantine_reasons", []))
        if not row.get("m3t_sparse_candidate_eligible"):
            quarantined.append(annotate_quarantined_edge(row, reasons or ["low_sparse_selection_score"]))
            continue
        if source in used_out or target in used_in:
            quarantined.append(annotate_quarantined_edge(row, ["one_to_one_matching_rejected"]))
            continue
        if (*transition, source) in used_out_transition or (*transition, target) in used_in_transition:
            quarantined.append(annotate_quarantined_edge(row, ["one_to_one_matching_rejected"]))
            continue
        selected_row = dict(row)
        selected_row.update(
            {
                "m3t_sparse_selected": True,
                "m3t_sparse_selection_reason": "one_to_one_sparse_match_selected",
                **m3t_guardrail_fields(),
            }
        )
        visual_stamp(selected_row)
        assert_no_forbidden_keys(selected_row)
        selected.append(selected_row)
        used_out.add(source)
        used_in.add(target)
        used_out_transition.add((*transition, source))
        used_in_transition.add((*transition, target))
    return selected, quarantined


def annotate_quarantined_edge(row: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    output = dict(row)
    output.update(
        {
            "m3t_topology_quarantined": True,
            "m3t_topology_quarantine_reasons": sorted(set(reasons)) or ["one_to_one_matching_rejected"],
            "not_safe_for_sparse_pathlet": True,
            **m3t_guardrail_fields(),
        }
    )
    visual_stamp(output)
    assert_no_forbidden_keys(output)
    return output


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def edge_sort_key(edge: dict[str, Any]) -> tuple[int, int, str]:
    return (
        safe_int(edge.get("source_frame_sequence"), -1),
        safe_int(edge.get("target_frame_sequence"), -1),
        edge_id(edge),
    )


def selected_edge_components(selected_edges: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    union = UnionFind()
    for edge in selected_edges:
        source = str(edge.get("source_visible_person_base_id", ""))
        target = str(edge.get("target_visible_person_base_id", ""))
        if source and target:
            union.union(source, target)
    component_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in selected_edges:
        source = str(edge.get("source_visible_person_base_id", ""))
        target = str(edge.get("target_visible_person_base_id", ""))
        component_edges[union.find(source or target or edge_id(edge))].append(edge)
    return [sorted(edges, key=edge_sort_key) for edges in component_edges.values()]


def split_component_by_span(edges: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_min = 10**9
    current_max = -1
    for edge in sorted(edges, key=edge_sort_key):
        start = safe_int(edge.get("source_frame_sequence"), -1)
        end = safe_int(edge.get("target_frame_sequence"), -1)
        proposed_min = min(current_min, start, end)
        proposed_max = max(current_max, start, end)
        if current and proposed_max - proposed_min > M3T_MAX_PATHLET_SPAN_FRAMES:
            chunks.append(current)
            current = []
            current_min = 10**9
            current_max = -1
        current.append(edge)
        current_min = min(current_min, start, end)
        current_max = max(current_max, start, end)
    if current:
        chunks.append(current)
    return chunks


def pathlet_metrics(edges: list[dict[str, Any]]) -> dict[str, Any]:
    frame_members: dict[int, set[str]] = defaultdict(set)
    out_degree: Counter[str] = Counter()
    in_degree: Counter[str] = Counter()
    for edge in edges:
        source = str(edge.get("source_visible_person_base_id", ""))
        target = str(edge.get("target_visible_person_base_id", ""))
        source_frame = safe_int(edge.get("source_frame_sequence"), -1)
        target_frame = safe_int(edge.get("target_frame_sequence"), -1)
        if source and target:
            out_degree[source] += 1
            in_degree[target] += 1
        if source and source_frame >= 0:
            frame_members[source_frame].add(source)
        if target and target_frame >= 0:
            frame_members[target_frame].add(target)
    frames = sorted(frame_members)
    member_pairs = sorted((frame, member) for frame, members in frame_members.items() for member in members)
    max_members = max((len(members) for members in frame_members.values()), default=0)
    max_in = max(in_degree.values(), default=0)
    max_out = max(out_degree.values(), default=0)
    branch_count = sum(1 for value in out_degree.values() if value > 1)
    merge_count = sum(1 for value in in_degree.values() if value > 1)
    frame_span = max(frames) - min(frames) if frames else 0
    return {
        "member_pairs": member_pairs,
        "min_frame_sequence": min(frames) if frames else -1,
        "max_frame_sequence": max(frames) if frames else -1,
        "frame_span": frame_span,
        "seconds_span": round(frame_span / 10.0, 4),
        "max_members_per_frame": max_members,
        "max_in_degree": max_in,
        "max_out_degree": max_out,
        "branch_count": branch_count,
        "merge_count": merge_count,
    }


def pathlet_violation_reasons(metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if safe_int(metrics.get("frame_span"), 0) > M3T_MAX_PATHLET_SPAN_FRAMES or safe_float(metrics.get("seconds_span"), 0.0) > M3T_MAX_PATHLET_SPAN_SECONDS:
        reasons.append("span_cap_violation")
    if safe_int(metrics.get("max_members_per_frame"), 0) > 1:
        reasons.append("duplicate_frame_member_conflict")
    if safe_int(metrics.get("branch_count"), 0) > 0 or safe_int(metrics.get("merge_count"), 0) > 0:
        reasons.append("branch_merge_not_resolved")
    if safe_int(metrics.get("max_in_degree"), 0) > 1 or safe_int(metrics.get("max_out_degree"), 0) > 1:
        reasons.append("branch_merge_not_resolved")
    return sorted(set(reasons))


def make_pathlet_row(index: int, edges: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    source_flags = {
        "from_branch_merge_heavy_region": any(edge.get("branch_merge_history") is True for edge in edges),
        "from_duplicate_frame_heavy_region": any(edge.get("duplicate_frame_member_history") is True for edge in edges),
        "from_role_context_mixed_region": any(edge.get("role_context_mixing_history") is True for edge in edges),
        "contains_m3s_handoff_seed_edge": any(edge.get("m3s_handoff_seed_edge") is True for edge in edges),
    }
    row = guardrail_stamp(
        {
            "pathlet_id": f"step2m3t_pathlet_{index:06d}",
            "member_visible_person_base_ids": [member for _frame, member in metrics["member_pairs"]],
            "member_frame_sequences": [frame for frame, _member in metrics["member_pairs"]],
            "accepted_continuity_edge_ids": [edge_id(edge) for edge in edges],
            "source_m3_visual_continuity_group_ids": sorted({str(edge.get("visual_continuity_group_id", "")) for edge in edges if edge.get("visual_continuity_group_id")}),
            "min_frame_sequence": metrics["min_frame_sequence"],
            "max_frame_sequence": metrics["max_frame_sequence"],
            "frame_span": metrics["frame_span"],
            "seconds_span": metrics["seconds_span"],
            "max_members_per_frame": metrics["max_members_per_frame"],
            "max_in_degree": metrics["max_in_degree"],
            "max_out_degree": metrics["max_out_degree"],
            "branch_count": metrics["branch_count"],
            "merge_count": metrics["merge_count"],
            "topology_safe_for_handoff_candidate": True,
            "pathlet_not_identity": True,
            "pathlet_not_player_slot": True,
            "pathlet_not_goalkeeper_slot": True,
            **source_flags,
            **m3t_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(row)
    return row


def make_quarantined_pathlet_row(index: int, edges: list[dict[str, Any]], metrics: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    row = make_pathlet_row(index, edges, metrics)
    row.update(
        {
            "pathlet_id": f"step2m3t_quarantined_pathlet_{index:06d}",
            "topology_safe_for_handoff_candidate": False,
            "m3t_topology_quarantined": True,
            "m3t_topology_quarantine_reasons": sorted(set(reasons)),
            **m3t_guardrail_fields(),
        }
    )
    visual_stamp(row)
    assert_no_forbidden_keys(row)
    return row


def build_pathlets_from_selected_edges(selected_edges: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    accepted_pathlets: list[dict[str, Any]] = []
    quarantined_pathlets: list[dict[str, Any]] = []
    quarantined_edge_ids: set[str] = set()
    accepted_index = 1
    quarantine_index = 1
    for component in selected_edge_components(selected_edges):
        for chunk in split_component_by_span(component):
            metrics = pathlet_metrics(chunk)
            reasons = pathlet_violation_reasons(metrics)
            if reasons:
                quarantined_pathlets.append(make_quarantined_pathlet_row(quarantine_index, chunk, metrics, reasons))
                quarantine_index += 1
                quarantined_edge_ids.update(edge_id(edge) for edge in chunk)
                continue
            pathlet = make_pathlet_row(accepted_index, chunk, metrics)
            accepted_index += 1
            accepted_pathlets.append(pathlet)
    return accepted_pathlets, quarantined_pathlets, quarantined_edge_ids


def annotate_selected_edge(edge: dict[str, Any], pathlet_id: str) -> dict[str, Any]:
    row = dict(edge)
    row.update(
        {
            "pathlet_id": pathlet_id,
            "m3t_selected_sparse_edge": True,
            "topology_safe_for_handoff_candidate": True,
            **m3t_guardrail_fields(),
        }
    )
    visual_stamp(row)
    assert_no_forbidden_keys(row)
    return row


def attach_pathlet_ids_to_edges(selected_edges: list[dict[str, Any]], pathlets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edge_to_pathlet: dict[str, str] = {}
    for pathlet in pathlets:
        for continuity_edge_id in pathlet.get("accepted_continuity_edge_ids", []):
            edge_to_pathlet[str(continuity_edge_id)] = str(pathlet.get("pathlet_id", ""))
    return [
        annotate_selected_edge(edge, edge_to_pathlet[edge_id(edge)])
        for edge in selected_edges
        if edge_id(edge) in edge_to_pathlet
    ]


def summarize_reason_counts(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        for reason in row.get(key, []):
            counter[str(reason)] += 1
    return dict(sorted(counter.items()))


def candidate_edge_payloads(candidate_edges: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = guardrail_stamp(
        {
            "artifact": "step2m3t_sparse_candidate_edge_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "candidate_edge_count": len(candidate_edges),
            "eligible_candidate_edge_count": sum(1 for edge in candidate_edges if edge.get("m3t_sparse_candidate_eligible") is True),
            "m3s_seed_candidate_edge_count": sum(1 for edge in candidate_edges if edge.get("m3s_handoff_seed_edge") is True),
            "m3r_accepted_edge_level_candidate_count": sum(1 for edge in candidate_edges if edge.get("m3r_edge_level_accepted") is True),
            "direct_m3r_rejected_edge_count": sum(1 for edge in candidate_edges if edge.get("m3r_edge_level_rejected") is True),
            "direct_m3r_unsure_edge_count": sum(1 for edge in candidate_edges if edge.get("m3r_edge_level_unsure") is True),
            **m3t_guardrail_fields(),
        }
    )
    sample = guardrail_stamp(
        {
            "artifact": "step2m3t_sparse_candidate_edge_sample",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "rows": sorted(candidate_edges, key=lambda row: safe_float(row.get("m3t_sparse_candidate_score")), reverse=True)[:50],
            **m3t_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(summary)
    assert_no_forbidden_keys(sample)
    return summary, sample


def selected_edge_payloads(selected_edges: list[dict[str, Any]], source_candidate_count: int) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = guardrail_stamp(
        {
            "artifact": "step2m3t_selected_sparse_visual_continuity_edge_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "source_candidate_edge_count": source_candidate_count,
            "selected_sparse_edge_count": len(selected_edges),
            "selected_m3s_seed_edge_count": sum(1 for edge in selected_edges if edge.get("m3s_handoff_seed_edge") is True),
            "selected_m3r_edge_level_accepted_count": sum(1 for edge in selected_edges if edge.get("m3r_edge_level_accepted") is True),
            **m3t_guardrail_fields(),
        }
    )
    sample = guardrail_stamp(
        {
            "artifact": "step2m3t_selected_sparse_visual_continuity_edge_sample",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "rows": selected_edges[:50],
            **m3t_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(summary)
    assert_no_forbidden_keys(sample)
    return summary, sample


def pathlet_payloads(pathlets: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    over_cap = sum(1 for row in pathlets if safe_int(row.get("frame_span"), 0) > M3T_MAX_PATHLET_SPAN_FRAMES or safe_float(row.get("seconds_span"), 0.0) > M3T_MAX_PATHLET_SPAN_SECONDS)
    duplicate = sum(1 for row in pathlets if safe_int(row.get("max_members_per_frame"), 0) > 1)
    branch_merge = sum(1 for row in pathlets if safe_int(row.get("branch_count"), 0) > 0 or safe_int(row.get("merge_count"), 0) > 0)
    payload = guardrail_stamp(
        {
            "artifact": "step2m3t_sparse_visual_continuity_pathlets",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "rows": pathlets,
            **m3t_guardrail_fields(),
        }
    )
    summary = guardrail_stamp(
        {
            "artifact": "step2m3t_sparse_visual_continuity_pathlet_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "sparse_pathlet_count": len(pathlets),
            "pathlets_over_cap_count": over_cap,
            "pathlets_with_duplicate_frame_members_count": duplicate,
            "pathlets_with_branch_merge_count": branch_merge,
            "max_pathlet_span_frames_observed": max((safe_int(row.get("frame_span"), 0) for row in pathlets), default=0),
            "max_pathlet_span_seconds_observed": max((safe_float(row.get("seconds_span"), 0.0) for row in pathlets), default=0.0),
            "pathlets_from_branch_merge_heavy_regions": sum(1 for row in pathlets if row.get("from_branch_merge_heavy_region") is True),
            "pathlets_from_duplicate_frame_heavy_regions": sum(1 for row in pathlets if row.get("from_duplicate_frame_heavy_region") is True),
            "pathlets_from_role_context_mixed_regions": sum(1 for row in pathlets if row.get("from_role_context_mixed_region") is True),
            "pathlets_with_m3s_seed_edges": sum(1 for row in pathlets if row.get("contains_m3s_handoff_seed_edge") is True),
            **m3t_guardrail_fields(),
        }
    )
    sample = guardrail_stamp(
        {
            "artifact": "step2m3t_sparse_visual_continuity_pathlet_sample",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "rows": pathlets[:30],
            **m3t_guardrail_fields(),
        }
    )
    for item in [payload, summary, sample]:
        assert_no_forbidden_keys(item)
    return payload, summary, sample


def quarantine_payloads(quarantined_edges: list[dict[str, Any]], quarantined_pathlets: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    pathlet_payload = guardrail_stamp(
        {
            "artifact": "step2m3t_topology_quarantined_pathlets",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "rows": quarantined_pathlets,
            **m3t_guardrail_fields(),
        }
    )
    summary = guardrail_stamp(
        {
            "artifact": "step2m3t_topology_quarantine_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "topology_quarantined_edge_count": len(quarantined_edges),
            "topology_quarantined_pathlet_count": len(quarantined_pathlets),
            "topology_quarantined_edge_reason_counts": summarize_reason_counts(quarantined_edges, "m3t_topology_quarantine_reasons"),
            "topology_quarantined_pathlet_reason_counts": summarize_reason_counts(quarantined_pathlets, "m3t_topology_quarantine_reasons"),
            **m3t_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(pathlet_payload)
    assert_no_forbidden_keys(summary)
    return pathlet_payload, summary


def build_sparse_outputs() -> dict[str, Any]:
    accepted_edges = read_jsonl_gz_rows(STEP2M3_ACCEPTED_EDGES_JSONL_GZ_PATH)
    edge_audit_rows = read_jsonl_gz_rows(STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_ROWS_JSONL_GZ_PATH)
    group_audit_rows = rows_from_payload(read_json(STEP2M3R_GROUP_TOPOLOGY_AUDIT_ROWS_PATH))
    decisions = load_m3r_reviewed_decisions(STEP2M3R_REVIEWED_TOPOLOGY_DECISIONS_PATH)
    m3s_seed_edges = read_jsonl_gz_rows(STEP2M3S_HANDOFF_SAFE_EDGES_JSONL_GZ_PATH)
    candidate_edges = build_sparse_candidate_edges(accepted_edges, edge_audit_rows, group_audit_rows, decisions, m3s_seed_edges)
    prelim_selected_edges, quarantined_edges = sparse_select_edges(candidate_edges)
    pathlets, quarantined_pathlets, pathlet_quarantined_edge_ids = build_pathlets_from_selected_edges(prelim_selected_edges)
    selected_edges = attach_pathlet_ids_to_edges(prelim_selected_edges, pathlets)
    selected_edge_ids = {edge_id(row) for row in selected_edges}
    for edge in prelim_selected_edges:
        if edge_id(edge) in pathlet_quarantined_edge_ids and edge_id(edge) not in selected_edge_ids:
            quarantined_edges.append(annotate_quarantined_edge(edge, ["branch_merge_not_resolved"]))
    quarantined_ids = {edge_id(row) for row in quarantined_edges}
    selected_edges = [edge for edge in selected_edges if edge_id(edge) not in quarantined_ids]
    return {
        "m3_source_accepted_edges_loaded": len(accepted_edges),
        "m3r_decisions_loaded": len(decisions),
        "m3s_handoff_seed_edges_loaded": len(m3s_seed_edges),
        "candidate_edges": candidate_edges,
        "selected_edges": selected_edges,
        "pathlets": pathlets,
        "quarantined_edges": quarantined_edges,
        "quarantined_pathlets": quarantined_pathlets,
    }


def render_pathlet_evidence(pathlet: dict[str, Any], nodes_by_id: dict[str, dict[str, Any]], frame_lookup: dict[int, str], stem: str) -> dict[str, Any]:
    ensure_dir(STEP2M3T_PATHLET_ANIMATIONS_DIR)
    ensure_dir(STEP2M3T_PATHLET_STRIPS_DIR)
    animation_path = STEP2M3T_PATHLET_ANIMATIONS_DIR / f"{stem}_pathlet.gif"
    strip_path = STEP2M3T_PATHLET_STRIPS_DIR / f"{stem}_pathlet.jpg"
    frames = sample_frame_sequences(
        safe_int(pathlet.get("min_frame_sequence"), -1),
        safe_int(pathlet.get("max_frame_sequence"), -1),
        pathlet.get("member_frame_sequences", []),
        max_frames=16,
    )
    members_by_frame: dict[int, list[str]] = defaultdict(list)
    for frame, member in zip(pathlet.get("member_frame_sequences", []), pathlet.get("member_visible_person_base_ids", []), strict=False):
        members_by_frame[safe_int(frame, -1)].append(str(member))
    rendered: list[Any] = []
    for frame in frames:
        tile, metadata = tile_from_frame(frame, frame_lookup)
        if tile is None:
            continue
        cv2.putText(tile, f"pathlet {pathlet.get('pathlet_id', '')[-12:]} frame {frame}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
        for member_id in members_by_frame.get(frame, []):
            node = nodes_by_id.get(member_id, {})
            draw_scaled_box(tile, node.get("bbox", {}), metadata, label=f"visual {member_id[-8:]}", colour=(0, 215, 255), thickness=2)
        cv2.putText(tile, "not identity / visual-only topology QA", (10, tile.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 230, 240), 1, cv2.LINE_AA)
        rendered.append(tile)
    strip = horizontal_image_strip(rendered)
    static_available = bool(strip is not None and write_image(strip_path, strip))
    animation_available = write_animation_gif(animation_path, rendered)
    return {
        "evidence_type": "pathlet_animation",
        "evidence_animation_gif_path": m3t_rel_path(animation_path) if animation_available else "",
        "evidence_static_strip_path": m3t_rel_path(strip_path) if static_available else "",
        "evidence_available": animation_available and static_available,
        "sampled_frame_sequences": frames,
        "current_visual_evidence_version": M3T_CURRENT_VISUAL_EVIDENCE_VERSION,
    }


def render_edge_evidence(edge: dict[str, Any], nodes_by_id: dict[str, dict[str, Any]], frame_lookup: dict[int, str], stem: str) -> dict[str, Any]:
    ensure_dir(STEP2M3T_EDGE_BURST_ANIMATIONS_DIR)
    ensure_dir(STEP2M3T_EDGE_BURST_STRIPS_DIR)
    animation_path = STEP2M3T_EDGE_BURST_ANIMATIONS_DIR / f"{stem}_edge.gif"
    strip_path = STEP2M3T_EDGE_BURST_STRIPS_DIR / f"{stem}_edge.jpg"
    source_frame = safe_int(edge.get("source_frame_sequence"), -1)
    target_frame = safe_int(edge.get("target_frame_sequence"), -1)
    frames = sample_frame_sequences(min(source_frame, target_frame), max(source_frame, target_frame), [source_frame, target_frame], max_frames=12)
    source_id = str(edge.get("source_visible_person_base_id", ""))
    target_id = str(edge.get("target_visible_person_base_id", ""))
    rendered: list[Any] = []
    for frame in frames:
        tile, metadata = tile_from_frame(frame, frame_lookup)
        if tile is None:
            continue
        cv2.putText(tile, f"edge {edge_id(edge)[-12:]} frame {frame}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
        if frame == source_frame:
            draw_scaled_box(tile, nodes_by_id.get(source_id, {}).get("bbox", {}), metadata, label=f"source {source_id[-8:]}", colour=(0, 215, 255), thickness=3)
        if frame == target_frame:
            draw_scaled_box(tile, nodes_by_id.get(target_id, {}).get("bbox", {}), metadata, label=f"target {target_id[-8:]}", colour=(90, 245, 120), thickness=3)
        cv2.putText(tile, "not identity / visual-only topology QA", (10, tile.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 230, 240), 1, cv2.LINE_AA)
        rendered.append(tile)
    strip = horizontal_image_strip(rendered)
    static_available = bool(strip is not None and write_image(strip_path, strip))
    animation_available = write_animation_gif(animation_path, rendered)
    return {
        "evidence_type": "edge_burst_animation",
        "evidence_animation_gif_path": m3t_rel_path(animation_path) if animation_available else "",
        "evidence_static_strip_path": m3t_rel_path(strip_path) if static_available else "",
        "evidence_available": animation_available and static_available,
        "sampled_frame_sequences": frames,
        "current_visual_evidence_version": M3T_CURRENT_VISUAL_EVIDENCE_VERSION,
    }


def add_unique_review_row(rows: list[dict[str, Any]], seen: set[str], row: dict[str, Any]) -> None:
    candidate_id = str(row.get("step2m3t_review_candidate_id", ""))
    if candidate_id and candidate_id not in seen and len(rows) < M3T_REVIEW_HARD_MAX_CARDS:
        rows.append(row)
        seen.add(candidate_id)


def build_review_candidates(pathlets: list[dict[str, Any]], selected_edges: list[dict[str, Any]], quarantined_edges: list[dict[str, Any]]) -> dict[str, Any]:
    edge_by_id = {edge_id(edge): edge for edge in selected_edges + quarantined_edges}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def pathlet_row(pathlet: dict[str, Any], category: str) -> dict[str, Any]:
        row = {
            "step2m3t_review_candidate_id": f"step2m3t_review_{category}_{pathlet.get('pathlet_id')}",
            "review_subject_type": "sparse_visual_continuity_pathlet",
            "step2m3t_review_category": category,
            "pathlet_id": pathlet.get("pathlet_id", ""),
            "continuity_edge_id": "",
            "accepted_continuity_edge_ids": pathlet.get("accepted_continuity_edge_ids", []),
            "min_frame_sequence": pathlet.get("min_frame_sequence"),
            "max_frame_sequence": pathlet.get("max_frame_sequence"),
            "review_instruction": "Judge only whether the sparse visual pathlet handoff is visually coherent. Do not infer identity.",
            **m3t_guardrail_fields(),
        }
        visual_stamp(row)
        return row

    buckets = [
        ("branch_merge_sparse_accept", [row for row in pathlets if row.get("from_branch_merge_heavy_region") is True], 10),
        ("duplicate_frame_sparse_accept", [row for row in pathlets if row.get("from_duplicate_frame_heavy_region") is True], 10),
        ("role_context_sparse_accept", [row for row in pathlets if row.get("from_role_context_mixed_region") is True], 10),
        ("m3s_seed_control", [row for row in pathlets if row.get("contains_m3s_handoff_seed_edge") is True], 5),
    ]
    for category, bucket_rows, limit in buckets:
        for pathlet in bucket_rows[:limit]:
            add_unique_review_row(rows, seen, pathlet_row(pathlet, category))
    for edge in quarantined_edges[:5]:
        row = {
            "step2m3t_review_candidate_id": f"step2m3t_review_quarantine_control_{edge_id(edge)}",
            "review_subject_type": "quarantined_sparse_candidate_edge",
            "step2m3t_review_category": "rejected_quarantined_control",
            "pathlet_id": "",
            "continuity_edge_id": edge_id(edge),
            "accepted_continuity_edge_ids": [],
            "source_frame_sequence": edge.get("source_frame_sequence"),
            "target_frame_sequence": edge.get("target_frame_sequence"),
            "m3t_topology_quarantine_reasons": edge.get("m3t_topology_quarantine_reasons", []),
            "review_instruction": "Control card: confirm the visual-only quarantine is reasonable. Do not infer identity.",
            **m3t_guardrail_fields(),
        }
        visual_stamp(row)
        add_unique_review_row(rows, seen, row)
    for pathlet in pathlets:
        if len(rows) >= M3T_REVIEW_TARGET_CARDS:
            break
        add_unique_review_row(rows, seen, pathlet_row(pathlet, "sparse_accept_fill"))

    node_payload = read_json(STEP2M1_NODE_ROWS_PATH)
    nodes_by_id, _nodes_by_frame, frame_lookup = node_lookup_from_payload(node_payload)
    pathlet_by_id = {str(row.get("pathlet_id", "")): row for row in pathlets}
    enriched_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows[:M3T_REVIEW_HARD_MAX_CARDS], start=1):
        stem = safe_asset_stem(str(row.get("step2m3t_review_candidate_id", index)))
        if row.get("review_subject_type") == "quarantined_sparse_candidate_edge":
            evidence = render_edge_evidence(edge_by_id.get(str(row.get("continuity_edge_id", "")), {}), nodes_by_id, frame_lookup, stem)
        else:
            evidence = render_pathlet_evidence(pathlet_by_id.get(str(row.get("pathlet_id", "")), {}), nodes_by_id, frame_lookup, stem)
        enriched = {**row, **evidence}
        assert_no_forbidden_keys(enriched)
        enriched_rows.append(enriched)
    category_counts = Counter(str(row.get("step2m3t_review_category", "")) for row in enriched_rows)
    payload = guardrail_stamp(
        {
            "artifact": "step2m3t_review_candidate_rows",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "target_review_cards": M3T_REVIEW_TARGET_CARDS,
            "hard_max_review_cards": M3T_REVIEW_HARD_MAX_CARDS,
            "summary": {
                "review_queue_size": len(enriched_rows),
                "review_category_counts": dict(sorted(category_counts.items())),
                "visual_evidence_available_count": sum(1 for row in enriched_rows if row.get("evidence_available") is True),
                "current_visual_evidence_version": M3T_CURRENT_VISUAL_EVIDENCE_VERSION,
            },
            "rows": enriched_rows,
            **m3t_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(payload)
    return payload


def render_review_contact_sheet(review_payload: dict[str, Any]) -> dict[str, Any]:
    ensure_dir(STEP2M3T_REVIEW_CONTACT_SHEET_PATH.parent)
    rows = rows_from_payload(review_payload)
    if cv2 is None or np is None:
        STEP2M3T_REVIEW_CONTACT_SHEET_PATH.write_bytes(b"STEP2M3T_CONTACT_SHEET_FALLBACK")
        return {"fallback_image": True, "visual_evidence_thumbnail_count": 0}
    cell_w, cell_h = 520, 246
    thumb_w, thumb_h = 500, 182
    rows_to_render = max(1, min(len(rows), M3T_REVIEW_TARGET_CARDS))
    sheet = np.full((cell_h * rows_to_render, cell_w, 3), 245, dtype=np.uint8)
    thumb_count = 0
    for idx, row in enumerate(rows[:rows_to_render]):
        y = idx * cell_h
        cv2.rectangle(sheet, (0, y), (cell_w - 1, y + cell_h - 1), (190, 190, 190), 1)
        evidence_path = m3t_abs_asset_path(str(row.get("evidence_static_strip_path", "")))
        image = cv2.imread(str(evidence_path)) if evidence_path.exists() else None
        if image is not None:
            thumb_count += 1
            thumbnail = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            sheet[y + 4 : y + 4 + thumb_h, 10 : 10 + thumb_w] = thumbnail
        cv2.putText(sheet, f"{idx + 1:02d} {row.get('step2m3t_review_category', '')}", (12, y + 210), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(sheet, "animation: yes" if row.get("evidence_animation_gif_path") else "animation: no", (12, y + 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 110, 50), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(STEP2M3T_REVIEW_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"Failed to write M3T review contact sheet: {STEP2M3T_REVIEW_CONTACT_SHEET_PATH}")
    return {"fallback_image": False, "visual_evidence_thumbnail_count": thumb_count}


def review_ui_html(review_payload: dict[str, Any]) -> str:
    payload_json = json.dumps(review_payload, indent=2).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Step2.M3T Sparse Pathlet Review</title>
<style>
:root{{font-family:Arial,sans-serif;color:#18202a;background:#f4f5f7;}}
body{{margin:0;}}
header{{position:sticky;top:0;background:#18202a;color:#fff;padding:12px 18px;z-index:2;display:flex;gap:16px;align-items:center;justify-content:space-between;}}
main{{max-width:1160px;margin:0 auto;padding:18px;}}
button{{border:0;border-radius:6px;padding:10px 14px;font-weight:700;cursor:pointer;}}
.accept{{background:#157347;color:#fff;}} .reject{{background:#b02a37;color:#fff;}} .unsure{{background:#8a6d1d;color:#fff;}} .nav{{background:#d8dee6;color:#18202a;}}
.export{{background:#34495e;color:#fff;}}
.card{{background:#fff;border:1px solid #d8dee6;border-radius:8px;padding:18px;margin-bottom:14px;box-shadow:0 1px 2px rgba(0,0,0,.05);}}
.evidence{{width:100%;max-height:64vh;object-fit:contain;background:#0f1720;border:1px solid #c8d0da;border-radius:6px;}}
.strip{{width:100%;max-height:24vh;object-fit:contain;background:#0f1720;border:1px solid #d8dee6;border-radius:6px;margin-top:8px;}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;}}
.field{{background:#f7f8fa;border:1px solid #e1e5ea;border-radius:6px;padding:10px;min-height:52px;}}
.label{{font-size:12px;color:#5b6675;text-transform:uppercase;letter-spacing:.02em;}}
.value{{font-size:15px;word-break:break-word;margin-top:4px;}}
.warning{{color:#8a2d12;font-weight:700;}}
.card-actions{{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0;}}
#status.error{{color:#ffb3b3;font-weight:700;}}
#status.saved{{color:#b5f2c9;font-weight:700;}}
textarea{{width:100%;min-height:64px;border:1px solid #c8d0da;border-radius:6px;padding:8px;font-family:inherit;box-sizing:border-box;}}
</style>
</head>
<body>
<header><div><strong>Step2.M3T Sparse Pathlet Review</strong> <span id="counter"></span></div><div id="status">ready</div></header>
<main>
<section class="card">
<p class="warning">Do not infer identity. Judge only sparse visual handoff coherence. Visual-only, not metric.</p>
<div class="card-actions">
<button class="accept" onclick="decide('accept_sparse_pathlet_for_visual_handoff')">A Accept</button>
<button class="reject" onclick="decide('reject_or_quarantine_sparse_pathlet')">X Reject</button>
<button class="unsure" onclick="decide('unsure_needs_later_review')">U Unsure</button>
<button class="nav" onclick="prevCard()">Left</button>
<button class="nav" onclick="nextCard()">Right</button>
<button class="export" onclick="exportDecisions()">Export M3T decisions JSON</button>
</div>
</section>
<section id="card" class="card"></section>
</main>
<script>
const state={payload_json};
const AUTOSAVE_ENDPOINT='/api/step2m3t/review-decision';
const STORAGE_KEY='step2m3t_review_decisions';
let index=Number(localStorage.getItem('step2m3t_review_index')||0);
const decisions=JSON.parse(localStorage.getItem(STORAGE_KEY)||'{{}}');
function esc(v){{return String(v??'').replace(/[&<>"]/g,s=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[s]));}}
function field(label,value){{return `<div class="field"><div class="label">${{esc(label)}}</div><div class="value">${{esc(value)}}</div></div>`;}}
function setStatus(message,isError=false){{
 const status=document.getElementById('status');
 status.textContent=message;
 status.className=isError?'error':(String(message).includes('saved')?'saved':'');
}}
function show(){{
 const rows=state.rows||[];
 if(!rows.length){{document.getElementById('card').innerHTML='<p>No review candidates.</p>';return;}}
 index=Math.max(0,Math.min(index,rows.length-1));
 localStorage.setItem('step2m3t_review_index',String(index));
 const row=rows[index];
 document.getElementById('counter').textContent=`${{index+1}} / ${{rows.length}}`;
 const animation=row.evidence_animation_gif_path ? `<img class="evidence" src="${{esc(row.evidence_animation_gif_path)}}" alt="animated sparse pathlet evidence">` : '<div class="field warning">Animated evidence missing.</div>';
 const strip=row.evidence_static_strip_path ? `<img class="strip" src="${{esc(row.evidence_static_strip_path)}}" alt="static evidence fallback">` : '';
 document.getElementById('card').innerHTML=`
 <h2>${{esc(row.step2m3t_review_category)}}</h2>
 <p class="warning">Do not infer identity. Judge only visual handoff coherence.</p>
 ${{animation}}${{strip}}
 <div class="card-actions">
 <button class="accept" onclick="decide('accept_sparse_pathlet_for_visual_handoff')">A Accept</button>
 <button class="reject" onclick="decide('reject_or_quarantine_sparse_pathlet')">X Reject</button>
 <button class="unsure" onclick="decide('unsure_needs_later_review')">U Unsure</button>
 <button class="export" onclick="exportDecisions()">Export M3T decisions JSON</button>
 </div>
 <label class="label" for="notes">Notes</label>
 <textarea id="notes" placeholder="Optional reviewer note">${{esc(decisions[row.step2m3t_review_candidate_id]?.notes||'')}}</textarea>
 <div class="grid">
 ${{field('candidate',row.step2m3t_review_candidate_id)}}
 ${{field('subject',row.review_subject_type)}}
 ${{field('pathlet',row.pathlet_id||'')}}
 ${{field('edge',row.continuity_edge_id||'')}}
 ${{field('frames',`${{row.min_frame_sequence ?? row.source_frame_sequence ?? ''}} -> ${{row.max_frame_sequence ?? row.target_frame_sequence ?? ''}}`)}}
 ${{field('evidence',`${{row.evidence_type || ''}} / ${{row.current_visual_evidence_version || ''}}`)}}
 </div>`;
 setStatus(decisions[row.step2m3t_review_candidate_id]?.human_review_decision||'ready');
}}
async function decide(decision){{
 const row=(state.rows||[])[index]; if(!row)return;
 const notes=(document.getElementById('notes')?.value)||'';
 const payload={{step2m3t_review_candidate_id:row.step2m3t_review_candidate_id,human_review_decision:decision,notes:notes,reviewer_name:localStorage.getItem('step2m3t_reviewer_name')||''}};
 setStatus('saving to disk');
 try{{
   const response=await fetch(AUTOSAVE_ENDPOINT,{{method:'POST',headers:{{'content-type':'application/json'}},body:JSON.stringify(payload)}});
   const data=await response.json().catch(()=>({{}}));
   if(!response.ok || data.success!==true){{throw new Error(data.error || `HTTP ${{response.status}}`);}}
   decisions[row.step2m3t_review_candidate_id]={{...row,...payload,reviewed_at:new Date().toISOString(),current_review_version:'{M3T_CURRENT_REVIEW_VERSION}',review_decisions_collected_with_review_version:'{M3T_CURRENT_REVIEW_VERSION}',current_visual_evidence_version:'{M3T_CURRENT_VISUAL_EVIDENCE_VERSION}',review_decisions_collected_with_visual_evidence_version:'{M3T_CURRENT_VISUAL_EVIDENCE_VERSION}',localStorage_backup_only:true}};
   localStorage.setItem(STORAGE_KEY,JSON.stringify(decisions));
   setStatus(`saved to disk (${{data.reviewed_count}}/${{data.total_review_candidates}})`);
   if(index<(state.rows||[]).length-1){{index++;show();}}
 }}catch(err){{
   setStatus(`save failed: ${{err.message || err}}`,true);
 }}
}}
function nextCard(){{index=Math.min((state.rows||[]).length-1,index+1);show();}}
function prevCard(){{index=Math.max(0,index-1);show();}}
function exportDecisions(){{
 const payload={{artifact:'step2m3t_local_review_decisions_export',exported_at:new Date().toISOString(),source:'browser_localStorage_backup_not_source_of_truth',rows:Object.values(decisions)}};
 const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}});
 const url=URL.createObjectURL(blob);
 const a=document.createElement('a');
 a.href=url;
 a.download='step2m3t_local_review_decisions_export.json';
 document.body.appendChild(a);
 a.click();
 a.remove();
 setTimeout(()=>URL.revokeObjectURL(url),500);
}}
document.addEventListener('keydown',ev=>{{if(ev.key==='a'||ev.key==='A')decide('accept_sparse_pathlet_for_visual_handoff');if(ev.key==='x'||ev.key==='X')decide('reject_or_quarantine_sparse_pathlet');if(ev.key==='u'||ev.key==='U')decide('unsure_needs_later_review');if(ev.key==='ArrowRight')nextCard();if(ev.key==='ArrowLeft')prevCard();}});
show();
</script>
</body>
</html>"""


def build_manifest_validation(
    *,
    candidate_summary: dict[str, Any],
    selected_summary: dict[str, Any],
    pathlet_summary: dict[str, Any],
    quarantine_summary: dict[str, Any],
    review_payload: dict[str, Any],
    pathlets: list[dict[str, Any]],
    selected_edges: list[dict[str, Any]],
    quarantined_edges: list[dict[str, Any]],
    review_progress: dict[str, Any] | None = None,
    review_persistence_issues: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    m3_summary = read_json(STEP2M3_ACCEPTED_EDGE_SUMMARY_PATH)
    m3_freeze = read_json(STEP2M3_FREEZE_CANDIDATE_MANIFEST_PATH)
    m3r_progress = read_json(STEP2M3R_REVIEW_PROGRESS_SUMMARY_PATH)
    m3s_manifest = read_json(STEP2M3S_HANDOFF_MANIFEST_PATH)
    m3s_freeze = read_json(STEP2M3S_FREEZE_CANDIDATE_MANIFEST_PATH)
    pathlets_over_cap = safe_int(pathlet_summary.get("pathlets_over_cap_count"), 1)
    pathlets_duplicate = safe_int(pathlet_summary.get("pathlets_with_duplicate_frame_members_count"), 1)
    pathlets_branch_merge = safe_int(pathlet_summary.get("pathlets_with_branch_merge_count"), 1)
    selected_direct_rejects = sum(1 for edge in selected_edges if edge.get("m3r_edge_level_rejected") is True)
    selected_direct_unsure = sum(1 for edge in selected_edges if edge.get("m3r_edge_level_unsure") is True)
    reviewed_payload = read_m3t_reviewed_decisions()
    review_progress = review_progress or m3t_review_progress_payload(review_payload, reviewed_payload)
    review_persistence_issues = review_persistence_issues or []
    persisted_rows = safe_int(review_progress.get("persisted_reviewed_decision_rows"), 0)
    persisted_review_errors = list(review_progress.get("validation_errors", []))
    persisted_review_valid = persisted_rows == 0 or (
        not persisted_review_errors
        and review_progress.get("persisted_decision_row_count_matches_reviewed_candidates") is True
        and review_progress.get("review_decisions_version_matches_current") is True
        and review_progress.get("review_decisions_visual_evidence_version_matches_current") is True
        and review_progress.get("forbidden_keys_present", []) == []
    )
    all_outputs = [
        candidate_summary,
        selected_summary,
        pathlet_summary,
        quarantine_summary,
        review_payload,
        reviewed_payload,
        review_progress,
        *pathlets,
        *selected_edges[:100],
        *quarantined_edges[:100],
    ]
    forbidden = sorted({key for payload in all_outputs for key in forbidden_keys_present(payload)})
    gate_checks = {
        "m3_freeze_candidate_created": m3_freeze.get("step2m3_freeze_candidate_created") is True,
        "m3r_topology_review_completed": m3r_progress.get("topology_review_completed") is True,
        "m3s_handoff_safe_candidate": m3s_manifest.get("handoff_safe_candidate") is True,
        "m3s_freeze_candidate_created": m3s_freeze.get("step2m3s_freeze_candidate_created") is True,
        "pathlets_cap_safe": pathlets_over_cap == 0,
        "pathlets_max_one_member_per_frame": pathlets_duplicate == 0,
        "pathlets_branch_merge_free": pathlets_branch_merge == 0,
        "selected_edges_no_direct_m3r_reject": selected_direct_rejects == 0,
        "selected_edges_no_direct_m3r_unsure": selected_direct_unsure == 0,
        "forbidden_keys_absent": forbidden == [],
        "m3t_persisted_review_decisions_valid": persisted_review_valid and not review_persistence_issues,
        "production_ready_false": PRODUCTION_READY is False,
        "no_auto_promotion_true": NO_AUTO_PROMOTION is True,
    }
    issues = [
        {"severity": "blocking", "issue_code": code, "message": f"Step2.M3T gate failed: {code}"}
        for code, passed in gate_checks.items()
        if not passed
    ]
    issues.extend(
        {
            "severity": "blocking",
            "issue_code": str(error.get("issue_code", "step2m3t_persisted_review_validation_error")),
            "message": f"Step2.M3T persisted review validation failed: {error}",
        }
        for error in persisted_review_errors
    )
    issues.extend(review_persistence_issues)
    freeze_candidate = all(gate_checks.values()) and not issues
    future_handoff_gate_checks = {
        **gate_checks,
        "persisted_review_file_exists": review_progress.get("persisted_review_decision_file_exists") is True,
        "persisted_reviewed_candidates_complete": safe_int(review_progress.get("reviewed_candidates"), 0) == M3T_REVIEW_TARGET_CARDS,
        "total_review_candidates_expected": safe_int(review_progress.get("total_review_candidates"), 0) == M3T_REVIEW_TARGET_CARDS,
        "sparse_pathlet_review_completed": review_progress.get("sparse_pathlet_review_completed") is True,
        "review_decisions_version_matches_current": review_progress.get("review_decisions_version_matches_current") is True,
        "review_decisions_visual_evidence_version_matches_current": review_progress.get("review_decisions_visual_evidence_version_matches_current") is True,
    }
    future_handoff_ready_candidate = all(future_handoff_gate_checks.values()) and not issues
    manifest = guardrail_stamp(
        {
            "artifact": "step2m3t_handoff_manifest",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "source_m3_folder": str(STEP2M3_OUTPUT_DIR.resolve()),
            "source_m3r_folder": str(STEP2M3R_OUTPUT_DIR.resolve()),
            "source_m3s_folder": str(STEP2M3S_OUTPUT_DIR.resolve()),
            "m3_source_accepted_edges_loaded": m3_summary.get("accepted_edge_count", 0),
            "m3r_decisions_loaded": m3r_progress.get("reviewed_candidates", 0),
            "m3s_handoff_seed_edges_loaded": m3s_manifest.get("m3s_accepted_handoff_edge_count", 0),
            "sparse_selected_edge_count": selected_summary.get("selected_sparse_edge_count", 0),
            "sparse_pathlet_count": pathlet_summary.get("sparse_pathlet_count", 0),
            "quarantined_edge_count": quarantine_summary.get("topology_quarantined_edge_count", 0),
            "quarantined_pathlet_count": quarantine_summary.get("topology_quarantined_pathlet_count", 0),
            "review_queue_size": len(rows_from_payload(review_payload)),
            "handoff_safe_candidate": freeze_candidate,
            "future_handoff_ready_candidate": future_handoff_ready_candidate,
            "forbidden_keys_present": forbidden,
            "gate_checks": gate_checks,
            "future_handoff_gate_checks": future_handoff_gate_checks,
            "reviewed_candidates": review_progress.get("reviewed_candidates", 0),
            "sparse_pathlet_review_completed": review_progress.get("sparse_pathlet_review_completed", False),
            "review_decisions_visual_evidence_version_matches_current": review_progress.get("review_decisions_visual_evidence_version_matches_current", False),
            **m3t_guardrail_fields(),
        }
    )
    validation = guardrail_stamp(
        {
            "artifact": "step2m3t_validation_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "sparse_selected_edge_count": selected_summary.get("selected_sparse_edge_count", 0),
            "sparse_pathlet_count": pathlet_summary.get("sparse_pathlet_count", 0),
            "pathlets_over_cap_count": pathlets_over_cap,
            "pathlets_with_duplicate_frame_members_count": pathlets_duplicate,
            "pathlets_with_branch_merge_count": pathlets_branch_merge,
            "quarantined_edge_count": quarantine_summary.get("topology_quarantined_edge_count", 0),
            "quarantined_pathlet_count": quarantine_summary.get("topology_quarantined_pathlet_count", 0),
            "review_queue_size": len(rows_from_payload(review_payload)),
            "step2m3t_freeze_candidate_created": freeze_candidate,
            "future_handoff_ready_candidate": future_handoff_ready_candidate,
            "total_review_candidates": review_progress.get("total_review_candidates", len(rows_from_payload(review_payload))),
            "reviewed_candidates": review_progress.get("reviewed_candidates", 0),
            "accepted_count": review_progress.get("accepted_count", 0),
            "rejected_count": review_progress.get("rejected_count", 0),
            "unsure_count": review_progress.get("unsure_count", 0),
            "sparse_pathlet_review_completed": review_progress.get("sparse_pathlet_review_completed", False),
            "review_decisions_version_matches_current": review_progress.get("review_decisions_version_matches_current", False),
            "review_decisions_visual_evidence_version_matches_current": review_progress.get("review_decisions_visual_evidence_version_matches_current", False),
            "forbidden_keys_present": forbidden,
            "gate_checks": gate_checks,
            "future_handoff_gate_checks": future_handoff_gate_checks,
            **m3t_guardrail_fields(),
        }
    )
    audit = guardrail_stamp(
        {
            "artifact": "step2m3t_safety_guardrail_audit",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "step2_visual_continuity_root": str(STEP2_VISUAL_CONTINUITY_DIR.resolve()),
            "m3_read_root": str(STEP2M3_OUTPUT_DIR.resolve()),
            "m3r_read_root": str(STEP2M3R_OUTPUT_DIR.resolve()),
            "m3s_read_root": str(STEP2M3S_OUTPUT_DIR.resolve()),
            "m3t_write_root": str(STEP2M3T_OUTPUT_DIR.resolve()),
            "no_m3t_writes_to_m1_m2_m3_m3r_m3s": True,
            "reviewed_sparse_pathlet_decisions_path": str(STEP2M3T_REVIEWED_SPARSE_PATHLET_DECISIONS_PATH.resolve()),
            "forbidden_keys_present": forbidden,
            **m3t_guardrail_fields(),
        }
    )
    issue_register = guardrail_stamp(
        {
            "artifact": "step2m3t_issue_register",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "blocking_issue_count": sum(1 for issue in issues if issue.get("severity") == "blocking"),
            "rows": issues,
            **m3t_guardrail_fields(),
        }
    )
    freeze_manifest = guardrail_stamp(
        {
            "artifact": "step2m3t_freeze_candidate_manifest",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "step2m3t_freeze_candidate_created": freeze_candidate,
            "future_handoff_ready_candidate": future_handoff_ready_candidate,
            "human_approved": False,
            "safe_to_apply_globally": False,
            "production_ready": PRODUCTION_READY,
            "no_auto_promotion": NO_AUTO_PROMOTION,
            "review_required_before_any_future_promotion": True,
            "persisted_review_required_before_future_handoff": True,
            "validation_summary_path": str(STEP2M3T_VALIDATION_SUMMARY_PATH.resolve()),
            "handoff_manifest_path": str(STEP2M3T_HANDOFF_MANIFEST_PATH.resolve()),
            "review_progress_summary_path": str(STEP2M3T_REVIEW_PROGRESS_SUMMARY_PATH.resolve()),
            "reviewed_sparse_pathlet_decisions_path": str(STEP2M3T_REVIEWED_SPARSE_PATHLET_DECISIONS_PATH.resolve()),
            "forbidden_keys_present": forbidden,
            "gate_checks": gate_checks,
            "future_handoff_gate_checks": future_handoff_gate_checks,
            **m3t_guardrail_fields(),
        }
    )
    for payload in [manifest, validation, audit, issue_register, freeze_manifest]:
        assert_no_forbidden_keys(payload)
    return manifest, validation, audit, issue_register, freeze_manifest


def build_step2m3t_sparse_pathlets() -> dict[str, Any]:
    assert_m3t_output_path_isolation()
    ensure_dir(STEP2M3T_OUTPUT_DIR)
    sparse = build_sparse_outputs()
    candidate_summary, candidate_sample = candidate_edge_payloads(sparse["candidate_edges"])
    selected_summary, selected_sample = selected_edge_payloads(sparse["selected_edges"], len(sparse["candidate_edges"]))
    pathlet_payload, pathlet_summary, pathlet_sample = pathlet_payloads(sparse["pathlets"])
    quarantined_pathlet_payload, quarantine_summary = quarantine_payloads(sparse["quarantined_edges"], sparse["quarantined_pathlets"])
    review_payload = build_review_candidates(sparse["pathlets"], sparse["selected_edges"], sparse["quarantined_edges"])
    contact_sheet = render_review_contact_sheet(review_payload)
    reviewed_payload = read_m3t_reviewed_decisions()
    progress, decision_summary = review_progress_payload(review_payload, reviewed_payload)
    handoff, validation, audit, issue_register, freeze_manifest = build_manifest_validation(
        candidate_summary=candidate_summary,
        selected_summary=selected_summary,
        pathlet_summary=pathlet_summary,
        quarantine_summary=quarantine_summary,
        review_payload=review_payload,
        pathlets=sparse["pathlets"],
        selected_edges=sparse["selected_edges"],
        quarantined_edges=sparse["quarantined_edges"],
        review_progress=progress,
    )
    for payload, path in [
        (candidate_summary, STEP2M3T_SPARSE_CANDIDATE_EDGE_SUMMARY_PATH),
        (candidate_sample, STEP2M3T_SPARSE_CANDIDATE_EDGE_SAMPLE_PATH),
        (selected_summary, STEP2M3T_SELECTED_SPARSE_EDGE_SUMMARY_PATH),
        (selected_sample, STEP2M3T_SELECTED_SPARSE_EDGE_SAMPLE_PATH),
        (pathlet_payload, STEP2M3T_SPARSE_PATHLETS_PATH),
        (pathlet_summary, STEP2M3T_SPARSE_PATHLET_SUMMARY_PATH),
        (pathlet_sample, STEP2M3T_SPARSE_PATHLET_SAMPLE_PATH),
        (quarantined_pathlet_payload, STEP2M3T_TOPOLOGY_QUARANTINED_PATHLETS_PATH),
        (quarantine_summary, STEP2M3T_TOPOLOGY_QUARANTINE_SUMMARY_PATH),
        (review_payload, STEP2M3T_REVIEW_CANDIDATE_ROWS_PATH),
        (progress, STEP2M3T_REVIEW_PROGRESS_SUMMARY_PATH),
        (decision_summary, STEP2M3T_REVIEW_DECISION_SUMMARY_PATH),
        (handoff, STEP2M3T_HANDOFF_MANIFEST_PATH),
        (validation, STEP2M3T_VALIDATION_SUMMARY_PATH),
        (audit, STEP2M3T_SAFETY_GUARDRAIL_AUDIT_PATH),
        (issue_register, STEP2M3T_ISSUE_REGISTER_PATH),
        (freeze_manifest, STEP2M3T_FREEZE_CANDIDATE_MANIFEST_PATH),
    ]:
        assert_no_forbidden_keys(payload)
        write_json(path, payload)
    write_text(STEP2M3T_REVIEW_UI_HTML_PATH, review_ui_html(review_payload))
    write_jsonl_gz(STEP2M3T_SPARSE_CANDIDATE_EDGES_JSONL_GZ_PATH, sparse["candidate_edges"])
    write_jsonl_gz(STEP2M3T_SELECTED_SPARSE_EDGES_JSONL_GZ_PATH, sparse["selected_edges"])
    write_jsonl_gz(STEP2M3T_TOPOLOGY_QUARANTINED_EDGES_JSONL_GZ_PATH, sparse["quarantined_edges"])
    return {
        "m3_source_accepted_edges_loaded": sparse["m3_source_accepted_edges_loaded"],
        "m3r_decisions_loaded": sparse["m3r_decisions_loaded"],
        "m3s_handoff_seed_edges_loaded": sparse["m3s_handoff_seed_edges_loaded"],
        "sparse_candidate_edge_summary": candidate_summary,
        "selected_sparse_edge_summary": selected_summary,
        "sparse_pathlet_summary": pathlet_summary,
        "topology_quarantine_summary": quarantine_summary,
        "review_candidates": review_payload,
        "contact_sheet": contact_sheet,
        "handoff_manifest": handoff,
        "validation_summary": validation,
        "safety_guardrail_audit": audit,
        "issue_register": issue_register,
        "freeze_candidate_manifest": freeze_manifest,
    }


def validate_step2m3t_sparse_pathlets() -> dict[str, Any]:
    assert_m3t_output_path_isolation()
    candidate_summary = read_json(STEP2M3T_SPARSE_CANDIDATE_EDGE_SUMMARY_PATH)
    selected_summary = read_json(STEP2M3T_SELECTED_SPARSE_EDGE_SUMMARY_PATH)
    pathlet_summary = read_json(STEP2M3T_SPARSE_PATHLET_SUMMARY_PATH)
    quarantine_summary = read_json(STEP2M3T_TOPOLOGY_QUARANTINE_SUMMARY_PATH)
    review_payload = read_json(STEP2M3T_REVIEW_CANDIDATE_ROWS_PATH)
    existing_progress = read_json(STEP2M3T_REVIEW_PROGRESS_SUMMARY_PATH) if STEP2M3T_REVIEW_PROGRESS_SUMMARY_PATH.exists() else {}
    review_persistence_issues: list[dict[str, Any]] = []
    if safe_int(existing_progress.get("reviewed_candidates"), 0) > 0 and not STEP2M3T_REVIEWED_SPARSE_PATHLET_DECISIONS_PATH.exists():
        review_persistence_issues.append(
            {
                "severity": "blocking",
                "issue_code": "step2m3t_ui_reviewed_but_persisted_decision_file_missing",
                "message": "Step2.M3T progress claimed reviewed candidates, but the persisted decision file is missing.",
            }
        )
    reviewed_payload = read_m3t_reviewed_decisions()
    progress, decision_summary = refresh_m3t_review_summaries(review_payload, reviewed_payload)
    pathlets = rows_from_payload(read_json(STEP2M3T_SPARSE_PATHLETS_PATH))
    selected_edges = read_jsonl_gz_rows(STEP2M3T_SELECTED_SPARSE_EDGES_JSONL_GZ_PATH)
    quarantined_edges = read_jsonl_gz_rows(STEP2M3T_TOPOLOGY_QUARANTINED_EDGES_JSONL_GZ_PATH)
    handoff, validation, audit, issue_register, freeze_manifest = build_manifest_validation(
        candidate_summary=candidate_summary,
        selected_summary=selected_summary,
        pathlet_summary=pathlet_summary,
        quarantine_summary=quarantine_summary,
        review_payload=review_payload,
        pathlets=pathlets,
        selected_edges=selected_edges,
        quarantined_edges=quarantined_edges,
        review_progress=progress,
        review_persistence_issues=review_persistence_issues,
    )
    for payload, path in [
        (handoff, STEP2M3T_HANDOFF_MANIFEST_PATH),
        (validation, STEP2M3T_VALIDATION_SUMMARY_PATH),
        (audit, STEP2M3T_SAFETY_GUARDRAIL_AUDIT_PATH),
        (issue_register, STEP2M3T_ISSUE_REGISTER_PATH),
        (freeze_manifest, STEP2M3T_FREEZE_CANDIDATE_MANIFEST_PATH),
    ]:
        assert_no_forbidden_keys(payload)
        write_json(path, payload)
    return {
        "handoff_manifest": handoff,
        "review_progress": progress,
        "review_decision": decision_summary,
        "validation_summary": validation,
        "safety_guardrail_audit": audit,
        "issue_register": issue_register,
        "freeze_candidate_manifest": freeze_manifest,
    }


def write_step2m3t_review_pack() -> dict[str, Any]:
    assert_m3t_output_path_isolation()
    ensure_dir(STEP2M3T_REVIEW_PACK_DIR)
    files = [
        STEP2M3T_SPARSE_CANDIDATE_EDGE_SUMMARY_PATH,
        STEP2M3T_SPARSE_CANDIDATE_EDGE_SAMPLE_PATH,
        STEP2M3T_SELECTED_SPARSE_EDGE_SUMMARY_PATH,
        STEP2M3T_SELECTED_SPARSE_EDGE_SAMPLE_PATH,
        STEP2M3T_SPARSE_PATHLETS_PATH,
        STEP2M3T_SPARSE_PATHLET_SUMMARY_PATH,
        STEP2M3T_SPARSE_PATHLET_SAMPLE_PATH,
        STEP2M3T_TOPOLOGY_QUARANTINED_PATHLETS_PATH,
        STEP2M3T_TOPOLOGY_QUARANTINE_SUMMARY_PATH,
        STEP2M3T_REVIEW_CANDIDATE_ROWS_PATH,
        STEP2M3T_REVIEW_UI_HTML_PATH,
        STEP2M3T_REVIEW_CONTACT_SHEET_PATH,
        STEP2M3T_REVIEW_PROGRESS_SUMMARY_PATH,
        STEP2M3T_REVIEW_DECISION_SUMMARY_PATH,
        STEP2M3T_REVIEWED_SPARSE_PATHLET_DECISIONS_PATH,
        STEP2M3T_HANDOFF_MANIFEST_PATH,
        STEP2M3T_VALIDATION_SUMMARY_PATH,
        STEP2M3T_SAFETY_GUARDRAIL_AUDIT_PATH,
        STEP2M3T_ISSUE_REGISTER_PATH,
        STEP2M3T_FREEZE_CANDIDATE_MANIFEST_PATH,
    ]
    copied: list[str] = []
    for path in files:
        if not path.exists():
            continue
        destination = STEP2M3T_REVIEW_PACK_DIR / path.name
        shutil.copyfile(path, destination)
        copied.append(str(destination.resolve()))
    copied_evidence: list[str] = []
    for source_root in [STEP2M3T_PATHLET_ANIMATIONS_DIR, STEP2M3T_PATHLET_STRIPS_DIR, STEP2M3T_EDGE_BURST_ANIMATIONS_DIR, STEP2M3T_EDGE_BURST_STRIPS_DIR]:
        if not source_root.exists():
            continue
        relative_root = source_root.relative_to(STEP2M3T_OUTPUT_DIR)
        for extension in ("*.gif", "*.jpg", "*.mp4"):
            for source_path in source_root.glob(extension):
                destination = STEP2M3T_REVIEW_PACK_DIR / relative_root / source_path.name
                ensure_dir(destination.parent)
                shutil.copyfile(source_path, destination)
                copied_evidence.append(str(destination.resolve()))
    validation = read_json(STEP2M3T_VALIDATION_SUMMARY_PATH)
    manifest = guardrail_stamp(
        {
            "artifact": "step2m3t_review_pack_manifest",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "review_pack_dir": str(STEP2M3T_REVIEW_PACK_DIR.resolve()),
            "copied_files": copied,
            "copied_visual_evidence_files": copied_evidence,
            "step2m3t_freeze_candidate_created": validation.get("step2m3t_freeze_candidate_created", False),
            "autosave_endpoint": "/api/step2m3t/review-decision",
            "reviewed_sparse_pathlet_decisions_path": str(STEP2M3T_REVIEWED_SPARSE_PATHLET_DECISIONS_PATH.resolve()),
            **m3t_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(manifest)
    write_json(STEP2M3T_REVIEW_PACK_MANIFEST_PATH, manifest)
    return manifest


def prepare_step2m3t_review_ui(host: str = "127.0.0.1", port: int = 8787) -> dict[str, Any]:
    assert_m3t_output_path_isolation()
    review_payload = read_json(STEP2M3T_REVIEW_CANDIDATE_ROWS_PATH)
    write_text(STEP2M3T_REVIEW_UI_HTML_PATH, review_ui_html(review_payload))
    reviewed_payload = read_m3t_reviewed_decisions()
    progress, decision_summary = refresh_m3t_review_summaries(review_payload, reviewed_payload)
    manifest = guardrail_stamp(
        {
            "artifact": "step2m3t_review_ui_manifest",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "url": f"http://{host}:{port}/",
            "review_ui_html_path": str(STEP2M3T_REVIEW_UI_HTML_PATH.resolve()),
            "review_candidate_rows_path": str(STEP2M3T_REVIEW_CANDIDATE_ROWS_PATH.resolve()),
            "reviewed_sparse_pathlet_decisions_path": str(STEP2M3T_REVIEWED_SPARSE_PATHLET_DECISIONS_PATH.resolve()),
            "review_progress_summary_path": str(STEP2M3T_REVIEW_PROGRESS_SUMMARY_PATH.resolve()),
            "review_decision_summary_path": str(STEP2M3T_REVIEW_DECISION_SUMMARY_PATH.resolve()),
            "total_review_candidates": progress.get("total_review_candidates", len(rows_from_payload(review_payload))),
            "reviewed_candidates": progress.get("reviewed_candidates", 0),
            "sparse_pathlet_review_completed": decision_summary.get("sparse_pathlet_review_completed", False),
            "current_review_version": M3T_CURRENT_REVIEW_VERSION,
            "current_visual_evidence_version": M3T_CURRENT_VISUAL_EVIDENCE_VERSION,
            "autosave_endpoint": "/api/step2m3t/review-decision",
            **m3t_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(manifest)
    return manifest


class Step2M3TReviewHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        raw_path = unquote(urlparse(self.path).path.lstrip("/"))
        file_path = STEP2M3T_REVIEW_UI_HTML_PATH.resolve() if not raw_path else (STEP2M3T_OUTPUT_DIR / raw_path).resolve()
        root = STEP2M3T_OUTPUT_DIR.resolve()
        if file_path != root and root not in file_path.parents:
            self.send_error(403)
            return
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_review_decision(self) -> None:
        if urlparse(self.path).path != "/api/step2m3t/review-decision":
            self._send_json(404, {"success": False, "error": "unknown_endpoint"})
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            _decision, reviewed_payload, progress = save_m3t_review_decision(payload)
            self._send_json(
                200,
                {
                    "success": True,
                    "reviewed_count": progress.get("reviewed_candidates", len(decision_rows_from_payload(reviewed_payload))),
                    "total_review_candidates": progress.get("total_review_candidates", 0),
                    "accepted_count": progress.get("accepted_count", 0),
                    "rejected_count": progress.get("rejected_count", 0),
                    "unsure_count": progress.get("unsure_count", 0),
                },
            )
        except Exception as exc:  # pragma: no cover - surfaced to browser and smoke tests
            self._send_json(400, {"success": False, "error": str(exc)})

    def do_POST(self) -> None:
        self._handle_review_decision()

    def do_PUT(self) -> None:
        self._handle_review_decision()


def serve_step2m3t_review_ui(host: str = "127.0.0.1", port: int = 8787) -> None:
    prepare_step2m3t_review_ui(host=host, port=port)
    server = ThreadingHTTPServer((host, port), Step2M3TReviewHandler)
    print(f"Serving Step2.M3T sparse pathlet review UI at http://{host}:{port}/")
    print(f"Autosave endpoint: http://{host}:{port}/api/step2m3t/review-decision")
    server.serve_forever()


def print_step2m3t_review_ui_console(manifest: dict[str, Any]) -> None:
    print(f"step2m3t_review_ui_html_path: {manifest['review_ui_html_path']}")
    print(f"step2m3t_review_candidate_rows_path: {manifest['review_candidate_rows_path']}")
    print(f"step2m3t_reviewed_sparse_pathlet_decisions_path: {manifest['reviewed_sparse_pathlet_decisions_path']}")
    print(f"step2m3t_review_progress_summary_path: {manifest['review_progress_summary_path']}")
    print(f"total_review_candidates: {manifest.get('total_review_candidates', 0)}")
    print(f"reviewed_candidates: {manifest.get('reviewed_candidates', 0)}")
    print(f"current_visual_evidence_version: {manifest.get('current_visual_evidence_version')}")
    print(f"autosave_endpoint: {manifest.get('autosave_endpoint')}")
    print("visual_only_warning=VISUAL_ONLY_NOT_METRIC")
    print("production_ready=false")
    print("no_auto_promotion=true")
    print("human_approved=false")


def print_step2m3t_console(outputs: dict[str, Any]) -> None:
    validation = outputs["validation_summary"]
    freeze = outputs["freeze_candidate_manifest"]
    print(f"step2m3t_output_dir: {STEP2M3T_OUTPUT_DIR.resolve()}")
    print(f"m3_source_accepted_edges_loaded: {outputs.get('m3_source_accepted_edges_loaded', 0)}")
    print(f"m3r_decisions_loaded: {outputs.get('m3r_decisions_loaded', 0)}")
    print(f"m3s_handoff_seed_edges_loaded: {outputs.get('m3s_handoff_seed_edges_loaded', 0)}")
    print(f"sparse_selected_edge_count: {validation.get('sparse_selected_edge_count', 0)}")
    print(f"sparse_pathlet_count: {validation.get('sparse_pathlet_count', 0)}")
    print(f"pathlets_over_cap_count: {validation.get('pathlets_over_cap_count', 0)}")
    print(f"pathlets_with_duplicate_frame_members_count: {validation.get('pathlets_with_duplicate_frame_members_count', 0)}")
    print(f"pathlets_with_branch_merge_count: {validation.get('pathlets_with_branch_merge_count', 0)}")
    print(f"quarantined_edge_count: {validation.get('quarantined_edge_count', 0)}")
    print(f"quarantined_pathlet_count: {validation.get('quarantined_pathlet_count', 0)}")
    print(f"review_queue_size: {validation.get('review_queue_size', 0)}")
    print(f"forbidden_keys_present: {json.dumps(validation.get('forbidden_keys_present', []))}")
    print(f"production_ready={str(validation.get('production_ready')).lower()}")
    print(f"no_auto_promotion={str(validation.get('no_auto_promotion')).lower()}")
    print(f"human_approved={str(validation.get('human_approved')).lower()}")
    print(f"step2m3t_freeze_candidate_created={str(freeze.get('step2m3t_freeze_candidate_created')).lower()}")


def print_step2m3t_validation_console(outputs: dict[str, Any]) -> None:
    validation = outputs["validation_summary"]
    freeze = outputs["freeze_candidate_manifest"]
    issues = outputs["issue_register"]
    print(f"step2m3t_validation_summary_path: {STEP2M3T_VALIDATION_SUMMARY_PATH.resolve()}")
    print(f"step2m3t_freeze_candidate_manifest_path: {STEP2M3T_FREEZE_CANDIDATE_MANIFEST_PATH.resolve()}")
    print(f"blocking_issue_count: {issues.get('blocking_issue_count', 0)}")
    print(f"forbidden_keys_present: {json.dumps(validation.get('forbidden_keys_present', []))}")
    print(f"step2m3t_freeze_candidate_created={str(freeze.get('step2m3t_freeze_candidate_created')).lower()}")


def print_step2m3t_review_pack_console(manifest: dict[str, Any]) -> None:
    print(f"step2m3t_review_pack_manifest_path: {STEP2M3T_REVIEW_PACK_MANIFEST_PATH.resolve()}")
    print(f"step2m3t_review_pack_dir: {STEP2M3T_REVIEW_PACK_DIR.resolve()}")
    print(f"copied_files: {len(manifest.get('copied_files', []))}")
    print(f"copied_visual_evidence_files: {len(manifest.get('copied_visual_evidence_files', []))}")
    print(f"production_ready={str(manifest.get('production_ready')).lower()}")
    print(f"no_auto_promotion={str(manifest.get('no_auto_promotion')).lower()}")
