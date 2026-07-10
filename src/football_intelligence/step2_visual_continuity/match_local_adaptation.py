# ruff: noqa: E501

from __future__ import annotations

import gzip
import hashlib
import json
import mimetypes
import shutil
from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

from football_intelligence.paths import CLIP_ID, MATCH_ID, ensure_dir
from football_intelligence.step2_visual_continuity import remediation as m1r_remediation
from football_intelligence.step2_visual_continuity import render_review
from football_intelligence.step2_visual_continuity.io import (
    STEP2M1_EDGE_CANDIDATE_ROWS_JSONL_GZ_PATH,
    STEP2M1_NODE_ROWS_PATH,
    STEP2M1_OUTPUT_DIR,
    STEP2M1_REVIEW_CANDIDATE_ROWS_PATH,
    STEP2M1_REVIEWED_DECISIONS_PATH,
    STEP2M1R_ADAPTATION_SAFETY_MANIFEST_PATH,
    STEP2M1R_TARGETED_REVIEW_CANDIDATE_ROWS_PATH,
    STEP2M2_ADAPTED_EDGE_CANDIDATE_SAMPLE_PATH,
    STEP2M2_ADAPTED_EDGE_CANDIDATE_SUMMARY_PATH,
    STEP2M2_ADAPTED_EDGE_CANDIDATES_JSONL_GZ_PATH,
    STEP2M2_BURST_OVERLAY_ALIGNMENT_SUMMARY_PATH,
    STEP2M2_BURST_OVERLAY_DEBUG_ROWS_PATH,
    STEP2M2_BURST_OVERLAY_QA_DIR,
    STEP2M2_FREEZE_CANDIDATE_MANIFEST_PATH,
    STEP2M2_ISSUE_REGISTER_PATH,
    STEP2M2_MATCH_LOCAL_ADAPTATION_PROFILE_PATH,
    STEP2M2_OUTPUT_DIR,
    STEP2M2_REVIEW_BURST_CLIPS_DIR,
    STEP2M2_REVIEW_BURST_COMPARISON_STRIPS_DIR,
    STEP2M2_REVIEW_BURST_RAW_STRIPS_DIR,
    STEP2M2_REVIEW_BURST_STRIPS_DIR,
    STEP2M2_REVIEW_CONTACT_SHEET_PATH,
    STEP2M2_REVIEW_DECISION_SUMMARY_PATH,
    STEP2M2_REVIEW_PACK_DIR,
    STEP2M2_REVIEW_PACK_MANIFEST_PATH,
    STEP2M2_REVIEW_PROGRESS_SUMMARY_PATH,
    STEP2M2_REVIEWED_DECISIONS_PATH,
    STEP2M2_REVIEW_UI_HTML_PATH,
    STEP2M2_REVIEWED_DECISION_TRAINING_ROWS_PATH,
    STEP2M2_REVIEWED_DECISION_TRAINING_SUMMARY_PATH,
    STEP2M2_SAFETY_GUARDRAIL_AUDIT_PATH,
    STEP2M2_SOURCE_CONTEXT_IMAGES_DIR,
    STEP2M2_SOURCE_CROP_IMAGES_DIR,
    STEP2M2_TARGET_CONTEXT_IMAGES_DIR,
    STEP2M2_TARGET_CROP_IMAGES_DIR,
    STEP2M2_TARGETED_REVIEW_CANDIDATE_ROWS_PATH,
    STEP2M2_VALIDATION_SUMMARY_PATH,
    read_json,
    write_json,
    write_text,
)
from football_intelligence.step2_visual_continuity.schema import (
    ACCEPT_DECISION,
    NO_AUTO_PROMOTION,
    PRODUCTION_READY,
    REJECT_DECISION,
    UNSURE_DECISION,
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


M2_TARGET_REVIEW_CARDS = 40
M2_HARD_MAX_REVIEW_CARDS = 60
M2_BUCKET_TARGETS = {
    "safe_auto_accept_audit": 8,
    "high_risk_adapted_accept": 8,
    "changed_state_candidate": 8,
    "merged_ambiguous_sentinel": 8,
    "role_state_mismatch_split": 8,
    "official_goalkeeper_sentinel": 4,
}
M2_CURRENT_OVERLAY_VERSION = m1r_remediation.CURRENT_BURST_OVERLAY_VERSION
M2_ADAPTED_ACCEPT_THRESHOLD = 0.72
M2_ADAPTED_REJECT_THRESHOLD = 0.38


def step2m2_output_paths() -> dict[str, Path]:
    return {
        "step2m2_output_dir": STEP2M2_OUTPUT_DIR,
        "training_rows": STEP2M2_REVIEWED_DECISION_TRAINING_ROWS_PATH,
        "training_summary": STEP2M2_REVIEWED_DECISION_TRAINING_SUMMARY_PATH,
        "adaptation_profile": STEP2M2_MATCH_LOCAL_ADAPTATION_PROFILE_PATH,
        "adapted_edge_rows_jsonl_gz": STEP2M2_ADAPTED_EDGE_CANDIDATES_JSONL_GZ_PATH,
        "adapted_edge_summary": STEP2M2_ADAPTED_EDGE_CANDIDATE_SUMMARY_PATH,
        "adapted_edge_sample": STEP2M2_ADAPTED_EDGE_CANDIDATE_SAMPLE_PATH,
        "targeted_review_candidates": STEP2M2_TARGETED_REVIEW_CANDIDATE_ROWS_PATH,
        "review_ui": STEP2M2_REVIEW_UI_HTML_PATH,
        "review_contact_sheet": STEP2M2_REVIEW_CONTACT_SHEET_PATH,
        "reviewed_decisions": STEP2M2_REVIEWED_DECISIONS_PATH,
        "review_progress": STEP2M2_REVIEW_PROGRESS_SUMMARY_PATH,
        "review_decision": STEP2M2_REVIEW_DECISION_SUMMARY_PATH,
        "burst_overlay_debug": STEP2M2_BURST_OVERLAY_DEBUG_ROWS_PATH,
        "burst_overlay_alignment_summary": STEP2M2_BURST_OVERLAY_ALIGNMENT_SUMMARY_PATH,
        "validation_summary": STEP2M2_VALIDATION_SUMMARY_PATH,
        "safety_guardrail_audit": STEP2M2_SAFETY_GUARDRAIL_AUDIT_PATH,
        "issue_register": STEP2M2_ISSUE_REGISTER_PATH,
        "freeze_candidate_manifest": STEP2M2_FREEZE_CANDIDATE_MANIFEST_PATH,
        "review_pack_manifest": STEP2M2_REVIEW_PACK_MANIFEST_PATH,
    }


def assert_m2_output_path_isolation() -> None:
    m2_root = STEP2M2_OUTPUT_DIR.resolve()
    m1_root = STEP2M1_OUTPUT_DIR.resolve()
    for path in step2m2_output_paths().values():
        resolved = path.resolve()
        if resolved != m2_root and m2_root not in resolved.parents:
            raise ValueError(f"Step2.M2 output path is outside the M2 root: {resolved}")
        if resolved == m1_root or m1_root in resolved.parents:
            raise ValueError(f"Step2.M2 output path points inside the Step2.M1 sandbox: {resolved}")


def m2_guardrail_fields() -> dict[str, Any]:
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


def decision_rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    candidate_rows = payload.get("rows", payload.get("decisions", []))
    return [dict(row) for row in candidate_rows if isinstance(row, dict)] if isinstance(candidate_rows, list) else []


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    temp_path = path.with_name(f"{path.stem}.tmp.{digest}{path.suffix}")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def m2_reviewed_decision_template() -> dict[str, Any]:
    return guardrail_stamp(
        {
            "artifact": "step2m2_reviewed_visual_continuity_decisions",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "current_overlay_version": M2_CURRENT_OVERLAY_VERSION,
            "reviewed_decision_rows": 0,
            "rows": [],
            **m2_guardrail_fields(),
        }
    )


def read_m2_reviewed_decisions() -> dict[str, Any]:
    if not STEP2M2_REVIEWED_DECISIONS_PATH.exists():
        return m2_reviewed_decision_template()
    payload = read_json(STEP2M2_REVIEWED_DECISIONS_PATH)
    rows = decision_rows_from_payload(payload)
    normalized = dict(payload) if isinstance(payload, dict) else {}
    normalized.pop("decisions", None)
    normalized.update(
        {
            "artifact": "step2m2_reviewed_visual_continuity_decisions",
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "current_overlay_version": normalized.get("current_overlay_version", M2_CURRENT_OVERLAY_VERSION),
            "reviewed_decision_rows": len(rows),
            "rows": rows,
            **m2_guardrail_fields(),
        }
    )
    return guardrail_stamp(normalized)


def m2_candidate_by_review_id(review_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("step2m2_review_candidate_id", "")): row
        for row in rows_from_payload(review_payload)
        if row.get("step2m2_review_candidate_id")
    }


def m2_review_decision_row(candidate: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    decision = normalize_decision(payload.get("human_review_decision", ""))
    if decision not in {ACCEPT_DECISION, REJECT_DECISION, UNSURE_DECISION}:
        raise ValueError(f"Step2.M2 human review decision is not allowed: {decision}")
    row = {
        "step2m2_review_candidate_id": str(candidate.get("step2m2_review_candidate_id", "")),
        "continuity_edge_id": str(candidate.get("continuity_edge_id", "")),
        "source_visible_person_base_id": str(candidate.get("source_visible_person_base_id", "")),
        "target_visible_person_base_id": str(candidate.get("target_visible_person_base_id", "")),
        "source_frame_sequence": safe_int(candidate.get("source_frame_sequence"), -1),
        "target_frame_sequence": safe_int(candidate.get("target_frame_sequence"), -1),
        "step2m2_target_review_bucket": str(candidate.get("step2m2_target_review_bucket", "")),
        "source_review_bucket": str(candidate.get("source_review_bucket", "")),
        "original_proposed_edge_state": str(candidate.get("original_proposed_edge_state", "")),
        "adapted_proposed_edge_state": str(candidate.get("adapted_proposed_edge_state", "")),
        "original_edge_score_sandbox": safe_float(candidate.get("original_edge_score_sandbox"), 0.0),
        "adapted_edge_score_sandbox": safe_float(candidate.get("adapted_edge_score_sandbox"), 0.0),
        "adapted_edge_state_changed": candidate.get("adapted_edge_state_changed") is True,
        "adaptation_reasons": list(candidate.get("adaptation_reasons", [])),
        "learned_from_m1_m1r_evidence": candidate.get("learned_from_m1_m1r_evidence") is True,
        "human_review_decision": decision,
        "reviewed_at": str(payload.get("reviewed_at", "")) or utc_iso(),
        "human_confirmed": True,
        "current_overlay_version": M2_CURRENT_OVERLAY_VERSION,
        "review_decisions_collected_with_overlay_version": M2_CURRENT_OVERLAY_VERSION,
        "approve_any_identity_tracking": False,
        "approve_any_player_slot_use": False,
        "approve_any_goalkeeper_slot_use": False,
        "approve_any_metric_use": False,
        "approve_event_or_tactical_analysis": False,
        "approve_exact_22_or_exact_two_goalkeeper_forcing": False,
        "approve_official_referee_exclusion": False,
        "approve_bad_detection_deletion": False,
        "approve_production_promotion": False,
        **m2_guardrail_fields(),
    }
    visual_stamp(row)
    assert_no_forbidden_keys(row)
    return row


def validate_m2_reviewed_rows(
    review_payload: dict[str, Any],
    reviewed_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = m2_candidate_by_review_id(review_payload)
    usable: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(decision_rows_from_payload(reviewed_payload)):
        candidate_id = str(row.get("step2m2_review_candidate_id", ""))
        candidate = candidates.get(candidate_id)
        if not candidate:
            errors.append({"row_index": index, "step2m2_review_candidate_id": candidate_id, "error": "unknown_review_candidate_id"})
            continue
        if candidate_id in seen:
            errors.append({"row_index": index, "step2m2_review_candidate_id": candidate_id, "error": "duplicate_review_candidate_id"})
        seen.add(candidate_id)
        for field in [
            "continuity_edge_id",
            "source_visible_person_base_id",
            "target_visible_person_base_id",
            "source_frame_sequence",
            "target_frame_sequence",
            "step2m2_target_review_bucket",
            "source_review_bucket",
            "original_proposed_edge_state",
            "adapted_proposed_edge_state",
            "original_edge_score_sandbox",
            "adapted_edge_score_sandbox",
            "adapted_edge_state_changed",
            "learned_from_m1_m1r_evidence",
        ]:
            if str(row.get(field, "")) != str(candidate.get(field, "")):
                errors.append({"row_index": index, "step2m2_review_candidate_id": candidate_id, "error": f"{field}_mismatch"})
        if list(row.get("adaptation_reasons", [])) != list(candidate.get("adaptation_reasons", [])):
            errors.append({"row_index": index, "step2m2_review_candidate_id": candidate_id, "error": "adaptation_reasons_mismatch"})
        decision = normalize_decision(row.get("human_review_decision", ""))
        if decision not in {ACCEPT_DECISION, REJECT_DECISION, UNSURE_DECISION}:
            errors.append({"row_index": index, "step2m2_review_candidate_id": candidate_id, "error": "human_review_decision_not_allowed"})
        if row.get("human_confirmed") is not True:
            errors.append({"row_index": index, "step2m2_review_candidate_id": candidate_id, "error": "human_confirmed_true_required"})
        if row.get("current_overlay_version") != M2_CURRENT_OVERLAY_VERSION or row.get("review_decisions_collected_with_overlay_version") != M2_CURRENT_OVERLAY_VERSION:
            errors.append({"row_index": index, "step2m2_review_candidate_id": candidate_id, "error": "overlay_version_mismatch"})
        for key in [
            "approve_any_identity_tracking",
            "approve_any_player_slot_use",
            "approve_any_goalkeeper_slot_use",
            "approve_any_metric_use",
            "approve_event_or_tactical_analysis",
            "approve_exact_22_or_exact_two_goalkeeper_forcing",
            "approve_official_referee_exclusion",
            "approve_bad_detection_deletion",
            "approve_production_promotion",
        ]:
            if row.get(key) is not False:
                errors.append({"row_index": index, "step2m2_review_candidate_id": candidate_id, "error": "forbidden_approval_flag_true_or_missing", "key": key})
        required_false = ["safe_to_apply_globally", "production_ready", "human_approved"]
        for key in required_false:
            if row.get(key) is not False:
                errors.append({"row_index": index, "step2m2_review_candidate_id": candidate_id, "error": f"{key}_must_be_false"})
        if row.get("match_local_only") is not True or row.get("requires_future_match_validation") is not True:
            errors.append({"row_index": index, "step2m2_review_candidate_id": candidate_id, "error": "match_local_guardrail_invalid"})
        if row.get("visual_only_warning") != VISUAL_ONLY_WARNING or row.get("do_not_use_for_metrics") is not True or row.get("no_auto_promotion") is not True:
            errors.append({"row_index": index, "step2m2_review_candidate_id": candidate_id, "error": "visual_or_promotion_guardrail_invalid"})
        forbidden = forbidden_keys_present(row)
        if forbidden:
            errors.append({"row_index": index, "step2m2_review_candidate_id": candidate_id, "error": "forbidden_keys_present", "keys": forbidden})
        usable.append({**row, "human_review_decision": decision})
    return usable if not errors else [], errors


def write_m2_reviewed_decisions_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = guardrail_stamp(
        {
            "artifact": "step2m2_reviewed_visual_continuity_decisions",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "current_overlay_version": M2_CURRENT_OVERLAY_VERSION,
            "reviewed_decision_rows": len(rows),
            "rows": sorted(rows, key=lambda row: str(row.get("step2m2_review_candidate_id", ""))),
            **m2_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(payload)
    write_json_atomic(STEP2M2_REVIEWED_DECISIONS_PATH, payload)
    return payload


def normalize_decision(value: Any) -> str:
    decision = str(value)
    if decision == "unsure":
        return UNSURE_DECISION
    return decision


def edge_feature(edge: dict[str, Any], name: str, default: Any = 0.0) -> Any:
    summary = edge.get("edge_feature_summary", {})
    if isinstance(summary, dict) and name in summary:
        return summary.get(name, default)
    return edge.get(name, default)


def candidate_lookup(candidate_payload: dict[str, Any], *, review_id_keys: Iterable[str]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_review_id: dict[str, dict[str, Any]] = {}
    by_edge_id: dict[str, dict[str, Any]] = {}
    for row in rows_from_payload(candidate_payload):
        for key in review_id_keys:
            value = str(row.get(key, ""))
            if value:
                by_review_id[value] = row
        edge_id = str(row.get("continuity_edge_id", ""))
        if edge_id:
            by_edge_id[edge_id] = row
    return by_review_id, by_edge_id


def decision_target_fields(decision: str) -> dict[str, Any]:
    return {
        "accepted_target_label": decision == ACCEPT_DECISION,
        "rejected_target_label": decision == REJECT_DECISION,
        "unsure_target_label": decision == UNSURE_DECISION,
        "target_label": "accepted" if decision == ACCEPT_DECISION else "rejected" if decision == REJECT_DECISION else "unsure",
    }


def training_row_from_decision(decision_row: dict[str, Any], candidate: dict[str, Any], *, source_stage: str) -> dict[str, Any]:
    decision = normalize_decision(decision_row.get("human_review_decision", ""))
    candidate_id = (
        decision_row.get("step2m1_review_candidate_id")
        or decision_row.get("step2m1r_review_candidate_id")
        or candidate.get("step2m1_review_candidate_id")
        or candidate.get("step2m1r_review_candidate_id")
        or candidate.get("continuity_edge_id", "")
    )
    row = {
        "training_decision_source": source_stage,
        "candidate_id": str(candidate_id),
        "step2m1_review_candidate_id": str(decision_row.get("step2m1_review_candidate_id", candidate.get("step2m1_review_candidate_id", ""))),
        "step2m1r_review_candidate_id": str(decision_row.get("step2m1r_review_candidate_id", candidate.get("step2m1r_review_candidate_id", ""))),
        "continuity_edge_id": str(decision_row.get("continuity_edge_id", candidate.get("continuity_edge_id", ""))),
        "source_frame_sequence": safe_int(decision_row.get("source_frame_sequence", candidate.get("source_frame_sequence")), -1),
        "target_frame_sequence": safe_int(decision_row.get("target_frame_sequence", candidate.get("target_frame_sequence")), -1),
        "source_visible_person_base_id": str(decision_row.get("source_visible_person_base_id", candidate.get("source_visible_person_base_id", ""))),
        "target_visible_person_base_id": str(decision_row.get("target_visible_person_base_id", candidate.get("target_visible_person_base_id", ""))),
        "review_bucket": str(decision_row.get("review_bucket", candidate.get("review_bucket", ""))),
        "human_review_decision": decision,
        **decision_target_fields(decision),
        "edge_score_sandbox": safe_float(edge_feature(candidate, "edge_score_sandbox"), 0.0),
        "uncertainty_score": safe_float(edge_feature(candidate, "uncertainty_score"), 0.0),
        "bbox_iou": safe_float(edge_feature(candidate, "bbox_iou"), 0.0),
        "bbox_center_delta_px": safe_float(edge_feature(candidate, "bbox_center_delta_px"), 0.0),
        "footpoint_delta_px": safe_float(edge_feature(candidate, "footpoint_delta_px"), 0.0),
        "bbox_area_ratio": safe_float(edge_feature(candidate, "bbox_area_ratio"), 0.0),
        "aspect_ratio_change": safe_float(edge_feature(candidate, "aspect_ratio_change"), 0.0),
        "role_state_compatibility": safe_float(edge_feature(candidate, "role_state_compatibility"), 0.0),
        "visual_team_context_compatibility": safe_float(edge_feature(candidate, "visual_team_context_compatibility"), 0.0),
        "step1_c2c_d1c_e1c_compatibility": safe_float(edge_feature(candidate, "step1_c2c_d1c_e1c_compatibility"), 0.0),
        "crop_quality_penalty": safe_float(edge_feature(candidate, "crop_quality_penalty"), 0.0),
        "warning_conflict_flag_penalty": safe_float(edge_feature(candidate, "warning_conflict_flag_penalty"), 0.0),
        "frame_gap": safe_int(candidate.get("frame_gap", decision_row.get("frame_gap")), 0),
        "frame_gap_penalty": safe_float(edge_feature(candidate, "frame_gap_penalty"), 0.0),
        "uncertainty_reasons": list(edge_feature(candidate, "uncertainty_reasons", candidate.get("uncertainty_reasons", [])) or []),
    }
    visual_stamp(row)
    assert_no_forbidden_keys(row)
    return row


def build_reviewed_decision_training_payloads(
    *,
    m1_reviewed_payload: Any,
    m1_candidate_payload: dict[str, Any],
    m1r_reviewed_payload: Any,
    m1r_candidate_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    m1_by_id, m1_by_edge = candidate_lookup(m1_candidate_payload, review_id_keys=["step2m1_review_candidate_id", "continuity_edge_id"])
    m1r_by_id, m1r_by_edge = candidate_lookup(m1r_candidate_payload, review_id_keys=["step2m1r_review_candidate_id"])
    training_rows: list[dict[str, Any]] = []
    missing_decisions: list[dict[str, Any]] = []
    m1_loaded = 0
    m1r_loaded = 0
    for index, decision in enumerate(decision_rows_from_payload(m1_reviewed_payload)):
        review_id = str(decision.get("step2m1_review_candidate_id", ""))
        edge_id = str(decision.get("continuity_edge_id", ""))
        candidate = m1_by_id.get(review_id) or m1_by_edge.get(edge_id)
        if not candidate:
            missing_decisions.append({"source": "step2m1", "row_index": index, "continuity_edge_id": edge_id, "candidate_id": review_id})
            continue
        training_rows.append(training_row_from_decision(decision, candidate, source_stage="step2m1"))
        m1_loaded += 1
    for index, decision in enumerate(decision_rows_from_payload(m1r_reviewed_payload)):
        review_id = str(decision.get("step2m1r_review_candidate_id", ""))
        edge_id = str(decision.get("continuity_edge_id", ""))
        candidate = m1r_by_id.get(review_id) or m1r_by_edge.get(edge_id)
        if not candidate:
            missing_decisions.append({"source": "step2m1r", "row_index": index, "continuity_edge_id": edge_id, "candidate_id": review_id})
            continue
        training_rows.append(training_row_from_decision(decision, candidate, source_stage="step2m1r"))
        m1r_loaded += 1
    counts = Counter(row["human_review_decision"] for row in training_rows)
    bucket_counts = Counter(row["review_bucket"] for row in training_rows)
    summary = guardrail_stamp(
        {
            "artifact": "step2m2_reviewed_decision_training_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "m1_decisions_loaded": m1_loaded,
            "m1r_decisions_loaded": m1r_loaded,
            "training_decision_count": len(training_rows),
            "accepted_count": counts.get(ACCEPT_DECISION, 0),
            "rejected_count": counts.get(REJECT_DECISION, 0),
            "unsure_count": counts.get(UNSURE_DECISION, 0),
            "bucket_training_counts": dict(sorted(bucket_counts.items())),
            "missing_decision_matches": missing_decisions,
            **m2_guardrail_fields(),
        }
    )
    payload = guardrail_stamp(
        {
            "artifact": "step2m2_reviewed_decision_training_rows",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "rows": training_rows,
            "summary": summary,
            **m2_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(payload)
    assert_no_forbidden_keys(summary)
    return payload, summary


def bucket_decision_rates(training_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets = sorted({str(row.get("review_bucket", "")) for row in training_rows})
    rates: dict[str, dict[str, Any]] = {}
    for bucket in buckets:
        rows = [row for row in training_rows if str(row.get("review_bucket", "")) == bucket]
        total = len(rows)
        accepted = sum(1 for row in rows if row.get("accepted_target_label") is True)
        rejected = sum(1 for row in rows if row.get("rejected_target_label") is True)
        unsure = sum(1 for row in rows if row.get("unsure_target_label") is True)
        rates[bucket] = {
            "total": total,
            "accepted": accepted,
            "rejected": rejected,
            "unsure": unsure,
            "acceptance_rate": round(accepted / max(1, total), 4),
            "rejection_rate": round(rejected / max(1, total), 4),
            "unsure_rate": round(unsure / max(1, total), 4),
        }
    return rates


def build_match_local_adaptation_profile(training_payload: dict[str, Any]) -> dict[str, Any]:
    training_rows = rows_from_payload(training_payload)
    counts = Counter(row.get("human_review_decision", "") for row in training_rows)
    rates = bucket_decision_rates(training_rows)
    safe_rate = rates.get("safe_auto_accept_audit", {}).get("acceptance_rate", 0.0)
    recommended_threshold_adjustments = {
        "base_accept_threshold": M2_ADAPTED_ACCEPT_THRESHOLD,
        "base_reject_threshold": M2_ADAPTED_REJECT_THRESHOLD,
        "safe_auto_accept_candidate_score_delta": 0.03 if safe_rate >= 0.7 else -0.05,
        "team_colour_ambiguity_strong_visual_delta": 0.12,
        "merged_or_ambiguous_score_delta": -0.18,
        "role_state_mismatch_strong_visual_delta": 0.06,
        "role_state_mismatch_swap_risk_delta": -0.08,
        "high_uncertainty_low_margin_delta": -0.03,
    }
    recommended_bucket_policy_adjustments = {
        "safe_auto_accept_candidate": {
            "policy": "allow_match_local_auto_accept_only_for_short_window_visual_candidates_after_safe_auto_accept_audit",
            "observed_safe_auto_accept_audit_acceptance_rate": safe_rate,
        },
        "merged_or_ambiguous": {
            "policy": "never_auto_accept; keep as needs_review or auto_reject_candidate",
        },
        "high_uncertainty_low_margin": {
            "policy": "active_learning_review; do_not_use_as_adaptation_safe_positive_without_human_acceptance",
        },
        "role_state_mismatch": {
            "policy": "split_into_strong_visual_overlap_or_person_swap_risk_using_bbox_iou_footpoint_delta_bbox_area_ratio_context_availability",
        },
        "team_colour_ambiguity": {
            "policy": "reduce_colour_penalty_when_bbox_iou_and_footpoint_continuity_are_strong",
        },
        "long_group_boundary_split": {
            "policy": "keep_boundary_cases in targeted review unless human accepted and span caps remain satisfied",
        },
        "official_goalkeeper_sentinel": {
            "policy": "sentinel_review_only; no official_referee_exclusion; no goalkeeper slot assignment",
        },
    }
    low_sample = [
        bucket
        for bucket, values in rates.items()
        if safe_int(values.get("total"), 0) < 8
    ]
    profile = guardrail_stamp(
        {
            "artifact": "step2m2_match_local_adaptation_profile",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "training_decision_count": len(training_rows),
            "accepted_count": counts.get(ACCEPT_DECISION, 0),
            "rejected_count": counts.get(REJECT_DECISION, 0),
            "unsure_count": counts.get(UNSURE_DECISION, 0),
            "bucket_decision_rates": rates,
            "safe_auto_accept_audit_reliability": {
                "reviewed_rows": rates.get("safe_auto_accept_audit", {}).get("total", 0),
                "acceptance_rate": safe_rate,
                "match_local_safe_auto_accept_reliability_observed": safe_rate >= 0.7,
            },
            "role_state_mismatch_split_profile": {
                "strong_visual_overlap_rule": "bbox_iou>=0.45 and footpoint_delta_px<=35 and 0.5<=bbox_area_ratio<=1.8",
                "person_swap_risk_rule": "bbox_iou<0.2 or footpoint_delta_px>90 or bbox_area_ratio outside 0.35..2.4",
            },
            "recommended_threshold_adjustments": recommended_threshold_adjustments,
            "recommended_bucket_policy_adjustments": recommended_bucket_policy_adjustments,
            "rules_not_learned_due_to_low_sample_size": low_sample,
            "global_thresholds_changed": False,
            **m2_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(profile)
    return profile


def strong_visual_continuity(edge: dict[str, Any]) -> bool:
    return (
        safe_float(edge_feature(edge, "bbox_iou"), 0.0) >= 0.45
        and safe_float(edge_feature(edge, "footpoint_delta_px"), 9999.0) <= 35.0
        and 0.5 <= safe_float(edge_feature(edge, "bbox_area_ratio"), 0.0) <= 1.8
    )


def role_state_split(edge: dict[str, Any]) -> str:
    if strong_visual_continuity(edge):
        return "role_state_mismatch_but_strong_visual_overlap"
    if safe_float(edge_feature(edge, "bbox_iou"), 0.0) < 0.2 or safe_float(edge_feature(edge, "footpoint_delta_px"), 0.0) > 90:
        return "role_state_mismatch_with_person_swap_risk"
    return "role_state_mismatch_needs_review"


def adapted_state_from_score(score: float, bucket: str, split_bucket: str) -> str:
    never_accept = bucket in {"merged_or_ambiguous", "high_uncertainty_low_margin", "bad_detection_proxy_adjacent"} or split_bucket == "role_state_mismatch_with_person_swap_risk"
    if score >= M2_ADAPTED_ACCEPT_THRESHOLD and not never_accept:
        return "auto_accept_candidate"
    if score <= M2_ADAPTED_REJECT_THRESHOLD:
        return "auto_reject_candidate"
    return "needs_review_candidate"


def adapt_edge_row(edge: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    bucket = str(edge.get("review_bucket", ""))
    original_score = safe_float(edge.get("edge_score_sandbox", edge_feature(edge, "edge_score_sandbox")), 0.0)
    adapted_score = original_score
    reasons: list[str] = []
    rates = profile.get("bucket_decision_rates", {}).get(bucket, {})
    split_bucket = ""
    if bucket == "safe_auto_accept_candidate":
        delta = safe_float(profile.get("recommended_threshold_adjustments", {}).get("safe_auto_accept_candidate_score_delta"), 0.0)
        adapted_score += delta
        reasons.append("safe_auto_accept_audit_match_local_reliability_delta")
    if bucket == "team_colour_ambiguity" and strong_visual_continuity(edge):
        adapted_score += safe_float(profile.get("recommended_threshold_adjustments", {}).get("team_colour_ambiguity_strong_visual_delta"), 0.12)
        reasons.append("team_colour_ambiguity_reduced_penalty_for_strong_visual_continuity")
    if bucket == "merged_or_ambiguous":
        adapted_score += safe_float(profile.get("recommended_threshold_adjustments", {}).get("merged_or_ambiguous_score_delta"), -0.18)
        reasons.append("merged_or_ambiguous_never_auto_accept_policy")
    if bucket == "high_uncertainty_low_margin":
        adapted_score += safe_float(profile.get("recommended_threshold_adjustments", {}).get("high_uncertainty_low_margin_delta"), -0.03)
        reasons.append("high_uncertainty_low_margin_active_learning_policy")
    if bucket == "role_state_mismatch":
        split_bucket = role_state_split(edge)
        if split_bucket == "role_state_mismatch_but_strong_visual_overlap":
            adapted_score += safe_float(profile.get("recommended_threshold_adjustments", {}).get("role_state_mismatch_strong_visual_delta"), 0.06)
        elif split_bucket == "role_state_mismatch_with_person_swap_risk":
            adapted_score += safe_float(profile.get("recommended_threshold_adjustments", {}).get("role_state_mismatch_swap_risk_delta"), -0.08)
        reasons.append(split_bucket)
    if safe_float(rates.get("rejection_rate"), 0.0) > 0.55:
        adapted_score -= 0.08
        reasons.append("bucket_rejection_rate_above_match_local_review_threshold")
    elif safe_float(rates.get("acceptance_rate"), 0.0) > 0.75:
        adapted_score += 0.05
        reasons.append("bucket_acceptance_rate_above_match_local_review_threshold")
    adapted_score = round(max(0.0, min(1.0, adapted_score)), 4)
    original_state = str(edge.get("proposed_edge_state", "needs_review_candidate"))
    adapted_state = adapted_state_from_score(adapted_score, bucket, split_bucket)
    row = {
        "continuity_edge_id": edge.get("continuity_edge_id", ""),
        "source_visible_person_base_id": edge.get("source_visible_person_base_id", ""),
        "target_visible_person_base_id": edge.get("target_visible_person_base_id", ""),
        "source_frame_sequence": edge.get("source_frame_sequence", -1),
        "target_frame_sequence": edge.get("target_frame_sequence", -1),
        "frame_gap": edge.get("frame_gap", 0),
        "review_bucket": bucket,
        "step2m2_role_state_split_bucket": split_bucket,
        "edge_feature_summary": edge.get("edge_feature_summary", {}),
        "uncertainty_reasons": edge.get("uncertainty_reasons", edge_feature(edge, "uncertainty_reasons", [])),
        "uncertainty_score": safe_float(edge.get("uncertainty_score", edge_feature(edge, "uncertainty_score")), 0.0),
        "original_edge_score_sandbox": round(original_score, 4),
        "adapted_edge_score_sandbox": adapted_score,
        "original_proposed_edge_state": original_state,
        "adapted_proposed_edge_state": adapted_state,
        "adapted_edge_state_changed": original_state != adapted_state,
        "adaptation_reasons": reasons or ["no_match_local_adjustment_applied"],
        "learned_from_m1_m1r_evidence": bool(rates),
        "match_local_only": True,
        "safe_to_apply_globally": False,
    }
    visual_stamp(row)
    assert_no_forbidden_keys(row)
    return row


def adapted_queue_bucket(row: dict[str, Any]) -> str | None:
    bucket = str(row.get("review_bucket", ""))
    if bucket == "safe_auto_accept_candidate" and row.get("adapted_proposed_edge_state") == "auto_accept_candidate":
        return "safe_auto_accept_audit"
    if row.get("adapted_proposed_edge_state") == "auto_accept_candidate" and safe_float(row.get("uncertainty_score"), 0.0) >= 0.25:
        return "high_risk_adapted_accept"
    if row.get("adapted_edge_state_changed") is True and row.get("adapted_proposed_edge_state") in {"needs_review_candidate", "auto_accept_candidate"}:
        return "changed_state_candidate"
    if bucket == "merged_or_ambiguous":
        return "merged_ambiguous_sentinel"
    if bucket == "role_state_mismatch":
        return "role_state_mismatch_split"
    if bucket in {"official_context_warning", "goalkeeper_context_warning"}:
        return "official_goalkeeper_sentinel"
    return None


def queue_priority(row: dict[str, Any], bucket: str) -> tuple[float, int, str]:
    if bucket == "safe_auto_accept_audit":
        score = -safe_float(row.get("adapted_edge_score_sandbox"), 0.0)
    elif bucket == "high_risk_adapted_accept":
        score = -safe_float(row.get("uncertainty_score"), 0.0)
    elif bucket == "changed_state_candidate":
        score = -abs(safe_float(row.get("adapted_edge_score_sandbox"), 0.0) - safe_float(row.get("original_edge_score_sandbox"), 0.0))
    else:
        score = -safe_float(row.get("uncertainty_score"), 0.0)
    return (score, safe_int(row.get("source_frame_sequence"), 0), str(row.get("continuity_edge_id", "")))


def add_pool_row(pools: dict[str, list[dict[str, Any]]], row: dict[str, Any], *, limit: int = 120) -> None:
    bucket = adapted_queue_bucket(row)
    if not bucket:
        return
    pools.setdefault(bucket, []).append(row)
    pools[bucket] = sorted(pools[bucket], key=lambda item: queue_priority(item, bucket))[:limit]


def iter_jsonl_gz_rows(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def write_adapted_edge_artifacts(profile: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    ensure_dir(STEP2M2_OUTPUT_DIR)
    ensure_dir(STEP2M2_ADAPTED_EDGE_CANDIDATES_JSONL_GZ_PATH.parent)
    row_count = 0
    sample_rows: list[dict[str, Any]] = []
    original_state_counts: Counter[str] = Counter()
    adapted_state_counts: Counter[str] = Counter()
    changed_state_counts: Counter[str] = Counter()
    review_bucket_counts: Counter[str] = Counter()
    pools: dict[str, list[dict[str, Any]]] = {}
    with gzip.open(STEP2M2_ADAPTED_EDGE_CANDIDATES_JSONL_GZ_PATH, "wt", encoding="utf-8", newline="\n") as handle:
        for edge in iter_jsonl_gz_rows(STEP2M1_EDGE_CANDIDATE_ROWS_JSONL_GZ_PATH):
            adapted = adapt_edge_row(edge, profile)
            handle.write(json.dumps(adapted, separators=(",", ":"), sort_keys=True))
            handle.write("\n")
            row_count += 1
            if len(sample_rows) < 80:
                sample_rows.append(adapted)
            original_state_counts[str(adapted.get("original_proposed_edge_state", ""))] += 1
            adapted_state_counts[str(adapted.get("adapted_proposed_edge_state", ""))] += 1
            review_bucket_counts[str(adapted.get("review_bucket", ""))] += 1
            if adapted.get("adapted_edge_state_changed") is True:
                changed_state_counts[f"{adapted.get('original_proposed_edge_state')}->{adapted.get('adapted_proposed_edge_state')}"] += 1
            add_pool_row(pools, adapted)
    summary = guardrail_stamp(
        {
            "artifact": "step2m2_adapted_visual_continuity_edge_candidate_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "adapted_edge_rows": row_count,
            "original_proposed_edge_state_counts": dict(sorted(original_state_counts.items())),
            "adapted_proposed_edge_state_counts": dict(sorted(adapted_state_counts.items())),
            "changed_edge_state_counts": dict(sorted(changed_state_counts.items())),
            "review_bucket_counts": dict(sorted(review_bucket_counts.items())),
            "full_rows_jsonl_gz_path": str(STEP2M2_ADAPTED_EDGE_CANDIDATES_JSONL_GZ_PATH.resolve()),
            "sample_json_path": str(STEP2M2_ADAPTED_EDGE_CANDIDATE_SAMPLE_PATH.resolve()),
            **m2_guardrail_fields(),
        }
    )
    sample = guardrail_stamp(
        {
            "artifact": "step2m2_adapted_visual_continuity_edge_candidate_sample",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "sample_rows": len(sample_rows),
            "total_rows": row_count,
            "rows": sample_rows,
            **m2_guardrail_fields(),
        }
    )
    write_json(STEP2M2_ADAPTED_EDGE_CANDIDATE_SUMMARY_PATH, summary)
    write_json(STEP2M2_ADAPTED_EDGE_CANDIDATE_SAMPLE_PATH, sample)
    assert_no_forbidden_keys(summary)
    assert_no_forbidden_keys(sample)
    return summary, sample, pools


def make_m2_review_candidate(row: dict[str, Any], index: int, queue_bucket: str) -> dict[str, Any]:
    candidate_id = f"step2m2_review_{index:03d}_{m1r_remediation.safe_stem(str(row.get('continuity_edge_id', 'edge')))}"
    candidate = {
        **row,
        "step2m2_review_candidate_id": candidate_id,
        "step2m1r_review_candidate_id": candidate_id,
        "review_card_index": index,
        "step2m2_target_review_bucket": queue_bucket,
        "review_bucket": queue_bucket,
        "source_review_bucket": row.get("review_bucket", ""),
        "review_decision_rule": "Accept only if the highlighted person continues through the burst without a visible person swap. Reject if the edge jumps to a nearby person. Use unsure if the burst is too blurry or occluded.",
        "m2_reuses_m1r_burst_overlay_renderer": True,
        "current_overlay_version": M2_CURRENT_OVERLAY_VERSION,
        "human_confirmed": False,
    }
    visual_stamp(candidate)
    assert_no_forbidden_keys(candidate)
    return candidate


def build_m2_review_queue_from_pools(pools: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    selected: dict[str, dict[str, Any]] = {}
    selected_bucket: dict[str, str] = {}
    for bucket in ["safe_auto_accept_audit", "high_risk_adapted_accept", "changed_state_candidate", "merged_ambiguous_sentinel", "role_state_mismatch_split"]:
        for row in sorted(pools.get(bucket, []), key=lambda item: queue_priority(item, bucket)):
            edge_id = str(row.get("continuity_edge_id", ""))
            if edge_id in selected:
                continue
            selected[edge_id] = row
            selected_bucket[edge_id] = bucket
            if sum(1 for value in selected_bucket.values() if value == bucket) >= M2_BUCKET_TARGETS[bucket]:
                break
    if len(selected) < M2_TARGET_REVIEW_CARDS:
        for bucket in ["official_goalkeeper_sentinel", "changed_state_candidate", "high_risk_adapted_accept"]:
            for row in sorted(pools.get(bucket, []), key=lambda item: queue_priority(item, bucket)):
                edge_id = str(row.get("continuity_edge_id", ""))
                if edge_id in selected:
                    continue
                selected[edge_id] = row
                selected_bucket[edge_id] = bucket
                if len(selected) >= M2_TARGET_REVIEW_CARDS:
                    break
            if len(selected) >= M2_TARGET_REVIEW_CARDS:
                break
    candidate_rows = [
        make_m2_review_candidate(row, index + 1, selected_bucket[str(row.get("continuity_edge_id", ""))])
        for index, row in enumerate(selected.values())
    ][:M2_HARD_MAX_REVIEW_CARDS]
    bucket_counts = Counter(row.get("step2m2_target_review_bucket", "") for row in candidate_rows)
    payload = guardrail_stamp(
        {
            "artifact": "step2m2_targeted_review_candidate_rows",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "review_flow": "Step2.M2 match-local adapted scoring review sandbox; visual-only; maximum 10 minutes; A/X/U decisions.",
            "keyboard_shortcuts": {
                "A": "accept_short_window_visual_continuity_edge",
                "X": "reject_edge",
                "U": "unsure_needs_later_review",
            },
            "target_review_cards": M2_TARGET_REVIEW_CARDS,
            "hard_max_review_cards": M2_HARD_MAX_REVIEW_CARDS,
            "safe_auto_accept_audit_rows": bucket_counts.get("safe_auto_accept_audit", 0),
            "bucket_counts": dict(sorted(bucket_counts.items())),
            "m2_reuses_m1r_burst_overlay_renderer": True,
            "rows": candidate_rows,
            "summary": {
                "targeted_review_candidate_rows": len(candidate_rows),
                "review_queue_hard_max": M2_HARD_MAX_REVIEW_CARDS,
                "safe_auto_accept_audit_rows": bucket_counts.get("safe_auto_accept_audit", 0),
                "bucket_counts": dict(sorted(bucket_counts.items())),
                "m2_reuses_m1r_burst_overlay_renderer": True,
                "current_overlay_version": M2_CURRENT_OVERLAY_VERSION,
                "visual_only_warning": VISUAL_ONLY_WARNING,
            },
            **m2_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(payload)
    return payload


@contextmanager
def redirected_m1r_burst_renderer_to_m2() -> Iterable[None]:
    replacements = {
        (m1r_remediation, "STEP2M1_OUTPUT_DIR"): STEP2M2_OUTPUT_DIR,
        (render_review, "STEP2M1_OUTPUT_DIR"): STEP2M2_OUTPUT_DIR,
        (m1r_remediation, "STEP2M1R_SOURCE_CONTEXT_IMAGES_DIR"): STEP2M2_SOURCE_CONTEXT_IMAGES_DIR,
        (m1r_remediation, "STEP2M1R_TARGET_CONTEXT_IMAGES_DIR"): STEP2M2_TARGET_CONTEXT_IMAGES_DIR,
        (m1r_remediation, "STEP2M1R_SOURCE_CROP_IMAGES_DIR"): STEP2M2_SOURCE_CROP_IMAGES_DIR,
        (m1r_remediation, "STEP2M1R_TARGET_CROP_IMAGES_DIR"): STEP2M2_TARGET_CROP_IMAGES_DIR,
        (m1r_remediation, "STEP2M1R_REVIEW_BURST_CLIPS_DIR"): STEP2M2_REVIEW_BURST_CLIPS_DIR,
        (m1r_remediation, "STEP2M1R_REVIEW_BURST_STRIPS_DIR"): STEP2M2_REVIEW_BURST_STRIPS_DIR,
        (m1r_remediation, "STEP2M1R_REVIEW_BURST_RAW_STRIPS_DIR"): STEP2M2_REVIEW_BURST_RAW_STRIPS_DIR,
        (m1r_remediation, "STEP2M1R_REVIEW_BURST_COMPARISON_STRIPS_DIR"): STEP2M2_REVIEW_BURST_COMPARISON_STRIPS_DIR,
        (m1r_remediation, "STEP2M1R_BURST_OVERLAY_DEBUG_ROWS_PATH"): STEP2M2_BURST_OVERLAY_DEBUG_ROWS_PATH,
        (m1r_remediation, "STEP2M1R_BURST_OVERLAY_ALIGNMENT_SUMMARY_PATH"): STEP2M2_BURST_OVERLAY_ALIGNMENT_SUMMARY_PATH,
        (m1r_remediation, "STEP2M1R_BURST_OVERLAY_QA_DIR"): STEP2M2_BURST_OVERLAY_QA_DIR,
        (m1r_remediation, "STEP2M1R_REVIEW_CONTACT_SHEET_PATH"): STEP2M2_REVIEW_CONTACT_SHEET_PATH,
    }
    old_values = {key: getattr(key[0], key[1]) for key in replacements}
    try:
        for (module, name), value in replacements.items():
            setattr(module, name, value)
        yield
    finally:
        for (module, name), value in old_values.items():
            setattr(module, name, value)


def render_m2_burst_evidence(review_payload: dict[str, Any], node_payload: dict[str, Any]) -> dict[str, Any]:
    with redirected_m1r_burst_renderer_to_m2():
        rendered = m1r_remediation.render_m1r_burst_evidence(review_payload, node_payload)
        m1r_remediation.render_m1r_contact_sheet(rows_from_payload(rendered))
    rendered["artifact"] = "step2m2_targeted_review_candidate_rows"
    rendered["summary"] = {
        **rendered.get("summary", {}),
        "m2_reuses_m1r_burst_overlay_renderer": True,
        "current_overlay_version": M2_CURRENT_OVERLAY_VERSION,
    }
    if STEP2M2_BURST_OVERLAY_DEBUG_ROWS_PATH.exists():
        debug_payload = read_json(STEP2M2_BURST_OVERLAY_DEBUG_ROWS_PATH)
        debug_payload["artifact"] = "step2m2_burst_overlay_debug_rows"
        debug_payload["m2_reuses_m1r_burst_overlay_renderer"] = True
        write_json(STEP2M2_BURST_OVERLAY_DEBUG_ROWS_PATH, debug_payload)
    if STEP2M2_BURST_OVERLAY_ALIGNMENT_SUMMARY_PATH.exists():
        overlay_summary = read_json(STEP2M2_BURST_OVERLAY_ALIGNMENT_SUMMARY_PATH)
        overlay_summary["artifact"] = "step2m2_burst_overlay_alignment_summary"
        overlay_summary["m2_reuses_m1r_burst_overlay_renderer"] = True
        write_json(STEP2M2_BURST_OVERLAY_ALIGNMENT_SUMMARY_PATH, overlay_summary)
    assert_no_forbidden_keys(rendered)
    return rendered


def m2_review_ui_html(review_payload: dict[str, Any]) -> str:
    state_json = json.dumps(
        {
            "artifact": "step2m2_review_ui_state",
            "visual_only_warning": VISUAL_ONLY_WARNING,
            "rows": rows_from_payload(review_payload),
        }
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Step2.M2 Match-Local Visual Continuity Review</title>
<style>
body{{margin:0;background:#111417;color:#ecf2f7;font-family:Arial,sans-serif}}
.top{{position:sticky;top:0;background:#191d22;padding:12px 18px;border-bottom:1px solid #303841;z-index:2}}
.wrap{{padding:18px;display:grid;grid-template-columns:1fr 360px;gap:16px}}
.card{{border:1px solid #303841;border-radius:8px;padding:12px;background:#171b20}}
.media{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}}
img{{max-width:100%;border:1px solid #303841;background:#0b0d0f}}
.burst{{grid-column:1 / span 2}}
.decision{{font-size:18px;line-height:1.35;color:#f4d38f}}
.badge{{display:inline-block;padding:3px 7px;border-radius:999px;background:#29313a;margin-right:6px;margin-bottom:6px}}
.muted{{color:#9fb0bd}}
button{{background:#27313b;color:#f4f8fb;border:1px solid #42505d;border-radius:6px;padding:8px 10px;margin-right:8px}}
</style>
</head>
<body>
<div class="top"><strong>Step2.M2 match-local adaptation review</strong> <span id="count"></span></div>
<div class="wrap">
  <div class="card">
    <div class="media">
      <img id="sourceStill" alt="source still">
      <img id="targetStill" alt="target still">
      <img id="sourceCrop" alt="source crop">
      <img id="targetCrop" alt="target crop">
      <img id="burstStrip" class="burst" alt="burst strip">
      <img id="burstClip" class="burst" alt="mini burst">
    </div>
  </div>
  <div class="card">
    <div id="meta"></div>
    <p class="decision">Accept only if the highlighted person continues through the burst without a visible person swap. Reject if the edge jumps to a nearby person. Use unsure if the burst is too blurry or occluded.</p>
    <button onclick="decide('accept_short_window_visual_continuity_edge')">A Accept</button>
    <button onclick="decide('reject_edge')">X Reject</button>
    <button onclick="decide('unsure_needs_later_review')">U Unsure</button>
    <button onclick="exportDecisions()">Export M2 decisions JSON</button>
    <p id="status" class="muted"></p>
  </div>
</div>
<script>
const state={state_json};
let index=0;
const decisions=JSON.parse(localStorage.getItem('step2m2_review_decisions')||'{{}}');
function asset(row,key){{return (row.ui_assets&&row.ui_assets[key])||'';}}
function setStatus(text, error=false){{
  const node=document.getElementById('status');
  node.textContent=text;
  node.style.color=error?'#ff8d8d':'#9fb0bd';
}}
function show(){{
  const rows=state.rows||[]; if(!rows.length) return;
  const row=rows[index];
  document.getElementById('count').textContent=`${{index+1}} / ${{rows.length}}`;
  document.getElementById('sourceStill').src=asset(row,'source_context_image');
  document.getElementById('targetStill').src=asset(row,'target_context_image');
  document.getElementById('sourceCrop').src=asset(row,'source_crop_image');
  document.getElementById('targetCrop').src=asset(row,'target_crop_image');
  document.getElementById('burstStrip').src=asset(row,'burst_strip');
  document.getElementById('burstClip').src=asset(row,'burst_clip');
  const saved=decisions[row.step2m2_review_candidate_id];
  setStatus(saved?`local backup: ${{saved.human_review_decision}}`:'not saved');
  document.getElementById('meta').innerHTML=[
    `<span class="badge">${{row.step2m2_target_review_bucket}}</span>`,
    `<span class="badge">${{row.source_frame_sequence}} -> ${{row.target_frame_sequence}}</span>`,
    `<span class="badge">original ${{row.original_proposed_edge_state}}</span>`,
    `<span class="badge">adapted ${{row.adapted_proposed_edge_state}}</span>`,
    `<p class="muted">${{row.continuity_edge_id}}</p>`,
    `<p class="muted">${{state.visual_only_warning}}</p>`
  ].join('');
}}
async function decide(value){{
  const row=state.rows[index];
  const payload={{step2m2_review_candidate_id:row.step2m2_review_candidate_id,continuity_edge_id:row.continuity_edge_id,human_review_decision:value,reviewed_at:new Date().toISOString(),visual_only_warning:state.visual_only_warning,production_ready:false,no_auto_promotion:true,human_approved:false}};
  decisions[row.step2m2_review_candidate_id]=payload;
  localStorage.setItem('step2m2_review_decisions',JSON.stringify(decisions));
  setStatus('saving to disk...');
  try{{
    const response=await fetch('/api/step2m2/review-decision',{{method:'POST',headers:{{'content-type':'application/json'}},body:JSON.stringify(payload)}});
    const body=await response.json();
    if(!response.ok||!body.success){{throw new Error(body.error||`HTTP ${{response.status}}`);}}
    setStatus(`saved to disk (${{body.reviewed_count}} / ${{body.total_review_candidates}})`);
    if(index < state.rows.length-1){{index++; show();}}
  }}catch(error){{
    setStatus(`disk autosave failed: ${{error.message}}`, true);
  }}
}}
function exportDecisions(){{
  const blob=new Blob([JSON.stringify({{artifact:'step2m2_review_decision_export_backup',created_at:new Date().toISOString(),rows:Object.values(decisions)}},null,2)],{{type:'application/json'}});
  const url=URL.createObjectURL(blob);
  const a=document.createElement('a');
  a.href=url; a.download='step2m2_review_decisions_export.json'; a.click();
  URL.revokeObjectURL(url);
}}
document.addEventListener('keydown',ev=>{{if(ev.key==='a'||ev.key==='A')decide('accept_short_window_visual_continuity_edge'); if(ev.key==='x'||ev.key==='X')decide('reject_edge'); if(ev.key==='u'||ev.key==='U')decide('unsure_needs_later_review'); if(ev.key==='ArrowRight'){{index=Math.min((state.rows||[]).length-1,index+1);show();}} if(ev.key==='ArrowLeft'){{index=Math.max(0,index-1);show();}}}});
show();
</script>
</body>
</html>"""


def m2_review_progress_payload(review_payload: dict[str, Any], reviewed_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = rows_from_payload(review_payload)
    bucket_counts = Counter(str(row.get("step2m2_target_review_bucket", "")) for row in rows)
    reviewed_payload = reviewed_payload or {"rows": []}
    usable_rows, validation_errors = validate_m2_reviewed_rows(review_payload, reviewed_payload)
    decision_counts = Counter(str(row.get("human_review_decision", "")) for row in usable_rows)
    candidates = m2_candidate_by_review_id(review_payload)
    reviewed_by_bucket = Counter(
        str(candidates.get(str(row.get("step2m2_review_candidate_id", "")), {}).get("step2m2_target_review_bucket", ""))
        for row in usable_rows
    )
    bucket_progress = {
        bucket: {"total": bucket_counts.get(bucket, 0), "reviewed": reviewed_by_bucket.get(bucket, 0)}
        for bucket in sorted(bucket_counts)
    }
    collected_versions = {
        str(row.get("review_decisions_collected_with_overlay_version", ""))
        for row in usable_rows
        if row.get("review_decisions_collected_with_overlay_version")
    }
    version_matches = bool(usable_rows) and collected_versions == {M2_CURRENT_OVERLAY_VERSION}
    targeted_review_completed = len(usable_rows) == len(rows) and not validation_errors
    forbidden = sorted(set(forbidden_keys_present(review_payload)) | set(forbidden_keys_present(reviewed_payload)))
    return guardrail_stamp(
        {
            "artifact": "step2m2_review_progress_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "total_review_candidates": len(rows),
            "reviewed_candidates": len(usable_rows),
            "accepted_count": decision_counts.get(ACCEPT_DECISION, 0),
            "rejected_count": decision_counts.get(REJECT_DECISION, 0),
            "unsure_count": decision_counts.get(UNSURE_DECISION, 0),
            "targeted_review_completed": targeted_review_completed,
            "bucket_counts": dict(sorted(bucket_counts.items())),
            "bucket_progress": bucket_progress,
            "safe_auto_accept_audit_rows": bucket_counts.get("safe_auto_accept_audit", 0),
            "safe_auto_accept_audit_reviewed": reviewed_by_bucket.get("safe_auto_accept_audit", 0),
            "review_queue_hard_max": M2_HARD_MAX_REVIEW_CARDS,
            "current_overlay_version": M2_CURRENT_OVERLAY_VERSION,
            "review_decisions_collected_with_overlay_version": sorted(collected_versions),
            "review_decisions_overlay_version_matches_current": version_matches,
            "burst_overlay_alignment_safe_for_review": review_payload.get("summary", {}).get("burst_overlay_alignment_safe_for_review", False),
            "validation_errors": validation_errors,
            "forbidden_keys_present": forbidden,
            **m2_guardrail_fields(),
        }
    )


def m2_review_decision_summary_payload(review_payload: dict[str, Any], reviewed_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    progress = m2_review_progress_payload(review_payload, reviewed_payload)
    usable_rows, _validation_errors = validate_m2_reviewed_rows(review_payload, reviewed_payload or {"rows": []})
    return guardrail_stamp(
        {
            "artifact": "step2m2_review_decision_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "total_review_candidates": progress["total_review_candidates"],
            "reviewed_candidates": progress["reviewed_candidates"],
            "accepted_count": progress["accepted_count"],
            "rejected_count": progress["rejected_count"],
            "unsure_count": progress["unsure_count"],
            "human_review_decision_counts": dict(sorted(Counter(str(row.get("human_review_decision", "")) for row in usable_rows).items())),
            "bucket_progress": progress["bucket_progress"],
            "safe_auto_accept_audit_rows": progress["safe_auto_accept_audit_rows"],
            "safe_auto_accept_audit_reviewed": progress["safe_auto_accept_audit_reviewed"],
            "targeted_review_completed": progress["targeted_review_completed"],
            "current_overlay_version": M2_CURRENT_OVERLAY_VERSION,
            "review_decisions_collected_with_overlay_version": progress["review_decisions_collected_with_overlay_version"],
            "review_decisions_overlay_version_matches_current": progress["review_decisions_overlay_version_matches_current"],
            "forbidden_keys_present": progress["forbidden_keys_present"],
            **m2_guardrail_fields(),
        }
    )


def refresh_m2_review_summaries(
    review_payload: dict[str, Any],
    reviewed_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    progress = m2_review_progress_payload(review_payload, reviewed_payload)
    decision_summary = m2_review_decision_summary_payload(review_payload, reviewed_payload)
    write_json_atomic(STEP2M2_REVIEW_PROGRESS_SUMMARY_PATH, progress)
    write_json_atomic(STEP2M2_REVIEW_DECISION_SUMMARY_PATH, decision_summary)
    return progress, decision_summary


def save_m2_review_decision(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    review_payload = read_json(STEP2M2_TARGETED_REVIEW_CANDIDATE_ROWS_PATH)
    candidates = m2_candidate_by_review_id(review_payload)
    candidate_id = str(payload.get("step2m2_review_candidate_id", ""))
    candidate = candidates.get(candidate_id)
    if not candidate:
        raise ValueError("unknown_step2m2_review_candidate")
    decision = m2_review_decision_row(candidate, payload)
    reviewed_payload = read_m2_reviewed_decisions()
    by_id = {
        str(row.get("step2m2_review_candidate_id", "")): row
        for row in decision_rows_from_payload(reviewed_payload)
        if row.get("step2m2_review_candidate_id")
    }
    by_id[str(decision["step2m2_review_candidate_id"])] = decision
    updated_payload = write_m2_reviewed_decisions_payload(list(by_id.values()))
    progress, _decision_summary = refresh_m2_review_summaries(review_payload, updated_payload)
    return decision, updated_payload, progress


def build_step2m2_validation_outputs(
    *,
    training_summary: dict[str, Any],
    profile: dict[str, Any],
    adapted_summary: dict[str, Any],
    review_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    m1r_manifest = read_json(STEP2M1R_ADAPTATION_SAFETY_MANIFEST_PATH)
    review_queue_size = len(rows_from_payload(review_payload))
    previous_progress = read_json(STEP2M2_REVIEW_PROGRESS_SUMMARY_PATH) if STEP2M2_REVIEW_PROGRESS_SUMMARY_PATH.exists() else {}
    decision_file_exists = STEP2M2_REVIEWED_DECISIONS_PATH.exists()
    reviewed_payload = read_m2_reviewed_decisions()
    progress, decision_summary = refresh_m2_review_summaries(review_payload, reviewed_payload)
    decision_row_count = len(decision_rows_from_payload(reviewed_payload)) if decision_file_exists else 0
    forbidden = sorted(
        set(forbidden_keys_present(training_summary))
        | set(forbidden_keys_present(profile))
        | set(forbidden_keys_present(adapted_summary))
        | set(forbidden_keys_present(review_payload))
        | set(forbidden_keys_present(reviewed_payload))
    )
    issues: list[dict[str, Any]] = []
    if not decision_file_exists and safe_int(previous_progress.get("reviewed_candidates"), 0) > 0:
        issues.append(
            {
                "severity": "blocking",
                "issue_code": "step2m2_review_progress_claims_reviewed_but_decision_file_missing",
                "message": "Step2.M2 review progress reported reviewed cards, but the persisted decision file is missing.",
            }
        )
    if safe_int(progress.get("reviewed_candidates"), 0) != decision_row_count:
        issues.append(
            {
                "severity": "blocking",
                "issue_code": "step2m2_reviewed_count_does_not_match_decision_row_count",
                "message": "Step2.M2 reviewed_candidates does not match persisted decision row count.",
                "reviewed_candidates": progress.get("reviewed_candidates", 0),
                "decision_row_count": decision_row_count,
            }
        )
    if progress.get("validation_errors"):
        issues.append(
            {
                "severity": "blocking",
                "issue_code": "step2m2_persisted_review_decision_validation_errors",
                "message": "Step2.M2 persisted review decisions failed schema or candidate validation.",
                "validation_errors": progress.get("validation_errors", []),
            }
        )
    if decision_row_count and progress.get("review_decisions_overlay_version_matches_current") is not True:
        issues.append(
            {
                "severity": "blocking",
                "issue_code": "step2m2_review_decisions_overlay_version_mismatch",
                "message": "Step2.M2 persisted review decisions were not collected with the current overlay version.",
            }
        )
    if forbidden:
        issues.append(
            {
                "severity": "blocking",
                "issue_code": "step2m2_forbidden_keys_present",
                "message": "Step2.M2 artifacts contain forbidden keys.",
                "forbidden_keys_present": forbidden,
            }
        )
    gate_checks = {
        "m1r_safe_for_step2m2_adaptation_candidate": m1r_manifest.get("safe_for_step2m2_adaptation_candidate") is True,
        "m1r_reviewed_candidates_is_55": safe_int(m1r_manifest.get("reviewed_candidates"), 0) == 55,
        "m1r_review_decisions_overlay_version_matches_current": m1r_manifest.get("review_decisions_overlay_version_matches_current") is True,
        "m1r_burst_overlay_alignment_safe_for_review": m1r_manifest.get("burst_overlay_alignment_safe_for_review") is True,
        "m1r_groups_over_cap_after_zero": safe_int(m1r_manifest.get("groups_over_cap_after"), -1) == 0,
        "m1r_forbidden_keys_absent": m1r_manifest.get("forbidden_keys_present") == [],
        "step2m2_forbidden_keys_absent": forbidden == [],
        "step2m2_review_queue_within_hard_max": review_queue_size <= M2_HARD_MAX_REVIEW_CARDS,
        "step2m2_persisted_review_state_valid": not issues,
        "production_ready_false": PRODUCTION_READY is False,
        "no_auto_promotion_true": NO_AUTO_PROMOTION is True,
    }
    for key, passed in gate_checks.items():
        if not passed:
            issues.append({"severity": "blocking", "issue_code": key, "message": f"Step2.M2 gate check failed: {key}"})
    freeze_candidate = all(gate_checks.values())
    validation = guardrail_stamp(
        {
            "artifact": "step2m2_validation_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "training_decision_count": training_summary.get("training_decision_count", 0),
            "adapted_edge_rows": adapted_summary.get("adapted_edge_rows", 0),
            "review_queue_size": review_queue_size,
            "review_queue_hard_max": M2_HARD_MAX_REVIEW_CARDS,
            "safe_auto_accept_audit_rows": review_payload.get("summary", {}).get("safe_auto_accept_audit_rows", 0),
            "reviewed_candidates": progress.get("reviewed_candidates", 0),
            "persisted_decision_row_count": decision_row_count,
            "targeted_review_completed": progress.get("targeted_review_completed", False),
            "review_decisions_overlay_version_matches_current": progress.get("review_decisions_overlay_version_matches_current", False),
            "step2m2_freeze_candidate_created": freeze_candidate,
            "forbidden_keys_present": forbidden,
            "gate_checks": gate_checks,
            **m2_guardrail_fields(),
        }
    )
    audit = guardrail_stamp(
        {
            "artifact": "step2m2_safety_guardrail_audit",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "forbidden_keys_present": forbidden,
            "m1_read_root": str(STEP2M1_OUTPUT_DIR.resolve()),
            "m2_write_root": str(STEP2M2_OUTPUT_DIR.resolve()),
            "reviewed_candidates": progress.get("reviewed_candidates", 0),
            "persisted_decision_row_count": decision_row_count,
            "review_decisions_overlay_version_matches_current": progress.get("review_decisions_overlay_version_matches_current", False),
            **m2_guardrail_fields(),
        }
    )
    issue_register = guardrail_stamp(
        {
            "artifact": "step2m2_issue_register",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "blocking_issue_count": sum(1 for issue in issues if issue.get("severity") == "blocking"),
            "rows": issues,
            **m2_guardrail_fields(),
        }
    )
    freeze_manifest = guardrail_stamp(
        {
            "artifact": "step2m2_freeze_candidate_manifest",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "step2m2_freeze_candidate_created": freeze_candidate,
            "human_approved": False,
            "review_required_before_any_future_promotion": True,
            "safe_to_apply_globally": False,
            "match_local_only": True,
            "validation_summary_path": str(STEP2M2_VALIDATION_SUMMARY_PATH.resolve()),
            "adaptation_profile_path": str(STEP2M2_MATCH_LOCAL_ADAPTATION_PROFILE_PATH.resolve()),
            "review_queue_size": review_queue_size,
            "reviewed_candidates": progress.get("reviewed_candidates", 0),
            "persisted_decision_row_count": decision_row_count,
            "targeted_review_completed": progress.get("targeted_review_completed", False),
            "forbidden_keys_present": forbidden,
            **m2_guardrail_fields(),
        }
    )
    for payload in [validation, audit, issue_register, freeze_manifest]:
        assert_no_forbidden_keys(payload)
    return validation, audit, issue_register, freeze_manifest


def write_step2m2_review_pack() -> dict[str, Any]:
    ensure_dir(STEP2M2_REVIEW_PACK_DIR)
    files = [
        STEP2M2_REVIEW_UI_HTML_PATH,
        STEP2M2_REVIEW_CONTACT_SHEET_PATH,
        STEP2M2_TARGETED_REVIEW_CANDIDATE_ROWS_PATH,
        STEP2M2_REVIEWED_DECISIONS_PATH,
        STEP2M2_REVIEW_PROGRESS_SUMMARY_PATH,
        STEP2M2_REVIEW_DECISION_SUMMARY_PATH,
        STEP2M2_MATCH_LOCAL_ADAPTATION_PROFILE_PATH,
        STEP2M2_VALIDATION_SUMMARY_PATH,
        STEP2M2_SAFETY_GUARDRAIL_AUDIT_PATH,
        STEP2M2_ISSUE_REGISTER_PATH,
        STEP2M2_FREEZE_CANDIDATE_MANIFEST_PATH,
    ]
    copied: list[str] = []
    for path in files:
        if not path.exists():
            continue
        destination = STEP2M2_REVIEW_PACK_DIR / path.name
        shutil.copyfile(path, destination)
        copied.append(str(destination.resolve()))
    manifest = guardrail_stamp(
        {
            "artifact": "step2m2_review_pack_manifest",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "review_pack_dir": str(STEP2M2_REVIEW_PACK_DIR.resolve()),
            "copied_files": copied,
            "burst_assets_root": str(STEP2M2_OUTPUT_DIR.resolve()),
            "m2_reuses_m1r_burst_overlay_renderer": True,
            **m2_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(manifest)
    write_json(STEP2M2_REVIEW_PACK_MANIFEST_PATH, manifest)
    return manifest


def build_step2m2_match_local_adaptation_profile() -> dict[str, Any]:
    assert_m2_output_path_isolation()
    ensure_dir(STEP2M2_OUTPUT_DIR)
    m1_reviewed_payload = read_json(STEP2M1_REVIEWED_DECISIONS_PATH)
    m1_candidate_payload = read_json(STEP2M1_REVIEW_CANDIDATE_ROWS_PATH)
    m1r_reviewed_payload = m1r_remediation.read_m1r_reviewed_decisions()
    m1r_candidate_payload = read_json(STEP2M1R_TARGETED_REVIEW_CANDIDATE_ROWS_PATH)
    training_payload, training_summary = build_reviewed_decision_training_payloads(
        m1_reviewed_payload=m1_reviewed_payload,
        m1_candidate_payload=m1_candidate_payload,
        m1r_reviewed_payload=m1r_reviewed_payload,
        m1r_candidate_payload=m1r_candidate_payload,
    )
    write_json(STEP2M2_REVIEWED_DECISION_TRAINING_ROWS_PATH, training_payload)
    write_json(STEP2M2_REVIEWED_DECISION_TRAINING_SUMMARY_PATH, training_summary)
    profile = build_match_local_adaptation_profile(training_payload)
    write_json(STEP2M2_MATCH_LOCAL_ADAPTATION_PROFILE_PATH, profile)
    adapted_summary, adapted_sample, pools = write_adapted_edge_artifacts(profile)
    review_payload = build_m2_review_queue_from_pools(pools)
    node_payload = read_json(STEP2M1_NODE_ROWS_PATH)
    review_payload = render_m2_burst_evidence(review_payload, node_payload)
    write_json(STEP2M2_TARGETED_REVIEW_CANDIDATE_ROWS_PATH, review_payload)
    write_text(STEP2M2_REVIEW_UI_HTML_PATH, m2_review_ui_html(review_payload))
    reviewed_payload = read_m2_reviewed_decisions()
    review_progress = m2_review_progress_payload(review_payload, reviewed_payload)
    review_decision = m2_review_decision_summary_payload(review_payload, reviewed_payload)
    write_json(STEP2M2_REVIEW_PROGRESS_SUMMARY_PATH, review_progress)
    write_json(STEP2M2_REVIEW_DECISION_SUMMARY_PATH, review_decision)
    validation, audit, issue_register, freeze_manifest = build_step2m2_validation_outputs(
        training_summary=training_summary,
        profile=profile,
        adapted_summary=adapted_summary,
        review_payload=review_payload,
    )
    write_json(STEP2M2_VALIDATION_SUMMARY_PATH, validation)
    write_json(STEP2M2_SAFETY_GUARDRAIL_AUDIT_PATH, audit)
    write_json(STEP2M2_ISSUE_REGISTER_PATH, issue_register)
    write_json(STEP2M2_FREEZE_CANDIDATE_MANIFEST_PATH, freeze_manifest)
    return {
        "training_payload": training_payload,
        "training_summary": training_summary,
        "adaptation_profile": profile,
        "adapted_edge_summary": adapted_summary,
        "adapted_edge_sample": adapted_sample,
        "review_payload": review_payload,
        "review_progress": review_progress,
        "review_decision": review_decision,
        "validation_summary": validation,
        "safety_guardrail_audit": audit,
        "issue_register": issue_register,
        "freeze_candidate_manifest": freeze_manifest,
    }


def validate_step2m2_match_local_adaptation() -> dict[str, Any]:
    assert_m2_output_path_isolation()
    training_summary = read_json(STEP2M2_REVIEWED_DECISION_TRAINING_SUMMARY_PATH)
    profile = read_json(STEP2M2_MATCH_LOCAL_ADAPTATION_PROFILE_PATH)
    adapted_summary = read_json(STEP2M2_ADAPTED_EDGE_CANDIDATE_SUMMARY_PATH)
    review_payload = read_json(STEP2M2_TARGETED_REVIEW_CANDIDATE_ROWS_PATH)
    validation, audit, issue_register, freeze_manifest = build_step2m2_validation_outputs(
        training_summary=training_summary,
        profile=profile,
        adapted_summary=adapted_summary,
        review_payload=review_payload,
    )
    write_json(STEP2M2_VALIDATION_SUMMARY_PATH, validation)
    write_json(STEP2M2_SAFETY_GUARDRAIL_AUDIT_PATH, audit)
    write_json(STEP2M2_ISSUE_REGISTER_PATH, issue_register)
    write_json(STEP2M2_FREEZE_CANDIDATE_MANIFEST_PATH, freeze_manifest)
    return {
        "validation_summary": validation,
        "safety_guardrail_audit": audit,
        "issue_register": issue_register,
        "freeze_candidate_manifest": freeze_manifest,
    }


def prepare_step2m2_review_ui(host: str = "127.0.0.1", port: int = 8785) -> dict[str, Any]:
    assert_m2_output_path_isolation()
    review_payload = read_json(STEP2M2_TARGETED_REVIEW_CANDIDATE_ROWS_PATH)
    write_text(STEP2M2_REVIEW_UI_HTML_PATH, m2_review_ui_html(review_payload))
    reviewed_payload = read_m2_reviewed_decisions()
    progress, decision_summary = refresh_m2_review_summaries(review_payload, reviewed_payload)
    manifest = guardrail_stamp(
        {
            "artifact": "step2m2_review_ui_manifest",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "url": f"http://{host}:{port}/",
            "review_ui_html_path": str(STEP2M2_REVIEW_UI_HTML_PATH.resolve()),
            "review_candidate_rows_path": str(STEP2M2_TARGETED_REVIEW_CANDIDATE_ROWS_PATH.resolve()),
            "reviewed_decisions_path": str(STEP2M2_REVIEWED_DECISIONS_PATH.resolve()),
            "review_progress_summary_path": str(STEP2M2_REVIEW_PROGRESS_SUMMARY_PATH.resolve()),
            "review_decision_summary_path": str(STEP2M2_REVIEW_DECISION_SUMMARY_PATH.resolve()),
            "total_review_candidates": progress.get("total_review_candidates", len(rows_from_payload(review_payload))),
            "reviewed_candidates": progress.get("reviewed_candidates", 0),
            "targeted_review_completed": decision_summary.get("targeted_review_completed", False),
            "autosave_endpoint": "/api/step2m2/review-decision",
            **m2_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(manifest)
    return manifest


def print_step2m2_review_ui_console(manifest: dict[str, Any]) -> None:
    print(f"step2m2_review_ui_html_path: {manifest['review_ui_html_path']}")
    print(f"step2m2_targeted_review_candidate_rows_path: {manifest['review_candidate_rows_path']}")
    print(f"step2m2_reviewed_visual_continuity_decisions_path: {manifest['reviewed_decisions_path']}")
    print(f"step2m2_review_progress_summary_path: {manifest['review_progress_summary_path']}")
    print(f"total_review_candidates: {manifest.get('total_review_candidates', 0)}")
    print(f"reviewed_candidates: {manifest.get('reviewed_candidates', 0)}")
    print(f"autosave_endpoint: {manifest.get('autosave_endpoint')}")
    print("visual_only_warning=VISUAL_ONLY_NOT_METRIC")
    print("production_ready=false")
    print("no_auto_promotion=true")


class Step2M2ReviewHandler(BaseHTTPRequestHandler):
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
        file_path = STEP2M2_REVIEW_UI_HTML_PATH if not raw_path else (STEP2M2_OUTPUT_DIR / raw_path).resolve()
        root = STEP2M2_OUTPUT_DIR.resolve()
        if not str(file_path).startswith(str(root)):
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
        if urlparse(self.path).path != "/api/step2m2/review-decision":
            self._send_json(404, {"success": False, "error": "unknown_endpoint"})
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            _decision, reviewed_payload, progress = save_m2_review_decision(payload)
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


def serve_step2m2_review_ui(host: str = "127.0.0.1", port: int = 8785) -> None:
    prepare_step2m2_review_ui(host=host, port=port)
    server = ThreadingHTTPServer((host, port), Step2M2ReviewHandler)
    print(f"Serving Step2.M2 review UI at http://{host}:{port}/")
    print(f"Autosave endpoint: http://{host}:{port}/api/step2m2/review-decision")
    server.serve_forever()


def print_step2m2_console(outputs: dict[str, Any]) -> None:
    training = outputs["training_summary"]
    profile = outputs["adaptation_profile"]
    adapted = outputs["adapted_edge_summary"]
    review = outputs["review_payload"]
    freeze = outputs["freeze_candidate_manifest"]
    validation = outputs["validation_summary"]
    print(f"step2m2_output_dir: {STEP2M2_OUTPUT_DIR.resolve()}")
    print(f"m1_decisions_loaded: {training.get('m1_decisions_loaded', 0)}")
    print(f"m1r_decisions_loaded: {training.get('m1r_decisions_loaded', 0)}")
    print(f"total_training_decisions: {training.get('training_decision_count', 0)}")
    print(f"accepted_count: {training.get('accepted_count', 0)}")
    print(f"rejected_count: {training.get('rejected_count', 0)}")
    print(f"unsure_count: {training.get('unsure_count', 0)}")
    print(f"bucket_decision_rates: {json.dumps(profile.get('bucket_decision_rates', {}), sort_keys=True)}")
    print(f"recommended_threshold_adjustments: {json.dumps(profile.get('recommended_threshold_adjustments', {}), sort_keys=True)}")
    print(f"adapted_edge_rows: {adapted.get('adapted_edge_rows', 0)}")
    print(f"changed_edge_state_counts: {json.dumps(adapted.get('changed_edge_state_counts', {}), sort_keys=True)}")
    print(f"review_queue_size: {len(rows_from_payload(review))}")
    print(f"safe_auto_accept_audit_count: {review.get('summary', {}).get('safe_auto_accept_audit_rows', 0)}")
    print(f"forbidden_keys_present: {validation.get('forbidden_keys_present', [])}")
    print(f"production_ready={str(validation.get('production_ready', False)).lower()}")
    print(f"no_auto_promotion={str(validation.get('no_auto_promotion', True)).lower()}")
    print(f"step2m2_freeze_candidate_created={str(freeze.get('step2m2_freeze_candidate_created', False)).lower()}")


def print_step2m2_validation_console(outputs: dict[str, Any]) -> None:
    validation = outputs["validation_summary"]
    freeze = outputs["freeze_candidate_manifest"]
    print(f"step2m2_validation_summary_path: {STEP2M2_VALIDATION_SUMMARY_PATH.resolve()}")
    print(f"step2m2_freeze_candidate_manifest_path: {STEP2M2_FREEZE_CANDIDATE_MANIFEST_PATH.resolve()}")
    print(f"blocking_issue_count: {outputs['issue_register'].get('blocking_issue_count', 0)}")
    print(f"forbidden_keys_present: {validation.get('forbidden_keys_present', [])}")
    print(f"step2m2_freeze_candidate_created={str(freeze.get('step2m2_freeze_candidate_created', False)).lower()}")


def print_step2m2_review_pack_console(manifest: dict[str, Any]) -> None:
    print(f"step2m2_review_pack_manifest_path: {STEP2M2_REVIEW_PACK_MANIFEST_PATH.resolve()}")
    print(f"step2m2_review_pack_dir: {manifest.get('review_pack_dir')}")
    print(f"copied_files: {len(manifest.get('copied_files', []))}")
    print("production_ready=false")
    print("no_auto_promotion=true")
