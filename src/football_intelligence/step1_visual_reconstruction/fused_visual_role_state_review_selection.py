# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from football_intelligence.paths import CLIP_ID, MATCH_ID, STAGE3C13_GOLD20_MANUAL_LABELS_PATH
from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_eval import (
    GOALKEEPER_ROLE_STATES,
    OFFICIAL_CONTEXT_ROLE_STATES,
    OUTFIELD_ROLE_STATES,
)
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import gold_visible_person_rows, strict_one_to_one_match
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1F1_FUSED_VISUAL_ROLE_STATE_EVAL_SUMMARY_PATH,
    STEP1F1_FUSED_VISUAL_ROLE_STATE_ROWS_PATH,
    STEP1F1_ROLE_STATE_CONFLICT_AUDIT_ROWS_PATH,
    STEP1F2_REVIEW_CANDIDATE_ROWS_PATH,
    STEP1F2_REVIEW_CANDIDATE_SELECTION_REPORT_PATH,
    STEP1F2_REVIEWED_DECISIONS_PATH,
    read_json,
    write_json,
    write_text,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    safe_float,
)


TARGET_MIN_CANDIDATES = 80
TARGET_MAX_CANDIDATES = 120
HARD_MAX_CANDIDATES = 180
PURE_UNKNOWN_AMBIGUITY_FLAG = "unknown_goalkeeper_context_ambiguous_c2c_d1c_evidence"
MANDATORY_BUCKETS = {"severe_fusion_conflict_all", "gold_proxy_problem_rows"}

ALLOWED_F2_FINAL_ROLE_STATES = {
    "team_1_outfield_visual_context",
    "team_2_outfield_visual_context",
    "team_unknown_outfield_visual_context",
    "team_1_goalkeeper_visual_context",
    "team_2_goalkeeper_visual_context",
    "goalkeeper_unknown_team_visual_context",
    "official_referee_visual_context",
    "assistant_or_line_official_visual_context",
    "off_pitch_context_person_visual_context",
    "bad_detection_or_not_person",
    "unknown_visible_person_visual_context",
}

DECISION_TO_ROLE_STATE = {
    "accept_f1_role_state": "",
    "bulk_accept_bucket": "",
    "correct_to_team_1_outfield_visual_context": "team_1_outfield_visual_context",
    "correct_to_team_2_outfield_visual_context": "team_2_outfield_visual_context",
    "correct_to_team_unknown_outfield_visual_context": "team_unknown_outfield_visual_context",
    "correct_to_team_1_goalkeeper_visual_context": "team_1_goalkeeper_visual_context",
    "correct_to_team_2_goalkeeper_visual_context": "team_2_goalkeeper_visual_context",
    "correct_to_goalkeeper_unknown_team_visual_context": "goalkeeper_unknown_team_visual_context",
    "correct_to_official_referee_visual_context": "official_referee_visual_context",
    "correct_to_assistant_or_line_official_visual_context": "assistant_or_line_official_visual_context",
    "correct_to_off_pitch_context_person_visual_context": "off_pitch_context_person_visual_context",
    "correct_to_bad_detection_or_not_person": "bad_detection_or_not_person",
    "correct_to_unknown_visible_person_visual_context": "unknown_visible_person_visual_context",
    "unsure_needs_later_review": "unsure_needs_later_review",
}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def rows_by_visible_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("visible_person_base_id", "")): row for row in rows if row.get("visible_person_base_id")}


def deterministic_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            int(safe_float(row.get("frame_sequence"), -1)),
            str(row.get("step1f1_fused_visual_role_state", "")),
            str(row.get("visible_person_base_id", "")),
        ),
    )


def diverse_sample(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ordered = deterministic_order(rows)
    selected = []
    seen_frames: set[int] = set()
    seen_ids: set[str] = set()
    for row in ordered:
        seq = int(safe_float(row.get("frame_sequence"), -1))
        visible_id = str(row.get("visible_person_base_id", ""))
        if seq in seen_frames or visible_id in seen_ids:
            continue
        selected.append(row)
        seen_frames.add(seq)
        seen_ids.add(visible_id)
        if len(selected) >= limit:
            return selected
    for row in ordered:
        visible_id = str(row.get("visible_person_base_id", ""))
        if visible_id in seen_ids:
            continue
        selected.append(row)
        seen_ids.add(visible_id)
        if len(selected) >= limit:
            break
    return selected


def deterministic_random_sample(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: stable_hash(f"{row.get('visible_person_base_id','')}|{row.get('frame_sequence','')}"),
    )[:limit]


def is_pure_unknown_ambiguity(row: dict[str, Any]) -> bool:
    flags = set(row.get("step1f1_conflict_flags", []))
    return flags == {PURE_UNKNOWN_AMBIGUITY_FLAG}


def severe_conflict_rows(f1_rows: list[dict[str, Any]], conflict_payload: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = rows_by_visible_id(f1_rows)
    out = []
    for conflict in conflict_payload.get("rows", []):
        if is_pure_unknown_ambiguity(conflict):
            continue
        row = by_id.get(str(conflict.get("visible_person_base_id", "")))
        if row:
            out.append(row)
    return deterministic_order(out)


def gold_proxy_problem_rows(f1_rows: list[dict[str, Any]], labels_payload: dict[str, Any] | None = None, limit: int = 40) -> list[dict[str, Any]]:
    labels_payload = labels_payload or read_json(STAGE3C13_GOLD20_MANUAL_LABELS_PATH)
    matches, _missed, _extra = strict_one_to_one_match(gold_visible_person_rows(labels_payload), f1_rows)
    problems = []
    for match in matches:
        gold = match["gold"]
        row = match["candidate"]
        gold_type = str(gold.get("visible_person_type_gold", ""))
        role = str(row.get("step1f1_fused_visual_role_state", ""))
        problem = False
        reason = ""
        if gold_type in {"gk_team_1", "gk_team_2"} and role not in GOALKEEPER_ROLE_STATES:
            problem = True
            reason = "gold_goalkeeper_proxy_not_goalkeeper_role"
        elif gold_type == "team_1_player" and role != "team_1_outfield_visual_context":
            problem = role in OUTFIELD_ROLE_STATES or role in GOALKEEPER_ROLE_STATES or role in {"bad_detection_or_not_person", "unknown_visible_person_visual_context"}
            reason = "gold_team_1_player_proxy_role_disagreement" if problem else ""
        elif gold_type == "team_2_player" and role != "team_2_outfield_visual_context":
            problem = role in OUTFIELD_ROLE_STATES or role in GOALKEEPER_ROLE_STATES or role in {"bad_detection_or_not_person", "unknown_visible_person_visual_context"}
            reason = "gold_team_2_player_proxy_role_disagreement" if problem else ""
        elif gold_type in {"official_referee", "off_pitch_person"} and role not in OFFICIAL_CONTEXT_ROLE_STATES:
            problem = True
            reason = "gold_official_context_proxy_not_official_context_role"
        if problem:
            problems.append(
                {
                    **row,
                    "f2_gold_proxy_problem_reason": reason,
                    "gold_visible_person_type_gold": gold_type,
                    "gold_row_id": gold.get("gold_row_id", ""),
                    "gold_proxy_visual_gap_px": match.get("match_features", {}).get("visual_gap_px", 0.0),
                    "gold_proxy_bbox_iou": match.get("match_features", {}).get("bbox_iou", 0.0),
                }
            )
    return diverse_sample(problems, limit)


def rows_with_role(rows: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("step1f1_fused_visual_role_state") == role]


def clean_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if not row.get("step1f1_conflict_flags")
        and row.get("step1f1_review_required") is not True
        and safe_float(row.get("step1f1_role_confidence"), 0.0) >= 0.5
    ]


def add_candidates(
    selected: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    bucket: str,
    reason: str,
    limit: int,
    seen: set[str],
    mandatory: bool = False,
) -> None:
    added = 0
    for row in rows:
        visible_id = str(row.get("visible_person_base_id", ""))
        if not visible_id or visible_id in seen:
            continue
        selected.append(
            {
                **row,
                "step1f2_review_bucket": bucket,
                "step1f2_selection_reason": reason,
                "step1f2_bucket_mandatory": mandatory,
            }
        )
        seen.add(visible_id)
        added += 1
        if added >= limit:
            break


def build_review_candidate_rows(
    f1_payload: dict[str, Any],
    conflict_payload: dict[str, Any],
    eval_summary: dict[str, Any],
    *,
    labels_payload: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    f1_rows = f1_payload.get("rows", [])
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    severe = severe_conflict_rows(f1_rows, conflict_payload)
    gold_problems = gold_proxy_problem_rows(f1_rows, labels_payload=labels_payload)
    mandatory_total = len({str(row.get("visible_person_base_id", "")) for row in severe + gold_problems})
    scope_too_large = mandatory_total > HARD_MAX_CANDIDATES

    add_candidates(selected, severe, bucket="severe_fusion_conflict_all", reason="mandatory severe F1 conflict warning", limit=HARD_MAX_CANDIDATES, seen=seen, mandatory=True)
    add_candidates(selected, gold_problems, bucket="gold_proxy_problem_rows", reason="mandatory Gold visual proxy role disagreement", limit=40, seen=seen, mandatory=True)

    if len(selected) < HARD_MAX_CANDIDATES:
        gk_rows = rows_with_role(f1_rows, "team_1_goalkeeper_visual_context") + rows_with_role(f1_rows, "team_2_goalkeeper_visual_context")
        add_candidates(selected, diverse_sample(gk_rows, 30), bucket="goalkeeper_sanity_sample", reason="balanced goalkeeper visual-context sanity sample", limit=30, seen=seen)
    if len(selected) < TARGET_MAX_CANDIDATES:
        unknown_rows = [
            row
            for row in f1_rows
            if row.get("step1f1_fused_visual_role_state") == "unknown_visible_person_visual_context"
            or PURE_UNKNOWN_AMBIGUITY_FLAG in set(row.get("step1f1_conflict_flags", []))
        ]
        add_candidates(selected, diverse_sample(unknown_rows, 20), bucket="unknown_ambiguous_sample", reason="sampled unknown/ambiguous visual role-state rows", limit=min(20, TARGET_MAX_CANDIDATES - len(selected)), seen=seen)
    if len(selected) < TARGET_MAX_CANDIDATES:
        bad_rows = rows_with_role(f1_rows, "bad_detection_or_not_person")
        add_candidates(selected, diverse_sample(bad_rows, 15), bucket="bad_detection_sample", reason="sampled bad detection visual QA beliefs", limit=min(15, TARGET_MAX_CANDIDATES - len(selected)), seen=seen)
    if len(selected) < TARGET_MAX_CANDIDATES:
        clean = clean_rows(f1_rows)
        clean_caps = [
            ("team_1_outfield_visual_context", 15),
            ("team_2_outfield_visual_context", 15),
            ("official_referee_visual_context", 10),
            ("assistant_or_line_official_visual_context", 10),
            ("off_pitch_context_person_visual_context", 10),
        ]
        for role, cap in clean_caps:
            if len(selected) >= TARGET_MAX_CANDIDATES:
                break
            role_rows = [row for row in clean if row.get("step1f1_fused_visual_role_state") == role]
            add_candidates(
                selected,
                diverse_sample(role_rows, cap),
                bucket="balanced_clean_role_sample",
                reason=f"clean high-confidence sanity sample for {role}",
                limit=min(cap, TARGET_MAX_CANDIDATES - len(selected)),
                seen=seen,
            )
    if len(selected) < TARGET_MIN_CANDIDATES:
        remaining_clean = [row for row in clean_rows(f1_rows) if str(row.get("visible_person_base_id", "")) not in seen]
        add_candidates(
            selected,
            deterministic_random_sample(remaining_clean, 10),
            bucket="random_clean_sanity_sample",
            reason="deterministic random clean sanity sample",
            limit=min(10, TARGET_MAX_CANDIDATES - len(selected)),
            seen=seen,
        )

    selected = selected[:HARD_MAX_CANDIDATES]
    bucket_counts = Counter(str(row.get("step1f2_review_bucket", "")) for row in selected)
    ordered = sorted(
        enumerate(selected),
        key=lambda item: (
            [
                "severe_fusion_conflict_all",
                "gold_proxy_problem_rows",
                "goalkeeper_sanity_sample",
                "unknown_ambiguous_sample",
                "bad_detection_sample",
                "balanced_clean_role_sample",
                "random_clean_sanity_sample",
            ].index(item[1].get("step1f2_review_bucket", "random_clean_sanity_sample")),
            int(safe_float(item[1].get("frame_sequence"), -1)),
            str(item[1].get("visible_person_base_id", "")),
        ),
    )
    rows = []
    for index, (_old_index, row) in enumerate(ordered):
        review_id = f"f2_review_{index:04d}_{row.get('visible_person_base_id','')}"
        rows.append(
            {
                **row,
                "step1f2_review_candidate_id": review_id,
                "step1f2_review_index": index,
                "proposed_f1_role_state": row.get("step1f1_fused_visual_role_state", ""),
                "saved_human_review_decision": "",
                "saved_human_corrected_role_state": "",
                "saved_notes": "",
                "ui_is_reviewed": False,
                "visual_only_warning": VISUAL_ONLY_WARNING,
                "do_not_use_for_metrics": True,
                "production_ready": PRODUCTION_READY,
            }
        )

    selection_summary = {
        "artifact": "step1f2_review_candidate_selection_summary",
        "created_at": utc_iso(),
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "goalkeeper_slot_assignment_performed": False,
        "expected_22_role_states_created": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "target_min_review_candidates": TARGET_MIN_CANDIDATES,
        "target_max_review_candidates": TARGET_MAX_CANDIDATES,
        "hard_max_review_candidates": HARD_MAX_CANDIDATES,
        "total_review_candidates": len(rows),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "mandatory_bucket_counts": {
            "severe_fusion_conflict_all": len(severe),
            "gold_proxy_problem_rows": len(gold_problems),
        },
        "unknown_ambiguity_full_count": sum(1 for row in f1_rows if PURE_UNKNOWN_AMBIGUITY_FLAG in set(row.get("step1f1_conflict_flags", []))),
        "unknown_ambiguity_sampled_count": bucket_counts.get("unknown_ambiguous_sample", 0),
        "f2_review_scope_too_large_rebuild_f1_rules": scope_too_large or len(rows) > HARD_MAX_CANDIDATES,
        "f1_safe_for_f2_human_review_candidate": eval_summary.get("f1_safe_for_f2_human_review_candidate", False),
    }
    return rows, selection_summary


def selection_report(selection_summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.F2 Review Candidate Selection Report",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            "- Scope: small triage review set for F1 fused visual role-state candidates.",
            "- The 2010 unknown-goalkeeper ambiguity rows are sampled, not exhaustively reviewed.",
            "- This is not canonical relabelling and not production-ready.",
            "",
            "## Counts",
            "",
            f"- Total review candidates: {selection_summary.get('total_review_candidates', 0)}",
            f"- Target range: {TARGET_MIN_CANDIDATES}-{TARGET_MAX_CANDIDATES}",
            f"- Hard cap: {HARD_MAX_CANDIDATES}",
            f"- Scope too large: {selection_summary.get('f2_review_scope_too_large_rebuild_f1_rules', False)}",
            "",
            "## Bucket Counts",
            "",
            "```json",
            json.dumps(selection_summary.get("bucket_counts", {}), indent=2),
            "```",
            "",
            "## Mandatory Buckets",
            "",
            "```json",
            json.dumps(selection_summary.get("mandatory_bucket_counts", {}), indent=2),
            "```",
        ]
    ) + "\n"


def empty_reviewed_decisions_payload() -> dict[str, Any]:
    return {
        "artifact": "step1f2_reviewed_fused_role_state_decisions",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "goalkeeper_slot_assignment_performed": False,
        "expected_22_role_states_created": False,
        "official_specialist_exclusion_performed": False,
        "rows": [],
    }


def build_and_write_f2_review_candidates() -> dict[str, Any]:
    f1_payload = read_json(STEP1F1_FUSED_VISUAL_ROLE_STATE_ROWS_PATH)
    conflict_payload = read_json(STEP1F1_ROLE_STATE_CONFLICT_AUDIT_ROWS_PATH)
    eval_summary = read_json(STEP1F1_FUSED_VISUAL_ROLE_STATE_EVAL_SUMMARY_PATH)
    rows, selection_summary = build_review_candidate_rows(f1_payload, conflict_payload, eval_summary)
    payload = {
        "artifact": "step1f2_fused_role_state_review_candidate_rows",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "goalkeeper_slot_assignment_performed": False,
        "expected_22_role_states_created": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "selection_summary": selection_summary,
        "rows": rows,
    }
    write_json(STEP1F2_REVIEW_CANDIDATE_ROWS_PATH, payload)
    if not STEP1F2_REVIEWED_DECISIONS_PATH.exists():
        write_json(STEP1F2_REVIEWED_DECISIONS_PATH, empty_reviewed_decisions_payload())
    write_text(STEP1F2_REVIEW_CANDIDATE_SELECTION_REPORT_PATH, selection_report(selection_summary))
    if selection_summary.get("f2_review_scope_too_large_rebuild_f1_rules"):
        raise RuntimeError("f2_review_scope_too_large_rebuild_f1_rules=true")
    return payload
