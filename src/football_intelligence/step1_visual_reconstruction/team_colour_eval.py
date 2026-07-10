# ruff: noqa: E501

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

from football_intelligence.paths import STAGE3C13_GOLD20_MANUAL_LABELS_PATH
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import (
    gold_visible_person_rows,
    load_completed_gold8_frames,
    strict_one_to_one_match,
)
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1B4_GOLD8_EVAL_SUMMARY_PATH,
    STEP1C1_COLOUR_FEATURE_ROWS_PATH,
    STEP1C1_GOLD8_COLOUR_EVAL_REPORT_PATH,
    STEP1C1_GOLD8_COLOUR_EVAL_SUMMARY_PATH,
    STEP1C1_REVIEW_DECISION_TEMPLATE_PATH,
    STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH,
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


GOLD_TEAM_COLOUR_PROXY_TYPES = {"team_1_player", "team_2_player"}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def review_decision_template_payload() -> dict[str, Any]:
    return {
        "reviewer_name": "",
        "reviewed_at": "",
        "contact_sheet_reviewed": False,
        "crop_contact_sheet_reviewed": False,
        "approve_step1c1_colour_beliefs_for_next_stage": False,
        "approve_cluster_to_team_mapping": False,
        "known_issues": [],
        "frames_requiring_manual_followup": [],
        "notes": "",
        "production_ready": PRODUCTION_READY,
        "no_auto_promotion": True,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
    }


def gold_colour_fields_available(labels_payload: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
    used = []
    missing = []
    has_visible_type = any(
        "visible_person_type_gold" in person
        for frame in labels_payload.get("frames", [])
        for person in frame.get("persons", [])
    )
    if has_visible_type:
        used.append("visible_person_type_gold")
    else:
        missing.append("visible_person_type_gold")
    has_team_proxy = any(
        str(person.get("visible_person_type_gold", "")) in GOLD_TEAM_COLOUR_PROXY_TYPES
        for frame in labels_payload.get("frames", [])
        for person in frame.get("persons", [])
    )
    return bool(has_visible_type and has_team_proxy), used, missing


def colour_eval_against_gold8(
    belief_payload: dict[str, Any],
    *,
    labels_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels_payload = labels_payload or read_json(STAGE3C13_GOLD20_MANUAL_LABELS_PATH)
    available, fields_used, fields_missing = gold_colour_fields_available(labels_payload)
    completed_frames = load_completed_gold8_frames(labels_payload)
    gold_rows = gold_visible_person_rows(labels_payload)
    frame_sequences = {int(safe_float(frame.get("frame_sequence"), -1)) for frame in completed_frames}
    belief_rows = [
        row
        for row in belief_payload.get("rows", [])
        if int(safe_float(row.get("frame_sequence"), -1)) in frame_sequences
    ]
    matches, missed, extra = strict_one_to_one_match(gold_rows, belief_rows)
    gold_proxy_matches = [
        match
        for match in matches
        if str(match["gold"].get("visible_person_type_gold", "")) in GOLD_TEAM_COLOUR_PROXY_TYPES
    ]
    distribution: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for match in gold_proxy_matches:
        gold_type = str(match["gold"].get("visible_person_type_gold", ""))
        belief = str(match["candidate"].get("team_colour_belief", ""))
        distribution[gold_type][belief] += 1
    issue_rows = []
    for row in missed:
        issue_rows.append(
            {
                "issue_type": "gold_visible_person_unmatched_by_c1_belief_row",
                "frame_id": row.get("frame_id", ""),
                "frame_sequence": row.get("frame_sequence", -1),
                "gold_row_id": row.get("gold_row_id", ""),
                "visible_person_type_gold": row.get("visible_person_type_gold", ""),
                "visual_only_warning": VISUAL_ONLY_WARNING,
                "do_not_use_for_metrics": True,
                "production_ready": PRODUCTION_READY,
            }
        )
    for row in extra:
        issue_rows.append(
            {
                "issue_type": "c1_belief_row_extra_vs_gold_visible",
                "frame_id": row.get("frame_id", ""),
                "frame_sequence": row.get("frame_sequence", -1),
                "visible_person_base_id": row.get("visible_person_base_id", ""),
                "team_colour_belief": row.get("team_colour_belief", ""),
                "visual_only_warning": VISUAL_ONLY_WARNING,
                "do_not_use_for_metrics": True,
                "production_ready": PRODUCTION_READY,
            }
        )
    frame_cluster_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for row in belief_rows:
        frame_cluster_counts[int(safe_float(row.get("frame_sequence"), -1))][str(row.get("team_colour_belief", ""))] += 1
    frame_contradictions = [
        {
            "frame_sequence": seq,
            "distinct_non_unknown_colour_beliefs": len(
                [
                    key
                    for key, value in counts.items()
                    if value > 0 and key not in {"unknown_ambiguous_colour", "crop_unusable"}
                ]
            ),
            "cluster_counts": dict(sorted(counts.items())),
        }
        for seq, counts in frame_cluster_counts.items()
        if len([key for key in counts if key not in {"unknown_ambiguous_colour", "crop_unusable"}]) > 3
    ]
    return (
        {
            "gold8_colour_eval_available": available,
            "gold8_colour_eval_fields_used": fields_used,
            "gold8_colour_eval_fields_missing": fields_missing,
            "gold8_colour_proxy_match_rows": len(gold_proxy_matches),
            "gold8_colour_proxy_distribution": {key: dict(sorted(value.items())) for key, value in sorted(distribution.items())},
            "gold8_visible_matches": len(matches),
            "gold8_visible_missed": len(missed),
            "gold8_visible_extra": len(extra),
            "one_frame_colour_contradiction_diagnostics": frame_contradictions[:20],
            "frames_needing_manual_followup": [item["frame_sequence"] for item in frame_contradictions[:12]],
            "note": "Gold visible_person_type_gold is used only as optional visual colour QA context, not role correctness.",
        },
        issue_rows,
    )


def build_eval_summary(
    feature_payload: dict[str, Any],
    belief_payload: dict[str, Any],
    *,
    b4_summary: dict[str, Any] | None = None,
    labels_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    b4_summary = b4_summary or read_json(STEP1B4_GOLD8_EVAL_SUMMARY_PATH)
    belief_summary = belief_payload.get("summary", {})
    gold_eval, issue_rows = colour_eval_against_gold8(belief_payload, labels_payload=labels_payload)
    cluster_counts = belief_summary.get("cluster_counts", {})
    mapped_counts = belief_summary.get("mapped_team_colour_counts", {})
    completed_frames = load_completed_gold8_frames(labels_payload) if labels_payload is not None else load_completed_gold8_frames()
    summary = {
        "artifact": "step1c1_gold8_colour_eval_summary",
        "created_at": utc_iso(),
        "b4_visible_person_base_rows": b4_summary.get("b4_total_visible_person_base_rows", belief_summary.get("b4_visible_person_base_rows", 0)),
        "step1c1_colour_feature_rows": len(feature_payload.get("rows", [])),
        "step1c1_team_colour_belief_rows": len(belief_payload.get("rows", [])),
        "unknown_ambiguous_colour_rows": belief_summary.get("unknown_ambiguous_colour_rows", 0),
        "crop_unusable_rows": belief_summary.get("crop_unusable_rows", 0),
        "high_confidence_visual_colour_rows": belief_summary.get("high_confidence_visual_colour_rows", 0),
        "medium_confidence_visual_colour_rows": belief_summary.get("medium_confidence_visual_colour_rows", 0),
        "low_confidence_visual_colour_rows": belief_summary.get("low_confidence_visual_colour_rows", 0),
        "review_required_rows": belief_summary.get("review_required_rows", 0),
        "source_disagreement_review_required_rows": belief_summary.get("source_disagreement_review_required_rows", 0),
        "cluster_counts": cluster_counts,
        "mapped_team_colour_counts": mapped_counts,
        "gold8_frames_used": [
            {
                "frame_id": frame.get("frame_id", ""),
                "frame_sequence": int(safe_float(frame.get("frame_sequence"), -1)),
                "timestamp_seconds": safe_float(frame.get("timestamp_seconds")),
            }
            for frame in completed_frames
        ],
        "gold8_colour_eval_available": gold_eval["gold8_colour_eval_available"],
        "gold8_colour_eval_fields_used": gold_eval["gold8_colour_eval_fields_used"],
        "gold8_colour_eval_summary": gold_eval,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "goalkeeper_classification_performed": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
    }
    return summary, issue_rows


def gold8_colour_eval_report(summary: dict[str, Any], issue_rows: list[dict[str, Any]]) -> str:
    gold_eval = summary.get("gold8_colour_eval_summary", {})
    lines = [
        "# Step1.C1 Gold-8 Colour Eval Report",
        "",
        f"- Warning: `{VISUAL_ONLY_WARNING}`.",
        "- Scope: visual colour belief QA only.",
        "- No identity, player-slot, expected-role, goalkeeper-role, official-specialist, tactical, football, or metric correctness was evaluated.",
        f"- Gold colour eval available: {str(summary.get('gold8_colour_eval_available', False)).lower()}",
        f"- Gold fields used: {summary.get('gold8_colour_eval_fields_used', [])}",
        f"- Gold fields missing: {gold_eval.get('gold8_colour_eval_fields_missing', [])}",
        "",
        "## Summary",
        "",
        f"- B4 visible-person base rows: {summary.get('b4_visible_person_base_rows', 0)}",
        f"- Step1.C1 belief rows: {summary.get('step1c1_team_colour_belief_rows', 0)}",
        f"- unknown/ambiguous rows: {summary.get('unknown_ambiguous_colour_rows', 0)}",
        f"- crop unusable rows: {summary.get('crop_unusable_rows', 0)}",
        f"- review-required rows: {summary.get('review_required_rows', 0)}",
        f"- issue rows: {len(issue_rows)}",
        "",
        "## Gold Proxy Distribution",
        "",
        "Gold visible-person type is used only as visual QA context when present, not as role correctness.",
        "",
        "```json",
        str(gold_eval.get("gold8_colour_proxy_distribution", {})),
        "```",
    ]
    return "\n".join(lines) + "\n"


def build_and_write_colour_eval() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    feature_payload = read_json(STEP1C1_COLOUR_FEATURE_ROWS_PATH)
    belief_payload = read_json(STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH)
    summary, issue_rows = build_eval_summary(feature_payload, belief_payload)
    write_json(STEP1C1_GOLD8_COLOUR_EVAL_SUMMARY_PATH, summary)
    write_text(STEP1C1_GOLD8_COLOUR_EVAL_REPORT_PATH, gold8_colour_eval_report(summary, issue_rows))
    write_json(
        STEP1C1_REVIEW_DECISION_TEMPLATE_PATH,
        review_decision_template_payload(),
    )
    return summary, issue_rows
