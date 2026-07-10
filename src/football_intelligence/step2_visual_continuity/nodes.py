from __future__ import annotations

from collections import Counter
from typing import Any

from football_intelligence.paths import CLIP_ID, MATCH_ID
from football_intelligence.step2_visual_continuity.schema import (
    VISUAL_ONLY_WARNING,
    assert_no_forbidden_keys,
    bbox_from_row,
    footpoint_from_row,
    guardrail_stamp,
    safe_int,
    utc_iso,
    visual_stamp,
)


PROVENANCE_PREFIXES = (
    "c2c_",
    "d1c_",
    "e1c_",
    "step1f1_",
    "step1f3_",
)

REQUIRED_NODE_FIELDS = [
    "step2m1_visual_continuity_node_id",
    "visible_person_base_id",
    "frame_sequence",
    "timestamp_seconds",
    "bbox",
    "footpoint",
    "crop_quality",
    "candidate_type",
    "roi_status",
    "step1f3_final_visual_role_state",
    "step1f3_final_visual_role_group",
    "step1f3_role_team_context",
    "step1f3_warning_flags",
    "step1f3_review_required",
    "retained_for_future_player_team_review",
]


def rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("rows", [])
    return value if isinstance(value, list) else []


def node_id_for_row(row: dict[str, Any]) -> str:
    return f"step2m1_vcnode_{row.get('visible_person_base_id', row.get('detection_id', 'unknown'))}"


def provenance_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key.startswith(PROVENANCE_PREFIXES)
    }


def merged_warning_flags(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    for key in [
        "step1f3_warning_flags",
        "step1f1_warning_flags",
        "step1f1_conflict_flags",
        "ambiguity_flags",
        "e1_goalkeeper_context_warning_flags",
        "e1c_warning_flags",
    ]:
        value = row.get(key, [])
        if isinstance(value, list):
            flags.extend(str(item) for item in value if str(item))
        elif value:
            flags.append(str(value))
    if row.get("step1f3_review_required") is True:
        flags.append("step1f3_review_required")
    return list(dict.fromkeys(flags))


def compact_context_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "c2c_final_colour_belief": row.get("c2c_final_colour_belief", ""),
        "c2c_colour_source": row.get("c2c_colour_source", row.get("c2c_context_source", "")),
        "d1c_final_official_context_belief": row.get("d1c_final_official_context_belief", ""),
        "d1c_context_source": row.get("d1c_context_source", ""),
        "e1c_final_goalkeeper_context_belief": row.get("e1c_final_goalkeeper_context_belief", ""),
        "e1c_context_source": row.get("e1c_context_source", ""),
        "step1f3_final_visual_role_state": row.get("step1f3_final_visual_role_state", ""),
        "step1f3_final_visual_role_group": row.get("step1f3_final_visual_role_group", ""),
        "step1f3_role_team_context": row.get("step1f3_role_team_context", ""),
    }


def build_node_row(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    bbox = bbox_from_row(row) or {}
    warning_flags = merged_warning_flags(row)
    out = {
        "step2m1_visual_continuity_node_id": node_id_for_row(row),
        "step2m1_source_row_index": row_index,
        "visible_person_base_id": str(row.get("visible_person_base_id", "")),
        "frame_id": row.get("frame_id", ""),
        "frame_sequence": safe_int(row.get("frame_sequence"), -1),
        "timestamp_seconds": row.get("timestamp_seconds", None),
        "detection_id": row.get("detection_id", ""),
        "source_detection_id": row.get("source_detection_id", ""),
        "bbox": bbox,
        "footpoint": footpoint_from_row(row),
        "crop_quality": str(row.get("crop_quality", "unknown")),
        "candidate_type": str(row.get("candidate_type", "unknown_person_candidate")),
        "roi_status": str(row.get("roi_status", "")),
        "state": row.get("state", ""),
        "step1f3_final_visual_role_state": str(
            row.get("step1f3_final_visual_role_state", "unknown_visible_person_visual_context")
        ),
        "step1f3_final_visual_role_group": str(row.get("step1f3_final_visual_role_group", "")),
        "step1f3_role_team_context": str(row.get("step1f3_role_team_context", "")),
        "step1f3_warning_flags": warning_flags,
        "step1f3_review_required": row.get("step1f3_review_required") is True,
        "retained_for_future_player_team_review": row.get("retained_for_future_player_team_review") is not False,
        "step2m1_context_snapshot": compact_context_snapshot(row),
        "step2m1_visual_continuity_node_is_identity": False,
        "step2m1_visual_continuity_node_is_player_slot": False,
        "step2m1_visual_continuity_node_is_goalkeeper_slot": False,
        "step2m1_visual_continuity_node_is_metric": False,
        "sandbox_only": True,
    }
    out.update(provenance_fields(row))
    visual_stamp(out)
    assert_no_forbidden_keys(out)
    return out


def validate_node_rows(f3_payload: dict[str, Any], node_rows: list[dict[str, Any]]) -> None:
    f3_rows = rows(f3_payload)
    if len(f3_rows) != len(node_rows):
        raise ValueError(f"Step2.M1 node row count mismatch: f3={len(f3_rows)} nodes={len(node_rows)}")
    f3_ids = [str(row.get("visible_person_base_id", "")) for row in f3_rows]
    node_ids = [str(row.get("visible_person_base_id", "")) for row in node_rows]
    if f3_ids != node_ids:
        raise ValueError("Step2.M1 node visible_person_base_id alignment was not preserved")
    for node in node_rows:
        missing = [field for field in REQUIRED_NODE_FIELDS if field not in node]
        if missing:
            raise ValueError(f"Step2.M1 node missing fields: {missing}")


def build_node_payload(f3_payload: dict[str, Any], g1_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    g1_manifest = g1_manifest or {}
    node_rows = [build_node_row(row, index) for index, row in enumerate(rows(f3_payload))]
    validate_node_rows(f3_payload, node_rows)
    role_counts = Counter(str(row.get("step1f3_final_visual_role_state", "")) for row in node_rows)
    candidate_counts = Counter(str(row.get("candidate_type", "")) for row in node_rows)
    review_required = sum(1 for row in node_rows if row.get("step1f3_review_required") is True)
    payload = guardrail_stamp(
        {
            "artifact": "step2m1_visual_continuity_node_rows",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "source_artifact": f3_payload.get("artifact", "step1f3_human_corrected_fused_visual_role_state_rows"),
            "step1g1_safe_for_step2_visual_continuity_candidate": g1_manifest.get(
                "step1g1_safe_for_step2_visual_continuity_candidate",
                False,
            ),
            "node_creation_rule": "Each Step1.F3 row becomes exactly one Step2.M1 visual-continuity node.",
            "visual_continuity_node_count": len(node_rows),
            "f3_row_count": len(rows(f3_payload)),
            "one_node_per_f3_row": len(node_rows) == len(rows(f3_payload)),
            "visible_person_base_id_alignment_preserved": [
                str(row.get("visible_person_base_id", "")) for row in rows(f3_payload)
            ]
            == [str(row.get("visible_person_base_id", "")) for row in node_rows],
            "rows": node_rows,
            "summary": {
                "visual_continuity_node_rows": len(node_rows),
                "f3_row_count": len(rows(f3_payload)),
                "one_node_per_f3_row": len(node_rows) == len(rows(f3_payload)),
                "step1f3_review_required_nodes": review_required,
                "role_state_counts": dict(sorted(role_counts.items())),
                "candidate_type_counts": dict(sorted(candidate_counts.items())),
                "visual_only_warning": VISUAL_ONLY_WARNING,
            },
        }
    )
    assert_no_forbidden_keys(payload)
    return payload
