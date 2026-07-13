from __future__ import annotations

from typing import Any

from football_intelligence.review.schemas import safety_payload


def rows_from_payload(payload: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [row for row in payload["rows"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError("payload must be rows or object with rows")


def build_rebuilt_continuity_nodes(
    visible_payload: dict[str, Any] | list[dict[str, Any]],
    entity_payload: dict[str, Any] | list[dict[str, Any]],
    fused_context_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    visible_rows = rows_from_payload(visible_payload)
    entity_rows = rows_from_payload(entity_payload)
    entity_by_source = {
        str(row.get("source_detection_id")): row for row in entity_rows if row.get("source_detection_id")
    }
    context_by_candidate = {str(row.get("candidate_id")): row for row in fused_context_rows}
    rows: list[dict[str, Any]] = []
    for index, visible in enumerate(visible_rows):
        entity = entity_by_source.get(str(visible.get("source_detection_id")), {})
        context = context_by_candidate.get(str(entity.get("candidate_id")), {})
        entity_state = entity.get("entity_validity_state", "ambiguous_entity_requires_review")
        context_state = context.get("visual_role_context_state", "unknown_visible_person_visual_context")
        rows.append(
            {
                "continuity_node_id": f"m5_4d_cnode_{index:06d}",
                "visible_person_base_id": visible.get("visible_person_base_id"),
                "candidate_id": entity.get("candidate_id"),
                "raw_detector_row_id": entity.get("raw_detector_row_id"),
                "source_detection_id": visible.get("source_detection_id"),
                "frame_sequence": int(visible.get("frame_sequence", 0)),
                "bbox": visible.get("bbox"),
                "entity_validity_state": entity_state,
                "visual_role_context_state": context_state,
                "continuity_eligible": entity_state in {"valid_on_pitch_person", "ambiguous_entity_requires_review"},
                "visual_continuity_is_real_identity": False,
                "visual_continuity_is_player_slot": False,
                "visual_continuity_is_metric": False,
                "raw_row_preserved": True,
                "detector_row_deleted": False,
                **safety_payload(),
            }
        )
    return {
        "artifact": "m5_4d_continuity_node_rows",
        "rows": rows,
        "node_count": len(rows),
        **safety_payload(),
    }
