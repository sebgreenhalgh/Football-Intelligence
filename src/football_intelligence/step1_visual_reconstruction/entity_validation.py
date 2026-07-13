from __future__ import annotations

from typing import Any


def validate_entity_spine(
    *,
    detector_count: int,
    candidate_payload: dict[str, Any],
    duplicate_payload: dict[str, Any],
    spatial_payload: dict[str, Any],
    feature_payload: dict[str, Any],
    entity_payload: dict[str, Any],
) -> dict[str, Any]:
    counts = {
        "detector_rows": detector_count,
        "candidate_rows": len(candidate_payload.get("rows", [])),
        "duplicate_rows": len(duplicate_payload.get("rows", [])),
        "spatial_rows": len(spatial_payload.get("rows", [])),
        "feature_rows": len(feature_payload.get("rows", [])),
        "entity_rows": len(entity_payload.get("rows", [])),
    }
    same_count = len(set(counts.values())) == 1
    all_person_candidate = all(
        row.get("candidate_type") == "person_candidate" for row in candidate_payload.get("rows", [])
    )
    no_auto_player = all(not row.get("auto_labelled_player", True) for row in candidate_payload.get("rows", []))
    no_deleted = all(not row.get("detector_row_deleted", True) for row in candidate_payload.get("rows", []))
    trace_ids = [
        {str(row.get("raw_detector_row_id")) for row in payload.get("rows", [])}
        for payload in (candidate_payload, duplicate_payload, spatial_payload, feature_payload, entity_payload)
    ]
    traceable = all(ids == trace_ids[0] for ids in trace_ids[1:]) if trace_ids else False
    return {
        "artifact": "m5_4d_entity_spine_validation",
        "counts": counts,
        "passed": same_count and all_person_candidate and no_auto_player and no_deleted and traceable,
        "every_detector_row_remains_traceable": traceable,
        "detector_outputs_begin_as_person_candidate": all_person_candidate,
        "detector_rows_auto_labelled_player": not no_auto_player,
        "raw_rows_deleted": not no_deleted,
    }
