from __future__ import annotations

from typing import Any

from football_intelligence.review.schemas import safety_payload


def build_continuity_segments(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, edge in enumerate(candidate_rows):
        rows.append(
            {
                "visual_continuity_segment_id": f"m5_4d_vcseg_{index:06d}",
                "visual_continuity_id": f"m5_4d_vcid_{index:06d}",
                "short_window_continuity_candidate_id": edge.get("short_window_continuity_candidate_id"),
                "source_visible_person_base_id": edge.get("source_visible_person_base_id"),
                "target_visible_person_base_id": edge.get("target_visible_person_base_id"),
                "frame_sequences": [edge.get("source_frame_sequence"), edge.get("target_frame_sequence")],
                "segment_state": "review_candidate_not_human_approved",
                "visual_continuity_is_real_identity": False,
                "visual_continuity_is_player_slot": False,
                "visual_continuity_is_metric": False,
                **safety_payload(),
            }
        )
    return {
        "artifact": "m5_4d_continuity_segment_rows",
        "rows": rows,
        "segment_count": len(rows),
        **safety_payload(),
    }
