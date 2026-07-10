from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING, visual_stamp  # noqa: E402
from football_intelligence.step1_visual_reconstruction.visible_person_base import build_visible_person_base_payloads  # noqa: E402


def row(
    det_id: str,
    *,
    counted: bool = True,
    action: str = "primary_observation_candidate",
    state: str = "observed_clear",
) -> dict[str, Any]:
    return visual_stamp(
        {
            "frame_id": "frame_001",
            "frame_sequence": 1,
            "timestamp_seconds": 1.0,
            "frame_file": "frame_001.jpg",
            "detection_id": det_id,
            "source_detection_id": f"source_{det_id}",
            "bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 80.0},
            "footpoint": {"x": 20.0, "y": 80.0, "method": "bbox_bottom_center", "confidence": 0.85},
            "candidate_type": "player_candidate_source",
            "original_role_source": "player",
            "source_role_labels": ["player_candidate"],
            "source_candidate_types": ["player_candidate_source"],
            "source_model_stages": ["fixture"],
            "bbox_confidence": 0.8,
            "bbox_quality_score": 0.8,
            "bbox_quality_reason": "bbox_plausible",
            "crop_quality": None,
            "roi_status": "inside_or_unverified_visual_roi",
            "duplicate_group_id": "group",
            "duplicate_action": "unique",
            "state": state,
            "confidence": 0.8,
            "reason": "fixture",
            "observed_visible_candidate": counted,
            "qa_warnings": [],
            "qa_render_tier": "primary_observed",
            "visual_object_group_id": f"group_{det_id}",
            "visual_object_group_size": 1,
            "reconciliation_action": action,
            "reconciliation_confidence": 0.88,
            "reconciliation_reason": "fixture",
            "count_as_observed_visible_candidate_b3": counted,
            "count_policy_reason": f"{action}_counted" if counted else "fixture_not_counted",
            "review_required": False,
            "source_disagreement_review_required": False,
        }
    )


def test_only_b3_counted_rows_enter_visible_person_base_and_shadows_are_excluded() -> None:
    payload, provenance = build_visible_person_base_payloads(
        {
            "artifact": "step1b3_count_policy_rows",
            "rows": [
                row("det_primary"),
                row("det_not_counted", counted=False),
                row("det_duplicate", action="duplicate_shadow_candidate"),
                row("det_source_shadow", action="source_overlap_shadow_candidate"),
            ],
        }
    )
    assert [item["detection_id"] for item in payload["rows"]] == ["det_primary"]
    assert provenance["summary"]["provenance_rows"] == 4
    exclusions = {item["detection_id"]: item["visible_person_base_exclusion_reason"] for item in provenance["rows"]}
    assert exclusions["det_not_counted"] == "b3_count_policy_false"
    assert exclusions["det_duplicate"] == "shadow_candidate_excluded_from_visible_base"
    assert exclusions["det_source_shadow"] == "shadow_candidate_excluded_from_visible_base"


def test_visible_person_base_id_is_unique_stable_and_schema_flags_are_visual_only() -> None:
    source = {"artifact": "step1b3_count_policy_rows", "rows": [row("det_a"), row("det_b")]}
    first_payload, _first_provenance = build_visible_person_base_payloads(source)
    second_payload, _second_provenance = build_visible_person_base_payloads(source)
    first_ids = [item["visible_person_base_id"] for item in first_payload["rows"]]
    second_ids = [item["visible_person_base_id"] for item in second_payload["rows"]]
    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))
    assert all(item["eligible_for_step1c_team_colour_candidate"] is True for item in first_payload["rows"])
    assert all(item["eligible_for_identity_tracking"] is False for item in first_payload["rows"])
    assert all(item["production_ready"] is False for item in first_payload["rows"])
    assert all(item["visual_only_warning"] == VISUAL_ONLY_WARNING for item in first_payload["rows"])
