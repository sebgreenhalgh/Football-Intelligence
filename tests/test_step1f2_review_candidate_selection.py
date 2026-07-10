from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_review_selection import (  # noqa: E402
    HARD_MAX_CANDIDATES,
    PURE_UNKNOWN_AMBIGUITY_FLAG,
    build_review_candidate_rows,
)


def f1_row(index: int, role: str, *, conflict_flags: list[str] | None = None) -> dict:
    x1 = 100.0 + float(index % 50) * 3.0
    return {
        "visible_person_base_id": f"base_{index}",
        "frame_id": f"frame_{index}",
        "frame_sequence": index,
        "timestamp_seconds": float(index),
        "detection_id": f"det_{index}",
        "bbox": {"x1": x1, "y1": 100.0, "x2": x1 + 30.0, "y2": 180.0},
        "footpoint": {"x": x1 + 15.0, "y": 180.0, "method": "bbox", "confidence": 0.9},
        "c2c_final_colour_belief": "team_1_outfield_colour_like",
        "d1c_final_official_context_belief": "player_like_not_official_context",
        "e1c_final_goalkeeper_context_belief": "outfield_player_like_not_goalkeeper",
        "step1f1_fused_visual_role_state": role,
        "step1f1_fused_visual_role_group": "player_outfield_visual_context",
        "step1f1_role_confidence": 0.8,
        "step1f1_review_required": bool(conflict_flags),
        "step1f1_conflict_flags": conflict_flags or [],
        "step1f1_warning_flags": [],
        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        "do_not_use_for_metrics": True,
        "production_ready": False,
    }


def labels_payload() -> dict:
    return {
        "frames": [
            {
                "frame_id": "frame_0",
                "frame_sequence": 0,
                "timestamp_seconds": 0.0,
                "labels_complete": True,
                "persons": [
                    {
                        "gold_person_id": "gold_gk",
                        "visible_person_type_gold": "gk_team_1",
                        "occlusion_state_gold": "observed_clear",
                        "bbox": {"x1": 100.0, "y1": 100.0, "x2": 130.0, "y2": 180.0},
                    }
                ],
            }
        ]
    }


def synthetic_payloads() -> tuple[dict, dict]:
    rows = [f1_row(0, "unknown_visible_person_visual_context")]
    rows.extend(f1_row(index, "team_1_goalkeeper_visual_context") for index in range(1, 20))
    rows.extend(f1_row(index, "team_2_goalkeeper_visual_context") for index in range(20, 40))
    rows.extend(
        f1_row(index, "unknown_visible_person_visual_context", conflict_flags=[PURE_UNKNOWN_AMBIGUITY_FLAG])
        for index in range(40, 2060)
    )
    rows.extend(f1_row(index, "bad_detection_or_not_person") for index in range(2060, 2100))
    rows.extend(f1_row(index, "team_1_outfield_visual_context") for index in range(2100, 2180))
    rows.extend(f1_row(index, "team_2_outfield_visual_context") for index in range(2180, 2260))
    rows.extend(f1_row(index, "official_referee_visual_context") for index in range(2260, 2290))
    rows.extend(f1_row(index, "assistant_or_line_official_visual_context") for index in range(2290, 2310))
    rows.extend(f1_row(index, "off_pitch_context_person_visual_context") for index in range(2310, 2330))
    rows.append(
        f1_row(
            3000,
            "team_1_goalkeeper_visual_context",
            conflict_flags=["official_context_goalkeeper_like_conflict"],
        )
    )
    rows.append(
        f1_row(
            3001,
            "team_1_outfield_visual_context",
            conflict_flags=["non_outfield_colour_final_player_or_goalkeeper_context"],
        )
    )
    conflict_rows = [
        {
            "visible_person_base_id": row["visible_person_base_id"],
            "frame_sequence": row["frame_sequence"],
            "step1f1_conflict_flags": row["step1f1_conflict_flags"],
        }
        for row in rows
        if row["step1f1_conflict_flags"]
    ]
    return {"rows": rows}, {"rows": conflict_rows}


def test_selection_target_hard_cap_mandatory_and_unknown_sampling() -> None:
    f1_payload, conflict_payload = synthetic_payloads()
    rows, summary = build_review_candidate_rows(
        f1_payload,
        conflict_payload,
        {"f1_safe_for_f2_human_review_candidate": True},
        labels_payload=labels_payload(),
    )
    assert 80 <= len(rows) <= 120
    assert len(rows) <= HARD_MAX_CANDIDATES
    assert summary["bucket_counts"]["severe_fusion_conflict_all"] == 2
    assert summary["bucket_counts"]["gold_proxy_problem_rows"] == 1
    assert summary["unknown_ambiguity_full_count"] >= 2010
    assert summary["unknown_ambiguity_sampled_count"] <= 20
    assert len({row["visible_person_base_id"] for row in rows}) == len(rows)


def test_selection_order_is_deterministic() -> None:
    f1_payload, conflict_payload = synthetic_payloads()
    first, _summary_a = build_review_candidate_rows(
        f1_payload,
        conflict_payload,
        {"f1_safe_for_f2_human_review_candidate": True},
        labels_payload=labels_payload(),
    )
    second, _summary_b = build_review_candidate_rows(
        f1_payload,
        conflict_payload,
        {"f1_safe_for_f2_human_review_candidate": True},
        labels_payload=labels_payload(),
    )
    assert [row["step1f2_review_candidate_id"] for row in first] == [
        row["step1f2_review_candidate_id"] for row in second
    ]
