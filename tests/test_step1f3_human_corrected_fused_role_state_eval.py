from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_human_correction_eval import (  # noqa: E402
    build_f3_eval_summary,
)
from test_step1f3_human_corrected_fused_role_state import synthetic_payloads  # noqa: E402
from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_human_corrections import (  # noqa: E402
    build_human_corrected_fused_role_state_payloads,
)


def labels_payload() -> dict:
    return {
        "frames": [
            {
                "frame_sequence": 1,
                "frame_id": "f1",
                "timestamp_seconds": 0.1,
                "labels_complete": True,
                "persons": [
                    {
                        "gold_person_id": "g1",
                        "candidate_row_id": "v1",
                        "visible_person_type_gold": "team_1_player",
                        "occlusion_state_gold": "observed_clear",
                        "bbox": {"x1": 11.0, "y1": 20.0, "x2": 41.0, "y2": 90.0},
                    }
                ],
            },
            {
                "frame_sequence": 2,
                "frame_id": "f2",
                "timestamp_seconds": 0.2,
                "labels_complete": True,
                "persons": [
                    {
                        "gold_person_id": "g2",
                        "candidate_row_id": "v2",
                        "visible_person_type_gold": "team_2_player",
                        "occlusion_state_gold": "observed_clear",
                        "bbox": {"x1": 12.0, "y1": 20.0, "x2": 42.0, "y2": 90.0},
                    }
                ],
            },
        ]
    }


def test_f3_eval_emits_before_after_gold_proxy_and_requires_safety_gate() -> None:
    f1_payload, candidate_payload, reviewed_payload = synthetic_payloads()
    corrected, audit = build_human_corrected_fused_role_state_payloads(f1_payload, candidate_payload, reviewed_payload)
    corrected["rows"] = corrected["rows"] * 1
    f1_eval_summary = {
        "missed_goalkeeper_proxy_count": 0,
        "outfield_team_proxy_counts": {},
        "official_context_proxy_counts": {},
    }
    f2_progress = {
        "f2_approve_f1_for_f3_human_correction_candidate": True,
        "reviewed_candidates": 4,
    }
    f2_decision_summary = {"accepted_count": 2, "corrected_count": 1, "unsure_count": 1}
    summary = build_f3_eval_summary(
        f1_payload,
        corrected,
        audit,
        f1_eval_summary,
        f2_progress,
        f2_decision_summary,
        labels_payload=labels_payload(),
    )
    assert "gold_proxy_distribution_before" in summary
    assert "gold_proxy_distribution_after" in summary
    assert "f1_missed_goalkeeper_proxy_count" in summary
    assert "f3_missed_goalkeeper_proxy_count" in summary
    assert "f1_outfield_proxy_match_mismatch_counts" in summary
    assert "f3_outfield_proxy_match_mismatch_counts" in summary
    assert "f1_official_context_proxy_match_miss_counts" in summary
    assert "f3_official_context_proxy_match_miss_counts" in summary
    assert summary["f3_safe_for_step1g_validation_candidate"] is False
    assert "f1_or_f3_row_count_not_10418" in summary["f3_safety_missing_reasons"]


def test_f3_eval_requires_f2_approval_gate() -> None:
    f1_payload, candidate_payload, reviewed_payload = synthetic_payloads()
    corrected, audit = build_human_corrected_fused_role_state_payloads(f1_payload, candidate_payload, reviewed_payload)
    summary = build_f3_eval_summary(
        f1_payload,
        corrected,
        audit,
        {},
        {"f2_approve_f1_for_f3_human_correction_candidate": False, "reviewed_candidates": 4},
        {},
        labels_payload=labels_payload(),
    )
    assert summary["f3_safe_for_step1g_validation_candidate"] is False
    assert "f2_approval_gate_not_true" in summary["f3_safety_missing_reasons"]
