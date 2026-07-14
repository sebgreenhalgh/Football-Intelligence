from __future__ import annotations

import json
from pathlib import Path

from football_intelligence.replay.occlusion_stateful_baseline import write_stateful_baseline_outputs


def _case(case_number: str, endpoint: str) -> dict[str, object]:
    return {
        "case_id": f"m5_4h1_cadence_matched_target_choice_case_{case_number}",
        "classification": "RULE_ABSTAINED",
        "endpoint_safe_group_id": endpoint,
        "frame_gap": 2,
        "human_decision": "target_a_continues_source",
        "human_selected_panel": "target_a",
        "random_control_status": False,
        "trajectory_safe_group_id": f"group_{case_number}",
    }


def _detail(case_number: str, endpoint: str, source_frame: int) -> dict[str, object]:
    return {
        "endpoint_safe_group_id": endpoint,
        "crossing_crowding_or_occlusion": True,
        "source_frame_sequence": source_frame,
        "target_frame_sequence": source_frame + 2,
        "source_bbox": {"x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 70.0},
        "target_options": [
            {
                "target_frame_sequence": source_frame + 2,
                "target_bbox": {"x1": 12.0, "y1": 12.0, "x2": 32.0, "y2": 72.0},
                "features": {"appearance_similarity": 0.8},
                "occlusion_or_crowding_evidence": True,
            },
            {
                "target_frame_sequence": source_frame + 2,
                "target_bbox": {"x1": 25.0, "y1": 10.0, "x2": 45.0, "y2": 70.0},
                "features": {"appearance_similarity": 0.9},
                "occlusion_or_crowding_evidence": True,
            },
        ],
    }


def _write_inputs(root: Path) -> None:
    primary_root = root / "continuity_v13" / "evaluation"
    primary_root.mkdir(parents=True)
    rows = [_case("008", "endpoint_008"), _case("010", "endpoint_010"), _case("013", "endpoint_013")]
    (primary_root / "corrected_primary_results.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
    (primary_root / "corrected_challenge_control_split.json").write_text("{}", encoding="utf-8")
    challenge_root = root / "continuity_v11" / "unseen_window"
    challenge_root.mkdir(parents=True)
    with (challenge_root / "challenge_candidate_rows.jsonl").open("w", encoding="utf-8") as handle:
        for number, endpoint, frame in [
            ("008", "endpoint_008", 100),
            ("010", "endpoint_010", 200),
            ("013", "endpoint_013", 300),
        ]:
            handle.write(json.dumps(_detail(number, endpoint, frame)) + "\n")


def test_stateful_baseline_escalates_crossings_without_forced_assignment(tmp_path: Path) -> None:
    historical = tmp_path / "historical"
    _write_inputs(historical)

    result = write_stateful_baseline_outputs(historical_stage_root=historical, output_root=tmp_path / "out")

    assert result["stateful_branch_status"] == "PASS_CROSSINGS_ESCALATED_WITHOUT_FORCED_SWAP"
    assert result["summary"]["wrong_forced_assignment_count"] == 0
    assert result["summary"]["review_escalation_count"] == 3
    assert (tmp_path / "out" / "HUMAN_REVIEW" / "reviewer_manifest.json").exists()
    assert (tmp_path / "out" / "candidate_generation_rows.jsonl").read_text(encoding="utf-8")
