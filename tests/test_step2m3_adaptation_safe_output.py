from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import football_intelligence.step2_visual_continuity.adaptation_safe_output as m3  # noqa: E402
from football_intelligence.step2_visual_continuity.adaptation_safe_output import (  # noqa: E402
    MAX_M3_GROUP_SPAN_FRAMES,
    build_m3_groups,
    build_step2m3_validation_outputs,
    classify_m3_edge,
    m2_bucket_review_results,
    quarantined_edge_row,
    step2m3_output_paths,
)
from football_intelligence.step2_visual_continuity.match_local_adaptation import M2_CURRENT_OVERLAY_VERSION  # noqa: E402
from football_intelligence.step2_visual_continuity.io import (  # noqa: E402
    STEP2M1_OUTPUT_DIR,
    STEP2M2_OUTPUT_DIR,
    STEP2M3_OUTPUT_DIR,
)
from football_intelligence.step2_visual_continuity.schema import (  # noqa: E402
    ACCEPT_DECISION,
    REJECT_DECISION,
    UNSURE_DECISION,
    VISUAL_ONLY_WARNING,
    assert_no_forbidden_keys,
    forbidden_keys_present,
    rows_from_payload,
)


def edge(index: int, *, bucket: str = "safe_auto_accept_candidate", state: str = "auto_accept_candidate") -> dict:
    return {
        "continuity_edge_id": f"edge_{index}",
        "source_visible_person_base_id": f"source_{index}",
        "target_visible_person_base_id": f"target_{index}",
        "source_frame_sequence": index,
        "target_frame_sequence": index + 1,
        "frame_gap": 1,
        "source_review_bucket": bucket,
        "review_bucket": bucket,
        "original_proposed_edge_state": state,
        "adapted_proposed_edge_state": state,
        "original_edge_score_sandbox": 0.8,
        "adapted_edge_score_sandbox": 0.86,
        "adapted_edge_state_changed": False,
        "adaptation_reasons": ["synthetic"],
        "learned_from_m1_m1r_evidence": True,
        "edge_feature_summary": {
            "bbox_iou": 0.68,
            "footpoint_delta_px": 8.0,
            "bbox_area_ratio": 1.0,
        },
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }


def decision(edge_id: str, decision_value: str, *, bucket: str = "safe_auto_accept_audit") -> dict:
    return {
        "continuity_edge_id": edge_id,
        "step2m2_target_review_bucket": bucket,
        "human_review_decision": decision_value,
        "human_confirmed": True,
        "review_decisions_collected_with_overlay_version": M2_CURRENT_OVERLAY_VERSION,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_step2m3_output_paths_are_isolated() -> None:
    m3_root = STEP2M3_OUTPUT_DIR.resolve()
    m1_root = STEP2M1_OUTPUT_DIR.resolve()
    m2_root = STEP2M2_OUTPUT_DIR.resolve()
    for path in step2m3_output_paths().values():
        resolved = path.resolve()
        assert resolved == m3_root or m3_root in resolved.parents
        assert resolved != m1_root
        assert resolved != m2_root
        assert m1_root not in resolved.parents
        assert m2_root not in resolved.parents


def test_m2_bucket_review_results_gate_counts() -> None:
    payload = {
        "rows": [
            *(decision(f"safe_{index}", ACCEPT_DECISION, bucket="safe_auto_accept_audit") for index in range(8)),
            *(decision(f"high_{index}", ACCEPT_DECISION, bucket="high_risk_adapted_accept") for index in range(8)),
            *(decision(f"merged_{index}", REJECT_DECISION, bucket="merged_ambiguous_sentinel") for index in range(8)),
            decision("role_accept", ACCEPT_DECISION, bucket="role_state_mismatch_split"),
            *(
                decision(f"role_reject_{index}", REJECT_DECISION, bucket="role_state_mismatch_split")
                for index in range(7)
            ),
        ]
    }
    results = m2_bucket_review_results(payload)
    assert results["safe_auto_accept_audit"]["accepted"] == 8
    assert results["high_risk_adapted_accept"]["accepted"] == 8
    assert results["merged_ambiguous_sentinel"]["rejected"] == 8
    assert results["role_state_mismatch_split"]["accepted"] == 1


def test_human_rejected_and_unsure_edges_are_quarantined() -> None:
    rejected = edge(1)
    state, _source, decision_value, reasons = classify_m3_edge(
        rejected,
        m1_decisions={},
        m1r_decisions={},
        m2_decisions={"edge_1": decision("edge_1", REJECT_DECISION)},
        m2_review_candidate_edges={"edge_1"},
    )
    assert state == "quarantined"
    assert decision_value == REJECT_DECISION
    assert "human_rejected" in reasons
    unsure = edge(2)
    state, _source, decision_value, reasons = classify_m3_edge(
        unsure,
        m1_decisions={},
        m1r_decisions={},
        m2_decisions={"edge_2": decision("edge_2", UNSURE_DECISION)},
        m2_review_candidate_edges={"edge_2"},
    )
    assert state == "quarantined"
    assert decision_value == UNSURE_DECISION
    assert "human_unsure" in reasons


def test_merged_and_role_state_policy_is_conservative() -> None:
    merged = edge(3, bucket="merged_or_ambiguous")
    state, _source, _decision, reasons = classify_m3_edge(
        merged,
        m1_decisions={},
        m1r_decisions={},
        m2_decisions={},
        m2_review_candidate_edges=set(),
    )
    assert state == "quarantined"
    assert "merged_or_ambiguous_policy" in reasons
    role = edge(4, bucket="role_state_mismatch")
    role["edge_feature_summary"]["bbox_iou"] = 0.1
    state, _source, _decision, reasons = classify_m3_edge(
        role,
        m1_decisions={},
        m1r_decisions={},
        m2_decisions={},
        m2_review_candidate_edges=set(),
    )
    assert state == "quarantined"
    assert "role_state_mismatch_policy" in reasons


def test_accepted_and_quarantine_rows_have_guardrails_and_reason_codes() -> None:
    accepted = m3.accepted_edge_row(
        edge(5),
        decision_source="step2m2",
        human_decision=ACCEPT_DECISION,
        reason="human_review_accepted",
    )
    assert_no_forbidden_keys(accepted)
    assert accepted["production_ready"] is False
    assert accepted["no_auto_promotion"] is True
    assert accepted["human_approved"] is False
    quarantined = quarantined_edge_row(
        edge(6),
        decision_source="step2m2",
        human_decision=REJECT_DECISION,
        reasons=["human_rejected"],
    )
    assert_no_forbidden_keys(quarantined)
    assert quarantined["m3_quarantine_reasons"] == ["human_rejected"]


def test_m3_group_span_cap_and_group_guardrails(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(m3, "STEP2M3_GROUP_ROWS_PATH", tmp_path / "m3_groups.json")
    monkeypatch.setattr(m3, "STEP2M3_GROUP_SAMPLE_PATH", tmp_path / "m3_group_sample.json")
    monkeypatch.setattr(m3, "STEP2M3_GROUP_SUMMARY_PATH", tmp_path / "m3_group_summary.json")
    accepted_rows = [
        m3.accepted_edge_row(
            edge(index),
            decision_source="",
            human_decision="",
            reason="m2_adapted_auto_accept_safety_filters_passed",
        )
        for index in [0, 1, 2, 40, 41]
    ]
    group_payload, _sample, summary = build_m3_groups(accepted_rows)
    assert summary["groups_over_cap_count"] == 0
    assert summary["max_group_span_frames_observed"] <= MAX_M3_GROUP_SPAN_FRAMES
    for row in rows_from_payload(group_payload):
        assert row["group_not_identity"] is True
        assert row["group_not_player_slot"] is True
        assert row["group_not_goalkeeper_slot"] is True
        assert row["do_not_use_for_metrics"] is True
        assert row["production_ready"] is False


def test_step2m3_validation_freeze_manifest_guardrails(monkeypatch, tmp_path: Path) -> None:
    paths = {
        "accepted": tmp_path / "accepted_summary.json",
        "quarantine": tmp_path / "quarantine_summary.json",
        "group": tmp_path / "group_summary.json",
        "handoff": tmp_path / "handoff.json",
        "m2_progress": tmp_path / "m2_progress.json",
        "m2_validation": tmp_path / "m2_validation.json",
        "validation": tmp_path / "m3_validation.json",
        "audit": tmp_path / "m3_audit.json",
        "issues": tmp_path / "m3_issues.json",
        "freeze": tmp_path / "m3_freeze.json",
    }
    for name, path in [
        ("STEP2M3_ACCEPTED_EDGE_SUMMARY_PATH", paths["accepted"]),
        ("STEP2M3_QUARANTINE_SUMMARY_PATH", paths["quarantine"]),
        ("STEP2M3_GROUP_SUMMARY_PATH", paths["group"]),
        ("STEP2M3_HANDOFF_MANIFEST_PATH", paths["handoff"]),
        ("STEP2M2_REVIEW_PROGRESS_SUMMARY_PATH", paths["m2_progress"]),
        ("STEP2M2_VALIDATION_SUMMARY_PATH", paths["m2_validation"]),
        ("STEP2M3_VALIDATION_SUMMARY_PATH", paths["validation"]),
        ("STEP2M3_SAFETY_GUARDRAIL_AUDIT_PATH", paths["audit"]),
        ("STEP2M3_ISSUE_REGISTER_PATH", paths["issues"]),
        ("STEP2M3_FREEZE_CANDIDATE_MANIFEST_PATH", paths["freeze"]),
    ]:
        monkeypatch.setattr(m3, name, path)
    m2_rows = [
        *(decision(f"safe_{index}", ACCEPT_DECISION, bucket="safe_auto_accept_audit") for index in range(8)),
        *(decision(f"high_{index}", ACCEPT_DECISION, bucket="high_risk_adapted_accept") for index in range(8)),
        *(decision(f"merged_{index}", REJECT_DECISION, bucket="merged_ambiguous_sentinel") for index in range(8)),
        decision("role_accept", ACCEPT_DECISION, bucket="role_state_mismatch_split"),
        *(decision(f"role_reject_{index}", REJECT_DECISION, bucket="role_state_mismatch_split") for index in range(7)),
    ]
    monkeypatch.setattr(m3, "read_m2_reviewed_decisions", lambda: {"rows": m2_rows})
    write_json(
        paths["accepted"],
        {
            "accepted_edge_count": 10,
            "accepted_human_decision_counts": {},
            "production_ready": False,
            "no_auto_promotion": True,
        },
    )
    write_json(
        paths["quarantine"],
        {
            "quarantined_edge_count": 20,
            "human_rejected_edges_quarantined": True,
            "human_unsure_edges_quarantined": True,
            "production_ready": False,
            "no_auto_promotion": True,
        },
    )
    write_json(
        paths["group"],
        {
            "adaptation_safe_group_count": 3,
            "groups_over_cap_count": 0,
            "max_group_span_frames_observed": 3,
            "max_group_span_seconds_observed": 0.3,
        },
    )
    write_json(paths["handoff"], {"production_ready": False, "no_auto_promotion": True})
    write_json(
        paths["m2_progress"],
        {
            "reviewed_candidates": 40,
            "targeted_review_completed": True,
            "review_decisions_overlay_version_matches_current": True,
        },
    )
    write_json(paths["m2_validation"], {"forbidden_keys_present": []})
    validation, _audit, issue_register, freeze = build_step2m3_validation_outputs()
    assert issue_register["blocking_issue_count"] == 0
    assert validation["step2m3_freeze_candidate_created"] is True
    assert freeze["step2m3_freeze_candidate_created"] is True
    assert freeze["production_ready"] is False
    assert freeze["no_auto_promotion"] is True
    assert freeze["human_approved"] is False
    assert forbidden_keys_present(freeze) == []
