from __future__ import annotations

import json
import sys
import threading
import urllib.request
from pathlib import Path
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import football_intelligence.step2_visual_continuity.match_local_adaptation as mla  # noqa: E402
from football_intelligence.step2_visual_continuity.io import (  # noqa: E402
    STEP2M1_OUTPUT_DIR,
    STEP2M1_REVIEWED_DECISIONS_PATH,
    STEP2M1R_REVIEWED_DECISIONS_PATH,
    STEP2M2_OUTPUT_DIR,
)
from football_intelligence.step2_visual_continuity.match_local_adaptation import (  # noqa: E402
    M2_CURRENT_OVERLAY_VERSION,
    adapt_edge_row,
    build_step2m2_validation_outputs,
    build_m2_review_queue_from_pools,
    build_match_local_adaptation_profile,
    build_reviewed_decision_training_payloads,
    bucket_decision_rates,
    save_m2_review_decision,
    step2m2_output_paths,
)
from football_intelligence.step2_visual_continuity.schema import (  # noqa: E402
    ACCEPT_DECISION,
    REJECT_DECISION,
    VISUAL_ONLY_WARNING,
    assert_no_forbidden_keys,
    forbidden_keys_present,
    rows_from_payload,
)


def edge_row(
    index: int,
    *,
    bucket: str = "safe_auto_accept_candidate",
    state: str = "auto_accept_candidate",
    score: float = 0.76,
) -> dict:
    return {
        "continuity_edge_id": f"edge_{index}",
        "source_visible_person_base_id": f"source_{index}",
        "target_visible_person_base_id": f"target_{index}",
        "source_frame_sequence": index,
        "target_frame_sequence": index + 1,
        "frame_gap": 1,
        "review_bucket": bucket,
        "proposed_edge_state": state,
        "edge_score_sandbox": score,
        "uncertainty_score": 0.31,
        "uncertainty_reasons": ["synthetic_uncertainty"],
        "edge_feature_summary": {
            "edge_score_sandbox": score,
            "uncertainty_score": 0.31,
            "bbox_iou": 0.58,
            "bbox_center_delta_px": 8.0,
            "footpoint_delta_px": 7.0,
            "bbox_area_ratio": 1.0,
            "aspect_ratio_change": 0.05,
            "role_state_compatibility": 0.75,
            "visual_team_context_compatibility": 0.7,
            "step1_c2c_d1c_e1c_compatibility": 0.72,
            "crop_quality_penalty": 0.0,
            "warning_conflict_flag_penalty": 0.0,
            "frame_gap_penalty": 0.0,
            "uncertainty_reasons": ["synthetic_uncertainty"],
        },
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }


def reviewed_decision(edge: dict, decision: str, *, m1r: bool = False, include_m1r_id: bool = True) -> dict:
    row = {
        "continuity_edge_id": edge["continuity_edge_id"],
        "source_visible_person_base_id": edge["source_visible_person_base_id"],
        "target_visible_person_base_id": edge["target_visible_person_base_id"],
        "source_frame_sequence": edge["source_frame_sequence"],
        "target_frame_sequence": edge["target_frame_sequence"],
        "review_bucket": edge["review_bucket"],
        "human_review_decision": decision,
        "human_confirmed": True,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }
    if m1r and include_m1r_id:
        row["step2m1r_review_candidate_id"] = edge["step2m1r_review_candidate_id"]
    elif not m1r:
        row["step2m1_review_candidate_id"] = edge["step2m1_review_candidate_id"]
    return row


def training_payload() -> dict:
    rows = [
        {
            **edge_row(1, bucket="safe_auto_accept_audit"),
            "human_review_decision": ACCEPT_DECISION,
            "accepted_target_label": True,
            "rejected_target_label": False,
            "unsure_target_label": False,
            "target_label": "accepted",
        },
        {
            **edge_row(2, bucket="merged_or_ambiguous", state="needs_review_candidate", score=0.51),
            "human_review_decision": REJECT_DECISION,
            "accepted_target_label": False,
            "rejected_target_label": True,
            "unsure_target_label": False,
            "target_label": "rejected",
        },
    ]
    return {"rows": rows}


def m2_candidate(index: int) -> dict:
    return {
        **edge_row(index, bucket="safe_auto_accept_candidate", score=0.82),
        "step2m2_review_candidate_id": f"step2m2_review_{index:03d}",
        "step2m1r_review_candidate_id": f"step2m2_review_{index:03d}",
        "step2m2_target_review_bucket": "safe_auto_accept_audit",
        "source_review_bucket": "safe_auto_accept_candidate",
        "original_proposed_edge_state": "auto_accept_candidate",
        "adapted_proposed_edge_state": "auto_accept_candidate",
        "original_edge_score_sandbox": 0.82,
        "adapted_edge_score_sandbox": 0.85,
        "adapted_edge_state_changed": False,
        "adaptation_reasons": ["safe_auto_accept_audit_match_local_reliability_delta"],
        "learned_from_m1_m1r_evidence": True,
        "match_local_only": True,
        "safe_to_apply_globally": False,
        "requires_future_match_validation": True,
        "current_overlay_version": M2_CURRENT_OVERLAY_VERSION,
    }


def write_json_for_test(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def patch_m2_review_paths(monkeypatch, tmp_path: Path) -> dict[str, Path]:
    paths = {
        "candidates": tmp_path / "step2m2_targeted_review_candidate_rows.json",
        "reviewed": tmp_path / "step2m2_reviewed_visual_continuity_decisions.json",
        "progress": tmp_path / "step2m2_review_progress_summary.json",
        "decision": tmp_path / "step2m2_review_decision_summary.json",
        "validation": tmp_path / "step2m2_validation_summary.json",
        "audit": tmp_path / "step2m2_safety_guardrail_audit.json",
        "issues": tmp_path / "step2m2_issue_register.json",
        "freeze": tmp_path / "step2m2_freeze_candidate_manifest.json",
        "m1r_manifest": tmp_path / "step2m1r_manifest.json",
    }
    for name, path in [
        ("STEP2M2_TARGETED_REVIEW_CANDIDATE_ROWS_PATH", paths["candidates"]),
        ("STEP2M2_REVIEWED_DECISIONS_PATH", paths["reviewed"]),
        ("STEP2M2_REVIEW_PROGRESS_SUMMARY_PATH", paths["progress"]),
        ("STEP2M2_REVIEW_DECISION_SUMMARY_PATH", paths["decision"]),
        ("STEP2M2_VALIDATION_SUMMARY_PATH", paths["validation"]),
        ("STEP2M2_SAFETY_GUARDRAIL_AUDIT_PATH", paths["audit"]),
        ("STEP2M2_ISSUE_REGISTER_PATH", paths["issues"]),
        ("STEP2M2_FREEZE_CANDIDATE_MANIFEST_PATH", paths["freeze"]),
        ("STEP2M1R_ADAPTATION_SAFETY_MANIFEST_PATH", paths["m1r_manifest"]),
    ]:
        monkeypatch.setattr(mla, name, path)
    write_json_for_test(
        paths["m1r_manifest"],
        {
            "safe_for_step2m2_adaptation_candidate": True,
            "reviewed_candidates": 55,
            "review_decisions_overlay_version_matches_current": True,
            "burst_overlay_alignment_safe_for_review": True,
            "groups_over_cap_after": 0,
            "forbidden_keys_present": [],
            "production_ready": False,
            "no_auto_promotion": True,
            "human_approved": False,
        },
    )
    return paths


def test_step2m2_loads_m1_and_m1r_decisions_with_edge_fallback() -> None:
    m1_edge = {**edge_row(1, bucket="high_uncertainty_low_margin"), "step2m1_review_candidate_id": "m1_edge_1"}
    m1r_edge = {**edge_row(2, bucket="safe_auto_accept_audit"), "step2m1r_review_candidate_id": "m1r_edge_2"}
    training, summary = build_reviewed_decision_training_payloads(
        m1_reviewed_payload={"rows": [reviewed_decision(m1_edge, REJECT_DECISION)]},
        m1_candidate_payload={"rows": [m1_edge]},
        m1r_reviewed_payload={
            "decisions": [reviewed_decision(m1r_edge, ACCEPT_DECISION, m1r=True, include_m1r_id=False)]
        },
        m1r_candidate_payload={"rows": [m1r_edge]},
    )
    assert summary["m1_decisions_loaded"] == 1
    assert summary["m1r_decisions_loaded"] == 1
    assert summary["training_decision_count"] == 2
    assert summary["accepted_count"] == 1
    assert summary["rejected_count"] == 1
    assert rows_from_payload(training)[1]["step2m1r_review_candidate_id"] == "m1r_edge_2"


def test_step2m2_bucket_decision_rate_calculation() -> None:
    rates = bucket_decision_rates(rows_from_payload(training_payload()))
    assert rates["safe_auto_accept_audit"]["acceptance_rate"] == 1.0
    assert rates["merged_or_ambiguous"]["rejection_rate"] == 1.0


def test_step2m2_profile_is_match_local_only_and_does_not_change_global_thresholds() -> None:
    profile = build_match_local_adaptation_profile(training_payload())
    assert profile["match_local_only"] is True
    assert profile["safe_to_apply_globally"] is False
    assert profile["requires_future_match_validation"] is True
    assert profile["global_thresholds_changed"] is False
    assert profile["production_ready"] is False
    assert profile["no_auto_promotion"] is True
    assert forbidden_keys_present(profile) == []


def test_step2m2_adapted_edge_row_contains_original_and_adapted_scores() -> None:
    profile = build_match_local_adaptation_profile(training_payload())
    row = adapt_edge_row(
        edge_row(3, bucket="team_colour_ambiguity", state="needs_review_candidate", score=0.66),
        profile,
    )
    assert row["original_edge_score_sandbox"] == 0.66
    assert row["adapted_edge_score_sandbox"] >= row["original_edge_score_sandbox"]
    assert "original_proposed_edge_state" in row
    assert "adapted_proposed_edge_state" in row
    assert row["production_ready"] is False
    assert row["no_auto_promotion"] is True
    assert forbidden_keys_present(row) == []


def test_step2m2_review_queue_hard_max_and_burst_overlay_reuse_flag() -> None:
    pools = {
        "safe_auto_accept_audit": [
            edge_row(index, bucket="safe_auto_accept_candidate", score=0.82) for index in range(10)
        ],
        "high_risk_adapted_accept": [
            edge_row(20 + index, bucket="team_colour_ambiguity", score=0.81) for index in range(10)
        ],
        "changed_state_candidate": [
            {
                **edge_row(40 + index, bucket="team_colour_ambiguity", state="auto_reject_candidate", score=0.71),
                "original_proposed_edge_state": "auto_reject_candidate",
                "adapted_proposed_edge_state": "needs_review_candidate",
                "adapted_edge_state_changed": True,
                "adapted_edge_score_sandbox": 0.71,
                "original_edge_score_sandbox": 0.61,
            }
            for index in range(10)
        ],
        "merged_ambiguous_sentinel": [
            {
                **edge_row(60 + index, bucket="merged_or_ambiguous", state="needs_review_candidate", score=0.51),
                "original_proposed_edge_state": "needs_review_candidate",
                "adapted_proposed_edge_state": "needs_review_candidate",
                "adapted_edge_state_changed": False,
                "adapted_edge_score_sandbox": 0.51,
                "original_edge_score_sandbox": 0.51,
            }
            for index in range(10)
        ],
        "role_state_mismatch_split": [
            {
                **edge_row(80 + index, bucket="role_state_mismatch", state="needs_review_candidate", score=0.52),
                "original_proposed_edge_state": "needs_review_candidate",
                "adapted_proposed_edge_state": "needs_review_candidate",
                "adapted_edge_state_changed": False,
                "adapted_edge_score_sandbox": 0.52,
                "original_edge_score_sandbox": 0.52,
            }
            for index in range(10)
        ],
    }
    queue = build_m2_review_queue_from_pools(pools)
    assert len(rows_from_payload(queue)) <= 60
    assert queue["summary"]["safe_auto_accept_audit_rows"] == 8
    assert queue["summary"]["m2_reuses_m1r_burst_overlay_renderer"] is True
    assert queue["summary"]["current_overlay_version"] == M2_CURRENT_OVERLAY_VERSION
    assert all(row["m2_reuses_m1r_burst_overlay_renderer"] is True for row in rows_from_payload(queue))


def test_step2m2_output_paths_are_isolated_from_step2m1_sandbox() -> None:
    m2_root = STEP2M2_OUTPUT_DIR.resolve()
    m1_root = STEP2M1_OUTPUT_DIR.resolve()
    for path in step2m2_output_paths().values():
        resolved = path.resolve()
        assert resolved == m2_root or m2_root in resolved.parents
        assert resolved != m1_root
        assert m1_root not in resolved.parents
    assert STEP2M1_REVIEWED_DECISIONS_PATH.resolve() != STEP2M1R_REVIEWED_DECISIONS_PATH.resolve()


def test_step2m2_payloads_have_no_identity_slot_metric_event_tactical_outputs() -> None:
    profile = build_match_local_adaptation_profile(training_payload())
    row = adapt_edge_row(edge_row(9), profile)
    queue = build_m2_review_queue_from_pools({"safe_auto_accept_audit": [row]})
    assert_no_forbidden_keys(profile)
    assert_no_forbidden_keys(row)
    assert_no_forbidden_keys(queue)
    assert profile["production_ready"] is False
    assert profile["no_auto_promotion"] is True
    assert queue["no_identity_tracking_performed"] is True
    assert queue["no_player_slots_assigned"] is True
    assert queue["no_goalkeeper_slots_assigned"] is True
    assert queue["no_metric_event_tactical_or_physical_performance_analysis"] is True


def test_step2m2_autosave_writes_schema_and_updates_duplicate(monkeypatch, tmp_path: Path) -> None:
    paths = patch_m2_review_paths(monkeypatch, tmp_path)
    review_payload = {"rows": [m2_candidate(1)], "summary": {"burst_overlay_alignment_safe_for_review": True}}
    write_json_for_test(paths["candidates"], review_payload)
    _decision, reviewed, progress = save_m2_review_decision(
        {
            "step2m2_review_candidate_id": "step2m2_review_001",
            "human_review_decision": "accept_short_window_visual_continuity_edge",
        }
    )
    assert reviewed["reviewed_decision_rows"] == 1
    assert progress["reviewed_candidates"] == 1
    _decision, reviewed, progress = save_m2_review_decision(
        {
            "step2m2_review_candidate_id": "step2m2_review_001",
            "human_review_decision": "reject_edge",
        }
    )
    row = reviewed["rows"][0]
    assert reviewed["reviewed_decision_rows"] == 1
    assert row["human_review_decision"] == "reject_edge"
    assert row["human_confirmed"] is True
    assert row["current_overlay_version"] == M2_CURRENT_OVERLAY_VERSION
    assert row["review_decisions_collected_with_overlay_version"] == M2_CURRENT_OVERLAY_VERSION
    assert row["match_local_only"] is True
    assert row["safe_to_apply_globally"] is False
    assert row["production_ready"] is False
    assert row["no_auto_promotion"] is True
    assert row["human_approved"] is False
    assert row["approve_any_identity_tracking"] is False
    assert row["approve_any_player_slot_use"] is False
    assert row["approve_any_goalkeeper_slot_use"] is False
    assert row["approve_any_metric_use"] is False
    assert row["approve_event_or_tactical_analysis"] is False
    assert row["approve_official_referee_exclusion"] is False
    assert row["approve_bad_detection_deletion"] is False
    assert forbidden_keys_present(reviewed) == []
    assert progress["rejected_count"] == 1


def test_step2m2_autosave_endpoint_writes_json(monkeypatch, tmp_path: Path) -> None:
    paths = patch_m2_review_paths(monkeypatch, tmp_path)
    write_json_for_test(paths["candidates"], {"rows": [m2_candidate(2)], "summary": {}})
    server = ThreadingHTTPServer(("127.0.0.1", 0), mla.Step2M2ReviewHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/api/step2m2/review-decision",
            data=json.dumps(
                {
                    "step2m2_review_candidate_id": "step2m2_review_002",
                    "human_review_decision": "unsure_needs_later_review",
                }
            ).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert body["success"] is True
    assert body["reviewed_count"] == 1
    reviewed = json.loads(paths["reviewed"].read_text(encoding="utf-8"))
    assert reviewed["rows"][0]["step2m2_review_candidate_id"] == "step2m2_review_002"


def test_step2m2_completed_review_sets_targeted_review_completed(monkeypatch, tmp_path: Path) -> None:
    paths = patch_m2_review_paths(monkeypatch, tmp_path)
    candidates = [m2_candidate(index) for index in range(40)]
    write_json_for_test(paths["candidates"], {"rows": candidates, "summary": {}})
    for candidate in candidates:
        save_m2_review_decision(
            {
                "step2m2_review_candidate_id": candidate["step2m2_review_candidate_id"],
                "human_review_decision": "accept_short_window_visual_continuity_edge",
            }
        )
    progress = json.loads(paths["progress"].read_text(encoding="utf-8"))
    assert progress["total_review_candidates"] == 40
    assert progress["reviewed_candidates"] == 40
    assert progress["accepted_count"] == 40
    assert progress["targeted_review_completed"] is True
    assert progress["review_decisions_overlay_version_matches_current"] is True


def test_step2m2_localstorage_only_progress_is_not_valid_persisted_review(monkeypatch, tmp_path: Path) -> None:
    paths = patch_m2_review_paths(monkeypatch, tmp_path)
    review_payload = {"rows": [m2_candidate(1)], "summary": {"safe_auto_accept_audit_rows": 1}}
    write_json_for_test(paths["progress"], {"reviewed_candidates": 1})
    validation, _audit, issue_register, freeze = build_step2m2_validation_outputs(
        training_summary={"training_decision_count": 1, "production_ready": False, "no_auto_promotion": True},
        profile={"production_ready": False, "no_auto_promotion": True},
        adapted_summary={"adapted_edge_rows": 1, "production_ready": False, "no_auto_promotion": True},
        review_payload=review_payload,
    )
    assert validation["reviewed_candidates"] == 0
    assert issue_register["blocking_issue_count"] >= 1
    assert freeze["step2m2_freeze_candidate_created"] is False
    codes = {row["issue_code"] for row in issue_register["rows"]}
    assert "step2m2_review_progress_claims_reviewed_but_decision_file_missing" in codes


def test_step2m2_outputs_do_not_overwrite_m1_or_m1r_artifacts() -> None:
    m2_paths = {path.resolve() for path in step2m2_output_paths().values()}
    assert STEP2M1_REVIEWED_DECISIONS_PATH.resolve() not in m2_paths
    assert STEP2M1R_REVIEWED_DECISIONS_PATH.resolve() not in m2_paths
    assert all(
        STEP2M2_OUTPUT_DIR.resolve() in path.parents or path == STEP2M2_OUTPUT_DIR.resolve()
        for path in m2_paths
    )
