from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from build_m5_5f0_stable_local_strand import (  # noqa: E402
    MODEL_BYTES,
    MODEL_SHA256,
    OUTCOMES,
    REVIEW_ID,
    REVIEW_PORT,
    REVIEW_ROOT,
    REVIEW_SESSION,
    SAFETY,
    STAGE_ROOT,
    audit_review_duration,
    validate_completed_review,
)
from football_intelligence.review_chassis.manifest import load_manifest  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_completed_m5_5e3_review_is_validated_without_rewriting_it() -> None:
    validation = validate_completed_review()
    assert validation["passed"] is True
    assert validation["decision_counts"] == {
        "STRAND_EVIDENCE_INCONSISTENT": 15,
        "ORDINARY_CROSSING_INDEPENDENT_OBSERVATIONS_REMAIN": 2,
        "GENUINE_MERGED_OBSERVATION_INTERVAL": 1,
    }
    assert validation["decision_event_count"] == 18
    assert validation["completion_event_count"] == 1


def test_zero_duration_is_reported_as_telemetry_defect() -> None:
    telemetry = audit_review_duration()
    assert telemetry["zero_duration_detected"] is True
    assert telemetry["interpretation"].startswith("zero review duration")


def test_failure_taxonomy_has_all_fifteen_inconsistent_cases() -> None:
    summary = read_json(STAGE_ROOT / "02_STRAND_FAILURE_TAXONOMY_AND_REPRODUCTION" / "failure_mode_summary.json")
    assert summary["human_inconsistency_count"] == 15
    assert summary["prior_machine_silent_switch_count"] == 0
    assert summary["prior_machine_impossible_jump_count"] == 0
    assert summary["explanation"]


def test_primary_benchmark_is_non_occlusion_and_supply_limited_without_padding() -> None:
    summary = read_json(STAGE_ROOT / "05_EASY_TO_HARD_BENCHMARK_CURATION" / "level_summary.json")
    assert summary["human_answers_used"] is False
    assert summary["primary_benchmark_excludes_genuine_occlusion"] is True
    assert summary["selected_case_count"] == 9
    assert summary["level_counts"] == {"1": 3, "2": 3, "3": 3}


def test_machine_tracker_gates_and_state_priority_are_explicit() -> None:
    gates = read_json(STAGE_ROOT / "06_MACHINE_ONLY_CONTINUITY_GATES" / "acceptance_checklist.json")
    assert gates["impossible_jumps"] == 0
    assert gates["double_assignments"] == 0
    assert gates["forced_assignments_below_margin"] == 0
    tracker = read_json(STAGE_ROOT / "04_ABSTENTION_FIRST_STRAND_TRACKER" / "tracker_summary.json")
    assert tracker["priority"] == [
        "CORRECT_CONTINUATION",
        "EXPLICIT_AMBIGUITY",
        "TEMPORARY_MISSING",
        "TERMINATION",
        "WRONG_CONTINUATION",
    ]
    assert tracker["appearance_conflict_gated"] is True


def test_detector_provenance_is_verified_but_runtime_limited() -> None:
    detector = read_json(STAGE_ROOT / "03_LOCAL_DETECTION_SUPPLY_REBUILD" / "detection_supply_summary.json")
    assert detector["checkpoint_sha256"] == MODEL_SHA256
    assert detector["checkpoint_bytes"] == MODEL_BYTES
    assert detector["global_defaults_changed"] is False
    assert detector["local_sandbox_only"] is True
    assert detector["status"] == "runtime_limited"
    assert detector["row_count"] == 0


def test_fresh_review_package_is_empty_and_stable_mode_is_configured() -> None:
    manifest = load_manifest(REVIEW_ROOT / "reviewer_manifest.json")
    assert 1 <= len(manifest.cases) <= 20
    assert all(case.task_type == "stable_local_strand_continuity_review" for case in manifest.cases)
    assert all(case.safety_payload == SAFETY for case in manifest.cases)
    decisions = read_json(REVIEW_ROOT / "decisions" / "review_decisions.json")
    assert decisions["decisions"] == {}
    assert decisions["reviewer_session_id"] == REVIEW_SESSION
    ui_config = read_json(REVIEW_ROOT / "ui_config.json")
    assert ui_config["presentation_mode"] == "stable_local_strand_continuity"
    assert ui_config["question_contract"]["outcomes"] == list(OUTCOMES)
    assert ui_config["question_contract"]["notes_optional_for_structured_outcomes"] is True


def test_package_and_launcher_are_fresh_and_port_is_correct() -> None:
    assert REVIEW_ID == "m5_5f0_stable_local_strand_continuity_review_v1"
    assert REVIEW_PORT == 8795
    assert (REVIEW_ROOT / "launch_review.ps1").read_text(encoding="utf-8").count("uv).Source run fi-pipeline") == 1
    event_log = REVIEW_ROOT / "decisions" / "review_decision_events.jsonl"
    assert event_log.exists() and event_log.read_text(encoding="utf-8") == ""


def test_flat_review_pack_is_bounded_and_excludes_sealed_inputs() -> None:
    pack = STAGE_ROOT / "11_REVIEW_PACK_FOR_CHATGPT"
    files = list(pack.iterdir())
    assert len(files) <= 20
    assert all(path.is_file() for path in files)
    assert not any("sealed" in path.name.lower() or "answer" in path.name.lower() for path in files)
    assert (pack / "04_SOURCE_DIFF.patch").exists()
    assert (pack / "17_FAILURE_EXAMPLES.jpg").exists()
    assert (pack / "18_BENCHMARK_REVIEW_UI.png").exists()
