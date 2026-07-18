from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).parents[1]
ROOT = REPO.parent
STAGE = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 2"
    / "M5_5F0C_SEED_CURATION_DEDUPLICATION_AND_ONE_FRAME_DROPOUT_REPAIR_v1"
)
PACKAGE = STAGE / "08_VALIDATED_LEVEL2_CONTINUITY_REVIEW_PACKAGE"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_completed_review_is_normalized_and_temporally_deduplicated() -> None:
    summary = read_json(STAGE / "02_TEMPORAL_EVENT_DEDUPLICATION" / "deduplicated_review_summary.json")
    assert summary["unique_dropout_frames"] == [32, 65]
    assert summary["duplicates"] == ["m5_5f0b_level2_case_005"]
    assert any(
        cluster["case_ids"] == ["m5_5f0b_level2_case_004", "m5_5f0b_level2_case_005"] for cluster in summary["clusters"]
    )
    assert all(row["elapsed_active_seconds"] == 0 for row in summary["rows"])


def test_failure_windows_have_fresh_cuda_supply_and_shared_gate_diagnosis() -> None:
    diagnostics = read_json(STAGE / "03_FRAME32_AND_FRAME65_DROPOUT_ROOT_CAUSE" / "dropout_diagnostics.json")
    assert {row["event_frame"] for row in diagnostics} == {32, 65}
    assert all(row["fresh_failure_window_detector"] for row in diagnostics)
    assert all(row["fresh_detector_device"] == "cuda:0" for row in diagnostics)
    assert all(row["fresh_detector_row_count_at_failure"] > 0 for row in diagnostics)
    assert all("GLOBAL_FRAME_LEVEL_ABSTENTION" in row["root_cause_classification"] for row in diagnostics)


def test_machine_preflight_selects_six_unique_level2_cases() -> None:
    summary = read_json(STAGE / "06_MACHINE_ONLY_LEVEL2_PREFLIGHT" / "level2_preflight_summary.json")
    assert summary["selected_count"] == 6
    assert summary["all_selected_pass"] is True
    assert summary["zero_bad_seeds"] is True
    assert summary["zero_bad_rois"] is True
    assert summary["zero_duplicate_temporal_events"] is True
    assert summary["zero_impossible_jumps"] is True


def test_completed_package_preserves_six_reviewed_decisions_and_thirteen_frame_cases() -> None:
    manifest = read_json(PACKAGE / "reviewer_manifest.json")
    decisions = read_json(PACKAGE / "decisions" / "review_decisions.json")
    assert manifest["review_id"] == "m5_5f0c_validated_level2_continuity_review_v1"
    assert len(manifest["cases"]) == 6
    assert all(case["visible_metadata"]["benchmark_level"] == 2 for case in manifest["cases"])
    assert all(len(case["visible_metadata"]["frame_records"]) == 13 for case in manifest["cases"])
    assert decisions["decisions"] == {
        "m5_5f0c_level2_candidate_002": "PASS",
        "m5_5f0c_level2_candidate_003": "B_SWITCH",
        "m5_5f0c_level2_candidate_004": "A_SWITCH",
        "m5_5f0c_level2_candidate_005": "A_SWITCH",
        "m5_5f0c_level2_candidate_006": "PASS",
        "m5_5f0c_level2_candidate_007": "PASS",
    }
    assert set(decisions["structured_reviews"]) == set(decisions["decisions"])
    assert "8798" in (PACKAGE / "launch_review.ps1").read_text(encoding="utf-8")


def test_tracker_rows_are_exact_source_bound_and_renderer_aware() -> None:
    rows_path = STAGE / "04_DETECTION_TO_STRAND_ASSIGNMENT_REPAIR" / "repaired_tracker_state_rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    observed = [row for row in rows if row["rendered_observed"]]
    assert observed
    assert all(row["source_observation_id"] for row in observed)
    assert all(row["render_style"] == "solid" for row in observed)
    assert len({row["benchmark_case_id"] for row in rows}) == 6
    assert all(not row["missing_reason"] for row in observed)


def test_active_time_telemetry_is_visibility_aware() -> None:
    app = (REPO / "src" / "football_intelligence" / "review_chassis" / "static" / "app.js").read_text(encoding="utf-8")
    assert "function activeTimeNow()" in app
    assert "document.visibilityState" in app
    assert "elapsed_active_seconds" in app
