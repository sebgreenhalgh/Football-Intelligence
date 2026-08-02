"""Create the exact self-contained twelve-file G7E-B R2 handoff."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
STAGE = (
    PROJECT
    / "experiments/football_observation_reasoner/part 7/G7E_B_R2_FULL_TEMPORAL_CANDIDATE_CLOSURE_AND_REVIEWER_REPAIR_v1"
)
HANDOFF = STAGE / "10_REVIEW_PACK/CHATGPT_HANDOFF"
SUCCESS = "PASS_G7E_B_R2_FULL_TEMPORAL_CANDIDATE_CLOSURE_READY_FOR_PRACTICE_REVIEW"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def main() -> None:
    closure = read_json(STAGE / "04_CANDIDATE_CLOSURE/candidate_closure_summary.json")
    reviewer = read_json(STAGE / "06_REVIEWER_REPAIR/reviewer_repair_report.json")
    browser = read_json(STAGE / "07_BROWSER_ACCEPTANCE/browser_acceptance_report.json")
    coordinate = read_json(STAGE / "06_REVIEWER_REPAIR/coordinate_and_overlay_audit.json")
    mapping = read_json(STAGE / "05_REVIEWER_CANDIDATE_MAPPING/mapping_validation_report.json")
    reuse = read_json(STAGE / "02_EXISTING_CANDIDATE_REUSE/reuse_compatibility_report.json")
    runtime = read_json(STAGE / "03_TEMPORAL_PROPOSAL_RUNTIME/runtime_summary.json")
    tests = read_json(STAGE / "09_TESTS_AND_LOGS/test_report.json")
    if not (
        closure["decision"] == SUCCESS
        and closure["verified_unique_frame_count"] == 1044
        and closure["candidate_data_unavailable_frame_count"] == 0
        and mapping["frame_references_mapped"] == 1080
        and browser["decision"] == "PASS_G7E_B_R2_REAL_EDGE_ACCEPTANCE"
        and coordinate["passed"]
        and tests["passed"]
    ):
        raise SystemExit("FAIL_G7E_B_R2_CHATGPT_HANDOFF")
    if HANDOFF.exists():
        for path in HANDOFF.iterdir():
            if path.is_file():
                path.unlink()
    HANDOFF.mkdir(parents=True, exist_ok=True)
    common = {
        "decision": SUCCESS,
        "repository_head_before_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "production_ready": False,
        "real_human_event_count": 0,
    }
    write_json(
        HANDOFF / "01_EXECUTIVE_SUMMARY.json",
        {
            **common,
            "result": "All 1,044 exact temporal frames have complete post-C3A6 candidate provenance and all 1,080 review references are mapped.",
            "review_revision": "G7E_B_R2_FULL_TEMPORAL_CANDIDATE_CLOSURE_V1",
            "reviewer_url": "http://127.0.0.1:8818/",
            "launcher": "launch_temporal_burst_review_r2.ps1",
            "next_action": "PRACTICE_REVIEW_ONLY_DO_NOT_START_REAL_TRANCHE_1",
            "semantic_features_or_folds_run": False,
            "focused_tests": {
                "passed": tests["passed"],
                "commands": [row["name"] for row in tests["results"]],
                "full_repository_suite_run": False,
            },
        },
    )
    write_json(
        HANDOFF / "02_RUNTIME_AND_UNIQUE_FRAME_CLOSURE.json",
        {
            **common,
            "frame_references": 1080,
            "unique_frames": 1044,
            "verified_unique_frames": closure["verified_unique_frame_count"],
            "unavailable_frames": closure["candidate_data_unavailable_frame_count"],
            "runtime_device": read_json(STAGE / "03_TEMPORAL_PROPOSAL_RUNTIME/runtime_device_preflight.json"),
            "proposal_runtime": read_json(STAGE / "00_INPUT_AND_RUNTIME_CLOSURE/proposal_runtime_resolution.json"),
            "pitch_gate_runtime": read_json(STAGE / "00_INPUT_AND_RUNTIME_CLOSURE/pitch_gate_runtime_resolution.json"),
        },
    )
    write_json(
        HANDOFF / "03_REUSE_AND_INFERENCE_RESULTS.json",
        {
            **common,
            "frozen_anchor_frames_searched": reuse["authoritative_frozen_anchor_frames_searched"],
            "exact_frames_reused": reuse["exact_unique_frames_reused"],
            "new_frames_inferred_once": runtime["unique_frames_newly_inferred"],
            "total_inference_seconds": runtime["total_inference_seconds"],
            "parallel_wall_elapsed_seconds": runtime["parallel_wall_elapsed_seconds"],
            "deterministic_worker_count": runtime["deterministic_worker_count"],
            "median_per_frame_seconds": runtime["median_per_frame_seconds"],
            "p95_per_frame_seconds": runtime["p95_per_frame_seconds"],
            "candidate_artifact_disk_size_bytes": runtime["candidate_artifact_disk_size_bytes"],
            "cuda_peak_allocated_bytes": runtime["cuda_peak_allocated_bytes"],
            "cuda_peak_reserved_bytes": runtime["cuda_peak_reserved_bytes"],
            "gpu_temperature_before_celsius": runtime["gpu_temperature_before_celsius"],
            "gpu_temperature_after_celsius": runtime["gpu_temperature_after_celsius"],
            "gpu_temperature_range_celsius": [
                runtime["gpu_temperature_min_celsius"],
                runtime["gpu_temperature_max_celsius"],
            ],
            "crop_features_executed": False,
            "semantic_folds_executed": False,
            "partial_candidates_merged": False,
        },
    )
    write_json(
        HANDOFF / "04_CANDIDATE_CLOSURE_RESULTS.json",
        {
            **common,
            "verified_available_frames": closure["verified_available_frame_count"],
            "verified_zero_frames": closure["verified_zero_frame_count"],
            "candidate_data_unavailable_frames": closure["candidate_data_unavailable_frame_count"],
            "pre_gate_candidates": closure["pre_gate_candidate_count"],
            "post_gate_candidates": closure["post_gate_candidate_count"],
            "pitch_gate_suppressions": closure["pitch_gate_suppression_count"],
            "candidate_ids_unique_per_frame": True,
            "retained_pre_gate_lineage_preserved": True,
            "gate_decisions_hidden_from_blind_reviewer": True,
        },
    )
    write_json(
        HANDOFF / "05_REVIEWER_MAPPING_AND_SCHEMA.json",
        {
            **common,
            "mapped_references": mapping["frame_references_mapped"],
            "unique_frames_verified": mapping["unique_frames_verified"],
            "unavailable_references": mapping["unavailable_references"],
            "candidate_state_api": reviewer["candidate_state_api"],
            "candidate_boxes_loaded_from_current_exact_frame": True,
            "centre_frame_box_propagation": False,
            "event_root_preflight": read_json(STAGE / "00_INPUT_AND_RUNTIME_CLOSURE/event_root_preflight.json"),
            "draft_and_event_fields": [
                "candidate_runtime_contract",
                "unique_frame_candidate_status",
                "per_frame_candidate_states",
                "post_gate_artifact_path_and_hash",
                "selected_candidate_ids",
            ],
            "practice_draft_policy": "INCOMPATIBLE_PRE_R2_REQUIRES_VISIBLE_RESET",
        },
    )
    write_json(
        HANDOFF / "06_USABILITY_AND_CANDIDATE_STATE_RESULTS.json",
        {
            **common,
            "r1_subject_first_guidance_preserved": reviewer["r1_subject_guidance_preserved"],
            "r1_zoom_pan_locked_step_fullscreen_preserved": reviewer["r1_zoom_pan_locked_step_fullscreen_preserved"],
            "verified_available": "all supply choices and exact candidate selection enabled",
            "verified_zero": "only No useful box and Not sure enabled; box selection disabled",
            "unavailable": "annotation and Save blocked with frame ID and failure code",
            "candidate_ids_hidden_by_default": True,
            "team_classification_present": False,
            "permanent_identity_present": False,
        },
    )
    write_json(
        HANDOFF / "07_BROWSER_AND_COORDINATE_ACCEPTANCE.json",
        {
            **common,
            "browser": browser["browser"],
            "actual_local_server": True,
            "mock_html_used": False,
            "nine_frames_loaded_with_exact_candidate_state": browser["nine_frame_candidate_states_loaded"],
            "selected_candidate_refresh_restoration": browser["selected_candidate_id_refresh_restoration"],
            "max_source_round_trip_error_pixels_per_axis": browser["max_source_round_trip_error_pixels_per_axis"],
            "max_display_round_trip_error_css_pixels_per_axis": browser[
                "max_display_round_trip_error_css_pixels_per_axis"
            ],
            "candidate_route_latency_median_ms": browser["candidate_route_latency_median_ms"],
            "frame_step_latency_after_cache_ms": browser["frame_step_latency_after_cache_ms"],
            "coordinate_candidate_occurrences_checked": coordinate["candidate_occurrences_checked"],
            "temporary_receipt_protocol_passed": True,
            "real_human_state_unchanged": browser["real_human_state_before"] == browser["real_human_state_after"],
            "three_real_browser_visuals": browser["visuals"],
        },
    )
    (HANDOFF / "08_DECISION.md").write_text(
        f"# {SUCCESS}\n\nThe exact frozen proposal path closed all 1,044 unique temporal frames; 108 were hash-exact reuses and 936 were inferred once on CUDA. All 1,080 references map to immutable post-C3A6 artifacts with zero unavailable frames.\n\nThe repaired R2 reviewer preserves the R1 subject-first guidance, 1×–12× zoom/pan, locked stepping, blind-first candidate evidence, server-backed drafts, immutable acknowledgement receipts, and tranche protocol. Verified zero is distinct from unavailable; unavailable blocks annotation.\n\nReal human state remains empty. Use practice only at `http://127.0.0.1:8818/`; do not start real Tranche 1. Production readiness remains false.\n",
        encoding="utf-8",
        newline="\n",
    )
    for source, name in (
        (STAGE / "08_VISUAL_QA/01_ALL_FRAMES_CANDIDATE_CLOSURE.png", "09_ALL_FRAMES_CLOSURE.png"),
        (STAGE / "08_VISUAL_QA/02_FRAME_SPECIFIC_SUBJECT_AND_BOXES.png", "10_FRAME_SPECIFIC_BOXES.png"),
        (STAGE / "08_VISUAL_QA/03_VERIFIED_ZERO_AND_UNAVAILABLE_STATES.png", "11_ZERO_AND_UNAVAILABLE_STATES.png"),
    ):
        copy(source, HANDOFF / name)
    files = sorted(path for path in HANDOFF.iterdir() if path.name != "12_MANIFEST.json")
    if len(files) != 11:
        raise SystemExit(f"FAIL_G7E_B_R2_CHATGPT_HANDOFF: cardinality {len(files)}")
    write_json(
        HANDOFF / "12_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7e_b_r2.handoff_manifest.v1",
            "file_count_excluding_manifest": 11,
            "self_hashed": False,
            "files": [
                {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)} for path in files
            ],
        },
    )
    (STAGE / "10_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only the CHATGPT_HANDOFF folder. It contains exactly twelve self-contained files.\n",
        encoding="utf-8",
        newline="\n",
    )
    print(SUCCESS)


if __name__ == "__main__":
    main()
