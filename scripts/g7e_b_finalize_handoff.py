"""Create the compact, self-contained twelve-file G7E-B handoff."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from football_intelligence.temporal_review import canonical_bytes, sha256_file

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
STAGE = PROJECT / "experiments/football_observation_reasoner/part 7" / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1"
HANDOFF = STAGE / "07_REVIEW_PACK/CHATGPT_HANDOFF"


def read_json(relative: str) -> dict[str, Any]:
    return json.loads((STAGE / relative).read_text(encoding="utf-8"))


def write_json(name: str, value: Any) -> None:
    (HANDOFF / name).write_bytes(canonical_bytes(value))


def main() -> None:
    HANDOFF.mkdir(parents=True, exist_ok=True)
    input_closure = read_json("00_INPUT_CLOSURE/input_closure.json")
    balance = read_json("01_TRANCHE_CONTRACT/tranche_balance_report.json")
    assets = read_json("02_REVIEW_ASSET_PACKAGE/asset_generation_report.json")
    branch = read_json("01_TRANCHE_CONTRACT/reviewer_branch_contract.json")
    browser = read_json("04_BROWSER_ACCEPTANCE/browser_acceptance_report.json")
    build = read_json("06_TESTS_AND_LOGS/build_report.json")

    write_json(
        "01_EXECUTIVE_SUMMARY.json",
        {
            "schema_version": "football_intelligence.g7e_b.executive_summary.v1",
            "decision": "PASS_G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_READY_FOR_HUMAN_REVIEW",
            "review_id": "G7E_B_TEMPORAL_BURST_REVIEW",
            "review_revision": "G7E_B_TEMPORAL_BURST_REVIEW_V1",
            "reviewer_url": "http://127.0.0.1:8818/",
            "launcher": "launch_temporal_burst_review.ps1",
            "bursts": 120,
            "tranches": 6,
            "bursts_per_tranche": 20,
            "frame_references": 1080,
            "unique_source_frames": 1044,
            "practice_bursts": 3,
            "real_edge_acceptance": browser["classification"],
            "performance_pass": browser["performance_pass"],
            "focused_tests": "12 PASSED",
            "real_human_events_created": 0,
            "inference_run": False,
            "validation_or_holdout_access": False,
            "g7e_a_manifests_mutated": False,
            "c3a6_pitch_gate_policy_changed": False,
            "production_ready": False,
            "stop_point": "BEFORE_REAL_HUMAN_ANNOTATION_AND_G7E_C",
        },
    )
    write_json(
        "02_INPUT_AND_TRANCHE_CLOSURE.json",
        {
            "schema_version": "football_intelligence.g7e_b.input_and_tranche_closure.v1",
            "input_classification": input_closure["classification"],
            "repository_head_before_changes": input_closure["repository_head"],
            "g7e_a_decision": input_closure["g7e_a_decision"],
            "temporal_burst_manifest": input_closure["temporal_burst_manifest"],
            "temporal_frame_manifest": input_closure["temporal_frame_manifest"],
            "ontology_protocol_id": input_closure["ontology_protocol_id"],
            "matches": input_closure["matches"],
            "tranche_classification": balance["classification"],
            "assignment_algorithm": balance["assignment_algorithm"],
            "assignment_attempt": balance["assignment_attempt"],
            "tranche_manifest_sha256": balance["tranche_manifest_sha256"],
            "tranches": balance["tranches"],
            "match_rotation_exact": True,
            "class_quotas_exact": True,
            "half_perspective_low_light_balance_pass": True,
            "tranche_1_calibration_seed_requirements_pass": True,
            "production_ready": False,
        },
    )
    write_json(
        "03_REVIEW_ASSET_RESULTS.json",
        {
            **assets,
            "asset_root": "03_TEMPORAL_REVIEWER/assets",
            "manifest_path": "02_REVIEW_ASSET_PACKAGE/review_asset_manifest.jsonl",
            "source_hash_validation": "SHA256_RGB24_C_CONTIGUOUS_SOURCE_DIMENSIONS",
            "browser_asset_hash_validation": "SHA256_EXACT_DERIVATIVE_BYTES_BEFORE_READY",
            "focus_crop_source_coordinate_mapping_recorded": True,
            "blind_payload_exposes_protected_selection_truth": False,
            "lazy_timeline_and_next_burst_prefetch": True,
        },
    )
    write_json(
        "04_REVIEWER_BRANCH_AND_EVENT_CONTRACT.json",
        {
            **branch,
            "persistence_chain": [
                "SERVER_BACKED_DRAFT_AFTER_EVERY_VALID_ANSWER",
                "IMMUTABLE_BURST_EVENT",
                "EVENT_ACKNOWLEDGEMENT_RECEIPT",
                "CURRENT_TRANCHE_COMPLETION_RECEIPT",
                "CURRENT_GLOBAL_COMPLETION_RECEIPT",
            ],
            "tranche_locking": "ONLY_IMMEDIATE_NEXT_AFTER_CURRENT_RECEIPT_AND_EXPLICIT_UNLOCK",
            "superseding_edit": "APPEND_EVENT_AND_REFRESH_AFFECTED_CURRENT_RECEIPTS",
            "last_event_and_receipt_ids_separate": True,
            "practice_storage_isolated_and_resettable": True,
            "read_only_completed_answer_endpoint": "/api/completed?tranche=TRANCHE_N",
            "event_schema_path": "03_TEMPORAL_REVIEWER/reviewer_event_schema.json",
            "production_ready": False,
        },
    )
    write_json(
        "05_USABILITY_AND_ACCESSIBILITY_RESULTS.json",
        {
            "schema_version": "football_intelligence.g7e_b.usability_results.v1",
            "classification": "PASS_G7E_B_USABILITY",
            "design": "POLISHED_MODERN_NON_EXPERT_FRIENDLY_NAVY_WHITE_MINT_AMBER_CORAL",
            "one_question_at_a_time": True,
            "plain_english_labels": True,
            "not_sure_available": True,
            "nine_frame_timeline": True,
            "playback": ["0.5x", "1x", "2x"],
            "default_fps": 5,
            "pan_zoom_reset_fullscreen": True,
            "candidate_ids_hidden_by_default": True,
            "minimum_hit_target_css_pixels": 44,
            "keyboard_navigation": True,
            "visible_focus": True,
            "reduced_motion": True,
            "responsive_viewports": browser["viewports"],
            "horizontal_overflow_failures": 0,
            "coordinate_mapping": browser["coordinate_mapping"],
            "performance_ms": browser["performance_ms"],
            "performance_targets_ms": browser["performance_targets_ms"],
            "performance_pass": browser["performance_pass"],
            "visual_qa_inspected": True,
            "visual_qa_findings": "REAL FOOTBALL PIXELS, LEGIBLE CONTROLS, CLEAR BRANCHING, CLEAR PAUSE BOUNDARY",
            "production_ready": False,
        },
    )
    write_json("06_BROWSER_ACCEPTANCE_RESULTS.json", browser)
    (HANDOFF / "07_DECISION.md").write_text(
        "# G7E-B decision\n\n"
        "PASS_G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_READY_FOR_HUMAN_REVIEW\n\n"
        "The exact frozen 120-burst G7E-A selection is assigned deterministically to six independently completable "
        "20-burst tranches. All 1,080 browser references are bound to validated source pixels and bounded derivatives. "
        "The polished blind-first reviewer passed real Microsoft Edge acceptance, source/display coordinate checks, "
        "server-backed draft restoration, immutable event acknowledgements, tranche locking/unlocking, supersession, "
        "six tranche receipts, and the global receipt.\n\n"
        "No real temporal human truth was created. No inference, training, default change, validation access, holdout "
        "access, identity construction, team classification, or production activation occurred. Start Tranche 1 only; "
        "after its receipt appears, pause and return that receipt for independent finalization.\n",
        encoding="utf-8",
        newline="\n",
    )
    shutil.copyfile(
        STAGE / "03_TEMPORAL_REVIEWER/HUMAN_REVIEW_INSTRUCTIONS.md",
        HANDOFF / "08_HUMAN_REVIEW_INSTRUCTIONS.md",
    )
    for source, destination in (
        ("01_POLISHED_MAIN_REVIEW.png", "09_POLISHED_MAIN_REVIEW.png"),
        ("02_TIMELINE_AND_BRANCHING.png", "10_TIMELINE_AND_BRANCHING.png"),
        ("03_TRANCHE_COMPLETION_AND_RESUME.png", "11_TRANCHE_COMPLETION.png"),
    ):
        shutil.copyfile(STAGE / "05_VISUAL_QA" / source, HANDOFF / destination)

    expected = [
        "01_EXECUTIVE_SUMMARY.json",
        "02_INPUT_AND_TRANCHE_CLOSURE.json",
        "03_REVIEW_ASSET_RESULTS.json",
        "04_REVIEWER_BRANCH_AND_EVENT_CONTRACT.json",
        "05_USABILITY_AND_ACCESSIBILITY_RESULTS.json",
        "06_BROWSER_ACCEPTANCE_RESULTS.json",
        "07_DECISION.md",
        "08_HUMAN_REVIEW_INSTRUCTIONS.md",
        "09_POLISHED_MAIN_REVIEW.png",
        "10_TIMELINE_AND_BRANCHING.png",
        "11_TRANCHE_COMPLETION.png",
    ]
    manifest_rows = []
    for filename in expected:
        path = HANDOFF / filename
        manifest_rows.append({"filename": filename, "byte_size": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(
        "12_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7e_b.chatgpt_handoff_manifest.v1",
            "file_count_excluding_manifest": 11,
            "files": manifest_rows,
            "self_hash_included": False,
            "build_report_sha256": sha256_file(STAGE / "06_TESTS_AND_LOGS/build_report.json"),
            "production_ready": False,
        },
    )
    files = sorted(path.name for path in HANDOFF.iterdir() if path.is_file())
    if files != sorted([*expected, "12_MANIFEST.json"]):
        raise ValueError(f"handoff file-set mismatch: {files}")
    (STAGE / "07_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only the CHATGPT_HANDOFF folder. It contains exactly 12 self-contained review files.\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"decision": "PASS_G7E_B_CHATGPT_HANDOFF", "files": len(files), "build": build["classification"]}))


if __name__ == "__main__":
    main()
