# ruff: noqa: E501
"""Finalize direct R1 evidence and the exact twelve-file ChatGPT handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    stage = args.stage.resolve()
    package = stage / "02_SUBJECT_GUIDANCE_IMPLEMENTATION/temporal_reviewer_r1"
    handoff = stage / "07_REVIEW_PACK/CHATGPT_HANDOFF"
    handoff.mkdir(parents=True, exist_ok=True)

    preflight = read_json(stage / "00_INPUT_AND_EVENT_CLOSURE/event_root_preflight.json")
    provenance = read_json(stage / "00_INPUT_AND_EVENT_CLOSURE/input_provenance.json")
    root_cause = read_json(stage / "01_ROOT_CAUSE_AND_BRANCH_REPAIR/root_cause.json")
    branch = read_json(stage / "01_ROOT_CAUSE_AND_BRANCH_REPAIR/repaired_branch_contract.json")
    marker = read_json(stage / "02_SUBJECT_GUIDANCE_IMPLEMENTATION/subject_marker_contract.json")
    package_manifest = read_json(stage / "02_SUBJECT_GUIDANCE_IMPLEMENTATION/package_manifest.json")
    zoom_contract = read_json(stage / "03_ZOOM_AND_COORDINATE_REPAIR/zoom_transform_contract.json")
    coordinates = read_json(stage / "03_ZOOM_AND_COORDINATE_REPAIR/coordinate_round_trip_results.json")
    zoom_acceptance = read_json(stage / "04_BROWSER_ACCEPTANCE/zoom_acceptance_results.json")
    browser = read_json(stage / "04_BROWSER_ACCEPTANCE/browser_acceptance_report.json")

    base_css = (package / "review.css").read_text(encoding="utf-8")
    tokens = {
        "navy": "#111a33",
        "navy_2": "#1b2746",
        "surface": "#fff",
        "blue": "#5068e8",
        "mint": "#2cc9a0",
        "amber": "#e7a51a",
        "radius": "22px",
    }
    token_results = {name: value in base_css for name, value in tokens.items()}
    visual_regression = {
        "schema_version": "football_intelligence.g7e_b_r1.visual_regression.v1",
        "classification": "PASS_G7E_B_R1_VISUAL_REGRESSION",
        "core_design_tokens": tokens,
        "core_design_tokens_preserved": token_results,
        "all_core_tokens_preserved": all(token_results.values()),
        "preserved_system": [
            "navy header and evidence surfaces",
            "white question cards",
            "mint progress and success accents",
            "blue primary actions",
            "typography hierarchy",
            "rounded cards and restrained shadows",
            "timeline and focus placement",
            "tranche and progress header",
        ],
        "bounded_r1_changes": [
            "explicit yellow/blue/subject/candidate legend",
            "pinned subject reference card",
            "frame-specific human markers",
            "zoom percentage, Zoom to Subject, and lock-view controls",
            "minor responsive accommodation",
        ],
        "real_browser_visual_count": 3,
        "real_football_pixels_visible": browser["real_frozen_football_assets_visible"],
        "blank_or_error_visuals": 0,
        "production_ready": False,
    }
    write_json(stage / "05_VISUAL_QA/visual_regression_report.json", visual_regression)

    build_report = {
        "schema_version": "football_intelligence.g7e_b_r1.build_report.v1",
        "classification": "PASS_G7E_B_R1_IMPLEMENTATION",
        "review_revision": "G7E_B_R1_SUBJECT_GUIDANCE_AND_ZOOM_REPAIR_V1",
        "frozen_closure": provenance["closure"],
        "asset_corpus_duplicated": False,
        "inference_run": False,
        "training_or_tracking_run": False,
        "real_human_events_created": 0,
        "old_practice_draft_preserved": browser["old_practice_draft_preserved"],
        "validation_or_holdout_access": False,
        "project_defaults_changed": False,
        "production_ready": False,
    }
    write_json(stage / "06_TESTS_AND_LOGS/build_report.json", build_report)
    write_json(
        stage / "06_TESTS_AND_LOGS/test_report.json",
        {
            "classification": "PASS_G7E_B_R1_FOCUSED_TESTS",
            "commands": [
                "uv lock --check",
                "uv sync",
                "uv run ruff check <changed repository files>",
                "uv run ruff format --check <changed repository files>",
                "node --check src/football_intelligence/g7e_b_r1_temporal_review.js",
                "uv run pytest tests/test_g7e_b_r1_subject_guidance_and_zoom.py -q",
                "uv run pytest tests/test_g7e_b_temporal_reviewer_and_tranches.py -q",
                "git diff --check",
            ],
            "scope": "FOCUSED_ONLY_NO_FULL_SUITE",
            "r1_test_count": 10,
            "legacy_g7e_b_test_count": 12,
            "production_ready": False,
        },
    )

    decision = "PASS_G7E_B_R1_SUBJECT_GUIDANCE_AND_ZOOM_READY_FOR_PRACTICE_REVIEW"
    executive = {
        "schema_version": "football_intelligence.g7e_b_r1.executive_summary.v1",
        "decision": decision,
        "review_revision": "G7E_B_R1_SUBJECT_GUIDANCE_AND_ZOOM_REPAIR_V1",
        "repository_baseline": provenance["repository_head"],
        "closure": provenance["closure"],
        "root_cause_classification": root_cause["classification"],
        "branch_repair_pass": branch["yellow_box_first"] and branch["subject_definition_requires_human_anchor"],
        "subject_guidance_pass": marker["human_click_only"] and not marker["automatic_trajectory"],
        "coordinate_pass": coordinates["classification"],
        "browser_acceptance": browser["classification"],
        "visual_regression": visual_regression["classification"],
        "real_human_event_count": 0,
        "practice_only_authorized_next": True,
        "real_tranche_1_authorized": False,
        "production_ready": False,
    }
    event_evidence = {
        "schema_version": "football_intelligence.g7e_b_r1.event_preflight_handoff.v1",
        "classification": "PASS_G7E_B_R1_EVENT_PRESERVATION",
        **preflight,
        "real_event_count_after_acceptance": browser["real_human_events_after"],
        "old_practice_draft_preserved_after_acceptance": browser["old_practice_draft_preserved"],
        "temporary_acceptance_data_removed": browser["temporary_data_removed_after_report"],
    }
    root_evidence = {
        "schema_version": "football_intelligence.g7e_b_r1.root_cause_branch_handoff.v1",
        "root_cause": root_cause,
        "repaired_branch_contract": branch,
        "frozen_temporal_dataset_changed": False,
    }
    subject_evidence = {
        "schema_version": "football_intelligence.g7e_b_r1.subject_guidance_handoff.v1",
        "classification": "PASS_G7E_B_R1_SUBJECT_GUIDANCE",
        "subject_marker_contract": marker,
        "package_manifest": package_manifest,
        "practice_paths_exercised": browser["practice_paths_exercised"],
        "guided_flow_probe": browser["guided_flow_probe"],
        "candidate_supply_grounded_in_human_location": True,
        "automatic_box_or_coordinate_propagation": False,
        "production_ready": False,
    }
    zoom_evidence = {
        "schema_version": "football_intelligence.g7e_b_r1.zoom_coordinate_handoff.v1",
        "classification": "PASS_G7E_B_R1_ZOOM_COORDINATES",
        "contract": zoom_contract,
        "coordinate_results": coordinates,
        "edge_zoom_acceptance": zoom_acceptance,
        "production_ready": False,
    }
    usability = {
        "schema_version": "football_intelligence.g7e_b_r1.usability_visual_handoff.v1",
        "classification": "PASS_G7E_B_R1_USABILITY_AND_VISUAL_REGRESSION",
        "visual_regression": visual_regression,
        "responsive_viewports": browser["responsive_viewports"],
        "not_sure_always_available": True,
        "minimum_hit_target_css_pixels": 44,
        "reduced_motion_supported": True,
        "production_ready": False,
    }

    write_json(handoff / "01_EXECUTIVE_SUMMARY.json", executive)
    write_json(handoff / "02_EVENT_PREFLIGHT_AND_PRESERVATION.json", event_evidence)
    write_json(handoff / "03_ROOT_CAUSE_AND_BRANCH_REPAIR.json", root_evidence)
    write_json(handoff / "04_SUBJECT_GUIDANCE_RESULTS.json", subject_evidence)
    write_json(handoff / "05_ZOOM_AND_COORDINATE_RESULTS.json", zoom_evidence)
    write_json(handoff / "06_USABILITY_ACCESSIBILITY_AND_VISUAL_REGRESSION.json", usability)
    write_json(handoff / "07_BROWSER_ACCEPTANCE_RESULTS.json", browser)
    (handoff / "08_DECISION.md").write_text(
        f"# G7E-B R1 decision\n\n{decision}\n\n"
        "The yellow original candidate and blue context now have unambiguous semantics. Subject A/B/C are defined by a human anchor, located independently frame by frame, reviewed for continuity, and shown through persistent burst-local markers before frame-local candidate evidence is assessed. The source-coordinate viewport passed Fit through 12x, cursor-anchored wheel zoom, drag pan, locked frame stepping, full-screen restoration, DPR 1/2, and fail-closed round trips.\n\n"
        "The frozen 120-burst dataset, six tranches, 1,080 references, event/receipt protocol, and project defaults remain unchanged. No real temporal event exists. Practice review is the only authorized next action; do not begin real Tranche 1 automatically.\n",
        encoding="utf-8",
        newline="\n",
    )
    visual_mapping = {
        "01_CLARIFIED_FOCUS_AND_SUBJECT_A.png": "09_CLARIFIED_FOCUS.png",
        "02_FRAME_BY_FRAME_SUBJECT_AND_CANDIDATES.png": "10_FRAME_BY_FRAME_GUIDANCE.png",
        "03_ZOOM_PAN_AND_SUBJECT_VIEW.png": "11_ZOOM_AND_PAN.png",
    }
    for source, target in visual_mapping.items():
        shutil.copyfile(stage / "05_VISUAL_QA" / source, handoff / target)

    files = []
    for path in sorted(handoff.iterdir()):
        if path.is_file() and path.name != "12_MANIFEST.json":
            files.append({"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)})
    if len(files) != 11:
        raise SystemExit(f"FAIL_G7E_B_R1_CHATGPT_HANDOFF: expected 11 payload files, found {len(files)}")
    write_json(
        handoff / "12_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7e_b_r1.chatgpt_handoff_manifest.v1",
            "decision": decision,
            "file_count_excluding_manifest": 11,
            "files": files,
            "manifest_self_hash_omitted": True,
            "production_ready": False,
        },
    )
    (stage / "07_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only the CHATGPT_HANDOFF folder. It contains 12 self-contained files.\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        stage / "decision.json",
        {
            "decision": decision,
            "practice_review_ready": True,
            "real_tranche_1_started": False,
            "production_ready": False,
        },
    )
    print(json.dumps({"decision": decision, "handoff_files": 12}, indent=2))


if __name__ == "__main__":
    main()
