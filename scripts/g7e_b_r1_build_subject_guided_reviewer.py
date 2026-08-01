# ruff: noqa: E501
"""Build the bounded G7E-B R1 subject-guided reviewer without duplicating assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

EXPECTED_HEAD = "f3243226c77c28323d78f8d00eb745f6980cde50"
REVIEW_REVISION = "G7E_B_R1_SUBJECT_GUIDANCE_AND_ZOOM_REPAIR_V1"
BURST_SHA256 = "619b6847fdde14ae13ec8a2618ac90c7ac9fc7f4d7445336bfde529e5746909d"
FRAME_SHA256 = "96688c685cc495a05af4c70003ea02b3e3f5b2dd66cc2c4813e581b10c42723d"
TRANCHE_SHA256 = "bdb9e0c2a54718124467600c2af2ebede70eda4f386c03b7374f9191dfa29466"
ASSET_MANIFEST_SHA256 = "cb2efc089ce45ccec9053c9c4751a39eb31a51d65c71a1a3415438780f7a92fb"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_count(root: Path, pattern: str) -> int:
    return len(list(root.glob(pattern))) if root.is_dir() else 0


def r1_cases(payload: dict[str, Any], practice: bool = False) -> dict[str, Any]:
    result = json.loads(json.dumps(payload))
    result["schema_version"] = (
        "football_intelligence.g7e_b_r1.practice_cases.v1"
        if practice
        else "football_intelligence.g7e_b_r1.blind_temporal_review_cases.v1"
    )
    result["review_revision"] = REVIEW_REVISION
    result["r1_semantics"] = "HUMAN_SUBJECT_LOCATION_PRECEDES_FRAME_LOCAL_CANDIDATE_SUPPLY"
    for case in result["cases"]:
        case["review_revision"] = REVIEW_REVISION
        case["schema_version"] = "football_intelligence.g7e_b_r1.blind_temporal_review_case.v1"
        case["frame_candidates"] = [[] for _ in range(9)]
        case["frame_candidates"][4] = json.loads(json.dumps(case.get("candidates", [])))
        case["candidate_availability_contract"] = "FROZEN_FRAME_LOCAL_RECORDS_ONLY_NO_PROPAGATION"
        case["yellow_original_focus_centre_frame_only"] = True
    return result


def branch_contract() -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.g7e_b_r1.reviewer_branch_contract.v1",
        "review_revision": REVIEW_REVISION,
        "first_question": "What does the yellow original focus box contain?",
        "yellow_box_first": True,
        "blue_context_is_not_counted": True,
        "subject_definition_requires_human_anchor": True,
        "subject_tokens": ["SUBJECT_A", "SUBJECT_B", "SUBJECT_C"],
        "frame_location_questions": 9,
        "visible_or_partial_requires_human_point": True,
        "automatic_coordinate_propagation": False,
        "marker_review_required": True,
        "candidate_supply_is_frame_by_frame": True,
        "candidate_selection_required_for": [
            "ONE_USEFUL_CANDIDATE",
            "MULTIPLE_CANDIDATES",
            "MERGED_WITH_OTHER_PEOPLE",
            "FRAGMENT_ONLY",
        ],
        "left_or_not_present_supply": "NOT_APPLICABLE",
        "relationship_condition": "MULTIPLE_OR_MERGED_OR_FRAGMENT_SUPPLY",
        "occlusion_condition": "PARTIAL_OR_HIDDEN_OR_DISAPPEAR_REAPPEAR",
        "continuity_is_burst_local_only": True,
        "team_classification": "INTENTIONALLY_EXCLUDED",
        "permanent_identity": "FORBIDDEN",
        "server_backed_draft_after_every_valid_answer": True,
        "immutable_event_then_acknowledgement": True,
        "production_ready": False,
    }


def event_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "football_intelligence.g7e_b_r1.burst_annotation_event.v1",
        "title": "G7E-B R1 subject-guided immutable temporal event",
        "type": "object",
        "required": [
            "event_id",
            "review_revision",
            "burst_id",
            "original_focus_box_answer",
            "subjects",
            "candidate_mappings",
            "source_frame_hashes",
            "summary_confirmed",
            "production_ready",
        ],
        "properties": {
            "review_revision": {"const": REVIEW_REVISION},
            "production_ready": {"const": False},
            "subjects": {"type": "array", "maxItems": 3},
            "source_frame_hashes": {"type": "array", "minItems": 9, "maxItems": 9},
        },
        "explicit_r1_fields": [
            "original_focus_box_answer",
            "subject_definition_source",
            "subject_location_source_x",
            "subject_location_source_y",
            "approximate_hidden_location",
            "selected_candidate_ids",
            "marker_continuity_confirmation",
        ],
        "forbidden_concepts": ["team", "shirt_number", "track_id", "cross_burst_identity", "permanent_identity"],
    }


def instructions() -> str:
    return """# G7E-B R1 temporal practice review

Launch `launch_temporal_burst_review_r1.ps1`, then open http://127.0.0.1:8818/.

Use practice only. Do not start real Tranche 1 yet. Yellow is the original focus candidate on the centre frame; blue dashed is context only. Define Subject A/B/C with your own click, then locate that same burst-local subject independently in every applicable frame. Model candidate boxes are separate evidence, not identity.

Wheel or use +/− to zoom, drag to pan, use Zoom to Subject, and Reset to return to Fit. Lock view across frames is on by default. Choose Not sure whenever needed. If an older practice draft is reported, use Reset practice; it is never silently migrated.

No team labels, permanent identities, shirt numbers, or cross-burst tracks are requested. Stop on any asset, hash, coordinate, or persistence error.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--existing-stage", type=Path, required=True)
    parser.add_argument("--g7e-a", type=Path, required=True)
    parser.add_argument("--output-stage", type=Path, required=True)
    args = parser.parse_args()

    repository = args.repository.resolve()
    old_stage = args.existing_stage.resolve()
    g7e_a = args.g7e_a.resolve()
    stage = args.output_stage.resolve()
    old_package = old_stage / "03_TEMPORAL_REVIEWER"
    package = stage / "02_SUBJECT_GUIDANCE_IMPLEMENTATION/temporal_reviewer_r1"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    if head != EXPECTED_HEAD:
        raise SystemExit(f"FAIL_BASELINE_OR_WORKTREE: {head}")
    burst_manifest = g7e_a / "02_BURST_SELECTION/temporal_burst_manifest.jsonl"
    frame_manifest = g7e_a / "02_BURST_SELECTION/temporal_frame_manifest.jsonl"
    tranche_manifest = old_stage / "01_TRANCHE_CONTRACT/tranche_manifest.jsonl"
    asset_manifest = old_stage / "02_REVIEW_ASSET_PACKAGE/review_asset_manifest.jsonl"
    expected = {
        str(burst_manifest): BURST_SHA256,
        str(frame_manifest): FRAME_SHA256,
        str(tranche_manifest): TRANCHE_SHA256,
        str(asset_manifest): ASSET_MANIFEST_SHA256,
    }
    observed = {path: sha256_file(Path(path)) for path in expected}
    if observed != expected:
        raise SystemExit("FAIL_G7E_B_R1_INPUT_PROVENANCE")

    real_root = old_package / "human_decisions"
    practice_root = old_package / "practice_decisions"
    event_preflight = {
        "schema_version": "football_intelligence.g7e_b_r1.event_root_preflight.v1",
        "real_root": str(real_root),
        "real_event_count": file_count(real_root, "events/*/*.json"),
        "real_acknowledgement_count": file_count(real_root, "receipts/acknowledgements/*.json"),
        "real_tranche_receipt_count": file_count(real_root, "receipts/tranche_completion/*.json"),
        "real_global_receipt_count": file_count(real_root, "receipts/global_completion/*.json"),
        "old_practice_draft_count": file_count(practice_root, "drafts/*.json"),
        "old_practice_draft_hashes": {
            str(path): sha256_file(path) for path in sorted(practice_root.glob("drafts/*.json"))
        },
        "old_practice_draft_migration": "REJECT_VISIBLY_RESET_REQUIRED_NO_SILENT_MIGRATION",
        "production_ready": False,
    }
    if any(
        event_preflight[key]
        for key in (
            "real_event_count",
            "real_acknowledgement_count",
            "real_tranche_receipt_count",
            "real_global_receipt_count",
        )
    ):
        raise SystemExit("FAIL_G7E_B_R1_REAL_EVENT_PREFLIGHT")

    stage.mkdir(parents=True, exist_ok=True)
    write_json(stage / "00_INPUT_AND_EVENT_CLOSURE/event_root_preflight.json", event_preflight)
    write_json(
        stage / "00_INPUT_AND_EVENT_CLOSURE/input_provenance.json",
        {
            "classification": "PASS_G7E_B_R1_INPUT_PROVENANCE",
            "repository_head": head,
            "validated_hashes": observed,
            "closure": {
                "bursts": 120,
                "tranches": 6,
                "bursts_per_tranche": 20,
                "frame_references": 1080,
                "unique_source_frames": 1044,
                "practice_bursts": 3,
            },
            "asset_corpus_duplicated": False,
            "source_assets_served_from": str(old_package / "assets"),
            "validation_or_holdout_access": False,
            "inference_run": False,
            "production_ready": False,
        },
    )

    root_cause = {
        "classification": "PASS_G7E_B_R1_ROOT_CAUSE_PROVEN",
        "causes": [
            "focus_confirmation named an undifferentiated highlighted area while both yellow and blue overlays were visible",
            "multiple-person flow created Subject A/B before an explicit person-selection explanation",
            "one centre-frame anchor was not replaced by per-frame human subject locations",
            "candidate supply was edited as an abstract nine-cell grid rather than after frame-local subject confirmation",
            "viewport stored pixel pan offsets, capped zoom at 5x, and wheel zoom was not cursor anchored",
        ],
        "frozen_data_or_model_cause": False,
        "visual_redesign_required": False,
        "production_ready": False,
    }
    write_json(stage / "01_ROOT_CAUSE_AND_BRANCH_REPAIR/root_cause.json", root_cause)
    write_json(stage / "01_ROOT_CAUSE_AND_BRANCH_REPAIR/repaired_branch_contract.json", branch_contract())

    package.mkdir(parents=True, exist_ok=True)
    real_cases = r1_cases(read_json(old_package / "review_cases.json"))
    practices = r1_cases(read_json(old_package / "practice_cases.json"), practice=True)
    write_json(package / "review_cases.json", real_cases)
    write_json(package / "practice_cases.json", practices)
    write_json(package / "reviewer_branch_contract.json", branch_contract())
    write_json(package / "reviewer_event_schema.json", event_schema())
    shutil.copyfile(tranche_manifest, package / "tranche_manifest.jsonl")
    (package / "index.html").write_bytes(
        (repository / "src/football_intelligence/g7e_b_r1_temporal_review.html").read_bytes()
    )
    (package / "review.js").write_bytes(
        (repository / "src/football_intelligence/g7e_b_r1_temporal_review.js").read_bytes()
    )
    base_css = (repository / "src/football_intelligence/g7e_b_temporal_review.css").read_text(encoding="utf-8")
    r1_css = (repository / "src/football_intelligence/g7e_b_r1_temporal_review.css").read_text(encoding="utf-8")
    (package / "review.css").write_text(base_css + "\n" + r1_css, encoding="utf-8", newline="\n")
    (package / "HUMAN_REVIEW_INSTRUCTIONS.md").write_text(instructions(), encoding="utf-8", newline="\n")
    server = """\"\"\"Launch the bounded G7E-B R1 reviewer.\"\"\"\nimport sys\nsys.path.insert(0, r'{}')\nfrom football_intelligence.temporal_review import main\nif __name__ == '__main__':\n    main()\n""".format(
        repository / "src"
    )
    (package / "review_server.py").write_text(server, encoding="utf-8", newline="\n")

    write_json(
        stage / "02_SUBJECT_GUIDANCE_IMPLEMENTATION/subject_marker_contract.json",
        {
            "review_revision": REVIEW_REVISION,
            "tokens": {"SUBJECT_A": "MINT_A", "SUBJECT_B": "VIOLET_B", "SUBJECT_C": "AMBER_C"},
            "human_click_only": True,
            "automatic_trajectory": False,
            "per_frame_locations": 9,
            "anchor_reference_card": [
                "anchor_frame_thumbnail",
                "human_anchor_point",
                "native_resolution_crop",
                "labelled_subject_token",
            ],
            "source_coordinate_fields": ["subject_location_source_x", "subject_location_source_y", "location_frame_id"],
            "production_ready": False,
        },
    )
    write_json(
        stage / "02_SUBJECT_GUIDANCE_IMPLEMENTATION/package_manifest.json",
        {
            "review_revision": REVIEW_REVISION,
            "package": str(package),
            "review_cases_sha256": sha256_file(package / "review_cases.json"),
            "practice_cases_sha256": sha256_file(package / "practice_cases.json"),
            "tranche_manifest_sha256": sha256_file(package / "tranche_manifest.jsonl"),
            "asset_root": str(old_package / "assets"),
            "asset_corpus_duplicated": False,
            "real_human_root": str(package / "human_decisions"),
            "real_human_root_exists": (package / "human_decisions").exists(),
            "legacy_practice_root": str(practice_root),
            "production_ready": False,
        },
    )
    write_json(
        stage / "03_ZOOM_AND_COORDINATE_REPAIR/zoom_transform_contract.json",
        {
            "schema_version": "football_intelligence.g7e_b_r1.zoom_transform_contract.v1",
            "views": ["PANORAMA", "FOCUS_DETAIL"],
            "canonical_pipeline": "SOURCE_TO_FITTED_IMAGE_TO_NORMALIZED_ZOOM_CENTRE_TO_CSS_DISPLAY_AND_EXACT_INVERSE",
            "zoom_multiplier": {"minimum": 1.0, "maximum": 12.0},
            "cursor_anchored_wheel_zoom": True,
            "button_zoom_anchor": "VIEWPORT_CENTRE",
            "lock_view_across_frames_default": True,
            "reset_on_new_burst": True,
            "source_round_trip_tolerance_px_per_axis": 0.5,
            "display_round_trip_tolerance_css_px_per_axis": 1.0,
            "dpr": [1, 2],
            "production_ready": False,
        },
    )

    launcher = stage / "launch_temporal_burst_review_r1.ps1"
    launcher.write_text(
        "$ErrorActionPreference = \"Stop\"\n"
        f"& \"{repository / '.venv/Scripts/python.exe'}\" \"{package / 'review_server.py'}\" `\n"
        f"  --package \"{package}\" `\n"
        f"  --asset-root \"{old_package / 'assets'}\" `\n"
        f"  --decisions-root \"{package / 'human_decisions'}\" `\n"
        f"  --practice-root \"{practice_root}\" `\n"
        "  --port 8818\n",
        encoding="utf-8",
        newline="\n",
    )
    (stage / "HUMAN_REVIEW_INSTRUCTIONS.md").write_text(instructions(), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "classification": "PASS_G7E_B_R1_DETERMINISTIC_BUILD",
                "package": str(package),
                "asset_corpus_duplicated": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
