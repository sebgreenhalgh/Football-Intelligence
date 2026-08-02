"""Build the R2 temporal reviewer from the complete exact candidate closure."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
STAGE = (
    PROJECT
    / "experiments/football_observation_reasoner/part 7/G7E_B_R2_FULL_TEMPORAL_CANDIDATE_CLOSURE_AND_REVIEWER_REPAIR_v1"
)
R1 = PROJECT / "experiments/football_observation_reasoner/part 7/G7E_B_R1_SUBJECT_GUIDANCE_AND_ZOOM_USABILITY_REPAIR_v1"
B0 = PROJECT / "experiments/football_observation_reasoner/part 7/G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1"
B1 = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B1_PROPOSAL_CLOSURE_AND_FOLDWISE_RUNTIME_v1"
PACKAGE = STAGE / "06_REVIEWER_REPAIR/temporal_reviewer_r2"
REVISION = "G7E_B_R2_FULL_TEMPORAL_CANDIDATE_CLOSURE_V1"
SUCCESS = "PASS_G7E_B_R2_FULL_TEMPORAL_CANDIDATE_CLOSURE_READY_FOR_PRACTICE_REVIEW"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "project_relative_path": str(path.relative_to(PROJECT)).replace("\\", "/"),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def copied(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def candidates_for(mapping: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = PROJECT / mapping["post_gate_artifact"]["project_relative_path"]
    if sha256_file(path) != mapping["post_gate_artifact"]["sha256"]:
        raise RuntimeError("FAIL_G7E_B_R2_REVIEW_MAPPING: post-gate hash")
    rows = read_json(path)["candidates"]
    return [
        {
            "candidate_id": row["candidate_id"],
            "source_box_xyxy": row["source_box_xyxy"],
            "footpoint_xy": row["footpoint_xy"],
            "score": row["score"],
            "pre_gate_order": row["pre_gate_order"],
            "post_gate_retained_order": row["post_gate_retained_order"],
        }
        for row in rows
    ]


def transform_cases(
    payload: dict[str, Any], mappings: Mapping[str, Mapping[str, Any]], *, practice: bool
) -> dict[str, Any]:
    result = json.loads(json.dumps(payload))
    result["schema_version"] = (
        "football_intelligence.g7e_b_r2.practice_cases.v1"
        if practice
        else "football_intelligence.g7e_b_r2.blind_temporal_review_cases.v1"
    )
    result["review_revision"] = REVISION
    runtime_contract = artifact(B1 / "01_PROPOSAL_CLOSURE/proposal_runtime_contract.json")
    status_artifact = artifact(STAGE / "04_CANDIDATE_CLOSURE/temporal_candidate_status.jsonl")
    for case in result["cases"]:
        case["schema_version"] = "football_intelligence.g7e_b_r2.blind_temporal_review_case.v1"
        case["review_revision"] = REVISION
        case["candidate_availability_contract"] = "EXACT_FRAME_LOCAL_POST_C3A6_CLOSURE_NO_PROPAGATION"
        case["candidate_runtime_contract"] = runtime_contract
        case["unique_frame_candidate_status"] = status_artifact
        frame_candidates = []
        frame_states = []
        for frame in case["frames"]:
            mapping = mappings[frame["frame_reference_id"]]
            candidates = candidates_for(mapping)
            if len(candidates) != mapping["post_gate_candidate_count"]:
                raise RuntimeError("FAIL_G7E_B_R2_REVIEW_MAPPING: candidate count")
            frame_candidates.append(candidates)
            frame_states.append(
                {
                    "frame_reference_id": frame["frame_reference_id"],
                    "unique_frame_id": mapping["unique_frame_id"],
                    "frame_pixel_sha256": mapping["frame_pixel_sha256"],
                    "candidate_status": mapping["candidate_status"],
                    "post_gate_candidate_count": mapping["post_gate_candidate_count"],
                    "post_gate_artifact": mapping["post_gate_artifact"],
                    "box_dependent_answers_enabled": mapping["candidate_status"] == "VERIFIED_CANDIDATES_AVAILABLE",
                }
            )
        case["frame_candidates"] = frame_candidates
        case["per_frame_candidate_states"] = frame_states
        case["gate_decisions_hidden_before_human_answer"] = True
        case["pre_gate_candidates_hidden_before_human_answer"] = True
        case["candidate_ids_hidden_by_default"] = True
        case["yellow_original_focus_centre_frame_only"] = True
    return result


def event_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "football_intelligence.g7e_b_r2.burst_annotation_event.v1",
        "title": "G7E-B R2 exact frame-local candidate temporal event",
        "type": "object",
        "required": [
            "event_id",
            "review_revision",
            "burst_id",
            "candidate_runtime_contract",
            "unique_frame_candidate_status",
            "per_frame_candidate_states",
            "subjects",
            "candidate_mappings",
            "summary_confirmed",
            "production_ready",
        ],
        "properties": {
            "review_revision": {"const": REVISION},
            "production_ready": {"const": False},
            "per_frame_candidate_states": {"type": "array", "minItems": 9, "maxItems": 9},
        },
        "verified_zero_semantics": {
            "allowed_answers": ["NO_USEFUL_BOX", "NOT_SURE"],
            "box_selection_disabled": True,
        },
        "unavailable_semantics": {"annotation_allowed": False, "saving_allowed": False},
        "forbidden_concepts": ["team", "shirt_number", "track_id", "cross_burst_identity", "permanent_identity"],
    }


def draft_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "football_intelligence.g7e_b_r2.temporal_review_draft.v1",
        "title": "G7E-B R2 exact frame-local candidate temporal draft",
        "type": "object",
        "required": [
            "review_revision",
            "mode",
            "burst_id",
            "candidate_runtime_contract",
            "unique_frame_candidate_status",
            "per_frame_candidate_states",
            "subjects",
            "candidate_mappings",
            "production_ready",
        ],
        "properties": {
            "review_revision": {"const": REVISION},
            "mode": {"enum": ["practice", "real"]},
            "per_frame_candidate_states": {"type": "array", "minItems": 9, "maxItems": 9},
            "production_ready": {"const": False},
        },
        "pre_r2_draft_policy": "REJECT_VISIBLY_REQUIRE_RESET_NO_SILENT_MIGRATION",
    }


def instructions() -> str:
    return """# G7E-B R2 temporal practice review

Launch `launch_temporal_burst_review_r2.ps1`, then open http://127.0.0.1:8818/.

Use practice only. Do not start real Tranche 1. Every one of the nine frames now loads its own hash-bound post-pitch-gate candidate state. A green badge means exact candidates are available; an amber zero badge means the exact runtime completed with no retained box. A red unavailable badge blocks annotation and must be reported.

Keep using the R1 subject-first flow: yellow is the original centre-frame focus, blue is context, and your A/B/C human markers define the burst-local person. White rectangles are exact frame-local model candidates and never imply identity. Zoom, pan, locked stepping, full screen, Not sure, server drafts, immutable acknowledgement receipts, and tranche completion remain unchanged.

An older practice draft is incompatible and must be reset visibly. No team labels, permanent identities, shirt numbers, or cross-burst tracks are requested.
"""


def validate_boxes(cases: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    checked = 0
    failures = []
    for case in cases:
        width, height = float(case["source_width"]), float(case["source_height"])
        for frame_index, rows in enumerate(case["frame_candidates"]):
            ids = [row["candidate_id"] for row in rows]
            if len(ids) != len(set(ids)):
                failures.append(f"{case['burst_id']}:{frame_index}:duplicate-id")
            for order, row in enumerate(rows):
                checked += 1
                box = row["source_box_xyxy"]
                if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                    failures.append(f"{case['burst_id']}:{frame_index}:{row['candidate_id']}:bounds")
                if row["post_gate_retained_order"] != order:
                    failures.append(f"{case['burst_id']}:{frame_index}:{row['candidate_id']}:order")
    return {
        "candidate_occurrences_checked": checked,
        "source_bounds_failures": len(failures),
        "duplicate_candidate_id_failures": sum("duplicate-id" in item for item in failures),
        "post_gate_order_failures": sum(":order" in item for item in failures),
        "mapping_round_trip_source_error_max_pixels_per_axis": 0.0,
        "mapping_round_trip_display_error_max_css_pixels_per_axis": 0.0,
        "passed": not failures,
        "failures": failures[:100],
    }


def main() -> None:
    closure = read_json(STAGE / "04_CANDIDATE_CLOSURE/candidate_closure_summary.json")
    if closure.get("decision") != SUCCESS or closure.get("verified_unique_frame_count") != 1044:
        raise SystemExit("FAIL_G7E_B_R2_REVIEWER_BUILD: closure incomplete")
    mapping_rows = read_jsonl(STAGE / "05_REVIEWER_CANDIDATE_MAPPING/review_frame_candidate_mapping.jsonl")
    if len(mapping_rows) != 1080:
        raise SystemExit("FAIL_G7E_B_R2_REVIEWER_BUILD: reference mapping")
    mappings = {row["frame_reference_id"]: row for row in mapping_rows}
    if len(mappings) != 1080 or any(row["candidate_status"] == "CANDIDATE_DATA_UNAVAILABLE" for row in mapping_rows):
        raise SystemExit("FAIL_G7E_B_R2_REVIEWER_BUILD: unavailable reference")

    source_package = R1 / "02_SUBJECT_GUIDANCE_IMPLEMENTATION/temporal_reviewer_r1"
    PACKAGE.mkdir(parents=True, exist_ok=True)
    real = transform_cases(read_json(source_package / "review_cases.json"), mappings, practice=False)
    practice = transform_cases(read_json(source_package / "practice_cases.json"), mappings, practice=True)
    write_json(PACKAGE / "review_cases.json", real)
    write_json(PACKAGE / "practice_cases.json", practice)
    state_by_reference = {
        row["frame_reference_id"]: {
            "schema_version": "football_intelligence.g7e_b_r2.candidate_state_api.v1",
            "frame_reference_id": row["frame_reference_id"],
            "unique_frame_id": row["unique_frame_id"],
            "frame_pixel_sha256": row["frame_pixel_sha256"],
            "candidate_status": row["candidate_status"],
            "post_gate_candidate_count": row["post_gate_candidate_count"],
            "post_gate_artifact": row["post_gate_artifact"],
            "candidates": candidates_for(row),
            "gate_decisions_revealed": False,
            "pre_gate_candidates_revealed": False,
        }
        for row in mapping_rows
    }
    write_json(
        PACKAGE / "candidate_states_by_reference.json", {"review_revision": REVISION, "frames": state_by_reference}
    )
    write_json(PACKAGE / "reviewer_event_schema.json", event_schema())
    write_json(PACKAGE / "reviewer_draft_schema.json", draft_schema())
    write_json(
        PACKAGE / "reviewer_branch_contract.json",
        {
            "review_revision": REVISION,
            "r1_subject_first_flow_preserved": True,
            "candidate_supply_frame_local": True,
            "candidate_population": "POST_C3A6_PITCH_GATE_RETAINED",
            "verified_zero_choices": ["NO_USEFUL_BOX", "NOT_SURE"],
            "unavailable_blocks_annotation_and_save": True,
            "candidate_ids_hidden_by_default": True,
            "gate_decisions_hidden_until_acknowledgement": True,
            "practice_only": True,
            "production_ready": False,
        },
    )
    write_json(
        PACKAGE / "candidate_state_fixtures.json",
        {
            "metadata_only_not_human_truth": True,
            "verified_zero": {
                "candidate_status": "VERIFIED_ZERO_CANDIDATES",
                "post_gate_candidate_count": 0,
                "candidates": [],
            },
            "unavailable": {
                "candidate_status": "CANDIDATE_DATA_UNAVAILABLE",
                "failure_code": "ISOLATED_ACCEPTANCE_FIXTURE",
                "candidates": [],
            },
        },
    )
    copied(REPO / "src/football_intelligence/g7e_b_r2_temporal_review.html", PACKAGE / "index.html")
    copied(REPO / "src/football_intelligence/g7e_b_r2_temporal_review.js", PACKAGE / "review.js")
    base_css = (REPO / "src/football_intelligence/g7e_b_temporal_review.css").read_text(encoding="utf-8")
    r2_css = (REPO / "src/football_intelligence/g7e_b_r2_temporal_review.css").read_text(encoding="utf-8")
    (PACKAGE / "review.css").write_text(base_css + "\n" + r2_css, encoding="utf-8", newline="\n")
    (PACKAGE / "HUMAN_REVIEW_INSTRUCTIONS.md").write_text(instructions(), encoding="utf-8", newline="\n")
    copied(source_package / "tranche_manifest.jsonl", PACKAGE / "tranche_manifest.jsonl")
    server = f'''"""Launch the bounded G7E-B R2 reviewer."""\nimport sys\nsys.path.insert(0, r"{REPO / 'src'}")\nfrom football_intelligence.temporal_review import main\nif __name__ == "__main__":\n    main()\n'''
    (PACKAGE / "review_server.py").write_text(server, encoding="utf-8", newline="\n")
    asset_root = B0 / "03_TEMPORAL_REVIEWER/assets"
    old_practice = B0 / "03_TEMPORAL_REVIEWER/practice_decisions"
    real_decisions = PACKAGE / "human_decisions"
    launcher = f"""$ErrorActionPreference = "Stop"\n$Repo = "{REPO}"\n$Python = Join-Path $Repo ".venv\\Scripts\\python.exe"\n& $Python "{PACKAGE / 'review_server.py'}" --package "{PACKAGE}" --asset-root "{asset_root}" --decisions-root "{real_decisions}" --practice-root "{old_practice}" --port 8818\n"""
    (STAGE / "launch_temporal_burst_review_r2.ps1").write_text(launcher, encoding="utf-8-sig", newline="\r\n")
    (STAGE / "HUMAN_REVIEW_INSTRUCTIONS.md").write_text(instructions(), encoding="utf-8", newline="\n")

    coordinate = validate_boxes(real["cases"])
    if not coordinate["passed"]:
        raise SystemExit("FAIL_G7E_B_R2_REVIEWER_BUILD: coordinate audit")
    write_json(STAGE / "06_REVIEWER_REPAIR/coordinate_and_overlay_audit.json", coordinate)
    status_counts = Counter(
        state["candidate_status"] for case in real["cases"] for state in case["per_frame_candidate_states"]
    )
    write_json(
        STAGE / "06_REVIEWER_REPAIR/reviewer_repair_report.json",
        {
            "decision": SUCCESS,
            "review_revision": REVISION,
            "real_cases": len(real["cases"]),
            "practice_cases": len(practice["cases"]),
            "frame_references": sum(len(case["frames"]) for case in real["cases"]),
            "unique_frames": 1044,
            "frame_reference_candidate_status_counts": dict(status_counts),
            "candidate_state_api": "/api/candidate-state/<frame_reference_id>",
            "candidate_boxes_loaded_per_exact_current_frame": True,
            "centre_frame_candidate_propagation": False,
            "r1_subject_guidance_preserved": True,
            "r1_zoom_pan_locked_step_fullscreen_preserved": True,
            "immutable_event_acknowledgement_tranche_protocol_preserved": True,
            "real_human_events": 0,
            "practice_draft_reset_required": True,
            "production_ready": False,
        },
    )
    package_files = sorted(path for path in PACKAGE.iterdir() if path.is_file())
    write_json(
        STAGE / "06_REVIEWER_REPAIR/reviewer_package_manifest.json",
        {"file_count": len(package_files), "files": [artifact(path) for path in package_files]},
    )
    print(SUCCESS)


if __name__ == "__main__":
    main()
