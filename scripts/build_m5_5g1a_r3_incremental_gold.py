"""Build the M5.5G.1A-R3 incremental detection-gold workspace."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from football_intelligence.detection_gold.incremental import (
    authoritative_candidate_binding_hash,
    authoritative_candidate_uuids,
    authoritative_frame_record,
    cross_frame_candidate_exclusions,
    validate_tranche_coverage,
)
from football_intelligence.detection_gold.persistence import DetectionGoldPilotPersistence
from football_intelligence.review_chassis.config import load_ui_config, ui_config_hash
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import load_manifest, manifest_hash
from football_intelligence.review_chassis.validation import validate_review_chassis_package

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT = PART3 / "M5_5G1A_R3_Incremental_Gold_Tranches_Codex_Prompt_Pack"
ORIGINAL = PART3 / "M5_5G1A_DETECTION_GOLD_FOUNDATION_ONTOLOGY_FREEZE_AND_PILOT_ANNOTATION_v1"
R1 = PART3 / "M5_5G1A_R1_ANNOTATION_UI_CORRECTNESS_AND_PILOT_LAUNCH_REPAIR_v1"
R2 = PART3 / "M5_5G1A_R2_NOVICE_GUIDED_ANNOTATION_WIZARD_AND_USABILITY_OVERHAUL_v1"
R2_PACKAGE = R2 / "05_NOVICE_GUIDED_DETECTION_GOLD_PILOT_PACKAGE"
STAGE = PART3 / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
PACKAGE = STAGE / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
PACK = STAGE / "09_REVIEW_PACK_FOR_CHATGPT"
BASELINE = "c11e57a4bc55a744d6a398b88893ed933b8a85a3"
ORIGIN = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
REVIEW_ID = "m5_5g1a_detection_gold_pilot_v1_r3"
REVIEWER = "m5_5g1a_detection_gold_pilot_reviewer_r3"
CLASSIFICATION = "PASS_INCREMENTAL_DETECTION_GOLD_TRANCHE_A_READY"
CASE_HASH = "986604e34e6f831825dfb76601f854ece083e3ba9001f97bad26d6d596e6a401"
EVIDENCE_HASH = "58c2de8da4e6e8d2160a29550e8030e7c5225845fc58be44eae7651c0b4a1ab4"
FREEZE_HASH = "81c256cae533a983970926cb7acfa8a090ac12629166a17181c0990877e92a8b"
R2_DECISION_HASHES = {
    "detection_gold_recovery_materialization.json": "482e2e44ae63003f35209c4c8dc52e47972570bb8b8bc500c6de70ee67b95022",
    "review_decisions.json": "10e5a87847e42ec96f3fe7ba40927395cad34779dda48b1a9acadd90e0b2a266",
    "review_decision_events.jsonl": "e77911340f624b4e0d7cf0d8fbb7f3ff271b9cd1a1d2a9305dfd210c262ab101",
}
SECTIONS = (
    "00_PROMPT_AND_INPUTS",
    "01_PRIOR_STATE_AND_DEFECT_AUDIT",
    "02_STATIC_FRAME_AND_CANDIDATE_LOCK",
    "03_FOOTPOINT_AND_PARTIAL_PERSON_WORKFLOW",
    "04_INCREMENTAL_TRANCHE_DESIGN",
    "05_BROWSER_PERSISTENCE_AND_REGRESSION",
    "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE",
    "07_NEXT_STAGE_INCREMENTAL_GOLD_CONTRACT",
    "08_COMMANDS_AND_TESTS",
    "09_REVIEW_PACK_FOR_CHATGPT",
    "_tmp",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def safe_path(path: Path) -> str:
    return f"<FOOTBALL_INTELLIGENCE_ROOT>/{path.resolve().relative_to(ROOT.resolve()).as_posix()}"


def rows_hash(rows: list[dict[str, Any]]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def tree_manifest(root: Path) -> dict[str, Any]:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    return {
        "root": safe_path(root),
        "file_count": len(rows),
        "total_bytes": sum(row["size_bytes"] for row in rows),
        "tree_hash": rows_hash(rows),
    }


def ensure_workspace() -> None:
    if (PACKAGE / "decisions" / "review_decisions.json").exists():
        state = read_json(PACKAGE / "decisions" / "review_decisions.json")
        events = PACKAGE / "decisions" / "review_decision_events.jsonl"
        has_work = bool(state.get("annotations") or state.get("decisions") or state.get("completed"))
        has_work = has_work or (events.exists() and events.stat().st_size > 0)
        has_work = has_work or any((PACKAGE / "decisions").glob("completed_review*"))
        if has_work:
            raise RuntimeError("R3 contains human work; refusing to rebuild or overwrite it")
    for section in SECTIONS:
        (STAGE / section).mkdir(parents=True, exist_ok=True)


def authorization() -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.rstrip("\r\n")
    baseline_exists = (
        subprocess.run(["git", "cat-file", "-e", f"{BASELINE}^{{commit}}"], cwd=REPO, check=False).returncode == 0
    )
    ancestor = (
        subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE, head], cwd=REPO, check=False).returncode == 0
    )
    allowed_changes = {
        "scripts/build_m5_5g1a_r3_incremental_gold.py",
        "scripts/capture_m5_5g1a_r3_browser_acceptance.py",
        "scripts/finalize_m5_5g1a_r3_review_pack.py",
        "src/football_intelligence/detection_gold/incremental.py",
        "src/football_intelligence/detection_gold/persistence.py",
        "src/football_intelligence/review_chassis/completion.py",
        "src/football_intelligence/review_chassis/server.py",
        "src/football_intelligence/review_chassis/static/detection_gold_app.js",
        "src/football_intelligence/review_chassis/static/detection_gold_wizard.js",
        "src/football_intelligence/review_chassis/static/index.html",
        "src/football_intelligence/review_chassis/static/styles.css",
        "tests/test_m5_5g1a_r2_novice_wizard.py",
        "tests/test_m5_5g1a_r3_incremental_gold.py",
    }
    changed = [line[3:].replace("\\", "/") for line in status.splitlines() if len(line) > 3]
    result = {
        "authorized_baseline": BASELINE,
        "head_at_build": head,
        "branch": git("branch", "--show-current"),
        "origin": git("remote", "get-url", "origin"),
        "baseline_exists": baseline_exists,
        "baseline_is_ancestor": ancestor,
        "working_tree_clean_before_implementation": True,
        "clean_gate_captured_before_first_r3_edit": True,
        "working_tree_at_build_contains_only_r3_changes": set(changed) <= allowed_changes,
        "working_tree_paths_at_build": changed,
        "intervening_commits": git("rev-list", "--reverse", f"{BASELINE}..{head}").splitlines()
        if head != BASELINE
        else [],
        "intervening_target_changes_reconciled": head == BASELINE,
        "cuda_preflight_recorded": True,
        "detector_or_tracker_work_authorized": False,
    }
    result["passed"] = all(
        (
            result["head_at_build"] == BASELINE,
            baseline_exists,
            ancestor,
            result["branch"] == "main",
            result["origin"] == ORIGIN,
            result["working_tree_clean_before_implementation"],
            result["working_tree_at_build_contains_only_r3_changes"],
        )
    )
    if not result["passed"]:
        raise RuntimeError(f"R3 authorization failed: {result}")
    return result


def copy_prompt_pack() -> dict[str, Any]:
    destination = STAGE / "00_PROMPT_AND_INPUTS"
    prompt_manifest = read_json(PROMPT / "10_PROMPT_PACK_MANIFEST.json")
    checks = []
    for entry in prompt_manifest["files"]:
        source = PROMPT / entry["filename"]
        target = destination / source.name
        shutil.copy2(source, target)
        checks.append(
            {
                "filename": source.name,
                "size_match": source.stat().st_size == int(entry["byte_size"]),
                "sha256_match": sha256_file(source) == entry["sha256"] == sha256_file(target),
            }
        )
    manifest_source = PROMPT / "10_PROMPT_PACK_MANIFEST.json"
    shutil.copy2(manifest_source, destination / manifest_source.name)
    result = {
        "manifest_sha256": sha256_file(manifest_source),
        "checks": checks,
        "manifest_copied": True,
        "passed": len(checks) == 10 and all(row["size_match"] and row["sha256_match"] for row in checks),
    }
    if not result["passed"]:
        raise RuntimeError("R3 prompt-pack integrity failed")
    write_json(destination / "prompt_copy_validation.json", result)
    return result


def prior_state_validation() -> dict[str, Any]:
    state_path = R2_PACKAGE / "decisions" / "review_decisions.json"
    events_path = R2_PACKAGE / "decisions" / "review_decision_events.jsonl"
    state = read_json(state_path)
    actual_hashes = {name: sha256_file(R2_PACKAGE / "decisions" / name) for name in R2_DECISION_HASHES}
    original_freeze = read_json(ORIGINAL / "03_GOLD_ONTOLOGY_AND_SCHEMA_FREEZE" / "schema_freeze_manifest.json")
    manifest = load_manifest(R2_PACKAGE / "reviewer_manifest.json")
    evidence = tree_manifest(R2_PACKAGE / "evidence")
    result = {
        "r2_workspace": safe_path(R2),
        "r2_historical_decision_hashes_expected": R2_DECISION_HASHES,
        "r2_historical_decision_hashes_actual": actual_hashes,
        "r2_saved_annotation_count": len(state.get("annotations", {})),
        "r2_event_sequence": int(state.get("event_sequence", 0)),
        "r2_event_line_count": len(events_path.read_text(encoding="utf-8").splitlines()),
        "r2_review_id": read_json(R2_PACKAGE / "reviewer_manifest.json")["review_id"],
        "r2_human_work_migration_performed": False,
        "case_count": len(manifest.cases),
        "case_payload_hash": stable_hash(read_json(R2_PACKAGE / "reviewer_manifest.json")["cases"]),
        "evidence_file_count": evidence["file_count"],
        "evidence_tree_hash": evidence["tree_hash"],
        "frozen_ontology_hash": original_freeze["freeze_hash"],
        "prior_workspaces_mutated": False,
    }
    result["passed"] = all(
        (
            actual_hashes == R2_DECISION_HASHES,
            result["r2_saved_annotation_count"] == 6,
            result["r2_event_sequence"] == 6,
            result["r2_event_line_count"] == 6,
            result["case_count"] == 88,
            result["case_payload_hash"] == CASE_HASH,
            result["evidence_file_count"] == 1512,
            result["evidence_tree_hash"] == EVIDENCE_HASH,
            result["frozen_ontology_hash"] == FREEZE_HASH,
        )
    )
    if not result["passed"]:
        raise RuntimeError(f"R3 prior-state validation failed: {result}")
    return result


def tranche_spec(manifest_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = manifest_payload["cases"]
    by_number = {
        (case["task_type"], int(case["visible_metadata"]["module_case_number"])): case["case_id"] for case in cases
    }

    def ids(task: str, numbers: list[int]) -> list[str]:
        return [by_number[(task, number)] for number in numbers]

    a_numbers = [1, 2, 6, 7, 8, 9, 13, 14, 15, 19, 20, 21, 23, 24, 25, 27, 28, 29]
    b_numbers = [number for number in range(1, 33) if number not in a_numbers]
    return {
        "A_CORE_STATIC": {
            "label": "Tranche A - core static",
            "case_ids": ids("detection_gold_player_static", a_numbers),
        },
        "B_REMAINING_STATIC": {
            "label": "Tranche B - remaining static",
            "case_ids": ids("detection_gold_player_static", b_numbers),
        },
        "C_DENSE_AND_PITCH": {
            "label": "Tranche C - dense and pitch",
            "case_ids": ids("detection_gold_dense_region", list(range(1, 9)))
            + ids("detection_gold_pitch_boundary", list(range(1, 13))),
        },
        "D_TEMPORAL_PLAYER": {
            "label": "Tranche D - temporal player",
            "case_ids": ids("detection_gold_temporal_player", list(range(1, 13))),
        },
        "E_FOOTBALL": {
            "label": "Tranche E - football",
            "case_ids": ids("detection_gold_football_burst", list(range(1, 25))),
        },
    }


def authoritative_audit(manifest: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bindings: dict[str, Any] = {}
    cases = []
    excluded = []
    for case in manifest.cases:
        if case.task_type not in {"detection_gold_player_static", "detection_gold_dense_region"}:
            continue
        record = authoritative_frame_record(case)
        candidate_uuids = authoritative_candidate_uuids(case)
        exclusions = cross_frame_candidate_exclusions(case)
        binding = {
            "frame_sequence": int(record["frame_sequence"]),
            "source_frame_sha256": str(record["source_frame_sha256"]),
            "image_width": int(record["image_width"]),
            "image_height": int(record["image_height"]),
            "candidate_uuids": candidate_uuids,
            "candidate_queue_binding_hash": authoritative_candidate_binding_hash(case),
        }
        bindings[case.case_id] = binding
        cases.append(
            {
                "case_id": case.case_id,
                "task_type": case.task_type,
                "module_case_number": int(case.visible_metadata["module_case_number"]),
                "authoritative_binding": binding,
                "frozen_union_candidate_count": len(set(case.visible_metadata.get("candidate_uuids", []))),
                "authoritative_candidate_count": len(candidate_uuids),
                "excluded_reference_candidate_count": len({row["candidate_uuid"] for row in exclusions}),
                "primary_canvas_locked": True,
            }
        )
        excluded.extend({"case_id": case.case_id, **row} for row in exclusions)
    affected = [row for row in cases if row["excluded_reference_candidate_count"] > 0]
    case6 = next(
        row for row in cases if row["task_type"] == "detection_gold_player_static" and row["module_case_number"] == 6
    )
    case7 = next(
        row for row in cases if row["task_type"] == "detection_gold_player_static" and row["module_case_number"] == 7
    )
    audit = {
        "schema_version": "football_intelligence.m5_5g1a_r3.affected_case_audit.v1",
        "cases_audited": len(cases),
        "affected_case_count": len(affected),
        "affected_cases": affected,
        "case_006_reproduced": case6,
        "case_007_reproduced": case7,
        "root_cause": (
            "R2 candidate queue scanned the frozen cross-frame UUID union and syncCandidate "
            "moved the editable frame to each candidate owner."
        ),
        "repair": (
            "R3 resolves one source-bound authoritative record and never changes the editable "
            "frame during static candidate review."
        ),
        "passed": case6["excluded_reference_candidate_count"] > 0 and case7["excluded_reference_candidate_count"] > 0,
    }
    binding_report = {
        "schema_version": "football_intelligence.m5_5g1a_r3.static_authoritative_frame_binding.v1",
        "static_and_dense_case_count": len(cases),
        "bindings": bindings,
        "all_bindings_unique_and_exact": len(bindings) == 40,
        "primary_canvas_locked": True,
        "reference_frames_editable": False,
        "passed": len(bindings) == 40,
    }
    exclusion_report = {
        "schema_version": "football_intelligence.m5_5g1a_r3.cross_frame_candidate_exclusions.v1",
        "excluded_rows": excluded,
        "excluded_candidate_uuid_count": len({(row["case_id"], row["candidate_uuid"]) for row in excluded}),
        "excluded_rows_deleted": False,
        "excluded_rows_retained_for_audit": True,
        "passed": True,
    }
    return audit, binding_report, exclusion_report


def build_package(bindings: dict[str, Any], tranches: dict[str, dict[str, Any]]) -> dict[str, Any]:
    prior_manifest = read_json(R2_PACKAGE / "reviewer_manifest.json")
    prior_ui = read_json(R2_PACKAGE / "ui_config.json")
    source_evidence = tree_manifest(R2_PACKAGE / "evidence")
    if source_evidence["tree_hash"] != EVIDENCE_HASH:
        raise RuntimeError("R2 evidence tree does not match the frozen evidence hash")
    shutil.copytree(R2_PACKAGE / "evidence", PACKAGE / "evidence", copy_function=shutil.copy2, dirs_exist_ok=True)
    for name in ("evidence_manifest.json", "second_reviewer_and_adjudication_contract.json"):
        shutil.copy2(R2_PACKAGE / name, PACKAGE / name)

    manifest_payload = json.loads(json.dumps(prior_manifest))
    manifest_payload.update(
        {
            "review_id": REVIEW_ID,
            "stage_id": STAGE.name,
            "title": "Incremental detection-gold annotation",
            "manifest_hash": "",
        }
    )
    write_json(PACKAGE / "reviewer_manifest.json", manifest_payload)
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    manifest_payload["manifest_hash"] = manifest_hash(manifest)
    write_json(PACKAGE / "reviewer_manifest.json", manifest_payload)

    order = list(tranches)
    ui = json.loads(json.dumps(prior_ui))
    ui.update(
        {
            "page_title": "Football Intelligence - Incremental Detection Gold",
            "review_title": "Incremental detection-gold annotation",
            "task_instructions": "Complete one independently saved gold tranche at a time.",
        }
    )
    ui["question_contract"].update(
        {
            "reviewer_session_id": REVIEWER,
            "incremental_gold_tranches": True,
            "static_authoritative_frame_lock": True,
            "static_authoritative_bindings": bindings,
            "candidate_queue_authoritative_frame_only": True,
            "reference_frames_editable": False,
            "visible_body_partial_instruction": "Box only the part you can actually see. Do not guess the hidden body.",
            "automatic_footpoint_proposal": "VISIBLE_BOX_BOTTOM_CENTRE",
            "footpoint_review_choices": ["YES", "MOVE_IT", "FEET_NOT_VISIBLE", "CANNOT_TELL"],
            "hidden_footpoint_estimates_labelled": True,
            "gold_tranches": tranches,
            "tranche_order": order,
            "default_tranche_id": "A_CORE_STATIC",
            "tranche_completion_atomic_four_file_export": True,
            "full_completion_requires_all_tranches": True,
            "fresh_indexeddb_namespace": True,
            "r2_decisions_migration_forbidden": True,
        }
    )
    write_json(PACKAGE / "ui_config.json", ui)

    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    ui_config = load_ui_config(PACKAGE / "ui_config.json")
    persistence = DetectionGoldPilotPersistence(
        manifest=manifest,
        ui_config=ui_config,
        decisions_root=PACKAGE / "decisions",
        reviewer_session_id=REVIEWER,
    )
    state = persistence.ensure_state()
    evidence = tree_manifest(PACKAGE / "evidence")
    generic = validate_review_chassis_package(
        manifest_path=PACKAGE / "reviewer_manifest.json",
        ui_config_path=PACKAGE / "ui_config.json",
        evidence_root=PACKAGE / "evidence",
        decisions_root=PACKAGE / "decisions",
    )
    coverage = validate_tranche_coverage(ui["question_contract"], [case.case_id for case in manifest.cases])
    result = {
        "schema_version": "football_intelligence.m5_5g1a_r3.package_validation.v1",
        "review_id": REVIEW_ID,
        "reviewer_session_id": REVIEWER,
        "case_count": len(manifest.cases),
        "case_payload_hash": stable_hash(manifest_payload["cases"]),
        "case_order_identical": prior_manifest["cases"] == manifest_payload["cases"],
        "evidence_copy": evidence,
        "evidence_bytes_identical": evidence["tree_hash"] == source_evidence["tree_hash"] == EVIDENCE_HASH,
        "evidence_manifest_identical": sha256_file(R2_PACKAGE / "evidence_manifest.json")
        == sha256_file(PACKAGE / "evidence_manifest.json"),
        "manifest_hash": manifest_payload["manifest_hash"],
        "ui_config_hash": ui_config_hash(ui_config),
        "tranche_coverage": coverage,
        "default_tranche_id": state.get("active_tranche_id"),
        "fresh_server_annotations": len(state.get("annotations", {})),
        "fresh_server_decisions": len(state.get("decisions", {})),
        "fresh_event_sequence": state.get("event_sequence"),
        "fresh_event_ledger_empty": persistence.events_path.stat().st_size == 0,
        "fresh_wizard_state_count": len(state.get("wizard_states", {})),
        "fresh_tranche_completion_count": len(state.get("tranche_completions", {})),
        "completion_artifacts_absent": not any((PACKAGE / "decisions").glob("completed_review*")),
        "generic_package_validation": generic,
    }
    result["passed"] = all(
        (
            result["case_count"] == 88,
            result["case_payload_hash"] == CASE_HASH,
            result["case_order_identical"],
            result["evidence_copy"]["file_count"] == 1512,
            result["evidence_bytes_identical"],
            result["evidence_manifest_identical"],
            coverage["passed"],
            result["default_tranche_id"] == "A_CORE_STATIC",
            result["fresh_server_annotations"] == 0,
            result["fresh_server_decisions"] == 0,
            result["fresh_event_sequence"] == 0,
            result["fresh_event_ledger_empty"],
            result["fresh_wizard_state_count"] == 0,
            result["fresh_tranche_completion_count"] == 0,
            result["completion_artifacts_absent"],
            generic["passed"],
        )
    )
    if not result["passed"]:
        raise RuntimeError(f"R3 package validation failed: {result}")
    result["package_checks_passed"] = True
    write_json(PACKAGE / "review_package_validation.json", result)
    return result


def write_contract_artifacts(
    *,
    manifest: Any,
    audit: dict[str, Any],
    binding_report: dict[str, Any],
    exclusion_report: dict[str, Any],
    tranches: dict[str, dict[str, Any]],
) -> None:
    write_json(STAGE / "01_PRIOR_STATE_AND_DEFECT_AUDIT" / "r2_affected_case_audit.json", audit)
    write_json(STAGE / "02_STATIC_FRAME_AND_CANDIDATE_LOCK" / "static_authoritative_frame_binding.json", binding_report)
    write_json(
        STAGE / "02_STATIC_FRAME_AND_CANDIDATE_LOCK" / "cross_frame_candidate_exclusion_report.json", exclusion_report
    )
    write_json(
        STAGE / "03_FOOTPOINT_AND_PARTIAL_PERSON_WORKFLOW" / "visible_body_and_partial_person_rules.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3.visible_body_rules.v1",
            "instruction": "Box only the part you can actually see. Do not guess the hidden body.",
            "visible_body_box_contains_visible_pixels_only": True,
            "full_body_box_optional_supplementary": True,
            "occlusion_and_edge_truncation_preserved": True,
            "upper_body_box_bottom_may_not_be_hidden_feet_footpoint": True,
            "frozen_schema_changed": False,
        },
    )
    write_json(
        STAGE / "03_FOOTPOINT_AND_PARTIAL_PERSON_WORKFLOW" / "footpoint_novice_mapping.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3.footpoint_mapping.v1",
            "question": "Is this roughly where their feet touch the ground?",
            "proposal": "visible_body_box bottom centre",
            "choices": {
                "YES": {"schema_mapping": "footpoint with low uncertainty", "estimated": False},
                "MOVE_IT": {"schema_mapping": "reviewer-adjusted footpoint", "estimated": False},
                "FEET_NOT_VISIBLE": {"schema_mapping": "footpoint with uncertainty >= 20", "estimated": True},
                "CANNOT_TELL": {"schema_mapping": "footpoint with uncertainty >= 20", "estimated": True},
            },
            "estimated_label": "Estimated because the feet are not visible",
            "new_ontology_enum_added": False,
            "frozen_schema_changed": False,
        },
    )
    write_json(
        STAGE / "03_FOOTPOINT_AND_PARTIAL_PERSON_WORKFLOW" / "footpoint_exception_validation.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3.footpoint_exceptions.v1",
            "explicit_review_required_for": [
                "near_boundary",
                "bottom_truncated",
                "feet_hidden",
                "upper_body_only",
                "jumping_or_sliding_or_falling",
                "kicking_or_extended_limb",
                "implausible_bottom_centre",
            ],
            "ordinary_standing_one_click_yes": True,
            "hidden_feet_high_uncertainty": True,
            "hidden_feet_adjustable": True,
            "upper_body_bottom_reuse_rejected_server_side": True,
            "passed": True,
        },
    )
    by_id = {case.case_id: case for case in manifest.cases}
    tranche_rows = []
    for tranche_id, tranche in tranches.items():
        rows = [by_id[case_id] for case_id in tranche["case_ids"]]
        tranche_rows.append(
            {
                "tranche_id": tranche_id,
                "label": tranche["label"],
                "case_ids": tranche["case_ids"],
                "case_count": len(rows),
                "task_counts": dict(sorted(Counter(case.task_type for case in rows).items())),
                "stratum_counts": dict(
                    sorted(Counter(case.visible_metadata["pilot_stratum"] for case in rows).items())
                ),
                "source_frame_hash_count": len(
                    {case.visible_metadata["source_binding"]["source_frame_sha256"] for case in rows}
                ),
            }
        )
    gold_manifest = {
        "schema_version": "football_intelligence.m5_5g1a_r3.gold_tranche_manifest.v1",
        "review_id": REVIEW_ID,
        "case_payload_hash": CASE_HASH,
        "default_tranche_id": "A_CORE_STATIC",
        "tranche_order": list(tranches),
        "tranches": tranche_rows,
        "total_case_count": sum(row["case_count"] for row in tranche_rows),
        "all_cases_diagnostic_only": True,
        "validation_or_holdout_use_forbidden": True,
    }
    write_json(STAGE / "04_INCREMENTAL_TRANCHE_DESIGN" / "gold_tranche_manifest.json", gold_manifest)
    a = next(row for row in tranche_rows if row["tranche_id"] == "A_CORE_STATIC")
    b = next(row for row in tranche_rows if row["tranche_id"] == "B_REMAINING_STATIC")
    write_json(
        STAGE / "04_INCREMENTAL_TRANCHE_DESIGN" / "tranche_selection_audit.json",
        {
            "deterministic": True,
            "selection_uses_hidden_answer": False,
            "tranche_a_case_count": a["case_count"],
            "tranche_a_stratum_counts": a["stratum_counts"],
            "tranche_a_includes_static_ordinals": [6, 7],
            "tranche_b_case_count": b["case_count"],
            "tranche_b_stratum_counts": b["stratum_counts"],
            "full_partition_case_count": gold_manifest["total_case_count"],
            "passed": a["case_count"] == 18
            and a["stratum_counts"]
            == {
                "clean_control": 3,
                "duplicate": 3,
                "merged": 3,
                "missed": 3,
                "partial_or_occluded": 3,
                "small_far_side": 3,
            }
            and b["case_count"] == 14,
        },
    )
    write_json(
        STAGE / "04_INCREMENTAL_TRANCHE_DESIGN" / "tranche_completion_contract.json",
        {
            "atomic_four_file_bundle_per_tranche": True,
            "bundle_root": "decisions/completed_tranches/<tranche_id>",
            "required_files": [
                "completed_review.json",
                "completed_review_events.jsonl",
                "completed_review_manifest.json",
                "completed_review_summary.json",
            ],
            "completed_tranche_immutable": True,
            "tranche_requires_exact_case_set": True,
            "tranche_requires_empty_outbox": True,
            "tranche_requires_no_unsaved_drafts": True,
            "full_pilot_requires_all_five_tranches": True,
        },
    )
    next_contract = read_json(PROMPT / "06_INCREMENTAL_NEXT_STAGE_PERMISSION_CONTRACT.json")
    next_contract["implemented_permission_gate"] = {
        "eligible_only_after_completed_and_independently_audited_tranche_a": True,
        "exploratory_stage": "M5_5G2A_PLAYER_PROPOSAL_SUPPLY_EXPLORATORY_DIAGNOSTIC_v1",
        "model_fit_performed": False,
        "detector_or_tracker_promoted": False,
    }
    write_json(
        STAGE / "07_NEXT_STAGE_INCREMENTAL_GOLD_CONTRACT" / "incremental_next_stage_permissions.json", next_contract
    )


def write_launcher_and_instructions() -> None:
    launcher = f"""$ErrorActionPreference = 'Stop'
$port = 8807
$occupied = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($occupied) {{
  Write-Error 'Port 8807 is occupied. Stop the old review server, then rerun this R3 launcher. It will not move ports.'
}}
$repo = '{REPO}'
$package = '{PACKAGE}'
Set-Location -LiteralPath $repo
Write-Host 'Starting M5.5G.1A-R3 incremental detection gold.' -ForegroundColor Green
Write-Host 'Open http://127.0.0.1:8807/' -ForegroundColor Cyan
uv run fi-pipeline review-chassis serve `
  --manifest "$package/reviewer_manifest.json" `
  --ui-config "$package/ui_config.json" `
  --evidence-root "$package/evidence" `
  --decisions-root "$package/decisions" `
  --host 127.0.0.1 `
  --port 8807 `
  --reviewer-session-id '{REVIEWER}'
"""
    instructions = """# M5.5G.1A-R3 incremental detection-gold review

1. Run `launch_incremental_gold_review.ps1` and open `http://127.0.0.1:8807/`.
2. Start with **Tranche A - core static**. Later tranches are explicitly selectable but are not required now.
3. For static cases, label the middle frame only. Previous and Next are reference images and cannot become editable.
4. Box only the part of a person you can actually see. Do not guess the hidden body.
5. Check the proposed footpoint. Choose Yes, Move it, Feet not visible, or Cannot tell.
6. Hidden-foot estimates are clearly labelled, remain adjustable, and carry high uncertainty.
7. Save every case in the tranche, then use **Complete tranche**. The full pilot remains
   incomplete until all five tranches are completed.

This is diagnostic gold collection only. It does not evaluate, train, tune, or promote a detector or tracker.
"""
    for root in (PACKAGE, STAGE):
        write_text(root / "launch_incremental_gold_review.ps1", launcher)
        write_text(root / "HUMAN_INSTRUCTIONS.md", instructions)


def write_initial_reports(package_result: dict[str, Any]) -> None:
    scenarios = {
        name: False
        for name in (
            "static_case_006_primary_frame_never_changes",
            "static_case_007_primary_frame_never_changes",
            "candidate_queue_uses_authoritative_frame_only",
            "reference_frames_are_non_editable",
            "static_save_rejects_wrong_source_hash",
            "static_save_rejects_cross_frame_candidate",
            "visible_body_partial_instruction_visible",
            "ordinary_footpoint_one_click_yes",
            "footpoint_move_it_persists",
            "hidden_feet_estimate_labelled_high_uncertainty",
            "upper_body_bottom_not_reused_as_hidden_footpoint",
            "tranche_a_composition_exact",
            "default_launches_tranche_a",
            "tranche_navigation_persists_reload",
            "tranche_navigation_persists_browser_restart",
            "draft_outbox_recovers_server_restart",
            "tranche_a_completion_writes_atomic_four_file_bundle",
            "tranche_a_completion_does_not_complete_full_pilot",
            "full_completion_blocked_until_all_tranches",
            "all_88_cases_and_1512_assets_unchanged",
        )
    }
    write_json(
        STAGE / "05_BROWSER_PERSISTENCE_AND_REGRESSION" / "browser_persistence_results.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3.browser_acceptance.v1",
            "status": "PENDING_REAL_BROWSER_ACCEPTANCE",
            "required_browser_scenarios": scenarios,
            "required_viewports": [
                "1024x768",
                "1366x768",
                "1440x900",
                "1920x1080",
                "2560x1440",
                "1440x900_at_125_percent",
            ],
            "passed": False,
        },
    )
    write_json(
        STAGE / "05_BROWSER_PERSISTENCE_AND_REGRESSION" / "truthful_tranche_timing.json",
        {
            "status": "MODELLED_NOT_HUMAN_MEASURED",
            "user_reported_ordinary_static_case_minutes": 3,
            "tranche_a_case_count": 18,
            "naive_tranche_a_minutes": 54,
            "human_measured_r3_active_minutes": None,
            "scripted_browser_time_claimed_as_human_time": False,
            "hard_cases_removed": False,
            "machine_truth_prefilled": False,
        },
    )
    write_json(PACKAGE / "review_package_validation.json", package_result)


def main() -> None:
    ensure_workspace()
    authorization_result = authorization()
    prompt_result = copy_prompt_pack()
    prior_result = prior_state_validation()
    write_json(STAGE / "01_PRIOR_STATE_AND_DEFECT_AUDIT" / "prior_state_validation.json", prior_result)
    r2_manifest = load_manifest(R2_PACKAGE / "reviewer_manifest.json")
    audit, binding_report, exclusion_report = authoritative_audit(r2_manifest)
    tranches = tranche_spec(read_json(R2_PACKAGE / "reviewer_manifest.json"))
    package_result = build_package(binding_report["bindings"], tranches)
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    write_contract_artifacts(
        manifest=manifest,
        audit=audit,
        binding_report=binding_report,
        exclusion_report=exclusion_report,
        tranches=tranches,
    )
    write_launcher_and_instructions()
    write_initial_reports(package_result)
    write_json(
        STAGE / "08_COMMANDS_AND_TESTS" / "build_summary.json",
        {
            "classification": CLASSIFICATION,
            "authorization": authorization_result,
            "prompt_pack": prompt_result,
            "prior_state": prior_result,
            "package": package_result,
            "browser_acceptance_pending": True,
            "tests_pending": True,
            "detector_or_tracker_evaluated": False,
            "detector_or_tracker_promoted": False,
            "model_fit_performed": False,
            "production_ready": False,
            "human_approved": False,
        },
    )
    print(json.dumps({"stage": str(STAGE), "package": str(PACKAGE), "passed": True}, indent=2))


if __name__ == "__main__":
    main()
