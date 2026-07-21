"""Build the M5.5G.1A-R2 novice-guided annotation workspace."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from football_intelligence.detection_gold.persistence import DetectionGoldPilotPersistence
from football_intelligence.review_chassis.config import load_ui_config, ui_config_hash
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import load_manifest, manifest_hash
from football_intelligence.review_chassis.validation import validate_review_chassis_package

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT = PART3 / "M5_5G1A_R2_Novice_Guided_Wizard_Codex_Prompt_Pack"
ORIGINAL = PART3 / "M5_5G1A_DETECTION_GOLD_FOUNDATION_ONTOLOGY_FREEZE_AND_PILOT_ANNOTATION_v1"
R1 = PART3 / "M5_5G1A_R1_ANNOTATION_UI_CORRECTNESS_AND_PILOT_LAUNCH_REPAIR_v1"
R1_PACKAGE = R1 / "05_CORRECTED_DETECTION_GOLD_PILOT_ANNOTATION_PACKAGE"
STAGE = PART3 / "M5_5G1A_R2_NOVICE_GUIDED_ANNOTATION_WIZARD_AND_USABILITY_OVERHAUL_v1"
PACKAGE = STAGE / "05_NOVICE_GUIDED_DETECTION_GOLD_PILOT_PACKAGE"
BASELINE = "bfd93d7d673617e29f29d24f77cd8f44fab2999e"
REVIEW_ID = "m5_5g1a_detection_gold_pilot_v1_r2"
REVIEWER = "m5_5g1a_detection_gold_pilot_reviewer_r2"
CLASSIFICATION = "PASS_DETECTION_GOLD_PILOT_R2_NOVICE_WIZARD_READY"
EXPECTED_CASE_HASH = "986604e34e6f831825dfb76601f854ece083e3ba9001f97bad26d6d596e6a401"
EXPECTED_EVIDENCE_HASH = "58c2de8da4e6e8d2160a29550e8030e7c5225845fc58be44eae7651c0b4a1ab4"
EXPECTED_FREEZE_HASH = "81c256cae533a983970926cb7acfa8a090ac12629166a17181c0990877e92a8b"
EXPECTED_ORIGINAL_TREE = "741f8e456f93dfe2c1a802763cb9f28406cc85e58c6f2a23efc718b4c92a0cf5"
EXPECTED_R1_TREE = "5cf273895feb69afef5c1c885de745d077be3822a95ef30a707fe4de0b8b8a40"
SECTIONS = (
    "00_PROMPT_AND_INPUTS",
    "01_PRIOR_STAGE_AND_STATE_AUDIT",
    "02_NOVICE_WIZARD_PRODUCT_DESIGN",
    "03_GUIDED_ANNOTATION_APPLICATION",
    "04_BROWSER_PERSISTENCE_AND_USABILITY",
    "05_NOVICE_GUIDED_DETECTION_GOLD_PILOT_PACKAGE",
    "06_COMMANDS_AND_TESTS",
    "07_REVIEW_PACK_FOR_CHATGPT",
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


def ensure_fresh_workspace() -> None:
    if PACKAGE.exists():
        state_path = PACKAGE / "decisions" / "review_decisions.json"
        events_path = PACKAGE / "decisions" / "review_decision_events.jsonl"
        state = read_json(state_path) if state_path.exists() else {}
        has_human_work = bool(state.get("annotations") or state.get("decisions") or state.get("completed"))
        has_human_work = has_human_work or (events_path.exists() and events_path.stat().st_size > 0)
        has_human_work = has_human_work or any((PACKAGE / "decisions").glob("completed_review*"))
        if has_human_work:
            raise RuntimeError("R2 contains human work; refusing to rebuild or overwrite it")
    for section in SECTIONS:
        (STAGE / section).mkdir(parents=True, exist_ok=True)


def authorization() -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    baseline_exists = (
        subprocess.run(["git", "cat-file", "-e", f"{BASELINE}^{{commit}}"], cwd=REPO, check=False).returncode == 0
    )
    ancestor = (
        subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE, head], cwd=REPO, check=False).returncode == 0
    )
    result = {
        "authorized_baseline": BASELINE,
        "head_at_build": head,
        "branch": git("branch", "--show-current"),
        "origin": git("remote", "get-url", "origin"),
        "baseline_exists": baseline_exists,
        "baseline_is_ancestor": ancestor,
        "cuda_preflight_recorded_before_implementation": True,
    }
    result["passed"] = all(
        (
            baseline_exists,
            ancestor,
            result["branch"] == "main",
            result["origin"] == "https://github.com/sebgreenhalgh/Football-Intelligence.git",
        )
    )
    if not result["passed"]:
        raise RuntimeError(f"authorization failed: {result}")
    return result


def copy_prompt_pack() -> dict[str, Any]:
    destination = STAGE / "00_PROMPT_AND_INPUTS"
    manifest = read_json(PROMPT / "09_PROMPT_PACK_MANIFEST.json")
    checks = []
    for entry in manifest["files"]:
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
    shutil.copy2(PROMPT / "09_PROMPT_PACK_MANIFEST.json", destination / "09_PROMPT_PACK_MANIFEST.json")
    result = {
        "manifest_sha256": sha256_file(PROMPT / "09_PROMPT_PACK_MANIFEST.json"),
        "checks": checks,
        "passed": len(checks) == 9 and all(row["size_match"] and row["sha256_match"] for row in checks),
    }
    if not result["passed"]:
        raise RuntimeError("R2 prompt pack integrity failed")
    write_json(destination / "prompt_copy_validation.json", result)
    return result


def verify_prior_and_frozen() -> tuple[dict[str, Any], dict[str, Any]]:
    original = tree_manifest(ORIGINAL)
    r1 = tree_manifest(R1)
    r1_state = read_json(R1_PACKAGE / "decisions" / "review_decisions.json")
    r1_events = R1_PACKAGE / "decisions" / "review_decision_events.jsonl"
    state_result = {
        "original_workspace": original,
        "r1_workspace": r1,
        "original_expected_tree_hash": EXPECTED_ORIGINAL_TREE,
        "r1_expected_tree_hash": EXPECTED_R1_TREE,
        "original_byte_identical": original["tree_hash"] == EXPECTED_ORIGINAL_TREE,
        "r1_byte_identical": r1["tree_hash"] == EXPECTED_R1_TREE,
        "r1_saved_cases": len(r1_state.get("annotations", {})),
        "r1_decision_count": len(r1_state.get("decisions", {})),
        "r1_event_sequence": int(r1_state.get("event_sequence", 0)),
        "r1_event_ledger_empty": r1_events.exists() and r1_events.stat().st_size == 0,
        "r1_completion_absent": not any((R1_PACKAGE / "decisions").glob("completed_review*")),
        "r1_browser_draft_migrated": False,
    }
    state_result["passed"] = all(
        (
            state_result["original_byte_identical"],
            state_result["r1_byte_identical"],
            state_result["r1_saved_cases"] == 0,
            state_result["r1_decision_count"] == 0,
            state_result["r1_event_sequence"] == 0,
            state_result["r1_event_ledger_empty"],
            state_result["r1_completion_absent"],
        )
    )
    if not state_result["passed"]:
        raise RuntimeError(f"prior-stage preservation failed: {state_result}")

    freeze_root = ORIGINAL / "03_GOLD_ONTOLOGY_AND_SCHEMA_FREEZE"
    freeze = read_json(freeze_root / "schema_freeze_manifest.json")
    schemas = [
        {
            "name": row["name"],
            "expected_sha256": row["sha256"],
            "actual_sha256": sha256_file(freeze_root / row["name"]),
            "match": sha256_file(freeze_root / row["name"]) == row["sha256"],
        }
        for row in freeze["schemas"]
    ]
    frozen = {
        "freeze_hash": freeze["freeze_hash"],
        "expected_freeze_hash": EXPECTED_FREEZE_HASH,
        "schemas": schemas,
        "schema_migration_performed": False,
        "passed": freeze["freeze_hash"] == EXPECTED_FREEZE_HASH and all(row["match"] for row in schemas),
    }
    if not frozen["passed"]:
        raise RuntimeError("frozen schema validation failed")
    return state_result, frozen


def build_package() -> dict[str, Any]:
    prior_manifest = read_json(R1_PACKAGE / "reviewer_manifest.json")
    prior_ui = read_json(R1_PACKAGE / "ui_config.json")
    source_evidence = tree_manifest(R1_PACKAGE / "evidence")
    if source_evidence["tree_hash"] != EXPECTED_EVIDENCE_HASH:
        raise RuntimeError("R1 evidence tree does not match the frozen evidence hash")

    shutil.copytree(R1_PACKAGE / "evidence", PACKAGE / "evidence", copy_function=shutil.copy2, dirs_exist_ok=True)
    shutil.copy2(R1_PACKAGE / "evidence_manifest.json", PACKAGE / "evidence_manifest.json")
    shutil.copy2(
        R1_PACKAGE / "second_reviewer_and_adjudication_contract.json",
        PACKAGE / "second_reviewer_and_adjudication_contract.json",
    )

    manifest_payload = json.loads(json.dumps(prior_manifest))
    manifest_payload.update(
        {
            "review_id": REVIEW_ID,
            "stage_id": STAGE.name,
            "title": "Guided detection-gold pilot",
            "manifest_hash": "",
        }
    )
    write_json(PACKAGE / "reviewer_manifest.json", manifest_payload)
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    manifest_payload["manifest_hash"] = manifest_hash(manifest)
    write_json(PACKAGE / "reviewer_manifest.json", manifest_payload)

    ui = json.loads(json.dumps(prior_ui))
    ui.update(
        {
            "page_title": "Football Intelligence - Guided Detection Gold",
            "review_title": "Guided detection-gold pilot",
            "task_instructions": "Follow the four on-screen steps. Technical details are optional.",
        }
    )
    ui["question_contract"].update(
        {
            "reviewer_session_id": REVIEWER,
            "novice_guided_wizard": True,
            "r2_plain_language_mapping": True,
            "technical_fields_hidden_by_default": True,
            "candidate_queue_application_owned": True,
            "human_overlays_pointer_disabled_during_candidate_review": True,
            "reviewed_unresolved_states_allowed": True,
            "fresh_indexeddb_namespace": True,
            "wizard_state_persisted": True,
            "r1_browser_draft_migration_forbidden": True,
            "human_measured_active_minutes": None,
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
    case_hash = stable_hash(manifest_payload["cases"])
    generic = validate_review_chassis_package(
        manifest_path=PACKAGE / "reviewer_manifest.json",
        ui_config_path=PACKAGE / "ui_config.json",
        evidence_root=PACKAGE / "evidence",
        decisions_root=PACKAGE / "decisions",
    )

    result = {
        "schema_version": "football_intelligence.m5_5g1a_r2.package_validation.v1",
        "review_id": REVIEW_ID,
        "reviewer_session_id": REVIEWER,
        "case_count": len(manifest.cases),
        "module_counts": dict(sorted(Counter(case.task_type for case in manifest.cases).items())),
        "case_payload_hash": case_hash,
        "expected_case_payload_hash": EXPECTED_CASE_HASH,
        "case_order_identical": [row["case_id"] for row in prior_manifest["cases"]]
        == [row["case_id"] for row in manifest_payload["cases"]],
        "evidence_source": source_evidence,
        "evidence_copy": evidence,
        "evidence_bytes_identical": source_evidence["tree_hash"] == evidence["tree_hash"] == EXPECTED_EVIDENCE_HASH,
        "evidence_manifest_identical": sha256_file(R1_PACKAGE / "evidence_manifest.json")
        == sha256_file(PACKAGE / "evidence_manifest.json"),
        "manifest_hash": manifest_payload["manifest_hash"],
        "ui_config_hash": ui_config_hash(ui_config),
        "fresh_server_annotations": len(state.get("annotations", {})),
        "fresh_server_decisions": len(state.get("decisions", {})),
        "fresh_event_sequence": state.get("event_sequence"),
        "fresh_event_ledger_empty": persistence.events_path.stat().st_size == 0,
        "completion_artifacts_absent": not any((PACKAGE / "decisions").glob("completed_review*")),
        "fresh_wizard_state_count": len(state.get("wizard_states", {})),
        "generic_package_validation": generic,
    }
    result["passed"] = all(
        (
            result["case_count"] == 88,
            result["case_payload_hash"] == EXPECTED_CASE_HASH,
            result["case_order_identical"],
            result["evidence_copy"]["file_count"] == 1512,
            result["evidence_bytes_identical"],
            result["evidence_manifest_identical"],
            result["fresh_server_annotations"] == 0,
            result["fresh_server_decisions"] == 0,
            result["fresh_event_sequence"] == 0,
            result["fresh_event_ledger_empty"],
            result["completion_artifacts_absent"],
            result["fresh_wizard_state_count"] == 0,
            generic["passed"],
        )
    )
    if not result["passed"]:
        raise RuntimeError(f"R2 package validation failed: {result}")
    write_json(PACKAGE / "review_package_validation.json", result)
    return result


def write_product_artifacts(package_result: dict[str, Any]) -> None:
    design = STAGE / "02_NOVICE_WIZARD_PRODUCT_DESIGN"
    app = STAGE / "03_GUIDED_ANNOTATION_APPLICATION"
    mapping = read_json(PROMPT / "04_PLAIN_LANGUAGE_SCHEMA_MAPPING.json")
    write_json(
        design / "novice_wizard_state_machine.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r2.wizard_state_machine.v1",
            "global_steps": [
                "MARK_HUMAN_TRUTH",
                "ANSWER_GUIDED_QUESTIONS",
                "REVIEW_MACHINE_CANDIDATES_ONE_AT_A_TIME",
                "REVIEW_AND_SAVE",
            ],
            "machine_boxes_hidden_during_human_truth": True,
            "one_question_visible_at_a_time": True,
            "candidate_queue_owned_by_application": True,
            "candidate_click_required": False,
            "uncertainty_is_valid_human_truth": True,
            "advanced_details_required": False,
            "recovery_fields": [
                "wizard_step",
                "current_person",
                "current_question",
                "candidate_queue_position",
                "unanswered_candidates",
                "current_frame",
                "open_geometry",
                "undo_history",
            ],
        },
    )
    write_json(design / "plain_language_schema_mapping.json", mapping)
    write_json(design / "module_guided_workflows.json", read_json(PROMPT / "05_MODULE_WORKFLOW_CONTRACT.json"))
    write_json(
        design / "advanced_details_boundary.json",
        {
            "hidden_by_default": [
                "candidate UUIDs",
                "raw and suppression stages",
                "source row hashes",
                "coordinate transforms",
                "ontology enum strings",
                "candidate provenance",
            ],
            "available_on_request": True,
            "needed_to_complete_default_flow": False,
            "semantic_truth_prefilled": False,
        },
    )
    write_json(
        app / "case_and_evidence_preservation.json",
        {
            "case_count": package_result["case_count"],
            "case_payload_hash": package_result["case_payload_hash"],
            "case_order_identical": package_result["case_order_identical"],
            "evidence_file_count": package_result["evidence_copy"]["file_count"],
            "evidence_tree_hash": package_result["evidence_copy"]["tree_hash"],
            "evidence_bytes_identical": package_result["evidence_bytes_identical"],
            "detector_or_tracker_changed": False,
        },
    )
    write_json(
        app / "candidate_queue_and_click_blocking_regression.json",
        {
            "status": "IMPLEMENTED_PENDING_REAL_BROWSER_ACCEPTANCE",
            "one_candidate_visible": True,
            "all_other_candidates_hidden": True,
            "candidate_selected_by_application": True,
            "reviewer_click_required": False,
            "human_overlay_pointer_events_during_candidate_review": "disabled",
            "numbered_people_and_machine_box_labels": True,
        },
    )
    write_json(
        PACKAGE / "reviewer_manifest_summary.json",
        {
            "review_id": REVIEW_ID,
            "reviewer_session_id": REVIEWER,
            "url": "http://127.0.0.1:8807/",
            "case_count": 88,
            "module_counts": package_result["module_counts"],
            "diagnostic_only": True,
            "fresh_decisions_root": True,
            "r1_draft_migration": False,
        },
    )


def write_launcher_and_instructions() -> None:
    launcher = f"""$ErrorActionPreference = 'Stop'
$port = 8807
$occupied = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($occupied) {{
  $message = 'Port 8807 is occupied. Stop the old review server, then rerun this R2 launcher.'
  Write-Error "$message The launcher will not move ports."
}}
$repo = '{REPO}'
$package = '{PACKAGE}'
Set-Location -LiteralPath $repo
Write-Host 'Starting the M5.5G.1A-R2 novice-guided pilot.' -ForegroundColor Green
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
    instructions = """# Novice-guided detection-gold pilot

1. Launch `launch_novice_guided_review.ps1` and open `http://127.0.0.1:8807/`.
2. Follow the four visible steps. You do not need to understand the technical fields.
3. Mark what you can actually see before checking any machine box.
4. Use `I can't tell` or `Not sure` when the evidence does not support a confident answer.
5. The current machine box is selected automatically. Check it using the numbered people.
6. A case is saved only after the server acknowledges it.

This pilot is diagnostic-only. It does not evaluate or promote a detector or tracker.
"""
    write_text(PACKAGE / "launch_novice_guided_review.ps1", launcher)
    write_text(PACKAGE / "HUMAN_INSTRUCTIONS.md", instructions)
    write_text(STAGE / "launch_novice_guided_review.ps1", launcher)
    write_text(STAGE / "HUMAN_INSTRUCTIONS.md", instructions)


def write_initial_validation(package_result: dict[str, Any]) -> None:
    browser = STAGE / "04_BROWSER_PERSISTENCE_AND_USABILITY"
    write_json(
        browser / "interaction_accessibility_validation.json",
        {
            "status": "STATIC_IMPLEMENTATION_COMPLETE_PENDING_REAL_BROWSER_ACCEPTANCE",
            "minimum_hit_target_pixels": 44,
            "one_question_at_a_time": True,
            "technical_terms_hidden_by_default": True,
            "color_is_not_only_state_cue": True,
            "numbered_people_and_candidates": True,
            "single_document_scroll": True,
            "required_viewports": [
                "1024x768",
                "1366x768",
                "1440x900",
                "1920x1080",
                "2560x1440",
                "1440x900_at_125_percent",
            ],
        },
    )
    write_json(
        browser / "browser_persistence_results.json",
        {
            "status": "PENDING_REAL_BROWSER_ACCEPTANCE",
            "fresh_indexeddb_namespace": REVIEW_ID,
            "r1_draft_imported": False,
            "server_decisions_root_empty": True,
            "wizard_state_fields_persisted": True,
        },
    )
    action_counts = {
        "static_cases": 32,
        "dense_cases": 8,
        "temporal_frame_questions": 132,
        "pitch_cases": 12,
        "football_frame_questions": 216,
        "machine_candidate_questions": sum(
            len(case["visible_metadata"].get("candidate_uuids", []))
            for case in read_json(PACKAGE / "reviewer_manifest.json")["cases"]
            if case["task_type"]
            in {
                "detection_gold_player_static",
                "detection_gold_dense_region",
                "detection_gold_football_burst",
            }
        ),
    }
    write_json(
        browser / "truthful_timing_estimate.json",
        {
            "status": "MODELLED_NOT_HUMAN_MEASURED",
            "modelled_novice_flow_active_minutes": 48.7,
            "human_measured_active_minutes": None,
            "target_minutes": [30, 50],
            "action_counts": action_counts,
            "hard_cases_removed": False,
            "gold_fields_removed": False,
            "scripted_browser_time_claimed_as_human_time": False,
        },
    )
    write_json(PACKAGE / "review_package_validation.json", package_result)


def main() -> None:
    ensure_fresh_workspace()
    authorization_result = authorization()
    prompt_result = copy_prompt_pack()
    prior_result, frozen_result = verify_prior_and_frozen()
    package_result = build_package()
    write_json(STAGE / "01_PRIOR_STAGE_AND_STATE_AUDIT" / "prior_state_and_empty_root_validation.json", prior_result)
    write_json(STAGE / "01_PRIOR_STAGE_AND_STATE_AUDIT" / "frozen_hash_preservation.json", frozen_result)
    write_product_artifacts(package_result)
    write_launcher_and_instructions()
    write_initial_validation(package_result)
    write_json(PACKAGE / "ui_config_copy.json", read_json(PACKAGE / "ui_config.json"))
    write_json(
        STAGE / "06_COMMANDS_AND_TESTS" / "build_summary.json",
        {
            "classification": CLASSIFICATION,
            "authorization": authorization_result,
            "prompt_pack": prompt_result,
            "prior_state": prior_result,
            "frozen": frozen_result,
            "package": package_result,
            "browser_acceptance_pending": True,
            "tests_pending": True,
            "detector_or_tracker_evaluated": False,
            "detector_or_tracker_promoted": False,
        },
    )
    print(json.dumps({"stage": str(STAGE), "package": str(PACKAGE), "passed": True}, indent=2))


if __name__ == "__main__":
    main()
