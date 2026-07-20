"""Build the bounded M5.5G.1A-R1 annotation-correctness repair workspace."""

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
PROMPT = PART3 / "M5_5G1A_R1_Codex_Prompt_Pack"
PRIOR_STAGE = PART3 / "M5_5G1A_DETECTION_GOLD_FOUNDATION_ONTOLOGY_FREEZE_AND_PILOT_ANNOTATION_v1"
PRIOR_PACKAGE = PRIOR_STAGE / "10_DETECTION_GOLD_PILOT_ANNOTATION_PACKAGE"
STAGE = PART3 / "M5_5G1A_R1_ANNOTATION_UI_CORRECTNESS_AND_PILOT_LAUNCH_REPAIR_v1"
PACKAGE = STAGE / "05_CORRECTED_DETECTION_GOLD_PILOT_ANNOTATION_PACKAGE"
BASELINE = "893e15959d43bee2e3ff9f609f71d4768c3cca5d"
REVIEW_ID = "m5_5g1a_detection_gold_pilot_v1_r1"
REVIEWER = "m5_5g1a_detection_gold_pilot_reviewer_r1"
CLASSIFICATION = "PASS_DETECTION_GOLD_PILOT_R1_READY"
SECTIONS = (
    "00_PROMPT_AND_INPUTS",
    "01_AUDIT_AND_PRIOR_STAGE_VERIFICATION",
    "02_UI_CORRECTNESS_REPAIR",
    "03_BROWSER_AND_PERSISTENCE_REGRESSION",
    "04_TIMING_AND_HUMAN_INSTRUCTIONS",
    "05_CORRECTED_DETECTION_GOLD_PILOT_ANNOTATION_PACKAGE",
    "06_COMMANDS_AND_TESTS",
    "07_REVIEW_PACK_FOR_CHATGPT",
    "_tmp",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def safe_path(path: Path) -> str:
    return f"<FOOTBALL_INTELLIGENCE_ROOT>/{path.resolve().relative_to(ROOT.resolve()).as_posix()}"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def rows_hash(rows: list[dict[str, Any]]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def tree_manifest(root: Path, *, include_files: bool = False) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    result: dict[str, Any] = {
        "root": safe_path(root),
        "file_count": len(rows),
        "total_bytes": sum(row["size_bytes"] for row in rows),
        "tree_hash": rows_hash(rows),
    }
    if include_files:
        result["files"] = rows
    return result


def prepare_workspace() -> None:
    if STAGE.exists():
        decisions = PACKAGE / "decisions"
        state_path = decisions / "review_decisions.json"
        events_path = decisions / "review_decision_events.jsonl"
        state = read_json(state_path) if state_path.exists() else {}
        human_work = bool(state.get("annotations") or state.get("decisions") or state.get("completed"))
        human_work = human_work or (events_path.exists() and events_path.stat().st_size > 0)
        human_work = human_work or any(decisions.glob("completed_review*"))
        if human_work:
            raise RuntimeError("R1 decisions contain human work; refusing to rebuild")
        shutil.rmtree(STAGE)
    for section in SECTIONS:
        (STAGE / section).mkdir(parents=True, exist_ok=True)


def verify_authorization() -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    baseline_exists = (
        subprocess.run(["git", "cat-file", "-e", f"{BASELINE}^{{commit}}"], cwd=REPO, check=False).returncode == 0
    )
    ancestor = (
        subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE, head], cwd=REPO, check=False).returncode == 0
    )
    result = {
        "authorized_baseline": BASELINE,
        "head_at_build": head,
        "branch": branch,
        "baseline_exists": baseline_exists,
        "baseline_is_ancestor": ancestor,
        "intervening_commit_count": int(git("rev-list", "--count", f"{BASELINE}..{head}")),
        "expected_origin": "https://github.com/sebgreenhalgh/Football-Intelligence.git",
        "actual_origin": git("remote", "get-url", "origin"),
    }
    result["passed"] = all(
        (
            result["baseline_exists"],
            result["baseline_is_ancestor"],
            result["branch"] == "main",
            result["actual_origin"] == result["expected_origin"],
        )
    )
    if not result["passed"]:
        raise RuntimeError(f"repository authorization failed: {result}")
    return result


def copy_and_validate_prompt() -> dict[str, Any]:
    destination = STAGE / "00_PROMPT_AND_INPUTS"
    source_manifest = read_json(PROMPT / "08_PROMPT_PACK_MANIFEST.json")
    checks = []
    for entry in source_manifest["files"]:
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
    shutil.copy2(PROMPT / "08_PROMPT_PACK_MANIFEST.json", destination / "08_PROMPT_PACK_MANIFEST.json")
    result = {
        "source_manifest_sha256": sha256_file(PROMPT / "08_PROMPT_PACK_MANIFEST.json"),
        "file_count": 9,
        "checks": checks,
        "passed": len(checks) == 8 and all(row["size_match"] and row["sha256_match"] for row in checks),
    }
    if not result["passed"]:
        raise RuntimeError("R1 prompt pack failed integrity validation")
    write_json(destination / "prompt_copy_validation.json", result)
    return result


def prior_frozen_hashes() -> dict[str, Any]:
    freeze_root = PRIOR_STAGE / "03_GOLD_ONTOLOGY_AND_SCHEMA_FREEZE"
    freeze = read_json(freeze_root / "schema_freeze_manifest.json")
    schema_checks = []
    for row in freeze["schemas"]:
        path = freeze_root / row["name"]
        schema_checks.append(
            {
                "name": row["name"],
                "expected_sha256": row["sha256"],
                "actual_sha256": sha256_file(path),
                "match": sha256_file(path) == row["sha256"],
            }
        )
    critical = {
        "schema_freeze_manifest": freeze_root / "schema_freeze_manifest.json",
        "matching_specification": PRIOR_STAGE
        / "04_MATCHING_METRICS_AND_ACCEPTANCE_GATES"
        / "matching_specification.json",
        "future_metric_schema": PRIOR_STAGE / "04_MATCHING_METRICS_AND_ACCEPTANCE_GATES" / "future_metric_schema.json",
        "frozen_acceptance_gates": PRIOR_STAGE
        / "04_MATCHING_METRICS_AND_ACCEPTANCE_GATES"
        / "frozen_acceptance_gates.json",
        "pilot_case_manifest": PRIOR_STAGE / "05_PILOT_CASE_SELECTION_AND_BINDING" / "pilot_case_manifest.json",
        "case_binding_validation": PRIOR_STAGE / "05_PILOT_CASE_SELECTION_AND_BINDING" / "case_binding_validation.json",
        "evidence_manifest": PRIOR_PACKAGE / "evidence_manifest.json",
    }
    result = {
        "ontology_version": freeze["ontology_version"],
        "freeze_hash": freeze["freeze_hash"],
        "expected_freeze_hash": "81c256cae533a983970926cb7acfa8a090ac12629166a17181c0990877e92a8b",
        "schema_checks": schema_checks,
        "critical_files": {
            name: {"path": safe_path(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in critical.items()
        },
    }
    result["passed"] = (
        result["freeze_hash"] == result["expected_freeze_hash"]
        and result["ontology_version"] == "m5_5g1a_detection_gold_v1"
        and all(row["match"] for row in schema_checks)
    )
    if not result["passed"]:
        raise RuntimeError("frozen G.1A ontology or schema hashes changed")
    return result


def build_corrected_package() -> dict[str, Any]:
    prior_manifest_payload = read_json(PRIOR_PACKAGE / "reviewer_manifest.json")
    prior_ui_payload = read_json(PRIOR_PACKAGE / "ui_config.json")
    prior_evidence = tree_manifest(PRIOR_PACKAGE / "evidence")
    shutil.copytree(PRIOR_PACKAGE / "evidence", PACKAGE / "evidence", copy_function=shutil.copy2)
    shutil.copy2(PRIOR_PACKAGE / "evidence_manifest.json", PACKAGE / "evidence_manifest.json")
    shutil.copy2(
        PRIOR_PACKAGE / "second_reviewer_and_adjudication_contract.json",
        PACKAGE / "second_reviewer_and_adjudication_contract.json",
    )

    manifest_payload = json.loads(json.dumps(prior_manifest_payload))
    manifest_payload.update(
        {
            "review_id": REVIEW_ID,
            "stage_id": STAGE.name,
            "title": "Detection-gold diagnostic pilot R1",
            "manifest_hash": "",
        }
    )
    manifest = load_manifest_from_payload(manifest_payload, PACKAGE / "reviewer_manifest.json")
    manifest_payload["manifest_hash"] = manifest_hash(manifest)
    write_json(PACKAGE / "reviewer_manifest.json", manifest_payload)

    ui_payload = json.loads(json.dumps(prior_ui_payload))
    ui_payload["page_title"] = "Football Intelligence - Detection Gold Pilot R1"
    ui_payload["review_title"] = "Detection-gold diagnostic pilot R1"
    ui_payload["task_instructions"] = (
        "Annotate the focal ROI with explicit human-object and candidate-target selection."
    )
    ui_payload["question_contract"].update(
        {
            "reviewer_session_id": REVIEWER,
            "r1_annotation_correctness": True,
            "annotation_scope": "FOCAL_ROI_ONLY_FOR_STATIC_AND_DENSE",
            "proposal_assistance_is_truth": False,
            "explicit_candidate_target_selection": True,
            "human_measured_active_minutes": None,
        }
    )
    write_json(PACKAGE / "ui_config.json", ui_payload)

    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    ui_config = load_ui_config(PACKAGE / "ui_config.json")
    persistence = DetectionGoldPilotPersistence(
        manifest=manifest,
        ui_config=ui_config,
        decisions_root=PACKAGE / "decisions",
        reviewer_session_id=REVIEWER,
    )
    state = persistence.ensure_state()

    launcher = f"""$ErrorActionPreference = 'Stop'
$port = 8807
$occupied = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($occupied) {{
  $message = 'Port 8807 is occupied. Stop the old review server, then rerun this R1 launcher.'
  Write-Error "$message The launcher will not move ports."
}}
$repo = '{REPO}'
$package = '{PACKAGE}'
Set-Location -LiteralPath $repo
Write-Host 'Starting the corrected M5.5G.1A-R1 package only.' -ForegroundColor Green
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
    write_text(PACKAGE / "launch_corrected_review.ps1", launcher)

    copied_evidence = tree_manifest(PACKAGE / "evidence")
    case_hash_before = stable_hash(prior_manifest_payload["cases"])
    case_hash_after = stable_hash(manifest_payload["cases"])
    evidence_manifest_identical = sha256_file(PRIOR_PACKAGE / "evidence_manifest.json") == sha256_file(
        PACKAGE / "evidence_manifest.json"
    )
    package_validation = validate_review_chassis_package(
        manifest_path=PACKAGE / "reviewer_manifest.json",
        ui_config_path=PACKAGE / "ui_config.json",
        evidence_root=PACKAGE / "evidence",
        decisions_root=PACKAGE / "decisions",
    )
    result = {
        "schema_version": "football_intelligence.m5_5g1a_r1.corrected_package_validation.v1",
        "review_id": REVIEW_ID,
        "reviewer_session_id": REVIEWER,
        "case_count": len(manifest.cases),
        "case_order_identical": [case["case_id"] for case in prior_manifest_payload["cases"]]
        == [case["case_id"] for case in manifest_payload["cases"]],
        "case_payload_hash_before": case_hash_before,
        "case_payload_hash_after": case_hash_after,
        "case_payload_identical": case_hash_before == case_hash_after,
        "evidence_manifest_sha256_before": sha256_file(PRIOR_PACKAGE / "evidence_manifest.json"),
        "evidence_manifest_sha256_after": sha256_file(PACKAGE / "evidence_manifest.json"),
        "evidence_manifest_identical": evidence_manifest_identical,
        "evidence_before": prior_evidence,
        "evidence_after": copied_evidence,
        "evidence_bytes_identical": prior_evidence["tree_hash"] == copied_evidence["tree_hash"],
        "manifest_hash": manifest_payload["manifest_hash"],
        "ui_config_hash": ui_config_hash(ui_config),
        "fresh_decisions_empty": not state["annotations"] and not state["decisions"],
        "fresh_event_sequence_zero": state["event_sequence"] == 0,
        "fresh_event_ledger_empty": persistence.events_path.stat().st_size == 0,
        "completion_artifacts_absent": not any((PACKAGE / "decisions").glob("completed_review*")),
        "generic_package_validation": package_validation,
    }
    result["passed"] = all(
        (
            result["case_count"] == 88,
            result["case_order_identical"],
            result["case_payload_identical"],
            result["evidence_manifest_identical"],
            result["evidence_bytes_identical"],
            result["fresh_decisions_empty"],
            result["fresh_event_sequence_zero"],
            result["fresh_event_ledger_empty"],
            result["completion_artifacts_absent"],
            package_validation["passed"],
        )
    )
    if not result["passed"]:
        raise RuntimeError(f"corrected package validation failed: {result}")
    write_json(PACKAGE / "review_package_validation.json", result)
    write_json(
        PACKAGE / "reviewer_manifest_summary.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r1.reviewer_manifest.v1",
            "review_id": REVIEW_ID,
            "reviewer_session_id": REVIEWER,
            "url": "http://127.0.0.1:8807/",
            "case_count": 88,
            "module_counts": dict(Counter(case.task_type for case in manifest.cases)),
            "diagnostic_only": True,
            "fresh_decisions_root": True,
            "use_original_package_forbidden": True,
        },
    )
    return result


def load_manifest_from_payload(payload: dict[str, Any], temporary_path: Path):
    write_json(temporary_path, payload)
    return load_manifest(temporary_path)


def write_repair_artifacts(package_result: dict[str, Any]) -> None:
    defect_rows = [
        ("temporal_manual_candidate_uuid", "fixed", "manual OBSERVED geometry saves candidate_uuids=[]"),
        ("implicit_candidate_targets", "fixed", "all relation targets are explicit checkboxes"),
        ("last_object_only_editing", "fixed", "UUID-backed player and mask selectors edit/remove any object"),
        ("dense_coverage_not_persisted", "fixed", "coverage input independently updates IndexedDB draft"),
        ("proposal_assistance_created_truth", "fixed", "geometry drafts start unresolved and unbound"),
        ("focal_scope_unclear", "fixed", "persistent badge and panorama ROI overlay"),
        ("timing_described_as_measured", "fixed", "human_measured_active_minutes remains null"),
    ]
    write_json(
        STAGE / "02_UI_CORRECTNESS_REPAIR" / "confirmed_defect_disposition.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r1.defect_disposition.v1",
            "all_confirmed_defects_addressed": True,
            "rows": [
                {"defect": defect, "status": status, "disposition": disposition}
                for defect, status, disposition in defect_rows
            ],
            "schema_migration_performed": False,
            "detector_or_tracker_changed": False,
        },
    )
    write_json(
        STAGE / "02_UI_CORRECTNESS_REPAIR" / "explicit_selection_and_validation_design.json",
        {
            "player_and_mask_selection": "stable annotation UUID; selected state visible in overlay and side panel",
            "candidate_target_binding": {
                "BACKGROUND": 0,
                "CLEAN_SINGLE_INSTANCE": 1,
                "DUPLICATE_OF_INSTANCE": 1,
                "PARTIAL_INSTANCE": 1,
                "MERGED_MULTIPLE_INSTANCES": "explicit subset with at least 2",
                "AMBIGUOUS": "0, 1, or many",
            },
            "removal_policy": "confirmed removal clears affected bindings and forces explicit re-review",
            "proposal_defaults": {
                "visibility_state": "UNRESOLVED",
                "occlusion_type": "UNKNOWN",
                "pitch_state": "BOUNDARY_UNCERTAIN",
                "coarse_role": "UNKNOWN",
                "mask_quality": "UNCERTAIN",
                "automatic_candidate_relation": None,
            },
            "save_validation": [
                "complete candidate coverage",
                "target cardinality and live references",
                "temporal frame/hash/candidate binding",
                "original-pixel geometry bounds",
                "dense coverage semantics",
                "full-strip review gates",
                "exact immutable source binding",
            ],
            "corrected_package_validation_sha256": sha256_file(PACKAGE / "review_package_validation.json"),
            "package_passed": package_result["passed"],
        },
    )


def write_timing_and_instructions() -> None:
    timing = {
        "schema_version": "football_intelligence.m5_5g1a_r1.timing_estimate.v1",
        "status": "MODELLED_NOT_HUMAN_MEASURED",
        "modelled_estimated_active_minutes": 56.4,
        "human_measured_active_minutes": None,
        "target_range_minutes": [30, 50],
        "estimate_above_target_range": True,
        "hard_cases_removed": False,
        "machine_truth_prefilled": False,
        "recommendation": "Complete one module at a time and take breaks; do not weaken the annotation task.",
        "seconds_per_action_assumptions": {
            "inspect_candidate_or_visible_person": 3.0,
            "select_or_draw_human_geometry": 8.0,
            "review_semantic_fields": 6.0,
            "bind_candidate_relation_and_targets": 5.0,
            "draw_and_review_dense_mask": 28.0,
            "review_temporal_or_football_frame": 3.0,
            "save_and_navigation": 2.0,
        },
        "per_module_action_counts": {
            "player_static": {"cases": 32, "modelled_seconds_per_case": 24, "modelled_minutes": 12.8},
            "dense_region": {"cases": 8, "modelled_seconds_per_case": 90, "modelled_minutes": 12.0},
            "temporal_player": {
                "cases": 12,
                "frames_per_case": 11,
                "modelled_seconds_per_case": 70,
                "modelled_minutes": 14.0,
            },
            "pitch_boundary": {"cases": 12, "modelled_seconds_per_case": 18, "modelled_minutes": 3.6},
            "football_burst": {
                "cases": 24,
                "frames_per_case": 9,
                "modelled_seconds_per_case": 35,
                "modelled_minutes": 14.0,
            },
        },
        "actual_active_human_time_will_be_measured_during_pilot": True,
    }
    write_json(STAGE / "04_TIMING_AND_HUMAN_INSTRUCTIONS" / "truthful_timing_report.json", timing)
    instructions = """# M5.5G.1A-R1 human annotation instructions

1. Use only the corrected R1 package at `http://127.0.0.1:8807/`.
2. Do not launch or annotate in the original non-R1 G.1A package.
3. For static and dense cases, annotate only the focal ROI. Panorama and adjacent frames are context.
4. Selecting a proposal creates draft geometry, not truth. Resolve its semantic fields deliberately.
5. Select the intended person or mask explicitly before editing geometry.
6. Select the intended human target subset explicitly before binding a machine candidate.
7. Review every enabled proposal layer and every visible person in the focal ROI
   before using the bulk-background action.
8. Review every temporal or football frame before accepting a stable run or burst-wide no-ball state.
9. Notes are optional.
10. Stop immediately if `Saved to server` disappears or the server event sequence stops advancing.

The `56.4` minute estimate is modelled, not human-measured. Actual active time will be recorded by this pilot.
"""
    write_text(STAGE / "04_TIMING_AND_HUMAN_INSTRUCTIONS" / "HUMAN_INSTRUCTIONS.md", instructions)
    write_text(PACKAGE / "HUMAN_INSTRUCTIONS.md", instructions)


def main() -> None:
    authorization = verify_authorization()
    prepare_workspace()
    prompt_validation = copy_and_validate_prompt()
    prior_before = tree_manifest(PRIOR_STAGE)
    frozen = prior_frozen_hashes()
    package = build_corrected_package()
    write_repair_artifacts(package)
    write_timing_and_instructions()
    prior_after = tree_manifest(PRIOR_STAGE)
    preservation = {
        "schema_version": "football_intelligence.m5_5g1a_r1.prior_stage_preservation.v1",
        "prior_stage": safe_path(PRIOR_STAGE),
        "before": prior_before,
        "after": prior_after,
        "byte_identical": prior_before["tree_hash"] == prior_after["tree_hash"],
        "frozen_hashes": frozen,
    }
    preservation["passed"] = preservation["byte_identical"] and frozen["passed"]
    if not preservation["passed"]:
        raise RuntimeError("prior G.1A workspace changed during R1 build")
    write_json(
        STAGE / "01_AUDIT_AND_PRIOR_STAGE_VERIFICATION" / "repository_authorization.json",
        authorization,
    )
    write_json(
        STAGE / "01_AUDIT_AND_PRIOR_STAGE_VERIFICATION" / "prior_stage_preservation_validation.json",
        preservation,
    )
    write_json(
        STAGE / "01_AUDIT_AND_PRIOR_STAGE_VERIFICATION" / "frozen_hash_validation.json",
        frozen,
    )
    write_json(
        STAGE / "06_COMMANDS_AND_TESTS" / "build_summary.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r1.build_summary.v1",
            "classification": CLASSIFICATION,
            "prompt_validation_passed": prompt_validation["passed"],
            "prior_stage_preserved": preservation["passed"],
            "frozen_hashes_preserved": frozen["passed"],
            "corrected_package_passed": package["passed"],
            "case_count": package["case_count"],
            "evidence_asset_count": package["evidence_after"]["file_count"],
            "schema_migration_performed": False,
            "detector_or_tracker_evaluated": False,
            "detector_or_tracker_promoted": False,
            "production_ready": False,
            "no_auto_promotion": True,
            "human_approved": False,
            "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "stage": str(STAGE),
                "package": str(PACKAGE),
                "case_count": package["case_count"],
                "evidence_files": package["evidence_after"]["file_count"],
                "prior_stage_preserved": preservation["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
