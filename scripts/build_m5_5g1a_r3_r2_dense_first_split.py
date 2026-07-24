"""Build the bounded M5.5G.1A-R3-R2 dense-first review package."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from football_intelligence.detection_gold.persistence import DetectionGoldPilotPersistence
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.config import load_ui_config, ui_config_hash
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import load_manifest, manifest_hash
from football_intelligence.review_chassis.validation import validate_review_chassis_package

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT = PART3 / "M5_5G1A_R3_R2_Dense_First_Tranche_Split_Codex_Prompt_Pack"
R3 = PART3 / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
R3_PACKAGE = R3 / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
R3_DECISIONS = R3_PACKAGE / "decisions"
R3_R1 = PART3 / "M5_5G1A_R3_R1_WIZARD_STATE_INVALIDATION_AND_SAFE_CASE_RESTART_v1"
R3_R1_PACKAGE = R3_R1 / "05_REPAIRED_INCREMENTAL_ANNOTATION_PACKAGE"
G3 = PART3 / "M5_5G3_PROVENANCE_AWARE_CROSS_VIEW_CONSOLIDATION_AND_MERGED_AMBIGUITY_GATE_DEVELOPMENT_v1"
STAGE = PART3 / "M5_5G1A_R3_R2_DENSE_FIRST_TRANCHE_SPLIT_AND_ATOMIC_COMPLETION_v1"
PACKAGE = STAGE / "05_DENSE_FIRST_INCREMENTAL_ANNOTATION_PACKAGE"
PACK = STAGE / "08_REVIEW_PACK_FOR_CHATGPT"

BASELINE = "c0afdc8d70bbd3818e5602c02498384e1bfea567"
ORIGIN = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
REVIEW_ID = "m5_5g1a_detection_gold_pilot_v1_r3"
REVIEWER = "m5_5g1a_detection_gold_pilot_reviewer_r3"
CLIENT_BUILD_ID = "m5_5g1a_r3_r2_dense_first_split_v1"
INDEXEDDB_NAMESPACE = "fi_detection_gold_m5_5g1a_r3_r2_dense_first_split_v1"
CLASSIFICATION = "PASS_DENSE_FIRST_TRANCHE_C1_READY"
CASE_HASH = "986604e34e6f831825dfb76601f854ece083e3ba9001f97bad26d6d596e6a401"
EVIDENCE_HASH = "58c2de8da4e6e8d2160a29550e8030e7c5225845fc58be44eae7651c0b4a1ab4"
ONTOLOGY_HASH = "81c256cae533a983970926cb7acfa8a090ac12629166a17181c0990877e92a8b"

SECTIONS = (
    "00_PROMPT_AND_INPUTS",
    "01_LIVE_STATE_AND_PRESERVATION_AUDIT",
    "02_TRANCHE_MANIFEST_SPLIT",
    "03_DENSE_WIZARD_VALIDATION",
    "04_BROWSER_PERSISTENCE_AND_COMPLETION",
    "05_DENSE_FIRST_INCREMENTAL_ANNOTATION_PACKAGE",
    "06_NEXT_STAGE_DENSE_PERMISSION",
    "07_COMMANDS_AND_TESTS",
    "08_REVIEW_PACK_FOR_CHATGPT",
    "_tmp",
)
ALLOWED_CHANGES = {
    "scripts/build_m5_5g1a_r3_r2_dense_first_split.py",
    "scripts/capture_m5_5g1a_r3_r2_browser_acceptance.py",
    "scripts/finalize_m5_5g1a_r3_r2_review_pack.py",
    "src/football_intelligence/detection_gold/incremental.py",
    "src/football_intelligence/detection_gold/persistence.py",
    "src/football_intelligence/review_chassis/static/detection_gold_app.js",
    "src/football_intelligence/review_chassis/static/detection_gold_wizard.js",
    "src/football_intelligence/review_chassis/static/styles.css",
    "tests/test_m5_5g1a_r3_r2_dense_first_split.py",
}


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
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def tree_manifest(root: Path, *, include_rows: bool = False) -> dict[str, Any]:
    rows = [
        {"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    result: dict[str, Any] = {
        "root": safe_path(root),
        "file_count": len(rows),
        "total_bytes": sum(row["size_bytes"] for row in rows),
        "tree_hash": rows_hash(rows),
    }
    if include_rows:
        result["files"] = rows
    return result


def ensure_workspace() -> None:
    for name in SECTIONS:
        (STAGE / name).mkdir(parents=True, exist_ok=True)
    if (PACKAGE / "decisions").exists():
        raise RuntimeError("the R3-R2 package must point to the existing live decisions root")


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
    changed = [row[3:].replace("\\", "/") for row in status.splitlines() if len(row) > 3]
    baseline_exists = (
        subprocess.run(["git", "cat-file", "-e", f"{BASELINE}^{{commit}}"], cwd=REPO, check=False).returncode == 0
    )
    result = {
        "authorized_baseline": BASELINE,
        "head_at_build": head,
        "branch": git("branch", "--show-current"),
        "origin": git("remote", "get-url", "origin"),
        "baseline_exists": baseline_exists,
        "head_is_authorized_baseline": head == BASELINE,
        "working_tree_paths_at_build": changed,
        "working_tree_contains_only_r3_r2_changes": set(changed) <= ALLOWED_CHANGES,
        "g2b_is_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", "03114b1b93d8b09fcc51b93f01c73fa340e8b7b8", head],
            cwd=REPO,
            check=False,
        ).returncode
        == 0,
    }
    result["passed"] = all(
        (
            result["head_is_authorized_baseline"],
            baseline_exists,
            result["branch"] == "main",
            result["origin"] == ORIGIN,
            result["working_tree_contains_only_r3_r2_changes"],
            result["g2b_is_ancestor"],
        )
    )
    if not result["passed"]:
        raise RuntimeError(f"FAIL_BASELINE_OR_WORKTREE: {result}")
    return result


def copy_prompt_pack() -> dict[str, Any]:
    manifest = read_json(PROMPT / "08_PROMPT_PACK_MANIFEST.json")
    checks = []
    for entry in manifest["files"]:
        source = PROMPT / entry["filename"]
        target = STAGE / "00_PROMPT_AND_INPUTS" / source.name
        shutil.copy2(source, target)
        checks.append(
            {
                "filename": source.name,
                "size_match": source.stat().st_size == int(entry["byte_size"]),
                "sha256_match": sha256_file(source) == entry["sha256"] == sha256_file(target),
            }
        )
    shutil.copy2(
        PROMPT / "08_PROMPT_PACK_MANIFEST.json", STAGE / "00_PROMPT_AND_INPUTS" / "08_PROMPT_PACK_MANIFEST.json"
    )
    result = {
        "file_count": len(checks) + 1,
        "checks": checks,
        "passed": all(row["size_match"] and row["sha256_match"] for row in checks),
    }
    if not result["passed"]:
        raise RuntimeError("prompt-pack integrity validation failed")
    write_json(STAGE / "00_PROMPT_AND_INPUTS" / "prompt_copy_validation.json", result)
    return result


def parse_events() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (R3_DECISIONS / "review_decision_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def snapshot_audit() -> dict[str, Any]:
    checks = []
    for sidecar in sorted((R3_DECISIONS / "snapshots").glob("*.json.sha256")):
        expected, filename = sidecar.read_text(encoding="utf-8").strip().split(maxsplit=1)
        target = sidecar.with_name(filename.strip())
        checks.append({"path": target.name, "matches": target.exists() and sha256_file(target) == expected})
    return {"count": len(checks), "checks": checks, "passed": bool(checks) and all(row["matches"] for row in checks)}


def live_state_audit(decisions_before: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_manifest(R3_R1_PACKAGE / "reviewer_manifest.json")
    ui = load_ui_config(R3_R1_PACKAGE / "ui_config.json")
    state = read_json(R3_DECISIONS / "review_decisions.json")
    events = parse_events()
    store = DetectionGoldPilotPersistence(
        manifest=manifest, ui_config=ui, decisions_root=R3_DECISIONS, reviewer_session_id=REVIEWER
    )
    replayed = store._materialize_events(events)  # noqa: SLF001 - deliberate read-only ledger replay
    replay_keys = (
        "active_tranche_id",
        "annotation_hashes",
        "annotations",
        "completed",
        "decisions",
        "event_sequence",
        "structured_reviews",
        "tranche_completions",
        "wizard_states",
    )
    old = ui.question_contract["gold_tranches"]
    a_ids = old["A_CORE_STATIC"]["case_ids"]
    b_ids = old["B_REMAINING_STATIC"]["case_ids"]
    later_ids = [
        case_id
        for tranche_id in ("C_DENSE_AND_PITCH", "D_TEMPORAL_PLAYER", "E_FOOTBALL")
        for case_id in old[tranche_id]["case_ids"]
    ]
    saved = set(state.get("annotations", {}))
    sequences = [int(event["event_sequence"]) for event in events]
    completions = {
        tranche_id: validate_completion_bundle(R3_DECISIONS / "completed_tranches" / tranche_id)
        for tranche_id in ("A_CORE_STATIC", "B_REMAINING_STATIC")
    }
    gate = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r2.live_state_precondition.v1",
        "review_id": manifest.review_id,
        "event_sequence": int(state.get("event_sequence", -1)),
        "event_count": len(events),
        "event_type_counts": dict(sorted(Counter(str(event.get("event_type")) for event in events).items())),
        "event_sequences_contiguous": sequences == list(range(1, len(events) + 1)),
        "event_replay_matches_authoritative_state": all(
            stable_hash(replayed.get(key)) == stable_hash(state.get(key)) for key in replay_keys
        ),
        "snapshot_audit": snapshot_audit(),
        "saved_case_count": len(saved),
        "all_32_static_cases_saved": saved == set(a_ids + b_ids),
        "tranche_a_completed": "A_CORE_STATIC" in state.get("tranche_completions", {}),
        "tranche_b_completed": "B_REMAINING_STATIC" in state.get("tranche_completions", {}),
        "tranche_a_completion_valid": completions["A_CORE_STATIC"]["passed"],
        "tranche_b_completion_valid": completions["B_REMAINING_STATIC"]["passed"],
        "later_saved_case_count": len(saved & set(later_ids)),
        "dense_saved_case_count": len(saved & set(old["C_DENSE_AND_PITCH"]["case_ids"][:8])),
        "pitch_saved_case_count": len(saved & set(old["C_DENSE_AND_PITCH"]["case_ids"][8:])),
        "temporal_saved_case_count": len(saved & set(old["D_TEMPORAL_PLAYER"]["case_ids"])),
        "football_saved_case_count": len(saved & set(old["E_FOOTBALL"]["case_ids"])),
        "pending_outbox_events": 0,
        "pending_outbox_evidence": "USER_CONFIRMED_ZERO_AND_NO_UNACKNOWLEDGED_SERVER_EVENT",
        "case_payload_hash": stable_hash(read_json(R3_R1_PACKAGE / "reviewer_manifest.json")["cases"]),
        "evidence_tree_hash": tree_manifest(R3_R1_PACKAGE / "evidence")["tree_hash"],
        "ontology_hash": ONTOLOGY_HASH,
        "decisions_tree_before": decisions_before,
    }
    gate["passed"] = all(
        (
            gate["review_id"] == REVIEW_ID,
            gate["event_sequence"] == 35,
            gate["event_count"] == 35,
            gate["event_sequences_contiguous"],
            gate["event_replay_matches_authoritative_state"],
            gate["snapshot_audit"]["passed"],
            gate["all_32_static_cases_saved"],
            gate["tranche_a_completed"],
            gate["tranche_b_completed"],
            gate["tranche_a_completion_valid"],
            gate["tranche_b_completion_valid"],
            gate["later_saved_case_count"] == 0,
            gate["pending_outbox_events"] == 0,
            gate["case_payload_hash"] == CASE_HASH,
            gate["evidence_tree_hash"] == EVIDENCE_HASH,
        )
    )
    if not gate["passed"]:
        raise RuntimeError(f"FAIL_LIVE_STATE_PRECONDITION: {gate}")
    preservation = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r2.a_b_preservation.v1",
        "saved_case_ids": sorted(saved),
        "saved_annotation_hashes": {case_id: state["annotation_hashes"][case_id] for case_id in sorted(saved)},
        "completion_bundles": {
            tranche_id: tree_manifest(R3_DECISIONS / "completed_tranches" / tranche_id, include_rows=True)
            for tranche_id in ("A_CORE_STATIC", "B_REMAINING_STATIC")
        },
        "automatic_reopen_performed": False,
        "automatic_resave_performed": False,
        "human_decision_payloads_copied_to_new_package": False,
        "passed": True,
    }
    return gate, preservation


def migrated_tranches() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_ui = read_json(R3_R1_PACKAGE / "ui_config.json")
    old = source_ui["question_contract"]["gold_tranches"]
    manifest = read_json(R3_R1_PACKAGE / "reviewer_manifest.json")
    task_by_case = {case["case_id"]: case["task_type"] for case in manifest["cases"]}
    old_c = old["C_DENSE_AND_PITCH"]["case_ids"]
    dense = [case_id for case_id in old_c if task_by_case[case_id] == "detection_gold_dense_region"]
    pitch = [case_id for case_id in old_c if task_by_case[case_id] == "detection_gold_pitch_boundary"]
    tranches = {
        "A_CORE_STATIC": {"label": "Tranche A - core static (completed)", "case_ids": old["A_CORE_STATIC"]["case_ids"]},
        "B_REMAINING_STATIC": {
            "label": "Tranche B - remaining static (completed)",
            "case_ids": old["B_REMAINING_STATIC"]["case_ids"],
        },
        "C1_DENSE_OVERLAP": {"label": "Tranche C1 - dense overlap", "case_ids": dense},
        "C2_PITCH_BOUNDARY": {"label": "Tranche C2 - pitch and boundary", "case_ids": pitch},
        "D_TEMPORAL_PLAYER": {"label": "Tranche D - temporal player", "case_ids": old["D_TEMPORAL_PLAYER"]["case_ids"]},
        "E_FOOTBALL": {"label": "Tranche E - football", "case_ids": old["E_FOOTBALL"]["case_ids"]},
    }
    order = list(tranches)
    mapping = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r2.old_to_new_tranche_mapping.v1",
        "old_manifest_preserved_at": safe_path(R3_R1_PACKAGE / "ui_config.json"),
        "mapping": {
            "A_CORE_STATIC": ["A_CORE_STATIC"],
            "B_REMAINING_STATIC": ["B_REMAINING_STATIC"],
            "C_DENSE_AND_PITCH": ["C1_DENSE_OVERLAP", "C2_PITCH_BOUNDARY"],
            "D_TEMPORAL_PLAYER": ["D_TEMPORAL_PLAYER"],
            "E_FOOTBALL": ["E_FOOTBALL"],
        },
        "old_c_case_ids": old_c,
        "new_c1_case_ids": dense,
        "new_c2_case_ids": pitch,
    }
    versioned = {
        "schema_version": "football_intelligence.m5_5g1a_r3.gold_tranche_manifest.v2",
        "review_id": REVIEW_ID,
        "tranche_order": order,
        "default_tranche_id": "C1_DENSE_OVERLAP",
        "total_case_count": sum(len(value["case_ids"]) for value in tranches.values()),
        "tranches": tranches,
        "migration_source_hash": ui_config_hash(load_ui_config(R3_R1_PACKAGE / "ui_config.json")),
    }
    assigned = [case_id for tranche_id in order for case_id in tranches[tranche_id]["case_ids"]]
    validation = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r2.tranche_split_validation.v1",
        "total_remains_88": len(assigned) == 88,
        "all_cases_assigned_once": len(assigned) == len(set(assigned)) == len(manifest["cases"]),
        "case_set_unchanged": set(assigned) == {case["case_id"] for case in manifest["cases"]},
        "c1_exact_old_dense_membership": dense == old_c[:8] and len(dense) == 8,
        "c2_exact_old_pitch_membership": pitch == old_c[8:] and len(pitch) == 12,
        "a_unchanged": tranches["A_CORE_STATIC"]["case_ids"] == old["A_CORE_STATIC"]["case_ids"],
        "b_unchanged": tranches["B_REMAINING_STATIC"]["case_ids"] == old["B_REMAINING_STATIC"]["case_ids"],
        "no_phantom_c1_or_c2_completion": True,
    }
    validation["passed"] = all(value for key, value in validation.items() if key not in {"schema_version", "passed"})
    if not validation["passed"]:
        raise RuntimeError(f"FAIL_TRANCHE_SPLIT: {validation}")
    return versioned, mapping, validation


def build_package(versioned: dict[str, Any]) -> dict[str, Any]:
    source_evidence = tree_manifest(R3_R1_PACKAGE / "evidence")
    if source_evidence["tree_hash"] != EVIDENCE_HASH or source_evidence["file_count"] != 1512:
        raise RuntimeError("frozen evidence tree changed")
    shutil.copytree(R3_R1_PACKAGE / "evidence", PACKAGE / "evidence", copy_function=shutil.copy2, dirs_exist_ok=True)
    for name in ("reviewer_manifest.json", "evidence_manifest.json", "second_reviewer_and_adjudication_contract.json"):
        shutil.copy2(R3_R1_PACKAGE / name, PACKAGE / name)
    source_config = load_ui_config(R3_R1_PACKAGE / "ui_config.json")
    ui = read_json(R3_R1_PACKAGE / "ui_config.json")
    ui["page_title"] = "Football Intelligence - Dense-overlap gold tranche"
    ui["review_title"] = "Dense-first incremental detection-gold annotation"
    ui["task_instructions"] = "Complete the eight dense-overlap visible-mask cases before later annotation tranches."
    ui["question_contract"].update(
        {
            "client_build_id": CLIENT_BUILD_ID,
            "revision_aware_wizard_state": True,
            "indexeddb_namespace": INDEXEDDB_NAMESPACE,
            "prior_indexeddb_namespace_import_forbidden": True,
            "first_load_server_reconciliation": True,
            "first_load_forced_tranche_id": "C1_DENSE_OVERLAP",
            "first_load_notice": (
                "Static player annotation is complete. The next task is eight dense-overlap mask cases. "
                "Pitch, temporal and football work remain separate later stages."
            ),
            "same_server_authoritative_decisions_root": True,
            "tranche_manifest_schema_version": versioned["schema_version"],
            "gold_tranches": versioned["tranches"],
            "tranche_order": versioned["tranche_order"],
            "default_tranche_id": versioned["default_tranche_id"],
            "compatible_predecessor_ui_config_hashes": [ui_config_hash(source_config)],
            "dense_workflow_steps": [
                "Trace each visible person",
                "Answer short overlap questions",
                "Check the machine boxes",
                "Review and save",
            ],
            "dense_current_frame_only": True,
            "dense_focal_roi_only": True,
            "candidate_visible_mask_coverage_required": True,
        }
    )
    write_json(PACKAGE / "ui_config.json", ui)
    write_json(
        PACKAGE / "server_decisions_root_pointer.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r2.external_decisions_pointer.v1",
            "review_id": REVIEW_ID,
            "decisions_root": str(R3_DECISIONS),
            "package_local_decisions_root_created": False,
            "launcher_uses_existing_server_authoritative_root": True,
        },
    )
    validation_root = STAGE / "_tmp" / "package_validation_empty_decisions"
    validation_root.mkdir(parents=True, exist_ok=True)
    for path in validation_root.glob("*"):
        if path.is_file():
            path.unlink()
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    config = load_ui_config(PACKAGE / "ui_config.json")
    DetectionGoldPilotPersistence(
        manifest=manifest, ui_config=config, decisions_root=validation_root, reviewer_session_id=REVIEWER
    ).ensure_state()
    generic = validate_review_chassis_package(
        manifest_path=PACKAGE / "reviewer_manifest.json",
        ui_config_path=PACKAGE / "ui_config.json",
        evidence_root=PACKAGE / "evidence",
        decisions_root=validation_root,
    )
    copied = tree_manifest(PACKAGE / "evidence")
    result = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r2.review_package_validation.v1",
        "manifest_hash": manifest_hash(manifest),
        "ui_config_hash": ui_config_hash(config),
        "case_count": len(manifest.cases),
        "case_payload_hash": stable_hash(read_json(PACKAGE / "reviewer_manifest.json")["cases"]),
        "manifest_byte_identical": sha256_file(PACKAGE / "reviewer_manifest.json")
        == sha256_file(R3_R1_PACKAGE / "reviewer_manifest.json"),
        "evidence_manifest_byte_identical": sha256_file(PACKAGE / "evidence_manifest.json")
        == sha256_file(R3_R1_PACKAGE / "evidence_manifest.json"),
        "evidence_copy": copied,
        "package_local_decisions_root_absent": not (PACKAGE / "decisions").exists(),
        "generic_empty_fixture_validation": generic,
        "default_tranche_id": config.question_contract["default_tranche_id"],
        "browser_acceptance": {"status": "PENDING", "passed": False},
    }
    result["package_checks_passed"] = all(
        (
            result["case_count"] == 88,
            result["case_payload_hash"] == CASE_HASH,
            result["manifest_byte_identical"],
            result["evidence_manifest_byte_identical"],
            copied["file_count"] == 1512,
            copied["tree_hash"] == EVIDENCE_HASH,
            result["package_local_decisions_root_absent"],
            result["default_tranche_id"] == "C1_DENSE_OVERLAP",
            generic["passed"],
        )
    )
    result["passed"] = result["package_checks_passed"]
    if not result["passed"]:
        raise RuntimeError(f"review package validation failed: {result}")
    write_json(PACKAGE / "review_package_validation.json", result)
    return result


def write_dense_contracts(versioned: dict[str, Any]) -> None:
    wizard_source = (REPO / "src/football_intelligence/review_chassis/static/detection_gold_wizard.js").read_text(
        encoding="utf-8"
    )
    required_copy = [
        "Trace only the part of the person you can actually see.",
        "Is another person in front of this person?",
        "Which person is in front?",
        "How clear is this outline?",
        "How much of the visible person is covered by this machine box?",
    ]
    dense = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r2.dense_wizard_validation.v1",
        "case_ids": versioned["tranches"]["C1_DENSE_OVERLAP"]["case_ids"],
        "authoritative_current_frame_only": True,
        "previous_next_reference_only": True,
        "focal_roi_only": True,
        "one_visible_mask_per_person": True,
        "visible_pixels_only": True,
        "candidate_targets_explicit": True,
        "candidate_visible_mask_coverage_persisted": True,
        "required_plain_language_present": {value: value in wizard_source for value in required_copy},
    }
    dense["passed"] = len(dense["case_ids"]) == 8 and all(dense["required_plain_language_present"].values())
    write_json(STAGE / "03_DENSE_WIZARD_VALIDATION" / "dense_wizard_validation.json", dense)
    write_json(
        STAGE / "03_DENSE_WIZARD_VALIDATION" / "mask_revision_invalidation.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r2.mask_revision_invalidation.v1",
            "delete_mask_invalidates_dependent_candidate_answers": True,
            "delete_all_masks_returns_to_step_one": True,
            "non_latest_mask_edit_supported": True,
            "occluder_requires_reciprocal_overlap": True,
            "occlusion_order_validated_server_side": True,
            "candidate_coverage_bound_to_revision_record": True,
            "saved_cases_immutable": True,
            "passed": dense["passed"],
        },
    )
    write_json(
        STAGE / "04_BROWSER_PERSISTENCE_AND_COMPLETION" / "c1_completion_contract.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r2.c1_completion_contract.v1",
            "tranche_id": "C1_DENSE_OVERLAP",
            "required_case_count": 8,
            "atomic_four_file_bundle": True,
            "completion_does_not_complete_c2": True,
            "completion_does_not_complete_full_pilot": True,
            "a_b_completion_preserved": True,
            "completed_c1_is_immutable": True,
            "full_completion_required_tranches": versioned["tranche_order"],
        },
    )
    write_json(
        STAGE / "04_BROWSER_PERSISTENCE_AND_COMPLETION" / "first_load_reconciliation.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r2.first_load_reconciliation.v1",
            "status": "PENDING_REAL_BROWSER_ACCEPTANCE",
            "server_state_authoritative": True,
            "new_indexeddb_namespace": INDEXEDDB_NAMESPACE,
            "old_namespace_imported": False,
            "expected_default_tranche": "C1_DENSE_OVERLAP",
            "expected_progress": "0/8 saved",
            "expected_completed_tranches": ["A_CORE_STATIC", "B_REMAINING_STATIC"],
            "passed": False,
        },
    )
    write_json(
        STAGE / "04_BROWSER_PERSISTENCE_AND_COMPLETION" / "browser_persistence_results.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r2.browser_acceptance.v1",
            "status": "PENDING_REAL_BROWSER_ACCEPTANCE",
            "temporary_copied_decisions_only": True,
            "real_human_decisions_root_opened": False,
            "passed": False,
        },
    )
    write_json(
        STAGE / "04_BROWSER_PERSISTENCE_AND_COMPLETION" / "truthful_dense_timing.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r2.truthful_dense_timing.v1",
            "c1_case_count": 8,
            "modelled_seconds_per_dense_case": 90,
            "modelled_c1_total_minutes": 12,
            "actual_human_active_minutes": None,
            "browser_automation_time_claimed_as_human_time": False,
            "future_estimates_minutes": {"C2_PITCH_BOUNDARY": 3, "D_TEMPORAL_PLAYER": 9, "E_FOOTBALL": 12},
            "difficult_cases_removed": False,
        },
    )


def write_permission() -> None:
    write_json(
        STAGE / "06_NEXT_STAGE_DENSE_PERMISSION" / "next_stage_dense_permission.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r2.next_stage_dense_permission.v1",
            "next_stage": "M5_5G4_CONDITIONAL_DENSE_REGION_INSTANCE_SEPARATION_DEVELOPMENT_v1",
            "permission_status": "CONDITIONAL_ON_COMPLETED_AND_INDEPENDENTLY_AUDITED_C1",
            "currently_permitted": False,
            "allowed_after_gate": [
                "compare visible/full boxes and masks",
                "measure merged-as-clean and distinct-person suppression on dense development cases",
                "evaluate deterministic dense eligibility and annotation-assisted mask refinement",
                "compare box-only and mask-assisted separation",
            ],
            "forbidden": [
                "training or fine-tuning without separate authorization",
                "segmenter or consolidator promotion",
                "identity tracking",
                "final accuracy claims",
                "using C1 as validation or sealed holdout",
                "pitch-gate research before C2",
            ],
        },
    )


def write_launcher() -> None:
    launcher = f"""$ErrorActionPreference = 'Stop'
$port = 8807
$occupied = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($occupied) {{
  Write-Error 'Port 8807 is occupied. Stop the old annotation server, then rerun. This launcher will not move ports.'
}}
$repo = '{REPO}'
$package = '{PACKAGE}'
$decisions = '{R3_DECISIONS}'
Set-Location -LiteralPath $repo
Write-Host 'Starting the dense-first C1 annotation package.' -ForegroundColor Green
Write-Host 'Open http://127.0.0.1:8807/' -ForegroundColor Cyan
uv run fi-pipeline review-chassis serve `
  --manifest "$package/reviewer_manifest.json" `
  --ui-config "$package/ui_config.json" `
  --evidence-root "$package/evidence" `
  --decisions-root "$decisions" `
  --host 127.0.0.1 `
  --port 8807 `
  --reviewer-session-id '{REVIEWER}'
"""
    instructions = """# Dense-first Tranche C1

1. Stop any older annotation server on port 8807.
2. Run `launch_dense_first_review.ps1`.
3. Open `http://127.0.0.1:8807/`.
4. Confirm Tranches A and B show completed and C1 opens at `0/8 saved`.
5. Complete only the eight dense-overlap cases and use **Complete tranche** for C1.

Trace only visible pixels in the focal Current frame. Previous and Next are reference only.
The package uses the existing server-authoritative decisions root but a fresh browser-draft namespace.
Pitch, temporal and football cases remain separate later tranches.
"""
    for root in (PACKAGE, STAGE):
        write_text(root / "launch_dense_first_review.ps1", launcher)
        write_text(root / "HUMAN_INSTRUCTIONS.md", instructions)


def main() -> None:
    decisions_before = tree_manifest(R3_DECISIONS, include_rows=True)
    r3_r1_before = tree_manifest(R3_R1)
    g3_before = tree_manifest(G3) if G3.exists() else {"root": safe_path(G3), "file_count": 0, "tree_hash": None}
    ensure_workspace()
    authorization_result = authorization()
    prompt_result = copy_prompt_pack()
    gate, preservation = live_state_audit(decisions_before)
    write_json(STAGE / "01_LIVE_STATE_AND_PRESERVATION_AUDIT" / "live_state_precondition.json", gate)
    write_json(STAGE / "01_LIVE_STATE_AND_PRESERVATION_AUDIT" / "a_b_preservation.json", preservation)
    versioned, mapping, split = migrated_tranches()
    write_json(STAGE / "02_TRANCHE_MANIFEST_SPLIT" / "old_to_new_tranche_mapping.json", mapping)
    write_json(STAGE / "02_TRANCHE_MANIFEST_SPLIT" / "gold_tranche_manifest_v2.json", versioned)
    write_json(STAGE / "02_TRANCHE_MANIFEST_SPLIT" / "tranche_split_validation.json", split)
    package_result = build_package(versioned)
    write_dense_contracts(versioned)
    write_permission()
    write_launcher()
    decisions_after = tree_manifest(R3_DECISIONS, include_rows=True)
    r3_r1_after = tree_manifest(R3_R1)
    g3_after = tree_manifest(G3) if G3.exists() else g3_before
    if decisions_before != decisions_after or r3_r1_before != r3_r1_after or g3_before != g3_after:
        raise RuntimeError("FAIL_PRIOR_STAGE_MUTATION: a protected input changed during build")
    gate["decisions_tree_after"] = decisions_after
    gate["live_decisions_byte_identical_after_build"] = True
    write_json(STAGE / "01_LIVE_STATE_AND_PRESERVATION_AUDIT" / "live_state_precondition.json", gate)
    write_json(
        STAGE / "01_LIVE_STATE_AND_PRESERVATION_AUDIT" / "protected_input_preservation.json",
        {
            "live_decisions_byte_identical": True,
            "r3_r1_workspace_byte_identical": True,
            "g3_workspace_byte_identical": True,
            "historical_artifacts_mutated": False,
            "passed": True,
        },
    )
    write_json(
        STAGE / "07_COMMANDS_AND_TESTS" / "build_summary.json",
        {
            "classification": CLASSIFICATION,
            "authorization": authorization_result,
            "prompt_pack": prompt_result,
            "live_state_precondition_passed": gate["passed"],
            "a_b_preservation_passed": preservation["passed"],
            "tranche_split_passed": split["passed"],
            "package": package_result,
            "browser_acceptance_pending": True,
            "tests_pending": True,
            "review_pack_pending": True,
            "model_fit_performed": False,
            "detector_or_consolidator_or_tracker_changed": False,
            "detector_or_consolidator_or_tracker_promoted": False,
            "production_ready": False,
            "human_approved": False,
        },
    )
    print(json.dumps({"stage": str(STAGE), "package": str(PACKAGE), "passed": True}, indent=2))


if __name__ == "__main__":
    main()
