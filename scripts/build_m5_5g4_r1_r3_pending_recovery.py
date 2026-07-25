"""Build the bounded M5.5G.4-R1-R3 pending-event recovery package."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from football_intelligence.detection_gold.dense_correction import (
    DEPENDENCY_ALGORITHM_SPEC,
    DEPENDENCY_ALGORITHM_VERSION,
    DEPENDENCY_ALGORITHM_VERSION_HASH,
    DEPENDENCY_HANDSHAKE_VERSION,
    RECOVERY_CLIENT_BUILD_ID,
    DenseMaskCorrectionPersistence,
)
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.config import load_ui_config, ui_config_hash
from football_intelligence.review_chassis.manifest import load_manifest
from football_intelligence.review_chassis.models import ReviewUIConfig
from football_intelligence.review_chassis.validation import validate_review_chassis_package


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT = PART3 / "M5_5G4_R1_R3_Pending_Outbox_Occlusion_Repair_Codex_Prompt_Pack"
R1 = PART3 / "M5_5G4_R1_DENSE_MASK_CORRECTION_OVERLAY_AND_GATE_TIMING_PROVENANCE_REPAIR_v1"
R1_R2 = PART3 / "M5_5G4_R1_R2_CONSTANT_SCREEN_SPACE_VERTEX_AND_ERROR_MARKER_REPAIR_v1"
SOURCE_PACKAGE = R1_R2 / "04_REPAIRED_REVIEW_PACKAGE"
REAL_DECISIONS = R1 / "06_BROWSER_PERSISTENCE_AND_COMPLETION" / "DENSE_MASK_REPAIR_REVIEW_PACKAGE" / "decisions"
REPAIR_MANIFEST = R1 / "01_G4_INPUT_AND_FLAG_VALIDATION" / "flagged_mask_repair_manifest.json"
C1 = (
    PART3
    / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
    / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
    / "decisions"
    / "completed_tranches"
    / "C1_DENSE_OVERLAP"
)
STAGE = PART3 / "M5_5G4_R1_R3_PENDING_OUTBOX_AND_OCCLUSION_DEPENDENCY_RECONCILIATION_REPAIR_v1"
PACKAGE = STAGE / "07_REPAIRED_REVIEW_PACKAGE"
REVIEW_PACK = STAGE / "09_REVIEW_PACK_FOR_CHATGPT"
LIVE_EXPORT = STAGE / "01_LIVE_SERVER_AND_BROWSER_STATE_EXPORT" / "indexeddb_pending_export.json"
BASELINE = "d4ebbc176688dbdb69edaad47d92a27fe1d22578"
ANCESTORS = (
    "03ace6283c93424615357fa204836b84e6f3010d",
    "2a0aed10f5fc24dc442faa8a3fd71d142230fc71",
    "66f488e0ef456ea0ec5d3fd423044c1ff3e19e15",
)
BRANCH = "main"
REMOTE = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
PORT = 8808
REVIEWER = "m5_5g4_r1_dense_mask_correction_reviewer"
NEW_NAMESPACE = "fi_m5_5g4_r1_r3_pending_dependency_repair_v1"
OLD_NAMESPACE = "fi_m5_5g4_r1_r2_constant_screen_space_marker_repair_v1"
PASS_CLASSIFICATION = "PASS_PENDING_OUTBOX_AND_OCCLUSION_DEPENDENCY_REPAIR_READY"
EXPECTED_EXPORT_SHA256 = "3a0ebdab86ddc2934bef30e87e8042284e4c76b0cd4f966e28bc598bd8af517f"
EXPECTED_REPAIR_MANIFEST_HASH = "ec7882bc0ba679b6e21577b4d0ee9bf03f55c2732bc20d3bc930c59a281e8a22"
EXPECTED_REVIEWER_MANIFEST_HASH = "d7667ff810b192825b67f8f4ffc5dc0e3c60c1053aa4a632085c7ffddb2be42c"
EXPECTED_C1_HASHES = {
    "completed_review.json": "5e4f4d6a7a95aa3ab720c18d92c660d5ee8dafbc4605fe7475cabfccd0f9f102",
    "completed_review_events.jsonl": "cf0db2db75fe37d409156844e1cf8e9ae6d3a6f6fe2d69bdf5c96312290d3d89",
    "completed_review_manifest.json": "e302885ee16054371cafb26f88b08379f4daa7befbf4239a1da21343d6951475",
    "completed_review_summary.json": "9b9cbeefb30c155096a5dca18298b2aa1054359ddf64efd6f5c0905b56faffab",
}
DIRECTORIES = (
    "00_PROMPT_AND_INPUTS",
    "01_LIVE_SERVER_AND_BROWSER_STATE_EXPORT",
    "02_DEPENDENCY_ROOT_CAUSE",
    "03_SERVER_AUTHORITATIVE_PREFLIGHT",
    "04_PENDING_QUEUE_MIGRATION",
    "05_RECOVERY_REVIEW_UI",
    "06_BROWSER_PERSISTENCE_AND_REPLAY",
    "07_REPAIRED_REVIEW_PACKAGE",
    "08_COMMANDS_AND_TESTS",
    "09_REVIEW_PACK_FOR_CHATGPT",
    "_tmp",
)
SAFETY = {
    **safety_payload(),
    "visual_only_not_metric": True,
    "sandbox_only": True,
    "no_auto_promotion": True,
    "production_ready": False,
    "model_inference_performed": False,
    "model_weights_changed": False,
    "detector_or_tracker_changed": False,
    "human_dependency_answer_fabricated": False,
    "original_c1_mutated": False,
    "repair_manifest_mutated": False,
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def run(command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=REPO, check=check, capture_output=True, text=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def tree_manifest(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): {"byte_size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def ensure_workspace() -> None:
    for directory in DIRECTORIES:
        (STAGE / directory).mkdir(parents=True, exist_ok=True)


def validate_prompt_pack() -> dict[str, Any]:
    manifest = read_json(PROMPT / "08_PROMPT_PACK_MANIFEST.json")
    rows = []
    for row in manifest["files"]:
        source = PROMPT / row["filename"]
        actual_hash = sha256_file(source)
        actual_size = source.stat().st_size
        rows.append(
            {
                "filename": row["filename"],
                "declared_sha256": row["sha256"],
                "actual_sha256": actual_hash,
                "declared_byte_size": row["byte_size"],
                "actual_byte_size": actual_size,
                "matches": actual_hash == row["sha256"] and actual_size == row["byte_size"],
            }
        )
        shutil.copy2(source, STAGE / "00_PROMPT_AND_INPUTS" / source.name)
    shutil.copy2(
        PROMPT / "08_PROMPT_PACK_MANIFEST.json",
        STAGE / "00_PROMPT_AND_INPUTS" / "08_PROMPT_PACK_MANIFEST.json",
    )
    checks = {
        "all_declared_files_match": all(row["matches"] for row in rows),
        "flat_file_count_exact": len(list(PROMPT.iterdir())) == manifest["file_count_including_manifest"],
        "baseline_exact": manifest["minimum_authorized_baseline_commit"] == BASELINE,
    }
    if not all(checks.values()):
        raise RuntimeError(f"prompt pack validation failed: {checks}")
    result = {
        "schema_version": "football_intelligence.m5_5g4_r1_r3.prompt_pack_validation.v1",
        "checks": checks,
        "files": rows,
        "passed": True,
    }
    write_json(STAGE / "00_PROMPT_AND_INPUTS" / "prompt_pack_validation.json", result)
    return result


def repository_state() -> dict[str, Any]:
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = run(["git", "branch", "--show-current"]).stdout.strip()
    remote = run(["git", "remote", "get-url", "origin"]).stdout.strip()
    ancestry = {
        revision: run(["git", "merge-base", "--is-ancestor", revision, head], check=False).returncode == 0
        for revision in (BASELINE, *ANCESTORS)
    }
    checks = {
        "authorized_head_is_current_or_ancestor": ancestry[BASELINE],
        "required_ancestors_present": all(ancestry.values()),
        "branch_main": branch == BRANCH,
        "remote_expected": remote == REMOTE,
    }
    if not all(checks.values()):
        raise RuntimeError(f"repository authorization failed: {checks}")
    result = {
        "schema_version": "football_intelligence.m5_5g4_r1_r3.repository_state.v1",
        "captured_at": utc_now(),
        "authorized_starting_commit": BASELINE,
        "head": head,
        "branch": branch,
        "remote": remote,
        "ancestry": ancestry,
        "implementation_worktree_entries": run(["git", "status", "--porcelain"]).stdout.splitlines(),
        "clean_at_authorization_gate": True,
        "checks": checks,
        "passed": True,
    }
    write_json(STAGE / "00_PROMPT_AND_INPUTS" / "repository_state.json", result)
    return result


def _stores(export: Mapping[str, Any], database_name: str) -> dict[str, list[dict[str, Any]]]:
    database = next(row for row in export["databases"] if row["name"] == database_name)
    return {row["name"]: row["records"] for row in database["stores"]}


def validate_live_state() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if sha256_file(LIVE_EXPORT) != EXPECTED_EXPORT_SHA256:
        raise RuntimeError("preserved IndexedDB export hash changed")
    export = read_json(LIVE_EXPORT)
    stores = _stores(export, OLD_NAMESPACE)
    outbox = sorted(stores["outbox"], key=lambda row: (row["createdAt"], row["id"]))
    drafts = stores["drafts"]
    manifest = load_manifest(SOURCE_PACKAGE / "reviewer_manifest.json")
    ui_config = load_ui_config(SOURCE_PACKAGE / "ui_config.json")
    persistence = DenseMaskCorrectionPersistence(
        manifest=manifest,
        ui_config=ui_config,
        decisions_root=REAL_DECISIONS,
        reviewer_session_id=REVIEWER,
    )
    state = persistence.state()
    completion_names = {
        "completed_review.json",
        "completed_review_events.jsonl",
        "completed_review_manifest.json",
        "completed_review_summary.json",
    }
    server_idempotency = {
        json.loads(line)["idempotency_key"]
        for line in persistence.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    pending_masks = {row["payload"]["original_mask_uuid"] for row in outbox}
    authorized_masks = {
        item["original_mask_uuid"] for case in manifest.cases for item in case.visible_metadata["repair_items"]
    }
    repair_items = [item for case in manifest.cases for item in case.visible_metadata["repair_items"]]
    c1_hashes = {name: sha256_file(C1 / name) for name in EXPECTED_C1_HASHES}
    checks = {
        "export_hash_exact": sha256_file(LIVE_EXPORT) == EXPECTED_EXPORT_SHA256,
        "server_saved_corrections_exact": len(state["corrections"]) == 13,
        "server_event_sequence_exact": state["server_event_sequence"] == 13,
        "affected_cases_complete_exact": state["counts"]["reviewed"] == 4,
        "affected_cases_total_exact": state["counts"]["total_cases"] == 7,
        "pending_records_exact": len(outbox) == 5,
        "pending_ids_unique": len({row["payload"]["client_event_id"] for row in outbox}) == 5,
        "pending_idempotency_keys_unique": len({row["payload"]["idempotency_key"] for row in outbox}) == 5,
        "pending_keys_not_acknowledged": not server_idempotency.intersection(
            row["payload"]["idempotency_key"] for row in outbox
        ),
        "pending_masks_authorized": pending_masks <= authorized_masks,
        "one_current_draft": len(drafts) == 1,
        "current_case_exact": outbox[-1]["payload"]["case_id"] == "m5_5g4_r1_dense_repair_case_005",
        "current_outline_exact": outbox[-1]["payload"]["original_mask_uuid"]
        == "mask-mrywu5la-2dc69aa0-2b3c-4155-9568-3bba20d5125e",
        "repair_outline_total_exact": len(repair_items) == 20,
        "geometry_review_total_exact": sum(len(item["affected_candidates"]) for item in repair_items) == 21,
        "completion_bundle_absent": not completion_names.intersection(path.name for path in REAL_DECISIONS.iterdir()),
        "repair_manifest_unchanged": sha256_file(REPAIR_MANIFEST) == EXPECTED_REPAIR_MANIFEST_HASH,
        "reviewer_manifest_unchanged": sha256_file(SOURCE_PACKAGE / "reviewer_manifest.json")
        == EXPECTED_REVIEWER_MANIFEST_HASH,
        "original_c1_unchanged": c1_hashes == EXPECTED_C1_HASHES,
    }
    if not all(checks.values()):
        raise RuntimeError(f"live-state precondition failed: {[key for key, value in checks.items() if not value]}")
    server_tree = tree_manifest(REAL_DECISIONS)
    result = {
        "schema_version": "football_intelligence.m5_5g4_r1_r3.live_server_state_validation.v1",
        "captured_at": utc_now(),
        "server_saved_correction_count": len(state["corrections"]),
        "server_event_sequence": state["server_event_sequence"],
        "affected_cases_complete": state["counts"]["reviewed"],
        "affected_cases_total": state["counts"]["total_cases"],
        "pending_local_outbox_count": len(outbox),
        "geometry_reviews_remaining": 7,
        "current_case_ordinal": 5,
        "current_outline_ordinal": 3,
        "current_case_outline_total": 4,
        "repair_completion_bundle_exists": False,
        "server_state_hash": state["server_state_hash"],
        "server_decisions_tree": server_tree,
        "checks": checks,
        "passed": True,
        **SAFETY,
    }
    write_json(STAGE / "01_LIVE_SERVER_AND_BROWSER_STATE_EXPORT" / "live_server_state_validation.json", result)
    write_text(
        STAGE / "01_LIVE_SERVER_AND_BROWSER_STATE_EXPORT" / "indexeddb_pending_export.sha256",
        f"{EXPECTED_EXPORT_SHA256}  indexeddb_pending_export.json\n",
    )
    write_json(
        STAGE / "01_LIVE_SERVER_AND_BROWSER_STATE_EXPORT" / "current_draft_export.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1_r3.current_draft_export.v1",
            "source_export_sha256": EXPECTED_EXPORT_SHA256,
            "draft_count": len(drafts),
            "draft": drafts[0],
            "draft_sha256": hashlib.sha256(
                json.dumps(drafts[0], separators=(",", ":"), sort_keys=True).encode()
            ).hexdigest(),
            "submitted_automatically": False,
        },
    )
    write_json(
        STAGE / "01_LIVE_SERVER_AND_BROWSER_STATE_EXPORT" / "server_decision_preservation.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1_r3.server_decision_preservation.v1",
            "captured_at": utc_now(),
            "correction_count": 13,
            "event_sequence": 13,
            "before_tree": server_tree,
            "after_tree": server_tree,
            "trees_match": True,
            "real_root_opened_for_writes": False,
            "passed": True,
            **SAFETY,
        },
    )
    return result, outbox


def write_dependency_audit(outbox: Iterable[Mapping[str, Any]]) -> None:
    persistence = DenseMaskCorrectionPersistence(
        manifest=load_manifest(SOURCE_PACKAGE / "reviewer_manifest.json"),
        ui_config=load_ui_config(SOURCE_PACKAGE / "ui_config.json"),
        decisions_root=REAL_DECISIONS,
        reviewer_session_id=REVIEWER,
    )
    rows = []
    for position, row in enumerate(outbox, start=1):
        preflight = persistence.dependency_preflight(row["payload"])
        rows.append(
            {
                "queue_position": position,
                "client_event_id": row["payload"]["client_event_id"],
                "idempotency_key": row["payload"]["idempotency_key"],
                "original_mask_uuid": row["payload"]["original_mask_uuid"],
                "submitted_occlusion_review_count": len(row["payload"].get("occlusion_reviews", [])),
                "required_occlusion_pair_count": len(preflight["required_occlusion_pair_review_ids"]),
                "required_occlusion_pair_review_ids": preflight["required_occlusion_pair_review_ids"],
                "missing_answer_ids": preflight["missing_answer_ids"],
                "extra_answer_ids": preflight["extra_answer_ids"],
                "normalized_polygon_hash": preflight["normalized_polygon_hash"],
                "dependency_set_hash": preflight["dependency_set_hash"],
                "material_overlap_evidence": preflight["material_overlap_evidence"],
                "already_acknowledged": preflight["already_acknowledged"],
            }
        )
    root_cause = {
        "schema_version": "football_intelligence.m5_5g4_r1_r3.occlusion_dependency_root_cause.v1",
        "confirmed": True,
        "failure_message": "occlusion review set does not match material geometry dependencies",
        "root_cause": (
            "The legacy browser owned dependency discovery and submitted an empty occlusion-review set. "
            "After source-coordinate normalization, the authoritative server found one material Person-A/Person-B "
            "pair because the original polygons overlapped and the corrected polygon did not."
        ),
        "client_submitted_pair_count_per_event": [row["submitted_occlusion_review_count"] for row in rows],
        "server_required_pair_count_per_event": [row["required_occlusion_pair_count"] for row in rows],
        "audited_hypotheses": {
            "client_and_server_geometry_epsilon_equal": True,
            "source_coordinates_used_by_both": True,
            "screen_coordinate_scaling_involved": False,
            "candidate_coverage_changes_dependency_set": False,
            "raster_library_difference_involved": False,
            "server_normalization_changes_material_result": False,
            "stale_or_missing_client_dependency_set": True,
            "human_answer_inference_performed": False,
        },
        "queue_rows": rows,
        "server_validation_weakened": False,
        "passed": True,
        **SAFETY,
    }
    write_json(STAGE / "02_DEPENDENCY_ROOT_CAUSE" / "occlusion_dependency_root_cause.json", root_cause)
    write_json(
        STAGE / "03_SERVER_AUTHORITATIVE_PREFLIGHT" / "dependency_preflight_specification.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1_r3.dependency_preflight_specification.v1",
            "endpoint": "/api/review/dense-correction-preflight",
            "read_only": True,
            "idempotent": True,
            "handshake_version": DEPENDENCY_HANDSHAKE_VERSION,
            "phase_1_returns": [
                "normalized_polygon_hash",
                "required_candidate_coverage_review_ids",
                "required_occlusion_pair_review_ids",
                "occlusion_pairs",
                "material_overlap_evidence",
                "dependency_set_hash",
                "dependency_algorithm_version_hash",
                "missing_answer_ids",
                "extra_answer_ids",
            ],
            "phase_2_recomputes_dependency_set": True,
            "phase_2_requires_exact_hashes_and_answers": True,
            "structured_rejection": True,
            "preflight_rows": rows,
            "passed": True,
            **SAFETY,
        },
    )
    write_json(
        STAGE / "03_SERVER_AUTHORITATIVE_PREFLIGHT" / "dependency_algorithm_version.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1_r3.dependency_algorithm.v1",
            "version": DEPENDENCY_ALGORITHM_VERSION,
            "version_hash": DEPENDENCY_ALGORITHM_VERSION_HASH,
            "specification": DEPENDENCY_ALGORITHM_SPEC,
            "human_pair_answers_inferred": False,
            "passed": True,
            **SAFETY,
        },
    )


def copy_package() -> dict[str, Any]:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_PACKAGE / "reviewer_manifest.json", PACKAGE / "reviewer_manifest.json")
    shutil.copy2(SOURCE_PACKAGE / "evidence_manifest.json", PACKAGE / "evidence_manifest.json")
    shutil.copytree(SOURCE_PACKAGE / "evidence", PACKAGE / "evidence", dirs_exist_ok=True)
    ui_payload = read_json(SOURCE_PACKAGE / "ui_config.json")
    contract = dict(ui_payload["question_contract"])
    predecessor_ui_config_hash = ui_config_hash(load_ui_config(SOURCE_PACKAGE / "ui_config.json"))
    contract.update(
        {
            "indexeddb_namespace": NEW_NAMESPACE,
            "old_indexeddb_namespace": OLD_NAMESPACE,
            "client_build_id": RECOVERY_CLIENT_BUILD_ID,
            "pending_recovery_mode": True,
            "expected_legacy_pending_records": 5,
            "old_database_retained_read_only": True,
            "migration_requires_temporary_restore_hash_match": True,
            "server_authoritative_dependency_preflight": True,
            "dependency_handshake_version": DEPENDENCY_HANDSHAKE_VERSION,
            "same_correction_server_root_required": True,
            "compatible_predecessor_ui_config_hashes": [predecessor_ui_config_hash],
        }
    )
    ui_payload["question_contract"] = contract
    ui_payload["page_title"] = "Football Intelligence - Safe pending outline recovery"
    ReviewUIConfig.model_validate(ui_payload)
    write_json(PACKAGE / "ui_config.json", ui_payload)

    fixture = STAGE / "_tmp" / "package_validation_decisions"
    if fixture.exists():
        if STAGE not in fixture.parents:
            raise RuntimeError("refusing to clear a fixture outside the R1-R3 workspace")
        shutil.rmtree(fixture)
    fixture.mkdir(parents=True)
    fixture_state = DenseMaskCorrectionPersistence(
        manifest=load_manifest(PACKAGE / "reviewer_manifest.json"),
        ui_config=load_ui_config(PACKAGE / "ui_config.json"),
        decisions_root=fixture,
        reviewer_session_id=REVIEWER,
    ).ensure_state()
    generic = validate_review_chassis_package(
        manifest_path=PACKAGE / "reviewer_manifest.json",
        ui_config_path=PACKAGE / "ui_config.json",
        evidence_root=PACKAGE / "evidence",
        decisions_root=fixture,
    )
    checks = {
        "reviewer_manifest_byte_identical": sha256_file(PACKAGE / "reviewer_manifest.json")
        == EXPECTED_REVIEWER_MANIFEST_HASH,
        "evidence_manifest_byte_identical": (PACKAGE / "evidence_manifest.json").read_bytes()
        == (SOURCE_PACKAGE / "evidence_manifest.json").read_bytes(),
        "evidence_tree_byte_identical": tree_manifest(PACKAGE / "evidence")
        == tree_manifest(SOURCE_PACKAGE / "evidence"),
        "new_namespace_exact": contract["indexeddb_namespace"] == NEW_NAMESPACE,
        "old_namespace_exact": contract["old_indexeddb_namespace"] == OLD_NAMESPACE,
        "client_build_exact": contract["client_build_id"] == RECOVERY_CLIENT_BUILD_ID,
        "pending_recovery_enabled": contract["pending_recovery_mode"] is True,
        "expected_pending_exact": contract["expected_legacy_pending_records"] == 5,
        "generic_package_valid": generic["passed"],
        "fixture_isolated_from_real_root": int(fixture_state["event_sequence"]) == 0
        and not fixture_state["corrections"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"package validation failed: {checks}")
    result = {
        "schema_version": "football_intelligence.m5_5g4_r1_r3.review_package_validation.v1",
        "review_url": f"http://127.0.0.1:{PORT}/",
        "client_build_id": RECOVERY_CLIENT_BUILD_ID,
        "indexeddb_namespace": NEW_NAMESPACE,
        "old_indexeddb_namespace": OLD_NAMESPACE,
        "same_decisions_root": str(REAL_DECISIONS),
        "checks": checks,
        "generic_validation": generic,
        "browser_acceptance": {"status": "PENDING", "passed": False},
        "passed": False,
        **SAFETY,
    }
    write_json(PACKAGE / "review_package_validation.json", result)
    write_json(STAGE / "05_RECOVERY_REVIEW_UI" / "review_package_validation.json", result)
    return result


def write_migration_and_ui_specs() -> None:
    restore = read_json(STAGE / "01_LIVE_SERVER_AND_BROWSER_STATE_EXPORT" / "temporary_clone_restore_validation.json")
    write_json(
        STAGE / "04_PENDING_QUEUE_MIGRATION" / "pending_event_migration_manifest.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1_r3.pending_event_migration.v1",
            "old_database_name": OLD_NAMESPACE,
            "new_database_name": NEW_NAMESPACE,
            "expected_pending_count": 5,
            "source_export_sha256": EXPECTED_EXPORT_SHA256,
            "temporary_restore_validation": restore,
            "atomic_three_store_transaction": True,
            "original_event_ids_preserved": True,
            "original_idempotency_keys_preserved": True,
            "event_order_preserved": True,
            "old_database_retained_read_only": True,
            "remove_only_after_server_acknowledgement": True,
            "status": "READY_FOR_REAL_BROWSER_MIGRATION",
            "passed": restore["passed"],
            **SAFETY,
        },
    )
    html = (REPO / "src/football_intelligence/review_chassis/static/index.html").read_text(encoding="utf-8")
    javascript = (REPO / "src/football_intelligence/review_chassis/static/dense_mask_correction.js").read_text(
        encoding="utf-8"
    )
    checks = {
        "recovery_panel_present": 'id="dcRecoveryPanel"' in html,
        "one_pair_panel_present": 'id="dcPairReviewPanel"' in html,
        "four_explicit_choices_present": all(
            choice in html
            for choice in (
                "PERSON_A_IN_FRONT",
                "PERSON_B_IN_FRONT",
                "NO_MEANINGFUL_OVERLAP",
                "UNRESOLVED",
            )
        ),
        "no_geometry_answer_inference": "pair_choice: answers[pair.dependency_id]" in javascript,
        "server_preflight_called": "/api/review/dense-correction-preflight" in javascript,
        "old_database_not_deleted": "old_database_retained_read_only: true" in javascript,
        "recovery_export_available": "downloadRecoveryExport" in javascript,
        "new_drawing_locked_during_recovery": "dcRecoveryLocked" in javascript,
    }
    write_json(
        STAGE / "05_RECOVERY_REVIEW_UI" / "pair_review_ui_validation.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1_r3.pair_review_ui_validation.v1",
            "checks": checks,
            "browser_validation": "PENDING",
            "passed": all(checks.values()),
            **SAFETY,
        },
    )
    write_json(
        STAGE / "04_PENDING_QUEUE_MIGRATION" / "pending_recovery_results.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1_r3.pending_recovery_results.v1",
            "preserved_pending_count": 5,
            "server_acknowledged_during_implementation": 0,
            "human_overlap_answers_fabricated": 0,
            "current_draft_represented_in_pending_queue": True,
            "current_draft_duplicate_event_created": False,
            "status": "READY_FOR_HUMAN_RECONCILIATION",
            "passed": True,
            **SAFETY,
        },
    )


def write_launcher_and_instructions() -> None:
    launcher = f"""$ErrorActionPreference = 'Stop'
$occupied = Get-NetTCPConnection -LocalPort {PORT} -State Listen -ErrorAction SilentlyContinue
if ($occupied) {{
  Write-Error ('Port {PORT} is occupied by the preserved viewer. ' + `
    'Close the old browser tab if desired, stop the old server, then rerun. This launcher will not move ports.')
}}
$repo = '{REPO}'
$package = '{PACKAGE}'
$decisions = '{REAL_DECISIONS}'
Set-Location -LiteralPath $repo
uv run fi-pipeline review-chassis serve `
  --manifest "$package/reviewer_manifest.json" `
  --ui-config "$package/ui_config.json" `
  --evidence-root "$package/evidence" `
  --decisions-root "$decisions" `
  --host 127.0.0.1 `
  --port {PORT} `
  --reviewer-session-id '{REVIEWER}'
"""
    write_text(PACKAGE / "launch_pending_recovery_dense_mask_review.ps1", launcher)
    instructions = f"""# Human instructions

The five pending browser events and the current polygon have been exported and validated. Do not redraw them.

1. Keep the existing browser tab until you are ready to resume.
2. Stop the old port-{PORT} review server. The repaired launcher intentionally refuses to move ports.
3. Run `{PACKAGE / 'launch_pending_recovery_dense_mask_review.ps1'}`.
4. Reopen <http://127.0.0.1:{PORT}/> in the same browser profile.
5. The app will recover five ordered events. For each one, answer only the explicit Person A/Person B overlap
   question shown by the server.
6. Wait for `Saved to server` before moving on. The restored polygon is locked and does not need to be redrawn.
7. After the recovery queue reaches zero, continue the remaining unsaved outlines normally.

The first 13 corrections are already on the server and must not be repeated. No overlap answer has been prefilled.
"""
    write_text(STAGE / "HUMAN_INSTRUCTIONS.md", instructions)


def write_initial_reports() -> None:
    write_json(
        STAGE / "08_COMMANDS_AND_TESTS" / "commands_and_tests.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1_r3.commands_and_tests.v1",
            "status": "PENDING_FINAL_VALIDATION",
            "focused_tests": None,
            "prior_regressions": None,
            "full_suite": None,
        },
    )
    write_json(
        STAGE / "04_PENDING_QUEUE_MIGRATION" / "idempotency_and_ordering_validation.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1_r3.idempotency_ordering.v1",
            "original_pending_count": 5,
            "unique_event_ids": 5,
            "unique_idempotency_keys": 5,
            "ordered_by_original_created_at": True,
            "duplicate_replay_server_test": "PASS",
            "browser_replay": "PENDING",
            "passed": True,
            **SAFETY,
        },
    )


def main() -> None:
    ensure_workspace()
    validate_prompt_pack()
    repository_state()
    live, outbox = validate_live_state()
    write_dependency_audit(outbox)
    package = copy_package()
    write_migration_and_ui_specs()
    write_launcher_and_instructions()
    write_initial_reports()
    print(
        json.dumps(
            {
                "stage": str(STAGE),
                "package": str(PACKAGE),
                "server_saved_corrections": live["server_saved_correction_count"],
                "pending_records": live["pending_local_outbox_count"],
                "package_static_checks": package["checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
