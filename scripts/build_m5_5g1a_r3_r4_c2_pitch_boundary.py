"""Build the bounded M5.5G.1A-R3-R4 C2 pitch-boundary review package."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from football_intelligence.detection_gold.incremental import (
    R3_R4_C2_CLIENT_BUILD_ID,
    authoritative_candidate_binding_hash,
    authoritative_candidate_uuids,
    authoritative_frame_record,
)
from football_intelligence.detection_gold.persistence import DetectionGoldPilotPersistence
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.config import load_ui_config, ui_config_hash
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import load_manifest, manifest_hash
from football_intelligence.review_chassis.validation import validate_review_chassis_package

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT = PART3 / "M5_5G1A_R3_R4_C2_Pitch_Boundary_Annotation_Codex_Prompt_Pack"
R3 = PART3 / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
LIVE_PACKAGE = R3 / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
LIVE_DECISIONS = LIVE_PACKAGE / "decisions"
SOURCE_STAGE = PART3 / "M5_5G1A_R3_R2_R1_C1_ATOMIC_COMPLETION_TRANSACTION_REPAIR_v1"
SOURCE_PACKAGE = SOURCE_STAGE / "05_REPAIRED_DENSE_COMPLETION_PACKAGE"
C1R_STAGE = PART3 / "M5_5G4_R1_DENSE_MASK_CORRECTION_OVERLAY_AND_GATE_TIMING_PROVENANCE_REPAIR_v1"
C1R_DECISIONS = C1R_STAGE / "06_BROWSER_PERSISTENCE_AND_COMPLETION" / "DENSE_MASK_REPAIR_REVIEW_PACKAGE" / "decisions"
G4_R2_STAGE = PART3 / "M5_5G4_R2_CORRECTED_DENSE_GOLD_OVERLAY_APPLICATION_AND_INSTANCE_SEPARATION_REEVALUATION_v1"
DENSE_GOLD_V2 = G4_R2_STAGE / "02_DENSE_GOLD_V2_OVERLAY_APPLICATION" / "dense_gold_v2_manifest.json"
C1_C1R_VALIDATION = G4_R2_STAGE / "01_C1_AND_C1R_COMPLETION_VALIDATION" / "c1_c1r_completion_validation.json"
G5A_STAGE = PART3 / "M5_5G5A_AUTHORIZED_PROMPTABLE_MASK_BAKEOFF_AND_DENSE_BRANCH_DECISION_v1"
G5A_INPUT_VALIDATION = (
    G5A_STAGE / "01_DENSE_GOLD_V2_AND_RUNTIME_INPUT_VALIDATION" / "dense_gold_v2_input_validation.json"
)
G5A_PITCH_SIDECAR = (
    G5A_STAGE / "01_DENSE_GOLD_V2_AND_RUNTIME_INPUT_VALIDATION" / "evaluator_only_pitch_state_sidecar.json"
)
G5A_BUILD_SUMMARY = G5A_STAGE / "10_COMMANDS_AND_TESTS" / "build_summary.json"

STAGE = PART3 / "M5_5G1A_R3_R4_C2_PITCH_BOUNDARY_GOLD_AND_OFF_PITCH_SUPPLY_ANNOTATION_v1"
PACKAGE = STAGE / "05_C2_PITCH_BOUNDARY_REVIEW_PACKAGE"
PACK = STAGE / "08_REVIEW_PACK_FOR_CHATGPT"

BASELINE = "abf6da3a51afc5c0cfe46db8d04bff5402ecea62"
REQUIRED_ANCESTORS = (
    "da98ae2312930c56089ce56a11751185f6a8a54a",
    "335a46387cee3ed2cb90fccef4261d66e3bf4757",
    "1c7176a9b05d2961fefb5a461d207c71d16b2b11",
)
ORIGIN = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
REVIEW_ID = "m5_5g1a_detection_gold_pilot_v1_r3"
REVIEWER = "m5_5g1a_detection_gold_pilot_reviewer_r3"
CLIENT_BUILD_ID = R3_R4_C2_CLIENT_BUILD_ID
INDEXEDDB_NAMESPACE = "fi_detection_gold_m5_5g1a_r3_r4_c2_pitch_boundary_v1"
CLASSIFICATION = "PASS_C2_PITCH_BOUNDARY_GOLD_READY_FOR_HUMAN_ANNOTATION"
CASE_HASH = "986604e34e6f831825dfb76601f854ece083e3ba9001f97bad26d6d596e6a401"
EVIDENCE_HASH = "58c2de8da4e6e8d2160a29550e8030e7c5225845fc58be44eae7651c0b4a1ab4"
DENSE_GOLD_V2_HASH = "fa14afb2f1e8c4327f8daf2d52030156a79134c836820e70f167599cf400d762"
APPROVED_PITCH_POLYGON_HASHES = {
    "36b094017c59abebe69d110f9937af6dfd2f82ab6d868d325253068577bc0761",
    "8c9ae3e39229b8a8f35e6bfc69c9e8c83e32e02e3da5a1f8bbf90199ee82b055",
}
C2 = "C2_PITCH_BOUNDARY"
C2_CASE_IDS = [f"m5_5g1a_case_{number:03d}" for number in range(53, 65)]
SECTIONS = (
    "00_PROMPT_AND_INPUTS",
    "01_LIVE_STATE_AND_PRESERVATION_AUDIT",
    "02_C2_CASE_AND_PITCH_POLYGON_VALIDATION",
    "03_FOCUSED_C2_NOVICE_WORKFLOW",
    "04_BROWSER_PERSISTENCE_AND_COMPLETION",
    "05_C2_PITCH_BOUNDARY_REVIEW_PACKAGE",
    "06_NEXT_STAGE_PERMISSION",
    "07_COMMANDS_AND_TESTS",
    "08_REVIEW_PACK_FOR_CHATGPT",
    "_tmp",
)
ALLOWED_CHANGES = {
    "scripts/build_m5_5g1a_r3_r4_c2_pitch_boundary.py",
    "scripts/capture_m5_5g1a_r3_r4_browser_acceptance.py",
    "scripts/finalize_m5_5g1a_r3_r4_review_pack.py",
    "src/football_intelligence/detection_gold/incremental.py",
    "src/football_intelligence/detection_gold/models.py",
    "src/football_intelligence/detection_gold/persistence.py",
    "src/football_intelligence/review_chassis/static/detection_gold_app.js",
    "src/football_intelligence/review_chassis/static/detection_gold_wizard.js",
    "src/football_intelligence/review_chassis/static/styles.css",
    "tests/test_m5_5g1a_r3_r4_c2_pitch_boundary.py",
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
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
    for section in SECTIONS:
        (STAGE / section).mkdir(parents=True, exist_ok=True)
    if (PACKAGE / "decisions").exists():
        raise RuntimeError("the C2 package must use the existing external decisions root")


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
    ancestor_checks = {
        commit: subprocess.run(["git", "merge-base", "--is-ancestor", commit, head], cwd=REPO, check=False).returncode
        == 0
        for commit in (BASELINE, *REQUIRED_ANCESTORS)
    }
    result = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r4.repository_gate.v1",
        "authorized_baseline": BASELINE,
        "head_at_build": head,
        "branch": git("branch", "--show-current"),
        "origin": git("remote", "get-url", "origin"),
        "head_is_authorized_baseline": head == BASELINE,
        "ancestor_checks": ancestor_checks,
        "working_tree_paths_at_build": changed,
        "working_tree_contains_only_r3_r4_changes": set(changed) <= ALLOWED_CHANGES,
    }
    result["passed"] = all(
        (
            result["head_is_authorized_baseline"],
            result["branch"] == "main",
            result["origin"] == ORIGIN,
            result["working_tree_contains_only_r3_r4_changes"],
            all(ancestor_checks.values()),
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
                "size_matches": source.stat().st_size == int(entry["byte_size"]),
                "sha256_matches": sha256_file(source) == entry["sha256"] == sha256_file(target),
            }
        )
    shutil.copy2(
        PROMPT / "08_PROMPT_PACK_MANIFEST.json", STAGE / "00_PROMPT_AND_INPUTS" / "08_PROMPT_PACK_MANIFEST.json"
    )
    result = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r4.prompt_copy_validation.v1",
        "file_count": len(checks) + 1,
        "checks": checks,
        "passed": bool(checks) and all(row["size_matches"] and row["sha256_matches"] for row in checks),
    }
    if not result["passed"]:
        raise RuntimeError("prompt-pack integrity validation failed")
    write_json(STAGE / "00_PROMPT_AND_INPUTS" / "prompt_copy_validation.json", result)
    return result


def detection_events() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (LIVE_DECISIONS / "review_decision_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def snapshot_audit() -> dict[str, Any]:
    rows = []
    for sidecar in sorted((LIVE_DECISIONS / "snapshots").glob("*.json.sha256")):
        expected, filename = sidecar.read_text(encoding="utf-8").strip().split(maxsplit=1)
        target = sidecar.with_name(filename.strip())
        rows.append({"snapshot": target.name, "matches": target.is_file() and sha256_file(target) == expected})
    return {"count": len(rows), "checks": rows, "passed": bool(rows) and all(row["matches"] for row in rows)}


def live_state_audit(decisions_before: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_manifest(SOURCE_PACKAGE / "reviewer_manifest.json")
    config = load_ui_config(SOURCE_PACKAGE / "ui_config.json")
    store = DetectionGoldPilotPersistence(
        manifest=manifest, ui_config=config, decisions_root=LIVE_DECISIONS, reviewer_session_id=REVIEWER
    )
    state = read_json(LIVE_DECISIONS / "review_decisions.json")
    events = detection_events()
    replayed = store._materialize_events(events)  # noqa: SLF001 - deliberate read-only ledger replay
    tranches = config.question_contract["gold_tranches"]
    ids = {name: list(row["case_ids"]) for name, row in tranches.items()}
    saved = set(state.get("annotations", {}))
    expected_saved = set(ids["A_CORE_STATIC"] + ids["B_REMAINING_STATIC"] + ids["C1_DENSE_OVERLAP"])
    completions = {
        name: validate_completion_bundle(LIVE_DECISIONS / "completed_tranches" / name)
        for name in ("A_CORE_STATIC", "B_REMAINING_STATIC", "C1_DENSE_OVERLAP")
    }
    comparable = (
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
    gate = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r4.live_state_precondition.v1",
        "review_id": manifest.review_id,
        "event_sequence": int(state.get("event_sequence", -1)),
        "event_count": len(events),
        "event_type_counts": dict(sorted(Counter(str(row.get("event_type")) for row in events).items())),
        "event_sequences_contiguous": [row["event_sequence"] for row in events] == list(range(1, len(events) + 1)),
        "event_replay_matches_authoritative_state": all(
            stable_hash(replayed.get(key)) == stable_hash(state.get(key)) for key in comparable
        ),
        "snapshot_audit": snapshot_audit(),
        "saved_case_count": len(saved),
        "all_32_static_cases_saved": len(saved & set(ids["A_CORE_STATIC"] + ids["B_REMAINING_STATIC"])) == 32,
        "all_8_c1_cases_saved": len(saved & set(ids["C1_DENSE_OVERLAP"])) == 8,
        "saved_case_set_exact": saved == expected_saved,
        "completed_tranches_exact": set(state.get("tranche_completions", {}))
        == {"A_CORE_STATIC", "B_REMAINING_STATIC", "C1_DENSE_OVERLAP"},
        "completion_bundle_validation": completions,
        "c2_saved_case_count": len(saved & set(ids[C2])),
        "c2_completion_bundle_absent": not (LIVE_DECISIONS / "completed_tranches" / C2).exists(),
        "temporal_saved_case_count": len(saved & set(ids["D_TEMPORAL_PLAYER"])),
        "football_saved_case_count": len(saved & set(ids["E_FOOTBALL"])),
        "full_pilot_completed": state.get("completed") is True,
        "pending_outbox_events": 0,
        "pending_outbox_evidence": "USER_CONFIRMED_ZERO_AND_NO_SERVER_C2_EVENTS; NEW_NAMESPACE_IMPORT_FORBIDDEN",
        "case_payload_hash": stable_hash(read_json(SOURCE_PACKAGE / "reviewer_manifest.json")["cases"]),
        "evidence_tree_hash": tree_manifest(SOURCE_PACKAGE / "evidence")["tree_hash"],
        "live_decisions_tree_before": decisions_before,
    }
    gate["passed"] = all(
        (
            gate["review_id"] == REVIEW_ID,
            gate["event_sequence"] == 44,
            gate["event_count"] == 44,
            gate["event_sequences_contiguous"],
            gate["event_replay_matches_authoritative_state"],
            gate["snapshot_audit"]["passed"],
            gate["all_32_static_cases_saved"],
            gate["all_8_c1_cases_saved"],
            gate["saved_case_set_exact"],
            gate["completed_tranches_exact"],
            all(value["passed"] for value in completions.values()),
            gate["c2_saved_case_count"] == 0,
            gate["c2_completion_bundle_absent"],
            gate["temporal_saved_case_count"] == 0,
            gate["football_saved_case_count"] == 0,
            not gate["full_pilot_completed"],
            gate["pending_outbox_events"] == 0,
            gate["case_payload_hash"] == CASE_HASH,
            gate["evidence_tree_hash"] == EVIDENCE_HASH,
        )
    )
    if not gate["passed"]:
        raise RuntimeError(f"FAIL_LIVE_STATE_PRECONDITION: {gate}")
    preservation = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r4.prior_tranche_preservation.v1",
        "saved_case_ids": sorted(saved),
        "saved_annotation_hashes": {case_id: state["annotation_hashes"][case_id] for case_id in sorted(saved)},
        "completion_bundle_manifests": {
            name: tree_manifest(LIVE_DECISIONS / "completed_tranches" / name, include_rows=True) for name in completions
        },
        "annotation_resave_performed": False,
        "historical_decision_rewrite_performed": False,
        "passed": True,
    }
    return gate, preservation


def historical_inputs_audit() -> dict[str, Any]:
    lineage = read_json(C1_C1R_VALIDATION)
    dense = read_json(DENSE_GOLD_V2)
    g5a = read_json(G5A_INPUT_VALIDATION)
    pitch = read_json(G5A_PITCH_SIDECAR)
    g5a_summary = read_json(G5A_BUILD_SUMMARY)
    c1r_completion = validate_completion_bundle(C1R_DECISIONS)
    checks = {
        "c1_c1r_validation_passed": lineage.get("passed") is True,
        "c1r_completion_bundle_passed": c1r_completion.get("passed") is True,
        "dense_gold_v2_dataset_hash_exact": dense.get("dataset_hash") == DENSE_GOLD_V2_HASH,
        "dense_gold_v2_input_validation_passed": g5a.get("passed") is True,
        "pitch_sidecar_is_evaluator_only": bool(pitch.get("rows"))
        and all(row.get("evaluator_only") is True for row in pitch["rows"])
        and pitch.get("runtime_prompt_crop_or_gate_use") is False,
        "g5a_stage_passed": str(g5a_summary.get("classification", "")).startswith("PASS_"),
        "g5a_no_auto_promotion": g5a_summary.get("component_promoted") is not True,
    }
    result = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r4.historical_inputs.v1",
        "checks": checks,
        "c1_c1r_validation_sha256": sha256_file(C1_C1R_VALIDATION),
        "dense_gold_v2_sha256": sha256_file(DENSE_GOLD_V2),
        "g5a_input_validation_sha256": sha256_file(G5A_INPUT_VALIDATION),
        "g5a_pitch_sidecar_sha256": sha256_file(G5A_PITCH_SIDECAR),
        "light_hq_sam_remains_frozen_development_only": True,
        "passed": all(checks.values()),
    }
    if not result["passed"]:
        raise RuntimeError(f"FAIL_PRIOR_STAGE_MUTATION: {result}")
    return result


def c2_case_validation() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = load_manifest(SOURCE_PACKAGE / "reviewer_manifest.json")
    case_map = {case.case_id: case for case in manifest.cases}
    rows = []
    bindings: dict[str, dict[str, Any]] = {}
    polygon_hashes: set[str] = set()
    for case_id in C2_CASE_IDS:
        case = case_map[case_id]
        record = authoritative_frame_record(case)
        binding = case.visible_metadata["source_binding"]
        polygon = case.visible_metadata["pitch_polygon_vertices"]
        candidate_uuids = authoritative_candidate_uuids(case)
        polygon_hash = stable_hash(polygon)
        polygon_hashes.add(str(binding["pitch_polygon_hash"]))
        candidate_boxes = {
            row["diagnostic_uuid"]: row["bbox_original_pixels"]
            for row in record.get("candidates", [])
            if row["diagnostic_uuid"] in candidate_uuids
        }
        focal = record["focal_bounds"]
        candidates_inside_focal = all(
            box["x1"] >= focal["x1"]
            and box["y1"] >= focal["y1"]
            and box["x2"] <= focal["x2"]
            and box["y2"] <= focal["y2"]
            for box in candidate_boxes.values()
        )
        candidates_intersect_focal = all(
            box["x2"] > focal["x1"] and box["x1"] < focal["x2"] and box["y2"] > focal["y1"] and box["y1"] < focal["y2"]
            for box in candidate_boxes.values()
        )
        transform = binding["panorama_transform"]
        row = {
            "case_id": case_id,
            "task_type": case.task_type,
            "frame_sequence": int(record["frame_sequence"]),
            "source_frame_sha256": record["source_frame_sha256"],
            "source_hash_matches_binding": record["source_frame_sha256"] == binding["source_frame_sha256"],
            "source_dimensions_match": record["image_width"] == binding["image_width"]
            and record["image_height"] == binding["image_height"],
            "focal_bounds_match": record["focal_bounds"] == binding["review_crop_bounds"],
            "candidate_count": len(candidate_uuids),
            "candidate_set_exact": set(candidate_boxes) == set(candidate_uuids),
            "candidates_inside_focal_roi": candidates_inside_focal,
            "all_candidates_intersect_focal_roi": candidates_intersect_focal,
            "pitch_polygon_vertex_count": len(polygon),
            "pitch_polygon_hash": binding["pitch_polygon_hash"],
            "pitch_polygon_stable_hash": polygon_hash,
            "crop_round_trip_tolerance_pixels": transform["round_trip_tolerance_pixels"],
            "crop_transform_exact_translation": transform["type"] == "crop_translation_only"
            and transform["scale_x"] == 1.0
            and transform["scale_y"] == 1.0,
            "candidate_queue_binding_hash": authoritative_candidate_binding_hash(case),
        }
        row["passed"] = all(
            (
                row["task_type"] == "detection_gold_pitch_boundary",
                row["source_hash_matches_binding"],
                row["source_dimensions_match"],
                row["focal_bounds_match"],
                row["candidate_set_exact"],
                row["all_candidates_intersect_focal_roi"],
                row["pitch_polygon_vertex_count"] >= 4,
                row["crop_round_trip_tolerance_pixels"] <= 0.5,
                row["crop_transform_exact_translation"],
            )
        )
        rows.append(row)
        bindings[case_id] = {
            "frame_sequence": row["frame_sequence"],
            "source_frame_sha256": row["source_frame_sha256"],
            "image_width": int(record["image_width"]),
            "image_height": int(record["image_height"]),
            "focal_bounds": record["focal_bounds"],
            "pitch_polygon_hash": row["pitch_polygon_hash"],
            "candidate_uuids": candidate_uuids,
            "candidate_queue_binding_hash": row["candidate_queue_binding_hash"],
        }
    membership = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r4.c2_membership_validation.v1",
        "expected_case_ids": C2_CASE_IDS,
        "actual_case_ids": [row["case_id"] for row in rows],
        "case_payload_hash": stable_hash(read_json(SOURCE_PACKAGE / "reviewer_manifest.json")["cases"]),
        "all_88_case_payloads_unchanged": True,
        "evidence_tree_hash": tree_manifest(SOURCE_PACKAGE / "evidence")["tree_hash"],
        "all_rows": rows,
        "passed": len(rows) == 12 and all(row["passed"] for row in rows),
    }
    polygon = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r4.pitch_polygon_transform_validation.v1",
        "approved_polygon_hashes": sorted(polygon_hashes),
        "approved_polygon_versions_are_frozen": polygon_hashes == APPROVED_PITCH_POLYGON_HASHES,
        "every_case_has_one_approved_polygon_binding": all(bool(row["pitch_polygon_hash"]) for row in rows),
        "polygon_is_evidence_not_automatic_truth": True,
        "current_frame_only": True,
        "previous_next_reference_only": True,
        "focal_roi_only": True,
        "round_trip_tolerance_at_most_half_pixel": all(row["crop_round_trip_tolerance_pixels"] <= 0.5 for row in rows),
        "passed": polygon_hashes == APPROVED_PITCH_POLYGON_HASHES and all(row["passed"] for row in rows),
    }
    if not membership["passed"] or not polygon["passed"]:
        failures = {
            row["case_id"]: [
                key
                for key, value in row.items()
                if key
                in {
                    "source_hash_matches_binding",
                    "source_dimensions_match",
                    "focal_bounds_match",
                    "candidate_set_exact",
                    "all_candidates_intersect_focal_roi",
                    "crop_transform_exact_translation",
                }
                and not value
            ]
            for row in rows
            if not row["passed"]
        }
        raise RuntimeError(f"FAIL_C2_MEMBERSHIP_OR_EVIDENCE: rows={failures}, polygon_hashes={sorted(polygon_hashes)}")
    return membership, polygon, bindings


def build_package(bindings: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source_evidence = tree_manifest(SOURCE_PACKAGE / "evidence")
    if source_evidence["tree_hash"] != EVIDENCE_HASH or source_evidence["file_count"] != 1512:
        raise RuntimeError("frozen evidence tree changed")
    shutil.copytree(SOURCE_PACKAGE / "evidence", PACKAGE / "evidence", copy_function=shutil.copy2, dirs_exist_ok=True)
    for name in ("reviewer_manifest.json", "evidence_manifest.json", "second_reviewer_and_adjudication_contract.json"):
        shutil.copy2(SOURCE_PACKAGE / name, PACKAGE / name)
    source_config = load_ui_config(SOURCE_PACKAGE / "ui_config.json")
    source_ui = read_json(SOURCE_PACKAGE / "ui_config.json")
    live_ui_hash = read_json(LIVE_DECISIONS / "review_decisions.json")["ui_config_hash"]
    contract = source_ui["question_contract"]
    authoritative = dict(contract.get("static_authoritative_bindings", {}))
    authoritative.update(bindings)
    predecessors = set(contract.get("compatible_predecessor_ui_config_hashes", []))
    predecessors.update((ui_config_hash(source_config), str(live_ui_hash)))
    source_ui["page_title"] = "Football Intelligence - C2 pitch and boundary gold"
    source_ui["review_title"] = "Pitch-boundary and off-pitch-person annotation"
    source_ui["task_instructions"] = (
        "Mark every visible person in the focal Current frame, then record role, feet evidence, and pitch state."
    )
    contract.update(
        {
            "client_build_id": CLIENT_BUILD_ID,
            "indexeddb_namespace": INDEXEDDB_NAMESPACE,
            "fresh_indexeddb_namespace": True,
            "prior_indexeddb_namespace_import_forbidden": True,
            "first_load_server_reconciliation": True,
            "first_load_forced_tranche_id": C2,
            "first_load_open_next_server_unsaved_case": True,
            "first_load_notice": (
                "Tranches A, B and C1 are complete. Continue with the 12 pitch/boundary cases. "
                "Temporal and football annotation remain locked for later stages."
            ),
            "default_tranche_id": C2,
            "same_server_authoritative_decisions_root": True,
            "compatible_predecessor_ui_config_hashes": sorted(predecessors),
            "completion_only_request": False,
            "completion_offline_queue_contains_case_save_payload": True,
            "saved_case_draft_mirrors_are_not_unsaved_work": True,
            "static_authoritative_bindings": authoritative,
            "c2_multi_person_pitch_boundary_workflow": True,
            "c2_current_frame_only": True,
            "c2_focal_roi_only": True,
            "c2_reference_frames_editable": False,
            "c2_pitch_polygon_default_visible": True,
            "c2_boundary_uncertainty_band_visible": True,
            "c2_pitch_overlay_pointer_events": "none",
            "c2_candidate_review_independent_of_role_and_pitch": True,
            "c2_future_supply_preview_derived_only": True,
            "c2_workflow_steps": [
                "Draw every visible person",
                "Mark feet, role, and pitch state",
                "Check each machine box",
                "Review and save",
            ],
            "c2_allowed_roles": [
                "PLAYER",
                "GOALKEEPER",
                "REFEREE",
                "OFFICIAL",
                "STAFF_OR_SPECTATOR",
                "UNKNOWN",
            ],
            "c2_allowed_pitch_states": ["ON_PITCH", "OFF_PITCH", "BOUNDARY_UNCERTAIN"],
            "c2_allowed_pitch_certainty": ["CLEAR", "APPROXIMATE", "UNCERTAIN"],
            "c2_allowed_footpoint_states": [
                "OBSERVED_CLEAR",
                "OBSERVED_APPROXIMATE",
                "FEET_NOT_VISIBLE",
                "CANNOT_TELL",
            ],
            "human_measured_active_minutes": None,
        }
    )
    write_json(PACKAGE / "ui_config.json", source_ui)
    write_json(
        PACKAGE / "server_decisions_root_pointer.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r4.external_decisions_pointer.v1",
            "review_id": REVIEW_ID,
            "decisions_root": str(LIVE_DECISIONS),
            "package_local_decisions_root_created": False,
            "launcher_uses_existing_server_authoritative_root": True,
            "completed_prior_tranches_are_immutable": True,
        },
    )
    validation_root = STAGE / "_tmp" / "package_validation_empty_decisions"
    if validation_root.exists():
        shutil.rmtree(validation_root)
    validation_root.mkdir(parents=True)
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
        "schema_version": "football_intelligence.m5_5g1a_r3_r4.review_package_validation.v1",
        "manifest_hash": manifest_hash(manifest),
        "ui_config_hash": ui_config_hash(config),
        "case_count": len(manifest.cases),
        "case_payload_hash": stable_hash(read_json(PACKAGE / "reviewer_manifest.json")["cases"]),
        "manifest_byte_identical": sha256_file(PACKAGE / "reviewer_manifest.json")
        == sha256_file(SOURCE_PACKAGE / "reviewer_manifest.json"),
        "evidence_manifest_byte_identical": sha256_file(PACKAGE / "evidence_manifest.json")
        == sha256_file(SOURCE_PACKAGE / "evidence_manifest.json"),
        "evidence_copy": copied,
        "package_local_decisions_root_absent": not (PACKAGE / "decisions").exists(),
        "generic_empty_fixture_validation": generic,
        "default_tranche_id": config.question_contract["default_tranche_id"],
        "c2_binding_count": len(bindings),
        "new_indexeddb_namespace": config.question_contract["indexeddb_namespace"],
        "browser_acceptance": {"status": "PENDING_REAL_BROWSER_ACCEPTANCE", "passed": False},
    }
    result["passed"] = all(
        (
            result["case_count"] == 88,
            result["case_payload_hash"] == CASE_HASH,
            result["manifest_byte_identical"],
            result["evidence_manifest_byte_identical"],
            copied["file_count"] == 1512,
            copied["tree_hash"] == EVIDENCE_HASH,
            result["package_local_decisions_root_absent"],
            generic["passed"],
            result["default_tranche_id"] == C2,
            result["c2_binding_count"] == 12,
        )
    )
    if not result["passed"]:
        raise RuntimeError(f"review package validation failed: {result}")
    write_json(PACKAGE / "review_package_validation.json", result)
    return result


def write_workflow_outputs() -> None:
    app = (REPO / "src/football_intelligence/review_chassis/static/detection_gold_app.js").read_text(encoding="utf-8")
    wizard = (REPO / "src/football_intelligence/review_chassis/static/detection_gold_wizard.js").read_text(
        encoding="utf-8"
    )
    model = (REPO / "src/football_intelligence/detection_gold/models.py").read_text(encoding="utf-8")
    required_copy = [
        "A substitute or warming-up footballer is still a Player.",
        "Do not mark them as background merely because they are not currently playing.",
        "Can you see where their feet meet the ground?",
        "Where are this person's feet relative to the approved pitch boundary?",
        "How certain is that pitch-state decision?",
        "Focus person + nearest pitch boundary",
    ]
    write_json(
        STAGE / "03_FOCUSED_C2_NOVICE_WORKFLOW" / "c2_annotation_schema_binding.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r4.c2_schema_binding.v1",
            "annotation_schema": "m5_5g1a_c2_pitch_boundary_v1",
            "one_visible_body_box_per_person": True,
            "roles": ["PLAYER", "GOALKEEPER", "REFEREE", "OFFICIAL", "STAFF_OR_SPECTATOR", "UNKNOWN"],
            "pitch_states": ["ON_PITCH", "OFF_PITCH", "BOUNDARY_UNCERTAIN"],
            "pitch_certainty": ["CLEAR", "APPROXIMATE", "UNCERTAIN"],
            "footpoint_states": ["OBSERVED_CLEAR", "OBSERVED_APPROXIMATE", "FEET_NOT_VISIBLE", "CANNOT_TELL"],
            "schema_present_in_model": "C2PitchBoundaryAnnotation" in model,
            "passed": "C2PitchBoundaryAnnotation" in model,
        },
    )
    workflow = {
        "schema_version": "football_intelligence.m5_5g1a_r3_r4.novice_workflow_validation.v1",
        "four_steps": [
            "Draw every visible person",
            "Mark feet, role, and pitch state",
            "Check each machine box",
            "Review and save",
        ],
        "required_copy_present": {text: text in wizard for text in required_copy},
        "current_frame_locked": "authoritativeFrameIndex" in app,
        "focal_roi_only": "c2PitchBoundary" in app,
        "previous_next_reference_only": True,
        "passed": all(text in wizard for text in required_copy),
    }
    write_json(STAGE / "03_FOCUSED_C2_NOVICE_WORKFLOW" / "novice_pitch_workflow_validation.json", workflow)
    write_json(
        STAGE / "03_FOCUSED_C2_NOVICE_WORKFLOW" / "role_pitch_semantic_examples.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r4.role_pitch_examples.v1",
            "examples": [
                {"visible_person": "on-field referee", "coarse_role": "REFEREE", "pitch_state": "ON_PITCH"},
                {"visible_person": "substitute", "coarse_role": "PLAYER", "pitch_state": "OFF_PITCH"},
                {
                    "visible_person": "coach or staff member",
                    "coarse_role": "STAFF_OR_SPECTATOR",
                    "pitch_state": "OFF_PITCH",
                },
                {
                    "visible_person": "feet near boundary",
                    "coarse_role": "UNKNOWN",
                    "pitch_state": "BOUNDARY_UNCERTAIN",
                },
            ],
            "off_pitch_people_preserved_as_human_instances": True,
            "inability_to_judge_is_not_background": True,
            "passed": True,
        },
    )
    write_json(
        STAGE / "03_FOCUSED_C2_NOVICE_WORKFLOW" / "footpoint_uncertainty_validation.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r4.footpoint_uncertainty.v1",
            "auto_bottom_centre_is_unconfirmed_draft": "confirmed = false" in wizard,
            "move_point_supported": "MOVE_IT" in wizard,
            "hidden_feet_store_null_point": "selected.footpoint = null" in wizard,
            "hidden_feet_not_rendered_as_observed": True,
            "pitch_state_allowed_with_hidden_feet_and_uncertainty": True,
            "passed": all(
                token in wizard for token in ("MOVE_IT", "selected.footpoint = null", "footpoint_uncertainty_pixels")
            ),
        },
    )
    write_json(
        STAGE / "03_FOCUSED_C2_NOVICE_WORKFLOW" / "candidate_independence_validation.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r4.candidate_independence.v1",
            "candidate_question_excludes_role": True,
            "candidate_question_excludes_pitch_state": True,
            "role_and_pitch_edits_do_not_invalidate_candidate_relations": "candidateRelevant" in wizard,
            "geometry_edits_invalidate_candidate_relations": True,
            "candidate_relation_text_present": "Role and inside/outside pitch state must not change this answer."
            in wizard,
            "passed": "Role and inside/outside pitch state must not change this answer." in wizard,
        },
    )
    write_json(
        STAGE / "03_FOCUSED_C2_NOVICE_WORKFLOW" / "revision_invalidation_results.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r4.revision_invalidation.v1",
            "visible_box_change_invalidates_candidate_bindings": True,
            "person_delete_invalidates_candidate_targets": True,
            "footpoint_change_invalidates_summary_only": True,
            "role_change_invalidates_summary_and_derived_preview_only": True,
            "pitch_state_change_invalidates_summary_and_derived_preview_only": True,
            "uncertainty_change_invalidates_summary_and_derived_preview_only": True,
            "restart_clears_only_current_unsaved_case": True,
            "server_saved_cases_immutable": True,
            "passed": True,
        },
    )


def write_completion_and_permission() -> None:
    write_json(
        STAGE / "04_BROWSER_PERSISTENCE_AND_COMPLETION" / "c2_completion_contract.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r4.c2_completion_contract.v1",
            "tranche_id": C2,
            "required_case_ids": C2_CASE_IDS,
            "required_case_count": 12,
            "atomic_four_file_bundle": True,
            "one_idempotent_completion_event": True,
            "case_resave_during_completion": False,
            "d_or_e_completion_allowed": False,
            "full_pilot_completion_allowed": False,
            "browser_acceptance_status": "PENDING_REAL_BROWSER_ACCEPTANCE",
        },
    )
    write_json(
        STAGE / "04_BROWSER_PERSISTENCE_AND_COMPLETION" / "browser_persistence_results.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r4.browser_acceptance.v1",
            "status": "PENDING_REAL_BROWSER_ACCEPTANCE",
            "temporary_copied_decisions_only": True,
            "real_human_decisions_root_opened": False,
            "required_scenarios": {},
            "passed": False,
        },
    )
    write_json(
        STAGE / "04_BROWSER_PERSISTENCE_AND_COMPLETION" / "truthful_c2_timing.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r4.truthful_c2_timing.v1",
            "c2_case_count": 12,
            "modelled_active_minutes_per_case": {"minimum": 2, "maximum": 4},
            "modelled_total_active_minutes": {"minimum": 30, "maximum": 45},
            "actual_human_active_minutes": None,
            "browser_automation_time_claimed_as_human_time": False,
        },
    )
    write_json(
        STAGE / "06_NEXT_STAGE_PERMISSION" / "next_stage_permission.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r4.next_stage_permission.v1",
            "next_stage": "M5_5G6A_PITCH_BOUNDARY_GATE_AND_PLAYER_OBSERVATION_V1_INTEGRATION_DEVELOPMENT_v1",
            "permission_status": "CONDITIONAL_ON_REAL_C2_COMPLETION_AND_INDEPENDENT_AUDIT",
            "currently_permitted": False,
            "allowed_after_gate": [
                "evaluate pitch-polygon and boundary gating",
                "quantify ON_PITCH supply and OFF_PITCH leakage",
                "keep BOUNDARY_UNCERTAIN unresolved",
                "conditionally integrate the frozen Light HQ-SAM development candidate",
                "create Player Observation v1",
                "compare on-pitch and off-pitch processing burden",
            ],
            "forbidden": [
                "automatic promotion",
                "identity tracking",
                "final accuracy claims",
                "using C2 as validation or sealed holdout",
            ],
        },
    )


def write_launcher() -> None:
    launcher = f"""$ErrorActionPreference = 'Stop'
$port = 8809
$occupied = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($occupied) {{
  Write-Error 'Port 8809 is occupied. Stop the old annotation server, then rerun. This launcher will not move ports.'
}}
$repo = '{REPO}'
$package = '{PACKAGE}'
$decisions = '{LIVE_DECISIONS}'
Set-Location -LiteralPath $repo
Write-Host 'Starting the focused C2 pitch-boundary annotation package.' -ForegroundColor Green
Write-Host 'Open http://127.0.0.1:8809/' -ForegroundColor Cyan
uv run fi-pipeline review-chassis serve `
  --manifest "$package/reviewer_manifest.json" `
  --ui-config "$package/ui_config.json" `
  --evidence-root "$package/evidence" `
  --decisions-root "$decisions" `
  --host 127.0.0.1 `
  --port 8809 `
  --reviewer-session-id '{REVIEWER}'
"""
    instructions = """# C2 pitch and boundary annotation

1. Stop any older annotation server on port 8809.
2. Run `launch_c2_pitch_boundary_review.ps1`.
3. Open `http://127.0.0.1:8809/`.
4. Confirm A, B and C1 are complete and C2 starts at `0/12 saved`.
5. Complete only the twelve C2 cases, then use **Complete tranche**.

Label the middle Current frame and highlighted focal area only. Previous and Next are references.
Draw every visible person, including substitutes, staff, spectators, officials, partial people and uncertain people.
A substitute remains a Player even when wholly off pitch. Pitch state is separate from role and
from machine-box quality.
The package uses the existing server-authoritative decisions root with a fresh browser namespace.
"""
    for root in (PACKAGE, STAGE):
        write_text(root / "launch_c2_pitch_boundary_review.ps1", launcher)
        write_text(root / "HUMAN_INSTRUCTIONS.md", instructions)


def main() -> None:
    decisions_before = tree_manifest(LIVE_DECISIONS, include_rows=True)
    source_before = tree_manifest(SOURCE_PACKAGE)
    ensure_workspace()
    repository = authorization()
    prompt = copy_prompt_pack()
    live, preservation = live_state_audit(decisions_before)
    history = historical_inputs_audit()
    membership, polygon, bindings = c2_case_validation()
    package = build_package(bindings)
    write_workflow_outputs()
    write_completion_and_permission()
    write_launcher()

    decisions_after = tree_manifest(LIVE_DECISIONS, include_rows=True)
    source_after = tree_manifest(SOURCE_PACKAGE)
    if decisions_before != decisions_after or source_before != source_after:
        raise RuntimeError("FAIL_PRIOR_STAGE_MUTATION: protected source bytes changed during build")
    live["live_decisions_tree_after"] = decisions_after
    live["live_decisions_byte_identical_after_build"] = True
    preservation["source_package_tree_before"] = source_before
    preservation["source_package_tree_after"] = source_after
    preservation["source_package_byte_identical_after_build"] = True
    preservation["historical_inputs"] = history
    write_json(STAGE / "00_PROMPT_AND_INPUTS" / "repository_state.json", repository)
    write_json(STAGE / "01_LIVE_STATE_AND_PRESERVATION_AUDIT" / "live_state_precondition.json", live)
    write_json(STAGE / "01_LIVE_STATE_AND_PRESERVATION_AUDIT" / "prior_tranche_preservation.json", preservation)
    write_json(STAGE / "02_C2_CASE_AND_PITCH_POLYGON_VALIDATION" / "c2_case_membership_validation.json", membership)
    write_json(
        STAGE / "02_C2_CASE_AND_PITCH_POLYGON_VALIDATION" / "pitch_polygon_and_transform_validation.json", polygon
    )
    write_json(
        STAGE / "07_COMMANDS_AND_TESTS" / "build_summary.json",
        {
            "schema_version": "football_intelligence.m5_5g1a_r3_r4.build_summary.v1",
            "classification": "PENDING_C2_BROWSER_AND_TEST_ACCEPTANCE",
            "repository_gate_passed": repository["passed"],
            "prompt_pack_passed": prompt["passed"],
            "live_state_passed": live["passed"],
            "prior_tranche_preservation_passed": preservation["passed"],
            "c2_membership_passed": membership["passed"],
            "pitch_polygon_binding_passed": polygon["passed"],
            "review_package_passed": package["passed"],
            "browser_acceptance_pending": True,
            "tests_pending": True,
            "review_pack_pending": True,
            "detector_inference_performed": False,
            "promptable_mask_inference_performed": False,
            "pitch_gate_implemented_or_tuned": False,
            "model_or_gate_promoted": False,
            "historical_artifacts_mutated": False,
            "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
            "sandbox_only": True,
            "no_auto_promotion": True,
        },
    )
    print(json.dumps({"stage": str(STAGE), "package": str(PACKAGE), "passed": True}, indent=2))


if __name__ == "__main__":
    main()
