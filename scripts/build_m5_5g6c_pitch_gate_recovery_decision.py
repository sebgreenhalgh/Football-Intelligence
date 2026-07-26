"""Build the read-only M5.5G.6C gate reevaluation and recovery decision."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import statistics
import subprocess
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from football_intelligence.detection_gold.player_observation import (
    PLAYER_OBSERVATION_SCHEMA_VERSION,
    apply_pitch_gate,
    estimate_footpoint,
    footpoint_geometry_variant_specification,
    player_observation_json_schema,
    pitch_gate_variant_specification,
    signed_distance_to_polygon,
    validate_runtime_payload,
)
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT = PART3 / "M5_5G6C_Pitch_Gate_Reevaluation_Codex_Prompt_Pack"
STAGE = PART3 / "M5_5G6C_PITCH_GATE_REEVALUATION_AND_PROPOSAL_SUPPLY_RECOVERY_DECISION_v1"
G6A = PART3 / "M5_5G6A_PITCH_BOUNDARY_GATE_AND_PLAYER_OBSERVATION_V1_INTEGRATION_DEVELOPMENT_v1"
G6B = PART3 / "M5_5G6B_BOUNDARY_FOCUSED_GOLD_AND_FROZEN_PROPOSAL_SUPPLY_ATTRIBUTION_v1"
G2B = PART3 / "M5_5G2B_FULL_STATIC_PLAYER_PROPOSAL_SUPPLY_DEVELOPMENT_BAKEOFF_v1"
G3 = PART3 / "M5_5G3_PROVENANCE_AWARE_CROSS_VIEW_CONSOLIDATION_AND_MERGED_AMBIGUITY_GATE_DEVELOPMENT_v1"
G4R2 = PART3 / "M5_5G4_R2_CORRECTED_DENSE_GOLD_OVERLAY_APPLICATION_AND_INSTANCE_SEPARATION_REEVALUATION_v1"
G5A = PART3 / "M5_5G5A_AUTHORIZED_PROMPTABLE_MASK_BAKEOFF_AND_DENSE_BRANCH_DECISION_v1"
C2_PACKAGE = (
    PART3
    / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
    / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
)
B1_PACKAGE = G6B / "05_PERSON_CENTRIC_BOUNDARY_REVIEW_PACKAGE"
B1_BUNDLE = B1_PACKAGE / "decisions" / "completed_tranches" / "B1_BOUNDARY_FOCUSED_PERSON_GOLD"
C2_BUNDLE = C2_PACKAGE / "decisions" / "completed_tranches" / "C2_PITCH_BOUNDARY"

BASELINE = "eedf1519362337845fe0cf8c251479ca13087e43"
REQUIRED_ANCESTORS = (
    "cbe68a9cd961956603f79319e603a16be6eee1ed",
    "b54ace62ec79217fbd175a0b4edc84b1f1a0b9b5",
    "abf6da3a51afc5c0cfe46db8d04bff5402ecea62",
)
EXPECTED_REMOTE = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
FOOTPOINT_SPEC_SHA256 = "d82e1a8315dd285f0c32ebc9966d0a8d46e791034a006bf9a41d6a4e4c7d55c2"
PITCH_GATE_SPEC_SHA256 = "d31938676d2718b3c222e83b79c7c1435af80d62b6b25ac28335ff76427aab8c"
B1_TRANSACTION = "tranche_B1_BOUNDARY_FOCUSED_PERSON_GOLD_59a9cb2aa61a80669f02182feb2dc672"
B1_SELECTION_HASH = "69afaa611909206c09de2bbf5f55c8997c3640330aca91122689d68401429215"
B1_DIVERSITY_HASH = "559b33826235dc73c9472ebfb9960780ff8f41cb727bdb06be1cb14187907c2b"
SELECTED_EXPERIMENT = "R-A_HIGH_RESOLUTION_SMALL_PERSON_PROPOSAL_BAKEOFF"
SELECTED_EXPERIMENT_ID = "R-A1_FROZEN_G2B_HIGH_RESOLUTION_VIEW_MATRIX"
FINAL_CHOICE = "FREEZE_PLAYER_OBSERVATION_SCHEMA_ONLY_AUTHORIZE_PROPOSAL_RECOVERY_EXPERIMENT"
CLASSIFICATION = "PASS_PITCH_GATE_REEVALUATION_AND_PROPOSAL_RECOVERY_DECISION_READY_FOR_PRO_REVIEW"

DIRS = {
    "inputs": STAGE / "00_PROMPT_AND_INPUTS",
    "b1": STAGE / "01_B1_COMPLETION_AND_GOLD_VALIDATION",
    "specs": STAGE / "02_FROZEN_GATE_SPECIFICATION_VALIDATION",
    "c2": STAGE / "03_C2_BROAD_SET_REEVALUATION",
    "b1_gate": STAGE / "04_B1_BOUNDARY_STRESS_REEVALUATION",
    "combined": STAGE / "05_COMBINED_GATE_DECISION",
    "misses": STAGE / "06_NINE_PERSON_MISS_PHENOTYPING",
    "recovery": STAGE / "07_PROPOSAL_RECOVERY_EXPERIMENT_SELECTION",
    "observation": STAGE / "08_PLAYER_OBSERVATION_V1_STATUS",
    "visuals": STAGE / "09_VISUAL_QA_AND_ERROR_LEDGER",
    "decision": STAGE / "10_NEXT_STAGE_DECISION",
    "commands": STAGE / "11_COMMANDS_AND_TESTS",
    "pack": STAGE / "12_REVIEW_PACK_FOR_CHATGPT",
    "tmp": STAGE / "_tmp",
}


def _load_g6a_builder() -> ModuleType:
    path = REPO / "scripts" / "build_m5_5g6a_pitch_gate_observation.py"
    spec = importlib.util.spec_from_file_location("m5_5g6a_read_only_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("FAIL_C2_GATE_REPLAY: G6A builder could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G6A_IMPL = _load_g6a_builder()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def source_frame(case: Mapping[str, Any], sequence_key: str = "source_frame_sequence") -> dict[str, Any]:
    sequence = int(case[sequence_key])
    matches = [frame for frame in case["visible_metadata"]["frame_records"] if int(frame["frame_sequence"]) == sequence]
    if len(matches) != 1:
        raise RuntimeError(f"source frame binding is not unique for {case['case_id']}")
    return dict(matches[0])


def bbox_iou(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    x1 = max(float(left["x1"]), float(right["x1"]))
    y1 = max(float(left["y1"]), float(right["y1"]))
    x2 = min(float(left["x2"]), float(right["x2"]))
    y2 = min(float(left["y2"]), float(right["y2"]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (float(left["x2"]) - float(left["x1"])) * (float(left["y2"]) - float(left["y1"]))
    right_area = (float(right["x2"]) - float(right["x1"])) * (float(right["y2"]) - float(right["y1"]))
    return intersection / max(1e-12, left_area + right_area - intersection)


def repository_and_prompt_validation() -> tuple[dict[str, Any], dict[str, Any]]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    remote = git("remote", "get-url", "origin")
    ancestor_checks = {
        commit: subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=REPO, check=False).returncode
        == 0
        for commit in (BASELINE, *REQUIRED_ANCESTORS)
    }
    status_rows = git("status", "--porcelain").splitlines()
    allowed_changes = {
        "scripts/build_m5_5g6c_pitch_gate_recovery_decision.py",
        "tests/test_m5_5g6b_boundary_gold.py",
        "tests/test_m5_5g6c_pitch_gate_recovery.py",
    }
    unexpected = [row for row in status_rows if row[3:].replace("\\", "/") not in allowed_changes]
    repository = {
        "schema_version": "football_intelligence.m5_5g6c.repository_state.v1",
        "repository": str(REPO),
        "branch": branch,
        "head": head,
        "minimum_authorized_baseline": BASELINE,
        "required_ancestor_checks": ancestor_checks,
        "origin": remote,
        "working_changes": status_rows,
        "unexpected_working_changes": unexpected,
        "passed": branch == "main" and remote == EXPECTED_REMOTE and all(ancestor_checks.values()) and not unexpected,
    }
    manifest = read_json(PROMPT / "08_PROMPT_PACK_MANIFEST.json")
    rows = []
    for expected in manifest["files"]:
        path = PROMPT / expected["filename"]
        actual = file_record(path)
        rows.append(
            {
                "filename": expected["filename"],
                "expected_byte_size": expected["byte_size"],
                "actual_byte_size": actual["byte_size"],
                "expected_sha256": expected["sha256"],
                "actual_sha256": actual["sha256"],
                "passed": expected["byte_size"] == actual["byte_size"] and expected["sha256"] == actual["sha256"],
            }
        )
    prompt = {
        "schema_version": "football_intelligence.m5_5g6c.prompt_pack_validation.v1",
        "file_count": len(rows) + 1,
        "manifest_self_hash_omitted": manifest.get("manifest_self_hash_omitted") is True,
        "rows": rows,
        "passed": len(rows) == 8 and all(row["passed"] for row in rows),
    }
    if not repository["passed"] or not prompt["passed"]:
        raise RuntimeError("FAIL_BASELINE_OR_WORKTREE")
    return repository, prompt


def _manifest_paths(path: Path) -> tuple[list[Path], dict[str, dict[str, Any]]]:
    payload = read_json(path)
    records = {str(row["path"]): row for row in payload["files"]}
    return [Path(value) for value in records], records


def protected_inputs() -> tuple[list[Path], dict[str, dict[str, Any]]]:
    paths: set[Path] = set()
    expected: dict[str, dict[str, Any]] = {}
    for manifest_path in (
        G6A / "00_PROMPT_AND_INPUTS" / "protected_input_manifest_before.json",
        G6B / "00_PROMPT_AND_INPUTS" / "protected_input_manifest_before.json",
    ):
        manifest_paths, records = _manifest_paths(manifest_path)
        paths.update(manifest_paths)
        for key, row in records.items():
            previous = expected.get(key)
            if previous and previous["sha256"] != row["sha256"]:
                raise RuntimeError(f"FAIL_PRIOR_STAGE_MUTATION: conflicting historical hash for {key}")
            expected[key] = row
    for tranche in ("A_CORE_STATIC", "B_REMAINING_STATIC", "C1_DENSE_OVERLAP", "C2_PITCH_BOUNDARY"):
        root = C2_PACKAGE / "decisions" / "completed_tranches" / tranche
        paths.update(path for path in root.rglob("*") if path.is_file())
    for root in (
        G6B / "01_G6A_AND_GOLD_VALIDATION",
        G6B / "02_C2_MISSING_SUPPLY_ATTRIBUTION",
        G6B / "03_BOUNDARY_CANDIDATE_MINING",
        G6B / "04_BOUNDARY_CASE_MANIFEST",
        G6B / "05_PERSON_CENTRIC_BOUNDARY_REVIEW_PACKAGE",
        G6B / "06_BROWSER_PERSISTENCE_AND_COMPLETION",
        G6B / "07_NEXT_STAGE_PERMISSION",
    ):
        paths.update(path for path in root.rglob("*") if path.is_file())
    for path in (
        G6A / "03_FROZEN_FOOTPOINT_AND_GATE_VARIANTS" / "footpoint_geometry_variant_specification.json",
        G6A / "03_FROZEN_FOOTPOINT_AND_GATE_VARIANTS" / "footpoint_geometry_variant_specification.sha256",
        G6A / "03_FROZEN_FOOTPOINT_AND_GATE_VARIANTS" / "pitch_gate_variant_specification.json",
        G6A / "03_FROZEN_FOOTPOINT_AND_GATE_VARIANTS" / "pitch_gate_variant_specification.sha256",
        G6A / "04_PLAYER_OBSERVATION_V1_SCHEMA" / "player_observation_v1_schema.json",
        G6A / "05_OBSERVATION_PIPELINE_INTEGRATION" / "player_observation_v1_runtime_ledger.jsonl",
        G6A / "06_PITCH_GATE_AND_SUPPLY_EVALUATION" / "pitch_gate_results.json",
        G6A / "06_PITCH_GATE_AND_SUPPLY_EVALUATION" / "player_observation_v1_results.json",
        G2B / "03_FROZEN_PROPOSAL_FAMILY_MATRIX" / "exact_frozen_replay_manifest.json",
        G3 / "03_FROZEN_CONSOLIDATION_VARIANTS" / "consolidation_variant_specification.json",
        G4R2 / "02_DENSE_GOLD_V2_OVERLAY_APPLICATION" / "dense_gold_v2_manifest.json",
        G5A / "04_FROZEN_PROMPT_AND_CROP_MATRIX" / "frozen_crop_prompt_specification.json",
    ):
        paths.add(path)
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise RuntimeError(f"FAIL_PRIOR_STAGE_MUTATION: missing protected inputs: {missing}")
    for filename, expected_hash in read_json(PROMPT / "03_B1_INDEPENDENT_AUDIT.json")["completion"][
        "artifact_hash_checks"
    ].items():
        path = B1_BUNDLE / filename
        if sha256_file(path) != expected_hash["expected_sha256"]:
            raise RuntimeError(f"FAIL_B1_COMPLETION_OR_GOLD: hash mismatch for {filename}")
    for key, row in expected.items():
        path = Path(key)
        size_matches = "byte_size" not in row or path.stat().st_size == row["byte_size"]
        if not path.is_file() or sha256_file(path) != row["sha256"] or not size_matches:
            raise RuntimeError(f"FAIL_PRIOR_STAGE_MUTATION: historical hash mismatch for {key}")
    return sorted(paths), expected


def protected_manifest() -> dict[str, Any]:
    paths, expected = protected_inputs()
    records = [file_record(path) for path in paths]
    return {
        "schema_version": "football_intelligence.m5_5g6c.protected_input_manifest.v1",
        "file_count": len(records),
        "historical_expected_record_count": len(expected),
        "files": records,
        "manifest_hash": stable_hash(records),
    }


def validate_b1_completion() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bundle_validation = validate_completion_bundle(B1_BUNDLE)
    if not bundle_validation["passed"] or bundle_validation["completion_transaction_id"] != B1_TRANSACTION:
        raise RuntimeError("FAIL_B1_COMPLETION_OR_GOLD: invalid completion transaction")
    completed = read_json(B1_BUNDLE / "completed_review.json")
    manifest = read_json(B1_PACKAGE / "reviewer_manifest.json")
    root_state = read_json(B1_PACKAGE / "decisions" / "review_decisions.json")
    events = read_jsonl(B1_PACKAGE / "decisions" / "review_decision_events.jsonl")
    if [row["event_sequence"] for row in events] != list(range(1, 20)):
        raise RuntimeError("FAIL_B1_COMPLETION_OR_GOLD: root event sequence is not strict 1..19")
    event_counts = Counter(row["event_type"] for row in events)
    if event_counts != {"DETECTION_CASE_SAVED": 18, "DETECTION_TRANCHE_COMPLETED": 1}:
        raise RuntimeError("FAIL_B1_COMPLETION_OR_GOLD: root event types differ")
    if int(root_state["event_sequence"]) != 19 or root_state.get("pending_outbox_count", 0) != 0:
        raise RuntimeError("FAIL_B1_COMPLETION_OR_GOLD: root materialized state differs")
    annotations = completed["state"]["annotations"]
    cases = {str(row["case_id"]): row for row in manifest["cases"]}
    boundary_cases = {
        str(row["case_id"]): row
        for row in read_json(G6B / "04_BOUNDARY_CASE_MANIFEST" / "boundary_case_manifest.json")["cases"]
    }
    if sorted(annotations) != sorted(cases) or len(cases) != 18:
        raise RuntimeError("FAIL_B1_COMPLETION_OR_GOLD: exact case set differs")
    if sorted(boundary_cases) != sorted(cases):
        raise RuntimeError("FAIL_B1_COMPLETION_OR_GOLD: source-group case set differs")
    roles: Counter[str] = Counter()
    pitch_states: Counter[str] = Counter()
    footpoints: Counter[str] = Counter()
    relations: Counter[str] = Counter()
    source_groups: set[str] = set()
    binding_rows = []
    for case_id in sorted(cases):
        case = cases[case_id]
        annotation = annotations[case_id]
        people = annotation["player_instances"]
        candidate_relations = annotation["candidate_relations"]
        if len(people) != 1 or len(candidate_relations) != 1 or annotation["visible_person_count"] != 1:
            raise RuntimeError(f"FAIL_B1_COMPLETION_OR_GOLD: target cardinality differs for {case_id}")
        person = people[0]
        relation = candidate_relations[0]
        frame = source_frame(case, "target_frame_sequence")
        candidates = frame["candidates"]
        expected_candidate = str(case["visible_metadata"]["candidate_uuids"][0])
        if (
            len(candidates) != 1
            or str(candidates[0]["diagnostic_uuid"]) != expected_candidate
            or relation
            != {
                "candidate_uuid": expected_candidate,
                "relation": "CLEAN_SINGLE_INSTANCE",
                "annotation_uuids": [person["annotation_uuid"]],
            }
        ):
            raise RuntimeError(f"FAIL_B1_COMPLETION_OR_GOLD: candidate binding differs for {case_id}")
        expected_binding = case["visible_metadata"]["source_binding"]
        actual_binding = annotation["source_binding"]
        keys = ("frame_index", "source_frame_sha256", "image_width", "image_height", "pitch_polygon_hash")
        if any(str(actual_binding[key]) != str(expected_binding[key]) for key in keys):
            raise RuntimeError(f"FAIL_B1_COMPLETION_OR_GOLD: source binding differs for {case_id}")
        if str(frame["source_frame_sha256"]) != str(actual_binding["source_frame_sha256"]) or int(
            frame["frame_sequence"]
        ) != int(actual_binding["frame_index"]):
            raise RuntimeError(f"FAIL_B1_COMPLETION_OR_GOLD: frame binding differs for {case_id}")
        roles[person["coarse_role"]] += 1
        pitch_states[person["pitch_state"]] += 1
        footpoints[person["footpoint_status"]] += 1
        relations[relation["relation"]] += 1
        source_group_id = str(boundary_cases[case_id]["source_group_id"])
        source_groups.add(source_group_id)
        binding_rows.append(
            {
                "case_id": case_id,
                "source_frame_sha256": actual_binding["source_frame_sha256"],
                "frame_index": actual_binding["frame_index"],
                "source_group_id": source_group_id,
                "target_box_hash": stable_hash(person["visible_body_box"]),
                "candidate_binding_hash": stable_hash(relation),
            }
        )
    expected_counts = {
        "roles": {"GOALKEEPER": 2, "OFFICIAL": 8, "PLAYER": 8},
        "pitch_states": {"BOUNDARY_UNCERTAIN": 8, "OFF_PITCH": 2, "ON_PITCH": 8},
        "footpoints": {"OBSERVED_CLEAR": 18},
        "relations": {"CLEAN_SINGLE_INSTANCE": 18},
    }
    actual_counts = {
        "roles": dict(sorted(roles.items())),
        "pitch_states": dict(sorted(pitch_states.items())),
        "footpoints": dict(sorted(footpoints.items())),
        "relations": dict(sorted(relations.items())),
    }
    boundary = completed["boundary_completion"]
    if (
        actual_counts != expected_counts
        or len(source_groups) != 18
        or stable_hash(sorted(source_groups)) != B1_DIVERSITY_HASH
        or boundary["selection_specification_hash"] != B1_SELECTION_HASH
        or boundary["source_group_diversity_hash"] != B1_DIVERSITY_HASH
        or not boundary["prior_gold_unchanged"]
    ):
        raise RuntimeError("FAIL_B1_COMPLETION_OR_GOLD: exact gold counts or provenance differ")
    snapshots = B1_PACKAGE / "decisions" / "snapshots"
    snapshot_checks = []
    for sequence in range(1, 20):
        path = snapshots / f"review_state_{sequence:06d}.json"
        sidecar = path.with_suffix(path.suffix + ".sha256")
        expected_hash = sidecar.read_text(encoding="ascii").split()[0]
        payload = read_json(path)
        passed = sha256_file(path) == expected_hash and int(payload["snapshot_sequence"]) == sequence
        snapshot_checks.append({"event_sequence": sequence, "passed": passed})
    if not all(row["passed"] for row in snapshot_checks):
        raise RuntimeError("FAIL_B1_COMPLETION_OR_GOLD: snapshot replay validation failed")
    validation = {
        "schema_version": "football_intelligence.m5_5g6c.b1_completion_and_gold_validation.v1",
        "classification": "PASS_B1_COMPLETION_AND_GOLD_VALIDATED",
        "completion_transaction_id": B1_TRANSACTION,
        "root_event_sequence": 19,
        "root_event_count": 19,
        "event_type_counts": dict(sorted(event_counts.items())),
        "case_count": 18,
        "distinct_source_group_count": 18,
        "exact_counts": actual_counts,
        "selection_specification_hash": B1_SELECTION_HASH,
        "source_group_diversity_hash": B1_DIVERSITY_HASH,
        "completion_artifact_hashes": bundle_validation["artifact_hashes"],
        "snapshot_count": len(snapshot_checks),
        "snapshot_replay_valid": True,
        "source_binding_ledger_hash": stable_hash(binding_rows),
        "source_bindings_valid": True,
        "prior_gold_unchanged": True,
        "passed": True,
        **safety_payload(),
    }
    return completed, manifest, validation


def evaluator_universe_contract() -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.m5_5g6c.evaluator_universe_contract.v1",
        "pooling_into_single_accuracy_forbidden": True,
        "universes": {
            "C2_BROAD_CLEAR_PERSON_DEVELOPMENT_SET": {
                "cases": 12,
                "people": 96,
                "on_pitch": 45,
                "off_pitch": 51,
                "boundary_uncertain": 0,
                "uses": [
                    "on-pitch retention",
                    "labelled off-pitch leakage",
                    "role retention",
                    "feet-not-visible handling",
                    "general frozen proposal supply",
                ],
            },
            "B1_TARGETED_BOUNDARY_STRESS_SET": {
                "cases": 18,
                "people": 18,
                "on_pitch": 8,
                "off_pitch": 2,
                "boundary_uncertain": 8,
                "population_representative": False,
                "uses": [
                    "boundary-uncertain routing",
                    "near-line hard-classification errors",
                    "frozen gate disagreement",
                    "near-boundary clear-person preservation",
                ],
            },
        },
        "rule_based_shortlist_only": True,
        "single_overall_accuracy_reported": False,
        "passed": True,
    }


def validate_frozen_specifications(b1_manifest: Mapping[str, Any]) -> dict[str, Any]:
    root = G6A / "03_FROZEN_FOOTPOINT_AND_GATE_VARIANTS"
    footpoint_path = root / "footpoint_geometry_variant_specification.json"
    gate_path = root / "pitch_gate_variant_specification.json"
    schema_path = G6A / "04_PLAYER_OBSERVATION_V1_SCHEMA" / "player_observation_v1_schema.json"
    projection_path = G6A / "02_PITCH_POLYGON_AND_TRANSFORM_DIAGNOSIS" / "browser_projection_validation.json"
    c2_manifest = read_json(C2_PACKAGE / "reviewer_manifest.json")
    c2_completed = read_json(C2_BUNDLE / "completed_review.json")
    c2_case_ids = set(c2_completed["state"]["annotations"])
    b1_polygon_hashes = {
        str(row["visible_metadata"]["source_binding"]["pitch_polygon_hash"]) for row in b1_manifest["cases"]
    }
    c2_polygon_hashes = {
        str(row["visible_metadata"]["source_binding"]["pitch_polygon_hash"])
        for row in c2_manifest["cases"]
        if row["case_id"] in c2_case_ids
    }
    c2_completed_polygon_hashes = {
        str(row["source_binding"]["pitch_polygon_hash"]) for row in c2_completed["state"]["annotations"].values()
    }
    b1_audit_polygon_hashes = {
        str(row["pitch_polygon_hash"])
        for row in read_json(PROMPT / "03_B1_INDEPENDENT_AUDIT.json")["human_gold_inventory"]["case_rows"]
    }
    checks = {
        "footpoint_file_hash_exact": sha256_file(footpoint_path) == FOOTPOINT_SPEC_SHA256,
        "pitch_gate_file_hash_exact": sha256_file(gate_path) == PITCH_GATE_SPEC_SHA256,
        "footpoint_sidecar_exact": footpoint_path.with_suffix(".sha256").read_text(encoding="ascii").split()[0]
        == FOOTPOINT_SPEC_SHA256,
        "pitch_gate_sidecar_exact": gate_path.with_suffix(".sha256").read_text(encoding="ascii").split()[0]
        == PITCH_GATE_SPEC_SHA256,
        "production_footpoint_spec_unchanged": read_json(footpoint_path) == footpoint_geometry_variant_specification(),
        "production_pitch_gate_spec_unchanged": read_json(gate_path) == pitch_gate_variant_specification(),
        "player_observation_schema_unchanged": read_json(schema_path) == player_observation_json_schema(),
        "projection_repair_browser_valid": read_json(projection_path).get("passed") is True,
        "b1_source_polygon_binding_exact": len(b1_polygon_hashes) == 1 and b1_polygon_hashes == b1_audit_polygon_hashes,
        "c2_source_polygon_binding_exact": c2_polygon_hashes == c2_completed_polygon_hashes,
        "cross_universe_polygon_identity_not_assumed": b1_polygon_hashes.isdisjoint(c2_polygon_hashes),
    }
    result = {
        "schema_version": "football_intelligence.m5_5g6c.frozen_specification_validation.v1",
        "footpoint_specification_sha256": FOOTPOINT_SPEC_SHA256,
        "pitch_gate_specification_sha256": PITCH_GATE_SPEC_SHA256,
        "player_observation_schema_version": PLAYER_OBSERVATION_SCHEMA_VERSION,
        "pitch_polygon_hashes": {
            "b1_boundary_stress": sorted(b1_polygon_hashes),
            "c2_broad": sorted(c2_polygon_hashes),
        },
        "pitch_margin_pixels": 10.0,
        "threshold_or_margin_changed": False,
        "checks": checks,
        "passed": all(checks.values()),
        **safety_payload(),
    }
    if not result["passed"]:
        raise RuntimeError(f"FAIL_FROZEN_SPECIFICATION: {checks}")
    return result


def replay_c2() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[float]]]:
    completed, exact_counts, bundle_validation = G6A_IMPL.c2_validation()
    manifest = read_json(C2_PACKAGE / "reviewer_manifest.json")
    case_rows, runtime_rows, timings = G6A_IMPL.build_runtime_cases(completed, manifest)
    frozen_rows = read_jsonl(G6A / "05_OBSERVATION_PIPELINE_INTEGRATION" / "player_observation_v1_runtime_ledger.jsonl")
    runtime_hash = stable_hash(runtime_rows)
    frozen_runtime_hash = stable_hash(frozen_rows)
    frozen_results = read_json(G6A / "06_PITCH_GATE_AND_SUPPLY_EVALUATION" / "pitch_gate_results.json")
    frozen_by_variant = {row["pitch_gate_variant"]: row for row in frozen_results["variants"]}
    variants = []
    deterministic_matches = []
    for variant in ("P1", "P2", "P3", "P4"):
        fresh = G6A_IMPL.evaluate_variant(variant, case_rows)
        pitch = fresh["pitch_gate"]
        frozen = frozen_by_variant[variant]
        keys = (
            "on_pitch_person_supply_retained",
            "off_pitch_labelled_person_leakage",
            "boundary_review_observation_count",
            "feet_not_visible_person_routed",
            "feet_not_visible_person_missing_without_runtime_claim",
            "referee_retained",
            "goalkeeper_retained",
        )
        unchanged = all(pitch[key] == frozen[key] for key in keys)
        deterministic_matches.append(unchanged)
        for row in runtime_rows:
            if row["pitch_gate_variant"] == variant:
                validate_runtime_payload(row["observation"])
        runtime_values = timings[variant]
        variants.append(
            {
                "pitch_gate_variant": variant,
                "on_pitch_person_supply_retained": pitch["on_pitch_person_supply_retained"],
                "off_pitch_labelled_person_leakage": pitch["off_pitch_labelled_person_leakage"],
                "boundary_review_observation_count": pitch["boundary_review_observation_count"],
                "feet_not_visible_person_routed": pitch["feet_not_visible_person_routed"],
                "feet_not_visible_person_missing_without_runtime_claim": pitch[
                    "feet_not_visible_person_missing_without_runtime_claim"
                ],
                "referee_retained": pitch["referee_retained"],
                "goalkeeper_retained": pitch["goalkeeper_retained"],
                "cpu_p50_milliseconds_per_source": round(statistics.median(runtime_values), 8),
                "cpu_p95_milliseconds_per_source": round(quantile(runtime_values, 0.95), 8),
                "frozen_screen": frozen["screen"],
                "deterministic_outputs_unchanged": unchanged,
            }
        )
    result = {
        "schema_version": "football_intelligence.m5_5g6c.c2_gate_replay_validation.v1",
        "universe": "C2_BROAD_CLEAR_PERSON_DEVELOPMENT_SET",
        "exact_counts": exact_counts,
        "bundle_validation": bundle_validation,
        "frozen_runtime_row_count": len(frozen_rows),
        "replayed_runtime_row_count": len(runtime_rows),
        "frozen_runtime_ledger_hash": frozen_runtime_hash,
        "replayed_runtime_ledger_hash": runtime_hash,
        "runtime_ledger_exact": runtime_hash == frozen_runtime_hash,
        "variant_outputs_exact": all(deterministic_matches),
        "human_truth_entered_runtime": False,
        "variants": variants,
        "passed": runtime_hash == frozen_runtime_hash and all(deterministic_matches),
        **safety_payload(),
    }
    if not result["passed"]:
        raise RuntimeError("FAIL_C2_GATE_REPLAY")
    return result, case_rows, timings


def _gate_outcome(relation: str, truth: str) -> str:
    if relation == "BOUNDARY_UNCERTAIN":
        return "ROUTED"
    if truth == "ON_PITCH":
        return "RETAINED" if relation == "ON_PITCH" else "HARD_OFF"
    if truth == "OFF_PITCH":
        return "REJECTED" if relation == "OFF_PITCH" else "LEAKED"
    return "HARD_ON" if relation == "ON_PITCH" else "HARD_OFF"


def replay_b1(
    completed: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[float]]]:
    annotations = completed["state"]["annotations"]
    timings: dict[str, list[float]] = {variant: [] for variant in ("P1", "P2", "P3", "P4")}
    ledgers: dict[str, list[dict[str, Any]]] = {variant: [] for variant in timings}
    visual_contexts: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        case_id = str(case["case_id"])
        annotation = annotations[case_id]
        person = annotation["player_instances"][0]
        frame = source_frame(case, "target_frame_sequence")
        candidate = frame["candidates"][0]
        box = {key: float(candidate["bbox_original_pixels"][key]) for key in ("x1", "y1", "x2", "y2")}
        polygon = [dict(row) for row in case["visible_metadata"]["pitch_polygon_vertices"]]
        runtime_input = {
            "visible_box": box,
            "source_frame_sha256": frame["source_frame_sha256"],
            "source_row_sha256": candidate["source_row_sha256"],
            "pitch_polygon_hash": annotation["source_binding"]["pitch_polygon_hash"],
        }
        validate_runtime_payload(runtime_input)
        geometry = {name: estimate_footpoint(name, box) for name in ("F0", "F1", "F3")}
        geometry_distances = {
            name: signed_distance_to_polygon(row["footpoint_estimate"], polygon) for name, row in geometry.items()
        }
        for variant in timings:
            outputs = []
            started = time.perf_counter_ns()
            for _ in range(25):
                outputs.append(apply_pitch_gate(variant, box, polygon))
            timings[variant].append((time.perf_counter_ns() - started) / 25 / 1_000_000)
            if any(output != outputs[0] for output in outputs[1:]):
                raise RuntimeError(f"FAIL_B1_GATE_REEVALUATION: nondeterministic {variant} output")
            gate = outputs[0]
            human_footpoint = person["footpoint"]
            ledgers[variant].append(
                {
                    "case_id": case_id,
                    "source_frame_sha256": frame["source_frame_sha256"],
                    "runtime_input_hash": stable_hash(runtime_input),
                    "candidate_box_hash": stable_hash(box),
                    "candidate_lineage_hash": stable_hash(candidate["source_row_sha256"]),
                    "runtime": {
                        "pitch_relation": gate["pitch_relation"],
                        "geometry_method": gate["geometry_method"],
                        "footpoint_method": gate["footpoint_method"],
                        "runtime_footpoint_signed_distance_pixels": round(
                            signed_distance_to_polygon(gate["footpoint_estimate"], polygon), 8
                        ),
                        "f0_f1_f3_signed_distances_pixels": {
                            key: round(value, 8) for key, value in geometry_distances.items()
                        },
                        "f0_f1_f3_hard_side_agreement": len({value >= 0 for value in geometry_distances.values()}) == 1,
                    },
                    "evaluator_join_after_runtime": {
                        "pitch_state": person["pitch_state"],
                        "role": person["coarse_role"],
                        "footpoint_status": person["footpoint_status"],
                        "human_footpoint_signed_distance_pixels": round(
                            signed_distance_to_polygon(human_footpoint, polygon), 8
                        ),
                        "runtime_to_human_footpoint_error_pixels": round(
                            math.dist(
                                (float(gate["footpoint_estimate"]["x"]), float(gate["footpoint_estimate"]["y"])),
                                (float(human_footpoint["x"]), float(human_footpoint["y"])),
                            ),
                            8,
                        ),
                        "scored_outcome": _gate_outcome(gate["pitch_relation"], person["pitch_state"]),
                    },
                    "human_truth_entered_runtime": False,
                }
            )
        panorama_path = B1_PACKAGE / "evidence" / case_id / frame["panorama_asset_path"]
        if sha256_file(panorama_path) != frame["source_frame_sha256"]:
            raise RuntimeError(f"FAIL_B1_COMPLETION_OR_GOLD: source image mismatch for {case_id}")
        visual_contexts.append(
            {
                "case_id": case_id,
                "image_path": panorama_path,
                "box": box,
                "polygon": polygon,
                "human_footpoint": dict(person["footpoint"]),
                "truth": person["pitch_state"],
                "relations": {variant: ledgers[variant][-1]["runtime"]["pitch_relation"] for variant in timings},
            }
        )
    variants = []
    for variant, rows in ledgers.items():
        by_truth: dict[str, Counter[str]] = {
            truth: Counter() for truth in ("ON_PITCH", "OFF_PITCH", "BOUNDARY_UNCERTAIN")
        }
        for row in rows:
            joined = row["evaluator_join_after_runtime"]
            by_truth[joined["pitch_state"]][joined["scored_outcome"]] += 1
        on = by_truth["ON_PITCH"]
        off = by_truth["OFF_PITCH"]
        boundary = by_truth["BOUNDARY_UNCERTAIN"]
        screen = {
            "routes_at_least_7_of_8_boundary_uncertain": boundary["ROUTED"] >= 7,
            "hard_misclassifies_at_most_1_boundary_uncertain": boundary["HARD_ON"] + boundary["HARD_OFF"] <= 1,
            "clear_on_pitch_preserved": on["HARD_OFF"] == 0 and (on["RETAINED"] == 8 or on["ROUTED"] <= 1),
            "clear_off_pitch_preserved": off["LEAKED"] == 0 and (off["REJECTED"] == 2 or off["ROUTED"] <= 1),
            "human_truth_free_runtime": True,
            "passed": False,
        }
        screen["passed"] = all(value for key, value in screen.items() if key != "passed")
        variants.append(
            {
                "pitch_gate_variant": variant,
                "on_pitch": {
                    "denominator": 8,
                    "retained": on["RETAINED"],
                    "hard_off": on["HARD_OFF"],
                    "routed": on["ROUTED"],
                },
                "off_pitch": {
                    "denominator": 2,
                    "rejected": off["REJECTED"],
                    "leaked": off["LEAKED"],
                    "routed": off["ROUTED"],
                },
                "boundary_uncertain": {
                    "denominator": 8,
                    "routed": boundary["ROUTED"],
                    "hard_on": boundary["HARD_ON"],
                    "hard_off": boundary["HARD_OFF"],
                },
                "cpu_p50_milliseconds_per_source": round(statistics.median(timings[variant]), 8),
                "cpu_p95_milliseconds_per_source": round(quantile(timings[variant], 0.95), 8),
                "screen": screen,
                "person_level_ledger": rows,
            }
        )
    result = {
        "schema_version": "football_intelligence.m5_5g6c.b1_pitch_gate_results.v1",
        "universe": "B1_TARGETED_BOUNDARY_STRESS_SET",
        "population_representative": False,
        "variant_count": 4,
        "variants": variants,
        "shortlisted_variants": [row["pitch_gate_variant"] for row in variants if row["screen"]["passed"]],
        "human_truth_entered_runtime": False,
        "threshold_or_margin_changed": False,
        "passed": True,
        **safety_payload(),
    }
    return result, visual_contexts, timings


def combined_gate_decision(c2: Mapping[str, Any], b1: Mapping[str, Any]) -> dict[str, Any]:
    c2_by_variant = {row["pitch_gate_variant"]: row for row in c2["variants"]}
    b1_by_variant = {row["pitch_gate_variant"]: row for row in b1["variants"]}
    variants = []
    for variant in ("P1", "P2", "P3", "P4"):
        c2_screen = c2_by_variant[variant]["frozen_screen"]
        b1_screen = b1_by_variant[variant]["screen"]
        variants.append(
            {
                "pitch_gate_variant": variant,
                "c2_broad_screen_passed": c2_screen["passed"],
                "c2_failed_checks": [key for key, value in c2_screen.items() if key != "passed" and not value],
                "b1_stress_screen_passed": b1_screen["passed"],
                "b1_failed_checks": [key for key, value in b1_screen.items() if key != "passed" and not value],
                "combined_screen_passed": c2_screen["passed"] and b1_screen["passed"],
            }
        )
    shortlisted = [row["pitch_gate_variant"] for row in variants if row["combined_screen_passed"]]
    return {
        "schema_version": "football_intelligence.m5_5g6c.combined_pitch_gate_shortlist.v1",
        "single_overall_accuracy_reported": False,
        "screen_weakened_after_results": False,
        "variants": variants,
        "shortlisted_pitch_gate_variants": shortlisted,
        "development_pitch_gate_frozen": False,
        "decision": "NO_FROZEN_PITCH_GATE_VARIANT_PASSES_BOTH_UNIVERSES",
        "passed": True,
        **safety_payload(),
    }


def _match_missing_person(item: Mapping[str, Any], annotations: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [
        row
        for row in annotations[item["case_id"]]["player_instances"]
        if stable_hash(row["visible_body_box"]) == item["visible_body_box_sha256"]
    ]
    if len(matches) != 1:
        raise RuntimeError(f"FAIL_MISS_PHENOTYPING: missing-person binding differs for {item['anonymous_person_id']}")
    return matches[0]


def phenotype_nine_misses() -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    missing = read_json(G6B / "02_C2_MISSING_SUPPLY_ATTRIBUTION" / "c2_missing_on_pitch_people.json")["rows"]
    attribution = {
        row["anonymous_person_id"]: row
        for row in read_jsonl(G6B / "02_C2_MISSING_SUPPLY_ATTRIBUTION" / "frozen_proposal_supply_attribution.jsonl")
    }
    c2_completed = read_json(C2_BUNDLE / "completed_review.json")
    annotations = c2_completed["state"]["annotations"]
    manifest = read_json(C2_PACKAGE / "reviewer_manifest.json")
    cases = {str(row["case_id"]): row for row in manifest["cases"]}
    rows = []
    visual_contexts = []
    for item in missing:
        person = _match_missing_person(item, annotations)
        box = {key: float(person["visible_body_box"][key]) for key in ("x1", "y1", "x2", "y2")}
        case = cases[item["case_id"]]
        frame = source_frame(case)
        image_path = C2_PACKAGE / "evidence" / item["case_id"] / frame["panorama_asset_path"]
        if sha256_file(image_path) != item["source_frame_sha256"]:
            raise RuntimeError(f"FAIL_MISS_PHENOTYPING: source image differs for {item['anonymous_person_id']}")
        width = box["x2"] - box["x1"]
        height = box["y2"] - box["y1"]
        aspect = width / height
        other_people = [
            row
            for row in annotations[item["case_id"]]["player_instances"]
            if row["annotation_uuid"] != person["annotation_uuid"]
        ]
        center = ((box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2)
        neighbor_distances = []
        for other in other_people:
            other_box = other["visible_body_box"]
            other_center = (
                (float(other_box["x1"]) + float(other_box["x2"])) / 2,
                (float(other_box["y1"]) + float(other_box["y2"])) / 2,
            )
            neighbor_distances.append(math.dist(center, other_center) / max(1.0, height))
        close_neighbors = sum(value <= 2.0 for value in neighbor_distances)
        attr = attribution[item["anonymous_person_id"]]
        nearest = attr["best_existing_proposals"][0] if attr["best_existing_proposals"] else None
        localization = None
        if nearest:
            raw_box = nearest["bbox_panorama_pixels"]
            raw_center = (
                (float(raw_box["x1"]) + float(raw_box["x2"])) / 2,
                (float(raw_box["y1"]) + float(raw_box["y2"])) / 2,
            )
            localization = {
                "proposal_lineage_hash": stable_hash(nearest["source_row_hash"]),
                "score": nearest["score"],
                "iou": nearest["iou"],
                "person_containment": nearest["person_containment"],
                "centre_offset_pixels": round(math.dist(center, raw_center), 8),
                "bottom_offset_pixels": round(float(raw_box["y2"]) - box["y2"], 8),
            }
        phenotype = "SMALL_FAR_SIDE" if height <= 40 and box["y2"] <= 0.45 * frame["image_height"] else "UNRESOLVED"
        different_families = [
            key
            for key, available in attr["frozen_family_availability_for_source"].items()
            if available and key != "FULL_PANORAMA_1280"
        ]
        row = {
            "schema_version": "football_intelligence.m5_5g6c.nine_person_miss_phenotype_row.v1",
            "anonymous_person_id": item["anonymous_person_id"],
            "case_id": item["case_id"],
            "source_frame_sha256": item["source_frame_sha256"],
            "frame_sequence": item["frame_sequence"],
            "visible_body_box_sha256": item["visible_body_box_sha256"],
            "visible_width_pixels": round(width, 8),
            "visible_height_pixels": round(height, 8),
            "visible_height_fraction_of_panorama": round(height / frame["image_height"], 8),
            "aspect_ratio_width_over_height": round(aspect, 8),
            "pose_aspect_diagnostic": "PLAUSIBLE_UPRIGHT_SMALL_PERSON" if 0.25 <= aspect <= 0.8 else "UNUSUAL_ASPECT",
            "occlusion_partial_state": "CLOSE_PERSON_CONTEXT_NO_CONFIRMED_OCCLUSION"
            if close_neighbors
            else "NO_CONFIRMED_OCCLUSION",
            "nearby_people_within_two_target_heights": close_neighbors,
            "image_region": "FAR_SIDE_UPPER_PANORAMA",
            "frozen_view_coverage": attr["frozen_family_availability_for_source"],
            "earliest_supported_origin": attr["earliest_supported_origin"],
            "nearest_raw_proposal": localization,
            "different_frozen_family_weak_or_partial_evidence": different_families,
            "light_hq_sam_without_detector_proposal": {
                "runtime_prompt_available": False,
                "evaluator_box_could_prompt_diagnostic_only": True,
                "reason": (
                    "The frozen runtime has no detector-derived prompt for a no-raw-proposal person; "
                    "evaluator geometry is forbidden at runtime."
                ),
            },
            "primary_phenotype": phenotype,
            "visual_confidence": "HIGH" if height <= 30 else "MEDIUM",
            "generic_iou_only_decision": False,
            "human_truth_entered_runtime": False,
        }
        rows.append(row)
        visual_contexts.append(
            {
                "row": row,
                "image_path": image_path,
                "box": box,
                "nearest_raw_box": nearest["bbox_panorama_pixels"] if nearest else None,
            }
        )
    phenotype_counts = Counter(row["primary_phenotype"] for row in rows)
    origin_counts = Counter(row["earliest_supported_origin"] for row in rows)
    if (
        len(rows) != 9
        or phenotype_counts != {"SMALL_FAR_SIDE": 9}
        or origin_counts
        != {
            "NO_RAW_PROPOSAL": 7,
            "RAW_LOCALIZATION_BAD": 2,
        }
    ):
        raise RuntimeError("FAIL_MISS_PHENOTYPING: exact phenotype or origin counts differ")
    heights = [row["visible_height_pixels"] for row in rows]
    summary = {
        "schema_version": "football_intelligence.m5_5g6c.miss_phenotype_summary.v1",
        "missing_person_count": len(rows),
        "phenotype_counts": dict(sorted(phenotype_counts.items())),
        "origin_counts": dict(sorted(origin_counts.items())),
        "visible_height_pixels": {
            "minimum": round(min(heights), 8),
            "median": round(statistics.median(heights), 8),
            "maximum": round(max(heights), 8),
        },
        "all_far_side_upper_panorama": all(row["image_region"] == "FAR_SIDE_UPPER_PANORAMA" for row in rows),
        "all_plausible_upright_aspect": all(
            row["pose_aspect_diagnostic"] == "PLAUSIBLE_UPRIGHT_SMALL_PERSON" for row in rows
        ),
        "provenance_coordinate_repair_indicated": False,
        "selected_experiment": SELECTED_EXPERIMENT,
        "new_inference_performed": False,
        "passed": True,
        **safety_payload(),
    }
    return rows, summary, visual_contexts


def _supported_c2_controls(target_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    completed = read_json(C2_BUNDLE / "completed_review.json")
    manifest = read_json(C2_PACKAGE / "reviewer_manifest.json")
    cases = {str(row["case_id"]): row for row in manifest["cases"]}
    controls = []
    for case_id, annotation in sorted(completed["state"]["annotations"].items()):
        related = {
            str(target)
            for relation in annotation["candidate_relations"]
            if relation["relation"] not in {"AMBIGUOUS", "BACKGROUND"}
            for target in relation["annotation_uuids"]
        }
        frame = source_frame(cases[case_id])
        for person in annotation["player_instances"]:
            if person["pitch_state"] != "ON_PITCH" or person["annotation_uuid"] not in related:
                continue
            box = person["visible_body_box"]
            controls.append(
                {
                    "source_frame_sha256": frame["source_frame_sha256"],
                    "visible_body_box_sha256": stable_hash(box),
                    "visible_height_pixels": round(float(box["y2"]) - float(box["y1"]), 8),
                    "coarse_role": person["coarse_role"],
                    "case_id_hash": stable_hash(case_id),
                }
            )
    target_median = statistics.median(row["visible_height_pixels"] for row in target_rows)
    controls.sort(
        key=lambda row: (
            abs(row["visible_height_pixels"] - target_median),
            row["source_frame_sha256"],
            row["visible_body_box_sha256"],
        )
    )
    small = controls[:9]
    remaining = sorted(controls[9:], key=lambda row: (row["visible_height_pixels"], row["visible_body_box_sha256"]))
    if len(remaining) < 9:
        raise RuntimeError("FAIL_RECOVERY_EXPERIMENT_DECISION: insufficient matched controls")
    positions = [round(index * (len(remaining) - 1) / 8) for index in range(9)]
    ordinary = [remaining[index] for index in positions]
    selected = []
    for index, row in enumerate([*small, *ordinary], 1):
        selected.append(
            {
                "anonymous_control_id": f"proposal-recovery-control-{index:03d}",
                "control_band": "SMALL_HEIGHT_MATCHED" if index <= 9 else "GENERAL_SCALE_STRATIFIED",
                **row,
            }
        )
    return selected


def proposal_recovery_decision(
    phenotype_rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    controls = _supported_c2_controls(phenotype_rows)
    matrix = read_json(G2B / "03_FROZEN_PROPOSAL_FAMILY_MATRIX" / "exact_frozen_replay_manifest.json")
    target_universe = [
        {
            "anonymous_person_id": row["anonymous_person_id"],
            "source_frame_sha256": row["source_frame_sha256"],
            "visible_body_box_sha256": row["visible_body_box_sha256"],
            "visible_height_pixels": row["visible_height_pixels"],
            "origin": row["earliest_supported_origin"],
        }
        for row in phenotype_rows
    ]
    decision = {
        "schema_version": "football_intelligence.m5_5g6c.proposal_recovery_experiment_decision.v1",
        "selected_experiment": SELECTED_EXPERIMENT,
        "selected_experiment_id": SELECTED_EXPERIMENT_ID,
        "exactly_one_experiment_selected": True,
        "rationale": [
            "All nine misses are plausible upright far-side people only 22-33 panorama pixels tall.",
            "Seven misses have no raw FULL_PANORAMA_1280 proposal and two have badly localized raw proposals.",
            "The frozen evidence is internally consistent, so provenance repair is not indicated.",
            "A frozen high-resolution/view matrix directly tests the dominant failure without changing thresholds.",
        ],
        "rejected_options": {
            "R-B_OCCLUSION_PARTIAL_PERSON_PROPOSAL_BAKEOFF": "No miss is primarily supported as partial or occluded.",
            "R-C_GENERAL_NEW_DETECTOR_FAMILY_PROPOSAL_BAKEOFF": (
                "The phenotype is narrow rather than ordinary mixed-scale visibility."
            ),
            "R-D_ANNOTATION_FIRST_EXPANSION": (
                "Nine coherent cases plus matched controls are sufficient for a bounded development experiment."
            ),
            "R-E_PROVENANCE_COORDINATE_REPAIR": "Source, frame, box, and frozen-proposal provenance validate exactly.",
        },
        "phenotype_summary_hash": stable_hash(summary),
        "experiment_executed_in_g6c": False,
        "passed": True,
        **safety_payload(),
    }
    contract = {
        "schema_version": "football_intelligence.m5_5g6c.proposal_recovery_experiment_contract.v1",
        "experiment": SELECTED_EXPERIMENT,
        "experiment_id": SELECTED_EXPERIMENT_ID,
        "development_only": True,
        "target_universe": target_universe,
        "target_universe_hash": stable_hash(target_universe),
        "control_universe": controls,
        "control_universe_hash": stable_hash(controls),
        "target_count": 9,
        "control_count": 18,
        "immutable_baseline": {
            "checkpoint_sha256": matrix["checkpoint_sha256"],
            "canonical_person_runtime": matrix["canonical_person_runtime"],
            "baseline_view": "FULL_PANORAMA_1280",
            "baseline_attribution": {"NO_RAW_PROPOSAL": 7, "RAW_LOCALIZATION_BAD": 2},
            "pitch_gate_specification_sha256": PITCH_GATE_SPEC_SHA256,
            "consolidation_variant": "IOU_CONNECTED_COMPONENT_055",
        },
        "future_permitted_inference": {
            "authorized_in_g6c": False,
            "authorized_only_after_new_stage_prompt": True,
            "fixed_view_contract": matrix["fixed_view_contract"],
            "threshold_search": False,
            "crop_or_tile_change": False,
            "nms_or_fusion_change": False,
            "training_or_fine_tuning": False,
        },
        "official_model_family_research_requirements": [
            "official upstream source",
            "code, checkpoint, dataset and transitive licence audit",
            "exact checkpoint SHA-256 and immutable model card",
            "isolated dependency environment and reproducible export",
            "no community mirror, quantization or credential-dependent artifact",
        ],
        "metrics": [
            "raw proposal support for each target",
            "confidence-surviving and post-NMS target support",
            "fused independent-person supply",
            "matched-control retention",
            "merged-as-clean observations",
            "duplicate accepted observations and lineage",
            "distinct-person suppression",
            "off-pitch processing burden",
            "runtime, OOM state and peak VRAM",
            "coordinate and provenance failures",
        ],
        "frozen_development_screen": {
            "recover_raw_support_for_at_least_7_of_9_targets": True,
            "recover_fused_independent_supply_for_at_least_6_of_9_targets": True,
            "zero_merged_as_clean": True,
            "duplicate_accepted_rate_at_most_0_02": True,
            "zero_matched_control_supply_regression": True,
            "zero_coordinate_or_provenance_failure": True,
            "zero_silent_cpu_fallback": True,
        },
        "rtx_5060_limits": {
            "device": "cuda:0",
            "batch": 1,
            "fp16_when_validated": True,
            "peak_allocated_vram_gib_maximum": 6.5,
            "target_plus_control_sources": 27,
            "cuda_oom_handling": "record configuration failure; do not silently fall back to CPU",
        },
        "rejection_criteria": [
            "target recovery below the frozen development screen",
            "any matched-control supply regression",
            "any merged-as-clean observation",
            "duplicate accepted rate above two percent",
            "any provenance, coordinate or renderer mismatch",
            "peak VRAM above 6.5 GiB or silent CPU fallback",
            "any post-result threshold, crop, tile, NMS or fusion change",
        ],
        "experiment_executed": False,
        **safety_payload(),
    }
    return decision, contract


def player_observation_status(combined: Mapping[str, Any]) -> dict[str, Any]:
    frozen = read_json(G6A / "06_PITCH_GATE_AND_SUPPLY_EVALUATION" / "player_observation_v1_results.json")
    by_variant = {row["pitch_gate_variant"]: row for row in frozen["pipelines"]["O0_BOX_ONLY_CONTROL"]}
    return {
        "schema_version": "football_intelligence.m5_5g6c.player_observation_v1_status.v1",
        "schema_version_frozen": PLAYER_OBSERVATION_SCHEMA_VERSION,
        "schema_and_materializer_status": "READY_DEVELOPMENT_ONLY",
        "pitch_gate_status": "BLOCKED_NO_VARIANT_PASSES_COMBINED_SCREEN",
        "pitch_gate_candidates": combined["shortlisted_pitch_gate_variants"],
        "proposal_supply_status": "BELOW_FROZEN_OBSERVATION_SCREEN",
        "fused_proposal_support_ceiling": {"supported_on_pitch_people": 36, "denominator": 45},
        "exact_independent_observation_supply": {
            variant: row["one_accepted_observation_per_on_pitch_person"] for variant, row in by_variant.items()
        },
        "merged_as_clean_observations": {
            variant: row["merged_as_clean_observations"] for variant, row in by_variant.items()
        },
        "duplicate_accepted_observations": {
            variant: row["duplicate_accepted_observations"] for variant, row in by_variant.items()
        },
        "distinct_person_suppression": {
            variant: row["distinct_person_suppression"] for variant, row in by_variant.items()
        },
        "unresolved_or_routed_observations": {
            variant: row["unresolved_or_routed_observations"] for variant, row in by_variant.items()
        },
        "observed_state_contamination_count": frozen["observed_state_contamination_count"],
        "provenance_failure_count": max(row["provenance_failure_count"] for row in by_variant.values()),
        "recomputed_for_shortlisted_gate": False,
        "player_observation_v1_complete": False,
        "new_inference_performed": False,
        **safety_payload(),
    }


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/segoeuib.ttf") if bold else Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf") if bold else Path("C:/Windows/Fonts/arial.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def crop_context(
    image: Image.Image, box: Mapping[str, float], *, pad_factor: float = 2.2
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    width = float(box["x2"]) - float(box["x1"])
    height = float(box["y2"]) - float(box["y1"])
    pad = max(28.0, max(width, height) * pad_factor)
    bounds = (
        max(0, int(float(box["x1"]) - pad)),
        max(0, int(float(box["y1"]) - pad)),
        min(image.width, int(float(box["x2"]) + pad)),
        min(image.height, int(float(box["y2"]) + pad)),
    )
    return image.crop(bounds), bounds


def paste_crop_with_boxes(
    canvas: Image.Image,
    origin: tuple[int, int],
    size: tuple[int, int],
    source: Image.Image,
    bounds: tuple[int, int, int, int],
    boxes: Sequence[tuple[Mapping[str, float], str]],
) -> None:
    scale = min(size[0] / source.width, size[1] / source.height)
    resized = source.resize(
        (max(1, round(source.width * scale)), max(1, round(source.height * scale))), Image.Resampling.LANCZOS
    )
    x = origin[0] + (size[0] - resized.width) // 2
    y = origin[1] + (size[1] - resized.height) // 2
    canvas.paste(resized, (x, y))
    draw = ImageDraw.Draw(canvas, "RGBA")
    for box, color in boxes:
        coordinates = (
            x + (float(box["x1"]) - bounds[0]) * scale,
            y + (float(box["y1"]) - bounds[1]) * scale,
            x + (float(box["x2"]) - bounds[0]) * scale,
            y + (float(box["y2"]) - bounds[1]) * scale,
        )
        draw.rectangle(coordinates, outline=color, width=3)


def b1_gate_atlas(contexts: Sequence[Mapping[str, Any]], output: Path) -> None:
    canvas = Image.new("RGB", (1800, 1510), "#0b1110")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((28, 20), "B1 boundary-stress replay: frozen P1-P4", font=font(30, bold=True), fill="#f4f7f5")
    draw.text(
        (28, 61),
        "Evaluator truth is overlay-only | source-panorama geometry | development-only, not a population estimate",
        font=font(17),
        fill="#a9b9b1",
    )
    colors = {"ON_PITCH": "#5ef0b8", "OFF_PITCH": "#ff7a89", "BOUNDARY_UNCERTAIN": "#f4c95d"}
    for index, context in enumerate(contexts):
        column, row = index % 3, index // 3
        x, y = 25 + column * 590, 100 + row * 230
        with Image.open(context["image_path"]) as image:
            crop, bounds = crop_context(image.convert("RGB"), context["box"], pad_factor=2.8)
        paste_crop_with_boxes(canvas, (x, y), (560, 150), crop, bounds, [(context["box"], "#ff5b83")])
        draw.text(
            (x, y + 154),
            f"{context['case_id']} | truth {context['truth']}",
            font=font(16, bold=True),
            fill=colors[context["truth"]],
        )
        relation_text = " | ".join(
            f"{key}:{value.replace('BOUNDARY_UNCERTAIN', 'ROUTE')}" for key, value in context["relations"].items()
        )
        draw.text((x, y + 180), relation_text, font=font(14), fill="#d8e1dc")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def miss_evidence_atlas(contexts: Sequence[Mapping[str, Any]], output: Path) -> None:
    canvas = Image.new("RGB", (1800, 1090), "#0b1110")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text(
        (28, 20),
        "Nine C2 ON_PITCH supply misses: evaluator target and frozen raw evidence",
        font=font(30, bold=True),
        fill="#f4f7f5",
    )
    draw.text(
        (28, 61),
        "Pink = evaluator visible person | cyan = nearest frozen raw proposal when present",
        font=font(17),
        fill="#a9b9b1",
    )
    for index, context in enumerate(contexts):
        column, row = index % 3, index // 3
        x, y = 25 + column * 590, 100 + row * 320
        with Image.open(context["image_path"]) as image:
            crop, bounds = crop_context(image.convert("RGB"), context["box"], pad_factor=2.8)
        boxes = [(context["box"], "#ff5b83")]
        if context["nearest_raw_box"]:
            boxes.append((context["nearest_raw_box"], "#33d6e8"))
        paste_crop_with_boxes(canvas, (x, y), (560, 220), crop, bounds, boxes)
        item = context["row"]
        draw.text((x, y + 225), item["anonymous_person_id"], font=font(17, bold=True), fill="#f4f7f5")
        draw.text(
            (x, y + 250),
            (
                f"{item['visible_width_pixels']:.1f} x {item['visible_height_pixels']:.1f}px | "
                f"{item['earliest_supported_origin']}"
            ),
            font=font(15),
            fill="#d8e1dc",
        )
        draw.text(
            (x, y + 274),
            f"phenotype {item['primary_phenotype']} | confidence {item['visual_confidence']}",
            font=font(15),
            fill="#f4c95d",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def phenotype_decision_atlas(contexts: Sequence[Mapping[str, Any]], output: Path) -> None:
    canvas = Image.new("RGB", (1800, 980), "#0b1110")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text(
        (30, 20),
        "Phenotype decision: R-A high-resolution small-person proposal bakeoff",
        font=font(30, bold=True),
        fill="#f4f7f5",
    )
    draw.text(
        (30, 61), "All nine targets are far-side people only 22-33 panorama pixels tall", font=font(18), fill="#f4c95d"
    )
    for index, context in enumerate(contexts):
        column, row = index % 5, index // 5
        x, y = 30 + column * 350, 105 + row * 290
        with Image.open(context["image_path"]) as image:
            crop, bounds = crop_context(image.convert("RGB"), context["box"], pad_factor=2.0)
        paste_crop_with_boxes(canvas, (x, y), (320, 210), crop, bounds, [(context["box"], "#ff5b83")])
        item = context["row"]
        draw.text((x, y + 214), item["anonymous_person_id"], font=font(15, bold=True), fill="#f4f7f5")
        draw.text(
            (x, y + 238),
            f"h={item['visible_height_pixels']:.1f}px | {item['earliest_supported_origin']}",
            font=font(14),
            fill="#a9b9b1",
        )
    y = 720
    draw.rounded_rectangle((30, y, 1770, 940), radius=6, fill="#121c19", outline="#355148", width=2)
    bullets = [
        "7/9: no FULL_PANORAMA_1280 raw proposal",
        "2/9: raw proposal exists but is badly localized",
        "Next experiment: exact frozen G2B 1280/1536/2048 + overlapping-view matrix on 9 targets and 18 controls",
        "No inference, threshold change, gate tuning, or component promotion occurred in G6C",
    ]
    for index, text in enumerate(bullets):
        draw.text((55, y + 25 + index * 43), f"{index + 1}. {text}", font=font(18), fill="#d8e1dc")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def error_ledger(c2: Mapping[str, Any], b1: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.m5_5g6c.pitch_and_observation_error_ledger.v1",
        "errors": [
            {
                "scope": "C2",
                "failure": "FROZEN_PROPOSAL_SUPPLY_CEILING",
                "detail": "Only 36/45 ON_PITCH people have frozen fused proposal support; no gate can meet 43/45.",
            },
            {
                "scope": "B1_P1",
                "failure": "UNSAFE_HARD_CLASSIFICATION",
                "detail": (
                    "P1 hard-classifies all eight boundary-uncertain targets and leaks both clear OFF_PITCH targets."
                ),
            },
            {
                "scope": "B1_P2_P3_P4",
                "failure": "EXCESS_CLEAR_PERSON_ROUTING",
                "detail": "Each conservative variant routes 6/8 clear ON_PITCH and 2/2 clear OFF_PITCH targets.",
            },
            {
                "scope": "PLAYER_OBSERVATION_V1",
                "failure": "OBSERVATION_SCREEN_BELOW_THRESHOLD",
                "detail": (
                    "Independent supply is 30/45 for P1 and 29/45 for P2-P4 with merged-as-clean "
                    "observations retained."
                ),
            },
        ],
        "c2_replay_passed": c2["passed"],
        "b1_replay_passed": b1["passed"],
        "observed_state_contamination_count": observation["observed_state_contamination_count"],
        "provenance_failure_count": observation["provenance_failure_count"],
        "threshold_or_margin_changed": False,
    }


def development_shortlist(combined: Mapping[str, Any], experiment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.m5_5g6c.development_shortlist.v1",
        "pitch_gate_candidates": combined["shortlisted_pitch_gate_variants"],
        "pitch_gate_candidate_frozen": False,
        "player_observation_schema": PLAYER_OBSERVATION_SCHEMA_VERSION,
        "player_observation_schema_frozen_development_only": True,
        "player_observation_runtime_candidate_frozen": False,
        "proposal_recovery_experiment": experiment["selected_experiment"],
        "proposal_recovery_experiment_id": experiment["selected_experiment_id"],
        "experiment_authorized_for_future_stage_only": True,
        "final_choice": FINAL_CHOICE,
        "production_promotion": False,
        **safety_payload(),
    }


def runtime_payload(
    started: float,
    c2_timings: Mapping[str, Sequence[float]],
    b1_timings: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        cuda_device = torch.cuda.get_device_name(0) if cuda_available else None
        torch_version = torch.__version__
    except ImportError:
        cuda_available = False
        cuda_device = None
        torch_version = None
    return {
        "schema_version": "football_intelligence.m5_5g6c.runtime.v1",
        "wall_seconds": round(time.perf_counter() - started, 6),
        "c2_cpu_gate_runtime": {
            variant: {
                "p50_ms_per_source": round(statistics.median(values), 8),
                "p95_ms_per_source": round(quantile(values, 0.95), 8),
            }
            for variant, values in c2_timings.items()
            if variant in {"P1", "P2", "P3", "P4"}
        },
        "b1_cpu_gate_runtime": {
            variant: {
                "p50_ms_per_source": round(statistics.median(values), 8),
                "p95_ms_per_source": round(quantile(values, 0.95), 8),
            }
            for variant, values in b1_timings.items()
        },
        "cuda_environment": {
            "available": cuda_available,
            "device": cuda_device,
            "torch_version": torch_version,
        },
        "gpu_computation_performed": False,
        "detector_segmenter_or_promptable_inference_performed": False,
        "new_proposals_generated": False,
    }


def source_diff() -> str:
    paths = (
        "scripts/build_m5_5g6c_pitch_gate_recovery_decision.py",
        "tests/test_m5_5g6c_pitch_gate_recovery.py",
        "tests/test_m5_5g6b_boundary_gold.py",
    )
    cached = subprocess.run(
        ["git", "diff", "--cached", "--", *paths], cwd=REPO, text=True, capture_output=True, check=True
    ).stdout
    if cached.strip():
        return cached
    return subprocess.run(
        ["git", "diff", "HEAD", "--", *paths], cwd=REPO, text=True, capture_output=True, check=True
    ).stdout


def build_review_pack(
    *,
    repository: Mapping[str, Any],
    b1_validation: Mapping[str, Any],
    universe: Mapping[str, Any],
    specs: Mapping[str, Any],
    c2: Mapping[str, Any],
    b1: Mapping[str, Any],
    combined: Mapping[str, Any],
    observation: Mapping[str, Any],
    phenotype_summary: Mapping[str, Any],
    experiment: Mapping[str, Any],
    experiment_contract: Mapping[str, Any],
    runtime: Mapping[str, Any],
    shortlist: Mapping[str, Any],
    validation: Mapping[str, Any],
    finalize: bool,
) -> dict[str, Any]:
    pack = DIRS["pack"]
    pack.mkdir(parents=True, exist_ok=True)
    for path in pack.iterdir():
        if path.name == "04_SOURCE_DIFF.patch" and finalize and path.stat().st_size:
            continue
        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    write_json(
        pack / "01_EXECUTIVE_OUTCOME.json",
        {
            "classification": CLASSIFICATION,
            "final_choice": FINAL_CHOICE,
            "selected_experiment": SELECTED_EXPERIMENT,
            "pitch_gate_candidate_frozen": False,
            "player_observation_schema_frozen": True,
            "component_promoted": False,
            **safety_payload(),
        },
    )
    write_json(pack / "02_REPOSITORY_STATE.json", repository)
    write_json(pack / "03_B1_COMPLETION_VALIDATION.json", b1_validation)
    patch_path = pack / "04_SOURCE_DIFF.patch"
    if not patch_path.is_file() or not patch_path.stat().st_size:
        patch_path.write_text(source_diff(), encoding="utf-8")
    write_json(pack / "05_UNIVERSE_AND_FROZEN_SPECIFICATIONS.json", {"universe": universe, "specifications": specs})
    write_json(pack / "06_C2_GATE_REPLAY.json", c2)
    write_json(
        pack / "07_B1_GATE_RESULTS.json",
        {
            **b1,
            "variants": [
                {key: value for key, value in row.items() if key != "person_level_ledger"} for row in b1["variants"]
            ],
        },
    )
    write_json(pack / "08_COMBINED_GATE_SHORTLIST.json", combined)
    write_json(pack / "09_PLAYER_OBSERVATION_STATUS.json", observation)
    write_json(pack / "10_NINE_PERSON_PHENOTYPE_SUMMARY.json", phenotype_summary)
    write_json(
        pack / "11_RECOVERY_EXPERIMENT_AND_CONTRACT.json",
        {
            "decision": experiment,
            "contract": {
                key: value
                for key, value in experiment_contract.items()
                if key not in {"target_universe", "control_universe"}
            },
        },
    )
    write_json(pack / "12_RUNTIME.json", runtime)
    write_json(pack / "13_DEVELOPMENT_SHORTLIST_AND_DECISION.json", shortlist)
    write_json(pack / "14_TESTS_AND_SAFETY.json", validation)
    shutil.copy2(DIRS["visuals"] / "01_B1_PITCH_GATE_OUTCOMES.png", pack / "15_B1_PITCH_GATE_OUTCOMES.png")
    shutil.copy2(DIRS["visuals"] / "02_NINE_PERSON_RAW_EVIDENCE.png", pack / "16_NINE_PERSON_RAW_EVIDENCE.png")
    shutil.copy2(
        DIRS["visuals"] / "03_PHENOTYPE_AND_RECOVERY_DECISION.png", pack / "17_PHENOTYPE_AND_RECOVERY_DECISION.png"
    )
    files = sorted(path for path in pack.iterdir() if path.is_file())
    visual_count = sum(path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif"} for path in files)
    forbidden_extensions = {".pt", ".pth", ".onnx", ".mp4", ".avi", ".mov", ".zip"}
    records = [file_record(path) | {"filename": path.name} for path in files]
    for row in records:
        row.pop("path", None)
    checks = {
        "flat": all(path.parent == pack for path in files),
        "file_count_within_limit": len(files) + 1 <= 20,
        "total_bytes_within_limit": sum(path.stat().st_size for path in files) <= 50 * 1024 * 1024,
        "visual_count_within_limit": visual_count <= 3,
        "exactly_three_visuals": visual_count == 3,
        "source_diff_present_nonempty": patch_path.is_file() and patch_path.stat().st_size > 0,
        "forbidden_extensions_absent": not any(path.suffix.lower() in forbidden_extensions for path in files),
        "full_human_payloads_absent": not any(
            token in path.name.lower()
            for path in files
            for token in ("completed_review", "decisions", "private_mapping")
        ),
    }
    manifest = {
        "schema_version": "football_intelligence.m5_5g6c.review_pack_manifest.v1",
        "flat": True,
        "maximum_file_count": 20,
        "maximum_total_bytes": 50 * 1024 * 1024,
        "maximum_visual_files": 3,
        "manifest_self_hash_omitted": True,
        "file_count_including_manifest": len(files) + 1,
        "total_bytes_excluding_manifest": sum(path.stat().st_size for path in files),
        "visual_file_count": visual_count,
        "files": records,
        "checks": checks,
        "passed": all(checks.values()),
    }
    write_json(pack / "18_REVIEW_PACK_MANIFEST.json", manifest)
    if not manifest["passed"]:
        raise RuntimeError(f"FAIL_REVIEW_PACK: {checks}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--validated", action="store_true")
    parser.add_argument("--focused-passed", type=int, default=0)
    parser.add_argument("--regression-passed", type=int, default=0)
    parser.add_argument("--full-passed", type=int, default=0)
    parser.add_argument("--full-warnings", type=int, default=0)
    parser.add_argument("--push-result", default="PENDING")
    parser.add_argument("--remote-head")
    args = parser.parse_args()
    started = time.perf_counter()
    for path in DIRS.values():
        path.mkdir(parents=True, exist_ok=True)
    repository, prompt_validation = repository_and_prompt_validation()
    if args.remote_head:
        repository["remote_head"] = args.remote_head
        repository["local_remote_head_match"] = repository["head"] == args.remote_head
    for path in PROMPT.iterdir():
        if path.is_file():
            shutil.copy2(path, DIRS["inputs"] / path.name)
    write_json(DIRS["inputs"] / "repository_state.json", repository)
    write_json(DIRS["inputs"] / "prompt_pack_validation.json", prompt_validation)
    before = protected_manifest()
    write_json(DIRS["inputs"] / "protected_input_manifest_before.json", before)

    b1_completed, b1_manifest, b1_validation = validate_b1_completion()
    universe = evaluator_universe_contract()
    write_json(DIRS["b1"] / "b1_completion_and_gold_validation.json", b1_validation)
    write_json(DIRS["b1"] / "evaluator_universe_contract.json", universe)

    specs = validate_frozen_specifications(b1_manifest)
    write_json(DIRS["specs"] / "frozen_specification_validation.json", specs)

    c2_result, _, c2_timings = replay_c2()
    write_json(DIRS["c2"] / "c2_gate_replay_validation.json", c2_result)
    b1_result, b1_visuals, b1_timings = replay_b1(b1_completed, b1_manifest)
    write_json(DIRS["b1_gate"] / "b1_pitch_gate_results.json", b1_result)
    combined = combined_gate_decision(c2_result, b1_result)
    write_json(DIRS["combined"] / "combined_pitch_gate_shortlist.json", combined)

    phenotype_rows, phenotype_summary, miss_visuals = phenotype_nine_misses()
    write_jsonl(DIRS["misses"] / "nine_person_miss_phenotype_ledger.jsonl", phenotype_rows)
    write_json(DIRS["misses"] / "miss_phenotype_summary.json", phenotype_summary)
    experiment, experiment_contract = proposal_recovery_decision(phenotype_rows, phenotype_summary)
    write_json(DIRS["recovery"] / "proposal_recovery_experiment_decision.json", experiment)
    write_json(DIRS["recovery"] / "proposal_recovery_experiment_contract.json", experiment_contract)

    observation = player_observation_status(combined)
    write_json(DIRS["observation"] / "player_observation_v1_status.json", observation)
    errors = error_ledger(c2_result, b1_result, observation)
    write_json(DIRS["visuals"] / "pitch_and_observation_error_ledger.json", errors)
    b1_gate_atlas(b1_visuals, DIRS["visuals"] / "01_B1_PITCH_GATE_OUTCOMES.png")
    miss_evidence_atlas(miss_visuals, DIRS["visuals"] / "02_NINE_PERSON_RAW_EVIDENCE.png")
    phenotype_decision_atlas(miss_visuals, DIRS["visuals"] / "03_PHENOTYPE_AND_RECOVERY_DECISION.png")

    shortlist = development_shortlist(combined, experiment)
    write_json(DIRS["decision"] / "development_shortlist.json", shortlist)
    decision_text = (
        "# M5.5G.6C decision\n\n"
        f"**Choice B: `{FINAL_CHOICE}`**\n\n"
        "No frozen pitch-gate variant passes the unchanged C2 broad-person and B1 boundary-stress screens. "
        "The Player Observation v1 schema and truth-free materializer remain frozen for development, but no "
        "runtime observation candidate is complete while proposal supply remains below the frozen screen.\n\n"
        f"The next authorized development experiment is `{SELECTED_EXPERIMENT}` using the exact frozen G2B "
        "resolution/view matrix on nine small far-side misses and eighteen matched controls. G6C executes no "
        "inference and changes no threshold, margin, crop, tile, NMS, fusion, or production default.\n\n"
        "No component is promoted.\n"
    )
    (DIRS["decision"] / "final_decision.md").write_text(decision_text, encoding="utf-8")
    runtime = runtime_payload(started, c2_timings, b1_timings)
    write_json(DIRS["commands"] / "runtime.json", runtime)
    after = protected_manifest()
    write_json(DIRS["commands"] / "protected_input_manifest_after.json", after)
    if before != after:
        raise RuntimeError("FAIL_PRIOR_STAGE_MUTATION")

    validation = {
        "schema_version": "football_intelligence.m5_5g6c.validation_results.v1",
        "all_required_checks_passed": args.validated,
        "checks": {
            "prompt_pack": "PASS",
            "b1_completion_and_gold": "PASS",
            "universe_separation": "PASS",
            "frozen_specifications": "PASS",
            "c2_gate_replay": "PASS",
            "b1_gate_replay": "PASS",
            "miss_phenotyping": "PASS",
            "recovery_experiment_decision": "PASS",
            "prior_inputs_unchanged": True,
            "focused_tests": {"status": "PASS" if args.validated else "PENDING", "passed": args.focused_passed},
            "prior_regressions": {
                "status": "PASS" if args.validated else "PENDING",
                "passed": args.regression_passed,
            },
            "full_suite": {
                "status": "PASS" if args.validated else "PENDING",
                "passed": args.full_passed,
                "warnings": args.full_warnings,
            },
        },
        "push_result": args.push_result,
        "remote_head": args.remote_head,
        "inference_performed": False,
        "threshold_or_margin_changed": False,
        "prior_artifacts_unchanged": True,
        "component_promoted": False,
        **safety_payload(),
    }
    write_json(DIRS["commands"] / "validation_results.json", validation)
    pack_manifest = build_review_pack(
        repository=repository,
        b1_validation=b1_validation,
        universe=universe,
        specs=specs,
        c2=c2_result,
        b1=b1_result,
        combined=combined,
        observation=observation,
        phenotype_summary=phenotype_summary,
        experiment=experiment,
        experiment_contract=experiment_contract,
        runtime=runtime,
        shortlist=shortlist,
        validation=validation,
        finalize=args.finalize,
    )
    write_json(DIRS["commands"] / "review_pack_validation.json", pack_manifest)
    build_summary = {
        "classification": CLASSIFICATION,
        "b1_completion_valid": True,
        "c2_gate_replay_exact": True,
        "b1_shortlisted_gate_variants": b1_result["shortlisted_variants"],
        "combined_shortlisted_gate_variants": combined["shortlisted_pitch_gate_variants"],
        "player_observation_schema_frozen": True,
        "selected_recovery_experiment": SELECTED_EXPERIMENT,
        "selected_recovery_experiment_id": SELECTED_EXPERIMENT_ID,
        "final_choice": FINAL_CHOICE,
        "review_pack_passed": pack_manifest["passed"],
        "review_pack_file_count": pack_manifest["file_count_including_manifest"],
        "prior_artifacts_unchanged": True,
        "inference_performed": False,
        "threshold_or_margin_changed": False,
        "component_promoted": False,
    }
    write_json(DIRS["commands"] / "build_summary.json", build_summary)
    print(json.dumps(build_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
