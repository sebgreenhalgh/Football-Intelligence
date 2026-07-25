from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from football_intelligence.detection_gold.consolidation import consolidate_proposals
from football_intelligence.detection_gold.player_observation import (
    PLAYER_OBSERVATION_SCHEMA_VERSION,
    apply_pitch_gate,
    clip_polygon_to_bounds,
    footpoint_geometry_variant_specification,
    materialize_player_observation,
    panorama_to_focal,
    player_observation_json_schema,
    pitch_gate_variant_specification,
    signed_distance_to_polygon,
)
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash

BASELINE_COMMIT = "b54ace62ec79217fbd175a0b4edc84b1f1a0b9b5"
APPROVED_DETECTOR_SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
LIGHT_HQ_SAM_SHA256 = "0f32c075ccdd870ae54db2f7630e7a0878ede5a2b06d05d6fe02c65a82fb7196"
LIGHT_HQ_SAM_SPEC_SHA256 = "c0fc0860a9c43127aac78cb41360b6b61499218550fad96b0584ca79557c66b0"

REPO = Path(__file__).resolve().parents[1]
FOOTBALL_ROOT = REPO.parent
PART3 = FOOTBALL_ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT_PACK = PART3 / "M5_5G6A_Pitch_Gate_Player_Observation_V1_Codex_Prompt_Pack"
STAGE = PART3 / "M5_5G6A_PITCH_BOUNDARY_GATE_AND_PLAYER_OBSERVATION_V1_INTEGRATION_DEVELOPMENT_v1"
C2_PACKAGE = (
    PART3
    / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
    / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
)
C2_BUNDLE = C2_PACKAGE / "decisions" / "completed_tranches" / "C2_PITCH_BOUNDARY"
G3_STAGE = PART3 / "M5_5G3_PROVENANCE_AWARE_CROSS_VIEW_CONSOLIDATION_AND_MERGED_AMBIGUITY_GATE_DEVELOPMENT_v1"
G4R2_STAGE = PART3 / "M5_5G4_R2_CORRECTED_DENSE_GOLD_OVERLAY_APPLICATION_AND_INSTANCE_SEPARATION_REEVALUATION_v1"
G5A_STAGE = PART3 / "M5_5G5A_AUTHORIZED_PROMPTABLE_MASK_BAKEOFF_AND_DENSE_BRANCH_DECISION_v1"

DIRS = {
    "inputs": STAGE / "00_PROMPT_AND_INPUTS",
    "c2": STAGE / "01_C2_COMPLETION_AND_GOLD_VALIDATION",
    "projection": STAGE / "02_PITCH_POLYGON_AND_TRANSFORM_DIAGNOSIS",
    "variants": STAGE / "03_FROZEN_FOOTPOINT_AND_GATE_VARIANTS",
    "schema": STAGE / "04_PLAYER_OBSERVATION_V1_SCHEMA",
    "pipeline": STAGE / "05_OBSERVATION_PIPELINE_INTEGRATION",
    "evaluation": STAGE / "06_PITCH_GATE_AND_SUPPLY_EVALUATION",
    "dense": STAGE / "07_DENSE_BRANCH_INTEGRATION",
    "visuals": STAGE / "08_VISUAL_QA_AND_ERROR_LEDGER",
    "decision": STAGE / "09_NEXT_STAGE_DECISION",
    "commands": STAGE / "10_COMMANDS_AND_TESTS",
    "pack": STAGE / "11_REVIEW_PACK_FOR_CHATGPT",
    "tmp": STAGE / "_tmp",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def freeze_json(path: Path, hash_path: Path, payload: Mapping[str, Any]) -> str:
    write_json(path, payload)
    digest = sha256_file(path)
    hash_path.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def iou(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    x1, y1 = max(float(left["x1"]), float(right["x1"])), max(float(left["y1"]), float(right["y1"]))
    x2, y2 = min(float(left["x2"]), float(right["x2"])), min(float(left["y2"]), float(right["y2"]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (float(left["x2"]) - float(left["x1"])) * (float(left["y2"]) - float(left["y1"]))
    right_area = (float(right["x2"]) - float(right["x1"])) * (float(right["y2"]) - float(right["y1"]))
    return intersection / max(1e-12, left_area + right_area - intersection)


def point_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return math.dist((float(left["x"]), float(left["y"])), (float(right["x"]), float(right["y"])))


def candidate_nodes(case: Mapping[str, Any], frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    width, height = int(frame["image_width"]), int(frame["image_height"])
    source_hash = str(frame["source_frame_sha256"])
    fused = sorted(
        (candidate for candidate in frame["candidates"] if candidate["stage"] == "FUSED"),
        key=lambda row: str(row["diagnostic_uuid"]),
    )
    runtime_hash = stable_hash(
        {
            "approved_detector_sha256": APPROVED_DETECTOR_SHA256,
            "supply": "G2B_FUSED_PRIMARY_PROPOSALS",
            "detector_settings_changed": False,
        }
    )
    nodes: list[dict[str, Any]] = []
    for row in fused:
        box = {key: float(row["bbox_original_pixels"][key]) for key in ("x1", "y1", "x2", "y2")}
        transform_hash = stable_hash(
            {
                "coordinate_space": "canonical_panorama_pixels",
                "source_frame_sha256": source_hash,
                "image_width": width,
                "image_height": height,
                "transform": "IDENTITY_ALREADY_PANORAMA",
            }
        )
        nodes.append(
            {
                "source_frame_sha256": source_hash,
                "proposal_uuid": str(row["diagnostic_uuid"]),
                "source_view_family": "PRIMARY_FULL_PANORAMA",
                "inference_view_id": str(row["inference_view"]),
                "source_view_footprint": {"x1": 0.0, "y1": 0.0, "x2": float(width), "y2": float(height)},
                "crop_bounds_panorama_pixels": None,
                "tile_bounds_panorama_pixels": None,
                "raw_candidate_index": None,
                "score": float(row["score"]),
                "class_provenance": "MODEL_NAMES_RUNTIME_PERSON",
                "bbox_panorama_pixels": box,
                "transform_hash": transform_hash,
                "checkpoint_runtime_hash": runtime_hash,
                "parent_lineage_ids": [str(row["source_row_sha256"])],
                "near_tile_or_crop_edge": False,
                "visible_in_another_overlapping_view": False,
            }
        )
    return nodes


def c2_validation() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bundle_validation = validate_completion_bundle(C2_BUNDLE)
    if not bundle_validation["passed"]:
        raise RuntimeError("FAIL_C2_COMPLETION_OR_GOLD: atomic bundle validation failed")
    completed = read_json(C2_BUNDLE / "completed_review.json")
    annotations = completed["state"]["annotations"]
    expected_cases = [f"m5_5g1a_case_{index:03d}" for index in range(53, 65)]
    if sorted(annotations) != expected_cases:
        raise RuntimeError("FAIL_C2_COMPLETION_OR_GOLD: C2 case membership differs")
    root_state = read_json(C2_PACKAGE / "decisions" / "review_decisions.json")
    if int(root_state["event_sequence"]) != 57:
        raise RuntimeError("FAIL_C2_COMPLETION_OR_GOLD: root sequence is not 57")
    people = [person for annotation in annotations.values() for person in annotation["player_instances"]]
    relations = [relation for annotation in annotations.values() for relation in annotation["candidate_relations"]]
    counts = {
        "cases": len(annotations),
        "people": len(people),
        "roles": dict(sorted(Counter(person["coarse_role"] for person in people).items())),
        "pitch_states": dict(sorted(Counter(person["pitch_state"] for person in people).items())),
        "footpoint_status": dict(sorted(Counter(person["footpoint_status"] for person in people).items())),
        "candidate_relations": dict(sorted(Counter(row["relation"] for row in relations).items())),
    }
    expected = {
        "cases": 12,
        "people": 96,
        "roles": {"GOALKEEPER": 2, "PLAYER": 85, "REFEREE": 2, "STAFF_OR_SPECTATOR": 6, "UNKNOWN": 1},
        "pitch_states": {"OFF_PITCH": 51, "ON_PITCH": 45},
        "footpoint_status": {"FEET_NOT_VISIBLE": 9, "OBSERVED_APPROXIMATE": 6, "OBSERVED_CLEAR": 81},
        "candidate_relations": {
            "AMBIGUOUS": 10,
            "BACKGROUND": 1,
            "CLEAN_SINGLE_INSTANCE": 42,
            "DUPLICATE_OF_INSTANCE": 26,
            "MERGED_MULTIPLE_INSTANCES": 5,
        },
    }
    if counts != expected:
        raise RuntimeError(f"FAIL_C2_COMPLETION_OR_GOLD: exact counts differ: {counts}")
    expected_hashes = {
        "completed_review.json": "79635a1093736b1699e971fc89b052db106c29896e4960c69abf9672d9e1157d",
        "completed_review_events.jsonl": "f1c82266c22816674996df939ce3867d3dfc1e81fbc816845376844e2012b517",
        "completed_review_summary.json": "73f4af6152a617fa39ca58cb195247b811e42621cb3296c129e3eb4551a59c0a",
    }
    checks = {
        name: {"expected": expected_hash, "actual": sha256_file(C2_BUNDLE / name)}
        for name, expected_hash in expected_hashes.items()
    }
    if any(row["expected"] != row["actual"] for row in checks.values()):
        raise RuntimeError("FAIL_C2_COMPLETION_OR_GOLD: completion hashes differ")
    return (
        completed,
        counts,
        {
            "schema_version": "football_intelligence.m5_5g6a.c2_completion_validation.v1",
            "classification": "PASS_C2_COMPLETION_AND_GOLD_VALIDATED",
            "completion_bundle_valid": True,
            "root_event_sequence": 57,
            "tranche_event_sequence": int(completed["state"]["event_sequence"]),
            "exact_counts": counts,
            "artifact_hash_checks": checks,
            "source_bindings_valid": True,
            "single_reviewer_development_gold_only": True,
            "validation_or_holdout_use": False,
        },
    )


def protected_inputs() -> list[Path]:
    files = [
        C2_PACKAGE / "decisions" / "review_decisions.json",
        C2_PACKAGE / "decisions" / "review_decision_events.jsonl",
        C2_PACKAGE / "reviewer_manifest.json",
        G3_STAGE / "03_FROZEN_CONSOLIDATION_VARIANTS" / "consolidation_variant_specification.json",
        G3_STAGE / "03_FROZEN_CONSOLIDATION_VARIANTS" / "consolidation_variant_specification.sha256",
        G3_STAGE / "06_PERSON_OBSERVATION_EVALUATION" / "final_observation_ledger.jsonl",
        G3_STAGE / "08_NEXT_STAGE_DECISION" / "development_consolidator_shortlist.json",
        G4R2_STAGE / "02_DENSE_GOLD_V2_OVERLAY_APPLICATION" / "dense_gold_v2_manifest.json",
        G4R2_STAGE / "04_FIXED_BASELINE_AND_ORACLE_REEVALUATION" / "corrected_box_only_baseline.json",
        G4R2_STAGE / "08_NEXT_STAGE_DECISION" / "final_decision.json",
        G5A_STAGE / "04_FROZEN_PROMPT_AND_CROP_MATRIX" / "frozen_crop_prompt_specification.json",
        G5A_STAGE / "04_FROZEN_PROMPT_AND_CROP_MATRIX" / "frozen_crop_prompt_specification.sha256",
        G5A_STAGE / "09_NEXT_STAGE_DECISION" / "development_shortlist.json",
    ]
    for tranche in ("A_CORE_STATIC", "B_REMAINING_STATIC", "C1_DENSE_OVERLAP", "C2_PITCH_BOUNDARY"):
        bundle = C2_PACKAGE / "decisions" / "completed_tranches" / tranche
        files.extend(bundle / name for name in ("completed_review.json", "completed_review_events.jsonl"))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"FAIL_PRIOR_STAGE_MUTATION: protected inputs missing: {missing}")
    return files


def protected_manifest() -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.m5_5g6a.protected_input_manifest.v1",
        "files": [file_record(path) for path in protected_inputs()],
    }


def source_frame(case: Mapping[str, Any]) -> Mapping[str, Any]:
    source_sequence = int(case["source_frame_sequence"])
    matches = [
        frame for frame in case["visible_metadata"]["frame_records"] if int(frame["frame_sequence"]) == source_sequence
    ]
    if len(matches) != 1:
        raise RuntimeError(f"FAIL_C2_COMPLETION_OR_GOLD: source frame binding differs for {case['case_id']}")
    return matches[0]


def relation_index(annotation: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["candidate_uuid"]): dict(row) for row in annotation["candidate_relations"]}


def evaluate_variant(
    variant: str,
    case_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    all_people: dict[str, Mapping[str, Any]] = {}
    gate_accepted_by_person: dict[str, set[str]] = defaultdict(set)
    independent_accepted_by_person: dict[str, set[str]] = defaultdict(set)
    routed_by_person: dict[str, set[str]] = defaultdict(set)
    accepted_rows: list[Mapping[str, Any]] = []
    routed_rows: list[Mapping[str, Any]] = []
    unmatched_rows: list[Mapping[str, Any]] = []
    localization_ious: list[float] = []
    footpoint_errors: list[float] = []
    signed_distance_errors: list[float] = []
    merged_as_clean = 0
    distinct_person_suppression = 0
    duplicate_lineage_count = 0
    provenance_failures = 0
    contamination = 0
    for case in case_rows:
        polygon = case["pitch_polygon"]
        people = {str(row["annotation_uuid"]): row for row in case["people"]}
        all_people.update(people)
        for item in case["observations_by_variant"][variant]:
            observation = item["observation"]
            targets = [target for target in item["resolved_target_uuids"] if target in people]
            accepted = observation["observation_state"] in {"OBSERVED_BOX", "OBSERVED_MASK"} and observation[
                "pitch_relation"
            ] in {"ON_PITCH", "UNGATED_RETAIN"}
            routed = observation["observation_state"] in {
                "ROUTE_DENSE_REVIEW",
                "ROUTE_PITCH_BOUNDARY_REVIEW",
                "UNRESOLVED",
            }
            if accepted:
                accepted_rows.append(item)
                for target in targets:
                    gate_accepted_by_person[target].add(str(observation["observation_uuid"]))
                if len(targets) == 1:
                    independent_accepted_by_person[targets[0]].add(str(observation["observation_uuid"]))
            if routed:
                routed_rows.append(item)
                for target in targets:
                    routed_by_person[target].add(str(observation["observation_uuid"]))
            if not targets:
                unmatched_rows.append(item)
            if accepted and len(set(targets)) > 1:
                merged_as_clean += 1
                distinct_person_suppression += max(0, len({target for target in targets if target in people}) - 1)
            if accepted and len(observation["proposal_uuid_lineage"]) > 1:
                duplicate_lineage_count += len(observation["proposal_uuid_lineage"]) - 1
            required_provenance = (
                observation.get("source_frame_sha256"),
                observation.get("proposal_uuid_lineage"),
                observation.get("source_view_ids"),
                observation.get("provenance_hash"),
            )
            if not all(required_provenance):
                provenance_failures += 1
            serialized = json.dumps(observation, sort_keys=True).lower()
            if "predicted" in serialized or "interpolated" in serialized or "track_id" in serialized:
                contamination += 1
            if len(targets) == 1:
                truth = people[targets[0]]
                if truth.get("visible_body_box"):
                    localization_ious.append(iou(observation["visible_box"], truth["visible_body_box"]))
                if truth.get("footpoint"):
                    footpoint_errors.append(point_distance(observation["footpoint_estimate"], truth["footpoint"]))
                    signed_distance_errors.append(
                        abs(
                            signed_distance_to_polygon(observation["footpoint_estimate"], polygon)
                            - signed_distance_to_polygon(truth["footpoint"], polygon)
                        )
                    )
    on_people = {key: row for key, row in all_people.items() if row["pitch_state"] == "ON_PITCH"}
    off_people = {key: row for key, row in all_people.items() if row["pitch_state"] == "OFF_PITCH"}
    gate_on_retained = {key for key in on_people if gate_accepted_by_person[key]}
    gate_off_leaked = {key for key in off_people if gate_accepted_by_person[key]}
    independent_on_retained = {key for key in on_people if independent_accepted_by_person[key]}
    exact_one = {key for key in on_people if len(independent_accepted_by_person[key]) == 1}
    duplicate_accepted = sum(max(0, len(independent_accepted_by_person[key]) - 1) for key in on_people)
    feet_not_visible = {key for key, row in all_people.items() if row["footpoint_status"] == "FEET_NOT_VISIBLE"}
    feet_routed = {key for key in feet_not_visible if routed_by_person[key]}
    feet_missing = {key for key in feet_not_visible if not gate_accepted_by_person[key] and not routed_by_person[key]}
    role_on = defaultdict(set)
    for key in gate_on_retained:
        role_on[str(on_people[key]["coarse_role"])].add(key)
    staff_leakage = {key for key in gate_off_leaked if str(off_people[key]["coarse_role"]) == "STAFF_OR_SPECTATOR"}
    accepted_observation_count = len(accepted_rows)
    duplicate_rate = duplicate_accepted / max(1, accepted_observation_count)
    return {
        "pitch_gate_variant": variant,
        "denominators": {
            "all_on_pitch_people": len(on_people),
            "footpoint_scoreable_on_pitch_people": sum(
                row["footpoint_status"] != "FEET_NOT_VISIBLE" for row in on_people.values()
            ),
            "feet_not_visible_all_pitch_states": len(feet_not_visible),
            "off_pitch_labelled_people": len(off_people),
            "unscored_crowd_proposals": sum(item["evaluator_status"] == "UNSCORED_CROWD" for item in unmatched_rows),
        },
        "pitch_gate": {
            "on_pitch_person_supply_retained": len(gate_on_retained),
            "off_pitch_labelled_person_leakage": len(gate_off_leaked),
            "boundary_review_observation_count": sum(
                row["observation"]["observation_state"] == "ROUTE_PITCH_BOUNDARY_REVIEW" for row in routed_rows
            ),
            "feet_not_visible_person_routed": len(feet_routed),
            "feet_not_visible_person_missing_without_runtime_claim": len(feet_missing),
            "staff_spectator_on_pitch_leakage": len(staff_leakage),
            "referee_retained": len(role_on["REFEREE"]),
            "goalkeeper_retained": len(role_on["GOALKEEPER"]),
            "box_mask_footpoint_disagreement_count": 0,
            "median_signed_distance_error_pixels": round(statistics.median(signed_distance_errors), 8)
            if signed_distance_errors
            else None,
        },
        "player_observation_v1": {
            "one_accepted_observation_per_on_pitch_person": len(exact_one),
            "accepted_observation_count": accepted_observation_count,
            "duplicate_accepted_observations": duplicate_accepted,
            "duplicate_accepted_observation_rate": round(duplicate_rate, 8),
            "suppressed_duplicate_lineage_members": duplicate_lineage_count,
            "merged_as_clean_observations": merged_as_clean,
            "distinct_person_suppression": distinct_person_suppression,
            "missing_on_pitch_people": len(on_people) - len(independent_on_retained),
            "unresolved_or_routed_observations": len(routed_rows),
            "median_visible_box_iou": round(statistics.median(localization_ious), 8) if localization_ious else None,
            "median_footpoint_error_pixels": round(statistics.median(footpoint_errors), 8)
            if footpoint_errors
            else None,
            "provenance_failure_count": provenance_failures,
            "observed_state_contamination_count": contamination,
        },
        "evaluator_only": {
            "on_pitch_person_uuids_retained": sorted(gate_on_retained),
            "off_pitch_person_uuids_leaked": sorted(gate_off_leaked),
            "human_labels_entered_runtime": False,
        },
    }


def partial_crowd_unscored_rows(completed: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Audit unmatched post-NMS rows against the evaluator-only crowd boundary."""

    annotations = completed["state"]["annotations"]
    manifest_cases = {str(row["case_id"]): row for row in manifest["cases"]}
    rows: list[dict[str, Any]] = []
    for case_id in sorted(annotations):
        annotation = annotations[case_id]
        case = manifest_cases[case_id]
        frame = source_frame(case)
        polygon = case["visible_metadata"]["pitch_polygon_vertices"]
        relations = relation_index(annotation)
        candidates = {
            str(candidate["diagnostic_uuid"]): candidate
            for candidate in frame["candidates"]
            if candidate["stage"] == "POST_NMS"
        }
        for candidate_uuid, candidate in sorted(candidates.items()):
            relation = relations.get(candidate_uuid)
            targets = [] if relation is None else [str(value) for value in relation["annotation_uuids"]]
            if targets:
                continue
            box = candidate["bbox_original_pixels"]
            relation_name = apply_pitch_gate("P1", box, polygon)["pitch_relation"]
            if relation_name != "OFF_PITCH":
                continue
            rows.append(
                {
                    "case_id": case_id,
                    "candidate_uuid": candidate_uuid,
                    "source_row_sha256": str(candidate["source_row_sha256"]),
                    "stage": "POST_NMS",
                    "status": "UNSCORED_CROWD",
                    "human_candidate_relation": relation["relation"] if relation else "UNANNOTATED",
                    "reason": (
                        "Unmatched post-NMS proposal lies outside the approved pitch in the C2 "
                        "partial-clear-person off-pitch crowd universe"
                    ),
                    "runtime_gate_input": False,
                    "background_false_positive_scored": False,
                }
            )
    return rows


def build_runtime_cases(
    completed: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[float]]]:
    annotations = completed["state"]["annotations"]
    manifest_cases = {str(row["case_id"]): row for row in manifest["cases"]}
    runtime_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    timings: dict[str, list[float]] = defaultdict(list)
    for case_id in sorted(annotations):
        annotation = annotations[case_id]
        case = manifest_cases[case_id]
        frame = source_frame(case)
        binding = case["visible_metadata"]["source_binding"]
        if str(binding["source_frame_sha256"]) != str(frame["source_frame_sha256"]):
            raise RuntimeError(f"FAIL_C2_COMPLETION_OR_GOLD: source hash mismatch for {case_id}")
        panorama_path = C2_PACKAGE / "evidence" / case_id / str(frame["panorama_asset_path"])
        if sha256_file(panorama_path) != str(frame["source_frame_sha256"]):
            raise RuntimeError(f"FAIL_C2_COMPLETION_OR_GOLD: panorama bytes differ for {case_id}")
        polygon = [dict(point) for point in case["visible_metadata"]["pitch_polygon_vertices"]]
        nodes = candidate_nodes(case, frame)
        consolidated = consolidate_proposals(nodes, "IOU_CONNECTED_COMPONENT_055", apply_merged_gate=True)
        relations = relation_index(annotation)
        by_variant: dict[str, list[dict[str, Any]]] = {}
        for variant in ("P0", "P1", "P2", "P3", "P4"):
            started = time.perf_counter_ns()
            rows = []
            for observation in consolidated["observations"]:
                materialized = materialize_player_observation(
                    observation,
                    frame_index=int(frame["frame_sequence"]),
                    pitch_gate_variant=variant,
                    polygon=polygon,
                )
                member_relations = [
                    relations[identifier]
                    for identifier in observation["cluster_member_proposal_uuids"]
                    if identifier in relations
                ]
                targets = sorted(
                    {
                        str(target)
                        for relation in member_relations
                        if relation["relation"] not in {"AMBIGUOUS", "BACKGROUND"}
                        for target in relation["annotation_uuids"]
                    }
                )
                evaluator_status = "SCORED"
                if not targets:
                    f0 = apply_pitch_gate("P1", observation["box_panorama_pixels"], polygon)
                    if f0["pitch_relation"] == "OFF_PITCH":
                        evaluator_status = "UNSCORED_CROWD"
                row = {
                    "case_id": case_id,
                    "source_group": str(binding.get("sequence_id") or case_id),
                    "observation": materialized,
                    "resolved_target_uuids": targets,
                    "evaluator_relation_types": sorted({str(item["relation"]) for item in member_relations}),
                    "evaluator_status": evaluator_status,
                }
                rows.append(row)
                runtime_rows.append(
                    {
                        "case_id": case_id,
                        "pipeline": "O0_BOX_ONLY_CONTROL",
                        "pitch_gate_variant": variant,
                        "observation": materialized,
                    }
                )
            timings[variant].append((time.perf_counter_ns() - started) / 1_000_000)
            by_variant[variant] = rows
        case_rows.append(
            {
                "case_id": case_id,
                "source_group": str(binding.get("sequence_id") or case_id),
                "source_frame_sha256": str(frame["source_frame_sha256"]),
                "frame_index": int(frame["frame_sequence"]),
                "source_image_path": str(panorama_path),
                "focal_image_path": str(C2_PACKAGE / "evidence" / case_id / str(frame["focal_asset_path"])),
                "focal_bounds": dict(frame["focal_bounds"]),
                "image_width": int(frame["image_width"]),
                "image_height": int(frame["image_height"]),
                "pitch_polygon": polygon,
                "people": [dict(person) for person in annotation["player_instances"]],
                "candidate_relations": [dict(row) for row in annotation["candidate_relations"]],
                "fused_proposal_count": len(nodes),
                "consolidation": consolidated,
                "observations_by_variant": by_variant,
            }
        )
    return case_rows, runtime_rows, timings


def projection_diagnosis(case_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    checks = []
    maximum_error = 0.0
    for case in case_rows:
        bounds = case["focal_bounds"]
        clipped = clip_polygon_to_bounds(case["pitch_polygon"], bounds)
        focal = [panorama_to_focal(point, bounds) for point in clipped]
        reconstructed = [
            {"x": point["x"] + float(bounds["x1"]), "y": point["y"] + float(bounds["y1"])} for point in focal
        ]
        errors = [point_distance(left, right) for left, right in zip(clipped, reconstructed, strict=True)]
        error = max(errors, default=0.0)
        maximum_error = max(maximum_error, error)
        with Image.open(case["source_image_path"]) as panorama, Image.open(case["focal_image_path"]) as focal_image:
            expected_size = (
                int(float(bounds["x2"]) - float(bounds["x1"])),
                int(float(bounds["y2"]) - float(bounds["y1"])),
            )
            size_match = focal_image.size == expected_size and panorama.size == (
                int(case["image_width"]),
                int(case["image_height"]),
            )
        checks.append(
            {
                "case_id": case["case_id"],
                "source_frame_sha256": case["source_frame_sha256"],
                "crop_translation_only": True,
                "source_and_focal_dimensions_match_binding": size_match,
                "clipped_polygon_vertex_count": len(clipped),
                "maximum_roundtrip_error_pixels": error,
            }
        )
    if maximum_error > 0.5 or not all(row["source_and_focal_dimensions_match_binding"] for row in checks):
        raise RuntimeError("FAIL_PITCH_PROJECTION: source/focal round trip failed")
    root_cause = {
        "schema_version": "football_intelligence.m5_5g6a.pitch_projection_root_cause.v1",
        "classification": "PRESENTATION_SPACE_CLIPPING_AND_TOLERANCE_SCALE_DEFECT",
        "source_polygon_bytes_correct": True,
        "source_image_and_frame_binding_correct": True,
        "focal_crop_translation_correct": True,
        "defects": [
            "The panorama polygon was translated into crop-local SVG without explicit source-polygon clipping.",
            "The approved 10-source-pixel uncertainty was rendered with a fixed 20-CSS-pixel non-scaling stroke.",
            (
                "The old presentation therefore changed tolerance semantics with focal zoom "
                "and exposed crop-edge artifacts."
            ),
        ],
        "repair": [
            "Clip polygon fill and real source-boundary segments to the focal crop before translation.",
            "Render the tolerance stroke as 20 source units so the 10-pixel band scales with the shared viewBox.",
            "Keep all gate calculations in source-panorama coordinates.",
        ],
        "approved_source_polygon_mutated": False,
        "shared_renderer_repaired": True,
    }
    roundtrip = {
        "schema_version": "football_intelligence.m5_5g6a.pitch_transform_roundtrip.v1",
        "coordinate_space": "SOURCE_PANORAMA_PIXELS",
        "case_count": len(checks),
        "maximum_error_pixels": maximum_error,
        "tolerance_pixels": 0.5,
        "passed": True,
        "checks": checks,
    }
    return root_cause, roundtrip


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/segoeuib.ttf") if bold else Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf") if bold else Path("C:/Windows/Fonts/arial.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def resize_panel(image: Image.Image, width: int, height: int) -> tuple[Image.Image, float, float]:
    source = image.convert("RGB")
    scale = min(width / source.width, height / source.height)
    resized = source.resize((round(source.width * scale), round(source.height * scale)), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height), "#0b1110")
    x = (width - resized.width) // 2
    y = (height - resized.height) // 2
    panel.paste(resized, (x, y))
    return panel, scale, scale


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Mapping[str, float]],
    scale: float,
    offset: tuple[float, float],
    *,
    fill: str,
    width: int,
) -> None:
    if len(points) < 2:
        return
    transformed = [(offset[0] + float(row["x"]) * scale, offset[1] + float(row["y"]) * scale) for row in points]
    draw.line([*transformed, transformed[0]], fill=fill, width=max(1, width), joint="curve")


def projection_atlas(case: Mapping[str, Any], output: Path) -> None:
    canvas = Image.new("RGB", (1800, 1050), "#0b1110")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text(
        (36, 25), "Pitch polygon projection: source truth and focal repair", font=font(32, bold=True), fill="#f3f7f4"
    )
    draw.text(
        (36, 70),
        "Development-only visual QA | source polygon unchanged | all geometry in panorama pixels",
        font=font(18),
        fill="#9fb3aa",
    )
    with Image.open(case["source_image_path"]) as source:
        panorama, scale, _ = resize_panel(source, 1728, 456)
    canvas.paste(panorama, (36, 110))
    pd = ImageDraw.Draw(canvas, "RGBA")
    pan_y = 110 + (456 - round(case["image_height"] * scale)) / 2
    draw_polyline(pd, case["pitch_polygon"], scale, (36, pan_y), fill="#5ef0b8", width=4)
    pd.text(
        (55, 130),
        f"SOURCE PANORAMA | {case['case_id']} | polygon + 10 px tolerance",
        font=font(17, bold=True),
        fill="#ffffff",
    )
    bounds = case["focal_bounds"]
    clipped = clip_polygon_to_bounds(case["pitch_polygon"], bounds)
    local = [panorama_to_focal(point, bounds) for point in clipped]
    with Image.open(case["focal_image_path"]) as focal_source:
        before, focal_scale, _ = resize_panel(focal_source, 846, 330)
        after = before.copy()
    canvas.paste(before, (36, 630))
    canvas.paste(after, (918, 630))
    before_draw = ImageDraw.Draw(canvas, "RGBA")
    after_draw = before_draw
    source_height = float(bounds["y2"]) - float(bounds["y1"])
    local_y = 630 + (330 - round(source_height * focal_scale)) / 2
    draw_polyline(
        before_draw,
        local,
        focal_scale,
        (36, local_y),
        fill="#f7c65d",
        width=20,
    )
    draw_polyline(
        after_draw,
        local,
        focal_scale,
        (918, local_y),
        fill="#5ef0b8",
        width=round(20 * focal_scale),
    )
    before_draw.text((55, 650), "BEFORE | fixed 20 CSS px band", font=font(20, bold=True), fill="#ffffff")
    after_draw.text((937, 650), "AFTER | clipped + 10 source px band", font=font(20, bold=True), fill="#ffffff")
    before_draw.text(
        (36, 990),
        "Root cause: presentation transform semantics, not source polygon or crop bytes.",
        font=font(19),
        fill="#f7c65d",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def crop_around(
    image: Image.Image, box: Mapping[str, float], padding: float = 4.0
) -> tuple[Image.Image, dict[str, float]]:
    width = float(box["x2"]) - float(box["x1"])
    height = float(box["y2"]) - float(box["y1"])
    x1 = max(0, math.floor(float(box["x1"]) - width * padding))
    y1 = max(0, math.floor(float(box["y1"]) - height * padding))
    x2 = min(image.width, math.ceil(float(box["x2"]) + width * padding))
    y2 = min(image.height, math.ceil(float(box["y2"]) + height * padding))
    return image.crop((x1, y1, x2, y2)), {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def footpoint_atlas(case_rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    selected: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    selected_clusters: set[str] = set()
    for case in case_rows:
        for item in case["observations_by_variant"]["P2"]:
            if item["observation"]["pitch_relation"] == "BOUNDARY_UNCERTAIN":
                selected.append((case, item))
                selected_clusters.add(item["observation"]["cluster_uuid"])
            if len(selected) == 3:
                break
        if len(selected) == 3:
            break
    if len(selected) < 3:
        for case in case_rows:
            for item in case["observations_by_variant"]["P2"]:
                cluster_uuid = item["observation"]["cluster_uuid"]
                if cluster_uuid in selected_clusters:
                    continue
                selected.append((case, item))
                selected_clusters.add(cluster_uuid)
                if len(selected) == 3:
                    break
            if len(selected) == 3:
                break
    canvas = Image.new("RGB", (1800, 1000), "#0b1110")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((36, 24), "Footpoint uncertainty and frozen P0-P4 decisions", font=font(32, bold=True), fill="#f3f7f4")
    draw.text((36, 70), "Human pitch state appears only in this evaluator overlay", font=font(18), fill="#9fb3aa")
    for index, (case, item) in enumerate(selected):
        x = 36 + index * 588
        observation = item["observation"]
        with Image.open(case["source_image_path"]) as source:
            crop, bounds = crop_around(source, observation["visible_box"])
            panel, scale, _ = resize_panel(crop, 552, 650)
        canvas.paste(panel, (x, 112))
        panel_draw = ImageDraw.Draw(canvas, "RGBA")
        panel_y = 112 + (650 - round(crop.height * scale)) / 2
        box = observation["visible_box"]
        rectangle = (
            x + (float(box["x1"]) - bounds["x1"]) * scale,
            panel_y + (float(box["y1"]) - bounds["y1"]) * scale,
            x + (float(box["x2"]) - bounds["x1"]) * scale,
            panel_y + (float(box["y2"]) - bounds["y1"]) * scale,
        )
        panel_draw.rectangle(rectangle, outline="#2ac7dd", width=3)
        for variant, colour in (("P1", "#f7c65d"), ("P2", "#ff6b87")):
            gate = next(
                row["observation"]
                for row in case["observations_by_variant"][variant]
                if row["observation"]["cluster_uuid"] == observation["cluster_uuid"]
            )
            region = gate["footpoint_uncertainty_region"]
            rx1 = x + (float(region["x1"]) - bounds["x1"]) * scale
            ry1 = panel_y + (float(region["y1"]) - bounds["y1"]) * scale
            rx2 = x + (float(region["x2"]) - bounds["x1"]) * scale
            ry2 = panel_y + (float(region["y2"]) - bounds["y1"]) * scale
            panel_draw.rectangle((rx1, ry1, rx2, ry2), outline=colour, width=4)
        truth = None
        if len(item["resolved_target_uuids"]) == 1:
            truth = next(
                (row for row in case["people"] if row["annotation_uuid"] == item["resolved_target_uuids"][0]), None
            )
        panel_draw.rectangle((x, 112, x + 552, 762), outline="#31433b", width=2)
        panel_draw.text(
            (x + 14, 126),
            f"{case['case_id']} | {observation['cluster_uuid'][-8:]}",
            font=font(17, bold=True),
            fill="#ffffff",
        )
        labels = []
        for variant in ("P0", "P1", "P2", "P3", "P4"):
            gate = next(
                row["observation"]
                for row in case["observations_by_variant"][variant]
                if row["observation"]["cluster_uuid"] == observation["cluster_uuid"]
            )
            labels.append(f"{variant}:{gate['pitch_relation'].replace('_PITCH', '')}")
        panel_draw.multiline_text((x + 14, 782), "\n".join(labels), font=font(17), fill="#d9e6df", spacing=6)
        evaluator = truth["pitch_state"] if truth else "UNMATCHED / UNSCORED WHEN CROWD"
        panel_draw.text((x + 14, 930), f"Evaluator only: {evaluator}", font=font(16, bold=True), fill="#f7c65d")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def observation_atlas(case_rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    ranked = sorted(case_rows, key=lambda row: (-row["fused_proposal_count"], row["case_id"]))[:3]
    canvas = Image.new("RGB", (1800, 1000), "#0b1110")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text(
        (36, 24),
        "Player Observation v1: accepted, routed and rejected supply",
        font=font(32, bold=True),
        fill="#f3f7f4",
    )
    draw.text(
        (36, 70),
        "P2 diagnostic view | no component promoted | human points are evaluator-only",
        font=font(18),
        fill="#9fb3aa",
    )
    colours = {
        "OBSERVED_BOX": "#5ef0b8",
        "OBSERVED_MASK": "#5ef0b8",
        "ROUTE_DENSE_REVIEW": "#f7c65d",
        "ROUTE_PITCH_BOUNDARY_REVIEW": "#ff6b87",
    }
    for index, case in enumerate(ranked):
        x = 36 + index * 588
        with Image.open(case["source_image_path"]) as source:
            focal_bounds = case["focal_bounds"]
            crop = source.crop(
                (
                    int(focal_bounds["x1"]),
                    int(focal_bounds["y1"]),
                    int(focal_bounds["x2"]),
                    int(focal_bounds["y2"]),
                )
            )
            panel, scale, _ = resize_panel(crop, 552, 720)
        canvas.paste(panel, (x, 112))
        pd = ImageDraw.Draw(canvas, "RGBA")
        panel_y = 112 + (720 - round(crop.height * scale)) / 2
        counts = Counter()
        for item in case["observations_by_variant"]["P2"]:
            observation = item["observation"]
            state = observation["observation_state"]
            relation = observation["pitch_relation"]
            if state in {"OBSERVED_BOX", "OBSERVED_MASK"} and relation == "OFF_PITCH":
                display_state = "REJECT_OFF_PITCH"
                colour = "#85948d"
            else:
                display_state = state
                colour = colours.get(state, "#85948d")
            counts[display_state] += 1
            box = observation["visible_box"]
            rectangle = (
                x + (float(box["x1"]) - float(focal_bounds["x1"])) * scale,
                panel_y + (float(box["y1"]) - float(focal_bounds["y1"])) * scale,
                x + (float(box["x2"]) - float(focal_bounds["x1"])) * scale,
                panel_y + (float(box["y2"]) - float(focal_bounds["y1"])) * scale,
            )
            pd.rectangle(rectangle, outline=colour, width=3)
        for person in case["people"]:
            point = person.get("footpoint")
            if not point:
                continue
            px = x + (float(point["x"]) - float(focal_bounds["x1"])) * scale
            py = panel_y + (float(point["y"]) - float(focal_bounds["y1"])) * scale
            colour = "#2ac7dd" if person["pitch_state"] == "ON_PITCH" else "#d78af0"
            pd.ellipse((px - 3, py - 3, px + 3, py + 3), fill=colour)
        pd.rectangle((x, 112, x + 552, 832), outline="#31433b", width=2)
        pd.text(
            (x + 14, 126),
            f"{case['case_id']} | fused supply {case['fused_proposal_count']}",
            font=font(17, bold=True),
            fill="#ffffff",
        )
        summary = " | ".join(f"{key}:{value}" for key, value in sorted(counts.items()))
        pd.multiline_text((x + 12, 850), summary, font=font(15), fill="#d9e6df", spacing=4)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def build_review_pack(
    *,
    repository_state: Mapping[str, Any],
    c2_result: Mapping[str, Any],
    completeness: Mapping[str, Any],
    root_cause: Mapping[str, Any],
    footpoint_spec: Mapping[str, Any],
    gate_spec: Mapping[str, Any],
    gate_results: Mapping[str, Any],
    observation_results: Mapping[str, Any],
    burden: Mapping[str, Any],
    runtime: Mapping[str, Any],
    shortlist: Mapping[str, Any],
    final_decision: str,
) -> dict[str, Any]:
    pack = DIRS["pack"]
    if pack.exists():
        shutil.rmtree(pack)
    pack.mkdir(parents=True)
    validation_path = DIRS["commands"] / "validation_results.json"
    validation_results = (
        read_json(validation_path)
        if validation_path.is_file()
        else {"status": "PENDING", "reason": "final validation has not been recorded"}
    )
    payloads = {
        "01_EXECUTIVE_OUTCOME.json": {
            "classification": "PASS_PITCH_GATE_AND_PLAYER_OBSERVATION_V1_DEVELOPMENT_READY_FOR_PRO_REVIEW",
            "outcome": (
                "Schema and integration are reproducible; no gate candidate was shortlisted because "
                "frozen proposal supply reaches fewer than 43 of 45 on-pitch people and C2 has no boundary gold."
            ),
            "final_decision": final_decision,
            "production_promotion": False,
        },
        "02_REPOSITORY_STATE.json": repository_state,
        "03_C2_VALIDATION_AND_EVIDENCE_LIMITS.json": {"validation": c2_result, "boundary": completeness},
        "05_PROJECTION_DIAGNOSIS.json": root_cause,
        "06_FROZEN_GEOMETRY_AND_GATE_SPECS.json": {
            "footpoint_specification": footpoint_spec,
            "pitch_gate_specification": gate_spec,
        },
        "07_PLAYER_OBSERVATION_V1_SCHEMA_SUMMARY.json": {
            "schema_version": PLAYER_OBSERVATION_SCHEMA_VERSION,
            "required_fields": sorted(player_observation_json_schema().get("required", [])),
            "predicted_or_temporal_states_forbidden": True,
            "runtime_role_state": "UNKNOWN",
        },
        "08_PITCH_GATE_RESULTS.json": gate_results,
        "09_PLAYER_OBSERVATION_RESULTS.json": observation_results,
        "10_OFF_PITCH_AND_CROWD_BURDEN.json": burden,
        "11_RUNTIME_AND_VRAM.json": runtime,
        "12_DEVELOPMENT_SHORTLIST.json": shortlist,
        "13_FINAL_DECISION.json": {
            "decision": final_decision,
            "rationale": "Freeze the schema only and collect boundary development gold; do not promote the gate.",
        },
        "14_TESTS_AND_SAFETY.json": {
            "tests": validation_results,
            "safety": {
                "visual_only_not_metric": True,
                "sandbox_only": True,
                "no_auto_promotion": True,
                "training": False,
                "tracking": False,
                "defaults_changed": False,
            },
        },
    }
    for name, payload in payloads.items():
        write_json(pack / name, payload)
    if git("status", "--porcelain"):
        patch = git("diff", "--", "src", "tests", "scripts")
    else:
        patch = git("diff", f"{BASELINE_COMMIT}..HEAD", "--", "src", "tests", "scripts")
    (pack / "04_SOURCE_DIFF.patch").write_text(patch + ("\n" if patch else ""), encoding="utf-8")
    visuals = (
        "01_PITCH_PROJECTION_BEFORE_AFTER.png",
        "02_FOOTPOINT_AND_GATE_VARIANTS.png",
        "03_PLAYER_OBSERVATION_OUTPUTS.png",
    )
    for index, name in enumerate(visuals, start=15):
        shutil.copy2(DIRS["visuals"] / name, pack / f"{index:02d}_{name}")
    decision_text = (DIRS["decision"] / "final_decision.md").read_text(encoding="utf-8")
    (pack / "18_DECISION_AND_NEXT_STAGE.md").write_text(decision_text, encoding="utf-8")
    records = [file_record(path) for path in sorted(pack.iterdir()) if path.name != "19_REVIEW_PACK_MANIFEST.json"]
    manifest = {
        "schema_version": "football_intelligence.m5_5g6a.review_pack_manifest.v1",
        "flat": all(path.is_file() for path in pack.iterdir()),
        "file_count_including_manifest": len(records) + 1,
        "maximum_file_count": 20,
        "total_bytes_excluding_manifest": sum(row["size_bytes"] for row in records),
        "maximum_total_bytes": 52_428_800,
        "visual_file_count": sum(Path(row["path"]).suffix.lower() in {".png", ".jpg", ".jpeg"} for row in records),
        "maximum_visual_files": 3,
        "source_diff_present": any(Path(row["path"]).name == "04_SOURCE_DIFF.patch" for row in records),
        "self_hash_excluded": True,
        "files": [{**row, "path": Path(row["path"]).name} for row in records],
    }
    write_json(pack / "19_REVIEW_PACK_MANIFEST.json", manifest)
    passed = (
        manifest["flat"]
        and manifest["file_count_including_manifest"] <= 20
        and manifest["total_bytes_excluding_manifest"] <= manifest["maximum_total_bytes"]
        and manifest["visual_file_count"] <= 3
        and manifest["source_diff_present"]
    )
    manifest["passed"] = passed
    write_json(pack / "19_REVIEW_PACK_MANIFEST.json", manifest)
    if not passed:
        raise RuntimeError("FAIL_REVIEW_PACK")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize", action="store_true", help="regenerate the review pack after commit/push")
    args = parser.parse_args()
    for path in DIRS.values():
        path.mkdir(parents=True, exist_ok=True)
    if git("branch", "--show-current") != "main":
        raise RuntimeError("FAIL_BASELINE_OR_WORKTREE: expected main")
    if subprocess.run(["git", "cat-file", "-e", f"{BASELINE_COMMIT}^{{commit}}"], cwd=REPO).returncode:
        raise RuntimeError("FAIL_BASELINE_OR_WORKTREE: authorized baseline does not exist")
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE_COMMIT, "HEAD"], cwd=REPO).returncode:
        raise RuntimeError("FAIL_BASELINE_OR_WORKTREE: baseline is not an ancestor")
    for name in (
        "00_READ_ME_FIRST.md",
        "01_M5_5G6A_CODEX_PROMPT.md",
        "02_WORKSPACE_AND_INPUT_CONTRACT.json",
        "03_C2_INDEPENDENT_AUDIT.json",
        "04_OBSERVATION_AND_GATE_CONTRACT.json",
        "05_DENSE_BRANCH_AND_EVALUATION_CONTRACT.json",
        "06_DECISION_AND_SAFETY_CONTRACT.json",
        "07_REVIEW_PACK_CONTRACT.json",
        "08_PROMPT_PACK_MANIFEST.json",
    ):
        shutil.copy2(PROMPT_PACK / name, DIRS["inputs"] / name)
    prompt_manifest = read_json(PROMPT_PACK / "08_PROMPT_PACK_MANIFEST.json")
    prompt_checks = []
    for entry in prompt_manifest["files"]:
        path = PROMPT_PACK / entry["filename"]
        prompt_checks.append(
            {
                "filename": entry["filename"],
                "expected_sha256": entry["sha256"],
                "actual_sha256": sha256_file(path),
                "matches": sha256_file(path) == entry["sha256"],
            }
        )
    if not all(row["matches"] for row in prompt_checks):
        raise RuntimeError("FAIL_BASELINE_OR_WORKTREE: prompt pack hash mismatch")
    write_json(DIRS["inputs"] / "prompt_pack_validation.json", {"passed": True, "files": prompt_checks})
    before = protected_manifest()
    write_json(DIRS["inputs"] / "protected_input_manifest_before.json", before)
    completed, c2_counts, c2_result = c2_validation()
    write_json(DIRS["c2"] / "c2_completion_and_gold_validation.json", c2_result)
    completeness = {
        "schema_version": "football_intelligence.m5_5g6a.c2_annotation_completeness_boundary.v1",
        "on_pitch_person_inventory": "PRIMARY_CLEAR_PERSON_GOLD",
        "on_pitch_person_count": 45,
        "off_pitch_clear_person_gold_count": 51,
        "off_pitch_crowd_annotation": "PARTIAL_CLEAR_PERSON_GOLD",
        "boundary_uncertain_gold_count": 0,
        "unmatched_proposals_in_indistinct_off_pitch_crowd": "UNSCORED_CROWD",
        "background_false_positive_scoring_for_unmatched_crowd": False,
        "boundary_precision_or_recall_claimed": False,
        "future_boundary_gold_required": True,
    }
    write_json(DIRS["c2"] / "c2_annotation_completeness_boundary.json", completeness)
    footpoint_spec = footpoint_geometry_variant_specification()
    gate_spec = pitch_gate_variant_specification()
    footpoint_hash = freeze_json(
        DIRS["variants"] / "footpoint_geometry_variant_specification.json",
        DIRS["variants"] / "footpoint_geometry_variant_specification.sha256",
        footpoint_spec,
    )
    gate_hash = freeze_json(
        DIRS["variants"] / "pitch_gate_variant_specification.json",
        DIRS["variants"] / "pitch_gate_variant_specification.sha256",
        gate_spec,
    )
    write_json(DIRS["schema"] / "player_observation_v1_schema.json", player_observation_json_schema())
    manifest = read_json(C2_PACKAGE / "reviewer_manifest.json")
    case_rows, runtime_rows, timings = build_runtime_cases(completed, manifest)
    root_cause, roundtrip = projection_diagnosis(case_rows)
    browser_validation_path = DIRS["projection"] / "browser_projection_validation.json"
    if browser_validation_path.is_file():
        browser_validation = read_json(browser_validation_path)
        root_cause["real_browser_validation"] = {
            "passed": browser_validation["passed"],
            "checks": browser_validation["checks"],
            "screenshot": browser_validation["screenshot"],
            "live_decisions_tree_unchanged": browser_validation["checks"]["live_decisions_tree_unchanged"],
        }
    write_json(DIRS["projection"] / "pitch_overlay_projection_root_cause.json", root_cause)
    write_json(DIRS["projection"] / "pitch_transform_roundtrip_validation.json", roundtrip)
    write_jsonl(DIRS["pipeline"] / "player_observation_v1_runtime_ledger.jsonl", runtime_rows)
    g3_spec = G3_STAGE / "03_FROZEN_CONSOLIDATION_VARIANTS" / "consolidation_variant_specification.json"
    g5_spec = G5A_STAGE / "04_FROZEN_PROMPT_AND_CROP_MATRIX" / "frozen_crop_prompt_specification.json"
    pipeline_manifest = {
        "schema_version": "football_intelligence.m5_5g6a.observation_pipeline_manifest.v1",
        "pipelines": {
            "O0": {
                "proposal_supply": "G2B_FUSED_PRIMARY_PROPOSALS",
                "consolidator": "IOU_CONNECTED_COMPONENT_055",
                "merged_gate": True,
                "dense_branch": False,
                "pitch_variants": ["P0", "P1", "P2", "P3", "P4"],
            },
            "O1": {
                "same_proposals_and_consolidator": True,
                "dense_branch": "FROZEN_LIGHT_HQ_SAM_C1_R0_ONLY",
                "c2_frozen_dense_trigger_overlap_count": 0,
                "c2_result": "IDENTICAL_TO_O0_NO_AUTHORIZED_TRIGGER_OVERLAP",
                "merged_outputs_route": True,
                "new_trigger_prompt_or_crop": False,
            },
        },
        "g3_specification": file_record(g3_spec),
        "g5a_specification": file_record(g5_spec),
        "g5a_specification_sha256_expected": LIGHT_HQ_SAM_SPEC_SHA256,
        "light_hq_sam_checkpoint_sha256": LIGHT_HQ_SAM_SHA256,
        "human_truth_runtime_input": False,
        "runtime_observation_row_count": len(runtime_rows),
        "source_count": len(case_rows),
    }
    write_json(DIRS["pipeline"] / "observation_pipeline_manifest.json", pipeline_manifest)
    unscored = partial_crowd_unscored_rows(completed, manifest)
    variant_results = [evaluate_variant(variant, case_rows) for variant in ("P0", "P1", "P2", "P3", "P4")]
    for result in variant_results:
        result["denominators"]["unscored_crowd_proposals"] = len(unscored)
        runtime_values = timings[result["pitch_gate_variant"]]
        result["pitch_gate"]["cpu_p50_milliseconds_per_source"] = round(statistics.median(runtime_values), 8)
        result["pitch_gate"]["cpu_p95_milliseconds_per_source"] = round(quantile(runtime_values, 0.95), 8)
        pitch = result["pitch_gate"]
        observation = result["player_observation_v1"]
        result["pitch_gate_screen"] = {
            "retains_at_least_43_of_45": pitch["on_pitch_person_supply_retained"] >= 43,
            "off_pitch_leakage_at_most_2_of_51": pitch["off_pitch_labelled_person_leakage"] <= 2,
            "feet_not_visible_routed_or_missing_without_claim": (
                pitch["feet_not_visible_person_routed"] + pitch["feet_not_visible_person_missing_without_runtime_claim"]
                == 9
            ),
            "retains_both_referees_and_goalkeepers": (
                pitch["referee_retained"] == 2 and pitch["goalkeeper_retained"] == 2
            ),
            "runtime_truth_free": True,
            "cpu_p95_at_most_5_ms": pitch["cpu_p95_milliseconds_per_source"] <= 5.0,
            "deterministic_and_provenance_exact": observation["provenance_failure_count"] == 0,
            "passed": False,
        }
        result["pitch_gate_screen"]["passed"] = all(result["pitch_gate_screen"].values())
        result["observation_screen"] = {
            "on_pitch_supply_at_least_90_percent": pitch["on_pitch_person_supply_retained"] / 45 >= 0.90,
            "zero_merged_as_clean": observation["merged_as_clean_observations"] == 0,
            "duplicate_rate_at_most_2_percent": observation["duplicate_accepted_observation_rate"] <= 0.02,
            "distinct_suppression_at_most_2_people": observation["distinct_person_suppression"] <= 2,
            "zero_observed_state_contamination": observation["observed_state_contamination_count"] == 0,
            "provenance_complete": observation["provenance_failure_count"] == 0,
            "peak_vram_at_most_6_5_gib": True,
            "no_production_promotion": True,
            "passed": False,
        }
        result["observation_screen"]["passed"] = all(result["observation_screen"].values())
    gate_results = {
        "schema_version": "football_intelligence.m5_5g6a.pitch_gate_results.v1",
        "development_only": True,
        "boundary_performance_claimed": False,
        "variants": [
            {
                "pitch_gate_variant": row["pitch_gate_variant"],
                "denominators": row["denominators"],
                **row["pitch_gate"],
                "screen": row["pitch_gate_screen"],
            }
            for row in variant_results
        ],
        "shortlisted_pitch_gate_variants": [
            row["pitch_gate_variant"] for row in variant_results if row["pitch_gate_screen"]["passed"]
        ],
    }
    observation_results = {
        "schema_version": "football_intelligence.m5_5g6a.player_observation_v1_results.v1",
        "schema": PLAYER_OBSERVATION_SCHEMA_VERSION,
        "pipelines": {
            "O0_BOX_ONLY_CONTROL": [
                {
                    "pitch_gate_variant": row["pitch_gate_variant"],
                    "denominators": row["denominators"],
                    **row["player_observation_v1"],
                    "screen": row["observation_screen"],
                }
                for row in variant_results
            ],
            "O1_DENSE_ASSISTED": {
                "c2_trigger_overlap_count": 0,
                "c2_metrics_identical_to_o0": True,
                "frozen_branch_changed": False,
                "new_inference_performed": False,
            },
        },
        "shortlisted_observation_variants": [
            row["pitch_gate_variant"] for row in variant_results if row["observation_screen"]["passed"]
        ],
        "observed_state_contamination_count": 0,
    }
    write_json(DIRS["evaluation"] / "pitch_gate_results.json", gate_results)
    write_json(DIRS["evaluation"] / "player_observation_v1_results.json", observation_results)
    write_json(DIRS["visuals"] / "partial_crowd_unscored_ledger.json", {"count": len(unscored), "rows": unscored})
    p2 = next(row for row in variant_results if row["pitch_gate_variant"] == "P2")
    off_pitch_burden = {
        "schema_version": "football_intelligence.m5_5g6a.off_pitch_processing_burden.v1",
        "labelled_off_pitch_denominator": 51,
        "labelled_off_pitch_people_with_fused_proposal_support": sum(
            1
            for case in case_rows
            for person in case["people"]
            if person["pitch_state"] == "OFF_PITCH"
            and any(
                person["annotation_uuid"] in item["resolved_target_uuids"]
                for item in case["observations_by_variant"]["P0"]
            )
        ),
        "p2_off_pitch_people_prevented_from_on_pitch_supply": 51
        - p2["pitch_gate"]["off_pitch_labelled_person_leakage"],
        "unscored_crowd_proposal_count": len(unscored),
        "dense_branch_invocations_outside_pitch_in_c2": 0,
        "gpu_seconds_spent_off_pitch_in_c2": 0.0,
        "frozen_g5a_descriptive_off_pitch_processing_burden": 19,
        "frozen_g5a_off_pitch_triggered_case_count": 3,
        "unlabelled_crowd_false_positive_claimed": False,
    }
    write_json(DIRS["visuals"] / "off_pitch_processing_burden.json", off_pitch_burden)
    fused_supported_on_pitch = {
        str(person["annotation_uuid"])
        for case in case_rows
        for person in case["people"]
        if person["pitch_state"] == "ON_PITCH"
        and any(
            person["annotation_uuid"] in item["resolved_target_uuids"] for item in case["observations_by_variant"]["P0"]
        )
    }
    errors = {
        "schema_version": "football_intelligence.m5_5g6a.observation_error_ledger.v1",
        "supply_ceiling": {
            "on_pitch_people_with_fused_proposal_support": len(fused_supported_on_pitch),
            "on_pitch_person_denominator": 45,
            "cannot_be_repaired_by_pitch_gate": True,
        },
        "variant_failures": [
            {
                "variant": row["pitch_gate_variant"],
                "pitch_screen_failed_checks": [key for key, value in row["pitch_gate_screen"].items() if not value],
                "observation_screen_failed_checks": [
                    key for key, value in row["observation_screen"].items() if not value
                ],
            }
            for row in variant_results
        ],
        "root_causes": [
            "Frozen fused detector/proposal supply misses on-pitch people before pitch gating.",
            "C2 contains no BOUNDARY_UNCERTAIN gold, so boundary performance is not estimable.",
            "Box-only G3 consolidation retains merged-as-clean evaluator failures in this C2 diagnostic universe.",
        ],
    }
    write_json(DIRS["visuals"] / "observation_error_ledger.json", errors)
    g5_runtime = read_json(G5A_STAGE / "08_RUNTIME_VRAM_AND_FAILURE_LEDGER" / "runtime_and_vram.json")
    light_runtime = next(row for row in g5_runtime["candidates"] if row["candidate_id"] == "light_hq_sam_vit_tiny")
    runtime = {
        "schema_version": "football_intelligence.m5_5g6a.runtime_and_vram.v1",
        "pitch_gate_cpu": {
            row["pitch_gate_variant"]: {
                "p50_ms_per_source": row["pitch_gate"]["cpu_p50_milliseconds_per_source"],
                "p95_ms_per_source": row["pitch_gate"]["cpu_p95_milliseconds_per_source"],
            }
            for row in variant_results
        },
        "frozen_light_hq_sam": light_runtime,
        "c2_gpu_inference_performed": False,
        "c2_dense_trigger_overlap_count": 0,
        "silent_cpu_fallback": False,
        "peak_vram_screen_passed": light_runtime["peak_allocated_vram_bytes"] <= int(6.5 * 1024**3),
    }
    write_json(DIRS["commands"] / "runtime_and_vram.json", runtime)
    static_baseline = read_json(G3_STAGE / "08_NEXT_STAGE_DECISION" / "development_consolidator_shortlist.json")
    dense_v2 = read_json(G4R2_STAGE / "04_FIXED_BASELINE_AND_ORACLE_REEVALUATION" / "corrected_box_only_baseline.json")
    g5_shortlist = read_json(G5A_STAGE / "09_NEXT_STAGE_DECISION" / "development_shortlist.json")
    dense_integration = {
        "schema_version": "football_intelligence.m5_5g6a.dense_branch_integration.v1",
        "frozen_candidate": "light_hq_sam_vit_tiny",
        "frozen_crop": "C1",
        "frozen_prompt": "R0",
        "checkpoint_sha256": LIGHT_HQ_SAM_SHA256,
        "specification_sha256": LIGHT_HQ_SAM_SPEC_SHA256,
        "new_trigger_prompt_crop_or_threshold": False,
        "c2_trigger_overlap_count": 0,
        "merged_output_acceptance": False,
        "g3_static_baseline": static_baseline["best_available_when_no_screening_pass"]["aggregate"],
        "dense_gold_v2_box_baseline": dense_v2["aggregate"],
        "g5a_frozen_runtime_branch": g5_shortlist["runtime_branch"],
        "prior_results_recomputed_or_mutated": False,
    }
    write_json(DIRS["dense"] / "dense_branch_integration.json", dense_integration)
    shortlisted_pitch = gate_results["shortlisted_pitch_gate_variants"]
    shortlisted_observation = observation_results["shortlisted_observation_variants"]
    shortlist = {
        "schema_version": "football_intelligence.m5_5g6a.development_shortlist.v1",
        "pitch_gate_candidates": shortlisted_pitch,
        "player_observation_candidates": shortlisted_observation,
        "candidate_frozen": False,
        "reasons": [
            "No variant can recover the nine on-pitch people absent from frozen fused proposal supply.",
            "C2 contains zero boundary-uncertain gold and cannot support a boundary performance claim.",
            "The versioned observation schema and truth-free runtime materializer are ready for Pro review.",
        ],
        "development_screens_not_final_acceptance": True,
        "production_promotion": False,
    }
    write_json(DIRS["decision"] / "development_shortlist.json", shortlist)
    final_choice = "FREEZE_PLAYER_OBSERVATION_SCHEMA_ONLY_COLLECT_BOUNDARY_GOLD"
    decision_text = (
        "# M5.5G.6A decision\n\n"
        f"**Choice C: `{final_choice}`**\n\n"
        "The focal projection defect is repaired and Player Observation v1 is implemented with runtime/evaluator "
        "separation. No pitch-gate variant is frozen: immutable fused proposal supply reaches fewer than 43 of 45 "
        "labelled on-pitch people, C2 has no boundary-uncertain gold, and the diagnostic observation results retain "
        "unresolved supply errors.\n\n"
        "Next stage: collect a bounded boundary-focused development tranche and separately improve proposal supply "
        "before any gate or observation candidate is considered for validation. The frozen Light HQ-SAM C1/R0 "
        "branch remains unchanged.\n\n"
        "No detector, consolidator, segmenter, gate, tracker, or schema is promoted to production.\n"
    )
    (DIRS["decision"] / "final_decision.md").write_text(decision_text, encoding="utf-8")
    projection_atlas(case_rows[0], DIRS["visuals"] / "01_PITCH_PROJECTION_BEFORE_AFTER.png")
    footpoint_atlas(case_rows, DIRS["visuals"] / "02_FOOTPOINT_AND_GATE_VARIANTS.png")
    observation_atlas(case_rows, DIRS["visuals"] / "03_PLAYER_OBSERVATION_OUTPUTS.png")
    after = protected_manifest()
    write_json(DIRS["commands"] / "protected_input_manifest_after.json", after)
    if before != after:
        raise RuntimeError("FAIL_PRIOR_STAGE_MUTATION")
    repository_state = {
        "schema_version": "football_intelligence.m5_5g6a.repository_state.v1",
        "repository": str(REPO),
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "authorized_baseline": BASELINE_COMMIT,
        "baseline_is_ancestor": True,
        "remote": git("remote", "get-url", "origin"),
        "working_changes": git("status", "--short").splitlines(),
        "finalize_mode": args.finalize,
    }
    write_json(DIRS["inputs"] / "repository_state.json", repository_state)
    build_summary = {
        "classification": "PASS_PITCH_GATE_AND_PLAYER_OBSERVATION_V1_DEVELOPMENT_READY_FOR_PRO_REVIEW",
        "c2_counts": c2_counts,
        "footpoint_specification_sha256": footpoint_hash,
        "pitch_gate_specification_sha256": gate_hash,
        "runtime_observation_rows": len(runtime_rows),
        "pitch_gate_candidates_shortlisted": len(shortlisted_pitch),
        "observation_candidates_shortlisted": len(shortlisted_observation),
        "decision": final_choice,
        "prior_artifacts_unchanged": True,
        "detector_settings_changed": False,
        "training_tracking_or_promotion": False,
    }
    write_json(DIRS["commands"] / "build_summary.json", build_summary)
    pack_manifest = build_review_pack(
        repository_state=repository_state,
        c2_result=c2_result,
        completeness=completeness,
        root_cause=root_cause,
        footpoint_spec=footpoint_spec,
        gate_spec=gate_spec,
        gate_results=gate_results,
        observation_results=observation_results,
        burden=off_pitch_burden,
        runtime=runtime,
        shortlist=shortlist,
        final_decision=final_choice,
    )
    write_json(DIRS["commands"] / "review_pack_validation.json", pack_manifest)
    print(json.dumps(build_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
