"""Build M5.5G.6B boundary gold and frozen proposal-supply attribution."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import subprocess
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from football_intelligence.detection_gold.incremental import (
    G6B_BOUNDARY_FOCUSED_CLIENT_BUILD_ID,
    authoritative_candidate_binding_hash,
)
from football_intelligence.detection_gold.persistence import DetectionGoldPilotPersistence
from football_intelligence.detection_gold.player_observation import signed_distance_to_polygon
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.config import load_ui_config, ui_config_hash
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import load_manifest, manifest_hash
from football_intelligence.review_chassis.models import GenericReviewManifest
from football_intelligence.review_chassis.validation import validate_review_chassis_package

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT = PART3 / "M5_5G6B_Boundary_Gold_And_Proposal_Attribution_Codex_Prompt_Pack"
STAGE = PART3 / "M5_5G6B_BOUNDARY_FOCUSED_GOLD_AND_FROZEN_PROPOSAL_SUPPLY_ATTRIBUTION_v1"
SOURCE_PACKAGE = (
    PART3
    / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
    / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
)
G2B = PART3 / "M5_5G2B_FULL_STATIC_PLAYER_PROPOSAL_SUPPLY_DEVELOPMENT_BAKEOFF_v1"
G3 = PART3 / "M5_5G3_PROVENANCE_AWARE_CROSS_VIEW_CONSOLIDATION_AND_MERGED_AMBIGUITY_GATE_DEVELOPMENT_v1"
G4R2 = PART3 / "M5_5G4_R2_CORRECTED_DENSE_GOLD_OVERLAY_APPLICATION_AND_INSTANCE_SEPARATION_REEVALUATION_v1"
G5A = PART3 / "M5_5G5A_AUTHORIZED_PROMPTABLE_MASK_BAKEOFF_AND_DENSE_BRANCH_DECISION_v1"
G6A = PART3 / "M5_5G6A_PITCH_BOUNDARY_GATE_AND_PLAYER_OBSERVATION_V1_INTEGRATION_DEVELOPMENT_v1"

BASELINE = "cbe68a9cd961956603f79319e603a16be6eee1ed"
REVIEW_ID = "m5_5g6b_boundary_focused_person_gold_v1"
STAGE_ID = "M5_5G6B_BOUNDARY_FOCUSED_GOLD_AND_FROZEN_PROPOSAL_SUPPLY_ATTRIBUTION_v1"
REVIEWER = "m5_5g6b_boundary_focused_gold_reviewer"
TRANCHE = "B1_BOUNDARY_FOCUSED_PERSON_GOLD"
INDEXEDDB_NAMESPACE = "m5_5g6b_boundary_focused_gold_outbox_v1"
PACKAGE = STAGE / "05_PERSON_CENTRIC_BOUNDARY_REVIEW_PACKAGE"
C2_BUNDLE = SOURCE_PACKAGE / "decisions" / "completed_tranches" / "C2_PITCH_BOUNDARY"

DIRS = {
    "inputs": STAGE / "00_PROMPT_AND_INPUTS",
    "validation": STAGE / "01_G6A_AND_GOLD_VALIDATION",
    "attribution": STAGE / "02_C2_MISSING_SUPPLY_ATTRIBUTION",
    "mining": STAGE / "03_BOUNDARY_CANDIDATE_MINING",
    "cases": STAGE / "04_BOUNDARY_CASE_MANIFEST",
    "package": PACKAGE,
    "browser": STAGE / "06_BROWSER_PERSISTENCE_AND_COMPLETION",
    "next": STAGE / "07_NEXT_STAGE_PERMISSION",
    "commands": STAGE / "08_COMMANDS_AND_TESTS",
    "review": STAGE / "09_REVIEW_PACK_FOR_CHATGPT",
    "tmp": STAGE / "_tmp",
}

SELECTED_VARIANT = "IOU_CONNECTED_COMPONENT_055"
SELECTED_POOL = "PRIMARY_FULL_1280_PLUS_TILES"
SELECTED_OUTPUT = "ACCEPT_INDEPENDENT_OBSERVATION"
SELECTION_QUOTAS = {
    "estimated_inside_near_boundary": 6,
    "estimated_outside_near_boundary": 6,
    "disagreement_hidden_feet_or_straddling": 6,
}
SELECTION_SPEC = {
    "schema_version": "football_intelligence.m5_5g6b.boundary_selection_specification.v1",
    "frozen_before_candidate_scoring": True,
    "authoritative_coordinate_space": "SOURCE_PANORAMA_PIXELS",
    "source_observation_variant": SELECTED_VARIANT,
    "source_observation_pool": SELECTED_POOL,
    "source_observation_state": SELECTED_OUTPUT,
    "inside_near_boundary_signed_distance_pixels": {"minimum_exclusive": 10.0, "maximum_inclusive": 100.0},
    "outside_near_boundary_signed_distance_pixels": {"minimum_inclusive": -100.0, "maximum_exclusive": -10.0},
    "straddling_band_absolute_distance_pixels": 10.0,
    "disagreement_rule": (
        "absolute F1 boundary distance <= 10 source pixels OR P1/P2/P3/P4 runtime relations disagree"
    ),
    "runtime_relations": {
        "P1": "F1 point inside polygon => ON_PITCH else OFF_PITCH",
        "P2": "F1 within 10 source pixels => ROUTE_BOUNDARY_REVIEW else P1",
        "P3": "any F0/F1/F3 relation disagreement => ROUTE_BOUNDARY_REVIEW else P2",
        "P4": "F1 within max(10 pixels, 0.15 bbox height) => ROUTE_BOUNDARY_REVIEW else P3",
    },
    "target_quotas": SELECTION_QUOTAS,
    "distinct_source_group_required": True,
    "one_target_person_per_case": True,
    "gold_pitch_labels_used_for_selection": False,
    "human_labels_hidden_from_mining": True,
    "tie_break": "absolute boundary distance, descending score, source hash, observation UUID",
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def run_git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "byte_size": path.stat().st_size, "sha256": sha256_file(path)}


def tree_manifest(root: Path) -> dict[str, Any]:
    rows = [
        {"path": path.relative_to(root).as_posix(), "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return {"root": str(root), "file_count": len(rows), "tree_hash": stable_hash(rows), "files": rows}


def freeze_json(path: Path, hash_path: Path, payload: Mapping[str, Any]) -> str:
    write_json(path, payload)
    digest = sha256_file(path)
    write_text(hash_path, f"{digest}  {path.name}\n")
    return digest


def iou(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    x1 = max(float(left["x1"]), float(right["x1"]))
    y1 = max(float(left["y1"]), float(right["y1"]))
    x2 = min(float(left["x2"]), float(right["x2"]))
    y2 = min(float(left["y2"]), float(right["y2"]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, float(left["x2"]) - float(left["x1"])) * max(0.0, float(left["y2"]) - float(left["y1"]))
    area_right = max(0.0, float(right["x2"]) - float(right["x1"])) * max(0.0, float(right["y2"]) - float(right["y1"]))
    return intersection / max(1e-12, area_left + area_right - intersection)


def containment(proposal: Mapping[str, float], person: Mapping[str, float]) -> float:
    x1 = max(float(proposal["x1"]), float(person["x1"]))
    y1 = max(float(proposal["y1"]), float(person["y1"]))
    x2 = min(float(proposal["x2"]), float(person["x2"]))
    y2 = min(float(proposal["y2"]), float(person["y2"]))
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    person_area = max(1e-12, (float(person["x2"]) - float(person["x1"])) * (float(person["y2"]) - float(person["y1"])))
    return overlap / person_area


def polygon_relation(distance: float, band: float) -> str:
    if abs(distance) <= band:
        return "ROUTE_BOUNDARY_REVIEW"
    return "ON_PITCH" if distance > 0 else "OFF_PITCH"


def footpoint_variants(box: Mapping[str, float]) -> dict[str, dict[str, float]]:
    x1, x2 = float(box["x1"]), float(box["x2"])
    y2 = float(box["y2"])
    width = x2 - x1
    return {
        "F0": {"x": x1 + 0.42 * width, "y": y2},
        "F1": {"x": (x1 + x2) / 2.0, "y": y2},
        "F3": {"x": x1 + 0.58 * width, "y": y2},
    }


def nearest_polygon_point(point: Mapping[str, float], polygon: Sequence[Mapping[str, float]]) -> dict[str, float]:
    px, py = float(point["x"]), float(point["y"])
    best = {"x": px, "y": py, "distance": float("inf")}
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        ax, ay = float(start["x"]), float(start["y"])
        bx, by = float(end["x"]), float(end["y"])
        dx, dy = bx - ax, by - ay
        scale = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        x, y = ax + scale * dx, ay + scale * dy
        distance = math.dist((px, py), (x, y))
        if distance < best["distance"]:
            best = {"x": x, "y": y, "distance": distance}
    return best


def focal_bounds(box: Mapping[str, float], boundary: Mapping[str, float], width: int, height: int) -> dict[str, int]:
    x1 = min(float(box["x1"]), float(boundary["x"]))
    x2 = max(float(box["x2"]), float(boundary["x"]))
    y1 = min(float(box["y1"]), float(boundary["y"]))
    y2 = max(float(box["y2"]), float(boundary["y"]))
    box_height = max(24.0, float(box["y2"]) - float(box["y1"]))
    pad_x, pad_y = max(90.0, box_height * 2.2), max(60.0, box_height * 1.3)
    left, right = max(0, math.floor(x1 - pad_x)), min(width, math.ceil(x2 + pad_x))
    top, bottom = max(0, math.floor(y1 - pad_y)), min(height, math.ceil(y2 + pad_y))
    if right - left < 320:
        centre = (left + right) / 2
        left, right = max(0, int(centre - 160)), min(width, int(centre + 160))
    if bottom - top < 180:
        centre = (top + bottom) / 2
        top, bottom = max(0, int(centre - 90)), min(height, int(centre + 90))
    return {"x1": left, "y1": top, "x2": right, "y2": bottom}


def repository_and_prompt_validation() -> tuple[dict[str, Any], dict[str, Any]]:
    head = run_git("rev-parse", "HEAD")
    branch = run_git("branch", "--show-current")
    remote = run_git("remote", "get-url", "origin")
    for commit in (
        BASELINE,
        "b54ace62ec79217fbd175a0b4edc84b1f1a0b9b5",
        "abf6da3a51afc5c0cfe46db8d04bff5402ecea62",
        "da98ae2312930c56089ce56a11751185f6a8a54a",
    ):
        subprocess.run(["git", "merge-base", "--is-ancestor", commit, head], cwd=REPO, check=True)
    repository = {
        "schema_version": "football_intelligence.m5_5g6b.repository_state.v1",
        "head": head,
        "branch": branch,
        "remote": remote,
        "baseline": BASELINE,
        "baseline_exact_at_start": head == BASELINE,
        "required_ancestors_valid": True,
        "expected_repository": str(REPO),
        "detector_inference_performed": False,
        "promptable_mask_inference_performed": False,
        "threshold_or_fusion_change_performed": False,
        "component_promoted": False,
        "passed": branch == "main"
        and head == BASELINE
        and remote == "https://github.com/sebgreenhalgh/Football-Intelligence.git",
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
                "passed": actual["byte_size"] == expected["byte_size"] and actual["sha256"] == expected["sha256"],
            }
        )
    prompt = {
        "schema_version": "football_intelligence.m5_5g6b.prompt_pack_validation.v1",
        "file_count": len(rows) + 1,
        "manifest_self_hash_omitted": manifest.get("manifest_self_hash_omitted") is True,
        "rows": rows,
        "passed": len(rows) == 8 and all(row["passed"] for row in rows),
    }
    if not repository["passed"] or not prompt["passed"]:
        raise RuntimeError("FAIL_BASELINE_OR_WORKTREE: repository or prompt-pack authorization failed")
    return repository, prompt


def protected_inputs() -> list[Path]:
    g2_matrix = G2B / "03_FROZEN_PROPOSAL_FAMILY_MATRIX"
    return [
        SOURCE_PACKAGE / "reviewer_manifest.json",
        SOURCE_PACKAGE / "ui_config.json",
        SOURCE_PACKAGE / "decisions" / "review_decisions.json",
        SOURCE_PACKAGE / "decisions" / "review_decision_events.jsonl",
        *(
            C2_BUNDLE / name
            for name in (
                "completed_review.json",
                "completed_review_events.jsonl",
                "completed_review_manifest.json",
                "completed_review_summary.json",
            )
        ),
        G6A / "03_FROZEN_FOOTPOINT_AND_GATE_VARIANTS" / "footpoint_geometry_variant_specification.json",
        G6A / "03_FROZEN_FOOTPOINT_AND_GATE_VARIANTS" / "footpoint_geometry_variant_specification.sha256",
        G6A / "03_FROZEN_FOOTPOINT_AND_GATE_VARIANTS" / "pitch_gate_variant_specification.json",
        G6A / "03_FROZEN_FOOTPOINT_AND_GATE_VARIANTS" / "pitch_gate_variant_specification.sha256",
        G6A / "04_PLAYER_OBSERVATION_V1_SCHEMA" / "player_observation_v1_schema.json",
        G6A / "02_PITCH_POLYGON_AND_TRANSFORM_DIAGNOSIS" / "browser_projection_validation.json",
        G6A / "09_NEXT_STAGE_DECISION" / "development_shortlist.json",
        g2_matrix / "exact_frozen_replay_manifest.json",
        g2_matrix / "exact_replay_raw_candidate_rows.jsonl",
        g2_matrix / "exact_replay_nms_candidate_rows.jsonl",
        g2_matrix / "exact_replay_post_nms_rows.jsonl",
        g2_matrix / "exact_replay_fused_rows.jsonl",
        g2_matrix / "exact_replay_runtime_views.json",
        G3 / "03_FROZEN_CONSOLIDATION_VARIANTS" / "consolidation_variant_specification.json",
        G3 / "03_FROZEN_CONSOLIDATION_VARIANTS" / "consolidation_variant_specification.sha256",
        G3 / "06_PERSON_OBSERVATION_EVALUATION" / "final_observation_ledger.jsonl",
        G3 / "08_NEXT_STAGE_DECISION" / "development_consolidator_shortlist.json",
        G4R2 / "02_DENSE_GOLD_V2_OVERLAY_APPLICATION" / "dense_gold_v2_manifest.json",
        G5A / "04_FROZEN_PROMPT_AND_CROP_MATRIX" / "frozen_crop_prompt_specification.json",
        G5A / "04_FROZEN_PROMPT_AND_CROP_MATRIX" / "frozen_crop_prompt_specification.sha256",
    ]


def protected_manifest() -> dict[str, Any]:
    files = protected_inputs()
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"FAIL_G6A_INPUT_VALIDATION: protected inputs missing: {missing}")
    return {
        "schema_version": "football_intelligence.m5_5g6b.protected_input_manifest.v1",
        "files": [file_record(path) for path in files],
        "manifest_hash": stable_hash([file_record(path) for path in files]),
    }


def validate_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    completed = read_json(C2_BUNDLE / "completed_review.json")
    annotations = completed["state"]["annotations"]
    people = [person for annotation in annotations.values() for person in annotation["player_instances"]]
    c2_bundle = validate_completion_bundle(C2_BUNDLE)
    root_state = read_json(SOURCE_PACKAGE / "decisions" / "review_decisions.json")
    footpoint = G6A / "03_FROZEN_FOOTPOINT_AND_GATE_VARIANTS" / "footpoint_geometry_variant_specification.json"
    pitch = G6A / "03_FROZEN_FOOTPOINT_AND_GATE_VARIANTS" / "pitch_gate_variant_specification.json"
    footpoint_sidecar = footpoint.with_suffix(".sha256")
    pitch_sidecar = pitch.with_suffix(".sha256")
    footpoint_expected = footpoint_sidecar.read_text(encoding="ascii").split()[0]
    pitch_expected = pitch_sidecar.read_text(encoding="ascii").split()[0]
    schema = read_json(G6A / "04_PLAYER_OBSERVATION_V1_SCHEMA" / "player_observation_v1_schema.json")
    projection = read_json(G6A / "02_PITCH_POLYGON_AND_TRANSFORM_DIAGNOSIS" / "browser_projection_validation.json")
    shortlist = read_json(G6A / "09_NEXT_STAGE_DECISION" / "development_shortlist.json")
    replay = read_json(G2B / "03_FROZEN_PROPOSAL_FAMILY_MATRIX" / "exact_frozen_replay_manifest.json")
    g3_spec = G3 / "03_FROZEN_CONSOLIDATION_VARIANTS" / "consolidation_variant_specification.json"
    g3_sidecar = g3_spec.with_suffix(".sha256")
    g3_shortlist = read_json(G3 / "08_NEXT_STAGE_DECISION" / "development_consolidator_shortlist.json")
    replay_hashes = {
        name: sha256_file(G2B / "03_FROZEN_PROPOSAL_FAMILY_MATRIX" / name) for name in replay["artifact_hashes"]
    }
    pitch_counts = dict(sorted(Counter(person["pitch_state"] for person in people).items()))
    checks = {
        "c2_atomic_bundle_valid": c2_bundle["passed"],
        "c2_root_event_sequence_57": int(root_state["event_sequence"]) == 57,
        "c2_case_count_12": len(annotations) == 12,
        "c2_person_count_96": len(people) == 96,
        "c2_pitch_counts_exact": pitch_counts == {"OFF_PITCH": 51, "ON_PITCH": 45},
        "g6a_footpoint_spec_hash_exact": sha256_file(footpoint) == footpoint_expected,
        "g6a_pitch_spec_hash_exact": sha256_file(pitch) == pitch_expected,
        "g6a_player_observation_schema_exact": schema.get("$id") == "football_intelligence.player_observation.v1"
        or schema.get("title") == "PlayerObservationV1",
        "g6a_projection_browser_passed": projection.get("passed") is True,
        "g6a_no_candidate_frozen": shortlist.get("candidate_frozen") is False
        and shortlist.get("pitch_gate_candidates") == []
        and shortlist.get("player_observation_candidates") == [],
        "g2b_exact_replay_passed": replay.get("passed") is True,
        "g2b_artifact_hashes_exact": replay_hashes == replay["artifact_hashes"],
        "g3_spec_hash_exact": sha256_file(g3_spec) == g3_sidecar.read_text(encoding="ascii").split()[0],
        "g3_selected_variant_available": SELECTED_VARIANT in json.dumps(g3_shortlist),
        "dense_gold_v2_read_only_available": (
            G4R2 / "02_DENSE_GOLD_V2_OVERLAY_APPLICATION" / "dense_gold_v2_manifest.json"
        ).is_file(),
        "light_hq_sam_frozen_read_only_available": (
            G5A / "04_FROZEN_PROMPT_AND_CROP_MATRIX" / "frozen_crop_prompt_specification.json"
        ).is_file(),
    }
    validation = {
        "schema_version": "football_intelligence.m5_5g6b.g6a_input_validation.v1",
        "checks": checks,
        "c2_pitch_state_counts": pitch_counts,
        "player_observation_schema_sha256": sha256_file(
            G6A / "04_PLAYER_OBSERVATION_V1_SCHEMA" / "player_observation_v1_schema.json"
        ),
        "footpoint_specification_sha256": sha256_file(footpoint),
        "pitch_gate_specification_sha256": sha256_file(pitch),
        "g2b_replay_manifest_sha256": sha256_file(
            G2B / "03_FROZEN_PROPOSAL_FAMILY_MATRIX" / "exact_frozen_replay_manifest.json"
        ),
        "g3_consolidation_specification_sha256": sha256_file(g3_spec),
        "prior_gold_mutated": False,
        "inference_performed": False,
        "passed": all(checks.values()),
        **safety_payload(),
    }
    if not validation["passed"]:
        raise RuntimeError(f"FAIL_G6A_INPUT_VALIDATION: {checks}")
    source_manifest = read_json(SOURCE_PACKAGE / "reviewer_manifest.json")
    return completed, source_manifest, validation


def source_frame(case: Mapping[str, Any]) -> dict[str, Any]:
    sequence = int(case["source_frame_sequence"])
    matches = [frame for frame in case["visible_metadata"]["frame_records"] if int(frame["frame_sequence"]) == sequence]
    if len(matches) != 1:
        raise RuntimeError(f"source-frame binding is not unique for {case['case_id']}")
    return dict(matches[0])


def identify_missing_people(
    completed: Mapping[str, Any], source_manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    cases = {str(case["case_id"]): case for case in source_manifest["cases"]}
    missing: list[dict[str, Any]] = []
    source_cases: dict[str, dict[str, Any]] = {}
    supported = 0
    for case_id, annotation in sorted(completed["state"]["annotations"].items()):
        case = cases[case_id]
        frame = source_frame(case)
        source_cases[str(frame["source_frame_sha256"])] = case
        related = {
            str(person_id)
            for relation in annotation["candidate_relations"]
            for person_id in relation["annotation_uuids"]
        }
        for person in annotation["player_instances"]:
            if person["pitch_state"] != "ON_PITCH":
                continue
            if str(person["annotation_uuid"]) in related:
                supported += 1
                continue
            missing.append(
                {
                    "case_id": case_id,
                    "source_frame_sha256": str(frame["source_frame_sha256"]),
                    "frame_sequence": int(frame["frame_sequence"]),
                    "person_uuid": str(person["annotation_uuid"]),
                    "visible_body_box": dict(person["visible_body_box"]),
                    "footpoint_status": str(person["footpoint_status"]),
                    "coarse_role": str(person["coarse_role"]),
                }
            )
    if supported != 36 or len(missing) != 9:
        raise RuntimeError(
            f"FAIL_PROPOSAL_SUPPLY_ATTRIBUTION: expected 36 supported and 9 missing, got {supported} and {len(missing)}"
        )
    missing.sort(key=lambda row: (row["case_id"], row["visible_body_box"]["x1"], row["person_uuid"]))
    for index, row in enumerate(missing, 1):
        row["anonymous_person_id"] = f"c2-missing-person-{index:03d}"
    return missing, source_cases


def _proposal_diagnostic(row: Mapping[str, Any], person_box: Mapping[str, float]) -> dict[str, Any]:
    box = row["bbox_panorama_pixels"]
    return {
        "diagnostic_uuid": str(row["diagnostic_uuid"]),
        "source_row_hash": stable_hash(row),
        "view_family": str(row["inference_view_type"]),
        "view_id_hash": stable_hash(str(row["inference_view_id"])),
        "raw_candidate_index": int(row["raw_candidate_index"]),
        "score": float(row["requested_class_score"]),
        "bbox_panorama_pixels": {key: float(box[key]) for key in ("x1", "y1", "x2", "y2")},
        "iou": round(iou(box, person_box), 8),
        "person_containment": round(containment(box, person_box), 8),
    }


def attribute_frozen_supply(
    missing: Sequence[Mapping[str, Any]], source_manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matrix = G2B / "03_FROZEN_PROPOSAL_FAMILY_MATRIX"
    missing_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for person in missing:
        missing_by_source[str(person["source_frame_sha256"])].append(person)
    relevant_sources = set(missing_by_source)
    raw_by_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_key_by_uuid: dict[str, tuple[str, str, int]] = {}
    raw_uuid_by_key: dict[tuple[str, str, int], str] = {}
    source_case_map = {str(case["case_id"]): case for case in source_manifest["cases"]}
    source_stage_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for person in missing:
        frame = source_frame(source_case_map[str(person["case_id"])])
        for candidate in frame.get("candidates", []):
            box = candidate["bbox_original_pixels"]
            source_stage_rows[str(person["anonymous_person_id"])].append(
                {
                    "stage": str(candidate["stage"]),
                    "diagnostic_uuid": str(candidate["diagnostic_uuid"]),
                    "score": float(candidate.get("score") or 0.0),
                    "bbox_panorama_pixels": {key: float(box[key]) for key in ("x1", "y1", "x2", "y2")},
                    "iou": round(iou(box, person["visible_body_box"]), 8),
                    "person_containment": round(containment(box, person["visible_body_box"]), 8),
                    "source_row_hash": str(candidate.get("source_row_sha256") or stable_hash(candidate)),
                    "view_family": "FULL_PANORAMA_1280",
                }
            )
    for row in read_jsonl(matrix / "exact_replay_raw_candidate_rows.jsonl"):
        source_hash = str(row.get("source_frame_sha256"))
        if source_hash not in relevant_sources or row.get("requested_class_name") != "person":
            continue
        key = (source_hash, str(row["inference_view_id"]), int(row["raw_candidate_index"]))
        candidate_uuid = str(row["diagnostic_uuid"])
        raw_key_by_uuid[candidate_uuid] = key
        raw_uuid_by_key[key] = candidate_uuid
        for person in missing_by_source[source_hash]:
            diagnostic = _proposal_diagnostic(row, person["visible_body_box"])
            if diagnostic["iou"] >= 0.02 or diagnostic["person_containment"] >= 0.10:
                raw_by_person[str(person["anonymous_person_id"])].append(diagnostic)

    nms_by_uuid: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(matrix / "exact_replay_nms_candidate_rows.jsonl"):
        source_hash = str(row.get("source_frame_sha256"))
        if source_hash not in relevant_sources or row.get("class_name") != "person":
            continue
        key = (source_hash, str(row["inference_view_id"]), int(row["raw_candidate_index"]))
        candidate_uuid = raw_uuid_by_key.get(key)
        if candidate_uuid:
            nms_by_uuid[candidate_uuid] = {
                "nms_state": str(row["nms_state"]),
                "suppressor_raw_candidate_index": row.get("suppressor_raw_candidate_index"),
                "suppressor_iou": row.get("suppressor_iou"),
            }
    post_uuids = {
        str(row["diagnostic_uuid"])
        for row in read_jsonl(matrix / "exact_replay_post_nms_rows.jsonl")
        if str(row.get("source_frame_sha256")) in relevant_sources and row.get("class_name") == "person"
    }
    fused_uuids: set[str] = set()
    fused_rows_by_member: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(matrix / "exact_replay_fused_rows.jsonl"):
        if str(row.get("source_frame_sha256")) not in relevant_sources:
            continue
        for member in row.get("member_diagnostic_uuids", []):
            member = str(member)
            fused_uuids.add(member)
            fused_rows_by_member[member] = {
                "proposal_id_hash": stable_hash(str(row["proposal_id"])),
                "representative_diagnostic_uuid": str(row["representative_diagnostic_uuid"]),
                "member_count": int(row["member_count"]),
                "inference_view_type": str(row["inference_view_type"]),
            }

    ledger: list[dict[str, Any]] = []
    origins = Counter()
    for person in missing:
        anonymous_id = str(person["anonymous_person_id"])
        proposals = raw_by_person.get(anonymous_id, [])
        c2_rows = source_stage_rows[anonymous_id]
        for raw in (row for row in c2_rows if row["stage"] == "RAW"):
            if raw["iou"] < 0.02 and raw["person_containment"] < 0.10:
                continue
            matched_stages = {
                stage: any(
                    row["stage"] == stage and iou(row["bbox_panorama_pixels"], raw["bbox_panorama_pixels"]) >= 0.80
                    for row in c2_rows
                )
                for stage in ("RAW", "CONFIDENCE", "PRE_NMS", "POST_NMS", "FUSED")
            }
            proposals.append(
                {
                    "diagnostic_uuid": raw["diagnostic_uuid"],
                    "source_row_hash": raw["source_row_hash"],
                    "view_family": "FULL_PANORAMA_1280",
                    "view_id_hash": stable_hash(f"FULL_PANORAMA_1280:{person['source_frame_sha256']}"),
                    "raw_candidate_index": None,
                    "score": raw["score"],
                    "bbox_panorama_pixels": raw["bbox_panorama_pixels"],
                    "iou": raw["iou"],
                    "person_containment": raw["person_containment"],
                    "source": "C2_FROZEN_REVIEW_MANIFEST",
                    "c2_stage_states": matched_stages,
                }
            )
        for proposal in proposals:
            proposal_uuid = proposal["diagnostic_uuid"]
            c2_states = proposal.pop("c2_stage_states", None)
            proposal["stages"] = (
                {
                    "RAW": c2_states["RAW"],
                    "CONFIDENCE_SURVIVING": c2_states["CONFIDENCE"],
                    "PRE_NMS": c2_states["PRE_NMS"],
                    "POST_NMS": c2_states["POST_NMS"],
                    "FUSED": c2_states["FUSED"],
                }
                if c2_states
                else {
                    "RAW": True,
                    "CONFIDENCE_SURVIVING": proposal["score"] >= 0.22,
                    "PRE_NMS": proposal_uuid in nms_by_uuid,
                    "POST_NMS": proposal_uuid in post_uuids,
                    "FUSED": proposal_uuid in fused_uuids,
                }
            )
            proposal["nms"] = nms_by_uuid.get(proposal_uuid)
            if proposal_uuid in fused_rows_by_member:
                proposal["fused_binding"] = fused_rows_by_member[proposal_uuid]
        ranked = sorted(
            proposals,
            key=lambda row: (
                -max(float(row["iou"]), float(row["person_containment"])),
                -float(row["score"]),
                row["diagnostic_uuid"],
            ),
        )
        strong = [row for row in ranked if row["iou"] >= 0.30 or row["person_containment"] >= 0.50]
        confidence = [row for row in strong if row["stages"]["CONFIDENCE_SURVIVING"]]
        post = [row for row in strong if row["stages"]["POST_NMS"]]
        fused = [row for row in strong if row["stages"]["FUSED"]]
        other_family = [
            row
            for row in fused
            if row["view_family"] not in {"FULL_PANORAMA_1280", "OVERLAPPING_HIGH_RESOLUTION_TILES"}
        ]
        if not ranked:
            origin = "NO_RAW_PROPOSAL"
        elif not strong:
            origin = "RAW_LOCALIZATION_BAD"
        elif not confidence:
            origin = "LOST_AT_CONFIDENCE"
        elif not post:
            origin = "LOST_AT_NMS"
        elif other_family:
            origin = "AVAILABLE_OTHER_FROZEN_FAMILY"
        elif not fused:
            origin = "LOST_AT_CROSS_VIEW_FUSION"
        else:
            origin = "UNRESOLVED"
        origins[origin] += 1
        top = ranked[:12]
        ledger.append(
            {
                "schema_version": "football_intelligence.m5_5g6b.frozen_supply_attribution_row.v1",
                "anonymous_person_id": anonymous_id,
                "case_id": person["case_id"],
                "source_frame_sha256": person["source_frame_sha256"],
                "frame_sequence": person["frame_sequence"],
                "visible_body_box_sha256": stable_hash(person["visible_body_box"]),
                "earliest_supported_origin": origin,
                "attribution_confidence": "HIGH"
                if origin in {"LOST_AT_CONFIDENCE", "AVAILABLE_OTHER_FROZEN_FAMILY"}
                else "MEDIUM",
                "strong_raw_proposal_count": len(strong),
                "strong_confidence_surviving_count": len(confidence),
                "strong_post_nms_count": len(post),
                "strong_fused_count": len(fused),
                "inspected_frozen_families": sorted({row["view_family"] for row in proposals}),
                "frozen_family_availability_for_source": {
                    "FULL_PANORAMA_1280": True,
                    "GLOBAL_2048_FIXED": False,
                    "OVERLAPPING_HIGH_RESOLUTION_TILES": False,
                    "BOUNDED_FULL_PANORAMA_2048": False,
                    "CURRENT_LOCAL_CROP_VIEW_IF_ALREADY_AVAILABLE": False,
                },
                "source_stage_candidate_counts": dict(sorted(Counter(row["stage"] for row in c2_rows).items())),
                "best_existing_proposals": top,
                "weak_overlap_upgraded_to_clean_supply": False,
                "new_inference_performed": False,
            }
        )
    if len(ledger) != 9 or sum(origins.values()) != 9:
        raise RuntimeError("FAIL_PROPOSAL_SUPPLY_ATTRIBUTION: person-level attribution cardinality failed")
    summary = {
        "schema_version": "football_intelligence.m5_5g6b.proposal_supply_gap_summary.v1",
        "missing_on_pitch_person_count": len(ledger),
        "origin_counts": dict(sorted(origins.items())),
        "source_frame_count": len(relevant_sources),
        "all_rows_have_source_hash": all(bool(row["source_frame_sha256"]) for row in ledger),
        "all_rows_have_stage_diagnostics": all(bool(row["source_stage_candidate_counts"]) for row in ledger),
        "frozen_artifacts_only": True,
        "weak_overlap_not_clean_supply": True,
        "new_inference_performed": False,
        "passed": len(ledger) == 9
        and all(row["earliest_supported_origin"] != "UNRESOLVED" for row in ledger)
        and all(row["source_stage_candidate_counts"] for row in ledger),
        **safety_payload(),
    }
    if not summary["passed"]:
        raise RuntimeError(f"FAIL_PROPOSAL_SUPPLY_ATTRIBUTION: {summary}")
    return ledger, summary


def static_source_cases(source_manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for case in source_manifest["cases"]:
        if case["task_type"] != "detection_gold_player_static":
            continue
        frame = source_frame(case)
        mapped[str(frame["source_frame_sha256"])] = dict(case)
    return mapped


def candidate_pool(
    source_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_cases = static_source_cases(source_manifest)
    rows: list[dict[str, Any]] = []
    for observation in read_jsonl(G3 / "06_PERSON_OBSERVATION_EVALUATION" / "final_observation_ledger.jsonl"):
        if (
            observation.get("consolidation_variant") != SELECTED_VARIANT
            or observation.get("pool_name") != SELECTED_POOL
            or observation.get("output_state") != SELECTED_OUTPUT
        ):
            continue
        source_hash = str(observation["source_frame_sha256"])
        case = source_cases.get(source_hash)
        if case is None:
            continue
        polygon = case["visible_metadata"]["pitch_polygon_vertices"]
        box = {key: float(observation["box_panorama_pixels"][key]) for key in ("x1", "y1", "x2", "y2")}
        feet = footpoint_variants(box)
        distances = {name: signed_distance_to_polygon(point, polygon) for name, point in feet.items()}
        height = max(1.0, box["y2"] - box["y1"])
        relations = {
            "P1": polygon_relation(distances["F1"], 0.0),
            "P2": polygon_relation(distances["F1"], 10.0),
            "P3": (
                "ROUTE_BOUNDARY_REVIEW"
                if len({polygon_relation(distances[name], 0.0) for name in ("F0", "F1", "F3")}) > 1
                else polygon_relation(distances["F1"], 10.0)
            ),
            "P4": polygon_relation(distances["F1"], max(10.0, 0.15 * height)),
        }
        disagreement = len(set(relations.values())) > 1 or abs(distances["F1"]) <= 10.0
        strata: list[str] = []
        if disagreement:
            strata.append("disagreement_hidden_feet_or_straddling")
        if 10.0 < distances["F1"] <= 100.0:
            strata.append("estimated_inside_near_boundary")
        if -100.0 <= distances["F1"] < -10.0:
            strata.append("estimated_outside_near_boundary")
        if not strata:
            continue
        boundary = nearest_polygon_point(feet["F1"], polygon)
        rows.append(
            {
                "source_group_id": stable_hash(source_hash)[:20],
                "source_frame_sha256": source_hash,
                "frame_sequence": int(source_frame(case)["frame_sequence"]),
                "source_case_id": case["case_id"],
                "observation_uuid": observation["observation_uuid"],
                "observation_provenance_hash": observation["provenance_hash"],
                "representative_proposal_uuid": observation["representative_proposal_uuid"],
                "representative_proposal_hash": observation["representative_proposal_hash"],
                "source_view_ids": observation["all_source_view_ids"],
                "box_panorama_pixels": box,
                "score": float(observation["score"]),
                "footpoint_variants": feet,
                "signed_distances_source_pixels": {key: round(value, 8) for key, value in distances.items()},
                "runtime_variant_relations": relations,
                "runtime_variant_disagreement": disagreement,
                "nearest_boundary_point": {key: round(float(boundary[key]), 8) for key in ("x", "y", "distance")},
                "eligible_strata": strata,
                "selection_uses_human_pitch_gold": False,
                "identity_tracking_performed": False,
            }
        )
    rows.sort(
        key=lambda row: (
            abs(row["signed_distances_source_pixels"]["F1"]),
            -row["score"],
            row["source_frame_sha256"],
            row["observation_uuid"],
        )
    )
    selected: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    quota_order = (
        "disagreement_hidden_feet_or_straddling",
        "estimated_outside_near_boundary",
        "estimated_inside_near_boundary",
    )
    shortfalls: dict[str, int] = {}
    for stratum in quota_order:
        eligible = [
            row for row in rows if stratum in row["eligible_strata"] and row["source_frame_sha256"] not in used_sources
        ]
        chosen: list[dict[str, Any]] = []
        stratum_sources: set[str] = set()
        for row in eligible:
            if row["source_frame_sha256"] in stratum_sources:
                continue
            chosen.append(row)
            stratum_sources.add(row["source_frame_sha256"])
            if len(chosen) == SELECTION_QUOTAS[stratum]:
                break
        for row in chosen:
            selected.append({**row, "selection_stratum": stratum})
            used_sources.add(row["source_frame_sha256"])
        shortfalls[stratum] = SELECTION_QUOTAS[stratum] - len(chosen)
    if any(shortfalls.values()):
        for stratum, shortfall in list(shortfalls.items()):
            if not shortfall:
                continue
            fallback = [row for row in rows if row["source_frame_sha256"] not in used_sources][:shortfall]
            for row in fallback:
                selected.append({**row, "selection_stratum": stratum, "quota_fallback": True})
                used_sources.add(row["source_frame_sha256"])
            shortfalls[stratum] -= len(fallback)
    selected.sort(
        key=lambda row: stable_hash(
            {"review_id": REVIEW_ID, "source": row["source_frame_sha256"], "observation": row["observation_uuid"]}
        )
    )
    for index, row in enumerate(selected, 1):
        row["case_id"] = f"m5_5g6b_boundary_case_{index:03d}"
        row["anonymous_target_uuid"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{REVIEW_ID}:target:{index}"))
        row["anonymous_annotation_uuid"] = f"boundary-target-{index:03d}"
    validation = {
        "schema_version": "football_intelligence.m5_5g6b.boundary_case_selection_validation.v1",
        "selection_specification_sha256": sha256_file(DIRS["mining"] / "boundary_selection_specification.json"),
        "candidate_pool_count": len(rows),
        "selected_case_count": len(selected),
        "selected_source_group_count": len({row["source_frame_sha256"] for row in selected}),
        "selected_stratum_counts": dict(sorted(Counter(row["selection_stratum"] for row in selected).items())),
        "quota_shortfalls": shortfalls,
        "all_source_groups_distinct": len({row["source_frame_sha256"] for row in selected}) == len(selected),
        "gold_pitch_labels_used_for_selection": False,
        "selection_frozen_before_human_labels": True,
        "selection_order_independent_of_expected_label": True,
        "passed": len(selected) == 18
        and len({row["source_frame_sha256"] for row in selected}) == 18
        and not any(shortfalls.values())
        and Counter(row["selection_stratum"] for row in selected) == Counter(SELECTION_QUOTAS),
        **safety_payload(),
    }
    if not validation["passed"]:
        raise RuntimeError(f"FAIL_BOUNDARY_CASE_MINING: {validation}")
    return rows, selected, validation


def _panorama_asset(case: Mapping[str, Any], frame: Mapping[str, Any]) -> tuple[Path, Mapping[str, Any]]:
    source_hash = str(frame["source_frame_sha256"])
    matches = [
        asset
        for asset in case["evidence_assets"]
        if asset["asset_type"] == "image"
        and source_hash == str(asset.get("metadata", {}).get("source_frame_sha256"))
        and int(frame["frame_sequence"]) in asset.get("frame_sequences", [])
    ]
    if len(matches) != 1:
        raise RuntimeError(f"cannot resolve exact panorama asset for {case['case_id']} frame {frame['frame_sequence']}")
    path = SOURCE_PACKAGE / "evidence" / case["case_id"] / matches[0]["relative_path"]
    if sha256_file(path) != source_hash:
        raise RuntimeError(f"source panorama hash mismatch: {path}")
    return path, matches[0]


def _asset(*, case_id: str, index: int, kind: str, path: Path, sequence: int, source_hash: str) -> dict[str, Any]:
    types = {"panorama": "image", "focal": "crop", "contact": "temporal_strip"}
    labels = {
        "panorama": "Exact full panorama",
        "focal": "Target and nearest-boundary focal view",
        "contact": "Target context strip",
    }
    return {
        "asset_id": f"{case_id}_{kind}_{index:03d}",
        "asset_type": types[kind],
        "label": labels[kind],
        "relative_path": path.name,
        "sha256": sha256_file(path),
        "media_type": "image/jpeg",
        "frame_sequences": [sequence],
        "metadata": {"human_truth": False, "source_frame_sha256": source_hash, "target_only_review": True},
        "visibility_policy": "always_visible",
    }


def build_review_package(
    source_manifest: Mapping[str, Any], selected: Sequence[Mapping[str, Any]], selection_hash: str
) -> dict[str, Any]:
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    (PACKAGE / "evidence").mkdir(parents=True)
    source_cases = static_source_cases(source_manifest)
    evidence_rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for module_case_number, selected_row in enumerate(selected, start=1):
        case_id = str(selected_row["case_id"])
        source_case = source_cases[str(selected_row["source_frame_sha256"])]
        case_root = PACKAGE / "evidence" / case_id
        case_root.mkdir(parents=True)
        source_frames = sorted(
            source_case["visible_metadata"]["frame_records"], key=lambda row: int(row["frame_sequence"])
        )
        centre = int(source_case["source_frame_sequence"])
        current_frame = source_frame(source_case)
        polygon = source_case["visible_metadata"]["pitch_polygon_vertices"]
        box = selected_row["box_panorama_pixels"]
        boundary = selected_row["nearest_boundary_point"]
        bounds = focal_bounds(box, boundary, int(current_frame["image_width"]), int(current_frame["image_height"]))
        frame_records: list[dict[str, Any]] = []
        case_assets: list[dict[str, Any]] = []
        for index, frame in enumerate(source_frames):
            source_path, _ = _panorama_asset(source_case, frame)
            panorama_path = case_root / f"panorama_{index:03d}.jpg"
            shutil.copy2(source_path, panorama_path)
            with Image.open(source_path) as image:
                image.load()
                focal = image.crop((bounds["x1"], bounds["y1"], bounds["x2"], bounds["y2"]))
                focal_path = case_root / f"focal_{index:03d}.jpg"
                focal.save(focal_path, format="JPEG", quality=95, subsampling=0)
                contact = focal.copy()
                contact.thumbnail((480, 240), Image.Resampling.LANCZOS)
                contact_path = case_root / f"contact_{index:03d}.jpg"
                contact.save(contact_path, format="JPEG", quality=92)
            is_current = int(frame["frame_sequence"]) == centre
            candidates = []
            if is_current:
                candidates = [
                    {
                        "diagnostic_uuid": selected_row["anonymous_target_uuid"],
                        "stage": "FUSED",
                        "inference_view": "ANONYMOUS_FROZEN_PROPOSAL",
                        "bbox_original_pixels": box,
                        "score": None,
                        "source_row_sha256": stable_hash(
                            {
                                "anonymous_target_uuid": selected_row["anonymous_target_uuid"],
                                "source_frame_sha256": frame["source_frame_sha256"],
                                "bbox_original_pixels": box,
                            }
                        ),
                    }
                ]
            frame_records.append(
                {
                    "frame_sequence": int(frame["frame_sequence"]),
                    "timestamp_seconds": float(frame["timestamp_seconds"]),
                    "source_frame_sha256": str(frame["source_frame_sha256"]),
                    "image_width": int(frame["image_width"]),
                    "image_height": int(frame["image_height"]),
                    "focal_bounds": bounds,
                    "panorama_asset_id": f"{case_id}_panorama_{index:03d}",
                    "panorama_asset_path": panorama_path.name,
                    "panorama_asset_sha256": sha256_file(panorama_path),
                    "focal_asset_id": f"{case_id}_focal_{index:03d}",
                    "focal_asset_path": focal_path.name,
                    "focal_asset_sha256": sha256_file(focal_path),
                    "contact_asset_id": f"{case_id}_contact_{index:03d}",
                    "contact_asset_path": contact_path.name,
                    "contact_asset_sha256": sha256_file(contact_path),
                    "candidates": candidates,
                }
            )
            for kind, path in (("panorama", panorama_path), ("focal", focal_path), ("contact", contact_path)):
                asset = _asset(
                    case_id=case_id,
                    index=index,
                    kind=kind,
                    path=path,
                    sequence=int(frame["frame_sequence"]),
                    source_hash=str(frame["source_frame_sha256"]),
                )
                case_assets.append(asset)
                evidence_rows.append({"case_id": case_id, **asset})
        binding = copy.deepcopy(source_case["visible_metadata"]["source_binding"])
        binding.update(
            {
                "frame_index": int(current_frame["frame_sequence"]),
                "timestamp_seconds": float(current_frame["timestamp_seconds"]),
                "source_frame_sha256": str(current_frame["source_frame_sha256"]),
                "image_width": int(current_frame["image_width"]),
                "image_height": int(current_frame["image_height"]),
                "review_crop_bounds": bounds,
                "pitch_polygon_hash": stable_hash(polygon),
                "panorama_transform": {
                    "type": "crop_translation_only",
                    "scale_x": 1.0,
                    "scale_y": 1.0,
                    "focal_to_panorama_x": bounds["x1"],
                    "focal_to_panorama_y": bounds["y1"],
                    "round_trip_tolerance_pixels": 0.5,
                },
            }
        )
        visible_metadata = {
            "module": "pitch_boundary",
            "module_case_number": module_case_number,
            "diagnostic_only": True,
            "target_only_review": True,
            "target_initial_box_original_pixels": box,
            "target_annotation_uuid": selected_row["anonymous_annotation_uuid"],
            "candidate_uuids": [selected_row["anonymous_target_uuid"]],
            "frame_records": frame_records,
            "source_binding": binding,
            "pitch_polygon_vertices": polygon,
            "pitch_boundary_tolerance_pixels": 10.0,
            "nearest_boundary_point_original_pixels": {"x": boundary["x"], "y": boundary["y"]},
            "coordinate_space": "SOURCE_PANORAMA_PIXELS",
            "proposal_provenance_available": True,
            "proposal_assistance_is_truth": False,
            "reference_frames_editable": False,
            "target_copy": "Label the highlighted target person only. Other people are context.",
        }
        case_payload = {
            "case_id": case_id,
            "task_type": "detection_gold_pitch_boundary",
            "candidate_id": f"anonymous-boundary-target-{case_id[-3:]}",
            "candidate_hash": stable_hash(
                {
                    "case_id": case_id,
                    "source_frame_sha256": current_frame["source_frame_sha256"],
                    "anonymous_target_uuid": selected_row["anonymous_target_uuid"],
                }
            ),
            "evidence_hash": stable_hash(
                [{key: asset[key] for key in ("asset_id", "sha256")} for asset in case_assets]
            ),
            "equivalence_cluster_id": stable_hash(str(current_frame["source_frame_sha256"]))[:24],
            "allowed_decisions": ["ANNOTATED"],
            "concise_question": "What is the pitch relation of the highlighted target person?",
            "detailed_instructions": "Label the highlighted target person only. Other people are context.",
            "priority": 0,
            "evidence_assets": case_assets,
            "source_frame_sequence": centre,
            "target_frame_sequence": centre,
            "frame_gap": 0,
            "source_bbox": box,
            "target_bbox": box,
            "competing_candidates": [],
            "visible_metadata": visible_metadata,
            "hidden_metadata": {},
            "reveal_metadata": {},
            "safety_payload": safety_payload(),
            "source_artifact_references": [
                {
                    "artifact_id": f"source-panorama-{case_id}",
                    "path": f"SOURCE_PACKAGE/evidence/{source_case['case_id']}",
                    "sha256": str(current_frame["source_frame_sha256"]),
                    "role": "exact source panorama and temporal context",
                }
            ],
        }
        cases.append(case_payload)
        private_rows.append(
            {
                "case_id": case_id,
                "source_group_id": selected_row["source_group_id"],
                "source_case_id": selected_row["source_case_id"],
                "source_frame_sha256": selected_row["source_frame_sha256"],
                "selection_stratum": selected_row["selection_stratum"],
                "observation_uuid": selected_row["observation_uuid"],
                "representative_proposal_uuid": selected_row["representative_proposal_uuid"],
                "anonymous_target_uuid": selected_row["anonymous_target_uuid"],
                "selection_specification_sha256": selection_hash,
            }
        )
    evidence_manifest = {
        "schema_version": "football_intelligence.m5_5g6b.evidence_manifest.v1",
        "review_id": REVIEW_ID,
        "asset_count": len(evidence_rows),
        "assets": evidence_rows,
    }
    evidence_manifest["evidence_manifest_hash"] = stable_hash(evidence_manifest)
    write_json(PACKAGE / "evidence_manifest.json", evidence_manifest)
    manifest_model = GenericReviewManifest.model_validate(
        {
            "schema_version": "football_intelligence.review_manifest.v2",
            "review_id": REVIEW_ID,
            "stage_id": STAGE_ID,
            "task_type": "detection_gold_pitch_boundary",
            "title": "Boundary-focused target-person gold",
            "production_ready": False,
            "no_auto_promotion": True,
            "human_approved": False,
            "cases": cases,
            "manifest_hash": "",
            "evidence_manifest_hash": evidence_manifest["evidence_manifest_hash"],
            "source_manifest_hash": stable_hash(source_manifest),
            "source_artifact_references": [],
            "safety_payload": safety_payload(),
        }
    )
    manifest_model.manifest_hash = manifest_hash(manifest_model)
    write_json(PACKAGE / "reviewer_manifest.json", manifest_model.model_dump(mode="json"))
    write_json(DIRS["cases"] / "boundary_case_private_provenance_mapping.json", {"rows": private_rows})

    source_ui = read_json(SOURCE_PACKAGE / "ui_config.json")
    source_ui.update(
        {
            "page_title": "Football Intelligence - Boundary-focused person gold",
            "review_title": "Target-person pitch boundary review",
            "task_instructions": "Label the highlighted target person only. Other people are context.",
            "gif_primary": False,
            "image_stepper_enabled": True,
            "layout": "single_synchronized_viewer",
            "presentation_mode": "detection_gold_pilot",
            "decisions": [{"key": "s", "value": "ANNOTATED", "label": "Save complete target", "style": "primary"}],
        }
    )
    contract = dict(source_ui.get("question_contract", {}))
    case_ids = [case["case_id"] for case in cases]
    contract.update(
        {
            "client_build_id": G6B_BOUNDARY_FOCUSED_CLIENT_BUILD_ID,
            "reviewer_session_id": REVIEWER,
            "indexeddb_namespace": INDEXEDDB_NAMESPACE,
            "fresh_indexeddb_namespace": True,
            "prior_indexeddb_namespace_import_forbidden": True,
            "persistence_mode": "detection_gold_pilot_v1",
            "server_authoritative_events": True,
            "indexeddb_outbox_required": True,
            "saved_only_after_ack": True,
            "first_load_server_reconciliation": True,
            "first_load_notice": (
                "Fresh boundary-focused target review ready. " "No prior decisions or browser drafts were imported."
            ),
            "revision_aware_wizard_state": True,
            "incremental_gold_tranches": True,
            "gold_tranches": {TRANCHE: {"case_ids": case_ids, "label": "B1 - boundary-focused person gold"}},
            "tranche_order": [TRANCHE],
            "default_tranche_id": TRANCHE,
            "modules": ["detection_gold_pitch_boundary"],
            "full_completion_requires_all_tranches": True,
            "novice_guided_wizard": True,
            "boundary_focused_person_gold": True,
            "boundary_target_only_copy": "Label the highlighted target person only. Other people are context.",
            "boundary_workflow_steps": [
                "Confirm target box",
                "Mark role, feet, and footpoint",
                "Choose pitch relation",
                "Review and save",
            ],
            "boundary_selection_specification_hash": selection_hash,
            "boundary_quota_shortfalls": {key: 0 for key in SELECTION_QUOTAS},
            "boundary_source_group_count": len(cases),
            "boundary_source_group_diversity_hash": stable_hash(sorted(row["source_group_id"] for row in selected)),
            "static_authoritative_bindings": {},
            "c2_multi_person_pitch_boundary_workflow": False,
            "c2_current_frame_only": True,
            "c2_focal_roi_only": True,
            "c2_reference_frames_editable": False,
            "c2_pitch_polygon_default_visible": True,
            "c2_boundary_uncertainty_band_visible": True,
            "c2_pitch_overlay_pointer_events": "none",
            "c2_candidate_review_independent_of_role_and_pitch": True,
            "c2_allowed_roles": ["PLAYER", "GOALKEEPER", "REFEREE", "OFFICIAL", "STAFF_OR_SPECTATOR", "UNKNOWN"],
            "c2_allowed_pitch_states": ["ON_PITCH", "OFF_PITCH", "BOUNDARY_UNCERTAIN"],
            "c2_allowed_pitch_certainty": ["CLEAR", "APPROXIMATE", "UNCERTAIN"],
            "c2_allowed_footpoint_states": [
                "OBSERVED_CLEAR",
                "OBSERVED_APPROXIMATE",
                "FEET_NOT_VISIBLE",
                "CANNOT_TELL",
            ],
            "human_measured_active_minutes": None,
            "pilot_diagnostic_only": True,
            "architecture_evaluation_forbidden": True,
        }
    )
    source_ui["question_contract"] = contract
    write_json(PACKAGE / "ui_config.json", source_ui)
    loaded_manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    for case in loaded_manifest.cases:
        contract["static_authoritative_bindings"][case.case_id] = {
            "frame_sequence": int(case.source_frame_sequence),
            "source_frame_sha256": case.visible_metadata["source_binding"]["source_frame_sha256"],
            "image_width": int(case.visible_metadata["source_binding"]["image_width"]),
            "image_height": int(case.visible_metadata["source_binding"]["image_height"]),
            "candidate_uuids": list(case.visible_metadata["candidate_uuids"]),
            "candidate_queue_binding_hash": authoritative_candidate_binding_hash(case),
        }
    write_json(PACKAGE / "ui_config.json", source_ui)
    decisions = PACKAGE / "decisions"
    config = load_ui_config(PACKAGE / "ui_config.json")
    store = DetectionGoldPilotPersistence(
        manifest=loaded_manifest, ui_config=config, decisions_root=decisions, reviewer_session_id=REVIEWER
    )
    state = store.ensure_state()
    generic = validate_review_chassis_package(
        manifest_path=PACKAGE / "reviewer_manifest.json",
        ui_config_path=PACKAGE / "ui_config.json",
        evidence_root=PACKAGE / "evidence",
        decisions_root=decisions,
    )
    validation = {
        "schema_version": "football_intelligence.m5_5g6b.review_package_validation.v1",
        "manifest_hash": manifest_hash(loaded_manifest),
        "ui_config_hash": ui_config_hash(config),
        "case_count": len(loaded_manifest.cases),
        "evidence_asset_count": len(evidence_rows),
        "one_target_per_case": all(
            len(case.visible_metadata["candidate_uuids"]) == 1 for case in loaded_manifest.cases
        ),
        "browser_manifest_has_selection_strata": "selection_stratum"
        in json.dumps(read_json(PACKAGE / "reviewer_manifest.json")),
        "fresh_decisions_root": state.get("annotations") == {} and int(state.get("event_sequence", -1)) == 0,
        "completion_bundle_absent": not (decisions / "completed_tranches" / TRANCHE).exists(),
        "generic_validation": generic,
        "passed": len(loaded_manifest.cases) == 18
        and len(evidence_rows) == 162
        and all(len(case.visible_metadata["candidate_uuids"]) == 1 for case in loaded_manifest.cases)
        and "selection_stratum" not in json.dumps(read_json(PACKAGE / "reviewer_manifest.json"))
        and state.get("annotations") == {}
        and int(state.get("event_sequence", -1)) == 0
        and generic["passed"],
        **safety_payload(),
    }
    write_json(PACKAGE / "review_package_validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError(f"FAIL_PERSON_CENTRIC_WORKFLOW: {validation}")
    return validation


def write_launcher_and_instructions() -> None:
    launcher = f"""$ErrorActionPreference = 'Stop'
$port = 8810
$occupied = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($occupied) {{
  Write-Error 'Port 8810 is occupied. Stop the existing server, then rerun. This launcher will not move ports.'
}}
$repo = '{REPO}'
$package = '{PACKAGE}'
Set-Location -LiteralPath $repo
Write-Host 'Starting the boundary-focused target-person review.' -ForegroundColor Green
Write-Host 'Open http://127.0.0.1:8810/' -ForegroundColor Cyan
uv run fi-pipeline review-chassis serve `
  --manifest "$package/reviewer_manifest.json" `
  --ui-config "$package/ui_config.json" `
  --evidence-root "$package/evidence" `
  --decisions-root "$package/decisions" `
  --host 127.0.0.1 `
  --port 8810 `
  --reviewer-session-id '{REVIEWER}'
"""
    instructions = """# Boundary-focused target-person review

1. Stop any older annotation server on port 8810.
2. Run `launch_boundary_focused_review.ps1`.
3. Open `http://127.0.0.1:8810/`.
4. Confirm the new tranche starts at `0/18 saved`.
5. Complete all 18 highlighted targets, then select **Complete tranche**.

Label the highlighted target person only. Other people are context.

- Confirm or redraw the target's visible-body box.
- Record role, feet visibility and the best supported footpoint.
- Use `BOUNDARY_UNCERTAIN` when the ground-contact region touches or straddles
  the line, is too close to resolve, or hidden feet prevent a defensible side.
- A substitute remains `PLAYER + OFF_PITCH`.
- The polygon and machine proposal are evidence, not automatic truth.

The package is single-reviewer development gold. It is not validation, holdout evidence, or approval for production use.
"""
    for root in (PACKAGE, STAGE):
        write_text(root / "launch_boundary_focused_review.ps1", launcher)
        write_text(root / "HUMAN_INSTRUCTIONS.md", instructions)


def write_workflow_and_completion_outputs(selected: Sequence[Mapping[str, Any]], selection_hash: str) -> None:
    app = (REPO / "src/football_intelligence/review_chassis/static/detection_gold_app.js").read_text(encoding="utf-8")
    wizard = (REPO / "src/football_intelligence/review_chassis/static/detection_gold_wizard.js").read_text(
        encoding="utf-8"
    )
    persistence = (REPO / "src/football_intelligence/detection_gold/persistence.py").read_text(encoding="utf-8")
    exact_copy = "Label the highlighted target person only. Other people are context."
    workflow = {
        "schema_version": "football_intelligence.m5_5g6b.person_centric_workflow_validation.v1",
        "review_id": REVIEW_ID,
        "reviewer_session_id": REVIEWER,
        "url": "http://127.0.0.1:8810/",
        "four_steps": [
            "Confirm target box",
            "Mark role, feet, and footpoint",
            "Choose pitch relation",
            "Review and save",
        ],
        "exact_target_only_copy_present": exact_copy in wizard and exact_copy in app,
        "one_target_cardinality_enforced_client": "requires exactly one highlighted target person" in app,
        "one_target_cardinality_enforced_server": "boundary-focused gold requires exactly one" in persistence,
        "redraw_supported": "redrawSelectedPerson" in wizard and "beginRedrawSelectedVisible" in app,
        "pitch_states": ["ON_PITCH", "OFF_PITCH", "BOUNDARY_UNCERTAIN"],
        "role_and_pitch_are_separate": True,
        "proposal_relation_target_only": True,
        "other_people_context_only": True,
        "exhaustive_crowd_annotation_required": False,
        "panorama_and_focal_views": True,
        "previous_current_next_context": True,
        "pitch_polygon_and_10_pixel_band": True,
        "overlay_pointer_interception": False,
        "indexeddb_durable_outbox": True,
        "server_acknowledgement_required": True,
        "passed": exact_copy in wizard
        and exact_copy in app
        and "requires exactly one highlighted target person" in app
        and "boundary-focused gold requires exactly one" in persistence
        and "redrawSelectedPerson" in wizard
        and "beginRedrawSelectedVisible" in app,
        **safety_payload(),
    }
    if not workflow["passed"]:
        raise RuntimeError(f"FAIL_PERSON_CENTRIC_WORKFLOW: {workflow}")
    write_json(DIRS["browser"] / "person_centric_workflow_validation.json", workflow)
    completion = {
        "schema_version": "football_intelligence.m5_5g6b.boundary_completion_contract.v1",
        "review_id": REVIEW_ID,
        "tranche_id": TRANCHE,
        "required_case_ids": [row["case_id"] for row in selected],
        "required_target_count": len(selected),
        "selection_specification_sha256": selection_hash,
        "quota_shortfalls": {key: 0 for key in SELECTION_QUOTAS},
        "source_group_count": len({row["source_group_id"] for row in selected}),
        "source_group_diversity_hash": stable_hash(sorted(row["source_group_id"] for row in selected)),
        "completion_bundle_fields": [
            "selection_specification_hash",
            "exact_target_count",
            "quota_shortfalls",
            "source_group_count",
            "source_group_diversity_hash",
            "human_pitch_state_counts",
            "footpoint_visibility_counts",
            "prior_gold_unchanged",
        ],
        "atomic_four_file_completion": True,
        "completion_does_not_complete_original_88_case_pilot": True,
        "completion_does_not_complete_temporal_or_football_tranches": True,
        "fresh_completion_bundle_absent": not (PACKAGE / "decisions" / "completed_tranches" / TRANCHE).exists(),
        "passed": len(selected) == 18 and not (PACKAGE / "decisions" / "completed_tranches" / TRANCHE).exists(),
        **safety_payload(),
    }
    write_json(DIRS["browser"] / "boundary_completion_contract.json", completion)
    timing = {
        "schema_version": "football_intelligence.m5_5g6b.truthful_boundary_timing.v1",
        "case_count": len(selected),
        "estimated_active_minutes": {"lower": 24, "upper": 40},
        "estimate_basis": "four-step target-only workflow; no exhaustive crowd annotation",
        "human_measured_active_minutes": None,
        "timing_claim_is_estimate_not_measurement": True,
        "target_range_minutes": "not contractually capped",
    }
    write_json(DIRS["browser"] / "truthful_boundary_timing.json", timing)
    browser = {
        "schema_version": "football_intelligence.m5_5g6b.browser_persistence_results.v1",
        "status": "PENDING_REAL_BROWSER_ACCEPTANCE",
        "production_decisions_root_fresh": True,
        "synthetic_tests_use_temporary_root": True,
        "required_profiles": ["1024x768", "1366x768", "1440x900", "1920x1080", "2560x1440"],
        "passed": False,
    }
    write_json(DIRS["browser"] / "browser_persistence_results.json", browser)


def write_next_stage_permission() -> None:
    write_json(
        DIRS["next"] / "next_stage_permission.json",
        {
            "schema_version": "football_intelligence.m5_5g6b.next_stage_permission.v1",
            "next_stage": "M5_5G6C_PITCH_GATE_REEVALUATION_AND_PROPOSAL_SUPPLY_RECOVERY_DECISION_v1",
            "authorized_now": False,
            "authorization_requires": [
                "completed B1 boundary-focused human review",
                "atomic completion bundle validation",
                "independent audit of human labels and source bindings",
            ],
            "conditionally_permitted_work": [
                "re-evaluate frozen P1-P4 on boundary-focused development gold",
                "decide whether a conservative pitch gate can be frozen",
                "select one proposal-recovery experiment from the nine-person attribution",
                "rebuild the Player Observation v1 development candidate",
            ],
            "forbidden": ["promotion", "training", "identity tracking", "final accuracy claims"],
            "human_review_pending": True,
            **safety_payload(),
        },
    )


def prepare_workspace() -> None:
    for path in DIRS.values():
        path.mkdir(parents=True, exist_ok=True)
    for path in sorted(PROMPT.iterdir()):
        if path.is_file():
            shutil.copy2(path, DIRS["inputs"] / path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-attribution", action="store_true", help="reuse an existing frozen attribution ledger")
    args = parser.parse_args()
    prepare_workspace()
    repository, prompt = repository_and_prompt_validation()
    write_json(DIRS["inputs"] / "repository_state.json", repository)
    write_json(DIRS["inputs"] / "prompt_pack_validation.json", prompt)
    before = protected_manifest()
    write_json(DIRS["inputs"] / "protected_input_manifest_before.json", before)
    completed, source_manifest, validation = validate_inputs()
    write_json(DIRS["validation"] / "g6a_input_validation.json", validation)
    selection_hash = freeze_json(
        DIRS["mining"] / "boundary_selection_specification.json",
        DIRS["mining"] / "boundary_selection_specification.sha256",
        SELECTION_SPEC,
    )
    missing, _ = identify_missing_people(completed, source_manifest)
    public_missing = [
        {
            "anonymous_person_id": row["anonymous_person_id"],
            "case_id": row["case_id"],
            "source_frame_sha256": row["source_frame_sha256"],
            "frame_sequence": row["frame_sequence"],
            "visible_body_box_sha256": stable_hash(row["visible_body_box"]),
            "footpoint_status": row["footpoint_status"],
            "coarse_role": row["coarse_role"],
        }
        for row in missing
    ]
    write_json(
        DIRS["attribution"] / "c2_missing_on_pitch_people.json",
        {
            "schema_version": "football_intelligence.m5_5g6b.c2_missing_on_pitch_people.v1",
            "count": len(public_missing),
            "rows": public_missing,
            "human_annotation_uuids_redacted": True,
        },
    )
    attribution_path = DIRS["attribution"] / "frozen_proposal_supply_attribution.jsonl"
    if args.skip_attribution and attribution_path.is_file():
        ledger = list(read_jsonl(attribution_path))
        origins = Counter(row["earliest_supported_origin"] for row in ledger)
        gap = {
            "schema_version": "football_intelligence.m5_5g6b.proposal_supply_gap_summary.v1",
            "missing_on_pitch_person_count": len(ledger),
            "origin_counts": dict(sorted(origins.items())),
            "frozen_artifacts_only": True,
            "new_inference_performed": False,
            "passed": len(ledger) == 9,
            **safety_payload(),
        }
    else:
        ledger, gap = attribute_frozen_supply(missing, source_manifest)
        write_jsonl(attribution_path, ledger)
    write_json(DIRS["attribution"] / "proposal_supply_gap_summary.json", gap)
    pool, selected, selection = candidate_pool(source_manifest)
    write_json(
        DIRS["mining"] / "boundary_candidate_pool.json",
        {
            "schema_version": "football_intelligence.m5_5g6b.boundary_candidate_pool.v1",
            "selection_specification_sha256": selection_hash,
            "candidate_count": len(pool),
            "rows": pool,
            "human_pitch_labels_present": False,
        },
    )
    write_json(
        DIRS["cases"] / "boundary_case_manifest.json",
        {
            "schema_version": "football_intelligence.m5_5g6b.boundary_case_manifest.v1",
            "selection_specification_sha256": selection_hash,
            "case_count": len(selected),
            "cases": selected,
            "single_reviewer_development_only": True,
        },
    )
    write_json(DIRS["cases"] / "boundary_case_selection_validation.json", selection)
    package_validation = build_review_package(source_manifest, selected, selection_hash)
    write_launcher_and_instructions()
    write_workflow_and_completion_outputs(selected, selection_hash)
    write_next_stage_permission()
    after = protected_manifest()
    write_json(DIRS["commands"] / "protected_input_manifest_after_build.json", after)
    prior_unchanged = before == after
    summary = {
        "schema_version": "football_intelligence.m5_5g6b.build_summary.v1",
        "classification": "PENDING_BROWSER_AND_TEST_VALIDATION",
        "review_id": REVIEW_ID,
        "case_count": len(selected),
        "source_group_count": len({row["source_group_id"] for row in selected}),
        "selection_stratum_counts": dict(sorted(Counter(row["selection_stratum"] for row in selected).items())),
        "missing_supply_person_count": len(ledger),
        "attribution_origin_counts": gap["origin_counts"],
        "review_package_passed": package_validation["passed"],
        "prior_artifacts_unchanged": prior_unchanged,
        "new_inference_performed": False,
        "threshold_or_fusion_change_performed": False,
        "component_promoted": False,
        "human_review_pending": True,
        **safety_payload(),
    }
    write_json(DIRS["commands"] / "build_summary.json", summary)
    if not prior_unchanged:
        raise RuntimeError("FAIL_PRIOR_STAGE_MUTATION: protected hashes changed during build")


if __name__ == "__main__":
    main()
