"""Reproducible G7B stage orchestration helpers.

This module is intentionally conservative: it validates every immutable input
before a caller creates the G7B workspace, keeps K1 candidate state unavailable,
and treats learned pitch predictions as descriptive evidence only.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from football_intelligence.review_chassis.hashing import stable_hash

STAGE_ID = "M5_5G7B_K1_SUPERVISED_MULTITASK_AND_HIERARCHICAL_OBSERVATION_SELECTION_v1"
DEVELOPMENT_SCOPE = "SINGLE_MATCH_GROUPED_DEVELOPMENT_ONLY"
BASELINE_COMMIT = "5aa841dd8107ebc5a2f2bb50831d3ed2c326bed9"
EXPECTED_ORIGIN = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
PASS_CLASSIFICATION = "PASS_K1_SUPERVISED_HIERARCHICAL_REASONER_READY_FOR_PRO_REVIEW"

WORKSPACE_DIRECTORIES = (
    "00_PROMPT_AND_INPUTS",
    "01_G7A_AND_K1_VALIDATION",
    "02_K1_TARGET_BINDING_AND_DATA_JOIN",
    "03_RETRAINING_DATASET",
    "04_FROZEN_ENCODER_AND_FEATURE_REUSE",
    "05_MULTITASK_NODE_MODELS",
    "06_PAIRWISE_DUPLICATE_MERGE_MODELS",
    "07_HIERARCHICAL_CLUSTERING_AND_SELECTION",
    "08_GROUPED_OUT_OF_FOLD_EVALUATION",
    "09_ROLE_TEAM_KIT_PARTICIPATION_EVALUATION",
    "10_CALIBRATION_AND_SELECTIVE_ROUTING",
    "11_ERROR_ANALYSIS_AND_VISUAL_QA",
    "12_DEVELOPMENT_SHORTLIST",
    "13_NEXT_STAGE_DECISION",
    "14_COMMANDS_AND_TESTS",
    "15_REVIEW_PACK_FOR_CHATGPT",
    "_tmp",
)

EXPECTED_K1_DISTRIBUTIONS: dict[str, dict[str, int]] = {
    "role": {
        "OUTFIELD_PLAYER": 91,
        "GOALKEEPER": 8,
        "REFEREE": 5,
        "OTHER_MATCH_OFFICIAL": 8,
        "STAFF_OR_SPECTATOR": 12,
        "UNKNOWN_ROLE": 4,
    },
    "team_affiliation": {"TEAM_1": 29, "TEAM_2": 37, "NO_TEAM": 21, "UNKNOWN_TEAM": 41},
    "kit_state": {
        "MATCH_OUTFIELD_KIT": 58,
        "MATCH_GOALKEEPER_KIT": 8,
        "WARMUP_OR_BIB": 33,
        "OFFICIAL_KIT": 13,
        "STAFF_OR_SPECTATOR_CLOTHING": 12,
        "UNKNOWN_KIT": 4,
    },
    "pitch_state": {"ON_PITCH": 70, "OFF_PITCH": 57, "BOUNDARY_UNCERTAIN": 1},
    "participation_state": {
        "ACTIVE_ON_PITCH": 71,
        "OFF_PITCH_SUBSTITUTE_OR_WARMING": 33,
        "OFF_PITCH_NON_PLAYER": 22,
        "UNKNOWN_PARTICIPATION": 2,
    },
    "certainty": {"CERTAIN": 128},
}

EXPECTED_INPUT_HASHES = {
    "k1_bundle_manifest": "281886eca5e471d93f4ee8a0b573ce082945de82d1ebf21744ad281209ace9cb",
    "k1_decisions": "833a517d2a57a7dd9654a2588bf65eb3bb0bdeef8d923b14db4e9c6d62edfed7",
    "k1_case_manifest": "e9c0aacb1ae15aab09a70e869fb7cac7177d5c40d41a81c10bd1a161a8f50512",
    "k1_selection_specification": "7eb187c6c18b016132a7717e5bf6c83663fa376003db3ca7ae96740ff006fb94",
    "g7a_dataset_manifest": "99d80fcbc2292ec792f87db249b383b756ec48c1f7f32da9adf5b2072aecf707",
    "g7a_node_rows": "c24bfc9837521d028e92b17026a6f72f80741307b1227f817aecd095862775fe",
    "g7a_edge_rows": "4e70f28d6e4bb9e89aa84e74d31b337e470788d79891c70075a03f3a6f17c84e",
    "g7a_scene_rows": "ac20e6572ec98bc56b83fadfc5925f0b3bf646af419a199547be832d812d2ca8",
    "g7a_grouped_split": "7e9c45953f16ae427a41674ee9dd4fb3c5ab5b46142d3c32ab51459a726a10b3",
    "g7a_pair_sampling": "3f1522688bec14160cee662ca4759542f41bc5e6962ab4a2229a26c4883f290c",
    "g7a_encoder_provenance": "d9ec3d8085608f4dbe8fc64e5dd81599f2f32da22c9a06820f81e604b250c84a",
    "g7a_embedding_cache": "a21dc16e2d5ba3060fa9fed5ed9fc4dd037e4d767d2da8594e6c0f73b95c5dce",
    "g7a_feature_cache_manifest": "a3824ac5dc1c859952e6887c5e62b3196dbd4933e5ff68009c4ad29f53278f15",
    "g7a_model_results": "0040210f7fd4b654e58d0c5db3fb93c79ea7c424d395a856fad4aebcc24eb0f1",
    "g7a_ablations": "ba7b857e07eea7d6491940296f76c92d9b3c97aa5f6d9edc5608b479ad44ab66",
    "g7a_model_training_specification": "8f305b28d0c90caad3bd126dd5ecc78d57603f36c584f9bcf576140266d88d8e",
    "g7a_model_weight_manifest": "0a96737a2cde6f6b5ead5802d94fa38d14291fa23a31736c6249b93ed5037ac6",
    "g7a_pair_relation_results": "8ad85118d13e2923cbd9863ac917322edec6c851d44d5f620ca076631f60c9dc",
    "g7a_evaluator_person_denominator": "714d136099c69e3c3edb36b93195efea4d84995287c29dc09a91cbe3314372f3",
    "g7a_stage_artifact_manifest": "248565548e27716c78ecebb355c8c2dc5b82efa56f04e8606c09c4b6f7fdec5d",
}

ROLE_CLASSES = (
    "OUTFIELD_PLAYER",
    "GOALKEEPER",
    "REFEREE",
    "OTHER_MATCH_OFFICIAL",
    "STAFF_OR_SPECTATOR",
    "UNKNOWN_ROLE",
)
TEAM_CLASSES = ("TEAM_1", "TEAM_2", "NO_TEAM", "UNKNOWN_TEAM")
KIT_CLASSES = (
    "MATCH_OUTFIELD_KIT",
    "MATCH_GOALKEEPER_KIT",
    "WARMUP_OR_BIB",
    "OFFICIAL_KIT",
    "STAFF_OR_SPECTATOR_CLOTHING",
    "UNKNOWN_KIT",
)
PITCH_CLASSES = ("ON_PITCH", "OFF_PITCH", "BOUNDARY_UNCERTAIN")
PARTICIPATION_CLASSES = (
    "ACTIVE_ON_PITCH",
    "OFF_PITCH_SUBSTITUTE_OR_WARMING",
    "OFF_PITCH_NON_PLAYER",
    "UNKNOWN_PARTICIPATION",
)
CANDIDATE_CLASSES = (
    "CLEAN_INDEPENDENT_PERSON",
    "DUPLICATE_OF_PERSON",
    "MERGED_MULTIPLE_PEOPLE",
    "PARTIAL_PERSON",
    "BACKGROUND",
    "AMBIGUOUS_UNRESOLVED",
)
HEAD_CLASSES = {
    "candidate_state": CANDIDATE_CLASSES,
    "role": ROLE_CLASSES,
    "team": TEAM_CLASSES,
    "kit": KIT_CLASSES,
    "pitch": PITCH_CLASSES,
    "participation": PARTICIPATION_CLASSES,
}


@dataclass(frozen=True)
class StageLocations:
    repo: Path
    prompt_pack: Path
    g7a: Path
    k1_completion: Path
    workspace: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix() if root is not None else str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def tree_records(root: Path, *, excluded_relative_paths: Iterable[str] = ()) -> list[dict[str, Any]]:
    excluded = {Path(value).as_posix() for value in excluded_relative_paths}
    return [
        file_record(path, root=root)
        for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix())
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    ]


def tree_hash(records: Sequence[Mapping[str, Any]]) -> str:
    return stable_hash([{"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]} for row in records])


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def validate_prompt_pack(prompt_pack: Path) -> dict[str, Any]:
    manifest_path = prompt_pack / "10_PROMPT_PACK_MANIFEST.json"
    manifest = read_json(manifest_path)
    failures = []
    declared = manifest.get("files", [])
    for row in declared:
        path = prompt_pack / str(row["filename"])
        if not path.is_file():
            failures.append({"path": str(path), "failure": "MISSING"})
            continue
        actual = file_record(path)
        if actual["bytes"] != int(row["byte_size"]) or actual["sha256"] != row["sha256"]:
            failures.append({"path": str(path), "failure": "HASH_OR_SIZE_MISMATCH", "actual": actual})
    actual_files = sorted(path.name for path in prompt_pack.iterdir() if path.is_file())
    declared_names = sorted([str(row["filename"]) for row in declared] + [manifest_path.name])
    if actual_files != declared_names:
        failures.append(
            {"failure": "PROMPT_PACK_FILE_SET_MISMATCH", "actual": actual_files, "declared": declared_names}
        )
    if manifest.get("minimum_authorized_baseline_commit") != BASELINE_COMMIT:
        failures.append({"failure": "BASELINE_CONTRACT_MISMATCH"})
    return {
        "passed": not failures,
        "manifest": file_record(manifest_path),
        "declared_payload_count": len(declared),
        "actual_file_count": len(actual_files),
        "failures": failures,
    }


def validate_repository(repo: Path) -> dict[str, Any]:
    head = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "branch", "--show-current")
    status = _git(repo, "status", "--porcelain")
    origin = _git(repo, "remote", "get-url", "origin")
    tracking = _git(repo, "rev-parse", "origin/main")
    failures = []
    if head != BASELINE_COMMIT:
        failures.append(f"HEAD {head} != baseline {BASELINE_COMMIT}")
    if branch != "main":
        failures.append(f"branch {branch!r} != 'main'")
    if status:
        # New G7B implementation files are expected while the builder is running.
        allowed_stage_paths = (
            "src/football_intelligence/football_observation_reasoner/g7b_",
            "src/football_intelligence/football_observation_reasoner/hierarchical_selection.py",
            "scripts/build_m5_5g7b_",
            "tests/test_m5_5g7b_",
        )
        unexpected = [
            line
            for line in status.splitlines()
            if not any(value in line.replace("\\", "/") for value in allowed_stage_paths)
        ]
        if unexpected:
            failures.append(f"unexpected pre-existing worktree changes: {unexpected}")
    if origin.rstrip("/") != EXPECTED_ORIGIN.rstrip("/"):
        failures.append(f"origin mismatch: {origin}")
    if tracking != BASELINE_COMMIT:
        failures.append(f"origin/main {tracking} != baseline {BASELINE_COMMIT}")
    ancestors = {}
    for commit in (
        "4b346ddf2209d64b6f13c6a42839f7ec10bc0ebe",
        "98eda1e1c6b3d151bc38782994f7c4c7199ede0a",
        "cbe68a9cd961956603f79319e603a16be6eee1ed",
    ):
        result = subprocess.run(["git", "merge-base", "--is-ancestor", commit, head], cwd=repo, check=False)
        ancestors[commit] = result.returncode == 0
        if not ancestors[commit]:
            failures.append(f"required ancestor missing: {commit}")
    return {
        "passed": not failures,
        "repository": str(repo),
        "head": head,
        "branch": branch,
        "origin": origin,
        "origin_main": tracking,
        "status_porcelain": status,
        "required_ancestors": ancestors,
        "failures": failures,
    }


def _validate_hash(path: Path, expected: str, label: str, failures: list[dict[str, Any]]) -> None:
    actual = sha256_file(path) if path.is_file() else None
    if actual != expected:
        failures.append({"artifact": label, "path": str(path), "expected_sha256": expected, "actual_sha256": actual})


def validate_k1_and_g7a(locations: StageLocations) -> dict[str, Any]:
    g7a = locations.g7a
    k1 = locations.k1_completion
    failures: list[dict[str, Any]] = []
    paths = {
        "k1_bundle_manifest": k1 / "K1_COMPLETION_BUNDLE_MANIFEST.json",
        "k1_decisions": k1 / "02_ACCEPTED_DECISIONS.jsonl",
        "k1_case_manifest": g7a / "03_SUPPLEMENTARY_TEAM_ROLE_KIT_GOLD" / "k1_case_manifest.json",
        "k1_selection_specification": g7a / "03_SUPPLEMENTARY_TEAM_ROLE_KIT_GOLD" / "k1_selection_specification.json",
        "g7a_dataset_manifest": g7a / "05_FOOTBALL_REASONER_DATASET" / "football_reasoner_dataset_manifest.json",
        "g7a_node_rows": g7a / "05_FOOTBALL_REASONER_DATASET" / "football_reasoner_node_rows.parquet",
        "g7a_edge_rows": g7a / "05_FOOTBALL_REASONER_DATASET" / "football_reasoner_edge_rows.parquet",
        "g7a_scene_rows": g7a / "05_FOOTBALL_REASONER_DATASET" / "football_reasoner_scene_rows.parquet",
        "g7a_grouped_split": g7a / "05_FOOTBALL_REASONER_DATASET" / "grouped_split_manifest.json",
        "g7a_pair_sampling": g7a / "05_FOOTBALL_REASONER_DATASET" / "pair_sampling_manifest.json",
        "g7a_encoder_provenance": g7a
        / "04_FROZEN_PRETRAINED_ENCODER_PROVENANCE"
        / "frozen_visual_encoder_provenance.json",
        "g7a_embedding_cache": g7a / "_tmp" / "embeddings" / "official_resnet18_candidate_embeddings.pt",
        "g7a_feature_cache_manifest": g7a / "07_VISUAL_AND_GEOMETRY_FEATURES" / "feature_cache_manifest.json",
        "g7a_model_results": g7a / "09_MODEL_VARIANTS_AND_TRAINING" / "model_variant_results.json",
        "g7a_ablations": g7a / "10_GROUPED_DEVELOPMENT_EVALUATION" / "ablation_results.json",
        "g7a_model_training_specification": g7a
        / "09_MODEL_VARIANTS_AND_TRAINING"
        / "model_training_specification.json",
        "g7a_model_weight_manifest": g7a / "09_MODEL_VARIANTS_AND_TRAINING" / "model_weight_manifest.json",
        "g7a_pair_relation_results": g7a / "10_GROUPED_DEVELOPMENT_EVALUATION" / "pair_relation_results.json",
        "g7a_evaluator_person_denominator": g7a / "05_FOOTBALL_REASONER_DATASET" / "evaluator_person_denominator.json",
        "g7a_stage_artifact_manifest": g7a / "14_COMMANDS_AND_TESTS" / "stage_artifact_manifest.json",
    }
    for label, expected in EXPECTED_INPUT_HASHES.items():
        _validate_hash(paths[label], expected, label, failures)

    stage_manifest = read_json(paths["g7a_stage_artifact_manifest"])
    stage_records = list(stage_manifest.get("files") or [])
    stage_record_failures = []
    for row in stage_records:
        path = g7a / str(row["path"])
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            stage_record_failures.append(str(row["path"]))
    if (
        len(stage_records) != 635
        or stage_manifest.get("payload_tree_hash") != "c4814352bea5bbd4e56457e8bbecc62d3f89c3458a841ccb71e2100f1e3ca861"
        or tree_hash(stage_records) != stage_manifest.get("payload_tree_hash")
        or stage_record_failures
    ):
        failures.append(
            {
                "artifact": "g7a_stage_artifact_manifest",
                "failure": "STAGE_PAYLOAD_TREE_MISMATCH",
                "record_failures": stage_record_failures,
            }
        )

    bundle = read_json(paths["k1_bundle_manifest"])
    if bundle.get("payload_tree_hash") != "27f04937b155386a7214a2f9600fef238a5ea05fe54444f5f62ca4db79615a58":
        failures.append({"artifact": "k1_bundle_manifest", "failure": "PAYLOAD_TREE_HASH_MISMATCH"})
    for row in bundle.get("files", []):
        path = k1 / str(row["path"])
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            failures.append({"artifact": "k1_bundle_payload", "path": str(path), "failure": "HASH_OR_SIZE_MISMATCH"})

    decisions = read_jsonl(paths["k1_decisions"])
    case_manifest = read_json(paths["k1_case_manifest"])
    cases = {str(row["case_id"]): row for row in case_manifest.get("cases", [])}
    decision_ids = [str(row.get("case_id", "")) for row in decisions]
    if len(decisions) != 128 or len(set(decision_ids)) != 128 or set(decision_ids) != set(cases):
        failures.append({"artifact": "k1_decisions", "failure": "CASE_COVERAGE_NOT_EXACTLY_128"})
    distributions = {
        axis: Counter(str(row["annotation"][axis]) for row in decisions) for axis in EXPECTED_K1_DISTRIBUTIONS
    }
    for axis, expected in EXPECTED_K1_DISTRIBUTIONS.items():
        if dict(distributions[axis]) != expected:
            failures.append(
                {
                    "artifact": "k1_decisions",
                    "failure": "DISTRIBUTION_MISMATCH",
                    "axis": axis,
                    "actual": distributions[axis],
                }
            )
    for row in decisions:
        ack = row.get("server_acknowledgement", {})
        if ack.get("event_type") != "K1_CASE_SAVED" or ack.get("event_sequence") != row.get(
            "final_server_event_sequence"
        ):
            failures.append({"case_id": row.get("case_id"), "failure": "FINAL_SERVER_ACK_INVALID"})
        case = cases.get(str(row.get("case_id")), {})
        annotation = row.get("annotation", {})
        for key, decision_key in (
            ("source_frame_sha256", "source_frame_sha256"),
            ("target_binding_sha256", "target_binding_sha256"),
            ("target_crop_sha256", "target_crop_sha256"),
        ):
            if annotation.get(decision_key) != case.get(key):
                failures.append({"case_id": row.get("case_id"), "failure": f"{key.upper()}_JOIN_MISMATCH"})
    warmups = [row for row in decisions if row["annotation"]["kit_state"] == "WARMUP_OR_BIB"]
    if len(warmups) != 33 or any(
        row["annotation"]["role"] != "OUTFIELD_PLAYER" or row["annotation"]["team_affiliation"] != "UNKNOWN_TEAM"
        for row in warmups
    ):
        failures.append({"artifact": "k1_decisions", "failure": "WARMUP_INVARIANT_FAILED"})
    keepers = [row for row in decisions if row["annotation"]["role"] == "GOALKEEPER"]
    keeper_teams = Counter(row["annotation"]["team_affiliation"] for row in keepers)
    if keeper_teams != {"TEAM_1": 4, "TEAM_2": 4}:
        failures.append({"artifact": "k1_decisions", "failure": "GOALKEEPER_TEAM_SPLIT_FAILED", "actual": keeper_teams})

    dataset_manifest = read_json(paths["g7a_dataset_manifest"])
    grouped_split = read_json(paths["g7a_grouped_split"])
    pair_sampling = read_json(paths["g7a_pair_sampling"])
    if dataset_manifest.get("dataset_hash") != "147bb0754cf1cafe05dfec3425afc06dc11becb3de12c0657617382c2d220bfd":
        failures.append({"artifact": "g7a_dataset_manifest", "failure": "SEMANTIC_DATASET_HASH_MISMATCH"})
    if grouped_split.get("manifest_hash") != "40a240fad085e62171553faccfe403373c5e9c172dd24ed03736e302dc31e243":
        failures.append({"artifact": "g7a_grouped_split", "failure": "SEMANTIC_SPLIT_HASH_MISMATCH"})
    if pair_sampling.get("manifest_hash") != "29f3bb734e8f35d30e70783593143ed6b811ad18ce47675d1899d7b1aa299cc3":
        failures.append({"artifact": "g7a_pair_sampling", "failure": "SEMANTIC_PAIR_HASH_MISMATCH"})
    leakage = grouped_split.get("leakage_checks", {})
    if not leakage.get("passed") or any(
        leakage.get(key) != 0
        for key in (
            "lineage_cross_fold_count",
            "positive_edge_cross_fold_count",
            "source_frame_cross_fold_count",
            "source_group_cross_fold_count",
        )
    ):
        failures.append({"artifact": "g7a_grouped_split", "failure": "LEAKAGE_CHECK_FAILED", "actual": leakage})

    return {
        "schema_version": "football_intelligence.m5_5g7b.input_validation.v1",
        "stage_id": STAGE_ID,
        "passed": not failures,
        "acceptance_blocked": bool(failures),
        "k1": {
            "accepted_decision_count": len(decisions),
            "unique_case_count": len(set(decision_ids)),
            "completion_transaction_id": bundle.get("completion_transaction_id"),
            "decision_state_hash": "92f90bbdb7fa194f7d10fedeb6e53d0ee34c7064bcdd8b87cd55bd166f55c274",
            "payload_tree_hash": bundle.get("payload_tree_hash"),
            "distributions": {key: dict(value) for key, value in distributions.items()},
            "warmup_count": len(warmups),
            "goalkeeper_team_counts": dict(keeper_teams),
            "candidate_state_collected": False,
        },
        "g7a": {
            "node_count": dataset_manifest.get("counts", {}).get("node_rows", 2812),
            "edge_count": dataset_manifest.get("counts", {}).get("edge_rows", 24566),
            "scene_count": dataset_manifest.get("counts", {}).get("scene_rows", 49),
            "fold_count": grouped_split.get("fold_count"),
            "dataset_hash": dataset_manifest.get("dataset_hash"),
            "grouped_split_hash": grouped_split.get("manifest_hash"),
            "pair_sampling_hash": pair_sampling.get("manifest_hash"),
            "leakage_checks": leakage,
        },
        "validated_artifacts": {label: file_record(path) for label, path in paths.items()},
        "failures": failures,
    }


def create_workspace_layout(locations: StageLocations) -> dict[str, Path]:
    paths = {name: locations.workspace / name for name in WORKSPACE_DIRECTORIES}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def bbox_iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    ix1 = max(float(left["x1"]), float(right["x1"]))
    iy1 = max(float(left["y1"]), float(right["y1"]))
    ix2 = min(float(left["x2"]), float(right["x2"]))
    iy2 = min(float(left["y2"]), float(right["y2"]))
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, float(left["x2"]) - float(left["x1"])) * max(0.0, float(left["y2"]) - float(left["y1"]))
    right_area = max(0.0, float(right["x2"]) - float(right["x1"])) * max(0.0, float(right["y2"]) - float(right["y1"]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def build_source_group_folds(node_rows: Sequence[Mapping[str, Any]], split: Mapping[str, Any]) -> dict[str, int]:
    assignment = split.get("assignment_by_example_uuid", {})
    by_group: dict[str, set[int]] = defaultdict(set)
    for row in node_rows:
        by_group[str(row["source_group_id"])].add(int(assignment[str(row["example_uuid"])]))
    invalid = {group: sorted(folds) for group, folds in by_group.items() if len(folds) != 1}
    if invalid:
        raise ValueError(f"G7A source groups cross folds: {invalid}")
    return {group: next(iter(folds)) for group, folds in by_group.items()}


def build_k1_join(
    decisions: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    node_rows: Sequence[Mapping[str, Any]],
    source_group_folds: Mapping[str, int],
    *,
    minimum_iou: float = 0.8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if minimum_iou <= 0.0 or minimum_iou > 1.0:
        raise ValueError("minimum_iou must be in (0, 1]")
    decision_by_case = {str(row["case_id"]): row for row in decisions}
    nodes_by_frame: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for node in node_rows:
        nodes_by_frame[str(node["source_frame_sha256"])].append(node)
    ledger = []
    authoritative_rows = []
    propagation_rows = []
    for case in sorted(cases, key=lambda row: str(row["case_id"])):
        case_id = str(case["case_id"])
        decision = decision_by_case[case_id]
        annotation = dict(decision["annotation"])
        group = str(case["source_group_id"])
        target_box = dict(case["target"]["bbox_original_pixels"])
        candidates = []
        for node in nodes_by_frame[str(case["source_frame_sha256"])]:
            overlap = bbox_iou(target_box, node["visible_box"])
            if overlap > 0.0:
                candidates.append((overlap, node))
        candidates.sort(key=lambda value: (-value[0], str(value[1]["example_uuid"])))
        accepted = []
        rejected = []
        for overlap, node in candidates:
            state = node.get("candidate_state_target")
            gold_person_ids = list(node.get("gold_person_ids") or [])
            reasons = []
            if overlap < minimum_iou:
                reasons.append("IOU_BELOW_FROZEN_BINDING_THRESHOLD")
            if state in {"MERGED_MULTIPLE_PEOPLE", "BACKGROUND"}:
                reasons.append("PRIOR_MERGED_OR_BACKGROUND")
            elif state not in {"CLEAN_INDEPENDENT_PERSON", "PARTIAL_PERSON", "DUPLICATE_OF_PERSON"}:
                reasons.append("PRIOR_STATE_NOT_AUTHORIZED_FOR_PERSON_AXIS_PROPAGATION")
            if len(gold_person_ids) > 1:
                reasons.append("MULTIPLE_EVALUATOR_PEOPLE_CONTAINED")
            record = {
                "example_uuid": str(node["example_uuid"]),
                "candidate_uuid": str(node["candidate_uuid"]),
                "iou": overlap,
                "prior_candidate_state": state,
                "gold_person_id_count": len(gold_person_ids),
            }
            if reasons:
                rejected.append({**record, "reasons": reasons})
            else:
                accepted.append(record)
                propagation_rows.append(
                    {
                        "schema_version": "football_intelligence.m5_5g7b.k1_propagation.v1",
                        "case_id": case_id,
                        "example_uuid": str(node["example_uuid"]),
                        "source_group_id": group,
                        "fold": int(source_group_folds[group]),
                        "candidate_state_target": state,
                        "candidate_state_source": "PRIOR_G7A_ONLY",
                        "candidate_state_from_k1": False,
                        "role": annotation["role"],
                        "team": annotation["team_affiliation"],
                        "kit": annotation["kit_state"],
                        "pitch": annotation["pitch_state"],
                        "participation": annotation["participation_state"],
                        "certainty_used_as_target": False,
                        "iou": overlap,
                    }
                )
        crop_path = None
        authoritative_rows.append(
            {
                "schema_version": "football_intelligence.m5_5g7b.k1_person_row.v1",
                "case_id": case_id,
                "example_uuid": f"g7b_{case_id}",
                "source_group_id": group,
                "source_frame_sha256": str(case["source_frame_sha256"]),
                "target_binding_sha256": str(case["target_binding_sha256"]),
                "target_crop_sha256": str(case["target_crop_sha256"]),
                "target_bbox_source_pixels": target_box,
                "fold": int(source_group_folds[group]),
                "candidate_state_target": None,
                "role_target": annotation["role"],
                "team_target": annotation["team_affiliation"],
                "kit_target": annotation["kit_state"],
                "pitch_target": annotation["pitch_state"],
                "participation_target": annotation["participation_state"],
                "certainty": annotation["certainty"],
                "certainty_target_available": False,
                "candidate_state_target_available": False,
                "human_identity_label_available": False,
                "temporal_target_available": False,
                "crop_path": crop_path,
            }
        )
        ledger.append(
            {
                "schema_version": "football_intelligence.m5_5g7b.k1_join_ledger.v1",
                "case_id": case_id,
                "source_group_id": group,
                "fold": int(source_group_folds[group]),
                "source_frame_sha256": case["source_frame_sha256"],
                "target_binding_sha256": case["target_binding_sha256"],
                "target_crop_sha256": case["target_crop_sha256"],
                "server_acknowledged": True,
                "authoritative_person_row_created": True,
                "candidate_state_inferred_from_k1": False,
                "accepted_candidate_bindings": accepted,
                "rejected_candidate_bindings": rejected,
            }
        )
    propagated_by_case = Counter(row["case_id"] for row in propagation_rows)
    for row in propagation_rows:
        row["case_normalized_supervision_weight"] = 1.0 / propagated_by_case[row["case_id"]]
    summary = {
        "schema_version": "football_intelligence.m5_5g7b.k1_propagation_summary.v1",
        "frozen_minimum_iou": minimum_iou,
        "authoritative_k1_person_rows": len(authoritative_rows),
        "cases_with_candidate_propagation": len(propagated_by_case),
        "candidate_propagation_rows": len(propagation_rows),
        "prior_clean_propagation_rows": sum(
            row["candidate_state_target"] == "CLEAN_INDEPENDENT_PERSON" for row in propagation_rows
        ),
        "prior_partial_propagation_rows": sum(
            row["candidate_state_target"] == "PARTIAL_PERSON" for row in propagation_rows
        ),
        "prior_duplicate_propagation_rows": sum(
            row["candidate_state_target"] == "DUPLICATE_OF_PERSON" for row in propagation_rows
        ),
        "merged_or_background_propagation_rows": sum(
            row["candidate_state_target"] in {"MERGED_MULTIPLE_PEOPLE", "BACKGROUND"} for row in propagation_rows
        ),
        "candidate_state_values_created_from_k1": 0,
        "certainty_targets_created": 0,
        "identity_or_temporal_targets_created": 0,
        "propagation_rows": propagation_rows,
    }
    return ledger, authoritative_rows, summary


def node_tabular_features(row: Mapping[str, Any]) -> np.ndarray:
    """Return the fixed 32-value node vector shared with K1 target crops.

    Polygon membership and pitch-distance values are intentionally absent, so
    the participation branch cannot learn polygon membership as a shortcut.
    """

    coordinates = row.get("source_coordinates") or {}
    box = row.get("visible_box") or {}
    image_width = max(float(coordinates.get("image_width", 2730.0)), 1.0)
    image_height = max(float(coordinates.get("image_height", 720.0)), 1.0)
    width = max(float(box.get("x2", 0.0)) - float(box.get("x1", 0.0)), 1e-6)
    height = max(float(box.get("y2", 0.0)) - float(box.get("y1", 0.0)), 1e-6)
    geometry = [
        float(coordinates.get("centre_x_normalized", 0.0)),
        float(coordinates.get("centre_y_normalized", 0.0)),
        width / image_width,
        height / image_height,
        width / height,
        math.log1p(width * height) / 16.0,
        float(box.get("x1", 0.0)) / image_width,
        float(box.get("y1", 0.0)) / image_height,
    ]
    provenance = row.get("proposal_provenance_features") or {}
    proposal = [
        float(provenance.get("score", row.get("score", 0.0)) or 0.0),
        float(provenance.get("lineage_depth", 0.0) or 0.0) / 8.0,
        float(provenance.get("cross_view_corroboration_count", 0.0) or 0.0) / 8.0,
        float(provenance.get("duplicate_cluster_size", 0.0) or 0.0) / 8.0,
        float(bool(provenance.get("stage_is_fused", False))),
        float(bool(provenance.get("stage_is_raw", False))),
        float(bool(provenance.get("stage_is_post_nms", False))),
        float(bool(provenance.get("nms_survived", False))),
    ]
    scale = row.get("expected_scale_features") or {}
    perspective_scale = [
        float(scale.get("height_z_score", 0.0) or 0.0) / 8.0,
        float(scale.get("width_z_score", 0.0) or 0.0) / 8.0,
        float(scale.get("aspect_z_score", 0.0) or 0.0) / 8.0,
        float(scale.get("plausible_scale_probability", 0.0) or 0.0),
        float(bool((row.get("shape_features") or {}).get("small_far_side", False))),
        float(scale.get("uncertainty_multiplier", 0.0) or 0.0) / 8.0,
    ]
    shape = row.get("shape_features") or {}
    shape_values = [
        float(shape.get("blur_evidence", 0.0) or 0.0),
        float(shape.get("truncation_flag_count", 0.0) or 0.0) / 4.0,
    ]
    colour = list((row.get("colour_kit_features") or {}).get("spatial_colour_layout_rgb") or [])[:8]
    colour = [float(value) for value in colour] + [0.0] * (8 - len(colour))
    values = np.asarray(geometry + proposal + perspective_scale + shape_values + colour, dtype=np.float32)
    if values.shape != (32,) or not np.isfinite(values).all():
        raise ValueError("node tabular feature vector must contain 32 finite values")
    return values


def k1_crop_features(path: Path, bbox: Mapping[str, Any]) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height = max(float(bbox["y2"]) - float(bbox["y1"]), 1e-6)
    width = max(float(bbox["x2"]) - float(bbox["x1"]), 1e-6)
    from football_intelligence.football_observation_reasoner.features import colour_kit_evidence_features

    # Use the exact same spatial-colour feature definition as G7A candidate
    # nodes; a different K1-only colour statistic would encode row provenance.
    colour_source = colour_kit_evidence_features(rgb)["spatial_colour_layout_rgb"][:8]
    centre_x = (float(bbox["x1"]) + float(bbox["x2"])) / 2.0
    centre_y = (float(bbox["y1"]) + float(bbox["y2"])) / 2.0
    geometry = [
        centre_x / 2730.0,
        centre_y / 720.0,
        width / 2730.0,
        height / 720.0,
        width / height,
        math.log1p(width * height) / 16.0,
        float(bbox["x1"]) / 2730.0,
        float(bbox["y1"]) / 720.0,
    ]
    # K1 targets are authoritative person rows, not detector proposals.  Their
    # proposal and fitted-perspective features therefore remain explicitly 0.
    values = np.asarray(geometry + [0.0] * 16 + list(colour_source), dtype=np.float32)
    if values.shape != (32,) or not np.isfinite(values).all():
        raise ValueError("K1 crop feature vector must contain 32 finite values")
    return values


def macro_metrics(
    targets: Sequence[str],
    predictions: Sequence[str],
    classes: Sequence[str],
    probabilities: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have equal length")
    confusion = {truth: {guess: 0 for guess in classes} for truth in classes}
    for truth, guess in zip(targets, predictions, strict=True):
        confusion[str(truth)][str(guess)] += 1
    classwise = {}
    f1_values = []
    for value in classes:
        tp = confusion[value][value]
        fp = sum(confusion[truth][value] for truth in classes if truth != value)
        fn = sum(confusion[value][guess] for guess in classes if guess != value)
        support = sum(confusion[value].values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        classwise[value] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)
    result: dict[str, Any] = {
        "denominator": len(targets),
        "correct": sum(left == right for left, right in zip(targets, predictions, strict=True)),
        "accuracy": sum(left == right for left, right in zip(targets, predictions, strict=True)) / len(targets)
        if targets
        else None,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "classwise": classwise,
        "confusion_matrix": confusion,
    }
    if probabilities is not None and targets:
        class_index = {value: index for index, value in enumerate(classes)}
        vectors = np.asarray(probabilities, dtype=np.float64)
        truth_matrix = np.zeros_like(vectors)
        for index, target in enumerate(targets):
            truth_matrix[index, class_index[target]] = 1.0
        result["multiclass_brier"] = float(np.mean(np.sum((vectors - truth_matrix) ** 2, axis=1)))
        confidence = vectors.max(axis=1)
        correct = np.asarray(
            [left == right for left, right in zip(targets, predictions, strict=True)], dtype=np.float64
        )
        bins = []
        ece = 0.0
        for bin_index in range(10):
            lower, upper = bin_index / 10.0, (bin_index + 1) / 10.0
            mask = (confidence >= lower) & ((confidence < upper) if bin_index < 9 else (confidence <= upper))
            count = int(mask.sum())
            mean_confidence = float(confidence[mask].mean()) if count else None
            empirical_accuracy = float(correct[mask].mean()) if count else None
            gap = abs(mean_confidence - empirical_accuracy) if count else None
            if count:
                ece += count / len(targets) * float(gap)
            bins.append(
                {
                    "bin": bin_index,
                    "lower_inclusive": lower,
                    "upper": upper,
                    "upper_inclusive": bin_index == 9,
                    "count": count,
                    "mean_confidence": mean_confidence,
                    "empirical_accuracy": empirical_accuracy,
                    "absolute_gap": gap,
                }
            )
        result["expected_calibration_error"] = ece
        result["calibration_bins"] = bins
    return result


def derive_primary_truth(annotation: Mapping[str, Any]) -> str:
    role = str(annotation["role"])
    pitch = str(annotation["pitch_state"])
    participation = str(annotation["participation_state"])
    kit = str(annotation["kit_state"])
    if (
        pitch == "ON_PITCH"
        and participation == "ACTIVE_ON_PITCH"
        and role
        in {
            "OUTFIELD_PLAYER",
            "GOALKEEPER",
            "REFEREE",
            "OTHER_MATCH_OFFICIAL",
        }
    ):
        return "ACTIVE_OBSERVATION"
    if kit == "WARMUP_OR_BIB" or role == "STAFF_OR_SPECTATOR":
        return "OUT_OF_SCOPE_PERSON"
    return "BOUNDARY_OR_PARTICIPATION_UNRESOLVED"


def review_pack_validation(review_dir: Path) -> dict[str, Any]:
    files = sorted(path for path in review_dir.iterdir() if path.is_file())
    records = [file_record(path, root=review_dir) for path in files]
    visuals = [row for row in records if Path(row["path"]).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    forbidden_suffixes = {".pt", ".pth", ".joblib", ".parquet", ".mp4", ".mov", ".avi", ".mkv", ".webm"}
    forbidden = [row["path"] for row in records if Path(row["path"]).suffix.lower() in forbidden_suffixes]
    sensitive_tokens = ("accepted_decisions", "credential", "secret", ".env", "raw_video")
    sensitive = [row["path"] for row in records if any(token in row["path"].lower() for token in sensitive_tokens)]
    total = sum(int(row["bytes"]) for row in records)
    manifest_path = review_dir / "REVIEW_PACK_MANIFEST.json"
    manifest_checks = {
        "present": manifest_path.is_file(),
        "payload_file_set_exact": False,
        "payload_file_count_exact": False,
        "payload_total_bytes_exact": False,
        "payload_tree_hash_exact": False,
    }
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        payload_records = [row for row in records if row["path"] != manifest_path.name]
        declared_records = list(manifest.get("files") or [])
        manifest_checks.update(
            {
                "payload_file_set_exact": declared_records == payload_records,
                "payload_file_count_exact": int(manifest.get("payload_file_count", -1)) == len(payload_records),
                "payload_total_bytes_exact": int(manifest.get("payload_total_bytes", -1))
                == sum(int(row["bytes"]) for row in payload_records),
                "payload_tree_hash_exact": manifest.get("payload_tree_hash") == tree_hash(payload_records),
            }
        )
    checks = {
        "flat": all("/" not in row["path"] for row in records),
        "file_count_at_most_20": len(records) <= 20,
        "total_bytes_at_most_50_mib": total <= 52_428_800,
        "visual_count_at_most_3": len(visuals) <= 3,
        "source_diff_present": (review_dir / "04_SOURCE_DIFF.patch").is_file(),
        "forbidden_binary_artifacts_absent": not forbidden,
        "sensitive_or_raw_artifacts_absent": not sensitive,
        "manifest_and_payload_hashes_valid": all(manifest_checks.values()),
    }
    return {
        "schema_version": "football_intelligence.m5_5g7b.review_pack_validation.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "file_count": len(records),
        "total_bytes": total,
        "visual_count": len(visuals),
        "forbidden_files": forbidden,
        "sensitive_files": sensitive,
        "manifest_checks": manifest_checks,
        "files": records,
    }


def artifact_manifest(workspace: Path, *, excluded_paths: Iterable[str] = ()) -> dict[str, Any]:
    records = tree_records(workspace, excluded_relative_paths=excluded_paths)
    return {
        "schema_version": "football_intelligence.m5_5g7b.stage_artifact_manifest.v1",
        "stage_id": STAGE_ID,
        "development_scope": DEVELOPMENT_SCOPE,
        "file_count": len(records),
        "files": records,
        "payload_tree_hash": tree_hash(records),
        "production_promoted": False,
        "identity_tracking_performed": False,
        "temporal_predictions_created": False,
        "count_prior_used": False,
    }
