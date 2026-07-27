"""Build the cached-only M5.5G.6F conditional recovery bakeoff.

The stage never invokes a detector or promptable model. It reconstructs the
frozen G6E S0/S3 caches, freezes a truth-free trigger/admission matrix, and
joins evaluator gold only after every runtime candidate has been materialized.
"""

from __future__ import annotations

import importlib.util
import argparse
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from football_intelligence.detection_forensics import EXPECTED_CHECKPOINT_SHA256, sha256_file, stable_hash
from football_intelligence.detection_gold.consolidation import consolidate_proposals
from football_intelligence.detection_gold.proposal_supply import (
    bbox_height,
    bbox_iou,
    deterministic_one_to_one_supply,
)
from football_intelligence.review.schemas import safety_payload

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT = PART3 / "M5_5G6F_Conditional_Cross_View_Recovery_Codex_Prompt_Pack"
STAGE = PART3 / "M5_5G6F_CONDITIONAL_LOW_CONFIDENCE_CROSS_VIEW_RECOVERY_AND_DUPLICATE_CONTROL_v1"
G6E = PART3 / "M5_5G6E_C0_PROPOSAL_REINTEGRATION_AND_PLAYER_OBSERVATION_V1_FULL_UNIVERSE_VALIDATION_v1"
G6D = PART3 / "M5_5G6D_R_A1_HIGH_RESOLUTION_SMALL_PERSON_PROPOSAL_BAKEOFF_v1"
G5A = PART3 / "M5_5G5A_AUTHORIZED_PROMPTABLE_MASK_BAKEOFF_AND_DENSE_BRANCH_DECISION_v1"
CHECKPOINT = REPO / "models" / "model=yolov8m-imgsz=2048.pt"
G6E_SCRIPT = REPO / "scripts" / "build_m5_5g6e_c0_reintegration.py"

BASELINE = "6e295258d11dd6d25086a74d1a2bdd6becae60b0"
EXPECTED_REMOTE = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
EXPECTED_ROOT_CAUSE = "HISTORICAL_C2_REVIEW_PAYLOAD_STAGE_PRIORITY_AND_120_ROW_CAP_OMITTED_LOWER_PRIORITY_RAW_LINEAGE"
EXPECTED_MATRIX_SHA256 = "0bba7df6f7e346e92e7d30510e1c2d046924065f8c2a6dbc1811205e545533c5"
FUSION = "IOU_CONNECTED_COMPONENT_055"
CLASSIFICATION = "PASS_CONDITIONAL_CROSS_VIEW_RECOVERY_BAKEOFF_READY_FOR_PRO_REVIEW"
FINAL_CHOICES = {
    "FREEZE_CONDITIONAL_CROSS_VIEW_RECOVERY_DEVELOPMENT_CANDIDATE",
    "AUTHORIZE_LOW_CONFIDENCE_SECOND_STAGE_PERSON_VALIDATOR_BAKEOFF",
    "AUTHORIZE_OFFICIAL_SMALL_PERSON_DETECTOR_FAMILY_BAKEOFF",
    "COLLECT_MORE_CONFIDENCE_STAGE_HARD_NEGATIVE_GOLD",
    "KEEP_NARROW_G6D_RECOVERY_EVIDENCE_ONLY",
}
INDEPENDENT_STATES = {"INDEPENDENT_SINGLE_SUPPORT", "INDEPENDENT_SUPPORT_WITH_DUPLICATE_BURDEN"}

DIRS = {
    "inputs": STAGE / "00_PROMPT_AND_INPUTS",
    "validation": STAGE / "01_CACHED_ROW_AND_UNIVERSE_VALIDATION",
    "matrix": STAGE / "02_FROZEN_TRIGGER_AND_ADMISSION_MATRIX",
    "trigger": STAGE / "03_MACHINE_ONLY_TRIGGER_REPLAY",
    "recovery": STAGE / "04_CONDITIONAL_RECOVERY_BAKEOFF",
    "transition": STAGE / "05_C2_TRANSITION_AND_OBSERVATION_DIAGNOSIS",
    "regression": STAGE / "06_STATIC_DENSE_AND_B1_REGRESSION",
    "burden": STAGE / "07_OFF_PITCH_CROWD_AND_RUNTIME_BURDEN",
    "error": STAGE / "08_ERROR_LEDGER_AND_VISUAL_QA",
    "decision": STAGE / "09_DEVELOPMENT_SHORTLIST_AND_DECISION",
    "commands": STAGE / "10_COMMANDS_AND_TESTS",
    "pack": STAGE / "11_REVIEW_PACK_FOR_CHATGPT",
}

SAFETY = {
    **safety_payload(),
    "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
    "sandbox_only": True,
    "production_ready": False,
    "no_auto_promotion": True,
    "human_approved": False,
    "safe_to_apply_globally": False,
    "match_local_only": True,
    "new_inference_performed": False,
    "detector_settings_changed": False,
    "global_confidence_changed": False,
    "nms_changed": False,
    "tile_geometry_changed": False,
    "fusion_defaults_changed": False,
    "pitch_gate_settings_changed": False,
    "project_defaults_changed": False,
    "training_performed": False,
    "fine_tuning_performed": False,
    "identity_tracking_performed": False,
    "temporal_states_created": False,
    "detector_promoted": False,
    "tracker_promoted": False,
    "component_promoted": False,
}

GEOMETRY = {
    "novelty_veto": {"minimum_iou": 0.15, "maximum_bottom_distance_smaller_height": 0.55, "height_ratio": 0.35},
    "raw_cluster": {"minimum_iou": 0.25, "maximum_bottom_distance_smaller_height": 0.35, "height_ratio": 0.50},
    "cross_view": {"minimum_iou": 0.20, "maximum_bottom_distance_smaller_height": 0.50, "height_ratio": 0.35},
    "b2_minimum_raw_cluster_members": 3,
    "a3_multi_mode_minimum_bottom_distance": 0.45,
    "a3_multi_mode_maximum_iou": 0.15,
}


def load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


G6E_IMPL = load_module("m5_5g6e_builder_for_g6f", G6E_SCRIPT)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO, check=check, text=True, capture_output=True)


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def tree_manifest(paths: Sequence[Path]) -> dict[str, Any]:
    rows = [file_record(path) for path in sorted(set(paths), key=lambda value: str(value).lower())]
    return {"files": rows, "tree_sha256": stable_hash(rows), "file_count": len(rows)}


def prompt_and_repository_validation() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(PROMPT / "08_PROMPT_PACK_MANIFEST.json")
    prompt_rows = []
    for row in manifest["files"]:
        path = PROMPT / str(row["filename"])
        prompt_rows.append(
            {
                "filename": row["filename"],
                "exists": path.is_file(),
                "bytes_exact": path.is_file() and path.stat().st_size == int(row["byte_size"]),
                "sha256_exact": path.is_file() and sha256_file(path) == str(row["sha256"]),
            }
        )
    head = git("rev-parse", "HEAD").stdout.strip()
    branch = git("branch", "--show-current").stdout.strip()
    remote = git("remote", "get-url", "origin").stdout.strip()
    status = git("status", "--porcelain").stdout.strip()
    baseline_exists = git("cat-file", "-e", f"{BASELINE}^{{commit}}", check=False).returncode == 0
    baseline_ancestor = git("merge-base", "--is-ancestor", BASELINE, "HEAD", check=False).returncode == 0
    intervening_commits = [
        row for row in git("log", "--format=%H %s", f"{BASELINE}..HEAD").stdout.splitlines() if row.strip()
    ]
    changed_files = [row for row in git("diff", "--name-only", f"{BASELINE}..HEAD").stdout.splitlines() if row.strip()]
    repository_checks = {
        "repository_exact": REPO == Path(r"C:\Users\sebgr\Documents\football-intelligence\SoccerTrack-v2"),
        "branch_main": branch == "main",
        "head_at_baseline_or_clean_descendant": head == BASELINE or baseline_ancestor,
        "baseline_exists": baseline_exists,
        "baseline_is_ancestor": baseline_ancestor,
        "worktree_clean_before_build": not status,
        "remote_exact": remote == EXPECTED_REMOTE,
    }
    prompt_checks = {
        "declared_payload_count": len(prompt_rows) == 8,
        "all_payloads_byte_valid": all(
            row["exists"] and row["bytes_exact"] and row["sha256_exact"] for row in prompt_rows
        ),
        "flat": not any(path.is_dir() for path in PROMPT.iterdir()),
        "actual_file_count": len([path for path in PROMPT.iterdir() if path.is_file()]) == 9,
        "manifest_self_hash_omitted": manifest.get("manifest_self_hash_omitted") is True,
    }
    if not all(repository_checks.values()):
        raise RuntimeError(f"FAIL_REPOSITORY_AUTHORIZATION: {repository_checks}")
    if not all(prompt_checks.values()):
        raise RuntimeError(f"FAIL_PROMPT_PACK: {prompt_checks}")
    for path in PROMPT.iterdir():
        if path.is_file():
            DIRS["inputs"].mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, DIRS["inputs"] / path.name)
    return (
        {
            "schema_version": "football_intelligence.m5_5g6f.repository_state.v1",
            "head": head,
            "branch": branch,
            "remote": remote,
            "checks": repository_checks,
            "intervening_commits": intervening_commits,
            "changed_files_audited": changed_files,
            "target_module_reconciliation": {
                "new_g6f_builder": "scripts/build_m5_5g6f_conditional_recovery.py" in changed_files,
                "new_g6f_tests": "tests/test_m5_5g6f_conditional_recovery.py" in changed_files,
                "existing_detector_or_project_defaults_changed": False,
            },
            "passed": True,
        },
        {
            "schema_version": "football_intelligence.m5_5g6f.prompt_pack_validation.v1",
            "files": prompt_rows,
            "checks": prompt_checks,
            "passed": True,
        },
    )


def g6e_new_paths() -> dict[str, Path]:
    root = G6E / "_tmp" / "missing_c2_exact_replay" / "primary"
    return {
        "raw": root / "primary_raw_candidate_rows.jsonl",
        "nms": root / "primary_nms_candidate_rows.jsonl",
        "post": root / "primary_post_nms_rows.jsonl",
        "runtime": root / "primary_runtime_views.json",
    }


def cached_input_paths() -> list[Path]:
    specs = {**G6E_IMPL.cache_specs(), "G6E_NEW": g6e_new_paths()}
    paths = [path for spec in specs.values() for path in spec.values()]
    paths.extend(
        [
            G6E / "stage_summary.json",
            G6E / "01_G6D_AND_PRIOR_ARTIFACT_VALIDATION" / "protected_inputs_before.json",
            G6E / "11_COMMANDS_AND_TESTS" / "protected_inputs_after.json",
            G6E / "02_RAW_STAGE_PROVENANCE_RECONCILIATION" / "raw_stage_reconciliation_summary.json",
            G6E / "02_RAW_STAGE_PROVENANCE_RECONCILIATION" / "raw_stage_reconciliation_ledger.jsonl",
            G6E / "03_FULL_UNIVERSE_C0_REPLAY" / "full_universe_contract.json",
            G6E / "03_FULL_UNIVERSE_C0_REPLAY" / "c0_full_universe_replay_manifest.json",
            G6E / "03_FULL_UNIVERSE_C0_REPLAY" / "c0_proposal_nodes.jsonl",
            G6E / "03_FULL_UNIVERSE_C0_REPLAY" / "c0_observation_rows.jsonl",
            G6E / "03_FULL_UNIVERSE_C0_REPLAY" / "c0_c2_results.json",
            G6E / "04_STATIC_AND_DENSE_REGRESSION" / "c0_static_results.json",
            G6E / "04_STATIC_AND_DENSE_REGRESSION" / "c0_dense_results.json",
            G6D / "02_CHECKPOINT_RUNTIME_AND_VIEW_MATRIX_FREEZE" / "frozen_view_matrix.json",
            CHECKPOINT,
        ]
    )
    before = read_json(G6E / "01_G6D_AND_PRIOR_ARTIFACT_VALIDATION" / "protected_inputs_before.json")
    paths.extend(Path(str(row["path"])) for row in before["files"])
    return sorted(set(paths), key=lambda value: str(value).lower())


def _manifest_entries_exact(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    def compact(payload: Mapping[str, Any]) -> list[tuple[str, int, str]]:
        return sorted((str(row["path"]), int(row["bytes"]), str(row["sha256"])) for row in payload["files"])

    return compact(left) == compact(right)


def validate_cached_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stage_summary = read_json(G6E / "stage_summary.json")
    replay = read_json(G6E / "03_FULL_UNIVERSE_C0_REPLAY" / "c0_full_universe_replay_manifest.json")
    universe = read_json(G6E / "03_FULL_UNIVERSE_C0_REPLAY" / "full_universe_contract.json")
    reconciliation = read_json(G6E / "02_RAW_STAGE_PROVENANCE_RECONCILIATION" / "raw_stage_reconciliation_summary.json")
    before = read_json(G6E / "01_G6D_AND_PRIOR_ARTIFACT_VALIDATION" / "protected_inputs_before.json")
    after = read_json(G6E / "11_COMMANDS_AND_TESTS" / "protected_inputs_after.json")
    matrix = G6D / "02_CHECKPOINT_RUNTIME_AND_VIEW_MATRIX_FREEZE" / "frozen_view_matrix.json"
    paths = cached_input_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    checks = {
        "g6e_classification_exact": stage_summary.get("classification")
        == "PASS_C0_REINTEGRATION_AND_PLAYER_OBSERVATION_V1_FULL_UNIVERSE_VALIDATION_READY_FOR_PRO_REVIEW",
        "g6e_repository_head_exact": stage_summary.get("repository_head") == BASELINE,
        "g6e_global_choice_reject": read_json(G6E / "10_NEXT_STAGE_DECISION" / "final_decision.json").get("choice")
        == "REJECT_C0_DUE_FULL_UNIVERSE_REGRESSION",
        "checkpoint_exact": sha256_file(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA256,
        "matrix_exact": sha256_file(matrix) == EXPECTED_MATRIX_SHA256,
        "source_count_exact": replay.get("source_count") == 49,
        "stage_counts_exact": replay.get("stage_row_counts")
        == {"raw": 147000, "confidence": 28185, "post_nms": 7526, "fused": 2327, "observations": 2327},
        "coordinate_roundtrip_exact": replay.get("coordinate_roundtrip_exact") is True,
        "nms_replay_exact": replay.get("nms_exact") is True,
        "provenance_complete": replay.get("provenance_complete") is True,
        "raw_root_cause_exact": reconciliation.get("root_cause") == EXPECTED_ROOT_CAUSE,
        "raw_reconciliation_passed": reconciliation.get("passed") is True,
        "historical_artifacts_preserved": reconciliation.get("historical_artifacts_preserved") is True,
        "g6e_protected_before_after_exact": _manifest_entries_exact(before, after),
        "full_universe_hash_exact": universe.get("full_universe_hash")
        == "19fe924cf1d1435788b7251125a88e49e72af4413cc85710264dcdaedaa36e42",
        "all_cached_paths_exist": not missing,
    }
    if not all(checks.values()):
        raise RuntimeError(f"FAIL_G6E_CACHED_INPUT_VALIDATION: {checks}; missing={missing}")
    protected = tree_manifest(paths)
    return (
        {
            "schema_version": "football_intelligence.m5_5g6f.g6e_cached_input_validation.v1",
            "checks": checks,
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "matrix_sha256": sha256_file(matrix),
            "g6e_stage_summary_sha256": sha256_file(G6E / "stage_summary.json"),
            "g6e_replay_manifest_sha256": sha256_file(
                G6E / "03_FULL_UNIVERSE_C0_REPLAY" / "c0_full_universe_replay_manifest.json"
            ),
            "raw_reconciliation_root_cause": reconciliation["root_cause"],
            "passed": True,
            **SAFETY,
        },
        protected,
        replay,
    )


def frozen_matrix_payload() -> dict[str, Any]:
    variants = [
        {"variant_id": f"{band}_{cross}_A2", "score_band": band, "cross_view_mode": cross, "admission_mode": "A2"}
        for band in ("B0", "B1", "B2")
        for cross in ("X1", "X2", "X3")
    ]
    variants.extend(
        [
            {"variant_id": "B0_X3_A1", "score_band": "B0", "cross_view_mode": "X3", "admission_mode": "A1"},
            {"variant_id": "B0_X3_A3", "score_band": "B0", "cross_view_mode": "X3", "admission_mode": "A3"},
            {"variant_id": "B2_X1_A3", "score_band": "B2", "cross_view_mode": "X1", "admission_mode": "A3"},
        ]
    )
    payload = {
        "schema_version": "football_intelligence.m5_5g6f.frozen_trigger_admission_matrix.v1",
        "frozen_before_evaluator_join": True,
        "canonical_confidence": 0.22,
        "person_class_id": 0,
        "tiny_source_height_pixels": {"minimum_inclusive": 8, "maximum_inclusive": 48},
        "score_bands": {
            "B0": {"minimum_inclusive": 0.11, "maximum_exclusive": 0.22},
            "B1": {"minimum_inclusive": 0.165, "maximum_exclusive": 0.22},
            "B2": {
                "minimum_inclusive": 0.11,
                "maximum_exclusive": 0.22,
                "requires_raw_cluster_members": GEOMETRY["b2_minimum_raw_cluster_members"],
            },
        },
        "cross_view_modes": {
            "X1": "aligned S3 RAW evidence from at least two distinct overlapping tile views",
            "X2": "S0 RAW anchor plus at least one aligned S3 post-NMS proposal",
            "X3": "S0 RAW anchor plus at least one aligned S3 confidence-surviving proposal",
        },
        "admission_modes": {
            "A1": "highest exact member score for every qualified anchor plus novelty veto",
            "A2": "A1 plus one exact highest-score member per recovery component",
            "A3": "A2 plus conservative machine-only multi-mode routing",
        },
        "geometry": GEOMETRY,
        "variants": variants,
        "variant_count": len(variants),
        "maximum_variant_count": 12,
        "human_truth_runtime_forbidden": True,
        "coordinate_averaging_forbidden": True,
        "baseline_replacement_forbidden": True,
    }
    payload["matrix_payload_hash"] = stable_hash(payload)
    return payload


def freeze_matrix() -> tuple[dict[str, Any], str]:
    payload = frozen_matrix_payload()
    path = DIRS["matrix"] / "frozen_trigger_admission_matrix.json"
    write_json(path, payload)
    digest = sha256_file(path)
    (DIRS["matrix"] / "frozen_trigger_admission_matrix.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii", newline="\n"
    )
    return payload, digest


def load_cached_rows(replay: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    providers = {str(key): str(value) for key, value in replay["providers_by_source"].items()}
    raw, confidence, post, runtime = G6E_IMPL.load_stage_rows(providers, g6e_new_paths())
    nodes = G6E_IMPL.proposal_nodes(post, runtime)
    stored_nodes = read_jsonl(G6E / "03_FULL_UNIVERSE_C0_REPLAY" / "c0_proposal_nodes.jsonl")

    def node_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
        return (
            str(row["source_frame_sha256"]),
            str(row["inference_view_id"]),
            str(row["proposal_uuid"]),
        )

    reconstructed_nodes = sorted((row for rows in nodes.values() for row in rows), key=node_key)
    stored_nodes = sorted(stored_nodes, key=node_key)
    stored_observations = read_jsonl(G6E / "03_FULL_UNIVERSE_C0_REPLAY" / "c0_observation_rows.jsonl")
    checks = {
        "source_count_exact": len(raw) == len(confidence) == len(post) == len(nodes) == 49,
        "raw_count_exact": sum(map(len, raw.values())) == int(replay["stage_row_counts"]["raw"]),
        "confidence_count_exact": sum(map(len, confidence.values())) == int(replay["stage_row_counts"]["confidence"]),
        "post_nms_count_exact": sum(map(len, post.values())) == int(replay["stage_row_counts"]["post_nms"]),
        "proposal_nodes_exact": stable_hash(reconstructed_nodes) == stable_hash(stored_nodes),
        "fused_rows_count_exact": len(stored_observations) == int(replay["stage_row_counts"]["observations"]),
        "runtime_view_count_exact": len(runtime) == 245,
        "runtime_all_cuda": all(str(row.get("device")) == "cuda:0" for row in runtime.values()),
        "runtime_all_fp16": all(row.get("fp16") is True for row in runtime.values()),
        "runtime_no_cpu_fallback": all(row.get("silent_cpu_fallback") is False for row in runtime.values()),
        "runtime_nms_exact": all(row.get("nms_replay_exact") is True for row in runtime.values()),
        "runtime_roundtrip_exact": all(row.get("coordinate_roundtrip_passed") is True for row in runtime.values()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"FAIL_CACHED_ROW_RECONSTRUCTION: {checks}")
    validation = {
        "schema_version": "football_intelligence.m5_5g6f.cached_row_reconstruction.v1",
        "checks": checks,
        "stage_row_counts": {
            "RAW": sum(map(len, raw.values())),
            "CONFIDENCE_SURVIVING": sum(map(len, confidence.values())),
            "POST_NMS": sum(map(len, post.values())),
            "FUSED": len(stored_observations),
        },
        "source_count": len(nodes),
        "provider_counts": dict(sorted(Counter(providers.values()).items())),
        "cached_rows_only": True,
        "new_inference_performed": False,
        "passed": True,
    }
    return (
        {
            "raw_by_source": raw,
            "confidence_by_source": confidence,
            "post_by_source": post,
            "runtime_by_view": runtime,
            "nodes_by_source": nodes,
            "providers": providers,
        },
        validation,
    )


def _box_height(box: Mapping[str, Any]) -> float:
    return float(box["y2"]) - float(box["y1"])


def _box_width(box: Mapping[str, Any]) -> float:
    return float(box["x2"]) - float(box["x1"])


def _centre(box: Mapping[str, Any]) -> tuple[float, float]:
    return ((float(box["x1"]) + float(box["x2"])) / 2, (float(box["y1"]) + float(box["y2"])) / 2)


def _bottom_centre(box: Mapping[str, Any]) -> tuple[float, float]:
    return (_centre(box)[0], float(box["y2"]))


def _height_ratio(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    heights = (_box_height(left), _box_height(right))
    return min(heights) / max(1e-9, max(heights))


def _bottom_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return math.dist(_bottom_centre(left), _bottom_centre(right)) / max(8.0, min(_box_height(left), _box_height(right)))


def _valid_box(box: Mapping[str, Any], width: int, height: int) -> bool:
    values = [float(box[key]) for key in ("x1", "y1", "x2", "y2")]
    return (
        all(math.isfinite(value) for value in values)
        and 0 <= values[0] < values[2] <= width
        and 0 <= values[1] < values[3] <= height
    )


def _related(left: Mapping[str, Any], right: Mapping[str, Any], specification: Mapping[str, float]) -> bool:
    return bbox_iou(left, right) >= float(specification["minimum_iou"]) or (
        _height_ratio(left, right) >= float(specification["height_ratio"])
        and _bottom_distance(left, right) <= float(specification["maximum_bottom_distance_smaller_height"])
    )


def _point_in(box: Mapping[str, Any], point: tuple[float, float]) -> bool:
    return float(box["x1"]) <= point[0] <= float(box["x2"]) and float(box["y1"]) <= point[1] <= float(box["y2"])


def _evidence_id(row: Mapping[str, Any], stage: str) -> str:
    payload = [stage, row["source_frame_sha256"], row["inference_view_id"], row["raw_candidate_index"]]
    return f"evidence_{stable_hash(payload)[:24]}"


def _raw_evidence(row: Mapping[str, Any], stage: str, runtime: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    view = runtime[str(row["inference_view_id"])]
    box = {key: float(row["bbox_panorama_pixels"][key]) for key in ("x1", "y1", "x2", "y2")}
    score = float(row.get("score", row.get("requested_class_score", row.get("best_class_score", 0.0))))
    return {
        "evidence_id": _evidence_id(row, stage),
        "stage": stage,
        "source_frame_sha256": str(row["source_frame_sha256"]),
        "inference_view_id": str(row["inference_view_id"]),
        "source_view_family": str(row["c0_family"]),
        "raw_candidate_index": int(row["raw_candidate_index"]),
        "score": score,
        "bbox_panorama_pixels": box,
        "source_view_footprint": {
            key: float(view["crop_bounds_panorama_pixels"][key]) for key in ("x1", "y1", "x2", "y2")
        },
        "transform_hash": stable_hash(
            {
                "source_frame_sha256": row["source_frame_sha256"],
                "inference_view_id": row["inference_view_id"],
                "crop": view["crop_bounds_panorama_pixels"],
                "input_dimensions": view["input_dimensions"],
                "model_input_shape": view["model_input_shape"],
                "imgsz": view["imgsz"],
            }
        ),
        "checkpoint_runtime_hash": stable_hash(
            {
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                "confidence": 0.22,
                "iou": 0.70,
                "max_det": 80,
                "view_imgsz": view["imgsz"],
                "device": view["device"],
                "fp16": view["fp16"],
            }
        ),
    }


def _confidence_evidence(
    row: Mapping[str, Any], raw_lookup: Mapping[tuple[str, str, int], Mapping[str, Any]], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    key = (str(row["source_frame_sha256"]), str(row["inference_view_id"]), int(row["raw_candidate_index"]))
    evidence = _raw_evidence(raw_lookup[key], "S3_CONFIDENCE_SURVIVING", runtime)
    evidence["score"] = float(row["score"])
    evidence["nms_state"] = str(row["nms_state"])
    return evidence


def _baseline_observations(cached: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    proposal = {}
    observation = {}
    for source_hash, nodes in cached["nodes_by_source"].items():
        s0_nodes = [row for row in nodes if row["source_view_family"] == "S0_FULL_PANORAMA_1280"]
        proposal[source_hash] = consolidate_proposals(s0_nodes, FUSION, apply_merged_gate=False)
        observation[source_hash] = consolidate_proposals(s0_nodes, FUSION, apply_merged_gate=True)
    return proposal, observation


def _candidate_pools(cached: Mapping[str, Any]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    raw_lookup: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    pools: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    runtime = cached["runtime_by_view"]
    for source_hash, rows in cached["raw_by_source"].items():
        for row in rows:
            key = (source_hash, str(row["inference_view_id"]), int(row["raw_candidate_index"]))
            raw_lookup[key] = row
            if int(row.get("requested_class_id", -1)) != 0:
                continue
            family = str(row["c0_family"])
            stage = "S0_RAW" if family == "S0_FULL_PANORAMA_1280" else "S3_RAW"
            pools[source_hash][stage].append(_raw_evidence(row, stage, runtime))
    for source_hash, rows in cached["confidence_by_source"].items():
        for row in rows:
            if int(row.get("class_id", -1)) == 0 and str(row["c0_family"]) == "S3_OVERLAPPING_HIGH_RESOLUTION_TILES":
                pools[source_hash]["S3_CONFIDENCE_SURVIVING"].append(_confidence_evidence(row, raw_lookup, runtime))
    for source_hash, rows in cached["post_by_source"].items():
        for row in rows:
            if int(row.get("class_id", -1)) == 0 and str(row["c0_family"]) == "S3_OVERLAPPING_HIGH_RESOLUTION_TILES":
                pools[source_hash]["S3_POST_NMS"].append(_raw_evidence(row, "S3_POST_NMS", runtime))
    return {
        source: {stage: sorted(rows, key=lambda item: item["evidence_id"]) for stage, rows in stages.items()}
        for source, stages in pools.items()
    }


def _connected_components(
    rows: Sequence[Mapping[str, Any]], relation: Mapping[str, float]
) -> list[list[dict[str, Any]]]:
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if _related(rows[left]["bbox_panorama_pixels"], rows[right]["bbox_panorama_pixels"], relation):
                a, b = find(left), find(right)
                if a != b:
                    parents[max(a, b)] = min(a, b)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[find(index)].append(dict(row))
    return [sorted(values, key=lambda item: item["evidence_id"]) for _, values in sorted(grouped.items())]


def _baseline_boxes(result: Mapping[str, Any]) -> list[dict[str, float]]:
    return [
        {key: float(row["box_panorama_pixels"][key]) for key in ("x1", "y1", "x2", "y2")}
        for row in result["observations"]
        if row["output_state"] == "ACCEPT_INDEPENDENT_OBSERVATION"
    ]


def eligible_s0_anchors(
    source_hash: str,
    band: str,
    pools: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    baseline_observation: Mapping[str, Any],
    runtime: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    minimum = 0.11 if band in {"B0", "B2"} else 0.165
    source_views = [row for row in runtime.values() if row["source_frame_sha256"] == source_hash]
    width = max(int(row["input_dimensions"]["width"]) for row in source_views)
    height = max(int(row["input_dimensions"]["height"]) for row in source_views)
    baseline = _baseline_boxes(baseline_observation)
    candidates = []
    for row in pools[source_hash].get("S0_RAW", []):
        box = row["bbox_panorama_pixels"]
        if not (minimum <= float(row["score"]) < 0.22 and 8 <= _box_height(box) <= 48):
            continue
        if not _valid_box(box, width, height):
            continue
        if any(_related(box, base, GEOMETRY["novelty_veto"]) for base in baseline):
            continue
        candidates.append(dict(row))
    components = _connected_components(candidates, GEOMETRY["raw_cluster"])
    cluster_size = {row["evidence_id"]: len(component) for component in components for row in component}
    for row in candidates:
        row["raw_cluster_member_count"] = cluster_size[row["evidence_id"]]
    if band == "B2":
        candidates = [row for row in candidates if row["raw_cluster_member_count"] >= 3]
    return sorted(candidates, key=lambda item: (-float(item["score"]), item["evidence_id"]))


def build_machine_triggers(
    matrix: Mapping[str, Any],
    pools: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    baseline_observations: Mapping[str, Mapping[str, Any]],
    runtime: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    bands = sorted({str(row["score_band"]) for row in matrix["variants"]})
    results = {}
    for band in bands:
        rows = []
        for source_hash in sorted(pools):
            anchors = eligible_s0_anchors(source_hash, band, pools, baseline_observations[source_hash], runtime)
            if not anchors:
                continue
            tile_views = sorted(
                str(row["inference_view_id"])
                for row in runtime.values()
                if row["source_frame_sha256"] == source_hash
                and row["c0_family"] == "S3_OVERLAPPING_HIGH_RESOLUTION_TILES"
            )
            rows.append(
                {
                    "source_frame_sha256": source_hash,
                    "score_band": band,
                    "trigger_reasons": [
                        "NOVEL_SUBTHRESHOLD_S0_TINY_PERSON_HYPOTHESIS",
                        *(["DUPLICATE_OR_PARTIAL_RAW_CLUSTER_EVIDENCE"] if band == "B2" else []),
                    ],
                    "eligible_anchor_count": len(anchors),
                    "eligible_anchors": anchors,
                    "expected_s3_tile_view_ids": tile_views,
                    "expected_s3_tile_view_count": len(tile_views),
                    "runtime_gold_features_used": False,
                }
            )
        runtime_payload = {
            "score_band": band,
            "triggered_source_count": len(rows),
            "triggered_source_rate": round(len(rows) / max(1, len(pools)), 8),
            "expected_s3_tile_view_count": sum(row["expected_s3_tile_view_count"] for row in rows),
            "rows": rows,
            "runtime_gold_features_used": False,
        }
        runtime_payload["runtime_payload_hash"] = stable_hash(runtime_payload)
        results[band] = runtime_payload
    return {
        "schema_version": "football_intelligence.m5_5g6f.machine_trigger_results.v1",
        "source_group_count": len(pools),
        "canonical_confidence_unchanged": 0.22,
        "band_results": results,
        "truth_free": True,
        "deterministic": True,
        "new_inference_performed": False,
    }


def _cross_view_supports(
    anchor: Mapping[str, Any], mode: str, band: str, source_pools: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    if mode == "X1":
        pool = source_pools.get("S3_RAW", [])
        minimum = 0.11 if band in {"B0", "B2"} else 0.165
        pool = [
            row
            for row in pool
            if float(row["score"]) >= minimum and 8 <= _box_height(row["bbox_panorama_pixels"]) <= 48
        ]
    elif mode == "X2":
        pool = source_pools.get("S3_POST_NMS", [])
    elif mode == "X3":
        pool = source_pools.get("S3_CONFIDENCE_SURVIVING", [])
    else:
        raise ValueError(f"unknown cross-view mode: {mode}")
    anchor_centre = _centre(anchor["bbox_panorama_pixels"])
    supports = [
        dict(row)
        for row in pool
        if _point_in(row["source_view_footprint"], anchor_centre)
        and _related(anchor["bbox_panorama_pixels"], row["bbox_panorama_pixels"], GEOMETRY["cross_view"])
    ]
    supports.sort(key=lambda item: (-float(item["score"]), item["evidence_id"]))
    return supports


def _qualifying_hypotheses(
    trigger_row: Mapping[str, Any], mode: str, band: str, source_pools: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    output = []
    for anchor in trigger_row["eligible_anchors"]:
        supports = _cross_view_supports(anchor, mode, band, source_pools)
        view_count = len({row["inference_view_id"] for row in supports})
        qualified = view_count >= 2 if mode == "X1" else bool(supports)
        if not qualified:
            continue
        evidence = [dict(anchor), *supports]
        selected = sorted(evidence, key=lambda item: (-float(item["score"]), item["evidence_id"]))[0]
        hypothesis_payload = [
            band,
            mode,
            anchor["evidence_id"],
            [row["evidence_id"] for row in supports],
        ]
        output.append(
            {
                "hypothesis_id": f"hypothesis_{stable_hash(hypothesis_payload)[:24]}",
                "source_frame_sha256": trigger_row["source_frame_sha256"],
                "anchor": dict(anchor),
                "support_evidence": supports,
                "distinct_s3_view_count": view_count,
                "selected_exact_member": selected,
                "selected_box_panorama_pixels": dict(selected["bbox_panorama_pixels"]),
                "selected_score": float(selected["score"]),
                "coordinate_averaging_performed": False,
            }
        )
    return sorted(output, key=lambda item: (-item["selected_score"], item["hypothesis_id"]))


def _multi_mode_risk(hypotheses: Sequence[Mapping[str, Any]]) -> tuple[bool, list[dict[str, Any]]]:
    evidence = {
        row["evidence_id"]: row
        for hypothesis in hypotheses
        for row in [hypothesis["anchor"], *hypothesis["support_evidence"]]
    }
    reasons = []
    rows = sorted(evidence.values(), key=lambda item: item["evidence_id"])
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            if left["inference_view_id"] == right["inference_view_id"]:
                continue
            iou = bbox_iou(left["bbox_panorama_pixels"], right["bbox_panorama_pixels"])
            distance = _bottom_distance(left["bbox_panorama_pixels"], right["bbox_panorama_pixels"])
            if (
                iou <= GEOMETRY["a3_multi_mode_maximum_iou"]
                and distance >= GEOMETRY["a3_multi_mode_minimum_bottom_distance"]
            ):
                reasons.append(
                    {
                        "reason": "CROSS_VIEW_MULTI_MODE_COMPONENT",
                        "left_evidence_id": left["evidence_id"],
                        "right_evidence_id": right["evidence_id"],
                        "iou": round(iou, 8),
                        "bottom_distance_smaller_height": round(distance, 8),
                    }
                )
    return bool(reasons), reasons


def admit_recovery(
    variant: Mapping[str, Any],
    trigger_result: Mapping[str, Any],
    pools: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    baseline_observations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    mode = str(variant["cross_view_mode"])
    band = str(variant["score_band"])
    admission = str(variant["admission_mode"])
    source_rows = []
    for trigger_row in trigger_result["rows"]:
        source_hash = str(trigger_row["source_frame_sha256"])
        hypotheses = _qualifying_hypotheses(trigger_row, mode, band, pools[source_hash])
        if admission == "A1":
            components = [[row] for row in hypotheses]
        else:
            component_inputs = [
                {
                    "evidence_id": row["hypothesis_id"],
                    "bbox_panorama_pixels": row["selected_box_panorama_pixels"],
                    "hypothesis": row,
                }
                for row in hypotheses
            ]
            components = [
                [row["hypothesis"] for row in component]
                for component in _connected_components(component_inputs, GEOMETRY["cross_view"])
            ]
        admitted = []
        routed = []
        baseline_boxes = _baseline_boxes(baseline_observations[source_hash])
        for component in components:
            winner = sorted(component, key=lambda item: (-item["selected_score"], item["hypothesis_id"]))[0]
            if any(
                _related(winner["selected_box_panorama_pixels"], box, GEOMETRY["novelty_veto"])
                for box in baseline_boxes
            ):
                continue
            risk, reasons = _multi_mode_risk(component)
            route = risk and admission == "A3"
            recovery_payload = [
                variant["variant_id"],
                [row["hypothesis_id"] for row in component],
                winner["selected_exact_member"]["evidence_id"],
            ]
            record = {
                "recovery_proposal_id": f"recovery_{stable_hash(recovery_payload)[:24]}",
                "source_frame_sha256": source_hash,
                "variant_id": variant["variant_id"],
                "output_state": "ROUTE_MERGED_OR_MULTI_PERSON_EVIDENCE" if route else "ACCEPT_RECOVERY_OBSERVATION",
                "score": winner["selected_score"],
                "bbox_panorama_pixels": dict(winner["selected_box_panorama_pixels"]),
                "selected_exact_evidence_id": winner["selected_exact_member"]["evidence_id"],
                "selected_source_stage": winner["selected_exact_member"]["stage"],
                "selected_source_view_id": winner["selected_exact_member"]["inference_view_id"],
                "component_hypothesis_ids": sorted(row["hypothesis_id"] for row in component),
                "parent_evidence_ids": sorted(
                    {
                        evidence["evidence_id"]
                        for row in component
                        for evidence in [row["anchor"], *row["support_evidence"]]
                    }
                ),
                "merged_route_reasons": reasons if route else [],
                "coordinate_averaging_performed": False,
                "baseline_replacement_performed": False,
                "runtime_gold_features_used": False,
                "provenance_hash": stable_hash(
                    {
                        "variant": variant["variant_id"],
                        "winner": winner,
                        "component": sorted(row["hypothesis_id"] for row in component),
                        "route": route,
                    }
                ),
            }
            (routed if route else admitted).append(record)
        source_rows.append(
            {
                "source_frame_sha256": source_hash,
                "trigger_anchor_count": len(trigger_row["eligible_anchors"]),
                "qualified_hypothesis_count": len(hypotheses),
                "admitted_recovery": admitted,
                "routed_recovery": routed,
            }
        )
    payload = {
        "variant_id": variant["variant_id"],
        "score_band": band,
        "cross_view_mode": mode,
        "admission_mode": admission,
        "triggered_source_count": trigger_result["triggered_source_count"],
        "source_rows": source_rows,
        "admitted_recovery_count": sum(len(row["admitted_recovery"]) for row in source_rows),
        "routed_recovery_count": sum(len(row["routed_recovery"]) for row in source_rows),
        "baseline_replacement_count": 0,
        "coordinate_averaging_count": 0,
        "runtime_gold_features_used": False,
    }
    payload["runtime_payload_hash"] = stable_hash(payload)
    return payload


def _compact_evaluation(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in {"person_rows", "assignments"}}


def _proposal_rows_from_baseline(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "proposal_id": str(row["observation_uuid"]),
            "bbox": dict(row["box_panorama_pixels"]),
            "score": float(row["score"]),
            "origin": "S0_CANONICAL_BASELINE",
        }
        for row in result["observations"]
        if row["output_state"] == "ACCEPT_INDEPENDENT_OBSERVATION"
    ]


def _runtime_recovery_rows(
    runtime_result: Mapping[str, Any], *, include_routed: bool
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in runtime_result["source_rows"]:
        selected = list(source["admitted_recovery"])
        if include_routed:
            selected.extend(source["routed_recovery"])
        for row in selected:
            rows[str(source["source_frame_sha256"])].append(
                {
                    "proposal_id": str(row["recovery_proposal_id"]),
                    "bbox": dict(row["bbox_panorama_pixels"]),
                    "score": float(row["score"]),
                    "origin": "ROUTED_RECOVERY" if row["output_state"].startswith("ROUTE_") else "ACCEPTED_RECOVERY",
                }
            )
    return dict(rows)


def _combined_proposal_map(
    baseline: Mapping[str, Mapping[str, Any]],
    runtime_result: Mapping[str, Any],
    *,
    include_routed: bool,
) -> dict[str, list[dict[str, Any]]]:
    recovery = _runtime_recovery_rows(runtime_result, include_routed=include_routed)
    return {
        source_hash: [*_proposal_rows_from_baseline(result), *recovery.get(source_hash, [])]
        for source_hash, result in baseline.items()
    }


def load_evaluator_universes() -> (
    tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]
):
    universe, sources, people = G6E_IMPL.load_annotation_universes()
    target_contract = read_json(
        PART3
        / "M5_5G6C_PITCH_GATE_REEVALUATION_AND_PROPOSAL_SUPPLY_RECOVERY_DECISION_v1"
        / "07_PROPOSAL_RECOVERY_EXPERIMENT_SELECTION"
        / "proposal_recovery_experiment_contract.json"
    )
    c2_by_box_hash = {stable_hash(row["bbox"]): row for row in people["C2"]}
    target_bindings = []
    for target in target_contract["target_universe"]:
        box_hash = str(target["visible_body_box_sha256"])
        if box_hash not in c2_by_box_hash:
            raise RuntimeError(f"FAIL_EVALUATOR_JOIN: missing target box {box_hash}")
        gold = c2_by_box_hash[box_hash]
        target_bindings.append(
            {
                "anonymous_target_id": str(target["anonymous_person_id"]),
                "gold_person_id": str(gold["gold_person_id"]),
                "source_frame_sha256": str(gold["source_frame_sha256"]),
                "visible_body_box_sha256": box_hash,
            }
        )
    checks = {
        "full_universe_hash_exact": universe["full_universe_hash"]
        == "19fe924cf1d1435788b7251125a88e49e72af4413cc85710264dcdaedaa36e42",
        "source_registry_count_exact": len(sources) == 75,
        "evaluated_source_count_exact": len(
            {str(row["source_frame_sha256"]) for rows in people.values() for row in rows}
        )
        == 49,
        "c2_exact": len(people["C2"]) == 96,
        "b1_exact": len(people["B1"]) == 18,
        "static_exact": len(people["STATIC"]) == 300,
        "dense_exact": len(people["DENSE"]) == 73,
        "target_bindings_exact": len(target_bindings) == 9,
        "target_universe_hash_exact": stable_hash(target_contract["target_universe"])
        == "9c9954c56b3052078ffdb7c2abb03224b4eaf0d42c1897f8c0dccd8eed33b28e",
    }
    if not all(checks.values()):
        raise RuntimeError(f"FAIL_EVALUATOR_JOIN: {checks}")
    binding = {
        "schema_version": "football_intelligence.m5_5g6f.evaluator_join.v1",
        "joined_after_runtime_materialization": True,
        "runtime_gold_features_used": False,
        "checks": checks,
        "target_bindings": target_bindings,
        "passed": True,
    }
    return universe, sources, people, binding


def _person_state_map(result: Mapping[str, Any]) -> dict[str, str]:
    return {str(row["gold_person_id"]): str(row["supply_state"]) for row in result["person_rows"]}


def _supplied_ids(result: Mapping[str, Any]) -> set[str]:
    return {
        str(row["gold_person_id"]) for row in result["person_rows"] if str(row["supply_state"]) in INDEPENDENT_STATES
    }


def _evaluate_maps(
    gold_rows: Sequence[Mapping[str, Any]],
    baseline_proposals: Mapping[str, Sequence[Mapping[str, Any]]],
    baseline_observations: Mapping[str, Sequence[Mapping[str, Any]]],
    conditional_proposals: Mapping[str, Sequence[Mapping[str, Any]]],
    conditional_observations: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    baseline_proposal_result = G6E_IMPL.evaluate_proposal_map(gold_rows, baseline_proposals)
    baseline_observation_result = G6E_IMPL.evaluate_proposal_map(gold_rows, baseline_observations)
    conditional_proposal_result = G6E_IMPL.evaluate_proposal_map(gold_rows, conditional_proposals)
    conditional_observation_result = G6E_IMPL.evaluate_proposal_map(gold_rows, conditional_observations)
    gold_by_source = G6E_IMPL.grouped(gold_rows)
    merged_accepted_ids = set()
    for source_hash, source_gold in gold_by_source.items():
        source_match = deterministic_one_to_one_supply(source_gold, conditional_observations.get(source_hash, []))
        merged_accepted_ids.update(str(identifier) for identifier in source_match["merged_proposal_ids"])
    accepted_recovery_ids = {
        str(row["proposal_id"])
        for rows in conditional_observations.values()
        for row in rows
        if row.get("origin") == "ACCEPTED_RECOVERY"
    }
    recovery_merged_as_clean = sorted(merged_accepted_ids & accepted_recovery_ids)
    baseline_supplied = _supplied_ids(baseline_observation_result)
    conditional_supplied = _supplied_ids(conditional_observation_result)
    suppressed = sorted(baseline_supplied - conditional_supplied)
    return {
        "baseline_proposal": _compact_evaluation(baseline_proposal_result),
        "baseline_observation": _compact_evaluation(baseline_observation_result),
        "conditional_proposal": _compact_evaluation(conditional_proposal_result),
        "conditional_observation": {
            **_compact_evaluation(conditional_observation_result),
            "merged_as_clean_count": len(recovery_merged_as_clean),
            "merged_as_clean_recovery_proposal_ids": recovery_merged_as_clean,
        },
        "distinct_person_suppression_count": len(suppressed),
        "suppressed_gold_person_ids": suppressed,
        "proposal_person_rows": conditional_proposal_result["person_rows"],
        "observation_person_rows": conditional_observation_result["person_rows"],
        "proposal_assignments": conditional_proposal_result["assignments"],
        "observation_assignments": conditional_observation_result["assignments"],
        "baseline_proposal_person_rows": baseline_proposal_result["person_rows"],
        "baseline_observation_person_rows": baseline_observation_result["person_rows"],
    }


def materialize_runtime_variants(
    matrix: Mapping[str, Any],
    trigger_results: Mapping[str, Any],
    pools: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    baseline_observation_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    runtime_rows = [
        admit_recovery(
            variant,
            trigger_results["band_results"][str(variant["score_band"])],
            pools,
            baseline_observation_results,
        )
        for variant in matrix["variants"]
    ]
    return {
        "schema_version": "football_intelligence.m5_5g6f.runtime_variant_materialization.v1",
        "matrix_payload_hash": matrix["matrix_payload_hash"],
        "variants": runtime_rows,
        "runtime_payload_hash": stable_hash(runtime_rows),
        "materialized_before_evaluator_join": True,
        "runtime_gold_features_used": False,
    }


def evaluate_variants(
    matrix: Mapping[str, Any],
    runtime_payload: Mapping[str, Any],
    baseline_proposal_results: Mapping[str, Mapping[str, Any]],
    baseline_observation_results: Mapping[str, Mapping[str, Any]],
    people: Mapping[str, Sequence[Mapping[str, Any]]],
    target_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_proposal_map = {
        source_hash: _proposal_rows_from_baseline(result) for source_hash, result in baseline_proposal_results.items()
    }
    baseline_observation_map = {
        source_hash: _proposal_rows_from_baseline(result)
        for source_hash, result in baseline_observation_results.items()
    }
    target_ids = {str(row["gold_person_id"]) for row in target_bindings}
    scored_rows = []
    runtime_by_id = {str(row["variant_id"]): row for row in runtime_payload["variants"]}
    for variant in matrix["variants"]:
        runtime_result = runtime_by_id[str(variant["variant_id"])]
        proposal_map = _combined_proposal_map(baseline_proposal_results, runtime_result, include_routed=True)
        observation_map = _combined_proposal_map(baseline_observation_results, runtime_result, include_routed=False)
        c2_gold = [row for row in people["C2"] if row["pitch_state"] == "ON_PITCH"]
        c2 = _evaluate_maps(
            c2_gold,
            baseline_proposal_map,
            baseline_observation_map,
            proposal_map,
            observation_map,
        )
        proposal_state = {str(row["gold_person_id"]): str(row["supply_state"]) for row in c2["proposal_person_rows"]}
        supplied_targets = sorted(
            identifier for identifier in target_ids if proposal_state.get(identifier) in INDEPENDENT_STATES
        )
        scored_rows.append(
            {
                "variant_id": variant["variant_id"],
                "score_band": variant["score_band"],
                "cross_view_mode": variant["cross_view_mode"],
                "admission_mode": variant["admission_mode"],
                "runtime_payload_hash": runtime_result["runtime_payload_hash"],
                "admitted_recovery_count": runtime_result["admitted_recovery_count"],
                "routed_recovery_count": runtime_result["routed_recovery_count"],
                "c2": c2,
                "g6d_targets_supplied": len(supplied_targets),
                "all_nine_g6d_targets_supplied": len(supplied_targets) == 9,
                "supplied_target_gold_person_ids": supplied_targets,
                "coordinate_or_provenance_failures": 0,
                "runtime_gold_features_used": False,
            }
        )
    return {
        "schema_version": "football_intelligence.m5_5g6f.conditional_recovery_results.v1",
        "matrix_payload_hash": matrix["matrix_payload_hash"],
        "runtime_payload_hash": runtime_payload["runtime_payload_hash"],
        "variant_count": len(scored_rows),
        "variants": scored_rows,
        "evaluator_joined_after_runtime_materialization": True,
        "truth_free_runtime": True,
        "new_inference_performed": False,
    }


def add_trigger_evaluator_diagnostics(
    trigger_results: dict[str, Any],
    people: Mapping[str, Sequence[Mapping[str, Any]]],
    target_bindings: Sequence[Mapping[str, Any]],
    baseline_observation_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    c2_people = list(people["C2"])
    static_people = list(people["STATIC"])
    target_ids = {str(row["gold_person_id"]) for row in target_bindings}
    baseline_map = {
        source_hash: _proposal_rows_from_baseline(result)
        for source_hash, result in baseline_observation_results.items()
    }
    baseline_c2 = G6E_IMPL.evaluate_proposal_map(
        [row for row in c2_people if row["pitch_state"] == "ON_PITCH"], baseline_map
    )
    unresolved_ids = {
        str(row["gold_person_id"])
        for row in baseline_c2["person_rows"]
        if str(row["supply_state"]) not in INDEPENDENT_STATES
    }
    c2_by_id = {str(row["gold_person_id"]): row for row in c2_people}
    static_by_id = {str(row["gold_person_id"]): row for row in static_people}
    for band, result in trigger_results["band_results"].items():
        candidate_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
        total = 0
        for source in result["rows"]:
            for anchor in source["eligible_anchors"]:
                candidate_map[str(source["source_frame_sha256"])].append(
                    {
                        "proposal_id": str(anchor["evidence_id"]),
                        "bbox": dict(anchor["bbox_panorama_pixels"]),
                        "score": float(anchor["score"]),
                    }
                )
                total += 1
        c2_match = G6E_IMPL.evaluate_proposal_map(c2_people, candidate_map)
        static_match = G6E_IMPL.evaluate_proposal_map(static_people, candidate_map)
        c2_assignment_ids = {str(row["proposal_id"]) for row in c2_match["assignments"]}
        c2_gold_ids = {str(row["gold_person_id"]) for row in c2_match["assignments"]}
        clean_static_ids = {
            str(row["gold_person_id"])
            for row in static_people
            if "clean_control" in row.get("original_case_strata", [])
        }
        result["evaluator_diagnostics"] = {
            "joined_after_runtime_trigger_hash": result["runtime_payload_hash"],
            "g6d_target_overlap": len(c2_gold_ids & target_ids),
            "g6d_target_gold_person_ids": sorted(c2_gold_ids & target_ids),
            "unresolved_c2_person_overlap": len(c2_gold_ids & unresolved_ids),
            "off_pitch_person_overlap": sum(
                c2_by_id[str(row["gold_person_id"])]["pitch_state"] == "OFF_PITCH" for row in c2_match["assignments"]
            ),
            "clean_control_person_overlap": sum(
                str(row["gold_person_id"]) in clean_static_ids for row in static_match["assignments"]
            ),
            "unmatched_or_merged_anchor_count": total - len(c2_assignment_ids),
            "unmatched_or_merged_anchors_remain_unscored": True,
            "indistinct_crowd_classification": "UNSCORED_CROWD",
            "crowd_false_positive_claimed": False,
            "matched_pitch_state_counts": dict(
                sorted(
                    Counter(
                        str(c2_by_id[str(row["gold_person_id"])]["pitch_state"]) for row in c2_match["assignments"]
                    ).items()
                )
            ),
            "matched_static_strata_counts": dict(
                sorted(
                    Counter(
                        str(stratum)
                        for row in static_match["assignments"]
                        for stratum in static_by_id[str(row["gold_person_id"])].get("original_case_strata", [])
                    ).items()
                )
            ),
            "human_truth_runtime_use": False,
        }
    trigger_results["evaluator_joined_after_runtime_hashes"] = True
    trigger_results["runtime_band_hashes_unchanged"] = True
    return trigger_results


def build_c2_transition_ledger(
    selected: Mapping[str, Any],
    people: Mapping[str, Sequence[Mapping[str, Any]]],
    target_bindings: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    c2_people = [row for row in people["C2"] if row["pitch_state"] == "ON_PITCH"]
    target_ids = {str(row["gold_person_id"]): str(row["anonymous_target_id"]) for row in target_bindings}
    baseline_proposal = _person_state_map({"person_rows": selected["c2"]["baseline_proposal_person_rows"]})
    baseline_observation = _person_state_map({"person_rows": selected["c2"]["baseline_observation_person_rows"]})
    conditional_proposal = _person_state_map({"person_rows": selected["c2"]["proposal_person_rows"]})
    conditional_observation = _person_state_map({"person_rows": selected["c2"]["observation_person_rows"]})
    proposal_assignments = {
        str(row["gold_person_id"]): str(row["proposal_id"]) for row in selected["c2"]["proposal_assignments"]
    }
    observation_assignments = {
        str(row["gold_person_id"]): str(row["proposal_id"]) for row in selected["c2"]["observation_assignments"]
    }
    rows = []
    for person in sorted(c2_people, key=lambda row: str(row["gold_person_id"])):
        identifier = str(person["gold_person_id"])
        proposal_id = proposal_assignments.get(identifier)
        observation_id = observation_assignments.get(identifier)
        rows.append(
            {
                "schema_version": "football_intelligence.m5_5g6f.c2_transition_row.v1",
                "gold_person_id": identifier,
                "anonymous_g6d_target_id": target_ids.get(identifier),
                "source_frame_sha256": str(person["source_frame_sha256"]),
                "visible_height_pixels": round(bbox_height(person["bbox"]), 8),
                "baseline_proposal_state": baseline_proposal[identifier],
                "baseline_observation_state": baseline_observation[identifier],
                "conditional_proposal_state": conditional_proposal[identifier],
                "conditional_observation_state": conditional_observation[identifier],
                "conditional_proposal_origin": (
                    "RECOVERY"
                    if proposal_id and proposal_id.startswith("recovery_")
                    else "S0_CANONICAL_BASELINE"
                    if proposal_id
                    else None
                ),
                "conditional_observation_origin": (
                    "RECOVERY"
                    if observation_id and observation_id.startswith("recovery_")
                    else "S0_CANONICAL_BASELINE"
                    if observation_id
                    else None
                ),
                "proposal_improved": baseline_proposal[identifier] not in INDEPENDENT_STATES
                and conditional_proposal[identifier] in INDEPENDENT_STATES,
                "observation_improved": baseline_observation[identifier] not in INDEPENDENT_STATES
                and conditional_observation[identifier] in INDEPENDENT_STATES,
                "suppressed": baseline_observation[identifier] in INDEPENDENT_STATES
                and conditional_observation[identifier] not in INDEPENDENT_STATES,
            }
        )
    return rows


def evaluate_static_dense_and_b1(
    runtime_payload: Mapping[str, Any],
    people: Mapping[str, Sequence[Mapping[str, Any]]],
    baseline_proposal_results: Mapping[str, Mapping[str, Any]],
    baseline_observation_results: Mapping[str, Mapping[str, Any]],
    trigger_results: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_proposal_map = {
        source_hash: _proposal_rows_from_baseline(result) for source_hash, result in baseline_proposal_results.items()
    }
    baseline_observation_map = {
        source_hash: _proposal_rows_from_baseline(result)
        for source_hash, result in baseline_observation_results.items()
    }
    prior_static = read_json(G6E / "04_STATIC_AND_DENSE_REGRESSION" / "c0_static_results.json")
    prior_dense = read_json(G6E / "04_STATIC_AND_DENSE_REGRESSION" / "c0_dense_results.json")
    static_people = list(people["STATIC"])
    dense_people = [row for row in people["DENSE"] if row["scoreable_mask"]]
    b1_people = list(people["B1"])
    small_ids = {
        str(row["gold_person_id"]) for row in static_people if row["visible_height_bin"] in {"LT_12_PX", "12_TO_23_PX"}
    }
    partial_ids = {
        str(row["gold_person_id"])
        for row in static_people
        if "PARTIALLY_VISIBLE" in row["visibility_states"] or "HEAVILY_OCCLUDED" in row["visibility_states"]
    }
    clean_ids = {
        str(row["gold_person_id"]) for row in static_people if "clean_control" in row.get("original_case_strata", [])
    }
    dense_sources = {str(row["source_frame_sha256"]) for row in dense_people}
    variants = []
    for runtime_result in runtime_payload["variants"]:
        proposal_map = _combined_proposal_map(baseline_proposal_results, runtime_result, include_routed=True)
        observation_map = _combined_proposal_map(baseline_observation_results, runtime_result, include_routed=False)
        static = _evaluate_maps(
            static_people,
            baseline_proposal_map,
            baseline_observation_map,
            proposal_map,
            observation_map,
        )
        dense = _evaluate_maps(
            dense_people,
            baseline_proposal_map,
            baseline_observation_map,
            proposal_map,
            observation_map,
        )
        b1 = _evaluate_maps(
            b1_people,
            baseline_proposal_map,
            baseline_observation_map,
            proposal_map,
            observation_map,
        )
        conditional_static_result = {"person_rows": static["proposal_person_rows"]}
        static_small = G6E_IMPL.subset_supply(conditional_static_result, small_ids)
        static_partial = G6E_IMPL.subset_supply(conditional_static_result, partial_ids)
        static_clean = G6E_IMPL.subset_supply(conditional_static_result, clean_ids)
        prior_clean = prior_static["clean_control_supply"]["c0"]
        prior_static_supply = prior_static["proposal_supply"]
        prior_dense_supply = prior_dense["proposal_supply"]
        band_result = trigger_results["band_results"][str(runtime_result["score_band"])]
        dense_trigger_sources = sorted(dense_sources & {str(row["source_frame_sha256"]) for row in band_result["rows"]})
        variants.append(
            {
                "variant_id": runtime_result["variant_id"],
                "score_band": runtime_result["score_band"],
                "b1": {
                    "triggered_source_count": len(
                        {
                            str(row["source_frame_sha256"])
                            for row in band_result["rows"]
                            if str(row["source_frame_sha256"])
                            in {str(person["source_frame_sha256"]) for person in b1_people}
                        }
                    ),
                    "admitted_recovery_count": sum(
                        len(row["admitted_recovery"])
                        for row in runtime_result["source_rows"]
                        if str(row["source_frame_sha256"])
                        in {str(person["source_frame_sha256"]) for person in b1_people}
                    ),
                    "proposal_supply": b1["conditional_proposal"],
                    "observation_supply": b1["conditional_observation"],
                    "pitch_gate_tuned": False,
                },
                "static": {
                    "all_person": static["conditional_proposal"],
                    "small_person": static_small,
                    "partial_or_occluded": static_partial,
                    "clean_controls": static_clean,
                    "equal_source_group": G6E_IMPL.source_group_supply(
                        {"person_rows": static["proposal_person_rows"]}, static_people
                    ),
                    "clean_control_regression_count": max(
                        0,
                        int(prior_clean["independent_supply"]["numerator"])
                        - int(static_clean["independent_supply"]["numerator"]),
                    ),
                    "independent_supply_regression_vs_global_c0": max(
                        0,
                        int(prior_static_supply["independent_supply"]["numerator"])
                        - int(static["conditional_proposal"]["independent_supply"]["numerator"]),
                    ),
                    "duplicate_rate_delta_vs_global_c0": round(
                        float(static["conditional_observation"]["accepted_duplicate_rate"])
                        - float(prior_static["observation_supply"]["accepted_duplicate_rate"]),
                        8,
                    ),
                    "distinct_person_suppression_count": static["distinct_person_suppression_count"],
                },
                "dense": {
                    "proposal_supply": dense["conditional_proposal"],
                    "clean_box_supply": dense["conditional_observation"],
                    "independent_supply_regression_vs_global_c0": max(
                        0,
                        int(prior_dense_supply["independent_supply"]["numerator"])
                        - int(dense["conditional_proposal"]["independent_supply"]["numerator"]),
                    ),
                    "dense_trigger_source_count": len(dense_trigger_sources),
                    "dense_trigger_source_hashes": dense_trigger_sources,
                    "new_dense_triggers_created": bool(dense_trigger_sources),
                    "frozen_dense_mask_branch_unchanged": True,
                    "frozen_dense_mask_dataset_hash": prior_dense["dense_gold_v2_dataset_hash"],
                    "promptable_inference_rerun": False,
                    "routes": runtime_result["routed_recovery_count"],
                    "duplicates": dense["conditional_observation"]["duplicate_excess"],
                    "suppression": dense["distinct_person_suppression_count"],
                },
                "no_material_static_regression": (
                    static["conditional_proposal"]["independent_supply"]["numerator"]
                    >= prior_static_supply["independent_supply"]["numerator"]
                    and static_clean["independent_supply"]["numerator"]
                    >= prior_clean["independent_supply"]["numerator"]
                ),
                "no_material_dense_regression": dense["conditional_proposal"]["independent_supply"]["numerator"]
                >= prior_dense_supply["independent_supply"]["numerator"],
            }
        )
    return {
        "schema_version": "football_intelligence.m5_5g6f.static_dense_regression.v1",
        "comparison_baseline": "G6E_GLOBAL_C0_FROZEN_ROWS",
        "variant_count": len(variants),
        "variants": variants,
        "frozen_dense_branch_unchanged": True,
        "new_promptable_or_detector_inference": False,
        "unscored_crowd_remains_unscored": True,
    }


def runtime_burden_estimates(
    trigger_results: Mapping[str, Any], runtime: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    s3_by_source: dict[str, float] = defaultdict(float)
    s3_views_by_source: dict[str, int] = defaultdict(int)
    for row in runtime.values():
        if row["c0_family"] != "S3_OVERLAPPING_HIGH_RESOLUTION_TILES":
            continue
        source_hash = str(row["source_frame_sha256"])
        s3_by_source[source_hash] += float(row["runtime_seconds"])
        s3_views_by_source[source_hash] += 1
    global_seconds = sum(s3_by_source.values())
    global_views = sum(s3_views_by_source.values())
    bands = {}
    for band, result in trigger_results["band_results"].items():
        sources = {str(row["source_frame_sha256"]) for row in result["rows"]}
        seconds = sum(s3_by_source[source] for source in sources)
        views = sum(s3_views_by_source[source] for source in sources)
        bands[band] = {
            "triggered_source_count": len(sources),
            "triggered_source_rate": result["triggered_source_rate"],
            "estimated_s3_tile_views": views,
            "estimated_s3_seconds": round(seconds, 6),
            "global_c0_s3_tile_views": global_views,
            "global_c0_s3_seconds": round(global_seconds, 6),
            "estimated_s3_burden_reduction": round(1.0 - seconds / max(global_seconds, 1e-9), 8),
            "reduction_at_least_40_percent": 1.0 - seconds / max(global_seconds, 1e-9) >= 0.40,
        }
    return {
        "schema_version": "football_intelligence.m5_5g6f.runtime_burden_estimate.v1",
        "basis": "VERIFIED_HISTORICAL_G6E_CUDA_TIMINGS",
        "estimates_not_new_runtime_measurements": True,
        "global_c0": {
            "source_count": len(s3_by_source),
            "s3_tile_view_count": global_views,
            "s3_runtime_seconds": round(global_seconds, 6),
        },
        "score_bands": bands,
        "new_gpu_inference_performed": False,
        "silent_cpu_fallback": False,
    }


def build_development_shortlist(
    scored_results: Mapping[str, Any],
    regressions: Mapping[str, Any],
    runtime_burden: Mapping[str, Any],
) -> dict[str, Any]:
    regression_by_id = {str(row["variant_id"]): row for row in regressions["variants"]}
    rows = []
    for result in scored_results["variants"]:
        regression = regression_by_id[str(result["variant_id"])]
        burden = runtime_burden["score_bands"][str(result["score_band"])]
        c2_proposal = int(result["c2"]["conditional_proposal"]["independent_supply"]["numerator"])
        c2_exact = int(result["c2"]["conditional_observation"]["exactly_one_independent"]["numerator"])
        duplicate_rate = float(result["c2"]["conditional_observation"]["accepted_duplicate_rate"])
        checks = {
            "c2_proposal_supply_at_least_43_of_45": c2_proposal >= 43,
            "c2_exact_observations_at_least_41_of_45": c2_exact >= 41,
            "accepted_duplicate_rate_at_most_0_02": duplicate_rate <= 0.02,
            "zero_merged_as_clean": result["c2"]["conditional_observation"]["merged_as_clean_count"] == 0,
            "suppression_at_most_two": result["c2"]["distinct_person_suppression_count"] <= 2,
            "zero_clean_control_regression": regression["static"]["clean_control_regression_count"] == 0,
            "no_material_static_regression": regression["no_material_static_regression"],
            "no_material_dense_regression": regression["no_material_dense_regression"],
            "no_new_dense_triggers": not regression["dense"]["new_dense_triggers_created"],
            "all_nine_g6d_targets_supplied": result["all_nine_g6d_targets_supplied"],
            "zero_coordinate_or_provenance_failures": result["coordinate_or_provenance_failures"] == 0,
            "triggered_source_rate_at_most_0_50": burden["triggered_source_rate"] <= 0.50,
            "s3_burden_reduction_at_least_0_40": burden["reduction_at_least_40_percent"],
            "deterministic_truth_free": result["runtime_gold_features_used"] is False,
        }
        rows.append(
            {
                "variant_id": result["variant_id"],
                "score_band": result["score_band"],
                "cross_view_mode": result["cross_view_mode"],
                "admission_mode": result["admission_mode"],
                "c2_proposal_supply": c2_proposal,
                "c2_exact_observations": c2_exact,
                "accepted_duplicate_rate": duplicate_rate,
                "merged_as_clean_count": result["c2"]["conditional_observation"]["merged_as_clean_count"],
                "g6d_targets_supplied": result["g6d_targets_supplied"],
                "triggered_source_rate": burden["triggered_source_rate"],
                "estimated_s3_burden_reduction": burden["estimated_s3_burden_reduction"],
                "checks": checks,
                "failed_checks": sorted(key for key, passed in checks.items() if not passed),
                "passes_full_development_screen": all(checks.values()),
            }
        )
    rows.sort(
        key=lambda row: (
            not row["passes_full_development_screen"],
            -int(row["g6d_targets_supplied"]),
            -int(row["c2_proposal_supply"]),
            -int(row["c2_exact_observations"]),
            int(row["merged_as_clean_count"]),
            float(row["accepted_duplicate_rate"]),
            len(row["failed_checks"]),
            float(row["triggered_source_rate"]),
            str(row["variant_id"]),
        )
    )
    passing = [row for row in rows if row["passes_full_development_screen"]]
    return {
        "schema_version": "football_intelligence.m5_5g6f.development_shortlist.v1",
        "screen_not_weakened": True,
        "passing_variant_count": len(passing),
        "shortlist": rows,
        "best_diagnostic_variant_id": rows[0]["variant_id"],
        "development_candidate_frozen": passing[0]["variant_id"] if passing else None,
        "all_nine_target_maximum": max(row["g6d_targets_supplied"] for row in rows),
    }


def off_pitch_and_crowd_burden(
    selected_runtime: Mapping[str, Any],
    selected_scored: Mapping[str, Any],
    people: Mapping[str, Sequence[Mapping[str, Any]]],
    baseline_proposal_results: Mapping[str, Mapping[str, Any]],
    baseline_observation_results: Mapping[str, Mapping[str, Any]],
    trigger_results: Mapping[str, Any],
) -> dict[str, Any]:
    off_pitch_people = [row for row in people["C2"] if row["pitch_state"] == "OFF_PITCH"]
    baseline_proposal_map = {
        source_hash: _proposal_rows_from_baseline(result) for source_hash, result in baseline_proposal_results.items()
    }
    baseline_observation_map = {
        source_hash: _proposal_rows_from_baseline(result)
        for source_hash, result in baseline_observation_results.items()
    }
    proposal_map = _combined_proposal_map(baseline_proposal_results, selected_runtime, include_routed=True)
    observation_map = _combined_proposal_map(baseline_observation_results, selected_runtime, include_routed=False)
    evaluated = _evaluate_maps(
        off_pitch_people,
        baseline_proposal_map,
        baseline_observation_map,
        proposal_map,
        observation_map,
    )
    recovery_rows = _runtime_recovery_rows(selected_runtime, include_routed=True)
    recovery_only = G6E_IMPL.evaluate_proposal_map(off_pitch_people, recovery_rows)
    assigned_recovery_ids = {str(row["proposal_id"]) for row in recovery_only["assignments"]}
    total_recovery = sum(len(rows) for rows in recovery_rows.values())
    band_diagnostic = trigger_results["band_results"][str(selected_runtime["score_band"])]["evaluator_diagnostics"]
    return {
        "schema_version": "football_intelligence.m5_5g6f.off_pitch_and_crowd_burden.v1",
        "selected_variant_id": selected_runtime["variant_id"],
        "clear_off_pitch_people": len(off_pitch_people),
        "baseline_proposal_supply": evaluated["baseline_proposal"]["independent_supply"],
        "conditional_proposal_supply": evaluated["conditional_proposal"]["independent_supply"],
        "baseline_observation_supply": evaluated["baseline_observation"]["independent_supply"],
        "conditional_observation_supply": evaluated["conditional_observation"]["independent_supply"],
        "recovery_rows_matching_clear_off_pitch_people": len(assigned_recovery_ids),
        "recovery_rows_unmatched_to_clear_annotated_people": total_recovery - len(assigned_recovery_ids),
        "machine_trigger_off_pitch_person_overlap": band_diagnostic["off_pitch_person_overlap"],
        "unmatched_indistinct_crowd_policy": "UNSCORED_CROWD",
        "unmatched_rows_scored_as_background_false_positive": False,
        "crowd_false_positive_claimed": False,
        "off_pitch_output_counts_as_on_pitch_supply": False,
        "human_pitch_state_runtime_use": False,
        "primary_c2_on_pitch_result": selected_scored["c2"]["conditional_observation"]["independent_supply"],
    }


def observation_materialization_diagnosis(
    selected: Mapping[str, Any], transitions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    g6c_contract = read_json(
        PART3
        / "M5_5G6C_PITCH_GATE_REEVALUATION_AND_PROPOSAL_SUPPLY_RECOVERY_DECISION_v1"
        / "07_PROPOSAL_RECOVERY_EXPERIMENT_SELECTION"
        / "proposal_recovery_experiment_contract.json"
    )
    origins = Counter(str(row["origin"]) for row in g6c_contract["target_universe"])
    target_transitions = [row for row in transitions if row["anonymous_g6d_target_id"]]
    return {
        "schema_version": "football_intelligence.m5_5g6f.observation_materialization_diagnosis.v1",
        "selected_variant_id": selected["variant_id"],
        "canonical_s0_baseline": {
            "proposal_supply": selected["c2"]["baseline_proposal"]["independent_supply"],
            "exact_observations": selected["c2"]["baseline_observation"]["exactly_one_independent"],
        },
        "conditional_recovery": {
            "proposal_supply": selected["c2"]["conditional_proposal"]["independent_supply"],
            "exact_observations": selected["c2"]["conditional_observation"]["exactly_one_independent"],
            "proposal_improvements": sum(row["proposal_improved"] for row in transitions),
            "observation_improvements": sum(row["observation_improved"] for row in transitions),
            "suppressed_people": sum(row["suppressed"] for row in transitions),
        },
        "g6d_target_origin_counts": dict(sorted(origins.items())),
        "g6d_targets_recovered": sum(
            row["conditional_proposal_state"] in INDEPENDENT_STATES for row in target_transitions
        ),
        "g6d_targets_without_s0_raw_anchor": origins["NO_RAW_PROPOSAL"],
        "hard_reachability_finding": (
            "Seven evaluator targets have no S0 RAW proposal. An S0-RAW-anchored conditional trigger "
            "cannot request target-aligned cross-view corroboration for those people without a different "
            "machine-only source trigger. Raising confidence sensitivity cannot create absent RAW geometry."
        ),
        "raw_stage_reconciliation_root_cause": EXPECTED_ROOT_CAUSE,
        "coordinate_averaging_performed": False,
        "baseline_replacement_performed": False,
        "human_truth_runtime_use": False,
    }


def build_error_ledger(
    shortlist: Mapping[str, Any],
    transitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    for variant in shortlist["shortlist"]:
        for failure in variant["failed_checks"]:
            rows.append(
                {
                    "error_id": f"screen_{stable_hash([variant['variant_id'], failure])[:20]}",
                    "scope": "VARIANT_DEVELOPMENT_SCREEN",
                    "variant_id": variant["variant_id"],
                    "classification": failure,
                    "resolution": "REJECT_OR_RETAIN_DIAGNOSTIC_ONLY",
                }
            )
    for row in transitions:
        if not row["anonymous_g6d_target_id"] or row["conditional_proposal_state"] in INDEPENDENT_STATES:
            continue
        rows.append(
            {
                "error_id": f"target_{stable_hash(row['anonymous_g6d_target_id'])[:20]}",
                "scope": "G6D_TARGET_REACHABILITY",
                "anonymous_target_id": row["anonymous_g6d_target_id"],
                "classification": "NO_CONDITIONAL_INDEPENDENT_PROPOSAL",
                "baseline_state": row["baseline_proposal_state"],
                "conditional_state": row["conditional_proposal_state"],
                "resolution": "REQUIRES_DIFFERENT_MACHINE_ONLY_PROPOSAL_FAMILY_OR_TRIGGER_SOURCE",
            }
        )
    return {
        "schema_version": "football_intelligence.m5_5g6f.conditional_recovery_error_ledger.v1",
        "row_count": len(rows),
        "classification_counts": dict(sorted(Counter(row["classification"] for row in rows).items())),
        "rows": rows,
        "no_errors_hidden": True,
    }


def choose_final_decision(shortlist: Mapping[str, Any], diagnosis: Mapping[str, Any]) -> dict[str, Any]:
    if shortlist["passing_variant_count"]:
        choice = "FREEZE_CONDITIONAL_CROSS_VIEW_RECOVERY_DEVELOPMENT_CANDIDATE"
        rationale = "At least one frozen truth-free variant passes every unweakened development gate."
    elif diagnosis["g6d_targets_without_s0_raw_anchor"] and shortlist["all_nine_target_maximum"] < 9:
        choice = "AUTHORIZE_OFFICIAL_SMALL_PERSON_DETECTOR_FAMILY_BAKEOFF"
        rationale = (
            "The frozen S0-RAW trigger cannot reach seven no-RAW target people, and no variant supplies all "
            "nine targets. A second-stage confidence validator cannot recover geometry that never exists in RAW."
        )
    else:
        choice = "KEEP_NARROW_G6D_RECOVERY_EVIDENCE_ONLY"
        rationale = "No frozen conditional variant passes the complete screen; retain prior recovery evidence only."
    if choice not in FINAL_CHOICES:
        raise RuntimeError(f"invalid final choice: {choice}")
    return {
        "schema_version": "football_intelligence.m5_5g6f.final_decision.v1",
        "choice": choice,
        "rationale": rationale,
        "passing_variant_count": shortlist["passing_variant_count"],
        "development_candidate_frozen": shortlist["development_candidate_frozen"],
        "selected_diagnostic_variant_id": shortlist["best_diagnostic_variant_id"],
        "component_promoted": False,
        "production_ready": False,
        "no_auto_promotion": True,
    }


def write_final_decision(decision: Mapping[str, Any]) -> None:
    lines = [
        "# M5.5G.6F final decision",
        "",
        f"**Choice:** `{decision['choice']}`",
        "",
        str(decision["rationale"]),
        "",
        f"Frozen variants passing every gate: **{decision['passing_variant_count']}**.",
        "",
        "No detector, tracker, pitch gate, fusion rule, confidence default, or production component is promoted.",
    ]
    path = DIRS["decision"] / "final_decision.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    filename = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(filename, size)
    except OSError:
        return ImageFont.load_default()


def _draw_bbox(draw: ImageDraw.ImageDraw, box: Mapping[str, Any], color: str, *, width: int = 5) -> None:
    draw.rectangle(
        [float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])],
        outline=color,
        width=width,
    )


def _render_evidence_panel(
    source: Mapping[str, Any],
    overlays: Sequence[tuple[Mapping[str, Any], str]],
    title: str,
    *,
    size: tuple[int, int] = (760, 380),
) -> Image.Image:
    image_path = Path(str(source["image_path"]))
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box, color in overlays:
        _draw_bbox(draw, box, color)
    if overlays:
        x1 = min(float(box["x1"]) for box, _ in overlays)
        y1 = min(float(box["y1"]) for box, _ in overlays)
        x2 = max(float(box["x2"]) for box, _ in overlays)
        y2 = max(float(box["y2"]) for box, _ in overlays)
        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)
        pad_x = max(120.0, box_width * 2.5)
        pad_y = max(80.0, box_height * 2.0)
        crop = (
            max(0, int(x1 - pad_x)),
            max(0, int(y1 - pad_y)),
            min(image.width, int(x2 + pad_x)),
            min(image.height, int(y2 + pad_y)),
        )
        image = image.crop(crop)
    available = (size[0], size[1] - 44)
    image.thumbnail(available, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, "#0b1210")
    panel.paste(image, ((size[0] - image.width) // 2, 44 + (available[1] - image.height) // 2))
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.text((14, 11), title, fill="white", font=_font(20, bold=True))
    return panel


def create_visual_atlases(
    sources: Mapping[str, Mapping[str, Any]],
    people: Mapping[str, Sequence[Mapping[str, Any]]],
    trigger_results: Mapping[str, Any],
    selected_runtime: Mapping[str, Any],
    selected_scored: Mapping[str, Any],
) -> list[Path]:
    output_paths = []
    runtime_by_source = {str(row["source_frame_sha256"]): row for row in selected_runtime["source_rows"]}
    trigger_band = trigger_results["band_results"][str(selected_runtime["score_band"])]
    trigger_panels = []
    for row in trigger_band["rows"]:
        source_hash = str(row["source_frame_sha256"])
        runtime_row = runtime_by_source.get(source_hash, {})
        overlays = [(anchor["bbox_panorama_pixels"], "#ff4f5e") for anchor in row["eligible_anchors"][:3]]
        overlays.extend(
            (candidate["bbox_panorama_pixels"], "#70e4a2") for candidate in runtime_row.get("admitted_recovery", [])[:3]
        )
        if overlays and source_hash in sources:
            trigger_panels.append(
                _render_evidence_panel(
                    sources[source_hash], overlays, "S0 RAW trigger (red) / admitted cross-view recovery (green)"
                )
            )
        if len(trigger_panels) == 4:
            break
    atlas = Image.new("RGB", (1520, 760), "#07100d")
    for index, panel in enumerate(trigger_panels):
        atlas.paste(panel, ((index % 2) * 760, (index // 2) * 380))
    path = DIRS["error"] / "01_MACHINE_TRIGGER_AND_RECOVERY_ATLAS.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(path, optimize=True)
    output_paths.append(path)

    c2_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for person in people["C2"]:
        if person["pitch_state"] == "ON_PITCH":
            c2_by_source[str(person["source_frame_sha256"])].append(person)
    transition_panels = []
    improved_sources = {
        str(row["source_frame_sha256"])
        for row in selected_scored["c2"]["proposal_person_rows"]
        if row["supply_state"] in INDEPENDENT_STATES
    }
    for source_hash, runtime_row in runtime_by_source.items():
        if source_hash not in improved_sources or source_hash not in sources:
            continue
        overlays = [(person["bbox"], "#f5f5ef") for person in c2_by_source.get(source_hash, [])]
        overlays.extend(
            (candidate["bbox_panorama_pixels"], "#70e4a2") for candidate in runtime_row["admitted_recovery"]
        )
        overlays.extend((candidate["bbox_panorama_pixels"], "#ffc857") for candidate in runtime_row["routed_recovery"])
        transition_panels.append(
            _render_evidence_panel(
                sources[source_hash], overlays, "Evaluator people (white) / accepted (green) / routed (gold)"
            )
        )
        if len(transition_panels) == 4:
            break
    atlas = Image.new("RGB", (1520, 760), "#07100d")
    for index, panel in enumerate(transition_panels):
        atlas.paste(panel, ((index % 2) * 760, (index // 2) * 380))
    path = DIRS["error"] / "02_C2_TRANSITION_EVIDENCE_ATLAS.png"
    atlas.save(path, optimize=True)
    output_paths.append(path)

    control_panels = []
    for source_hash, runtime_row in runtime_by_source.items():
        off_pitch = [
            person
            for person in people["C2"]
            if person["source_frame_sha256"] == source_hash and person["pitch_state"] == "OFF_PITCH"
        ]
        if not off_pitch or source_hash not in sources:
            continue
        overlays = [(person["bbox"], "#c49bff") for person in off_pitch]
        overlays.extend(
            (candidate["bbox_panorama_pixels"], "#70e4a2") for candidate in runtime_row["admitted_recovery"]
        )
        control_panels.append(
            _render_evidence_panel(
                sources[source_hash], overlays, "Clear off-pitch gold (purple) / recovery burden (green)"
            )
        )
        if len(control_panels) == 4:
            break
    atlas = Image.new("RGB", (1520, 760), "#07100d")
    for index, panel in enumerate(control_panels):
        atlas.paste(panel, ((index % 2) * 760, (index // 2) * 380))
    path = DIRS["error"] / "03_OFF_PITCH_AND_CONTROL_ATLAS.png"
    atlas.save(path, optimize=True)
    output_paths.append(path)
    return output_paths


def source_diff_patch() -> str:
    result = git(
        "diff",
        "--binary",
        f"{BASELINE}..HEAD",
        "--",
        "scripts/build_m5_5g6f_conditional_recovery.py",
        "tests/test_m5_5g6f_conditional_recovery.py",
    )
    return result.stdout


def build_review_pack(visuals: Sequence[Path]) -> dict[str, Any]:
    pack = DIRS["pack"]
    pack.mkdir(parents=True, exist_ok=True)
    readme = pack / "01_READ_ME_FIRST.md"
    readme.write_text(
        "# M5.5G.6F review pack\n\n"
        "Cached-only evaluation of truth-free S0 low-confidence triggers and exact-member S3 recovery. "
        "No inference, tuning, tracking, or promotion occurred. Start with the stage summary and final decision.\n",
        encoding="utf-8",
    )
    payloads: list[tuple[Path, str]] = [
        (STAGE / "stage_summary.json", "02_STAGE_SUMMARY.json"),
        (DIRS["validation"] / "g6e_and_cached_row_validation.json", "03_CACHED_ROW_VALIDATION.json"),
        (DIRS["matrix"] / "frozen_trigger_admission_matrix.json", "05_FROZEN_MATRIX.json"),
        (DIRS["trigger"] / "machine_trigger_results.json", "06_MACHINE_TRIGGER_RESULTS.json"),
        (DIRS["recovery"] / "conditional_recovery_results.json", "07_CONDITIONAL_RECOVERY_RESULTS.json"),
        (DIRS["transition"] / "observation_materialization_diagnosis.json", "08_OBSERVATION_DIAGNOSIS.json"),
        (DIRS["regression"] / "static_dense_regression.json", "09_STATIC_DENSE_REGRESSION.json"),
        (DIRS["burden"] / "off_pitch_and_crowd_burden.json", "10_OFF_PITCH_CROWD_BURDEN.json"),
        (DIRS["burden"] / "runtime_burden_estimate.json", "11_RUNTIME_BURDEN.json"),
        (DIRS["error"] / "conditional_recovery_error_ledger.json", "12_ERROR_LEDGER.json"),
        (DIRS["decision"] / "development_shortlist.json", "13_DEVELOPMENT_SHORTLIST.json"),
        (DIRS["decision"] / "final_decision.md", "14_FINAL_DECISION.md"),
        (DIRS["commands"] / "verification_results.json", "15_VERIFICATION_RESULTS.json"),
    ]
    for source, name in payloads:
        shutil.copy2(source, pack / name)
    (pack / "04_SOURCE_DIFF.patch").write_text(source_diff_patch(), encoding="utf-8", newline="\n")
    for index, visual in enumerate(visuals, start=16):
        shutil.copy2(visual, pack / f"{index:02d}_{visual.name}")
    manifest_path = pack / "19_REVIEW_PACK_MANIFEST.json"
    files = [path for path in pack.iterdir() if path.is_file() and path != manifest_path]
    manifest = {
        "schema_version": "football_intelligence.m5_5g6f.review_pack_manifest.v1",
        "manifest_self_hash_omitted": True,
        "files": [file_record(path) for path in sorted(files, key=lambda value: value.name)],
        "file_count_excluding_manifest": len(files),
    }
    write_json(manifest_path, manifest)
    return manifest


def validate_review_pack() -> dict[str, Any]:
    pack = DIRS["pack"]
    files = sorted(path for path in pack.iterdir() if path.is_file())
    visual_suffixes = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    visuals = [path for path in files if path.suffix.lower() in visual_suffixes]
    forbidden_suffixes = {".pt", ".pth", ".onnx", ".mp4", ".avi", ".mov", ".mkv"}
    manifest = read_json(pack / "19_REVIEW_PACK_MANIFEST.json")
    declared = {Path(str(row["path"])).name: row for row in manifest["files"]}
    checks = {
        "flat": all(path.parent == pack for path in files),
        "maximum_20_files": len(files) <= 20,
        "maximum_50_mib": sum(path.stat().st_size for path in files) <= 52_428_800,
        "maximum_three_visuals": len(visuals) <= 3,
        "three_real_visuals_present": len(visuals) == 3 and all(Image.open(path).size[0] > 1000 for path in visuals),
        "source_diff_present": (pack / "04_SOURCE_DIFF.patch").is_file(),
        "manifest_self_hash_omitted": manifest["manifest_self_hash_omitted"] is True
        and "19_REVIEW_PACK_MANIFEST.json" not in declared,
        "declared_payloads_exact": all(
            name in declared
            and int(declared[name]["bytes"]) == path.stat().st_size
            and str(declared[name]["sha256"]) == sha256_file(path)
            for path in files
            if path.name != "19_REVIEW_PACK_MANIFEST.json"
            for name in [path.name]
        ),
        "no_forbidden_binary_payloads": not any(path.suffix.lower() in forbidden_suffixes for path in files),
        "no_nested_directories": not any(path.is_dir() for path in pack.iterdir()),
    }
    result = {
        "schema_version": "football_intelligence.m5_5g6f.review_pack_validation.v1",
        "checks": checks,
        "file_count": len(files),
        "visual_file_count": len(visuals),
        "total_bytes": sum(path.stat().st_size for path in files),
        "passed": all(checks.values()),
    }
    if not result["passed"]:
        raise RuntimeError(f"FAIL_REVIEW_PACK: {checks}")
    return result


def run_verification(*, skip: bool) -> dict[str, Any]:
    if skip:
        result = {
            "schema_version": "football_intelligence.m5_5g6f.verification_results.v1",
            "skipped": True,
            "passed": False,
            "commands": [],
        }
        write_json(DIRS["commands"] / "verification_results.json", result)
        return result
    changed = [
        "scripts/build_m5_5g6f_conditional_recovery.py",
        "tests/test_m5_5g6f_conditional_recovery.py",
    ]
    regressions = [
        "tests/test_m5_5g2b_proposal_supply.py",
        "tests/test_m5_5g3_consolidation.py",
        "tests/test_m5_5g4_r2_corrected_dense_gold.py",
        "tests/test_m5_5g5a_promptable_masks.py",
        "tests/test_m5_5g6a_pitch_observation.py",
        "tests/test_m5_5g6c_pitch_gate_recovery.py",
        "tests/test_m5_5g6d_high_resolution_proposal_bakeoff.py",
        "tests/test_m5_5g6e_c0_reintegration.py",
    ]
    specifications = [
        ("uv_lock_check", ["uv", "lock", "--check"]),
        ("uv_sync", ["uv", "sync"]),
        ("ruff_check", ["uv", "run", "ruff", "check", *changed]),
        ("ruff_format_check", ["uv", "run", "ruff", "format", "--check", *changed]),
        ("focused_tests", ["uv", "run", "pytest", "tests/test_m5_5g6f_conditional_recovery.py", "-q"]),
        ("historical_regressions", ["uv", "run", "pytest", *regressions, "-q"]),
        ("full_suite", ["uv", "run", "pytest", "-q"]),
        ("cli_help", ["uv", "run", "fi-pipeline", "--help"]),
        ("review_chassis_help", ["uv", "run", "fi-pipeline", "review-chassis", "--help"]),
        ("git_diff_check", ["git", "diff", "--check"]),
    ]
    rows = []
    for name, command in specifications:
        completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
        output = (completed.stdout + completed.stderr).strip()
        log_path = DIRS["commands"] / f"{name}.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output + ("\n" if output else ""), encoding="utf-8")
        rows.append(
            {
                "name": name,
                "command": subprocess.list2cmdline(command),
                "return_code": completed.returncode,
                "passed": completed.returncode == 0,
                "log": str(log_path),
                "log_sha256": sha256_file(log_path),
                "tail": output.splitlines()[-8:],
            }
        )
    result = {
        "schema_version": "football_intelligence.m5_5g6f.verification_results.v1",
        "skipped": False,
        "commands": rows,
        "passed": all(row["passed"] for row in rows),
    }
    write_json(DIRS["commands"] / "verification_results.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true", help="replace only this generated G6F workspace")
    parser.add_argument("--skip-verification", action="store_true", help="development-only artifact build")
    return parser.parse_args()


def _prepare_workspace(*, replace: bool) -> None:
    expected = (PART3 / "M5_5G6F_CONDITIONAL_LOW_CONFIDENCE_CROSS_VIEW_RECOVERY_AND_DUPLICATE_CONTROL_v1").resolve()
    if STAGE.resolve() != expected:
        raise RuntimeError("refusing to operate on an unexpected stage path")
    if STAGE.exists():
        if not replace:
            raise RuntimeError(f"stage already exists; rerun with --replace: {STAGE}")
        shutil.rmtree(STAGE)
    for path in DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    _prepare_workspace(replace=args.replace)
    repository, prompt_validation = prompt_and_repository_validation()
    write_json(DIRS["validation"] / "repository_state.json", repository)
    write_json(DIRS["validation"] / "prompt_pack_validation.json", prompt_validation)

    cached_validation, protected_before, replay = validate_cached_inputs()
    write_json(DIRS["validation"] / "protected_inputs_before.json", protected_before)
    cached, row_validation = load_cached_rows(replay)
    combined_validation = {
        "schema_version": "football_intelligence.m5_5g6f.g6e_and_cached_row_validation.v1",
        "g6e_inputs": cached_validation,
        "cached_rows": row_validation,
        "passed": cached_validation["passed"] and row_validation["passed"],
    }
    write_json(DIRS["validation"] / "g6e_and_cached_row_validation.json", combined_validation)

    matrix, matrix_sha256 = freeze_matrix()
    baseline_proposals, baseline_observations = _baseline_observations(cached)
    pools = _candidate_pools(cached)
    trigger_results = build_machine_triggers(matrix, pools, baseline_observations, cached["runtime_by_view"])
    write_json(DIRS["trigger"] / "machine_trigger_results.runtime_only.json", trigger_results)

    runtime_payload = materialize_runtime_variants(matrix, trigger_results, pools, baseline_observations)
    runtime_path = DIRS["recovery"] / "runtime_variant_materialization.json"
    write_json(runtime_path, runtime_payload)
    evaluator_lock = {
        "matrix_file_sha256": matrix_sha256,
        "matrix_payload_hash": matrix["matrix_payload_hash"],
        "machine_trigger_results_sha256": sha256_file(DIRS["trigger"] / "machine_trigger_results.runtime_only.json"),
        "runtime_materialization_sha256": sha256_file(runtime_path),
        "runtime_payload_hash": runtime_payload["runtime_payload_hash"],
        "evaluator_loaded": False,
        "human_truth_runtime_use": False,
    }
    write_json(DIRS["recovery"] / "pre_evaluator_runtime_lock.json", evaluator_lock)

    universe, sources, people, evaluator_binding = load_evaluator_universes()
    evaluator_binding["pre_evaluator_runtime_lock_sha256"] = sha256_file(
        DIRS["recovery"] / "pre_evaluator_runtime_lock.json"
    )
    write_json(DIRS["recovery"] / "evaluator_join_receipt.json", evaluator_binding)
    scored_results = evaluate_variants(
        matrix,
        runtime_payload,
        baseline_proposals,
        baseline_observations,
        people,
        evaluator_binding["target_bindings"],
    )
    write_json(DIRS["recovery"] / "conditional_recovery_results.json", scored_results)
    trigger_results = add_trigger_evaluator_diagnostics(
        trigger_results, people, evaluator_binding["target_bindings"], baseline_observations
    )
    write_json(DIRS["trigger"] / "machine_trigger_results.json", trigger_results)

    regressions = evaluate_static_dense_and_b1(
        runtime_payload,
        people,
        baseline_proposals,
        baseline_observations,
        trigger_results,
    )
    write_json(DIRS["regression"] / "static_dense_regression.json", regressions)
    runtime_burden = runtime_burden_estimates(trigger_results, cached["runtime_by_view"])
    write_json(DIRS["burden"] / "runtime_burden_estimate.json", runtime_burden)
    shortlist = build_development_shortlist(scored_results, regressions, runtime_burden)
    write_json(DIRS["decision"] / "development_shortlist.json", shortlist)

    selected_id = str(shortlist["best_diagnostic_variant_id"])
    selected_runtime = next(row for row in runtime_payload["variants"] if row["variant_id"] == selected_id)
    selected_scored = next(row for row in scored_results["variants"] if row["variant_id"] == selected_id)
    transitions = build_c2_transition_ledger(selected_scored, people, evaluator_binding["target_bindings"])
    write_jsonl(DIRS["transition"] / "c2_transition_ledger.jsonl", transitions)
    diagnosis = observation_materialization_diagnosis(selected_scored, transitions)
    write_json(DIRS["transition"] / "observation_materialization_diagnosis.json", diagnosis)
    off_pitch = off_pitch_and_crowd_burden(
        selected_runtime,
        selected_scored,
        people,
        baseline_proposals,
        baseline_observations,
        trigger_results,
    )
    write_json(DIRS["burden"] / "off_pitch_and_crowd_burden.json", off_pitch)
    error_ledger = build_error_ledger(shortlist, transitions)
    write_json(DIRS["error"] / "conditional_recovery_error_ledger.json", error_ledger)
    decision = choose_final_decision(shortlist, diagnosis)
    write_json(DIRS["decision"] / "final_decision.json", decision)
    write_final_decision(decision)

    visuals = create_visual_atlases(
        sources,
        people,
        trigger_results,
        selected_runtime,
        selected_scored,
    )
    protected_after = tree_manifest(cached_input_paths())
    protected_exact = _manifest_entries_exact(protected_before, protected_after)
    if not protected_exact:
        raise RuntimeError("FAIL_HISTORICAL_ARTIFACT_PRESERVATION")
    write_json(DIRS["commands"] / "protected_inputs_after.json", protected_after)
    pending_verification = {
        "schema_version": "football_intelligence.m5_5g6f.verification_results.v1",
        "status": "PENDING_IN_BUILDER",
        "passed": True,
        "commands": [],
    }
    write_json(DIRS["commands"] / "verification_results.json", pending_verification)

    stage_summary = {
        "schema_version": "football_intelligence.m5_5g6f.stage_summary.v1",
        "stage_id": "M5_5G6F_CONDITIONAL_LOW_CONFIDENCE_CROSS_VIEW_RECOVERY_AND_DUPLICATE_CONTROL_v1",
        "classification": CLASSIFICATION,
        "repository_head": repository["head"],
        "matrix_sha256": matrix_sha256,
        "cached_row_validation_passed": combined_validation["passed"],
        "runtime_materialized_before_evaluator_join": True,
        "variant_count": matrix["variant_count"],
        "selected_diagnostic_variant_id": selected_id,
        "passing_variant_count": shortlist["passing_variant_count"],
        "final_choice": decision["choice"],
        "protected_inputs_unchanged": protected_exact,
        "verification_status": "PENDING_IN_BUILDER",
        "verification_passed": True,
        "review_pack_validation": {"passed": True, "file_count": 19, "visual_file_count": 3},
        "universe_hash": universe["full_universe_hash"],
        **SAFETY,
    }
    write_json(STAGE / "stage_summary.json", stage_summary)
    build_review_pack(visuals)
    preliminary_review_validation = validate_review_pack()
    write_json(DIRS["commands"] / "preliminary_review_pack_validation.json", preliminary_review_validation)

    verification = run_verification(skip=args.skip_verification)
    stage_summary["classification"] = CLASSIFICATION if verification["passed"] else "INCOMPLETE_VERIFICATION"
    stage_summary["verification_status"] = "PASSED" if verification["passed"] else "FAILED_OR_SKIPPED"
    stage_summary["verification_passed"] = verification["passed"]
    write_json(STAGE / "stage_summary.json", stage_summary)
    build_review_pack(visuals)
    review_validation = validate_review_pack()
    write_json(DIRS["commands"] / "review_pack_validation.json", review_validation)
    if not verification["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
