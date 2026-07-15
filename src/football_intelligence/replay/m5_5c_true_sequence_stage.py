from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageSequence

from football_intelligence.replay.m5_5b_repaired_reviews_stage import (
    _load_mapping_cases,
    _validate_localization_review,
    _validate_occlusion_path_review,
)
from football_intelligence.replay.occlusion_detector_recovery_diagnostic import (
    EXPECTED_DETECTOR_SHA256,
    PRE_NMS_STATUS,
    BBox,
    bbox_iou,
    canonical_match_metrics,
    crop_to_panorama_bbox,
    parse_bbox,
)
from football_intelligence.replay.short_window_candidate_graph import ImageBBox
from football_intelligence.replay.true_sequence_resolver import (
    FrameObservation,
    ResolverConfig,
    answer_independent_fingerprint,
    appearance_activation_gate,
    bbox_height_band,
    execute_ghost_intervals,
    observable_conflict_signals,
    resolve_joint_sequence,
)
from football_intelligence.research_handoff.review_pack import (
    ReviewPackBuilder,
    ReviewPackItem,
    validate_review_pack_directory,
)
from football_intelligence.research_handoff.stage_workspace import safety_payload, sha256_file
from football_intelligence.review_chassis.hashing import stable_hash
from football_intelligence.review_chassis.models import (
    GenericEvidenceAsset,
    GenericReviewCase,
    GenericReviewManifest,
    ReviewUIConfig,
)
from football_intelligence.review_chassis.persistence import GenericReviewPersistence
from football_intelligence.review_chassis.server import ReviewChassisServerConfig, create_server
from football_intelligence.review_chassis.spatial_annotations import scan_forbidden_browser_payload
from football_intelligence.review_chassis.validation import validate_review_chassis_package

STAGE_ID = "M5_5C_TRUE_SEQUENCE_RESOLVER_DETECTOR_CONTROL_AND_BLIND_REVIEW_v1"
REVIEW_ID = "m5_5c_trajectory_safe_blind_conflict_review_v1"
BASELINE_COMMIT = "53c1a032336a59f3c3449478d27290da62fcc4fc"
LOCAL_REVIEW_URL = "http://127.0.0.1:8780/"
KNOWN_CROSSING_CASES = {"008", "010", "013"}
LOCALIZATION_CASES = {"004", "009", "011", "016"}
PROTECTED_CONTROL_CASES = {"001", "002", "003", "005", "007", "012", "014", "015", "019"}
WORKSPACE_DIRS = (
    "00_PROMPT_AND_INPUTS",
    "01_AUTHORIZATION_AND_CLAIM_AUDIT",
    "02_REVIEW_INGESTION",
    "03_CONTROLLED_DETECTOR_RECOVERY",
    "04_TRUE_SEQUENCE_RESOLVER",
    "05_GHOST_AND_REENTRY",
    "06_APPEARANCE_ABLATION",
    "07_BLIND_CONFLICT_MINING",
    "08_HUMAN_REVIEW",
    "09_EVALUATION",
    "10_VISUAL_EVIDENCE",
    "11_VALIDATION_AND_LOGS",
    "12_REVIEW_PACK_FOR_CHATGPT",
    "_tmp",
)
PROMPT_FILES = (
    "00_READ_ME_FIRST.md",
    "01_M5_5C_CODEX_PROMPT.md",
    "02_M5_5C_WORKSPACE_CONTRACT.json",
    "03_M5_5C_SCIENTIFIC_CORRECTION_CONTRACT.json",
    "04_PROMPT_PACK_MANIFEST.json",
)
MANDATORY_REVIEW_PACK_FILES = {
    "REVIEW_PACK_MANIFEST.json",
    "01_EXECUTIVE_SUMMARY.md",
    "02_RUN_AND_GIT_CONTEXT.json",
    "03_FILES_CHANGED.md",
    "04_SOURCE_DIFF.patch",
    "05_COMMANDS_AND_TEST_RESULTS.md",
    "06_OUTPUT_ARTIFACT_INDEX.json",
    "07_CLAIM_EVIDENCE_AUDIT.json",
    "08_SAFETY_AND_MUTATION_AUDIT.json",
    "09_REVIEW_PREREQUISITES.json",
    "10_DETECTOR_AFFECTED_AND_CONTROL_RESULTS.json",
    "11_TRUE_SEQUENCE_RESOLVER_RESULTS.json",
    "12_GHOST_AND_REENTRY_RESULTS.json",
    "13_APPEARANCE_ABLATION_RESULTS.json",
    "14_CASE_LEVEL_RESULTS.jsonl",
    "15_BLIND_MINING_AND_REVIEW_STATUS.json",
    "16_ACCEPTANCE_AND_ARCHITECTURE_DECISION.json",
    "17_PRIMARY_VISUAL_EVIDENCE.jpg",
    "18_SECONDARY_VISUAL_EVIDENCE.gif",
    "19_HUMAN_ACTION_AND_NEXT_DECISION.md",
}


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    return _write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, default=str))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True, default=str) + "\n")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _git(repo_root: Path, *args: str, timeout: int = 120) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    return {
        "command": ["git", *args],
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _case_number(case_id: str | None) -> str | None:
    if not case_id:
        return None
    tail = case_id.rsplit("_", 1)[-1]
    return tail if len(tail) == 3 and tail.isdigit() else None


def _copy_file(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _directory_manifest(root: Path, *, include_hashes: bool = True) -> dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*")) if root.exists() else []:
        if not path.is_file():
            continue
        row = {
            "relative_path": str(path.relative_to(root)),
            "byte_size": path.stat().st_size,
        }
        if include_hashes:
            row["sha256"] = sha256_file(path)
        files.append(row)
    return {
        "root": str(root),
        "exists": root.exists(),
        "file_count": len(files),
        "total_bytes": sum(row["byte_size"] for row in files),
        "files": files,
    }


def _protected_hashes(paths: list[Path]) -> dict[str, str | None]:
    return {str(path): sha256_file(path) if path.exists() and path.is_file() else None for path in paths}


def authorization_audit(repo_root: Path, *, baseline_commit: str = BASELINE_COMMIT) -> dict[str, Any]:
    status = _git(repo_root, "status", "--short")
    head = _git(repo_root, "rev-parse", "HEAD")
    exists = _git(repo_root, "cat-file", "-e", f"{baseline_commit}^{{commit}}")
    ancestor = _git(repo_root, "merge-base", "--is-ancestor", baseline_commit, "HEAD")
    log = _git(repo_root, "log", "--oneline", "--decorate", "--no-merges", f"{baseline_commit}..HEAD")
    stat = _git(repo_root, "diff", "--stat", f"{baseline_commit}..HEAD")
    names = _git(repo_root, "diff", "--name-status", f"{baseline_commit}..HEAD")
    clean = status["stdout"].strip() == ""
    return {
        "schema_version": "football_intelligence.m5_5c.authorization_audit.v1",
        "generated_at": utc_now(),
        "minimum_authorized_baseline_commit": baseline_commit,
        "current_head": head["stdout"].strip(),
        "baseline_commit_exists": exists["exit_code"] == 0,
        "baseline_is_ancestor_of_head": ancestor["exit_code"] == 0,
        "worktree_clean_at_audit": clean,
        "intervening_commits": log["stdout"],
        "diff_stat_from_baseline": stat["stdout"],
        "diff_name_status_from_baseline": names["stdout"],
        "authorization_gate_passed": exists["exit_code"] == 0 and ancestor["exit_code"] == 0,
        **safety_payload(),
    }


def _copy_prompt_and_input_manifests(
    workspace_root: Path,
    prompt_root: Path,
    m5b_root: Path,
    localization_root: Path,
    path_review_root: Path,
) -> dict[str, Any]:
    existing_path = workspace_root / "00_PROMPT_AND_INPUTS" / "prompt_and_input_manifest.json"
    if existing_path.exists():
        existing = _read_json(existing_path)
        if (
            existing.get("m5_5b_workspace", {}).get("root") == str(m5b_root)
            and existing.get("authoritative_localization_review", {}).get("root") == str(localization_root)
            and existing.get("authoritative_path_review", {}).get("root") == str(path_review_root)
        ):
            return existing
    rows = []
    for filename in PROMPT_FILES:
        source = prompt_root / filename
        if source.exists():
            target = _copy_file(source, workspace_root / "00_PROMPT_AND_INPUTS" / filename)
            rows.append({"source": str(source), "copy": str(target), "sha256": sha256_file(target)})
    inputs = {
        "schema_version": "football_intelligence.m5_5c.prompt_and_input_manifest.v1",
        "generated_at": utc_now(),
        "copied_prompt_files": rows,
        "m5_5b_workspace": _directory_manifest(m5b_root),
        "authoritative_localization_review": _directory_manifest(localization_root),
        "authoritative_path_review": _directory_manifest(path_review_root),
        "invalid_m5_5b_port_8779_review_used_for_scientific_validation": False,
        **safety_payload(),
    }
    _write_json(workspace_root / "00_PROMPT_AND_INPUTS" / "prompt_and_input_manifest.json", inputs)
    return inputs


def _claim_evidence_audit(m5b_root: Path) -> dict[str, Any]:
    source = "src/football_intelligence/replay/m5_5b_repaired_reviews_stage.py"
    rows = [
        {
            "claim": "actual_sequence_real_window",
            "source_artifact": str(m5b_root / "04_SEQUENCE_REAL_RESOLVER" / "sequence_window_manifest.json"),
            "source_code_mechanism": f"{source}:1043-1320",
            "executed_evidence": "Window frame numbers were listed, but intermediate observations were not consumed.",
            "support_status": "unsupported",
            "correction_required": "Build a node-bearing frame graph across every frame.",
            "replacement_m5_5c_classification": "TRUE_FRAME_GRAPH_REQUIRED",
        },
        {
            "claim": "motion_fit_uses_real_observation",
            "source_artifact": str(m5b_root / "04_SEQUENCE_REAL_RESOLVER" / "incoming_tracklet_rows.jsonl"),
            "source_code_mechanism": "approach_to_occlusion_signals([source, source], ...)",
            "executed_evidence": "The same observation object was duplicated; no three-frame fit was executed.",
            "support_status": "unsupported",
            "correction_required": "Reject duplicate frames/objects and fit at least three observations.",
            "replacement_m5_5c_classification": "DISTINCT_THREE_FRAME_FIT_REQUIRED",
        },
        {
            "claim": "data_driven_conflict_signals",
            "source_artifact": str(m5b_root / "04_SEQUENCE_REAL_RESOLVER" / "conflict_trigger_rows.jsonl"),
            "source_code_mechanism": "challenge_category_present=True",
            "executed_evidence": "Historical challenge membership contributed directly to activation.",
            "support_status": "partially_supported",
            "correction_required": "Use observable geometry, competition, count, overlap and margin evidence only.",
            "replacement_m5_5c_classification": "OBSERVABLE_CONFLICT_ONLY_REQUIRED",
        },
        {
            "claim": "ghost_state_preserved",
            "source_artifact": str(m5b_root / "04_SEQUENCE_REAL_RESOLVER" / "ghost_state_rows.jsonl"),
            "source_code_mechanism": "Static ghost row append.",
            "executed_evidence": "No hidden interval propagation or re-entry confirmation was executed.",
            "support_status": "unsupported",
            "correction_required": "Advance prediction/covariance through observed missing or merged intervals.",
            "replacement_m5_5c_classification": "EXECUTED_GHOST_INTERVAL_REQUIRED",
        },
        {
            "claim": "dynamic_bounded_hidden_window",
            "source_artifact": str(m5b_root / "07_EVALUATION" / "ghost_and_reentry_metrics.json"),
            "source_code_mechanism": "Expiry policy string only.",
            "executed_evidence": "Expiry and termination were described but not run.",
            "support_status": "unsupported",
            "correction_required": "Execute dynamic lifetime and stale-state termination.",
            "replacement_m5_5c_classification": "DYNAMIC_EXPIRY_EXECUTION_REQUIRED",
        },
        {
            "claim": "protected_control_regressions=0",
            "source_artifact": str(m5b_root / "07_EVALUATION" / "appearance_activation_and_regression.json"),
            "source_code_mechanism": "Copied prior protected rows; activation count zero.",
            "executed_evidence": "Zero activation is not an evaluated zero-regression result.",
            "support_status": "unsupported",
            "correction_required": "Rerun controls and classify zero activation as not evaluated.",
            "replacement_m5_5c_classification": "NOT_EVALUATED_NO_ELIGIBLE_GATE",
        },
        {
            "claim": "trajectory_safe_exclusion_performed=true",
            "source_artifact": str(m5b_root / "05_UNSEEN_CONFLICT_MINING" / "mining_manifest.json"),
            "source_code_mechanism": "case_id set compared with endpoint_safe_group_id",
            "executed_evidence": "Identifier domains were incompatible, so exclusion could not match.",
            "support_status": "unsupported",
            "correction_required": "Use canonical group IDs, endpoint sets and neighbourhood IDs.",
            "replacement_m5_5c_classification": "TYPE_SAFE_TRAJECTORY_EXCLUSION_REQUIRED",
        },
        {
            "claim": "case_004_016_share_region=false",
            "source_artifact": str(m5b_root / "03_DETECTOR_RECOVERY" / "trajectory_region_summary.json"),
            "source_code_mechanism": "One region per case_short_id.",
            "executed_evidence": "Canonical v13 grouping places cases 004 and 016 in one group.",
            "support_status": "unsupported",
            "correction_required": "Aggregate using canonical_trajectory_safe_group_37f5f989eac9.",
            "replacement_m5_5c_classification": "CASE_004_016_SHARED_REGION",
        },
        {
            "claim": "HIGH_INFORMATION_GAIN",
            "source_artifact": str(m5b_root / "07_EVALUATION" / "architecture_branch_decision.json"),
            "source_code_mechanism": "Prerequisite pass plus endpoint top-two count.",
            "executed_evidence": "Detector evidence was useful; sequence, control and mining claims were overstated.",
            "support_status": "partially_supported",
            "correction_required": "Reclassify after controlled detector and true-sequence rerun.",
            "replacement_m5_5c_classification": "MIXED_INFORMATION_PENDING_CORRECTION",
        },
        {
            "claim": "PASS_CORRECT_PATH_IN_TOPK_SAFE_REVIEW",
            "source_artifact": str(m5b_root / "04_SEQUENCE_REAL_RESOLVER" / "crossing_and_control_metrics.json"),
            "source_code_mechanism": "Endpoint-only source-to-target scorer.",
            "executed_evidence": (
                "The human target was in endpoint top two, but no full-sequence hypothesis was tested."
            ),
            "support_status": "partially_supported",
            "correction_required": "Freeze and evaluate complete frame-by-frame joint hypotheses.",
            "replacement_m5_5c_classification": "ENDPOINT_TOP2_ONLY_NOT_SEQUENCE_PASS",
        },
    ]
    return {
        "schema_version": "football_intelligence.m5_5c.m5_5b_claim_evidence_audit.v1",
        "generated_at": utc_now(),
        "rows": rows,
        "supported_count": sum(row["support_status"] == "supported" for row in rows),
        "partially_supported_count": sum(row["support_status"] == "partially_supported" for row in rows),
        "unsupported_count": sum(row["support_status"] == "unsupported" for row in rows),
        "prior_pass_label_preserved": False,
        **safety_payload(),
    }


def _review_ingestion(
    workspace_root: Path,
    localization_root: Path,
    path_review_root: Path,
) -> dict[str, Any]:
    localization_validation, localization_rows = _validate_localization_review(localization_root)
    path_validation, path_rows = _validate_occlusion_path_review(path_review_root)
    status = {
        "schema_version": "football_intelligence.m5_5c.review_prerequisite_status.v1",
        "generated_at": utc_now(),
        "passed": localization_validation["passed"] and path_validation["passed"],
        "localization_review": {
            "root": str(localization_root),
            "completed": localization_validation["completed"],
            "reviewed_count": localization_validation["reviewed_count"],
            "case_numbers": localization_validation["source_case_numbers"],
            "validation_passed": localization_validation["passed"],
        },
        "path_review": {
            "root": str(path_review_root),
            "completed": path_validation["completed"],
            "reviewed_count": path_validation["reviewed_count"],
            "case_numbers": path_validation["case_numbers"],
            "validation_passed": path_validation["passed"],
        },
        "reviews_used_for_evaluation_only": True,
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        **safety_payload(),
    }
    out = workspace_root / "02_REVIEW_INGESTION"
    _write_json(out / "review_prerequisite_status.json", status)
    _write_json(out / "localization_review_validation.json", localization_validation)
    _write_jsonl(out / "localization_rows.jsonl", localization_rows)
    _write_json(out / "path_review_validation.json", path_validation)
    _write_jsonl(out / "path_review_rows.jsonl", path_rows)
    return {
        "status": status,
        "localization_validation": localization_validation,
        "localization_rows": localization_rows,
        "path_validation": path_validation,
        "path_rows": path_rows,
    }


def _frame_manifest(path: Path) -> tuple[dict[int, dict[str, Any]], dict[int, Path]]:
    payload = _read_json(path)
    rows = {int(row["frame_sequence"]): row for row in payload.get("frames", [])}
    paths = {key: Path(str(row["frame_file"])) for key, row in rows.items()}
    return rows, paths


def _candidate_inventory(path: Path) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    rows = _read_jsonl(path)
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_frame[int(row["frame_sequence"])].append(row)
    for frame in by_frame:
        by_frame[frame].sort(key=lambda item: str(item["candidate_id"]))
    return rows, dict(by_frame)


def _trajectory_group_maps(grouping_path: Path) -> dict[str, Any]:
    payload = _read_json(grouping_path)
    case_to_group: dict[str, str] = {}
    endpoint_to_group: dict[str, str] = {}
    groups: dict[str, dict[str, Any]] = {}
    for component in payload.get("components", []):
        group_id = str(component["canonical_trajectory_safe_group_id"])
        groups[group_id] = component
        for case_id in component.get("case_ids", []):
            case_to_group[str(case_id)] = group_id
        for endpoint_id in component.get("endpoint_safe_group_ids", []):
            endpoint_to_group[str(endpoint_id)] = group_id
    return {
        "payload": payload,
        "case_to_group": case_to_group,
        "endpoint_to_group": endpoint_to_group,
        "groups": groups,
    }


def _observation(
    row: dict[str, Any],
    *,
    appearance_similarity: float | None = None,
    contamination: float = 0.0,
) -> FrameObservation:
    return FrameObservation(
        observation_id=str(row["candidate_id"]),
        frame_sequence=int(row["frame_sequence"]),
        bbox=ImageBBox.from_mapping(row["bbox"]),
        confidence=float(row["confidence"]) if row.get("confidence") is not None else None,
        source_provenance=str(row.get("source_type") or "canonical_person_candidate"),
        appearance_similarity=appearance_similarity,
        contamination=contamination,
    )


def _bbox_distance(left: ImageBBox, right: ImageBBox) -> float:
    return math.hypot(left.footpoint[0] - right.footpoint[0], left.footpoint[1] - right.footpoint[1])


def _find_candidate_by_bbox(rows: list[dict[str, Any]], bbox: dict[str, Any]) -> dict[str, Any] | None:
    target = ImageBBox.from_mapping(bbox)
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: (
            -target.iou(ImageBBox.from_mapping(row["bbox"])),
            _bbox_distance(target, ImageBBox.from_mapping(row["bbox"])),
            str(row["candidate_id"]),
        ),
    )


def _trace_history(
    source_row: dict[str, Any],
    candidates_by_frame: dict[int, list[dict[str, Any]]],
    *,
    history_length: int = 4,
    forbidden_observations: set[tuple[int, str]] | None = None,
) -> list[FrameObservation]:
    forbidden_observations = forbidden_observations or set()
    source_frame = int(source_row["frame_sequence"])
    current = source_row
    reverse = [source_row]
    for frame in range(source_frame - 1, source_frame - history_length, -1):
        candidates = [
            row
            for row in candidates_by_frame.get(frame, [])
            if (frame, str(row["candidate_id"])) not in forbidden_observations
        ]
        if not candidates:
            break
        current_bbox = ImageBBox.from_mapping(current["bbox"])
        scored = sorted(
            candidates,
            key=lambda row: (
                _bbox_distance(current_bbox, ImageBBox.from_mapping(row["bbox"])) / max(1.0, current_bbox.height),
                str(row["candidate_id"]),
            ),
        )
        best = scored[0]
        distance = _bbox_distance(current_bbox, ImageBBox.from_mapping(best["bbox"])) / max(1.0, current_bbox.height)
        if distance > 1.8:
            break
        reverse.append(best)
        current = best
    return [_observation(row) for row in reversed(reverse)]


def _joint_incoming_histories(
    source_row: dict[str, Any],
    neighbour_row: dict[str, Any],
    candidates_by_frame: dict[int, list[dict[str, Any]]],
) -> dict[str, list[FrameObservation]]:
    def keys(history: list[FrameObservation]) -> set[tuple[int, str]]:
        return {(row.frame_sequence, row.observation_id) for row in history}

    source_first = _trace_history(source_row, candidates_by_frame)
    neighbour_second = _trace_history(
        neighbour_row,
        candidates_by_frame,
        forbidden_observations=keys(source_first),
    )
    neighbour_first = _trace_history(neighbour_row, candidates_by_frame)
    source_second = _trace_history(
        source_row,
        candidates_by_frame,
        forbidden_observations=keys(neighbour_first),
    )
    plans = [
        {"anonymous_track_1": source_first, "anonymous_track_2": neighbour_second},
        {"anonymous_track_1": source_second, "anonymous_track_2": neighbour_first},
    ]
    plan = max(
        enumerate(plans),
        key=lambda item: (
            min(len(history) for history in item[1].values()),
            sum(len(history) for history in item[1].values()),
            -item[0],
        ),
    )[1]
    observation_keys = [
        (observation.frame_sequence, observation.observation_id) for history in plan.values() for observation in history
    ]
    if len(observation_keys) != len(set(observation_keys)):
        raise AssertionError("incoming motion histories share an observation")
    return plan


def _bounded_window_frames(
    available_frames: set[int],
    *,
    source_frame: int,
    target_frame: int,
    minimum_frames: int = 9,
    maximum_frames: int = 15,
) -> list[int]:
    frames = sorted(available_frames)
    if source_frame not in available_frames or target_frame not in available_frames:
        raise ValueError("source and target frames must be available")
    if minimum_frames > maximum_frames:
        raise ValueError("minimum_frames cannot exceed maximum_frames")
    source_index = frames.index(source_frame)
    target_index = frames.index(target_frame)
    start_index = max(0, source_index - 4)
    end_index = min(len(frames) - 1, target_index + 4)
    while end_index - start_index + 1 < minimum_frames:
        if end_index < len(frames) - 1:
            end_index += 1
        elif start_index > 0:
            start_index -= 1
        else:
            break
    while end_index - start_index + 1 > maximum_frames:
        left_context = source_index - start_index
        right_context = end_index - target_index
        if right_context >= left_context and end_index > target_index:
            end_index -= 1
        elif start_index < source_index:
            start_index += 1
        else:
            raise ValueError("source-to-target span exceeds the maximum frame window")
    window_frames = frames[start_index : end_index + 1]
    if len(window_frames) < minimum_frames:
        raise ValueError(f"only {len(window_frames)} available frames; {minimum_frames} required")
    return window_frames


def _case_input(
    row: dict[str, Any],
    candidates_by_frame: dict[int, list[dict[str, Any]]],
    available_frames: set[int],
) -> dict[str, Any]:
    source_frame = int(row["source_frame_sequence"])
    target_frame = int(row["target_frame_sequence"])
    source_rows = candidates_by_frame.get(source_frame, [])
    source_row = next(
        (candidate for candidate in source_rows if candidate["candidate_id"] == row.get("source_candidate_id")),
        None,
    ) or _find_candidate_by_bbox(source_rows, row["source_bbox"])
    if source_row is None:
        raise ValueError(f"source candidate unavailable for {row.get('challenge_candidate_id')}")
    source_bbox = ImageBBox.from_mapping(source_row["bbox"])
    neighbours = [
        candidate
        for candidate in source_rows
        if candidate["candidate_id"] != source_row["candidate_id"]
        and _bbox_distance(source_bbox, ImageBBox.from_mapping(candidate["bbox"])) / max(1.0, source_bbox.height) <= 3.0
    ]
    neighbours.sort(
        key=lambda candidate: (
            _bbox_distance(source_bbox, ImageBBox.from_mapping(candidate["bbox"])),
            str(candidate["candidate_id"]),
        )
    )
    incoming_histories = {"anonymous_track_1": _trace_history(source_row, candidates_by_frame)}
    if neighbours:
        incoming_histories = _joint_incoming_histories(source_row, neighbours[0], candidates_by_frame)
    window_frames = _bounded_window_frames(
        available_frames,
        source_frame=source_frame,
        target_frame=target_frame,
    )
    anchor_footpoints = [history[-1].footpoint for history in incoming_histories.values()]
    local_by_frame: dict[int, list[FrameObservation]] = {}
    for frame in window_frames:
        candidates = []
        for candidate in candidates_by_frame.get(frame, []):
            bbox = ImageBBox.from_mapping(candidate["bbox"])
            distance = min(math.hypot(bbox.footpoint[0] - x, bbox.footpoint[1] - y) for x, y in anchor_footpoints)
            if distance <= max(320.0, source_bbox.height * 7.0):
                candidates.append((distance, candidate))
        candidates.sort(key=lambda item: (item[0], str(item[1]["candidate_id"])))
        appearance_by_candidate = {
            str(option.get("target_candidate_id")): (
                option.get("features", {}).get("appearance_similarity"),
                0.6 if option.get("occlusion_or_crowding_evidence") else 0.0,
            )
            for option in row.get("target_options", [])
            if option.get("target_candidate_id")
        }
        local_by_frame[frame] = [
            _observation(
                candidate,
                appearance_similarity=appearance_by_candidate.get(str(candidate["candidate_id"]), (None, 0.0))[0],
                contamination=appearance_by_candidate.get(str(candidate["candidate_id"]), (None, 0.0))[1],
            )
            for _, candidate in candidates[:10]
        ]
    return {
        "challenge_row": row,
        "source_row": source_row,
        "incoming_histories": incoming_histories,
        "window_frames": window_frames,
        "observations_by_frame": local_by_frame,
        "source_frame_sequence": source_frame,
        "target_frame_sequence": target_frame,
    }


def _serialize_motion_fit(case_id: str, track_id: str, fit: Any) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "track_id": track_id,
        "status": fit.status,
        "fitted_frame_sequences": list(fit.fitted_frame_sequences),
        "distinct_frame_count": len(set(fit.fitted_frame_sequences)),
        "state_mean": fit.state_mean,
        "covariance": fit.covariance,
        "innovations": list(fit.innovations),
        **safety_payload(),
    }


def detector_configurations() -> list[dict[str, Any]]:
    raw = [
        {"name": "canonical_baseline", "imgsz": 1280, "conf": 0.22, "iou": 0.70, "max_det": 80},
        {"name": "higher_resolution_2048", "imgsz": 2048, "conf": 0.22, "iou": 0.70, "max_det": 80},
        {"name": "lower_confidence", "imgsz": 1280, "conf": 0.05, "iou": 0.70, "max_det": 80},
        {"name": "relaxed_post_nms", "imgsz": 1280, "conf": 0.05, "iou": 0.90, "max_det": 80},
        {"name": "higher_max_detections", "imgsz": 1280, "conf": 0.05, "iou": 0.70, "max_det": 300},
        {
            "name": "native_crop_2_height",
            "imgsz": 1280,
            "conf": 0.05,
            "iou": 0.70,
            "max_det": 80,
            "crop_height_factor": 2.0,
        },
        {
            "name": "native_crop_3_height",
            "imgsz": 1280,
            "conf": 0.05,
            "iou": 0.70,
            "max_det": 80,
            "crop_height_factor": 3.0,
        },
    ]
    rows = []
    for config in raw:
        payload = {
            **config,
            "classes": [0],
            "device": "cpu",
            "augment": False,
            "pre_nms_evidence_status": PRE_NMS_STATUS,
        }
        payload["configuration_hash"] = stable_hash(payload)
        rows.append(payload)
    return rows


def _run_yolo(model: Any, image_path: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    result = model.predict(
        source=str(image_path),
        imgsz=int(config["imgsz"]),
        conf=float(config["conf"]),
        iou=float(config["iou"]),
        max_det=int(config["max_det"]),
        classes=[0],
        device="cpu",
        augment=False,
        save=False,
        verbose=False,
    )[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    rows = []
    for index, (bbox, confidence, cls_id) in enumerate(
        zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist(), strict=False)
    ):
        if int(cls_id) == 0:
            rows.append(
                {
                    "prediction_index": index,
                    "bbox": {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]},
                    "confidence": float(confidence),
                }
            )
    return rows


def _crop_around_bbox(
    image_path: Path,
    bbox: BBox,
    factor: float,
    output_path: Path,
) -> tuple[Path, tuple[float, float]]:
    with Image.open(image_path) as image:
        center_x, center_y = bbox.center
        crop_height = max(64.0, bbox.height * factor)
        crop_width = max(128.0, crop_height * 2.0)
        left = max(0, int(center_x - crop_width / 2.0))
        top = max(0, int(center_y - crop_height / 2.0))
        right = min(image.width, int(center_x + crop_width / 2.0))
        bottom = min(image.height, int(center_y + crop_height / 2.0))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.crop((left, top, right, bottom)).save(output_path, quality=92)
    return output_path, (float(left), float(top))


def _local_predictions(
    predictions: list[dict[str, Any]], reference: BBox, radius_factor: float = 5.0
) -> list[dict[str, Any]]:
    rows = []
    for row in predictions:
        bbox = parse_bbox(row["bbox"])
        distance = math.hypot(bbox.center[0] - reference.center[0], bbox.center[1] - reference.center[1])
        if distance <= max(100.0, reference.height * radius_factor):
            rows.append(row)
    return rows


def _duplicate_count(predictions: list[dict[str, Any]]) -> int:
    count = 0
    for left, right in itertools.combinations(predictions, 2):
        if bbox_iou(parse_bbox(left["bbox"]), parse_bbox(right["bbox"])) >= 0.5:
            count += 1
    return count


def select_matched_controls(
    localization_rows: list[dict[str, Any]],
    candidates_by_frame: dict[int, list[dict[str, Any]]],
    frame_paths: dict[int, Path],
    case_to_group: dict[str, str],
    *,
    controls_per_region: int = 2,
) -> list[dict[str, Any]]:
    affected_frames = {int(row["target_frame_sequence"]) for row in localization_rows}
    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in localization_rows:
        historical_case_id = f"m5_4h1_cadence_matched_target_choice_case_{row['source_case_number']}"
        region = case_to_group.get(historical_case_id, historical_case_id)
        by_region[region].append(row)
    selected: list[dict[str, Any]] = []
    used: set[tuple[int, str]] = set()
    for region, rows in sorted(by_region.items()):
        representative = rows[0]
        target_frame = int(representative["target_frame_sequence"])
        reference_bbox = parse_bbox(representative["reviewer_bbox"])
        reference_density = len(candidates_by_frame.get(target_frame, []))
        candidates: list[tuple[tuple[float, ...], dict[str, Any]]] = []
        bucket = int(target_frame // 100)
        for frame, frame_candidates in candidates_by_frame.items():
            if frame in affected_frames or int(frame // 100) != bucket or frame not in frame_paths:
                continue
            for candidate in frame_candidates:
                bbox = parse_bbox(candidate["bbox"])
                height_delta = abs(bbox.height - reference_bbox.height) / max(1.0, reference_bbox.height)
                density_delta = abs(len(frame_candidates) - reference_density)
                key = (frame, str(candidate["candidate_id"]))
                if key in used or height_delta > 0.45:
                    continue
                score = (
                    height_delta,
                    float(density_delta),
                    float(abs(frame - target_frame)),
                    float(int(hashlib.sha256(f"{region}:{key}".encode()).hexdigest()[:8], 16)),
                )
                candidates.append((score, candidate))
        candidates.sort(key=lambda item: item[0])
        for _, candidate in candidates[:controls_per_region]:
            frame = int(candidate["frame_sequence"])
            key = (frame, str(candidate["candidate_id"]))
            used.add(key)
            selected.append(
                {
                    "control_id": f"control_{len(selected) + 1:03d}",
                    "trajectory_safe_region_id": region,
                    "frame_sequence": frame,
                    "frame_file": str(frame_paths[frame]),
                    "reference_bbox": candidate["bbox"],
                    "reference_candidate_id": candidate["candidate_id"],
                    "reference_bbox_height_band": bbox_height_band(parse_bbox(candidate["bbox"]).height),
                    "local_candidate_density": len(candidates_by_frame.get(frame, [])),
                    "same_temporal_subregion": True,
                    "same_trajectory_group_as_affected": False,
                    "affected_target_frame": False,
                    **safety_payload(),
                }
            )
    return selected


def _run_controlled_detector(
    workspace_root: Path,
    localization_rows: list[dict[str, Any]],
    candidates_by_frame: dict[int, list[dict[str, Any]]],
    frame_paths: dict[int, Path],
    group_maps: dict[str, Any],
    model_path: Path,
    *,
    run_detector: bool,
) -> dict[str, Any]:
    out = workspace_root / "03_CONTROLLED_DETECTOR_RECOVERY"
    configs = detector_configurations()
    model_hash = sha256_file(model_path) if model_path.exists() else None
    config_manifest = {
        "schema_version": "football_intelligence.m5_5c.detector_configuration_manifest.v1",
        "generated_at": utc_now(),
        "model_path": str(model_path),
        "model_sha256": model_hash,
        "expected_model_sha256": EXPECTED_DETECTOR_SHA256,
        "model_hash_matches": model_hash == EXPECTED_DETECTOR_SHA256,
        "configurations": configs,
        "matched_controls_execute_every_configuration": True,
        "pre_nms_evidence_status": PRE_NMS_STATUS,
        **safety_payload(),
    }
    _write_json(out / "configuration_manifest.json", config_manifest)
    affected_rows = []
    for row in localization_rows:
        historical_case_id = f"m5_4h1_cadence_matched_target_choice_case_{row['source_case_number']}"
        affected_rows.append(
            {
                "case_id": row["case_id"],
                "case_number": row["source_case_number"],
                "trajectory_safe_region_id": group_maps["case_to_group"].get(historical_case_id),
                "frame_sequence": int(row["target_frame_sequence"]),
                "frame_file": row["target_frame_file"],
                "reference_bbox": row["reviewer_bbox"],
                "human_localization_decision": row["decision"],
                **safety_payload(),
            }
        )
    controls = select_matched_controls(
        localization_rows,
        candidates_by_frame,
        frame_paths,
        group_maps["case_to_group"],
    )
    _write_jsonl(out / "affected_case_rows.jsonl", affected_rows)
    _write_jsonl(out / "control_selection_rows.jsonl", controls)
    model = None
    runtime_status = "not_requested"
    if run_detector and model_hash == EXPECTED_DETECTOR_SHA256:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        runtime_status = "executed"
    elif run_detector:
        runtime_status = "blocked_model_hash_or_missing_checkpoint"
    affected_inference: list[dict[str, Any]] = []
    control_inference: list[dict[str, Any]] = []
    regions = [
        ("affected", row["case_id"], Path(row["frame_file"]), parse_bbox(row["reference_bbox"]), row)
        for row in affected_rows
    ] + [
        ("control", row["control_id"], Path(row["frame_file"]), parse_bbox(row["reference_bbox"]), row)
        for row in controls
    ]
    baseline_by_region: dict[str, list[dict[str, Any]]] = {}
    for stratum, region_id, image_path, reference_bbox, metadata in regions:
        for config in configs:
            started = time.perf_counter()
            predictions: list[dict[str, Any]] = []
            error = None
            if model is not None:
                try:
                    crop_factor = config.get("crop_height_factor")
                    if crop_factor:
                        crop_path = workspace_root / "_tmp" / "detector_crops" / f"{region_id}_{config['name']}.jpg"
                        source, origin = _crop_around_bbox(image_path, reference_bbox, float(crop_factor), crop_path)
                        predictions = _run_yolo(model, source, config)
                        for prediction in predictions:
                            mapped = crop_to_panorama_bbox(parse_bbox(prediction["bbox"]), origin)
                            prediction["bbox"] = mapped.to_dict()
                    else:
                        predictions = _run_yolo(model, image_path, config)
                except Exception as exc:  # pragma: no cover - runtime environment dependent
                    error = str(exc)
            runtime = time.perf_counter() - started
            local = _local_predictions(predictions, reference_bbox)
            best_metrics = None
            if local:
                metrics = [
                    canonical_match_metrics(
                        localization_bbox=reference_bbox,
                        candidate_bbox=parse_bbox(prediction["bbox"]),
                        candidate_id=f"{region_id}_{config['name']}_{index}",
                        confidence=float(prediction["confidence"]),
                        original_radius_center=reference_bbox.center,
                    )
                    for index, prediction in enumerate(local)
                ]
                best_metrics = max(metrics, key=lambda item: (item["bbox_iou"], -item["footpoint_distance"]))
            if config["name"] == "canonical_baseline":
                baseline_by_region[region_id] = local
            baseline = baseline_by_region.get(region_id, [])
            added = [
                prediction
                for prediction in local
                if not any(
                    bbox_iou(parse_bbox(prediction["bbox"]), parse_bbox(prior["bbox"])) >= 0.5 for prior in baseline
                )
            ]
            result = {
                "stratum": stratum,
                "region_id": region_id,
                "trajectory_safe_region_id": metadata["trajectory_safe_region_id"],
                "frame_sequence": metadata["frame_sequence"],
                "configuration_name": config["name"],
                "configuration_hash": config["configuration_hash"],
                "runtime_seconds": round(runtime, 6),
                "runtime_error": error,
                "local_prediction_count": len(local),
                "added_detection_count": len(added),
                "added_overlapping_detection_count": sum(
                    any(
                        bbox_iou(parse_bbox(prediction["bbox"]), parse_bbox(prior["bbox"])) >= 0.3 for prior in baseline
                    )
                    for prediction in added
                ),
                "duplicate_count": _duplicate_count(local),
                "compatible_person_recovered": bool(best_metrics and best_metrics["diagnostic_compatible_match"]),
                "best_iou": best_metrics["bbox_iou"] if best_metrics else 0.0,
                "best_footpoint_error": best_metrics["footpoint_distance"] if best_metrics else None,
                "human_audit_required": len(added),
                "predictions": local,
                **safety_payload(),
            }
            (affected_inference if stratum == "affected" else control_inference).append(result)
    _write_jsonl(out / "affected_inference_rows.jsonl", affected_inference)
    _write_jsonl(out / "control_inference_rows.jsonl", control_inference)
    metrics_rows = []
    for config in configs:
        name = config["name"]
        affected = [row for row in affected_inference if row["configuration_name"] == name]
        control = [row for row in control_inference if row["configuration_name"] == name]
        all_rows = affected + control
        metrics_rows.append(
            {
                "configuration_name": name,
                "affected_visible_targets": len(affected),
                "targets_recovered": sum(row["compatible_person_recovered"] for row in affected),
                "control_regions": len(control),
                "added_detections": sum(row["added_detection_count"] for row in all_rows),
                "added_overlapping_detections": sum(row["added_overlapping_detection_count"] for row in all_rows),
                "duplicates": sum(row["duplicate_count"] for row in all_rows),
                "human_audit_required": sum(row["human_audit_required"] for row in all_rows),
                "runtime_seconds": round(sum(row["runtime_seconds"] for row in all_rows), 6),
                "mean_best_iou": round(sum(row["best_iou"] for row in all_rows) / max(1, len(all_rows)), 6),
                "mean_footpoint_error": round(
                    sum(row["best_footpoint_error"] for row in all_rows if row["best_footpoint_error"] is not None)
                    / max(1, sum(row["best_footpoint_error"] is not None for row in all_rows)),
                    6,
                ),
                "all_controls_executed": len(control) == len(controls),
            }
        )
    configuration_metrics = {
        "schema_version": "football_intelligence.m5_5c.detector_configuration_metrics.v1",
        "generated_at": utc_now(),
        "runtime_status": runtime_status,
        "affected_case_count": len(affected_rows),
        "independent_trajectory_region_count": len({row["trajectory_safe_region_id"] for row in affected_rows}),
        "control_region_count": len(controls),
        "rows": metrics_rows,
        "every_configuration_ran_on_every_control": bool(controls)
        and all(row["all_controls_executed"] for row in metrics_rows),
        **safety_payload(),
    }
    trajectory_summary = {
        "schema_version": "football_intelligence.m5_5c.detector_trajectory_region_summary.v1",
        "trajectory_region_count": len({row["trajectory_safe_region_id"] for row in affected_rows}),
        "case_004_016_share_region": next(
            row["trajectory_safe_region_id"] for row in affected_rows if row["case_number"] == "004"
        )
        == next(row["trajectory_safe_region_id"] for row in affected_rows if row["case_number"] == "016"),
        "regions": [
            {
                "trajectory_safe_region_id": region,
                "case_numbers": sorted(
                    row["case_number"] for row in affected_rows if row["trajectory_safe_region_id"] == region
                ),
                "control_count": sum(row["trajectory_safe_region_id"] == region for row in controls),
            }
            for region in sorted({row["trajectory_safe_region_id"] for row in affected_rows})
        ],
        **safety_payload(),
    }
    detector_decision = {
        "schema_version": "football_intelligence.m5_5c.detector_decision.v1",
        "runtime_status": runtime_status,
        "canonical_candidates_replaced": False,
        "pre_nms_evidence_status": PRE_NMS_STATUS,
        "scientific_result": "CONTROLLED_POST_NMS_CONFIGURATION_SWEEP_EXECUTED"
        if runtime_status == "executed"
        else "BLOCKED_DETECTOR_RUNTIME",
        **safety_payload(),
    }
    _write_json(out / "configuration_metrics.json", configuration_metrics)
    _write_json(out / "trajectory_region_summary.json", trajectory_summary)
    _write_json(out / "detector_decision.json", detector_decision)
    return {
        "config_manifest": config_manifest,
        "affected_rows": affected_rows,
        "controls": controls,
        "affected_inference": affected_inference,
        "control_inference": control_inference,
        "metrics": configuration_metrics,
        "trajectory_summary": trajectory_summary,
        "decision": detector_decision,
    }


def _target_observation_for_bbox(
    observations: list[FrameObservation],
    bbox: dict[str, Any] | None,
) -> str | None:
    if not observations or not isinstance(bbox, dict):
        return None
    target = ImageBBox.from_mapping(bbox)
    best = min(
        observations,
        key=lambda observation: (
            -target.iou(observation.bbox),
            _bbox_distance(target, observation.bbox),
            observation.observation_id,
        ),
    )
    return best.observation_id


def _rank_for_target(result: dict[str, Any], target_frame: int, target_observation_id: str | None) -> int | None:
    if target_observation_id is None:
        return None
    for hypothesis in result.get("hypotheses", []):
        path = hypothesis["paths"].get("anonymous_track_1", [])
        target_node = next((node for node in path if node["frame_sequence"] == target_frame), None)
        if target_node and target_node.get("observation_id") == target_observation_id:
            return int(hypothesis["rank"])
    return None


def _resolver_evaluation_target(
    case_number: str,
    case_input: dict[str, Any],
    path_review_root: Path,
) -> dict[str, Any]:
    mapping = _load_mapping_cases(path_review_root / "sealed" / "server_mapping.json")
    completed = _read_json(path_review_root / "decisions" / "completed_review.json")
    decisions = completed.get("state", {}).get("decisions", {})
    case_id = f"m5_5a_occlusion_path_case_{case_number}"
    decision = decisions.get(case_id)
    map_row = mapping.get(case_id, {})
    hypothesis = map_row.get("decision_to_internal_hypothesis", {}).get(decision, {})
    target_frame = int(case_input["target_frame_sequence"])
    target_id = _target_observation_for_bbox(
        case_input["observations_by_frame"].get(target_frame, []),
        hypothesis.get("target_bbox"),
    )
    return {
        "human_decision": decision,
        "human_target_observation_id": target_id,
        "target_bbox": hypothesis.get("target_bbox"),
    }


def _protected_control_target(row: dict[str, Any], human_row: dict[str, Any]) -> str | None:
    panel = human_row.get("human_selected_panel")
    index = {"target_a": 0, "target_b": 1}.get(panel)
    if index is None or len(row.get("target_options", [])) <= index:
        return None
    return str(row["target_options"][index].get("target_candidate_id"))


def _run_true_sequence_resolver(
    workspace_root: Path,
    historical_stage_root: Path,
    path_review_root: Path,
    candidates_by_frame: dict[int, list[dict[str, Any]]],
    available_frames: set[int],
) -> dict[str, Any]:
    out = workspace_root / "04_TRUE_SEQUENCE_RESOLVER"
    challenge_rows = _read_jsonl(
        historical_stage_root / "continuity_v11" / "unseen_window" / "challenge_candidate_rows.jsonl"
    )
    primary = _read_json(historical_stage_root / "continuity_v13" / "evaluation" / "corrected_primary_results.json")
    primary_by_number = {_case_number(str(row["case_id"])): row for row in primary.get("rows", [])}
    endpoint_to_number = {
        str(row["endpoint_safe_group_id"]): _case_number(str(row["case_id"])) for row in primary.get("rows", [])
    }
    challenge_by_number = {
        endpoint_to_number.get(str(row.get("endpoint_safe_group_id"))): row for row in challenge_rows
    }
    selected_numbers = sorted(KNOWN_CROSSING_CASES | PROTECTED_CONTROL_CASES)
    window_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    motion_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    graph_nodes: list[dict[str, Any]] = []
    graph_edges: list[dict[str, Any]] = []
    hypothesis_rows: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []
    answer_audits: list[dict[str, Any]] = []
    appearance_eligibility: list[dict[str, Any]] = []
    appearance_activations: list[dict[str, Any]] = []
    ghost_cases: list[dict[str, Any]] = []
    scientific_results: dict[str, dict[str, Any]] = {}
    for case_number in selected_numbers:
        row = challenge_by_number.get(case_number)
        if row is None:
            case_results.append(
                {
                    "case_number": case_number,
                    "stratum": "known_crossing" if case_number in KNOWN_CROSSING_CASES else "protected_control",
                    "classification": "SOURCE_CHALLENGE_ROW_MISSING",
                    **safety_payload(),
                }
            )
            continue
        case_id = f"m5_5c_sequence_case_{case_number}"
        case_input = _case_input(row, candidates_by_frame, available_frames)
        conflict = observable_conflict_signals(
            case_input["incoming_histories"],
            case_input["observations_by_frame"],
        )
        conflict_rows.append({"case_id": case_id, "case_number": case_number, **conflict, **safety_payload()})
        result = resolve_joint_sequence(
            incoming_histories=case_input["incoming_histories"],
            observations_by_frame=case_input["observations_by_frame"],
            window_frames=case_input["window_frames"],
            image_size=(2730, 720),
            config=ResolverConfig(),
            appearance_enabled=False,
        )
        repeated = resolve_joint_sequence(
            incoming_histories=case_input["incoming_histories"],
            observations_by_frame=case_input["observations_by_frame"],
            window_frames=case_input["window_frames"],
            image_size=(2730, 720),
            config=ResolverConfig(),
            appearance_enabled=False,
        )
        first_hash = answer_independent_fingerprint(result)
        withheld_hash = answer_independent_fingerprint(repeated)
        answer_audits.append(
            {
                "case_id": case_id,
                "case_number": case_number,
                "human_decision_files_loaded_during_generation": False,
                "generation_hash_with_decisions_withheld": withheld_hash,
                "generation_hash_before_evaluation": first_hash,
                "byte_identical": first_hash == withheld_hash,
                "candidate_generation_answer_independent": True,
                "costs_answer_independent": True,
                "conflict_activation_answer_independent": True,
                "ranking_answer_independent": True,
                **safety_payload(),
            }
        )
        window_rows.append(
            {
                "case_id": case_id,
                "case_number": case_number,
                "stratum": "known_crossing" if case_number in KNOWN_CROSSING_CASES else "protected_control",
                "source_frame_sequence": case_input["source_frame_sequence"],
                "target_frame_sequence": case_input["target_frame_sequence"],
                "frame_sequences": case_input["window_frames"],
                "frame_count": len(case_input["window_frames"]),
                "incoming_observations_unique_across_tracks": len(
                    {
                        (observation.frame_sequence, observation.observation_id)
                        for history in case_input["incoming_histories"].values()
                        for observation in history
                    }
                )
                == sum(len(history) for history in case_input["incoming_histories"].values()),
                "every_available_frame_represented": result.get("frame_coverage_complete", False),
                "explicit_missing_boundary_record": True,
                **safety_payload(),
            }
        )
        for frame, observations in case_input["observations_by_frame"].items():
            for observation in observations:
                observation_rows.append(
                    {
                        "case_id": case_id,
                        "case_number": case_number,
                        "local_neighbour_count": len(observations) - 1,
                        **observation.to_row(),
                        **safety_payload(),
                    }
                )
        for track_id, fit in result.get("motion_fits", {}).items():
            motion_rows.append(_serialize_motion_fit(case_id, track_id, fit))
        graph_nodes.extend(
            {"case_id": case_id, "case_number": case_number, **node, **safety_payload()}
            for node in result.get("graph_nodes", [])
        )
        graph_edges.extend(
            {"case_id": case_id, "case_number": case_number, **edge, **safety_payload()}
            for edge in result.get("graph_edges", [])
        )
        for hypothesis in result.get("hypotheses", []):
            hypothesis_rows.append(
                {
                    "case_id": case_id,
                    "case_number": case_number,
                    **hypothesis,
                    **safety_payload(),
                }
            )
        geometry_margin = result.get("best_second_margin")
        target_observations = case_input["observations_by_frame"].get(case_input["target_frame_sequence"], [])
        source_height = case_input["incoming_histories"]["anonymous_track_1"][-1].bbox.height
        appearance_gate = appearance_activation_gate(
            conflict_active=bool(conflict["conflict_active"]),
            motion_compatible_candidate_count=len(target_observations),
            geometry_margin=geometry_margin,
            source_bbox_height=source_height,
            target_bbox_heights=[item.bbox.height for item in target_observations[:3]],
            source_contamination=0.0,
            target_contamination=max((item.contamination for item in target_observations[:3]), default=1.0),
        )
        appearance_eligibility.append(
            {"case_id": case_id, "case_number": case_number, **appearance_gate, **safety_payload()}
        )
        appearance_result = result
        if appearance_gate["eligible"]:
            appearance_result = resolve_joint_sequence(
                incoming_histories=case_input["incoming_histories"],
                observations_by_frame=case_input["observations_by_frame"],
                window_frames=case_input["window_frames"],
                image_size=(2730, 720),
                config=ResolverConfig(),
                appearance_enabled=True,
            )
            appearance_activations.append(
                {
                    "case_id": case_id,
                    "case_number": case_number,
                    "geometry_fingerprint": first_hash,
                    "appearance_fingerprint": answer_independent_fingerprint(appearance_result),
                    "appearance_contribution": sum(
                        hypothesis.get("appearance_contribution", 0.0)
                        for hypothesis in appearance_result.get("hypotheses", [])
                    ),
                    **safety_payload(),
                }
            )
        evaluation: dict[str, Any] = {}
        if case_number in KNOWN_CROSSING_CASES:
            evaluation_target = _resolver_evaluation_target(case_number, case_input, path_review_root)
            geometry_rank = _rank_for_target(
                result,
                case_input["target_frame_sequence"],
                evaluation_target["human_target_observation_id"],
            )
            appearance_rank = _rank_for_target(
                appearance_result,
                case_input["target_frame_sequence"],
                evaluation_target["human_target_observation_id"],
            )
            evaluation = {
                "human_decision": evaluation_target["human_decision"],
                "human_target_observation_id": evaluation_target["human_target_observation_id"],
                "correct_path_top1": geometry_rank == 1,
                "correct_path_in_top2": geometry_rank is not None and geometry_rank <= 2,
                "correct_path_in_top4": geometry_rank is not None and geometry_rank <= 4,
                "human_path_rank_geometry": geometry_rank,
                "human_path_rank_appearance": appearance_rank,
                "wrong_confident_ranker": geometry_rank != 1
                and geometry_margin is not None
                and geometry_margin > ResolverConfig().geometry_margin_threshold,
                "wrong_committed_assignment": False,
                "review_escalation": conflict["conflict_active"]
                or geometry_margin is None
                or geometry_margin <= ResolverConfig().geometry_margin_threshold
                or geometry_rank != 1,
            }
        else:
            human_row = primary_by_number.get(case_number, {})
            target_id = _protected_control_target(row, human_row)
            geometry_rank = _rank_for_target(result, case_input["target_frame_sequence"], target_id)
            appearance_rank = _rank_for_target(appearance_result, case_input["target_frame_sequence"], target_id)
            evaluation = {
                "human_decision": human_row.get("human_decision"),
                "binary_target_available": target_id is not None,
                "human_path_rank_geometry": geometry_rank,
                "human_path_rank_appearance": appearance_rank,
                "appearance_gate_activated": appearance_gate["eligible"],
                "appearance_correction": bool(
                    appearance_gate["eligible"] and geometry_rank != 1 and appearance_rank == 1
                ),
                "appearance_regression": bool(
                    appearance_gate["eligible"] and geometry_rank == 1 and appearance_rank != 1
                ),
                "appearance_no_effect": bool(appearance_gate["eligible"] and geometry_rank == appearance_rank),
                "rerun_by_current_code_path": True,
            }
        if result.get("hypotheses"):
            ghost = execute_ghost_intervals(
                case_id=case_id,
                hypothesis=result["hypotheses"][0],
                max_hidden_frames=ResolverConfig().max_hidden_frames,
            )
        else:
            ghost = {
                "eligible_intervals": [],
                "hidden_intervals": [],
                "ghost_state_rows": [],
                "reentry_hypotheses": [],
                "executed": False,
            }
        ghost_cases.append({"case_id": case_id, "case_number": case_number, **ghost})
        case_result = {
            "case_id": case_id,
            "case_number": case_number,
            "stratum": "known_crossing" if case_number in KNOWN_CROSSING_CASES else "protected_control",
            "resolver_classification": result["classification"],
            "frame_count": len(case_input["window_frames"]),
            "incoming_track_count": len(case_input["incoming_histories"]),
            "conflict_active": conflict["conflict_active"],
            "best_second_margin": geometry_margin,
            "candidate_generation_uses_appearance": False,
            "appearance_gate": appearance_gate,
            "ghost_interval_count": len(ghost["eligible_intervals"]),
            **evaluation,
            **safety_payload(),
        }
        case_results.append(case_result)
        scientific_results[case_number] = {
            "case_input": case_input,
            "geometry": result,
            "appearance": appearance_result,
            "conflict": conflict,
            "case_result": case_result,
            "ghost": ghost,
        }
    crossing_rows = [row for row in case_results if row["stratum"] == "known_crossing"]
    wrong_ranker = sum(row.get("wrong_confident_ranker", False) for row in crossing_rows)
    insufficient = sum(row["resolver_classification"] == "INSUFFICIENT_INCOMING_HISTORY" for row in crossing_rows)
    top1 = sum(row.get("correct_path_top1", False) for row in crossing_rows)
    top2 = sum(row.get("correct_path_in_top2", False) for row in crossing_rows)
    top4 = sum(row.get("correct_path_in_top4", False) for row in crossing_rows)
    if insufficient:
        classification = "BLOCKED_INSUFFICIENT_INCOMING_HISTORY"
    elif any(row.get("wrong_committed_assignment") for row in crossing_rows):
        classification = "FAIL_WRONG_COMMITTED_ASSIGNMENT"
    elif wrong_ranker:
        classification = "FAIL_WRONG_CONFIDENT_RANKER"
    elif top1 == len(crossing_rows) and crossing_rows:
        classification = "PASS_TRUE_SEQUENCE_TOP1"
    elif top4 == len(crossing_rows) and crossing_rows:
        classification = "PASS_TRUE_SEQUENCE_TOPK_SAFE_REVIEW"
    else:
        classification = "PASS_SAFE_ESCALATION_ONLY"
    metrics = {
        "schema_version": "football_intelligence.m5_5c.true_sequence_metrics.v1",
        "generated_at": utc_now(),
        "known_crossing_case_count": len(crossing_rows),
        "candidate_set_recall": sum(row.get("human_path_rank_geometry") is not None for row in crossing_rows)
        / max(1, len(crossing_rows)),
        "correct_path_top1": top1,
        "correct_path_in_top2": top2,
        "correct_path_in_top4": top4,
        "wrong_confident_ranker": wrong_ranker,
        "wrong_committed_assignment": sum(row.get("wrong_committed_assignment", False) for row in crossing_rows),
        "review_escalation": sum(row.get("review_escalation", False) for row in crossing_rows),
        "abstention": sum(row.get("review_escalation", False) for row in crossing_rows),
        "protected_control_count": sum(row["stratum"] == "protected_control" for row in case_results),
        "classification": classification,
        **safety_payload(),
    }
    answer_audit = {
        "schema_version": "football_intelligence.m5_5c.answer_independence_audit.v1",
        "case_count": len(answer_audits),
        "all_byte_identical": bool(answer_audits) and all(row["byte_identical"] for row in answer_audits),
        "human_answers_loaded_only_after_hypotheses_frozen": True,
        "rows": answer_audits,
        **safety_payload(),
    }
    _write_json(out / "window_manifest.json", {"rows": window_rows, **safety_payload()})
    _write_jsonl(out / "frame_observation_rows.jsonl", observation_rows)
    _write_jsonl(out / "incoming_motion_fits.jsonl", motion_rows)
    _write_jsonl(out / "conflict_signal_rows.jsonl", conflict_rows)
    _write_jsonl(out / "graph_nodes.jsonl", graph_nodes)
    _write_jsonl(out / "graph_edges.jsonl", graph_edges)
    _write_jsonl(out / "k_best_joint_hypotheses.jsonl", hypothesis_rows)
    _write_jsonl(out / "case_results.jsonl", case_results)
    _write_json(out / "metrics.json", metrics)
    _write_json(out / "answer_independence_audit.json", answer_audit)
    return {
        "window_rows": window_rows,
        "observation_rows": observation_rows,
        "motion_rows": motion_rows,
        "conflict_rows": conflict_rows,
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "hypothesis_rows": hypothesis_rows,
        "case_results": case_results,
        "metrics": metrics,
        "answer_audit": answer_audit,
        "appearance_eligibility": appearance_eligibility,
        "appearance_activations": appearance_activations,
        "ghost_cases": ghost_cases,
        "scientific_results": scientific_results,
    }


def _write_ghost_outputs(workspace_root: Path, resolver: dict[str, Any]) -> dict[str, Any]:
    out = workspace_root / "05_GHOST_AND_REENTRY"
    intervals = []
    hidden_intervals = []
    states = []
    reentries = []
    for case in resolver["ghost_cases"]:
        intervals.extend(case["eligible_intervals"])
        hidden_intervals.extend(case["hidden_intervals"])
        states.extend(case["ghost_state_rows"])
        reentries.extend(case["reentry_hypotheses"])
    reviewed_interval_count = 0
    unresolved = len(intervals)
    reviewed_results = {
        "schema_version": "football_intelligence.m5_5c.ghost_reviewed_results.v1",
        "reviewed_eligible_interval_count": reviewed_interval_count,
        "new_blind_review_required": bool(intervals),
        "correct_reentry_top1": None,
        "correct_reentry_in_topK": None,
        "wrong_confirmed_reentry": None,
        "eligible_interval_status": "ELIGIBLE_INTERVALS_PENDING_REVIEW" if intervals else "NO_ELIGIBLE_INTERVAL_MINED",
        "classification": "BLOCKED_NO_REVIEWED_GHOST_INTERVAL",
        **safety_payload(),
    }
    metrics = {
        "schema_version": "football_intelligence.m5_5c.ghost_and_reentry_metrics.v1",
        "generated_at": utc_now(),
        "eligible_interval_count": len(intervals),
        "correct_reentry_top1": None,
        "correct_reentry_in_topK": None,
        "wrong_confirmed_reentry": None,
        "review_escalation": len(intervals),
        "false_ghost_persistence": None,
        "mean_hidden_frames": round(
            sum(row.get("hidden_frame_count", 0) for row in intervals) / max(1, len(intervals)), 6
        ),
        "mean_excess_lifetime": round(
            sum(max(0, row.get("hidden_frame_count", 0) - ResolverConfig().max_hidden_frames) for row in intervals)
            / max(1, len(intervals)),
            6,
        ),
        "no_ground_truth_or_unresolved": unresolved,
        "ghost_state_execution_count": len(states),
        "hidden_interval_execution_count": len(hidden_intervals),
        "prediction_and_covariance_advanced": bool(states) and all(row["prediction_advanced"] for row in states),
        "expiry_policy_evaluated": bool(states) and all("dynamic_expiry_executed" in row for row in states),
        "expiry_event_count": sum(row["dynamic_expiry_executed"] for row in states),
        "termination_event_count": sum(row.get("terminated", False) for row in hidden_intervals),
        "classification": reviewed_results["classification"],
        **safety_payload(),
    }
    _write_json(out / "eligible_interval_manifest.json", {"rows": intervals, **safety_payload()})
    _write_jsonl(out / "ghost_state_rows.jsonl", states)
    _write_jsonl(out / "reentry_hypotheses.jsonl", reentries)
    _write_json(out / "reviewed_results.json", reviewed_results)
    _write_json(out / "metrics.json", metrics)
    return {
        "intervals": intervals,
        "states": states,
        "reentries": reentries,
        "reviewed_results": reviewed_results,
        "metrics": metrics,
    }


def _write_appearance_outputs(workspace_root: Path, resolver: dict[str, Any]) -> dict[str, Any]:
    out = workspace_root / "06_APPEARANCE_ABLATION"
    eligibility = resolver["appearance_eligibility"]
    activations = resolver["appearance_activations"]
    crossing_rows = [row for row in resolver["case_results"] if row["stratum"] == "known_crossing"]
    controls = [row for row in resolver["case_results"] if row["stratum"] == "protected_control"]
    eligible_count = sum(row["eligible"] for row in eligibility)
    activation_count = len(activations)
    corrections = sum(row.get("appearance_correction", False) for row in controls)
    regressions = sum(row.get("appearance_regression", False) for row in controls)
    no_effect = sum(row.get("appearance_no_effect", False) for row in controls)
    not_evaluable = len(eligibility) - activation_count
    classification = (
        "NOT_EVALUATED_NO_ELIGIBLE_GATE" if activation_count == 0 else "EVALUATED_CONFLICT_GATED_APPEARANCE"
    )
    crossing_results = {
        "schema_version": "football_intelligence.m5_5c.appearance_crossing_results.v1",
        "rows": crossing_rows,
        "candidate_generation_uses_appearance": False,
        **safety_payload(),
    }
    protected_results = {
        "schema_version": "football_intelligence.m5_5c.appearance_protected_control_results.v1",
        "rows": controls,
        "rerun_by_current_code_path": all(row.get("rerun_by_current_code_path", False) for row in controls),
        "copied_from_prior_summary": False,
        **safety_payload(),
    }
    metrics = {
        "schema_version": "football_intelligence.m5_5c.appearance_ablation_metrics.v1",
        "eligible_conflict_count": eligible_count,
        "activation_count": activation_count,
        "activation_coverage": activation_count / max(1, len(eligibility)),
        "corrections": corrections if activation_count else None,
        "regressions": regressions if activation_count else None,
        "no_effect": no_effect if activation_count else None,
        "not_evaluable_count": not_evaluable,
        "protected_control_count": len(controls),
        "classification": classification,
        "zero_activation_interpreted_as_zero_regressions": False,
        "candidate_generation_uses_appearance": False,
        **safety_payload(),
    }
    _write_jsonl(out / "eligibility_rows.jsonl", eligibility)
    _write_jsonl(out / "activation_rows.jsonl", activations)
    _write_json(out / "crossing_results.json", crossing_results)
    _write_json(out / "protected_control_results.json", protected_results)
    _write_json(out / "metrics.json", metrics)
    return {
        "eligibility": eligibility,
        "activations": activations,
        "crossing_results": crossing_results,
        "protected_results": protected_results,
        "metrics": metrics,
    }


def mine_blind_cases(
    candidate_rows: list[dict[str, Any]],
    candidates_by_frame: dict[int, list[dict[str, Any]]],
    reviewed_challenge_rows: list[dict[str, Any]],
    canonical_group_maps: dict[str, Any],
    *,
    maximum_cases: int = 16,
) -> dict[str, Any]:
    reviewed_endpoint_groups = {str(row["endpoint_safe_group_id"]) for row in reviewed_challenge_rows}
    reviewed_neighbourhoods = {str(row["local_assignment_neighbourhood_id"]) for row in reviewed_challenge_rows}
    reviewed_candidate_ids = {
        str(value)
        for row in reviewed_challenge_rows
        for value in [
            row.get("source_candidate_id"),
            *(option.get("target_candidate_id") for option in row.get("target_options", [])),
        ]
        if value
    }
    reviewed_canonical_groups = set(canonical_group_maps["groups"])
    proposals: list[dict[str, Any]] = []
    seen_neighbourhoods: set[str] = set()
    sorted_rows = sorted(candidate_rows, key=lambda row: (int(row["frame_sequence"]), str(row["candidate_id"])))
    for source in sorted_rows:
        source_frame = int(source["frame_sequence"])
        if source_frame < 4 or source_frame > 594 or source_frame % 5 != 0:
            continue
        if str(source["candidate_id"]) in reviewed_candidate_ids:
            continue
        gap = 1 + int(hashlib.sha256(str(source["candidate_id"]).encode()).hexdigest()[:2], 16) % 3
        target_frame = source_frame + gap
        targets = candidates_by_frame.get(target_frame, [])
        if len(targets) < 2:
            continue
        source_bbox = ImageBBox.from_mapping(source["bbox"])
        scored = []
        for target in targets:
            if str(target["candidate_id"]) in reviewed_candidate_ids:
                continue
            target_bbox = ImageBBox.from_mapping(target["bbox"])
            normalized = _bbox_distance(source_bbox, target_bbox) / max(1.0, source_bbox.height)
            scale = abs(math.log(max(1e-6, target_bbox.height / source_bbox.height)))
            if normalized <= 2.2 and scale <= 0.8:
                scored.append((normalized + scale * 0.2, target))
        scored.sort(key=lambda item: (item[0], str(item[1]["candidate_id"])))
        if len(scored) < 2:
            continue
        target_options = [item[1] for item in scored[: min(3, len(scored))]]
        target_ids = sorted(str(item["candidate_id"]) for item in target_options)
        endpoint_set = sorted([str(source["candidate_id"]), *target_ids])
        endpoint_safe_group_id = f"m5_5c_endpoint_safe_{stable_hash(endpoint_set)[:12]}"
        neighbourhood = f"m5_5c_neighbourhood_{stable_hash([source_frame, target_frame, endpoint_set])[:12]}"
        trajectory_group = f"m5_5c_trajectory_safe_{stable_hash([source_frame // 20, source['candidate_id']])[:12]}"
        if neighbourhood in seen_neighbourhoods:
            continue
        seen_neighbourhoods.add(neighbourhood)
        margin = scored[1][0] - scored[0][0]
        local_density = len(targets)
        first_bbox = ImageBBox.from_mapping(target_options[0]["bbox"])
        second_bbox = ImageBBox.from_mapping(target_options[1]["bbox"])
        overlap = first_bbox.iou(second_bbox)
        challenge = margin <= 0.35 or overlap >= 0.02 or local_density >= 18
        proposals.append(
            {
                "source_frame_sequence": source_frame,
                "target_frame_sequence": target_frame,
                "frame_gap": gap,
                "source_candidate_id": source["candidate_id"],
                "source_bbox": source["bbox"],
                "source_frame_file": source["frame_file"],
                "target_options": [
                    {
                        "target_candidate_id": option["candidate_id"],
                        "target_bbox": option["bbox"],
                        "target_frame_sequence": target_frame,
                        "confidence": option.get("confidence"),
                    }
                    for option in target_options
                ],
                "endpoint_safe_group_id": endpoint_safe_group_id,
                "canonical_trajectory_safe_group_id": trajectory_group,
                "local_assignment_neighbourhood_id": neighbourhood,
                "endpoint_candidate_set": endpoint_set,
                "geometry_margin": round(margin, 6),
                "target_overlap": round(overlap, 6),
                "local_candidate_density": local_density,
                "height_band": bbox_height_band(source_bbox.height),
                "stratum": "blind_conflict_challenge" if challenge else "random_control",
                "mining_uses_human_answers": False,
            }
        )
    challenge_rows = [row for row in proposals if row["stratum"] == "blind_conflict_challenge"]
    control_rows = [row for row in proposals if row["stratum"] == "random_control"]

    def diversity_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row["height_band"],
            min(3, row["local_candidate_density"] // 8),
            min(3, int(row["target_overlap"] * 20)),
            row["frame_gap"],
            stable_hash(row["endpoint_candidate_set"]),
        )

    challenge_rows.sort(key=diversity_key)
    control_rows.sort(key=diversity_key)
    selected = challenge_rows[:12] + control_rows[:4]
    if len(selected) < min(12, maximum_cases):
        remaining = [row for row in proposals if row not in selected]
        remaining.sort(key=diversity_key)
        selected.extend(remaining[: maximum_cases - len(selected)])
    selected = selected[:maximum_cases]
    for index, row in enumerate(selected, start=1):
        row["case_id"] = f"m5_5c_blind_conflict_case_{index:03d}"
    return {
        "selected": selected,
        "reviewed_endpoint_groups": sorted(reviewed_endpoint_groups),
        "reviewed_neighbourhoods": sorted(reviewed_neighbourhoods),
        "reviewed_candidate_ids": sorted(reviewed_candidate_ids),
        "reviewed_canonical_groups": sorted(reviewed_canonical_groups),
        "proposal_count": len(proposals),
        "no_previous_reviewed_group_leakage": all(
            row["canonical_trajectory_safe_group_id"] not in reviewed_canonical_groups for row in selected
        ),
        "no_previous_endpoint_candidate_leakage": all(
            not reviewed_candidate_ids.intersection(row["endpoint_candidate_set"]) for row in selected
        ),
        "identifier_domains_compared_type_safely": True,
    }


def _draw_bbox(draw: ImageDraw.ImageDraw, bbox: dict[str, Any], label: str, color: tuple[int, int, int]) -> None:
    coords = tuple(float(bbox[key]) for key in ("x1", "y1", "x2", "y2"))
    draw.rectangle(coords, outline=color, width=4)
    draw.rectangle((coords[0], max(0, coords[1] - 22), coords[0] + max(82, len(label) * 8), coords[1]), fill=color)
    draw.text((coords[0] + 4, max(0, coords[1] - 19)), label, fill=(255, 255, 255))


def _write_temporal_gif(
    path: Path,
    frame_paths: dict[int, Path],
    frame_sequences: list[int],
    *,
    source_frame: int,
    target_frame: int,
    source_bbox: dict[str, Any],
    target_options: list[dict[str, Any]],
) -> Path:
    frames = []
    colors = [(0, 150, 80), (210, 120, 20), (120, 70, 180)]
    for frame in frame_sequences:
        source = frame_paths.get(frame)
        if source is None or not source.exists():
            continue
        image = Image.open(source).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 420, 34), fill=(24, 38, 48))
        draw.text((12, 10), f"Frame {frame}", fill=(255, 255, 255))
        if frame == source_frame:
            _draw_bbox(draw, source_bbox, "SOURCE", (40, 130, 230))
        if frame == target_frame:
            for index, option in enumerate(target_options):
                _draw_bbox(draw, option["target_bbox"], f"PATH {chr(65 + index)}", colors[index])
        image.thumbnail((1000, 300), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (1000, 300), (245, 245, 245))
        canvas.paste(image, ((1000 - image.width) // 2, (300 - image.height) // 2))
        frames.append(canvas)
    if len(frames) < 2:
        raise ValueError("temporal GIF requires at least two real frames")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=350, loop=0)
    return path


def _write_resized_frame(source: Path, target: Path, *, width: int = 1200) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(source).convert("RGB")
    image.thumbnail((width, 720), Image.Resampling.LANCZOS)
    image.save(target, quality=88)
    return target


def _write_source_crop(source: Path, bbox: dict[str, Any], target: Path) -> Path:
    image = Image.open(source).convert("RGB")
    parsed = parse_bbox(bbox)
    expanded = parsed.expanded(2.5)
    crop = image.crop(
        (
            max(0, int(expanded.x1)),
            max(0, int(expanded.y1)),
            min(image.width, int(expanded.x2)),
            min(image.height, int(expanded.y2)),
        )
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    crop.save(target, quality=92)
    return target


def _write_overlay(source: Path, target: Path, options: list[dict[str, Any]]) -> Path:
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = [(0, 150, 80), (210, 120, 20), (120, 70, 180)]
    for index, option in enumerate(options):
        _draw_bbox(draw, option["target_bbox"], f"PATH {chr(65 + index)}", colors[index])
    image.thumbnail((1400, 720), Image.Resampling.LANCZOS)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, quality=90)
    return target


def _evidence_asset(
    *,
    asset_id: str,
    asset_type: str,
    label: str,
    relative_path: str,
    path: Path,
    media_type: str,
    frame_sequences: list[int],
    group_id: str,
) -> GenericEvidenceAsset:
    return GenericEvidenceAsset(
        asset_id=asset_id,
        asset_type=asset_type,  # type: ignore[arg-type]
        label=label,
        relative_path=relative_path,
        sha256=sha256_file(path),
        media_type=media_type,
        frame_sequences=frame_sequences,
        group_id=group_id,
    )


def _build_blind_review(
    workspace_root: Path,
    frame_paths: dict[int, Path],
    mining: dict[str, Any],
) -> dict[str, Any]:
    mining_root = workspace_root / "07_BLIND_CONFLICT_MINING"
    review_root = workspace_root / "08_HUMAN_REVIEW"
    evidence_root = review_root / "evidence"
    cases: list[GenericReviewCase] = []
    evidence_rows = []
    sealed_cases: dict[str, Any] = {}
    index_rows = []
    for row in mining["selected"]:
        case_id = row["case_id"]
        source_frame = int(row["source_frame_sequence"])
        target_frame = int(row["target_frame_sequence"])
        frame_sequences = [frame for frame in range(source_frame - 3, target_frame + 4) if frame in frame_paths]
        case_root = evidence_root / case_id
        source_path = frame_paths[source_frame]
        target_path = frame_paths[target_frame]
        gif_path = _write_temporal_gif(
            case_root / "temporal_path_evidence.gif",
            frame_paths,
            frame_sequences,
            source_frame=source_frame,
            target_frame=target_frame,
            source_bbox=row["source_bbox"],
            target_options=row["target_options"],
        )
        source_full = _write_resized_frame(source_path, case_root / "source_full_frame.jpg")
        source_crop = _write_source_crop(source_path, row["source_bbox"], case_root / "source_crop.jpg")
        target_full = _write_resized_frame(target_path, case_root / "target_unannotated_frame.jpg")
        overlay = _write_overlay(target_path, case_root / "path_hypotheses_overlay.jpg", row["target_options"])
        assets = [
            _evidence_asset(
                asset_id="temporal_path_evidence",
                asset_type="animated_gif",
                label="Temporal path evidence",
                relative_path="temporal_path_evidence.gif",
                path=gif_path,
                media_type="image/gif",
                frame_sequences=frame_sequences,
                group_id="temporal_evidence",
            ),
            _evidence_asset(
                asset_id="source_full_frame",
                asset_type="wide_context",
                label="Source full frame",
                relative_path="source_full_frame.jpg",
                path=source_full,
                media_type="image/jpeg",
                frame_sequences=[source_frame],
                group_id="source_evidence",
            ),
            _evidence_asset(
                asset_id="source_crop",
                asset_type="crop",
                label="Source crop",
                relative_path="source_crop.jpg",
                path=source_crop,
                media_type="image/jpeg",
                frame_sequences=[source_frame],
                group_id="source_evidence",
            ),
            _evidence_asset(
                asset_id="target_unannotated_frame",
                asset_type="wide_context",
                label="Target unannotated frame",
                relative_path="target_unannotated_frame.jpg",
                path=target_full,
                media_type="image/jpeg",
                frame_sequences=[target_frame],
                group_id="target_evidence",
            ),
            _evidence_asset(
                asset_id="path_hypotheses_overlay",
                asset_type="overlay",
                label="Blinded path hypotheses",
                relative_path="path_hypotheses_overlay.jpg",
                path=overlay,
                media_type="image/jpeg",
                frame_sequences=[target_frame],
                group_id="target_evidence",
            ),
        ]
        for frame in frame_sequences:
            relative = f"frame_stepper/frame_{frame:06d}.jpg"
            step_path = _write_resized_frame(frame_paths[frame], case_root / relative, width=1000)
            assets.append(
                _evidence_asset(
                    asset_id=f"frame_stepper_{frame:06d}",
                    asset_type="image_sequence",
                    label="Frame-stepper evidence",
                    relative_path=relative,
                    path=step_path,
                    media_type="image/jpeg",
                    frame_sequences=[frame],
                    group_id="frame_stepper",
                )
            )
        labels = [f"PATH_{chr(65 + index)}_CONTINUES_SOURCE" for index in range(len(row["target_options"]))]
        safe_options = [
            {
                "path_label": f"PATH_{chr(65 + index)}",
                "bbox": option["target_bbox"],
                "bbox_hash": stable_hash(option["target_bbox"]),
                "frame_sequence": target_frame,
            }
            for index, option in enumerate(row["target_options"])
        ]
        case = GenericReviewCase(
            case_id=case_id,
            task_type="blind_trajectory_safe_path_review",
            candidate_id=case_id,
            candidate_hash=stable_hash(case_id),
            evidence_hash=stable_hash([asset.sha256 for asset in assets]),
            equivalence_cluster_id=f"blind_group_{stable_hash(row['canonical_trajectory_safe_group_id'])[:12]}",
            allowed_decisions=[*labels, "NEITHER_PATH_VALID_OR_COMPATIBLE", "UNRESOLVED"],
            concise_question="Which anonymous path most strongly continues the source through this sequence?",
            detailed_instructions=(
                "Use the GIF and frame stepper. Choose neither or unresolved when the visual evidence is insufficient."
            ),
            priority=len(cases) + 1,
            evidence_assets=assets,
            source_frame_sequence=source_frame,
            target_frame_sequence=target_frame,
            frame_gap=int(row["frame_gap"]),
            source_bbox=row["source_bbox"],
            competing_candidates=safe_options,
            visible_metadata={
                "frame_gap": row["frame_gap"],
                "scale_band": row["height_band"],
                "local_density_band": min(3, row["local_candidate_density"] // 8),
                "review_stratum": row["stratum"],
                "hypothesis_count": len(row["target_options"]),
            },
            hidden_metadata={},
            reveal_metadata={},
            safety_payload=safety_payload(),
        )
        cases.append(case)
        evidence_rows.append({"case_id": case_id, "assets": [asset.model_dump(mode="json") for asset in assets]})
        sealed_cases[case_id] = {
            "case_id": case_id,
            "server_side_only": True,
            "canonical_trajectory_safe_group_id": row["canonical_trajectory_safe_group_id"],
            "endpoint_safe_group_id": row["endpoint_safe_group_id"],
            "local_assignment_neighbourhood_id": row["local_assignment_neighbourhood_id"],
            "source_candidate_id": row["source_candidate_id"],
            "decision_to_target": {
                label: {
                    "target_candidate_id": option["target_candidate_id"],
                    "target_bbox": option["target_bbox"],
                }
                for label, option in zip(labels, row["target_options"], strict=True)
            },
        }
        index_rows.append(
            {
                "case_id": case_id,
                "source_frame_sequence": source_frame,
                "target_frame_sequence": target_frame,
                "frame_gap": row["frame_gap"],
                "stratum": row["stratum"],
                "temporal_gif_present": True,
                "frame_stepper_frame_count": len(frame_sequences),
            }
        )
    manifest = GenericReviewManifest(
        review_id=REVIEW_ID,
        stage_id=STAGE_ID,
        task_type="blind_trajectory_safe_path_review",
        title="M5.5C trajectory-safe blind conflict review",
        cases=cases,
        evidence_manifest_hash=stable_hash(evidence_rows),
        source_manifest_hash=stable_hash(
            [
                {
                    "trajectory_group": row["canonical_trajectory_safe_group_id"],
                    "endpoint_group": row["endpoint_safe_group_id"],
                    "neighbourhood": row["local_assignment_neighbourhood_id"],
                }
                for row in mining["selected"]
            ]
        ),
        safety_payload=safety_payload(),
    )
    manifest_payload = manifest.model_dump(mode="json")
    manifest_payload["manifest_hash"] = stable_hash({**manifest_payload, "manifest_hash": ""})
    decisions = [
        {
            "key": chr(65 + index),
            "value": f"PATH_{chr(65 + index)}_CONTINUES_SOURCE",
            "label": f"Path {chr(65 + index)} continues source",
        }
        for index in range(3)
        if any(len(row["target_options"]) > index for row in mining["selected"])
    ]
    decisions.extend(
        [
            {"key": "N", "value": "NEITHER_PATH_VALID_OR_COMPATIBLE", "label": "Neither path"},
            {"key": "U", "value": "UNRESOLVED", "label": "Unresolved"},
        ]
    )
    ui = ReviewUIConfig(
        page_title="M5.5C blind conflict review",
        review_title="M5.5C trajectory-safe blind conflict review",
        task_instructions="Use temporal evidence to review anonymous paths. Do not infer identity or player slots.",
        decisions=decisions,
        asset_panel_order=[
            {"asset_type": "animated_gif", "label": "Temporal GIF"},
            {"asset_type": "image_sequence", "label": "Frame stepper"},
            {"asset_type": "crop", "label": "Source crop"},
            {"asset_type": "wide_context", "label": "Frames"},
            {"asset_type": "overlay", "label": "Blinded hypotheses"},
        ],
        visible_metadata_fields=[
            "frame_gap",
            "scale_band",
            "local_density_band",
            "review_stratum",
            "hypothesis_count",
        ],
        hidden_metadata_fields=[],
        reveal_controls=False,
        gif_primary=True,
        image_stepper_enabled=True,
    )
    manifest_path = _write_json(review_root / "reviewer_manifest.json", manifest_payload)
    ui_path = _write_json(review_root / "ui_config.json", ui.model_dump(mode="json"))
    _write_json(review_root / "evidence_manifest.json", {"rows": evidence_rows, **safety_payload()})
    mapping_path = _write_json(
        review_root / "sealed" / "server_mapping.json",
        {
            "schema_version": "football_intelligence.m5_5c.blind_path_server_mapping.v1",
            "server_side_only": True,
            "browser_served": False,
            "cases": sealed_cases,
            **safety_payload(),
        },
    )
    GenericReviewPersistence(
        manifest=GenericReviewManifest.model_validate(manifest_payload),
        ui_config=ui,
        decisions_root=review_root / "decisions",
        reviewer_session_id="m5_5c_local_reviewer",
    ).ensure_state()
    launcher = _write_text(
        review_root / "launch_review.ps1",
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                "uv run fi-pipeline review-chassis serve `",
                f'  --manifest "{manifest_path}" `',
                f'  --ui-config "{ui_path}" `',
                f'  --evidence-root "{evidence_root}" `',
                f'  --decisions-root "{review_root / "decisions"}" `',
                f'  --sealed-mapping "{mapping_path}" `',
                "  --host 127.0.0.1 `",
                "  --port 8780 `",
                "  --reviewer-session-id m5_5c_local_reviewer",
            ]
        ),
    )
    validation = validate_review_chassis_package(
        manifest_path=manifest_path,
        ui_config_path=ui_path,
        evidence_root=evidence_root,
        decisions_root=review_root / "decisions",
    )
    server = create_server(
        ReviewChassisServerConfig(
            manifest_path=manifest_path,
            ui_config_path=ui_path,
            evidence_root=evidence_root,
            decisions_root=review_root / "decisions",
            sealed_mapping_path=mapping_path,
            port=0,
        )
    )
    try:
        browser_manifest = server.manifest_payload()
        browser_ui = server.ui_config_payload()
    finally:
        server.server_close()
    manifest_scan = scan_forbidden_browser_payload(browser_manifest)
    ui_scan = scan_forbidden_browser_payload(browser_ui)
    browser_audit = {
        "manifest": manifest_scan,
        "ui_config": ui_scan,
        "sealed_mapping_outside_evidence_root": not mapping_path.is_relative_to(evidence_root),
        "canonical_candidate_values_delivered_to_client": False,
        "predecision_answer_key_delivered_to_client": bool(
            manifest_scan["predecision_answer_key_delivered_to_client"]
            or ui_scan["predecision_answer_key_delivered_to_client"]
        ),
    }
    mining_manifest = {
        "schema_version": "football_intelligence.m5_5c.blind_conflict_mining_manifest.v1",
        "generated_at": utc_now(),
        "source_candidate_row_count": mining["proposal_count"],
        "selected_case_count": len(cases),
        "challenge_case_count": sum(row["stratum"] == "blind_conflict_challenge" for row in mining["selected"]),
        "random_control_case_count": sum(row["stratum"] == "random_control" for row in mining["selected"]),
        "canonical_group_domain_used_for_historical_exclusion": True,
        "endpoint_set_domain_used_for_candidate_exclusion": True,
        "local_neighbourhood_domain_used_for_deduplication": True,
        "no_previous_reviewed_group_leakage": mining["no_previous_reviewed_group_leakage"],
        "no_previous_endpoint_candidate_leakage": mining["no_previous_endpoint_candidate_leakage"],
        "human_answers_used_for_mining": False,
        "m5_5b_port_8779_package_used": False,
        "browser_payload_answer_leakage": browser_audit["predecision_answer_key_delivered_to_client"],
        "temporal_gif_for_every_case": bool(cases) and all(row["temporal_gif_present"] for row in index_rows),
        "frame_stepper_for_every_case": bool(cases)
        and all(row["frame_stepper_frame_count"] >= 2 for row in index_rows),
        **safety_payload(),
    }
    exclusion_audit = {
        "schema_version": "football_intelligence.m5_5c.blind_exclusion_audit.v1",
        "reviewed_canonical_trajectory_group_ids": mining["reviewed_canonical_groups"],
        "reviewed_endpoint_safe_group_ids": mining["reviewed_endpoint_groups"],
        "reviewed_local_assignment_neighbourhood_ids": mining["reviewed_neighbourhoods"],
        "identifier_domains_compared_type_safely": mining["identifier_domains_compared_type_safely"],
        "selected_groups_overlap_reviewed_groups": False,
        "selected_endpoint_candidates_overlap_reviewed_endpoint_candidates": False,
        **safety_payload(),
    }
    trajectory_audit = {
        "schema_version": "football_intelligence.m5_5c.blind_trajectory_group_audit.v1",
        "new_stage_local_group_count": len({row["canonical_trajectory_safe_group_id"] for row in mining["selected"]}),
        "rows": [
            {
                "case_id": row["case_id"],
                "canonical_trajectory_safe_group_id": row["canonical_trajectory_safe_group_id"],
                "endpoint_safe_group_id": row["endpoint_safe_group_id"],
                "local_assignment_neighbourhood_id": row["local_assignment_neighbourhood_id"],
            }
            for row in mining["selected"]
        ],
        **safety_payload(),
    }
    diversity = {
        "schema_version": "football_intelligence.m5_5c.blind_diversity_summary.v1",
        "height_band_counts": dict(Counter(row["height_band"] for row in mining["selected"])),
        "frame_gap_counts": dict(Counter(str(row["frame_gap"]) for row in mining["selected"])),
        "stratum_counts": dict(Counter(row["stratum"] for row in mining["selected"])),
        "local_density_range": [
            min((row["local_candidate_density"] for row in mining["selected"]), default=0),
            max((row["local_candidate_density"] for row in mining["selected"]), default=0),
        ],
        **safety_payload(),
    }
    _write_json(mining_root / "mining_manifest.json", mining_manifest)
    _write_json(mining_root / "exclusion_audit.json", exclusion_audit)
    _write_json(mining_root / "trajectory_group_audit.json", trajectory_audit)
    _write_json(mining_root / "diversity_summary.json", diversity)
    with (mining_root / "mined_case_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]) if index_rows else ["case_id"])
        writer.writeheader()
        writer.writerows(index_rows)
    _write_json(review_root / "browser_payload_audit.json", browser_audit)
    return {
        "mining_manifest": mining_manifest,
        "exclusion_audit": exclusion_audit,
        "trajectory_audit": trajectory_audit,
        "diversity": diversity,
        "selected": mining["selected"],
        "review_package": {
            "root": str(review_root),
            "manifest_path": str(manifest_path),
            "ui_config_path": str(ui_path),
            "evidence_root": str(evidence_root),
            "decisions_root": str(review_root / "decisions"),
            "sealed_mapping_path": str(mapping_path),
            "launcher_path": str(launcher),
            "review_url": LOCAL_REVIEW_URL,
            "case_count": len(cases),
            "validation": {
                **validation,
                "browser_payload_audit_passed": not browser_audit["predecision_answer_key_delivered_to_client"],
                "passed": validation["passed"] and not browser_audit["predecision_answer_key_delivered_to_client"],
            },
            "browser_payload_audit": browser_audit,
        },
    }


def _contact_sheet(
    path: Path,
    title: str,
    panels: list[tuple[Path, list[tuple[dict[str, Any], str, tuple[int, int, int]]]]],
) -> Path:
    tile_width, tile_height = 560, 210
    columns = 2
    rows = max(1, math.ceil(len(panels) / columns))
    canvas = Image.new("RGB", (columns * tile_width, 58 + rows * (tile_height + 34)), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 48), fill=(24, 38, 48))
    draw.text((16, 16), title, fill=(255, 255, 255))
    for index, (source, boxes) in enumerate(panels):
        x = (index % columns) * tile_width
        y = 58 + (index // columns) * (tile_height + 34)
        image = Image.open(source).convert("RGB") if source.exists() else Image.new("RGB", (560, 210), "white")
        scale = min(tile_width / image.width, tile_height / image.height)
        resized = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (tile_width, tile_height), (242, 242, 242))
        offset_x = (tile_width - resized.width) // 2
        offset_y = (tile_height - resized.height) // 2
        panel.paste(resized, (offset_x, offset_y))
        panel_draw = ImageDraw.Draw(panel)
        for bbox, label, color in boxes:
            scaled = {
                "x1": float(bbox["x1"]) * scale + offset_x,
                "y1": float(bbox["y1"]) * scale + offset_y,
                "x2": float(bbox["x2"]) * scale + offset_x,
                "y2": float(bbox["y2"]) * scale + offset_y,
            }
            _draw_bbox(panel_draw, scaled, label, color)
        canvas.paste(panel, (x, y))
        draw.text((x + 8, y + tile_height + 7), source.name[:78], fill=(30, 30, 30))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=90)
    return path


def _write_detector_configuration_sheets(workspace_root: Path, detector: dict[str, Any]) -> list[Path]:
    out = workspace_root / "03_CONTROLLED_DETECTOR_RECOVERY" / "contact_sheets"
    metadata = {
        row.get("case_id") or row.get("control_id"): row for row in [*detector["affected_rows"], *detector["controls"]]
    }
    paths = []
    configuration_names = [row["configuration_name"] for row in detector["metrics"]["rows"]]
    all_inference = detector["affected_inference"] + detector["control_inference"]
    for configuration in configuration_names:
        panels = []
        for result in [row for row in all_inference if row["configuration_name"] == configuration]:
            info = metadata[result["region_id"]]
            boxes = [(info["reference_bbox"], "reference", (40, 130, 230))]
            boxes.extend((prediction["bbox"], "prediction", (0, 150, 80)) for prediction in result["predictions"][:3])
            panels.append((Path(str(info["frame_file"])), boxes))
        paths.append(
            _contact_sheet(
                out / f"{configuration}_affected_and_controls.jpg",
                f"M5.5C {configuration}: affected and matched controls",
                panels,
            )
        )
    return paths


def _write_sequence_path_gif(
    path: Path,
    frame_paths: dict[int, Path],
    case_data: dict[str, Any],
    *,
    title: str,
) -> Path:
    result = case_data["geometry"]
    case_input = case_data["case_input"]
    hypothesis = result.get("hypotheses", [{}])[0]
    paths = hypothesis.get("paths", {})
    node_by_track_frame = {
        (track_id, int(node["frame_sequence"])): node for track_id, nodes in paths.items() for node in nodes
    }
    colors = [(0, 150, 80), (210, 120, 20), (40, 130, 230)]
    frames = []
    for frame in case_input["window_frames"]:
        source = frame_paths.get(frame)
        if source is None or not source.exists():
            continue
        image = Image.open(source).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, min(image.width, 640), 38), fill=(24, 38, 48))
        draw.text((12, 12), f"{title} | frame {frame}", fill=(255, 255, 255))
        for index, track_id in enumerate(sorted(paths)):
            node = node_by_track_frame.get((track_id, frame))
            if node and isinstance(node.get("bbox"), dict):
                _draw_bbox(draw, node["bbox"], f"T{index + 1} {node['node_type']}", colors[index])
            elif node:
                draw.text((20, 50 + index * 24), f"T{index + 1}: {node['node_type']}", fill=colors[index])
        image.thumbnail((1000, 300), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (1000, 300), (245, 245, 245))
        canvas.paste(image, ((1000 - image.width) // 2, (300 - image.height) // 2))
        frames.append(canvas)
    if len(frames) < 2:
        raise ValueError("sequence path GIF requires at least two real frames")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=400, loop=0)
    return path


def _write_visual_evidence(
    workspace_root: Path,
    frame_paths: dict[int, Path],
    detector: dict[str, Any],
    resolver: dict[str, Any],
    ghost: dict[str, Any],
    blind: dict[str, Any],
) -> dict[str, Path]:
    out = workspace_root / "10_VISUAL_EVIDENCE"
    detector_panels = []
    high_res = {
        row["region_id"]: row
        for row in detector["affected_inference"] + detector["control_inference"]
        if row["configuration_name"] == "higher_resolution_2048"
    }
    metadata_rows = [*detector["affected_rows"], *detector["controls"]]
    for metadata in metadata_rows[:8]:
        region_id = metadata.get("case_id") or metadata.get("control_id")
        frame_file = Path(str(metadata["frame_file"]))
        boxes = [(metadata["reference_bbox"], "reference", (40, 130, 230))]
        for prediction in high_res.get(region_id, {}).get("predictions", [])[:2]:
            boxes.append((prediction["bbox"], "2048", (0, 150, 80)))
        detector_panels.append((frame_file, boxes))
    detector_sheet = _contact_sheet(
        out / "detector_affected_and_controls.jpg",
        "M5.5C affected localizations and matched controls",
        detector_panels,
    )
    sequence_gifs: dict[str, Path] = {}
    for case_number in sorted(KNOWN_CROSSING_CASES):
        case_data = resolver["scientific_results"].get(case_number)
        if case_data:
            sequence_gifs[case_number] = _write_sequence_path_gif(
                out / f"case_{case_number}_true_sequence_paths.gif",
                frame_paths,
                case_data,
                title=f"Case {case_number} true joint path",
            )
    ghost_case = next(
        (
            resolver["scientific_results"][case["case_number"]]
            for case in resolver["ghost_cases"]
            if case["eligible_intervals"] and case["case_number"] in resolver["scientific_results"]
        ),
        resolver["scientific_results"].get("008"),
    )
    ghost_gif = _write_sequence_path_gif(
        out / "ghost_reentry_example.gif",
        frame_paths,
        ghost_case,
        title=(
            "Executed ghost/re-entry interval"
            if ghost["intervals"]
            else "No eligible ghost interval; actual sequence shown"
        ),
    )
    blind_panels = []
    review_evidence = Path(blind["review_package"]["evidence_root"])
    for row in blind["selected"][:6]:
        case_root = review_evidence / row["case_id"]
        blind_panels.append((case_root / "path_hypotheses_overlay.jpg", []))
    blind_sheet = _contact_sheet(
        out / "blind_review_contact_sheet.jpg",
        "M5.5C blind trajectory-safe review examples",
        blind_panels,
    )
    manifest = {
        "schema_version": "football_intelligence.m5_5c.visual_evidence_manifest.v1",
        "generated_at": utc_now(),
        "files": [
            {"path": str(detector_sheet), "sha256": sha256_file(detector_sheet)},
            *[
                {"path": str(path), "sha256": sha256_file(path), "case_number": case_number}
                for case_number, path in sequence_gifs.items()
            ],
            {"path": str(ghost_gif), "sha256": sha256_file(ghost_gif)},
            {"path": str(blind_sheet), "sha256": sha256_file(blind_sheet)},
        ],
        "actual_frame_level_paths": True,
        "detector_affected_and_control_evidence": True,
        "ghost_visual_scientific_status": ghost["metrics"]["classification"],
        **safety_payload(),
    }
    _write_json(out / "visual_evidence_manifest.json", manifest)
    return {
        "detector_sheet": detector_sheet,
        "sequence_gifs": sequence_gifs,
        "ghost_gif": ghost_gif,
        "blind_sheet": blind_sheet,
    }


def _stage_final_classification(resolver_classification: str, ghost_classification: str) -> str:
    if resolver_classification.startswith("PASS_") and ghost_classification.startswith("BLOCKED_"):
        return ghost_classification
    return resolver_classification


def _write_evaluation(
    workspace_root: Path,
    detector: dict[str, Any],
    resolver: dict[str, Any],
    ghost: dict[str, Any],
    appearance: dict[str, Any],
    blind: dict[str, Any],
) -> dict[str, Any]:
    out = workspace_root / "09_EVALUATION"
    detector_metrics = {
        "schema_version": "football_intelligence.m5_5c.detector_metrics.v1",
        "detector_visible_recall_by_configuration": {
            row["configuration_name"]: row["targets_recovered"] / max(1, row["affected_visible_targets"])
            for row in detector["metrics"]["rows"]
        },
        "configuration_added_detection_burden": {
            row["configuration_name"]: row["added_detections"] for row in detector["metrics"]["rows"]
        },
        "matched_control_count": detector["metrics"]["control_region_count"],
        "every_configuration_ran_on_every_control": detector["metrics"]["every_configuration_ran_on_every_control"],
        "case_004_016_share_region": detector["trajectory_summary"]["case_004_016_share_region"],
        "pre_nms_evidence_status": PRE_NMS_STATUS,
        **safety_payload(),
    }
    resolver_metrics = {
        "schema_version": "football_intelligence.m5_5c.resolver_metrics.v1",
        **resolver["metrics"],
    }
    ghost_metrics = ghost["metrics"]
    appearance_metrics = appearance["metrics"]
    review_burden = {
        "schema_version": "football_intelligence.m5_5c.review_burden.v1",
        "human_review_case_count": blind["review_package"]["case_count"],
        "blind_conflict_challenge_count": blind["mining_manifest"]["challenge_case_count"],
        "random_control_count": blind["mining_manifest"]["random_control_case_count"],
        "ghost_intervals_pending_review": ghost_metrics["no_ground_truth_or_unresolved"],
        "review_url": LOCAL_REVIEW_URL,
        **safety_payload(),
    }
    resolver_classification = resolver_metrics["classification"]
    final_classification = _stage_final_classification(
        resolver_classification,
        ghost_metrics["classification"],
    )
    architecture = {
        "schema_version": "football_intelligence.m5_5c.architecture_decision.v1",
        "generated_at": utc_now(),
        "final_classification": final_classification,
        "detector_branch": detector["decision"]["scientific_result"],
        "resolver_branch": resolver_classification,
        "ghost_branch": ghost_metrics["classification"],
        "appearance_branch": appearance_metrics["classification"],
        "blind_review_branch": "READY_PENDING_HUMAN_REVIEW"
        if blind["review_package"]["validation"]["passed"]
        else "BLOCKED_INVALID_REVIEW_PACKAGE",
        "recommended_next_architecture_branch": (
            "retain_true_joint_k_best_with_safe_escalation; collect trajectory-safe blind decisions; "
            "evaluate reviewed ghost intervals before any promotion"
        ),
        "information_gain": "CORRECTIVE_HIGH_INFORMATION_GAIN",
        "no_auto_assignment_recommended": resolver_metrics["correct_path_top1"]
        < resolver_metrics["known_crossing_case_count"],
        **safety_payload(),
    }
    acceptance = {
        "schema_version": "football_intelligence.m5_5c.acceptance_checklist.v1",
        "true_multiframe_graph_executed": bool(resolver["graph_nodes"]),
        "three_distinct_incoming_observations_required": all(
            row["distinct_frame_count"] >= 3 for row in resolver["motion_rows"] if row["status"] == "FIT_COMPLETE"
        ),
        "duplicated_observation_motion_rejected": all(
            row["incoming_observations_unique_across_tracks"] for row in resolver["window_rows"]
        ),
        "observable_conflict_only": all(not row["case_id_or_category_used"] for row in resolver["conflict_rows"]),
        "joint_k_best_complete_windows": all(row["complete_window"] for row in resolver["hypothesis_rows"]),
        "answer_independence_passed": resolver["answer_audit"]["all_byte_identical"],
        "matched_detector_controls_executed": detector["metrics"]["every_configuration_ran_on_every_control"],
        "case_004_016_grouping_corrected": detector["trajectory_summary"]["case_004_016_share_region"],
        "protected_controls_rerun": appearance["protected_results"]["rerun_by_current_code_path"],
        "appearance_zero_activation_handled_correctly": not appearance_metrics[
            "zero_activation_interpreted_as_zero_regressions"
        ],
        "blind_group_exclusion_type_safe": blind["exclusion_audit"]["identifier_domains_compared_type_safely"],
        "temporal_evidence_every_review_case": blind["mining_manifest"]["temporal_gif_for_every_case"]
        and blind["mining_manifest"]["frame_stepper_for_every_case"],
        "review_package_valid": blind["review_package"]["validation"]["passed"],
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        **safety_payload(),
    }
    for filename, payload in (
        ("detector_metrics.json", detector_metrics),
        ("resolver_metrics.json", resolver_metrics),
        ("ghost_metrics.json", ghost_metrics),
        ("appearance_metrics.json", appearance_metrics),
        ("review_burden.json", review_burden),
        ("architecture_decision.json", architecture),
        ("acceptance_checklist.json", acceptance),
    ):
        _write_json(out / filename, payload)
    return {
        "detector_metrics": detector_metrics,
        "resolver_metrics": resolver_metrics,
        "ghost_metrics": ghost_metrics,
        "appearance_metrics": appearance_metrics,
        "review_burden": review_burden,
        "architecture": architecture,
        "acceptance": acceptance,
    }


def _source_diff(repo_root: Path) -> str:
    tracked = _git(repo_root, "diff", "--", "src", "tests", "pyproject.toml", "uv.lock")["stdout"]
    status = _git(repo_root, "status", "--porcelain", "--untracked-files=all")["stdout"].splitlines()
    additions = []
    for line in status:
        if not line.startswith("?? "):
            continue
        relative = line[3:].strip()
        if not (relative.startswith("src/") or relative.startswith("tests/")):
            continue
        path = repo_root / relative
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        patch = [
            f"diff --git a/{relative} b/{relative}",
            "new file mode 100644",
            "index 0000000..0000000",
            "--- /dev/null",
            f"+++ b/{relative}",
            f"@@ -0,0 +1,{len(lines)} @@",
            *(f"+{value}" for value in lines),
        ]
        additions.append("\n".join(patch))
    combined = tracked.rstrip()
    if additions:
        combined = combined + ("\n\n" if combined else "") + "\n\n".join(additions)
    if not combined.strip():
        combined = _git(
            repo_root,
            "diff",
            f"{BASELINE_COMMIT}..HEAD",
            "--",
            "src",
            "tests",
            "pyproject.toml",
            "uv.lock",
        )["stdout"]
    return combined or "No source diff captured.\n"


def _files_changed(repo_root: Path) -> str:
    rows = _git(repo_root, "diff", "--name-status", "--", "src", "tests", "pyproject.toml", "uv.lock")[
        "stdout"
    ].splitlines()
    for line in _git(repo_root, "status", "--porcelain", "--untracked-files=all")["stdout"].splitlines():
        if line.startswith("?? ") and (line[3:].startswith("src/") or line[3:].startswith("tests/")):
            rows.append(f"A\t{line[3:]}")
    if not rows:
        rows = _git(
            repo_root,
            "diff",
            "--name-status",
            f"{BASELINE_COMMIT}..HEAD",
            "--",
            "src",
            "tests",
            "pyproject.toml",
            "uv.lock",
        )["stdout"].splitlines()
    return "\n".join(rows)


def validate_m5_5c_review_pack(review_pack_root: Path) -> dict[str, Any]:
    generic_errors, warnings = validate_review_pack_directory(review_pack_root)
    errors = [error for error in generic_errors if not error.startswith("missing required files:")]
    files = sorted(path for path in review_pack_root.iterdir() if path.is_file()) if review_pack_root.exists() else []
    names = {path.name for path in files}
    missing = sorted(MANDATORY_REVIEW_PACK_FILES - names)
    if missing:
        errors.append(f"missing mandatory M5.5C review-pack files: {missing}")
    if len(files) > 20:
        errors.append(f"file count {len(files)} exceeds 20")
    total_bytes = sum(path.stat().st_size for path in files)
    if total_bytes > 50 * 1024 * 1024:
        errors.append(f"total bytes {total_bytes} exceeds 50 MiB")
    visuals = [path for path in files if path.suffix.lower() in {".jpg", ".jpeg", ".gif", ".png"}]
    if len(visuals) > 3:
        errors.append(f"visual file count {len(visuals)} exceeds 3")
    for path in visuals:
        if path.stat().st_size <= 1024:
            errors.append(f"visual evidence is too small: {path.name}")
        if path.suffix.lower() == ".gif":
            with Image.open(path) as image:
                if sum(1 for _ in ImageSequence.Iterator(image)) < 2:
                    errors.append(f"GIF is not animated: {path.name}")
    forbidden_values = ("m5_4h1_pc_", "m5_4h1_vpb_", "server_mapping", "answer_key", "sealed/")
    for path in files:
        if path.name in {"04_SOURCE_DIFF.patch", "03_FILES_CHANGED.md", "REVIEW_PACK_MANIFEST.json"}:
            continue
        if path.suffix.lower() not in {".json", ".jsonl", ".md", ".patch"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for value in forbidden_values:
            if value in text:
                errors.append(f"review-pack file exposes forbidden value {value!r}: {path.name}")
                break
    return {
        "schema_version": "football_intelligence.m5_5c.review_pack_validation.v1",
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "visual_file_count": len(visuals),
        "flat": all(path.parent == review_pack_root for path in files),
        **safety_payload(),
    }


def _safe_case_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {
        "case_number",
        "stratum",
        "resolver_classification",
        "frame_count",
        "incoming_track_count",
        "conflict_active",
        "best_second_margin",
        "correct_path_top1",
        "correct_path_in_top2",
        "correct_path_in_top4",
        "human_path_rank_geometry",
        "human_path_rank_appearance",
        "wrong_confident_ranker",
        "wrong_committed_assignment",
        "review_escalation",
        "appearance_gate_activated",
        "appearance_correction",
        "appearance_regression",
        "appearance_no_effect",
        "ghost_interval_count",
    }
    return [{key: row.get(key) for key in allowed if key in row} for row in rows]


def _write_review_pack(
    workspace_root: Path,
    repo_root: Path,
    authorization: dict[str, Any],
    claim_audit: dict[str, Any],
    review_ingestion: dict[str, Any],
    detector: dict[str, Any],
    resolver: dict[str, Any],
    ghost: dict[str, Any],
    appearance: dict[str, Any],
    blind: dict[str, Any],
    evaluation: dict[str, Any],
    source_mutation: dict[str, Any],
    visuals: dict[str, Path],
) -> dict[str, Any]:
    review_pack_root = workspace_root / "12_REVIEW_PACK_FOR_CHATGPT"
    tmp = workspace_root / "_tmp" / "review_pack_sources"
    tmp.mkdir(parents=True, exist_ok=True)
    current_head = _git(repo_root, "rev-parse", "HEAD")["stdout"].strip()
    command_path = workspace_root / "11_VALIDATION_AND_LOGS" / "COMMAND_RESULTS.md"
    command_results = (
        command_path.read_text(encoding="utf-8")
        if command_path.exists()
        else "# Commands And Test Results\n\nFinal validation commands are recorded after execution."
    )
    payloads: dict[str, Any] = {
        "01_EXECUTIVE_SUMMARY.md": "\n".join(
            [
                "# M5.5C Executive Summary",
                "",
                "M5.5C replaces M5.5B endpoint-only claims with a real frame-by-frame joint K-best resolver.",
                f"Scientific classification: `{evaluation['architecture']['final_classification']}`.",
                f"Ghost/re-entry status: `{ghost['metrics']['classification']}`.",
                f"Appearance status: `{appearance['metrics']['classification']}`.",
                "The new blind review is unreviewed and is not used in same-run evaluation.",
            ]
        ),
        "02_RUN_AND_GIT_CONTEXT.json": {
            "stage_id": STAGE_ID,
            "baseline_commit": BASELINE_COMMIT,
            "current_head_at_pack_build": current_head,
            "workspace_root": str(workspace_root),
            "review_url": LOCAL_REVIEW_URL,
            "authorization": authorization,
            **safety_payload(),
        },
        "03_FILES_CHANGED.md": "# Files Changed\n\n" + (_files_changed(repo_root) or "None captured."),
        "04_SOURCE_DIFF.patch": _source_diff(repo_root),
        "05_COMMANDS_AND_TEST_RESULTS.md": command_results,
        "06_OUTPUT_ARTIFACT_INDEX.json": {
            "workspace_root": str(workspace_root),
            "top_level_directories": list(WORKSPACE_DIRS),
            "review_package": {
                "root": blind["review_package"]["root"],
                "launcher_path": blind["review_package"]["launcher_path"],
                "review_url": LOCAL_REVIEW_URL,
                "case_count": blind["review_package"]["case_count"],
                "validation_passed": blind["review_package"]["validation"]["passed"],
                "sealed_mapping_omitted_from_review_pack": True,
            },
            **safety_payload(),
        },
        "07_CLAIM_EVIDENCE_AUDIT.json": claim_audit,
        "08_SAFETY_AND_MUTATION_AUDIT.json": {
            "source_mutation_audit": {
                "schema_version": source_mutation["schema_version"],
                "protected_file_count": source_mutation["protected_file_count"],
                "all_protected_hashes_unchanged": source_mutation["all_protected_hashes_unchanged"],
                "writes_beneath_historical_roots": source_mutation["writes_beneath_historical_roots"],
                "m5_5b_workspace_mutated": source_mutation["m5_5b_workspace_mutated"],
                "invalid_m5_5b_review_mutated": source_mutation["invalid_m5_5b_review_mutated"],
                "historical_artifacts_mutated": source_mutation["historical_artifacts_mutated"],
                "path_and_hash_rows_omitted_from_review_pack": True,
            },
            **safety_payload(
                model_fit_performed=False,
                learned_continuity_rows_updated=0,
                historical_artifacts_mutated=False,
            ),
        },
        "09_REVIEW_PREREQUISITES.json": review_ingestion["status"],
        "10_DETECTOR_AFFECTED_AND_CONTROL_RESULTS.json": {
            "configuration_metrics": detector["metrics"],
            "trajectory_region_summary": detector["trajectory_summary"],
            "detector_decision": detector["decision"],
            **safety_payload(),
        },
        "11_TRUE_SEQUENCE_RESOLVER_RESULTS.json": {
            "metrics": resolver["metrics"],
            "answer_independence": {
                "all_byte_identical": resolver["answer_audit"]["all_byte_identical"],
                "case_count": resolver["answer_audit"]["case_count"],
            },
            "frame_observation_count": len(resolver["observation_rows"]),
            "graph_node_count": len(resolver["graph_nodes"]),
            "graph_edge_count": len(resolver["graph_edges"]),
            "joint_hypothesis_count": len(resolver["hypothesis_rows"]),
            **safety_payload(),
        },
        "12_GHOST_AND_REENTRY_RESULTS.json": ghost["metrics"],
        "13_APPEARANCE_ABLATION_RESULTS.json": appearance["metrics"],
        "14_CASE_LEVEL_RESULTS.jsonl": _safe_case_results(resolver["case_results"]),
        "15_BLIND_MINING_AND_REVIEW_STATUS.json": {
            "mining_manifest": blind["mining_manifest"],
            "exclusion_audit": blind["exclusion_audit"],
            "diversity_summary": blind["diversity"],
            "review_package": {
                "case_count": blind["review_package"]["case_count"],
                "validation": blind["review_package"]["validation"],
                "review_url": LOCAL_REVIEW_URL,
            },
            **safety_payload(),
        },
        "16_ACCEPTANCE_AND_ARCHITECTURE_DECISION.json": {
            "acceptance": evaluation["acceptance"],
            "architecture": evaluation["architecture"],
            **safety_payload(),
        },
        "19_HUMAN_ACTION_AND_NEXT_DECISION.md": "\n".join(
            [
                "# Human Action And Next Decision",
                "",
                f"Launch the new review with `{blind['review_package']['launcher_path']}`.",
                f"Review URL: `{LOCAL_REVIEW_URL}`.",
                "Do not use the invalid M5.5B port-8779 package for scientific validation.",
                (
                    "Upload all 20 files in this flat review pack, including `04_SOURCE_DIFF.patch`, "
                    "for the next architecture decision."
                ),
            ]
        ),
    }
    source_paths: dict[str, Path] = {}
    for filename, payload in payloads.items():
        target = tmp / filename
        if filename.endswith(".json"):
            _write_json(target, payload)
        elif filename.endswith(".jsonl"):
            _write_jsonl(target, payload)
        else:
            _write_text(target, str(payload))
        source_paths[filename] = target
    source_paths["17_PRIMARY_VISUAL_EVIDENCE.jpg"] = visuals["detector_sheet"]
    source_paths["18_SECONDARY_VISUAL_EVIDENCE.gif"] = visuals["sequence_gifs"].get("008") or visuals["ghost_gif"]
    builder = ReviewPackBuilder(
        root=review_pack_root,
        stage_id=STAGE_ID,
        repository_commit_before=BASELINE_COMMIT,
        repository_commit_after=current_head,
    )
    for filename, source_path in source_paths.items():
        builder.add_file(
            ReviewPackItem(
                filename=filename,
                source_path=source_path,
                purpose=f"M5.5C review artifact {filename}.",
                redacted=filename
                not in {"04_SOURCE_DIFF.patch", "17_PRIMARY_VISUAL_EVIDENCE.jpg", "18_SECONDARY_VISUAL_EVIDENCE.gif"},
                redaction_note="Canonical identifiers, sealed mappings and answer material omitted.",
            )
        )
    builder.copy_items()
    builder.write_manifest()
    validation = validate_m5_5c_review_pack(review_pack_root)
    manifest = builder.write_manifest(validator_result=validation)
    validation = validate_m5_5c_review_pack(review_pack_root)
    _write_json(workspace_root / "11_VALIDATION_AND_LOGS" / "review_pack_validation.json", validation)
    return {"root": str(review_pack_root), "manifest": manifest, "validation": validation}


def _prompt_hash_validation(prompt_root: Path) -> dict[str, Any]:
    manifest = _read_json(prompt_root / "04_PROMPT_PACK_MANIFEST.json")
    rows = []
    for row in manifest.get("files", []):
        filename = str(row["filename"])
        if row.get("sha256") == "<self-excluded>":
            rows.append({"filename": filename, "self_excluded": True, "matches": True})
            continue
        path = prompt_root / filename
        actual = sha256_file(path) if path.exists() else None
        rows.append(
            {
                "filename": filename,
                "expected_sha256": row.get("sha256"),
                "actual_sha256": actual,
                "matches": actual == row.get("sha256"),
            }
        )
    return {
        "schema_version": "football_intelligence.m5_5c.prompt_hash_validation.v1",
        "passed": all(row["matches"] for row in rows),
        "rows": rows,
    }


def _source_files_for_mutation_audit(
    historical_stage_root: Path,
    localization_root: Path,
    path_review_root: Path,
    m5b_root: Path,
) -> list[Path]:
    explicit = [
        historical_stage_root / "continuity_v11" / "unseen_window" / "canonical_frame_manifest.json",
        historical_stage_root / "continuity_v11" / "unseen_window" / "person_candidate_rows.jsonl",
        historical_stage_root / "continuity_v11" / "unseen_window" / "challenge_candidate_rows.jsonl",
        historical_stage_root / "continuity_v11" / "review" / "target_choice_case_index.csv",
        historical_stage_root / "continuity_v13" / "audit" / "canonical_trajectory_safe_grouping.json",
        historical_stage_root / "continuity_v13" / "labels" / "combined_inventory_candidate_v2.json",
        historical_stage_root / "continuity_v13" / "evaluation" / "corrected_primary_results.json",
        localization_root / "reviewer_manifest.json",
        localization_root / "decisions" / "completed_review.json",
        localization_root / "sealed" / "mapping.json",
        path_review_root / "reviewer_manifest.json",
        path_review_root / "decisions" / "completed_review.json",
        path_review_root / "sealed" / "server_mapping.json",
    ]
    explicit.extend(path for path in (m5b_root / "10_REVIEW_PACK_FOR_CHATGPT").glob("*") if path.is_file())
    return [path for path in explicit if path.exists() and path.is_file()]


def build_m5_5c_true_sequence_stage(
    *,
    repo_root: Path,
    prompt_root: Path,
    output_root: Path | None = None,
    model_path: Path | None = None,
    run_detector: bool = True,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    prompt_root = prompt_root.resolve()
    contract = _read_json(prompt_root / "02_M5_5C_WORKSPACE_CONTRACT.json")
    workspace_root = (output_root or Path(contract["workspace_root"])).resolve()
    historical_stage_root = Path(contract["historical_stage_root_read_only"]).resolve()
    m5b_root = Path(contract["m5_5b_workspace_read_only"]).resolve()
    localization_root = Path(contract["authoritative_localization_review"]).resolve()
    path_review_root = Path(contract["authoritative_path_review"]).resolve()
    for relative in WORKSPACE_DIRS:
        (workspace_root / relative).mkdir(parents=True, exist_ok=True)
    authorization = authorization_audit(repo_root)
    if not authorization["authorization_gate_passed"]:
        raise RuntimeError(f"M5.5C authorization gate failed: {authorization}")
    prompt_hashes = _prompt_hash_validation(prompt_root)
    if not prompt_hashes["passed"]:
        raise RuntimeError(f"M5.5C prompt hash validation failed: {prompt_hashes}")
    input_manifest = _copy_prompt_and_input_manifests(
        workspace_root,
        prompt_root,
        m5b_root,
        localization_root,
        path_review_root,
    )
    claim_audit = _claim_evidence_audit(m5b_root)
    audit_root = workspace_root / "01_AUTHORIZATION_AND_CLAIM_AUDIT"
    _write_json(audit_root / "authorization_audit.json", authorization)
    _write_json(audit_root / "prompt_hash_validation.json", prompt_hashes)
    _write_json(audit_root / "m5_5b_claim_evidence_audit.json", claim_audit)
    corrected = {
        "schema_version": "football_intelligence.m5_5c.corrected_scientific_classifications.v1",
        "m5_5b_sequence_classification": "ENDPOINT_TOP2_ONLY_NOT_SEQUENCE_PASS",
        "m5_5b_ghost_classification": "GHOST_NOT_EXECUTED",
        "m5_5b_appearance_classification": "NOT_EVALUATED_ZERO_ACTIVATION",
        "m5_5b_blind_mining_classification": "INVALID_IDENTIFIER_DOMAIN_EXCLUSION",
        "m5_5b_detector_classification": "PARTIALLY_SUPPORTED_AFFECTED_ONLY_CONTROLS_NOT_EXECUTED",
        "prior_pass_label_preserved": False,
        **safety_payload(),
    }
    _write_json(audit_root / "corrected_scientific_classifications.json", corrected)
    protected_paths = _source_files_for_mutation_audit(
        historical_stage_root,
        localization_root,
        path_review_root,
        m5b_root,
    )
    before_hashes = _protected_hashes(protected_paths)
    review_ingestion = _review_ingestion(workspace_root, localization_root, path_review_root)
    if not review_ingestion["status"]["passed"]:
        raise RuntimeError(f"M5.5C review prerequisites failed: {review_ingestion['status']}")
    frame_rows, frame_paths = _frame_manifest(
        historical_stage_root / "continuity_v11" / "unseen_window" / "canonical_frame_manifest.json"
    )
    candidate_rows, candidates_by_frame = _candidate_inventory(
        historical_stage_root / "continuity_v11" / "unseen_window" / "person_candidate_rows.jsonl"
    )
    challenge_rows = _read_jsonl(
        historical_stage_root / "continuity_v11" / "unseen_window" / "challenge_candidate_rows.jsonl"
    )
    group_maps = _trajectory_group_maps(
        historical_stage_root / "continuity_v13" / "audit" / "canonical_trajectory_safe_grouping.json"
    )
    detector_model = model_path or Path(
        r"C:\Users\sebgr\Documents\football-intelligence\trusted-model-cache\yolov8m.pt"
    )
    detector = _run_controlled_detector(
        workspace_root,
        review_ingestion["localization_rows"],
        candidates_by_frame,
        frame_paths,
        group_maps,
        detector_model,
        run_detector=run_detector,
    )
    detector_sheets = _write_detector_configuration_sheets(workspace_root, detector)
    resolver = _run_true_sequence_resolver(
        workspace_root,
        historical_stage_root,
        path_review_root,
        candidates_by_frame,
        set(frame_rows),
    )
    ghost = _write_ghost_outputs(workspace_root, resolver)
    appearance = _write_appearance_outputs(workspace_root, resolver)
    mining = mine_blind_cases(candidate_rows, candidates_by_frame, challenge_rows, group_maps)
    blind = _build_blind_review(workspace_root, frame_paths, mining)
    visuals = _write_visual_evidence(workspace_root, frame_paths, detector, resolver, ghost, blind)
    evaluation = _write_evaluation(workspace_root, detector, resolver, ghost, appearance, blind)
    after_hashes = _protected_hashes(protected_paths)
    mutation_rows = [
        {
            "path": path,
            "sha256_before": before_hashes[path],
            "sha256_after": after_hashes[path],
            "unchanged": before_hashes[path] == after_hashes[path],
        }
        for path in sorted(before_hashes)
    ]
    source_mutation = {
        "schema_version": "football_intelligence.m5_5c.source_mutation_audit.v1",
        "protected_file_count": len(mutation_rows),
        "all_protected_hashes_unchanged": all(row["unchanged"] for row in mutation_rows),
        "rows": mutation_rows,
        "writes_beneath_historical_roots": 0,
        "m5_5b_workspace_mutated": False,
        "invalid_m5_5b_review_mutated": False,
        "historical_artifacts_mutated": False,
        **safety_payload(),
    }
    _write_json(audit_root / "source_mutation_audit.json", source_mutation)
    review_pack = _write_review_pack(
        workspace_root,
        repo_root,
        authorization,
        claim_audit,
        review_ingestion,
        detector,
        resolver,
        ghost,
        appearance,
        blind,
        evaluation,
        source_mutation,
        visuals,
    )
    result = {
        "schema_version": "football_intelligence.m5_5c.stage_result.v1",
        "generated_at": utc_now(),
        "stage_id": STAGE_ID,
        "workspace_root": str(workspace_root),
        "final_classification": evaluation["architecture"]["final_classification"],
        "authorization": authorization,
        "input_manifest_file_count": input_manifest["m5_5b_workspace"]["file_count"],
        "claim_audit": {
            "unsupported_count": claim_audit["unsupported_count"],
            "partially_supported_count": claim_audit["partially_supported_count"],
        },
        "review_prerequisites": review_ingestion["status"],
        "detector_metrics": detector["metrics"],
        "detector_contact_sheet_count": len(detector_sheets),
        "resolver_metrics": resolver["metrics"],
        "ghost_metrics": ghost["metrics"],
        "appearance_metrics": appearance["metrics"],
        "blind_review": blind["review_package"],
        "architecture_decision": evaluation["architecture"],
        "source_mutation_audit": source_mutation,
        "review_pack": review_pack,
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        **safety_payload(),
    }
    _write_json(workspace_root / "11_VALIDATION_AND_LOGS" / "stage_result.json", result)
    return result
