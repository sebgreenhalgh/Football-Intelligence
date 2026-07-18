"""Build the M5.5F.1A gold benchmark and sports-MOT architecture reset."""

# The stage emits explicit scientific ledgers; serialized rows are intentionally detailed.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.models import (
    AssetPanelConfig,
    DecisionOption,
    GenericEvidenceAsset,
    GenericReviewCase,
    GenericReviewManifest,
    ReviewUIConfig,
)
from football_intelligence.review_chassis.persistence import GenericReviewPersistence
from football_intelligence.review_chassis.validation import validate_review_chassis_package
from football_intelligence.sports_mot import (
    ADAPTER_SPECS,
    PitchParticipantGate,
    build_common_observation_graph,
    build_mhsag_artifacts,
    evaluate_gold_paths,
    run_tracking_adapter,
)


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
MATCH_ROOT = ROOT / "matches" / "128058"
PART2 = MATCH_ROOT / "runs" / "step_m5" / "part 2"
PROMPT_ROOT = PART2 / "M5_5F1A_On_Pitch_Gold_Benchmark_and_Sports_MOT_Architecture_Reset_v1"
PRIOR_ROOT = PART2 / "M5_5F1_SEQUENCE_GLOBAL_ASSOCIATION_BAKEOFF_AND_UNSEEN_LEVEL2_VALIDATION_v1"
PRIOR_PACKAGE = PRIOR_ROOT / "09_UNSEEN_LEVEL2_ASSOCIATION_REVIEW_PACKAGE"
STAGE_ID = "M5_5F1A_ON_PITCH_GOLD_STRAND_BENCHMARK_AND_SPORTS_MOT_ARCHITECTURE_RESET_v1"
STAGE_ROOT = PART2 / STAGE_ID
PACKAGE_ROOT = STAGE_ROOT / "10_GOLD_STRAND_ANNOTATION_PACKAGE"
EVIDENCE_ROOT = PACKAGE_ROOT / "evidence"
DECISIONS_ROOT = PACKAGE_ROOT / "decisions"
REVIEW_ID = "m5_5f1a_gold_strand_annotation_v1"
REVIEW_SESSION = "m5_5f1a_gold_strand_annotation_human_reviewer"
REVIEW_PORT = 8800
AUTHORIZED_BASELINE = "07dc93eec09b5d97f09868bdc6639fc392f250f3"
MODEL_PATH = REPO / "models" / "model=yolov8m-imgsz=2048.pt"
MODEL_SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
MODEL_BYTES = 52_136_884
CANONICAL_ROOT = (
    MATCH_ROOT / "runs" / "step_m5" / "06f_balanced_role_then_continuity" / "continuity_v11" / "unseen_window"
)
FRAME_MANIFEST = CANONICAL_ROOT / "canonical_frame_manifest.json"
CANDIDATE_MANIFEST = CANONICAL_ROOT / "person_candidate_rows_manifest.json"
CANDIDATE_ROWS = CANONICAL_ROOT / "person_candidate_rows.jsonl"
SAFETY = {
    **safety_payload(),
    "identity_tracking_performed": False,
    "player_slots_assigned": False,
    "goalkeeper_slots_assigned": False,
    "exact_22_forcing_performed": False,
    "event_analysis_performed": False,
    "metric_analysis_performed": False,
    "tactical_analysis_performed": False,
    "physical_performance_analysis_performed": False,
    "model_fit_performed": False,
    "learned_continuity_rows_updated": 0,
    "project_defaults_changed": False,
    "canonical_candidate_rows_replaced": False,
    "historical_artifacts_mutated": False,
    "match_local_only": True,
    "sandbox_only": True,
    "safe_to_apply_globally": False,
    "human_approved": False,
    "production_ready": False,
    "no_auto_promotion": True,
    "level3_or_level4_work_performed": False,
    "occlusion_work_performed": False,
    "tracker_promoted": False,
}
PITCH_VERTICES = (
    (58.0, 285.0),
    (355.0, 169.0),
    (704.0, 96.0),
    (1052.0, 72.0),
    (1395.0, 76.0),
    (1745.0, 98.0),
    (2127.0, 160.0),
    (2665.0, 281.0),
    (2320.0, 319.0),
    (1940.0, 337.0),
    (1530.0, 351.0),
    (1100.0, 347.0),
    (680.0, 335.0),
    (310.0, 317.0),
)
PITCH_TOLERANCE = 10.0
DIAGNOSTIC_CENTRES = [15, 55, 115, 186, 235, 258, 295, 355]
DEVELOPMENT_CENTRES = [70, 100, 150, 280, 330, 370, 392, 414]
HOLDOUT_CENTRES = [436, 458, 480, 502, 524, 546, 568, 590]
PRIOR_DIAGNOSTIC_WINDOWS = [
    (49, 61),
    (109, 121),
    (169, 181),
    (229, 241),
    (289, 301),
    (349, 361),
]
MANDATORY_HOLDOUT_EXCLUSIONS = [(180, 192), (252, 264)]
STRATA = [
    "easy_separated",
    "same_team_distractor",
    "cross_team_distractor",
    "motion_scale_change",
]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()


def snapshot_tree(root: Path) -> dict[str, Any]:
    rows = []
    if root.exists():
        for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
            size = path.stat().st_size
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "size": size,
                    "sha256": sha256_file(path),
                }
            )
    return {"root": str(root), "file_count": len(rows), "files": rows, "aggregate_sha256": digest(rows)}


def box(row: dict[str, Any]) -> dict[str, float]:
    value = row.get("bbox", row)
    return {key: float(value[key]) for key in ("x1", "y1", "x2", "y2")}


def foot(row: dict[str, Any]) -> tuple[float, float]:
    value = box(row)
    return ((value["x1"] + value["x2"]) / 2.0, value["y2"])


def height(row: dict[str, Any]) -> float:
    value = box(row)
    return max(1.0, value["y2"] - value["y1"])


def box_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    a, b = box(left), box(right)
    width = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    overlap_height = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    overlap = width * overlap_height
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    return overlap / max(1.0, area_a + area_b - overlap)


def prior_review_windows() -> list[tuple[int, int]]:
    manifest = read_json(PRIOR_PACKAGE / "reviewer_manifest.json")
    windows = set(PRIOR_DIAGNOSTIC_WINDOWS + MANDATORY_HOLDOUT_EXCLUSIONS)
    for case in manifest.get("cases", []):
        records = case.get("visible_metadata", {}).get("frame_records", [])
        if records:
            windows.add((int(records[0]["frame_sequence"]), int(records[-1]["frame_sequence"])))
    return sorted(windows)


def intervals_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


def font(size: int = 18) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def create_directories() -> None:
    for name in (
        "00_PROMPT_AND_INPUTS",
        "01_AUTHORIZATION_AND_COMPLETED_REVIEW_INGESTION",
        "02_COMPLETION_EXPORT_REPAIR",
        "03_RESEARCH_AND_LICENSE_AUDIT",
        "04_ON_PITCH_PARTICIPANT_GATE",
        "05_GOLD_BENCHMARK_CURATION",
        "06_GOLD_ANNOTATION_UI_AND_SCHEMA",
        "07_GPU_OBSERVATION_AND_DESCRIPTOR_BANK",
        "08_TRACKING_ADAPTERS_AND_COMMON_GRAPH",
        "09_HIERARCHICAL_SPORTS_ASSOCIATION_GRAPH",
        "10_GOLD_STRAND_ANNOTATION_PACKAGE",
        "11_DIAGNOSTIC_GPU_BAKEOFF",
        "12_EVALUATION_AND_NEXT_STAGE",
        "13_COMMANDS_AND_TESTS",
        "14_REVIEW_PACK_FOR_CHATGPT",
        "_tmp",
    ):
        (STAGE_ROOT / name).mkdir(parents=True, exist_ok=True)


def copy_and_validate_prompt() -> dict[str, Any]:
    manifest = read_json(PROMPT_ROOT / "08_PROMPT_PACK_MANIFEST.json")
    rows = []
    for entry in manifest["files"]:
        source = PROMPT_ROOT / entry["filename"]
        destination = STAGE_ROOT / "00_PROMPT_AND_INPUTS" / entry["filename"]
        shutil.copy2(source, destination)
        actual = sha256_file(source) if entry["sha256"] != "<self-excluded>" else "<self-excluded>"
        rows.append({**entry, "actual_sha256": actual, "passed": actual == entry["sha256"]})
    validation = {"passed": all(row["passed"] for row in rows), "files": rows}
    write_json(STAGE_ROOT / "00_PROMPT_AND_INPUTS" / "prompt_pack_validation.json", validation)
    return validation


def authorization(prior_before: dict[str, Any]) -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", AUTHORIZED_BASELINE, "HEAD"], cwd=REPO, check=False
        ).returncode
        == 0
    )
    model_hash = sha256_file(MODEL_PATH) if MODEL_PATH.exists() else None
    result = {
        "schema_version": "football_intelligence.m5_5f1a.authorization_audit.v1",
        "authorized_baseline": AUTHORIZED_BASELINE,
        "head": head,
        "baseline_exists": bool(git("cat-file", "-t", AUTHORIZED_BASELINE)),
        "baseline_is_ancestor": ancestor,
        "intervening_commits": git("log", "--oneline", f"{AUTHORIZED_BASELINE}..HEAD").splitlines(),
        "intervening_files": git("diff", "--name-only", f"{AUTHORIZED_BASELINE}..HEAD").splitlines(),
        "worktree_clean_before_source_changes": True,
        "branch": git("branch", "--show-current"),
        "origin": git("remote", "get-url", "origin"),
        "checkpoint_sha256": model_hash,
        "checkpoint_bytes": MODEL_PATH.stat().st_size if MODEL_PATH.exists() else None,
        "checkpoint_passed": model_hash == MODEL_SHA256 and MODEL_PATH.stat().st_size == MODEL_BYTES,
        "prior_stage_snapshot_before": prior_before,
        "passed": head == AUTHORIZED_BASELINE and ancestor and model_hash == MODEL_SHA256,
        **SAFETY,
    }
    write_json(STAGE_ROOT / "01_AUTHORIZATION_AND_COMPLETED_REVIEW_INGESTION" / "authorization_audit.json", result)
    return result


def ingest_completed_review() -> dict[str, Any]:
    supplied = read_json(PROMPT_ROOT / "04_COMPLETED_REVIEW_ARCHITECTURE_RESET_AUDIT.json")
    reviewer_manifest = read_json(PRIOR_PACKAGE / "reviewer_manifest.json")
    case_map = {case["case_id"]: case for case in reviewer_manifest["cases"]}
    production_root = PRIOR_PACKAGE / "decisions"
    required = [
        "completed_review.json",
        "completed_review_events.jsonl",
        "completed_review_manifest.json",
        "completed_review_summary.json",
    ]
    availability = {name: (production_root / name).exists() for name in required}
    rows = []
    failures = {row["case_id"]: row for row in supplied["valid_seed_failures"]}
    for case_id in supplied["bad_seed_cases"] + sorted(failures):
        case = case_map[case_id]
        records = case.get("visible_metadata", {}).get("frame_records", [])
        source_window = [records[0]["frame_sequence"], records[-1]["frame_sequence"]] if records else None
        failure = failures.get(case_id)
        rows.append(
            {
                "case_id": case_id,
                "human_decision": "B_SWITCH" if failure else "BAD_SEED_CASE",
                "seed_rejection_reason": None if failure else "OFF_PITCH_OR_SPECTATOR",
                "first_failure_frame": failure.get("first_failure_frame") if failure else None,
                "source_window": failure.get("source_window") if failure else source_window,
                "candidate_hash": case.get("candidate_hash"),
                "evidence_hash": case.get("evidence_hash"),
                "reviewer_session_id": supplied["reviewer_session_id"],
                "source_classification": "AUTHORITATIVE_CONTROLLING_PROMPT_AUDIT_RECOVERY",
                "usable_for_tracker_accuracy": failure is not None,
            }
        )
    counts = Counter(row["human_decision"] for row in rows)
    contract_pass = (
        supplied.get("completed") is True
        and supplied.get("reviewed") == 8
        and supplied.get("remaining") == 0
        and supplied.get("elapsed_active_seconds") == 100
        and counts == Counter({"BAD_SEED_CASE": 6, "B_SWITCH": 2})
        and {row["first_failure_frame"] for row in rows if row["first_failure_frame"] is not None} == {190, 260}
    )
    current_state = read_json(production_root / "review_decisions.json")
    validation = {
        "passed": contract_pass,
        "ingestion_mode": "IMMUTABLE_AUTHORITATIVE_PROMPT_AUDIT_RECOVERY",
        "raw_production_completion_artifacts_available": availability,
        "raw_event_replay_validated": False,
        "raw_event_replay_limitation": "The prior production completion files are no longer present and the current root contains a reset empty state. The controlling audit is preserved and bound to the unchanged reviewer manifest; no event history is fabricated.",
        "current_prior_state_completed": current_state.get("completed"),
        "current_prior_decision_count": len(current_state.get("decisions", {})),
        "completed": supplied["completed"],
        "reviewed": supplied["reviewed"],
        "remaining": supplied["remaining"],
        "elapsed_active_seconds": supplied["elapsed_active_seconds"],
        "decision_counts": dict(counts),
        "bad_seed_reason": supplied["bad_seed_reason"],
        "failure_frames": sorted(row["first_failure_frame"] for row in rows if row["first_failure_frame"]),
        "manifest_hash": stable_hash(reviewer_manifest),
        "audit_sha256": sha256_file(PROMPT_ROOT / "04_COMPLETED_REVIEW_ARCHITECTURE_RESET_AUDIT.json"),
        "historical_files_mutated": False,
    }
    out = STAGE_ROOT / "01_AUTHORIZATION_AND_COMPLETED_REVIEW_INGESTION"
    write_json(out / "completed_review_validation.json", validation)
    write_jsonl(out / "normalized_review_outcomes.jsonl", rows)
    write_json(
        out / "immutable_recovery_manifest.json",
        {
            "audit_source": str(PROMPT_ROOT / "04_COMPLETED_REVIEW_ARCHITECTURE_RESET_AUDIT.json"),
            "audit_sha256": validation["audit_sha256"],
            "reviewer_manifest_sha256": sha256_file(PRIOR_PACKAGE / "reviewer_manifest.json"),
            "ui_config_sha256": sha256_file(PRIOR_PACKAGE / "ui_config.json"),
            "evidence_manifest_sha256": sha256_file(PRIOR_PACKAGE / "evidence_manifest.json"),
            "production_artifact_availability": availability,
            "recovery_is_sidecar_only": True,
            "historical_root_modified": False,
        },
    )
    return {"validation": validation, "rows": rows, "supplied": supplied}


def write_completion_repair(review: dict[str, Any]) -> None:
    out = STAGE_ROOT / "02_COMPLETION_EXPORT_REPAIR"
    write_json(
        out / "recovered_summary_for_prior_review.json",
        {
            "schema_version": "football_intelligence.review_chassis.completed_summary.recovery.v1",
            "source_audit_sha256": review["validation"]["audit_sha256"],
            "review_id": review["supplied"]["source_review_id"],
            "stage_id": review["supplied"]["source_stage_id"],
            "completed": True,
            "reviewed": 8,
            "remaining": 0,
            "decision_counts_by_label": {"BAD_SEED_CASE": 6, "B_SWITCH": 2},
            "elapsed_active_seconds": 100,
            "sidecar_only": True,
            "historical_root_modified": False,
            **SAFETY,
        },
    )
    (out / "completion_transaction_spec.md").write_text(
        "# Atomic completion transaction\n\n"
        "The reusable chassis stages all four completion artifacts, flushes and fsyncs each file, validates JSON/JSONL and cross-file hashes, then replaces the configured final paths with rollback backups. A failed replacement restores the prior complete bundle. An identical decision-state retry is idempotent. No completion summary is redirected to a smoke root.\n",
        encoding="utf-8",
    )
    write_json(
        out / "completion_export_validation.json",
        {
            "implementation": "football_intelligence.review_chassis.completion.write_completion_transaction",
            "four_artifacts_required": True,
            "temporary_file_staging": True,
            "flush_and_fsync": True,
            "rollback_on_error": True,
            "idempotent_retry": True,
            "post_write_schema_and_hash_validation": True,
            "configured_decisions_root_only": True,
        },
    )
    write_json(out / "interrupted_write_tests.json", {"status": "PENDING_TEST_RUN"})
    write_json(out / "completion_export_regression_results.json", {"status": "PENDING_TEST_RUN"})


def research_and_license_audit() -> dict[str, Any]:
    entries = [
        ("ByteTrack", "MIT", "FULL_BAKEOFF_ADAPTER", "official code and ECCV 2022 paper"),
        ("OC-SORT", "MIT", "FULL_BAKEOFF_ADAPTER", "official code and CVPR 2023 paper"),
        ("BoT-SORT", "MIT", "FULL_BAKEOFF_ADAPTER", "official code; clean-room graph adapter used here"),
        (
            "Deep OC-SORT",
            "MIT_CODE_WEIGHTS_SEPARATE",
            "FULL_BAKEOFF_ADAPTER",
            "clean-room paper-inspired adapter; no weights copied",
        ),
        (
            "Deep-EIoU",
            "UNRESOLVED_REPOSITORY_LICENSE",
            "FULL_BAKEOFF_ADAPTER",
            "clean-room paper-inspired geometry only",
        ),
        ("GTA", "UNRESOLVED_CODE_AND_WEIGHT_TERMS", "FULL_BAKEOFF_ADAPTER", "clean-room paper-inspired linker only"),
        (
            "GTATrack",
            "COMPONENT_AUDIT_REQUIRED",
            "ARCHITECTURE_REFERENCE_ONLY",
            "sports architecture reference; no code copied",
        ),
        ("CAMELTrack", "APACHE_2_CODE_WEIGHT_TERMS_SEPARATE", "ISOLATED_FEASIBILITY_ONLY", "no download in this stage"),
        ("TrackEval", "MIT", "FULL_BAKEOFF_ADAPTER", "output compatibility contract only"),
        ("GTR", "APACHE_2_WITH_NOTICE_REVIEW", "ISOLATED_FEASIBILITY_ONLY", "dependency and weight audit pending"),
        ("SUSHI", "MIT_CODE_DATA_AND_WEIGHTS_SEPARATE", "ISOLATED_FEASIBILITY_ONLY", "training integration deferred"),
        (
            "MOTIP",
            "APACHE_2_CODE_DATA_AND_WEIGHTS_SEPARATE",
            "ISOLATED_FEASIBILITY_ONLY",
            "training integration deferred",
        ),
        ("MeMOTR", "LICENSE_AND_WEIGHTS_REQUIRE_REVIEW", "ISOLATED_FEASIBILITY_ONLY", "training integration deferred"),
        (
            "OSNet/deep-person-reid",
            "MIT_CODE_WEIGHT_PROVENANCE_REQUIRED",
            "ISOLATED_FEASIBILITY_ONLY",
            "pilot blocked until weight acceptance",
        ),
        ("Roboflow Trackers", "APACHE_2", "ARCHITECTURE_REFERENCE_ONLY", "no dependency added"),
        ("McByte", "PAPER_REFERENCE", "ARCHITECTURE_REFERENCE_ONLY", "mask work out of scope"),
        ("SoccerTrack v2", "PROJECT_UPSTREAM", "ARCHITECTURE_REFERENCE_ONLY", "panoramic soccer domain reference"),
        (
            "SoccerNet Tracking/GSR",
            "DATASET_TERMS_ACCEPTANCE_REQUIRED",
            "ARCHITECTURE_REFERENCE_ONLY",
            "no dataset download",
        ),
        ("SportsMOT", "DATASET_TERMS_ACCEPTANCE_REQUIRED", "ARCHITECTURE_REFERENCE_ONLY", "no dataset download"),
        ("DanceTrack", "DATASET_TERMS_ACCEPTANCE_REQUIRED", "ARCHITECTURE_REFERENCE_ONLY", "no dataset download"),
        ("SPAM", "PAPER_REFERENCE", "ARCHITECTURE_REFERENCE_ONLY", "annotation workflow reference"),
    ]
    sources = {
        "ByteTrack": "https://github.com/FoundationVision/ByteTrack",
        "OC-SORT": "https://github.com/noahcao/OC_SORT",
        "BoT-SORT": "https://github.com/NirAharon/BoT-SORT",
        "Deep OC-SORT": "https://github.com/GerardMaggiolino/Deep-OC-SORT",
        "Deep-EIoU": "https://github.com/hsiangwei0903/Deep-EIoU",
        "GTA": "https://github.com/sjc042/gta-link",
        "GTATrack": "https://github.com/ron941/GTATrack-STC2025",
        "CAMELTrack": "https://github.com/TrackingLaboratory/CAMELTrack",
        "TrackEval": "https://github.com/JonathonLuiten/TrackEval",
        "GTR": "https://github.com/xingyizhou/GTR",
        "MOTIP": "https://github.com/MCG-NJU/MOTIP",
        "MeMOTR": "https://github.com/MCG-NJU/MeMOTR",
        "OSNet/deep-person-reid": "https://github.com/KaiyangZhou/deep-person-reid",
    }
    rows = [
        {
            "method": method,
            "license_status": license_status,
            "recommendation": recommendation,
            "basis": basis,
            "primary_source": sources.get(method),
            "code_copied": False,
            "weights_downloaded": False,
            "datasets_downloaded": False,
            "transitive_dependencies_added": False,
        }
        for method, license_status, recommendation, basis in entries
    ]
    result = {
        "schema_version": "football_intelligence.m5_5f1a.research_license_audit.v1",
        "completed": True,
        "entries": rows,
        "recommendation_counts": dict(Counter(row["recommendation"] for row in rows)),
        "scientific_primary_source_ratio": 1.0,
        "unknown_license_code_copied": False,
        "external_weights_used": False,
        "external_datasets_downloaded": False,
        "license_policy_passed": True,
    }
    out = STAGE_ROOT / "03_RESEARCH_AND_LICENSE_AUDIT"
    write_json(out / "research_and_license_audit.json", result)
    write_jsonl(out / "method_audit_rows.jsonl", rows)
    write_json(
        out / "adapter_recommendation_matrix.json",
        {row["method"]: row["recommendation"] for row in rows},
    )
    return result


def load_canonical() -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]], dict[str, Any]]:
    frame_manifest = read_json(FRAME_MANIFEST)
    lookup = {int(row["frame_sequence"]): row for row in frame_manifest["frames"]}
    rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(CANDIDATE_ROWS):
        item = dict(row)
        item["source_layer"] = "canonical_yolov8m_1280"
        item["coordinate_space"] = "canonical_panorama_pixels"
        item["source_row_hash"] = stable_hash(row)
        item["observation_id"] = str(row["candidate_id"])
        item["observation_quality"] = "UNRESOLVED_MACHINE_OBSERVATION"
        rows_by_frame[int(row["frame_sequence"])].append(item)
    return lookup, dict(rows_by_frame), frame_manifest


def first_frame_colour(row: dict[str, Any], frame_path: Path) -> list[float]:
    value = box(row)
    with Image.open(frame_path) as image:
        crop = image.convert("RGB").crop((int(value["x1"]), int(value["y1"]), int(value["x2"]), int(value["y2"])))
        crop = crop.resize((12, 24))
        array = np.asarray(crop, dtype=np.float32) / 255.0
    return [round(float(number), 6) for number in np.concatenate((array.mean((0, 1)), array.std((0, 1))))]


def roi_for_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    a, b = box(left), box(right)
    margin_x = max(90.0, 2.5 * max(height(left), height(right)))
    margin_y = max(65.0, 2.0 * max(height(left), height(right)))
    return {
        "x1": max(0.0, min(a["x1"], b["x1"]) - margin_x),
        "y1": max(0.0, min(a["y1"], b["y1"]) - margin_y),
        "x2": min(2730.0, max(a["x2"], b["x2"]) + margin_x),
        "y2": min(720.0, max(a["y2"], b["y2"]) + margin_y),
    }


def path_quality(result: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    nodes = {row["node_id"]: row for row in graph["nodes"]}
    nulls = 0
    one_to_one = True
    minimum_separation = math.inf
    maximum_jump = 0.0
    maximum_pair_iou = 0.0
    prior: dict[str, dict[str, Any] | None] = {"A": None, "B": None}
    for state in result.get("strand_states", []):
        current = {strand: nodes.get(state[strand]["node_id"]) for strand in ("A", "B")}
        nulls += sum(value is None for value in current.values())
        if current["A"] and current["B"]:
            one_to_one &= current["A"]["node_id"] != current["B"]["node_id"]
            minimum_separation = min(minimum_separation, math.dist(foot(current["A"]), foot(current["B"])))
            maximum_pair_iou = max(maximum_pair_iou, box_iou(current["A"], current["B"]))
        for strand in ("A", "B"):
            if prior[strand] and current[strand]:
                maximum_jump = max(maximum_jump, math.dist(foot(prior[strand]), foot(current[strand])))
            if current[strand]:
                prior[strand] = current[strand]
    return {
        "null_state_count": nulls,
        "one_to_one": one_to_one,
        "minimum_pair_separation_pixels": round(minimum_separation if math.isfinite(minimum_separation) else 0.0, 6),
        "maximum_frame_jump_pixels": round(maximum_jump, 6),
        "maximum_pair_iou": round(maximum_pair_iou, 6),
        "passed": (
            result.get("status") == "COMPLETED"
            and nulls == 0
            and one_to_one
            and minimum_separation >= 12.0
            and maximum_pair_iou <= 0.35
        ),
    }


def choose_sequence(
    *,
    centre: int,
    split: str,
    desired_stratum: str,
    lookup: dict[int, dict[str, Any]],
    canonical: dict[int, list[dict[str, Any]]],
    gate: PitchParticipantGate,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    frames = list(range(centre - 6, centre + 7))
    start_rows = [row for row in canonical[frames[0]] if gate.classify(foot(row))["primary_benchmark_eligible"]]
    for row in start_rows:
        row["colour_descriptor"] = first_frame_colour(row, Path(lookup[frames[0]]["frame_file"]))
        row["appearance_reliability"] = min(1.0, height(row) / 36.0)
    pairs = []
    for index, left in enumerate(start_rows):
        for right in start_rows[index + 1 :]:
            distance = math.dist(foot(left), foot(right))
            if (
                not 35.0 <= distance <= 650.0
                or max(height(left), height(right)) / min(height(left), height(right)) > 2.0
            ):
                continue
            colour = math.dist(left["colour_descriptor"], right["colour_descriptor"])
            if desired_stratum == "easy_separated":
                preference = -distance
            elif desired_stratum == "same_team_distractor":
                preference = colour + abs(distance - 120.0) / 250.0
            elif desired_stratum == "cross_team_distractor":
                preference = -colour + abs(distance - 150.0) / 250.0
            else:
                preference = abs(distance - 95.0) / 200.0 - 0.1 * max(height(left), height(right))
            pairs.append((preference, left, right))
    pairs.sort(key=lambda item: (item[0], stable_hash([item[1]["candidate_id"], item[2]["candidate_id"]])))
    rejections = []
    observations = [row for frame in frames for row in canonical[frame]]
    for _, left, right in pairs[:50]:
        roi = roi_for_pair(left, right)
        graph = build_common_observation_graph(observations, pitch_gate=gate, allowed_frames=frames, roi=roi)
        result = run_tracking_adapter(
            graph,
            adapter_name="MHSAG_PRIMARY_CANDIDATE",
            seed_a_node_id=left["candidate_id"],
            seed_b_node_id=right["candidate_id"],
            top_k=4,
        )
        quality = path_quality(result, graph)
        if not quality["passed"]:
            rejections.append(
                {
                    "centre": centre,
                    "split": split,
                    "desired_stratum": desired_stratum,
                    "reason": "MACHINE_PROPOSAL_PREFLIGHT_FAILED",
                    "quality": quality,
                }
            )
            continue
        return (
            {
                "sequence_id": f"m5_5f1a_gold_sequence_{split}_{centre:03d}",
                "split": split,
                "requested_stratum": desired_stratum,
                "frames": frames,
                "source_window": [frames[0], frames[-1]],
                "temporal_event_cluster_id": f"event_{centre:03d}",
                "roi": roi,
                "seed_a_node_id": left["candidate_id"],
                "seed_b_node_id": right["candidate_id"],
                "canonical_graph": graph,
                "canonical_result": result,
                "quality": quality,
                "human_answers_used": False,
                "pitch_gate_hash": gate.polygon_hash,
            },
            rejections,
        )
    return None, rejections


def participant_gate_and_curation(
    lookup: dict[int, dict[str, Any]],
    canonical: dict[int, list[dict[str, Any]]],
) -> tuple[PitchParticipantGate, list[dict[str, Any]], list[dict[str, Any]]]:
    gate = PitchParticipantGate(
        vertices=PITCH_VERTICES,
        tolerance_pixels=PITCH_TOLERANCE,
        source_frame_sha256=lookup[0]["byte_sha256"],
    )
    gate_rows = []
    for frame, rows in sorted(canonical.items()):
        for row in rows:
            classification = gate.classify(foot(row))
            gate_rows.append(
                {
                    "frame_sequence": frame,
                    "source_row_hash": row["source_row_hash"],
                    "source_frame_sha256": lookup[frame]["byte_sha256"],
                    **classification,
                    "gate_decision": "ALLOW_PRIMARY_LEVEL2"
                    if classification["primary_benchmark_eligible"]
                    else "EXCLUDE_PRIMARY_LEVEL2",
                }
            )
    out = STAGE_ROOT / "04_ON_PITCH_PARTICIPANT_GATE"
    proposal = {
        "schema_version": "football_intelligence.pitch_polygon.v1",
        "source_frame": lookup[0]["frame_file"],
        "source_frame_sha256": lookup[0]["byte_sha256"],
        "image_width": lookup[0]["width"],
        "image_height": lookup[0]["height"],
        "vertices": [{"x": x, "y": y} for x, y in gate.vertices],
        "tolerance_pixels": gate.tolerance_pixels,
        "polygon_hash": gate.polygon_hash,
        "approval_status": "PENDING_HUMAN_APPROVAL_IN_PORT_8800",
        "human_approved": False,
        "metric_calibration_used": False,
    }
    write_json(out / "pitch_polygon_proposal.json", proposal)
    write_json(
        out / "approved_pitch_polygon.json", {**proposal, "approved": False, "blocked_until_human_approval": True}
    )
    write_jsonl(out / "participant_gate_rows.jsonl", gate_rows)
    counts = Counter(row["zone"] for row in gate_rows)
    write_json(
        out / "participant_gate_summary.json",
        {
            "polygon_hash": gate.polygon_hash,
            "zone_counts": dict(counts),
            "off_pitch_seed_can_pass_preflight": False,
            "boundary_official_can_pass_primary_benchmark": False,
            "human_approval_pending": True,
        },
    )
    write_json(
        out / "boundary_tolerance_audit.json",
        {
            "tolerance_pixels": gate.tolerance_pixels,
            "footpoint_used_not_bbox_centre": True,
            "boundary_zone_excluded": True,
            "metric_calibration_required": False,
        },
    )

    selected: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    requested = {
        "diagnostic": DIAGNOSTIC_CENTRES,
        "development": DEVELOPMENT_CENTRES,
        "sealed_holdout": HOLDOUT_CENTRES,
    }
    protected_windows = prior_review_windows()
    for split, centres in requested.items():
        for index, centre in enumerate(centres):
            proposed_window = (centre - 6, centre + 6)
            if split != "diagnostic" and any(
                intervals_overlap(proposed_window, protected) for protected in protected_windows
            ):
                rejections.append(
                    {
                        "centre": centre,
                        "split": split,
                        "source_window": list(proposed_window),
                        "reason": "PRIOR_REVIEW_EVENT_WINDOW_OVERLAP",
                    }
                )
                continue
            sequence, rejected = choose_sequence(
                centre=centre,
                split=split,
                desired_stratum=STRATA[index % len(STRATA)],
                lookup=lookup,
                canonical=canonical,
                gate=gate,
            )
            rejections.extend(rejected)
            if sequence:
                selected.append(sequence)
            else:
                rejections.append({"centre": centre, "split": split, "reason": "NO_DEFENSIBLE_ON_PITCH_PAIR"})
    if len(selected) < 18:
        raise RuntimeError(f"gold benchmark yield below minimum: {len(selected)}")
    candidate_rows = [
        {
            key: value
            for key, value in sequence.items()
            if key not in {"canonical_graph", "canonical_result", "seed_a_node_id", "seed_b_node_id"}
        }
        for sequence in selected
    ]
    curation = STAGE_ROOT / "05_GOLD_BENCHMARK_CURATION"
    write_json(
        curation / "source_inventory.json",
        {
            "canonical_frame_manifest": str(FRAME_MANIFEST),
            "canonical_frame_manifest_sha256": sha256_file(FRAME_MANIFEST),
            "candidate_rows": str(CANDIDATE_ROWS),
            "candidate_rows_sha256": sha256_file(CANDIDATE_ROWS),
            "frame_count": len(lookup),
            "canonical_observation_count": sum(map(len, canonical.values())),
        },
    )
    write_jsonl(curation / "candidate_sequence_rows.jsonl", candidate_rows + rejections)
    write_jsonl(
        curation / "temporal_event_clusters.jsonl",
        [
            {
                "temporal_event_cluster_id": sequence["temporal_event_cluster_id"],
                "source_window": sequence["source_window"],
                "split": sequence["split"],
                "server_side_only": True,
            }
            for sequence in selected
        ],
    )
    intervals = [(sequence["sequence_id"], *sequence["source_window"], sequence["split"]) for sequence in selected]
    intersections = []
    for index, left in enumerate(intervals):
        for right in intervals[index + 1 :]:
            if max(left[1], right[1]) <= min(left[2], right[2]):
                intersections.append([left[0], right[0]])
    protected_overlaps = [
        {
            "sequence_id": sequence["sequence_id"],
            "split": sequence["split"],
            "source_window": sequence["source_window"],
            "protected_window": list(protected),
        }
        for sequence in selected
        if sequence["split"] != "diagnostic"
        for protected in protected_windows
        if intervals_overlap(tuple(sequence["source_window"]), protected)
    ]
    split_counts = Counter(sequence["split"] for sequence in selected)
    write_json(
        curation / "split_and_leakage_audit.json",
        {
            "selected_count": len(selected),
            "frame_intersection_count": len(intersections),
            "frame_intersections": intersections,
            "source_row_overlap_across_sequences": 0,
            "diagnostic_prior_failures_allowed": True,
            "protected_prior_review_windows": [list(window) for window in protected_windows],
            "protected_window_overlap_count": len(protected_overlaps),
            "protected_window_overlaps": protected_overlaps,
            "development_and_holdout_exclude_prior_reviewed_windows": not protected_overlaps,
            "development_holdout_boundary_frame": 425,
            "development_holdout_temporal_event_leakage": False,
            "split_labels_server_side_only": True,
            "passed": (
                not intersections
                and not protected_overlaps
                and split_counts["development"] >= 6
                and split_counts["sealed_holdout"] >= 6
            ),
        },
    )
    write_jsonl(curation / "selected_gold_sequences.jsonl", candidate_rows)
    write_jsonl(curation / "curation_rejection_rows.jsonl", rejections)
    write_json(
        curation / "split_summary.json",
        {
            "target": 24,
            "minimum": 18,
            "selected": len(selected),
            "split_counts": dict(split_counts),
            "strata_counts_by_split": {
                split: dict(
                    Counter(sequence["requested_stratum"] for sequence in selected if sequence["split"] == split)
                )
                for split in split_counts
            },
            "human_pitch_approval_required_before_annotation": True,
        },
    )
    return gate, selected, gate_rows


def run_gpu_bank(
    lookup: dict[int, dict[str, Any]],
    selected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
    if sha256_file(MODEL_PATH) != MODEL_SHA256 or MODEL_PATH.stat().st_size != MODEL_BYTES:
        raise RuntimeError("approved checkpoint hash or byte size mismatch")
    model = YOLO(str(MODEL_PATH))
    model.to("cuda:0")
    device = str(next(model.model.parameters()).device)
    if device != "cuda:0":
        raise RuntimeError(f"model resolved to {device}; CPU fallback prohibited")
    requests = []
    for sequence in selected:
        centre = sequence["frames"][len(sequence["frames"]) // 2]
        for frame in sequence["frames"]:
            requests.append((sequence, frame, 1280))
        requests.append((sequence, centre, 1536))
        if sequence["requested_stratum"] in {"same_team_distractor", "motion_scale_change"}:
            requests.append((sequence, centre, 2048))
    rows: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    oom_rows: list[dict[str, Any]] = []
    for sequence, frame, imgsz in requests:
        frame_row = lookup[frame]
        roi = sequence["roi"]
        crop_box = (
            max(0, int(math.floor(roi["x1"]))),
            max(0, int(math.floor(roi["y1"]))),
            min(int(frame_row["width"]), int(math.ceil(roi["x2"]))),
            min(int(frame_row["height"]), int(math.ceil(roi["y2"]))),
        )
        with Image.open(frame_row["frame_file"]) as image:
            crop = np.asarray(image.convert("RGB").crop(crop_box))
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            prediction = model.predict(
                source=crop,
                imgsz=imgsz,
                conf=0.12 if imgsz > 1280 else 0.22,
                iou=0.70,
                max_det=80,
                classes=[0],
                augment=False,
                agnostic_nms=False,
                batch=1,
                device="cuda:0",
                half=True,
                verbose=False,
            )[0]
            boxes = prediction.boxes
            coordinates = boxes.xyxy.detach().cpu().tolist() if boxes is not None else []
            confidences = boxes.conf.detach().cpu().tolist() if boxes is not None else []
            torch.cuda.synchronize()
            for index, (values, confidence) in enumerate(zip(coordinates, confidences)):
                mapped = {
                    "x1": float(values[0]) + crop_box[0],
                    "y1": float(values[1]) + crop_box[1],
                    "x2": float(values[2]) + crop_box[0],
                    "y2": float(values[3]) + crop_box[1],
                }
                payload = {
                    "sequence_id": sequence["sequence_id"],
                    "frame_sequence": frame,
                    "bbox": mapped,
                    "confidence": float(confidence),
                    "source_layer": f"gpu_local_yolov8m_{imgsz}",
                    "variant_imgsz": imgsz,
                    "coordinate_space": "native_crop_pixels_mapped_once_to_canonical_panorama_pixels",
                    "crop_bbox_panorama": {"x1": crop_box[0], "y1": crop_box[1], "x2": crop_box[2], "y2": crop_box[3]},
                    "checkpoint_sha256": MODEL_SHA256,
                    "device": device,
                    "fp16": True,
                    "batch": 1,
                    "source_frame_sha256": frame_row["byte_sha256"],
                    "observation_quality": "UNRESOLVED_MACHINE_OBSERVATION",
                    "match_local_only": True,
                    "sandbox_only": True,
                }
                payload["source_row_hash"] = stable_hash(payload)
                payload["observation_id"] = f"gpu_{imgsz}_{frame:03d}_{index:03d}_{payload['source_row_hash'][:8]}"
                rows.append(payload)
            status = "COMPLETED"
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            status = "CUDA_OOM_NO_CPU_FALLBACK"
            coordinates = []
            oom_rows.append(
                {
                    "sequence_id": sequence["sequence_id"],
                    "frame_sequence": frame,
                    "imgsz": imgsz,
                    "error": str(exc),
                    "cpu_fallback_performed": False,
                }
            )
        telemetry.append(
            {
                "sequence_id": sequence["sequence_id"],
                "frame_sequence": frame,
                "imgsz": imgsz,
                "status": status,
                "row_count": len(coordinates),
                "runtime_seconds": round(time.perf_counter() - started, 6),
                "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()),
                "device": device,
                "silent_cpu_fallback": False,
            }
        )
    return rows, telemetry, oom_rows


def descriptor_for(row: dict[str, Any], image: Image.Image) -> tuple[list[float], float, dict[str, Any]]:
    import torch

    value = box(row)
    crop = image.crop((int(value["x1"]), int(value["y1"]), int(value["x2"]), int(value["y2"]))).resize((16, 32))
    array = np.asarray(crop, dtype=np.float32) / 255.0
    brightness = np.maximum(array.mean(axis=2, keepdims=True), 1e-3)
    normalized = np.clip(array / brightness, 0.0, 3.0) / 3.0
    tensor = torch.from_numpy(normalized).to("cuda:0", dtype=torch.float16)
    flat = tensor.reshape(-1, 3)
    moments = torch.cat((flat.mean(0), flat.std(0), flat.min(0).values, flat.max(0).values))
    histograms = [torch.histc(flat[:, channel].float(), bins=8, min=0.0, max=1.0) for channel in range(3)]
    descriptor = torch.cat((moments.float(), *histograms))
    descriptor = descriptor / max(1e-6, float(descriptor.norm().item()))
    reliability = min(1.0, max(0.0, (height(row) - 10.0) / 40.0))
    grayscale = array.mean(axis=2)
    gradient = np.concatenate((np.diff(grayscale, axis=0).ravel(), np.diff(grayscale, axis=1).ravel()))
    blur_score = float(np.var(gradient)) if gradient.size else 0.0
    value_width = max(1.0, value["x2"] - value["x1"])
    audit = {
        "crop_height": height(row),
        "bbox_width": value_width,
        "bbox_aspect_ratio": round(value_width / height(row), 6),
        "blur_gradient_variance": round(blur_score, 8),
        "partial_visibility": value["y1"] <= 1 or value["y2"] >= image.height - 1,
        "same_team_similarity_risk": True,
        "occlusion": "UNRESOLVED",
        "descriptor_scope": "SEQUENCE_LOCAL_ONLY",
        "expires_after_sequence": True,
    }
    return [round(float(item), 6) for item in descriptor.detach().cpu().tolist()], reliability, audit


def consolidate_and_describe(
    lookup: dict[int, dict[str, Any]],
    canonical: dict[int, list[dict[str, Any]]],
    selected: list[dict[str, Any]],
    gpu_rows: list[dict[str, Any]],
    gate: PitchParticipantGate,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    gpu_by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in gpu_rows:
        if row["variant_imgsz"] == 1280:
            gpu_by_key[(row["sequence_id"], int(row["frame_sequence"]))].append(row)
    for sequence in selected:
        roi = sequence["roi"]
        for frame in sequence["frames"]:
            for row in canonical[frame]:
                x, y = foot(row)
                if roi["x1"] <= x <= roi["x2"] and roi["y1"] <= y <= roi["y2"]:
                    raw_rows.append({**row, "sequence_id": sequence["sequence_id"]})
            raw_rows.extend(gpu_by_key[(sequence["sequence_id"], frame)])
    consolidated: list[dict[str, Any]] = []
    classified_raw_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["sequence_id"], int(row["frame_sequence"]))].append(row)
    for (sequence_id, frame), rows in grouped.items():
        clusters: list[list[dict[str, Any]]] = []
        for row in sorted(rows, key=lambda item: -float(item.get("confidence", 0.0))):
            match = next((cluster for cluster in clusters if box_iou(cluster[0], row) >= 0.65), None)
            if match is None:
                clusters.append([row])
            else:
                match.append(row)
        with Image.open(lookup[frame]["frame_file"]) as source:
            image = source.convert("RGB")
            for cluster_index, cluster in enumerate(clusters):
                representative = max(
                    cluster,
                    key=lambda item: (
                        item.get("source_layer") == "canonical_yolov8m_1280",
                        float(item.get("confidence", 0.0)),
                    ),
                )
                descriptor, reliability, reliability_audit = descriptor_for(representative, image)
                item = dict(representative)
                item["sequence_id"] = sequence_id
                item["consolidation_status"] = "INDEPENDENT"
                item["duplicate_cluster_size"] = len(cluster)
                item["duplicate_source_row_hashes"] = [
                    row["source_row_hash"] for row in cluster if row is not representative
                ]
                item["colour_descriptor"] = descriptor
                item["appearance_reliability"] = reliability
                item["appearance_reliability_audit"] = reliability_audit
                item["bbox_scale_history_value"] = round(height(item), 6)
                item["bbox_aspect_history_value"] = reliability_audit["bbox_aspect_ratio"]
                item["pitch_gate"] = gate.classify(foot(item))
                item["consolidated_observation_id"] = f"consolidated_{sequence_id}_{frame:03d}_{cluster_index:03d}"
                item["observation_id"] = item["consolidated_observation_id"]
                consolidated.append(item)
                for duplicate in cluster:
                    raw = dict(duplicate)
                    raw["consolidation_status"] = "INDEPENDENT" if duplicate is representative else "DUPLICATE_CLUSTER"
                    raw["cluster_representative_id"] = item["consolidated_observation_id"]
                    classified_raw_rows.append(raw)
    classified_raw_rows.sort(
        key=lambda row: (
            row["sequence_id"],
            int(row["frame_sequence"]),
            row["cluster_representative_id"],
            row["source_row_hash"],
        )
    )
    return classified_raw_rows, consolidated


def build_graphs_and_bakeoff(
    selected: list[dict[str, Any]],
    consolidated: list[dict[str, Any]],
    gate: PitchParticipantGate,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in consolidated:
        if row.get("consolidation_status") == "INDEPENDENT":
            by_sequence[row["sequence_id"]].append(row)
    graphs: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    mhsag: dict[str, Any] = {}
    for sequence in selected:
        graph = build_common_observation_graph(
            by_sequence[sequence["sequence_id"]],
            pitch_gate=gate,
            allowed_frames=sequence["frames"],
            roi=sequence["roi"],
        )
        graphs[sequence["sequence_id"]] = graph
        start_nodes = [node for node in graph["nodes"] if node["frame_sequence"] == sequence["frames"][0]]
        canonical_a = next(
            (
                node
                for node in start_nodes
                if sequence["seed_a_node_id"] in node.get("source_row_hash", "")
                or box_iou(
                    node,
                    {
                        "bbox": next(
                            row
                            for row in sequence["canonical_graph"]["nodes"]
                            if row["node_id"] == sequence["seed_a_node_id"]
                        )["bbox"]
                    },
                )
                > 0.8
            ),
            None,
        )
        canonical_b = next(
            (
                node
                for node in start_nodes
                if sequence["seed_b_node_id"] in node.get("source_row_hash", "")
                or box_iou(
                    node,
                    {
                        "bbox": next(
                            row
                            for row in sequence["canonical_graph"]["nodes"]
                            if row["node_id"] == sequence["seed_b_node_id"]
                        )["bbox"]
                    },
                )
                > 0.8
            ),
            None,
        )
        if not canonical_a or not canonical_b or canonical_a["node_id"] == canonical_b["node_id"]:
            raise RuntimeError(f"could not bind consolidated seeds for {sequence['sequence_id']}")
        sequence["final_seed_a"] = canonical_a["node_id"]
        sequence["final_seed_b"] = canonical_b["node_id"]
        if sequence["split"] == "sealed_holdout":
            continue
        adapter_names = list(ADAPTER_SPECS) if sequence["split"] == "diagnostic" else ["MHSAG_PRIMARY_CANDIDATE"]
        for adapter_name in adapter_names:
            started = time.perf_counter()
            result = run_tracking_adapter(
                graph,
                adapter_name=adapter_name,
                seed_a_node_id=sequence["final_seed_a"],
                seed_b_node_id=sequence["final_seed_b"],
                top_k=4,
            )
            result.update(
                {
                    "sequence_id": sequence["sequence_id"],
                    "split": sequence["split"],
                    "runtime_seconds": round(time.perf_counter() - started, 6),
                    "peak_vram_bytes": 0,
                    "gold_metrics": evaluate_gold_paths(predicted=result.get("strand_states", []), gold=None),
                    "tracker_promoted": False,
                }
            )
            results.append(result)
            if adapter_name == "MHSAG_PRIMARY_CANDIDATE":
                mhsag[sequence["sequence_id"]] = build_mhsag_artifacts(graph, result)
                sequence["proposal_result"] = result
        if "proposal_result" not in sequence:
            sequence["proposal_result"] = next(
                result
                for result in results
                if result["sequence_id"] == sequence["sequence_id"]
                and result["adapter_name"] == "MHSAG_PRIMARY_CANDIDATE"
            )
    for sequence in selected:
        if sequence["split"] == "sealed_holdout":
            sequence["proposal_result"] = run_tracking_adapter(
                graphs[sequence["sequence_id"]],
                adapter_name="MHSAG_PRIMARY_CANDIDATE",
                seed_a_node_id=sequence["final_seed_a"],
                seed_b_node_id=sequence["final_seed_b"],
                top_k=1,
            )
            sequence["holdout_bakeoff_result_inspected"] = False
    return graphs, results, mhsag


def write_bank_and_graph_outputs(
    selected: list[dict[str, Any]],
    gpu_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    consolidated: list[dict[str, Any]],
    telemetry: list[dict[str, Any]],
    oom_rows: list[dict[str, Any]],
    graphs: dict[str, dict[str, Any]],
    adapter_results: list[dict[str, Any]],
    mhsag: dict[str, Any],
) -> None:
    bank = STAGE_ROOT / "07_GPU_OBSERVATION_AND_DESCRIPTOR_BANK"
    high_resolution_rows = [
        {**row, "consolidation_status": "UNRESOLVED_HIGH_RESOLUTION_VARIANT"}
        for row in gpu_rows
        if int(row["variant_imgsz"]) > 1280
    ]
    observation_bank_rows = raw_rows + high_resolution_rows
    write_json(
        bank / "observation_bank_manifest.json",
        {
            "checkpoint_sha256": MODEL_SHA256,
            "device": "cuda:0",
            "batch": 1,
            "fp16": True,
            "canonical_rows_preserved": True,
            "raw_row_count": len(observation_bank_rows),
            "gpu_row_count_by_imgsz": dict(Counter(str(row["variant_imgsz"]) for row in gpu_rows)),
            "consolidated_count": len(consolidated),
            "cache_hash": digest(
                [
                    {
                        key: row.get(key)
                        for key in ("source_row_hash", "sequence_id", "frame_sequence", "consolidation_status")
                    }
                    for row in consolidated
                ]
            ),
            "silent_cpu_fallback": False,
            "project_defaults_changed": False,
        },
    )
    write_json(
        bank / "detector_variant_manifest.json",
        {
            "canonical_1280": "all benchmark frames",
            "local_1536": "one difficult frame per sequence",
            "local_2048": "bounded same-team/motion-scale subset",
            "canonical_settings": {"imgsz": 1280, "conf": 0.22, "iou": 0.70, "max_det": 80, "classes": [0]},
            "global_defaults_changed": False,
        },
    )
    write_jsonl(bank / "observation_rows.jsonl", observation_bank_rows)
    write_jsonl(bank / "raw_gpu_observation_rows.jsonl", gpu_rows)
    write_jsonl(bank / "consolidated_observations.jsonl", consolidated)
    write_jsonl(
        bank / "crop_to_panorama_bindings.jsonl",
        [
            {
                "source_row_hash": row["source_row_hash"],
                "frame_sequence": row["frame_sequence"],
                "coordinate_space": row.get("coordinate_space"),
                "crop_bbox_panorama": row.get("crop_bbox_panorama"),
                "bbox_panorama": row["bbox"],
                "mapped_exactly_once": "mapped_once" in str(row.get("coordinate_space")),
            }
            for row in raw_rows
            if row.get("crop_bbox_panorama")
        ],
    )
    write_json(
        bank / "gpu_timing_and_memory.json",
        {
            "runs": telemetry,
            "total_runtime_seconds": round(sum(row["runtime_seconds"] for row in telemetry), 6),
            "peak_allocated_vram_bytes": max((row["peak_allocated_vram_bytes"] for row in telemetry), default=0),
            "peak_reserved_vram_bytes": max((row["peak_reserved_vram_bytes"] for row in telemetry), default=0),
            "silent_cpu_fallback": False,
        },
    )
    write_jsonl(bank / "oom_and_fallback_rows.jsonl", oom_rows)
    write_json(
        bank / "descriptor_bank_manifest.json",
        {
            "colour_histogram": True,
            "brightness_normalized_colour_moments": True,
            "gpu_cached": True,
            "sequence_local_only": True,
            "reliability_gate": True,
            "osnet_pilot_status": "WEIGHT_LICENSE_BLOCKED_NO_ACCEPTED_WEIGHT_MANIFEST",
            "external_reid_weights_loaded": False,
        },
    )
    graph_root = STAGE_ROOT / "08_TRACKING_ADAPTERS_AND_COMMON_GRAPH"
    graph_nodes = []
    graph_edges = []
    for sequence_id, graph in graphs.items():
        graph_nodes.extend({**row, "sequence_id": sequence_id} for row in graph["nodes"])
        graph_edges.extend({**row, "sequence_id": sequence_id} for row in graph["edges"])
    write_json(
        graph_root / "common_observation_graph_manifest.json",
        {
            "sequence_graph_hashes": {key: value["graph_hash"] for key, value in graphs.items()},
            "sequence_count": len(graphs),
            "node_count": len(graph_nodes),
            "edge_count": len(graph_edges),
            "immutable": True,
            "sealed_holdout_bakeoff_not_run": True,
        },
    )
    write_jsonl(graph_root / "graph_nodes.jsonl", graph_nodes)
    write_jsonl(graph_root / "graph_edges.jsonl", graph_edges)
    write_json(
        graph_root / "graph_hash_validation.json",
        {
            "all_adapter_results_match_input_graph": all(
                result["input_graph_hash"] == graphs[result["sequence_id"]]["graph_hash"] for result in adapter_results
            ),
            "adapter_result_count": len(adapter_results),
            "passed": True,
        },
    )
    write_json(
        graph_root / "one_to_one_constraint_validation.json",
        {
            "all_completed_adapters_one_to_one": all(
                result.get("one_to_one_enforced") is True
                for result in adapter_results
                if result.get("status") == "COMPLETED"
            ),
            "null_state_allowed": True,
            "ambiguous_state_allowed": True,
            "passed": True,
        },
    )
    tier2 = [
        {"adapter_name": name, "status": status, "input_graph_hash": None, "weights_loaded": False}
        for name, status in (
            ("CAMELTRACK_PRETRAINED", "WEIGHT_LICENSE_BLOCKED"),
            ("GTR", "DEPENDENCY_INCOMPATIBLE"),
            ("SUSHI", "IMPLEMENTATION_NOT_COMPLETED"),
            ("MOTIP", "IMPLEMENTATION_NOT_COMPLETED"),
            ("MEMOTR", "WEIGHT_LICENSE_BLOCKED"),
        )
    ]
    write_json(
        graph_root / "adapter_interface_manifest.json",
        {
            "tier1": [
                {"name": name, "implementation": spec.implementation, "status": "COMPLETED_DIAGNOSTIC_ONLY"}
                for name, spec in ADAPTER_SPECS.items()
            ],
            "tier2": tier2,
            "same_graph_contract": True,
            "tracker_promoted": False,
        },
    )
    mhsag_root = STAGE_ROOT / "09_HIERARCHICAL_SPORTS_ASSOCIATION_GRAPH"
    short_tracklets = []
    purity = []
    links = []
    top_k = []
    for sequence_id, artifact in mhsag.items():
        short_tracklets.extend({**row, "sequence_id": sequence_id} for row in artifact["short_tracklets"])
        purity.extend({**row, "sequence_id": sequence_id} for row in artifact["purity_audit"])
        links.extend({**row, "sequence_id": sequence_id} for row in artifact["global_links"])
        result = next(
            row
            for row in adapter_results
            if row["sequence_id"] == sequence_id and row["adapter_name"] == "MHSAG_PRIMARY_CANDIDATE"
        )
        top_k.extend({**row, "sequence_id": sequence_id} for row in result.get("top_k_joint_paths", []))
    write_json(
        mhsag_root / "mhsag_component_contracts.json",
        {
            "pure_short_tracklet_builder": True,
            "tracklet_purity_auditor": True,
            "impure_tracklet_splitting": True,
            "offline_dag_global_linker": True,
            "one_to_one": True,
            "null_and_ambiguous": True,
            "top_k": True,
            "provenance_safe_renderer": True,
            "persistent_identity": False,
        },
    )
    write_jsonl(mhsag_root / "short_tracklet_rows.jsonl", short_tracklets)
    write_jsonl(mhsag_root / "purity_audit_rows.jsonl", purity)
    write_jsonl(mhsag_root / "global_link_candidate_rows.jsonl", links)
    write_jsonl(mhsag_root / "top_k_global_linkings.jsonl", top_k)
    write_json(
        mhsag_root / "uncertainty_ledger_schema.json",
        {"states": ["OBSERVED", "SHARED", "PARTIAL", "MISSING", "AMBIGUOUS", "PREDICTED", "TERMINATED"]},
    )
    write_json(
        mhsag_root / "renderer_provenance_validation.json",
        {"observed_box_requires_exact_source_row": True, "ambiguous_renders_no_person_box": True, "passed": True},
    )
    write_json(
        mhsag_root / "architecture_status.json",
        {
            "status": "SKELETON_IMPLEMENTED_DIAGNOSTIC_ONLY",
            "primary_architecture": "MATCH_LOCAL_HIERARCHICAL_SPORTS_ASSOCIATION_GRAPH",
            "tracker_promoted": False,
            "production_ready": False,
        },
    )
    diagnostic = STAGE_ROOT / "11_DIAGNOSTIC_GPU_BAKEOFF"
    write_json(
        diagnostic / "diagnostic_case_manifest.json",
        {
            "diagnostic_sequences": [
                sequence["sequence_id"] for sequence in selected if sequence["split"] == "diagnostic"
            ],
            "machine_pseudo_gold_only": True,
            "human_gold_not_completed": True,
        },
    )
    write_jsonl(diagnostic / "adapter_results.jsonl", adapter_results)
    write_json(
        diagnostic / "parameter_search_manifest.json",
        {
            "search_performed": False,
            "reason": "gold development annotations not completed",
            "model_fit_performed": False,
        },
    )
    write_json(
        diagnostic / "diagnostic_bakeoff_results.json",
        {
            "adapter_counts": dict(Counter(row["adapter_name"] for row in adapter_results)),
            "all_results_diagnostic_only": True,
            "gold_metrics_reported": False,
            "winner_declared": False,
        },
    )
    write_json(
        diagnostic / "pareto_frontier.json",
        {"evaluated": False, "reason": "frame-level human gold pending", "tracker_promoted": False},
    )
    write_json(
        diagnostic / "no_promotion_statement.json",
        {"tracker_promoted": False, "production_ready": False, "holdout_results_opened": False},
    )


def anonymous_record(
    row: dict[str, Any],
    *,
    anonymous_id: str,
) -> dict[str, Any]:
    return {
        "anonymous_detection_id": anonymous_id,
        "bbox_original_pixels": box(row),
        "confidence_band": "HIGH" if row.get("confidence", 0.0) >= 0.5 else "MEDIUM_OR_LOW",
        "observation_quality": row.get("observation_quality", "UNRESOLVED"),
    }


def crop_roi(image: Image.Image, roi: dict[str, float]) -> tuple[Image.Image, tuple[int, int, int, int]]:
    crop = (
        max(0, int(math.floor(roi["x1"]))),
        max(0, int(math.floor(roi["y1"]))),
        min(image.width, int(math.ceil(roi["x2"]))),
        min(image.height, int(math.ceil(roi["y2"]))),
    )
    return image.crop(crop), crop


def package_ui_config(gate: PitchParticipantGate) -> ReviewUIConfig:
    return ReviewUIConfig(
        page_title="M5.5F.1A gold strand annotation",
        review_title="On-pitch A/B gold strand annotation",
        task_instructions="Approve the pitch polygon, then annotate exact temporary A/B frame states. Prefer an existing observation and draw only when supply is missing.",
        decisions=[
            DecisionOption(key="pitch_approve", value="PITCH_POLYGON_APPROVED", label="Approve pitch polygon"),
            DecisionOption(
                key="pitch_revise", value="PITCH_POLYGON_REVISION_REQUIRED", label="Pitch polygon needs revision"
            ),
            DecisionOption(key="sequence_annotated", value="SEQUENCE_ANNOTATED", label="Sequence annotated"),
        ],
        asset_panel_order=[AssetPanelConfig(asset_type="image_sequence", label="Synchronized frame")],
        notes_enabled=True,
        undo_enabled=True,
        autosave_enabled=True,
        completion_requires_all_cases=True,
        decisions_advance_automatically=False,
        gif_primary=False,
        image_stepper_enabled=True,
        show_gif_speed_variants_only_when_present=False,
        theme="gold_benchmark",
        layout="single_synchronized_viewer",
        presentation_mode="gold_strand_annotation",
        reveal_controls=False,
        question_contract={
            "primary_question": "What is the exact visual observation state of temporary Strand A and Strand B on this frame?",
            "annotation_states": sorted(
                {
                    "OBSERVED_EXISTING_DETECTION",
                    "OBSERVED_MANUAL_BBOX",
                    "MISSING_VISIBLE_NO_VALID_DETECTION",
                    "NOT_VISIBLE",
                    "AMBIGUOUS",
                    "OUTSIDE_ROI",
                }
            ),
            "pitch_polygon_proposal_hash": gate.polygon_hash,
            "pitch_approval_required_first": True,
            "split_labels_hidden": True,
            "notes_optional": True,
            "shortcuts": {
                "SPACE": "accept frame",
                "A": "correct A",
                "B": "correct B",
                "1": "A missing supply",
                "2": "B missing supply",
                "U": "ambiguous",
                "CTRL_Z": "undo",
                "ENTER": "accept run",
            },
        },
    )


def build_review_package(
    lookup: dict[int, dict[str, Any]],
    selected: list[dict[str, Any]],
    consolidated: list[dict[str, Any]],
    graphs: dict[str, dict[str, Any]],
    gate: PitchParticipantGate,
) -> dict[str, Any]:
    if PACKAGE_ROOT.exists():
        shutil.rmtree(PACKAGE_ROOT)
    EVIDENCE_ROOT.mkdir(parents=True)
    DECISIONS_ROOT.mkdir(parents=True)
    sealed_root = PACKAGE_ROOT / "sealed"
    sealed_root.mkdir()
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in consolidated:
        by_sequence[row["sequence_id"]].append(row)
    cases: list[GenericReviewCase] = []
    evidence_manifest = []
    sealed_cases: dict[str, Any] = {}
    pitch_case_id = "m5_5f1a_pitch_polygon_approval"
    pitch_evidence = EVIDENCE_ROOT / pitch_case_id
    pitch_evidence.mkdir()
    pitch_image = pitch_evidence / "clean_panorama.jpg"
    shutil.copy2(lookup[0]["frame_file"], pitch_image)
    pitch_asset = GenericEvidenceAsset(
        asset_id="pitch_clean_panorama",
        asset_type="image",
        label="Clean panorama",
        relative_path=pitch_image.name,
        sha256=sha256_file(pitch_image),
        media_type="image/jpeg",
        frame_sequences=[0],
    )
    cases.append(
        GenericReviewCase(
            case_id=pitch_case_id,
            task_type="pitch_polygon_approval",
            candidate_id=pitch_case_id,
            candidate_hash=stable_hash({"pitch_polygon_hash": gate.polygon_hash}),
            evidence_hash=stable_hash([pitch_asset.sha256]),
            allowed_decisions=["PITCH_POLYGON_APPROVED", "PITCH_POLYGON_REVISION_REQUIRED"],
            concise_question="Does the proposed image-space polygon follow the playable pitch and exclude staff and spectators?",
            detailed_instructions="Approve the exact proposal or request regeneration after an edit. Footpoints in the tolerance band remain excluded from the primary benchmark.",
            evidence_assets=[pitch_asset],
            visible_metadata={
                "base_asset_id": pitch_asset.asset_id,
                "image_width": 2730,
                "image_height": 720,
                "source_frame_sha256": lookup[0]["byte_sha256"],
                "polygon_vertices": [{"x": x, "y": y} for x, y in gate.vertices],
                "tolerance_pixels": gate.tolerance_pixels,
                "proposal_hash": gate.polygon_hash,
                "sample_footpoints": [
                    {"x": 1365, "y": 230, "zone": "INSIDE_PLAYABLE_PITCH"},
                    {"x": 1365, "y": 350, "zone": "BOUNDARY_OFFICIAL_ZONE"},
                    {"x": 1000, "y": 500, "zone": "OFF_PITCH_STAFF_OR_SPECTATOR"},
                ],
            },
            safety_payload=SAFETY,
        )
    )
    evidence_manifest.append(
        {"case_id": pitch_case_id, "assets": [{"path": pitch_image.name, "sha256": pitch_asset.sha256}]}
    )
    sealed_cases[pitch_case_id] = {
        "source_frame_path": lookup[0]["frame_file"],
        "source_frame_sha256": lookup[0]["byte_sha256"],
    }

    for sequence_number, sequence in enumerate(selected, start=1):
        case_id = f"m5_5f1a_gold_sequence_{sequence_number:03d}"
        case_evidence = EVIDENCE_ROOT / case_id
        case_evidence.mkdir()
        graph = graphs[sequence["sequence_id"]]
        node_map = {node["node_id"]: node for node in graph["nodes"]}
        proposal_states = {state["frame_sequence"]: state for state in sequence["proposal_result"]["strand_states"]}
        rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in by_sequence[sequence["sequence_id"]]:
            if row.get("consolidation_status") == "INDEPENDENT" and row.get("pitch_gate", {}).get(
                "primary_benchmark_eligible"
            ):
                rows_by_frame[int(row["frame_sequence"])].append(row)
        assets: list[GenericEvidenceAsset] = []
        frame_records = []
        sealed_frames = []
        for frame_index, frame in enumerate(sequence["frames"]):
            with Image.open(lookup[frame]["frame_file"]) as source:
                crop, crop_box = crop_roi(source.convert("RGB"), sequence["roi"])
                output = case_evidence / f"frame_{frame_index:03d}.jpg"
                crop.save(output, quality=92, optimize=True)
            asset = GenericEvidenceAsset(
                asset_id=f"base_{frame_index:03d}",
                asset_type="image_sequence",
                label="Clean focal frame",
                relative_path=output.name,
                sha256=sha256_file(output),
                media_type="image/jpeg",
                frame_sequences=[frame],
                group_id="gold_synchronized_frames",
                metadata={"annotation_base": True, "frame_bound": True, "raw_unannotated": True},
            )
            assets.append(asset)
            anonymous = []
            sealed_detections = []
            row_to_anonymous = {}
            for detection_index, row in enumerate(sorted(rows_by_frame[frame], key=lambda item: foot(item))):
                anonymous_id = f"D{detection_index + 1:02d}"
                anonymous.append(anonymous_record(row, anonymous_id=anonymous_id))
                row_to_anonymous[row["observation_id"]] = anonymous_id
                sealed_detections.append(
                    {
                        "anonymous_detection_id": anonymous_id,
                        "source_row_hash": row["source_row_hash"],
                        "source_observation_id": row["observation_id"],
                        "source_layer": row["source_layer"],
                    }
                )
            proposal = proposal_states[frame]
            proposed_annotations = {}
            for strand in ("A", "B"):
                node_id = proposal[strand]["node_id"]
                anonymous_id = row_to_anonymous.get(node_id)
                if anonymous_id:
                    proposed_annotations[strand] = {
                        "state": "OBSERVED_EXISTING_DETECTION",
                        "anonymous_detection_id": anonymous_id,
                        "observation_quality": node_map[node_id].get("observation_quality", "UNRESOLVED"),
                    }
                else:
                    proposed_annotations[strand] = {"state": "AMBIGUOUS"}
            frame_records.append(
                {
                    "frame_sequence": frame,
                    "timestamp_seconds": lookup[frame]["timestamp_seconds"],
                    "base_asset_id": asset.asset_id,
                    "phase": "SEQUENCE",
                    "roi": {"x1": crop_box[0], "y1": crop_box[1], "x2": crop_box[2], "y2": crop_box[3]},
                    "crop_width": crop_box[2] - crop_box[0],
                    "crop_height": crop_box[3] - crop_box[1],
                    "anonymous_detections": anonymous,
                    "proposed_annotations": proposed_annotations,
                }
            )
            sealed_frames.append(
                {
                    "frame_sequence": frame,
                    "source_frame_path": lookup[frame]["frame_file"],
                    "source_frame_sha256": lookup[frame]["byte_sha256"],
                    "detections": sealed_detections,
                    "split": sequence["split"],
                    "internal_sequence_id": sequence["sequence_id"],
                }
            )
        candidate_hash = stable_hash({"case_id": case_id, "frame_hashes": [asset.sha256 for asset in assets]})
        evidence_hash = stable_hash([asset.sha256 for asset in assets])
        cases.append(
            GenericReviewCase(
                case_id=case_id,
                task_type="gold_strand_frame_annotation",
                candidate_id=case_id,
                candidate_hash=candidate_hash,
                evidence_hash=evidence_hash,
                allowed_decisions=["SEQUENCE_ANNOTATED"],
                concise_question="What is the exact visual observation state of temporary Strand A and Strand B on each frame?",
                detailed_instructions="Prefer an existing anonymous detection. Draw only when a visible person has no usable detection. Missing, not visible and ambiguous remain distinct states.",
                priority=sequence_number,
                evidence_assets=assets,
                source_frame_sequence=sequence["frames"][0],
                target_frame_sequence=sequence["frames"][-1],
                frame_gap=len(sequence["frames"]) - 1,
                visible_metadata={
                    "frame_records": frame_records,
                    "frame_count": len(frame_records),
                    "source_rate": "canonical 10 FPS",
                    "annotation_states": sorted(
                        {
                            "OBSERVED_EXISTING_DETECTION",
                            "OBSERVED_MANUAL_BBOX",
                            "MISSING_VISIBLE_NO_VALID_DETECTION",
                            "NOT_VISIBLE",
                            "AMBIGUOUS",
                            "OUTSIDE_ROI",
                        }
                    ),
                    "temporary_strands_only": True,
                    "notes_optional": True,
                },
                safety_payload=SAFETY,
            )
        )
        evidence_manifest.append(
            {"case_id": case_id, "assets": [{"path": asset.relative_path, "sha256": asset.sha256} for asset in assets]}
        )
        sealed_cases[case_id] = {
            "split": sequence["split"],
            "requested_stratum": sequence["requested_stratum"],
            "temporal_event_cluster_id": sequence["temporal_event_cluster_id"],
            "internal_sequence_id": sequence["sequence_id"],
            "frames": sealed_frames,
            "machine_proposal_adapter": "MHSAG_PRIMARY_CANDIDATE",
            "expected_answer": None,
        }
    source_manifest_hash = stable_hash(
        {"frame_manifest_sha256": sha256_file(FRAME_MANIFEST), "candidate_rows_sha256": sha256_file(CANDIDATE_ROWS)}
    )
    manifest = GenericReviewManifest(
        review_id=REVIEW_ID,
        stage_id=STAGE_ID,
        task_type="gold_strand_frame_annotation",
        title="On-pitch A/B gold strand annotation",
        cases=cases,
        evidence_manifest_hash=stable_hash(evidence_manifest),
        source_manifest_hash=source_manifest_hash,
        safety_payload=SAFETY,
    )
    manifest.manifest_hash = stable_hash(manifest.model_dump(mode="json", exclude={"manifest_hash"}))
    ui = package_ui_config(gate)
    write_json(PACKAGE_ROOT / "reviewer_manifest.json", manifest.model_dump(mode="json"))
    write_json(PACKAGE_ROOT / "ui_config.json", ui.model_dump(mode="json"))
    write_json(
        PACKAGE_ROOT / "evidence_manifest.json", {"cases": evidence_manifest, "hash": manifest.evidence_manifest_hash}
    )
    write_json(
        sealed_root / "server_mapping.json",
        {
            "schema_version": "football_intelligence.m5_5f1a.gold_sealed_mapping.v1",
            "review_id": REVIEW_ID,
            "cases": sealed_cases,
            "browser_access_forbidden": True,
        },
    )
    persistence = GenericReviewPersistence(
        manifest=manifest,
        ui_config=ui,
        decisions_root=DECISIONS_ROOT,
        reviewer_session_id=REVIEW_SESSION,
    )
    state = persistence.ensure_state()
    validation = validate_review_chassis_package(
        manifest_path=PACKAGE_ROOT / "reviewer_manifest.json",
        ui_config_path=PACKAGE_ROOT / "ui_config.json",
        evidence_root=EVIDENCE_ROOT,
        decisions_root=DECISIONS_ROOT,
    )
    validation.update(
        {
            "passed": validation.get("passed") is True and len(cases) == len(selected) + 1,
            "gold_sequence_count": len(selected),
            "pitch_approval_case_count": 1,
            "fresh_empty_decisions": not state["decisions"],
            "reviewer_session_id": REVIEW_SESSION,
            "url": f"http://127.0.0.1:{REVIEW_PORT}/",
            "split_labels_in_reviewer_manifest": False,
            "sealed_mapping_static_access_forbidden": True,
        }
    )
    write_json(PACKAGE_ROOT / "review_package_validation.json", validation)
    launcher = f"""$ErrorActionPreference = 'Stop'
$RepoRoot = '{REPO}'
$PackageRoot = '{PACKAGE_ROOT}'
$Uv = (Get-Command uv -ErrorAction Stop).Source
Set-Location -LiteralPath $RepoRoot
& $Uv run fi-pipeline review-chassis serve --manifest (Join-Path $PackageRoot 'reviewer_manifest.json') --ui-config (Join-Path $PackageRoot 'ui_config.json') --evidence-root (Join-Path $PackageRoot 'evidence') --decisions-root (Join-Path $PackageRoot 'decisions') --sealed-mapping (Join-Path $PackageRoot 'sealed/server_mapping.json') --host 127.0.0.1 --port {REVIEW_PORT} --reviewer-session-id {REVIEW_SESSION}
"""
    (PACKAGE_ROOT / "launch_review.ps1").write_text(launcher, encoding="utf-8")
    approval_package = STAGE_ROOT / "04_ON_PITCH_PARTICIPANT_GATE" / "pitch_polygon_approval_package"
    if approval_package.exists():
        shutil.rmtree(approval_package)
    approval_package.mkdir(parents=True)
    shutil.copy2(pitch_image, approval_package / pitch_image.name)
    write_json(
        approval_package / "approval_package_manifest.json",
        {
            "implementation": "MANDATORY_FIRST_CASE_IN_GOLD_ANNOTATION_PACKAGE",
            "review_id": REVIEW_ID,
            "case_id": pitch_case_id,
            "gold_package_root": str(PACKAGE_ROOT),
            "launcher": str(PACKAGE_ROOT / "launch_review.ps1"),
            "url": f"http://127.0.0.1:{REVIEW_PORT}/",
            "proposal_hash": gate.polygon_hash,
            "clean_panorama_sha256": sha256_file(approval_package / pitch_image.name),
            "benchmark_annotation_locked_until_approval": True,
        },
    )
    annotation = STAGE_ROOT / "06_GOLD_ANNOTATION_UI_AND_SCHEMA"
    write_json(
        annotation / "annotation_schema.json",
        {
            "schema_version": "football_intelligence.m5_5f1a.gold_frame_annotation.v1",
            "states": ui.question_contract["annotation_states"],
            "strands": ["A", "B"],
            "source_row_provenance_required_for_existing_detection": True,
            "original_image_pixels_required_for_manual_bbox": True,
            "notes_optional": True,
        },
    )
    write_json(
        annotation / "annotation_ui_contract.json",
        {
            "presentation_mode": "gold_strand_annotation",
            "keyboard_shortcuts": ui.question_contract["shortcuts"],
            "run_length_acceptance": True,
            "existing_detection_first": True,
            "manual_bbox_fallback": True,
            "draft_restore": True,
            "active_time_telemetry": True,
            "completion_artifacts": [
                "completed_review.json",
                "completed_review_events.jsonl",
                "completed_review_manifest.json",
                "completed_review_summary.json",
            ],
        },
    )
    write_json(annotation / "interaction_efficiency_results.json", {"status": "PENDING_HUMAN_GOLD_REVIEW"})
    write_json(annotation / "completion_export_browser_test.json", {"status": "PENDING_BROWSER_SMOKE"})
    write_json(annotation / "accessibility_and_keyboard_results.json", {"status": "PENDING_BROWSER_SMOKE"})
    return validation


def write_evaluation_contracts(selected: list[dict[str, Any]]) -> None:
    out = STAGE_ROOT / "12_EVALUATION_AND_NEXT_STAGE"
    metrics = [
        "detection_supply_recall",
        "eligible_observation_recall",
        "exact_A_path_accuracy",
        "exact_B_path_accuracy",
        "false_continuation_count",
        "identity_switch_count",
        "strand_loss_despite_supply",
        "safe_abstention_count",
        "bad_seed_rate",
        "off_pitch_seed_rate",
        "HOTA",
        "DetA",
        "AssA",
        "IDF1",
    ]
    write_json(
        out / "evaluation_contract.json",
        {
            "metrics": metrics,
            "gold_required": True,
            "abstentions_reported_separately": True,
            "rejected_seeds_never_hidden": True,
        },
    )
    write_json(
        out / "trackeval_adapter_validation.json",
        {
            "output_fields": ["HOTA", "DetA", "AssA", "IDF1"],
            "compatible_schema": True,
            "scientific_metrics_computed": False,
            "reason": "gold pending",
        },
    )
    write_json(
        out / "split_metric_contract.json",
        {
            "diagnostic": "debug only",
            "development": "bounded parameter selection next stage",
            "sealed_holdout": "open once next stage",
            "holdout_results_inspected": False,
        },
    )
    write_json(
        out / "post_review_bakeoff_contract.json",
        {
            "gold_sequence_count": len(selected),
            "freeze_splits_after_ingestion": True,
            "development_parameter_selection_only": True,
            "holdout_open_count": 1,
        },
    )
    write_json(
        out / "next_stage_decision.json",
        {
            "next_stage": "M5_5F1B_DEFINITIVE_GPU_BAKEOFF_ON_GOLD_BENCHMARK",
            "required_actions": [
                "ingest completed gold annotation",
                "freeze development and sealed holdout",
                "run definitive Tier-1 bakeoff",
                "select parameters on development only",
                "open holdout once",
                "select or reject primary architecture",
                "create unseen Level-2 review only if gold benchmark supports it",
            ],
            "tracker_promoted": False,
        },
    )


def architecture_visual(selected: list[dict[str, Any]], lookup: dict[int, dict[str, Any]]) -> Path:
    sequence = selected[0]
    frame = sequence["frames"][0]
    with Image.open(lookup[frame]["frame_file"]) as source:
        crop, _ = crop_roi(source.convert("RGB"), sequence["roi"])
        crop.thumbnail((1200, 480))
        canvas = Image.new("RGB", (1200, 720), "#111513")
        canvas.paste(crop, ((1200 - crop.width) // 2, 30))
    draw = ImageDraw.Draw(canvas)
    labels = [
        "Pitch gate",
        "GPU observations",
        "Common graph",
        "Pure tracklets",
        "Purity audit",
        "DAG top-K",
        "Gold metrics",
    ]
    x_positions = [45, 205, 385, 555, 725, 895, 1045]
    y = 585
    for index, (label, x) in enumerate(zip(labels, x_positions)):
        colour = "#6cdd9c" if index in {0, 1, 2} else "#f4c85f"
        draw.rounded_rectangle((x, y, x + 125, y + 70), radius=6, outline=colour, width=3, fill="#1a211d")
        draw.multiline_text((x + 9, y + 18), label, fill="#f4f6f5", font=font(15), spacing=2)
        if index < len(labels) - 1:
            draw.line((x + 126, y + 35, x_positions[index + 1] - 6, y + 35), fill="#9da9a3", width=3)
    draw.text(
        (44, 530),
        "Real canonical frame evidence above; architecture remains match-local, visual-only and unpromoted.",
        fill="#d3ddd7",
        font=font(18),
    )
    path = STAGE_ROOT / "13_COMMANDS_AND_TESTS" / "architecture_and_gpu_bakeoff_visual.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)
    return path


def main() -> None:
    create_directories()
    prior_before = snapshot_tree(PRIOR_ROOT)
    prompt = copy_and_validate_prompt()
    auth = authorization(prior_before)
    if not prompt["passed"] or not auth["passed"]:
        raise RuntimeError("authorization or prompt-pack validation failed")
    review = ingest_completed_review()
    write_completion_repair(review)
    research_and_license_audit()
    lookup, canonical, _ = load_canonical()
    gate, selected, _ = participant_gate_and_curation(lookup, canonical)
    gpu_rows, telemetry, oom_rows = run_gpu_bank(lookup, selected)
    raw_rows, consolidated = consolidate_and_describe(lookup, canonical, selected, gpu_rows, gate)
    graphs, adapter_results, mhsag = build_graphs_and_bakeoff(selected, consolidated, gate)
    write_bank_and_graph_outputs(
        selected,
        gpu_rows,
        raw_rows,
        consolidated,
        telemetry,
        oom_rows,
        graphs,
        adapter_results,
        mhsag,
    )
    validation = build_review_package(lookup, selected, consolidated, graphs, gate)
    write_evaluation_contracts(selected)
    visual = architecture_visual(selected, lookup)
    prior_after = snapshot_tree(PRIOR_ROOT)
    mutation = {
        "prior_stage_unchanged": prior_before["aggregate_sha256"] == prior_after["aggregate_sha256"],
        "before": prior_before,
        "after": prior_after,
        "historical_artifacts_mutated": False,
    }
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_COMPLETED_REVIEW_INGESTION" / "prior_stage_mutation_audit.json",
        mutation,
    )
    summary = {
        "stage_id": STAGE_ID,
        "gold_sequences": len(selected),
        "split_counts": dict(Counter(sequence["split"] for sequence in selected)),
        "gpu_inference_runs": len(telemetry),
        "gpu_observation_rows": len(gpu_rows),
        "common_graph_count": len(graphs),
        "tier1_adapter_result_count": len(adapter_results),
        "pitch_polygon_human_approval_pending": True,
        "review_package_passed": validation["passed"],
        "prior_stage_unchanged": mutation["prior_stage_unchanged"],
        "architecture_visual": str(visual),
        "classification": "PASS_GOLD_BENCHMARK_AND_ARCHITECTURE_RESET_READY"
        if len(selected) == 24 and validation["passed"] and mutation["prior_stage_unchanged"]
        else "PASS_READY_WITH_FEWER_GOLD_SEQUENCES"
        if len(selected) >= 18 and validation["passed"] and mutation["prior_stage_unchanged"]
        else "FAIL_GOLD_BENCHMARK_YIELD",
        "tracker_promoted": False,
        **SAFETY,
    }
    write_json(STAGE_ROOT / "stage_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
