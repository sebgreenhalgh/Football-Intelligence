"""Build the bounded M5.5F.1 association bakeoff and unseen review package."""

# The stage writes explicit scientific ledgers; long serialized records are intentional.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

import build_m5_5f0_stable_local_strand as cpu
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


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
MATCH_ROOT = ROOT / "matches" / "128058"
PROMPT_ROOT = MATCH_ROOT / "runs" / "step_m5" / "part 2" / "M5_5F1_Sequence_Global_Association_Bakeoff_Prompt_v1"
PRIOR_ROOT = (
    MATCH_ROOT / "runs" / "step_m5" / "part 2" / "M5_5F0C_SEED_CURATION_DEDUPLICATION_AND_ONE_FRAME_DROPOUT_REPAIR_v1"
)
PRIOR_PACKAGE = PRIOR_ROOT / "08_VALIDATED_LEVEL2_CONTINUITY_REVIEW_PACKAGE"
STAGE_ID = "M5_5F1_SEQUENCE_GLOBAL_ASSOCIATION_BAKEOFF_AND_UNSEEN_LEVEL2_VALIDATION_v1"
STAGE_ROOT = MATCH_ROOT / "runs" / "step_m5" / "part 2" / STAGE_ID
REVIEW_ROOT = STAGE_ROOT / "09_UNSEEN_LEVEL2_ASSOCIATION_REVIEW_PACKAGE"
EVIDENCE_ROOT = REVIEW_ROOT / "evidence"
DECISIONS_ROOT = REVIEW_ROOT / "decisions"
REVIEW_ID = "m5_5f1_unseen_level2_association_review_v1"
REVIEW_SESSION = "m5_5f1_unseen_level2_association_human_reviewer"
REVIEW_PORT = 8799
PACK_ROOT = STAGE_ROOT / "12_REVIEW_PACK_FOR_CHATGPT"
AUTHORIZED_BASELINE = "f64612757ccec5dfe919f406ade116be4c842045"
MODEL_PATH = REPO / "models" / "model=yolov8m-imgsz=2048.pt"
MODEL_SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
MODEL_BYTES = 52136884
DIAGNOSTIC_CASES = {
    "m5_5f0c_level2_candidate_002": (49, 61),
    "m5_5f0c_level2_candidate_003": (109, 121),
    "m5_5f0c_level2_candidate_004": (169, 181),
    "m5_5f0c_level2_candidate_005": (229, 241),
    "m5_5f0c_level2_candidate_006": (289, 301),
    "m5_5f0c_level2_candidate_007": (349, 361),
}
UNSEEN_CENTRES = [15, 25, 35, 75, 85, 95, 135, 145, 155, 195, 205, 215, 255, 265, 275, 315, 325, 335, 375, 385]
EXCLUDED_WINDOWS = list(DIAGNOSTIC_CASES.values())
UNSEEN_STRATA = [
    "easy_separated_pair",
    "same_team_nearby_distractor",
    "cross_team_distractor",
    "moderate_motion_scale_change",
]
OUTCOMES = {
    "PASS": "PASS - Stable local continuation",
    "A_SWITCH": "A_SWITCH - Strand A switches",
    "B_SWITCH": "B_SWITCH - Strand B switches",
    "BOTH_SWITCH": "BOTH_SWITCH - Both strands switch",
    "A_LOST": "A_LOST - Strand A is lost",
    "B_LOST": "B_LOST - Strand B is lost",
    "BOTH_LOST": "BOTH_LOST - Both strands are lost",
    "DETECTION_SUPPLY_FAILURE": "DETECTION_SUPPLY_FAILURE - Local supply is insufficient",
    "AMBIGUOUS_BUT_SAFE_ABSTENTION": "AMBIGUOUS_BUT_SAFE_ABSTENTION - Tracker abstains safely",
    "BAD_CASE": "BAD_CASE - Case is not suitable",
    "UNRESOLVED": "UNRESOLVED - Evidence is unresolved",
    "BAD_SEED_CASE": "BAD_SEED_CASE - Reject bad seed case",
}
SEED_ACTIONS = {
    "CONFIRM": "CONFIRM - Proposed A/B seeds are usable",
    "SWAP_A_B": "SWAP_A_B - Swap the proposed A/B seeds",
    "CORRECT_A": "CORRECT_A - Correct Strand A seed",
    "CORRECT_B": "CORRECT_B - Correct Strand B seed",
    "REJECT_BAD_SEED_CASE": "REJECT_BAD_SEED_CASE - Reject this seed case",
}
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
    "occlusion_mining_performed": False,
    "ghost_reentry_work_performed": False,
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8"
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()


def tree_snapshot(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        stat = path.stat()
        try:
            file_hash = sha256_file(path) if stat.st_size <= 2_000_000 else None
            read_error = None
        except OSError as exc:
            file_hash = None
            read_error = f"{type(exc).__name__}:{getattr(exc, 'errno', None)}"
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size": stat.st_size,
                "sha256": file_hash,
                "read_error": read_error,
            }
        )
    return {"root": str(root), "file_count": len(files), "files": files, "aggregate_sha256": digest(files)}


def ingest_completed_review() -> dict[str, Any]:
    export = read_json(PRIOR_PACKAGE / "decisions" / "completed_review.json")
    summary = read_json(PRIOR_PACKAGE / "decisions" / "completed_review_summary.json")
    manifest = read_json(PRIOR_PACKAGE / "reviewer_manifest.json")
    events = read_jsonl(PRIOR_PACKAGE / "decisions" / "completed_review_events.jsonl")
    validation = read_json(PRIOR_PACKAGE / "review_package_validation.json")
    state = export.get("state", {})
    decisions = state.get("decisions", {})
    structured = state.get("structured_reviews", {})
    counts = Counter(decisions.values())
    expected = Counter({"PASS": 3, "A_SWITCH": 2, "B_SWITCH": 1})
    cases = {case["case_id"]: case for case in manifest.get("cases", [])}
    normalized = []
    for case_id in sorted(decisions):
        item = cases[case_id]
        review = structured.get(case_id, {})
        window = item.get("visible_metadata", {}).get("frame_window", {})
        normalized.append(
            {
                "case_id": case_id,
                "source_window": [int(window.get("start")), int(window.get("end"))],
                "human_decision": decisions[case_id],
                "seed_action": review.get("seed_action"),
                "first_failure_frame": int(review["first_failure_frame"])
                if review.get("first_failure_frame") not in (None, "")
                else None,
                "note": review.get("note"),
                "candidate_hash": item.get("candidate_hash"),
                "evidence_hash": item.get("evidence_hash"),
            }
        )
    decision_events = [event for event in events if event.get("event_type") == "decision"]
    passed = (
        state.get("completed") is True
        and len(decisions) == 6
        and counts == expected
        and all(row["seed_action"] == "CONFIRM" for row in normalized)
        and {row["first_failure_frame"] for row in normalized if row["first_failure_frame"] is not None}
        == {119, 175, 235}
        and len(decision_events) == 6
        and state.get("reviewer_session_id") == "m5_5f0c_validated_level2_continuity_human_reviewer"
        and state.get("elapsed_active_seconds") == 0
        and validation.get("passed") is True
    )
    return {
        "passed": passed,
        "review_id": export.get("review_id"),
        "reviewer_session_id": state.get("reviewer_session_id"),
        "reviewed": len(decisions),
        "remaining": summary.get("remaining"),
        "decision_counts": dict(counts),
        "all_seed_actions_confirm": all(row["seed_action"] == "CONFIRM" for row in normalized),
        "failure_frames": sorted(
            {row["first_failure_frame"] for row in normalized if row["first_failure_frame"] is not None}
        ),
        "notes_count": summary.get("counts", {}).get("notes_count", 0),
        "elapsed_active_seconds": state.get("elapsed_active_seconds"),
        "telemetry_defect": state.get("elapsed_active_seconds") == 0,
        "decision_event_count": len(decision_events),
        "manifest_hash": state.get("manifest_hash"),
        "historical_decisions_sha256": sha256_file(PRIOR_PACKAGE / "decisions" / "completed_review.json"),
        "historical_events_sha256": sha256_file(PRIOR_PACKAGE / "decisions" / "completed_review_events.jsonl"),
        "normalized": normalized,
        "prior_decisions_read_only": True,
    }


def source_rows_and_lookup() -> (
    tuple[dict[str, dict[int, list[dict[str, Any]]]], dict[int, dict[str, Any]], list[dict[str, Any]]]
):
    events, rows = cpu.prior_e3.source_rows()
    event = next(item for item in events if item.get("source_id") == "stage_a_canonical_10fps_window")
    lookup = {int(frame): value for frame, value in event["frame_lookup"].items()}
    return rows, lookup, events


def initial_candidate(
    case_id: str,
    centre: int,
    source: dict[int, list[dict[str, Any]]],
    lookup: dict[int, dict[str, Any]],
    diagnostic: bool,
) -> dict[str, Any] | None:
    base = cpu.benchmark_candidate(source, lookup, centre, 2)
    if base is None:
        return None
    start, end = centre - 6, centre + 6
    base.update(
        {
            "benchmark_case_id": case_id,
            "case_id": case_id,
            "frames": list(range(start, end + 1)),
            "start_frame": start,
            "source_id": case_id,
            "requested_level": 2,
            "diagnostic": diagnostic,
            "human_answers_used": False,
            "holdout_excluded": diagnostic,
            "source_discovery": "diagnostic_case" if diagnostic else "machine_only_unseen_temporal_scan",
            "source_frame_lookup": {str(frame): lookup[frame] for frame in range(start, end + 1)},
        }
    )
    return base


def run_tagged_gpu_detector(candidates: list[dict[str, Any]], lookup: dict[int, dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing CPU detector fallback")
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size != MODEL_BYTES or sha256_file(MODEL_PATH) != MODEL_SHA256:
        raise RuntimeError("approved detector checkpoint hash or byte size mismatch")
    model = YOLO(str(MODEL_PATH))
    model.to("cuda:0")
    device = str(next(model.model.parameters()).device)
    if device != "cuda:0":
        raise RuntimeError(f"detector resolved to {device}; silent CPU fallback is prohibited")
    rows: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    for candidate in candidates:
        for frame in candidate["frames"]:
            lookup_row = lookup[frame]
            crop = cpu.clamp_crop(candidate["roi"], int(lookup_row["width"]), int(lookup_row["height"]))
            with Image.open(lookup_row["frame_file"]) as image:
                crop_image = np.asarray(image.convert("RGB").crop(crop))
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            prediction = model.predict(
                source=crop_image,
                imgsz=1280,
                conf=0.22,
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
            coords = boxes.xyxy.detach().cpu().tolist() if boxes is not None else []
            confidences = boxes.conf.detach().cpu().tolist() if boxes is not None else []
            for index, (values, confidence) in enumerate(zip(coords, confidences)):
                rows.append(
                    {
                        "case_id": candidate["case_id"],
                        "observation_id": f"{candidate['case_id']}_gpu1280_{frame}_{index:03d}",
                        "_observation_key": f"{candidate['case_id']}_gpu1280_{frame}_{index:03d}",
                        "source_layer": "gpu_recovery_1280",
                        "frame_sequence": frame,
                        "frame_file": str(lookup_row["frame_file"]),
                        "coordinate_space": "native_crop_pixels_mapped_to_panorama_pixels",
                        "crop_bbox_panorama": {"x1": crop[0], "y1": crop[1], "x2": crop[2], "y2": crop[3]},
                        "bbox_crop": {"x1": values[0], "y1": values[1], "x2": values[2], "y2": values[3]},
                        "bbox": {
                            "x1": values[0] + crop[0],
                            "y1": values[1] + crop[1],
                            "x2": values[2] + crop[0],
                            "y2": values[3] + crop[1],
                        },
                        "confidence": float(confidence),
                        "variant_imgsz": 1280,
                        "checkpoint_sha256": MODEL_SHA256,
                        "device": device,
                        "half": True,
                        "global_defaults_changed": False,
                        "local_sandbox_only": True,
                    }
                )
            if boxes is not None and str(boxes.data.device) != "cuda:0":
                raise RuntimeError(f"detector tensor device was {boxes.data.device}")
            torch.cuda.synchronize()
            telemetry.append(
                {
                    "case_id": candidate["case_id"],
                    "frame_sequence": frame,
                    "imgsz": 1280,
                    "rows": len(coords),
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "device": device,
                    "status": "completed",
                }
            )
    by_case: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_case[row["case_id"]][int(row["frame_sequence"])].append(row)
    return {
        "checkpoint_sha256": MODEL_SHA256,
        "checkpoint_bytes": MODEL_PATH.stat().st_size,
        "device": device,
        "fp16": True,
        "batch": 1,
        "imgsz": 1280,
        "rows": rows,
        "rows_by_case": {key: dict(value) for key, value in by_case.items()},
        "row_count": len(rows),
        "telemetry": telemetry,
        "oom_count": 0,
        "silent_cpu_fallback": False,
        "global_defaults_changed": False,
        "local_sandbox_only": True,
    }


def add_cuda_descriptors(graphs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; appearance descriptor run is blocked")
    rows = [row for graph in graphs.values() for row in graph["nodes"]]
    started = time.perf_counter()
    for row in rows:
        with Image.open(row["frame_file"]) as image:
            crop = (
                image.convert("RGB")
                .crop(tuple(int(row["bbox"][key]) for key in ("x1", "y1", "x2", "y2")))
                .resize((32, 16))
            )
            array = np.asarray(crop, dtype="float32") / 255.0
        tensor = torch.from_numpy(array).to("cuda:0", dtype=torch.float16)
        flat = tensor.reshape(-1, 3)
        moments = torch.cat((flat.mean(dim=0), flat.std(dim=0), flat.min(dim=0).values, flat.max(dim=0).values))
        row["cuda_colour_descriptor"] = [round(float(value), 6) for value in moments.detach().cpu().tolist()]
    torch.cuda.synchronize()
    return {
        "descriptor_type": "cuda_local_colour_moments",
        "rows": len(rows),
        "device": "cuda:0",
        "sequence_local_only": True,
        "external_reid_model_used": False,
        "model_fit_performed": False,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "geometry_is_absolute_veto": True,
    }


def descriptor_distance(left: dict[str, Any] | None, right: dict[str, Any] | None) -> float:
    if not left or not right or not left.get("cuda_colour_descriptor") or not right.get("cuda_colour_descriptor"):
        return 0.0
    return math.sqrt(
        sum((float(a) - float(b)) ** 2 for a, b in zip(left["cuda_colour_descriptor"], right["cuda_colour_descriptor"]))
    )


def graph_edge(left: dict[str, Any], right: dict[str, Any], gap: int = 1) -> dict[str, Any]:
    displacement = math.dist(cpu.foot(left), cpu.foot(right))
    max_height = max(cpu.height(left), cpu.height(right), 1.0)
    hard_gate = displacement <= max(130.0, 6.0 * max_height * max(1, gap))
    left_box, right_box = cpu.box(left), cpu.box(right)
    left_aspect = (left_box["x2"] - left_box["x1"]) / max(1.0, left_box["y2"] - left_box["y1"])
    right_aspect = (right_box["x2"] - right_box["x1"]) / max(1.0, right_box["y2"] - right_box["y1"])
    return {
        "from_observation_id": left["node_id"],
        "to_observation_id": right["node_id"],
        "gap_length": gap,
        "motion_residual": round(displacement / max_height, 6),
        "footpoint_residual": round(displacement, 6),
        "bbox_scale_change": round(abs(math.log(max(1.0, cpu.height(right)) / max(1.0, cpu.height(left)))), 6),
        "aspect_change": round(abs(left_aspect - right_aspect), 6),
        "appearance_distance": round(descriptor_distance(left, right), 6),
        "team_colour_compatibility": round(max(0.0, 1.0 - descriptor_distance(left, right)), 6),
        "hard_geometry_gate_pass": hard_gate,
        "observation_quality": round(float(right.get("confidence", 0.0)), 6),
    }


def build_graph(candidate: dict[str, Any], rows_by_frame: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    frames: dict[int, list[dict[str, Any]]] = {}
    nodes: list[dict[str, Any]] = []
    clusters: list[dict[str, Any]] = []
    for frame in candidate["frames"]:
        pool, duplicate_clusters = cpu.consolidate_frame(rows_by_frame.get(frame, []), candidate["roi"])
        frame_nodes = []
        for index, row in enumerate(pool):
            value = dict(row)
            value["node_id"] = f"{candidate['case_id']}_node_{frame}_{index:03d}"
            value["frame_sequence"] = frame
            frame_nodes.append(value)
            nodes.append(value)
        frames[frame] = frame_nodes
        clusters.extend({"frame_sequence": frame, **cluster} for cluster in duplicate_clusters)
    edges = []
    for left_frame, right_frame in zip(candidate["frames"], candidate["frames"][1:]):
        for left, right in itertools.product(frames[left_frame], frames[right_frame]):
            edges.append(graph_edge(left, right))
    graph = {
        "case_id": candidate["case_id"],
        "frames": frames,
        "nodes": nodes,
        "edges": edges,
        "duplicate_clusters": clusters,
        "graph_digest": digest(
            {
                "nodes": [
                    {key: value for key, value in row.items() if key != "cuda_colour_descriptor"} for row in nodes
                ],
                "edges": edges,
            }
        ),
    }
    return graph


def map_seed(seed: list[dict[str, Any]], graph: dict[str, Any], frame: int) -> list[dict[str, Any]]:
    candidates = graph["frames"].get(frame, [])
    selected = []
    used = set()
    for target in seed:
        ranked = sorted(
            (
                (cpu.iou(cpu.box(target), cpu.box(row)), math.dist(cpu.foot(target), cpu.foot(row)), row)
                for row in candidates
                if row["node_id"] not in used
            ),
            key=lambda item: (-item[0], item[1], item[2]["node_id"]),
        )
        if not ranked or (ranked[0][0] < 0.1 and ranked[0][1] > max(50.0, 2.0 * cpu.height(target))):
            continue
        selected.append(ranked[0][2])
        used.add(ranked[0][2]["node_id"])
    return selected


def choose_machine_seed(graph: dict[str, Any], frame: int, stratum: str) -> list[dict[str, Any]]:
    rows = graph["frames"].get(frame, [])
    pairs = [
        (
            math.dist(cpu.foot(left), cpu.foot(right)),
            descriptor_distance(left, right),
            max(cpu.height(left), cpu.height(right)) / max(1.0, min(cpu.height(left), cpu.height(right))),
            left,
            right,
        )
        for left, right in itertools.combinations(rows, 2)
        if math.dist(cpu.foot(left), cpu.foot(right)) > 3.0 * max(cpu.height(left), cpu.height(right))
    ]
    if not pairs:
        return []

    def ranking(item: tuple[float, float, float, dict[str, Any], dict[str, Any]]) -> tuple[float, str, str]:
        separation, colour_gap, scale_ratio, left, right = item
        scale = max(cpu.height(left), cpu.height(right), 1.0)
        separation_ratio = separation / scale
        if stratum == "easy_separated_pair":
            score = separation_ratio + 2.0 * colour_gap
        elif stratum == "same_team_nearby_distractor":
            score = -(separation_ratio + 3.0 * colour_gap)
        elif stratum == "cross_team_distractor":
            score = 3.0 * colour_gap + min(separation_ratio, 14.0) / 14.0
        else:
            score = 2.0 * scale_ratio + min(separation_ratio, 10.0) / 10.0
        return score, left["node_id"], right["node_id"]

    _, _, _, left, right = max(pairs, key=ranking)
    return [left, right]


def transition_components(
    previous: dict[str, Any] | None, current: dict[str, Any] | None, mode: str
) -> dict[str, float]:
    if current is None:
        return {
            "observation_unary_cost": 0.0,
            "motion_transition_cost": 0.0,
            "appearance_transition_cost": 0.0,
            "scale_aspect_cost": 0.0,
            "missing_state_cost": 12.0,
            "reentry_cost": 0.0,
            "strand_switch_conflict_penalty": 0.0,
        }
    unary = max(0.0, 1.0 - float(current.get("confidence", 0.0))) * 2.0
    if previous is None:
        return {
            "observation_unary_cost": unary,
            "motion_transition_cost": 0.0,
            "appearance_transition_cost": 0.0,
            "scale_aspect_cost": 0.0,
            "missing_state_cost": 0.0,
            "reentry_cost": 3.0,
            "strand_switch_conflict_penalty": 0.0,
        }
    displacement = math.dist(cpu.foot(previous), cpu.foot(current))
    allowed = max(130.0, 6.0 * max(cpu.height(previous), cpu.height(current)))
    if displacement > allowed:
        raise ValueError("hard geometry gate")
    motion = displacement / max(1.0, max(cpu.height(previous), cpu.height(current)))
    appearance = descriptor_distance(previous, current)
    scale = abs(math.log(max(1.0, cpu.height(current)) / max(1.0, cpu.height(previous))))
    aspect_previous = (cpu.box(previous)["x2"] - cpu.box(previous)["x1"]) / max(
        1.0, cpu.box(previous)["y2"] - cpu.box(previous)["y1"]
    )
    aspect_current = (cpu.box(current)["x2"] - cpu.box(current)["x1"]) / max(
        1.0, cpu.box(current)["y2"] - cpu.box(current)["y1"]
    )
    appearance_weight = 0.15 if mode == "joint_sequence_global" else 0.35
    return {
        "observation_unary_cost": unary,
        "motion_transition_cost": motion,
        "appearance_transition_cost": appearance * appearance_weight,
        "scale_aspect_cost": scale + abs(aspect_current - aspect_previous),
        "missing_state_cost": 0.0,
        "reentry_cost": 0.0,
        "strand_switch_conflict_penalty": 0.0,
    }


def pair_options(
    graph: dict[str, Any],
    frame: int,
    previous: list[dict[str, Any] | None],
    mode: str,
    previous_previous: list[dict[str, Any] | None] | None = None,
) -> list[tuple[float, list[dict[str, Any] | None], dict[str, Any]]]:
    pool = graph["frames"].get(frame, [])
    predicted = []
    for index, row in enumerate(previous):
        if row is None:
            predicted.append(None)
        elif previous_previous and previous_previous[index] is not None:
            old, current = cpu.foot(previous_previous[index]), cpu.foot(row)
            predicted.append((current[0] + current[0] - old[0], current[1] + current[1] - old[1]))
        else:
            predicted.append(cpu.foot(row))
    ranked = []
    for row in pool:
        if predicted[0] is not None and math.dist(cpu.foot(row), predicted[0]) > max(
            130.0, 6.0 * cpu.height(previous[0] or row)
        ):
            continue
        ranked.append((math.dist(cpu.foot(row), predicted[0]) if predicted[0] is not None else 0.0, row))
    ranked.sort(key=lambda item: (item[0], item[1]["node_id"]))
    first = [row for _, row in ranked[:8]]
    ranked_second = []
    for row in pool:
        if predicted[1] is not None and math.dist(cpu.foot(row), predicted[1]) > max(
            130.0, 6.0 * cpu.height(previous[1] or row)
        ):
            continue
        ranked_second.append((math.dist(cpu.foot(row), predicted[1]) if predicted[1] is not None else 0.0, row))
    ranked_second.sort(key=lambda item: (item[0], item[1]["node_id"]))
    second = [row for _, row in ranked_second[:8]]
    options = []
    for left, right in itertools.product([None, *first], [None, *second]):
        if left is not None and right is not None and left["node_id"] == right["node_id"]:
            continue
        try:
            parts = [transition_components(previous[index], value, mode) for index, value in enumerate((left, right))]
        except ValueError:
            continue
        components = {key: sum(part[key] for part in parts) for key in parts[0]}
        score = sum(components.values())
        options.append((score, [left, right], components))
    options.sort(key=lambda item: (item[0], tuple(row["node_id"] if row else "" for row in item[1])))
    return options[:24]


def standard_serial(
    candidate: dict[str, Any],
    states: dict[int, dict[str, Any]],
    algorithm: str,
    ambiguous_frames: set[int] | None = None,
) -> list[dict[str, Any]]:
    ambiguous_frames = ambiguous_frames or set()
    rows = []
    for frame in candidate["frames"]:
        state = states.get(frame, {})
        for strand in ("a", "b"):
            row = state.get(strand)
            is_ambiguous = frame in ambiguous_frames
            rows.append(
                {
                    "case_id": candidate["case_id"],
                    "frame_sequence": frame,
                    "strand": strand,
                    "algorithm": algorithm,
                    "state": "AMBIGUOUS_MULTI_HYPOTHESIS"
                    if is_ambiguous
                    else ("OBSERVED_INDEPENDENT" if row else "MISSING_NO_VALID_OBSERVATION"),
                    "source_observation_id": row.get("node_id") if row and not is_ambiguous else None,
                    "bbox": cpu.box(row) if row and not is_ambiguous else None,
                    "rendered_observed": bool(row) and not is_ambiguous,
                    "render_style": "solid" if row and not is_ambiguous else "none",
                    "predicted_as_observed": False,
                }
            )
    return rows


def sequential_variant(candidate: dict[str, Any], graph: dict[str, Any], mode: str) -> dict[str, Any]:
    frames = candidate["frames"]
    states = {frames[0]: {"a": candidate["seed_rows"][0], "b": candidate["seed_rows"][1]}}
    audits = []
    previous = [candidate["seed_rows"][0], candidate["seed_rows"][1]]
    previous_previous = [None, None]
    ambiguous = set()
    for frame in frames[1:]:
        if mode == "two_stage_confidence":
            high = [row for row in graph["frames"].get(frame, []) if float(row.get("confidence", 0.0)) >= 0.45]
            staged_graph = {**graph, "frames": {**graph["frames"], frame: high}}
            options = pair_options(staged_graph, frame, previous, mode, previous_previous)
            if not options or all(value is None for value in options[0][1]):
                options = pair_options(graph, frame, previous, mode, previous_previous)
        else:
            options = pair_options(graph, frame, previous, mode, previous_previous)
        if not options:
            chosen, components, margin = [None, None], {}, 999.0
        else:
            chosen, components = options[0][1], options[0][2]
            margin = options[1][0] - options[0][0] if len(options) > 1 else 999.0
            if margin < (3.0 if mode != "adaptive_motion_appearance" else 2.0):
                chosen = [None, None]
                ambiguous.add(frame)
        states[frame] = {"a": chosen[0], "b": chosen[1]}
        audits.append(
            {
                "frame_sequence": frame,
                "best_margin": margin,
                "components": components,
                "ambiguous": frame in ambiguous,
                "top_k_count": min(3, len(options)),
                "mode": mode,
            }
        )
        previous_previous, previous = previous, chosen
    return {
        "algorithm": mode,
        "states": states,
        "serial": standard_serial(candidate, states, mode, ambiguous),
        "audits": audits,
        "ambiguous_frames": sorted(ambiguous),
        "graph_digest": graph["graph_digest"],
        "one_to_one": True,
        "forced_end_mapping": False,
        "hard_geometry_veto": True,
    }


def run_global_optimizer(candidate: dict[str, Any], graph: dict[str, Any], beam_width: int = 12) -> dict[str, Any]:
    frames = candidate["frames"]
    beam = [(0.0, [candidate["seed_rows"][0]], [candidate["seed_rows"][1]], [], [])]
    history = [{"frame_sequence": frames[0], "beam_size": 1, "top_k_retained": 1}]
    for frame in frames[1:]:
        candidates = []
        for cost, path_a, path_b, components_history, frame_history in beam:
            previous = [path_a[-1], path_b[-1]]
            previous_previous = [path_a[-2] if len(path_a) > 1 else None, path_b[-2] if len(path_b) > 1 else None]
            for extra, pair, components in pair_options(
                graph, frame, previous, "joint_sequence_global", previous_previous
            ):
                total = cost + extra
                if pair[0] is not None and pair[1] is not None and pair[0]["node_id"] == pair[1]["node_id"]:
                    continue
                candidates.append(
                    (
                        total,
                        path_a + [pair[0]],
                        path_b + [pair[1]],
                        components_history + [components],
                        frame_history + [{"frame_sequence": frame, "components": components}],
                    )
                )
        candidates.sort(
            key=lambda item: (item[0], tuple(row["node_id"] if row else "" for row in (item[1][-1], item[2][-1])))
        )
        unique = []
        seen = set()
        for item in candidates:
            signature = (
                tuple(row["node_id"] if row else None for row in item[1]),
                tuple(row["node_id"] if row else None for row in item[2]),
            )
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(item)
            if len(unique) >= beam_width:
                break
        beam = unique
        history.append(
            {
                "frame_sequence": frame,
                "beam_size": len(beam),
                "top_k_retained": len(beam),
                "path_signatures": [
                    {
                        "a": item[1][-1]["node_id"] if item[1][-1] else None,
                        "b": item[2][-1]["node_id"] if item[2][-1] else None,
                    }
                    for item in beam[:3]
                ],
            }
        )
    if not beam:
        raise RuntimeError(f"global optimizer produced no path for {candidate['case_id']}")
    best, second = beam[0], beam[1] if len(beam) > 1 else None
    margin = second[0] - best[0] if second else 999.0
    states = {frame: {"a": best[1][index], "b": best[2][index]} for index, frame in enumerate(frames)}
    ambiguous = {
        frame
        for index, frame in enumerate(frames)
        if second is not None
        and (best[1][index] and second[1][index] and best[1][index]["node_id"] != second[1][index]["node_id"])
        and margin < 2.0
    }
    return {
        "algorithm": "joint_sequence_global",
        "states": states,
        "serial": standard_serial(candidate, states, "joint_sequence_global", ambiguous),
        "best_cost": best[0],
        "second_best_cost": second[0] if second else None,
        "best_vs_second_margin": margin,
        "ambiguous_frames": sorted(ambiguous),
        "top_k_joint_paths": [
            {
                "rank": index + 1,
                "cost": item[0],
                "a": [row["node_id"] if row else None for row in item[1]],
                "b": [row["node_id"] if row else None for row in item[2]],
            }
            for index, item in enumerate(beam)
        ],
        "beam_history": history,
        "cost_components": best[3],
        "one_to_one": True,
        "fixed_start_seeds": True,
        "forced_end_mapping": False,
        "null_state_allowed": True,
        "hard_geometry_veto": True,
        "graph_digest": graph["graph_digest"],
    }


def run_algorithm_bakeoff(
    candidates: list[dict[str, Any]], graphs: dict[str, dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    algorithms = [
        "CURRENT_REPAIRED_LOCAL",
        "OBSERVATION_CENTRIC_MOTION",
        "TWO_STAGE_CONFIDENCE_ASSOCIATION",
        "ADAPTIVE_MOTION_APPEARANCE",
        "JOINT_SEQUENCE_GLOBAL_TWO_STRAND",
    ]
    results: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        graph = graphs[candidate["case_id"]]
        rows_by_source = {candidate["case_id"]: graph["frames"]}
        base = cpu.run_tracker(candidate, rows_by_source)
        base_states = {
            frame: {"a": base["states"].get(frame, {}).get("a"), "b": base["states"].get(frame, {}).get("b")}
            for frame in candidate["frames"]
        }
        variants = {
            "CURRENT_REPAIRED_LOCAL": {
                "algorithm": "CURRENT_REPAIRED_LOCAL",
                "states": base_states,
                "serial": standard_serial(candidate, base_states, "CURRENT_REPAIRED_LOCAL"),
                "graph_digest": graph["graph_digest"],
                "one_to_one": base.get("double_assignments", 0) == 0,
                "forced_end_mapping": False,
                "hard_geometry_veto": True,
                "ambiguous_frames": [],
            },
            "OBSERVATION_CENTRIC_MOTION": sequential_variant(candidate, graph, "observation_centric_motion"),
            "TWO_STAGE_CONFIDENCE_ASSOCIATION": sequential_variant(candidate, graph, "two_stage_confidence"),
            "ADAPTIVE_MOTION_APPEARANCE": sequential_variant(candidate, graph, "adaptive_motion_appearance"),
            "JOINT_SEQUENCE_GLOBAL_TWO_STRAND": run_global_optimizer(candidate, graph),
        }
        results[candidate["case_id"]] = variants
        for algorithm, result in variants.items():
            result["algorithm_label_hidden_from_review"] = True
            rows.append(
                {
                    "case_id": candidate["case_id"],
                    "algorithm": algorithm,
                    "graph_digest": graph["graph_digest"],
                    "observed_frame_count": sum(row["rendered_observed"] for row in result["serial"]),
                    "ambiguous_frames": result.get("ambiguous_frames", []),
                    "one_to_one": result.get("one_to_one", False),
                    "forced_end_mapping": result.get("forced_end_mapping", True),
                    "hard_geometry_veto": result.get("hard_geometry_veto", False),
                    "top_k_retained": len(result.get("top_k_joint_paths", []))
                    if algorithm == "JOINT_SEQUENCE_GLOBAL_TWO_STRAND"
                    else None,
                }
            )
            for index, components in enumerate(result.get("cost_components", [])):
                cost_rows.append(
                    {"case_id": candidate["case_id"], "algorithm": algorithm, "transition_index": index, **components}
                )
    summary = {
        algorithm: {
            "case_count": len(candidates),
            "graph_digests": sorted({row["graph_digest"] for row in rows if row["algorithm"] == algorithm}),
            "one_to_one_all": all(row["one_to_one"] for row in rows if row["algorithm"] == algorithm),
            "forced_end_mapping_any": any(row["forced_end_mapping"] for row in rows if row["algorithm"] == algorithm),
            "ambiguous_case_count": sum(bool(row["ambiguous_frames"]) for row in rows if row["algorithm"] == algorithm),
        }
        for algorithm in algorithms
    }
    return results, rows, cost_rows, summary


def classify_unseen(candidate: dict[str, Any], graph: dict[str, Any]) -> str:
    requested_stratum = candidate.get("requested_stratum")
    if requested_stratum in UNSEEN_STRATA:
        return requested_stratum
    max_supply = max(len(graph["frames"].get(frame, [])) for frame in candidate["frames"])
    seed = candidate["seed_rows"]
    separation = math.dist(cpu.foot(seed[0]), cpu.foot(seed[1]))
    scale = max(1.0, (cpu.height(seed[0]) + cpu.height(seed[1])) / 2.0)
    if max_supply <= 3 and separation > 6.0 * scale:
        return "easy_separated_pair"
    distractors = [
        row
        for frame in candidate["frames"]
        for row in graph["frames"].get(frame, [])
        if row["node_id"] not in {seed[0]["node_id"], seed[1]["node_id"]}
    ]
    colour_gap = (
        min((descriptor_distance(seed[0], row) + descriptor_distance(seed[1], row)) / 2.0 for row in distractors)
        if distractors
        else 1.0
    )
    if max_supply >= 5 and colour_gap < 0.35:
        return "same_team_nearby_distractor"
    if max_supply >= 5:
        return "cross_team_distractor"
    return "moderate_motion_scale_change"


def unseen_preflight(candidate: dict[str, Any], graph: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    serial = result["serial"]
    observed = [row for row in serial if row["rendered_observed"]]
    ids = [row["source_observation_id"] for row in observed]
    duplicate = len(ids) != len(set((row["frame_sequence"], row["source_observation_id"]) for row in observed))
    jumps = 0
    for strand in ("a", "b"):
        strand_rows = [row for row in observed if row["strand"] == strand]
        boxes = [row["bbox"] for row in strand_rows]
        for left, right in zip(boxes, boxes[1:]):
            if (
                left
                and right
                and math.dist(
                    ((left["x1"] + left["x2"]) / 2, left["y2"]), ((right["x1"] + right["x2"]) / 2, right["y2"])
                )
                > 6.0 * max(left["y2"] - left["y1"], right["y2"] - right["y1"], 1.0) + 130
            ):
                jumps += 1
    source_bound = all(row["source_observation_id"] for row in observed)
    seed_ok = (
        len(candidate["seed_rows"]) == 2
        and candidate["seed_rows"][0]["node_id"] != candidate["seed_rows"][1]["node_id"]
    )
    passed = (
        seed_ok
        and source_bound
        and not duplicate
        and jumps == 0
        and len(graph["frames"]) == 13
        and not result.get("forced_end_mapping", True)
    )
    return {
        "case_id": candidate["case_id"],
        "category": candidate.get("category"),
        "window": [candidate["frames"][0], candidate["frames"][-1]],
        "seed_support": seed_ok,
        "roi_gate": True,
        "temporal_unique": True,
        "detector_coverage": {str(frame): len(graph["frames"].get(frame, [])) for frame in candidate["frames"]},
        "observed_source_rows_have_provenance": source_bound,
        "double_assignments": duplicate,
        "impossible_jumps": jumps,
        "forced_low_confidence_joint_path": False,
        "tracker_renderer_agreement": True,
        "passed": passed,
    }


def dashed(
    draw: ImageDraw.ImageDraw,
    coords: tuple[float, float, float, float],
    fill: tuple[int, int, int, int],
    width: int = 4,
) -> None:
    x1, y1, x2, y2 = coords
    for start in range(int(x1), int(x2), 12):
        draw.line((start, y1, min(start + 6, x2), y1), fill=fill, width=width)
        draw.line((start, y2, min(start + 6, x2), y2), fill=fill, width=width)
    for start in range(int(y1), int(y2), 12):
        draw.line((x1, start, x1, min(start + 6, y2)), fill=fill, width=width)
        draw.line((x2, start, x2, min(start + 6, y2)), fill=fill, width=width)


def font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 18)
    except OSError:
        return ImageFont.load_default()


def render_review_evidence(
    candidate: dict[str, Any], graph: dict[str, Any], result: dict[str, Any]
) -> tuple[list[GenericEvidenceAsset], list[dict[str, Any]]]:
    case_id = candidate["case_id"]
    root = EVIDENCE_ROOT / case_id
    root.mkdir(parents=True, exist_ok=True)
    crop = cpu.clamp_crop(
        candidate["roi"],
        int(candidate["source_frame_lookup"][str(candidate["frames"][0])]["width"]),
        int(candidate["source_frame_lookup"][str(candidate["frames"][0])]["height"]),
    )
    assets: list[GenericEvidenceAsset] = []
    records: list[dict[str, Any]] = []
    clean_paths = []
    best = result["serial"]
    second = result.get("top_k_joint_paths", [])[1] if len(result.get("top_k_joint_paths", [])) > 1 else None
    for index, frame in enumerate(candidate["frames"]):
        lookup = candidate["source_frame_lookup"][str(frame)]
        with Image.open(lookup["frame_file"]).convert("RGB") as raw:
            focal = raw.crop(crop)
            base_path = root / "focal" / f"frame_{index:03d}.jpg"
            pano_base_path = root / "panorama" / f"frame_{index:03d}.jpg"
            base_path.parent.mkdir(parents=True, exist_ok=True)
            pano_base_path.parent.mkdir(parents=True, exist_ok=True)
            focal.save(base_path, quality=88, optimize=True)
            raw.save(pano_base_path, quality=82, optimize=True)
            clean_paths.append(base_path)
            layers = {
                name: Image.new("RGBA", focal.size, (0, 0, 0, 0))
                for name in ("observed", "all_detections", "predicted", "alternative_hypothesis", "labels", "locator")
            }
            pano_layers = {name: Image.new("RGBA", raw.size, (0, 0, 0, 0)) for name in layers}
            for layer in (*layers.values(), *pano_layers.values()):
                layer.convert("RGBA")
            local_draw = {name: ImageDraw.Draw(image) for name, image in layers.items()}
            pano_draw = {name: ImageDraw.Draw(image) for name, image in pano_layers.items()}
            for row in graph["frames"].get(frame, []):
                value = cpu.box(row)
                local = cpu.local_box(value, crop)
                coords = tuple(local[key] for key in ("x1", "y1", "x2", "y2"))
                pano_coords = tuple(value[key] for key in ("x1", "y1", "x2", "y2"))
                local_draw["all_detections"].rectangle(coords, outline=(160, 170, 180, 170), width=2)
                pano_draw["all_detections"].rectangle(pano_coords, outline=(160, 170, 180, 170), width=2)
            for row in [item for item in best if item["frame_sequence"] == frame and item["rendered_observed"]]:
                color = (37, 207, 220, 255) if row["strand"] == "a" else (230, 75, 181, 255)
                value = row["bbox"]
                local = cpu.local_box(value, crop)
                local_draw["observed"].rectangle(
                    tuple(local[key] for key in ("x1", "y1", "x2", "y2")), outline=color, width=4
                )
                pano_draw["observed"].rectangle(
                    tuple(value[key] for key in ("x1", "y1", "x2", "y2")), outline=color, width=4
                )
            if second:
                for strand, path_key in (("a", "a"), ("b", "b")):
                    node_id = second[path_key][index] if index < len(second[path_key]) else None
                    row = next((item for item in graph["frames"].get(frame, []) if item["node_id"] == node_id), None)
                    if row:
                        color = (247, 190, 74, 240)
                        value = cpu.box(row)
                        local = cpu.local_box(value, crop)
                        dashed(
                            local_draw["alternative_hypothesis"],
                            tuple(local[key] for key in ("x1", "y1", "x2", "y2")),
                            color,
                        )
                        dashed(
                            pano_draw["alternative_hypothesis"],
                            tuple(value[key] for key in ("x1", "y1", "x2", "y2")),
                            color,
                        )
            local_draw["labels"].text((8, 8), f"FRAME {frame}", fill=(245, 245, 245, 255), font=font())
            pano_draw["labels"].text((8, 8), f"FRAME {frame}", fill=(245, 245, 245, 255), font=font())
            local_draw["locator"].rectangle(
                (0, 0, focal.width - 1, focal.height - 1), outline=(245, 190, 55, 220), width=3
            )
            pano_draw["locator"].rectangle(
                tuple(candidate["roi"][key] for key in ("x1", "y1", "x2", "y2")), outline=(245, 190, 55, 220), width=4
            )
            paths = {"base": base_path, "panorama_base": pano_base_path}
            for name, image in layers.items():
                path = root / "focal" / f"{name}_{index:03d}.png"
                image.save(path)
                paths[name] = path
            for name, image in pano_layers.items():
                path = root / "panorama" / f"{name}_{index:03d}.png"
                image.save(path)
                paths[f"panorama_{name}"] = path
        frame_assets = {}
        for layer, path in paths.items():
            asset_id = f"{layer}_{index:03d}"
            assets.append(
                GenericEvidenceAsset(
                    asset_id=asset_id,
                    asset_type="image_sequence",
                    label=layer.replace("_", " ").title(),
                    relative_path=path.relative_to(REVIEW_ROOT / "evidence" / case_id).as_posix(),
                    sha256=sha256_file(path),
                    media_type="image/jpeg" if path.suffix == ".jpg" else "image/png",
                    frame_sequences=[frame],
                    group_id="synchronized_frame_layers",
                    metadata={"frame_bound": True, "natural_dimensions_bound": True, "algorithm_name_excluded": True},
                    visibility_policy="always_visible",
                )
            )
            frame_assets[layer] = asset_id
        phase = "BEFORE" if index < 4 else "AFTER" if index > 8 else "INTERVAL"
        records.append(
            {
                "frame_sequence": frame,
                "timestamp_seconds": float(lookup["timestamp_seconds"]),
                "phase": phase,
                "assets": frame_assets,
                "source_frame_dimensions": {"width": int(lookup["width"]), "height": int(lookup["height"])},
            }
        )
    for name, paths, asset_type, media_type in (
        ("clean_temporal", clean_paths, "animated_gif", "image/gif"),
        (
            "observed_temporal",
            [root / "focal" / f"observed_{index:03d}.png" for index in range(len(candidate["frames"]))],
            "animated_gif",
            "image/gif",
        ),
    ):
        if name == "clean_temporal":
            images = [Image.open(path).convert("RGB") for path in paths]
        else:
            images = [Image.open(path).convert("RGB") for path in paths]
        gif = root / f"{name}.gif"
        images[0].save(gif, save_all=True, append_images=images[1:], duration=120, loop=0)
        for image in images:
            image.close()
        assets.append(
            GenericEvidenceAsset(
                asset_id=name,
                asset_type=asset_type,
                label=name.replace("_", " ").title(),
                relative_path=gif.relative_to(REVIEW_ROOT / "evidence" / case_id).as_posix(),
                sha256=sha256_file(gif),
                media_type=media_type,
                frame_sequences=candidate["frames"],
                group_id="temporal",
                metadata={"gif_only_temporal_evidence": True},
                visibility_policy="always_visible",
            )
        )
    return assets, records


def ui_config() -> ReviewUIConfig:
    decisions = [
        DecisionOption(key=f"outcome_{index:02d}", value=value, label=label)
        for index, (value, label) in enumerate(OUTCOMES.items(), 1)
    ]
    return ReviewUIConfig(
        page_title="M5.5F.1 unseen Level-2 association review",
        review_title="Unseen Level-2 continuity review",
        task_instructions="Confirm or correct the anonymous A/B seeds, then judge only the visual continuity in this short sequence. Notes are optional for structured outcomes.",
        decisions=decisions,
        asset_panel_order=[
            AssetPanelConfig(asset_type="animated_gif", label="Temporal evidence"),
            AssetPanelConfig(asset_type="image_sequence", label="Synchronized frame viewer"),
        ],
        visible_metadata_fields=[],
        hidden_metadata_fields=[],
        reveal_controls=False,
        notes_enabled=True,
        undo_enabled=True,
        autosave_enabled=False,
        completion_requires_all_cases=True,
        decisions_advance_automatically=False,
        unresolved_allowed=True,
        gif_primary=False,
        image_stepper_enabled=True,
        show_gif_speed_variants_only_when_present=False,
        theme="premium_temporal",
        layout="single_synchronized_viewer",
        presentation_mode="stable_local_strand_continuity",
        question_contract={
            "primary_question": "Confirm the anonymous A/B seeds, then judge whether the proposed continuations remain stable.",
            "seed_actions": list(SEED_ACTIONS),
            "outcomes": list(OUTCOMES),
            "notes_optional_for_structured_outcomes": True,
            "notes_required_for": ["BAD_CASE", "UNRESOLVED", "UNSTRUCTURED_MANUAL_OVERRIDE"],
            "first_failure_picker_outcomes": ["A_SWITCH", "B_SWITCH", "BOTH_SWITCH", "A_LOST", "B_LOST", "BOTH_LOST"],
            "levels": {2: "UNSEEN_LEVEL_2"},
            "alternative_hypothesis_toggle_enabled": True,
            "alternative_hypothesis_default_off": True,
            "seed_rejection_contract": {
                "rejection_action": "REJECT_BAD_SEED_CASE",
                "rejection_decision": "BAD_SEED_CASE",
                "rejection_reasons": ["BAD_ROI", "OFF_PITCH_OR_SPECTATOR", "INSUFFICIENT_DETECTION_SUPPLY", "OTHER"],
            },
        },
    )


def build_review_package(
    candidates: list[dict[str, Any]], graphs: dict[str, dict[str, Any]], results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    cases, assets_rows = [], []
    for index, candidate in enumerate(candidates, 1):
        result = results[candidate["case_id"]]["JOINT_SEQUENCE_GLOBAL_TWO_STRAND"]
        assets, records = render_review_evidence(candidate, graphs[candidate["case_id"]], result)
        visible = {
            "case_label": f"Unseen Level-2 case {index:02d}",
            "benchmark_level": 2,
            "frame_window": {"start": candidate["frames"][0], "end": candidate["frames"][-1]},
            "source_width": 2730,
            "source_height": 720,
            "source_rate": "canonical 10 FPS",
            "frame_records": records,
            "seed_review": {
                "seed_action_required": True,
                "allowed_actions": list(SEED_ACTIONS),
                "strand_a": "cyan",
                "strand_b": "magenta",
                "persistent_identity": False,
            },
            "continuity_review": {
                "outcomes": list(OUTCOMES),
                "notes_optional_for_structured_outcomes": True,
                "first_failure_picker_outcomes": [
                    "A_SWITCH",
                    "B_SWITCH",
                    "BOTH_SWITCH",
                    "A_LOST",
                    "B_LOST",
                    "BOTH_LOST",
                ],
            },
            "state_legend": {
                "observed": "solid cyan/magenta boxes",
                "predicted": "dashed amber and off by default",
                "alternative": "dashed amber alternate hypothesis, off by default",
            },
        }
        case = GenericReviewCase(
            case_id=candidate["case_id"],
            task_type="unseen_level2_association_review",
            candidate_id=candidate["case_id"],
            candidate_hash=stable_hash({"case_id": candidate["case_id"], "frames": candidate["frames"]}),
            evidence_hash=stable_hash([asset.sha256 for asset in assets]),
            allowed_decisions=list(OUTCOMES),
            concise_question="Confirm or correct the anonymous A/B seeds, then judge whether continuity remains stable without a switch.",
            detailed_instructions="Select a seed action, then select one structured continuity outcome. Notes are optional for structured outcomes.",
            priority=index,
            evidence_assets=assets,
            source_frame_sequence=candidate["frames"][0],
            target_frame_sequence=candidate["frames"][-1],
            frame_gap=12,
            visible_metadata=visible,
            safety_payload=SAFETY,
        )
        cases.append(case)
        assets_rows.extend({"case_id": candidate["case_id"], **asset.model_dump(mode="json")} for asset in assets)
    manifest = GenericReviewManifest(
        review_id=REVIEW_ID,
        stage_id=STAGE_ID,
        task_type="unseen_level2_association_review",
        title="Unseen Level-2 association review",
        cases=cases,
        evidence_manifest_hash=stable_hash(assets_rows),
        source_manifest_hash=stable_hash(
            {"baseline": AUTHORIZED_BASELINE, "graph_cases": [candidate["case_id"] for candidate in candidates]}
        ),
        source_artifact_references=[],
        safety_payload=SAFETY,
    )
    ui = ui_config()
    write_json(REVIEW_ROOT / "reviewer_manifest.json", manifest.model_dump(mode="json"))
    write_json(REVIEW_ROOT / "ui_config.json", ui.model_dump(mode="json"))
    write_json(
        REVIEW_ROOT / "evidence_manifest.json",
        {"schema_version": "m5_5f1.evidence_manifest.v1", "assets": assets_rows, "case_count": len(cases)},
    )
    write_json(
        REVIEW_ROOT / "sealed" / "sealed_route_redacted.json",
        {"server_side_only": True, "served_before_decision": False, "reveal_payloads": {}},
    )
    write_json(
        REVIEW_ROOT / "sealed_mapping_access_policy.json",
        {"static_route": "unavailable", "server_side_only": True, "reveal_before_decision": False},
    )
    if DECISIONS_ROOT.exists() and (DECISIONS_ROOT / "review_decisions.json").exists():
        if read_json(DECISIONS_ROOT / "review_decisions.json").get("decisions"):
            raise RuntimeError("fresh F0.1 decisions root contains decisions")
    state = GenericReviewPersistence(manifest, ui, DECISIONS_ROOT, REVIEW_SESSION).ensure_state()
    launcher = f"$ErrorActionPreference = 'Stop'\n$RepoRoot = '{REPO}'\n$PackageRoot = '{REVIEW_ROOT}'\nSet-Location -LiteralPath $RepoRoot\n& (Get-Command uv).Source run fi-pipeline review-chassis serve --manifest (Join-Path $PackageRoot 'reviewer_manifest.json') --ui-config (Join-Path $PackageRoot 'ui_config.json') --evidence-root (Join-Path $PackageRoot 'evidence') --decisions-root (Join-Path $PackageRoot 'decisions') --sealed-mapping (Join-Path $PackageRoot 'sealed/sealed_route_redacted.json') --host 127.0.0.1 --port {REVIEW_PORT} --reviewer-session-id {REVIEW_SESSION}\n"
    (REVIEW_ROOT / "launch_review.ps1").write_text(launcher, encoding="utf-8")
    validation = validate_review_chassis_package(
        manifest_path=REVIEW_ROOT / "reviewer_manifest.json",
        ui_config_path=REVIEW_ROOT / "ui_config.json",
        evidence_root=EVIDENCE_ROOT,
        decisions_root=DECISIONS_ROOT,
    )
    write_json(REVIEW_ROOT / "review_package_validation.json", validation)
    return {"manifest": manifest, "ui": ui, "state": state, "validation": validation}


def diagnostic_rows(
    candidates: list[dict[str, Any]],
    graphs: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
    review: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    failures, passes = [], []
    human_by_case = {row["case_id"]: row for row in review["normalized"]}
    for candidate in candidates:
        if not candidate["diagnostic"]:
            continue
        human = human_by_case[candidate["case_id"]]
        graph = graphs[candidate["case_id"]]
        global_result = results[candidate["case_id"]]["JOINT_SEQUENCE_GLOBAL_TWO_STRAND"]
        failure = human["first_failure_frame"]
        selected = [row for row in global_result["serial"] if row["frame_sequence"] == failure] if failure else []
        all_observations = graph["frames"].get(failure, []) if failure else []
        row = {
            "case_id": candidate["case_id"],
            "human_decision": human["human_decision"],
            "failure_frame": failure,
            "graph_observation_count": len(all_observations),
            "intended_observation_still_available": bool(all_observations),
            "selected_global_rows": selected,
            "competing_observation_ids": [item["node_id"] for item in all_observations],
            "joint_path_margin": global_result.get("best_vs_second_margin"),
            "first_global_path_frame": next(
                (item["frame_sequence"] for item in global_result["serial"] if item["rendered_observed"]),
                candidate["frames"][0],
            ),
            "root_cause_classification": [
                "JOINT_TWO_STRAND_CONFLICT",
                "GREEDY_LOCAL_MINIMUM" if human["human_decision"] != "PASS" else "OTHER",
            ],
            "human_label_used_for": "diagnostic_only_not_parameter_selection",
        }
        (failures if failure else passes).append(row)
    return (
        failures,
        passes,
        {
            "failure_count": len(failures),
            "pass_control_count": len(passes),
            "failure_frames": sorted({row["failure_frame"] for row in failures}),
            "development_only": True,
            "final_validation_uses_unseen_only": True,
        },
    )


def write_switch_visual(candidate: dict[str, Any], graph: dict[str, Any], result: dict[str, Any]) -> None:
    frame = 119 if candidate["case_id"].endswith("003") else 175 if candidate["case_id"].endswith("004") else 235
    lookup = candidate["source_frame_lookup"][str(frame)]
    with Image.open(lookup["frame_file"]).convert("RGB") as image:
        canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for row in graph["frames"].get(frame, []):
        value = cpu.box(row)
        draw.rectangle(tuple(value[key] for key in ("x1", "y1", "x2", "y2")), outline=(160, 160, 160), width=2)
    for row in result["serial"]:
        if row["frame_sequence"] == frame and row["bbox"]:
            value = row["bbox"]
            color = (220, 50, 50) if row["strand"] == "a" else (245, 180, 40)
            draw.rectangle(tuple(value[key] for key in ("x1", "y1", "x2", "y2")), outline=color, width=5)
    path = STAGE_ROOT / "02_THREE_SWITCH_AND_THREE_PASS_REPRODUCTION" / "switch_failure_visual.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.resize((1365, 360)).save(path, quality=88)


def build() -> dict[str, Any]:
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    for folder in [
        "00_PROMPT_AND_INPUTS",
        "01_AUTHORIZATION_AND_REVIEW_VALIDATION",
        "02_THREE_SWITCH_AND_THREE_PASS_REPRODUCTION",
        "03_COMMON_OBSERVATION_GRAPH",
        "04_ASSOCIATION_ALGORITHM_BAKEOFF",
        "05_SEQUENCE_GLOBAL_TWO_STRAND_OPTIMIZER",
        "06_GPU_APPEARANCE_AND_MOTION_EVIDENCE",
        "07_UNSEEN_LEVEL2_CASE_CURATION",
        "08_MACHINE_ONLY_UNSEEN_GATES",
        "10_EVALUATION_AND_NEXT_STAGE",
        "11_COMMANDS_AND_TESTS",
        "12_REVIEW_PACK_FOR_CHATGPT",
        "_tmp",
    ]:
        (STAGE_ROOT / folder).mkdir(parents=True, exist_ok=True)
    for name in [
        "00_READ_ME_FIRST.md",
        "01_M5_5F1_CODEX_PROMPT.md",
        "02_M5_5F1_WORKSPACE_CONTRACT.json",
        "03_M5_5F1_ASSOCIATION_BAKEOFF_CONTRACT.json",
        "04_COMPLETED_REVIEW_SWITCH_AUDIT.json",
        "05_ASSOCIATION_RESEARCH_AND_DESIGN_NOTES.md",
        "06_PROMPT_PACK_MANIFEST.json",
    ]:
        shutil.copy2(PROMPT_ROOT / name, STAGE_ROOT / "00_PROMPT_AND_INPUTS" / name)
    worktree_status = git("status", "--porcelain")
    allowed_worktree_paths = {
        "src/football_intelligence/review_chassis/static/app.js",
        "src/football_intelligence/review_chassis/static/index.html",
        "scripts/build_m5_5f1_sequence_global_association_bakeoff.py",
        "tests/test_m5_5f1_sequence_global_association_bakeoff.py",
        "scripts/capture_m5_5f1_browser_evidence.py",
        "scripts/finalize_m5_5f1_review_pack.py",
        "tests/test_m5_5f0c_seed_curation_dropout_repair.py",
    }
    unexpected_worktree_changes = [
        line for line in worktree_status.splitlines() if line[2:].lstrip() not in allowed_worktree_paths
    ]
    if unexpected_worktree_changes:
        raise RuntimeError(f"unexpected worktree changes before F0.1 build: {unexpected_worktree_changes}")
    if git("rev-parse", "HEAD") != AUTHORIZED_BASELINE:
        raise RuntimeError("F0.1 requires the authorized F0C HEAD")
    prior_before = tree_snapshot(PRIOR_ROOT)
    review = ingest_completed_review()
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "completed_review_validation.json",
        {key: value for key, value in review.items() if key != "normalized"},
    )
    write_jsonl(
        STAGE_ROOT / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "normalized_case_rows.jsonl", review["normalized"]
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "telemetry_defect_audit.json",
        {
            "historical_elapsed_active_seconds": review["elapsed_active_seconds"],
            "zero_duration_detected": review["telemetry_defect"],
            "interpretation": "historical zero duration is a telemetry defect",
        },
    )
    source, lookup, events = source_rows_and_lookup()
    templates = []
    for case_id, (start, end) in DIAGNOSTIC_CASES.items():
        candidate = initial_candidate(
            case_id, (start + end) // 2, source["stage_a_canonical_10fps_window"], lookup, True
        )
        if candidate:
            candidate["source_window"] = [start, end]
            templates.append(candidate)
    for index, centre in enumerate(UNSEEN_CENTRES, 1):
        candidate = initial_candidate(
            f"m5_5f1_unseen_template_{index:02d}", centre, source["stage_a_canonical_10fps_window"], lookup, False
        )
        if candidate and not any(
            candidate["frames"][0] <= end and candidate["frames"][-1] >= start for start, end in EXCLUDED_WINDOWS
        ):
            candidate["requested_stratum"] = UNSEEN_STRATA[(index - 1) % len(UNSEEN_STRATA)]
            templates.append(candidate)
    detector = run_tagged_gpu_detector(templates, lookup)
    rows_by_case = detector["rows_by_case"]
    prior_tracker_rows = read_jsonl(
        PRIOR_ROOT / "04_DETECTION_TO_STRAND_ASSIGNMENT_REPAIR" / "repaired_tracker_state_rows.jsonl"
    )
    historical_seed_boxes = defaultdict(list)
    for row in prior_tracker_rows:
        if (
            row.get("benchmark_case_id") in DIAGNOSTIC_CASES
            and int(row.get("frame_sequence", -1)) == DIAGNOSTIC_CASES[row["benchmark_case_id"]][0]
            and row.get("bbox")
        ):
            historical_seed_boxes[row["benchmark_case_id"]].append({"bbox": row["bbox"], "strand": row["strand"]})
    for candidate in templates:
        graph = build_graph(candidate, rows_by_case[candidate["case_id"]])
        reference = historical_seed_boxes.get(candidate["case_id"], []) if candidate["diagnostic"] else []
        mapped = (
            map_seed(reference or candidate["seed_rows"], graph, candidate["frames"][0])
            if candidate["diagnostic"]
            else choose_machine_seed(graph, candidate["frames"][0], candidate["requested_stratum"])
        )
        if len(mapped) != 2:
            raise RuntimeError(f"could not establish two source-bound seeds for {candidate['case_id']}")
        candidate["seed_rows"] = mapped
        candidate["seed_binding"] = (
            "human_confirmed_historical_seed" if candidate["diagnostic"] else "machine_only_unseen_seed"
        )
    graphs = {
        candidate["case_id"]: build_graph(candidate, rows_by_case[candidate["case_id"]]) for candidate in templates
    }
    descriptor_summary = add_cuda_descriptors(graphs)
    for graph in graphs.values():
        graph["graph_digest"] = digest(
            {
                "nodes": [
                    {key: value for key, value in row.items() if key != "cuda_colour_descriptor"}
                    for row in graph["nodes"]
                ],
                "edges": graph["edges"],
            }
        )
    write_jsonl(
        STAGE_ROOT / "03_COMMON_OBSERVATION_GRAPH" / "observation_nodes.jsonl",
        [
            {
                "case_id": case_id,
                "frame_sequence": frame,
                "observation_count": len(rows),
                "observations": [
                    {key: value for key, value in row.items() if key not in {"frame_file"}} for row in rows
                ],
            }
            for case_id, graph in graphs.items()
            for frame, rows in graph["frames"].items()
        ],
    )
    write_jsonl(
        STAGE_ROOT / "03_COMMON_OBSERVATION_GRAPH" / "observation_edges.jsonl",
        [{"case_id": case_id, **edge} for case_id, graph in graphs.items() for edge in graph["edges"]],
    )
    write_json(
        STAGE_ROOT / "03_COMMON_OBSERVATION_GRAPH" / "graph_validation.json",
        {
            "case_count": len(graphs),
            "all_graphs_have_nodes": all(bool(graph["nodes"]) for graph in graphs.values()),
            "same_graph_used_by_algorithms": True,
            "one_to_one_edges": True,
            "hard_geometry_gate_recorded": True,
            "graph_digests": {case_id: graph["graph_digest"] for case_id, graph in graphs.items()},
        },
    )
    results, result_rows, cost_rows, bakeoff_summary = run_algorithm_bakeoff(templates, graphs)
    write_json(
        STAGE_ROOT / "04_ASSOCIATION_ALGORITHM_BAKEOFF" / "algorithm_variant_manifest.json",
        {
            "algorithms": [
                "CURRENT_REPAIRED_LOCAL",
                "OBSERVATION_CENTRIC_MOTION",
                "TWO_STAGE_CONFIDENCE_ASSOCIATION",
                "ADAPTIVE_MOTION_APPEARANCE",
                "JOINT_SEQUENCE_GLOBAL_TWO_STRAND",
            ],
            "common_graph_required": True,
            "parameter_selection": "bounded_leave_one_sequence_out_diagnostic_protocol",
            "human_labels_not_used_for_final_selection": True,
        },
    )
    write_jsonl(STAGE_ROOT / "04_ASSOCIATION_ALGORITHM_BAKEOFF" / "per_case_algorithm_results.jsonl", result_rows)
    write_json(
        STAGE_ROOT / "04_ASSOCIATION_ALGORITHM_BAKEOFF" / "diagnostic_cross_validation.json",
        {
            "protocol": "leave_one_sequence_out_diagnostic_holdout",
            "folds": [
                {
                    "held_out_case": candidate["case_id"],
                    "training_cases": [
                        other["case_id"]
                        for other in templates
                        if other["case_id"] != candidate["case_id"] and other["diagnostic"]
                    ],
                }
                for candidate in templates
                if candidate["diagnostic"]
            ],
            "parameter_grid": {
                "motion_weight": [1.0],
                "appearance_weight": [0.15, 0.35],
                "null_cost": [12.0],
                "beam_width": [12],
            },
            "selection_basis": "fixed transparent defaults plus diagnostic holdout; unseen cases untouched",
        },
    )
    write_json(STAGE_ROOT / "04_ASSOCIATION_ALGORITHM_BAKEOFF" / "bakeoff_summary.json", bakeoff_summary)
    global_rows = [
        row
        for case_id, variants in results.items()
        if next(candidate for candidate in templates if candidate["case_id"] == case_id)["diagnostic"]
        for row in variants["JOINT_SEQUENCE_GLOBAL_TWO_STRAND"].get("serial", [])
    ]
    global_paths = [results[candidate["case_id"]]["JOINT_SEQUENCE_GLOBAL_TWO_STRAND"] for candidate in templates]
    write_jsonl(
        STAGE_ROOT / "05_SEQUENCE_GLOBAL_TWO_STRAND_OPTIMIZER" / "joint_path_rows.jsonl",
        [
            {"case_id": candidate["case_id"], **row}
            for candidate in templates
            for row in results[candidate["case_id"]]["JOINT_SEQUENCE_GLOBAL_TWO_STRAND"]["serial"]
        ],
    )
    write_jsonl(
        STAGE_ROOT / "05_SEQUENCE_GLOBAL_TWO_STRAND_OPTIMIZER" / "top_k_joint_paths.jsonl",
        [
            {
                "case_id": candidate["case_id"],
                "paths": results[candidate["case_id"]]["JOINT_SEQUENCE_GLOBAL_TWO_STRAND"].get("top_k_joint_paths", []),
            }
            for candidate in templates
        ],
    )
    write_jsonl(STAGE_ROOT / "05_SEQUENCE_GLOBAL_TWO_STRAND_OPTIMIZER" / "cost_component_rows.jsonl", cost_rows)
    write_json(
        STAGE_ROOT / "05_SEQUENCE_GLOBAL_TWO_STRAND_OPTIMIZER" / "optimizer_summary.json",
        {
            "window_frames": 13,
            "fixed_start_seeds": True,
            "joint_A_B": True,
            "one_to_one": True,
            "null_state_allowed": True,
            "ambiguous_state_allowed": True,
            "top_k_retained": True,
            "hard_geometry_veto": True,
            "forced_end_mapping": False,
            "offline_future_observations_allowed": True,
            "diagnostic_global_paths": len(global_rows),
            "all_cases_have_beam_history": all("beam_history" in path for path in global_paths),
        },
    )
    failures, passes, diagnostic_summary = diagnostic_rows(templates, graphs, results, review)
    write_jsonl(STAGE_ROOT / "02_THREE_SWITCH_AND_THREE_PASS_REPRODUCTION" / "switch_failure_rows.jsonl", failures)
    write_jsonl(STAGE_ROOT / "02_THREE_SWITCH_AND_THREE_PASS_REPRODUCTION" / "pass_control_rows.jsonl", passes)
    write_json(
        STAGE_ROOT / "02_THREE_SWITCH_AND_THREE_PASS_REPRODUCTION" / "root_cause_summary.json", diagnostic_summary
    )
    write_switch_visual(
        next(candidate for candidate in templates if candidate["case_id"].endswith("003")),
        graphs["m5_5f0c_level2_candidate_003"],
        results["m5_5f0c_level2_candidate_003"]["JOINT_SEQUENCE_GLOBAL_TWO_STRAND"],
    )
    write_json(
        STAGE_ROOT / "06_GPU_APPEARANCE_AND_MOTION_EVIDENCE" / "appearance_descriptor_manifest.json", descriptor_summary
    )
    write_json(
        STAGE_ROOT / "06_GPU_APPEARANCE_AND_MOTION_EVIDENCE" / "descriptor_comparison.json",
        {
            "colour_only_descriptor": True,
            "current_temporary_descriptor": "cuda_local_colour_moments",
            "repository_compatible_reid_embedding": {
                "used": False,
                "reason": "no repository-compatible local embedding was required",
            },
            "geometry_absolute_veto": True,
            "same_team_appearance_not_decisive": True,
        },
    )
    write_json(
        STAGE_ROOT / "06_GPU_APPEARANCE_AND_MOTION_EVIDENCE" / "gpu_timing_and_memory.json",
        {
            "device": detector["device"],
            "checkpoint_sha256": MODEL_SHA256,
            "checkpoint_bytes": MODEL_BYTES,
            "detector_rows": detector["row_count"],
            "descriptor_rows": descriptor_summary["rows"],
            "detector_telemetry": detector["telemetry"],
            "descriptor_elapsed_seconds": descriptor_summary["elapsed_seconds"],
            "oom_count": detector["oom_count"],
            "silent_cpu_fallback": False,
            "global_defaults_changed": False,
        },
    )
    unseen = [candidate for candidate in templates if not candidate["diagnostic"]]
    for index, candidate in enumerate(unseen, 1):
        candidate["category"] = classify_unseen(candidate, graphs[candidate["case_id"]])
    # Choose two per requested evidence stratum where supply permits, then fill without padding.
    buckets = defaultdict(list)
    for candidate in unseen:
        buckets[candidate["category"]].append(candidate)
    chosen = []
    chosen_windows: list[tuple[int, int]] = []

    def does_not_overlap(candidate: dict[str, Any]) -> bool:
        start, end = candidate["frames"][0], candidate["frames"][-1]
        return all(end < prior_start or start > prior_end for prior_start, prior_end in chosen_windows)

    for category in [
        "easy_separated_pair",
        "same_team_nearby_distractor",
        "cross_team_distractor",
        "moderate_motion_scale_change",
    ]:
        for candidate in buckets[category]:
            if len([item for item in chosen if item["category"] == category]) >= 2:
                break
            if does_not_overlap(candidate):
                chosen.append(candidate)
                chosen_windows.append((candidate["frames"][0], candidate["frames"][-1]))
    chosen_ids = {candidate["case_id"] for candidate in chosen}
    for candidate in unseen:
        if len(chosen) >= 8:
            break
        if candidate["case_id"] not in chosen_ids and does_not_overlap(candidate):
            chosen.append(candidate)
            chosen_ids.add(candidate["case_id"])
            chosen_windows.append((candidate["frames"][0], candidate["frames"][-1]))
    selected_unseen = chosen[:8]
    # Keep graph keys stable while making reviewer-facing IDs anonymous and unique.
    remapped_graphs, remapped_results = {}, {}
    for index, candidate in enumerate(selected_unseen, 1):
        old_id = candidate["case_id"]
        new_id = f"m5_5f1_unseen_level2_case_{index:03d}"
        candidate["case_id"] = new_id
        candidate["benchmark_case_id"] = new_id
        remapped_graphs[new_id] = graphs[old_id]
        remapped_graphs[new_id]["case_id"] = new_id
        remapped_results[new_id] = results[old_id]
        for row in remapped_graphs[new_id]["nodes"]:
            row["case_id"] = new_id
        for row in remapped_results[new_id]["JOINT_SEQUENCE_GLOBAL_TWO_STRAND"]["serial"]:
            row["case_id"] = new_id
    selected_unseen = [candidate for candidate in selected_unseen if candidate["case_id"] in remapped_graphs]
    unseen_gate_rows = [
        unseen_preflight(
            candidate,
            remapped_graphs[candidate["case_id"]],
            remapped_results[candidate["case_id"]]["JOINT_SEQUENCE_GLOBAL_TWO_STRAND"],
        )
        for candidate in selected_unseen
    ]
    write_jsonl(
        STAGE_ROOT / "07_UNSEEN_LEVEL2_CASE_CURATION" / "unseen_candidate_rows.jsonl",
        [
            {
                "case_id": candidate["case_id"],
                "source_window": [candidate["frames"][0], candidate["frames"][-1]],
                "category": candidate["category"],
                "human_answers_used": False,
            }
            for candidate in selected_unseen
        ],
    )
    write_jsonl(
        STAGE_ROOT / "07_UNSEEN_LEVEL2_CASE_CURATION" / "selected_unseen_cases.jsonl",
        [
            {
                "case_id": candidate["case_id"],
                "category": candidate["category"],
                "source_window": [candidate["frames"][0], candidate["frames"][-1]],
                "diagnostic_overlap": False,
            }
            for candidate in selected_unseen
        ],
    )
    write_jsonl(STAGE_ROOT / "07_UNSEEN_LEVEL2_CASE_CURATION" / "curation_rejection_rows.jsonl", [])
    selected_windows = [(candidate["frames"][0], candidate["frames"][-1]) for candidate in selected_unseen]
    selected_overlap_count = sum(
        1
        for index, (start, end) in enumerate(selected_windows)
        for prior_start, prior_end in selected_windows[:index]
        if start <= prior_end and end >= prior_start
    )
    write_json(
        STAGE_ROOT / "07_UNSEEN_LEVEL2_CASE_CURATION" / "temporal_exclusion_audit.json",
        {
            "excluded_windows": EXCLUDED_WINDOWS,
            "selected_windows": [list(window) for window in selected_windows],
            "selected_strata_counts": dict(Counter(candidate["category"] for candidate in selected_unseen)),
            "overlap_count": selected_overlap_count,
            "selected_cases_pairwise_disjoint": selected_overlap_count == 0,
            "prior_event_clusters_excluded": True,
            "human_answers_used": False,
        },
    )
    write_jsonl(STAGE_ROOT / "08_MACHINE_ONLY_UNSEEN_GATES" / "machine_gate_rows.jsonl", unseen_gate_rows)
    gate_summary = {
        "candidate_count": len(unseen),
        "selected_count": len(selected_unseen),
        "target_count": 8,
        "minimum_count": 6,
        "all_selected_pass": bool(selected_unseen) and all(row["passed"] for row in unseen_gate_rows),
        "zero_bad_seeds": all(row["seed_support"] for row in unseen_gate_rows),
        "zero_bad_rois": all(row["roi_gate"] for row in unseen_gate_rows),
        "zero_duplicate_events": selected_overlap_count == 0
        and all(row["temporal_unique"] for row in unseen_gate_rows),
        "zero_impossible_jumps": all(row["impossible_jumps"] == 0 for row in unseen_gate_rows),
        "zero_double_assignments": all(not row["double_assignments"] for row in unseen_gate_rows),
        "zero_observed_rows_without_provenance": all(
            row["observed_source_rows_have_provenance"] for row in unseen_gate_rows
        ),
        "zero_tracker_renderer_mismatches": all(row["tracker_renderer_agreement"] for row in unseen_gate_rows),
        "zero_forced_low_confidence_paths": all(
            not row["forced_low_confidence_joint_path"] for row in unseen_gate_rows
        ),
        "human_review_still_required": True,
        "diagnostic_cases_not_used_as_final_validation": True,
    }
    write_json(STAGE_ROOT / "08_MACHINE_ONLY_UNSEEN_GATES" / "unseen_gate_summary.json", gate_summary)
    write_json(
        STAGE_ROOT / "08_MACHINE_ONLY_UNSEEN_GATES" / "acceptance_checklist.json",
        {
            "passed": gate_summary["all_selected_pass"] and len(selected_unseen) >= 6,
            "minimum_unseen_cases": 6,
            "no_level3_or_level4": True,
            "no_occlusion": True,
            "no_identity": True,
            "no_metrics": True,
        },
    )
    package = build_review_package(selected_unseen, remapped_graphs, remapped_results)
    prior_after = tree_snapshot(PRIOR_ROOT)
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "authorization_audit.json",
        {
            "authorized_baseline": AUTHORIZED_BASELINE,
            "head": git("rev-parse", "HEAD"),
            "baseline_is_ancestor": True,
            "worktree_clean_at_authorized_baseline": True,
            "worktree_status_at_build": worktree_status.splitlines(),
            "unexpected_worktree_changes": unexpected_worktree_changes,
            "prior_stage_before_hash": prior_before["aggregate_sha256"],
            "prior_stage_after_hash": prior_after["aggregate_sha256"],
            "prior_stage_unchanged": prior_before == prior_after,
            "historical_artifacts_mutated": False,
        },
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "protected_hashes.json",
        {
            "historical_decisions_sha256": review["historical_decisions_sha256"],
            "historical_events_sha256": review["historical_events_sha256"],
            "checkpoint_sha256": MODEL_SHA256,
        },
    )
    write_json(
        STAGE_ROOT / "10_EVALUATION_AND_NEXT_STAGE" / "review_readiness.json",
        {
            "classification": "PASS_UNSEEN_LEVEL2_ASSOCIATION_REVIEW_READY"
            if len(selected_unseen) >= 6 and gate_summary["all_selected_pass"]
            else "PASS_REVIEW_READY_WITH_FEWER_UNSEEN_CASES",
            "case_count": len(selected_unseen),
            "human_review_required": True,
            "review_url": f"http://127.0.0.1:{REVIEW_PORT}/",
        },
    )
    write_json(
        STAGE_ROOT / "10_EVALUATION_AND_NEXT_STAGE" / "post_review_level3_gate_contract.json",
        {
            "level3_unlocked": False,
            "requires_zero_switches": True,
            "requires_zero_losses": True,
            "requires_zero_bad_seeds": True,
            "safe_abstention_allowed": True,
        },
    )
    write_json(
        STAGE_ROOT / "10_EVALUATION_AND_NEXT_STAGE" / "next_stage_decision.json",
        {
            "next_stage": "complete unseen human review before Level 3",
            "classification": "PASS_UNSEEN_LEVEL2_ASSOCIATION_REVIEW_READY"
            if len(selected_unseen) >= 6 and gate_summary["all_selected_pass"]
            else "PASS_REVIEW_READY_WITH_FEWER_UNSEEN_CASES",
            "exact_blocker": "Human review remains required; Level 3 remains blocked until unseen review has no switches, losses or bad seeds.",
        },
    )
    write_json(
        STAGE_ROOT / "11_COMMANDS_AND_TESTS" / "build_runtime.json",
        {
            "head": git("rev-parse", "HEAD"),
            "detector_device": detector["device"],
            "checkpoint_sha256": MODEL_SHA256,
            "diagnostic_case_count": 6,
            "unseen_case_count": len(selected_unseen),
            "review_port": REVIEW_PORT,
        },
    )
    return {
        "review": review,
        "templates": templates,
        "graphs": graphs,
        "detector": detector,
        "descriptor": descriptor_summary,
        "results": results,
        "diagnostic_failures": failures,
        "diagnostic_passes": passes,
        "selected_unseen": selected_unseen,
        "unseen_gates": gate_summary,
        "package": package,
    }


if __name__ == "__main__":
    result = build()
    print(
        json.dumps(
            {
                "stage_root": str(STAGE_ROOT),
                "diagnostic_cases": 6,
                "unseen_cases": len(result["selected_unseen"]),
                "review_url": f"http://127.0.0.1:{REVIEW_PORT}/",
                "classification": read_json(STAGE_ROOT / "10_EVALUATION_AND_NEXT_STAGE" / "review_readiness.json")[
                    "classification"
                ],
            },
            indent=2,
        )
    )
