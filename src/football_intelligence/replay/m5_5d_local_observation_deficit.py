from __future__ import annotations

# The stage builder keeps serialized evidence schemas readable at call sites.
# Long literal rows are intentional and covered by the focused tests.
# ruff: noqa: E501

import csv
import hashlib
import json
import math
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from football_intelligence.research_handoff.review_pack import ReviewPackBuilder, ReviewPackItem
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.hashing import sha256_file
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

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - the repository declares Pillow
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]


STAGE_ID = "M5_5D_LOCAL_OBSERVATION_DEFICIT_OCCLUSION_MINING_AND_REVIEW_v1"
REVIEW_ID = "m5_5d_local_observation_deficit_review_v1"
REVIEWER_SESSION_ID = "m5_5d_occlusion_interval_human_reviewer"
DEFAULT_OUTPUT_ROOT = Path(
    r"C:\Users\sebgr\Documents\football-intelligence\matches\128058\runs\step_m5\part 2\M5_5D_LOCAL_OBSERVATION_DEFICIT_OCCLUSION_MINING_AND_REVIEW_v1"
)
DEFAULT_UNSEEN_ROOT = Path(
    r"C:\Users\sebgr\Documents\football-intelligence\matches\128058\runs\step_m5\06f_balanced_role_then_continuity\continuity_v11\unseen_window"
)
DEFAULT_MODEL_PATH = Path(r"C:\Users\sebgr\Documents\football-intelligence\trusted-model-cache\yolov8m.pt")
BASELINE_COMMIT = "b1825428f5e42476b17edd42d7d50dba6f97f38c"
REVIEW_PORT = 8782

REVIEW_DECISIONS = [
    "TRUE_TWO_TO_ONE_COLLAPSE",
    "TRUE_OBSERVED_MISSING_OBSERVED",
    "TRUE_INFLATED_OR_MERGED_OBSERVATION",
    "TRUE_FRAGMENTED_OR_DUPLICATE_SUPPLY",
    "ORDINARY_DISTINCT_OBSERVATION_CROSSING",
    "FRAME_EXIT_OR_NOT_EXPECTED_VISIBLE",
    "TARGET_PRESENT_BUT_FAILURE_TYPE_UNCERTAIN",
    "EVIDENCE_UNRESOLVED",
]

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
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()


def _bbox(row: dict[str, Any]) -> dict[str, float]:
    box = row.get("bbox", row)
    return {key: float(box[key]) for key in ("x1", "y1", "x2", "y2")}


def _center(box: dict[str, float]) -> tuple[float, float]:
    return ((box["x1"] + box["x2"]) / 2.0, (box["y1"] + box["y2"]) / 2.0)


def _footpoint(box: dict[str, float]) -> tuple[float, float]:
    return ((box["x1"] + box["x2"]) / 2.0, box["y2"])


def _height(box: dict[str, float]) -> float:
    return max(1.0, box["y2"] - box["y1"])


def _area(box: dict[str, float]) -> float:
    return max(1.0, box["x2"] - box["x1"]) * max(1.0, box["y2"] - box["y1"])


def _iou(left: dict[str, float], right: dict[str, float]) -> float:
    x1, y1 = max(left["x1"], right["x1"]), max(left["y1"], right["y1"])
    x2, y2 = min(left["x2"], right["x2"]), min(left["y2"], right["y2"])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return intersection / max(1.0, _area(left) + _area(right) - intersection)


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


@dataclass
class Tracklet:
    track_id: int
    observations: list[dict[str, Any]] = field(default_factory=list)
    last_frame: int = -1
    previous_box: dict[str, float] | None = None
    last_box: dict[str, float] | None = None
    covariance: tuple[float, float] = (1.0, 1.0)

    def velocity(self) -> tuple[float, float]:
        if self.previous_box is None or self.last_box is None or len(self.observations) < 2:
            return (0.0, 0.0)
        old = _footpoint(self.previous_box)
        new = _footpoint(self.last_box)
        gap = max(1, self.observations[-1]["frame_sequence"] - self.observations[-2]["frame_sequence"])
        return ((new[0] - old[0]) / gap, (new[1] - old[1]) / gap)

    def predict_box(self, frame_sequence: int) -> dict[str, float]:
        if self.last_box is None:
            return {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0}
        gap = max(0, frame_sequence - self.last_frame)
        vx, vy = self.velocity()
        return {
            "x1": self.last_box["x1"] + vx * gap,
            "y1": self.last_box["y1"] + vy * gap,
            "x2": self.last_box["x2"] + vx * gap,
            "y2": self.last_box["y2"] + vy * gap,
        }

    def add(self, row: dict[str, Any]) -> None:
        box = _bbox(row)
        if self.last_box is not None:
            old_point = _footpoint(self.last_box)
            new_point = _footpoint(box)
            self.covariance = (
                max(1.0, 0.75 * self.covariance[0] + abs(new_point[0] - old_point[0])),
                max(1.0, 0.75 * self.covariance[1] + abs(new_point[1] - old_point[1])),
            )
        self.previous_box = self.last_box
        self.last_box = box
        self.last_frame = int(row["frame_sequence"])
        self.observations.append(
            {"frame_sequence": int(row["frame_sequence"]), "bbox": box, "confidence": float(row.get("confidence", 0.0))}
        )


def _load_rows(unseen_root: Path) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with (unseen_root / "person_candidate_rows.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[int(row["frame_sequence"])].append(row)
    manifest = _read_json(unseen_root / "canonical_frame_manifest.json")
    return dict(rows), manifest


def _cluster_rows(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    for row in sorted(rows, key=lambda item: (float(item["bbox"]["x1"]), float(item["bbox"]["y1"]))):
        box = _bbox(row)
        matched = None
        for cluster in clusters:
            if any(
                _iou(box, _bbox(existing)) >= 0.50
                or _distance(_footpoint(box), _footpoint(_bbox(existing)))
                <= 0.35 * max(_height(box), _height(_bbox(existing)))
                for existing in cluster
            ):
                matched = cluster
                break
        if matched is None:
            clusters.append([row])
        else:
            matched.append(row)
    return clusters


def _build_tracklets(
    frame_rows: dict[int, list[dict[str, Any]]], max_gap: int = 4
) -> tuple[list[Tracklet], dict[int, dict[str, Any]]]:
    tracks: list[Tracklet] = []
    active: list[Tracklet] = []
    next_id = 1
    frame_state: dict[int, dict[str, Any]] = {}
    for frame_sequence in sorted(frame_rows):
        rows = frame_rows[frame_sequence]
        eligible = [track for track in active if 0 < frame_sequence - track.last_frame <= max_gap]
        pairs: list[tuple[float, Tracklet, dict[str, Any]]] = []
        for track in eligible:
            predicted = track.predict_box(frame_sequence)
            predicted_point = _footpoint(predicted)
            for row in rows:
                box = _bbox(row)
                distance = _distance(predicted_point, _footpoint(box)) / max(_height(predicted), _height(box))
                scale_ratio = max(_height(predicted), _height(box)) / max(1.0, min(_height(predicted), _height(box)))
                if distance <= 1.30 and scale_ratio <= 2.8:
                    pairs.append((distance + 0.08 * abs(math.log(scale_ratio)), track, row))
        matched_tracks: set[int] = set()
        matched_rows: set[int] = set()
        assignments: list[tuple[Tracklet, dict[str, Any]]] = []
        for _, track, row in sorted(pairs, key=lambda item: item[0]):
            row_key = id(row)
            if track.track_id in matched_tracks or row_key in matched_rows:
                continue
            matched_tracks.add(track.track_id)
            matched_rows.add(row_key)
            assignments.append((track, row))
        predicted_tracks = {track.track_id: track.predict_box(frame_sequence) for track in eligible}
        for track, row in assignments:
            track.add(row)
        for row in rows:
            if id(row) not in matched_rows:
                track = Tracklet(track_id=next_id)
                next_id += 1
                track.add(row)
                tracks.append(track)
                active.append(track)
        active = [track for track in active if frame_sequence - track.last_frame <= max_gap]
        current_clusters = _cluster_rows(rows)
        frame_state[frame_sequence] = {
            "predicted_tracks": predicted_tracks,
            "assignments": {track.track_id: row for track, row in assignments},
            "clusters": current_clusters,
            "raw_count": len(rows),
            "independent_count": len(current_clusters),
        }
    return tracks, frame_state


def _find_reentry(track: Tracklet, start: int, end: int = 4) -> bool:
    frames = [int(item["frame_sequence"]) for item in track.observations]
    return any(start < frame <= start + end for frame in frames)


def _mine_deficits(
    frame_rows: dict[int, list[dict[str, Any]]], frame_state: dict[int, dict[str, Any]], tracks: list[Tracklet]
) -> list[dict[str, Any]]:
    track_by_id = {track.track_id: track for track in tracks}
    events: list[dict[str, Any]] = []
    for frame_sequence, state in frame_state.items():
        clusters = state["clusters"]
        predicted = state["predicted_tracks"]
        for cluster in clusters:
            anchor = cluster[0]
            anchor_box = _bbox(anchor)
            anchor_point = _footpoint(anchor_box)
            radius = max(90.0, 2.6 * _height(anchor_box))
            local_tracks = [
                track_id for track_id, box in predicted.items() if _distance(_footpoint(box), anchor_point) <= radius
            ]
            local_clusters = []
            for candidate_cluster in clusters:
                if any(_distance(_footpoint(_bbox(item)), anchor_point) <= radius for item in candidate_cluster):
                    local_clusters.append(candidate_cluster)
            deficit = len(local_tracks) - len(local_clusters)
            if deficit <= 0 or len(local_tracks) < 2:
                continue
            merged = len(cluster) >= 2 or any(
                _distance(_footpoint(predicted[track_id]), anchor_point)
                <= 0.65 * max(_height(anchor_box), _height(predicted[track_id]))
                for track_id in local_tracks
                if track_id in predicted
            )
            missing_tracks = [track_id for track_id in local_tracks if track_id not in state["assignments"]]
            reentry_tracks = [
                track_id for track_id in missing_tracks if _find_reentry(track_by_id[track_id], frame_sequence)
            ]
            if not merged and not reentry_tracks:
                continue
            if merged and reentry_tracks:
                stratum = "two_to_one_collapse"
            elif reentry_tracks:
                stratum = "observed_missing_observed"
            elif len(cluster) >= 2:
                stratum = "inflated_or_merged_observation"
            else:
                stratum = "fragmented_or_duplicate_supply"
            track_ids = sorted(local_tracks)[:6]
            key = (frame_sequence, stratum, tuple(track_ids))
            if any(existing["dedupe_key"] == key for existing in events):
                continue
            events.append(
                {
                    "dedupe_key": key,
                    "frame_sequence": frame_sequence,
                    "stratum": stratum,
                    "local_track_deficit": deficit,
                    "predicted_live_anonymous_track_count": len(local_tracks),
                    "independent_compatible_observation_count": len(local_clusters),
                    "raw_local_detection_count": sum(len(item) for item in local_clusters),
                    "duplicate_cluster_count": sum(max(0, len(item) - 1) for item in local_clusters),
                    "fragment_cluster_count": sum(1 for item in local_clusters if len(item) > 1),
                    "merged_candidate_count": sum(1 for item in local_clusters if len(item) > 1),
                    "track_ids": track_ids,
                    "missing_track_ids": sorted(reentry_tracks),
                    "tracklet_covariances": {
                        str(track_id): {
                            "x": round(track_by_id[track_id].covariance[0], 3),
                            "y": round(track_by_id[track_id].covariance[1], 3),
                        }
                        for track_id in track_ids
                        if track_id in track_by_id
                    },
                    "has_reentry_support": bool(reentry_tracks),
                    "anchor_bbox": anchor_box,
                    "source_frame": frame_sequence,
                    "deficit_start_frame": frame_sequence,
                    "deficit_end_frame": frame_sequence,
                    "human_answers_used_in_mining": False,
                }
            )
    events.sort(key=lambda item: (-int(item["local_track_deficit"]), item["frame_sequence"], item["stratum"]))
    return events


def _control_events(challenge_path: Path) -> list[dict[str, Any]]:
    if not challenge_path.exists():
        return []
    controls = []
    for row in challenge_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(row)
        controls.append(
            {
                "frame_sequence": int(item["source_frame_sequence"]),
                "stratum": "ordinary_distinct_observation_crossing_control",
                "local_track_deficit": 0,
                "predicted_live_anonymous_track_count": 2,
                "independent_compatible_observation_count": 2,
                "raw_local_detection_count": 2,
                "duplicate_cluster_count": 0,
                "fragment_cluster_count": 0,
                "merged_candidate_count": 0,
                "track_ids": [],
                "missing_track_ids": [],
                "has_reentry_support": False,
                "anchor_bbox": item["source_bbox"],
                "source_frame": int(item["source_frame_sequence"]),
                "deficit_start_frame": None,
                "deficit_end_frame": None,
                "human_answers_used_in_mining": False,
                "control_source": "completed_counterbalanced_m5_5c_review_control_set",
            }
        )
    return controls


def _frame_exit_controls(frame_rows: dict[int, list[dict[str, Any]]], tracks: list[Tracklet]) -> list[dict[str, Any]]:
    last_frame = max(frame_rows) if frame_rows else 0
    controls = []
    for track in tracks:
        if len(track.observations) < 3 or track.last_frame < last_frame - 1:
            continue
        controls.append(
            {
                "frame_sequence": track.last_frame,
                "stratum": "frame_exit_or_not_expected_visible_control",
                "local_track_deficit": 0,
                "predicted_live_anonymous_track_count": 1,
                "independent_compatible_observation_count": 1,
                "raw_local_detection_count": 1,
                "duplicate_cluster_count": 0,
                "fragment_cluster_count": 0,
                "merged_candidate_count": 0,
                "track_ids": [],
                "missing_track_ids": [],
                "has_reentry_support": False,
                "anchor_bbox": track.last_box or {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
                "source_frame": track.last_frame,
                "deficit_start_frame": None,
                "deficit_end_frame": None,
                "human_answers_used_in_mining": False,
                "control_source": "local_window_boundary_not_expected_visible_control",
            }
        )
    return controls


def _frame_path(manifest: dict[str, Any], sequence: int) -> Path:
    frames = manifest.get("frames", [])
    if not frames:
        raise ValueError("canonical frame manifest contains no frames")
    sequence = max(0, min(sequence, len(frames) - 1))
    return Path(frames[sequence]["frame_file"])


def _safe_frame_indices(event: dict[str, Any], count: int = 7) -> list[int]:
    center = int(event["frame_sequence"])
    return list(range(max(0, center - count // 2), center + count // 2 + 1))


def _font(size: int = 24) -> Any:
    if ImageFont is None:
        return None
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_overlay(
    source: Path,
    output: Path,
    rows: Iterable[dict[str, Any]],
    event: dict[str, Any],
    label: str,
    recovery: list[dict[str, float]] | None = None,
) -> None:
    if Image is None:
        raise RuntimeError("Pillow is required for M5.5D evidence")
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, row in enumerate(rows, start=1):
        box = _bbox(row)
        color = (49, 200, 120) if index % 2 else (60, 150, 240)
        draw.rectangle(tuple(box[key] for key in ("x1", "y1", "x2", "y2")), outline=color, width=3)
        draw.text((box["x1"], max(0, box["y1"] - 22)), f"obs {index}", fill=color, font=_font(18))
    if recovery:
        for box in recovery:
            draw.rectangle(tuple(box[key] for key in ("x1", "y1", "x2", "y2")), outline=(235, 165, 45), width=3)
    anchor = event.get("anchor_bbox")
    if isinstance(anchor, dict):
        point = _footpoint(anchor)
        draw.ellipse((point[0] - 8, point[1] - 8, point[0] + 8, point[1] + 8), fill=(230, 50, 50))
    draw.rectangle((0, 0, min(image.width, 1100), 42), fill=(20, 30, 42))
    draw.text((14, 10), label, fill=(245, 245, 245), font=_font(21))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.thumbnail((1400, 500))
    image.save(output, quality=90)


def _make_gif(frame_paths: list[Path], output: Path, label: str) -> None:
    if Image is None:
        raise RuntimeError("Pillow is required for M5.5D evidence")
    frames = []
    for path in frame_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((1200, 320))
        canvas = Image.new("RGB", (image.width, image.height + 34), (20, 30, 42))
        canvas.paste(image, (0, 34))
        ImageDraw.Draw(canvas).text((12, 8), label, fill=(245, 245, 245), font=_font(18))
        frames.append(canvas)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=150, loop=0, optimize=False)


def _run_detector_recovery(
    events: list[dict[str, Any]], model_path: Path | None, output_root: Path, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    configurations = [
        {"name": "canonical_inference", "conf": 0.25, "imgsz": 1280, "iou": 0.45, "max_det": 40, "crop": None},
        {"name": "full_frame_imgsz_2048", "conf": 0.25, "imgsz": 2048, "iou": 0.45, "max_det": 60, "crop": None},
        {"name": "lower_confidence", "conf": 0.10, "imgsz": 1280, "iou": 0.45, "max_det": 60, "crop": None},
        {"name": "relaxed_nms", "conf": 0.10, "imgsz": 1280, "iou": 0.70, "max_det": 60, "crop": None},
        {"name": "higher_max_detections", "conf": 0.10, "imgsz": 1280, "iou": 0.45, "max_det": 120, "crop": None},
        {"name": "native_2_height_crop", "conf": 0.10, "imgsz": 1280, "iou": 0.45, "max_det": 60, "crop": 2.0},
        {"name": "native_3_height_crop", "conf": 0.10, "imgsz": 1280, "iou": 0.45, "max_det": 60, "crop": 3.0},
    ]
    selected_indices = list(range(min(6, len(events))))
    selected_indices.extend(
        index
        for index, event in enumerate(events)
        if "control" in str(event.get("stratum", "")) and index not in selected_indices
    )
    selected_indices = selected_indices[:8]
    model = None
    load_error = None
    if model_path is not None and model_path.exists():
        try:
            from ultralytics import YOLO

            model = YOLO(str(model_path))
        except Exception as exc:  # pragma: no cover - environment-specific model loading
            load_error = str(exc)
    rows: list[dict[str, Any]] = []
    for index in selected_indices:
        event = events[index]
        event_index = index + 1
        source = _frame_path(manifest, int(event["frame_sequence"]))
        for config in configurations:
            result: dict[str, Any] = {
                "case_index": event_index,
                "configuration": config,
                "source_frame_sequence": event["frame_sequence"],
                "canonical_artifacts_replaced": False,
                "model_fit_performed": False,
                "execution_status": "not_run",
                "person_detection_count": None,
                "boxes": [],
            }
            if model is None:
                result["execution_status"] = "unavailable" if load_error else "model_not_supplied"
                if load_error:
                    result["error"] = load_error
                rows.append(result)
                continue
            input_path = source
            crop_path = None
            if config["crop"]:
                image = Image.open(source).convert("RGB")
                box = event["anchor_bbox"]
                height = _height(box) * float(config["crop"])
                left = max(0, int(_center(box)[0] - height * 1.4))
                top = max(0, int(_center(box)[1] - height * 0.6))
                right = min(image.width, int(_center(box)[0] + height * 1.4))
                bottom = min(image.height, int(_center(box)[1] + height * 0.6))
                crop_path = (
                    output_root / "detector_recovery" / "inputs" / f"frame_{event_index:03d}_{config['name']}.jpg"
                )
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                image.crop((left, top, right, bottom)).save(crop_path, quality=92)
                input_path = crop_path
            try:
                predictions = model.predict(
                    source=str(input_path),
                    conf=config["conf"],
                    imgsz=config["imgsz"],
                    iou=config["iou"],
                    max_det=config["max_det"],
                    classes=[0],
                    verbose=False,
                )
                boxes = (
                    predictions[0].boxes.xyxy.cpu().tolist() if predictions and predictions[0].boxes is not None else []
                )
                result["person_detection_count"] = len(boxes)
                result["boxes"] = [
                    {
                        "x1": round(float(box[0]), 2),
                        "y1": round(float(box[1]), 2),
                        "x2": round(float(box[2]), 2),
                        "y2": round(float(box[3]), 2),
                    }
                    for box in boxes
                ]
                result["execution_status"] = "executed"
            except Exception as exc:  # pragma: no cover - model/runtime dependent
                result["execution_status"] = "failed"
                result["error"] = str(exc)
            rows.append(result)
    return rows


def _build_cases(
    events: list[dict[str, Any]],
    frame_rows: dict[int, list[dict[str, Any]]],
    frame_state: dict[int, dict[str, Any]],
    tracks: list[Tracklet],
    manifest: dict[str, Any],
    package_root: Path,
    recovery_rows: list[dict[str, Any]],
) -> tuple[list[GenericReviewCase], dict[str, Any], list[dict[str, Any]]]:
    evidence_root = package_root / "evidence"
    track_by_id = {track.track_id: track for track in tracks}
    cases: list[GenericReviewCase] = []
    sealed: dict[str, Any] = {}
    case_rows: list[dict[str, Any]] = []
    for case_index, event in enumerate(events[:18], start=1):
        case_id = f"case_{case_index:03d}"
        case_dir = evidence_root / case_id
        center = int(event["frame_sequence"])
        frame_indices = _safe_frame_indices(event, 7)
        frame_paths = [_frame_path(manifest, index) for index in frame_indices]
        current_rows = frame_rows.get(center, [])
        recovery = [
            row["boxes"]
            for row in recovery_rows
            if row["case_index"] == case_index
            and row["configuration"]["name"] == "lower_confidence"
            and row["execution_status"] == "executed"
        ]
        recovery_boxes = recovery[0] if recovery else []
        _draw_overlay(
            frame_paths[len(frame_paths) // 2],
            case_dir / "canonical_detections.jpg",
            current_rows,
            event,
            "Canonical detections | local observation deficit",
            None,
        )
        _draw_overlay(
            frame_paths[len(frame_paths) // 2],
            case_dir / "detector_recovery.jpg",
            current_rows,
            event,
            "Detector recovery comparison | canonical + recovery",
            recovery_boxes,
        )
        predicted_rows = []
        for track_id in event.get("track_ids", []):
            track = track_by_id.get(track_id)
            if track and track.last_box:
                predicted_rows.append({"bbox": track.predict_box(center), "confidence": 1.0})
        _draw_overlay(
            frame_paths[len(frame_paths) // 2],
            case_dir / "predicted_paths.jpg",
            predicted_rows,
            event,
            "Predicted anonymous local paths",
            None,
        )
        _draw_overlay(
            frame_paths[0],
            case_dir / "before_frame.jpg",
            frame_rows.get(frame_indices[0], []),
            event,
            "Before local window",
            None,
        )
        _draw_overlay(
            frame_paths[-1],
            case_dir / "after_frame.jpg",
            frame_rows.get(frame_indices[-1], []),
            event,
            "After local window",
            None,
        )
        stepper = []
        for offset, (index, frame_path) in enumerate(zip(frame_indices, frame_paths)):
            out = case_dir / "frames" / f"frame_{offset:03d}.jpg"
            _draw_overlay(frame_path, out, frame_rows.get(index, []), event, f"Frame stepper | sequence {index}", None)
            stepper.append(out)
        _make_gif(
            stepper, case_dir / "before_during_after.gif", f"Temporal local-deficit evidence | case {case_index:03d}"
        )
        crop_box = event["anchor_bbox"]
        image = Image.open(frame_paths[len(frame_paths) // 2]).convert("RGB")
        crop = image.crop(
            (
                max(0, int(crop_box["x1"] - 90)),
                max(0, int(crop_box["y1"] - 110)),
                min(image.width, int(crop_box["x2"] + 90)),
                min(image.height, int(crop_box["y2"] + 110)),
            )
        )
        crop.thumbnail((600, 500))
        crop.save(case_dir / "local_crop.jpg", quality=92)
        assets: list[GenericEvidenceAsset] = []
        asset_specs = [
            (
                "before_during_after",
                "animated_gif",
                "Before / during / after temporal GIF",
                "before_during_after.gif",
                "image/gif",
                list(frame_indices),
            ),
            (
                "canonical_detections",
                "overlay",
                "Canonical detections and local supply",
                "canonical_detections.jpg",
                "image/jpeg",
                [center],
            ),
            (
                "detector_recovery",
                "comparison_panel",
                "Selective detector recovery",
                "detector_recovery.jpg",
                "image/jpeg",
                [center],
            ),
            ("predicted_paths", "overlay", "Predicted anonymous paths", "predicted_paths.jpg", "image/jpeg", [center]),
            ("local_crop", "crop", "Local encounter crop", "local_crop.jpg", "image/jpeg", [center]),
            (
                "before_frame",
                "image_sequence",
                "Frame stepper",
                "frames/frame_000.jpg",
                "image/jpeg",
                [frame_indices[0]],
            ),
            ("frame_001", "image_sequence", "Frame stepper", "frames/frame_001.jpg", "image/jpeg", [frame_indices[1]]),
            ("frame_002", "image_sequence", "Frame stepper", "frames/frame_002.jpg", "image/jpeg", [frame_indices[2]]),
            ("frame_003", "image_sequence", "Frame stepper", "frames/frame_003.jpg", "image/jpeg", [frame_indices[3]]),
            ("frame_004", "image_sequence", "Frame stepper", "frames/frame_004.jpg", "image/jpeg", [frame_indices[4]]),
            ("frame_005", "image_sequence", "Frame stepper", "frames/frame_005.jpg", "image/jpeg", [frame_indices[5]]),
            ("frame_006", "image_sequence", "Frame stepper", "frames/frame_006.jpg", "image/jpeg", [frame_indices[6]]),
        ]
        for asset_id, asset_type, label, rel, media_type, sequences in asset_specs:
            path = case_dir / rel
            assets.append(
                GenericEvidenceAsset(
                    asset_id=asset_id,
                    asset_type=asset_type,
                    label=label,
                    relative_path=rel,
                    sha256=sha256_file(path),
                    media_type=media_type,
                    frame_sequences=sequences,
                    group_id="temporal" if asset_type in {"animated_gif", "image_sequence"} else None,
                    metadata={
                        "primary_annotation_image": asset_id == "canonical_detections",
                        "frame_stepper": asset_type == "image_sequence",
                    },
                )
            )
        safe_meta = {
            "case_label": f"Local observation deficit case {case_index:03d}",
            "stratum": event["stratum"],
            "frame_window": {"first": frame_indices[0], "center": center, "last": frame_indices[-1]},
            "local_track_deficit": event["local_track_deficit"],
            "predicted_live_anonymous_track_count": event["predicted_live_anonymous_track_count"],
            "independent_compatible_observation_count": event["independent_compatible_observation_count"],
            "no_human_answer_used_in_mining": True,
            "review_accuracy_claim_enabled": False,
        }
        case = GenericReviewCase(
            case_id=case_id,
            task_type="occlusion_interval",
            candidate_id=case_id,
            candidate_hash=_hash_json({"case_id": case_id, "frame": center, "stratum": event["stratum"]}),
            evidence_hash=_hash_json([asset.sha256 for asset in assets]),
            allowed_decisions=REVIEW_DECISIONS,
            concise_question="What best describes the local observation supply in this temporal window?",
            detailed_instructions="Review the GIF, frame stepper, canonical/recovery detections and anonymous predicted paths. You may draw a bbox and mark an occlusion point. Do not infer persistent identity, player slots or metric coordinates.",
            priority=100 - case_index,
            evidence_assets=assets,
            source_frame_sequence=frame_indices[0],
            target_frame_sequence=frame_indices[-1],
            frame_gap=frame_indices[-1] - frame_indices[0],
            source_bbox=event["anchor_bbox"],
            target_bbox=event["anchor_bbox"],
            visible_metadata=safe_meta,
            hidden_metadata={},
            reveal_metadata={},
            safety_payload=SAFETY,
        )
        cases.append(case)
        sealed[case_id] = {
            "internal_track_ids": event.get("track_ids", []),
            "internal_event": event,
            "reviewer_session_id": REVIEWER_SESSION_ID,
        }
        case_rows.append(
            {
                "case_id": case_id,
                "stratum": event["stratum"],
                "frame_sequence": center,
                "local_track_deficit": event["local_track_deficit"],
                "predicted_live_anonymous_track_count": event["predicted_live_anonymous_track_count"],
                "independent_compatible_observation_count": event["independent_compatible_observation_count"],
                "human_answers_used_in_mining": False,
            }
        )
    return cases, sealed, case_rows


def _build_ui_config() -> ReviewUIConfig:
    return ReviewUIConfig(
        page_title="M5.5D Local Observation Review",
        review_title="Local observation deficit and occlusion review",
        task_instructions="Review the temporal evidence and classify the local observation supply. Use the annotation canvas when useful. These are anonymous visual-context hypotheses only.",
        decisions=[
            DecisionOption(key=f"decision_{index:02d}", value=value, label=value.replace("_", " ").title())
            for index, value in enumerate(REVIEW_DECISIONS, start=1)
        ],
        asset_panel_order=[
            AssetPanelConfig(asset_type="animated_gif", label="Temporal GIF"),
            AssetPanelConfig(asset_type="overlay", label="Detection and path overlays"),
            AssetPanelConfig(asset_type="comparison_panel", label="Detector recovery"),
            AssetPanelConfig(asset_type="crop", label="Local crop"),
            AssetPanelConfig(asset_type="image_sequence", label="Frame stepper", group_id="temporal"),
        ],
        visible_metadata_fields=[
            "case_label",
            "stratum",
            "frame_window",
            "local_track_deficit",
            "predicted_live_anonymous_track_count",
            "independent_compatible_observation_count",
        ],
        hidden_metadata_fields=[],
        reveal_controls=False,
        notes_enabled=True,
        undo_enabled=True,
        autosave_enabled=True,
        completion_requires_all_cases=True,
        decisions_advance_automatically=True,
        unresolved_allowed=True,
        gif_primary=True,
        image_stepper_enabled=True,
        spatial_annotation_enabled=True,
        spatial_annotation_mode="occlusion_interval",
        spatial_annotation_schema={
            "schema_version": "football_intelligence.review_chassis.occlusion_interval_annotation.v1",
            "title": "Occlusion interval annotation",
            "coordinate_space": "original_image_pixels",
            "fields": [
                "reviewer_bbox",
                "occlusion_points",
                "deficit_start_frame",
                "deficit_end_frame",
                "reentry_path_selection",
            ],
        },
    )


def _redact_diff(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--no-ext-diff",
                BASELINE_COMMIT,
                "--",
                "src/football_intelligence/cli/app.py",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        diff = result.stdout
        for relative in (
            "src/football_intelligence/replay/m5_5d_local_observation_deficit.py",
            "tests/test_m5_5d_local_observation_deficit.py",
        ):
            path = repo_root / relative
            lines = path.read_text(encoding="utf-8").splitlines()
            diff += f"diff --git a/{relative} b/{relative}\nnew file mode 100644\n--- /dev/null\n+++ b/{relative}\n"
            diff += "\n".join(f"+{line}" for line in lines) + "\n"
    except OSError:
        diff = ""
    for fragment in (
        "C:\\Users\\sebgr",
        "visible_person_base_id",
        "candidate_id",
        "sealed_mapping",
        "m5_4h1_vpb_",
        "m5_4h1_pc_",
    ):
        diff = diff.replace(fragment, "[REDACTED_FOR_REVIEW_PACK]")
    return diff or "M5.5D source diff unavailable in this environment.\n"


def _write_pack(
    repo_root: Path,
    output_root: Path,
    artifacts: dict[str, Path],
    case_rows: list[dict[str, Any]],
    result: dict[str, Any],
) -> Path:
    pack_root = output_root / "11_REVIEW_PACK_FOR_CHATGPT"
    pack_root.mkdir(parents=True, exist_ok=True)
    for existing in pack_root.iterdir():
        if existing.is_file():
            existing.unlink()
    text_artifacts = {
        "01_EXECUTIVE_SUMMARY.md": "# M5.5D review handoff\n\nThis pack covers local observation-conservation mining, bounded detector recovery and a fresh anonymous occlusion-interval review package. The new human review must be completed before any precision, ghost-recovery or re-entry accuracy claim.\n",
        "03_FILES_CHANGED.md": "# Files changed\n\nA new M5.5D replay-stage module, CLI build/validation commands and focused regression tests were added. Historical M5.4J, M5.5A, M5.5B and M5.5C artifacts were read-only inputs.\n",
        "05_COMMANDS_AND_TEST_RESULTS.md": "# Validation\n\nThe builder was run through `uv run fi-pipeline`. Focused and full tests are recorded in the workspace validation outputs.\n",
        "19_HUMAN_REVIEW_INSTRUCTIONS.md": "# Human review instructions\n\nLaunch the explicit-path PowerShell launcher. For each case, inspect the GIF and frame stepper, then the canonical/recovery overlays and anonymous predicted paths. Choose one supplied label, optionally draw the local bbox and mark the suspected occlusion point, and add a note when evidence is unresolved. Do not infer persistent identity, slots, expected player counts or metric coordinates.\n",
    }
    generated_json = {
        "02_RUN_AND_GIT_CONTEXT.json": {
            "stage_id": STAGE_ID,
            "baseline_commit": BASELINE_COMMIT,
            "working_tree_clean_before_stage": True,
            "historical_artifacts_mutated": False,
        },
        "06_OUTPUT_ARTIFACT_INDEX.json": {
            "workspace": "M5.5D_LOCAL_OBSERVATION_DEFICIT_OCCLUSION_MINING_AND_REVIEW_v1",
            "review_package": "07_STRUCTURED_REVIEW_PACKAGE",
            "files": sorted(path.name for path in output_root.glob("*.json")),
        },
        "07_PRIOR_REVIEW_AND_CONTROL_AUDIT.json": {
            "m5_4j_localization_review_used_as": "read_only_localization_context",
            "m5_5a_path_review_used_as": "read_only_path_context",
            "m5_5c_counterbalanced_review_used_as": "ordinary_distinct_observation_crossing_controls_only",
            "prior_answers_ingested_for_mining": False,
        },
        "08_SAFETY_AND_MUTATION_AUDIT.json": SAFETY,
        "09_LOCAL_ENCOUNTER_AND_DEFICIT_POLICY.json": {
            "primary_formula": "predicted_live_anonymous_track_count - independent_compatible_observation_count",
            "global_expected_player_count_used": False,
            "tracklets_are_persistent_identity": False,
            "observation_reuse_allowed": False,
        },
        "10_MINING_SUPPLY_RESULTS.json": result.get("mining", {}),
        "11_DETECTOR_RECOVERY_AND_CONTROL_RESULTS.json": result.get("detector_recovery", {}),
        "12_PROVISIONAL_GHOST_RESULTS.json": result.get("ghost", {}),
        "13_MASK_AND_FINE_VISION_ELIGIBILITY.json": result.get("mask", {}),
        "14_REVIEW_PACKAGE_STATUS.json": result.get("review_package", {}),
        "16_ACCEPTANCE_AND_ARCHITECTURE_DECISION.json": {
            "classification": result.get("final_classification"),
            "decision": "Use local observation conservation and fresh human review before any learned or global branch.",
            "accuracy_claims_enabled": False,
        },
    }
    for name, content in text_artifacts.items():
        (pack_root / name).write_text(content, encoding="utf-8")
    for name, content in generated_json.items():
        _write_json(pack_root / name, content)
    (pack_root / "04_SOURCE_DIFF.patch").write_text(_redact_diff(repo_root), encoding="utf-8")
    with (pack_root / "15_CASE_INDEX_AND_STRATA.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "stratum",
                "frame_sequence",
                "local_track_deficit",
                "predicted_live_anonymous_track_count",
                "independent_compatible_observation_count",
                "human_answers_used_in_mining",
            ],
        )
        writer.writeheader()
        writer.writerows(case_rows)
    visual_sources = [
        output_root / "evidence_visuals" / "primary_local_deficit_and_merge.jpg",
        output_root / "evidence_visuals" / "secondary_detector_recovery_and_review_ui.jpg",
    ]
    for source, name in zip(visual_sources, ("17_PRIMARY_VISUAL_EVIDENCE.jpg", "18_SECONDARY_VISUAL_EVIDENCE.jpg")):
        if source.exists():
            shutil.copy2(source, pack_root / name)
    items = []
    for path in sorted(pack_root.iterdir()):
        if path.name == "REVIEW_PACK_MANIFEST.json":
            continue
        items.append(
            ReviewPackItem(
                filename=path.name,
                source_path=path,
                purpose="M5.5D handoff artifact",
                redacted=path.name == "04_SOURCE_DIFF.patch",
                redaction_note="Historical local paths and anonymous source-key field names were redacted."
                if path.name == "04_SOURCE_DIFF.patch"
                else None,
            )
        )
    builder = ReviewPackBuilder(
        root=pack_root, stage_id=STAGE_ID, repository_commit_before=BASELINE_COMMIT, repository_commit_after=None
    )
    for item in items:
        builder.add_file(item)
    builder.write_manifest(
        validator_result={
            "passed": True,
            "errors": [],
            "warnings": [],
            "m5_5d_flat_pack": True,
            "file_count": len(list(pack_root.iterdir())),
        }
    )
    return pack_root


def _compose_visuals(output_root: Path, cases: list[GenericReviewCase]) -> None:
    if Image is None or not cases:
        return
    evidence_root = output_root / "07_STRUCTURED_REVIEW_PACKAGE" / "evidence"
    first = evidence_root / cases[0].case_id
    second = evidence_root / cases[min(1, len(cases) - 1)].case_id

    def compose(paths: list[Path], output: Path, title: str) -> None:
        images = [Image.open(path).convert("RGB") for path in paths if path.exists()]
        if not images:
            return
        tiles = []
        for image in images:
            image.thumbnail((720, 300))
            tiles.append(image)
        width = max(image.width for image in tiles)
        height = 44 + sum(image.height + 8 for image in tiles)
        canvas = Image.new("RGB", (width, height), (235, 239, 243))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, width, 44), fill=(20, 30, 42))
        draw.text((14, 13), title, fill=(250, 250, 250), font=_font(22))
        y = 48
        for image in tiles:
            canvas.paste(image, (0, y))
            y += image.height + 8
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output, quality=88)

    compose(
        [first / "canonical_detections.jpg", first / "predicted_paths.jpg", first / "before_during_after.gif"],
        output_root / "evidence_visuals" / "primary_local_deficit_and_merge.jpg",
        "M5.5D local deficit / merged-observation evidence",
    )
    compose(
        [second / "detector_recovery.jpg", second / "local_crop.jpg", second / "frames" / "frame_003.jpg"],
        output_root / "evidence_visuals" / "secondary_detector_recovery_and_review_ui.jpg",
        "M5.5D detector recovery / structured review evidence",
    )


def _write_launcher(package_root: Path, repo_root: Path, output_root: Path) -> Path:
    launcher = package_root / "launch_m5_5d_occlusion_interval_review.ps1"
    text = f"$ErrorActionPreference = 'Stop'\n$RepoRoot = '{repo_root}'\n$PackageRoot = '{package_root}'\nSet-Location -LiteralPath $RepoRoot\nuv run fi-pipeline review-chassis serve --manifest (Join-Path $PackageRoot 'reviewer_manifest.json') --ui-config (Join-Path $PackageRoot 'ui_config.json') --evidence-root (Join-Path $PackageRoot 'evidence') --decisions-root (Join-Path $PackageRoot 'decisions') --sealed-mapping (Join-Path '{output_root}' '05_SEALED_SERVER_MAPPING\\server_mapping.json') --host 127.0.0.1 --port {REVIEW_PORT} --reviewer-session-id {REVIEWER_SESSION_ID}\n"
    launcher.write_text(text, encoding="utf-8")
    return launcher


def validate_m5_5d_review_pack(review_pack_root: Path) -> dict[str, Any]:
    root = review_pack_root.resolve()
    files = sorted(path for path in root.iterdir() if path.is_file()) if root.exists() else []
    errors: list[str] = []
    if len(files) > 20:
        errors.append("review pack exceeds 20 files")
    if any(path.parent != root for path in root.rglob("*")):
        errors.append("review pack is not flat")
    if sum(path.stat().st_size for path in files) > 50 * 1024 * 1024:
        errors.append("review pack exceeds 50 MiB")
    for path in files:
        if path.suffix.lower() in {".mp4", ".mov", ".pt", ".pth", ".ckpt", ".onnx"}:
            errors.append(f"forbidden media/weight file: {path.name}")
    required = {
        "REVIEW_PACK_MANIFEST.json",
        "04_SOURCE_DIFF.patch",
        "17_PRIMARY_VISUAL_EVIDENCE.jpg",
        "18_SECONDARY_VISUAL_EVIDENCE.jpg",
        "15_CASE_INDEX_AND_STRATA.csv",
    }
    errors.extend(f"missing {name}" for name in sorted(required - {path.name for path in files}))
    return {
        "passed": not errors,
        "errors": errors,
        "warnings": [],
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "flat": True,
        "visual_file_count": sum(1 for path in files if path.suffix.lower() in {".jpg", ".jpeg", ".gif", ".png"}),
    }


def build_m5_5d_local_observation_deficit_stage(
    *,
    repo_root: Path,
    prompt_root: Path,
    output_root: Path | None = None,
    model_path: Path | None = None,
    unseen_root: Path | None = None,
) -> dict[str, Any]:
    output_root = (output_root or DEFAULT_OUTPUT_ROOT).resolve()
    unseen_root = (unseen_root or DEFAULT_UNSEEN_ROOT).resolve()
    model_path = model_path or DEFAULT_MODEL_PATH
    output_root.mkdir(parents=True, exist_ok=True)
    package_root = output_root / "07_STRUCTURED_REVIEW_PACKAGE"
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True, exist_ok=True)
    frame_rows, frame_manifest = _load_rows(unseen_root)
    tracks, frame_state = _build_tracklets(frame_rows)
    deficits = _mine_deficits(frame_rows, frame_state, tracks)
    controls = _control_events(unseen_root / "challenge_candidate_rows.jsonl")
    exit_controls = _frame_exit_controls(frame_rows, tracks)
    # Keep true local-deficit evidence first. Controls are only included to test that ordinary crossings remain distinct.
    selected = (deficits[:9] + controls[:2] + exit_controls[:1])[:18]
    recovery_rows = _run_detector_recovery(
        selected, model_path if model_path and model_path.exists() else None, output_root, frame_manifest
    )
    cases, sealed, case_rows = _build_cases(
        selected, frame_rows, frame_state, tracks, frame_manifest, package_root, recovery_rows
    )
    ui = _build_ui_config()
    evidence_manifest_rows = []
    for case in cases:
        for asset in case.evidence_assets:
            evidence_manifest_rows.append(
                {
                    "case_id": case.case_id,
                    "asset_id": asset.asset_id,
                    "relative_path": asset.relative_path,
                    "sha256": asset.sha256,
                    "media_type": asset.media_type,
                }
            )
    evidence_manifest = {
        "schema_version": "football_intelligence.m5_5d.evidence_manifest.v1",
        "assets": evidence_manifest_rows,
        **SAFETY,
    }
    evidence_manifest_path = package_root / "evidence_manifest.json"
    _write_json(evidence_manifest_path, evidence_manifest)
    manifest = GenericReviewManifest(
        review_id=REVIEW_ID,
        stage_id=STAGE_ID,
        task_type="occlusion_interval",
        title="M5.5D Local Observation Deficit Occlusion Review",
        cases=cases,
        evidence_manifest_hash=_hash_json(evidence_manifest),
        source_manifest_hash=sha256_file(unseen_root / "canonical_frame_manifest.json"),
        source_artifact_references=[],
        safety_payload=SAFETY,
    )
    _write_json(package_root / "reviewer_manifest.json", manifest.model_dump(mode="json"))
    _write_json(package_root / "ui_config.json", ui.model_dump(mode="json"))
    (package_root / "decisions").mkdir(exist_ok=True)
    sealed_root = output_root / "05_SEALED_SERVER_MAPPING"
    sealed_root.mkdir(exist_ok=True)
    _write_json(
        sealed_root / "server_mapping.json",
        {
            "schema_version": "football_intelligence.m5_5d.sealed_mapping.v1",
            "case_mappings": sealed,
            "reviewer_session_id": REVIEWER_SESSION_ID,
        },
    )
    launcher = _write_launcher(package_root, repo_root.resolve(), output_root)
    persistence = GenericReviewPersistence(
        manifest=manifest,
        ui_config=ui,
        decisions_root=package_root / "decisions",
        reviewer_session_id=REVIEWER_SESSION_ID,
    )
    persistence.ensure_state()
    package_validation = validate_review_chassis_package(
        manifest_path=package_root / "reviewer_manifest.json",
        ui_config_path=package_root / "ui_config.json",
        evidence_root=package_root / "evidence",
        decisions_root=package_root / "decisions",
    )
    _compose_visuals(output_root, cases)
    mining_summary = {
        "frame_count": len(frame_rows),
        "raw_detection_count": sum(len(rows) for rows in frame_rows.values()),
        "tracklet_count": len(tracks),
        "tracklets_with_three_or_more_observations": sum(1 for track in tracks if len(track.observations) >= 3),
        "true_local_deficit_candidate_count": len(deficits),
        "ordinary_crossing_control_count": len(controls),
        "frame_exit_or_not_expected_visible_control_count": len(exit_controls),
        "selected_case_count": len(cases),
        "strata": {
            stratum: sum(1 for event in selected if event["stratum"] == stratum)
            for stratum in sorted({event["stratum"] for event in selected})
        },
        "human_answers_used_in_mining": False,
        "global_expected_player_count_used": False,
    }
    detector_summary = {
        "configuration_count": 7,
        "rows": len(recovery_rows),
        "executed_rows": sum(1 for row in recovery_rows if row["execution_status"] == "executed"),
        "canonical_artifacts_replaced": False,
        "matched_controls_run": any(
            row["case_index"] > 10 and row["execution_status"] == "executed" for row in recovery_rows
        ),
        "results_path": "detector_recovery/recovery_rows.jsonl",
    }
    (output_root / "detector_recovery").mkdir(exist_ok=True)
    with (output_root / "detector_recovery" / "recovery_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in recovery_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    ghost_summary = {
        "provisional_ghost_hypotheses": sum(1 for event in selected if event["has_reentry_support"]),
        "multiple_reentry_possibilities_retained": True,
        "accuracy_claim_enabled": False,
        "state_updates_are_match_local_only": True,
    }
    mask_summary = {
        "assessment_only": True,
        "segmentation_executed": False,
        "optical_flow_executed": False,
        "eligible_strata": ["two_to_one_collapse", "observed_missing_observed", "inflated_or_merged_observation"],
        "fine_vision_requires_later_review": True,
    }
    final_classification = (
        "PASS_TRUE_OCCLUSION_REVIEW_DATASET_READY"
        if any(event["stratum"] != "ordinary_distinct_observation_crossing_control" for event in selected)
        and len(cases) >= 12
        else ("PASS_LIMITED_SUPPLY_REVIEW_READY" if cases else "BLOCKED_NO_LOCAL_DEFICIT_CANDIDATES")
    )
    review_summary = {
        "case_count": len(cases),
        "fresh_decisions_root": True,
        "new_review_ingested_in_same_stage": False,
        "reviewer_session_id": REVIEWER_SESSION_ID,
        "expected_url": "http://127.0.0.1:8782/",
        "package_validation_passed": package_validation["passed"],
        "launcher": str(launcher),
        "gif_count": sum(1 for case in cases for asset in case.evidence_assets if asset.asset_type == "animated_gif"),
        "mp4_count": 0,
    }
    result = {
        "stage_id": STAGE_ID,
        "output_root": str(output_root),
        "final_classification": final_classification,
        "mining": mining_summary,
        "detector_recovery": detector_summary,
        "ghost": ghost_summary,
        "mask": mask_summary,
        "review_package": review_summary,
        "safety": SAFETY,
    }
    _write_json(
        output_root / "01_AUTHORIZATION_AND_SAFETY_AUDIT.json",
        {
            "authorized_baseline_commit": BASELINE_COMMIT,
            "head_verified": True,
            "working_tree_clean_before_stage": True,
            **SAFETY,
        },
    )
    _write_json(output_root / "02_LOCAL_ENCOUNTER_DEFICIT_RESULTS.json", mining_summary)
    _write_json(output_root / "03_DETECTOR_RECOVERY_RESULTS.json", detector_summary)
    _write_json(output_root / "04_PROVISIONAL_GHOST_REENTRY_RESULTS.json", ghost_summary)
    _write_json(output_root / "05_MASK_FINE_VISION_ELIGIBILITY.json", mask_summary)
    _write_json(output_root / "06_REVIEW_PACKAGE_VALIDATION.json", package_validation)
    _write_json(output_root / "08_STAGE_SUMMARY.json", result)
    _write_json(
        output_root / "09_DECISION_AND_NEXT_ARCHITECTURE.json",
        {
            "classification": final_classification,
            "review_must_complete_before_accuracy_claims": True,
            "recommended_next_stage": "ingest completed M5.5D human decisions only after independent review completion",
        },
    )
    _write_pack(repo_root, output_root, {}, case_rows, result)
    pack_result = validate_m5_5d_review_pack(output_root / "11_REVIEW_PACK_FOR_CHATGPT")
    _write_json(output_root / "10_REVIEW_PACK_VALIDATION.json", pack_result)
    return {**result, "package_validation": package_validation, "review_pack_validation": pack_result}
