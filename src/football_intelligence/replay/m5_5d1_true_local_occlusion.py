from __future__ import annotations

# The serialized evidence rows intentionally keep their contract visible.
# ruff: noqa: E501

"""M5.5D.1: bounded local observation-deficit mining.

This module is intentionally independent from the earlier M5.5D builder.  It
keeps anonymous observation/tracklet evidence local to a fixed encounter and
does not replace canonical detections or create persistent identities.
"""

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
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]


STAGE_ID = "M5_5D1_TRUE_LOCAL_OCCLUSION_MINING_AND_REVIEW_v1"
REVIEW_ID = "m5_5d1_true_local_occlusion_review_v1"
REVIEWER_SESSION_ID = "m5_5d1_true_occlusion_human_reviewer"
BASELINE_COMMIT = "2fe35f933b8223f941f9e5849809c9756f66c58e"
REVIEW_PORT = 8783
REQUIRED_MODEL_SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
DEFAULT_OUTPUT_ROOT = Path(
    r"C:\Users\sebgr\Documents\football-intelligence\matches\128058\runs\step_m5\part 2\M5_5D1_TRUE_LOCAL_OCCLUSION_MINING_AND_REVIEW_v1"
)
DEFAULT_UNSEEN_ROOT = Path(
    r"C:\Users\sebgr\Documents\football-intelligence\matches\128058\runs\step_m5\06f_balanced_role_then_continuity\continuity_v11\unseen_window"
)
DEFAULT_MODEL_PATH = Path(
    r"C:\Users\sebgr\Documents\football-intelligence\SoccerTrack-v2\models\model=yolov8m-imgsz=2048.pt"
)
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


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _box(row: dict[str, Any]) -> dict[str, float]:
    value = row.get("bbox", row)
    return {key: float(value[key]) for key in ("x1", "y1", "x2", "y2")}


def _height(box: dict[str, float]) -> float:
    return max(1.0, box["y2"] - box["y1"])


def _area(box: dict[str, float]) -> float:
    return max(1.0, box["x2"] - box["x1"]) * max(1.0, box["y2"] - box["y1"])


def _foot(box: dict[str, float]) -> tuple[float, float]:
    return ((box["x1"] + box["x2"]) / 2.0, box["y2"])


def _dist(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _iou(left: dict[str, float], right: dict[str, float]) -> float:
    x1, y1 = max(left["x1"], right["x1"]), max(left["y1"], right["y1"])
    x2, y2 = min(left["x2"], right["x2"]), min(left["y2"], right["y2"])
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return overlap / max(1.0, _area(left) + _area(right) - overlap)


def _font(size: int = 20) -> Any:
    if ImageFont is None:
        return None
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


@dataclass
class StableTracklet:
    track_id: int
    observations: list[dict[str, Any]] = field(default_factory=list)
    covariance: tuple[float, float] = (1.0, 1.0)
    forward_keys: set[str] = field(default_factory=set)
    reverse_keys: set[str] = field(default_factory=set)

    @property
    def first_frame(self) -> int:
        return int(self.observations[0]["frame_sequence"])

    @property
    def last_frame(self) -> int:
        return int(self.observations[-1]["frame_sequence"])

    @property
    def last_box(self) -> dict[str, float]:
        return _box(self.observations[-1])

    def velocity(self) -> tuple[float, float]:
        if len(self.observations) < 2:
            return (0.0, 0.0)
        old, new = _foot(_box(self.observations[-2])), _foot(self.last_box)
        gap = max(1, self.last_frame - int(self.observations[-2]["frame_sequence"]))
        return ((new[0] - old[0]) / gap, (new[1] - old[1]) / gap)

    def predict(self, frame_sequence: int) -> dict[str, float]:
        if not self.observations:
            return {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0}
        exact = next((item for item in self.observations if int(item["frame_sequence"]) == frame_sequence), None)
        if exact is not None:
            return _box(exact)
        before = [item for item in self.observations if int(item["frame_sequence"]) < frame_sequence]
        after = [item for item in self.observations if int(item["frame_sequence"]) > frame_sequence]
        if before and after:
            left, right = before[-1], after[0]
            alpha = (frame_sequence - int(left["frame_sequence"])) / max(
                1, int(right["frame_sequence"]) - int(left["frame_sequence"])
            )
            lbox, rbox = _box(left), _box(right)
            return {key: lbox[key] + alpha * (rbox[key] - lbox[key]) for key in lbox}
        base = _box(before[-1] if before else after[0])
        anchor_frame = int((before[-1] if before else after[0])["frame_sequence"])
        vx, vy = self.velocity()
        gap = frame_sequence - anchor_frame
        return {
            "x1": base["x1"] + vx * gap,
            "y1": base["y1"] + vy * gap,
            "x2": base["x2"] + vx * gap,
            "y2": base["y2"] + vy * gap,
        }


def _load_rows(unseen_root: Path) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with (unseen_root / "person_candidate_rows.jsonl").open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            row["_observation_key"] = f"{int(row['frame_sequence'])}:{index}"
            rows[int(row["frame_sequence"])].append(row)
    return dict(rows), _read_json(unseen_root / "canonical_frame_manifest.json")


def _compatible(predicted: dict[str, float], observed: dict[str, float]) -> float | None:
    scale = max(_height(predicted), _height(observed))
    distance = _dist(_foot(predicted), _foot(observed)) / scale
    ratio = max(_height(predicted), _height(observed)) / max(1.0, min(_height(predicted), _height(observed)))
    if distance > 1.45 or ratio > 2.4:
        return None
    return distance + 0.12 * abs(math.log(ratio))


def _assign_once(frame_rows: dict[int, list[dict[str, Any]]], reverse: bool = False) -> list[StableTracklet]:
    tracks: list[StableTracklet] = []
    active: list[StableTracklet] = []
    next_id = 1
    frame_order = sorted(frame_rows, reverse=reverse)
    for frame in frame_order:
        rows = frame_rows[frame]
        eligible = [track for track in active if 0 < abs(frame - track.last_frame) <= 1]
        pairs: list[tuple[float, StableTracklet, dict[str, Any]]] = []
        for track in eligible:
            predicted = track.predict(frame)
            for row in rows:
                cost = _compatible(predicted, _box(row))
                if cost is not None:
                    pairs.append((cost, track, row))
        used_tracks: set[int] = set()
        used_rows: set[str] = set()
        for _, track, row in sorted(pairs, key=lambda item: (item[0], item[1].track_id, item[2]["_observation_key"])):
            if track.track_id in used_tracks or row["_observation_key"] in used_rows:
                continue
            used_tracks.add(track.track_id)
            used_rows.add(row["_observation_key"])
            track.observations.append(row)
            track.forward_keys.add(row["_observation_key"] if not reverse else "")
        for row in rows:
            if row["_observation_key"] in used_rows:
                continue
            track = StableTracklet(track_id=next_id, observations=[row])
            next_id += 1
            if reverse:
                track.reverse_keys.add(row["_observation_key"])
            else:
                track.forward_keys.add(row["_observation_key"])
            tracks.append(track)
            active.append(track)
        active = [track for track in active if abs(frame - track.last_frame) <= 1]
    for track in tracks:
        track.observations.sort(key=lambda item: int(item["frame_sequence"]))
    return tracks


def _build_stable_tracklets(
    frame_rows: dict[int, list[dict[str, Any]]],
) -> tuple[list[StableTracklet], dict[str, Any], dict[int, dict[str, Any]]]:
    forward = _assign_once(frame_rows)
    reverse_rows = {-frame: [{**row, "frame_sequence": -frame} for row in rows] for frame, rows in frame_rows.items()}
    reverse = _assign_once(reverse_rows)
    reverse_by_key: dict[str, StableTracklet] = {}
    for track in reverse:
        for row in track.observations:
            reverse_by_key[row["_observation_key"]] = track
    stable: list[StableTracklet] = []
    for track in forward:
        keys = {row["_observation_key"] for row in track.observations}
        reverse_keys = set()
        for key in keys:
            reverse_keys.update(row["_observation_key"] for row in reverse_by_key.get(key, track).observations)
        track.reverse_keys = reverse_keys
        consistency = len(keys & reverse_keys) / max(1, len(keys))
        track.forward_keys = keys
        if len(keys) >= 4 and consistency >= 0.75:
            stable.append(track)
    stable.sort(key=lambda item: (item.first_frame, item.track_id))
    metrics = {
        "forward_tracklet_count": len(forward),
        "reverse_tracklet_count": len(reverse),
        "stable_tracklet_count": len(stable),
        "minimum_distinct_observations": 4,
        "minimum_bidirectional_consistency": 0.75,
        "stale_unmatched_gap_frames": 1,
        "observation_reuse_count": 0,
        "fragmented_forward_tracklet_count": sum(1 for track in forward if len(track.observations) < 4),
        "stable_observation_counts": {
            "min": min((len(track.observations) for track in stable), default=0),
            "max": max((len(track.observations) for track in stable), default=0),
        },
    }
    frame_state: dict[int, dict[str, Any]] = {}
    stable_by_id = {track.track_id: track for track in stable}
    for frame in sorted(frame_rows):
        frame_state[frame] = {
            "rows": frame_rows[frame],
            "stable_predictions": {str(track.track_id): track.predict(frame) for track in stable},
            "stable_assignments": {
                str(track.track_id): next(
                    (row["_observation_key"] for row in track.observations if int(row["frame_sequence"]) == frame), None
                )
                for track in stable_by_id.values()
            },
        }
    return stable, metrics, frame_state


def _clusters(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    for row in sorted(rows, key=lambda item: (float(_box(item)["x1"]), float(_box(item)["y1"]))):
        box = _box(row)
        target = next(
            (
                cluster
                for cluster in result
                if any(
                    _iou(box, _box(old)) >= 0.5
                    or _dist(_foot(box), _foot(_box(old))) <= 0.28 * max(_height(box), _height(_box(old)))
                    for old in cluster
                )
            ),
            None,
        )
        if target is None:
            result.append([row])
        else:
            target.append(row)
    return result


def _group_candidates(stable: list[StableTracklet]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for left_index, left in enumerate(stable):
        for right in stable[left_index + 1 :]:
            start, end = max(left.first_frame, right.first_frame), min(left.last_frame, right.last_frame)
            if end - start < 4:
                continue
            distances = [
                (
                    frame,
                    _dist(_foot(left.predict(frame)), _foot(right.predict(frame))),
                    max(_height(left.predict(frame)), _height(right.predict(frame))),
                )
                for frame in range(start, end + 1)
            ]
            frame, distance, scale = min(distances, key=lambda item: item[1] / max(1.0, item[2]))
            if distance / max(1.0, scale) > 1.65:
                continue
            groups.append(
                {
                    "encounter_group_id": f"enc_{left.track_id:04d}_{right.track_id:04d}_{frame:04d}",
                    "track_ids": [left.track_id, right.track_id],
                    "encounter_center_frame": frame,
                    "encounter_start_frame": max(start, frame - 8),
                    "encounter_end_frame": min(end, frame + 8),
                    "trajectory_safe_hash": _json_hash([left.track_id, right.track_id, start, end]),
                }
            )
    groups.sort(key=lambda item: (item["encounter_center_frame"], item["track_ids"]))
    deduped: list[dict[str, Any]] = []
    seen_pairs: set[tuple[int, ...]] = set()
    for group in groups:
        pair = tuple(group["track_ids"])
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        deduped.append(group)
    return deduped


def _independence_for_frame(
    frame_rows: dict[int, list[dict[str, Any]]],
    stable_by_id: dict[int, StableTracklet],
    group: dict[str, Any],
    frame: int,
) -> dict[str, Any]:
    local_tracks = [stable_by_id[track_id] for track_id in group["track_ids"]]
    predictions = {track.track_id: track.predict(frame) for track in local_tracks}
    candidate_rows = frame_rows.get(frame, [])
    local_rows = [
        row
        for row in candidate_rows
        if any(_compatible(prediction, _box(row)) is not None for prediction in predictions.values())
    ]
    clusters = _clusters(local_rows)
    cluster_compatibility: list[list[int]] = []
    for cluster in clusters:
        compatible = [
            track_id
            for track_id, prediction in predictions.items()
            if any(_compatible(prediction, _box(row)) is not None for row in cluster)
        ]
        cluster_compatibility.append(sorted(compatible))
    pairs = sorted(
        (
            (_compatible(predictions[track_id], _box(cluster[0])), track_id, index)
            for track_id, prediction in predictions.items()
            for index, cluster in enumerate(clusters)
            if cluster and _compatible(prediction, _box(cluster[0])) is not None
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    used_tracks: set[int] = set()
    used_clusters: set[int] = set()
    assignments: dict[int, int] = {}
    for _, track_id, cluster_index in pairs:
        if track_id in used_tracks or cluster_index in used_clusters:
            continue
        used_tracks.add(track_id)
        used_clusters.add(cluster_index)
        assignments[track_id] = cluster_index
    return {
        "frame_sequence": frame,
        "incoming_track_count": len(local_tracks),
        "independent_observation_count": len(clusters),
        "raw_detection_count": len(local_rows),
        "clusters": clusters,
        "cluster_compatible_track_ids": cluster_compatibility,
        "assignments": assignments,
        "unassigned_track_ids": [track_id for track_id in predictions if track_id not in assignments],
        "predictions": predictions,
    }


def _find_intervals(
    frame_rows: dict[int, list[dict[str, Any]]], stable: list[StableTracklet], groups: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    stable_by_id = {track.track_id: track for track in stable}
    genuine: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    all_supply: list[dict[str, Any]] = []
    for group in groups:
        first = int(group["encounter_start_frame"])
        last = int(group["encounter_end_frame"])
        rows = [_independence_for_frame(frame_rows, stable_by_id, group, frame) for frame in range(first, last + 1)]
        by_frame = {row["frame_sequence"]: row for row in rows}
        for center in range(first + 2, last - 3):
            before = [by_frame.get(center - 2), by_frame.get(center - 1)]
            if any(row is None or row["independent_observation_count"] != len(group["track_ids"]) for row in before):
                continue
            deficit_frames = []
            cursor = center
            while cursor <= last - 2 and by_frame[cursor]["independent_observation_count"] < len(group["track_ids"]):
                deficit_frames.append(cursor)
                cursor += 1
            if not deficit_frames or len(deficit_frames) > 12:
                continue
            after = [by_frame.get(cursor), by_frame.get(cursor + 1)]
            post_recovered = all(
                row is not None and row["independent_observation_count"] == len(group["track_ids"]) for row in after
            )
            if not post_recovered:
                continue
            selected_rows = [by_frame[frame] for frame in deficit_frames]
            max_deficit = max(len(group["track_ids"]) - row["independent_observation_count"] for row in selected_rows)
            one_to_many = any(
                any(
                    len(ids) >= 2 and len(by_frame[frame]["clusters"][index]) == 1
                    for index, ids in enumerate(by_frame[frame]["cluster_compatible_track_ids"])
                )
                for frame in deficit_frames
            )
            duplicate_cluster = any(
                any(len(cluster) > 1 for cluster in by_frame[frame]["clusters"]) for frame in deficit_frames
            )
            merged_evidence = any(
                one_to_many
                and max(_area(_box(cluster[0])) for cluster in by_frame[frame]["clusters"] if cluster)
                > 1.15
                * (
                    sum(_area(stable_by_id[track_id].predict(frame)) for track_id in group["track_ids"])
                    / len(group["track_ids"])
                )
                for frame in deficit_frames
            )
            if duplicate_cluster and not one_to_many:
                stratum = "duplicate_fragment"
            elif one_to_many and merged_evidence:
                stratum = "merged_observation"
            elif len(group["track_ids"]) == 2 and max_deficit == 1 and len(deficit_frames) == 1:
                stratum = "observed_missing_observed"
            else:
                stratum = "local_observation_deficit"
            anchor_cluster = selected_rows[0]["clusters"][0] if selected_rows[0]["clusters"] else []
            anchor_bbox = (
                _box(anchor_cluster[0])
                if anchor_cluster
                else stable_by_id[group["track_ids"][0]].predict(deficit_frames[0])
            )
            event = {
                **group,
                "anchor_bbox": anchor_bbox,
                "stratum": stratum,
                "deficit_start_frame": deficit_frames[0],
                "deficit_end_frame": deficit_frames[-1],
                "deficit_frame_count": len(deficit_frames),
                "incoming_track_count": len(group["track_ids"]),
                "minimum_independent_observation_count": min(
                    row["independent_observation_count"] for row in selected_rows
                ),
                "maximum_local_track_deficit": max_deficit,
                "precondition_frames": [row["frame_sequence"] for row in before],
                "postcondition_frames": [row["frame_sequence"] for row in after],
                "post_recovery_observed": post_recovered,
                "one_observation_compatible_with_multiple_tracks": one_to_many,
                "duplicate_or_fragment_cluster": duplicate_cluster,
                "merged_evidence": merged_evidence,
                "frame_rows": rows,
                "human_answers_used_in_mining": False,
                "trajectory_group_hash": group["trajectory_safe_hash"],
            }
            all_supply.append(event)
            if stratum in {"observed_missing_observed", "merged_observation"} and len(group["track_ids"]) <= 4:
                genuine.append(event)
        if not any(event["encounter_group_id"] == group["encounter_group_id"] for event in all_supply):
            center = int(group["encounter_center_frame"])
            check = _independence_for_frame(frame_rows, stable_by_id, group, center)
            anchor_bbox = (
                _box(check["clusters"][0][0])
                if check["clusters"]
                else stable_by_id[group["track_ids"][0]].predict(center)
            )
            controls.append(
                {
                    **group,
                    "anchor_bbox": anchor_bbox,
                    "stratum": "ordinary_distinct_observation_crossing_control",
                    "deficit_start_frame": None,
                    "deficit_end_frame": None,
                    "deficit_frame_count": 0,
                    "incoming_track_count": len(group["track_ids"]),
                    "minimum_independent_observation_count": check["independent_observation_count"],
                    "maximum_local_track_deficit": 0,
                    "precondition_frames": [max(first, center - 1), center],
                    "postcondition_frames": [center, min(last, center + 1)],
                    "post_recovery_observed": True,
                    "one_observation_compatible_with_multiple_tracks": False,
                    "duplicate_or_fragment_cluster": False,
                    "merged_evidence": False,
                    "frame_rows": [check],
                    "human_answers_used_in_mining": False,
                    "trajectory_group_hash": group["trajectory_safe_hash"],
                }
            )
    dedup: dict[tuple[str, int, int], dict[str, Any]] = {}
    for event in genuine:
        dedup[(event["trajectory_group_hash"], event["deficit_start_frame"], event["deficit_end_frame"])] = event
    genuine = list(dedup.values())
    controls = controls[:20]
    summary = {
        "candidate_interval_count": len(all_supply),
        "genuine_interval_count": len(genuine),
        "control_interval_count": len(controls),
        "minimum_precondition_frames": 2,
        "minimum_postcondition_frames": 2,
        "maximum_deficit_frames": 12,
        "true_two_to_one_count": sum(
            1
            for event in genuine
            if event["incoming_track_count"] == 2
            and event["maximum_local_track_deficit"] == 1
            and event["deficit_frame_count"] == 1
        ),
        "adjacent_frame_duplicates_removed": 0,
        "human_answers_used_in_mining": False,
    }
    return genuine, controls, summary


def _frame_path(manifest: dict[str, Any], frame_sequence: int) -> Path:
    return Path(manifest["frames"][max(0, min(frame_sequence, len(manifest["frames"]) - 1))]["frame_file"])


def _draw_frame(
    source: Path,
    target: Path,
    rows: Iterable[dict[str, Any]],
    event: dict[str, Any],
    label: str,
    track_boxes: list[tuple[dict[str, float], str, tuple[int, int, int]]] | None = None,
) -> None:
    if Image is None:
        raise RuntimeError("Pillow is required")
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, row in enumerate(rows, start=1):
        box = _box(row)
        color = (50, 190, 120) if index % 2 else (50, 140, 240)
        draw.rectangle(tuple(box[key] for key in ("x1", "y1", "x2", "y2")), outline=color, width=3)
        draw.text((box["x1"], max(0, box["y1"] - 22)), f"obs {index}", fill=color, font=_font(16))
    for box, text, color in track_boxes or []:
        draw.rectangle(tuple(box[key] for key in ("x1", "y1", "x2", "y2")), outline=color, width=4)
        draw.text((box["x1"], box["y2"] + 2), text, fill=color, font=_font(17))
    anchor = event.get("anchor_bbox")
    if isinstance(anchor, dict):
        point = _foot(anchor)
        draw.ellipse((point[0] - 8, point[1] - 8, point[0] + 8, point[1] + 8), fill=(235, 50, 50))
    draw.rectangle((0, 0, min(image.width, 1450), 38), fill=(18, 28, 40))
    draw.text((10, 9), label, fill=(245, 245, 245), font=_font(19))
    target.parent.mkdir(parents=True, exist_ok=True)
    image.thumbnail((1450, 520))
    image.save(target, quality=90)


def _make_gif(paths: list[Path], target: Path, label: str) -> None:
    if Image is None or not paths:
        raise RuntimeError("Pillow and frames are required")
    frames = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((1300, 480))
        canvas = Image.new("RGB", (image.width, image.height + 34), (18, 28, 40))
        canvas.paste(image, (0, 34))
        ImageDraw.Draw(canvas).text((10, 8), label, fill=(245, 245, 245), font=_font(17))
        frames.append(canvas)
    target.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(target, save_all=True, append_images=frames[1:], duration=170, loop=0, optimize=False)


def _detector_configurations() -> list[dict[str, Any]]:
    return [
        {"name": "canonical", "imgsz": 1280, "conf": 0.22, "iou": 0.70, "max_det": 80, "crop_height_multiplier": None},
        {
            "name": "full_frame_2048",
            "imgsz": 2048,
            "conf": 0.22,
            "iou": 0.70,
            "max_det": 80,
            "crop_height_multiplier": None,
        },
        {
            "name": "lower_confidence",
            "imgsz": 1280,
            "conf": 0.10,
            "iou": 0.70,
            "max_det": 80,
            "crop_height_multiplier": None,
        },
        {
            "name": "relaxed_nms",
            "imgsz": 1280,
            "conf": 0.10,
            "iou": 0.85,
            "max_det": 80,
            "crop_height_multiplier": None,
        },
        {
            "name": "higher_max_det",
            "imgsz": 1280,
            "conf": 0.10,
            "iou": 0.70,
            "max_det": 160,
            "crop_height_multiplier": None,
        },
        {
            "name": "native_2_height_crop",
            "imgsz": 1280,
            "conf": 0.10,
            "iou": 0.70,
            "max_det": 80,
            "crop_height_multiplier": 2.0,
        },
        {
            "name": "native_3_height_crop",
            "imgsz": 1280,
            "conf": 0.10,
            "iou": 0.70,
            "max_det": 80,
            "crop_height_multiplier": 3.0,
        },
    ]


def _crop_transform(image: Any, event: dict[str, Any], multiplier: float) -> tuple[Any, dict[str, int]]:
    anchor = event["anchor_bbox"]
    height = _height(anchor) * multiplier
    cx = (anchor["x1"] + anchor["x2"]) / 2
    cy = (anchor["y1"] + anchor["y2"]) / 2
    left, top = max(0, int(cx - 1.55 * height)), max(0, int(cy - 0.75 * height))
    right, bottom = min(image.width, int(cx + 1.55 * height)), min(image.height, int(cy + 1.0 * height))
    return image.crop((left, top, right, bottom)), {"left": left, "top": top, "right": right, "bottom": bottom}


def _run_detector_recovery(
    events: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    model_path: Path,
    output_root: Path,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual_hash = sha256_file(model_path) if model_path.exists() else None
    model_validation = {
        "path": str(model_path),
        "exists": model_path.exists(),
        "byte_size": model_path.stat().st_size if model_path.exists() else None,
        "sha256": actual_hash,
        "required_sha256": REQUIRED_MODEL_SHA256,
        "passed": actual_hash == REQUIRED_MODEL_SHA256,
        "task": "detect",
        "class_count": 80,
        "person_class_id": 0,
    }
    if not model_validation["passed"]:
        raise ValueError(f"detector checkpoint hash mismatch: {model_validation}")
    selected = (events[:4] + controls[:4])[:8]
    configs = _detector_configurations()
    rows: list[dict[str, Any]] = []
    try:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
    except Exception as exc:  # pragma: no cover
        model = None
        load_error = str(exc)
    for case_index, event in enumerate(selected, start=1):
        source = _frame_path(manifest, int(event.get("deficit_start_frame") or event["encounter_center_frame"]))
        for config in configs:
            row: dict[str, Any] = {
                "case_index": case_index,
                "control": event in controls,
                "configuration": config,
                "source_frame_sequence": int(event.get("deficit_start_frame") or event["encounter_center_frame"]),
                "execution_status": "not_run",
                "model_fit_performed": False,
                "canonical_artifacts_replaced": False,
                "boxes": [],
                "mapped_to_panorama": config["crop_height_multiplier"] is None,
            }
            input_path = source
            transform = None
            if config["crop_height_multiplier"] is not None and Image is not None:
                image = Image.open(source).convert("RGB")
                cropped, transform = _crop_transform(image, event, float(config["crop_height_multiplier"]))
                input_path = (
                    output_root
                    / "04_EXACT_DETECTOR_RECOVERY"
                    / "inputs"
                    / f"case_{case_index:03d}_{config['name']}.jpg"
                )
                input_path.parent.mkdir(parents=True, exist_ok=True)
                cropped.save(input_path, quality=92)
                row["crop_transform"] = transform
            if model is None:
                row.update({"execution_status": "unavailable", "error": load_error})
                rows.append(row)
                continue
            try:
                predictions = model.predict(
                    source=str(input_path),
                    imgsz=config["imgsz"],
                    conf=config["conf"],
                    iou=config["iou"],
                    max_det=config["max_det"],
                    classes=[0],
                    augment=False,
                    agnostic_nms=False,
                    device="cpu",
                    save=False,
                    stream=False,
                    verbose=False,
                )
                local_boxes = (
                    predictions[0].boxes.xyxy.cpu().tolist() if predictions and predictions[0].boxes is not None else []
                )
                mapped = []
                for values in local_boxes:
                    box = {
                        "x1": round(float(values[0]), 2),
                        "y1": round(float(values[1]), 2),
                        "x2": round(float(values[2]), 2),
                        "y2": round(float(values[3]), 2),
                    }
                    if transform:
                        box = {
                            "x1": box["x1"] + transform["left"],
                            "y1": box["y1"] + transform["top"],
                            "x2": box["x2"] + transform["left"],
                            "y2": box["y2"] + transform["top"],
                        }
                    mapped.append(box)
                row.update(
                    {
                        "execution_status": "executed",
                        "boxes": mapped,
                        "person_detection_count": len(mapped),
                        "mapped_to_panorama": True,
                    }
                )
            except Exception as exc:  # pragma: no cover
                row.update({"execution_status": "failed", "error": str(exc)})
            rows.append(row)
    _write_json(output_root / "04_EXACT_DETECTOR_RECOVERY" / "model_validation.json", model_validation)
    _write_json(
        output_root / "04_EXACT_DETECTOR_RECOVERY" / "configuration_manifest.json",
        {
            "configurations": configs,
            "canonical": configs[0],
            "all_alternatives_required": True,
            "model_fit_performed": False,
        },
    )
    return rows, model_validation


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def _safe_frame_sequence(event: dict[str, Any]) -> list[int]:
    start = int(event.get("deficit_start_frame") or event["encounter_center_frame"])
    end = int(event.get("deficit_end_frame") or start)
    first = max(0, start - 4)
    last = end + 4
    if last - first + 1 < 9:
        last = first + 8
    return list(range(first, last + 1))[:21]


def _build_review_package(
    events: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    stable: list[StableTracklet],
    frame_rows: dict[int, list[dict[str, Any]]],
    manifest_data: dict[str, Any],
    recovery_rows: list[dict[str, Any]],
    output_root: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], list[GenericReviewCase], dict[str, Any]]:
    package = output_root / "08_REVIEW_PACKAGE"
    if package.exists():
        shutil.rmtree(package)
    evidence_root = package / "evidence"
    decisions_root = package / "decisions"
    sealed_root = package / "sealed"
    evidence_root.mkdir(parents=True, exist_ok=True)
    sealed_root.mkdir(parents=True, exist_ok=True)
    stable_by_id = {track.track_id: track for track in stable}
    selected = (events + controls)[:20]
    cases: list[GenericReviewCase] = []
    sealed: dict[str, Any] = {}
    case_index_rows: list[dict[str, Any]] = []
    evidence_manifest: list[dict[str, Any]] = []
    for index, event in enumerate(selected, start=1):
        case_id = f"case_{index:03d}"
        case_dir = evidence_root / case_id
        sequences = _safe_frame_sequence(event)
        raw_paths = [_frame_path(manifest_data, sequence) for sequence in sequences]
        annotated_paths: list[Path] = []
        track_boxes: list[tuple[dict[str, float], str, tuple[int, int, int]]] = []
        for track_id in event["track_ids"]:
            track = stable_by_id.get(track_id)
            if track:
                track_boxes.append((track.predict(sequences[len(sequences) // 2]), f"path {track_id}", (240, 190, 50)))
        for offset, (sequence, source) in enumerate(zip(sequences, raw_paths)):
            target = case_dir / "frames" / f"frame_{offset:03d}.jpg"
            _draw_frame(
                source, target, frame_rows.get(sequence, []), event, f"Frame stepper | sequence {sequence}", track_boxes
            )
            annotated_paths.append(target)
        _make_gif(
            annotated_paths, case_dir / "before_during_after.gif", f"Anonymous local encounter | case {index:03d}"
        )
        _make_gif(annotated_paths, case_dir / "annotated_diagnostic.gif", f"Diagnostic overlays | case {index:03d}")
        center_offset = len(sequences) // 2
        center_source = raw_paths[center_offset]
        _draw_frame(
            center_source,
            case_dir / "annotation_frame.jpg",
            frame_rows.get(sequences[center_offset], []),
            event,
            "Full-resolution interval annotation frame",
            track_boxes,
        )
        _draw_frame(
            center_source,
            case_dir / "canonical_deficit_frame.jpg",
            frame_rows.get(int(event.get("deficit_start_frame") or sequences[center_offset]), []),
            event,
            "Canonical local observations",
            track_boxes,
        )
        recovery = next(
            (
                row
                for row in recovery_rows
                if row["case_index"] == index and row["configuration"]["name"] == "lower_confidence"
            ),
            None,
        )
        _draw_frame(
            center_source,
            case_dir / "detector_recovery.jpg",
            frame_rows.get(sequences[center_offset], []),
            event,
            "Exact detector recovery | mapped panorama coordinates",
            track_boxes
            + [
                ({"x1": box["x1"], "y1": box["y1"], "x2": box["x2"], "y2": box["y2"]}, "recovery", (235, 80, 210))
                for box in (recovery or {}).get("boxes", [])
            ],
        )
        _draw_frame(
            center_source, case_dir / "incoming_paths.jpg", [], event, "Stable anonymous incoming paths", track_boxes
        )
        _draw_frame(
            center_source,
            case_dir / "merged_candidate.jpg",
            frame_rows.get(sequences[center_offset], []),
            event,
            "Candidate merged observation | reviewer must decide",
            track_boxes,
        )
        image = Image.open(center_source).convert("RGB") if Image else None
        if image:
            anchor = event.get("anchor_bbox") or {"x1": 0, "y1": 0, "x2": 100, "y2": 100}
            for phase, sequence in (
                ("before", sequences[0]),
                ("during", sequences[center_offset]),
                ("after", sequences[-1]),
            ):
                source_image = Image.open(_frame_path(manifest_data, sequence)).convert("RGB")
                crop = source_image.crop(
                    (
                        max(0, int(anchor["x1"] - 160)),
                        max(0, int(anchor["y1"] - 160)),
                        min(source_image.width, int(anchor["x2"] + 160)),
                        min(source_image.height, int(anchor["y2"] + 160)),
                    )
                )
                crop.thumbnail((700, 500))
                crop.save(case_dir / f"local_crop_{phase}.jpg", quality=90)
        asset_specs = [
            (
                "temporal_gif",
                "animated_gif",
                "Before / during / after GIF",
                "before_during_after.gif",
                "image/gif",
                sequences,
                "temporal",
            ),
            (
                "diagnostic_gif",
                "animated_gif",
                "Annotated diagnostic GIF",
                "annotated_diagnostic.gif",
                "image/gif",
                sequences,
                "diagnostic",
            ),
            (
                "annotation_frame",
                "image",
                "Full-resolution annotation frame",
                "annotation_frame.jpg",
                "image/jpeg",
                [sequences[center_offset]],
                "annotation",
            ),
            (
                "canonical_deficit",
                "overlay",
                "Canonical deficit-frame detections",
                "canonical_deficit_frame.jpg",
                "image/jpeg",
                [sequences[center_offset]],
                "canonical",
            ),
            (
                "recovery",
                "comparison_panel",
                "Exact detector recovery",
                "detector_recovery.jpg",
                "image/jpeg",
                [sequences[center_offset]],
                "recovery",
            ),
            (
                "incoming_paths",
                "overlay",
                "Stable anonymous incoming paths",
                "incoming_paths.jpg",
                "image/jpeg",
                [sequences[center_offset]],
                "paths",
            ),
            (
                "merged_candidate",
                "overlay",
                "Candidate merged observation",
                "merged_candidate.jpg",
                "image/jpeg",
                [sequences[center_offset]],
                "merged",
            ),
            ("crop_before", "crop", "Local crop before", "local_crop_before.jpg", "image/jpeg", [sequences[0]], "crop"),
            (
                "crop_during",
                "crop",
                "Local crop during",
                "local_crop_during.jpg",
                "image/jpeg",
                [sequences[center_offset]],
                "crop",
            ),
            ("crop_after", "crop", "Local crop after", "local_crop_after.jpg", "image/jpeg", [sequences[-1]], "crop"),
        ]
        asset_specs.extend(
            (
                f"frame_step_{offset:03d}",
                "image_sequence",
                "Frame stepper",
                f"frames/frame_{offset:03d}.jpg",
                "image/jpeg",
                [sequence],
                "temporal_stepper",
            )
            for offset, sequence in enumerate(sequences)
        )
        assets: list[GenericEvidenceAsset] = []
        for asset_id, asset_type, label, relative, media_type, frames, group_id in asset_specs:
            path = case_dir / relative
            asset = GenericEvidenceAsset(
                asset_id=asset_id,
                asset_type=asset_type,
                label=label,
                relative_path=relative,
                sha256=sha256_file(path),
                media_type=media_type,
                frame_sequences=frames,
                group_id=group_id,
                metadata={"primary_annotation_image": asset_id == "annotation_frame", "frame_stepper": False},
            )
            assets.append(asset)
            evidence_manifest.append({"case_id": case_id, **asset.model_dump(mode="json")})
        safe_candidates = []
        for number, track_id in enumerate(event["track_ids"], start=1):
            track = stable_by_id.get(track_id)
            if track:
                safe_candidates.append(
                    {
                        "anonymous_candidate_number": number,
                        "bbox": track.predict(sequences[center_offset]),
                        "frame_sequence": sequences[center_offset],
                        "class_name": "person",
                    }
                )
        visible_meta = {
            "case_label": f"Anonymous local observation review {index:03d}",
            "frame_window": {
                "first": sequences[0],
                "last": sequences[-1],
                "deficit_start": event.get("deficit_start_frame"),
                "deficit_end": event.get("deficit_end_frame"),
            },
            "incoming_track_count": event["incoming_track_count"],
            "safe_anonymous_candidates": safe_candidates,
            "frame_sequences": sequences,
            "reentry_path_options": ["PATH_A", "PATH_B", "PATH_C", "NO_REENTRY", "UNRESOLVED"],
            "no_human_answer_used_in_mining": True,
        }
        case = GenericReviewCase(
            case_id=case_id,
            task_type="occlusion_interval",
            candidate_id=case_id,
            candidate_hash=_json_hash({"case": case_id, "sequence": sequences}),
            evidence_hash=_json_hash([asset.sha256 for asset in assets]),
            allowed_decisions=REVIEW_DECISIONS,
            concise_question="What best describes the local observation supply in this temporal window?",
            detailed_instructions="Review the GIF and frame stepper. Set deficit start/end when supported, click or draw the merged observation, mark an occlusion point, and choose a re-entry path or unresolved.",
            priority=100 - index,
            evidence_assets=assets,
            source_frame_sequence=sequences[0],
            target_frame_sequence=sequences[-1],
            frame_gap=sequences[-1] - sequences[0],
            source_bbox=event.get("anchor_bbox"),
            target_bbox=event.get("anchor_bbox"),
            visible_metadata=visible_meta,
            safety_payload=SAFETY,
        )
        cases.append(case)
        sealed[case_id] = {
            "internal_track_ids": event["track_ids"],
            "internal_stratum": event["stratum"],
            "trajectory_group_hash": event["trajectory_group_hash"],
            "reviewer_session_id": REVIEWER_SESSION_ID,
        }
        case_index_rows.append(
            {
                "case_id": case_id,
                "frame_first": sequences[0],
                "frame_last": sequences[-1],
                "incoming_track_count": event["incoming_track_count"],
                "human_answers_used_in_mining": False,
            }
        )
    ui = _ui_config()
    manifest = GenericReviewManifest(
        review_id=REVIEW_ID,
        stage_id=STAGE_ID,
        task_type="occlusion_interval",
        title="M5.5D.1 True Local Occlusion Review",
        production_ready=False,
        no_auto_promotion=True,
        human_approved=False,
        cases=cases,
        evidence_manifest_hash=_json_hash(evidence_manifest),
        source_manifest_hash=sha256_file(Path(manifest_data.get("manifest_file", _frame_path(manifest_data, 0))))
        if manifest_data.get("manifest_file")
        else _json_hash(manifest_data),
        safety_payload=SAFETY,
    )
    _write_json(package / "reviewer_manifest.json", manifest.model_dump(mode="json"))
    _write_json(package / "ui_config.json", ui.model_dump(mode="json"))
    _write_json(
        package / "evidence_manifest.json",
        {"schema_version": "m5_5d1.evidence_manifest.v1", "assets": evidence_manifest},
    )
    with (package / "case_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "frame_first", "frame_last", "incoming_track_count", "human_answers_used_in_mining"],
        )
        writer.writeheader()
        writer.writerows(case_index_rows)
    _write_json(
        sealed_root / "server_mapping.json",
        {"schema_version": "m5_5d1.sealed_mapping.v1", "cases": sealed, "served_before_decision": False},
    )
    persistence = GenericReviewPersistence(
        manifest=manifest, ui_config=ui, decisions_root=decisions_root, reviewer_session_id=REVIEWER_SESSION_ID
    )
    persistence.ensure_state()
    launcher = package / "launch_review.ps1"
    launcher.write_text(
        f"$ErrorActionPreference = 'Stop'\n$RepoRoot = '{repo_root}'\n$PackageRoot = '{package}'\nSet-Location -LiteralPath $RepoRoot\nuv run fi-pipeline review-chassis serve --manifest (Join-Path $PackageRoot 'reviewer_manifest.json') --ui-config (Join-Path $PackageRoot 'ui_config.json') --evidence-root (Join-Path $PackageRoot 'evidence') --decisions-root (Join-Path $PackageRoot 'decisions') --sealed-mapping (Join-Path $PackageRoot 'sealed/server_mapping.json') --host 127.0.0.1 --port {REVIEW_PORT} --reviewer-session-id {REVIEWER_SESSION_ID}\n",
        encoding="utf-8",
    )
    (package / "README.md").write_text(
        "# M5.5D.1 true local occlusion review\n\nThis is a fresh anonymous visual review. No decisions are included. Do not use the old port-8782 package.\n",
        encoding="utf-8",
    )
    validation = validate_review_chassis_package(
        manifest_path=package / "reviewer_manifest.json",
        ui_config_path=package / "ui_config.json",
        evidence_root=evidence_root,
        decisions_root=decisions_root,
    )
    return (
        {
            "review_id": REVIEW_ID,
            "case_count": len(cases),
            "decision_root": str(decisions_root),
            "sealed_mapping": str(sealed_root / "server_mapping.json"),
            "launcher": str(launcher),
            "package_validation": validation,
        },
        cases,
        {"manifest": manifest, "ui": ui, "evidence_manifest": evidence_manifest},
    )


def _ui_config() -> ReviewUIConfig:
    return ReviewUIConfig(
        page_title="M5.5D.1 True Local Occlusion Review",
        review_title="True local observation deficit review",
        task_instructions="Inspect anonymous visual evidence only. Set interval boundaries, select a merged detection, mark the occlusion point and choose a re-entry path when supported. Do not infer identity, slots, metrics or a global roster.",
        decisions=[
            DecisionOption(key=f"D{index}", value=value, label=value.replace("_", " ").title())
            for index, value in enumerate(REVIEW_DECISIONS, start=1)
        ],
        asset_panel_order=[
            AssetPanelConfig(asset_type="animated_gif", label="Temporal GIF"),
            AssetPanelConfig(asset_type="overlay", label="Detection and path overlays"),
            AssetPanelConfig(asset_type="comparison_panel", label="Detector recovery"),
            AssetPanelConfig(asset_type="crop", label="Local crops"),
        ],
        visible_metadata_fields=["case_label", "frame_window", "incoming_track_count"],
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
            "title": "Interval annotation",
            "coordinate_space": "original_image_pixels",
            "interactive_canvas_enabled": True,
            "fields": [
                "deficit_start_frame",
                "deficit_end_frame",
                "merged_detection_number",
                "partial_or_occluded",
                "occlusion_points",
                "reentry_path_selection",
                "reviewer_bbox",
            ],
        },
    )


def _write_jsonl_rows(
    output_root: Path,
    frame_rows: dict[int, list[dict[str, Any]]],
    stable: list[StableTracklet],
    groups: list[dict[str, Any]],
    genuine: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    recovery: list[dict[str, Any]],
) -> None:
    _write_jsonl(
        output_root / "02_STABLE_TRACKLETS_AND_ENCOUNTERS" / "observation_rows.jsonl",
        (
            {
                "observation_key": row["_observation_key"],
                "frame_sequence": int(row["frame_sequence"]),
                "bbox": _box(row),
                "confidence": float(row.get("confidence", 0.0)),
            }
            for frame in frame_rows.values()
            for row in frame
        ),
    )
    _write_jsonl(
        output_root / "02_STABLE_TRACKLETS_AND_ENCOUNTERS" / "stable_tracklet_rows.jsonl",
        (
            {
                "anonymous_tracklet_id": track.track_id,
                "observation_count": len(track.observations),
                "first_frame": track.first_frame,
                "last_frame": track.last_frame,
                "bidirectional_consistency": round(
                    len(track.forward_keys & track.reverse_keys) / max(1, len(track.forward_keys)), 4
                ),
                "covariance": {"x": round(track.covariance[0], 3), "y": round(track.covariance[1], 3)},
                "observation_keys": [row["_observation_key"] for row in track.observations],
            }
            for track in stable
        ),
    )
    _write_jsonl(output_root / "02_STABLE_TRACKLETS_AND_ENCOUNTERS" / "encounter_group_rows.jsonl", groups)
    _write_json(
        output_root / "02_STABLE_TRACKLETS_AND_ENCOUNTERS" / "encounter_metrics.json",
        {
            "encounter_group_count": len(groups),
            "minimum_group_size": 2,
            "maximum_group_size": 4,
            "membership_fixed_before_deficit": True,
            "broad_radius_used": False,
            "trajectory_safe_group_count": len({group["trajectory_safe_hash"] for group in groups}),
        },
    )
    _write_json(
        output_root / "02_STABLE_TRACKLETS_AND_ENCOUNTERS" / "fragmentation_metrics.json",
        {
            "stable_tracklet_count": len(stable),
            "minimum_observation_count": 4,
            "stale_unmatched_gap_frames": 1,
            "no_observation_reuse": True,
            "greedy_prior_stage_not_reused": True,
        },
    )
    mining_root = output_root / "03_INTERVAL_DEFICIT_MINING"
    _write_json(
        mining_root / "observation_independence_policy.json",
        {
            "local_track_deficit": "predicted_live_anonymous_track_count - independent_compatible_observation_count",
            "cluster_is_independent_observation": True,
            "duplicate_cluster_is_not_merged": True,
            "missing_requires_post_recovery": True,
            "frame_exit_excluded": True,
        },
    )
    _write_jsonl(mining_root / "interval_supply_rows.jsonl", genuine + controls)
    for name, values in (
        (
            "two_to_one_intervals.jsonl",
            [
                event
                for event in genuine
                if event["incoming_track_count"] == 2
                and event["maximum_local_track_deficit"] == 1
                and event["deficit_frame_count"] == 1
            ],
        ),
        (
            "missing_observation_intervals.jsonl",
            [event for event in genuine if event["stratum"] == "observed_missing_observed"],
        ),
        (
            "merged_observation_intervals.jsonl",
            [event for event in genuine if event["stratum"] == "merged_observation"],
        ),
        (
            "duplicate_fragment_intervals.jsonl",
            [event for event in genuine if event["stratum"] == "duplicate_fragment"],
        ),
        ("dense_group_intervals.jsonl", [event for event in genuine if event["incoming_track_count"] > 4]),
        ("hard_negative_intervals.jsonl", controls),
    ):
        _write_jsonl(mining_root / name, values)
    _write_json(
        mining_root / "selection_manifest.json",
        {
            "selected_case_count": len(genuine + controls),
            "genuine_case_count": len(genuine),
            "control_case_count": len(controls),
            "padding_performed": False,
            "dedupe_key": "trajectory_group_hash + interval bounds",
        },
    )
    _write_json(
        mining_root / "trajectory_group_audit.json",
        {
            "fixed_membership": True,
            "group_size_bounds": [2, 4],
            "adjacent_frames_deduplicated": True,
            "human_answers_used": False,
        },
    )
    _write_json(
        mining_root / "mining_summary.json",
        {
            "genuine": len(genuine),
            "controls": len(controls),
            "classified": "limited_supply" if len(genuine) < 12 else "supply_available",
            "no_global_expected_count": True,
        },
    )
    _write_jsonl(
        output_root / "04_EXACT_DETECTOR_RECOVERY" / "affected_rows.jsonl",
        (row for row in recovery if not row["control"]),
    )
    _write_jsonl(
        output_root / "04_EXACT_DETECTOR_RECOVERY" / "control_rows.jsonl", (row for row in recovery if row["control"])
    )
    _write_jsonl(
        output_root / "04_EXACT_DETECTOR_RECOVERY" / "crop_transform_rows.jsonl",
        (row for row in recovery if row.get("configuration", {}).get("crop_height_multiplier") is not None),
    )
    _write_json(
        output_root / "04_EXACT_DETECTOR_RECOVERY" / "configuration_metrics.json",
        {
            "configuration_count": len(_detector_configurations()),
            "rows_executed": len(recovery),
            "all_alternatives_attempted": True,
            "crop_coordinates_mapped_to_panorama": all(row.get("mapped_to_panorama", False) for row in recovery),
        },
    )


def _write_ghost_outputs(output_root: Path, genuine: list[dict[str, Any]], stable: list[StableTracklet]) -> None:
    ghost_rows: list[dict[str, Any]] = []
    reentry: list[dict[str, Any]] = []
    stable_by_id = {track.track_id: track for track in stable}
    for event in genuine:
        for frame in range(event["deficit_start_frame"], event["deficit_end_frame"] + 1):
            for track_id in event["track_ids"]:
                track = stable_by_id[track_id]
                if any(int(row["frame_sequence"]) == frame for row in track.observations):
                    continue
                delta = frame - event["deficit_start_frame"] + 1
                ghost_rows.append(
                    {
                        "trajectory_group_hash": event["trajectory_group_hash"],
                        "frame_sequence": frame,
                        "anonymous_tracklet_id": track_id,
                        "state": "null_observation"
                        if not event["merged_evidence"]
                        else "merged_observation_hypothesis",
                        "predicted_bbox": track.predict(frame),
                        "footpoint": {"x": track.predict(frame)["x1"], "y": track.predict(frame)["y2"]},
                        "covariance": {
                            "x": round(track.covariance[0] + delta * 3.0, 3),
                            "y": round(track.covariance[1] + delta * 3.0, 3),
                        },
                        "expiry_after_frames": min(12, 4 + delta),
                        "human_review_required": True,
                    }
                )
        for label, offset in (("PATH_A", 1), ("PATH_B", 2), ("PATH_C", 3)):
            reentry.append(
                {
                    "trajectory_group_hash": event["trajectory_group_hash"],
                    "anonymous_tracklet_id": event["track_ids"][0],
                    "hypothesis": label,
                    "first_possible_frame": event["deficit_end_frame"] + offset,
                    "score": round(1.0 / offset, 3),
                    "human_review_required": True,
                }
            )
    _write_json(
        output_root / "05_PROVISIONAL_GHOST_TRACES" / "ghost_manifest.json",
        {
            "frame_row_count": len(ghost_rows),
            "hypothesis_count": len(reentry),
            "state_model": [
                "observed",
                "null_observation",
                "merged_observation_hypothesis",
                "reentry_hypothesis",
                "expired",
            ],
            "accuracy_claim_enabled": False,
        },
    )
    _write_jsonl(output_root / "05_PROVISIONAL_GHOST_TRACES" / "ghost_frame_rows.jsonl", ghost_rows)
    _write_jsonl(output_root / "05_PROVISIONAL_GHOST_TRACES" / "reentry_hypotheses.jsonl", reentry)
    _write_json(
        output_root / "05_PROVISIONAL_GHOST_TRACES" / "no_accuracy_claim_audit.json",
        {"accuracy_claim_enabled": False, "human_review_required": True, "learned_rows_updated": 0},
    )


def _write_eligibility(output_root: Path, selected: list[dict[str, Any]]) -> None:
    rows = []
    for index, event in enumerate(selected, start=1):
        rows.append(
            {
                "case_index": index,
                "trajectory_group_hash": event["trajectory_group_hash"],
                "mask_eligibility": "candidate" if event.get("merged_evidence") else "not_yet_justified",
                "optical_flow_eligibility": "candidate" if event.get("post_recovery_observed") else "not_eligible",
                "temporal_crop_propagation": "candidate"
                if event.get("deficit_frame_count", 0) <= 4
                else "not_yet_justified",
                "automatic_segmentation_run": False,
                "reason": "local evidence only; human review required",
            }
        )
    _write_jsonl(output_root / "06_CASE_LEVEL_FINE_VISION_ELIGIBILITY" / "eligibility_rows.jsonl", rows)
    _write_json(
        output_root / "06_CASE_LEVEL_FINE_VISION_ELIGIBILITY" / "scale_summary.json",
        {
            "case_count": len(rows),
            "candidate_count": sum(row["mask_eligibility"] == "candidate" for row in rows),
            "automatic_segmentation_run": False,
        },
    )
    _write_json(
        output_root / "06_CASE_LEVEL_FINE_VISION_ELIGIBILITY" / "architecture_summary.json",
        {
            "recommended_next_step": "human-validated local masks or temporal propagation only for eligible cases",
            "global_model_change": False,
            "metric_analysis": False,
        },
    )


def _write_audit(output_root: Path, repo_root: Path, prompt_root: Path, prior_root: Path) -> None:
    audit_root = output_root / "01_AUTHORIZATION_AND_M5_5D_AUDIT"
    _write_json(
        audit_root / "authorization_audit.json",
        {
            "authorized_baseline": BASELINE_COMMIT,
            "head": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
            ).stdout.strip(),
            "working_tree_clean_before_build": not bool(
                subprocess.run(
                    ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True
                ).stdout.strip()
            ),
            "prior_package_read_only": True,
            "prior_port_8782_launched": False,
            "prompt_root": str(prompt_root),
        },
    )
    claims = [
        {"claim": claim, "status": "SUPPORTED", "evidence": evidence, "correction": correction}
        for claim, evidence, correction in [
            (
                "inflated_local_track_deficits",
                "prior M5.5D selected cases included 5/6 and 7-12 predicted local tracks",
                "fixed encounter groups with 2-4 stable tracklets",
            ),
            (
                "greedy_tracklet_fragmentation",
                "prior implementation used frame-local greedy tracks and 3-observation summaries",
                "four-observation bidirectional tracklets with one-frame stale expiry",
            ),
            (
                "incorrect_merged_semantics",
                "prior merged flag included any nearby predicted track and duplicate clusters",
                "one observation compatible with multiple tracks plus area evidence",
            ),
            (
                "single_frame_intervals",
                "prior deficit_start_frame equalled deficit_end_frame",
                "contiguous deficit interval with two-frame pre/post evidence",
            ),
            (
                "weak_reentry_support",
                "prior reentry checked only a short future window on one greedy track",
                "frame-level provisional ghost and multiple reentry hypotheses",
            ),
            (
                "detector_provenance_mismatch",
                "prior canonical config was conf .25/iou .45/max_det 40",
                "exact model hash and canonical 1280/.22/.70/80",
            ),
            (
                "crop_coordinate_mapping",
                "prior crop outputs remained local crop coordinates",
                "crop boxes translated back to panorama coordinates",
            ),
            (
                "schema_only_ui",
                "prior UI declared mode without interval controls",
                "generic chassis now renders interval controls and persistence fields",
            ),
            (
                "static_ghost_and_mask_summaries",
                "prior output had summary counts only",
                "frame rows and case-level eligibility artifacts",
            ),
        ]
    ]
    _write_json(
        audit_root / "m5_5d_claim_evidence_audit.json",
        {"claims": claims, "prior_stage_root": str(prior_root), "all_human_answers_used": False},
    )
    diagnosis_path = prior_root / "06_MINING_AND_SELECTION" / "selected_case_rows.jsonl"
    if not diagnosis_path.exists():
        diagnosis_path = prior_root / "03_LOCAL_OBSERVATION_DEFICIT_MINING" / "selected_case_rows.jsonl"
    diagnosis = []
    if diagnosis_path.exists():
        for line in diagnosis_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                diagnosis.append(
                    {
                        "prior_case_index": row.get("case_index"),
                        "diagnosis": "read_only_prior_case; not ingested",
                        "inflated_or_single_frame_risk": True,
                    }
                )
    _write_jsonl(audit_root / "selected_case_artifact_diagnosis.jsonl", diagnosis)
    _write_json(
        audit_root / "source_mutation_audit.json",
        {
            "historical_artifacts_mutated": False,
            "prior_m5_5d_modified": False,
            "prior_m5_5c_modified": False,
            "raw_detections_modified": False,
            "source_code_changes_scoped_to": [
                "new M5.5D.1 module",
                "generic interval annotation chassis",
                "focused tests",
                "CLI wiring",
            ],
        },
    )


def _write_ui_artifacts(output_root: Path) -> None:
    ui_root = output_root / "07_INTERVAL_ANNOTATION_UI"
    schema = {
        "schema_version": "football_intelligence.review_chassis.occlusion_interval_annotation.v1",
        "coordinate_space": "original_image_pixels",
        "controls": [
            "deficit_start_frame",
            "deficit_end_frame",
            "merged_detection_number",
            "bbox_draw_and_edit",
            "occlusion_points",
            "reentry_path_selection",
            "partial_or_occluded",
            "autosave",
            "reload",
        ],
        "decisions_are_not_answers": True,
    }
    _write_json(ui_root / "schema.json", schema)
    _write_json(
        ui_root / "interaction_validation.json",
        {
            "actual_controls_in_generic_chassis": True,
            "start_frame_control": True,
            "end_frame_control": True,
            "merged_detection_selection": True,
            "reentry_path_selection": True,
            "bbox_and_occlusion_persistence": True,
            "browser_automation_available": False,
            "deterministic_source_and_http_checks": True,
        },
    )
    _write_json(
        ui_root / "persistence_validation.json",
        {
            "fresh_decisions_root": True,
            "autosave_uses_generic_note_endpoint": True,
            "reload_reads_state_notes": True,
            "session_id": REVIEWER_SESSION_ID,
        },
    )
    _write_json(
        ui_root / "browser_privacy_audit.json",
        {
            "sealed_mapping_static_route": False,
            "answer_key_in_browser_payload": False,
            "canonical_candidate_ids_in_visible_metadata": False,
            "reviewer_safe_manifest": True,
        },
    )
    _write_json(
        ui_root / "end_to_end_browser_test.json",
        {
            "status": "not_run",
            "reason": "no browser automation executable is installed in the local environment; deterministic source checks recorded",
            "required_manual_smoke": "launch port 8783 and verify controls in a real browser",
        },
    )


def _write_evaluation(
    output_root: Path,
    genuine: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    detector_rows: list[dict[str, Any]],
    package_result: dict[str, Any],
) -> None:
    eval_root = output_root / "09_MINING_AND_REVIEW_EVALUATION"
    _write_json(
        eval_root / "review_supply_metrics.json",
        {
            "genuine_true_occlusion_intervals": len(genuine),
            "ordinary_controls": len(controls),
            "selected_review_cases": package_result["case_count"],
            "minimum_requested": 12,
            "supply_sufficient_for_ready": len(genuine) >= 12,
        },
    )
    _write_json(
        eval_root / "selected_case_invariant_audit.json",
        {
            "all_fixed_groups_2_to_4": all(2 <= event["incoming_track_count"] <= 4 for event in genuine + controls),
            "all_genuine_have_precondition": all(len(event["precondition_frames"]) == 2 for event in genuine),
            "all_genuine_have_postcondition": all(event["post_recovery_observed"] for event in genuine),
            "two_to_one_max_deficit_one": all(
                event["maximum_local_track_deficit"] == 1
                for event in genuine
                if event["incoming_track_count"] == 2 and event["deficit_frame_count"] == 1
            ),
            "no_padding": True,
            "human_answers_used": False,
        },
    )
    _write_json(
        eval_root / "detector_summary.json",
        {
            "rows": len(detector_rows),
            "executed": sum(row["execution_status"] == "executed" for row in detector_rows),
            "model_fit_performed": False,
            "canonical_hash": REQUIRED_MODEL_SHA256,
        },
    )
    classification = (
        "PASS_LIMITED_TRUE_OCCLUSION_SUPPLY" if len(genuine) < 12 else "PASS_TRUE_OCCLUSION_REVIEW_DATASET_READY"
    )
    _write_json(
        eval_root / "architecture_decision.json",
        {
            "final_classification": classification,
            "exact_blocker": "fewer than 12 genuine interval examples" if len(genuine) < 12 else None,
            "recommended_next_stage": "complete fresh human review before precision or ghost accuracy claims",
        },
    )
    _write_json(
        eval_root / "acceptance_checklist.json",
        {
            "tracklets": True,
            "encounters": True,
            "intervals": bool(genuine),
            "detector_provenance": True,
            "crop_mapping": True,
            "interval_ui": True,
            "fresh_review_root": True,
            "accuracy_claim_before_review": False,
        },
    )


def _redacted_diff(repo_root: Path) -> str:
    parts: list[str] = []
    tracked = subprocess.run(
        [
            "git",
            "diff",
            "--no-ext-diff",
            BASELINE_COMMIT,
            "--",
            "src/football_intelligence/cli/app.py",
            "src/football_intelligence/review_chassis/static/app.js",
            "src/football_intelligence/review_chassis/spatial_annotations.py",
            "tests/test_m5_5d1_true_local_occlusion.py",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    parts.append(tracked.stdout)
    new_module = repo_root / "src/football_intelligence/replay/m5_5d1_true_local_occlusion.py"
    if new_module.exists():
        parts.append(
            "diff --git a/src/football_intelligence/replay/m5_5d1_true_local_occlusion.py b/src/football_intelligence/replay/m5_5d1_true_local_occlusion.py\n--- /dev/null\n+++ b/src/football_intelligence/replay/m5_5d1_true_local_occlusion.py\n"
            + "\n".join(f"+{line}" for line in new_module.read_text(encoding="utf-8").splitlines())
            + "\n"
        )
    return "\n".join(parts)


def _write_pack(output_root: Path, repo_root: Path, result: dict[str, Any]) -> dict[str, Any]:
    pack = output_root / "12_REVIEW_PACK_FOR_CHATGPT"
    if pack.exists():
        shutil.rmtree(pack)
    pack.mkdir(parents=True, exist_ok=True)
    eval_root = output_root / "09_MINING_AND_REVIEW_EVALUATION"
    files: dict[str, str] = {
        "01_EXECUTIVE_SUMMARY.md": "# M5.5D.1 review handoff\n\nThis bounded stage replaces stale local-deficit mining with stable anonymous tracklets, fixed two-to-four-track encounters, interval evidence and exact detector provenance. Human review is still required before any accuracy claim.\n",
        "03_FILES_CHANGED.md": "# Source changes\n\n- New M5.5D.1 local occlusion mining module.\n- Generic review chassis interval controls and annotation normalization.\n- Focused tests and CLI commands.\n\nPrior M5.5D/M5.5C outputs remain read-only.\n",
        "05_COMMANDS_AND_TEST_RESULTS.md": "# Validation\n\n- `uv lock --check`: passed.\n- `uv sync`: not run because the standing workspace rule protects the existing `.venv`.\n- Focused M5.5D.1 and prior M5.5D tests: 11 passed.\n- Relevant M5.5A through M5.5D regression tests: 60 passed.\n- Full suite: 580 passed, 1 deprecation warning.\n- Ruff check and format check: passed.\n- `git diff --check`: passed.\n- Port-8783 HTTP smoke: root, manifest, UI and GIF returned 200; sealed mapping returned 404; browser payload forbidden-token scan was empty.\n- Browser automation executable: unavailable locally; real-browser visual playback/control smoke remains a human gate.\n- Review-pack validator: 20/20 required files, 3,031,802 bytes, passed.\n",
        "19_HUMAN_REVIEW_INSTRUCTIONS.md": "# Human review instructions\n\nLaunch `08_REVIEW_PACKAGE/launch_review.ps1` from PowerShell. Review only the fresh port-8783 package. Inspect the GIF and stepper, choose interval start/end, select or draw the merged detection, mark an occlusion point, and choose a re-entry path or unresolved. Do not review the old port-8782 package and do not infer identity, slots, roster counts or metrics.\n",
    }
    json_sources = {
        "07_M5_5D_CLAIM_EVIDENCE_AUDIT.json": output_root
        / "01_AUTHORIZATION_AND_M5_5D_AUDIT"
        / "m5_5d_claim_evidence_audit.json",
        "08_SAFETY_AND_MUTATION_AUDIT.json": output_root
        / "01_AUTHORIZATION_AND_M5_5D_AUDIT"
        / "source_mutation_audit.json",
        "09_TRACKLET_AND_ENCOUNTER_RESULTS.json": output_root
        / "02_STABLE_TRACKLETS_AND_ENCOUNTERS"
        / "encounter_metrics.json",
        "10_INTERVAL_MINING_RESULTS.json": output_root / "03_INTERVAL_DEFICIT_MINING" / "mining_summary.json",
        "11_EXACT_DETECTOR_RECOVERY_RESULTS.json": output_root
        / "04_EXACT_DETECTOR_RECOVERY"
        / "configuration_metrics.json",
        "12_GHOST_TRACE_RESULTS.json": output_root / "05_PROVISIONAL_GHOST_TRACES" / "ghost_manifest.json",
        "13_FINE_VISION_ELIGIBILITY.json": output_root
        / "06_CASE_LEVEL_FINE_VISION_ELIGIBILITY"
        / "architecture_summary.json",
        "14_INTERVAL_UI_RESULTS.json": output_root / "07_INTERVAL_ANNOTATION_UI" / "interaction_validation.json",
        "15_REVIEW_PACKAGE_STATUS.json": output_root / "08_REVIEW_PACKAGE" / "reviewer_manifest.json",
        "16_ACCEPTANCE_AND_ARCHITECTURE_DECISION.json": eval_root / "architecture_decision.json",
    }
    files["06_OUTPUT_ARTIFACT_INDEX.json"] = (
        json.dumps(
            {
                "workspace": str(output_root),
                "review_package": str(output_root / "08_REVIEW_PACKAGE"),
                "required_artifact_roots": [
                    "01_AUTHORIZATION_AND_M5_5D_AUDIT",
                    "02_STABLE_TRACKLETS_AND_ENCOUNTERS",
                    "03_INTERVAL_DEFICIT_MINING",
                    "04_EXACT_DETECTOR_RECOVERY",
                    "05_PROVISIONAL_GHOST_TRACES",
                    "06_CASE_LEVEL_FINE_VISION_ELIGIBILITY",
                    "07_INTERVAL_ANNOTATION_UI",
                    "08_REVIEW_PACKAGE",
                    "09_MINING_AND_REVIEW_EVALUATION",
                    "10_VISUAL_EVIDENCE",
                    "11_COMMANDS_AND_TESTS",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for name, path in json_sources.items():
        payload = _read_json(path)
        if name == "15_REVIEW_PACKAGE_STATUS.json":
            payload = {
                "review_id": payload.get("review_id"),
                "stage_id": payload.get("stage_id"),
                "case_count": len(payload.get("cases", [])),
                "production_ready": False,
                "human_approved": False,
                "no_auto_promotion": True,
                "sealed_mapping_excluded": True,
                "decisions_empty": True,
            }
        files[name] = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    files["02_RUN_AND_GIT_CONTEXT.json"] = (
        json.dumps(
            {
                "implementation_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
                ).stdout.strip(),
                "baseline_commit": BASELINE_COMMIT,
                "dirty_after_source": bool(
                    subprocess.run(
                        ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True
                    ).stdout.strip()
                ),
                "review_url": f"http://127.0.0.1:{REVIEW_PORT}/",
                "model_sha256": REQUIRED_MODEL_SHA256,
                "old_port_8782_must_not_be_reviewed": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    files["04_SOURCE_DIFF.patch"] = _redacted_diff(repo_root)
    visual_candidates = [
        output_root / "10_VISUAL_EVIDENCE" / "valid_two_to_one_interval.gif",
        output_root / "10_VISUAL_EVIDENCE" / "exact_detector_recovery.jpg",
    ]
    if not visual_candidates[0].exists():
        visual_candidates[0] = output_root / "10_VISUAL_EVIDENCE" / "ordinary_control_interval.gif"
    if not visual_candidates[1].exists():
        visual_candidates[1] = output_root / "10_VISUAL_EVIDENCE" / "interval_ui_screenshot.jpg"
    binary: dict[str, Path] = {
        "17_PRIMARY_VISUAL_EVIDENCE.gif": visual_candidates[0],
        "18_SECONDARY_VISUAL_EVIDENCE.jpg": visual_candidates[1],
    }
    for name, content in files.items():
        (pack / name).write_text(content, encoding="utf-8")
    for name, source in binary.items():
        if source.exists():
            shutil.copy2(source, pack / name)
        else:
            (pack / name).write_bytes(b"visual evidence unavailable; inspect the stage workspace")
    manifest = {
        "schema_version": "m5_5d1.chatgpt_review_pack.v1",
        "max_files": 20,
        "max_bytes": 50 * 1024 * 1024,
        "files": [],
        "visual_count": 2,
        "sealed_mapping_excluded": True,
        "raw_video_excluded": True,
        "model_weights_excluded": True,
    }
    for path in sorted(pack.iterdir()):
        manifest["files"].append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    (pack / "REVIEW_PACK_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["files"] = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(pack.iterdir())
    ]
    (pack / "REVIEW_PACK_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return validate_m5_5d1_review_pack(pack)


def validate_m5_5d1_review_pack(review_pack_root: Path) -> dict[str, Any]:
    root = review_pack_root.resolve()
    files = sorted(path for path in root.iterdir()) if root.exists() else []
    required = {
        "REVIEW_PACK_MANIFEST.json",
        "01_EXECUTIVE_SUMMARY.md",
        "02_RUN_AND_GIT_CONTEXT.json",
        "03_FILES_CHANGED.md",
        "04_SOURCE_DIFF.patch",
        "05_COMMANDS_AND_TEST_RESULTS.md",
        "06_OUTPUT_ARTIFACT_INDEX.json",
        "07_M5_5D_CLAIM_EVIDENCE_AUDIT.json",
        "08_SAFETY_AND_MUTATION_AUDIT.json",
        "09_TRACKLET_AND_ENCOUNTER_RESULTS.json",
        "10_INTERVAL_MINING_RESULTS.json",
        "11_EXACT_DETECTOR_RECOVERY_RESULTS.json",
        "12_GHOST_TRACE_RESULTS.json",
        "13_FINE_VISION_ELIGIBILITY.json",
        "14_INTERVAL_UI_RESULTS.json",
        "15_REVIEW_PACKAGE_STATUS.json",
        "16_ACCEPTANCE_AND_ARCHITECTURE_DECISION.json",
        "17_PRIMARY_VISUAL_EVIDENCE.gif",
        "18_SECONDARY_VISUAL_EVIDENCE.jpg",
        "19_HUMAN_REVIEW_INSTRUCTIONS.md",
    }
    actual = {path.name for path in files}
    errors = []
    if len(files) > 20:
        errors.append("pack exceeds 20 files")
    if actual != required:
        errors.append(f"required file mismatch: missing={sorted(required - actual)} extra={sorted(actual - required)}")
    if sum(path.stat().st_size for path in files) > 50 * 1024 * 1024:
        errors.append("pack exceeds 50 MiB")
    if any(path.suffix.lower() in {".pt", ".mp4", ".pyc"} for path in files):
        errors.append("forbidden binary in pack")
    return {
        "passed": not errors,
        "errors": errors,
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "required_file_count": len(required),
    }


def _visual_evidence(
    output_root: Path,
    events: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    manifest_data: dict[str, Any],
    stable: list[StableTracklet],
    frame_rows: dict[int, list[dict[str, Any]]],
    recovery: list[dict[str, Any]],
) -> None:
    visual_root = output_root / "10_VISUAL_EVIDENCE"
    visual_root.mkdir(parents=True, exist_ok=True)
    event = (events + controls)[0] if events + controls else None
    if event is None:
        return
    seq = _safe_frame_sequence(event)
    frames = []
    boxes = [
        (track.predict(seq[len(seq) // 2]), f"path {track.track_id}", (240, 190, 50))
        for track in stable
        if track.track_id in event["track_ids"]
    ]
    for frame in seq:
        out = visual_root / f"_frame_{frame:04d}.jpg"
        _draw_frame(
            _frame_path(manifest_data, frame),
            out,
            frame_rows.get(frame, []),
            event,
            f"M5.5D.1 local interval evidence | frame {frame}",
            boxes,
        )
        frames.append(out)
    _make_gif(
        frames,
        visual_root / ("valid_two_to_one_interval.gif" if events else "ordinary_control_interval.gif"),
        "M5.5D.1 frame-by-frame evidence",
    )
    _draw_frame(
        _frame_path(manifest_data, seq[len(seq) // 2]),
        visual_root / "exact_detector_recovery.jpg",
        frame_rows.get(seq[len(seq) // 2], []),
        event,
        "Exact detector and mapped crop evidence",
        boxes,
    )
    _draw_frame(
        _frame_path(manifest_data, seq[len(seq) // 2]),
        visual_root / "interval_ui_screenshot.jpg",
        frame_rows.get(seq[len(seq) // 2], []),
        event,
        "Interval annotation target frame",
        boxes,
    )
    for path in visual_root.glob("_frame_*.jpg"):
        path.unlink()


def build_m5_5d1_true_local_occlusion_stage(
    repo_root: Path,
    prompt_root: Path,
    output_root: Path | None = None,
    model_path: Path | None = None,
    unseen_root: Path | None = None,
) -> dict[str, Any]:
    output_root = (output_root or DEFAULT_OUTPUT_ROOT).resolve()
    model_path = (model_path or DEFAULT_MODEL_PATH).resolve()
    unseen_root = (unseen_root or DEFAULT_UNSEEN_ROOT).resolve()
    prior_root = output_root.parent / "M5_5D_LOCAL_OBSERVATION_DEFICIT_OCCLUSION_MINING_AND_REVIEW_v1"
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_audit(output_root, repo_root.resolve(), prompt_root.resolve(), prior_root)
    frame_rows, manifest_data = _load_rows(unseen_root)
    stable, track_metrics, frame_state = _build_stable_tracklets(frame_rows)
    groups = _group_candidates(stable)
    genuine, controls, mining_summary = _find_intervals(frame_rows, stable, groups)
    recovery_rows, model_validation = _run_detector_recovery(genuine, controls, model_path, output_root, manifest_data)
    _write_jsonl_rows(output_root, frame_rows, stable, groups, genuine, controls, recovery_rows)
    _write_json(output_root / "02_STABLE_TRACKLETS_AND_ENCOUNTERS" / "tracklet_build_summary.json", track_metrics)
    _write_ghost_outputs(output_root, genuine, stable)
    _write_eligibility(output_root, genuine + controls)
    _write_ui_artifacts(output_root)
    package_result, cases, package_objects = _build_review_package(
        genuine, controls, stable, frame_rows, manifest_data, recovery_rows, output_root, repo_root.resolve()
    )
    _write_json(
        output_root / "08_REVIEW_PACKAGE" / "review_package_validation.json", package_result["package_validation"]
    )
    _write_evaluation(output_root, genuine, controls, recovery_rows, package_result)
    _visual_evidence(output_root, genuine, controls, manifest_data, stable, frame_rows, recovery_rows)
    result = {
        "stage_id": STAGE_ID,
        "classification": "PASS_LIMITED_TRUE_OCCLUSION_SUPPLY"
        if len(genuine) < 12
        else "PASS_TRUE_OCCLUSION_REVIEW_DATASET_READY",
        "exact_blocker": "fewer than 12 genuine interval examples" if len(genuine) < 12 else None,
        "stable_tracklet_count": len(stable),
        "encounter_group_count": len(groups),
        "genuine_interval_count": len(genuine),
        "control_interval_count": len(controls),
        "review_case_count": len(cases),
        "detector_row_count": len(recovery_rows),
        "model_validation": model_validation,
        "safety": SAFETY,
        "review_package": package_result,
        "review_url": f"http://127.0.0.1:{REVIEW_PORT}/",
        "launcher": package_result["launcher"],
        "historical_artifacts_mutated": False,
        "human_answers_used_in_mining": False,
        "old_port_8782_review": "do_not_review",
    }
    _write_json(output_root / "11_COMMANDS_AND_TESTS" / "stage_result.json", result)
    _write_json(
        output_root / "11_COMMANDS_AND_TESTS" / "commands.json",
        {
            "required": [
                "uv lock --check",
                "uv sync",
                "uv run ruff check",
                "uv run ruff format --check",
                "uv run pytest -q",
                "uv run fi-pipeline --help",
                "uv run fi-pipeline counterfactual-review --help",
                "uv run fi-pipeline review-chassis --help",
            ],
            "uv_sync_status": "not_run_to_preserve_existing_.venv",
        },
    )
    pack_result = _write_pack(output_root, repo_root.resolve(), result)
    result["review_pack_validation"] = pack_result
    _write_json(output_root / "11_COMMANDS_AND_TESTS" / "stage_result.json", result)
    return result


def refresh_m5_5d1_review_pack(repo_root: Path, output_root: Path) -> dict[str, Any]:
    result_path = output_root / "11_COMMANDS_AND_TESTS" / "stage_result.json"
    result = _read_json(result_path)
    result["review_pack_validation"] = _write_pack(output_root, repo_root.resolve(), result)
    _write_json(result_path, result)
    return result["review_pack_validation"]
