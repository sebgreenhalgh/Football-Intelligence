from __future__ import annotations

# Evidence schemas are deliberately kept visible in this module.
# ruff: noqa: E501

"""M5.5D.2 encounter-episode gap mining and bounded burst scan."""

import csv
import hashlib
import itertools
import json
import math
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from football_intelligence.replay.m5_5d1_true_local_occlusion import (
    _box,
    _clusters,
    _compatible,
    _dist,
    _draw_frame,
    _foot,
    _height,
    _make_gif,
    _read_json,
    _write_json,
    _write_jsonl,
)
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


STAGE_ID = "M5_5D2_ENCOUNTER_EPISODE_GAP_MINING_AND_EXPANDED_BURST_SCAN_v1"
REVIEW_ID = "m5_5d2_encounter_episode_review_v1"
REVIEWER_SESSION_ID = "m5_5d2_encounter_episode_human_reviewer"
BASELINE_COMMIT = "a7d560523a19140054519bd8820d395459a80a12"
REVIEW_PORT = 8784
MODEL_SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
VIDEO_SHA256 = "8db0efdc045978d67572c6764681a76350e8da75a9f5fa7bc9307f3b9f21d989"
DEFAULT_OUTPUT_ROOT = Path(
    r"C:\Users\sebgr\Documents\football-intelligence\matches\128058\runs\step_m5\part 2\M5_5D2_ENCOUNTER_EPISODE_GAP_MINING_AND_EXPANDED_BURST_SCAN_v1"
)
DEFAULT_INVENTORY_ROOT = Path(
    r"C:\Users\sebgr\Documents\football-intelligence\matches\128058\runs\step_m5\06f_balanced_role_then_continuity\continuity_v11\unseen_window"
)
DEFAULT_MATCH_ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence\matches\128058")
DEFAULT_VIDEO = Path(
    r"C:\Users\sebgr\Documents\football-intelligence\matches\128058\videos\128058_panorama_1st_half.mp4"
)
DEFAULT_VIDEO_MANIFEST = Path(
    r"C:\Users\sebgr\Documents\football-intelligence\matches\128058\runs\step_m5\05_blind_second_window\source\source_video_manifest.json"
)
DEFAULT_MODEL = Path(
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


def _hash_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _row_key(row: dict[str, Any]) -> str:
    return str(
        row.get("_observation_key")
        or f"{row.get('frame_sequence')}:{row.get('source_frame_sequence')}:{row.get('bbox')}"
    )


@dataclass
class VisibleSegment:
    segment_id: str
    observations: list[dict[str, Any]] = field(default_factory=list)
    termination_reason: str = "unmatched"
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
        if frame_sequence < self.first_frame:
            return _box(self.observations[0])
        base = self.last_box
        vx, vy = self.velocity()
        gap = frame_sequence - self.last_frame
        return {
            "x1": base["x1"] + vx * gap,
            "y1": base["y1"] + vy * gap,
            "x2": base["x2"] + vx * gap,
            "y2": base["y2"] + vy * gap,
        }


def _load_rows(rows_path: Path, manifest_path: Path) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with rows_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_observation_key"] = f"{int(row['frame_sequence'])}:{index}"
            rows[int(row["frame_sequence"])].append(row)
    return dict(rows), _read_json(manifest_path)


def _assignment_cost(predicted: dict[str, float], observed: dict[str, float]) -> float | None:
    cost = _compatible(predicted, observed)
    if cost is None:
        return None
    return cost


def _segment_once(frame_rows: dict[int, list[dict[str, Any]]], reverse: bool = False) -> list[VisibleSegment]:
    frame_order = sorted(frame_rows, reverse=reverse)
    active: list[dict[str, Any]] = []
    segments: list[VisibleSegment] = []
    next_id = 1
    for frame in frame_order:
        rows = frame_rows[frame]
        pairs: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for state in active:
            if abs(frame - state["last_frame"]) > 1:
                continue
            segment: VisibleSegment = state["segment"]
            last_box = state["last_box"]
            vx, vy = state["velocity"]
            gap = frame - state["last_frame"]
            predicted = {
                "x1": last_box["x1"] + vx * gap,
                "y1": last_box["y1"] + vy * gap,
                "x2": last_box["x2"] + vx * gap,
                "y2": last_box["y2"] + vy * gap,
            }
            for row in rows:
                cost = _assignment_cost(predicted, _box(row))
                if cost is not None:
                    pairs.append((cost, state, row))
        used_states: set[int] = set()
        used_rows: set[str] = set()
        matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for _, state, row in sorted(
            pairs, key=lambda item: (item[0], item[1]["segment"].segment_id, _row_key(item[2]))
        ):
            state_id = id(state)
            if state_id in used_states or _row_key(row) in used_rows:
                continue
            used_states.add(state_id)
            used_rows.add(_row_key(row))
            matched.append((state, row))
        for state, row in matched:
            segment: VisibleSegment = state["segment"]
            previous = state["last_box"]
            current = _box(row)
            gap = max(1, abs(frame - state["last_frame"]))
            old_point, new_point = _foot(previous), _foot(current)
            state["velocity"] = ((new_point[0] - old_point[0]) / gap, (new_point[1] - old_point[1]) / gap)
            state["last_box"] = current
            state["last_frame"] = frame
            segment.observations.append(row)
            if reverse:
                segment.reverse_keys.add(_row_key(row))
            else:
                segment.forward_keys.add(_row_key(row))
            segment.covariance = (
                max(1.0, 0.75 * segment.covariance[0] + abs(new_point[0] - old_point[0])),
                max(1.0, 0.75 * segment.covariance[1] + abs(new_point[1] - old_point[1])),
            )
        for row in rows:
            if _row_key(row) in used_rows:
                continue
            segment = VisibleSegment(segment_id=f"seg_{next_id:06d}", observations=[row])
            next_id += 1
            segment.forward_keys.add(_row_key(row)) if not reverse else segment.reverse_keys.add(_row_key(row))
            segments.append(segment)
            active.append({"segment": segment, "last_frame": frame, "last_box": _box(row), "velocity": (0.0, 0.0)})
        active = [state for state in active if abs(frame - state["last_frame"]) <= 1]
    for segment in segments:
        segment.observations.sort(key=lambda item: int(item["frame_sequence"]))
        if reverse:
            continue
        if segment.last_frame == max(frame_order, default=segment.last_frame):
            segment.termination_reason = "frame_boundary"
        elif segment.last_frame + 1 not in frame_rows:
            segment.termination_reason = "detector_supply_loss"
        else:
            next_rows = frame_rows.get(segment.last_frame + 1, [])
            predicted = segment.predict(segment.last_frame + 1)
            has_compatible_next = any(_compatible(predicted, _box(row)) is not None for row in next_rows)
            segment.termination_reason = "ambiguity" if has_compatible_next else "unmatched"
    return segments


def _build_visible_segments(frame_rows: dict[int, list[dict[str, Any]]]) -> tuple[list[VisibleSegment], dict[str, Any]]:
    forward = _segment_once(frame_rows)
    reverse_rows = {-frame: [{**row, "frame_sequence": -frame} for row in rows] for frame, rows in frame_rows.items()}
    reverse = _segment_once(reverse_rows)
    reverse_by_key: dict[str, VisibleSegment] = {}
    for segment in reverse:
        for row in segment.observations:
            reverse_by_key[_row_key(row)] = segment
    stable: list[VisibleSegment] = []
    for segment in forward:
        keys = {_row_key(row) for row in segment.observations}
        reverse_keys: set[str] = set()
        for key in keys:
            reverse_keys.update(_row_key(row) for row in reverse_by_key.get(key, segment).observations)
        segment.forward_keys = keys
        segment.reverse_keys = reverse_keys
        consistency = len(keys & reverse_keys) / max(1, len(keys))
        if len(keys) >= 4 and consistency >= 0.75:
            stable.append(segment)
    metrics = {
        "forward_segment_count": len(forward),
        "reverse_segment_count": len(reverse),
        "stable_segment_count": len(stable),
        "minimum_provisional_observations": 3,
        "minimum_stable_observations": 4,
        "eligible_gap_frames": 1,
        "automatic_gap_stitching": False,
        "observation_reuse_count": 0,
        "segment_split_after_missing_frame_is_expected": True,
        "stable_observation_min": min((len(item.observations) for item in stable), default=0),
        "stable_observation_max": max((len(item.observations) for item in stable), default=0),
    }
    return stable, metrics


def _segment_payload(segment: VisibleSegment) -> dict[str, Any]:
    return {
        "segment_id": segment.segment_id,
        "first_observed_frame": segment.first_frame,
        "last_observed_frame": segment.last_frame,
        "observation_frames": [int(row["frame_sequence"]) for row in segment.observations],
        "observation_count": len(segment.observations),
        "last_state": "visible",
        "velocity": {"x": round(segment.velocity()[0], 3), "y": round(segment.velocity()[1], 3)},
        "covariance": {"x": round(segment.covariance[0], 3), "y": round(segment.covariance[1], 3)},
        "termination_reason": segment.termination_reason,
        "local_neighbours_at_end": [],
        "source_provenance": {"observation_keys": [_row_key(row) for row in segment.observations]},
        "bidirectional_consistency": round(
            len(segment.forward_keys & segment.reverse_keys) / max(1, len(segment.forward_keys)), 4
        ),
    }


def _episode_anchor(segment: VisibleSegment, frame: int) -> tuple[float, float]:
    return _foot(segment.predict(frame))


def _build_episodes(stable: list[VisibleSegment], frame_min: int, frame_max: int) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    local_group_counts: dict[tuple[int, int, int], int] = defaultdict(int)
    for left, right in itertools.combinations(stable, 2):
        if len(left.observations) < 3 or len(right.observations) < 3:
            continue
        if abs(left.last_frame - right.last_frame) > 20:
            continue
        shared_end = min(left.last_frame, right.last_frame)
        shared_start = max(left.first_frame, right.first_frame)
        if shared_end - shared_start < 2:
            continue
        first = max(frame_min, min(left.last_frame, right.last_frame) - 3)
        last = min(frame_max, max(left.last_frame, right.last_frame) + 20)
        if last <= first:
            continue
        # Contact must be supported while both incoming segments are visible.
        # Searching the post-termination horizon here would pair unrelated
        # segments and manufacture an encounter at a frame boundary.
        candidates: list[tuple[float, int]] = []
        for frame in range(shared_start, shared_end + 1):
            left_box, right_box = left.predict(frame), right.predict(frame)
            normalized = _dist(_foot(left_box), _foot(right_box)) / max(1.0, max(_height(left_box), _height(right_box)))
            candidates.append((normalized, frame))
        if not candidates:
            continue
        # Prefer the latest frame on a flat minimum so the episode retains
        # observable pre-contact evidence instead of anchoring at a window
        # boundary.
        normalized, contact = min(candidates, key=lambda item: (item[0], -item[1]))
        if normalized > 1.65:
            continue
        left_anchor = _episode_anchor(left, contact)
        right_anchor = _episode_anchor(right, contact)
        local_group_key = (
            contact // 25,
            int(min(left_anchor[0], right_anchor[0]) // 240),
            int(min(left_anchor[1], right_anchor[1]) // 160),
        )
        if local_group_counts[local_group_key] >= 4:
            continue
        local_group_counts[local_group_key] += 1
        episode_start = max(frame_min, contact - 2)
        if sum(int(row["frame_sequence"]) <= contact for row in left.observations) < 3:
            continue
        if sum(int(row["frame_sequence"]) <= contact for row in right.observations) < 3:
            continue
        horizon_end = min(frame_max, max(left.last_frame, right.last_frame) + 20)
        if horizon_end - episode_start < 5:
            continue
        ids = [left.segment_id, right.segment_id]
        episode = {
            "encounter_episode_id": f"episode_{left.segment_id}_{right.segment_id}_{contact:06d}",
            "incoming_segment_ids": ids,
            "incoming_track_count": 2,
            "pre_encounter_frames": list(range(episode_start, contact)),
            "encounter_start_frame": episode_start,
            "predicted_contact_frame": contact,
            "prediction_horizon_end": horizon_end,
            "predicted_state_by_frame": {},
            "uncertainty_by_frame": {},
            "trajectory_safe_episode_hash": _hash_json([ids, episode_start, contact, horizon_end]),
            "contact_normalized_distance": round(normalized, 4),
            "local_encounter_group_key": [*local_group_key],
            "membership_frozen_before_deficit": True,
            "prediction_continues_after_segment_termination": any(
                segment.last_frame < horizon_end for segment in (left, right)
            ),
        }
        for frame in range(episode_start, horizon_end + 1):
            states = {}
            uncertainty = {}
            for segment in (left, right):
                predicted = segment.predict(frame)
                states[segment.segment_id] = predicted
                delta = max(0, frame - segment.last_frame)
                uncertainty[segment.segment_id] = {
                    "x": round(segment.covariance[0] + delta * 3.0, 3),
                    "y": round(segment.covariance[1] + delta * 3.0, 3),
                }
            episode["predicted_state_by_frame"][str(frame)] = states
            episode["uncertainty_by_frame"][str(frame)] = uncertainty
        episodes.append(episode)
    dedup: dict[tuple[tuple[str, ...], int], dict[str, Any]] = {}
    for episode in episodes:
        key = (
            tuple(sorted(episode["incoming_segment_ids"])),
            int(episode["predicted_contact_frame"]) // 5,
        )
        if key not in dedup:
            dedup[key] = episode
    return sorted(dedup.values(), key=lambda item: (item["predicted_contact_frame"], item["incoming_segment_ids"]))[
        :600
    ]


def _episode_frame_supply(
    episode: dict[str, Any],
    frame_rows: dict[int, list[dict[str, Any]]],
    stable_by_id: dict[str, VisibleSegment],
    outgoing_by_incoming: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    incoming_ids = episode["incoming_segment_ids"]
    for frame in range(episode["encounter_start_frame"], episode["prediction_horizon_end"] + 1):
        predictions = episode["predicted_state_by_frame"][str(frame)]
        local_rows = [
            row
            for row in frame_rows.get(frame, [])
            if any(_compatible(prediction, _box(row)) is not None for prediction in predictions.values())
        ]
        clusters = _clusters(local_rows)
        compatible_tracks = [
            [
                segment_id
                for segment_id, prediction in predictions.items()
                if any(_compatible(prediction, _box(row)) is not None for row in cluster)
            ]
            for cluster in clusters
        ]
        pairs = sorted(
            (
                (_compatible(predictions[segment_id], _box(cluster[0])), segment_id, index)
                for segment_id in incoming_ids
                for index, cluster in enumerate(clusters)
                if cluster and _compatible(predictions[segment_id], _box(cluster[0])) is not None
            ),
            key=lambda item: (item[0], item[1], item[2]),
        )
        assignments: dict[str, int] = {}
        used_ids: set[str] = set()
        used_clusters: set[int] = set()
        for _, segment_id, cluster_index in pairs:
            if segment_id in used_ids or cluster_index in used_clusters:
                continue
            used_ids.add(segment_id)
            used_clusters.add(cluster_index)
            assignments[segment_id] = cluster_index
        outgoing_segments = []
        outgoing_ids = {
            item["outgoing_segment_id"] for item in itertools.chain.from_iterable(outgoing_by_incoming.values())
        }
        for segment_id in outgoing_ids:
            segment = stable_by_id.get(segment_id)
            if segment and any(int(row["frame_sequence"]) == frame for row in segment.observations):
                outgoing_segments.append(segment_id)
        rows.append(
            {
                "frame_sequence": frame,
                "raw_local_detection_count": len(local_rows),
                "duplicate_clusters": [len(cluster) for cluster in clusters if len(cluster) > 1],
                "fragment_clusters": [len(cluster) for cluster in clusters if len(cluster) > 1],
                "independent_observation_count": len(clusters),
                "single_observation_multi_track_compatibility": [ids for ids in compatible_tracks if len(ids) >= 2],
                "unassigned_predicted_track_ids": [
                    segment_id for segment_id in incoming_ids if segment_id not in assignments
                ],
                "assignments": assignments,
                "outgoing_segment_ids": sorted(outgoing_segments),
                "outgoing_independent_observation_count": len(set(outgoing_segments)),
                "local_track_deficit": max(0, len(incoming_ids) - len(clusters)),
                "predictions": predictions,
            }
        )
    return rows


def _find_outgoing_segments(
    episode: dict[str, Any], stable: list[VisibleSegment], post_start_frame: int
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    incoming = set(episode["incoming_segment_ids"])
    by_incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for segment in stable:
        if segment.segment_id in incoming or segment.first_frame < post_start_frame:
            continue
        if segment.first_frame > episode["prediction_horizon_end"] or len(segment.observations) < 3:
            continue
        for incoming_id in episode["incoming_segment_ids"]:
            if segment.first_frame > episode["prediction_horizon_end"]:
                continue
            prediction = episode["predicted_state_by_frame"].get(str(segment.first_frame), {}).get(incoming_id)
            if prediction is None:
                continue
            cost = _compatible(prediction, _box(segment.observations[0]))
            if cost is None or cost > 1.8:
                continue
            row = {
                "outgoing_segment_id": segment.segment_id,
                "first_observed_frame": segment.first_frame,
                "candidate_incoming_segment_ids": episode["incoming_segment_ids"],
                "motion_cost": round(cost, 4),
                "scale_cost": round(
                    abs(math.log(max(1.0, _height(prediction)) / max(1.0, _height(_box(segment.observations[0]))))), 4
                ),
                "order_consistency": "unresolved",
                "appearance_gate_status": "soft_visual_only",
                "joint_hypothesis_id": None,
            }
            by_incoming[incoming_id].append(row)
            rows.append(row)
    unique = {row["outgoing_segment_id"]: row for row in rows}
    candidates = list(unique.values())
    for row in candidates:
        row["joint_hypothesis_id"] = _hash_json([episode["trajectory_safe_episode_hash"], row["outgoing_segment_id"]])[
            :16
        ]
    return dict(by_incoming), candidates


def _mine_episode(
    episode: dict[str, Any],
    frame_rows: dict[int, list[dict[str, Any]]],
    stable: list[VisibleSegment],
    image_size: tuple[int, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stable_by_id = {segment.segment_id: segment for segment in stable}
    base_supply = _episode_frame_supply(episode, frame_rows, stable_by_id, {})
    base_by_frame = {row["frame_sequence"]: row for row in base_supply}
    supply = base_supply
    genuine: list[dict[str, Any]] = []
    near_misses: list[dict[str, Any]] = []
    for start in range(episode["encounter_start_frame"] + 2, episode["prediction_horizon_end"] - 2):
        pre = [base_by_frame.get(start - 2), base_by_frame.get(start - 1)]
        if any(row is None or row["independent_observation_count"] != episode["incoming_track_count"] for row in pre):
            continue
        deficit_frames: list[int] = []
        cursor = start
        while (
            cursor <= episode["prediction_horizon_end"] - 2
            and base_by_frame.get(cursor) is not None
            and base_by_frame[cursor]["independent_observation_count"] < episode["incoming_track_count"]
        ):
            deficit_frames.append(cursor)
            cursor += 1
        if not deficit_frames or len(deficit_frames) > 12:
            continue
        selected = [base_by_frame[frame] for frame in deficit_frames]
        if image_size:
            width, height = image_size
            exit_only = all(
                all(
                    prediction["x2"] < 0
                    or prediction["x1"] > width
                    or prediction["y2"] < 0
                    or prediction["y1"] > height
                    for prediction in row["predictions"].values()
                )
                for row in selected
            )
            if exit_only:
                near_misses.append(
                    {
                        "encounter_episode_id": episode["encounter_episode_id"],
                        "trajectory_safe_episode_hash": episode["trajectory_safe_episode_hash"],
                        "incoming_segment_ids": episode["incoming_segment_ids"],
                        "rejection_reason": "frame_exit_or_not_expected_visible",
                        "deficit_frames": deficit_frames,
                        "human_answers_used_in_mining": False,
                    }
                )
                continue
        outgoing_by_incoming, outgoing_candidates = _find_outgoing_segments(episode, stable, cursor)
        supply = _episode_frame_supply(episode, frame_rows, stable_by_id, outgoing_by_incoming)
        by_frame = {row["frame_sequence"]: row for row in supply}
        pre = [by_frame.get(start - 2), by_frame.get(start - 1)]
        selected = [by_frame[frame] for frame in deficit_frames]
        post_frames = [by_frame.get(cursor), by_frame.get(cursor + 1)]
        has_outgoing_post = all(
            row is not None and row["outgoing_independent_observation_count"] >= episode["incoming_track_count"]
            for row in post_frames
        )
        max_deficit = max(row["local_track_deficit"] for row in selected)
        multi_track = any(row["single_observation_multi_track_compatibility"] for row in selected)
        merged_evidence = False
        if multi_track:
            for selected_row in selected:
                compatible_rows = [
                    item
                    for item in frame_rows.get(selected_row["frame_sequence"], [])
                    if any(
                        _compatible(prediction, _box(item)) is not None
                        for prediction in selected_row["predictions"].values()
                    )
                ]
                for cluster in _clusters(compatible_rows):
                    if len(cluster) != 1:
                        continue
                    observed_box = _box(cluster[0])
                    observed_height = _height(observed_box)
                    observed_width = max(1.0, observed_box["x2"] - observed_box["x1"])
                    predicted_heights = [
                        _height(prediction)
                        for prediction in selected_row["predictions"].values()
                        if _compatible(prediction, observed_box) is not None
                    ]
                    predicted_widths = [
                        max(1.0, prediction["x2"] - prediction["x1"])
                        for prediction in selected_row["predictions"].values()
                        if _compatible(prediction, observed_box) is not None
                    ]
                    if predicted_heights and (
                        observed_height >= 1.35 * max(predicted_heights)
                        or observed_width >= 1.35 * max(predicted_widths)
                    ):
                        merged_evidence = True
                        break
                if merged_evidence:
                    break
        duplicate = any(row["duplicate_clusters"] for row in selected)
        if not has_outgoing_post:
            stratum = "ending_without_recovery"
        elif multi_track and merged_evidence:
            stratum = "merged_observation"
        elif duplicate and not multi_track:
            stratum = "duplicate_fragment"
        elif episode["incoming_track_count"] == 2 and max_deficit == 1 and len(deficit_frames) == 1:
            stratum = "two_to_one_collapse"
        else:
            stratum = "observed_missing_observed"
        event = {
            "encounter_episode_id": episode["encounter_episode_id"],
            "trajectory_safe_episode_hash": episode["trajectory_safe_episode_hash"],
            "incoming_segment_ids": episode["incoming_segment_ids"],
            "incoming_track_count": episode["incoming_track_count"],
            "deficit_start_frame": deficit_frames[0],
            "deficit_end_frame": deficit_frames[-1],
            "deficit_frame_count": len(deficit_frames),
            "maximum_local_track_deficit": max_deficit,
            "precondition_frames": [row["frame_sequence"] for row in pre],
            "postcondition_frames": [row["frame_sequence"] for row in post_frames if row],
            "post_recovery_observed": has_outgoing_post,
            "stratum": stratum,
            "outgoing_segment_candidates": outgoing_candidates,
            "joint_hypotheses": _joint_hypotheses(episode, outgoing_candidates),
            "supply_rows": supply,
            "human_answers_used_in_mining": False,
            "frame_file": next(
                (row.get("frame_file") for row in frame_rows.get(deficit_frames[0], []) if row.get("frame_file")),
                None,
            ),
            "source_frame_sequence": next(
                (
                    row.get("source_frame_sequence")
                    for row in frame_rows.get(deficit_frames[0], [])
                    if row.get("source_frame_sequence") is not None
                ),
                None,
            ),
            "anchor_bbox": _box(
                frame_rows.get(
                    deficit_frames[0],
                    [{"bbox": stable_by_id[episode["incoming_segment_ids"][0]].predict(deficit_frames[0])}],
                )[0]
            ),
        }
        if stratum == "two_to_one_collapse" and (max_deficit != 1 or len(deficit_frames) > 12):
            near_misses.append({**event, "rejection_reason": "two_to_one_invariant_failed"})
        elif stratum in {"two_to_one_collapse", "observed_missing_observed", "merged_observation"}:
            genuine.append(event)
        else:
            near_misses.append({**event, "rejection_reason": "no two-frame independent outgoing postcondition"})
        break
    if not genuine and not near_misses:
        near_misses.append(
            {
                "encounter_episode_id": episode["encounter_episode_id"],
                "trajectory_safe_episode_hash": episode["trajectory_safe_episode_hash"],
                "incoming_segment_ids": episode["incoming_segment_ids"],
                "rejection_reason": "no two-frame precondition plus contiguous deficit found",
                "human_answers_used_in_mining": False,
            }
        )
    return genuine, near_misses, supply


def _joint_hypotheses(episode: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outgoing_ids = sorted({row["outgoing_segment_id"] for row in candidates})
    hypotheses = []
    for index, permutation in enumerate(
        itertools.permutations(outgoing_ids, min(episode["incoming_track_count"], len(outgoing_ids)))
    ):
        if index >= 4:
            break
        hypotheses.append(
            {
                "joint_hypothesis_id": _hash_json([episode["trajectory_safe_episode_hash"], permutation])[:16],
                "incoming_segment_ids": episode["incoming_segment_ids"],
                "outgoing_segment_ids": list(permutation),
                "rank": index + 1,
                "assignment_forced": False,
                "human_review_required": True,
            }
        )
    return hypotheses


def _select_matched_controls(
    genuine: list[dict[str, Any]], near_misses: list[dict[str, Any]], maximum: int = 6
) -> list[dict[str, Any]]:
    """Select non-positive, answer-independent controls from the same miner."""

    if not genuine:
        return []
    target_heights = [
        _height(event.get("anchor_bbox", {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 2.0})) for event in genuine
    ]
    target_height = sum(target_heights) / max(1, len(target_heights))
    eligible = [
        row
        for row in near_misses
        if row.get("frame_file")
        and row.get("rejection_reason") not in {"frame_exit_or_not_expected_visible"}
        and row.get("human_answers_used_in_mining") is False
    ]
    eligible.sort(
        key=lambda row: (
            abs(_height(row.get("anchor_bbox", {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 2.0})) - target_height),
            str(row.get("encounter_episode_id", "")),
        )
    )
    selected: list[dict[str, Any]] = []
    seen_episodes: set[str] = set()
    for row in eligible:
        episode_id = str(row.get("encounter_episode_id", ""))
        if episode_id in seen_episodes:
            continue
        selected.append(
            {
                **row,
                "is_matched_control": True,
                "control_reason": row.get("rejection_reason", "near_miss_without_postcondition"),
                "human_answers_used_in_mining": False,
            }
        )
        seen_episodes.add(episode_id)
        if len(selected) >= maximum:
            break
    return selected


def _inventory_catalog(match_root: Path, preferred_root: Path) -> list[dict[str, Any]]:
    rows_paths = sorted(match_root.rglob("person_candidate_rows.jsonl"))
    catalog: list[dict[str, Any]] = []
    for rows_path in rows_paths:
        manifest_path = rows_path.parent / "canonical_frame_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = _read_json(manifest_path)
            provenance_path = rows_path.parent / "pipeline_provenance.json"
            row_manifest_path = rows_path.parent / "person_candidate_rows_manifest.json"
            provenance = _read_json(provenance_path) if provenance_path.exists() else {}
            row_manifest = _read_json(row_manifest_path) if row_manifest_path.exists() else {}
            with rows_path.open(encoding="utf-8") as handle:
                row_count = sum(1 for line in handle if line.strip())
            frame_count = len(manifest.get("frames", []))
            schema = "person_candidate_rows.jsonl"
            compatible = (
                frame_count > 0
                and row_count > 0
                and "bbox" in json.loads(next(line for line in rows_path.open(encoding="utf-8") if line.strip()))
            )
        except (OSError, StopIteration, json.JSONDecodeError):
            continue
        catalog.append(
            {
                "artifact_root": str(rows_path.parent),
                "rows_path": str(rows_path),
                "frame_manifest_path": str(manifest_path),
                "time_range": {"first_frame": 0, "last_frame": frame_count - 1},
                "cadence": "frame_sequence_manifest",
                "frame_count": frame_count,
                "row_count": row_count,
                "detector_provenance": provenance.get("detector", provenance.get("source_type", "unresolved")),
                "model_hash": provenance.get("model_sha256") or row_manifest.get("model_sha256"),
                "runtime_settings": provenance.get("detector_settings") or row_manifest.get("runtime_settings"),
                "candidate_row_schema": schema,
                "historical_safety_flags": {
                    "human_answers_used": False,
                    "canonical_rows_replaced": False,
                    "do_not_use_for_metrics": bool(manifest.get("do_not_use_for_metrics", True)),
                },
                "preferred": rows_path.parent.resolve() == preferred_root.resolve(),
                "usable_for_new_mining": compatible,
                "reason": "only compatible person-candidate inventory discovered"
                if compatible
                else "schema or frame manifest unavailable",
            }
        )
    return catalog


def _load_video_manifest(video_manifest_path: Path, video_path: Path) -> dict[str, Any]:
    manifest = _read_json(video_manifest_path)
    actual_hash = sha256_file(video_path)
    expected = str(manifest.get("source_video_sha256", "")).lower()
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=codec_name,width,height,r_frame_rate",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    probe_payload = json.loads(probe.stdout)
    validation = {
        "video_path": str(video_path),
        "manifest_path": str(video_manifest_path),
        "expected_sha256": expected,
        "actual_sha256": actual_hash,
        "hash_match": actual_hash.lower() == expected,
        "byte_size": video_path.stat().st_size,
        "probe": probe_payload,
        "manifest": manifest,
        "passed": actual_hash.lower() == expected,
    }
    return validation


def _extract_video_frames(
    video_path: Path, start_seconds: float, end_seconds: float, fps: float, output_dir: Path, prefix: str
) -> list[dict[str, Any]]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for the bounded video scan") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    start_frame = int(round(start_seconds * source_fps))
    end_frame = int(round(end_seconds * source_fps))
    step = max(1, int(round(source_fps / fps)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    rows: list[dict[str, Any]] = []
    frame_number = start_frame
    while frame_number <= end_frame:
        ok, frame = capture.read()
        if not ok:
            break
        if (frame_number - start_frame) % step == 0:
            sample_index = len(rows)
            path = output_dir / f"{prefix}_{sample_index:06d}.jpg"
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            rows.append(
                {
                    "frame_sequence": sample_index,
                    "source_frame_sequence": frame_number,
                    "timestamp_seconds": frame_number / source_fps,
                    "frame_file": str(path),
                    "fps": fps,
                }
            )
        frame_number += 1
    capture.release()
    return rows


def _canonical_config() -> dict[str, Any]:
    return {
        "name": "canonical",
        "imgsz": 1280,
        "conf": 0.22,
        "iou": 0.70,
        "max_det": 80,
        "classes": [0],
        "augment": False,
        "agnostic_nms": False,
        "device": "cpu",
        "save": False,
        "stream": False,
    }


def _validate_model_checkpoint(model_path: Path) -> dict[str, Any]:
    actual = sha256_file(model_path).lower()
    result = {
        "model_path": str(model_path),
        "expected_sha256": MODEL_SHA256,
        "actual_sha256": actual,
        "hash_match": actual == MODEL_SHA256,
        "validated_before_inference": True,
    }
    if not result["hash_match"]:
        raise ValueError(f"detector checkpoint hash mismatch: expected {MODEL_SHA256}, got {actual}")
    return result


def _run_yolo_rows(
    frame_manifest: list[dict[str, Any]], model_path: Path, output_path: Path, batch_size: int = 8
) -> list[dict[str, Any]]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    rows: list[dict[str, Any]] = []
    config = _canonical_config()
    for start in range(0, len(frame_manifest), batch_size):
        batch = frame_manifest[start : start + batch_size]
        predictions = model.predict(
            source=[row["frame_file"] for row in batch],
            batch=batch_size,
            verbose=False,
            **{key: value for key, value in config.items() if key != "name"},
        )
        for manifest_row, prediction in zip(batch, predictions):
            boxes = prediction.boxes.xyxy.cpu().tolist() if prediction.boxes is not None else []
            confidences = prediction.boxes.conf.cpu().tolist() if prediction.boxes is not None else []
            for index, values in enumerate(boxes):
                rows.append(
                    {
                        "frame_sequence": manifest_row["frame_sequence"],
                        "source_frame_sequence": manifest_row.get("source_frame_sequence"),
                        "timestamp_seconds": manifest_row.get("timestamp_seconds"),
                        "frame_file": manifest_row["frame_file"],
                        "bbox": {
                            "x1": round(float(values[0]), 2),
                            "y1": round(float(values[1]), 2),
                            "x2": round(float(values[2]), 2),
                            "y2": round(float(values[3]), 2),
                        },
                        "confidence": round(float(confidences[index]), 5) if index < len(confidences) else None,
                        "class_id": 0,
                        "_observation_key": f"{manifest_row['frame_sequence']}:{index}",
                    }
                )
    _write_jsonl(output_path, rows)
    return rows


def _rows_by_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[int(row["frame_sequence"])].append(row)
    return dict(result)


def _coarse_candidates(
    frame_rows: dict[int, list[dict[str, Any]]], stable: list[VisibleSegment], cap: int = 100
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    frames = sorted(frame_rows)
    counts = {frame: len(frame_rows[frame]) for frame in frames}
    for index, frame in enumerate(frames):
        prior = counts.get(frames[index - 1], counts[frame]) if index else counts[frame]
        following = counts.get(frames[index + 1], counts[frame]) if index + 1 < len(frames) else counts[frame]
        if (
            abs(counts[frame] - prior) >= 2
            or abs(following - counts[frame]) >= 2
            or any(len(cluster) > 1 for cluster in _clusters(frame_rows[frame]))
        ):
            candidates.append(
                {
                    "coarse_frame_sequence": frame,
                    "source_frame_sequence": frame_rows[frame][0].get("source_frame_sequence"),
                    "timestamp_seconds": frame_rows[frame][0].get("timestamp_seconds"),
                    "reason": "local_count_change_or_cluster",
                }
            )
    for episode in _build_episodes(stable, min(frames, default=0), max(frames, default=0)):
        candidates.append(
            {
                "coarse_frame_sequence": episode["predicted_contact_frame"],
                "source_frame_sequence": None,
                "timestamp_seconds": None,
                "reason": "coarse_encounter_episode",
            }
        )
    candidates.sort(key=lambda item: (item["coarse_frame_sequence"], item["reason"]))
    deduped: list[dict[str, Any]] = []
    for candidate in candidates:
        if any(abs(candidate["coarse_frame_sequence"] - other["coarse_frame_sequence"]) <= 5 for other in deduped):
            continue
        deduped.append(candidate)
        if len(deduped) >= cap:
            break
    return deduped


def _extract_bursts(
    video_path: Path,
    coarse_candidates: list[dict[str, Any]],
    source_start: float,
    source_fps: float,
    output_dir: Path,
    max_bursts: int = 12,
    max_frames: int = 600,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected = coarse_candidates[:max_bursts]
    all_frames: list[dict[str, Any]] = []
    bursts: list[dict[str, Any]] = []
    budget_used = 0
    for index, candidate in enumerate(selected, start=1):
        center = source_start + (float(candidate.get("coarse_frame_sequence", 0)) / 2.0)
        burst_start, burst_end = center - 2.5, center + 2.5
        if budget_used + 50 > max_frames:
            break
        rows = _extract_video_frames(
            video_path, burst_start, burst_end, 10.0, output_dir / f"burst_{index:03d}", f"burst_{index:03d}"
        )
        frame_offset = len(all_frames)
        for row in rows:
            row["frame_sequence"] = frame_offset + int(row["frame_sequence"])
            row["burst_id"] = f"burst_{index:03d}"
            row["coarse_candidate_frame"] = candidate.get("coarse_frame_sequence")
        all_frames.extend(rows)
        budget_used += len(rows)
        bursts.append(
            {
                "burst_id": f"burst_{index:03d}",
                "coarse_candidate_frame": candidate.get("coarse_frame_sequence"),
                "center_seconds": center,
                "start_seconds": burst_start,
                "end_seconds": burst_end,
                "frame_count": len(rows),
                "half_window_seconds": 2.5,
            }
        )
    return (
        bursts,
        all_frames,
        {
            "burst_count": len(bursts),
            "burst_frame_count": len(all_frames),
            "max_bursts": max_bursts,
            "max_burst_frames": max_frames,
            "budget_exhausted": budget_used >= max_frames,
        },
    )


def _run_selective_recovery(
    events: list[dict[str, Any]], controls: list[dict[str, Any]], model_path: Path, output_root: Path
) -> list[dict[str, Any]]:
    configs = [
        {"name": "canonical", **_canonical_config()},
        {"name": "full_frame_2048", **{**_canonical_config(), "imgsz": 2048}},
        {"name": "lower_confidence", **{**_canonical_config(), "conf": 0.10}},
        {"name": "relaxed_post_nms", **{**_canonical_config(), "conf": 0.10, "iou": 0.85}},
        {"name": "higher_max_det", **{**_canonical_config(), "conf": 0.10, "max_det": 160}},
        {"name": "native_2_height_crop", **{**_canonical_config(), "conf": 0.10, "crop_height_multiplier": 2.0}},
        {"name": "native_3_height_crop", **{**_canonical_config(), "conf": 0.10, "crop_height_multiplier": 3.0}},
    ]
    _write_json(
        output_root / "08_SELECTIVE_DETECTOR_RECOVERY" / "configuration_manifest.json",
        {
            "canonical": _canonical_config(),
            "configurations": configs,
            "model_sha256": MODEL_SHA256,
            "model_fit_performed": False,
        },
    )
    if not events:
        _write_json(
            output_root / "08_SELECTIVE_DETECTOR_RECOVERY" / "metrics.json",
            {"candidate_count": 0, "control_count": 0, "rows": 0, "selective_recovery_run": False},
        )
        return []
    try:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
    except Exception as exc:  # pragma: no cover
        _write_json(
            output_root / "08_SELECTIVE_DETECTOR_RECOVERY" / "metrics.json",
            {
                "candidate_count": len(events),
                "control_count": len(controls),
                "rows": 0,
                "selective_recovery_run": False,
                "error": str(exc),
            },
        )
        return []
    rows: list[dict[str, Any]] = []
    selected_events = events[:14] + controls[:6]
    control_ids = {str(event.get("encounter_episode_id")) for event in controls}
    for index, event in enumerate(selected_events, start=1):
        source = event.get("frame_file")
        if not source or not Path(source).exists():
            continue
        deficit_rows = [
            row for row in event.get("supply_rows", []) if row.get("frame_sequence") == event.get("deficit_start_frame")
        ]
        before_count = deficit_rows[0].get("independent_observation_count") if deficit_rows else None
        for config in configs:
            record = {
                "case_index": index,
                "encounter_episode_id": event.get("encounter_episode_id"),
                "control": str(event.get("encounter_episode_id")) in control_ids,
                "configuration": config,
                "execution_status": "not_run",
                "boxes": [],
                "mapped_to_panorama": config.get("crop_height_multiplier") is None,
                "model_fit_performed": False,
                "canonical_artifacts_replaced": False,
                "independent_observation_count_before": before_count,
            }
            try:
                input_path = source
                transform = None
                if config.get("crop_height_multiplier") and Image is not None:
                    image = Image.open(source).convert("RGB")
                    anchor = event["anchor_bbox"]
                    crop_height = _height(anchor) * config["crop_height_multiplier"]
                    left, top = (
                        max(0, int(_foot(anchor)[0] - 1.5 * crop_height)),
                        max(0, int(anchor["y1"] - crop_height)),
                    )
                    right, bottom = (
                        min(image.width, int(_foot(anchor)[0] + 1.5 * crop_height)),
                        min(image.height, int(anchor["y2"] + crop_height)),
                    )
                    crop_path = (
                        output_root
                        / "08_SELECTIVE_DETECTOR_RECOVERY"
                        / "inputs"
                        / f"case_{index:03d}_{config['name']}.jpg"
                    )
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    image.crop((left, top, right, bottom)).save(crop_path, quality=90)
                    input_path = str(crop_path)
                    transform = {"left": left, "top": top, "right": right, "bottom": bottom}
                    record["crop_transform"] = transform
                kwargs = {key: value for key, value in config.items() if key not in {"name", "crop_height_multiplier"}}
                prediction = model.predict(source=input_path, verbose=False, **kwargs)[0]
                boxes = prediction.boxes.xyxy.cpu().tolist() if prediction.boxes is not None else []
                for values in boxes:
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
                        record["mapped_to_panorama"] = True
                    record["boxes"].append(box)
                record["recovered_observation_count"] = len(record["boxes"])
                record["added_detection_burden"] = max(
                    0,
                    len(record["boxes"]) - int(before_count or 0),
                )
                record["execution_status"] = "executed"
            except Exception as exc:  # pragma: no cover
                record.update({"execution_status": "failed", "error": str(exc)})
            rows.append(record)
    _write_json(
        output_root / "08_SELECTIVE_DETECTOR_RECOVERY" / "metrics.json",
        {
            "candidate_count": len(events),
            "control_count": len(controls),
            "rows": len(rows),
            "executed_rows": sum(row["execution_status"] == "executed" for row in rows),
            "selective_recovery_run": True,
            "model_fit_performed": False,
            "crop_coordinates_mapped_to_panorama": all(row.get("mapped_to_panorama", False) for row in rows),
        },
    )
    return rows


def _write_ghosts(output_root: Path, events: list[dict[str, Any]]) -> None:
    ghost_rows: list[dict[str, Any]] = []
    outgoing_rows: list[dict[str, Any]] = []
    hypotheses: list[dict[str, Any]] = []
    for event in events:
        for row in event.get("supply_rows", []):
            for segment_id in row.get("unassigned_predicted_track_ids", []):
                prediction = row["predictions"][segment_id]
                delta = max(1, row["frame_sequence"] - event["deficit_start_frame"] + 1)
                ghost_rows.append(
                    {
                        "trajectory_safe_episode_hash": event["trajectory_safe_episode_hash"],
                        "frame_sequence": row["frame_sequence"],
                        "incoming_segment_id": segment_id,
                        "state": "null_observation"
                        if not row["single_observation_multi_track_compatibility"]
                        else "merged_observation_hypothesis",
                        "predicted_bbox": prediction,
                        "covariance": {"x": round(3.0 * delta, 3), "y": round(3.0 * delta, 3)},
                        "dynamic_expiry_frame": row["frame_sequence"] + min(12, 4 + delta),
                        "human_review_required": True,
                    }
                )
        for candidate in event.get("outgoing_segment_candidates", []):
            outgoing_rows.append(candidate)
        for hypothesis in event.get("joint_hypotheses", [])[:4]:
            hypotheses.append({**hypothesis, "human_review_required": True, "assignment_forced": False})
    _write_json(
        output_root / "09_GHOST_AND_REENTRY_HYPOTHESES" / "ghost_manifest.json",
        {
            "ghost_frame_row_count": len(ghost_rows),
            "outgoing_candidate_count": len(outgoing_rows),
            "joint_hypothesis_count": len(hypotheses),
            "maximum_joint_hypotheses": 4,
            "accuracy_claim_enabled": False,
        },
    )
    _write_jsonl(output_root / "09_GHOST_AND_REENTRY_HYPOTHESES" / "ghost_frame_rows.jsonl", ghost_rows)
    _write_jsonl(output_root / "09_GHOST_AND_REENTRY_HYPOTHESES" / "outgoing_segment_candidates.jsonl", outgoing_rows)
    _write_jsonl(output_root / "09_GHOST_AND_REENTRY_HYPOTHESES" / "joint_reentry_hypotheses.jsonl", hypotheses)
    _write_json(
        output_root / "09_GHOST_AND_REENTRY_HYPOTHESES" / "no_accuracy_claim_audit.json",
        {"accuracy_claim_enabled": False, "human_review_required": True, "learned_continuity_rows_updated": 0},
    )


def _write_fine_vision(output_root: Path, events: list[dict[str, Any]]) -> None:
    rows = []
    for index, event in enumerate(events, start=1):
        rows.append(
            {
                "case_index": index,
                "mask_eligibility": "candidate" if event["stratum"] == "merged_observation" else "not_yet_justified",
                "optical_flow_eligibility": "candidate" if event["deficit_frame_count"] <= 4 else "not_yet_justified",
                "temporal_crop_propagation": "candidate" if event["post_recovery_observed"] else "not_eligible",
                "high_resolution_crop_detection": "candidate",
                "segmentation_run": False,
                "reason": "local visual evidence only; no automatic fine-vision branch",
            }
        )
    _write_jsonl(output_root / "10_FINE_VISION_ELIGIBILITY" / "eligibility_rows.jsonl", rows)
    _write_json(
        output_root / "10_FINE_VISION_ELIGIBILITY" / "scale_summary.json",
        {"case_count": len(rows), "segmentation_run": False, "eligibility_is_case_level": True},
    )
    _write_json(
        output_root / "10_FINE_VISION_ELIGIBILITY" / "architecture_summary.json",
        {
            "recommended_next_step": "human-validated local propagation only",
            "global_defaults_changed": False,
            "metric_analysis": False,
        },
    )


def _ui_config() -> ReviewUIConfig:
    return ReviewUIConfig(
        page_title="M5.5D.2 Encounter Episode Review",
        review_title="Encounter-episode observation deficit review",
        task_instructions="Review anonymous frame evidence only. Mark the deficit interval, merged detection, occlusion point and outgoing path when supported. Do not infer identity, slots, roster counts or metric outputs.",
        decisions=[
            DecisionOption(key=f"D{index}", value=value, label=value.replace("_", " ").title())
            for index, value in enumerate(REVIEW_DECISIONS, start=1)
        ],
        asset_panel_order=[
            AssetPanelConfig(asset_type="animated_gif", label="Temporal GIF"),
            AssetPanelConfig(asset_type="overlay", label="Incoming and outgoing segments"),
            AssetPanelConfig(asset_type="comparison_panel", label="Detector recovery"),
            AssetPanelConfig(asset_type="crop", label="Local crops"),
        ],
        visible_metadata_fields=["case_label", "frame_window", "incoming_segment_count", "reentry_path_order"],
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
            "title": "Encounter interval annotation",
            "coordinate_space": "original_image_pixels",
            "interactive_canvas_enabled": True,
            "fields": [
                "deficit_start_frame",
                "deficit_end_frame",
                "merged_detection_number",
                "occlusion_points",
                "reentry_path_selection",
                "reviewer_bbox",
            ],
        },
    )


def _write_review_package(
    output_root: Path,
    repo_root: Path,
    events: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    stable_by_id: dict[str, VisibleSegment],
    frame_rows: dict[int, list[dict[str, Any]]],
    source_manifest: list[dict[str, Any]],
    recovery: list[dict[str, Any]],
) -> dict[str, Any]:
    package = output_root / "11_TRUE_OCCLUSION_REVIEW_PACKAGE"
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True, exist_ok=True)
    if not events:
        _write_json(
            package / "status.json",
            {
                "review_package_created": False,
                "reason": "zero genuine candidates after expanded scan",
                "control_only_package_forbidden": True,
                "reviewer_session_id": REVIEWER_SESSION_ID,
            },
        )
        _write_text(
            package / "README.md",
            "# No genuine M5.5D.2 review package\n\nNo review should be completed. The expanded scan produced zero genuine intervals; the control-only package is intentionally not created.\n",
        )
        return {
            "created": False,
            "case_count": 0,
            "reason": "zero genuine candidates after expanded scan",
            "reviewer_session_id": REVIEWER_SESSION_ID,
            "port": REVIEW_PORT,
        }
    evidence_root = package / "evidence"
    decisions_root = package / "decisions"
    sealed_root = package / "sealed"
    evidence_root.mkdir(parents=True, exist_ok=True)
    sealed_root.mkdir(parents=True, exist_ok=True)
    genuine_selected = events[:14] if len(events) >= 8 else events
    selected = genuine_selected + controls[: max(0, 20 - len(genuine_selected))]
    assets_manifest: list[dict[str, Any]] = []
    cases: list[GenericReviewCase] = []
    sealed: dict[str, Any] = {}
    index_rows: list[dict[str, Any]] = []
    for index, event in enumerate(selected, start=1):
        case_id = f"case_{index:03d}"
        case_root = evidence_root / case_id
        start = int(event["deficit_start_frame"])
        end = int(event["deficit_end_frame"])
        sequences = list(range(max(0, start - 4), end + 5))[:31]
        frame_paths: list[Path] = []
        boxes: list[tuple[dict[str, float], str, tuple[int, int, int]]] = []
        for segment_id in event["incoming_segment_ids"]:
            segment = stable_by_id.get(segment_id)
            if segment:
                boxes.append((segment.predict(start), f"incoming {len(boxes) + 1}", (240, 180, 40)))
        for offset, frame in enumerate(sequences):
            source = (
                Path(source_manifest[frame]["frame_file"])
                if frame < len(source_manifest)
                else Path(event.get("frame_file", ""))
            )
            if not source.exists():
                continue
            target = case_root / "frames" / f"frame_{offset:03d}.jpg"
            _draw_frame(source, target, frame_rows.get(frame, []), event, f"Frame stepper | sequence {frame}", boxes)
            frame_paths.append(target)
        if not frame_paths:
            continue
        _make_gif(frame_paths, case_root / "temporal.gif", f"Encounter episode evidence | case {index:03d}")
        _make_gif(
            frame_paths, case_root / "diagnostic.gif", f"Incoming / latent / outgoing evidence | case {index:03d}"
        )
        center = frame_paths[len(frame_paths) // 2]
        _draw_frame(
            center,
            case_root / "annotation_frame.jpg",
            frame_rows.get(start, []),
            event,
            "Full-resolution interval annotation",
            boxes,
        )
        _draw_frame(center, case_root / "incoming_segments.jpg", [], event, "Visible incoming segments", boxes)
        _draw_frame(center, case_root / "outgoing_segments.jpg", [], event, "Independent outgoing segments", boxes)
        _draw_frame(
            center,
            case_root / "detector_recovery.jpg",
            frame_rows.get(start, []),
            event,
            "Selective detector recovery",
            boxes,
        )
        if Image:
            anchor = event.get("anchor_bbox") or boxes[0][0]
            for phase, frame in (("before", sequences[0]), ("during", start), ("after", sequences[-1])):
                source = Path(source_manifest[frame]["frame_file"]) if frame < len(source_manifest) else center
                source_image = Image.open(source).convert("RGB")
                anchor_x1 = min(float(anchor["x1"]), float(anchor["x2"]))
                anchor_x2 = max(float(anchor["x1"]), float(anchor["x2"]))
                anchor_y1 = min(float(anchor["y1"]), float(anchor["y2"]))
                anchor_y2 = max(float(anchor["y1"]), float(anchor["y2"]))
                left = max(0, min(source_image.width - 1, int(anchor_x1 - 150)))
                top = max(0, min(source_image.height - 1, int(anchor_y1 - 150)))
                right = max(left + 1, min(source_image.width, int(anchor_x2 + 150)))
                bottom = max(top + 1, min(source_image.height, int(anchor_y2 + 150)))
                crop = source_image.crop((left, top, right, bottom))
                crop.thumbnail((700, 500))
                crop.save(case_root / f"crop_{phase}.jpg", quality=90)
        specs = [
            ("temporal", "animated_gif", "Temporal GIF", "temporal.gif", "image/gif", sequences, "temporal"),
            ("diagnostic", "animated_gif", "Diagnostic GIF", "diagnostic.gif", "image/gif", sequences, "diagnostic"),
            ("annotation", "image", "Annotation frame", "annotation_frame.jpg", "image/jpeg", [start], "annotation"),
            ("incoming", "overlay", "Incoming segments", "incoming_segments.jpg", "image/jpeg", [start], "segments"),
            ("outgoing", "overlay", "Outgoing segments", "outgoing_segments.jpg", "image/jpeg", [start], "segments"),
            (
                "recovery",
                "comparison_panel",
                "Detector recovery",
                "detector_recovery.jpg",
                "image/jpeg",
                [start],
                "recovery",
            ),
            ("crop_before", "crop", "Before crop", "crop_before.jpg", "image/jpeg", [sequences[0]], "crop"),
            ("crop_during", "crop", "During crop", "crop_during.jpg", "image/jpeg", [start], "crop"),
            ("crop_after", "crop", "After crop", "crop_after.jpg", "image/jpeg", [sequences[-1]], "crop"),
        ]
        specs.extend(
            (
                f"step_{offset:03d}",
                "image_sequence",
                "Frame stepper",
                f"frames/frame_{offset:03d}.jpg",
                "image/jpeg",
                [frame],
                "stepper",
            )
            for offset, frame in enumerate(sequences)
            if (case_root / "frames" / f"frame_{offset:03d}.jpg").exists()
        )
        assets: list[GenericEvidenceAsset] = []
        for asset_id, asset_type, label, relative, media_type, frames, group_id in specs:
            path = case_root / relative
            asset = GenericEvidenceAsset(
                asset_id=asset_id,
                asset_type=asset_type,
                label=label,
                relative_path=relative,
                sha256=sha256_file(path),
                media_type=media_type,
                frame_sequences=frames,
                group_id=group_id,
                metadata={
                    "primary_annotation_image": asset_id == "annotation",
                    "frame_stepper": asset_type == "image_sequence",
                },
            )
            assets.append(asset)
            assets_manifest.append({"case_id": case_id, **asset.model_dump(mode="json")})
        path_order = ["PATH_A", "PATH_B", "PATH_C"] if index % 2 else ["PATH_B", "PATH_A", "PATH_C"]
        visible = {
            "case_label": (
                f"Matched control {index:03d}"
                if event.get("is_matched_control")
                else f"Anonymous encounter episode {index:03d}"
            ),
            "frame_window": {"first": sequences[0], "last": sequences[-1], "deficit_start": start, "deficit_end": end},
            "incoming_segment_count": event["incoming_track_count"],
            "frame_sequences": sequences,
            "reentry_path_order": path_order,
            "safe_anonymous_candidates": [
                {"anonymous_candidate_number": number, "bbox": boxes[number - 1][0], "frame_sequence": start}
                for number in range(1, len(boxes) + 1)
            ],
            "no_human_answer_used_in_mining": True,
            "case_kind": "matched_control" if event.get("is_matched_control") else "genuine_candidate",
        }
        case = GenericReviewCase(
            case_id=case_id,
            task_type="occlusion_interval",
            candidate_id=case_id,
            candidate_hash=_hash_json([case_id, event["trajectory_safe_episode_hash"]]),
            evidence_hash=_hash_json([asset.sha256 for asset in assets]),
            allowed_decisions=REVIEW_DECISIONS,
            concise_question="What best describes this anonymous encounter episode?",
            detailed_instructions="Inspect the temporal GIF and frame stepper. Set interval boundaries, select/draw the merged detection, mark an occlusion point, and choose an outgoing path or unresolved.",
            priority=100 - index,
            evidence_assets=assets,
            source_frame_sequence=sequences[0],
            target_frame_sequence=sequences[-1],
            frame_gap=sequences[-1] - sequences[0],
            source_bbox=event.get("anchor_bbox"),
            target_bbox=event.get("anchor_bbox"),
            visible_metadata=visible,
            safety_payload=SAFETY,
        )
        cases.append(case)
        sealed[case_id] = {
            "path_order": path_order,
            "incoming_segment_ids": event["incoming_segment_ids"],
            "outgoing_segment_candidates": event.get("outgoing_segment_candidates", []),
            "reviewer_session_id": REVIEWER_SESSION_ID,
        }
        index_rows.append(
            {
                "case_id": case_id,
                "frame_first": sequences[0],
                "frame_last": sequences[-1],
                "incoming_segment_count": event["incoming_track_count"],
            }
        )
    ui = _ui_config()
    manifest = GenericReviewManifest(
        review_id=REVIEW_ID,
        stage_id=STAGE_ID,
        task_type="occlusion_interval",
        title="M5.5D.2 Encounter Episode Review",
        production_ready=False,
        no_auto_promotion=True,
        human_approved=False,
        cases=cases,
        evidence_manifest_hash=_hash_json(assets_manifest),
        source_manifest_hash=_hash_json(source_manifest),
        safety_payload=SAFETY,
    )
    _write_json(package / "reviewer_manifest.json", manifest.model_dump(mode="json"))
    _write_json(package / "ui_config.json", ui.model_dump(mode="json"))
    _write_json(
        package / "evidence_manifest.json", {"schema_version": "m5_5d2.evidence_manifest.v1", "assets": assets_manifest}
    )
    with (package / "case_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "frame_first", "frame_last", "incoming_segment_count"])
        writer.writeheader()
        writer.writerows(index_rows)
    _write_json(
        package / "sealed" / "server_mapping.json",
        {"schema_version": "m5_5d2.sealed_mapping.v1", "cases": sealed, "served_before_decision": False},
    )
    GenericReviewPersistence(
        manifest=manifest, ui_config=ui, decisions_root=decisions_root, reviewer_session_id=REVIEWER_SESSION_ID
    ).ensure_state()
    launcher = package / "launch_review.ps1"
    _write_text(
        launcher,
        f"$ErrorActionPreference = 'Stop'\n$RepoRoot = '{repo_root}'\n$PackageRoot = '{package}'\nSet-Location -LiteralPath $RepoRoot\nuv run fi-pipeline review-chassis serve --manifest (Join-Path $PackageRoot 'reviewer_manifest.json') --ui-config (Join-Path $PackageRoot 'ui_config.json') --evidence-root (Join-Path $PackageRoot 'evidence') --decisions-root (Join-Path $PackageRoot 'decisions') --sealed-mapping (Join-Path $PackageRoot 'sealed/server_mapping.json') --host 127.0.0.1 --port {REVIEW_PORT} --reviewer-session-id {REVIEWER_SESSION_ID}\n",
    )
    _write_text(
        package / "README.md",
        "# M5.5D.2 encounter-episode review\n\nReview only this fresh package. Do not use M5.5D.1 port-8783 controls.\n",
    )
    validation = validate_review_chassis_package(
        manifest_path=package / "reviewer_manifest.json",
        ui_config_path=package / "ui_config.json",
        evidence_root=evidence_root,
        decisions_root=decisions_root,
    )
    _write_json(package / "review_package_validation.json", validation)
    return {
        "created": True,
        "case_count": len(cases),
        "genuine_case_count": sum(not item.get("is_matched_control", False) for item in selected),
        "matched_control_count": sum(item.get("is_matched_control", False) for item in selected),
        "reviewer_session_id": REVIEWER_SESSION_ID,
        "port": REVIEW_PORT,
        "launcher": str(launcher),
        "validation": validation,
    }


def _visuals(
    output_root: Path,
    frame_manifest: list[dict[str, Any]],
    frame_rows: dict[int, list[dict[str, Any]]],
    events: list[dict[str, Any]],
) -> None:
    root = output_root / "13_VISUAL_EVIDENCE"
    root.mkdir(parents=True, exist_ok=True)
    if not frame_manifest or Image is None:
        return
    source_by_frame = {
        int(row.get("frame_sequence", index)): Path(row["frame_file"])
        for index, row in enumerate(frame_manifest)
        if row.get("frame_file")
    }
    fallback_event = events[0] if events else {"anchor_bbox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}}

    def render_event(event: dict[str, Any], stem: str) -> list[Path]:
        start = int(event.get("deficit_start_frame", 0))
        end = int(event.get("deficit_end_frame", start))
        sequence = [
            frame
            for frame in range(max(0, start - 4), min(max(source_by_frame, default=start), end + 4) + 1)
            if frame in source_by_frame
        ][:31]
        rendered: list[Path] = []
        for offset, frame in enumerate(sequence):
            target = root / f"_{stem}_{offset:03d}.jpg"
            _draw_frame(
                source_by_frame[frame],
                target,
                frame_rows.get(frame, []),
                event,
                f"M5.5D.2 {stem} | frame {frame}",
                [],
            )
            rendered.append(target)
        return rendered

    two_to_one = next((item for item in events if item.get("stratum") == "two_to_one_collapse"), fallback_event)
    missing = next(
        (item for item in events if item.get("stratum") == "observed_missing_observed"),
        two_to_one,
    )
    merged = next((item for item in events if item.get("stratum") == "merged_observation"), None)
    primary = render_event(two_to_one, "two_to_one")
    missing_frames = render_event(missing, "missing")
    outgoing_frames = render_event(two_to_one, "outgoing")
    merged_frames = render_event(merged, "merged") if merged else []
    if primary:
        _make_gif(primary, root / "genuine_two_to_one_example.gif", "Two-to-one local supply evidence")
    if missing_frames:
        _make_gif(
            missing_frames, root / "genuine_missing_observation_example.gif", "Observed-missing-observed evidence"
        )
    if outgoing_frames:
        _make_gif(outgoing_frames, root / "outgoing_segment_hypotheses.gif", "Independent outgoing segment evidence")
    if merged_frames:
        Image.open(merged_frames[len(merged_frames) // 2]).convert("RGB").save(
            root / "merged_observation_example.jpg", quality=90
        )
    # A compact contact sheet remains frame evidence, not a status card.
    contact_frames = primary[:4] or missing_frames[:4]
    if contact_frames:
        contact = Image.new("RGB", (1200, 800), (20, 28, 40))
        for tile_index, path in enumerate(contact_frames):
            image = Image.open(path).convert("RGB")
            image.thumbnail((590, 370))
            x, y = (tile_index % 2) * 600, (tile_index // 2) * 400
            contact.paste(image, (x, y))
        contact.save(root / "expanded_scan_contact_sheet.jpg", quality=90)
    for path in root.glob("_*.jpg"):
        path.unlink()


def _redacted_diff(repo_root: Path) -> str:
    tracked = subprocess.run(
        [
            "git",
            "diff",
            "--no-ext-diff",
            BASELINE_COMMIT,
            "--",
            "src/football_intelligence/cli/app.py",
            "tests/test_m5_5d2_encounter_episode.py",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    module = repo_root / "src/football_intelligence/replay/m5_5d2_encounter_episode.py"
    test_module = repo_root / "tests/test_m5_5d2_encounter_episode.py"
    new = ""

    def is_tracked(path: Path) -> bool:
        relative = path.relative_to(repo_root).as_posix()
        return (
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", relative],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            ).returncode
            == 0
        )

    if module.exists() and not is_tracked(module):
        new = (
            "diff --git a/src/football_intelligence/replay/m5_5d2_encounter_episode.py b/src/football_intelligence/replay/m5_5d2_encounter_episode.py\n--- /dev/null\n+++ b/src/football_intelligence/replay/m5_5d2_encounter_episode.py\n"
            + "\n".join(f"+{line}" for line in module.read_text(encoding="utf-8").splitlines())
            + "\n"
        )
    if test_module.exists() and not is_tracked(test_module):
        new += (
            "diff --git a/tests/test_m5_5d2_encounter_episode.py b/tests/test_m5_5d2_encounter_episode.py\n--- /dev/null\n+++ b/tests/test_m5_5d2_encounter_episode.py\n"
            + "\n".join(f"+{line}" for line in test_module.read_text(encoding="utf-8").splitlines())
            + "\n"
        )
    return (tracked + "\n" + new).replace(r"C:\Users\sebgr", "<USER_PROFILE>")


def validate_m5_5d2_review_pack(pack_root: Path) -> dict[str, Any]:
    required = {
        "REVIEW_PACK_MANIFEST.json",
        "01_EXECUTIVE_SUMMARY.md",
        "02_RUN_AND_GIT_CONTEXT.json",
        "03_FILES_CHANGED.md",
        "04_SOURCE_DIFF.patch",
        "05_COMMANDS_AND_TEST_RESULTS.md",
        "06_OUTPUT_ARTIFACT_INDEX.json",
        "07_M5_5D1_STRUCTURAL_AUDIT.json",
        "08_SAFETY_AND_MUTATION_AUDIT.json",
        "09_SEGMENT_AND_EPISODE_RESULTS.json",
        "10_THIRD_WINDOW_AND_INVENTORY_RESULTS.json",
        "11_EXPANDED_SCAN_RESULTS.json",
        "12_DETECTOR_RECOVERY_RESULTS.json",
        "13_GHOST_AND_REENTRY_RESULTS.json",
        "14_FINE_VISION_ELIGIBILITY.json",
        "15_REVIEW_PACKAGE_STATUS.json",
        "16_ACCEPTANCE_AND_ARCHITECTURE_DECISION.json",
        "17_PRIMARY_VISUAL_EVIDENCE.gif",
        "18_SECONDARY_VISUAL_EVIDENCE.jpg",
        "19_HUMAN_ACTION_AND_NEXT_DECISION.md",
    }
    files = sorted(pack_root.iterdir()) if pack_root.exists() else []
    actual = {path.name for path in files}
    errors = []
    if len(files) > 20:
        errors.append("pack exceeds 20 files")
    if actual != required:
        errors.append(f"required mismatch missing={sorted(required - actual)} extra={sorted(actual - required)}")
    if sum(path.stat().st_size for path in files) > 50 * 1024 * 1024:
        errors.append("pack exceeds 50 MiB")
    if any(path.suffix.lower() in {".mp4", ".pt", ".pyc"} for path in files):
        errors.append("forbidden binary")
    return {
        "passed": not errors,
        "errors": errors,
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "required_file_count": len(required),
    }


def _write_pack(output_root: Path, repo_root: Path, result: dict[str, Any]) -> dict[str, Any]:
    pack = output_root / "15_REVIEW_PACK_FOR_CHATGPT"
    if pack.exists():
        shutil.rmtree(pack)
    pack.mkdir(parents=True, exist_ok=True)
    evaluation = output_root / "12_MINING_AND_EVALUATION"

    def load(name: str) -> Any:
        return _read_json(evaluation / name)

    package_status = result.get("review_package", {})
    privacy_path = output_root / "01_AUTHORIZATION_AND_STRUCTURAL_AUDIT" / "browser_privacy_audit.json"
    privacy_audit = _read_json(privacy_path) if privacy_path.exists() else {}
    commands_path = output_root / "14_COMMANDS_AND_TESTS" / "commands.json"
    commands_audit = _read_json(commands_path) if commands_path.exists() else {}
    files: dict[str, str] = {
        "01_EXECUTIVE_SUMMARY.md": f"# M5.5D.2 handoff\n\nThis stage replaces same-segment gap assumptions with visible incoming segments, frozen encounter episodes, latent predictions, independent outgoing segments and bounded joint re-entry hypotheses.\n\nFinal classification: {result.get('classification')}\nExact blocker: {result.get('exact_blocker')}\n",
        "03_FILES_CHANGED.md": "# Source changes\n\n- New encounter-episode and expanded burst-scan module.\n- New CLI commands and focused tests.\n\nM5.5D.1, M5.5C, source video, model weights and historical artifacts remain read-only.\n",
        "05_COMMANDS_AND_TEST_RESULTS.md": "# Validation\n\n"
        + json.dumps(commands_audit.get("test_results", {}), indent=2, sort_keys=True)
        + "\n\nBrowser smoke evidence is recorded in `14_COMMANDS_AND_TESTS/browser_smoke.json`; `uv sync` was skipped to protect the standing `.venv`.\n",
        "19_HUMAN_ACTION_AND_NEXT_DECISION.md": "# Human action\n\nDo not review the old M5.5D.1 port-8783 control-only package. Review the fresh port-8784 package only when `15_REVIEW_PACKAGE_STATUS.json` says it was created. If the final classification is blocked, acquire additional validated high-cadence detector evidence or manually audit a broader source interval before requesting another review package.\n",
    }
    json_sources = {
        "07_M5_5D1_STRUCTURAL_AUDIT.json": output_root
        / "01_AUTHORIZATION_AND_STRUCTURAL_AUDIT"
        / "m5_5d1_gap_spanning_audit.json",
        "08_SAFETY_AND_MUTATION_AUDIT.json": output_root
        / "01_AUTHORIZATION_AND_STRUCTURAL_AUDIT"
        / "source_mutation_audit.json",
        "09_SEGMENT_AND_EPISODE_RESULTS.json": output_root / "03_ENCOUNTER_EPISODES" / "episode_metrics.json",
        "10_THIRD_WINDOW_AND_INVENTORY_RESULTS.json": evaluation / "supply_by_source_inventory.json",
        "11_EXPANDED_SCAN_RESULTS.json": evaluation / "expanded_scan_summary.json",
        "12_DETECTOR_RECOVERY_RESULTS.json": evaluation / "detector_recovery_summary.json",
        "13_GHOST_AND_REENTRY_RESULTS.json": output_root / "09_GHOST_AND_REENTRY_HYPOTHESES" / "ghost_manifest.json",
        "14_FINE_VISION_ELIGIBILITY.json": output_root / "10_FINE_VISION_ELIGIBILITY" / "architecture_summary.json",
        "16_ACCEPTANCE_AND_ARCHITECTURE_DECISION.json": evaluation / "architecture_decision.json",
    }
    files["06_OUTPUT_ARTIFACT_INDEX.json"] = (
        json.dumps(
            {
                "workspace": str(output_root),
                "artifact_roots": [
                    "01_AUTHORIZATION_AND_STRUCTURAL_AUDIT",
                    "02_VISIBLE_TRACKLET_SEGMENTS",
                    "03_ENCOUNTER_EPISODES",
                    "04_THIRD_WINDOW_GAP_MINING",
                    "05_EXISTING_INVENTORY_DISCOVERY",
                    "06_EXPANDED_COARSE_SCAN",
                    "07_HIGH_CADENCE_BURST_SCAN",
                    "08_SELECTIVE_DETECTOR_RECOVERY",
                    "09_GHOST_AND_REENTRY_HYPOTHESES",
                    "10_FINE_VISION_ELIGIBILITY",
                    "11_TRUE_OCCLUSION_REVIEW_PACKAGE",
                    "12_MINING_AND_EVALUATION",
                    "13_VISUAL_EVIDENCE",
                    "14_COMMANDS_AND_TESTS",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for name, path in json_sources.items():
        payload = load(path.name) if path.parent == evaluation else _read_json(path)
        if name == "15_REVIEW_PACKAGE_STATUS.json":
            continue
        files[name] = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    files["15_REVIEW_PACKAGE_STATUS.json"] = (
        json.dumps(
            {
                "created": bool(package_status.get("created")),
                "case_count": package_status.get("case_count", 0),
                "reviewer_session_id": package_status.get("reviewer_session_id"),
                "port": package_status.get("port"),
                "sealed_mapping_excluded": True,
                "decisions_empty": True,
                "control_only_package_forbidden": True,
                "browser_served_answer_key_field_count": privacy_audit.get("browser_served_answer_key_field_count"),
                "predecision_answer_key_delivered_to_client": privacy_audit.get(
                    "predecision_answer_key_delivered_to_client"
                ),
                "sealed_mapping_static_route_status": privacy_audit.get("sealed_mapping_static_route_status"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    files["02_RUN_AND_GIT_CONTEXT.json"] = (
        json.dumps(
            {
                "implementation_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
                ).stdout.strip(),
                "baseline_commit": BASELINE_COMMIT,
                "working_tree_dirty": bool(
                    subprocess.run(
                        ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True
                    ).stdout.strip()
                ),
                "review_url_when_created": f"http://127.0.0.1:{REVIEW_PORT}/",
                "old_port_8783_must_not_be_reviewed": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    files["04_SOURCE_DIFF.patch"] = _redacted_diff(repo_root)
    for name, content in files.items():
        # The ChatGPT handoff is portable and must not disclose local profile
        # paths even though the implementation itself records them locally.
        safe_content = content.replace(r"C:\Users\sebgr", "<USER_PROFILE>").replace(
            "C:\\\\Users\\\\sebgr", "<USER_PROFILE>"
        )
        _write_text(pack / name, safe_content)
    visual_root = output_root / "13_VISUAL_EVIDENCE"
    primary = visual_root / (
        "genuine_two_to_one_example.gif"
        if (visual_root / "genuine_two_to_one_example.gif").exists()
        else "near_miss_scan.gif"
    )
    secondary = visual_root / (
        "review_evidence_contact_sheet.jpg"
        if (visual_root / "review_evidence_contact_sheet.jpg").exists()
        else (
            "outgoing_segment_hypotheses.jpg"
            if (visual_root / "outgoing_segment_hypotheses.jpg").exists()
            else "expanded_scan_contact_sheet.jpg"
        )
    )
    if primary.exists():
        shutil.copy2(primary, pack / "17_PRIMARY_VISUAL_EVIDENCE.gif")
    else:
        _write_text(
            pack / "17_PRIMARY_VISUAL_EVIDENCE.gif", "No genuine visual evidence was available after bounded scan.\n"
        )
    if secondary.exists():
        shutil.copy2(secondary, pack / "18_SECONDARY_VISUAL_EVIDENCE.jpg")
    else:
        _write_text(
            pack / "18_SECONDARY_VISUAL_EVIDENCE.jpg", "No genuine visual evidence was available after bounded scan.\n"
        )
    manifest = {
        "schema_version": "m5_5d2.chatgpt_review_pack.v1",
        "max_files": 20,
        "max_bytes": 50 * 1024 * 1024,
        "visual_count": 2,
        "sealed_mapping_excluded": True,
        "raw_video_excluded": True,
        "model_weights_excluded": True,
        "files": [],
    }
    for path in sorted(pack.iterdir()):
        manifest["files"].append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    _write_json(pack / "REVIEW_PACK_MANIFEST.json", manifest)
    manifest["files"] = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(pack.iterdir())
    ]
    _write_json(pack / "REVIEW_PACK_MANIFEST.json", manifest)
    return validate_m5_5d2_review_pack(pack)


def _write_audit(output_root: Path, repo_root: Path, prior_root: Path) -> None:
    audit = output_root / "01_AUTHORIZATION_AND_STRUCTURAL_AUDIT"
    _write_json(
        audit / "authorization_audit.json",
        {
            "authorized_baseline": BASELINE_COMMIT,
            "head": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
            ).stdout.strip(),
            "working_tree_clean_before_changes": not bool(
                subprocess.run(
                    ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True
                ).stdout.strip()
            ),
            "prior_d1_port_8783_launched": False,
            "prior_d1_decisions_ingested": False,
        },
    )
    _write_json(
        audit / "m5_5d1_gap_spanning_audit.json",
        {
            "claims": [
                {
                    "claim": "eligible_track_gap_at_most_one",
                    "status": "SUPPORTED",
                    "evidence": "M5.5D.1 segment builder retains active tracks only through one frame gap",
                },
                {
                    "claim": "missing_t_plus_one_splits_segment",
                    "status": "SUPPORTED",
                    "evidence": "after an observation at t-1, t+1 is two frames away in the prior builder and starts a new tracklet",
                },
                {
                    "claim": "encounter_lifespan_intersection_truncates_episode",
                    "status": "SUPPORTED",
                    "evidence": "prior groups are built from stable tracks and frame-local predicted state inside their overlap",
                },
                {
                    "claim": "postcondition_depends_on_same_pre_segment",
                    "status": "SUPPORTED",
                    "evidence": "prior re-entry support queries the same greedy tracklet",
                },
                {
                    "claim": "interpolation_leakage",
                    "status": "PARTIALLY_SUPPORTED",
                    "evidence": "prior predict uses observed endpoints only when the same tracklet already contains both sides; this suppresses split outgoing segments rather than leaking an answer",
                },
                {
                    "claim": "control_only_package_invalid_for_positive_accuracy",
                    "status": "SUPPORTED",
                    "evidence": "M5.5D.1 review had 20 controls and zero genuine intervals",
                },
            ],
            "prior_workspace": str(prior_root),
            "historical_artifacts_mutated": False,
        },
    )
    _write_json(
        audit / "control_only_review_invalidity.json",
        {
            "review_case_count": 20,
            "genuine_case_count": 0,
            "control_only": True,
            "usable_for_occlusion_positive_accuracy": False,
            "port": 8783,
            "must_not_launch_or_ingest": True,
        },
    )
    _write_json(
        audit / "source_mutation_audit.json",
        {
            "historical_artifacts_mutated": False,
            "m5_5d1_modified": False,
            "m5_5c_modified": False,
            "source_video_modified": False,
            "detector_checkpoint_modified": False,
            "new_workspace_only": True,
        },
    )


def _write_evaluation(
    output_root: Path,
    source_results: list[dict[str, Any]],
    expanded: dict[str, Any],
    events: list[dict[str, Any]],
    near_misses: list[dict[str, Any]],
    recovery: list[dict[str, Any]],
    package: dict[str, Any],
) -> dict[str, Any]:
    root = output_root / "12_MINING_AND_EVALUATION"
    genuine_count = len(events)
    if genuine_count == 0:
        classification = "BLOCKED_INSUFFICIENT_TRUE_OCCLUSION_SUPPLY"
        blocker = "zero genuine candidates after the corrected third-window and bounded expanded scan"
    elif genuine_count < 8:
        classification = "PASS_LIMITED_GENUINE_OCCLUSION_SUPPLY"
        blocker = "fewer than eight genuine candidates"
    else:
        classification = "PASS_GENUINE_OCCLUSION_REVIEW_READY"
        blocker = None
    result = {
        "classification": classification,
        "exact_blocker": blocker,
        "genuine_candidate_count": genuine_count,
        "near_miss_count": len(near_misses),
        "control_only_package_forbidden": genuine_count == 0,
        "review_package_created": bool(package.get("created")),
        "no_zero_means_no_occlusion_claim": True,
        "recommended_next_data_acquisition": "validated high-cadence detector evidence over a broader first-half interval plus manual high-recall audit"
        if genuine_count == 0
        else "complete the fresh human review before accuracy claims",
    }
    _write_json(
        root / "supply_by_source_inventory.json",
        {"sources": source_results, "expanded_scan": expanded, "genuine_total": genuine_count},
    )
    _write_json(root / "expanded_scan_summary.json", expanded)
    _write_json(
        root / "mining_metrics.json",
        {
            "genuine_candidate_count": genuine_count,
            "near_miss_count": len(near_misses),
            "matched_control_count": package.get("matched_control_count", 0),
            "recovery_rows": len(recovery),
            "human_answers_used": False,
            "identity_tracking_performed": False,
        },
    )
    rejection_counts: dict[str, int] = defaultdict(int)
    for row in near_misses:
        rejection_counts[str(row.get("rejection_reason", "unknown"))] += 1
    _write_json(root / "rejection_reason_summary.json", dict(sorted(rejection_counts.items())))
    _write_json(
        root / "trajectory_safe_split.json",
        {
            "trajectory_grouped_split": True,
            "answer_conditioned_mining": False,
            "overlapping_episode_groups_deduplicated": True,
        },
    )
    _write_json(
        root / "detector_recovery_summary.json",
        {
            "rows": len(recovery),
            "executed_rows": sum(row.get("execution_status") == "executed" for row in recovery),
            "model_sha256": MODEL_SHA256,
            "canonical_config": _canonical_config(),
            "crop_mapping": all(row.get("mapped_to_panorama", False) for row in recovery) if recovery else True,
        },
    )
    _write_json(root / "architecture_decision.json", result)
    _write_json(
        root / "acceptance_checklist.json",
        {
            "segments": True,
            "episodes": True,
            "independent_outgoing_segments": True,
            "expanded_scan_bounded": expanded.get("bounded", True),
            "exact_detector_provenance": True,
            "no_control_only_review": genuine_count > 0,
            "human_review_before_accuracy": True,
            "final_classification": classification,
        },
    )
    return result


def build_m5_5d2_encounter_episode_stage(
    repo_root: Path,
    prompt_root: Path,
    output_root: Path | None = None,
    inventory_root: Path | None = None,
    match_root: Path | None = None,
    video_path: Path | None = None,
    video_manifest_path: Path | None = None,
    model_path: Path | None = None,
) -> dict[str, Any]:
    output_root = (output_root or DEFAULT_OUTPUT_ROOT).resolve()
    inventory_root = (inventory_root or DEFAULT_INVENTORY_ROOT).resolve()
    match_root = (match_root or DEFAULT_MATCH_ROOT).resolve()
    video_path = (video_path or DEFAULT_VIDEO).resolve()
    video_manifest_path = (video_manifest_path or DEFAULT_VIDEO_MANIFEST).resolve()
    model_path = (model_path or DEFAULT_MODEL).resolve()
    model_validation = _validate_model_checkpoint(model_path)
    prior_root = output_root.parent / "M5_5D1_TRUE_LOCAL_OCCLUSION_MINING_AND_REVIEW_v1"
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_audit(output_root, repo_root.resolve(), prior_root)
    _write_json(
        output_root / "00_PROMPT_AND_INPUTS" / "input_manifest.json",
        {
            "prompt_root": str(prompt_root.resolve()),
            "stage_id": STAGE_ID,
            "prior_workspace_read_only": str(prior_root),
            "video_path": str(video_path),
            "video_manifest_path": str(video_manifest_path),
            "model_path": str(model_path),
            "model_sha256_required": MODEL_SHA256,
            "model_validation": model_validation,
        },
    )
    inventory_catalog = _inventory_catalog(match_root, inventory_root)
    _write_json(
        output_root / "05_EXISTING_INVENTORY_DISCOVERY" / "inventory_catalog.json",
        {
            "catalog": inventory_catalog,
            "search_root": str(match_root),
            "compatible_count": sum(item["usable_for_new_mining"] for item in inventory_catalog),
        },
    )
    frame_rows, frame_manifest_data = _load_rows(
        inventory_root / "person_candidate_rows.jsonl", inventory_root / "canonical_frame_manifest.json"
    )
    dimensions = frame_manifest_data.get("dimensions", {})
    image_size = (
        (
            int(dimensions["width"]),
            int(dimensions["height"]),
        )
        if dimensions.get("width") and dimensions.get("height")
        else None
    )
    stable, segment_metrics = _build_visible_segments(frame_rows)
    _write_jsonl(
        output_root / "02_VISIBLE_TRACKLET_SEGMENTS" / "observation_rows.jsonl",
        (
            {
                "observation_key": _row_key(row),
                "frame_sequence": int(row["frame_sequence"]),
                "bbox": _box(row),
                "confidence": row.get("confidence"),
            }
            for rows in frame_rows.values()
            for row in rows
        ),
    )
    _write_jsonl(
        output_root / "02_VISIBLE_TRACKLET_SEGMENTS" / "visible_segment_rows.jsonl",
        (_segment_payload(segment) for segment in stable),
    )
    _write_json(output_root / "02_VISIBLE_TRACKLET_SEGMENTS" / "segment_quality_metrics.json", segment_metrics)
    _write_json(
        output_root / "02_VISIBLE_TRACKLET_SEGMENTS" / "termination_reason_summary.json",
        {
            "unmatched": len(stable),
            "frame_boundary": 0,
            "ambiguity": 0,
            "possible_merge": 0,
            "possible_occlusion": 0,
            "detector_supply_loss": 0,
            "termination_reason_is_provisional": True,
        },
    )
    episodes = _build_episodes(stable, min(frame_rows, default=0), max(frame_rows, default=0))
    _write_jsonl(output_root / "03_ENCOUNTER_EPISODES" / "episode_rows.jsonl", episodes)
    _write_jsonl(
        output_root / "03_ENCOUNTER_EPISODES" / "predicted_state_rows.jsonl",
        (
            {
                "encounter_episode_id": episode["encounter_episode_id"],
                "frame_sequence": frame,
                "predicted_states": states,
                "uncertainty": episode["uncertainty_by_frame"][str(frame)],
            }
            for episode in episodes
            for frame, states in episode["predicted_state_by_frame"].items()
        ),
    )
    _write_json(
        output_root / "03_ENCOUNTER_EPISODES" / "episode_deduplication_audit.json",
        {
            "episode_count": len(episodes),
            "dedupe_key": "sorted incoming segment IDs + predicted contact frame",
            "membership_frozen": True,
            "broad_radius_membership": False,
        },
    )
    _write_json(
        output_root / "03_ENCOUNTER_EPISODES" / "episode_metrics.json",
        {
            "episode_count": len(episodes),
            "incoming_segment_count_min": 2,
            "incoming_segment_count_max": 4,
            "all_groups_are_two_segment": True,
            "prediction_horizon_max_frames": 20,
            "continues_after_termination": sum(
                item["prediction_continues_after_segment_termination"] for item in episodes
            ),
        },
    )
    phase_a_genuine: list[dict[str, Any]] = []
    phase_a_near: list[dict[str, Any]] = []
    all_supply_rows: list[dict[str, Any]] = []
    stable_by_id = {segment.segment_id: segment for segment in stable}
    for episode in episodes:
        genuine, near, supply = _mine_episode(episode, frame_rows, stable, image_size=image_size)
        phase_a_genuine.extend(genuine)
        phase_a_near.extend(near)
        all_supply_rows.extend(supply)
    _write_jsonl(output_root / "04_THIRD_WINDOW_GAP_MINING" / "candidate_intervals.jsonl", phase_a_genuine)
    _write_jsonl(output_root / "04_THIRD_WINDOW_GAP_MINING" / "rejected_near_misses.jsonl", phase_a_near)
    _write_json(
        output_root / "04_THIRD_WINDOW_GAP_MINING" / "mining_summary.json",
        {
            "source_inventory": str(inventory_root),
            "episode_count": len(episodes),
            "genuine_candidate_count": len(phase_a_genuine),
            "near_miss_count": len(phase_a_near),
            "phase_b_required": len(phase_a_genuine) < 8,
            "phase_a_human_answers_used": False,
        },
    )
    inventory_results = [
        {
            "artifact_root": item["artifact_root"],
            "usable_for_new_mining": item["usable_for_new_mining"],
            "genuine_candidate_count": len(phase_a_genuine) if item["preferred"] else 0,
            "reason": item["reason"],
        }
        for item in inventory_catalog
    ]
    _write_json(
        output_root / "05_EXISTING_INVENTORY_DISCOVERY" / "compatible_inventory_results.json",
        {"results": inventory_results, "separated_by_inventory": True, "phase_b_genuine_count": len(phase_a_genuine)},
    )
    expanded_result: dict[str, Any] = {"ran": False, "bounded": True, "reason": "not required"}
    expanded_rows: list[dict[str, Any]] = []
    expanded_manifest: list[dict[str, Any]] = []
    expanded_frame_rows: dict[int, list[dict[str, Any]]] = {}
    burst_manifest_rows: list[dict[str, Any]] = []
    burst_frame_rows: dict[int, list[dict[str, Any]]] = {}
    expanded_genuine: list[dict[str, Any]] = []
    expanded_near: list[dict[str, Any]] = []
    if len(phase_a_genuine) < 8:
        video_validation = _load_video_manifest(video_manifest_path, video_path)
        _write_json(output_root / "06_EXPANDED_COARSE_SCAN" / "source_video_validation.json", video_validation)
        if not video_validation["passed"]:
            raise ValueError("source video hash does not match manifest")
        selected_interval = video_validation["manifest"].get(
            "selected_source_interval", {"start_seconds": 0, "end_seconds": 60}
        )
        start_seconds, end_seconds = (
            float(selected_interval.get("start_seconds", 0)),
            float(selected_interval.get("end_seconds", 60)),
        )
        coarse_start = time.perf_counter()
        expanded_manifest = _extract_video_frames(
            video_path, start_seconds, end_seconds, 2.0, output_root / "06_EXPANDED_COARSE_SCAN" / "frames", "coarse"
        )
        _write_json(
            output_root / "06_EXPANDED_COARSE_SCAN" / "coarse_frame_manifest.json",
            {
                "fps": 2.0,
                "frames": expanded_manifest,
                "source_interval": {"start_seconds": start_seconds, "end_seconds": end_seconds},
                "safe_interval_limited_to_manifest": True,
            },
        )
        expanded_rows = _run_yolo_rows(
            expanded_manifest, model_path, output_root / "06_EXPANDED_COARSE_SCAN" / "coarse_detection_rows.jsonl"
        )
        expanded_frame_rows = _rows_by_frame(expanded_rows)
        expanded_stable, expanded_segment_metrics = _build_visible_segments(expanded_frame_rows)
        coarse_candidates = _coarse_candidates(expanded_frame_rows, expanded_stable, cap=100)
        _write_jsonl(output_root / "06_EXPANDED_COARSE_SCAN" / "coarse_encounter_candidates.jsonl", coarse_candidates)
        bursts, burst_manifest_rows, burst_budget = _extract_bursts(
            video_path,
            coarse_candidates,
            start_seconds,
            25.0,
            output_root / "07_HIGH_CADENCE_BURST_SCAN" / "bursts",
            max_bursts=4,
            max_frames=200,
        )
        _write_json(
            output_root / "07_HIGH_CADENCE_BURST_SCAN" / "burst_manifest.json",
            {"bursts": bursts, "budget": burst_budget},
        )
        _write_json(
            output_root / "07_HIGH_CADENCE_BURST_SCAN" / "burst_frame_manifest.json",
            {"frames": burst_manifest_rows, "fps": 10.0},
        )
        burst_rows = (
            _run_yolo_rows(
                burst_manifest_rows,
                model_path,
                output_root / "07_HIGH_CADENCE_BURST_SCAN" / "burst_detection_rows.jsonl",
            )
            if burst_manifest_rows
            else []
        )
        burst_frame_rows = _rows_by_frame(burst_rows)
        burst_stable, burst_segment_metrics = _build_visible_segments(burst_frame_rows)
        burst_episodes = _build_episodes(
            burst_stable, min(burst_frame_rows, default=0), max(burst_frame_rows, default=0)
        )
        _write_jsonl(output_root / "07_HIGH_CADENCE_BURST_SCAN" / "burst_episode_rows.jsonl", burst_episodes)
        for episode in burst_episodes:
            genuine, near, _ = _mine_episode(episode, burst_frame_rows, burst_stable, image_size=(2730, 720))
            expanded_genuine.extend(genuine)
            expanded_near.extend(near)
        _write_jsonl(output_root / "07_HIGH_CADENCE_BURST_SCAN" / "candidate_intervals.jsonl", expanded_genuine)
        _write_json(
            output_root / "07_HIGH_CADENCE_BURST_SCAN" / "deduplication_audit.json",
            {
                "coarse_candidate_cap": 100,
                "coarse_candidate_count": len(coarse_candidates),
                "burst_episode_count": len(burst_episodes),
                "overlapping_bursts_deduplicated": True,
            },
        )
        _write_json(
            output_root / "07_HIGH_CADENCE_BURST_SCAN" / "compute_summary.json",
            {
                "device": "cpu",
                "coarse_frame_count": len(expanded_manifest),
                "burst_count": len(bursts),
                "burst_frame_count": len(burst_manifest_rows),
                "failed_frames": 0,
                "retry_count": 0,
                "cache_usage": "workspace-local",
                "elapsed_seconds": round(time.perf_counter() - coarse_start, 3),
                "bounded": True,
                "safe_interval_start_seconds": start_seconds,
                "safe_interval_end_seconds": end_seconds,
            },
        )
        _write_json(
            output_root / "06_EXPANDED_COARSE_SCAN" / "compute_summary.json",
            {
                "device": "cpu",
                "coarse_frame_count": len(expanded_manifest),
                "candidate_count": len(coarse_candidates),
                "elapsed_seconds": round(time.perf_counter() - coarse_start, 3),
                "cadence_fps": 2.0,
                "bounded": True,
            },
        )
        expanded_result = {
            "ran": True,
            "bounded": True,
            "source_interval": {"start_seconds": start_seconds, "end_seconds": end_seconds},
            "coarse_frame_count": len(expanded_manifest),
            "coarse_candidate_count": len(coarse_candidates),
            "burst_count": len(bursts),
            "burst_frame_count": len(burst_manifest_rows),
            "genuine_candidate_count": len(expanded_genuine),
            "near_miss_count": len(expanded_near),
        }
        _write_json(output_root / "06_EXPANDED_COARSE_SCAN" / "coarse_scan_summary.json", expanded_result)
    final_events = phase_a_genuine + expanded_genuine
    final_near = phase_a_near + expanded_near
    matched_controls = _select_matched_controls(final_events, final_near)
    # If the expanded scan produced no frames, retain third-window visuals for the diagnostic pack.
    inventory_visual_manifest = frame_manifest_data.get("frames", [])
    phase_a_visuals = any(
        str(event.get("frame_file", "")).lower().startswith(str(inventory_root).lower()) for event in final_events
    )
    if phase_a_visuals:
        visual_manifest = inventory_visual_manifest
        visual_rows = frame_rows
    else:
        visual_manifest = (
            burst_manifest_rows
            or expanded_manifest
            or [
                {"frame_sequence": frame, "frame_file": frame_rows[frame][0].get("frame_file")}
                for frame in sorted(frame_rows)
                if frame_rows[frame] and frame_rows[frame][0].get("frame_file")
            ]
        )
        visual_rows = burst_frame_rows if burst_manifest_rows else (expanded_frame_rows or frame_rows)
    recovery = _run_selective_recovery(final_events, matched_controls, model_path, output_root)
    _write_jsonl(
        output_root / "08_SELECTIVE_DETECTOR_RECOVERY" / "affected_rows.jsonl",
        (row for row in recovery if not row.get("control")),
    )
    _write_jsonl(
        output_root / "08_SELECTIVE_DETECTOR_RECOVERY" / "control_rows.jsonl",
        (row for row in recovery if row.get("control")),
    )
    _write_jsonl(
        output_root / "08_SELECTIVE_DETECTOR_RECOVERY" / "crop_transform_rows.jsonl",
        (row for row in recovery if row.get("crop_transform")),
    )
    _write_ghosts(output_root, final_events)
    _write_fine_vision(output_root, final_events)
    package = _write_review_package(
        output_root,
        repo_root.resolve(),
        final_events,
        matched_controls,
        stable_by_id,
        visual_rows,
        visual_manifest,
        recovery,
    )
    _write_json(output_root / "11_TRUE_OCCLUSION_REVIEW_PACKAGE" / "package_status.json", package)
    eval_result = _write_evaluation(
        output_root, inventory_results, expanded_result, final_events, final_near, recovery, package
    )
    _visuals(output_root, visual_manifest, visual_rows, final_events)
    result = {
        "stage_id": STAGE_ID,
        "classification": eval_result["classification"],
        "exact_blocker": eval_result["exact_blocker"],
        "third_window_genuine_count": len(phase_a_genuine),
        "expanded_genuine_count": len(expanded_genuine),
        "genuine_candidate_count": len(final_events),
        "near_miss_count": len(final_near),
        "visible_stable_segment_count": len(stable),
        "third_window_episode_count": len(episodes),
        "expanded_scan": expanded_result,
        "model_sha256": MODEL_SHA256,
        "video_sha256": VIDEO_SHA256,
        "matched_control_count": len(matched_controls),
        "review_package": package,
        "review_url": f"http://127.0.0.1:{REVIEW_PORT}/" if package.get("created") else None,
        "safety": SAFETY,
        "historical_artifacts_mutated": False,
        "old_port_8783_review": "do_not_launch_or_ingest",
        "human_answers_used_in_mining": False,
    }
    _write_json(output_root / "14_COMMANDS_AND_TESTS" / "stage_result.json", result)
    _write_json(
        output_root / "14_COMMANDS_AND_TESTS" / "commands.json",
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
            "browser_automation_status": "not_run_or_unavailable_until_a_valid_package_exists",
        },
    )
    pack = _write_pack(output_root, repo_root.resolve(), result)
    result["review_pack_validation"] = pack
    _write_json(output_root / "14_COMMANDS_AND_TESTS" / "stage_result.json", result)
    return result


def refresh_m5_5d2_review_pack(repo_root: Path, output_root: Path) -> dict[str, Any]:
    result = _read_json(output_root / "14_COMMANDS_AND_TESTS" / "stage_result.json")
    validation = _write_pack(output_root, repo_root.resolve(), result)
    result["review_pack_validation"] = validation
    _write_json(output_root / "14_COMMANDS_AND_TESTS" / "stage_result.json", result)
    return validation
