"""Machine-only mining helpers for fresh short sports-MOT challenge sequences.

These helpers propose temporary A/B strands for human annotation.  They never
produce gold labels, persistent identity, player slots, or football metrics.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from football_intelligence.review_chassis.hashing import stable_hash


CHALLENGE_STRATA = (
    "sharp_direction_or_acceleration",
    "same_team_nearby_distractors",
    "cross_team_or_parallel_motion_distractors",
    "large_scale_or_depth_change",
    "low_confidence_or_partial_observation",
    "panorama_crop_handoff",
    "fast_diagonal_or_long_displacement",
    "crossing_without_overlap_or_dense_local_motion",
)


def bbox_height(row: Mapping[str, Any]) -> float:
    box = row["bbox"]
    return max(1.0, float(box["y2"]) - float(box["y1"]))


def bbox_iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    a, b = left["bbox"], right["bbox"]
    x1, y1 = max(float(a["x1"]), float(b["x1"])), max(float(a["y1"]), float(b["y1"]))
    x2, y2 = min(float(a["x2"]), float(b["x2"])), min(float(a["y2"]), float(b["y2"]))
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(a["x2"]) - float(a["x1"])) * max(0.0, float(a["y2"]) - float(a["y1"]))
    right_area = max(0.0, float(b["x2"]) - float(b["x1"])) * max(0.0, float(b["y2"]) - float(b["y1"]))
    return overlap / max(1e-9, left_area + right_area - overlap)


def footpoint(row: Mapping[str, Any]) -> tuple[float, float]:
    box = row["bbox"]
    return ((float(box["x1"]) + float(box["x2"])) / 2.0, float(box["y2"]))


def colour_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    a = left.get("torso_mean_rgb") or [0.0, 0.0, 0.0]
    b = right.get("torso_mean_rgb") or [0.0, 0.0, 0.0]
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b, strict=True))) / 441.673


def _assignment_cost(
    history: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any], *, direction: int
) -> float | None:
    previous = history[-1]
    prior_point = footpoint(previous)
    velocity = (0.0, 0.0)
    if len(history) >= 2:
        older_point = footpoint(history[-2])
        velocity = (prior_point[0] - older_point[0], prior_point[1] - older_point[1])
    prediction = (prior_point[0] + velocity[0], prior_point[1] + velocity[1])
    distance = math.dist(prediction, footpoint(candidate))
    height = max(bbox_height(previous), bbox_height(candidate))
    hard_limit = max(24.0, 2.25 * height)
    ratio = max(bbox_height(previous), bbox_height(candidate)) / min(bbox_height(previous), bbox_height(candidate))
    if distance > hard_limit or ratio > 1.7 or candidate.get("pitch_zone") == "OFF_PITCH_STAFF_OR_SPECTATOR":
        return None
    motion = distance / hard_limit
    scale = abs(math.log(ratio))
    confidence = 1.0 - float(candidate.get("confidence", 0.0))
    # Colour is intentionally only a small local tie-breaker.
    colour = colour_distance(previous, candidate)
    return motion + 0.22 * scale + 0.04 * confidence + 0.025 * colour + (0.0 if direction else 0.0)


def _extend_joint(
    *,
    histories: dict[str, list[dict[str, Any]]],
    observations: Sequence[Mapping[str, Any]],
    direction: int,
) -> tuple[dict[str, dict[str, Any]] | None, dict[str, Any]]:
    eligible = [dict(row) for row in observations if row.get("pitch_zone") != "OFF_PITCH_STAFF_OR_SPECTATOR"]
    ranked: list[tuple[float, dict[str, dict[str, Any]]]] = []
    for left, right in itertools.permutations(eligible, 2):
        if left["observation_id"] == right["observation_id"]:
            continue
        left_cost = _assignment_cost(histories["A"], left, direction=direction)
        right_cost = _assignment_cost(histories["B"], right, direction=direction)
        if left_cost is None or right_cost is None:
            continue
        if bbox_iou(left, right) > 0.12:
            continue
        ranked.append((left_cost + right_cost, {"A": left, "B": right}))
    ranked.sort(key=lambda item: (item[0], item[1]["A"]["observation_id"], item[1]["B"]["observation_id"]))
    if not ranked:
        return None, {"candidate_count": 0, "margin": None, "ambiguous": True}
    best_cost, best = ranked[0]
    second_cost = ranked[1][0] if len(ranked) > 1 else None
    margin = second_cost - best_cost if second_cost is not None else None
    return best, {
        "candidate_count": len(ranked),
        "best_cost": round(best_cost, 6),
        "second_cost": round(second_cost, 6) if second_cost is not None else None,
        "margin": round(margin, 6) if margin is not None else None,
        "ambiguous": margin is not None and margin < 0.08,
    }


def build_joint_proposal(
    observations_by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    frames: Sequence[int],
    seed_a_id: str,
    seed_b_id: str,
) -> dict[str, Any]:
    """Build a bounded two-strand proposal with forward/backward consistency."""

    ordered_frames = [int(frame) for frame in frames]
    seed_index = len(ordered_frames) // 2
    seed_frame = ordered_frames[seed_index]
    seed_rows = {str(row["observation_id"]): dict(row) for row in observations_by_frame[seed_frame]}
    if seed_a_id not in seed_rows or seed_b_id not in seed_rows or seed_a_id == seed_b_id:
        raise ValueError("proposal requires two distinct seed observations")

    states: dict[int, dict[str, dict[str, Any]]] = {seed_frame: {"A": seed_rows[seed_a_id], "B": seed_rows[seed_b_id]}}
    diagnostics: dict[int, dict[str, Any]] = {
        seed_frame: {"candidate_count": len(seed_rows), "margin": None, "ambiguous": False}
    }
    for direction in (1, -1):
        histories = {"A": [seed_rows[seed_a_id]], "B": [seed_rows[seed_b_id]]}
        indices = (
            range(seed_index + direction, len(ordered_frames), direction)
            if direction > 0
            else range(seed_index - 1, -1, -1)
        )
        for index in indices:
            frame = ordered_frames[index]
            selected, audit = _extend_joint(
                histories=histories,
                observations=observations_by_frame.get(frame, []),
                direction=direction,
            )
            diagnostics[frame] = audit
            if selected is None:
                return {
                    "passed": False,
                    "failure_reason": "PROPOSAL_OBSERVATION_GAP",
                    "first_failure_frame": frame,
                    "states": states,
                    "diagnostics": diagnostics,
                }
            states[frame] = selected
            for strand in ("A", "B"):
                histories[strand].append(selected[strand])

    ordered_states = [states[frame] for frame in ordered_frames]
    impossible_jumps = 0
    maximum_iou = 0.0
    minimum_separation = float("inf")
    for index, state in enumerate(ordered_states):
        maximum_iou = max(maximum_iou, bbox_iou(state["A"], state["B"]))
        minimum_separation = min(minimum_separation, math.dist(footpoint(state["A"]), footpoint(state["B"])))
        if index:
            for strand in ("A", "B"):
                height = max(bbox_height(ordered_states[index - 1][strand]), bbox_height(state[strand]))
                impossible_jumps += int(
                    math.dist(footpoint(ordered_states[index - 1][strand]), footpoint(state[strand]))
                    > max(24.0, 2.25 * height)
                )
    return {
        "passed": impossible_jumps == 0 and maximum_iou <= 0.12,
        "failure_reason": None if impossible_jumps == 0 and maximum_iou <= 0.12 else "GEOMETRY_GATE_FAILED",
        "states": states,
        "diagnostics": diagnostics,
        "impossible_jump_count": impossible_jumps,
        "maximum_pair_iou": round(maximum_iou, 6),
        "minimum_pair_separation_pixels": round(minimum_separation, 6),
        "machine_uncertain_frame_count": sum(bool(row.get("ambiguous")) for row in diagnostics.values()),
    }


def choose_seed_pair(
    observations_by_frame: Mapping[int, Sequence[Mapping[str, Any]]], *, frames: Sequence[int]
) -> dict[str, Any]:
    """Choose a difficult, fully preflightable pair without assigning truth."""

    seed_frame = int(frames[len(frames) // 2])
    seeds = [
        dict(row)
        for row in observations_by_frame.get(seed_frame, [])
        if row.get("pitch_zone") == "INSIDE_PLAYABLE_PITCH" and 14 <= bbox_height(row) <= 130
    ]
    pair_rows: list[tuple[float, dict[str, Any]]] = []
    for left, right in itertools.combinations(seeds, 2):
        if bbox_iou(left, right) > 0.05:
            continue
        mean_height = (bbox_height(left) + bbox_height(right)) / 2.0
        separation = math.dist(footpoint(left), footpoint(right)) / max(1.0, mean_height)
        if not 1.25 <= separation <= 13.0:
            continue
        proposal = build_joint_proposal(
            observations_by_frame,
            frames=frames,
            seed_a_id=str(left["observation_id"]),
            seed_b_id=str(right["observation_id"]),
        )
        if not proposal["passed"]:
            continue
        local_density = sum(
            math.dist(footpoint(left), footpoint(other)) <= 7.0 * mean_height
            for other in seeds
            if other["observation_id"] not in {left["observation_id"], right["observation_id"]}
        )
        difficulty = 1.0 / separation + 0.10 * local_density + 0.08 * proposal["machine_uncertain_frame_count"]
        difficulty += 0.15 * abs(colour_distance(left, right) - 0.35)
        pair_rows.append(
            (
                -difficulty,
                {
                    "seed_a_id": left["observation_id"],
                    "seed_b_id": right["observation_id"],
                    "seed_frame": seed_frame,
                    "seed_pair_separation_heights": round(separation, 6),
                    "seed_local_distractor_count": local_density,
                    "proposal": proposal,
                },
            )
        )
    if not pair_rows:
        return {"passed": False, "failure_reason": "NO_STABLE_TWO_SEED_PAIR"}
    pair_rows.sort(key=lambda item: (item[0], stable_hash(item[1]["seed_a_id"] + item[1]["seed_b_id"])))
    return {"passed": True, **pair_rows[0][1]}


def challenge_score_components(candidate: Mapping[str, Any]) -> dict[str, float]:
    """Derive interpretable difficulty components from proposal observations."""

    frames = [int(value) for value in candidate["frames"]]
    states = candidate["proposal"]["states"]
    strand_rows = {strand: [states[frame][strand] for frame in frames] for strand in ("A", "B")}
    displacements: dict[str, list[float]] = {}
    acceleration = 0.0
    for strand, rows in strand_rows.items():
        points = [footpoint(row) for row in rows]
        steps = [math.dist(a, b) for a, b in zip(points, points[1:])]
        displacements[strand] = steps
        acceleration = max(acceleration, *(abs(a - b) for a, b in zip(steps, steps[1:])))
    heights = [bbox_height(row) for rows in strand_rows.values() for row in rows]
    confidence = [float(row.get("confidence", 0.0)) for rows in strand_rows.values() for row in rows]
    colour = sum(colour_distance(states[frame]["A"], states[frame]["B"]) for frame in frames) / len(frames)
    minimum_pair_distance = min(
        math.dist(footpoint(states[frame]["A"]), footpoint(states[frame]["B"]))
        / max(1.0, (bbox_height(states[frame]["A"]) + bbox_height(states[frame]["B"])) / 2.0)
        for frame in frames
    )
    points = [footpoint(row) for rows in strand_rows.values() for row in rows]
    boundary = min(min(abs(point[0] - boundary) for boundary in (910.0, 1820.0)) for point in points)
    diagonal = max(sum(values) for values in displacements.values()) / max(1.0, sum(heights) / len(heights))
    return {
        "sharp_direction_or_acceleration": round(acceleration / max(1.0, sum(heights) / len(heights)), 6),
        "same_team_nearby_distractors": round((1.0 - colour) + 0.12 * candidate["seed_local_distractor_count"], 6),
        "cross_team_or_parallel_motion_distractors": round(colour + 0.08 * candidate["seed_local_distractor_count"], 6),
        "large_scale_or_depth_change": round(max(heights) / min(heights) - 1.0, 6),
        "low_confidence_or_partial_observation": round(1.0 - min(confidence), 6),
        "panorama_crop_handoff": round(1.0 / max(1.0, boundary), 6),
        "fast_diagonal_or_long_displacement": round(diagonal, 6),
        "crossing_without_overlap_or_dense_local_motion": round(
            1.0 / max(0.25, minimum_pair_distance) + 0.10 * candidate["seed_local_distractor_count"], 6
        ),
    }


def select_stratified_challenges(
    candidates: Sequence[Mapping[str, Any]], *, target: int = 32, per_stratum: int = 4
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cover each challenge stratum first, then fill by aggregate difficulty."""

    prepared = []
    for source in candidates:
        row = dict(source)
        row["challenge_score_components"] = challenge_score_components(row)
        prepared.append(row)
    selected: dict[str, dict[str, Any]] = {}
    for stratum in CHALLENGE_STRATA:
        ranked = sorted(
            prepared,
            key=lambda row: (-row["challenge_score_components"][stratum], row["candidate_key"]),
        )
        for row in ranked[:per_stratum]:
            selected[row["candidate_key"]] = row
    for row in sorted(
        prepared,
        key=lambda item: (-sum(item["challenge_score_components"].values()), item["candidate_key"]),
    ):
        if len(selected) >= min(target, len(prepared)):
            break
        selected[row["candidate_key"]] = row
    output = list(selected.values())
    output.sort(key=lambda row: row["event_cluster_id"])
    for row in output:
        row["challenge_tags"] = [
            stratum
            for stratum in CHALLENGE_STRATA
            if row
            in sorted(
                prepared,
                key=lambda item: (-item["challenge_score_components"][stratum], item["candidate_key"]),
            )[: max(per_stratum, 8)]
        ]
    coverage = Counter(tag for row in output for tag in row["challenge_tags"])
    return output, {
        "selected_count": len(output),
        "target_count": target,
        "minimum_count": 24,
        "stratum_counts": {stratum: coverage[stratum] for stratum in CHALLENGE_STRATA},
        "every_stratum_has_four": all(coverage[stratum] >= per_stratum for stratum in CHALLENGE_STRATA),
        "machine_selection_only": True,
        "human_labels_created": False,
    }


def event_cluster_leakage_audit(
    candidates: Sequence[Mapping[str, Any]], assignments: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_cluster: dict[str, set[str]] = defaultdict(set)
    by_candidate = {str(row["candidate_key"]): row for row in candidates}
    for assignment in assignments:
        by_cluster[str(assignment["event_cluster_id"])].add(str(assignment["hidden_split"]))
    frame_hashes = [str(value) for row in candidates for value in row.get("source_frame_hashes", []) if value]
    assignment_keys = [str(row["candidate_key"]) for row in assignments]
    split_counts = Counter(str(row["hidden_split"]) for row in assignments)
    violations = {
        "event_cluster_in_multiple_splits": sorted(key for key, values in by_cluster.items() if len(values) > 1),
        "duplicate_frame_hashes": len(frame_hashes) - len(set(frame_hashes)),
        "duplicate_candidate_assignments": len(assignment_keys) - len(set(assignment_keys)),
        "unknown_candidate_assignments": sorted(set(assignment_keys) - set(by_candidate)),
    }
    return {
        "schema_version": "football_intelligence.m5_5f1e.split_leakage_audit.v1",
        "split_counts": dict(split_counts),
        "violations": violations,
        "passed": not violations["event_cluster_in_multiple_splits"]
        and not violations["duplicate_frame_hashes"]
        and not violations["duplicate_candidate_assignments"]
        and not violations["unknown_candidate_assignments"],
    }


def estimate_annotation_time(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sequence_count = len(candidates)
    uncertain = sum(int(row["proposal"].get("machine_uncertain_frame_count", 0)) for row in candidates)
    seed_actions = sequence_count
    stable_runs = sequence_count
    expected_corrections = max(4, round(0.12 * uncertain + 0.12 * sequence_count))
    predicted_clicks = seed_actions + stable_runs + expected_corrections * 2 + sequence_count
    predicted_minutes = 0.55 * sequence_count + 0.05 * uncertain + 0.15 * expected_corrections
    return {
        "schema_version": "football_intelligence.m5_5f1e.annotation_time_estimate.v1",
        "sequence_count": sequence_count,
        "seed_confirmation_actions": seed_actions,
        "stable_run_actions": stable_runs,
        "machine_uncertain_frames": uncertain,
        "expected_manual_corrections": expected_corrections,
        "predicted_clicks": predicted_clicks,
        "predicted_active_minutes": round(predicted_minutes, 2),
        "target_active_minutes": [30, 40],
        "warning_threshold_minutes": 45,
        "within_budget": predicted_minutes <= 45,
        "notes_optional": True,
    }
