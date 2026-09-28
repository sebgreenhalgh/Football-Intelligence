"""Candidate-blind, additive temporal Gold contracts and immutable events.

TRACKLET is anonymous confirmed continuity over existing DETECTION instances.
BALL and MATCH_STATE are independently reviewed. None of these contracts
implies roster identity, calibrated pitch coordinates or model predictions.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

from .corpus import GoldError, canonical, digest, require


SELECTION_SCHEMA = "football_intelligence.gold.temporal_sequence_selection.v1"
SCHEMAS = {
    "TRACKLET": "football_intelligence.gold.tracklet_sequence.v1",
    "BALL": "football_intelligence.gold.ball_sequence.v1",
    "MATCH_STATE": "football_intelligence.gold.match_state_sequence.v1",
}
ACK_SCHEMA = "football_intelligence.gold.temporal_acknowledgement.v1"
SEQUENCE_RECEIPT_SCHEMA = "football_intelligence.gold.temporal_sequence_completion.v1"
BALL_STATES = ("VISIBLE", "OCCLUDED", "OFF_SCREEN", "UNCERTAIN")
MATCH_STATES = ("OPEN_PLAY", "THROW_IN", "FREE_KICK", "CORNER", "GOAL_KICK", "KICK_OFF", "STOPPAGE", "OTHER", "UNKNOWN")
ASSERTIONS = {
    "TRACKLET": "I reviewed every MATCH_RELEVANT visible person across this sequence and linked only continuity I can support from the images.",
    "BALL": "I reviewed the ball visibility/location state in every annotation frame and did not infer hidden positions.",
    "MATCH_STATE": "I reviewed the football match state for every annotation frame using the available visual context.",
    "SEQUENCE": "All required Gold layers for this sequence have been reviewed and finalized.",
}
HASH = re.compile(r"[0-9a-f]{64}\Z")


def _hash(value: object) -> bool:
    return isinstance(value, str) and HASH.fullmatch(value) is not None


def validate_sequence(selection: Mapping, sequence: Mapping) -> list[dict]:
    require(selection.get("schema_version") == SELECTION_SCHEMA, "Unknown temporal selection schema")
    require(selection.get("candidate_data_used") is False, "Candidate-exposed selection is forbidden")
    require(sequence in selection.get("sequences", []), "Sequence not in frozen selection")
    frames = sequence.get("annotation_frames")
    require(isinstance(frames, list) and len(frames) == 9, "A sequence requires exactly nine frames")
    require([frame.get("order") for frame in frames] == list(range(1, 10)), "Frame order is not exact")
    indices = [frame.get("source_frame_number_zero_based") for frame in frames]
    require(all(type(index) is int for index in indices) and indices == sorted(set(indices)), "Source frame indices must be increasing and unique")
    hashes = [frame.get("source_rgb_sha256") for frame in frames]
    require(all(_hash(value) for value in hashes), "Invalid source RGB hash")
    ids = [frame.get("gold_frame_id") for frame in frames]
    require(ids == ["gf-" + value for value in hashes], "Gold frame ID/source hash mismatch")
    require(frames[4]["gold_frame_id"] == sequence.get("anchor_gold_frame_id"), "Anchor must be central frame")
    require(frames[4].get("existing_canonical_detection_event_sha256") == sequence.get("anchor_detection_event_sha256"), "Canonical anchor event mismatch")
    require(sum(frame.get("detection_read_only") is True for frame in frames) >= 1, "No canonical anchor reused")
    require(_hash(sequence.get("source_video_sha256")), "Missing source video hash")
    identity = {
        "source_video_sha256": sequence["source_video_sha256"],
        "source_frame_numbers_zero_based": indices,
        "source_rgb_sha256": hashes,
        "sampling_rule": "nearest source frame to fixed offsets -1.00 through +1.00 seconds; Python round ties-to-even",
    }
    require(sequence.get("gold_sequence_id") == "gs-" + digest(canonical(identity)), "Sequence ID is not deterministic")
    require(sequence.get("role") in {"PRIMARY", "RESERVE"}, "Unknown sequence role")
    return frames


def validate_detection_bindings(sequence: Mapping, detection: Mapping[str, Mapping]) -> None:
    for frame in sequence["annotation_frames"]:
        frame_id = frame["gold_frame_id"]
        require(frame_id in detection, f"Missing authoritative DETECTION event: {frame_id}")
        item = detection[frame_id]
        require(_hash(item.get("annotation_event_sha256")), "Invalid detection event hash")
        require(item.get("source_frame_sha256") == frame["source_rgb_sha256"], "Detection source hash mismatch")
        if frame.get("detection_read_only"):
            require(item["annotation_event_sha256"] == frame["existing_canonical_detection_event_sha256"], "Canonical detection anchor changed")
        people = item.get("people")
        require(isinstance(people, list) and len({p.get("instance_id") for p in people}) == len(people), "Invalid detection people")


def validate_tracklet(sequence: Mapping, annotation: Mapping, detection: Mapping[str, Mapping]) -> dict:
    validate_detection_bindings(sequence, detection)
    ordered = [frame["gold_frame_id"] for frame in sequence["annotation_frames"]]
    positions = {frame_id: index for index, frame_id in enumerate(ordered)}
    tracklets = annotation.get("confirmed_tracklets")
    uncertain = annotation.get("uncertain_continuity_relations", [])
    require(isinstance(tracklets, list) and isinstance(uncertain, list), "Tracklets/uncertain relations must be arrays")
    ids = set()
    assigned = set()
    for tracklet in tracklets:
        tracklet_id = tracklet.get("tracklet_id")
        require(isinstance(tracklet_id, str) and re.fullmatch(r"tracklet-[0-9]{3,}", tracklet_id), "Invalid sequence-local tracklet ID")
        require(tracklet_id not in ids, "Duplicate tracklet ID")
        ids.add(tracklet_id)
        members = tracklet.get("members")
        require(isinstance(members, list) and members, "Confirmed tracklet must have members")
        prior = -1
        for member in members:
            frame_id = member.get("gold_frame_id")
            require(frame_id in positions and positions[frame_id] > prior, "Tracklet members must follow sequence order")
            prior = positions[frame_id]
            source = detection[frame_id]
            require(member.get("detection_event_sha256") == source["annotation_event_sha256"], "Tracklet detection reference changed")
            person = next((p for p in source["people"] if p["instance_id"] == member.get("instance_id")), None)
            require(person is not None and person["relevance"] == "MATCH_RELEVANT", "Only confirmed match-relevant detections may be linked")
            ref = (frame_id, member["instance_id"])
            require(ref not in assigned, "Detection assigned to multiple confirmed tracklets")
            assigned.add(ref)
            require(set(member) == {"gold_frame_id", "detection_event_sha256", "instance_id"}, "Tracklet must not duplicate geometry")
    for relation in uncertain:
        require(relation.get("relation") == "POSSIBLY_SAME_PERSON", "Unknown uncertain relation")
        require(relation.get("from_tracklet_id") in ids and relation.get("to_tracklet_id") in ids, "Uncertain relation references absent tracklet")
        require(relation["from_tracklet_id"] != relation["to_tracklet_id"], "Self relation is invalid")
    require(annotation.get("primary_evaluation_excludes_uncertain_continuity") is True, "Uncertain continuity must be excluded from primary evaluation")
    require(annotation.get("player_identity_implemented") is False, "Tracklet is not roster identity")
    return {"confirmed_tracklets": tracklets, "uncertain_continuity_relations": uncertain, "primary_evaluation_excludes_uncertain_continuity": True, "player_identity_implemented": False}


def validate_ball(sequence: Mapping, annotation: Mapping) -> dict:
    frames = sequence["annotation_frames"]
    rows = annotation.get("frames")
    require(isinstance(rows, list) and len(rows) == 9, "BALL requires nine reviewed frame states")
    width, height = sequence["source_width"], sequence["source_height"]
    clean = []
    for source, row in zip(frames, rows, strict=True):
        require(row.get("gold_frame_id") == source["gold_frame_id"] and row.get("human_reviewed") is True, "BALL frame binding/review mismatch")
        state = row.get("visibility")
        require(state in BALL_STATES, "Unknown ball visibility state")
        point = row.get("point")
        if state == "VISIBLE":
            require(isinstance(point, dict) and set(point) == {"x", "y"}, "Visible ball requires exactly one source-coordinate point")
            require(all(type(point[k]) in (float, int) and math.isfinite(point[k]) for k in ("x", "y")), "Ball point must be finite")
            require(0 <= point["x"] < width and 0 <= point["y"] < height, "Ball point outside source frame")
        else:
            require(point is None, "Non-visible ball must not carry an inferred point")
        clean.append({"gold_frame_id": source["gold_frame_id"], "visibility": state, "point": point, "human_reviewed": True})
    return {"frames": clean}


def validate_match_state(sequence: Mapping, annotation: Mapping) -> dict:
    frames = sequence["annotation_frames"]
    ids = [frame["gold_frame_id"] for frame in frames]
    if "intervals" in annotation:
        require("frames" not in annotation, "Cannot mix interval and frame encodings")
        compiled = [None] * 9
        for interval in annotation["intervals"]:
            start, end = interval.get("start_order"), interval.get("end_order")
            state = interval.get("state")
            require(type(start) is int and type(end) is int and 1 <= start <= end <= 9, "Invalid match-state interval")
            require(state in MATCH_STATES and interval.get("human_reviewed") is True, "Unreviewed/unknown match state")
            for index in range(start - 1, end):
                require(compiled[index] is None, "Overlapping match-state intervals")
                compiled[index] = state
        require(all(state is not None for state in compiled), "Match-state interval coverage incomplete")
        rows = [{"gold_frame_id": frame_id, "state": state, "human_reviewed": True} for frame_id, state in zip(ids, compiled, strict=True)]
    else:
        rows = annotation.get("frames")
        require(isinstance(rows, list) and len(rows) == 9, "MATCH_STATE requires nine reviewed frame states")
        require([row.get("gold_frame_id") for row in rows] == ids, "Match-state frame order changed")
        require(all(row.get("state") in MATCH_STATES and row.get("human_reviewed") is True for row in rows), "Invalid/unreviewed match state")
    return {"ontology_version": "football_intelligence.gold.match_state_ontology.v1", "frames": rows}


def build_layer_event(layer: str, sequence: Mapping, annotation: Mapping, *, selection_sha256: str, reviewer_sha256: str, schema_sha256: str, completion_assertion: str, final_revision: int, detection: Mapping[str, Mapping] | None = None, supersedes_event_sha256: str | None = None) -> tuple[dict, dict]:
    require(layer in SCHEMAS, "Unknown temporal layer")
    require(completion_assertion == ASSERTIONS[layer], "Exact human completion assertion required")
    require(all(_hash(value) for value in (selection_sha256, reviewer_sha256, schema_sha256)), "Missing immutable source/reviewer/schema binding")
    require(type(final_revision) is int and final_revision > 0, "Invalid final revision")
    require(supersedes_event_sha256 is None or _hash(supersedes_event_sha256), "Invalid predecessor hash")
    if layer == "TRACKLET":
        require(detection is not None, "TRACKLET requires authoritative detection bindings")
        normalized = validate_tracklet(sequence, annotation, detection)
    elif layer == "BALL":
        normalized = validate_ball(sequence, annotation)
    else:
        normalized = validate_match_state(sequence, annotation)
    payload = {
        "schema_version": SCHEMAS[layer], "layer": layer, "gold_sequence_id": sequence["gold_sequence_id"],
        "ordered_gold_frame_ids": [frame["gold_frame_id"] for frame in sequence["annotation_frames"]],
        "annotation": normalized, "completion_assertion": completion_assertion,
        "selection_sha256": selection_sha256, "source_video_sha256": sequence["source_video_sha256"],
        "reviewer_sha256": reviewer_sha256, "schema_sha256": schema_sha256,
        "final_revision": final_revision, "supersedes_event_sha256": supersedes_event_sha256,
        "candidate_data_used": False, "immutable": True, "production_ready": False,
    }
    logical = digest(canonical(payload))
    event = {**payload, "event_id": f"{layer.lower()}-{logical[:24]}", "event_sha256": logical}
    ack_payload = {"schema_version": ACK_SCHEMA, "layer": layer, "gold_sequence_id": sequence["gold_sequence_id"], "event_id": event["event_id"], "event_sha256": logical, "status": "IMMUTABLY_FINALIZED"}
    ack_hash = digest(canonical(ack_payload))
    ack = {**ack_payload, "acknowledgement_id": f"ack-{ack_hash[:24]}", "acknowledgement_sha256": ack_hash}
    return event, ack


def validate_event_pair(event: Mapping, ack: Mapping) -> None:
    layer = event.get("layer")
    require(layer in SCHEMAS and event.get("schema_version") == SCHEMAS[layer], "Unknown temporal event schema")
    require(event.get("immutable") is True and event.get("candidate_data_used") is False and event.get("production_ready") is False, "Temporal event safety failure")
    payload = {k: v for k, v in event.items() if k not in {"event_id", "event_sha256"}}
    logical = digest(canonical(payload))
    require(event.get("event_sha256") == logical and event.get("event_id") == f"{layer.lower()}-{logical[:24]}", "Temporal event hash mismatch")
    ack_payload = {k: v for k, v in ack.items() if k not in {"acknowledgement_id", "acknowledgement_sha256"}}
    ack_hash = digest(canonical(ack_payload))
    require(ack.get("acknowledgement_sha256") == ack_hash and ack.get("acknowledgement_id") == f"ack-{ack_hash[:24]}", "Temporal acknowledgement hash mismatch")
    require(ack.get("schema_version") == ACK_SCHEMA and ack.get("status") == "IMMUTABLY_FINALIZED", "Unacknowledged temporal event")
    require(all(ack.get(key) == event.get(key) for key in ("layer", "gold_sequence_id", "event_id", "event_sha256")), "Temporal acknowledgement/event mismatch")


def completion_receipt(sequence: Mapping, detection: Mapping[str, Mapping], layer_pairs: Mapping[str, tuple[Mapping, Mapping]], *, assertion: str) -> dict:
    require(assertion == ASSERTIONS["SEQUENCE"], "Exact sequence completion assertion required")
    validate_detection_bindings(sequence, detection)
    require(set(layer_pairs) == set(SCHEMAS), "All three temporal layers must finalize independently")
    for layer, (event, ack) in layer_pairs.items():
        validate_event_pair(event, ack)
        require(event["layer"] == layer and event["gold_sequence_id"] == sequence["gold_sequence_id"], "Cross-layer/sequence event mix")
    payload = {
        "schema_version": SEQUENCE_RECEIPT_SCHEMA,
        "gold_sequence_id": sequence["gold_sequence_id"],
        "detection_event_sha256_by_gold_frame_id": {frame["gold_frame_id"]: detection[frame["gold_frame_id"]]["annotation_event_sha256"] for frame in sequence["annotation_frames"]},
        "layer_events": {layer: {"event_file_sha256": digest(canonical(pair[0])), "acknowledgement_file_sha256": digest(canonical(pair[1])), "event_payload_sha256": pair[0]["event_sha256"]} for layer, pair in sorted(layer_pairs.items())},
        "completion_assertion": assertion,
        "candidate_data_used": False,
        "immutable": True,
        "production_ready": False,
    }
    logical = digest(canonical(payload))
    return {**payload, "receipt_id": f"sequence-complete-{logical[:24]}", "receipt_sha256": logical}
