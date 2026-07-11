import math
from typing import Any


def _distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    dx = float(left["center_x"]) - float(right["center_x"])
    dy = float(left["center_y"]) - float(right["center_y"])
    return math.hypot(dx, dy)


def build_nearest_neighbor_tracks(
    frames: list[dict[str, Any]],
    *,
    track_prefix: str,
    distance_threshold_pixels: float,
    max_missing_frames: int,
) -> list[dict[str, Any]]:
    active: dict[str, dict[str, Any]] = {}
    finished: list[dict[str, Any]] = []
    next_track_number = 1

    for frame_order, frame in enumerate(frames):
        detections = frame["detections"]
        candidates: list[tuple[float, str, int]] = []

        for track_id, track in active.items():
            if frame_order - int(track["last_frame_order"]) > max_missing_frames + 1:
                continue
            for detection_index, detection in enumerate(detections):
                distance = _distance(track["last_detection"], detection)
                if distance <= distance_threshold_pixels:
                    candidates.append((distance, track_id, detection_index))

        candidates.sort(key=lambda item: item[0])
        matched_tracks: set[str] = set()
        matched_detections: set[int] = set()

        for _, track_id, detection_index in candidates:
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            detection = detections[detection_index]
            track = active[track_id]
            track["observations"].append(detection)
            track["last_detection"] = detection
            track["last_frame_order"] = frame_order
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            track_id = f"{track_prefix}_t{next_track_number:04d}"
            next_track_number += 1
            active[track_id] = {
                "track_id": track_id,
                "object_type": detection["object_type"],
                "observations": [detection],
                "last_detection": detection,
                "last_frame_order": frame_order,
            }

        stale_track_ids = [
            track_id
            for track_id, track in active.items()
            if frame_order - int(track["last_frame_order"]) > max_missing_frames
        ]
        for track_id in stale_track_ids:
            finished.append(active.pop(track_id))

    finished.extend(active.values())

    for track in finished:
        observations = track["observations"]
        track["start_time_seconds"] = float(observations[0]["timestamp_seconds"])
        track["end_time_seconds"] = float(observations[-1]["timestamp_seconds"])
        track["num_observations"] = len(observations)
        track.pop("last_detection", None)
        track.pop("last_frame_order", None)

    finished.sort(key=lambda item: (item["start_time_seconds"], item["track_id"]))
    return finished
