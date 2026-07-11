from pathlib import Path
from typing import Any


COCO_PERSON_CLASS_ID = 0
COCO_SPORTS_BALL_CLASS_ID = 32
COCO_CLASS_IDS_TO_KEEP = [COCO_PERSON_CLASS_ID, COCO_SPORTS_BALL_CLASS_ID]
COCO_OBJECT_TYPES = {
    COCO_PERSON_CLASS_ID: "player_candidate",
    COCO_SPORTS_BALL_CLASS_ID: "ball_candidate",
}


def object_type_for_class(class_id: int, class_name: str) -> str:
    if class_id in COCO_OBJECT_TYPES:
        return COCO_OBJECT_TYPES[class_id]
    if class_name == "person":
        return "player_candidate"
    if class_name == "sports ball":
        return "ball_candidate"
    return "unknown"


def detection_geometry(x1: float, y1: float, x2: float, y2: float) -> dict[str, float]:
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return {
        "center_x": x1 + width / 2.0,
        "center_y": y1 + height / 2.0,
        "width": width,
        "height": height,
        "area": width * height,
    }

def build_detection_record(
    *,
    detection_id: str,
    frame_id: str,
    frame_file: Path,
    timestamp_seconds: float,
    class_id: int,
    class_name: str,
    confidence: float,
    xyxy: list[float],
) -> dict[str, Any]:
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    geometry = detection_geometry(x1, y1, x2, y2)
    return {
        "detection_id": detection_id,
        "frame_id": frame_id,
        "frame_file": str(frame_file.resolve()),
        "timestamp_seconds": float(timestamp_seconds),
        "object_type": object_type_for_class(class_id, class_name),
        "class_id": int(class_id),
        "class_name": class_name,
        "confidence": float(confidence),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        **geometry,
    }
