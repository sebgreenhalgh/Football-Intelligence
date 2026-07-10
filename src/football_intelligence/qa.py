from pathlib import Path
from typing import Iterable

import cv2


BOX_COLORS = {
    "player_candidate": (30, 220, 60),
    "ball_candidate": (0, 165, 255),
}


def draw_detection_overlays(image, detections: Iterable[dict]) -> None:
    for detection in detections:
        object_type = detection["object_type"]
        color = BOX_COLORS.get(object_type, (255, 255, 255))
        x1 = int(round(detection["x1"]))
        y1 = int(round(detection["y1"]))
        x2 = int(round(detection["x2"]))
        y2 = int(round(detection["y2"]))
        confidence = float(detection["confidence"])
        label = f"{object_type} {confidence:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_y = max(16, y1 - 6)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(
            image,
            (x1, label_y - label_size[1] - 4),
            (x1 + label_size[0] + 6, label_y + 3),
            color,
            thickness=-1,
        )
        cv2.putText(
            image,
            label,
            (x1 + 3, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

