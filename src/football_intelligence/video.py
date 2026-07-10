import math
from dataclasses import dataclass
from pathlib import Path

import cv2

from football_intelligence.paths import require_file


@dataclass(frozen=True)
class VideoMetadata:
    file_path: Path
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float


def read_video_metadata(path: Path) -> VideoMetadata:
    require_file(path, "Video file")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if fps <= 0:
            raise RuntimeError(f"OpenCV reported invalid fps={fps} for video: {path}")

        return VideoMetadata(
            file_path=path,
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            duration_seconds=frame_count / fps,
        )
    finally:
        cap.release()


def sample_timestamps(duration_seconds: float, sample_rate_fps: float) -> list[float]:
    if sample_rate_fps <= 0:
        raise ValueError(f"sample_rate_fps must be positive, got {sample_rate_fps}")

    sample_count = max(1, int(math.floor(duration_seconds * sample_rate_fps)))
    return [index / sample_rate_fps for index in range(sample_count)]


def read_frame_at(cap: cv2.VideoCapture, frame_index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame index {frame_index}")
    return frame

