from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.core.fingerprints import semantic_hash, sha256_file

SCHEMA_VERSION = "m5.blind_window.selection.v1"
DEFAULT_SEED_STRING = "128058|M5.3|blind-second-window-v1"
DEFAULT_HISTORICAL_INTERVAL = (1882, 2062)
DEFAULT_EXCLUDED_INTERVAL = (1582, 2362)
EXPECTED_SOURCE_SHA256 = "8db0efdc045978d67572c6764681a76350e8da75a9f5fa7bc9307f3b9f21d989"


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    codec: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_seconds": self.duration_seconds,
            "codec": self.codec,
        }


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def git_status(repo_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def git_dirty(repo_root: Path) -> bool | None:
    status = git_status(repo_root)
    return None if status is None else bool(status)


def seed_hash(seed_string: str = DEFAULT_SEED_STRING) -> str:
    import hashlib

    return hashlib.sha256(seed_string.encode("utf-8")).hexdigest()


def probe_video_metadata(source_video: Path) -> VideoMetadata:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency is present in the project env
        raise RuntimeError("opencv-python is required to probe video metadata") from exc

    cap = cv2.VideoCapture(str(source_video))
    try:
        if not cap.isOpened():
            raise ValueError(f"source video could not be opened: {source_video}")
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((codec_int >> 8 * i) & 0xFF) for i in range(4)).strip("\x00") or None
        if fps <= 0 or frame_count <= 0:
            raise ValueError(f"invalid source metadata fps={fps} frame_count={frame_count}")
        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=frame_count / fps,
            codec=codec,
        )
    finally:
        cap.release()


def candidate_intervals(
    metadata: VideoMetadata,
    *,
    earliest_start_seconds: int = 300,
    latest_end_buffer_seconds: int = 300,
    candidate_spacing_seconds: int = 120,
    duration_seconds: int = 60,
    excluded_interval: tuple[int, int] = DEFAULT_EXCLUDED_INTERVAL,
) -> list[dict[str, Any]]:
    latest_permitted_end = metadata.duration_seconds - latest_end_buffer_seconds
    candidates: list[dict[str, Any]] = []
    all_index = 0
    start = earliest_start_seconds
    while start + duration_seconds <= latest_permitted_end + 1e-9:
        end = start + duration_seconds
        overlaps_excluded = start < excluded_interval[1] and end > excluded_interval[0]
        row = {
            "all_candidate_index": all_index,
            "start_seconds": start,
            "end_seconds": end,
            "duration_seconds": duration_seconds,
            "excluded": overlaps_excluded,
            "exclusion_reason": "historical_interval_plus_300s_buffer" if overlaps_excluded else None,
        }
        if not overlaps_excluded:
            row["eligible_candidate_index"] = len(candidates)
            candidates.append(row)
        all_index += 1
        start += candidate_spacing_seconds
    if not candidates:
        raise ValueError("candidate generation produced no eligible blind windows")
    return candidates


def deterministic_selection(candidates: list[dict[str, Any]], seed_string: str = DEFAULT_SEED_STRING) -> dict[str, Any]:
    digest = seed_hash(seed_string)
    selected_index = int(digest, 16) % len(candidates)
    selected = dict(candidates[selected_index])
    return {
        "seed_string": seed_string,
        "seed_hash": digest,
        "chosen_candidate_index": selected_index,
        "selected_start_seconds": selected["start_seconds"],
        "selected_end_seconds": selected["end_seconds"],
        "duration_seconds": selected["duration_seconds"],
        "selected_candidate": selected,
    }


def build_selection_payload(
    *,
    source_video: Path,
    metadata: VideoMetadata,
    source_sha256: str,
    repo_root: Path,
    selected_at: str | None = None,
    seed_string: str = DEFAULT_SEED_STRING,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidates = candidate_intervals(metadata)
    selection = deterministic_selection(candidates, seed_string)
    generation_rules = {
        "earliest_permitted_start_seconds": 300,
        "latest_permitted_end_rule": "source_duration_seconds - 300",
        "candidate_starts_spaced_seconds": 120,
        "fixed_duration_seconds": 60,
        "historical_interval_seconds": list(DEFAULT_HISTORICAL_INTERVAL),
        "excluded_interval_seconds": list(DEFAULT_EXCLUDED_INTERVAL),
        "selection_does_not_inspect_frames": True,
    }
    candidate_payload = {
        "schema_version": SCHEMA_VERSION,
        "source_video": str(source_video),
        "source_metadata": metadata.as_dict(),
        "candidate_generation_rules": generation_rules,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    selection_payload = {
        "schema_version": SCHEMA_VERSION,
        "source_video": str(source_video),
        "source_video_sha256": source_sha256,
        "source_metadata": metadata.as_dict(),
        "candidate_generation_rules": generation_rules,
        "seed_string": seed_string,
        "seed_hash": selection["seed_hash"],
        "chosen_candidate_index": selection["chosen_candidate_index"],
        "selected_start_seconds": selection["selected_start_seconds"],
        "selected_end_seconds": selection["selected_end_seconds"],
        "duration_seconds": selection["duration_seconds"],
        "selected_candidate": selection["selected_candidate"],
        "code_commit": git_commit(repo_root),
        "dirty_state": git_dirty(repo_root),
        "selected_at": selected_at or utc_now(),
    }
    seal = {
        **selection_payload,
        "complete_candidate_list": candidates,
        "selection_sealed_without_frame_inspection": True,
    }
    seal["selection_seal_hash"] = semantic_hash(seal)
    return candidate_payload, selection_payload, seal


def seal_blind_window_selection(
    *,
    repo_root: Path,
    stage_root: Path,
    source_video: Path,
    expected_source_sha256: str = EXPECTED_SOURCE_SHA256,
    require_clean_git: bool = True,
) -> dict[str, Any]:
    if require_clean_git and git_dirty(repo_root):
        raise RuntimeError("Git must be clean before sealing blind-window selection")
    actual_sha = sha256_file(source_video)
    if actual_sha != expected_source_sha256:
        raise ValueError(f"source SHA-256 mismatch: expected {expected_source_sha256}, got {actual_sha}")
    metadata = probe_video_metadata(source_video)
    candidates, selection, seal = build_selection_payload(
        source_video=source_video,
        metadata=metadata,
        source_sha256=actual_sha,
        repo_root=repo_root,
    )
    selection_root = stage_root / "selection"
    write_json(selection_root / "candidate_intervals.json", candidates)
    write_json(selection_root / "blind_window_selection.json", selection)
    write_json(selection_root / "blind_window_selection_seal.json", seal)
    return seal
