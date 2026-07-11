from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from football_intelligence.core.config import validate_root_relative_posix_uri
from football_intelligence.core.fingerprint_policy import (
    SemanticFingerprintPolicy,
    canonical_semantic_json as policy_canonical_semantic_json,
)

MEDIA_EXTENSIONS = {
    ".avi",
    ".gif",
    ".jpeg",
    ".jpg",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".png",
    ".webp",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def media_type_for_path(path: Path) -> str:
    media_type, _ = mimetypes.guess_type(path.name)
    return media_type or "application/octet-stream"


def canonical_semantic_json(value: Any, policy: SemanticFingerprintPolicy | None = None) -> str:
    return policy_canonical_semantic_json(value, policy=policy)


def semantic_hash(value: Any, policy: SemanticFingerprintPolicy | None = None) -> str:
    return sha256_bytes(canonical_semantic_json(value, policy=policy).encode("utf-8"))


def inventory_directory(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"inventory root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"inventory root is not a directory: {root}")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_uri = validate_root_relative_posix_uri(path.relative_to(root).as_posix())
        stat = path.stat()
        records.append(
            {
                "relative_uri": relative_uri,
                "byte_size": stat.st_size,
                "content_hash": sha256_file(path),
                "media_type": media_type_for_path(path),
            }
        )
    return records


def directory_inventory_hash(records: list[dict[str, Any]]) -> str:
    return semantic_hash(records)


def is_media_file(path: Path) -> bool:
    return path.suffix.lower() in MEDIA_EXTENSIONS


def locate_ffprobe() -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return {
            "available": False,
            "path": None,
            "diagnostic": "ffprobe executable was not found on PATH; decoded-media fingerprints were not produced",
            "version": None,
        }
    return {"available": True, "path": ffprobe, "diagnostic": None, "version": ffprobe_version(ffprobe)}


def ffprobe_version(ffprobe_path: str | None = None) -> str | None:
    path = ffprobe_path or shutil.which("ffprobe")
    if path is None:
        return None
    try:
        completed = subprocess.run([path, "-version"], check=True, capture_output=True, text=True, timeout=10)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return completed.stdout.splitlines()[0] if completed.stdout else None


def _stable_ffprobe_payload(raw: dict[str, Any]) -> dict[str, Any]:
    streams = []
    for stream in raw.get("streams", []):
        streams.append(
            {
                key: stream.get(key)
                for key in (
                    "codec_type",
                    "codec_name",
                    "width",
                    "height",
                    "pix_fmt",
                    "avg_frame_rate",
                    "r_frame_rate",
                    "nb_frames",
                    "sample_rate",
                    "channels",
                )
                if key in stream
            }
        )
    fmt = raw.get("format", {})
    return {
        "format": {key: fmt.get(key) for key in ("format_name", "format_long_name", "size", "bit_rate") if key in fmt},
        "streams": streams,
    }


def ffprobe_fingerprint(path: Path, ffprobe_path: str | None = None) -> dict[str, Any]:
    probe = {
        "available": True,
        "path": ffprobe_path or shutil.which("ffprobe"),
        "diagnostic": None,
        "version": ffprobe_version(ffprobe_path),
    }
    if probe["path"] is None:
        return locate_ffprobe()
    command = [
        str(probe["path"]),
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        os.fspath(path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {
            "available": True,
            "path": probe["path"],
            "diagnostic": f"ffprobe failed for selected media: {exc}",
            "version": probe["version"],
        }
    payload = json.loads(completed.stdout or "{}")
    stable_payload = _stable_ffprobe_payload(payload)
    return {
        "available": True,
        "path": probe["path"],
        "diagnostic": None,
        "version": probe["version"],
        "metadata": stable_payload,
        "semantic_hash": semantic_hash(stable_payload),
    }


def media_fingerprint(path: Path, relative_uri: str, ffprobe_path: str | None = None) -> dict[str, Any]:
    validate_root_relative_posix_uri(relative_uri)
    return {
        "relative_uri": relative_uri,
        "byte_size": path.stat().st_size,
        "content_hash": sha256_file(path),
        "media_type": media_type_for_path(path),
        "ffprobe": ffprobe_fingerprint(path, ffprobe_path=ffprobe_path),
    }
