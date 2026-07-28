"""Deterministic, inventory-only helpers for the G7C dataset stage."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

SOURCE_MODALITIES = ("bas", "videos", "raw", "gsr")
FOLDERS = ("source", "manifests", "calibration", "annotations", "derived", "runs")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return ()
    return (p for p in sorted(root.rglob("*")) if p.is_file())


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def discover_matches(matches_root: Path, expected_count: int) -> list[Path]:
    matches = sorted((p for p in matches_root.iterdir() if p.is_dir()), key=lambda p: p.name)
    if len(matches) != expected_count or len({p.name for p in matches}) != expected_count:
        raise ValueError(f"FAIL_MATCH_COUNT: expected {expected_count}, found {len(matches)}")
    return matches


def _source_roots(match: Path) -> list[tuple[Path, str, str]]:
    roots: list[tuple[Path, str, str]] = []
    source = match / "source"
    for modality in SOURCE_MODALITIES:
        canonical = source / modality
        direct = match / modality
        if canonical.is_dir():
            roots.append((canonical, modality, "PRESENT_CANONICAL"))
        if direct.is_dir() and not canonical.is_dir():
            roots.append((direct, modality, "PRESENT_NONCANONICAL"))
    return roots


def inventory_sources(matches: list[Path], project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    fingerprints: dict[str, str] = {}
    for match in matches:
        for root, modality, location in _source_roots(match):
            for path in _files(root):
                rel = path.relative_to(project_root).as_posix()
                digest = sha256_file(path)
                records.append(
                    {
                        "match_id": match.name,
                        "relative_path": rel,
                        "resolved_modality": modality,
                        "canonical_location": location,
                        "byte_size": path.stat().st_size,
                        "sha256": digest,
                        "extension": path.suffix.lower(),
                        "media_type_guess": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                        "modified_timestamp_for_diagnostics_only": path.stat().st_mtime_ns,
                    }
                )
                fingerprints[rel] = digest
    by_hash: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_hash[record["sha256"]].append(record["relative_path"])
    duplicates = [paths for paths in by_hash.values() if len(paths) > 1]
    return records, {"files": fingerprints, "duplicate_groups": sorted(duplicates)}


def probe_media(path: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-of", "json", "-show_streams", "-show_format", str(path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=45,
        )
        payload = json.loads(result.stdout)
    except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError):
        return {"status": "METADATA_TOOL_UNAVAILABLE"}
    streams = payload.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    rate = video.get("r_frame_rate", "")
    try:
        frame_rate = round(float(rate.split("/")[0]) / float(rate.split("/")[1]), 6) if "/" in rate else float(rate)
    except (ValueError, ZeroDivisionError):
        frame_rate = None
    return {
        "status": "OK",
        "duration": float(payload.get("format", {}).get("duration", 0) or 0),
        "width": video.get("width"),
        "height": video.get("height"),
        "frame_rate": frame_rate,
        "codec": video.get("codec_name"),
        "stream_count": len(streams),
        "audio_presence": any(s.get("codec_type") == "audio" for s in streams),
    }


def media_metadata(records: list[dict[str, Any]], project_root: Path) -> list[dict[str, Any]]:
    result = []
    for record in records:
        if record["extension"].lower() not in VIDEO_EXTENSIONS:
            continue
        path = project_root / Path(record["relative_path"])
        result.append({"relative_path": record["relative_path"], "sha256": record["sha256"], **probe_media(path)})
    return result


def legacy_inventory(match: Path, project_root: Path) -> list[dict[str, Any]]:
    canonical = set(FOLDERS) | {"README_LEGACY_LAYOUT.md"}
    result = []
    for item in sorted(match.iterdir(), key=lambda p: p.name):
        if item.name in canonical:
            continue
        count = 0
        size = 0
        metadata = hashlib.sha256()
        items = [item] if item.is_file() else list(_files(item))
        for path in items:
            rel = path.relative_to(match).as_posix()
            stat = path.stat()
            count += 1
            size += stat.st_size
            metadata.update(f"{rel}\0{stat.st_size}\0file\n".encode())
        result.append(
            {
                "relative_path": item.relative_to(project_root).as_posix(),
                "item_type": "file" if item.is_file() else "directory",
                "recursive_file_count": count,
                "recursive_byte_count": size,
                "metadata_tree_sha256": metadata.hexdigest(),
                "likely_category": "legacy_or_stage_artifact",
                "movement_authorized": False,
            }
        )
    return result


def write_manifests(
    project_root: Path, matches: list[Path], records: list[dict[str, Any]], metadata: list[dict[str, Any]]
) -> None:
    by_match: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_match[record["match_id"]].append(record)
    for match in matches:
        files = sorted(by_match[match.name], key=lambda r: r["relative_path"])
        statuses = {m: "MISSING" for m in SOURCE_MODALITIES}
        for r in files:
            statuses[r["resolved_modality"]] = r["canonical_location"]
        manifest = {
            "schema_version": "g7c.match_manifest.v1",
            "match_id": match.name,
            "source_folder_status": statuses,
            "team_1_colour_status": "HUMAN_CONFIRMATION_REQUIRED",
            "team_2_colour_status": "HUMAN_CONFIRMATION_REQUIRED",
            "goalkeeper_colour_status": "HUMAN_CONFIRMATION_REQUIRED",
            "pitch_polygon_status": "HUMAN_REQUIRED",
            "source_file_count": len(files),
        }
        _json_dump(match / "manifests" / "match_manifest.json", manifest)
        _json_dump(match / "manifests" / "source_file_manifest.json", {"match_id": match.name, "files": files})
        _json_dump(match / "manifests" / "source_file_hashes.json", {r["relative_path"]: r["sha256"] for r in files})
        if match.name == "128058":
            _json_dump(match / "manifests" / "legacy_content_inventory.json", legacy_inventory(match, project_root))


def run_inventory(
    project_root: Path,
    matches_root: Path,
    dataset_root: Path,
    experiment_root: Path,
    expected_count: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    matches = discover_matches(matches_root, expected_count)
    records, fingerprint = inventory_sources(matches, project_root)
    metadata = media_metadata(records, project_root)
    folders = {"canonical": 0, "noncanonical": 0, "missing": 0}
    for match in matches:
        for modality in SOURCE_MODALITIES:
            canonical = (match / "source" / modality).is_dir()
            direct = (match / modality).is_dir()
            if canonical:
                folders["canonical"] += 1
            elif direct:
                folders["noncanonical"] += 1
            else:
                folders["missing"] += 1
    if not dry_run:
        write_manifests(project_root, matches, records, metadata)
    return {
        "matches": [m.name for m in matches],
        "source_records": records,
        "fingerprint": fingerprint,
        "media_metadata": metadata,
        "folder_counts": folders,
    }
