from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.replay.frame_set_validator import (
    derived_classification,
    directory_inventory_hash,
    inspect_derived_asset_exclusions,
    sha256_file,
    validate_candidate_frame_set,
)


MATCH_ID = "128058"
FRAME_ROOT_URI = "matches/128058/frames/goal_window_stage3c_hq_short"
MISSING_FRAME_ROOT = Path(FRAME_ROOT_URI)
STAGE_URI = "matches/128058/runs/step_m5/04a_source_frame_recovery"
BLOCKED_RUN_URI = (
    "matches/128058/runs/step_m5/04_true_m4_reconstruction/runs/" "m5_true_m4_replay_20260711T162334Z_ca5c6979"
)
PROMPT_PATH = Path(r"C:\Users\sebgr\.codex\attachments\a85737ba-5e82-4365-a647-8cd9777f7135\pasted-text.txt")
TRUE_REPLAY_COMMIT = "fe2b679a9f67adf5d01a32c3296538063017a0d4"

MEDIA_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}
ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".7z", ".rar"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif"}
TEXT_SUFFIXES = {".py", ".ps1", ".bat", ".cmd", ".md", ".txt", ".json", ".yaml", ".yml", ".log"}
SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    ".mypy_cache",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def stable_id(prefix: str, path: Path) -> str:
    return f"{prefix}_{hashlib.sha1(str(path.resolve()).encode('utf-8')).hexdigest()[:12]}"


def file_metadata(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "name": path.name,
        "suffix": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        "sha256": sha256_file(path) if hash_file else None,
    }


def safe_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


@dataclass
class ScanState:
    seen_dirs: set[str]
    errors: list[dict[str, Any]]
    directory_hits: list[dict[str, Any]]
    exact_hits: list[dict[str, Any]]
    video_candidates: list[dict[str, Any]]
    archive_candidates: list[dict[str, Any]]
    quarantine_candidates: list[dict[str, Any]]
    generation_candidates: list[dict[str, Any]]
    scanned_file_count: int = 0
    scanned_directory_count: int = 0
    truncated: bool = False
    time_exhausted: bool = False


def collect_search_roots(artifact_root: Path) -> list[dict[str, Any]]:
    user_root = Path.home()
    candidates = [
        ("artifact_root", artifact_root),
        ("documents", user_root / "Documents"),
        ("downloads", user_root / "Downloads"),
        ("desktop", user_root / "Desktop"),
    ]
    for path in sorted(user_root.glob("OneDrive*")):
        candidates.append(("onedrive", path))
    for drive_letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
        drive = Path(f"{drive_letter}:\\")
        if drive.exists():
            candidates.append((f"drive_{drive_letter}", drive))
    for drive_letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        drive = Path(f"{drive_letter}:\\")
        recycle = drive / "$Recycle.Bin"
        if recycle.exists():
            candidates.append((f"recycle_bin_{drive_letter}", recycle))
    quarantine_roots = []
    for root in [
        artifact_root,
        artifact_root / "matches/128058",
        artifact_root / "matches/128058/runs/step_m5",
        artifact_root / "matches/128058/runs",
    ]:
        if not root.exists():
            continue
        try:
            for path in root.iterdir():
                if path.is_dir() and ("quarantine" in path.name.lower() or "archive" in path.name.lower()):
                    quarantine_roots.append(path)
        except OSError:
            continue
    for path in sorted(set(quarantine_roots)):
        candidates.append(("project_quarantine", path))
    records = []
    seen = set()
    for role, path in candidates:
        key = str(path.resolve()).lower() if path.exists() else str(path).lower()
        duplicate = key in seen
        seen.add(key)
        records.append(
            {
                "role": role,
                "path": str(path),
                "exists": path.exists(),
                "duplicate_of_prior_search_root": duplicate,
                "searched": path.exists() and not duplicate,
                "read_only": True,
            }
        )
    return records


def frame_requirement_manifest(
    *,
    artifact_root: Path,
    stage_root: Path,
    blocked_run: Path,
) -> dict[str, Any]:
    frame_manifest_path = artifact_root / FRAME_ROOT_URI / "frame_manifest.json"
    frame_manifest = read_json(frame_manifest_path)
    frame_lookup = read_json(blocked_run / "recovered_m1/frame_lookup.json")
    frames = [row for row in frame_manifest.get("frames", []) if isinstance(row, dict)]
    by_sequence = {int(row["frame_sequence"]): row for row in frames if "frame_sequence" in row}
    required_sequences = [
        int(record["frame_sequence"])
        for record in frame_lookup.get("records", [])
        if isinstance(record, dict) and "frame_sequence" in record
    ]
    declared_frames = [
        {
            "frame_sequence": int(row.get("frame_sequence")),
            "frame_id": row.get("frame_id"),
            "expected_filename": Path(str(row.get("frame_file", ""))).name,
            "declared_frame_file": row.get("frame_file"),
            "timestamp_seconds": row.get("timestamp_seconds"),
            "window_timestamp_seconds": row.get("window_timestamp_seconds"),
            "frame_index": row.get("frame_index"),
            "source_frame_index": row.get("source_frame_index"),
            "width": row.get("width"),
            "height": row.get("height"),
        }
        for row in sorted(frames, key=lambda item: int(item.get("frame_sequence", -1)))
    ]
    required_frames = []
    for sequence in sorted(required_sequences):
        row = by_sequence.get(sequence, {})
        required_frames.append(
            {
                "frame_sequence": sequence,
                "frame_id": row.get("frame_id"),
                "expected_filename": Path(str(row.get("frame_file", ""))).name,
                "declared_frame_file": row.get("frame_file"),
                "timestamp_seconds": row.get("timestamp_seconds"),
                "window_timestamp_seconds": row.get("window_timestamp_seconds"),
                "frame_index": row.get("frame_index"),
                "source_frame_index": row.get("source_frame_index"),
                "width": row.get("width"),
                "height": row.get("height"),
            }
        )
    parameters = frame_manifest.get("parameters", {}) if isinstance(frame_manifest.get("parameters"), dict) else {}
    payload = {
        "schema_version": "m5_2r_a.frame_requirement_manifest.v1",
        "created_at": utc_now(),
        "declared_frame_root": str((artifact_root / FRAME_ROOT_URI).resolve()),
        "declared_frame_manifest_uri": FRAME_ROOT_URI + "/frame_manifest.json",
        "declared_source_video_or_clip_path": parameters.get("clip_path"),
        "frame_count": len(frames),
        "required_frame_count_for_true_m4_renderer": len(required_frames),
        "declared_frame_sequence_values": [row["frame_sequence"] for row in declared_frames],
        "required_frame_sequence_values": [row["frame_sequence"] for row in required_frames],
        "required_filename_sequence": [row["expected_filename"] for row in required_frames],
        "source_timestamps_where_present": [
            {
                "frame_sequence": row["frame_sequence"],
                "timestamp_seconds": row.get("timestamp_seconds"),
                "window_timestamp_seconds": row.get("window_timestamp_seconds"),
            }
            for row in required_frames
        ],
        "expected_dimensions": sorted(
            {
                f"{row.get('width')}x{row.get('height')}"
                for row in declared_frames
                if row.get("width") is not None and row.get("height") is not None
            }
        ),
        "expected_extension": ".jpg",
        "sampling_rate": parameters.get("fps"),
        "first_required_frame": required_frames[0] if required_frames else None,
        "last_required_frame": required_frames[-1] if required_frames else None,
        "byte_hashes_present_in_manifest": any("sha256" in row or "byte_hash" in row for row in frames),
        "pixel_hashes_present_in_manifest": any("pixel_hash" in row or "decoded_pixel_hash" in row for row in frames),
        "declared_frames": declared_frames,
        "required_frames": required_frames,
        "source_values_not_inferred": True,
        "blocked_run_frame_lookup": str((blocked_run / "recovered_m1/frame_lookup.json").resolve()),
    }
    write_json(stage_root / "recovery/frame_requirement_manifest.json", payload)
    return payload


def text_matches(path: Path, terms: list[str]) -> list[str]:
    if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 5_000_000:
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return []
    return [term for term in terms if term.lower() in text]


def walk_root(
    *,
    root: Path,
    root_role: str,
    state: ScanState,
    exact_names: set[str],
    frame_root_name: str,
    source_clip_name: str | None,
    generation_terms: list[str],
    media_keywords: list[str],
    max_files: int,
    max_dirs: int,
    deadline: float,
) -> dict[str, Any]:
    if not root.exists():
        return {"role": root_role, "path": str(root), "exists": False, "searched": False, "reason": "missing"}
    if str(root.resolve()).lower() in state.seen_dirs:
        return {"role": root_role, "path": str(root), "exists": True, "searched": False, "reason": "duplicate"}
    stack = [root]
    scanned_here = 0
    directories_here = 0
    while stack:
        if time.monotonic() > deadline:
            state.truncated = True
            state.time_exhausted = True
            break
        if state.scanned_directory_count >= max_dirs:
            state.truncated = True
            break
        current = stack.pop()
        current_key = str(current.resolve()).lower()
        if current_key in state.seen_dirs:
            continue
        state.seen_dirs.add(current_key)
        try:
            with os.scandir(current) as handle:
                entries = list(handle)
        except OSError as exc:
            state.errors.append({"root_role": root_role, "path": str(current), "error": str(exc)})
            continue
        directories_here += 1
        state.scanned_directory_count += 1
        if current.name.lower() == frame_root_name.lower():
            state.directory_hits.append(
                {
                    "path": str(current),
                    "root_role": root_role,
                    "name": current.name,
                    "jpg_count": len(list(current.glob("*.jpg"))),
                    "classification": derived_classification(current) or "provenance_uncertain_requires_review",
                }
            )
        for entry in entries:
            path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                if entry.name in SKIP_DIR_NAMES or entry.name.startswith("_tmp"):
                    continue
                stack.append(path)
                continue
            if not entry.is_file(follow_symlinks=False):
                continue
            scanned_here += 1
            state.scanned_file_count += 1
            if state.scanned_file_count >= max_files:
                state.truncated = True
                stack.clear()
                break
            suffix = path.suffix.lower()
            name = path.name
            name_lower = name.lower()
            path_lower = path.as_posix().lower()
            if name in exact_names:
                record = file_metadata(path)
                record.update(
                    {
                        "root_role": root_role,
                        "classification": derived_classification(path) or "provenance_uncertain_requires_review",
                        "allowed_for_true_replay": derived_classification(path) is None,
                    }
                )
                state.exact_hits.append(record)
            is_media_candidate = suffix in MEDIA_SUFFIXES and (
                (source_clip_name and name_lower == source_clip_name.lower())
                or any(keyword.lower() in path_lower for keyword in media_keywords)
            )
            if is_media_candidate:
                record = file_metadata(path)
                record.update(
                    {
                        "root_role": root_role,
                        "type": "video",
                        "classification": (
                            "original_source_clip"
                            if source_clip_name and name_lower == source_clip_name.lower()
                            else "provenance_uncertain_requires_review"
                        ),
                        "allowed_for_true_replay": bool(source_clip_name and name_lower == source_clip_name.lower()),
                    }
                )
                state.video_candidates.append(record)
            if suffix in ARCHIVE_SUFFIXES and any(keyword.lower() in path_lower for keyword in media_keywords):
                record = file_metadata(path)
                record.update(
                    {
                        "root_role": root_role,
                        "type": "archive",
                        "classification": "provenance_uncertain_requires_review",
                        "allowed_for_true_replay": False,
                    }
                )
                state.archive_candidates.append(record)
            if ("quarantine" in path_lower or "$recycle.bin" in path_lower) and any(
                keyword.lower() in path_lower for keyword in media_keywords
            ):
                record = file_metadata(path, hash_file=suffix in (MEDIA_SUFFIXES | ARCHIVE_SUFFIXES | IMAGE_SUFFIXES))
                record.update(
                    {
                        "root_role": root_role,
                        "classification": derived_classification(path) or "provenance_uncertain_requires_review",
                        "allowed_for_true_replay": False,
                    }
                )
                state.quarantine_candidates.append(record)
            matched_terms = text_matches(path, generation_terms)
            if matched_terms:
                state.generation_candidates.append(
                    {
                        "path": str(path.resolve()),
                        "root_role": root_role,
                        "size_bytes": path.stat().st_size,
                        "matched_terms": matched_terms,
                        "classification": "generation_command_reference",
                        "read_only": True,
                    }
                )
    return {
        "role": root_role,
        "path": str(root),
        "exists": True,
        "searched": True,
        "scanned_file_count": scanned_here,
        "scanned_directory_count": directories_here,
        "truncated_by_bound": state.truncated,
        "time_exhausted": state.time_exhausted,
    }


def forensic_search(
    *,
    artifact_root: Path,
    repo_root: Path,
    stage_root: Path,
    requirement: dict[str, Any],
    max_files: int = 60_000,
    max_dirs: int = 20_000,
    max_seconds: int = 240,
) -> dict[str, Any]:
    required_names = set(requirement["required_filename_sequence"])
    frame_root_name = Path(FRAME_ROOT_URI).name
    clip_path = requirement.get("declared_source_video_or_clip_path")
    source_clip_name = Path(str(clip_path)).name if clip_path else None
    media_keywords = [MATCH_ID, "goal_window", "stage3c", "hq_short", "1882", "2062"]
    if source_clip_name:
        media_keywords.append(source_clip_name)
    generation_terms = [
        "goal_window_stage3c_hq_short",
        "stage3c_00_extract_hq_short_window_frames",
        "frame_manifest",
        "ffmpeg",
        "cv2.videocapture",
        source_clip_name or "",
    ]
    generation_terms = [term for term in generation_terms if term]
    search_roots = collect_search_roots(artifact_root)
    deadline = time.monotonic() + max_seconds
    state = ScanState(
        seen_dirs=set(),
        errors=[],
        directory_hits=[],
        exact_hits=[],
        video_candidates=[],
        archive_candidates=[],
        quarantine_candidates=[],
        generation_candidates=[],
    )
    location_results = []
    for record in search_roots:
        if not record["searched"]:
            location_results.append(record | {"scan_result": "not_searched"})
            continue
        result = walk_root(
            root=Path(record["path"]),
            root_role=record["role"],
            state=state,
            exact_names=required_names,
            frame_root_name=frame_root_name,
            source_clip_name=source_clip_name,
            generation_terms=generation_terms,
            media_keywords=media_keywords,
            max_files=max_files,
            max_dirs=max_dirs,
            deadline=deadline,
        )
        location_results.append(record | result)
    shell_history = Path.home() / "AppData/Roaming/Microsoft/Windows/PowerShell/PSReadLine/ConsoleHost_history.txt"
    if shell_history.exists():
        matched_terms = text_matches(shell_history, generation_terms)
        if matched_terms:
            state.generation_candidates.append(
                {
                    "path": str(shell_history.resolve()),
                    "root_role": "shell_history",
                    "size_bytes": shell_history.stat().st_size,
                    "matched_terms": matched_terms,
                    "classification": "generation_command_reference",
                    "read_only": True,
                }
            )
    git_lfs = {
        "git_lfs_directory_exists": (repo_root / ".git/lfs").exists(),
        "git_attributes_mentions_lfs": False,
    }
    git_attributes = repo_root / ".gitattributes"
    if git_attributes.exists():
        git_lfs["git_attributes_mentions_lfs"] = "filter=lfs" in git_attributes.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    dvc = {
        "dvc_directory_exists": (repo_root / ".dvc").exists(),
        "dvc_config_exists": (repo_root / ".dvc/config").exists(),
    }
    search_locations = {
        "schema_version": "m5_2r_a.search_locations.v1",
        "created_at": utc_now(),
        "read_only": True,
        "max_files_bound": max_files,
        "max_directories_bound": max_dirs,
        "max_seconds_bound": max_seconds,
        "scanned_file_count": state.scanned_file_count,
        "scanned_directory_count": state.scanned_directory_count,
        "truncated": state.truncated,
        "time_exhausted": state.time_exhausted,
        "locations": location_results,
        "scan_errors": state.errors,
        "git_lfs_metadata": git_lfs,
        "dvc_metadata": dvc,
    }
    exact_hits = {
        "schema_version": "m5_2r_a.exact_filename_hits.v1",
        "required_filename_count": len(required_names),
        "hit_count": len(state.exact_hits),
        "hits": state.exact_hits,
    }
    videos = {
        "schema_version": "m5_2r_a.source_video_candidates.v1",
        "declared_source_clip_filename": source_clip_name,
        "candidate_count": len(state.video_candidates),
        "candidates": state.video_candidates,
    }
    archives = {
        "schema_version": "m5_2r_a.archive_candidates.v1",
        "candidate_count": len(state.archive_candidates),
        "candidates": state.archive_candidates,
    }
    quarantines = {
        "schema_version": "m5_2r_a.quarantine_candidates.v1",
        "candidate_count": len(state.quarantine_candidates),
        "candidates": state.quarantine_candidates,
    }
    generation = {
        "schema_version": "m5_2r_a.generation_command_candidates.v1",
        "candidate_count": len(state.generation_candidates),
        "candidates": state.generation_candidates[:200],
        "truncated_candidate_list": len(state.generation_candidates) > 200,
    }
    write_json(stage_root / "recovery/search_locations.json", search_locations)
    write_json(stage_root / "recovery/exact_filename_hits.json", exact_hits)
    write_json(stage_root / "recovery/source_video_candidates.json", videos)
    write_json(stage_root / "recovery/archive_candidates.json", archives)
    write_json(stage_root / "recovery/quarantine_candidates.json", quarantines)
    write_json(stage_root / "recovery/generation_command_candidates.json", generation)
    return {
        "search_locations": search_locations,
        "exact_hits": exact_hits,
        "source_video_candidates": videos,
        "archive_candidates": archives,
        "quarantine_candidates": quarantines,
        "generation_command_candidates": generation,
        "directory_hits": state.directory_hits,
    }


def summarize_candidates(
    *,
    artifact_root: Path,
    stage_root: Path,
    requirement: dict[str, Any],
    search_result: dict[str, Any],
    derived_report: dict[str, Any],
) -> dict[str, Any]:
    required_names = set(requirement["required_filename_sequence"])
    declared_names = {row["expected_filename"] for row in requirement["declared_frames"]}
    candidates = []
    missing_root = artifact_root / FRAME_ROOT_URI
    roots_to_score = [missing_root]
    for hit in search_result["directory_hits"]:
        path = Path(hit["path"])
        if path.exists() and path not in roots_to_score:
            roots_to_score.append(path)
    for path in roots_to_score:
        jpg_names = {item.name for item in path.glob("*.jpg")} if path.exists() else set()
        derived = derived_classification(path)
        exact_required = sorted(required_names & jpg_names)
        exact_declared = sorted(declared_names & jpg_names)
        if derived:
            classification = derived
        elif len(exact_declared) == len(declared_names) and len(exact_required) == len(required_names):
            classification = "exact_original_frame_set"
        elif exact_required:
            classification = "partial_original_frame_set"
        else:
            classification = "provenance_uncertain_requires_review"
        allowed = classification == "exact_original_frame_set"
        candidates.append(
            {
                "candidate_id": stable_id("frames", path),
                "current_path": str(path.resolve()),
                "original_path_where_known": str(path.resolve()) if path == missing_root else None,
                "type": "frame_directory",
                "exists": path.exists(),
                "size": directory_inventory_hash(path, extensions={".jpg", ".jpeg"})["total_bytes"]
                if path.exists()
                else 0,
                "modification_time": path.stat().st_mtime if path.exists() else None,
                "sha256": directory_inventory_hash(path, extensions={".jpg", ".jpeg"})["inventory_hash"]
                if path.exists()
                else None,
                "raw_or_derived": "derived" if derived else "unknown",
                "annotations_visible_or_implied_by_provenance": bool(derived),
                "filename_coverage": {
                    "required_present": len(exact_required),
                    "required_total": len(required_names),
                    "declared_present": len(exact_declared),
                    "declared_total": len(declared_names),
                },
                "frame_sequence_coverage": {
                    "required_present": len(exact_required),
                    "required_total": len(required_names),
                },
                "dimension_compatibility": "not_assessed_without_exact_filename_coverage",
                "manifest_compatibility": allowed,
                "recovery_confidence": "high" if allowed else "none",
                "classification": classification,
                "allowed_for_true_replay": allowed,
            }
        )
    for source_name, key in [
        ("source_video_candidates", "original_source_clip"),
        ("archive_candidates", "archive_containing_source_video"),
        ("quarantine_candidates", "provenance_uncertain_requires_review"),
    ]:
        for record in search_result[source_name]["candidates"]:
            candidates.append(
                {
                    "candidate_id": stable_id(source_name, Path(record["path"])),
                    "current_path": record["path"],
                    "original_path_where_known": None,
                    "type": record.get("type", "file"),
                    "size": record.get("size_bytes"),
                    "modification_time": record.get("modified_at"),
                    "sha256": record.get("sha256"),
                    "raw_or_derived": "raw_candidate" if record.get("allowed_for_true_replay") else "unknown",
                    "annotations_visible_or_implied_by_provenance": derived_classification(Path(record["path"]))
                    is not None,
                    "filename_coverage": None,
                    "frame_sequence_coverage": None,
                    "dimension_compatibility": "not_applicable",
                    "manifest_compatibility": record.get("allowed_for_true_replay", False),
                    "recovery_confidence": "medium" if record.get("allowed_for_true_replay") else "low",
                    "classification": record.get("classification", key),
                    "allowed_for_true_replay": record.get("allowed_for_true_replay", False),
                }
            )
    for record in derived_report["records"]:
        candidates.append(
            {
                "candidate_id": stable_id("derived", Path(record["path"])),
                "current_path": record["path"],
                "original_path_where_known": None,
                "type": record["type"],
                "size": record["size_bytes"],
                "modification_time": None,
                "sha256": record["sha256_or_inventory_hash"],
                "raw_or_derived": "derived",
                "annotations_visible_or_implied_by_provenance": True,
                "filename_coverage": None,
                "frame_sequence_coverage": None,
                "dimension_compatibility": "not_applicable",
                "manifest_compatibility": False,
                "recovery_confidence": "none",
                "classification": record["classification"],
                "allowed_for_true_replay": False,
            }
        )
    payload = {
        "schema_version": "m5_2r_a.recovery_candidate_summary.v1",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "eligible_candidate_count": sum(1 for item in candidates if item["allowed_for_true_replay"]),
    }
    write_json(stage_root / "recovery/recovery_candidate_summary.json", payload)
    return payload


def run_git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def not_applicable_payload(schema_version: str, reason: str) -> dict[str, Any]:
    return {"schema_version": schema_version, "status": "not_applicable", "reason": reason, "passed": True}


def write_blocked_markdown(stage_root: Path, decision: dict[str, Any]) -> None:
    text = f"""# M5.2R Blocked: Missing Source Media

Final recovery classification: `{decision["final_classification"]}`

The true M4 reconstruction gate remains intact. M5.2R cannot resume because the
declared source-frame directory contains `frame_manifest.json` but zero JPG
source frames, and no eligible original source frame set or source video was
recovered in this bounded search.

The stage deliberately rejected preserved M4 evidence, Stage3D static-freeze
images, review contact sheets, screenshots, and other derived media as true
replay frame sources.

Recommended next step: move to M5.3 on a newly selected blind window with an
artifact-retention contract requiring raw source video hash, extracted-frame
manifest, all extracted source frames, extraction command, ffmpeg/ffprobe
versions, frame inventory hash, immutable input closure, and backup or remote
storage confirmation.
"""
    write_text(stage_root / "M5_2R_BLOCKED_MISSING_SOURCE_MEDIA.md", text)


def copy_review_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())


def build_review_pack(
    *,
    repo_root: Path,
    stage_root: Path,
    prompt_path: Path,
    decision: dict[str, Any],
) -> Path:
    review_pack = stage_root / "review_pack"
    review_pack.mkdir(parents=True, exist_ok=True)
    guide = f"""# M5.2R-A Review Guide

Stage root:
`{stage_root}`

Final classification:
`{decision["final_classification"]}`

M5.2R resume readiness:
`{decision["m5_2r_can_resume"]}`

Main blocker:
The declared Stage3C HQ source-frame directory has a manifest but zero JPGs. The
latest M5.2R frame lookup required 289 frames and resolved 0.

What was achieved:
- Parsed the frame manifest and blocked true-replay frame requirements.
- Performed a bounded read-only forensic search.
- Hashed and classified narrowed frame/video/archive/derived candidates.
- Explicitly rejected preserved M4 evidence and Stage3D static-freeze media.
- Preserved the true-replay gate without weakening algorithms or guardrails.
"""
    write_text(review_pack / "00_REVIEW_GUIDE.md", guide)
    copy_review_file(prompt_path, review_pack / "01_ORIGINAL_PROMPT.txt")
    mappings = [
        ("recovery/frame_requirement_manifest.json", "02_FRAME_REQUIREMENT_MANIFEST.json"),
        ("recovery/search_locations.json", "03_SEARCH_LOCATIONS.json"),
        ("recovery/exact_filename_hits.json", "04_EXACT_FILENAME_HITS.json"),
        ("recovery/source_video_candidates.json", "05_SOURCE_VIDEO_CANDIDATES.json"),
        ("recovery/archive_candidates.json", "06_ARCHIVE_CANDIDATES.json"),
        ("recovery/quarantine_candidates.json", "07_QUARANTINE_CANDIDATES.json"),
        ("recovery/generation_command_candidates.json", "08_GENERATION_COMMAND_CANDIDATES.json"),
        ("recovery/recovery_candidate_summary.json", "09_RECOVERY_CANDIDATE_SUMMARY.json"),
        ("validation/recovered_frame_set_validation.json", "10_RECOVERED_FRAME_SET_VALIDATION.json"),
        ("validation/regenerated_frame_set_validation.json", "11_REGENERATED_FRAME_SET_VALIDATION.json"),
        ("validation/regeneration_repeatability_report.json", "12_REGENERATION_REPEATABILITY_REPORT.json"),
        ("validation/derived_asset_exclusion_report.json", "13_DERIVED_ASSET_EXCLUSION_REPORT.json"),
        ("SOURCE_MEDIA_RECOVERY_DECISION.json", "14_SOURCE_MEDIA_RECOVERY_DECISION.json"),
        ("M5_2R_RESUME_READINESS.json", "15_M5_2R_RESUME_READINESS.json"),
    ]
    for relative, name in mappings:
        copy_review_file(stage_root / relative, review_pack / name)
    copy_review_file(
        repo_root / "src/football_intelligence/replay/source_frame_recovery.py",
        review_pack / "16_search_and_classification.py",
    )
    copy_review_file(
        repo_root / "src/football_intelligence/replay/frame_set_validator.py",
        review_pack / "17_frame_set_validator.py",
    )
    copy_review_file(
        repo_root / "tests/unit/replay/test_source_frame_recovery.py",
        review_pack / "18_test_derived_assets_rejected.py",
    )
    files = sorted(
        path.name for path in review_pack.iterdir() if path.is_file() and path.name != "19_REVIEW_PACK_MANIFEST.json"
    )
    files.append("19_REVIEW_PACK_MANIFEST.json")
    manifest = {
        "schema_version": "m5_2r_a.review_pack_manifest.v1",
        "file_count": len(files),
        "files": files,
        "max_files_allowed": 20,
    }
    write_json(review_pack / "19_REVIEW_PACK_MANIFEST.json", manifest)
    actual = sorted(path.name for path in review_pack.iterdir() if path.is_file())
    if len(actual) != 20:
        raise RuntimeError(f"review pack must contain exactly 20 files, found {len(actual)}: {actual}")
    return review_pack


def run_source_frame_recovery(
    *,
    repo_root: Path,
    artifact_root: Path,
    prompt_path: Path = PROMPT_PATH,
) -> dict[str, Any]:
    stage_root = artifact_root / STAGE_URI
    stage_root.mkdir(parents=True, exist_ok=True)
    blocked_run = artifact_root / BLOCKED_RUN_URI
    git_commit = run_git(repo_root, "rev-parse", "HEAD")
    git_status = run_git(repo_root, "status", "--porcelain")
    requirement = frame_requirement_manifest(
        artifact_root=artifact_root,
        stage_root=stage_root,
        blocked_run=blocked_run,
    )
    search_result = forensic_search(
        artifact_root=artifact_root,
        repo_root=repo_root,
        stage_root=stage_root,
        requirement=requirement,
    )
    derived_report = inspect_derived_asset_exclusions(artifact_root)
    write_json(stage_root / "validation/derived_asset_exclusion_report.json", derived_report)
    candidate_summary = summarize_candidates(
        artifact_root=artifact_root,
        stage_root=stage_root,
        requirement=requirement,
        search_result=search_result,
        derived_report=derived_report,
    )
    eligible_frame_dirs = [
        candidate
        for candidate in candidate_summary["candidates"]
        if candidate["classification"] == "exact_original_frame_set" and candidate["allowed_for_true_replay"]
    ]
    recovered_validation = not_applicable_payload(
        "m5_2r_a.recovered_frame_set_validation.v1",
        "No exact original frame set candidate was found.",
    )
    if eligible_frame_dirs:
        source = Path(eligible_frame_dirs[0]["current_path"])
        candidate_id = eligible_frame_dirs[0]["candidate_id"]
        recovered_root = stage_root / "recovered_candidates" / candidate_id
        recovered_root.mkdir(parents=True, exist_ok=True)
        for path in source.glob("*.jpg"):
            shutil.copy2(path, recovered_root / path.name)
        recovered_validation = validate_candidate_frame_set(
            candidate_root=recovered_root,
            required_frames=requirement["required_frames"],
            declared_frames=requirement["declared_frames"],
        )
    write_json(stage_root / "validation/recovered_frame_set_validation.json", recovered_validation)
    regenerated_validation = not_applicable_payload(
        "m5_2r_a.regenerated_frame_set_validation.v1",
        "No eligible original source video or source clip was recovered.",
    )
    repeatability = not_applicable_payload(
        "m5_2r_a.regeneration_repeatability_report.v1",
        "No video-based regeneration was attempted because no eligible original source video was found.",
    )
    write_json(stage_root / "validation/regenerated_frame_set_validation.json", regenerated_validation)
    write_json(stage_root / "validation/regeneration_repeatability_report.json", repeatability)
    if recovered_validation.get("classification") == "RECOVERED_EXACT_SOURCE_FRAME_SET" and recovered_validation.get(
        "passed"
    ):
        final_classification = "RECOVERED_EXACT_SOURCE_FRAME_SET"
        can_resume = True
        approved_uri = safe_relative(Path(recovered_validation["candidate_root"]), artifact_root)
    elif search_result["source_video_candidates"]["candidate_count"] > 0:
        final_classification = "PROVENANCE_UNCERTAIN_REQUIRES_HUMAN_REVIEW"
        can_resume = False
        approved_uri = None
    else:
        final_classification = "SOURCE_MEDIA_NOT_FOUND"
        can_resume = False
        approved_uri = None
    decision = {
        "schema_version": "m5_2r_a.source_media_recovery_decision.v1",
        "created_at": utc_now(),
        "final_classification": final_classification,
        "m5_2r_can_resume": can_resume,
        "approved_recovered_candidate_uri": approved_uri,
        "source_byte_hash": None,
        "recovered_frame_inventory_hash": recovered_validation.get("inventory", {}).get("inventory_hash"),
        "required_frame_coverage": {
            "required": requirement["required_frame_count_for_true_m4_renderer"],
            "resolved": recovered_validation.get("required_frames_present", 0),
        },
        "total_frame_coverage": {
            "declared": requirement["frame_count"],
            "resolved": recovered_validation.get("declared_frames_present", 0),
        },
        "known_limitations": [
            "The declared frame root contains zero JPG source frames.",
            "No eligible original source clip was recovered during the bounded search.",
        ],
        "prohibited_candidate_sources": [
            "preserved M4 overlays",
            "M4 strips",
            "M4 GIFs",
            "Stage3D static-freeze images",
            "review contact sheets",
            "screenshots or annotated derived images",
        ],
        "recommended_next_command": (
            "Restore the original source frame JPGs or source clip, then rerun "
            "scripts/m5_2r_source_frame_recovery.py. If not available, move to M5.3 with a stricter "
            "artifact-retention contract."
        ),
        "true_replay_code_commit_at_start": TRUE_REPLAY_COMMIT,
        "recovery_tool_git_commit": git_commit,
        "recovery_tool_git_dirty": bool(git_status),
    }
    write_json(stage_root / "SOURCE_MEDIA_RECOVERY_DECISION.json", decision)
    resume = {
        "schema_version": "m5_2r_a.resume_readiness.v1",
        "m5_2r_can_resume": can_resume,
        "eligible_frame_set_found": final_classification == "RECOVERED_EXACT_SOURCE_FRAME_SET",
        "eligible_source_video_found": False,
        "new_true_replay_config_created": False,
        "reason": "Source media not recovered." if not can_resume else "Recovered frame set is available.",
        "environment": {"git": {"commit": git_commit, "dirty": bool(git_status), "status_porcelain": git_status}},
    }
    write_json(stage_root / "M5_2R_RESUME_READINESS.json", resume)
    if not can_resume:
        write_blocked_markdown(stage_root, decision)
    review_pack = build_review_pack(
        repo_root=repo_root,
        stage_root=stage_root,
        prompt_path=prompt_path,
        decision=decision,
    )
    return {
        "stage_root": str(stage_root.resolve()),
        "review_pack": str(review_pack.resolve()),
        "decision": decision,
        "candidate_summary": candidate_summary,
        "search_result": search_result,
    }
