from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

MAX_REVIEW_PACK_FILES = 20
MAX_REVIEW_PACK_BYTES = 50 * 1024 * 1024
MAX_REVIEW_PACK_VISUALS = 3
REQUIRED_SOURCE_DIFF_NAME = "04_SOURCE_DIFF.patch"
REVIEW_PACK_MANIFEST_NAME = "REVIEW_PACK_MANIFEST.json"
REVIEW_PACK_MANIFEST_SCHEMA_VERSION = "football_intelligence.m5_5g7a.review_pack_manifest.v1"
REVIEW_PACK_VALIDATION_SCHEMA_VERSION = "football_intelligence.m5_5g7a.review_pack_validation.v1"

VISUAL_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".svg",
    ".pdf",
}
RASTER_VISUAL_SUFFIXES = VISUAL_SUFFIXES - {".svg", ".pdf"}
ALLOWED_REVIEW_SUFFIXES = {".json", ".jsonl", ".md", ".patch", ".txt", *VISUAL_SUFFIXES}
MODEL_WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".engine",
    ".h5",
    ".hdf5",
    ".joblib",
    ".onnx",
    ".pb",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
}
EMBEDDING_CACHE_SUFFIXES = {".cache", ".npy", ".npz"}
CREDENTIAL_SUFFIXES = {".env", ".jks", ".kdbx", ".key", ".p12", ".pem", ".pfx"}
RAW_VIDEO_SUFFIXES = {
    ".3gp",
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".mts",
    ".ts",
    ".webm",
}

_MODEL_WEIGHT_NAME_FRAGMENTS = ("checkpoint", "model_weights", "state_dict")
_EMBEDDING_CACHE_NAME_FRAGMENTS = ("cached_embedding", "embedding_cache", "embeddings_cache")
_FULL_DECISION_NAME_FRAGMENTS = (
    "completed_review",
    "decision_events",
    "full_decisions",
    "full_human_decisions",
    "human_decisions",
    "review_decisions",
)
_CREDENTIAL_NAME_FRAGMENTS = ("access_token", "api_key", "credential", "private_key", "secret")
_PRIVATE_KEY_MARKERS = (
    b"-----BEGIN PRIVATE KEY-----",
    b"-----BEGIN RSA PRIVATE KEY-----",
    b"-----BEGIN OPENSSH PRIVATE KEY-----",
    b"-----BEGIN EC PRIVATE KEY-----",
)
_BINARY_ARCHIVE_MARKERS = (
    b"PK\x03\x04",
    b"PK\x05\x06",
    b"PK\x07\x08",
    b"\x80\x02",
    b"\x80\x03",
    b"\x80\x04",
    b"\x80\x05",
)
_FORBIDDEN_JSON_KEYS = {
    "annotations",
    "decision_events",
    "embedding",
    "embeddings",
    "full_human_decisions",
    "human_decisions",
    "model_state_dict",
    "optimizer_state_dict",
    "private_key",
    "state_dict",
}


class ReviewPackValidationError(ValueError):
    """Raised when a G7A review pack violates its bounded-content contract."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


def stage_safety_summary() -> dict[str, bool | str]:
    """Return the non-overridable G7A development and promotion boundaries."""

    return {
        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        "development_scope": "SINGLE_MATCH_GROUPED_DEVELOPMENT_ONLY",
        "sandbox_only": True,
        "match_local_only": True,
        "no_auto_promotion": True,
        "production_ready": False,
        "detector_changes_performed": False,
        "tracker_changes_performed": False,
        "detector_defaults_changed": False,
        "tracker_defaults_changed": False,
        "project_defaults_changed": False,
        "identity_predictions_performed": False,
        "identity_tracking_performed": False,
        "temporal_predictions_performed": False,
        "temporal_acceptance_predictions_performed": False,
        "exact_visible_person_count_forcing_performed": False,
        "exact_22_forcing_performed": False,
        "exact_two_visible_goalkeepers_forcing_performed": False,
        "exactly_one_goalkeeper_per_team_forcing_performed": False,
        "hard_goalkeeper_count_forcing_performed": False,
    }


def sha256_file(path: Path) -> str:
    """Hash a file without loading the whole payload into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with source.open("rb") as source_handle, temporary.open("xb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def _is_token_named(path: Path, token: str) -> bool:
    parts = re.split(r"[^a-z0-9]+", path.stem.casefold())
    return token in parts


def _walk_json_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            keys.update(str(key).casefold() for key in current)
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return keys


def _payload_type_errors(path: Path) -> list[str]:
    name = path.name
    lower_name = name.casefold()
    suffix = path.suffix.casefold()
    errors: list[str] = []
    if suffix not in ALLOWED_REVIEW_SUFFIXES:
        errors.append(f"review-pack payload suffix is not allowlisted: {name}")
    if suffix == ".parquet":
        errors.append(f"Parquet training rows are forbidden: {name}")
    if suffix in MODEL_WEIGHT_SUFFIXES or any(fragment in lower_name for fragment in _MODEL_WEIGHT_NAME_FRAGMENTS):
        errors.append(f"model weights are forbidden: {name}")
    if suffix in EMBEDDING_CACHE_SUFFIXES or any(
        fragment in lower_name for fragment in _EMBEDDING_CACHE_NAME_FRAGMENTS
    ):
        errors.append(f"cached embeddings are forbidden: {name}")
    if any(fragment in lower_name for fragment in _FULL_DECISION_NAME_FRAGMENTS):
        errors.append(f"full human decisions are forbidden: {name}")
    if suffix in CREDENTIAL_SUFFIXES or any(fragment in lower_name for fragment in _CREDENTIAL_NAME_FRAGMENTS):
        errors.append(f"credentials are forbidden: {name}")
    if _is_token_named(path, "token") or _is_token_named(path, "password"):
        errors.append(f"credentials are forbidden: {name}")
    if suffix in RAW_VIDEO_SUFFIXES:
        errors.append(f"raw video is forbidden: {name}")

    try:
        with path.open("rb") as handle:
            head = handle.read(4096)
            if path.stat().st_size >= 4:
                handle.seek(-4, os.SEEK_END)
                tail = handle.read(4)
            else:
                tail = b""
    except OSError as exc:
        errors.append(f"review-pack file cannot be inspected ({name}): {exc}")
        return errors
    if head[:4] == b"PAR1" and tail == b"PAR1" and suffix != ".parquet":
        errors.append(f"Parquet training rows are forbidden regardless of filename: {name}")
    if any(marker in head for marker in _PRIVATE_KEY_MARKERS):
        errors.append(f"private-key credentials are forbidden: {name}")
    if len(head) >= 12 and (head[4:8] == b"ftyp" or (head[:4] == b"RIFF" and head[8:12] == b"AVI ")):
        errors.append(f"raw video is forbidden regardless of filename: {name}")
    if head.startswith(b"\x1aE\xdf\xa3"):
        errors.append(f"raw Matroska/WebM video is forbidden regardless of filename: {name}")
    if any(head.startswith(marker) for marker in _BINARY_ARCHIVE_MARKERS):
        errors.append(f"binary archive/model payload is forbidden regardless of filename: {name}")

    if suffix in {".json", ".jsonl", ".md", ".patch", ".txt", ".svg"}:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"text review-pack payload is not valid UTF-8 ({name}): {exc}")
            text = ""
        if "\x00" in text:
            errors.append(f"text review-pack payload contains binary NUL bytes: {name}")
        if suffix == ".json" and text:
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(f"review-pack JSON is invalid ({name}): {exc}")
            else:
                forbidden = sorted(_walk_json_keys(decoded) & _FORBIDDEN_JSON_KEYS)
                if forbidden:
                    errors.append(f"forbidden full decisions/weights/embeddings JSON keys in {name}: {forbidden}")
        elif suffix == ".jsonl" and text:
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    decoded = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"review-pack JSONL is invalid ({name}:{line_number}): {exc}")
                    break
                forbidden = sorted(_walk_json_keys(decoded) & _FORBIDDEN_JSON_KEYS)
                if forbidden:
                    errors.append(
                        f"forbidden full decisions/weights/embeddings JSON keys in {name}:{line_number}: {forbidden}"
                    )
                    break
        if suffix == ".svg" and text and "<svg" not in text[:2048].casefold():
            errors.append(f"SVG visual does not contain an SVG root: {name}")
    elif suffix == ".pdf" and not head.startswith(b"%PDF-"):
        errors.append(f"PDF visual has invalid magic bytes: {name}")
    elif suffix in RASTER_VISUAL_SUFFIXES:
        try:
            from PIL import Image

            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            errors.append(f"raster visual cannot be decoded ({name}): {exc}")
    return errors


def _normalise_sources(source_paths: Iterable[Path]) -> tuple[Path, ...]:
    sources = tuple(Path(path).resolve() for path in source_paths)
    errors: list[str] = []
    names_seen: dict[str, Path] = {}
    for source in sources:
        if not source.is_file():
            errors.append(f"review-pack source is not a file: {source}")
            continue
        if source.name == REVIEW_PACK_MANIFEST_NAME:
            errors.append(f"the manifest is generated and cannot be supplied as a source: {source}")
        casefolded = source.name.casefold()
        if casefolded in names_seen:
            errors.append(f"duplicate flat review-pack filename {source.name!r}: {names_seen[casefolded]} and {source}")
        else:
            names_seen[casefolded] = source
        errors.extend(_payload_type_errors(source))
    names = {source.name for source in sources}
    if REQUIRED_SOURCE_DIFF_NAME not in names:
        errors.append(f"required source diff is missing: {REQUIRED_SOURCE_DIFF_NAME}")
    if len(sources) + 1 > MAX_REVIEW_PACK_FILES:
        errors.append(f"file count including generated manifest {len(sources) + 1} exceeds {MAX_REVIEW_PACK_FILES}")
    visual_count = sum(source.suffix.casefold() in VISUAL_SUFFIXES for source in sources)
    if visual_count > MAX_REVIEW_PACK_VISUALS:
        errors.append(f"visual count {visual_count} exceeds {MAX_REVIEW_PACK_VISUALS}")
    if errors:
        raise ReviewPackValidationError(errors)
    return tuple(sorted(sources, key=lambda path: (path.name.casefold(), path.name)))


def _manifest_for_sources(sources: Sequence[Path]) -> dict[str, Any]:
    rows = [
        {
            "filename": source.name,
            "byte_size": source.stat().st_size,
            "sha256": sha256_file(source),
        }
        for source in sources
    ]
    payload_bytes = sum(row["byte_size"] for row in rows)
    safety = stage_safety_summary()
    return {
        "schema_version": REVIEW_PACK_MANIFEST_SCHEMA_VERSION,
        "flat": True,
        "file_count_including_manifest": len(rows) + 1,
        "payload_file_count": len(rows),
        "payload_total_bytes": payload_bytes,
        "visual_file_count": sum(Path(row["filename"]).suffix.casefold() in VISUAL_SUFFIXES for row in rows),
        "maximum_file_count": MAX_REVIEW_PACK_FILES,
        "maximum_total_bytes": MAX_REVIEW_PACK_BYTES,
        "maximum_visual_files": MAX_REVIEW_PACK_VISUALS,
        "source_diff_required": True,
        "source_diff_present": any(row["filename"] == REQUIRED_SOURCE_DIFF_NAME for row in rows),
        "manifest_self_hash_omitted": True,
        "files": rows,
        "safety": safety,
        **safety,
    }


def _manifest_errors(root: Path, payload_files: Sequence[Path]) -> list[str]:
    manifest_path = root / REVIEW_PACK_MANIFEST_NAME
    if not manifest_path.is_file():
        return [f"required manifest is missing: {REVIEW_PACK_MANIFEST_NAME}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"review-pack manifest is not valid UTF-8 JSON: {exc}"]
    if not isinstance(manifest, dict):
        return ["review-pack manifest must contain a JSON object"]
    errors: list[str] = []
    if manifest.get("schema_version") != REVIEW_PACK_MANIFEST_SCHEMA_VERSION:
        errors.append("review-pack manifest schema_version differs from the G7A contract")
    if manifest.get("flat") is not True:
        errors.append("review-pack manifest must declare flat=true")
    if manifest.get("manifest_self_hash_omitted") is not True:
        errors.append("review-pack manifest must declare that its self hash is omitted")
    rows = manifest.get("files")
    if not isinstance(rows, list):
        return [*errors, "review-pack manifest files must be a list"]
    row_names: list[str] = []
    row_by_name: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"review-pack manifest row {index} must be an object")
            continue
        name = row.get("filename")
        if not isinstance(name, str) or not name or "/" in name or "\\" in name:
            errors.append(f"review-pack manifest row {index} has a non-flat filename")
            continue
        row_names.append(name)
        if name in row_by_name:
            errors.append(f"review-pack manifest contains duplicate filename: {name}")
        row_by_name[name] = row
    if REVIEW_PACK_MANIFEST_NAME in row_by_name:
        errors.append("review-pack manifest must omit its own SHA-256/size row")
    expected_names = [path.name for path in payload_files]
    if row_names != expected_names:
        errors.append(f"review-pack manifest filenames differ: expected {expected_names}, found {row_names}")
    for path in payload_files:
        row = row_by_name.get(path.name)
        if row is None:
            continue
        if row.get("byte_size") != path.stat().st_size:
            errors.append(f"review-pack manifest byte size mismatch: {path.name}")
        if row.get("sha256") != sha256_file(path):
            errors.append(f"review-pack manifest SHA-256 mismatch: {path.name}")
    if manifest.get("payload_file_count") != len(payload_files):
        errors.append("review-pack manifest payload_file_count mismatch")
    if manifest.get("file_count_including_manifest") != len(payload_files) + 1:
        errors.append("review-pack manifest file_count_including_manifest mismatch")
    if manifest.get("payload_total_bytes") != sum(path.stat().st_size for path in payload_files):
        errors.append("review-pack manifest payload_total_bytes mismatch")
    if manifest.get("visual_file_count") != sum(path.suffix.casefold() in VISUAL_SUFFIXES for path in payload_files):
        errors.append("review-pack manifest visual_file_count mismatch")
    if manifest.get("source_diff_present") is not True:
        errors.append("review-pack manifest must record the required source diff")
    expected_safety = stage_safety_summary()
    if manifest.get("safety") != expected_safety:
        errors.append("review-pack manifest safety summary differs from the hardcoded G7A boundary")
    for key, value in expected_safety.items():
        if manifest.get(key) != value:
            errors.append(f"review-pack manifest safety field mismatch: {key}")
    return errors


def _review_pack_audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    if not root.exists() or not root.is_dir():
        errors.append(f"review-pack directory does not exist: {root}")
        files: list[Path] = []
        nested_entries: list[Path] = []
    else:
        entries = sorted(root.iterdir(), key=lambda path: (path.name.casefold(), path.name))
        files = [path for path in entries if path.is_file()]
        nested_entries = [path for path in entries if not path.is_file()]
    if nested_entries:
        errors.append(f"review pack must be flat; nested entries found: {[path.name for path in nested_entries]}")
    if len(files) > MAX_REVIEW_PACK_FILES:
        errors.append(f"file count {len(files)} exceeds {MAX_REVIEW_PACK_FILES}")
    total_bytes = sum(path.stat().st_size for path in files)
    if total_bytes > MAX_REVIEW_PACK_BYTES:
        errors.append(f"total bytes {total_bytes} exceeds {MAX_REVIEW_PACK_BYTES}")
    names = {path.name for path in files}
    if REQUIRED_SOURCE_DIFF_NAME not in names:
        errors.append(f"required source diff is missing: {REQUIRED_SOURCE_DIFF_NAME}")
    visual_count = sum(path.suffix.casefold() in VISUAL_SUFFIXES for path in files)
    if visual_count > MAX_REVIEW_PACK_VISUALS:
        errors.append(f"visual count {visual_count} exceeds {MAX_REVIEW_PACK_VISUALS}")
    payload_files = [path for path in files if path.name != REVIEW_PACK_MANIFEST_NAME]
    for path in payload_files:
        errors.extend(_payload_type_errors(path))
    if root.is_dir():
        errors.extend(_manifest_errors(root, payload_files))
    safety = stage_safety_summary()
    return {
        "schema_version": REVIEW_PACK_VALIDATION_SCHEMA_VERSION,
        "passed": not errors,
        "errors": errors,
        "flat": not nested_entries,
        "file_count": len(files),
        "payload_file_count": len(payload_files),
        "total_bytes": total_bytes,
        "visual_file_count": visual_count,
        "source_diff_present": REQUIRED_SOURCE_DIFF_NAME in names,
        "manifest_self_hash_omitted": REVIEW_PACK_MANIFEST_NAME not in {path.name for path in payload_files},
        "files": [
            {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in payload_files
        ],
        "safety": safety,
        **safety,
    }


def review_pack_validation_errors(root: Path) -> tuple[str, ...]:
    """Return deterministic contract errors without raising."""

    return tuple(_review_pack_audit(root)["errors"])


def validate_review_pack(root: Path) -> dict[str, Any]:
    """Validate a completed flat review pack and return its audit."""

    audit = _review_pack_audit(root)
    if audit["errors"]:
        raise ReviewPackValidationError(audit["errors"])
    return audit


def assemble_review_pack(source_paths: Iterable[Path], output_root: Path) -> dict[str, Any]:
    """Atomically assemble a deterministic flat review pack using source basenames."""

    sources = _normalise_sources(source_paths)
    manifest = _manifest_for_sources(sources)
    manifest_bytes = _json_bytes(manifest)
    total_bytes = sum(source.stat().st_size for source in sources) + len(manifest_bytes)
    if total_bytes > MAX_REVIEW_PACK_BYTES:
        raise ReviewPackValidationError(
            [f"total bytes including generated manifest {total_bytes} exceeds {MAX_REVIEW_PACK_BYTES}"]
        )

    output_root = output_root.resolve()
    if output_root.exists():
        if not output_root.is_dir():
            raise ReviewPackValidationError([f"review-pack output is not a directory: {output_root}"])
        existing = sorted(path.name for path in output_root.iterdir())
        if existing:
            raise ReviewPackValidationError(
                [f"refusing to replace non-empty review-pack output {output_root}: {existing}"]
            )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", suffix=".tmp", dir=output_root.parent)).resolve()
    committed = False
    try:
        for source in sources:
            _atomic_copy(source, temporary / source.name)
        _atomic_write_bytes(temporary / REVIEW_PACK_MANIFEST_NAME, manifest_bytes)
        validate_review_pack(temporary)
        if output_root.exists():
            output_root.rmdir()
        os.replace(temporary, output_root)
        committed = True
    finally:
        if not committed and temporary.exists():
            shutil.rmtree(temporary)
    validate_review_pack(output_root)
    return manifest


build_review_pack = assemble_review_pack


__all__ = [
    "MAX_REVIEW_PACK_BYTES",
    "MAX_REVIEW_PACK_FILES",
    "MAX_REVIEW_PACK_VISUALS",
    "RASTER_VISUAL_SUFFIXES",
    "REQUIRED_SOURCE_DIFF_NAME",
    "REVIEW_PACK_MANIFEST_NAME",
    "ReviewPackValidationError",
    "assemble_review_pack",
    "build_review_pack",
    "review_pack_validation_errors",
    "sha256_file",
    "stage_safety_summary",
    "validate_review_pack",
]
