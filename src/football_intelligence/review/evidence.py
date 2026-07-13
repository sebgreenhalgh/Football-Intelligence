from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2

from football_intelligence.core.fingerprints import sha256_file
from football_intelligence.review.schemas import (
    CONTINUITY_DECISIONS,
    CONTINUITY_QUESTION,
    EvidenceAsset,
    EvidenceManifest,
    ReviewCase,
    ReviewManifest,
    SourceArtifactReference,
    safety_payload,
    stable_hash,
    utc_now,
)

try:  # Pillow is already present through the media stack in this project environment.
    from PIL import Image
except ImportError:  # pragma: no cover - exercised only in unusually lean installs.
    Image = None  # type: ignore[assignment]


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("payload rows must be a list")
    return [row for row in rows if isinstance(row, dict)]


def _bbox(row: dict[str, Any]) -> dict[str, float]:
    bbox = row.get("bbox")
    if not isinstance(bbox, dict):
        raise ValueError(f"visible-person row has no bbox: {row.get('visible_person_base_id')}")
    return {key: float(bbox[key]) for key in ("x1", "y1", "x2", "y2")}


def _frame_record_by_sequence(frame_manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    records = {}
    for frame in frame_manifest.get("frames", []):
        if isinstance(frame, dict):
            records[int(frame.get("sequence", frame.get("frame_sequence", 0)))] = frame
    return records


def _frame_path(frame_root: Path, frame: dict[str, Any]) -> Path:
    relative = str(frame.get("relative_uri", frame.get("filename", "")))
    path = (frame_root / relative).resolve()
    if not path.exists():
        raise FileNotFoundError(f"canonical frame is missing: {path}")
    return path


def _read_frame(path: Path) -> Any:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"frame did not decode: {path}")
    return image


def _bounds_for_bbox(
    bbox: dict[str, float],
    *,
    width: int,
    height: int,
    scale: float,
    min_size: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max((x2 - x1) * scale, float(min_size))
    bh = max((y2 - y1) * scale, float(min_size))
    left = max(0, int(round(cx - bw / 2.0)))
    top = max(0, int(round(cy - bh / 2.0)))
    right = min(width, int(round(cx + bw / 2.0)))
    bottom = min(height, int(round(cy + bh / 2.0)))
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    return left, top, right, bottom


def _draw_box(image: Any, bbox: dict[str, float], *, label: str, color: tuple[int, int, int]) -> Any:
    out = image.copy()
    x1, y1, x2, y2 = [int(round(bbox[key])) for key in ("x1", "y1", "x2", "y2")]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
    cv2.putText(out, label, (max(0, x1), max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return out


def _label_frame(image: Any, label: str) -> Any:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 540), 42), (255, 255, 255), -1)
    cv2.putText(out, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (22, 24, 28), 2, cv2.LINE_AA)
    return out


def _write_jpg(path: Path, image: Any) -> EvidenceAsset:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ValueError(f"failed to write evidence asset: {path}")
    return EvidenceAsset(
        asset_id=path.stem,
        asset_type="image",
        relative_path=path.name,
        sha256=sha256_file(path),
        media_type="image/jpeg",
    )


def _crop(image: Any, bbox: dict[str, float], *, scale: float = 4.0, min_size: int = 180) -> Any:
    height, width = image.shape[:2]
    left, top, right, bottom = _bounds_for_bbox(bbox, width=width, height=height, scale=scale, min_size=min_size)
    return image[top:bottom, left:right]


def _context(image: Any, bbox: dict[str, float], *, label: str, color: tuple[int, int, int]) -> Any:
    height, width = image.shape[:2]
    left, top, right, bottom = _bounds_for_bbox(bbox, width=width, height=height, scale=8.0, min_size=420)
    context = image[top:bottom, left:right].copy()
    translated = {
        "x1": bbox["x1"] - left,
        "y1": bbox["y1"] - top,
        "x2": bbox["x2"] - left,
        "y2": bbox["y2"] - top,
    }
    return _draw_box(context, translated, label=label, color=color)


def _fit_height(image: Any, height: int) -> Any:
    h, w = image.shape[:2]
    if h == height:
        return image
    width = max(1, int(round(w * (height / h))))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _temporal_sequences(source: int, target: int, frame_count: int) -> list[int]:
    start = max(0, min(source, target) - 1)
    end = min(frame_count - 1, max(source, target) + 1)
    return list(range(start, end + 1))


def _write_gif(path: Path, frames: list[Any]) -> EvidenceAsset | None:
    if Image is None or not frames:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    pil_frames[0].save(path, save_all=True, append_images=pil_frames[1:], duration=260, loop=0)
    return EvidenceAsset(
        asset_id=path.stem,
        asset_type="animated_gif",
        relative_path=path.name,
        sha256=sha256_file(path),
        media_type="image/gif",
    )


def _write_mp4(path: Path, frames: list[Any]) -> EvidenceAsset | None:
    if not frames:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (width, height))
    if not writer.isOpened():
        return None
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    if not path.exists() or path.stat().st_size == 0:
        return None
    return EvidenceAsset(
        asset_id=path.stem,
        asset_type="temporal_clip",
        relative_path=path.name,
        sha256=sha256_file(path),
        media_type="video/mp4",
    )


def _source_ref(artifact_id: str, path: Path, role: str) -> SourceArtifactReference:
    return SourceArtifactReference(
        artifact_id=artifact_id,
        path=str(path),
        sha256=sha256_file(path) if path.exists() and path.is_file() else None,
        role=role,
    )


def _case_source_refs(
    *,
    candidate_rows_path: Path,
    frame_manifest_path: Path,
    source_frame_path: Path,
    target_frame_path: Path,
    visible_person_base_path: Path,
) -> list[SourceArtifactReference]:
    return [
        _source_ref("m5_4a_review_candidates", candidate_rows_path, "sealed M5.4A review candidate rows"),
        _source_ref("canonical_frame_manifest", frame_manifest_path, "canonical blind frame manifest"),
        _source_ref("source_raw_frame", source_frame_path, "source raw frame"),
        _source_ref("target_raw_frame", target_frame_path, "target raw frame"),
        _source_ref("visible_person_base_rows", visible_person_base_path, "run-local visible-person base rows"),
    ]


def _review_case_id(row: dict[str, Any], index: int) -> str:
    candidate_id = str(row.get("portable_review_candidate_id") or row.get("candidate_id") or f"case_{index + 1:03d}")
    return candidate_id.replace("portable_review_", "m5_4b_continuity_")


def _candidate_hash(row: dict[str, Any]) -> str:
    return stable_hash(row)


def _evidence_hash_payload(
    *,
    source_frame: dict[str, Any],
    target_frame: dict[str, Any],
    source_bbox: dict[str, float],
    target_bbox: dict[str, float],
    assets: list[EvidenceAsset],
) -> dict[str, Any]:
    return {
        "source_frame_sequence": source_frame.get("sequence"),
        "target_frame_sequence": target_frame.get("sequence"),
        "source_frame_byte_sha256": source_frame.get("byte_sha256"),
        "target_frame_byte_sha256": target_frame.get("byte_sha256"),
        "source_bbox": source_bbox,
        "target_bbox": target_bbox,
        "assets": [asset.model_dump(mode="json") for asset in assets],
    }


def build_visual_continuity_workbench(
    *,
    stage_root: Path,
    source_stage_root: Path,
    frame_manifest_path: Path,
    frame_root: Path,
    candidate_rows_path: Path,
    visible_person_base_path: Path,
) -> dict[str, Any]:
    review_root = stage_root / "review"
    evidence_root = review_root / "evidence"
    workbench_root = review_root / "workbench"
    decision_root = review_root / "decisions"
    evidence_root.mkdir(parents=True, exist_ok=True)
    workbench_root.mkdir(parents=True, exist_ok=True)
    decision_root.mkdir(parents=True, exist_ok=True)

    candidate_payload = read_json(candidate_rows_path)
    candidate_rows = _rows(candidate_payload)
    visible_rows = _rows(read_json(visible_person_base_path))
    visible_by_id = {str(row.get("visible_person_base_id")): row for row in visible_rows}
    frame_manifest = read_json(frame_manifest_path)
    frames = _frame_record_by_sequence(frame_manifest)
    source_refs = [
        _source_ref("m5_4a_stage_root", source_stage_root, "read-only M5.4A source stage"),
        _source_ref("m5_4a_review_candidates", candidate_rows_path, "sealed M5.4A review candidate rows"),
        _source_ref("canonical_frame_manifest", frame_manifest_path, "canonical blind frame manifest"),
        _source_ref("visible_person_base_rows", visible_person_base_path, "run-local visible-person base rows"),
    ]

    review_cases: list[ReviewCase] = []
    evidence_summaries: list[dict[str, Any]] = []
    for index, row in enumerate(candidate_rows):
        review_case_id = _review_case_id(row, index)
        case_dir = evidence_root / review_case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        source_sequence = int(row["source_frame_sequence"])
        target_sequence = int(row["target_frame_sequence"])
        source_frame = frames[source_sequence]
        target_frame = frames[target_sequence]
        source_path = _frame_path(frame_root, source_frame)
        target_path = _frame_path(frame_root, target_frame)
        source_visible = visible_by_id[str(row["source_visible_person_base_id"])]
        target_visible = visible_by_id[str(row["target_visible_person_base_id"])]
        source_bbox = _bbox(source_visible)
        target_bbox = _bbox(target_visible)
        source_image = _read_frame(source_path)
        target_image = _read_frame(target_path)
        assets: list[EvidenceAsset] = []

        source_crop = _crop(
            _draw_box(source_image, source_bbox, label=f"S f{source_sequence}", color=(0, 88, 255)),
            source_bbox,
        )
        target_crop = _crop(
            _draw_box(target_image, target_bbox, label=f"T f{target_sequence}", color=(0, 180, 0)),
            target_bbox,
        )
        assets.append(
            _write_jpg(case_dir / "source_crop.jpg", _label_frame(source_crop, f"Source frame {source_sequence}"))
        )
        assets.append(
            _write_jpg(case_dir / "target_crop.jpg", _label_frame(target_crop, f"Target frame {target_sequence}"))
        )

        source_context = _context(source_image, source_bbox, label=f"S f{source_sequence}", color=(0, 88, 255))
        target_context = _context(target_image, target_bbox, label=f"T f{target_sequence}", color=(0, 180, 0))
        assets.append(
            _write_jpg(
                case_dir / "source_context.jpg",
                _label_frame(source_context, f"Source frame {source_sequence}"),
            )
        )
        assets.append(
            _write_jpg(
                case_dir / "target_context.jpg",
                _label_frame(target_context, f"Target frame {target_sequence}"),
            )
        )

        h = max(source_context.shape[0], target_context.shape[0])
        side_by_side = cv2.hconcat([_fit_height(source_context, h), _fit_height(target_context, h)])
        assets.append(_write_jpg(case_dir / "side_by_side.jpg", side_by_side))

        temporal_frames: list[Any] = []
        for sequence in _temporal_sequences(source_sequence, target_sequence, len(frames)):
            frame = frames[sequence]
            path = _frame_path(frame_root, frame)
            image = _read_frame(path)
            if sequence == source_sequence:
                image = _draw_box(image, source_bbox, label=f"S f{sequence}", color=(0, 88, 255))
            elif sequence == target_sequence:
                image = _draw_box(image, target_bbox, label=f"T f{sequence}", color=(0, 180, 0))
            image = _label_frame(image, f"Frame {sequence}; gap {abs(target_sequence - source_sequence)}")
            temporal_frames.append(cv2.resize(image, (682, 180), interpolation=cv2.INTER_AREA))
        temporal_strip = cv2.hconcat(temporal_frames)
        assets.append(_write_jpg(case_dir / "temporal_strip.jpg", temporal_strip))
        mp4 = _write_mp4(case_dir / "temporal_clip.mp4", temporal_frames)
        if mp4 is not None:
            assets.append(mp4)
        gif = _write_gif(case_dir / "temporal_clip.gif", temporal_frames)
        if gif is not None:
            assets.append(gif)

        for asset in assets:
            asset.frame_sequences = [source_sequence, target_sequence]

        evidence_payload = _evidence_hash_payload(
            source_frame=source_frame,
            target_frame=target_frame,
            source_bbox=source_bbox,
            target_bbox=target_bbox,
            assets=assets,
        )
        evidence_hash = stable_hash(evidence_payload)
        evidence_manifest = EvidenceManifest(
            evidence_id=f"{review_case_id}_evidence",
            evidence_assets=assets,
            source_frame_hashes=[
                {
                    "frame_sequence": source_sequence,
                    "source_frame_uri": str(source_path),
                    "source_frame_byte_sha256": source_frame.get("byte_sha256"),
                    "decoded_pixel_sha256": source_frame.get("decoded_pixel_sha256"),
                },
                {
                    "frame_sequence": target_sequence,
                    "source_frame_uri": str(target_path),
                    "source_frame_byte_sha256": target_frame.get("byte_sha256"),
                    "decoded_pixel_sha256": target_frame.get("decoded_pixel_sha256"),
                },
            ],
            source_frame_sequence=source_sequence,
            target_frame_sequence=target_sequence,
            source_bbox=source_bbox,
            target_bbox=target_bbox,
            frame_gap=abs(target_sequence - source_sequence),
            temporal_evidence_available=mp4 is not None or gif is not None,
            evidence_hash=evidence_hash,
        )
        write_json(case_dir / "evidence_manifest.json", evidence_manifest.model_dump(mode="json"))
        case_hash = _candidate_hash(row)
        case = ReviewCase(
            review_case_id=review_case_id,
            task_type="visual_continuity_edge_review",
            concise_question=CONTINUITY_QUESTION,
            allowed_decisions=CONTINUITY_DECISIONS,
            candidate_artifact_id=str(row.get("portable_review_candidate_id")),
            source_artifact_references=_case_source_refs(
                candidate_rows_path=candidate_rows_path,
                frame_manifest_path=frame_manifest_path,
                source_frame_path=source_path,
                target_frame_path=target_path,
                visible_person_base_path=visible_person_base_path,
            ),
            source_frame_sequence=source_sequence,
            target_frame_sequence=target_sequence,
            evidence_manifest=evidence_manifest,
            uncertainty_reasons=[str(reason) for reason in row.get("uncertainty_reasons", [])],
            category=str(row.get("review_category", row.get("review_bucket", "continuity_review"))),
            priority=index + 1,
            control_status="not_control",
            candidate_hash=case_hash,
            evidence_hash=evidence_hash,
            safety_payload=safety_payload(),
        )
        review_cases.append(case)
        evidence_summaries.append(
            {
                "review_case_id": review_case_id,
                "asset_count": len(assets),
                "temporal_evidence_available": evidence_manifest.temporal_evidence_available,
                "evidence_hash": evidence_hash,
            }
        )

    candidate_manifest_hash = stable_hash(candidate_payload)
    evidence_manifest_hash = stable_hash([case.evidence_manifest.model_dump(mode="json") for case in review_cases])
    source_manifest_hash = stable_hash(frame_manifest)
    manifest = ReviewManifest(
        title="M5.4B Unified Autosaving Visual Review Workbench",
        review_task_family="visual_continuity",
        review_cases=review_cases,
        candidate_manifest_hash=candidate_manifest_hash,
        evidence_manifest_hash=evidence_manifest_hash,
        source_manifest_hash=source_manifest_hash,
        source_artifact_references=source_refs,
    )
    write_json(review_root / "review_manifest.json", manifest.model_dump(mode="json"))
    write_json(
        review_root / "evidence_manifest_summary.json",
        {
            "artifact": "m5_4b_evidence_manifest_summary",
            "created_at": utc_now(),
            "case_count": len(review_cases),
            "evidence_count": len(evidence_summaries),
            "temporal_evidence_count": sum(1 for item in evidence_summaries if item["temporal_evidence_available"]),
            "candidate_manifest_hash": candidate_manifest_hash,
            "evidence_manifest_hash": evidence_manifest_hash,
            "rows": evidence_summaries,
            **safety_payload(),
        },
    )
    write_json(
        decision_root / "review_decisions.json",
        {
            "schema_version": "m5_4b.review_decisions.v1",
            "created_at": utc_now(),
            "workbench_version": manifest.workbench_version,
            "candidate_manifest_hash": manifest.candidate_manifest_hash,
            "evidence_manifest_hash": manifest.evidence_manifest_hash,
            "reviewer_session_id": None,
            "event_sequence": 0,
            "decisions": {},
            "notes": {},
            "last_viewed_case_id": None,
            "elapsed_active_seconds": 0,
            "completed": False,
            **safety_payload(),
        },
    )
    events_path = decision_root / "review_decision_events.jsonl"
    events_path.write_text("", encoding="utf-8")
    return {
        "manifest_path": str(review_root / "review_manifest.json"),
        "evidence_root": str(evidence_root),
        "workbench_root": str(workbench_root),
        "decision_root": str(decision_root),
        "review_case_count": len(review_cases),
        "evidence_count": len(evidence_summaries),
        "temporal_evidence_count": sum(1 for item in evidence_summaries if item["temporal_evidence_available"]),
        "candidate_manifest_hash": candidate_manifest_hash,
        "evidence_manifest_hash": evidence_manifest_hash,
    }


def copy_workbench_assets(source_root: Path, target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    for path in source_root.iterdir():
        if path.is_file():
            shutil.copy2(path, target_root / path.name)
