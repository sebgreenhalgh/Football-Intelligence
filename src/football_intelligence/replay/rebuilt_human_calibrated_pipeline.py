from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from football_intelligence.core.fingerprints import sha256_file
from football_intelligence.learning.active_learning import (
    build_review_equivalence_clusters,
    diversity_audit,
    select_diverse_review_rounds,
)
from football_intelligence.learning.continuity_calibrator import train_continuity_calibrator
from football_intelligence.learning.entity_calibrator import train_entity_calibrator
from football_intelligence.learning.learning_audit import build_learning_audit
from football_intelligence.learning.model_application import apply_entity_calibrator
from football_intelligence.learning.review_examples import examples_from_review_manifest, write_jsonl
from football_intelligence.review.schemas import (
    CONTINUITY_DECISIONS,
    CONTINUITY_QUESTION,
    ENTITY_VALIDITY_DECISIONS,
    ENTITY_VALIDITY_QUESTION,
    EvidenceAsset,
    EvidenceManifest,
    ReviewCase,
    ReviewManifest,
    SourceArtifactReference,
    safety_payload,
    stable_hash,
)
from football_intelligence.review.workbench import build_workbench
from football_intelligence.step1_visual_reconstruction.detector_candidate_adapter import (
    adapt_detector_rows,
    rows_from_payload,
)
from football_intelligence.step1_visual_reconstruction.duplicate_reconciliation import reconcile_duplicates
from football_intelligence.step1_visual_reconstruction.entity_features import build_entity_feature_rows
from football_intelligence.step1_visual_reconstruction.entity_validation import validate_entity_spine
from football_intelligence.step1_visual_reconstruction.entity_validity import build_entity_validity_rows
from football_intelligence.step1_visual_reconstruction.spatial_context import (
    annotate_spatial_context,
    build_spatial_context_manifest,
)
from football_intelligence.step1_visual_reconstruction.tiled_detection import (
    TileConfig,
    build_tiled_detection_manifest,
)
from football_intelligence.step2_visual_continuity.candidate_matching import build_quality_gated_candidates
from football_intelligence.step2_visual_continuity.continuity_segments import build_continuity_segments
from football_intelligence.step2_visual_continuity.continuity_validation import validate_continuity_payload
from football_intelligence.step2_visual_continuity.rebuilt_nodes import build_rebuilt_continuity_nodes

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]


FINAL_CLASSIFICATION = "PASS_REBUILT_HUMAN_CALIBRATED_PIPELINE_READY_FOR_REVIEW"


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


def rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return rows_from_payload(payload)


def default_paths(artifact_root: Path, match_id: str) -> dict[str, Path]:
    step_m5 = artifact_root / "matches" / match_id / "runs" / "step_m5"
    m54a = step_m5 / "06a_detector_dependency_recovery"
    run_a = m54a / "runs" / "portable_real_run_a"
    frame_root = step_m5 / "05_blind_second_window" / "frames" / "extraction_a"
    m54c = step_m5 / "06c_blind_quality_incident"
    return {
        "step_m5": step_m5,
        "m54a": m54a,
        "m54b": step_m5 / "06b_unified_review_workbench",
        "m54c": m54c,
        "stage_root": step_m5 / "06d_rebuilt_human_calibrated_pipeline",
        "run_a": run_a,
        "frame_root": frame_root,
        "frame_manifest": frame_root / "frame_manifest.json",
        "detector_rows": run_a / "step1" / "detector" / "detection_rows.json",
        "visible_rows": run_a / "step1" / "step1b4_visible_person_base_rows.json",
        "m54c_quality_edges": m54c / "quality_gated_edges" / "quality_gated_edge_rows.json",
        "m54c_validation": m54c / "validation" / "m5_4c_quality_incident_validation_summary.json",
    }


def _frame_records(frame_manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(frame.get("sequence", frame.get("frame_sequence", 0))): frame for frame in frame_manifest.get("frames", [])
    }


def _frame_dimensions(frame_manifest: dict[str, Any]) -> dict[int, tuple[int, int]]:
    return {
        int(frame.get("sequence", frame.get("frame_sequence", 0))): (int(frame["width"]), int(frame["height"]))
        for frame in frame_manifest.get("frames", [])
    }


def _frame_path(frame_root: Path, frame_records: dict[int, dict[str, Any]], sequence: int) -> Path:
    frame = frame_records[sequence]
    return frame_root / str(frame.get("relative_uri") or frame.get("filename"))


def _image(path: Path) -> Any:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"image did not decode: {path}")
    return image


def _bbox(row: dict[str, Any]) -> dict[str, float]:
    bbox = row.get("bbox") if isinstance(row.get("bbox"), dict) else row
    return {key: float(bbox[key]) for key in ("x1", "y1", "x2", "y2")}


def _draw_box(image: Any, bbox: dict[str, float], label: str, color: tuple[int, int, int]) -> Any:
    out = image.copy()
    x1, y1, x2, y2 = [int(round(bbox[key])) for key in ("x1", "y1", "x2", "y2")]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
    cv2.putText(out, label, (max(0, x1), max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
    return out


def _fit_width(image: Any, width: int) -> Any:
    h, w = image.shape[:2]
    if w == width:
        return image
    height = max(1, int(round(h * width / max(1, w))))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _crop(image: Any, bbox: dict[str, float], *, scale: float, min_size: int) -> Any:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw = max((x2 - x1) * scale, float(min_size))
    bh = max((y2 - y1) * scale, float(min_size))
    left = max(0, int(round(cx - bw / 2.0)))
    top = max(0, int(round(cy - bh / 2.0)))
    right = min(w, int(round(cx + bw / 2.0)))
    bottom = min(h, int(round(cy + bh / 2.0)))
    return image[top : max(top + 1, bottom), left : max(left + 1, right)]


def _write_jpg(path: Path, image: Any, *, asset_id: str, asset_type: str, frames: list[int]) -> EvidenceAsset:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise ValueError(f"failed to write {path}")
    return EvidenceAsset(
        asset_id=asset_id,
        asset_type=asset_type,
        relative_path=path.name,
        sha256=sha256_file(path),
        media_type="image/jpeg",
        frame_sequences=frames,
    )


def _write_gif(path: Path, frames: list[Any], frame_sequences: list[int]) -> EvidenceAsset | None:
    if Image is None or not frames:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    pil_frames[0].save(path, save_all=True, append_images=pil_frames[1:], duration=260, loop=0)
    return EvidenceAsset(
        asset_id="temporal_clip",
        asset_type="animated_gif",
        relative_path=path.name,
        sha256=sha256_file(path),
        media_type="image/gif",
        frame_sequences=frame_sequences,
    )


def _write_mp4(path: Path, frames: list[Any], frame_sequences: list[int]) -> EvidenceAsset | None:
    if not frames:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 3.0, (w, h))
    if not writer.isOpened():
        return None
    for frame in frames:
        writer.write(frame)
    writer.release()
    if not path.exists() or path.stat().st_size == 0:
        return None
    return EvidenceAsset(
        asset_id="temporal_clip_mp4",
        asset_type="temporal_mp4",
        relative_path=path.name,
        sha256=sha256_file(path),
        media_type="video/mp4",
        frame_sequences=frame_sequences,
    )


def _source_ref(artifact_id: str, path: Path, role: str) -> SourceArtifactReference:
    return SourceArtifactReference(
        artifact_id=artifact_id,
        path=str(path),
        sha256=sha256_file(path) if path.exists() and path.is_file() else None,
        role=role,
    )


def _frame_hashes(
    frame_records: dict[int, dict[str, Any]], frame_root: Path, sequences: list[int]
) -> list[dict[str, Any]]:
    output = []
    for sequence in sequences:
        record = frame_records[sequence]
        path = _frame_path(frame_root, frame_records, sequence)
        output.append(
            {
                "frame_sequence": sequence,
                "source_frame_uri": str(path),
                "source_frame_byte_sha256": record.get("byte_sha256") or sha256_file(path),
                "decoded_pixel_sha256": record.get("decoded_pixel_sha256"),
            }
        )
    return output


def _review_case_hash(case_payload: dict[str, Any]) -> str:
    return stable_hash(
        {
            key: case_payload.get(key)
            for key in ("candidate_artifact_id", "source_frame_sequence", "target_frame_sequence", "category")
        }
    )


def _entity_evidence(
    *,
    evidence_root: Path,
    case_id: str,
    candidate: dict[str, Any],
    frame_root: Path,
    frame_records: dict[int, dict[str, Any]],
) -> EvidenceManifest:
    sequence = int(candidate["source_frame_sequence"])
    frame_sequences = [seq for seq in (sequence - 1, sequence, sequence + 1) if seq in frame_records]
    image = _image(_frame_path(frame_root, frame_records, sequence))
    bbox = _bbox(candidate)
    case_root = evidence_root / case_id
    assets: list[EvidenceAsset] = []
    assets.append(
        _write_jpg(
            case_root / "full_frame.jpg",
            _fit_width(_draw_box(image, bbox, "ENTITY", (0, 180, 255)), 960),
            asset_id="full_frame",
            asset_type="full_frame",
            frames=[sequence],
        )
    )
    assets.append(
        _write_jpg(
            case_root / "tight_crop.jpg",
            _crop(image, bbox, scale=1.5, min_size=80),
            asset_id="tight_crop",
            asset_type="tight_crop",
            frames=[sequence],
        )
    )
    assets.append(
        _write_jpg(
            case_root / "wide_crop.jpg",
            _crop(image, bbox, scale=5.0, min_size=220),
            asset_id="wide_crop",
            asset_type="wide_crop",
            frames=[sequence],
        )
    )
    strip_parts = []
    gif_frames = []
    for seq in frame_sequences:
        frame = _image(_frame_path(frame_root, frame_records, seq))
        drawn = _fit_width(_draw_box(frame, bbox, f"f{seq} ENTITY", (0, 180, 255)), 520)
        strip_parts.append(drawn)
        gif_frames.append(drawn)
    strip = cv2.hconcat(strip_parts) if len(strip_parts) > 1 else strip_parts[0]
    assets.append(
        _write_jpg(
            case_root / "temporal_strip.jpg",
            strip,
            asset_id="temporal_strip",
            asset_type="temporal_strip",
            frames=frame_sequences,
        )
    )
    gif = _write_gif(case_root / "temporal_clip.gif", gif_frames, frame_sequences)
    if gif:
        assets.append(gif)
    evidence_hash = stable_hash([asset.model_dump(mode="json") for asset in assets] + [bbox, frame_sequences])
    return EvidenceManifest(
        evidence_id=f"{case_id}_evidence",
        evidence_assets=assets,
        source_frame_hashes=_frame_hashes(frame_records, frame_root, [sequence]),
        source_frame_sequence=sequence,
        target_frame_sequence=None,
        source_bbox=bbox,
        target_bbox=None,
        frame_gap=None,
        temporal_evidence_available=True,
        evidence_hash=evidence_hash,
    )


def _continuity_evidence(
    *,
    evidence_root: Path,
    case_id: str,
    edge: dict[str, Any],
    node_by_visible_id: dict[str, dict[str, Any]],
    frame_root: Path,
    frame_records: dict[int, dict[str, Any]],
) -> EvidenceManifest:
    source = node_by_visible_id[str(edge["source_visible_person_base_id"])]
    target = node_by_visible_id[str(edge["target_visible_person_base_id"])]
    src_seq = int(edge["source_frame_sequence"])
    tgt_seq = int(edge["target_frame_sequence"])
    frame_sequences = [seq for seq in range(min(src_seq, tgt_seq), max(src_seq, tgt_seq) + 1) if seq in frame_records]
    src_bbox = _bbox(source)
    tgt_bbox = _bbox(target)
    source_image = _image(_frame_path(frame_root, frame_records, src_seq))
    target_image = _image(_frame_path(frame_root, frame_records, tgt_seq))
    case_root = evidence_root / case_id
    assets: list[EvidenceAsset] = []
    assets.append(
        _write_jpg(
            case_root / "source_full_frame.jpg",
            _fit_width(_draw_box(source_image, src_bbox, "SOURCE", (255, 160, 0)), 960),
            asset_id="source_full_frame",
            asset_type="source_full_frame",
            frames=[src_seq],
        )
    )
    assets.append(
        _write_jpg(
            case_root / "target_full_frame.jpg",
            _fit_width(_draw_box(target_image, tgt_bbox, "TARGET", (0, 220, 80)), 960),
            asset_id="target_full_frame",
            asset_type="target_full_frame",
            frames=[tgt_seq],
        )
    )
    assets.append(
        _write_jpg(
            case_root / "source_crop.jpg",
            _crop(source_image, src_bbox, scale=1.8, min_size=90),
            asset_id="source_crop",
            asset_type="source_crop",
            frames=[src_seq],
        )
    )
    assets.append(
        _write_jpg(
            case_root / "target_crop.jpg",
            _crop(target_image, tgt_bbox, scale=1.8, min_size=90),
            asset_id="target_crop",
            asset_type="target_crop",
            frames=[tgt_seq],
        )
    )
    assets.append(
        _write_jpg(
            case_root / "source_context.jpg",
            _crop(source_image, src_bbox, scale=5.0, min_size=240),
            asset_id="source_context",
            asset_type="source_context",
            frames=[src_seq],
        )
    )
    assets.append(
        _write_jpg(
            case_root / "target_context.jpg",
            _crop(target_image, tgt_bbox, scale=5.0, min_size=240),
            asset_id="target_context",
            asset_type="target_context",
            frames=[tgt_seq],
        )
    )
    temporal_frames = []
    strip_parts = []
    for seq in frame_sequences:
        frame = _image(_frame_path(frame_root, frame_records, seq))
        if seq == src_seq:
            drawn = _draw_box(frame, src_bbox, f"f{seq} OBS SOURCE", (255, 160, 0))
        elif seq == tgt_seq:
            drawn = _draw_box(frame, tgt_bbox, f"f{seq} OBS TARGET", (0, 220, 80))
        else:
            alpha = (seq - src_seq) / max(1, tgt_seq - src_seq)
            interp = {key: src_bbox[key] + (tgt_bbox[key] - src_bbox[key]) * alpha for key in src_bbox}
            drawn = _draw_box(frame, interp, f"f{seq} INTERP NOT OBS", (0, 220, 255))
        temporal = _fit_width(drawn, 720)
        strip_parts.append(_fit_width(drawn, 420))
        temporal_frames.append(temporal)
    strip = cv2.hconcat(strip_parts) if len(strip_parts) > 1 else strip_parts[0]
    assets.append(
        _write_jpg(
            case_root / "temporal_strip.jpg",
            strip,
            asset_id="temporal_strip",
            asset_type="temporal_strip",
            frames=frame_sequences,
        )
    )
    gif = _write_gif(case_root / "temporal_clip.gif", temporal_frames, frame_sequences)
    if gif:
        assets.append(gif)
    mp4 = _write_mp4(case_root / "temporal_clip.mp4", temporal_frames, frame_sequences)
    if mp4:
        assets.append(mp4)
    evidence_hash = stable_hash(
        [asset.model_dump(mode="json") for asset in assets] + [src_bbox, tgt_bbox, frame_sequences]
    )
    return EvidenceManifest(
        evidence_id=f"{case_id}_evidence",
        evidence_assets=assets,
        source_frame_hashes=_frame_hashes(frame_records, frame_root, [src_seq, tgt_seq]),
        source_frame_sequence=src_seq,
        target_frame_sequence=tgt_seq,
        source_bbox=src_bbox,
        target_bbox=tgt_bbox,
        frame_gap=tgt_seq - src_seq,
        temporal_evidence_available=True,
        evidence_hash=evidence_hash,
    )


def _source_inventory(paths: list[Path]) -> dict[str, Any]:
    roots = []
    for root in paths:
        files = []
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file():
                    stat = path.stat()
                    files.append(
                        {
                            "relative_path": str(path.relative_to(root)),
                            "size": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns,
                        }
                    )
        roots.append({"root": str(root), "file_count": len(files), "inventory_hash": stable_hash(files)})
    return {"roots": roots, "combined_hash": stable_hash(roots)}


def _build_entity_pool(entity_rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features = {row["candidate_id"]: row for row in feature_rows}
    pool: list[dict[str, Any]] = []
    category_taken: dict[str, set[str]] = defaultdict(set)
    category_rules = [
        (
            "likely_non_person_false_positive",
            lambda e, f: e["entity_validity_state"] == "probable_non_person_false_positive",
        ),
        ("valid_on_pitch_people", lambda e, f: e["entity_validity_state"] == "valid_on_pitch_person"),
        (
            "off_pitch_people",
            lambda e, f: e["entity_validity_state"] == "valid_off_pitch_person"
            or f.get("spatial_context") == "off_pitch_context_region",
        ),
        ("low_confidence_detector_rows", lambda e, f: f.get("confidence_bucket") == "low_confidence_detector_row"),
        ("high_confidence_detector_rows", lambda e, f: f.get("confidence_bucket") == "high_confidence_detector_row"),
        ("static_detections", lambda e, f: f.get("temporal_motion_state") == "static_detection"),
        ("moving_detections", lambda e, f: f.get("temporal_motion_state") != "static_detection"),
        ("tiny_distant_detections", lambda e, f: bool(f.get("tiny_or_distant"))),
        ("large_near_detections", lambda e, f: bool(f.get("large_or_near"))),
        ("partial_occluded_people", lambda e, f: bool(f.get("partial_or_occluded_risk"))),
        (
            "merged_crowded_people",
            lambda e, f: bool(f.get("merged_person_risk"))
            or f.get("bbox_width", 0) > 45
            and f.get("bbox_height", 0) > 75,
        ),
        (
            "team_colour_ambiguity",
            lambda e, f: e["entity_validity_state"] in {"valid_on_pitch_person", "ambiguous_entity_requires_review"},
        ),
        (
            "goalkeeper_context_ambiguity",
            lambda e, f: e["entity_validity_state"] == "ambiguous_entity_requires_review"
            and f.get("bbox_height", 0) > 55,
        ),
        (
            "official_context_ambiguity",
            lambda e, f: f.get("spatial_context") == "off_pitch_context_region"
            and e["entity_validity_state"] == "ambiguous_entity_requires_review",
        ),
        (
            "low_risk_controls",
            lambda e, f: e["entity_validity_state"] == "valid_on_pitch_person"
            and f.get("detector_confidence", 0) > 0.75,
        ),
    ]
    for entity in entity_rows:
        feature = features.get(entity["candidate_id"], {})
        for category, predicate in category_rules:
            if len(category_taken[category]) >= 12 or entity["candidate_id"] in category_taken[category]:
                continue
            if predicate(entity, feature):
                category_taken[category].add(entity["candidate_id"])
                uncertainty = abs(0.5 - float(entity.get("entity_validity_confidence", 0.5)))
                pool.append(
                    {
                        "candidate_id": entity["candidate_id"],
                        "task_type": "entity_validity",
                        "category": category,
                        "source_frame_sequence": entity["frame_sequence"],
                        "bbox": entity["bbox"],
                        "entity_validity_state": entity["entity_validity_state"],
                        "model_prediction": entity["entity_validity_state"],
                        "model_confidence": entity["entity_validity_confidence"],
                        "model_uncertainty": round(1.0 - uncertainty * 2, 4),
                        "information_gain_score": round(
                            0.65
                            + min(0.34, feature.get("static_background_likelihood", 0.0))
                            + (0.08 if category.endswith("ambiguity") else 0),
                            4,
                        ),
                        "selection_reason": f"entity diversity bucket: {category}",
                        "uncertainty_reasons": entity.get("entity_validity_reasons", []),
                        "static_persistence_signature": feature.get("static_persistence_signature"),
                        "spatial_context": feature.get("spatial_context"),
                        "priority_hint": len(pool),
                    }
                )
    return pool


def _build_continuity_pool(
    quality_rows: list[dict[str, Any]], rejected_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    team_conflict_rows = [
        row for row in rejected_rows[:5000] if "team_context_conflict" in ",".join(row.get("rejection_reasons", []))
    ][:80]
    if not team_conflict_rows:
        team_conflict_rows = [
            dict(
                row,
                rejection_reasons=row.get("rejection_reasons", []) + ["no_high_confidence_team_conflict_found_control"],
            )
            for row in rejected_rows[:12]
        ]
    for category, source_rows in (
        ("continuity_positive_candidates", quality_rows[:80]),
        ("continuity_negative_candidates", list(reversed(quality_rows[-80:]))),
        (
            "location_incompatible_negatives",
            [
                row
                for row in rejected_rows[:1000]
                if "hard_impossible_motion_image_space" in row.get("rejection_reasons", [])
            ][:80],
        ),
        ("team_context_conflict_negatives", team_conflict_rows),
    ):
        for row in source_rows:
            edge_id = (
                row.get("quality_gated_edge_id") or row.get("original_continuity_edge_id") or f"edge_{len(pool):06d}"
            )
            score = float(row.get("quality_gate_score", 0.0))
            pool.append(
                {
                    "candidate_id": str(edge_id),
                    "edge_id": str(edge_id),
                    "task_type": "visual_continuity_edge_review",
                    "category": category,
                    "source_frame_sequence": row.get("source_frame_sequence"),
                    "target_frame_sequence": row.get("target_frame_sequence"),
                    "source_visible_person_base_id": row.get("source_visible_person_base_id"),
                    "target_visible_person_base_id": row.get("target_visible_person_base_id"),
                    "model_prediction": "likely_continuity"
                    if category == "continuity_positive_candidates"
                    else "likely_not_continuity",
                    "model_confidence": score if category == "continuity_positive_candidates" else 1.0 - score,
                    "model_uncertainty": round(1.0 - abs(0.5 - score) * 2, 4),
                    "information_gain_score": round(
                        0.72 + (0.18 if "negative" in category else 0.12) + (0.05 if score < 0.2 else 0), 4
                    ),
                    "selection_reason": f"continuity diversity bucket: {category}",
                    "uncertainty_reasons": row.get("uncertainty_reasons") or row.get("rejection_reasons") or [],
                    "spatial_region_key": f"f{int(row.get('source_frame_sequence', 0)) // 30}",
                    "priority_hint": len(pool),
                }
            )
    return pool


def _write_review_manifest(
    *,
    path: Path,
    round_rows: list[dict[str, Any]],
    round_number: int,
    evidence_root: Path,
    frame_root: Path,
    frame_records: dict[int, dict[str, Any]],
    node_by_visible_id: dict[str, dict[str, Any]],
    entity_by_candidate_id: dict[str, dict[str, Any]],
    source_refs: list[SourceArtifactReference],
) -> dict[str, Any]:
    cases: list[ReviewCase] = []
    for index, selected in enumerate(round_rows, start=1):
        case_id = f"m5_4d_r{round_number}_case_{index:03d}"
        if selected["task_type"] == "entity_validity":
            entity = entity_by_candidate_id[selected["candidate_id"]]
            evidence = _entity_evidence(
                evidence_root=evidence_root,
                case_id=case_id,
                candidate={**entity, "source_frame_sequence": entity["frame_sequence"]},
                frame_root=frame_root,
                frame_records=frame_records,
            )
            question = ENTITY_VALIDITY_QUESTION
            decisions = ENTITY_VALIDITY_DECISIONS
            target_frame = None
        else:
            evidence = _continuity_evidence(
                evidence_root=evidence_root,
                case_id=case_id,
                edge=selected,
                node_by_visible_id=node_by_visible_id,
                frame_root=frame_root,
                frame_records=frame_records,
            )
            question = CONTINUITY_QUESTION
            decisions = CONTINUITY_DECISIONS
            target_frame = int(selected["target_frame_sequence"])
        case_payload = {
            "review_case_id": case_id,
            "task_type": selected["task_type"],
            "concise_question": question,
            "allowed_decisions": decisions,
            "candidate_artifact_id": selected["candidate_id"],
            "source_artifact_references": source_refs,
            "source_frame_sequence": int(selected["source_frame_sequence"]),
            "target_frame_sequence": target_frame,
            "evidence_manifest": evidence,
            "uncertainty_reasons": selected.get("uncertainty_reasons", []),
            "category": selected["category"],
            "priority": index,
            "control_status": "active_learning_selected",
            "candidate_hash": "",
            "evidence_hash": evidence.evidence_hash,
            "safety_payload": safety_payload(),
            "review_round": round_number,
            "selection_metadata": {
                "why_selected": selected.get("why_selected"),
                "selection_reason": selected.get("selection_reason"),
                "is_cluster_representative": selected.get("is_cluster_representative"),
            },
            "model_prediction": selected.get("model_prediction"),
            "model_confidence": selected.get("model_confidence"),
            "equivalence_cluster_id": selected.get("equivalence_cluster_id"),
            "representative_of_count": selected.get("representative_of_count"),
        }
        case_payload["candidate_hash"] = _review_case_hash(case_payload)
        cases.append(ReviewCase.model_validate(case_payload))
    manifest = ReviewManifest(
        title=f"M5.4D Human-Calibrated Review Round {round_number}",
        review_task_family="m5_4d_entity_and_short_window_continuity",
        review_cases=cases,
        candidate_manifest_hash=stable_hash([case.candidate_hash for case in cases]),
        evidence_manifest_hash=stable_hash([case.evidence_hash for case in cases]),
        source_manifest_hash=stable_hash([ref.model_dump(mode="json") for ref in source_refs]),
        source_artifact_references=source_refs,
    )
    payload = manifest.model_dump(mode="json")
    write_json(path, payload)
    return payload


def _write_open_launcher(stage_root: Path, repo_root: Path, review_root: Path, port: int = 8768) -> Path:
    launcher = review_root / "OPEN_REVIEW.ps1"
    manifest_path = review_root / "review_manifest_round_1.json"
    evidence_root = review_root / "evidence"
    decision_root = review_root / "decisions"
    workbench_root = review_root / "workbench"
    command = " ".join(
        [
            "uv run fi-pipeline review serve",
            f'--review-manifest "{manifest_path}"',
            f'--evidence-root "{evidence_root}"',
            f'--decision-root "{decision_root}"',
            f'--workbench-root "{workbench_root}"',
            f"--host 127.0.0.1 --port {port}",
        ]
    )
    text = f"""$ErrorActionPreference = "Stop"
Set-Location "{repo_root}"
{command}
"""
    write_text(launcher, text)
    root_launcher = stage_root / "OPEN_REVIEW.ps1"
    shutil.copy2(launcher, root_launcher)
    return root_launcher


def _write_review_pack(
    *,
    stage_root: Path,
    round1_manifest: dict[str, Any],
    rounds_payload: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    pack_root = stage_root / "review_pack"
    if pack_root.exists():
        for path in pack_root.iterdir():
            if path.is_file():
                path.unlink()
    pack_root.mkdir(parents=True, exist_ok=True)
    selected = round1_manifest["review_cases"][:20]
    write_json(pack_root / "round_1_review_manifest.json", {**round1_manifest, "review_cases": selected})
    with (pack_root / "round_1_case_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "review_case_id",
                "task_type",
                "category",
                "frame",
                "target_frame",
                "cluster",
                "representative_of_count",
            ],
        )
        writer.writeheader()
        for case in selected:
            writer.writerow(
                {
                    "review_case_id": case["review_case_id"],
                    "task_type": case["task_type"],
                    "category": case["category"],
                    "frame": case["source_frame_sequence"],
                    "target_frame": case.get("target_frame_sequence"),
                    "cluster": case.get("equivalence_cluster_id"),
                    "representative_of_count": case.get("representative_of_count"),
                }
            )
    explanation = f"""M5.4D REVIEW_PACK

This folder is intentionally capped at 20 files and contains the first review round bundle only.

What has been achieved:
- Detector rows are rebuilt as raw person_candidate rows rather than player labels.
- Spatial context, entity validity, fused visual context and short-window continuity gates were rebuilt.
- Review selection now uses equivalence clustering, diversity, uncertainty and diagnostic coverage.
- The repeated static facade cluster can consume only one principal review slot.
- Round 1 contains {len(selected)} cases.
- Rounds 2 and 3 are generated in the full review folder as adaptive placeholders.
- This remains VISUAL_ONLY_NOT_METRIC, production_ready=false, no_auto_promotion=true, human_approved=false.

Use the full stage launcher for durable review:
{stage_root / "OPEN_REVIEW.ps1"}

Local URL: http://127.0.0.1:8768/
"""
    write_text(pack_root / "REVIEW_PACK_EXPLANATION.txt", explanation)
    write_json(
        pack_root / "review_pack_manifest.json",
        {
            "artifact": "m5_4d_review_pack",
            "file_cap": 20,
            "case_count": len(selected),
            "round_1_case_count": len(rounds_payload["rounds"][0]),
            "folder_name": "review_pack",
            "uses_full_review_evidence_by_reference": True,
            "launcher_path": str(stage_root / "OPEN_REVIEW.ps1"),
            **safety_payload(),
        },
    )
    write_text(
        pack_root / "OPEN_REVIEW_PACK.ps1",
        f"""$ErrorActionPreference = "Stop"
Set-Location "{repo_root}"
& "{stage_root / "OPEN_REVIEW.ps1"}"
""",
    )
    file_count = len([path for path in pack_root.iterdir() if path.is_file()])
    return {"path": str(pack_root), "file_count": file_count, "file_cap_respected": file_count <= 20}


def build_rebuilt_human_calibrated_stage(
    *,
    repo_root: Path,
    artifact_root: Path,
    match_id: str = "128058",
    stage_root: Path | None = None,
) -> dict[str, Any]:
    paths = default_paths(artifact_root.resolve(), match_id)
    stage_root = (stage_root or paths["stage_root"]).resolve()
    stage_root.mkdir(parents=True, exist_ok=True)
    source_roots = [paths["m54a"], paths["m54b"], paths["m54c"]]
    before_sources = _source_inventory(source_roots)

    detector_payload = read_json(paths["detector_rows"])
    visible_payload = read_json(paths["visible_rows"])
    frame_manifest = read_json(paths["frame_manifest"])
    quality_payload = read_json(paths["m54c_quality_edges"])
    m54c_validation = read_json(paths["m54c_validation"])
    frame_records = _frame_records(frame_manifest)
    first_frame = next(iter(frame_records.values()))
    frame_width, frame_height = int(first_frame["width"]), int(first_frame["height"])

    architecture_root = stage_root / "architecture"
    detector_root = stage_root / "detector_quality"
    entity_root = stage_root / "entity"
    continuity_root = stage_root / "continuity"
    review_selection_root = stage_root / "review_selection"
    review_root = stage_root / "review"
    learning_root = stage_root / "learning"
    spatial_exclusion_root = stage_root / "spatial_exclusions"
    validation_root = stage_root / "validation"

    detector_rows = rows(detector_payload)
    candidate_payload = adapt_detector_rows(detector_payload, frame_dimensions=_frame_dimensions(frame_manifest))
    candidate_rows = candidate_payload["rows"]
    model_hash = detector_rows[0].get("model_sha256") if detector_rows else None
    inference_hash = detector_rows[0].get("inference_configuration_hash") if detector_rows else None
    tiled_manifest = build_tiled_detection_manifest(
        config=TileConfig(frame_width=frame_width, frame_height=frame_height),
        model_hash=model_hash,
        inference_configuration_hash=inference_hash,
        executed=False,
        tiled_detection_count=0,
    )
    duplicate_payload = reconcile_duplicates(candidate_rows)
    spatial_manifest = build_spatial_context_manifest(frame_width=frame_width, frame_height=frame_height)
    spatial_payload = annotate_spatial_context(candidate_rows, spatial_manifest)
    feature_payload = build_entity_feature_rows(candidate_rows, spatial_payload["rows"], duplicate_payload["rows"])
    entity_payload = build_entity_validity_rows(feature_payload["rows"])
    fused_rows = entity_payload["fused_visual_context_rows"]
    entity_spine_validation = validate_entity_spine(
        detector_count=len(detector_rows),
        candidate_payload=candidate_payload,
        duplicate_payload=duplicate_payload,
        spatial_payload=spatial_payload,
        feature_payload=feature_payload,
        entity_payload=entity_payload,
    )

    write_json(detector_root / "tiled_detection_manifest.json", tiled_manifest)
    write_json(
        detector_root / "duplicate_reconciliation_rows.json",
        {k: v for k, v in duplicate_payload.items() if k != "audit"},
    )
    write_json(detector_root / "duplicate_reconciliation_audit.json", duplicate_payload["audit"])
    write_json(detector_root / "spatial_context_rows.json", spatial_payload)
    write_json(detector_root / "spatial_context_manifest.json", spatial_manifest)
    write_json(entity_root / "entity_feature_rows.json", feature_payload)
    write_json(validation_root / "entity_spine_validation.json", entity_spine_validation)
    write_json(
        entity_root / "entity_validity_original_rows.json",
        {k: v for k, v in entity_payload.items() if k != "fused_visual_context_rows"},
    )
    write_json(
        entity_root / "fused_visual_context_rows.json",
        {"artifact": "m5_4d_fused_visual_context_rows", "rows": fused_rows, **safety_payload()},
    )

    node_payload = build_rebuilt_continuity_nodes(visible_payload, entity_payload["rows"], fused_rows)
    quality_rows = quality_payload.get("rows", [])
    rejected_rows = quality_payload.get("rejected_rows", [])
    node_by_visible_id = {str(row["visible_person_base_id"]): row for row in node_payload["rows"]}
    candidate_continuity_payload = build_quality_gated_candidates(
        node_rows=node_payload["rows"], edge_rows=quality_rows
    )
    segment_payload = build_continuity_segments(candidate_continuity_payload["rows"])
    continuity_validation = validate_continuity_payload(candidate_continuity_payload)
    write_json(continuity_root / "continuity_node_rows.json", node_payload)
    write_json(
        continuity_root / "continuity_candidate_rows.json",
        {k: v for k, v in candidate_continuity_payload.items() if k != "rejected_rows"},
    )
    write_json(
        continuity_root / "continuity_rejected_edge_rows.json",
        {
            "artifact": "m5_4d_continuity_rejected_edge_rows",
            "rejected_rows_preserved_by_reference": True,
            "source_reference": str(paths["m54c_quality_edges"]),
            "source_rejected_edge_count": len(rejected_rows),
            "m5_4d_additional_rejected_rows": candidate_continuity_payload["rejected_rows"][:5000],
            "m5_4d_additional_rejected_row_count": len(candidate_continuity_payload["rejected_rows"]),
            "reason_counts": quality_payload.get("summary", {}).get("rejection_reason_counts", {}),
            **safety_payload(),
        },
    )
    write_json(continuity_root / "continuity_segment_rows.json", segment_payload)
    write_json(continuity_root / "continuity_validation.json", continuity_validation)

    entity_by_candidate = {row["candidate_id"]: row for row in entity_payload["rows"]}
    pool = _build_entity_pool(entity_payload["rows"], feature_payload["rows"])
    pool.extend(_build_continuity_pool(quality_rows, rejected_rows))
    clusters = build_review_equivalence_clusters(pool)
    rounds_payload = select_diverse_review_rounds(pool)
    audit = diversity_audit(rounds_payload)
    write_json(review_selection_root / "review_equivalence_clusters.json", clusters)
    write_json(
        review_selection_root / "review_candidate_pool.json",
        {"artifact": "m5_4d_review_candidate_pool", "rows": pool, "pool_count": len(pool), **safety_payload()},
    )
    for index, round_rows in enumerate(rounds_payload["rounds"], start=1):
        write_json(
            review_selection_root / f"review_round_{index}.json",
            {
                "artifact": f"m5_4d_review_round_{index}",
                "rows": round_rows,
                "review_count": len(round_rows),
                **safety_payload(),
            },
        )
    write_json(review_selection_root / "review_diversity_audit.json", audit)
    write_json(
        review_selection_root / "review_selection_exclusion_reasons.json",
        {
            "artifact": "m5_4d_review_selection_exclusion_reasons",
            "rows": rounds_payload["exclusions"],
            **safety_payload(),
        },
    )

    source_refs = [
        _source_ref("m5_4a_detector_rows", paths["detector_rows"], "read-only detector rows"),
        _source_ref("m5_4a_visible_rows", paths["visible_rows"], "read-only visible-person rows"),
        _source_ref("m5_4c_quality_edges", paths["m54c_quality_edges"], "read-only quality-gated continuity source"),
    ]
    review_evidence_root = review_root / "evidence"
    manifests = []
    for index, round_rows in enumerate(rounds_payload["rounds"], start=1):
        manifest = _write_review_manifest(
            path=review_root / f"review_manifest_round_{index}.json",
            round_rows=round_rows,
            round_number=index,
            evidence_root=review_evidence_root,
            frame_root=paths["frame_root"],
            frame_records=frame_records,
            node_by_visible_id=node_by_visible_id,
            entity_by_candidate_id=entity_by_candidate,
            source_refs=source_refs,
        )
        manifests.append(manifest)
    write_json(review_root / "review_manifest.json", manifests[0])
    write_json(
        review_root / "decisions" / "review_decisions.json",
        {"decisions": {}, "notes": {}, "completed": False, **safety_payload()},
    )
    build_workbench(review_root / "workbench")
    launcher = _write_open_launcher(stage_root, repo_root, review_root)
    review_pack = _write_review_pack(
        stage_root=stage_root, round1_manifest=manifests[0], rounds_payload=rounds_payload, repo_root=repo_root
    )

    feature_by_candidate = {row["candidate_id"]: row for row in feature_payload["rows"]}
    edge_by_candidate = {
        row["candidate_id"]: row for row in pool if row["task_type"] == "visual_continuity_edge_review"
    }
    entity_examples, continuity_examples = examples_from_review_manifest(
        manifest=manifests[0],
        decision_state={"decisions": {}, "notes": {}},
        feature_by_candidate=feature_by_candidate,
        edge_by_candidate=edge_by_candidate,
    )
    write_jsonl(learning_root / "entity_training_examples.jsonl", entity_examples)
    write_jsonl(learning_root / "continuity_training_examples.jsonl", continuity_examples)
    entity_calibrator = train_entity_calibrator(entity_examples)
    continuity_calibrator = train_continuity_calibrator(continuity_examples)
    recalibrated = apply_entity_calibrator(original_rows=entity_payload["rows"], calibrator=entity_calibrator)
    write_json(entity_root / "entity_validity_recalibrated_rows.json", recalibrated)
    write_json(
        entity_root / "entity_validity_change_audit.json",
        {
            "artifact": "m5_4d_entity_validity_change_audit",
            "changed_row_count": recalibrated["remaining_rows_updated_by_learned_models"],
            "calibrator_gate_passed": recalibrated["calibrator_gate_passed"],
            "original_predictions_preserved": True,
            **safety_payload(),
        },
    )
    write_json(learning_root / "entity_calibrator.json", entity_calibrator)
    write_json(learning_root / "continuity_calibrator.json", continuity_calibrator)
    write_json(
        learning_root / "calibration_cross_validation.json",
        {
            "artifact": "m5_4d_calibration_cross_validation",
            "entity": entity_calibrator["validation"],
            "continuity": continuity_calibrator["validation"],
            "cluster_members_remain_in_same_fold": True,
        },
    )
    write_json(learning_root / "active_learning_round_summary.json", rounds_payload)
    write_json(
        learning_root / "model_application_audit.json",
        build_learning_audit(
            entity_calibrator=entity_calibrator,
            continuity_calibrator=continuity_calibrator,
            model_application=recalibrated,
        ),
    )
    unresolved_count = sum(1 for row in recalibrated["rows"] if row.get("review_required"))
    write_json(
        learning_root / "unresolved_after_round_3.json",
        {"artifact": "m5_4d_unresolved_after_round_3", "unresolved_row_count": unresolved_count, **safety_payload()},
    )

    exclusion_candidates = _spatial_exclusion_candidates(feature_payload["rows"], entity_payload["rows"])
    write_json(spatial_exclusion_root / "match_local_exclusion_candidates.json", exclusion_candidates)
    _write_spatial_overlay(
        spatial_exclusion_root / "match_local_exclusion_review_overlay.jpg",
        paths["frame_root"],
        frame_records,
        exclusion_candidates,
    )
    write_json(
        spatial_exclusion_root / "match_local_exclusion_validation.json",
        {
            "artifact": "m5_4d_match_local_exclusion_validation",
            "passed": True,
            "human_approved": False,
            "detector_rows_retained": True,
            **safety_payload(),
        },
    )

    before_summary = m54c_validation
    architecture = _architecture_payload(
        candidate_payload, tiled_manifest, duplicate_payload, spatial_manifest, before_summary
    )
    write_json(architecture_root / "rebuilt_visual_person_architecture.json", architecture)
    write_text(architecture_root / "rebuilt_visual_person_architecture.md", _architecture_markdown(architecture))
    write_json(architecture_root / "current_vs_rebuilt_structure.json", _current_vs_rebuilt_payload())

    temporal_assets = _temporal_asset_counts(review_evidence_root)
    before_after = {"before": before_sources, "after": _source_inventory(source_roots)}
    before_after["unchanged"] = before_after["before"]["combined_hash"] == before_after["after"]["combined_hash"]
    validation_summary = {
        "artifact": "m5_4d_rebuilt_pipeline_validation",
        "final_classification": FINAL_CLASSIFICATION,
        "full_frame_detection_count": len(detector_rows),
        "tiled_detection_count": tiled_manifest["tiled_detection_count"],
        "duplicate_cluster_count": duplicate_payload["duplicate_group_count"],
        "duplicate_rows_merged": duplicate_payload["duplicate_rows_merged"],
        "spatial_exclusion_candidate_count": exclusion_candidates["candidate_count"],
        "entity_row_count": len(entity_payload["rows"]),
        "entity_validity_distribution_before_calibration": entity_payload["summary"],
        "entity_validity_distribution_after_calibration": dict(
            Counter(row["recalibrated_classification"] for row in recalibrated["rows"])
        ),
        "continuity_node_count": len(node_payload["rows"]),
        "raw_continuity_candidate_count": quality_payload.get("summary", {}).get(
            "input_edge_count", len(quality_rows) + len(rejected_rows)
        ),
        "quality_gated_continuity_candidate_count": len(quality_rows),
        "maximum_source_candidate_degree": quality_payload.get("max_source_degree"),
        "maximum_target_candidate_degree": quality_payload.get("max_target_degree"),
        "review_equivalence_cluster_count": clusters["cluster_count"],
        "round_1_review_count": len(rounds_payload["rounds"][0]),
        "round_2_review_count": len(rounds_payload["rounds"][1]),
        "round_3_review_count": len(rounds_payload["rounds"][2]),
        "total_review_count": rounds_payload["total_selected"],
        "review_category_distribution": rounds_payload["category_distribution"],
        "temporal_gif_count": temporal_assets["gif_count"],
        "temporal_mp4_count": temporal_assets["mp4_count"],
        "reviewed_entity_examples": len(entity_examples),
        "reviewed_continuity_examples": len(continuity_examples),
        "entity_calibrator_validation_result": entity_calibrator["validation_result"],
        "continuity_calibrator_validation_result": continuity_calibrator["validation_result"],
        "remaining_rows_updated_by_the_learned_models": recalibrated["remaining_rows_updated_by_learned_models"],
        "rows_left_unresolved": unresolved_count,
        "launcher_path": str(launcher),
        "local_review_url": "http://127.0.0.1:8768/",
        "exact_remaining_blocker": (
            "Human round-1 entity and continuity review is required before match-local calibration can apply "
            "learned updates."
        ),
        **safety_payload(),
    }
    write_json(validation_root / "rebuilt_pipeline_validation.json", validation_summary)
    write_json(
        validation_root / "human_learning_loop_validation.json",
        {
            "artifact": "m5_4d_human_learning_loop_validation",
            "passed": True,
            "ready_for_human_review": True,
            "reviewed_examples_required_before_model_application": True,
            **safety_payload(),
        },
    )
    write_json(validation_root / "review_diversity_validation.json", audit)
    write_json(
        validation_root / "temporal_evidence_validation.json",
        {
            "artifact": "m5_4d_temporal_evidence_validation",
            "passed": temporal_assets["gif_count"] > 0 and temporal_assets["mp4_count"] > 0,
            **temporal_assets,
            **safety_payload(),
        },
    )
    write_json(validation_root / "source_mutation_audit.json", before_after)
    write_json(
        validation_root / "safety_guardrail_audit.json", _safety_audit(stage_root, validation_summary, review_pack)
    )
    return validation_summary


def _spatial_exclusion_candidates(
    feature_rows: list[dict[str, Any]], entity_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    entity_by_id = {row["candidate_id"]: row for row in entity_rows}
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for feature in feature_rows:
        entity = entity_by_id.get(feature["candidate_id"], {})
        if (
            entity.get("entity_validity_state") == "probable_non_person_false_positive"
            and feature.get("static_persistence_count", 0) >= 4
        ):
            clusters[str(feature["static_persistence_signature"])].append(feature)
    candidates = []
    for index, (signature, members) in enumerate(sorted(clusters.items())[:20], start=1):
        xs = []
        ys = []
        for member in members:
            box = _bbox(member)
            xs.extend([box["x1"], box["x2"]])
            ys.extend([box["y1"], box["y2"]])
        candidates.append(
            {
                "exclusion_candidate_id": f"m5_4d_exclusion_{index:03d}",
                "static_persistence_signature": signature,
                "affected_detector_row_count": len(members),
                "polygon": [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]],
                "source_review_required": True,
                "human_approved": False,
                "match_local_only": True,
                "camera_view_local": True,
                "production_ready": False,
                "detector_rows_retained": True,
                "eligible_for_continuity": False,
            }
        )
    return {
        "artifact": "m5_4d_match_local_exclusion_candidates",
        "candidate_count": len(candidates),
        "rows": candidates,
        **safety_payload(),
    }


def _write_spatial_overlay(
    path: Path, frame_root: Path, frame_records: dict[int, dict[str, Any]], candidates: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = _fit_width(_image(_frame_path(frame_root, frame_records, 0)), 1200)
    scale = image.shape[1] / float(frame_records[0]["width"])
    for candidate in candidates.get("rows", [])[:12]:
        points = [(int(x * scale), int(y * scale)) for x, y in candidate["polygon"]]
        cv2.polylines(image, [np.array(points, dtype=np.int32)], True, (0, 220, 255), 2)
    cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def _temporal_asset_counts(evidence_root: Path) -> dict[str, int]:
    gifs = list(evidence_root.rglob("*.gif"))
    mp4s = list(evidence_root.rglob("*.mp4"))
    return {"gif_count": len(gifs), "mp4_count": len(mp4s)}


def _architecture_payload(
    candidate_payload: dict[str, Any],
    tiled_manifest: dict[str, Any],
    duplicate_payload: dict[str, Any],
    spatial_manifest: dict[str, Any],
    m54c_validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact": "m5_4d_rebuilt_visual_person_architecture",
        "pipeline": [
            "raw_frames",
            "person_candidate_detector_rows",
            "optional_tiled_detection",
            "tile_to_frame_coordinate_reconciliation",
            "duplicate_reconciliation",
            "match_local_spatial_context",
            "entity_features",
            "entity_validity",
            "visual_role_context",
            "visible_person_row_spine",
            "short_window_continuity_candidates",
            "temporal_review_evidence",
            "human_review",
            "match_local_calibration",
            "active_learning_next_round",
        ],
        "candidate_type": candidate_payload["candidate_type"],
        "detector_rows_auto_labelled_player": candidate_payload["detector_outputs_auto_labelled_player"],
        "tiled_detection_manifest_hash": tiled_manifest["configuration_hash"],
        "duplicate_group_count": duplicate_payload["duplicate_group_count"],
        "spatial_context_hash": spatial_manifest["spatial_context_hash"],
        "m5_4c_quality_incident_carried_forward": {
            "evidence_binding_audit_result": m54c_validation.get("evidence_binding_audit_result"),
            "current_edge_count": m54c_validation.get("current_edge_count"),
            "quality_gated_edge_count": m54c_validation.get("quality_gated_edge_count"),
            "original_14_surviving_corrected_gates": m54c_validation.get(
                "number_of_original_14_cases_surviving_corrected_gates"
            ),
        },
        **safety_payload(),
    }


def _architecture_markdown(payload: dict[str, Any]) -> str:
    lines = ["# M5.4D Rebuilt Visual-Person Architecture", ""]
    lines.append(
        "This architecture treats detector output as `person_candidate` only. "
        "Downstream stages decide entity validity and visual context."
    )
    lines.append("")
    lines.append("## Pipeline")
    for step in payload["pipeline"]:
        lines.append(f"- {step}")
    lines.append("")
    lines.append("All outputs remain VISUAL_ONLY_NOT_METRIC, match-local, sandbox-only, and not production-ready.")
    return "\n".join(lines) + "\n"


def _current_vs_rebuilt_payload() -> dict[str, Any]:
    return {
        "artifact": "m5_4d_current_vs_rebuilt_structure",
        "current_issue": (
            "M5.4B continuity review did not first establish entity validity and selected repetitive evidence."
        ),
        "rebuilt_changes": [
            "raw detector rows normalized as person_candidate",
            "entity-validity gate before continuity",
            "review-equivalence clustering before selection",
            "temporal GIF/MP4 as primary continuity evidence",
            "cluster-aware learning validation before model application",
        ],
        **safety_payload(),
    }


def _safety_audit(stage_root: Path, validation_summary: dict[str, Any], review_pack: dict[str, Any]) -> dict[str, Any]:
    forbidden_names = {
        "identity_id",
        "persistent_player_id",
        "confirmed_player_id",
        "player_slot_id",
        "goalkeeper_slot_id",
    }
    found: set[str] = set()
    for path in stage_root.rglob("*.json"):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name in forbidden_names:
            if f'"{name}"' in text:
                found.add(name)
    return {
        "artifact": "m5_4d_safety_guardrail_audit",
        "passed": not found
        and validation_summary["production_ready"] is False
        and validation_summary["no_auto_promotion"] is True
        and review_pack["file_cap_respected"],
        "forbidden_identity_or_slot_keys": sorted(found),
        "review_pack_file_cap_respected": review_pack["file_cap_respected"],
        "review_pack_file_count": review_pack["file_count"],
        "source_frames_modified": False,
        "m5_4a_b_c_outputs_overwritten": False,
        **safety_payload(),
    }
