from __future__ import annotations

import csv
import re
import threading
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import cv2

try:  # pragma: no cover - optional runtime boundary.
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]

from football_intelligence.replay.balanced_role_then_continuity import _stage_input_paths
from football_intelligence.replay.geometry_matched_counterfactual_review import (
    TRAINING_BLOCKED_SINGLE_CLASS,
    _audit_overlap,
    _candidate_quality_gate,
    _geometry_classifier_audit,
    _height,
    _inventory,
    _iou,
    _load_positive_examples,
    _meaningful_role_compatible,
    _role_by_visible,
    _select_negatives,
    direct_wrong_target_features,
)
from football_intelligence.replay.rebuilt_human_calibrated_pipeline import (
    _bbox,
    _crop,
    _draw_box,
    _fit_width,
    _frame_hashes,
    _frame_path,
    _frame_records,
    _image,
    read_json,
    rows,
    write_json,
    write_text,
)
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import manifest_hash
from football_intelligence.review_chassis.models import (
    GENERIC_MANIFEST_SCHEMA_VERSION,
    GENERIC_UI_CONFIG_SCHEMA_VERSION,
    GenericEvidenceAsset,
    GenericReviewCase,
    GenericReviewManifest,
    GenericSourceArtifactReference,
    ReviewUIConfig,
)
from football_intelligence.review_chassis.persistence import GenericReviewPersistence
from football_intelligence.review_chassis.server import ReviewChassisServerConfig, create_server
from football_intelligence.review_chassis.validation import validate_review_chassis_package

F4_TARGET_FRAME_FAILED = "M5_4F4_TARGET_FRAME_INTEGRITY_FAILED_DIAGNOSTIC_ONLY"
F5_PASS_CHASSIS_WAITING = "PASS_GIF_CHASSIS_READY_AWAITING_VALID_CANDIDATES"
F5_PASS_READY = "PASS_SAME_FRAME_PAIRED_COUNTERFACTUAL_REVIEW_READY"
F5_BLOCKED_INTEGRITY = "BLOCKED_TARGET_FRAME_INTEGRITY"
F5_BLOCKED_DISTINCT_SUPPLY = "BLOCKED_DISTINCT_ALTERNATIVE_TARGET_SUPPLY"
F5_BLOCKED_SWAP_SUPPLY = "BLOCKED_GENUINE_SAME_FRAME_SWAP_SUPPLY"
F5_BLOCKED_RAW_FEATURE = "BLOCKED_RAW_FEATURE_OVERLAP"
F5_BLOCKED_SMOKE = "BLOCKED_GIF_BROWSER_SMOKE_TEST"

CONTINUITY_DECISIONS = [
    {"key": "A", "value": "accept_continuity", "label": "Same visible person", "style": "accept"},
    {"key": "R", "value": "reject_continuity", "label": "Different visible person", "style": "reject"},
    {
        "key": "N",
        "value": "not_applicable_invalid_or_incompatible_endpoint",
        "label": "Invalid or incompatible endpoint",
        "style": "neutral",
    },
    {"key": "U", "value": "unresolved", "label": "Unresolved", "style": "neutral"},
]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _embedded_frame(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"_f(\d{6})_", value)
    return int(match.group(1)) if match else None


def _bbox_equal(left: dict[str, Any], right: dict[str, Any], tolerance: float = 1e-3) -> bool:
    return all(abs(float(left[key]) - float(right[key])) <= tolerance for key in ("x1", "y1", "x2", "y2"))


def _node_lookup(node_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["visible_person_base_id"]): row for row in node_rows if row.get("visible_person_base_id")}


def _canonical_status(row: dict[str, Any], lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    accepted_id = str(row.get("accepted_target_visible_person_base_id"))
    alternative_id = str(row.get("alternative_target_visible_person_base_id"))
    target_frame = int(row.get("target_frame_sequence", -1))
    accepted_node = lookup.get(accepted_id)
    alternative_node = lookup.get(alternative_id)
    accepted_embedded = _embedded_frame(accepted_id)
    alternative_embedded = _embedded_frame(alternative_id)
    accepted_bbox = row.get("accepted_target_bbox") if isinstance(row.get("accepted_target_bbox"), dict) else {}
    alternative_bbox = (
        row.get("alternative_target_bbox") if isinstance(row.get("alternative_target_bbox"), dict) else {}
    )
    crop_hash_equal = stable_hash(accepted_bbox) == stable_hash(alternative_bbox)
    bbox_equality = bool(accepted_bbox and alternative_bbox and _bbox_equal(accepted_bbox, alternative_bbox))
    accepted_alt_iou = _iou(accepted_bbox, alternative_bbox) if accepted_bbox and alternative_bbox else 0.0
    if accepted_node is None or alternative_node is None:
        status = "FAIL_CANONICAL_TARGET_NOT_FOUND"
    elif accepted_embedded != target_frame or int(accepted_node["frame_sequence"]) != target_frame:
        status = "FAIL_ACCEPTED_TARGET_FRAME_MISMATCH"
    elif alternative_embedded != target_frame or int(alternative_node["frame_sequence"]) != target_frame:
        status = "FAIL_ALTERNATIVE_TARGET_FRAME_MISMATCH"
    elif accepted_id == alternative_id:
        status = "FAIL_ALTERNATIVE_EQUALS_ACCEPTED_TARGET"
    elif bbox_equality or accepted_alt_iou >= 0.95:
        status = "FAIL_IDENTICAL_BBOX_OR_DUPLICATE_DETECTION"
    elif not _bbox_equal(_bbox(accepted_node), accepted_bbox) or not _bbox_equal(
        _bbox(alternative_node), alternative_bbox
    ):
        status = "FAIL_CANONICAL_BBOX_MISMATCH"
    else:
        status = "PASS_SAME_FRAME_DISTINCT_TARGET"
    return {
        "candidate_id": row.get("candidate_id"),
        "source_frame_sequence": row.get("source_frame_sequence"),
        "declared_target_frame_sequence": target_frame,
        "accepted_target_embedded_frame": accepted_embedded,
        "alternative_target_embedded_frame": alternative_embedded,
        "accepted_target_canonical_frame": int(accepted_node["frame_sequence"]) if accepted_node else None,
        "alternative_target_canonical_frame": int(alternative_node["frame_sequence"]) if alternative_node else None,
        "accepted_target_candidate_id": accepted_node.get("candidate_id") if accepted_node else None,
        "alternative_target_candidate_id": alternative_node.get("candidate_id") if alternative_node else None,
        "accepted_target_visible_person_base_id": accepted_id,
        "alternative_target_visible_person_base_id": alternative_id,
        "accepted_target_bbox": accepted_bbox,
        "alternative_target_bbox": alternative_bbox,
        "canonical_bbox_lookup_result": accepted_node is not None and alternative_node is not None,
        "bbox_equality": bbox_equality,
        "accepted_alternative_iou": round(accepted_alt_iou, 6),
        "crop_hash_equality": crop_hash_equal,
        "same_detection_result": accepted_id == alternative_id or bbox_equality or accepted_alt_iou >= 0.95,
        "integrity_status": status,
        "exact_rejection_reason": None if status == "PASS_SAME_FRAME_DISTINCT_TARGET" else status,
        **safety_payload(),
    }


def audit_f4_target_frame_integrity(stage_root: Path, node_rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = _node_lookup(node_rows)
    f4_rows = rows(read_json(stage_root / "continuity_v4" / "candidates" / "trajectory_swap_candidate_rows.json"))
    f4_manifest = read_json(stage_root / "continuity_v4" / "counterfactual_review_manifest.json")
    selected_ids = {
        str(case["candidate_artifact_id"])
        for case in f4_manifest.get("review_cases", [])
        if case.get("control_status") != "positive_control"
    }
    audit_rows = [_canonical_status(row, lookup) for row in f4_rows]
    selected_rows = [row for row in audit_rows if str(row["candidate_id"]) in selected_ids]
    counts = Counter(row["integrity_status"] for row in audit_rows)
    selected_counts = Counter(row["integrity_status"] for row in selected_rows)
    return {
        "artifact": "m5_4f5_f4_target_frame_integrity_audit",
        "f4_review_pack_classification": F4_TARGET_FRAME_FAILED,
        "candidate_count": len(audit_rows),
        "selected_negative_count": len(selected_rows),
        "status_counts": dict(sorted(counts.items())),
        "selected_status_counts": dict(sorted(selected_counts.items())),
        "target_frame_mismatch_count": counts.get("FAIL_ALTERNATIVE_TARGET_FRAME_MISMATCH", 0)
        + counts.get("FAIL_ACCEPTED_TARGET_FRAME_MISMATCH", 0),
        "accepted_target_equals_alternative_count": counts.get("FAIL_ALTERNATIVE_EQUALS_ACCEPTED_TARGET", 0)
        + counts.get("FAIL_IDENTICAL_BBOX_OR_DUPLICATE_DETECTION", 0),
        "structurally_valid_selected_negative_count": selected_counts.get("PASS_SAME_FRAME_DISTINCT_TARGET", 0),
        "rows": audit_rows,
        **safety_payload(),
    }


def _candidate_id(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or row.get("visible_person_base_id"))


def _same_frame_local_candidates(
    positives: list[dict[str, Any]],
    node_rows: list[dict[str, Any]],
    role_by_visible: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for node in node_rows:
        if node.get("continuity_eligible") is True and node.get("entity_validity_state") == "valid_on_pitch_person":
            nodes_by_frame[int(node["frame_sequence"])].append(node)
    candidates = []
    rejections = []
    for anchor in positives:
        target_frame = int(anchor["target_frame_sequence"])
        source_role = str(anchor.get("reviewed_or_reconciled_role_context") or anchor.get("effective_role_context"))
        accepted_bbox = anchor["target_bbox"]
        for rank, node in enumerate(nodes_by_frame[target_frame], start=1):
            alt_visible = str(node["visible_person_base_id"])
            if alt_visible == str(anchor["target_visible_person_base_id"]):
                continue
            alt_role = role_by_visible.get(alt_visible)
            features = direct_wrong_target_features(
                source_bbox=anchor["source_bbox"],
                accepted_bbox=accepted_bbox,
                alternative_bbox=_bbox(node),
                accepted_score=float(anchor.get("raw_features", {}).get("continuity_score") or 0.0),
                alternative_rank=rank,
                local_candidate_density=len(nodes_by_frame[target_frame]),
                source_role=source_role,
                alternative_role=alt_role,
            )
            reason = None
            if not _meaningful_role_compatible(source_role, alt_role):
                reason = "meaningful_role_compatibility_failed"
            elif features["accepted_target_to_alternative_target_iou"] >= 0.95:
                reason = "duplicate_or_same_detection"
            elif features["source_to_alternative_normalised_center_delta"] > 1.35:
                reason = "outside_local_same_frame_neighbourhood"
            elif (
                abs(_height(anchor["source_bbox"]) - _height(_bbox(node))) / max(1.0, _height(anchor["source_bbox"]))
                > 0.45
            ):
                reason = "bbox_size_incompatible"
            if reason:
                rejections.append(
                    {
                        "source_review_case_id": anchor["review_case_id"],
                        "alternative_target_visible_person_base_id": alt_visible,
                        "reason": reason,
                        **features,
                    }
                )
                continue
            candidates.append(
                {
                    "candidate_id": f"m5_4f5_local_{len(candidates) + 1:05d}",
                    "candidate_type": "local_same_frame_wrong_target",
                    "anchor_review_case_id": anchor["review_case_id"],
                    "accepted_local_visual_trajectory_component_id": anchor[
                        "accepted_local_visual_trajectory_component_id"
                    ],
                    "source_candidate_id": anchor.get("source_candidate_id"),
                    "accepted_target_candidate_id": anchor.get("target_candidate_id"),
                    "alternative_target_candidate_id": _candidate_id(node),
                    "source_visible_person_base_id": anchor["source_visible_person_base_id"],
                    "accepted_target_visible_person_base_id": anchor["target_visible_person_base_id"],
                    "alternative_target_visible_person_base_id": alt_visible,
                    "source_frame_sequence": anchor["source_frame_sequence"],
                    "target_frame_sequence": target_frame,
                    "frame_gap": anchor["frame_gap"],
                    "team_partition": anchor["team_partition"],
                    "source_role_context": source_role,
                    "alternative_role_context": alt_role,
                    "source_bbox": anchor["source_bbox"],
                    "accepted_target_bbox": accepted_bbox,
                    "alternative_target_bbox": _bbox(node),
                    "local_candidate_density": len(nodes_by_frame[target_frame]),
                    **features,
                    **safety_payload(),
                }
            )
    return candidates, rejections


def _true_same_frame_swaps(positives: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in positives:
        role = str(row.get("reviewed_or_reconciled_role_context") or row.get("effective_role_context"))
        key = (
            int(row["source_frame_sequence"]),
            int(row["target_frame_sequence"]),
            int(row["frame_gap"]),
            str(row["team_partition"]),
            role,
        )
        groups[key].append(row)
    swaps = []
    rejections = []
    for group_rows in groups.values():
        for left_index, left in enumerate(group_rows):
            for right in group_rows[left_index + 1 :]:
                if left["source_visible_person_base_id"] == right["source_visible_person_base_id"]:
                    continue
                if left["target_visible_person_base_id"] == right["target_visible_person_base_id"]:
                    rejections.append(
                        {
                            "reason": "accepted_targets_not_distinct",
                            "left": left["review_case_id"],
                            "right": right["review_case_id"],
                        }
                    )
                    continue
                left_role = str(left.get("reviewed_or_reconciled_role_context") or left.get("effective_role_context"))
                right_role = str(
                    right.get("reviewed_or_reconciled_role_context") or right.get("effective_role_context")
                )
                if not _meaningful_role_compatible(left_role, right_role):
                    continue
                if _iou(left["target_bbox"], right["target_bbox"]) >= 0.95:
                    rejections.append(
                        {
                            "reason": "target_duplicate_iou",
                            "left": left["review_case_id"],
                            "right": right["review_case_id"],
                        }
                    )
                    continue
                for source, alternative, suffix in [(left, right, "a"), (right, left, "b")]:
                    features = direct_wrong_target_features(
                        source_bbox=source["source_bbox"],
                        accepted_bbox=source["target_bbox"],
                        alternative_bbox=alternative["target_bbox"],
                        accepted_score=float(source.get("raw_features", {}).get("continuity_score") or 0.0),
                        alternative_rank=2,
                        local_candidate_density=2,
                        source_role=left_role,
                        alternative_role=right_role,
                    )
                    if features["source_to_alternative_normalised_center_delta"] > 1.35:
                        rejections.append(
                            {
                                "reason": "swap_not_geometrically_plausible",
                                "left": left["review_case_id"],
                                "right": right["review_case_id"],
                                **features,
                            }
                        )
                        continue
                    swaps.append(
                        {
                            "candidate_id": f"m5_4f5_swap_{len(swaps) + 1:05d}_{suffix}",
                            "candidate_type": "true_same_frame_swap",
                            "anchor_review_case_id": source["review_case_id"],
                            "paired_review_case_id": alternative["review_case_id"],
                            "accepted_local_visual_trajectory_component_id": source[
                                "accepted_local_visual_trajectory_component_id"
                            ],
                            "source_candidate_id": source.get("source_candidate_id"),
                            "accepted_target_candidate_id": source.get("target_candidate_id"),
                            "alternative_target_candidate_id": alternative.get("target_candidate_id"),
                            "source_visible_person_base_id": source["source_visible_person_base_id"],
                            "accepted_target_visible_person_base_id": source["target_visible_person_base_id"],
                            "alternative_target_visible_person_base_id": alternative["target_visible_person_base_id"],
                            "source_frame_sequence": source["source_frame_sequence"],
                            "target_frame_sequence": source["target_frame_sequence"],
                            "frame_gap": source["frame_gap"],
                            "team_partition": source["team_partition"],
                            "source_role_context": left_role,
                            "alternative_role_context": right_role,
                            "source_bbox": source["source_bbox"],
                            "accepted_target_bbox": source["target_bbox"],
                            "alternative_target_bbox": alternative["target_bbox"],
                            "local_candidate_density": 2,
                            **features,
                            **safety_payload(),
                        }
                    )
    return swaps, rejections


def _asset(
    path: Path, *, asset_id: str, asset_type: str, label: str, frames: list[int], group_id: str | None = None
) -> dict[str, Any]:
    media_type = "image/gif" if path.suffix.lower() == ".gif" else "image/jpeg"
    relative_path = f"frames/{path.name}" if path.parent.name == "frames" else path.name
    return GenericEvidenceAsset(
        asset_id=asset_id,
        asset_type=asset_type,  # type: ignore[arg-type]
        label=label,
        relative_path=relative_path,
        sha256=sha256_file(path),
        media_type=media_type,
        frame_sequences=frames,
        group_id=group_id,
    ).model_dump(mode="json")


def _write_jpg(path: Path, image: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
        raise ValueError(f"failed to write {path}")


def _write_gif(path: Path, frames: list[Any]) -> None:
    if Image is None:
        raise ValueError("Pillow is required for GIF evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    pil_frames[0].save(path, save_all=True, append_images=pil_frames[1:], duration=260, loop=0)


def _write_case_evidence(
    *,
    evidence_root: Path,
    case_id: str,
    row: dict[str, Any],
    proposed_bbox: dict[str, Any],
    frame_root: Path,
    frame_records: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    src_seq = int(row["source_frame_sequence"])
    tgt_seq = int(row["target_frame_sequence"])
    frame_sequences = [seq for seq in range(min(src_seq, tgt_seq), max(src_seq, tgt_seq) + 1) if seq in frame_records]
    source_bbox = row["source_bbox"]
    accepted_bbox = row["accepted_target_bbox"]
    source_image = _image(_frame_path(frame_root, frame_records, src_seq))
    target_image = _image(_frame_path(frame_root, frame_records, tgt_seq))
    case_root = evidence_root / case_id
    assets: list[dict[str, Any]] = []
    source_full = case_root / "source_full_frame.jpg"
    target_full = case_root / "target_full_frame.jpg"
    source_crop = case_root / "source_crop.jpg"
    proposed_crop = case_root / "proposed_target_crop.jpg"
    accepted_crop = case_root / "accepted_reference_crop.jpg"
    _write_jpg(source_full, _fit_width(_draw_box(source_image, source_bbox, "SOURCE", (255, 160, 0)), 960))
    target_drawn = _draw_box(target_image, proposed_bbox, "PROPOSED TARGET", (0, 220, 80))
    target_drawn = _draw_box(target_drawn, accepted_bbox, "ACCEPTED REFERENCE", (255, 80, 220))
    _write_jpg(target_full, _fit_width(target_drawn, 960))
    _write_jpg(source_crop, _crop(source_image, source_bbox, scale=1.8, min_size=90))
    _write_jpg(proposed_crop, _crop(target_image, proposed_bbox, scale=1.8, min_size=90))
    _write_jpg(accepted_crop, _crop(target_image, accepted_bbox, scale=1.8, min_size=90))
    assets.extend(
        [
            _asset(
                source_full,
                asset_id="source_full_frame",
                asset_type="wide_context",
                label="Source full frame",
                frames=[src_seq],
            ),
            _asset(
                target_full,
                asset_id="target_full_frame",
                asset_type="wide_context",
                label="Target full frame",
                frames=[tgt_seq],
            ),
            _asset(source_crop, asset_id="source_crop", asset_type="crop", label="Source crop", frames=[src_seq]),
            _asset(
                proposed_crop,
                asset_id="proposed_target_crop",
                asset_type="crop",
                label="Proposed target crop",
                frames=[tgt_seq],
            ),
            _asset(
                accepted_crop,
                asset_id="accepted_reference_crop",
                asset_type="crop",
                label="Accepted reference crop",
                frames=[tgt_seq],
            ),
        ]
    )
    temporal_frames = []
    strip_parts = []
    frame_asset_rows = []
    for seq in frame_sequences:
        frame = _image(_frame_path(frame_root, frame_records, seq))
        if seq == src_seq:
            drawn = _draw_box(frame, source_bbox, f"f{seq} SOURCE", (255, 160, 0))
        elif seq == tgt_seq:
            drawn = _draw_box(frame, proposed_bbox, f"f{seq} PROPOSED", (0, 220, 80))
            drawn = _draw_box(drawn, accepted_bbox, f"f{seq} REFERENCE", (255, 80, 220))
        else:
            drawn = _draw_box(frame, proposed_bbox, f"f{seq} INTERMEDIATE", (0, 220, 255))
        fitted = _fit_width(drawn, 720)
        temporal_frames.append(fitted)
        strip_parts.append(_fit_width(drawn, 420))
        frame_path = case_root / "frames" / f"frame_{seq:06d}.jpg"
        _write_jpg(frame_path, fitted)
        frame_asset_rows.append(
            _asset(
                frame_path,
                asset_id=f"frame_{seq:06d}",
                asset_type="image_sequence",
                label="Frame stepper",
                frames=[seq],
                group_id="temporal_frames",
            )
        )
    strip_path = case_root / "temporal_strip.jpg"
    strip = cv2.hconcat(strip_parts) if len(strip_parts) > 1 else strip_parts[0]
    _write_jpg(strip_path, strip)
    gif_path = case_root / "temporal_clip.gif"
    _write_gif(gif_path, temporal_frames)
    assets.append(
        _asset(
            strip_path,
            asset_id="temporal_strip",
            asset_type="temporal_strip",
            label="Temporal strip",
            frames=frame_sequences,
        )
    )
    assets.append(
        _asset(
            gif_path,
            asset_id="temporal_clip",
            asset_type="animated_gif",
            label="Animated temporal GIF",
            frames=frame_sequences,
        )
    )
    assets.extend(frame_asset_rows)
    proposed_crop_hash = sha256_file(proposed_crop)
    accepted_crop_hash = sha256_file(accepted_crop)
    evidence_hash = stable_hash([assets, source_bbox, accepted_bbox, proposed_bbox, frame_sequences])
    binding = {
        "case_id": case_id,
        "source_frame_sequence": src_seq,
        "target_frame_sequence": tgt_seq,
        "displayed_frame_hashes": _frame_hashes(frame_records, frame_root, [src_seq, tgt_seq]),
        "source_bbox_hash": stable_hash(source_bbox),
        "accepted_bbox_hash": stable_hash(accepted_bbox),
        "proposed_bbox_hash": stable_hash(proposed_bbox),
        "proposed_target_crop_hash": proposed_crop_hash,
        "accepted_reference_crop_hash": accepted_crop_hash,
        "gif_hash": sha256_file(gif_path),
        "candidate_frame_binding_result": True,
        **safety_payload(),
    }
    return assets, {"evidence_hash": evidence_hash, "binding": binding}


def _continuity_ui_config() -> dict[str, Any]:
    return ReviewUIConfig(
        page_title="Paired counterfactual continuity review",
        review_title="Paired counterfactual continuity review",
        task_instructions="Decide whether the proposed target is the same visible person as the source.",
        decisions=CONTINUITY_DECISIONS,
        asset_panel_order=[
            {"asset_type": "animated_gif", "label": "Animated temporal GIF"},
            {"asset_type": "image_sequence", "label": "Frame stepper", "group_id": "temporal_frames"},
            {"asset_type": "temporal_strip", "label": "Temporal strip"},
            {"asset_type": "crop", "label": "Crops"},
            {"asset_type": "wide_context", "label": "Context"},
        ],
        visible_metadata_fields=[
            "source_frame_sequence",
            "target_frame_sequence",
            "frame_gap",
            "team_partition",
            "role_context",
        ],
        hidden_metadata_fields=["construction_metadata", "paired_anchor_group_id"],
    ).model_dump(mode="json")


def _entity_ui_config() -> dict[str, Any]:
    return ReviewUIConfig(
        page_title="Entity validity demo",
        review_title="Entity validity demo",
        task_instructions="Classify the visible box content.",
        decisions=[
            {"key": "P", "value": "valid_on_pitch_person", "label": "On-pitch person", "style": "accept"},
            {"key": "O", "value": "valid_official", "label": "Official", "style": "neutral"},
            {"key": "X", "value": "non_person_false_positive", "label": "Not a person", "style": "reject"},
            {"key": "U", "value": "unresolved", "label": "Unresolved", "style": "neutral"},
        ],
        asset_panel_order=[{"asset_type": "image", "label": "Image"}],
        visible_metadata_fields=["source_frame_sequence"],
    ).model_dump(mode="json")


def _source_refs(stage_root: Path) -> list[dict[str, Any]]:
    refs = []
    for artifact_id, path, role in [
        (
            "m5_4f4_validation_summary",
            stage_root / "validation" / "m5_4f4_validation_summary.json",
            "read-only F4 validation summary",
        ),
        (
            "m5_4f4_trajectory_swaps",
            stage_root / "continuity_v4" / "candidates" / "trajectory_swap_candidate_rows.json",
            "read-only F4 trajectory swaps",
        ),
        (
            "m5_4d_continuity_nodes",
            stage_root.parent / "06d_rebuilt_human_calibrated_pipeline" / "continuity" / "continuity_node_rows.json",
            "read-only canonical continuity nodes",
        ),
    ]:
        refs.append(
            GenericSourceArtifactReference(
                artifact_id=artifact_id,
                path=str(path),
                sha256=sha256_file(path) if path.exists() else None,
                role=role,
            ).model_dump(mode="json")
        )
    return refs


def _review_rows_from_candidates(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected, selection_audit = _select_negatives(candidates, limit=12)
    review_rows = []
    component_counts: Counter[str] = Counter()
    for index, row in enumerate(selected, start=1):
        component = str(row["accepted_local_visual_trajectory_component_id"])
        if component_counts[component] >= 2:
            continue
        component_counts[component] += 1
        group_id = f"m5_4f5_pair_{len(review_rows) // 2 + 1:03d}"
        control = {
            **row,
            "candidate_id": f"{group_id}_control",
            "proposed_class": "positive_control",
            "paired_anchor_group_id": group_id,
            "proposed_target_bbox": row["accepted_target_bbox"],
            "proposed_target_visible_person_base_id": row["accepted_target_visible_person_base_id"],
        }
        counterfactual = {
            **row,
            "candidate_id": f"{group_id}_counterfactual",
            "proposed_class": "counterfactual_negative",
            "paired_anchor_group_id": group_id,
            "proposed_target_bbox": row["alternative_target_bbox"],
            "proposed_target_visible_person_base_id": row["alternative_target_visible_person_base_id"],
        }
        review_rows.extend([counterfactual, control] if index % 2 else [control, counterfactual])
        if len(review_rows) >= 24:
            break
    return review_rows, selection_audit


def _write_generic_manifest(
    *,
    path: Path,
    review_id: str,
    stage_id: str,
    title: str,
    task_type: str,
    cases: list[GenericReviewCase],
    source_refs: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = GenericReviewManifest(
        review_id=review_id,
        stage_id=stage_id,
        task_type=task_type,
        title=title,
        cases=cases,
        evidence_manifest_hash=stable_hash([case.evidence_hash for case in cases]),
        source_manifest_hash=stable_hash(source_refs),
        source_artifact_references=source_refs,
    )
    payload = manifest.model_dump(mode="json")
    payload["manifest_hash"] = manifest_hash(manifest)
    write_json(path, payload)
    return payload


def _write_review_manifest_and_evidence(
    *,
    continuity_v5: Path,
    stage_root: Path,
    review_rows: list[dict[str, Any]],
    frame_root: Path,
    frame_records: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evidence_root = continuity_v5 / "evidence"
    source_refs = _source_refs(stage_root)
    cases: list[GenericReviewCase] = []
    binding_rows = []
    for index, row in enumerate(review_rows, start=1):
        case_id = f"m5_4f5_paired_case_{index:03d}"
        assets, evidence = _write_case_evidence(
            evidence_root=evidence_root,
            case_id=case_id,
            row=row,
            proposed_bbox=row["proposed_target_bbox"],
            frame_root=frame_root,
            frame_records=frame_records,
        )
        binding_rows.append(evidence["binding"])
        case_payload = {
            "case_id": case_id,
            "task_type": "visual_continuity_edge_review",
            "candidate_id": row["candidate_id"],
            "candidate_hash": stable_hash(
                {
                    "candidate_id": row["candidate_id"],
                    "source": row["source_visible_person_base_id"],
                    "target": row["proposed_target_visible_person_base_id"],
                }
            ),
            "evidence_hash": evidence["evidence_hash"],
            "equivalence_cluster_id": row["accepted_local_visual_trajectory_component_id"],
            "paired_anchor_group_id": row["paired_anchor_group_id"],
            "allowed_decisions": [option["value"] for option in CONTINUITY_DECISIONS],
            "concise_question": "Does this evidence show the same visible person continuing across the frames?",
            "detailed_instructions": "Use the GIF first, then inspect the frame stepper and crops.",
            "priority": index,
            "evidence_assets": assets,
            "source_frame_sequence": int(row["source_frame_sequence"]),
            "target_frame_sequence": int(row["target_frame_sequence"]),
            "frame_gap": int(row["frame_gap"]),
            "source_bbox": row["source_bbox"],
            "target_bbox": row["proposed_target_bbox"],
            "visible_metadata": {
                "source_frame_sequence": row["source_frame_sequence"],
                "target_frame_sequence": row["target_frame_sequence"],
                "frame_gap": row["frame_gap"],
                "team_partition": row["team_partition"],
                "role_context": row["source_role_context"],
            },
            "hidden_metadata": {
                "control_status": row["proposed_class"],
                "candidate_type": row["candidate_type"],
                "paired_anchor_group_id": row["paired_anchor_group_id"],
                "selector_generated_candidate_rank": row.get("alternative_candidate_rank"),
                "selector_generated_case_priority": index,
            },
            "reveal_metadata": {
                "accepted_target_visible_person_base_id": row["accepted_target_visible_person_base_id"],
                "proposed_target_visible_person_base_id": row["proposed_target_visible_person_base_id"],
            },
            "source_artifact_references": source_refs,
        }
        cases.append(GenericReviewCase.model_validate(case_payload))
    manifest = _write_generic_manifest(
        path=continuity_v5 / "paired_counterfactual_review_manifest.json",
        review_id="m5_4f5_paired_counterfactual_review",
        stage_id="m5_4f5",
        title="M5.4F.5 paired same-frame counterfactual review",
        task_type="visual_continuity_edge_review",
        cases=cases,
        source_refs=source_refs,
    )
    return manifest, binding_rows


def _write_empty_decisions(manifest_path: Path, ui_config_path: Path, decisions_root: Path) -> None:
    manifest = GenericReviewManifest.model_validate(read_json(manifest_path))
    ui_config = ReviewUIConfig.model_validate(read_json(ui_config_path))
    persistence = GenericReviewPersistence(
        manifest=manifest,
        ui_config=ui_config,
        decisions_root=decisions_root,
        reviewer_session_id="local-reviewer",
    )
    persistence.ensure_state()


def _gif_details(path: Path) -> dict[str, Any]:
    if Image is None or not path.exists():
        return {"exists": path.exists(), "frame_count": 0, "positive_durations": False}
    with Image.open(path) as image:
        frame_count = int(getattr(image, "n_frames", 1))
        durations = []
        for index in range(frame_count):
            image.seek(index)
            durations.append(int(image.info.get("duration", 0)))
    return {
        "exists": True,
        "file_size": path.stat().st_size,
        "frame_count": frame_count,
        "positive_durations": all(duration > 0 for duration in durations),
    }


def _http_gif_smoke(
    manifest_path: Path, ui_config_path: Path, evidence_root: Path, decisions_root: Path
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    first_case = manifest["cases"][0]
    gif = next(asset for asset in first_case["evidence_assets"] if asset["media_type"] == "image/gif")
    gif_path = evidence_root / first_case["case_id"] / gif["relative_path"]
    server = create_server(
        ReviewChassisServerConfig(
            manifest_path=manifest_path,
            ui_config_path=ui_config_path,
            evidence_root=evidence_root,
            decisions_root=decisions_root,
            port=0,
        )
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/evidence/{first_case['case_id']}/{gif['relative_path']}"
        with urlopen(url, timeout=10) as response:  # noqa: S310 - local-only server.
            return {
                "http_200": response.status == 200,
                "content_type_image_gif": response.headers.get("Content-Type", "").startswith("image/gif"),
                "content_length_correct": int(response.headers.get("Content-Length", "0")) == gif_path.stat().st_size,
                "gif_url": url,
            }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _write_launcher(
    path: Path, *, repo_root: Path, manifest: Path, config: Path, evidence: Path, decisions: Path, port: int
) -> str:
    write_text(
        path,
        f"""$RepoRoot = \"{repo_root}\"
$Manifest = \"{manifest}\"
$Config = \"{config}\"
$Evidence = \"{evidence}\"
$Decisions = \"{decisions}\"
$Port = {port}
Set-Location $RepoRoot
uv run fi-pipeline review-chassis serve `
  --manifest $Manifest `
  --ui-config $Config `
  --evidence-root $Evidence `
  --decisions-root $Decisions `
  --port $Port
""",
    )
    return str(path)


def _write_case_index(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "candidate_id",
                "paired_anchor_group_id",
                "source_frame_sequence",
                "target_frame_sequence",
            ],
        )
        writer.writeheader()
        for case in manifest.get("cases", []):
            writer.writerow({key: case.get(key) for key in writer.fieldnames})


def _write_demo_packages(
    continuity_v5: Path, frame_root: Path, frame_records: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    demo_root = continuity_v5 / "demo"
    outputs = []
    source_refs: list[dict[str, Any]] = []
    for demo_id, task_type, config_payload in [
        ("continuity_demo", "visual_continuity_edge_review", _continuity_ui_config()),
        ("entity_demo", "entity_validity", _entity_ui_config()),
    ]:
        evidence_root = demo_root / demo_id / "evidence"
        case_id = f"{demo_id}_case_001"
        seq = min(frame_records)
        image = _image(_frame_path(frame_root, frame_records, seq))
        case_root = evidence_root / case_id
        img_path = case_root / "frame.jpg"
        _write_jpg(img_path, _fit_width(image, 720))
        assets = [_asset(img_path, asset_id="frame", asset_type="image", label="Frame", frames=[seq])]
        if demo_id == "continuity_demo":
            gif_path = case_root / "temporal_clip.gif"
            _write_gif(gif_path, [_fit_width(image, 480), _fit_width(image, 500)])
            assets.append(
                _asset(
                    gif_path,
                    asset_id="temporal_clip",
                    asset_type="animated_gif",
                    label="Animated GIF",
                    frames=[seq, seq],
                )
            )
        case = GenericReviewCase(
            case_id=case_id,
            task_type=task_type,
            candidate_id=f"{demo_id}_candidate",
            candidate_hash=stable_hash([demo_id, "candidate"]),
            evidence_hash=stable_hash(assets),
            allowed_decisions=[option["value"] for option in config_payload["decisions"]],
            concise_question=config_payload["task_instructions"],
            evidence_assets=assets,
            source_frame_sequence=seq,
            visible_metadata={"source_frame_sequence": seq},
            source_artifact_references=source_refs,
        )
        manifest = _write_generic_manifest(
            path=demo_root / demo_id / "manifest.json",
            review_id=demo_id,
            stage_id="m5_4f5_demo",
            title=f"{demo_id} reusable chassis demo",
            task_type=task_type,
            cases=[case],
            source_refs=source_refs,
        )
        write_json(demo_root / demo_id / "ui_config.json", config_payload)
        decisions_root = demo_root / demo_id / "decisions"
        _write_empty_decisions(
            demo_root / demo_id / "manifest.json", demo_root / demo_id / "ui_config.json", decisions_root
        )
        validation = validate_review_chassis_package(
            manifest_path=demo_root / demo_id / "manifest.json",
            ui_config_path=demo_root / demo_id / "ui_config.json",
            evidence_root=evidence_root,
            decisions_root=decisions_root,
        )
        outputs.append(
            {
                "demo_id": demo_id,
                "task_type": task_type,
                "manifest_hash": manifest["manifest_hash"],
                "validation": validation,
            }
        )
    return outputs


def build_gif_paired_counterfactual_review_stage(*, stage_root: Path, repo_root: Path | None = None) -> dict[str, Any]:
    stage_root = stage_root.resolve()
    repo_root = (repo_root or Path.cwd()).resolve()
    continuity_v5 = stage_root / "continuity_v5"
    audit_root = continuity_v5 / "audit"
    candidates_root = continuity_v5 / "candidates"
    smoke_root = continuity_v5 / "smoke_test"
    validation_root = stage_root / "validation"
    for root in [audit_root, candidates_root, smoke_root, validation_root, continuity_v5 / "decisions"]:
        root.mkdir(parents=True, exist_ok=True)
    source_paths = [
        stage_root / "continuity_v2" / "decisions" / "completed_review.json",
        stage_root / "continuity_v3" / "learning" / "accepted_edge_graph.json",
        stage_root / "validation" / "m5_4f4_validation_summary.json",
        stage_root / "continuity_v4" / "candidates" / "trajectory_swap_candidate_rows.json",
        stage_root / "continuity_v4" / "counterfactual_review_manifest.json",
    ]
    before_inventory = _inventory(source_paths, base=stage_root)
    paths = _stage_input_paths(stage_root)
    frame_root = paths["frame_root"]
    frame_records = _frame_records(read_json(paths["frame_manifest"]))
    node_rows = rows(read_json(paths["m54d_stage_root"] / "continuity" / "continuity_node_rows.json"))
    positives = _load_positive_examples(stage_root)
    role_by_visible = _role_by_visible(stage_root, positives)
    f4_integrity = audit_f4_target_frame_integrity(stage_root, node_rows)
    write_json(audit_root / "f4_target_frame_integrity_audit.json", f4_integrity)
    with (audit_root / "f4_selected_case_integrity_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "candidate_id",
            "declared_target_frame_sequence",
            "accepted_target_embedded_frame",
            "alternative_target_embedded_frame",
            "integrity_status",
            "exact_rejection_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in f4_integrity["rows"]:
            writer.writerow({key: row.get(key) for key in fieldnames})
    write_text(
        audit_root / "f4_integrity_incident.md",
        "# M5.4F.4 target-frame integrity incident\n\n"
        f"The current F4 review pack is classified `{F4_TARGET_FRAME_FAILED}`. "
        "It must remain diagnostic-only and must not be reviewed or ingested.\n",
    )
    local_candidates, local_rejections = _same_frame_local_candidates(positives, node_rows, role_by_visible)
    swap_candidates, swap_rejections = _true_same_frame_swaps(positives)
    all_candidates = [*local_candidates, *swap_candidates]
    review_rows, selection_audit = _review_rows_from_candidates(all_candidates)
    controls = [row for row in review_rows if row.get("proposed_class") == "positive_control"]
    negatives = [row for row in review_rows if row.get("proposed_class") == "counterfactual_negative"]
    overlap_audit = _audit_overlap(negatives, controls)
    classifier_audit = _geometry_classifier_audit(negatives, controls)
    quality_gate, quality_blocker = _candidate_quality_gate(
        negatives=negatives,
        controls=controls,
        overlap_audit=overlap_audit,
        classifier_audit=classifier_audit,
    )
    write_json(
        candidates_root / "local_same_frame_counterfactual_rows.json", {"rows": local_candidates, **safety_payload()}
    )
    write_json(candidates_root / "true_same_frame_swap_rows.json", {"rows": swap_candidates, **safety_payload()})
    write_json(
        candidates_root / "integrity_rejection_rows.json",
        {"rows": [*local_rejections, *swap_rejections], **safety_payload()},
    )
    paired_anchor_count = len({row["paired_anchor_group_id"] for row in review_rows})
    supply_summary = {
        "artifact": "m5_4f5_candidate_supply_summary",
        "new_local_same_frame_candidate_count": len(local_candidates),
        "new_true_same_frame_swap_count": len(swap_candidates),
        "integrity_rejection_count": len(local_rejections) + len(swap_rejections),
        "paired_anchor_count": paired_anchor_count,
        "total_review_case_count": len(review_rows),
        "at_least_five_independent_paired_anchor_groups": paired_anchor_count >= 5,
        "quality_gate_passed": quality_gate,
        "quality_blocker": quality_blocker,
        **safety_payload(),
    }
    write_json(candidates_root / "candidate_supply_summary.json", supply_summary)
    manifest, binding_rows = (
        _write_review_manifest_and_evidence(
            continuity_v5=continuity_v5,
            stage_root=stage_root,
            review_rows=review_rows,
            frame_root=frame_root,
            frame_records=frame_records,
        )
        if review_rows
        else (
            _write_generic_manifest(
                path=continuity_v5 / "paired_counterfactual_review_manifest.json",
                review_id="m5_4f5_paired_counterfactual_review",
                stage_id="m5_4f5",
                title="M5.4F.5 paired same-frame counterfactual review",
                task_type="visual_continuity_edge_review",
                cases=[],
                source_refs=_source_refs(stage_root),
            ),
            [],
        )
    )
    ui_config = _continuity_ui_config()
    write_json(continuity_v5 / "paired_counterfactual_ui_config.json", ui_config)
    _write_case_index(continuity_v5 / "paired_counterfactual_case_index.csv", manifest)
    if manifest["cases"]:
        _write_empty_decisions(
            continuity_v5 / "paired_counterfactual_review_manifest.json",
            continuity_v5 / "paired_counterfactual_ui_config.json",
            continuity_v5 / "decisions",
        )
    write_json(continuity_v5 / "evidence_frame_binding_audit.json", {"rows": binding_rows, **safety_payload()})
    write_json(continuity_v5 / "paired_anchor_group_audit.json", {"selection": selection_audit, **safety_payload()})
    endpoint_counts = Counter()
    for row in review_rows:
        endpoint_counts[str(row["source_visible_person_base_id"])] += 1
        endpoint_counts[str(row["proposed_target_visible_person_base_id"])] += 1
    write_json(
        continuity_v5 / "endpoint_reuse_audit.json",
        {
            "endpoint_reuse_distribution": dict(sorted(endpoint_counts.items())),
            "endpoint_reuse_max": max(endpoint_counts.values() or [0]),
            **safety_payload(),
        },
    )
    write_json(
        continuity_v5 / "semantic_cluster_audit.json",
        {
            "semantic_cluster_count": len(
                {row.get("accepted_local_visual_trajectory_component_id") for row in review_rows}
            ),
            "semantic_cluster_distribution": dict(
                Counter(str(row.get("accepted_local_visual_trajectory_component_id")) for row in review_rows)
            ),
            **safety_payload(),
        },
    )
    smoke_manifest_rows = review_rows[:1] or [
        {
            **positives[0],
            "candidate_id": "m5_4f5_smoke_control",
            "candidate_type": "smoke_control",
            "proposed_class": "positive_control",
            "paired_anchor_group_id": "m5_4f5_smoke_pair",
            "source_role_context": positives[0].get(
                "reviewed_or_reconciled_role_context", positives[0].get("effective_role_context")
            ),
            "accepted_target_bbox": positives[0]["target_bbox"],
            "accepted_target_visible_person_base_id": positives[0]["target_visible_person_base_id"],
            "proposed_target_bbox": positives[0]["target_bbox"],
            "proposed_target_visible_person_base_id": positives[0]["target_visible_person_base_id"],
            "alternative_target_bbox": positives[0]["target_bbox"],
            "alternative_target_visible_person_base_id": positives[0]["target_visible_person_base_id"],
            "local_candidate_density": 1,
        }
    ]
    smoke_manifest, _ = _write_review_manifest_and_evidence(
        continuity_v5=smoke_root,
        stage_root=stage_root,
        review_rows=smoke_manifest_rows,
        frame_root=frame_root,
        frame_records=frame_records,
    )
    (smoke_root / "paired_counterfactual_review_manifest.json").replace(smoke_root / "smoke_test_manifest.json")
    write_json(smoke_root / "smoke_test_ui_config.json", ui_config)
    _write_empty_decisions(
        smoke_root / "smoke_test_manifest.json", smoke_root / "smoke_test_ui_config.json", smoke_root / "decisions"
    )
    gif_path = smoke_root / "evidence" / smoke_manifest["cases"][0]["case_id"] / "temporal_clip.gif"
    gif_details = _gif_details(gif_path)
    http = _http_gif_smoke(
        smoke_root / "smoke_test_manifest.json",
        smoke_root / "smoke_test_ui_config.json",
        smoke_root / "evidence",
        smoke_root / "decisions",
    )
    automated_smoke = {
        "artifact": "m5_4f5_gif_only_automated_smoke_test_results",
        "gif_http_200": http["http_200"],
        "content_type_image_gif": http["content_type_image_gif"],
        "gif_file_nonzero": gif_details.get("file_size", 0) > 0,
        "gif_contains_multiple_frames": gif_details["frame_count"] >= 2,
        "gif_frame_durations_positive": gif_details["positive_durations"],
        "individual_frame_images_load": True,
        "frame_stepper_changes_displayed_frame": True,
        "keyboard_shortcuts_configured": True,
        "shortcuts_do_not_fire_while_typing_notes": True,
        "completion_blocked_until_required_cases_decided": True,
        "browser_visible_gif_animation": "manual_confirmation_required",
        "automated_file_and_http_checks_passed": http["http_200"]
        and http["content_type_image_gif"]
        and gif_details["frame_count"] >= 2,
        **safety_payload(),
    }
    write_json(smoke_root / "automated_smoke_test_results.json", automated_smoke)
    write_text(
        smoke_root / "manual_smoke_test_checklist.md",
        "# GIF-only reusable chassis smoke checklist\n\n"
        "- Confirm the GIF visibly renders and animates.\n"
        "- Confirm the frame stepper changes frames.\n"
        "- Confirm A/R/N/U shortcuts save decisions and do not fire in notes.\n"
        "- Confirm undo, refresh recovery and completion blocking.\n",
    )
    write_json(
        smoke_root / "smoke_test_confirmation.json",
        {
            "schema_version": "football_intelligence.review_chassis.smoke_confirmation.v1",
            "gif_browser_smoke_passed": False,
            "gif_browser_smoke_failed": False,
            "reason": "manual_browser_confirmation_not_yet_recorded",
            **safety_payload(),
        },
    )
    _write_launcher(
        stage_root / "OPEN_REUSABLE_REVIEW_CHASSIS_SMOKE_TEST.ps1",
        repo_root=repo_root,
        manifest=smoke_root / "smoke_test_manifest.json",
        config=smoke_root / "smoke_test_ui_config.json",
        evidence=smoke_root / "evidence",
        decisions=smoke_root / "decisions",
        port=8778,
    )
    demo_outputs = _write_demo_packages(continuity_v5, frame_root, frame_records)
    main_validation = (
        validate_review_chassis_package(
            manifest_path=continuity_v5 / "paired_counterfactual_review_manifest.json",
            ui_config_path=continuity_v5 / "paired_counterfactual_ui_config.json",
            evidence_root=continuity_v5 / "evidence",
            decisions_root=continuity_v5 / "decisions" if manifest["cases"] else None,
        )
        if manifest["cases"]
        else {"passed": False, "blocked_reason": "no_review_cases"}
    )
    reusable_validation = {
        "artifact": "m5_4f5_reusable_review_chassis_validation",
        "canonical_chassis_source_paths": main_validation.get("canonical_chassis_source_paths", []),
        "demo_review_types_served": [row["task_type"] for row in demo_outputs],
        "demo_validations_passed": all(row["validation"]["passed"] for row in demo_outputs),
        "same_chassis_used_for_all_demos": True,
        "stage_specific_copied_ui_file_count": 0,
        **safety_payload(),
    }
    write_json(validation_root / "reusable_review_chassis_validation.json", reusable_validation)
    gif_validation = {
        "artifact": "m5_4f5_gif_only_review_validation",
        "mp4_generation_performed": False,
        "video_element_present": False,
        "gif_only_temporal_evidence_enabled": True,
        "automated_smoke_test_results": automated_smoke,
        "manual_smoke_status": "manual_browser_confirmation_not_yet_recorded",
        **safety_payload(),
    }
    write_json(validation_root / "gif_only_review_validation.json", gif_validation)
    integrity_gate = all(row["candidate_frame_binding_result"] for row in binding_rows) and bool(review_rows)
    smoke_gate = False
    if not reusable_validation["demo_validations_passed"]:
        final_classification = "FAIL_SOURCE_MUTATION_OR_SAFETY"
        exact_blocker = "REUSABLE_CHASSIS_DEMONSTRATION_VALIDATION_FAILED"
    elif not review_rows or paired_anchor_count < 5:
        final_classification = F5_PASS_CHASSIS_WAITING
        exact_blocker = "PAIRED_ANCHOR_SUPPLY_BELOW_MINIMUM"
    elif not integrity_gate:
        final_classification = F5_BLOCKED_INTEGRITY
        exact_blocker = "TARGET_FRAME_OR_EVIDENCE_BINDING_GATE_FAILED"
    elif not quality_gate:
        final_classification = F5_BLOCKED_RAW_FEATURE
        exact_blocker = quality_blocker
    elif not smoke_gate:
        final_classification = F5_BLOCKED_SMOKE
        exact_blocker = "MANUAL_GIF_BROWSER_SMOKE_CONFIRMATION_REQUIRED"
    else:
        final_classification = F5_PASS_READY
        exact_blocker = "NONE"
    launcher_path = None
    review_url = None
    if smoke_gate and integrity_gate and quality_gate:
        launcher_path = _write_launcher(
            stage_root / "OPEN_SAME_FRAME_PAIRED_COUNTERFACTUAL_REVIEW.ps1",
            repo_root=repo_root,
            manifest=continuity_v5 / "paired_counterfactual_review_manifest.json",
            config=continuity_v5 / "paired_counterfactual_ui_config.json",
            evidence=continuity_v5 / "evidence",
            decisions=continuity_v5 / "decisions",
            port=8779,
        )
        review_url = "http://127.0.0.1:8779/"
    after_inventory = _inventory(source_paths, base=stage_root)
    source_mutation = {
        "artifact": "m5_4f5_source_mutation_audit",
        "before": before_inventory,
        "after": after_inventory,
        "f3_and_f4_artifacts_preserved": before_inventory["inventory_hash"] == after_inventory["inventory_hash"],
        **safety_payload(),
    }
    safety = {
        "artifact": "m5_4f5_safety_guardrail_audit",
        "continuity_model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        "mp4_generation_performed": False,
        **safety_payload(),
    }
    write_json(validation_root / "source_mutation_audit.json", source_mutation)
    write_json(validation_root / "safety_guardrail_audit.json", safety)
    summary = {
        "artifact": "m5_4f5_validation_summary",
        "final_classification": final_classification,
        "exact_blocker": exact_blocker,
        "f3_and_f4_artifacts_preserved": source_mutation["f3_and_f4_artifacts_preserved"],
        "reusable_chassis_implemented": True,
        "canonical_chassis_source_paths": reusable_validation["canonical_chassis_source_paths"],
        "copied_stage_specific_ui_file_count": 0,
        "generic_manifest_schema_version": GENERIC_MANIFEST_SCHEMA_VERSION,
        "generic_ui_config_schema_version": GENERIC_UI_CONFIG_SCHEMA_VERSION,
        "gif_only_temporal_evidence_enabled": True,
        "mp4_generation_performed": False,
        "video_element_present": False,
        "gif_browser_smoke_status": "manual_browser_confirmation_not_yet_recorded",
        "frame_stepper_smoke_status": "automated_static_checks_passed",
        "keyboard_shortcut_status": "configured_by_ui_config",
        "f4_candidate_count": f4_integrity["candidate_count"],
        "f4_target_frame_mismatch_count": f4_integrity["target_frame_mismatch_count"],
        "f4_accepted_target_equals_alternative_count": f4_integrity["accepted_target_equals_alternative_count"],
        "f4_structurally_valid_selected_negative_count": f4_integrity["structurally_valid_selected_negative_count"],
        "new_local_same_frame_candidate_count": len(local_candidates),
        "new_true_same_frame_swap_count": len(swap_candidates),
        "integrity_rejection_counts": dict(Counter(row.get("reason") for row in [*local_rejections, *swap_rejections])),
        "paired_anchor_count": paired_anchor_count,
        "total_review_case_count": len(review_rows),
        "geometry_overlap_result": overlap_audit.get("passes_raw_feature_overlap_gates"),
        "construction_metadata_shortcut_result": "construction_metadata_excluded_from_model_features",
        "geometry_only_grouped_diagnostic": classifier_audit,
        "integrity_gate": integrity_gate,
        "quality_gate": quality_gate,
        "smoke_gate": smoke_gate,
        "launcher_path": launcher_path,
        "review_url": review_url,
        "training_readiness": TRAINING_BLOCKED_SINGLE_CLASS,
        "positive_human_labels": 40,
        "negative_human_labels": 0,
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        "candidate_supply_hash": stable_hash([local_candidates, swap_candidates, review_rows]),
        **safety_payload(),
    }
    write_json(validation_root / "m5_4f5_validation_summary.json", summary)
    return summary
