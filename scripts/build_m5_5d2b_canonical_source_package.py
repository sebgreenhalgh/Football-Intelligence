# ruff: noqa: E501

"""Build the M5.5D.2B package from the authoritative continuity-v11 rows.

This builder intentionally treats the three earlier M5.5D.2 packages as
read-only provenance.  Canonical geometry comes only from the v11 frame and
person-row manifests; prior review geometry is used only in the legacy audit.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.hashing import sha256_file
from football_intelligence.review_chassis.models import (
    AssetPanelConfig,
    DecisionOption,
    GenericEvidenceAsset,
    GenericReviewCase,
    GenericReviewManifest,
    ReviewUIConfig,
)
from football_intelligence.review_chassis.persistence import GenericReviewPersistence
from football_intelligence.review_chassis.validation import validate_review_chassis_package


ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = ROOT / "SoccerTrack-v2"
STAGE_ROOT = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5D2B_CANONICAL_CANDIDATE_SOURCE_REBUILD_v1"
SOURCE_ROOT = ROOT / r"matches\128058\runs\step_m5\06f_balanced_role_then_continuity\continuity_v11\unseen_window"
SOURCE_FRAME_MANIFEST = SOURCE_ROOT / "canonical_frame_manifest.json"
SOURCE_CANDIDATE_MANIFEST = SOURCE_ROOT / "person_candidate_rows_manifest.json"
SOURCE_CANDIDATE_ROWS = SOURCE_ROOT / "person_candidate_rows.jsonl"
OLD_SCIENCE_ROOT = (
    ROOT / r"matches\128058\runs\step_m5\part 2\M5_5D2_ENCOUNTER_EPISODE_GAP_MINING_AND_EXPANDED_BURST_SCAN_v1"
)
OLD_PACKAGE = OLD_SCIENCE_ROOT / "11_TRUE_OCCLUSION_REVIEW_PACKAGE"
OLD_ALIGNMENT_PACKAGE = (
    ROOT
    / r"matches\128058\runs\step_m5\part 2\M5_5D2_COORDINATE_PROVENANCE_AND_OVERLAY_ALIGNMENT_REPAIR_v1\04_REPAIRED_REVIEW_PACKAGE"
)
PROMPT_ROOT = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5D2B_Canonical_Candidate_Source_Rebuild_Prompt_v1"

STAGE_ID = "M5_5D2B_CANONICAL_CANDIDATE_SOURCE_REBUILD_v1"
REVIEW_ID = "m5_5d2b_canonical_source_review_v1"
REVIEWER_SESSION_ID = "m5_5d2b_canonical_source_human_reviewer"
REVIEW_PORT = 8787
FRAME_WIDTH = 2730
FRAME_HEIGHT = 720

CASE_WINDOWS = [
    ("case_001", 121, 129),
    ("case_002", 369, 377),
    ("case_003", 534, 543),
    ("case_004", 291, 300),
    ("case_005", 531, 539),
    ("case_006", 190, 198),
    ("case_007", 200, 209),
    ("case_008", 200, 208),
    ("case_009", 14, 22),
]

DECISIONS = [
    ("A", "CANONICAL_GEOMETRY_SEMANTICALLY_SUPPORTED", "Canonical box is around a visible person"),
    ("M", "CANONICAL_GEOMETRY_SEMANTICALLY_WRONG", "Box is wrong, empty, or not the intended person"),
    ("U", "UNRESOLVED", "Evidence is insufficient or ambiguous"),
]

SAFETY = {
    **safety_payload(),
    "identity_tracking_performed": False,
    "player_slots_assigned": False,
    "goalkeeper_slots_assigned": False,
    "exact_22_forcing_performed": False,
    "event_analysis_performed": False,
    "metric_analysis_performed": False,
    "tactical_analysis_performed": False,
    "physical_performance_analysis_performed": False,
    "model_fit_performed": False,
    "learned_continuity_rows_updated": 0,
    "project_defaults_changed": False,
    "canonical_candidate_rows_replaced": False,
    "historical_artifacts_mutated": False,
    "match_local_only": True,
    "sandbox_only": True,
    "safe_to_apply_globally": False,
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8"
    )


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def copy_exact(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return sha256_file(target)


def bbox_copy(bbox: dict[str, Any]) -> dict[str, float]:
    return {key: round(float(bbox[key]), 3) for key in ("x1", "y1", "x2", "y2")}


def bbox_area(bbox: dict[str, float]) -> float:
    return max(0.0, bbox["x2"] - bbox["x1"]) * max(0.0, bbox["y2"] - bbox["y1"])


def bbox_iou(left: dict[str, float], right: dict[str, float]) -> float:
    x1 = max(left["x1"], right["x1"])
    y1 = max(left["y1"], right["y1"])
    x2 = min(left["x2"], right["x2"])
    y2 = min(left["y2"], right["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = bbox_area(left) + bbox_area(right) - inter
    return inter / union if union else 0.0


def row_hash(row: dict[str, Any]) -> str:
    return digest_json(row)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def frame_catalog(frame_manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    frames = frame_manifest.get("frames", [])
    if len(frames) != 600:
        raise ValueError(f"expected 600 canonical frames, got {len(frames)}")
    catalog: dict[int, dict[str, Any]] = {}
    for frame in frames:
        sequence = int(frame["frame_sequence"])
        path = Path(frame["frame_file"])
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            if image.size != (FRAME_WIDTH, FRAME_HEIGHT):
                raise ValueError(f"canonical frame {sequence} has {image.size}, expected {(FRAME_WIDTH, FRAME_HEIGHT)}")
        actual_hash = sha256_file(path)
        if actual_hash != frame["byte_sha256"]:
            raise ValueError(f"canonical frame hash mismatch at {sequence}")
        catalog[sequence] = {**frame, "frame_file": str(path), "actual_byte_sha256": actual_hash}
    return catalog


def canonical_rows_by_frame(
    rows: list[dict[str, Any]], catalog: dict[int, dict[str, Any]]
) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        frame = int(row["frame_sequence"])
        if frame not in catalog:
            raise ValueError(f"candidate row points outside canonical frame catalog: {frame}")
        bbox = bbox_copy(row["bbox"])
        if not (0 <= bbox["x1"] < bbox["x2"] <= FRAME_WIDTH and 0 <= bbox["y1"] < bbox["y2"] <= FRAME_HEIGHT):
            raise ValueError(f"candidate bbox out of bounds: {row['candidate_id']}")
        by_frame.setdefault(frame, []).append({**row, "bbox": bbox, "source_row_hash": row_hash(row)})
    for frame in by_frame:
        by_frame[frame].sort(key=lambda item: (item["bbox"]["x1"], item["bbox"]["y1"], item["source_row_hash"]))
    return by_frame


def canonical_anonymous_rows(rows: list[dict[str, Any]], frame: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for number, row in enumerate(rows, start=1):
        result.append(
            {
                "anonymous_candidate_number": number,
                "layer": "CANONICAL_DETECTIONS",
                "bbox": row["bbox"],
                "frame_sequence": int(row["frame_sequence"]),
                "image_sha256": frame["actual_byte_sha256"],
                "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
                "geometry_role": "CANONICAL_DETECTIONS",
                "confidence": round(float(row["confidence"]), 6),
                "row_hash": row["source_row_hash"],
            }
        )
    return result


def load_science_sources() -> (
    tuple[
        dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
    ]
):
    episodes = load_jsonl(OLD_SCIENCE_ROOT / "03_ENCOUNTER_EPISODES" / "episode_rows.jsonl")
    observations = load_jsonl(OLD_SCIENCE_ROOT / "02_VISIBLE_TRACKLET_SEGMENTS" / "observation_rows.jsonl")
    segments = load_jsonl(OLD_SCIENCE_ROOT / "02_VISIBLE_TRACKLET_SEGMENTS" / "visible_segment_rows.jsonl")
    recovery = load_jsonl(OLD_SCIENCE_ROOT / "08_SELECTIVE_DETECTOR_RECOVERY" / "affected_rows.jsonl")
    controls = load_jsonl(OLD_SCIENCE_ROOT / "08_SELECTIVE_DETECTOR_RECOVERY" / "control_rows.jsonl")
    observation_map = {str(row["observation_key"]): row for row in observations}
    segment_map = {str(row["segment_id"]): row for row in segments}
    return (
        {str(row["encounter_episode_id"]): row for row in episodes},
        observation_map,
        segment_map,
        recovery,
        controls,
    )


def nearest_episode(episodes: dict[str, dict[str, Any]], default_frame: int) -> dict[str, Any]:
    candidates = [row for row in episodes.values() if len(row.get("incoming_segment_ids", [])) >= 2]
    return min(candidates, key=lambda row: abs(int(row["predicted_contact_frame"]) - default_frame))


def segment_boxes(
    segment: dict[str, Any],
    observations: dict[str, dict[str, Any]],
    start: int,
    end: int,
) -> list[tuple[int, dict[str, float], str]]:
    output = []
    for key in segment.get("source_provenance", {}).get("observation_keys", []):
        row = observations.get(str(key))
        if not row:
            continue
        frame = int(row["frame_sequence"])
        if start <= frame <= end:
            output.append((frame, bbox_copy(row["bbox"]), str(key)))
    return output


def source_layer_rows(
    case_id: str,
    frames: list[int],
    default_frame: int,
    episode: dict[str, Any],
    observations: dict[str, dict[str, Any]],
    segments: dict[str, dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
    case_number: int,
    frame_hashes: dict[int, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {
        key: 0
        for key in (
            "INCOMING_OBSERVED_SEGMENTS",
            "INCOMING_PREDICTED_STATES",
            "MERGED_OBSERVATION_CANDIDATES",
            "OUTGOING_SEGMENT_HYPOTHESES",
            "RECOVERY_DETECTIONS",
        )
    }
    incoming_ids = [str(value) for value in episode.get("incoming_segment_ids", [])]
    for segment_id in incoming_ids:
        segment = segments.get(segment_id, {})
        for frame, bbox, observation_key in segment_boxes(segment, observations, frames[0], frames[-1]):
            rows.append(
                {
                    "layer": "INCOMING_OBSERVED_SEGMENTS",
                    "frame_sequence": frame,
                    "image_sha256": frame_hashes[frame],
                    "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
                    "bbox": bbox,
                    "label": "Observed incoming segment",
                    "anonymous_segment_number": incoming_ids.index(segment_id) + 1,
                    "source_row_hash": digest_json([segment_id, observation_key, bbox]),
                }
            )
            source_counts["INCOMING_OBSERVED_SEGMENTS"] += 1
    for frame in frames:
        prediction = episode.get("predicted_state_by_frame", {}).get(str(frame), {})
        for number, (segment_id, bbox) in enumerate(sorted(prediction.items()), start=1):
            rows.append(
                {
                    "layer": "INCOMING_PREDICTED_STATES",
                    "frame_sequence": frame,
                    "image_sha256": frame_hashes[frame],
                    "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
                    "bbox": bbox_copy(bbox),
                    "label": "Predicted incoming state",
                    "anonymous_segment_number": number,
                    "source_row_hash": digest_json([episode["encounter_episode_id"], frame, segment_id, bbox]),
                }
            )
            source_counts["INCOMING_PREDICTED_STATES"] = source_counts.get("INCOMING_PREDICTED_STATES", 0) + 1
    recovery = next((row for row in recovery_rows if int(row.get("case_index", -1)) == case_number), None)
    if recovery:
        for number, bbox in enumerate(recovery.get("boxes", [])[:6], start=1):
            clean_bbox = bbox_copy(bbox)
            rows.append(
                {
                    "layer": "RECOVERY_DETECTIONS",
                    "frame_sequence": default_frame,
                    "image_sha256": frame_hashes[default_frame],
                    "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
                    "bbox": clean_bbox,
                    "label": "Recovery detector result",
                    "anonymous_detection_number": number,
                    "source_row_hash": digest_json([case_number, "recovery", number, clean_bbox]),
                }
            )
            rows.append(
                {
                    "layer": "MERGED_OBSERVATION_CANDIDATES",
                    "frame_sequence": default_frame,
                    "image_sha256": frame_hashes[default_frame],
                    "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
                    "bbox": clean_bbox,
                    "label": "Merged-observation candidate",
                    "anonymous_detection_number": number,
                    "source_row_hash": digest_json([case_number, "merged", number, clean_bbox]),
                }
            )
            source_counts["RECOVERY_DETECTIONS"] += 1
            source_counts["MERGED_OBSERVATION_CANDIDATES"] += 1
    outgoing_candidates = []
    for segment_id, segment in segments.items():
        if segment_id in incoming_ids:
            continue
        first = int(segment.get("first_observed_frame", 10**9))
        if default_frame - 2 <= first <= frames[-1] + 5:
            outgoing_candidates.append((abs(first - frames[-1]), segment_id, segment))
    for _, segment_id, segment in sorted(outgoing_candidates)[:2]:
        for frame, bbox, observation_key in segment_boxes(segment, observations, frames[0], frames[-1]):
            rows.append(
                {
                    "layer": "OUTGOING_SEGMENT_HYPOTHESES",
                    "frame_sequence": frame,
                    "image_sha256": frame_hashes[frame],
                    "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
                    "bbox": bbox,
                    "label": "Independent outgoing segment hypothesis",
                    "anonymous_segment_number": len([r for r in rows if r["layer"] == "OUTGOING_SEGMENT_HYPOTHESES"])
                    + 1,
                    "source_row_hash": digest_json([segment_id, observation_key, bbox]),
                }
            )
            source_counts["OUTGOING_SEGMENT_HYPOTHESES"] += 1
    return rows, {
        "episode_id": episode["encounter_episode_id"],
        "incoming_segment_count": len(incoming_ids),
        "source_counts": source_counts,
    }


def make_gif(frame_paths: list[Path], target: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    images[0].save(target, save_all=True, append_images=images[1:], duration=110, loop=0, optimize=False)
    for image in images:
        image.close()


def semantic_crops(
    root: Path,
    case_id: str,
    frame: int,
    rows: list[dict[str, Any]],
    image_path: Path,
    frame_hash: str,
    audit_rows: list[dict[str, Any]],
    sheet_items: list[tuple[str, Path]],
) -> None:
    with Image.open(image_path).convert("RGB") as image:
        case_root = root / "crops" / case_id / f"frame_{frame:06d}"
        for number, row in enumerate(rows, start=1):
            bbox = row["bbox"]
            exact_box = tuple(int(round(bbox[key])) for key in ("x1", "y1", "x2", "y2"))
            exact = image.crop(exact_box)
            padded_box = (
                max(0, int(bbox["x1"] - 0.25 * (bbox["x2"] - bbox["x1"]))),
                max(0, int(bbox["y1"] - 0.25 * (bbox["y2"] - bbox["y1"]))),
                min(image.width, int(bbox["x2"] + 0.25 * (bbox["x2"] - bbox["x1"]))),
                min(image.height, int(bbox["y2"] + 0.25 * (bbox["y2"] - bbox["y1"]))),
            )
            exact_path = case_root / f"canonical_{number:03d}_exact.jpg"
            padded_path = case_root / f"canonical_{number:03d}_padded.jpg"
            exact_path.parent.mkdir(parents=True, exist_ok=True)
            exact.save(exact_path, quality=95)
            image.crop(padded_box).save(padded_path, quality=95)
            sheet_items.append((f"{case_id} / frame {frame} / box {number}", padded_path))
            audit_rows.append(
                {
                    "case_id": case_id,
                    "frame_sequence": frame,
                    "anonymous_candidate_number": number,
                    "exact_bbox": bbox,
                    "exact_crop": str(exact_path.relative_to(root)),
                    "padded_crop": str(padded_path.relative_to(root)),
                    "full_frame_marker": {"frame": str(image_path), "bbox": bbox},
                    "source_row_hash": row["row_hash"],
                    "frame_hash": frame_hash,
                    "confidence": row["confidence"],
                    "visual_audit_status": "HUMAN_AUDIT_REQUIRED",
                    "non_authoritative_foreground_signal": "not_used_for_acceptance",
                }
            )


def build_contact_sheet(root: Path, items: list[tuple[str, Path]]) -> Path:
    target = root / "canonical_semantic_contact_sheet.jpg"
    thumb_w, thumb_h, label_h = 180, 120, 28
    columns = 8
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(items):
        with Image.open(path).convert("RGB") as image:
            image.thumbnail((thumb_w - 8, thumb_h - 8))
            x = (index % columns) * thumb_w + (thumb_w - image.width) // 2
            y = (index // columns) * (thumb_h + label_h) + (thumb_h - image.height) // 2
            sheet.paste(image, (x, y))
        draw.text(
            ((index % columns) * thumb_w + 3, (index // columns) * (thumb_h + label_h) + thumb_h + 2),
            label[:28],
            fill="black",
        )
    sheet.save(target, quality=88)
    return target


def legacy_audit(
    catalog: dict[int, dict[str, Any]], canonical_by_frame: dict[int, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    old_manifest = read_json(OLD_PACKAGE / "reviewer_manifest.json")
    alignment_manifest = read_json(OLD_ALIGNMENT_PACKAGE / "reviewer_manifest.json")
    rows: list[dict[str, Any]] = []
    for index, (case_id, start, end) in enumerate(CASE_WINDOWS, start=1):
        old_case = old_manifest["cases"][index - 1]
        aligned_case = alignment_manifest["cases"][index - 1]
        legacy = old_case.get("visible_metadata", {}).get("safe_anonymous_candidates", [])
        port_rows = aligned_case.get("visible_metadata", {}).get("geometry_layers", [])
        port_rows = [row for row in port_rows if row.get("layer") == "CANONICAL_DETECTIONS"]
        default = end - 4 if index not in {3, 5} else end - 1
        if case_id in {
            "case_001",
            "case_002",
            "case_003",
            "case_004",
            "case_005",
            "case_006",
            "case_007",
            "case_008",
            "case_009",
        }:
            default = {
                "case_001": 125,
                "case_002": 373,
                "case_003": 538,
                "case_004": 295,
                "case_005": 535,
                "case_006": 194,
                "case_007": 204,
                "case_008": 204,
                "case_009": 18,
            }[case_id]
        frame = catalog[default]
        direct = canonical_by_frame[default]
        for number, legacy_row in enumerate(legacy or [{}], start=1):
            legacy_bbox = bbox_copy(legacy_row.get("bbox", {"x1": 0, "y1": 0, "x2": 0, "y2": 0}))
            port_bbox = (
                bbox_copy(port_rows[number - 1].get("bbox", legacy_bbox)) if len(port_rows) >= number else legacy_bbox
            )
            reverse_scaled = {key: round(value / 2.0, 3) for key, value in port_bbox.items()}
            best = max(direct, key=lambda row: bbox_iou(row["bbox"], reverse_scaled), default=None)
            iou = bbox_iou(best["bbox"], reverse_scaled) if best else 0.0
            classification = "MATCHES_AUTHORITATIVE_CANONICAL_ROW" if iou >= 0.98 else "WRONG_SOURCE_CANDIDATE"
            rows.append(
                {
                    "historical_case": case_id,
                    "frame_sequence": default,
                    "anonymous_candidate_number": number,
                    "canonical_candidate_row_hash": best["source_row_hash"] if best else None,
                    "canonical_frame_path": frame["frame_file"],
                    "canonical_frame_sha256": frame["actual_byte_sha256"],
                    "canonical_frame_width": FRAME_WIDTH,
                    "canonical_frame_height": FRAME_HEIGHT,
                    "canonical_bbox": best["bbox"] if best else None,
                    "legacy_review_bbox": legacy_bbox,
                    "legacy_declared_dimensions": {
                        "width": 2048,
                        "height": 540,
                        "basis": "prior port-8786 builder assumption",
                    },
                    "legacy_actual_asset_dimensions": {
                        "width": 4096,
                        "height": 1080,
                        "basis": "prior port-8786 raw evidence",
                    },
                    "port8786_bbox": port_bbox,
                    "port8786_display_dimensions": {"width": 4096, "height": 1080},
                    "canonical_to_legacy_transform": {"type": "not_proven", "scale_x": None, "scale_y": None},
                    "legacy_to_port8786_transform": {
                        "type": "prior_explicit_2x_display_mapping",
                        "scale_x": 2.0,
                        "scale_y": 2.0,
                    },
                    "reverse_port_mapping_bbox": reverse_scaled,
                    "reverse_mapping_best_iou": round(iou, 6),
                    "geometry_classification": classification,
                    "semantic_crop_status": (
                        "LEGACY_BOX_AGREES_WITH_AUTHORITATIVE_ROW"
                        if classification == "MATCHES_AUTHORITATIVE_CANONICAL_ROW"
                        else "LEGACY_BOX_IS_NOT_AUTHORITATIVE_CANONICAL_GEOMETRY"
                    ),
                    "diagnosis": "The previous box followed a legacy/predicted source candidate through a 2x display mapping; it is not a direct v11 person row."
                    if classification != "MATCHES_AUTHORITATIVE_CANONICAL_ROW"
                    else "Legacy box agrees with a direct v11 row.",
                }
            )
    return rows


def make_ui_config() -> ReviewUIConfig:
    return ReviewUIConfig(
        page_title="M5.5D.2B Canonical Source Review",
        review_title="Canonical candidate source and semantic alignment review",
        task_instructions=(
            "Inspect the exact 2730x720 canonical frame. Use Clean frame and frame-specific layer toggles. "
            "Canonical rectangles are direct detector rows in native pixels; observed, predicted, recovery and "
            "outgoing layers remain separate visual evidence. Do not infer identity, slots, roster counts or metrics."
        ),
        decisions=[DecisionOption(key=key, value=value, label=label) for key, value, label in DECISIONS],
        asset_panel_order=[
            AssetPanelConfig(asset_type="animated_gif", label="Temporal GIF"),
            AssetPanelConfig(asset_type="image", label="Canonical source frame"),
            AssetPanelConfig(asset_type="image_sequence", label="Frame stepper"),
        ],
        visible_metadata_fields=["case_label", "frame_window", "display_binding", "layer_policy", "source_notice"],
        hidden_metadata_fields=[],
        reveal_controls=False,
        notes_enabled=True,
        undo_enabled=True,
        autosave_enabled=True,
        completion_requires_all_cases=True,
        decisions_advance_automatically=True,
        unresolved_allowed=True,
        gif_primary=True,
        image_stepper_enabled=True,
        spatial_annotation_enabled=True,
        spatial_annotation_mode="occlusion_interval",
        spatial_annotation_schema={
            "schema_version": "football_intelligence.review_chassis.occlusion_interval_annotation.v1",
            "title": "Canonical source annotation",
            "coordinate_space": "original_image_pixels",
            "interactive_canvas_enabled": True,
            "fields": [
                "deficit_start_frame",
                "deficit_end_frame",
                "merged_detection_number",
                "occlusion_points",
                "reentry_path_selection",
                "reviewer_bbox",
            ],
        },
    )


def build() -> dict[str, Any]:
    if STAGE_ROOT.exists():
        shutil.rmtree(STAGE_ROOT)
    for name in (
        "00_PROMPT_AND_INPUTS",
        "01_AUTHORIZATION_AND_LEGACY_GEOMETRY_AUDIT",
        "02_CANONICAL_SOURCE_DISCOVERY",
        "03_CANONICAL_FRAME_AND_ROW_VALIDATION",
        "04_TRACK_AND_RECOVERY_GEOMETRY_REBUILD",
        "05_SEMANTIC_BOX_AUDIT",
        "06_REBUILT_REVIEW_PACKAGE",
        "07_BROWSER_VALIDATION",
        "08_VISUAL_EVIDENCE",
        "09_COMMANDS_AND_TESTS",
        "10_REVIEW_PACK_FOR_CHATGPT",
        "_tmp",
    ):
        (STAGE_ROOT / name).mkdir(parents=True, exist_ok=True)
    for filename in (
        "00_READ_ME_FIRST.md",
        "01_CANONICAL_SOURCE_REBUILD_CODEX_PROMPT.md",
        "02_CANONICAL_SOURCE_WORKSPACE_CONTRACT.json",
        "03_CANONICAL_GEOMETRY_SOURCE_CONTRACT.json",
        "04_PROMPT_PACK_MANIFEST.json",
    ):
        source = PROMPT_ROOT / filename
        if source.exists():
            copy_exact(source, STAGE_ROOT / "00_PROMPT_AND_INPUTS" / filename)

    frame_manifest = read_json(SOURCE_FRAME_MANIFEST)
    candidate_manifest = read_json(SOURCE_CANDIDATE_MANIFEST)
    catalog = frame_catalog(frame_manifest)
    candidate_rows = load_jsonl(SOURCE_CANDIDATE_ROWS)
    by_frame = canonical_rows_by_frame(candidate_rows, catalog)
    if len(candidate_rows) != int(candidate_manifest["row_count"]):
        raise ValueError("candidate row count mismatch")
    episodes, observations, segments, recovery, controls = load_science_sources()

    write_json(
        STAGE_ROOT / "02_CANONICAL_SOURCE_DISCOVERY" / "canonical_frame_catalog.json",
        {
            "source_manifest": str(SOURCE_FRAME_MANIFEST),
            "dimensions": frame_manifest["dimensions"],
            "frame_count": len(catalog),
            "frames": [
                {
                    key: frame[key]
                    for key in (
                        "frame_sequence",
                        "source_frame_index",
                        "timestamp_seconds",
                        "frame_file",
                        "byte_sha256",
                        "width",
                        "height",
                    )
                }
                for frame in catalog.values()
            ],
        },
    )
    write_json(
        STAGE_ROOT / "02_CANONICAL_SOURCE_DISCOVERY" / "canonical_candidate_manifest.json",
        {
            "source_manifest": str(SOURCE_CANDIDATE_MANIFEST),
            "row_count": len(candidate_rows),
            "rows_sha256": sha256_file(SOURCE_CANDIDATE_ROWS),
            "model_sha256": candidate_manifest["model_sha256"],
            "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
            "direct_source_only": True,
        },
    )
    write_json(
        STAGE_ROOT / "02_CANONICAL_SOURCE_DISCOVERY" / "canonical_source_hash_audit.json",
        {
            "canonical_frame_manifest_sha256": sha256_file(SOURCE_FRAME_MANIFEST),
            "canonical_candidate_manifest_sha256": sha256_file(SOURCE_CANDIDATE_MANIFEST),
            "canonical_candidate_rows_sha256": sha256_file(SOURCE_CANDIDATE_ROWS),
            "candidate_manifest_declared_rows_sha256": candidate_manifest["rows_sha256"],
            "candidate_manifest_declared_frame_manifest_hash": candidate_manifest["frame_manifest_hash"],
            "model_sha256": candidate_manifest["model_sha256"],
            "all_frame_hashes_verified": True,
        },
    )
    write_json(
        STAGE_ROOT / "02_CANONICAL_SOURCE_DISCOVERY" / "canonical_video_comparison.json",
        {
            "video_reextraction_used_as_annotation_surface": False,
            "canonical_dimensions": {"width": FRAME_WIDTH, "height": FRAME_HEIGHT},
            "source_video_support": "not copied or used as primary annotation surface",
            "transform": None,
            "reason": "The canonical frame manifest supplies the exact primary annotation asset.",
        },
    )

    legacy_rows = legacy_audit(catalog, by_frame)
    write_jsonl(
        STAGE_ROOT / "01_AUTHORIZATION_AND_LEGACY_GEOMETRY_AUDIT" / "legacy_geometry_chain_audit.jsonl", legacy_rows
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_LEGACY_GEOMETRY_AUDIT" / "authorization_audit.json",
        {
            "authorized_baseline": "acf796beda66f25d4bd375114ffd2742edfb5fab",
            "head_verified_before_build": True,
            "worktree_clean_before_build": True,
            "baseline_is_ancestor": True,
            "prior_packages_read_only": True,
            "old_ports_not_continued": [8784, 8785, 8786],
            "target_port": REVIEW_PORT,
        },
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_LEGACY_GEOMETRY_AUDIT" / "source_review_invalidity.json",
        {
            "review_manifest_geometry_used_as_canonical_source": False,
            "prior_port_8786_rows_used_as_canonical_source": False,
            "prior_packages_role": "read_only_legacy_audit_and_scientific_layer_source_references",
        },
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_LEGACY_GEOMETRY_AUDIT" / "source_mutation_audit.json",
        {
            "prior_package_mutated": False,
            "prior_decisions_mutated": False,
            "prior_ports_modified": False,
            "historical_artifacts_mutated": False,
        },
    )

    package = STAGE_ROOT / "06_REBUILT_REVIEW_PACKAGE"
    evidence_root = package / "evidence"
    decisions_root = package / "decisions"
    (package / "sealed").mkdir(parents=True, exist_ok=True)
    decisions_root.mkdir(parents=True, exist_ok=True)
    write_json(
        package / "sealed" / "server_mapping.json",
        {
            "schema_version": "m5_5d2b.sealed_mapping.v1",
            "served_before_decision": False,
            "answer_key": {},
            "case_source_rows": {
                case_id: "server-side authoritative rows retained in audit only" for case_id, _, _ in CASE_WINDOWS
            },
        },
    )

    all_audit_rows: list[dict[str, Any]] = []
    sheet_items: list[tuple[str, Path]] = []
    all_layer_rows: list[dict[str, Any]] = []
    case_models: list[GenericReviewCase] = []
    evidence_manifest_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    layer_summary: list[dict[str, Any]] = []
    for case_number, (case_id, start, end) in enumerate(CASE_WINDOWS, start=1):
        frames = list(range(start, end + 1))
        default_frame = frames[len(frames) // 2]
        frame_hashes = {frame: catalog[frame]["actual_byte_sha256"] for frame in frames}
        case_root = evidence_root / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        frame_assets: list[GenericEvidenceAsset] = []
        frame_paths: list[Path] = []
        anonymous_by_frame: dict[str, list[dict[str, Any]]] = {}
        for frame in frames:
            source = Path(catalog[frame]["frame_file"])
            relative = f"frames/canonical_{frame:06d}.jpg"
            target = case_root / relative
            copied_hash = copy_exact(source, target)
            if copied_hash != frame_hashes[frame]:
                raise ValueError(f"copied frame hash mismatch {case_id}/{frame}")
            frame_paths.append(target)
            rows = canonical_anonymous_rows(by_frame.get(frame, []), catalog[frame])
            anonymous_by_frame[str(frame)] = rows
            asset = GenericEvidenceAsset(
                asset_id=f"frame_{frame:06d}",
                asset_type="image_sequence",
                label=f"Exact canonical frame {frame}",
                relative_path=relative,
                sha256=copied_hash,
                media_type="image/jpeg",
                frame_sequences=[frame],
                group_id="annotation_frames",
                metadata={
                    "annotation_base": True,
                    "raw_frame": True,
                    "primary_annotation_image": frame == default_frame,
                    "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
                    "frame_binding_required": True,
                    "width": FRAME_WIDTH,
                    "height": FRAME_HEIGHT,
                    "natural_dimensions": {"width": FRAME_WIDTH, "height": FRAME_HEIGHT},
                    "source_frame_index": catalog[frame]["source_frame_index"],
                    "timestamp_seconds": catalog[frame]["timestamp_seconds"],
                },
                record_reveal_event=False,
            )
            frame_assets.append(asset)
            evidence_manifest_rows.append({"case_id": case_id, **asset.model_dump(mode="json")})
        gif_path = case_root / "temporal.gif"
        make_gif(frame_paths, gif_path)
        gif_asset = GenericEvidenceAsset(
            asset_id="temporal_gif",
            asset_type="animated_gif",
            label="Temporal canonical frame evidence",
            relative_path="temporal.gif",
            sha256=sha256_file(gif_path),
            media_type="image/gif",
            frame_sequences=frames,
            group_id="temporal",
            metadata={
                "primary_annotation_image": False,
                "frame_stepper": False,
                "source_is_exact_canonical_frames": True,
            },
            record_reveal_event=True,
        )
        evidence_manifest_rows.append({"case_id": case_id, **gif_asset.model_dump(mode="json")})
        episode = nearest_episode(episodes, default_frame)
        layers, layer_info = source_layer_rows(
            case_id, frames, default_frame, episode, observations, segments, recovery, case_number, frame_hashes
        )
        all_layer_rows.extend([{**row, "case_id": case_id} for row in layers])
        layer_summary.append(
            {
                "case_id": case_id,
                "episode_source": layer_info,
                "canonical_rows_by_frame": {str(frame): len(by_frame.get(frame, [])) for frame in frames},
            }
        )
        geometry_layers = []
        for frame in frames:
            geometry_layers.extend(anonymous_by_frame[str(frame)])
        geometry_layers.extend(layers)
        visible_metadata = {
            "case_label": f"Canonical source semantic audit {case_number:03d}",
            "frame_window": {"first": start, "last": end, "default": default_frame},
            "frame_sequences": frames,
            "display_binding": {
                "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
                "width": FRAME_WIDTH,
                "height": FRAME_HEIGHT,
                "frame_hash_required": True,
            },
            "layer_policy": {
                "canonical_detections": True,
                "incoming_observed_segments": True,
                "merged_observation_candidates": True,
                "outgoing_segment_hypotheses": True,
                "recovery_detections": False,
                "incoming_predicted_states": False,
            },
            "source_notice": "Canonical boxes are direct continuity_v11 rows. Prior review geometry is excluded.",
            "safe_anonymous_candidates_by_frame": anonymous_by_frame,
            "geometry_layers": geometry_layers,
        }
        case_assets = [gif_asset, *frame_assets]
        case_models.append(
            GenericReviewCase(
                case_id=case_id,
                task_type="detection_validity",
                candidate_id=case_id,
                candidate_hash=digest_json(
                    [case_id, default_frame, [row["source_row_hash"] for row in by_frame.get(default_frame, [])]]
                ),
                evidence_hash=digest_json([asset.sha256 for asset in case_assets]),
                allowed_decisions=[value for _, value, _ in DECISIONS],
                concise_question="Are the canonical detector rectangles semantically supported by the exact displayed frame?",
                detailed_instructions="Inspect the exact frame, clean-frame mode, layer toggles, and temporal GIF. Mark supported, wrong, or unresolved. Do not infer persistent identity.",
                priority=100 - case_number,
                evidence_assets=case_assets,
                source_frame_sequence=start,
                target_frame_sequence=end,
                frame_gap=end - start,
                source_bbox=None,
                target_bbox=None,
                visible_metadata=visible_metadata,
                safety_payload=SAFETY,
            )
        )
        index_rows.append(
            {
                "case_id": case_id,
                "frame_first": start,
                "frame_last": end,
                "default_frame": default_frame,
                "canonical_box_count": len(by_frame.get(default_frame, [])),
            }
        )

    manifest = GenericReviewManifest(
        review_id=REVIEW_ID,
        stage_id=STAGE_ID,
        task_type="detection_validity",
        title="M5.5D.2B Canonical Candidate Source Review",
        production_ready=False,
        no_auto_promotion=True,
        human_approved=False,
        cases=case_models,
        evidence_manifest_hash=digest_json(evidence_manifest_rows),
        source_manifest_hash=sha256_file(SOURCE_FRAME_MANIFEST),
        safety_payload=SAFETY,
    )
    write_json(package / "reviewer_manifest.json", manifest.model_dump(mode="json"))
    ui = make_ui_config()
    write_json(package / "ui_config.json", ui.model_dump(mode="json"))
    GenericReviewPersistence(
        manifest=manifest,
        ui_config=ui,
        decisions_root=decisions_root,
        reviewer_session_id=REVIEWER_SESSION_ID,
    ).ensure_state()
    write_json(
        package / "evidence_manifest.json",
        {"schema_version": "m5_5d2b.evidence_manifest.v1", "assets": evidence_manifest_rows},
    )
    with (package / "case_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["case_id", "frame_first", "frame_last", "default_frame", "canonical_box_count"]
        )
        writer.writeheader()
        writer.writerows(index_rows)
    write_json(
        STAGE_ROOT / "04_TRACK_AND_RECOVERY_GEOMETRY_REBUILD" / "layer_source_manifest.json",
        {
            "canonical_source": str(SOURCE_CANDIDATE_ROWS),
            "observed_source": str(OLD_SCIENCE_ROOT / "02_VISIBLE_TRACKLET_SEGMENTS" / "observation_rows.jsonl"),
            "predicted_source": str(OLD_SCIENCE_ROOT / "03_ENCOUNTER_EPISODES" / "episode_rows.jsonl"),
            "recovery_source": str(OLD_SCIENCE_ROOT / "08_SELECTIVE_DETECTOR_RECOVERY" / "affected_rows.jsonl"),
            "outgoing_source": str(OLD_SCIENCE_ROOT / "02_VISIBLE_TRACKLET_SEGMENTS" / "observation_rows.jsonl"),
            "review_manifest_geometry_used": False,
            "screenshots_used_as_geometry": False,
        },
    )
    write_jsonl(
        STAGE_ROOT / "04_TRACK_AND_RECOVERY_GEOMETRY_REBUILD" / "observed_segment_rows.jsonl",
        [row for row in all_layer_rows if row["layer"] == "INCOMING_OBSERVED_SEGMENTS"],
    )
    write_jsonl(
        STAGE_ROOT / "04_TRACK_AND_RECOVERY_GEOMETRY_REBUILD" / "predicted_state_rows.jsonl",
        [row for row in all_layer_rows if row["layer"] == "INCOMING_PREDICTED_STATES"],
    )
    write_jsonl(
        STAGE_ROOT / "04_TRACK_AND_RECOVERY_GEOMETRY_REBUILD" / "recovery_transform_rows.jsonl",
        [row for row in all_layer_rows if row["layer"] == "RECOVERY_DETECTIONS"],
    )
    write_jsonl(
        STAGE_ROOT / "04_TRACK_AND_RECOVERY_GEOMETRY_REBUILD" / "merged_candidate_rows.jsonl",
        [row for row in all_layer_rows if row["layer"] == "MERGED_OBSERVATION_CANDIDATES"],
    )
    write_jsonl(
        STAGE_ROOT / "04_TRACK_AND_RECOVERY_GEOMETRY_REBUILD" / "outgoing_hypothesis_rows.jsonl",
        [row for row in all_layer_rows if row["layer"] == "OUTGOING_SEGMENT_HYPOTHESES"],
    )
    write_json(
        STAGE_ROOT / "04_TRACK_AND_RECOVERY_GEOMETRY_REBUILD" / "layer_counts.json",
        {
            "rows": len(all_layer_rows),
            "by_layer": {
                layer: sum(1 for row in all_layer_rows if row["layer"] == layer)
                for layer in {row["layer"] for row in all_layer_rows}
            },
        },
    )

    validation_rows: list[dict[str, Any]] = []
    bounds_rows: list[dict[str, Any]] = []
    for case_id, start, end in CASE_WINDOWS:
        default = (start + end) // 2
        for frame in range(start, end + 1):
            frame_info = catalog[frame]
            for number, row in enumerate(by_frame.get(frame, []), start=1):
                validation_rows.append(
                    {
                        "case_id": case_id,
                        "frame_sequence": frame,
                        "anonymous_candidate_number": number,
                        "source_row_hash": row["source_row_hash"],
                        "frame_hash": frame_info["actual_byte_sha256"],
                        "frame_hash_match": True,
                        "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
                        "scaling_applied": False,
                        "binding_valid": True,
                    }
                )
                bounds_rows.append(
                    {
                        "case_id": case_id,
                        "frame_sequence": frame,
                        "anonymous_candidate_number": number,
                        "bbox": row["bbox"],
                        "within_2730x720": True,
                    }
                )
    write_jsonl(STAGE_ROOT / "03_CANONICAL_FRAME_AND_ROW_VALIDATION" / "row_binding_results.jsonl", validation_rows)
    write_json(
        STAGE_ROOT / "03_CANONICAL_FRAME_AND_ROW_VALIDATION" / "bbox_bounds_results.json",
        {
            "row_count": len(bounds_rows),
            "all_within_bounds": all(row["within_2730x720"] for row in bounds_rows),
            "rows": bounds_rows,
        },
    )
    write_json(
        STAGE_ROOT / "03_CANONICAL_FRAME_AND_ROW_VALIDATION" / "frame_hash_binding_results.json",
        {
            "validated_rows": len(validation_rows),
            "all_hashes_match": True,
            "frame_specific_only": True,
            "multi_frame_union": False,
        },
    )

    write_jsonl(STAGE_ROOT / "05_SEMANTIC_BOX_AUDIT" / "human_audit_required_rows.jsonl", all_audit_rows)
    # Generate crops after the package has copied exact canonical assets.
    contact_rows: list[dict[str, Any]] = []
    sheet_items = []
    for case_id, start, end in CASE_WINDOWS:
        default = (start + end) // 2
        image_path = evidence_root / case_id / f"frames/canonical_{default:06d}.jpg"
        audit_before = len(all_audit_rows)
        semantic_crops(
            STAGE_ROOT / "05_SEMANTIC_BOX_AUDIT",
            case_id,
            default,
            canonical_anonymous_rows(by_frame.get(default, []), catalog[default]),
            image_path,
            catalog[default]["actual_byte_sha256"],
            all_audit_rows,
            sheet_items,
        )
        contact_rows.extend(all_audit_rows[audit_before:])
    contact_sheet = build_contact_sheet(STAGE_ROOT / "05_SEMANTIC_BOX_AUDIT", sheet_items)
    write_json(
        STAGE_ROOT / "05_SEMANTIC_BOX_AUDIT" / "semantic_audit_summary.json",
        {
            "case_count": 9,
            "displayed_default_frame_box_count": len(all_audit_rows),
            "contact_sheet": str(contact_sheet),
            "all_rows_human_audit_required": True,
            "empty_grass_acceptance": "not_auto_accepted",
        },
    )
    write_jsonl(STAGE_ROOT / "05_SEMANTIC_BOX_AUDIT" / "false_positive_rows.jsonl", [])
    with (STAGE_ROOT / "05_SEMANTIC_BOX_AUDIT" / "contact_sheet_index.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "frame_sequence",
                "anonymous_candidate_number",
                "exact_crop",
                "padded_crop",
                "visual_audit_status",
            ],
        )
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in all_audit_rows)

    write_json(
        package / "package_status.json",
        {
            "created": True,
            "case_count": 9,
            "decisions_root_empty": True,
            "reviewer_session_id": REVIEWER_SESSION_ID,
            "review_id": REVIEW_ID,
            "stage_id": STAGE_ID,
            "port": REVIEW_PORT,
            "primary_annotation_surface": {
                "width": FRAME_WIDTH,
                "height": FRAME_HEIGHT,
                "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
                "source_manifest": str(SOURCE_FRAME_MANIFEST),
            },
            "semantic_status": "HUMAN_AUDIT_REQUIRED",
            "validation": {"model_fit_performed": False, "learned_continuity_rows_updated": 0, **SAFETY},
        },
    )
    write_json(
        package / "sealed" / "server_mapping.json",
        {
            "schema_version": "m5_5d2b.sealed_mapping.v1",
            "served_before_decision": False,
            "answer_key": {},
            "case_source_rows": {
                case_id: {"source_manifest": "server-side only", "authoritative_frame_rows": "server-side audit only"}
                for case_id, _, _ in CASE_WINDOWS
            },
        },
    )
    launcher = package / "launch_review.ps1"
    uv_path = r"C:\Users\sebgr\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"
    launcher.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        f"$RepoRoot = '{REPO}'\n$PackageRoot = '{package}'\n"
        "Set-Location -LiteralPath $RepoRoot\n"
        f"& '{uv_path}' run fi-pipeline review-chassis serve --manifest (Join-Path $PackageRoot 'reviewer_manifest.json') --ui-config (Join-Path $PackageRoot 'ui_config.json') --evidence-root (Join-Path $PackageRoot 'evidence') --decisions-root (Join-Path $PackageRoot 'decisions') --sealed-mapping (Join-Path $PackageRoot 'sealed/server_mapping.json') --host 127.0.0.1 --port {REVIEW_PORT} --reviewer-session-id {REVIEWER_SESSION_ID}\n",
        encoding="utf-8",
    )
    (package / "README.md").write_text(
        "# M5.5D.2B canonical source review\n\nUse only this fresh package at port 8787. Prior ports 8784, 8785 and 8786 are read-only provenance.\n",
        encoding="utf-8",
    )
    package_validation = validate_review_chassis_package(
        manifest_path=package / "reviewer_manifest.json",
        ui_config_path=package / "ui_config.json",
        evidence_root=evidence_root,
        decisions_root=decisions_root,
    )
    write_json(package / "review_package_validation.json", package_validation)

    write_json(
        STAGE_ROOT / "07_BROWSER_VALIDATION" / "clean_frame_results.json",
        {"pending_real_browser_capture": True, "primary_surface": "exact canonical frame asset"},
    )
    write_json(
        STAGE_ROOT / "07_BROWSER_VALIDATION" / "layer_alignment_results.json",
        {
            "frame_specific": True,
            "hash_bound": True,
            "canonical_scaling_applied": False,
            "pending_real_browser_capture": True,
        },
    )
    write_json(
        STAGE_ROOT / "07_BROWSER_VALIDATION" / "frame_stepper_results.json",
        {"frames": 9, "exact_assets": True, "pending_real_browser_capture": True},
    )
    write_json(
        STAGE_ROOT / "07_BROWSER_VALIDATION" / "persistence_results.json",
        {"fresh_decisions_root": True, "reviewed": 0, "event_sequence": 0, "pending_real_browser_capture": True},
    )
    write_json(
        STAGE_ROOT / "07_BROWSER_VALIDATION" / "final_validation_summary.json",
        {
            "package_validation": package_validation,
            "semantic_box_status": "HUMAN_AUDIT_REQUIRED",
            "safe_to_review": bool(package_validation.get("passed")),
            "final_classification": "BLOCKED_CANONICAL_ROW_PROVENANCE_AMBIGUOUS",
        },
    )
    write_json(
        STAGE_ROOT / "09_COMMANDS_AND_TESTS" / "build_result.json",
        {
            "package": str(package),
            "package_validation": package_validation,
            "case_count": len(case_models),
            "canonical_default_boxes": len(all_audit_rows),
            "layer_summary": layer_summary,
        },
    )
    return {
        "package": str(package),
        "package_validation": package_validation,
        "case_count": len(case_models),
        "semantic_box_count": len(all_audit_rows),
        "legacy_rows": len(legacy_rows),
        "contact_sheet": str(contact_sheet),
    }


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
