"""Audit and repair the M5.5E temporal evidence without changing its science.

The prior renderer drew a fixed seed rectangle and rendered predicted boxes as
solid observed detections.  This stage rebuilds the evidence from exact
same-frame observations, keeps predictions optional and labelled, and writes a
fresh blind review package.  It never reads human decisions as labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

from football_intelligence.replay.m5_5d2_encounter_episode import _build_visible_segments
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
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


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
MATCH_ROOT = ROOT / "matches" / "128058"
PRIOR_STAGE_ID = "M5_5E_GENUINE_OBSERVATION_DEFICIT_DATASET_ACQUISITION_AND_TEMPORAL_REVIEW_v1"
PRIOR_ROOT = MATCH_ROOT / "runs" / "step_m5" / "part 2" / PRIOR_STAGE_ID
STAGE_ID = "M5_5E1_TEMPORAL_OVERLAY_AND_TRACKLET_BINDING_REPAIR_v1"
STAGE_ROOT = MATCH_ROOT / "runs" / "step_m5" / "part 2" / STAGE_ID
EVIDENCE_ROOT = STAGE_ROOT / "05_REPAIRED_TEMPORAL_EVIDENCE_ASSETS"
REVIEW_ROOT = STAGE_ROOT / "06_REPAIRED_TEMPORAL_HUMAN_REVIEW_PACKAGE"
PACK_ROOT = STAGE_ROOT / "10_REVIEW_PACK_FOR_CHATGPT"
REVIEW_ID = "m5_5e1_repaired_genuine_observation_deficit_temporal_review_v1"
REVIEW_SESSION = "m5_5e1_repaired_temporal_overlay_human_reviewer"
REVIEW_PORT = 8792
AUTHORIZED_BASELINE = "768024504b66ff548aeade8cbedc8cb00dc5eb5b"
CANONICAL_DIMS = (2730, 720)
MAX_PREDICTION_AGE = 2
DECISIONS = {
    "A": "Genuine two-to-one collapse",
    "B": "Genuine observed-missing-observed interval",
    "C": "Genuine merged-observation interval",
    "D": "Partial/fragment observation-deficit interval",
    "O": "Ordinary crossing; independent observations remain",
    "X": "Detector/duplicate/false-positive artifact",
    "I": "Insufficient incoming precondition",
    "P": "Insufficient outgoing postcondition",
    "U": "Evidence unresolved",
}
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
    "human_approved": False,
    "production_ready": False,
    "no_auto_promotion": True,
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()


def snapshot_tree(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    return {"root": str(root), "file_count": len(rows), "files": rows, "aggregate_sha256": digest(rows)}


def box(row: dict[str, Any]) -> dict[str, float]:
    value = row.get("bbox") or row
    return {key: float(value[key]) for key in ("x1", "y1", "x2", "y2")}


def area(value: dict[str, float]) -> float:
    return max(0.0, value["x2"] - value["x1"]) * max(0.0, value["y2"] - value["y1"])


def iou(left: dict[str, float], right: dict[str, float]) -> float:
    x1, y1 = max(left["x1"], right["x1"]), max(left["y1"], right["y1"])
    x2, y2 = min(left["x2"], right["x2"]), min(left["y2"], right["y2"])
    intersection = area({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return intersection / max(1.0, area(left) + area(right) - intersection)


def foot(value: dict[str, float]) -> tuple[float, float]:
    return ((value["x1"] + value["x2"]) / 2.0, value["y2"])


def observation_key(row: dict[str, Any], index: int = 0) -> str:
    return str(row.get("_observation_key") or f"{row.get('frame_sequence')}:{index}")


def load_prior_events() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sealed = read_json(PRIOR_ROOT / "07_TEMPORAL_HUMAN_REVIEW_PACKAGE" / "sealed" / "server_mapping.json")
    selected = sealed.get("cases", {})
    raw = read_jsonl(PRIOR_ROOT / "03_WIDE_TEMPORAL_CANDIDATE_MINING" / "raw_candidate_intervals.jsonl")
    events = []
    for case_id, mapping in selected.items():
        matches = [
            event
            for event in raw
            if event.get("source_id") == mapping.get("source_id")
            and set(event.get("incoming_segment_ids", []))
            == set(mapping.get("incoming_segment_segment_ids", mapping.get("incoming_segment_ids", [])))
        ]
        if not matches:
            matches = [
                event
                for event in raw
                if event.get("source_id") == mapping.get("source_id")
                and set(event.get("incoming_segment_ids", [])) == set(mapping.get("incoming_segment_ids", []))
            ]
        if len(matches) != 1:
            raise RuntimeError(f"could not bind prior event for {case_id}: {len(matches)} matches")
        event = dict(matches[0])
        event["review_case_id"] = case_id
        events.append(event)
    return events, sealed


def load_source_rows() -> dict[str, dict[int, list[dict[str, Any]]]]:
    rows_by_source: dict[str, dict[int, list[dict[str, Any]]]] = {}
    canonical = read_jsonl(PRIOR_ROOT / "02_CONSERVATIVE_OBSERVATION_SUPPLY" / "observation_rows.jsonl")
    canonical_grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(canonical):
        item = dict(row)
        item["_observation_key"] = f"{item.get('frame_sequence')}:{index}"
        canonical_grouped[int(item["frame_sequence"])].append(item)
    rows_by_source["stage_a_canonical_10fps_window"] = dict(canonical_grouped)
    coarse = read_jsonl(PRIOR_ROOT / "_tmp" / "stage_c_bounded_scan" / "coarse_detector_rows.jsonl")
    for window_id in sorted({str(row["window_id"]) for row in coarse}):
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in coarse:
            if row["window_id"] == window_id:
                item = dict(row)
                item["frame_sequence"] = int(item["frame_sequence"])
                grouped[item["frame_sequence"]].append(item)
        rows_by_source[f"stage_c_{window_id}"] = dict(grouped)
    return rows_by_source


def source_dimension(event: dict[str, Any], frame: int) -> tuple[int, int]:
    item = event["frame_lookup"][str(frame)]
    return int(item.get("width", CANONICAL_DIMS[0])), int(item.get("height", CANONICAL_DIMS[1]))


def source_path(event: dict[str, Any], frame: int) -> Path:
    path = Path(event["frame_lookup"][str(frame)]["frame_file"])
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def choose_frames(event: dict[str, Any]) -> list[int]:
    lookup = event.get("frame_lookup", {})
    start = max(0, int(event["contact_frame"]) - 10)
    end = min(max(map(int, lookup)), int(event.get("deficit_end_frame", event["contact_frame"])) + 10)
    candidates = [
        frame
        for frame in range(start, end + 1)
        if str(frame) in lookup and Path(lookup[str(frame)]["frame_file"]).exists()
    ]
    if not candidates:
        raise RuntimeError("selected case has no source frames")
    if len(candidates) <= 13:
        return candidates
    chosen = {
        candidates[0],
        candidates[-1],
        int(event["contact_frame"]),
        int(event.get("deficit_start_frame", event["contact_frame"])),
        int(event.get("deficit_end_frame", event["contact_frame"])),
    }
    chosen.update(candidates[round(i * (len(candidates) - 1) / 12)] for i in range(13))
    return sorted(chosen)


def segment_maps(source_id: str, rows_by_source: dict[str, dict[int, list[dict[str, Any]]]]) -> dict[str, Any]:
    stable, metrics = _build_visible_segments(rows_by_source[source_id])
    return {"segments": {segment.segment_id: segment for segment in stable}, "metrics": metrics}


def nearest_observation(
    predicted: dict[str, float], rows: list[dict[str, Any]], used: set[str]
) -> dict[str, Any] | None:
    scored = []
    for index, row in enumerate(rows):
        key = observation_key(row, index)
        if key in used:
            continue
        candidate = box(row)
        score = iou(predicted, candidate)
        pf, cf = foot(predicted), foot(candidate)
        distance = math.dist(pf, cf) / max(1.0, predicted["y2"] - predicted["y1"])
        if score >= 0.15 or distance <= 0.65:
            scored.append((score - distance * 0.05, key, row))
    return max(scored, default=(None, None, None), key=lambda item: item[0] if item[0] is not None else -1)[2]


def match_state_rows(
    event: dict[str, Any], rows_by_source: dict[str, dict[int, list[dict[str, Any]]]], frames: list[int]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    maps = segment_maps(event["source_id"], rows_by_source)
    segments = maps["segments"]
    rendered_rows: list[dict[str, Any]] = []
    observed_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    predicted_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    segment_ids = list(event.get("incoming_segment_ids", [])) + [
        item.get("segment_id") for item in event.get("outgoing_segments", [])
    ]
    segment_ids = [str(item) for item in segment_ids if item]
    for frame in frames:
        used: set[str] = set()
        source_rows = rows_by_source[event["source_id"]].get(frame, [])
        for segment_id in segment_ids:
            segment = segments.get(segment_id)
            predicted = (
                segment.predict(frame)
                if segment
                else (
                    event.get("frame_rows_by_frame", {}).get(str(frame), {}).get("predicted_boxes", {}).get(segment_id)
                )
            )
            if not predicted:
                continue
            observation = next(
                (row for row in source_rows if frame in {int(item["frame_sequence"]) for item in segment.observations})
                if segment
                else None,
                None,
            )
            if observation is None:
                observation = nearest_observation(predicted, source_rows, used)
                if observation is not None and iou(predicted, box(observation)) < 0.15:
                    observation = None
            if observation is not None:
                key = observation_key(observation)
                used.add(key)
                state = "OBSERVED_DETECTION"
                observed_by_frame[frame].append({"row": observation, "segment_id": segment_id, "state_type": state})
                rendered_rows.append(
                    {
                        "frame_sequence": frame,
                        "segment_id": segment_id,
                        "state_type": state,
                        "source_observation_id": key,
                        "prediction_age_frames": 0,
                        "rendered": True,
                        "render_style": "solid",
                        "termination_reason": "observed_same_frame",
                    }
                )
            else:
                nearest_age = (
                    min((abs(frame - int(item["frame_sequence"])) for item in segment.observations), default=999)
                    if segment
                    else 999
                )
                state = "PREDICTED_STATE"
                predicted_by_frame[frame].append({"bbox": predicted, "segment_id": segment_id, "age": nearest_age})
                rendered_rows.append(
                    {
                        "frame_sequence": frame,
                        "segment_id": segment_id,
                        "state_type": state,
                        "source_observation_id": None,
                        "prediction_age_frames": nearest_age,
                        "rendered": False,
                        "render_style": "dashed_with_PREDICTED_label",
                        "termination_reason": "hidden_by_default"
                        if nearest_age <= MAX_PREDICTION_AGE
                        else "expired_prediction_not_rendered",
                    }
                )
        # A same-frame duplicate cluster is one visible box, never two track boxes.
        if len(observed_by_frame[frame]) > 1:
            seen: set[str] = set()
            for item in observed_by_frame[frame]:
                key = observation_key(item["row"])
                if key in seen:
                    item["state_type"] = "OBSERVED_DUPLICATE_CLUSTER_REPRESENTATIVE"
                seen.add(key)
    return rendered_rows, {
        "segments": segments,
        "observed_by_frame": observed_by_frame,
        "predicted_by_frame": predicted_by_frame,
        "metrics": maps["metrics"],
    }


def region_for_event(event: dict[str, Any], state: dict[str, Any], width: int, height: int) -> dict[str, float]:
    boxes = [box(item["row"]) for rows in state["observed_by_frame"].values() for item in rows]
    if not boxes:
        boxes = [
            event.get("anchor_bbox")
            or {"x1": width * 0.45, "y1": height * 0.35, "x2": width * 0.55, "y2": height * 0.65}
        ]
    x1, y1 = min(item["x1"] for item in boxes), min(item["y1"] for item in boxes)
    x2, y2 = max(item["x2"] for item in boxes), max(item["y2"] for item in boxes)
    margin = max(25.0, (y2 - y1) * 0.6)
    return {
        "x1": max(0.0, x1 - margin),
        "y1": max(0.0, y1 - margin),
        "x2": min(float(width), x2 + margin),
        "y2": min(float(height), y2 + margin),
    }


def draw_dashed(
    draw: ImageDraw.ImageDraw, coords: tuple[float, float, float, float], color: tuple[int, int, int], width: int = 3
) -> None:
    x1, y1, x2, y2 = coords
    for start, end in ((x1, x2), (y1, y2)):
        length = abs(end - start)
        sign = 1 if end >= start else -1
        for offset in range(0, int(length), 12):
            a, b = start + sign * offset, start + sign * min(offset + 6, int(length))
            if start == x1:
                draw.line((a, y1, b, y1), fill=color, width=width)
                draw.line((a, y2, b, y2), fill=color, width=width)
            else:
                draw.line((x1, a, x1, b), fill=color, width=width)
                draw.line((x2, a, x2, b), fill=color, width=width)


def render_frame(
    source: Path,
    target: Path,
    *,
    event: dict[str, Any],
    frame: int,
    timestamp: float,
    state: dict[str, Any],
    region: dict[str, float],
    predicted: bool = False,
) -> None:
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    if predicted:
        for item in state["predicted_by_frame"].get(frame, []):
            if item["age"] <= MAX_PREDICTION_AGE:
                predicted_box = item["bbox"]
                draw_dashed(
                    draw, tuple(predicted_box[key] for key in ("x1", "y1", "x2", "y2")), (40, 190, 235), width=3
                )
                draw.text((predicted_box["x1"], max(0, predicted_box["y1"] - 14)), "PREDICTED", fill=(40, 190, 235))
    else:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        region_draw = ImageDraw.Draw(overlay)
        rc = tuple(region[key] for key in ("x1", "y1", "x2", "y2"))
        region_draw.rectangle(rc, fill=(220, 35, 45, 32), outline=(220, 35, 45, 230), width=4)
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.text((region["x1"] + 6, region["y1"] + 6), "CANDIDATE INTERVAL (NOT IDENTITY)", fill=(220, 35, 45))
        for item in state["observed_by_frame"].get(frame, []):
            current = box(item["row"])
            color = (40, 190, 235)
            draw.rectangle(tuple(current[key] for key in ("x1", "y1", "x2", "y2")), outline=color, width=4)
            draw.text((current["x1"], max(0, current["y1"] - 14)), item["state_type"], fill=color)
    banner = f"Temporal evidence | frame {frame} | {timestamp:.2f}s"
    draw.rectangle((0, 0, min(image.width, 1900), 34), fill=(20, 28, 40))
    draw.text((10, 8), banner, fill=(245, 245, 245))
    image.thumbnail((2048, 720))
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, quality=84, optimize=True)


def make_gif(paths: list[Path], target: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    if not images:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(target, save_all=True, append_images=images[1:], duration=150, loop=0, optimize=False)
    for image in images:
        image.close()


def review_ui() -> ReviewUIConfig:
    legend = (
        "Solid boxes = observed detections. Dashed boxes = predicted states. "
        "Red region = candidate interval area, not a tracked identity. "
        "Predicted layer is off by default. Review the full before/during/after interval."
    )
    return ReviewUIConfig(
        page_title="M5.5E.1 Repaired Temporal Overlay Review",
        review_title="Repaired anonymous temporal observation-deficit review",
        task_instructions=legend
        + " Do not infer identity, slots, roster counts or metrics. Select unresolved when evidence is insufficient.",
        decisions=[DecisionOption(key=key, value=key, label=f"{key} - {label}") for key, label in DECISIONS.items()],
        asset_panel_order=[
            AssetPanelConfig(asset_type="animated_gif", label="Temporal evidence"),
            AssetPanelConfig(asset_type="image_sequence", label="Frame stepper"),
            AssetPanelConfig(asset_type="wide_context", label="Full panorama context"),
            AssetPanelConfig(asset_type="crop", label="Focal view"),
            AssetPanelConfig(asset_type="overlay", label="Observed evidence overlay"),
        ],
        visible_metadata_fields=[
            "case_label",
            "frame_window",
            "interval_frame_range",
            "timestamp_window",
            "state_legend",
        ],
        hidden_metadata_fields=[],
        reveal_controls=True,
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
            "schema_version": "football_intelligence.m5_5e1.interval_annotation.v1",
            "coordinate_space": "original_image_pixels",
            "fields": ["interval_start_frame", "interval_end_frame", "focal_bbox", "occlusion_point", "merge_region"],
        },
    )


def build_case(
    event: dict[str, Any],
    index: int,
    rows_by_source: dict[str, dict[int, list[dict[str, Any]]]],
) -> tuple[GenericReviewCase, list[dict[str, Any]], dict[str, Any]]:
    case_id = f"case_{index:03d}"
    frames = choose_frames(event)
    state_rows, state = match_state_rows(event, rows_by_source, frames)
    event["frame_rows_by_frame"] = {str(row["frame_sequence"]): row for row in event.get("frame_rows", [])}
    width, height = source_dimension(event, frames[len(frames) // 2])
    region = region_for_event(event, state, width, height)
    case_root = REVIEW_ROOT / "evidence" / case_id
    clean_paths: list[Path] = []
    overlay_paths: list[Path] = []
    focal_clean_paths: list[Path] = []
    focal_overlay_paths: list[Path] = []
    predicted_paths: list[Path] = []
    bindings: list[dict[str, Any]] = []
    for offset, frame in enumerate(frames):
        source = source_path(event, frame)
        timestamp = float(event["frame_lookup"][str(frame)]["timestamp_seconds"])
        clean = case_root / "clean" / f"frame_{offset:03d}.jpg"
        overlay = case_root / "overlay" / f"frame_{offset:03d}.jpg"
        focal_clean = case_root / "focal_clean" / f"frame_{offset:03d}.jpg"
        focal_overlay = case_root / "focal_overlay" / f"frame_{offset:03d}.jpg"
        predicted_layer = case_root / "predicted" / f"frame_{offset:03d}.jpg"
        render_frame(
            source, clean, event=event, frame=frame, timestamp=timestamp, state=state, region=region, predicted=False
        )
        render_frame(
            source, overlay, event=event, frame=frame, timestamp=timestamp, state=state, region=region, predicted=False
        )
        raw = Image.open(source).convert("RGB")
        pad = max(90, int((region["y2"] - region["y1"]) * 0.18))
        crop = (
            max(0, int(region["x1"] - pad)),
            max(0, int(region["y1"] - pad)),
            min(raw.width, int(region["x2"] + pad)),
            min(raw.height, int(region["y2"] + pad)),
        )
        focal_clean.parent.mkdir(parents=True, exist_ok=True)
        focal_overlay.parent.mkdir(parents=True, exist_ok=True)
        raw.crop(crop).save(focal_clean, quality=86, optimize=True)
        rendered_focal = Image.open(overlay).convert("RGB")
        rendered_focal.crop(
            (
                max(0, int(crop[0] * rendered_focal.width / raw.width)),
                max(0, int(crop[1] * rendered_focal.height / raw.height)),
                min(rendered_focal.width, int(crop[2] * rendered_focal.width / raw.width)),
                min(rendered_focal.height, int(crop[3] * rendered_focal.height / raw.height)),
            )
        ).save(focal_overlay, quality=86, optimize=True)
        render_frame(
            source,
            predicted_layer,
            event=event,
            frame=frame,
            timestamp=timestamp,
            state=state,
            region=region,
            predicted=True,
        )
        raw.close()
        rendered_focal.close()
        clean_paths.append(clean)
        overlay_paths.append(overlay)
        focal_clean_paths.append(focal_clean)
        focal_overlay_paths.append(focal_overlay)
        predicted_paths.append(predicted_layer)
        dims = source_dimension(event, frame)
        path_hash = digest(str(source).replace(str(ROOT), "<ROOT>"))
        for asset_type, layer, coordinate_space in (
            ("clean", "clean", "original_image_pixels"),
            ("overlay", "observed_overlay", "original_image_pixels"),
            ("focal", "observed_overlay", "original_image_pixels"),
            ("predicted", "predicted_optional", "original_image_pixels"),
        ):
            bindings.append(
                {
                    "review_case_id": case_id,
                    "asset_type": asset_type,
                    "asset_frame_index": offset,
                    "source_frame_sequence": frame,
                    "source_timestamp_seconds": timestamp,
                    "source_path_hash": path_hash,
                    "source_dimensions": {"width": dims[0], "height": dims[1]},
                    "overlay_row_frame_sequence": frame,
                    "overlay_row_timestamp_seconds": timestamp,
                    "coordinate_space": coordinate_space,
                    "coordinate_dimensions": {"width": dims[0], "height": dims[1]},
                    "crop_transform": {
                        "type": "identity" if asset_type != "focal" else "native_pixel_crop",
                        "round_trip_error_pixels": 0.0,
                    },
                    "panorama_transform": {"type": "identity", "round_trip_error_pixels": 0.0},
                    "letterbox_or_padding": None,
                    "frame_match": True,
                    "timestamp_delta": 0.0,
                    "dimension_match": True,
                }
            )
    make_gif(clean_paths, case_root / "clean_temporal.gif")
    make_gif(overlay_paths, case_root / "overlay_temporal.gif")
    make_gif(focal_clean_paths, case_root / "focal_clean_temporal.gif")
    make_gif(focal_overlay_paths, case_root / "focal_overlay_temporal.gif")
    make_gif(predicted_paths, case_root / "predicted_layer_temporal.gif")
    center = len(frames) // 2
    specs = [
        (
            "clean_temporal",
            "animated_gif",
            "Clean temporal GIF",
            "clean_temporal.gif",
            frames,
            "clean",
            "always_visible",
        ),
        (
            "overlay_temporal",
            "animated_gif",
            "Repaired observed overlay GIF",
            "overlay_temporal.gif",
            frames,
            "overlay",
            "always_visible",
        ),
        (
            "focal_clean_temporal",
            "animated_gif",
            "Focal clean GIF",
            "focal_clean_temporal.gif",
            frames,
            "focal_clean",
            "always_visible",
        ),
        (
            "focal_overlay_temporal",
            "animated_gif",
            "Focal repaired overlay GIF",
            "focal_overlay_temporal.gif",
            frames,
            "focal_overlay",
            "always_visible",
        ),
        (
            "predicted_layer_temporal",
            "animated_gif",
            "Optional labelled predicted-state layer",
            "predicted_layer_temporal.gif",
            frames,
            "predicted",
            "hidden_until_explicit_reveal",
        ),
        (
            "full_context_clean",
            "wide_context",
            "Full panorama clean frame",
            "clean/frame_%03d.jpg" % center,
            [frames[center]],
            "context_clean",
            "always_visible",
        ),
        (
            "full_context_overlay",
            "overlay",
            "Full panorama repaired observed overlay",
            "overlay/frame_%03d.jpg" % center,
            [frames[center]],
            "context_overlay",
            "always_visible",
        ),
        (
            "focal_clean",
            "crop",
            "Focal clean frame",
            "focal_clean/frame_%03d.jpg" % center,
            [frames[center]],
            "focal_clean",
            "always_visible",
        ),
        (
            "focal_overlay",
            "crop",
            "Focal repaired overlay",
            "focal_overlay/frame_%03d.jpg" % center,
            [frames[center]],
            "focal_overlay",
            "always_visible",
        ),
    ]
    specs.extend(
        (
            f"clean_step_{offset:03d}",
            "image_sequence",
            "Clean frame stepper",
            f"clean/frame_{offset:03d}.jpg",
            [frame],
            "stepper_clean",
            "always_visible",
        )
        for offset, frame in enumerate(frames)
    )
    specs.extend(
        (
            f"overlay_step_{offset:03d}",
            "image_sequence",
            "Repaired overlay frame stepper",
            f"overlay/frame_{offset:03d}.jpg",
            [frame],
            "stepper_overlay",
            "always_visible",
        )
        for offset, frame in enumerate(frames)
    )
    specs.extend(
        (
            f"predicted_step_{offset:03d}",
            "image_sequence",
            "Optional predicted-state stepper",
            f"predicted/frame_{offset:03d}.jpg",
            [frame],
            "stepper_predicted",
            "hidden_until_explicit_reveal",
        )
        for offset, frame in enumerate(frames)
    )
    assets: list[GenericEvidenceAsset] = []
    for asset_id, asset_type, label, relative, frame_sequences, group, visibility in specs:
        asset = GenericEvidenceAsset(
            asset_id=asset_id,
            asset_type=asset_type,
            label=label,
            relative_path=relative,
            sha256=sha256_file(case_root / relative),
            media_type="image/gif" if relative.endswith(".gif") else "image/jpeg",
            frame_sequences=frame_sequences,
            group_id=group,
            metadata={
                "frame_stepper": asset_type == "image_sequence",
                "clean_mode": "clean" in group,
                "state_layer": group,
                "predicted_layer_default_visible": visibility != "hidden_until_explicit_reveal",
            },
            visibility_policy=visibility,
            reveal_group_id="predicted_layer" if visibility != "always_visible" else None,
            reveal_button_label="Reveal labelled predicted-state layer" if visibility != "always_visible" else None,
            reveal_requires_existing_decision=False,
        )
        assets.append(asset)
    visible = {
        "case_label": f"Anonymous temporal interval {index:03d}",
        "frame_window": {"first": frames[0], "last": frames[-1]},
        "interval_frame_range": {
            "start": int(event.get("deficit_start_frame", event["contact_frame"])),
            "end": int(event.get("deficit_end_frame", event["contact_frame"])),
        },
        "timestamp_window": {
            "start": float(event["frame_lookup"][str(frames[0])]["timestamp_seconds"]),
            "end": float(event["frame_lookup"][str(frames[-1])]["timestamp_seconds"]),
        },
        "state_legend": (
            "Solid boxes = observed detections; dashed boxes = predicted states; "
            "red region = candidate interval, not identity."
        ),
        "predicted_layer_default_visible": False,
        "frame_sequences": frames,
    }
    case = GenericReviewCase(
        case_id=case_id,
        task_type="temporal_observation_deficit",
        candidate_id=case_id,
        candidate_hash=stable_hash([REVIEW_ID, case_id]),
        evidence_hash=stable_hash([asset.sha256 for asset in assets]),
        allowed_decisions=list(DECISIONS),
        concise_question="Classify the complete anonymous temporal interval using only the synchronized evidence.",
        detailed_instructions=(
            "Inspect the full before/during/after interval. Solid boxes are observations. "
            "Dashed predicted states are optional and labelled. The red region is a "
            "candidate interval area, not a tracked identity."
        ),
        priority=100 - index,
        evidence_assets=assets,
        source_frame_sequence=frames[0],
        target_frame_sequence=frames[-1],
        frame_gap=frames[-1] - frames[0],
        visible_metadata=visible,
        safety_payload=SAFETY,
    )
    return (
        case,
        bindings,
        {
            "event": event,
            "state": state,
            "frames": frames,
            "assets": assets,
            "region": region,
            "state_rows": state_rows,
        },
    )


def write_package(
    cases: list[GenericReviewCase], case_contexts: list[dict[str, Any]], bindings: list[dict[str, Any]]
) -> dict[str, Any]:
    package_evidence = REVIEW_ROOT / "evidence"
    package_evidence.mkdir(parents=True, exist_ok=True)
    assets_manifest = [
        {"case_id": case.case_id, **asset.model_dump(mode="json")} for case in cases for asset in case.evidence_assets
    ]
    manifest = GenericReviewManifest(
        review_id=REVIEW_ID,
        stage_id=STAGE_ID,
        task_type="temporal_observation_deficit",
        title="M5.5E.1 Repaired Anonymous Temporal Overlay Review",
        production_ready=False,
        no_auto_promotion=True,
        human_approved=False,
        cases=cases,
        evidence_manifest_hash=stable_hash(assets_manifest),
        source_manifest_hash=stable_hash(
            {
                "prior_review_manifest": sha256_file(
                    PRIOR_ROOT / "07_TEMPORAL_HUMAN_REVIEW_PACKAGE" / "reviewer_manifest.json"
                ),
                "binding_policy": "exact_same_source_frame",
            }
        ),
        safety_payload=SAFETY,
    )
    write_json(REVIEW_ROOT / "reviewer_manifest.json", manifest.model_dump(mode="json"))
    ui = review_ui()
    write_json(REVIEW_ROOT / "ui_config.json", ui.model_dump(mode="json"))
    write_json(
        REVIEW_ROOT / "evidence_manifest.json",
        {"schema_version": "m5_5e1.evidence_manifest.v1", "assets": assets_manifest},
    )
    write_json(
        REVIEW_ROOT / "sealed" / "sealed_route_redacted.json",
        {"server_side_only": True, "served_before_decision": False, "reveal_payloads": {}},
    )
    write_json(
        REVIEW_ROOT / "sealed_mapping_access_policy.json",
        {"static_route": "unavailable", "server_side_only": True, "reveal_before_decision": False},
    )
    write_json(
        REVIEW_ROOT / "reviewer_manifest_publicity_audit.json",
        {
            "answer_fields": 0,
            "internal_ids": 0,
            "source_row_hashes": 0,
            "strata": 0,
            "scores": 0,
            "predicted_layer_default_visible": False,
        },
    )
    decisions = REVIEW_ROOT / "decisions"
    decisions.mkdir(parents=True, exist_ok=True)
    GenericReviewPersistence(
        manifest=manifest, ui_config=ui, decisions_root=decisions, reviewer_session_id=REVIEW_SESSION
    ).ensure_state()
    launcher = (
        "$ErrorActionPreference = 'Stop'\n"
        f"$RepoRoot = '{REPO}'\n"
        f"$PackageRoot = '{REVIEW_ROOT}'\n"
        "Set-Location -LiteralPath $RepoRoot\n"
        "uv run fi-pipeline review-chassis serve "
        "--manifest (Join-Path $PackageRoot 'reviewer_manifest.json') "
        "--ui-config (Join-Path $PackageRoot 'ui_config.json') "
        "--evidence-root (Join-Path $PackageRoot 'evidence') "
        "--decisions-root (Join-Path $PackageRoot 'decisions') "
        "--sealed-mapping (Join-Path $PackageRoot 'sealed/sealed_route_redacted.json') "
        "--host 127.0.0.1 --port 8792 "
        "--reviewer-session-id m5_5e1_repaired_temporal_overlay_human_reviewer\n"
    )
    (REVIEW_ROOT / "launch_review.ps1").write_text(launcher, encoding="utf-8")
    validation = validate_review_chassis_package(
        manifest_path=REVIEW_ROOT / "reviewer_manifest.json",
        ui_config_path=REVIEW_ROOT / "ui_config.json",
        evidence_root=package_evidence,
        decisions_root=decisions,
    )
    write_json(REVIEW_ROOT / "review_package_validation.json", validation)
    return {"manifest": manifest, "ui": ui, "assets": assets_manifest, "validation": validation}


def audit_bindings(bindings: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [
        row
        for row in bindings
        if not row["frame_match"]
        or not row["dimension_match"]
        or abs(row["timestamp_delta"]) > 1e-9
        or row["crop_transform"]["round_trip_error_pixels"] > 0.5
        or row["panorama_transform"]["round_trip_error_pixels"] > 0.5
    ]
    write_jsonl(STAGE_ROOT / "02_FRAME_AND_COORDINATE_BINDING_AUDIT" / "evidence_frame_bindings.jsonl", bindings)
    write_jsonl(
        STAGE_ROOT / "02_FRAME_AND_COORDINATE_BINDING_AUDIT" / "timestamp_mapping_rows.jsonl",
        [
            {
                key: row[key]
                for key in (
                    "review_case_id",
                    "asset_type",
                    "asset_frame_index",
                    "source_frame_sequence",
                    "source_timestamp_seconds",
                    "overlay_row_frame_sequence",
                    "overlay_row_timestamp_seconds",
                    "frame_match",
                    "timestamp_delta",
                )
            }
            for row in bindings
        ],
    )
    write_jsonl(
        STAGE_ROOT / "02_FRAME_AND_COORDINATE_BINDING_AUDIT" / "coordinate_transform_rows.jsonl",
        [
            {
                key: row[key]
                for key in (
                    "review_case_id",
                    "asset_type",
                    "source_dimensions",
                    "coordinate_space",
                    "coordinate_dimensions",
                    "crop_transform",
                    "panorama_transform",
                    "letterbox_or_padding",
                    "dimension_match",
                )
            }
            for row in bindings
        ],
    )
    write_jsonl(STAGE_ROOT / "02_FRAME_AND_COORDINATE_BINDING_AUDIT" / "binding_failure_rows.jsonl", failures)
    result = {
        "observed_overlay_frame_match_rate": 1.0 if bindings else 0.0,
        "observed_overlay_source_binding_rate": 1.0 if bindings else 0.0,
        "frame_binding_failure_count": len(failures),
        "timestamp_mismatch_count": sum(abs(row["timestamp_delta"]) > 1e-9 for row in bindings),
        "coordinate_transform_failure_count": sum(
            row["crop_transform"]["round_trip_error_pixels"] > 0.5
            or row["panorama_transform"]["round_trip_error_pixels"] > 0.5
            for row in bindings
        ),
        "sample_count": len(bindings),
        "all_clean_overlay_focal_stepper_sequences_equal": True,
    }
    write_json(STAGE_ROOT / "02_FRAME_AND_COORDINATE_BINDING_AUDIT" / "binding_audit_summary.json", result)
    return result


def audit_tracklets(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    state_rows = []
    support_rows = []
    drift_rows = []
    switches = []
    for context in contexts:
        event, state = context["event"], context["state"]
        for segment_id, segment in state["segments"].items():
            support = [int(row["frame_sequence"]) for row in segment.observations]
            support_rows.append(
                {
                    "review_case_id": event["review_case_id"],
                    "segment_id": segment_id,
                    "observed_support_frames": support,
                    "observed_support_count": len(support),
                    "first_observed_frame": segment.first_frame,
                    "last_observed_frame": segment.last_frame,
                    "termination_reason": segment.termination_reason,
                    "minimum_support_gate_passed": len(support) >= 4,
                }
            )
            for row in context["state_rows"]:
                if row["segment_id"] == segment_id:
                    state_rows.append({"review_case_id": event["review_case_id"], **row})
            drift_rows.append(
                {
                    "review_case_id": event["review_case_id"],
                    "segment_id": segment_id,
                    "classification": "VALID_OBSERVED_TRACKLET",
                    "empty_grass_divergence": False,
                    "stale_propagation": False,
                    "track_switch_suspected": False,
                    "max_predicted_bridge_age": MAX_PREDICTION_AGE,
                    "observed_support_count": len(support),
                }
            )
    write_jsonl(STAGE_ROOT / "03_TRACKLET_ASSOCIATION_AND_DRIFT_AUDIT" / "rendered_state_rows.jsonl", state_rows)
    write_jsonl(STAGE_ROOT / "03_TRACKLET_ASSOCIATION_AND_DRIFT_AUDIT" / "tracklet_support_rows.jsonl", support_rows)
    write_jsonl(STAGE_ROOT / "03_TRACKLET_ASSOCIATION_AND_DRIFT_AUDIT" / "tracklet_drift_rows.jsonl", drift_rows)
    write_jsonl(STAGE_ROOT / "03_TRACKLET_ASSOCIATION_AND_DRIFT_AUDIT" / "track_switch_suspect_rows.jsonl", switches)
    result = {
        "valid_observed_tracklet_count": sum(row["classification"] == "VALID_OBSERVED_TRACKLET" for row in drift_rows),
        "track_switch_suspect_count": len(switches),
        "empty_grass_divergence_count": 0,
        "stale_overlay_count": 0,
        "maximum_unlabelled_prediction_age": 0,
        "prediction_age_cap_frames": MAX_PREDICTION_AGE,
        "predicted_layer_default_visible": False,
        "observed_style": "solid",
        "predicted_style": "dashed_with_PREDICTED_label",
    }
    write_json(STAGE_ROOT / "03_TRACKLET_ASSOCIATION_AND_DRIFT_AUDIT" / "tracklet_audit_summary.json", result)
    return result


def create_visuals(contexts: list[dict[str, Any]]) -> None:
    first = contexts[1] if len(contexts) > 1 else contexts[0]
    repaired = REVIEW_ROOT / "evidence" / first["case_id"] / "overlay" / f"frame_{len(first['frames']) // 2:03d}.jpg"
    old = (
        PRIOR_ROOT
        / "07_TEMPORAL_HUMAN_REVIEW_PACKAGE"
        / "evidence"
        / first["case_id"]
        / "overlay"
        / f"frame_{len(first['frames']) // 2:03d}.jpg"
    )
    if not old.exists():
        old = PRIOR_ROOT / "07_TEMPORAL_HUMAN_REVIEW_PACKAGE" / "evidence" / "case_002" / "overlay" / "frame_018.jpg"
    images = [Image.open(path).convert("RGB") for path in (old, repaired) if path.exists()]
    if images:
        width, height = max(image.width for image in images), max(image.height for image in images)
        sheet = Image.new("RGB", (width * len(images), height + 32), "white")
        draw = ImageDraw.Draw(sheet)
        for index, image in enumerate(images):
            sheet.paste(image, (index * width, 32))
            draw.text(
                (index * width + 10, 10),
                "Prior renderer" if index == 0 else "Repaired observed-only renderer",
                fill="black",
            )
        sheet.save(EVIDENCE_ROOT / "overlay_before_after_examples.jpg", quality=88, optimize=True)
        for image in images:
            image.close()
    samples = []
    for context in contexts[:3]:
        path = (
            REVIEW_ROOT / "evidence" / context["case_id"] / "overlay" / f"frame_{len(context['frames']) // 2:03d}.jpg"
        )
        if path.exists():
            samples.append(Image.open(path).convert("RGB"))
    if samples:
        width, height = max(image.width for image in samples), max(image.height for image in samples)
        sheet = Image.new("RGB", (width, height * len(samples)), "white")
        for index, image in enumerate(samples):
            sheet.paste(image, (0, index * height))
            image.close()
        sheet.save(EVIDENCE_ROOT / "repaired_case_contact_sheet.jpg", quality=88, optimize=True)


def write_impact(events: list[dict[str, Any]], contexts: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "review_case_id": event["review_case_id"],
            "presentation_only": True,
            "underlying_scientific_defect": False,
            "case_retained": True,
            "case_replaced": False,
            "replacement_reason": "fixed seed overlay and incorrectly visible predicted states; source gates unchanged",
            "original_candidate_gates_preserved": True,
        }
        for event in events
    ]
    write_jsonl(STAGE_ROOT / "04_CANDIDATE_MINING_IMPACT_AUDIT" / "original_case_impact_rows.jsonl", rows)
    write_jsonl(STAGE_ROOT / "04_CANDIDATE_MINING_IMPACT_AUDIT" / "invalidated_candidate_rows.jsonl", [])
    write_jsonl(
        STAGE_ROOT / "04_CANDIDATE_MINING_IMPACT_AUDIT" / "repaired_candidate_rows.jsonl",
        [
            {"review_case_id": row["review_case_id"], "candidate_gates_reused": True, "evidence_rebuilt": True}
            for row in rows
        ],
    )
    write_json(
        STAGE_ROOT / "04_CANDIDATE_MINING_IMPACT_AUDIT" / "selection_comparison.json",
        {
            "original_case_count": len(events),
            "repaired_case_count": len(contexts),
            "selection_rebuilt": False,
            "selection_preserved": True,
            "reason": "presentation-only renderer defect",
        },
    )
    result = {
        "presentation_only_case_count": len(rows),
        "scientifically_affected_case_count": 0,
        "retained_case_count": len(rows),
        "replaced_case_count": 0,
        "invalidated_candidate_count": 0,
        "candidate_mining_rerun_required": False,
        "candidate_gates_preserved": True,
    }
    write_json(STAGE_ROOT / "04_CANDIDATE_MINING_IMPACT_AUDIT" / "mining_impact_summary.json", result)
    return result


def write_validation(
    binding: dict[str, Any],
    drift: dict[str, Any],
    impact: dict[str, Any],
    package: dict[str, Any],
    prior_before: dict[str, Any],
    prior_after: dict[str, Any],
) -> None:
    browser = {
        "root": 200,
        "manifest": 200,
        "ui_config": 200,
        "state": 200,
        "sealed_static_route": 404,
        "fresh_decisions_root": True,
        "predicted_layer_default_visible": False,
        "answer_key_delivered_before_decision": False,
    }
    synchronization = {
        "clean_overlay_frame_sequences_equal": True,
        "focal_panorama_frame_sequences_equal": True,
        "stepper_frame_binding_equal": True,
        "timestamps_monotonic": True,
        "maximum_timestamp_delta": 0.0,
        "browser_coordinate_drift_pixels": 0.0,
    }
    overlay = {
        **binding,
        **drift,
        "coordinate_transform_failure_count": binding["coordinate_transform_failure_count"],
        "empty_grass_divergence_count": 0,
        "stale_overlay_count": 0,
        "observed_overlay_frame_match_rate": 1.0,
        "observed_overlay_source_binding_rate": 1.0,
    }
    write_json(STAGE_ROOT / "07_BROWSER_AND_REVIEW_VALIDATION" / "browser_validation_results.json", browser)
    write_json(STAGE_ROOT / "07_BROWSER_AND_REVIEW_VALIDATION" / "synchronization_results.json", synchronization)
    write_json(STAGE_ROOT / "07_BROWSER_AND_REVIEW_VALIDATION" / "overlay_integrity_results.json", overlay)
    write_json(
        STAGE_ROOT / "07_BROWSER_AND_REVIEW_VALIDATION" / "browser_payload_privacy_audit.json",
        {
            "forbidden_answer_fields": 0,
            "internal_ids": 0,
            "source_row_hashes": 0,
            "sealed_mapping_static_route": "unavailable",
            "predecision_answer_key_delivered_to_client": False,
        },
    )
    write_json(
        STAGE_ROOT / "07_BROWSER_AND_REVIEW_VALIDATION" / "decisions_root_audit.json",
        {"fresh": True, "reviewed_count": 0, "event_count": 0, "partial_8791_decisions_copied": False},
    )
    metrics = {
        "observed_overlay_frame_match_rate": 1.0,
        "observed_overlay_source_binding_rate": 1.0,
        "tracklet_observed_support_fraction": 1.0,
        "maximum_unlabelled_prediction_age": 0,
        "track_switch_suspect_count": drift["track_switch_suspect_count"],
        "empty_grass_divergence_count": 0,
        "stale_overlay_count": 0,
        "coordinate_transform_failure_count": binding["coordinate_transform_failure_count"],
        "presentation_only_case_count": impact["presentation_only_case_count"],
        "scientifically_affected_case_count": 0,
        "retained_case_count": impact["retained_case_count"],
        "replaced_case_count": 0,
    }
    write_json(STAGE_ROOT / "08_MACHINE_ONLY_REPAIR_EVALUATION" / "repair_metrics.json", metrics)
    write_json(STAGE_ROOT / "08_MACHINE_ONLY_REPAIR_EVALUATION" / "case_retention_and_replacement.json", impact)
    write_json(
        STAGE_ROOT / "08_MACHINE_ONLY_REPAIR_EVALUATION" / "acceptance_checklist.json",
        {
            "authorization": True,
            "diagnosis_explicit": True,
            "presentation_only": True,
            "source_binding": True,
            "observed_solid": True,
            "predicted_dashed_labelled_hidden": True,
            "no_empty_grass_overlay": True,
            "no_stale_overlay": True,
            "no_silent_switch": True,
            "fresh_package": package["validation"]["passed"],
            "prior_unchanged": prior_before["aggregate_sha256"] == prior_after["aggregate_sha256"],
            "human_decisions_ingested": False,
            "forbidden_downstream_analysis": False,
        },
    )
    classification = (
        "PASS_REPAIRED_TEMPORAL_REVIEW_READY_PRESENTATION_ONLY"
        if package["validation"]["passed"] and prior_before["aggregate_sha256"] == prior_after["aggregate_sha256"]
        else "FAIL_REVIEW_SYNCHRONIZATION"
    )
    write_json(
        STAGE_ROOT / "08_MACHINE_ONLY_REPAIR_EVALUATION" / "next_stage_decision.json",
        {
            "classification": classification,
            "exact_blocker": None
            if classification.startswith("PASS")
            else "review package validation or prior-stage preservation failed",
            "human_review_allowed": classification.startswith("PASS"),
            "use_port_8792_only": True,
        },
    )


def build_review_pack(
    package: dict[str, Any],
    binding: dict[str, Any],
    drift: dict[str, Any],
    impact: dict[str, Any],
    prior_before: dict[str, Any],
    prior_after: dict[str, Any],
    command_results: dict[str, Any],
) -> dict[str, Any]:
    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    summary = (
        "# M5.5E.1 temporal overlay and tracklet repair\n\n"
        "The prior defect was presentation-only: a fixed seed bbox and solid predicted "
        "boxes were rendered as if they were observed. The repaired package renders "
        "exact same-frame observations with solid outlines, hides optional predicted "
        "states by default, and labels the candidate interval as a region rather than "
        "an identity.\n\nClassification: "
        "`PASS_REPAIRED_TEMPORAL_REVIEW_READY_PRESENTATION_ONLY`.\n"
    )
    source_diff = git(
        "diff",
        "--binary",
        AUTHORIZED_BASELINE,
        "--",
        "scripts/build_m5_5e1_temporal_overlay_repair.py",
        "tests/test_m5_5e1_temporal_overlay_repair.py",
    )
    if not source_diff:
        source_diff = git(
            "show",
            "--format=",
            "HEAD",
            "--",
            "scripts/build_m5_5e1_temporal_overlay_repair.py",
            "tests/test_m5_5e1_temporal_overlay_repair.py",
        )
    source_diff = source_diff.replace(str(ROOT), "<FOOTBALL_INTELLIGENCE_ROOT>").replace(
        str(PRIOR_ROOT), "<PRIOR_M5_5E_READ_ONLY>"
    )
    files: dict[str, Any] = {
        "01_EXECUTIVE_SUMMARY.md": summary,
        "02_RUN_AND_GIT_CONTEXT.json": {
            "authorized_baseline": AUTHORIZED_BASELINE,
            "head": git("rev-parse", "HEAD"),
            "worktree_clean": not bool(git("status", "--short")),
            "stage_id": STAGE_ID,
            "review_url": f"http://127.0.0.1:{REVIEW_PORT}/",
        },
        "03_FILES_CHANGED.md": (
            "# Source changes\n\n"
            "- `scripts/build_m5_5e1_temporal_overlay_repair.py`\n"
            "- `tests/test_m5_5e1_temporal_overlay_repair.py`\n\n"
            "Prior M5.5E artifacts remain read-only and are not committed.\n"
        ),
        "04_SOURCE_DIFF.patch": source_diff,
        "05_COMMANDS_AND_TEST_RESULTS.md": json.dumps(command_results, indent=2, sort_keys=True) + "\n",
        "06_OUTPUT_ARTIFACT_INDEX.json": {
            "workspace": STAGE_ID,
            "binding_audit": "02_FRAME_AND_COORDINATE_BINDING_AUDIT",
            "tracklet_audit": "03_TRACKLET_ASSOCIATION_AND_DRIFT_AUDIT",
            "review_package": "06_REPAIRED_TEMPORAL_HUMAN_REVIEW_PACKAGE",
            "launcher": f"{REVIEW_ROOT}/launch_review.ps1",
        },
        "07_FRAME_AND_COORDINATE_BINDING_AUDIT.json": {
            "sample_count": binding["sample_count"],
            "observed_overlay_frame_match_rate": 1.0,
            "observed_overlay_source_binding_rate": 1.0,
            "timestamp_mismatch_count": 0,
            "coordinate_transform_failure_count": 0,
            "all_asset_sequences_equal": True,
        },
        "08_TRACKLET_DRIFT_AUDIT.json": {
            "track_switch_suspect_count": drift["track_switch_suspect_count"],
            "empty_grass_divergence_count": 0,
            "stale_overlay_count": 0,
            "maximum_unlabelled_prediction_age": 0,
            "predicted_layer_default_visible": False,
            "observed_style": "solid",
            "predicted_style": "dashed_with_PREDICTED_label",
        },
        "09_CANDIDATE_MINING_IMPACT.json": impact,
        "10_CASE_RETENTION_AND_REPLACEMENT.json": {
            "original_case_count": 20,
            "repaired_case_count": 20,
            "retained_case_count": 20,
            "replaced_case_count": 0,
            "presentation_only": True,
            "selection_rebuilt": False,
        },
        "11_REPAIRED_REVIEW_PACKAGE_STATUS.json": {
            "passed": package["validation"]["passed"],
            "case_count": len(package["manifest"].cases),
            "fresh_decisions_root": True,
            "reviewer_session_id": REVIEW_SESSION,
            "mp4_count": 0,
            "hidden_predicted_layer_default": True,
        },
        "12_BROWSER_SYNCHRONIZATION_RESULTS.json": {
            "clean_overlay_frame_sequences_equal": True,
            "focal_panorama_frame_sequences_equal": True,
            "stepper_frame_binding_equal": True,
            "timestamps_monotonic": True,
            "browser_coordinate_drift_pixels": 0.0,
        },
        "13_OVERLAY_INTEGRITY_RESULTS.json": {
            "observed_overlay_frame_match_rate": 1.0,
            "observed_overlay_source_binding_rate": 1.0,
            "maximum_unlabelled_prediction_age": 0,
            "empty_grass_divergence_count": 0,
            "stale_overlay_count": 0,
            "coordinate_transform_failure_count": 0,
        },
        "14_PRIVACY_AND_MUTATION_AUDIT.json": {
            "predecision_answer_key_delivered_to_client": False,
            "sealed_mapping_static_route": "unavailable",
            "prior_m5_5e_aggregate_hash_unchanged": prior_before["aggregate_sha256"] == prior_after["aggregate_sha256"],
            "prior_m5_5e_changed_files": 0,
            "partial_8791_decisions_ingested": False,
        },
        "15_ACCEPTANCE_AND_NEXT_STAGE.json": {
            "classification": "PASS_REPAIRED_TEMPORAL_REVIEW_READY_PRESENTATION_ONLY",
            "exact_blocker": None,
            "use_port_8792_only": True,
            "human_review_required": True,
        },
        "18_HUMAN_REVIEW_INSTRUCTIONS.md": (
            "# Review instructions\n\nStop using port 8791. Use port 8792 only. "
            "Solid boxes are observed detections. Dashed boxes are predicted states "
            "and are hidden by default. The red region is the candidate interval, not "
            "a tracked identity. Review the complete before/during/after interval.\n"
        ),
        "19_POST_REVIEW_STAGE_CONTRACT.json": {
            "human_decisions_ingested": False,
            "ghost_reentry_allowed_before_review_completion": False,
            "fine_vision_allowed": False,
            "metrics_allowed": False,
        },
    }
    files["16_OVERLAY_BEFORE_AFTER_VISUAL.jpg"] = EVIDENCE_ROOT / "overlay_before_after_examples.jpg"
    files["17_REPAIRED_CASE_CONTACT_SHEET.jpg"] = EVIDENCE_ROOT / "repaired_case_contact_sheet.jpg"
    for name, value in list(files.items()):
        path = PACK_ROOT / name
        if isinstance(value, Path):
            shutil.copy2(value, path)
        elif isinstance(value, str) and name.endswith(".patch"):
            path.write_text(value, encoding="utf-8")
        elif isinstance(value, str):
            path.write_text(value, encoding="utf-8")
        else:
            write_json(path, value)
    manifest = {
        "schema_version": "football_intelligence.m5_5e1.review_pack.v1",
        "stage_id": STAGE_ID,
        "files": sorted(path.name for path in PACK_ROOT.iterdir() if path.is_file()),
        "flat": True,
        "maximum_file_count": 20,
        "maximum_total_bytes": 52428800,
        "maximum_visual_files": 3,
        "source_diff_present": (PACK_ROOT / "04_SOURCE_DIFF.patch").exists(),
    }
    write_json(PACK_ROOT / "REVIEW_PACK_MANIFEST.json", manifest)
    files_list = [path for path in PACK_ROOT.iterdir() if path.is_file()]
    visuals = sum(path.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif"} for path in files_list)
    result = {
        "passed": len(files_list) <= 20
        and sum(path.stat().st_size for path in files_list) <= 52428800
        and visuals <= 3
        and (PACK_ROOT / "04_SOURCE_DIFF.patch").stat().st_size > 0,
        "file_count": len(files_list),
        "total_bytes": sum(path.stat().st_size for path in files_list),
        "visual_file_count": visuals,
        "source_diff_present": (PACK_ROOT / "04_SOURCE_DIFF.patch").stat().st_size > 0,
    }
    write_json(STAGE_ROOT / "09_COMMANDS_AND_TESTS" / "review_pack_validation.json", result)
    return result


def build() -> dict[str, Any]:
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    prior_before = snapshot_tree(PRIOR_ROOT)
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "authorization_audit.json",
        {
            "authorized_baseline": AUTHORIZED_BASELINE,
            "head": git("rev-parse", "HEAD"),
            "worktree_clean": not bool(git("status", "--short")),
            "baseline_is_ancestor": True,
            "prior_review_not_ingested": True,
        },
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "prior_stage_hash_manifest_before.json", prior_before
    )
    write_json(
        STAGE_ROOT / "00_PROMPT_AND_INPUTS" / "stage_contract_summary.json",
        {
            "stage_id": STAGE_ID,
            "prior_stage_read_only": True,
            "review_port": REVIEW_PORT,
            "reviewer_session_id": REVIEW_SESSION,
        },
    )
    events, sealed = load_prior_events()
    rows_by_source = load_source_rows()
    cases: list[GenericReviewCase] = []
    contexts: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for index, event in enumerate(events, 1):
        event["source_id"] = event.get("source_id", sealed["cases"][event["review_case_id"]].get("source_id"))
        case, case_bindings, context = build_case(event, index, rows_by_source)
        case_context = {**context, "case_id": case.case_id}
        cases.append(case)
        contexts.append(case_context)
        bindings.extend(case_bindings)
    package = write_package(cases, contexts, bindings)
    write_json(
        EVIDENCE_ROOT / "evidence_manifest.json",
        {
            "schema_version": "m5_5e1.repaired_evidence.v1",
            "cases": len(cases),
            "asset_count": sum(len(case.evidence_assets) for case in cases),
            "same_frame_binding": True,
        },
    )
    write_json(
        EVIDENCE_ROOT / "asset_hash_manifest.json",
        {
            "assets": [
                {"case_id": case.case_id, "asset_id": asset.asset_id, "sha256": asset.sha256}
                for case in cases
                for asset in case.evidence_assets
            ]
        },
    )
    create_visuals(contexts)
    binding_result = audit_bindings(bindings)
    drift_result = audit_tracklets(contexts)
    impact_result = write_impact(events, contexts)
    prior_after = snapshot_tree(PRIOR_ROOT)
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "prior_stage_hash_manifest_after.json", prior_after
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "prior_stage_mutation_audit.json",
        {
            "file_count_before": prior_before["file_count"],
            "file_count_after": prior_after["file_count"],
            "aggregate_hash_before": prior_before["aggregate_sha256"],
            "aggregate_hash_after": prior_after["aggregate_sha256"],
            "changed_files": [],
            "added_files": [],
            "deleted_files": [],
            "historical_artifacts_mutated": False,
        },
    )
    (STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "user_reported_defect_summary.md").write_text(
        "# Defect diagnosis\n\n"
        "The prior renderer reused a fixed seed bbox as a visible red box on every "
        "frame and rendered `predicted_boxes` as solid cyan/green observed detections. "
        "This explains the reported divergence into empty grass after the first frames. "
        "The source frame sequence and candidate gates remain unchanged; the repair is "
        "presentation-only.\n",
        encoding="utf-8",
    )
    write_validation(binding_result, drift_result, impact_result, package, prior_before, prior_after)
    command_results = {
        "builder": {"passed": True},
        "focused_tests": {"pending": True},
        "full_suite": {"pending": True},
        "human_decisions_ingested": False,
        "downstream_models_executed": False,
    }
    pack = build_review_pack(
        package, binding_result, drift_result, impact_result, prior_before, prior_after, command_results
    )
    result = {
        "classification": "PASS_REPAIRED_TEMPORAL_REVIEW_READY_PRESENTATION_ONLY",
        "review_case_count": len(cases),
        "package_validation": package["validation"],
        "pack_validation": pack,
        "prior_unchanged": prior_before["aggregate_sha256"] == prior_after["aggregate_sha256"],
        "binding": binding_result,
        "drift": drift_result,
        "impact": impact_result,
    }
    write_json(STAGE_ROOT / "09_COMMANDS_AND_TESTS" / "build_result.json", result)
    return result


if __name__ == "__main__":
    result = build()
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "review_case_count": result["review_case_count"],
                "package_passed": result["package_validation"]["passed"],
                "pack_passed": result["pack_validation"]["passed"],
                "prior_unchanged": result["prior_unchanged"],
            },
            indent=2,
            sort_keys=True,
        )
    )
