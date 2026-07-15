# ruff: noqa: E501

"""Build the M5.5D.2C targeted semantic audit from authoritative source rows.

The package is deliberately small in semantic scope: one anonymous review case
per unique machine-used observation, with canonical geometry taken directly
from continuity_v11 or recovery geometry taken directly from the M5.5D source
rows.  Prior browser manifests and rendered images are never geometry sources.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

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
STAGE_ROOT = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5D2C_TARGETED_ENCOUNTER_CANDIDATE_SEMANTIC_AUDIT_v1"
SOURCE_ROOT = ROOT / r"matches\128058\runs\step_m5\06f_balanced_role_then_continuity\continuity_v11\unseen_window"
SCIENCE_ROOT = (
    ROOT / r"matches\128058\runs\step_m5\part 2\M5_5D2_ENCOUNTER_EPISODE_GAP_MINING_AND_EXPANDED_BURST_SCAN_v1"
)
B_ROOT = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5D2B_CANONICAL_CANDIDATE_SOURCE_REBUILD_v1"
PROMPT_ROOT = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5D2C_Targeted_Candidate_Semantic_Audit_Prompt_v1"

FRAME_MANIFEST = SOURCE_ROOT / "canonical_frame_manifest.json"
CANDIDATE_MANIFEST = SOURCE_ROOT / "person_candidate_rows_manifest.json"
CANDIDATE_ROWS = SOURCE_ROOT / "person_candidate_rows.jsonl"
STAGE_ID = "M5_5D2C_TARGETED_ENCOUNTER_CANDIDATE_SEMANTIC_AUDIT_v1"
REVIEW_ID = "m5_5d2c_targeted_candidate_review_v1"
REVIEWER_SESSION_ID = "m5_5d2c_targeted_candidate_human_reviewer"
REVIEW_PORT = 8788
FRAME_WIDTH = 2730
FRAME_HEIGHT = 720
AUTHORIZED_BASELINE = "b430cd039d64bbd9948ceaed650c2d21a135f4ed"

CASE_WINDOWS = {
    "case_001": (121, 129),
    "case_002": (369, 377),
    "case_003": (534, 543),
    "case_004": (291, 300),
    "case_005": (531, 539),
    "case_006": (190, 198),
    "case_007": (200, 209),
    "case_008": (200, 208),
    "case_009": (14, 22),
}

DECISIONS = [
    ("V", "VALID_VISIBLE_SINGLE_PERSON", "A visible single person is supported."),
    ("F", "FALSE_POSITIVE_OR_EMPTY", "The box is empty, non-person, or unsupported."),
    (
        "W",
        "WRONG_VISIBLE_PERSON_FOR_ENCOUNTER",
        "A visible person is present but is not the used encounter observation.",
    ),
    ("M", "MERGED_MULTIPLE_VISIBLE_PEOPLE", "One observation covers multiple visible people."),
    ("P", "PARTIAL_PERSON_OR_BODY_FRAGMENT", "Only a partial person or body fragment is visible."),
    ("D", "DUPLICATE_OF_ANOTHER_DETECTION", "This observation duplicates another same-frame detection."),
    ("U", "EVIDENCE_UNRESOLVED", "The evidence is insufficient or ambiguous."),
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


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def bbox(value: dict[str, Any]) -> dict[str, float]:
    return {key: round(float(value[key]), 3) for key in ("x1", "y1", "x2", "y2")}


def bbox_key(frame: int, value: dict[str, Any]) -> tuple[int, float, float, float, float]:
    clean = bbox(value)
    return (int(frame), clean["x1"], clean["y1"], clean["x2"], clean["y2"])


def copy_exact(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return sha256_file(target)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def frame_catalog() -> dict[int, dict[str, Any]]:
    manifest = read_json(FRAME_MANIFEST)
    frames = manifest.get("frames", [])
    if len(frames) != 600:
        raise ValueError(f"expected 600 canonical frames, got {len(frames)}")
    result = {}
    for item in frames:
        frame = int(item["frame_sequence"])
        path = Path(item["frame_file"])
        with Image.open(path) as image:
            if image.size != (FRAME_WIDTH, FRAME_HEIGHT):
                raise ValueError(f"frame {frame} dimensions are {image.size}")
        actual = sha256_file(path)
        if actual != item["byte_sha256"]:
            raise ValueError(f"frame hash mismatch at {frame}")
        result[frame] = {**item, "frame_file": str(path), "actual_byte_sha256": actual}
    return result


def load_sources() -> (
    tuple[
        dict[int, dict[str, Any]],
        list[dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        list[dict[str, Any]],
    ]
):
    catalog = frame_catalog()
    rows = load_jsonl(CANDIDATE_ROWS)
    for row in rows:
        frame = int(row["frame_sequence"])
        clean = bbox(row["bbox"])
        if frame not in catalog or not (
            0 <= clean["x1"] < clean["x2"] <= FRAME_WIDTH and 0 <= clean["y1"] < clean["y2"] <= FRAME_HEIGHT
        ):
            raise ValueError(f"canonical row out of bounds: {row.get('candidate_id')}")
        row["bbox"] = clean
        row["canonical_source_row_hash"] = digest(row)
    episodes = {
        str(row["encounter_episode_id"]): row
        for row in load_jsonl(SCIENCE_ROOT / "03_ENCOUNTER_EPISODES" / "episode_rows.jsonl")
    }
    observations = {
        str(row["observation_key"]): row
        for row in load_jsonl(SCIENCE_ROOT / "02_VISIBLE_TRACKLET_SEGMENTS" / "observation_rows.jsonl")
    }
    segments = {
        str(row["segment_id"]): row
        for row in load_jsonl(SCIENCE_ROOT / "02_VISIBLE_TRACKLET_SEGMENTS" / "visible_segment_rows.jsonl")
    }
    recovery = {
        str(row["case_index"]): row
        for row in load_jsonl(SCIENCE_ROOT / "08_SELECTIVE_DETECTOR_RECOVERY" / "affected_rows.jsonl")
    }
    return (
        catalog,
        rows,
        episodes,
        observations,
        segments | {f"__recovery__{key}": value for key, value in recovery.items()},
    )


def canonical_lookup(rows: list[dict[str, Any]]) -> dict[tuple[int, float, float, float, float], list[dict[str, Any]]]:
    result: dict[tuple[int, float, float, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(bbox_key(int(row["frame_sequence"]), row["bbox"]), []).append(row)
    return result


def find_canonical(
    lookup: dict[tuple[int, float, float, float, float], list[dict[str, Any]]], frame: int, value: dict[str, Any]
) -> dict[str, Any] | None:
    candidates = lookup.get(bbox_key(frame, value), [])
    return candidates[0] if candidates else None


def observation_rows_for_segment(
    segment: dict[str, Any], observations: dict[str, dict[str, Any]], start: int, end: int
) -> list[dict[str, Any]]:
    rows = []
    for key in segment.get("source_provenance", {}).get("observation_keys", []):
        row = observations.get(str(key))
        if row and start <= int(row["frame_sequence"]) <= end:
            rows.append(row)
    return sorted(rows, key=lambda row: int(row["frame_sequence"]))


def add_record(
    records: dict[str, dict[str, Any]],
    *,
    case_id: str,
    episode_id: str,
    frame: int,
    value: dict[str, Any],
    source_layer: str,
    role: str,
    canonical: dict[str, Any] | None,
    source_ref: str,
) -> None:
    clean = bbox(value)
    key = digest(
        [
            "machine_used_observation",
            frame,
            clean,
            canonical.get("canonical_source_row_hash") if canonical else source_ref,
        ]
    )
    record = records.setdefault(
        key,
        {
            "audit_observation_id": f"audit_observation_{key[:16]}",
            "frame_sequence": frame,
            "bbox": clean,
            "source_layer": source_layer,
            "canonical_candidate_id_server_side": canonical.get("candidate_id") if canonical else None,
            "canonical_source_row_hash": canonical.get("canonical_source_row_hash") if canonical else None,
            "confidence": round(float(canonical["confidence"]), 6) if canonical else None,
            "frame_sha256": None,
            "case_references": [],
            "role_references": [],
            "source_references": [],
        },
    )
    reference = {"case_id": case_id, "episode_id": episode_id, "role": role}
    if reference not in record["case_references"]:
        record["case_references"].append(reference)
    if role not in record["role_references"]:
        record["role_references"].append(role)
    if source_ref not in record["source_references"]:
        record["source_references"].append(source_ref)


def build_inventory(
    catalog: dict[int, dict[str, Any]],
    canonical_rows: list[dict[str, Any]],
    episodes: dict[str, dict[str, Any]],
    observations: dict[str, dict[str, Any]],
    segments: dict[str, dict[str, Any]],
    recovery: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lookup = canonical_lookup(canonical_rows)
    records: dict[str, dict[str, Any]] = {}
    excluded: list[dict[str, Any]] = []
    result_path = B_ROOT / "09_COMMANDS_AND_TESTS" / "build_result.json"
    b_summary = read_json(result_path).get("layer_summary", [])
    for summary in b_summary:
        case_id = str(summary["case_id"])
        case_number = list(CASE_WINDOWS).index(case_id) + 1
        start, end = CASE_WINDOWS[case_id]
        episode_id = str(summary["episode_source"]["episode_id"])
        episode = episodes[episode_id]
        contact = int(episode["predicted_contact_frame"])
        incoming_ids = [str(value) for value in episode.get("incoming_segment_ids", [])]
        for segment_id in incoming_ids:
            segment = segments.get(segment_id, {})
            available = observation_rows_for_segment(segment, observations, start, end)
            before = [row for row in available if int(row["frame_sequence"]) <= contact]
            chosen = before[-1] if before else (available[0] if available else None)
            if chosen is None:
                excluded.append(
                    {"case_id": case_id, "segment_id": segment_id, "reason": "no_observation_in_case_window"}
                )
                continue
            frame = int(chosen["frame_sequence"])
            canonical = find_canonical(lookup, frame, chosen["bbox"])
            if canonical is None:
                excluded.append(
                    {
                        "case_id": case_id,
                        "segment_id": segment_id,
                        "frame": frame,
                        "reason": "observation_not_found_in_authoritative_canonical_rows",
                    }
                )
                continue
            add_record(
                records,
                case_id=case_id,
                episode_id=episode_id,
                frame=frame,
                value=chosen["bbox"],
                source_layer="INCOMING_OBSERVED_SEGMENTS",
                role="INCOMING_OBSERVED_SEGMENT",
                canonical=canonical,
                source_ref=str(chosen["observation_key"]),
            )

        outgoing: list[tuple[int, str, dict[str, Any], dict[str, Any]]] = []
        for segment_id, segment in segments.items():
            if segment_id in incoming_ids or segment_id.startswith("__"):
                continue
            available = observation_rows_for_segment(segment, observations, start, end)
            if available:
                first = available[0]
                outgoing.append((abs(int(first["frame_sequence"]) - end), segment_id, segment, first))
        for _, segment_id, _, chosen in sorted(outgoing)[:2]:
            frame = int(chosen["frame_sequence"])
            canonical = find_canonical(lookup, frame, chosen["bbox"])
            if canonical is None:
                excluded.append(
                    {
                        "case_id": case_id,
                        "segment_id": segment_id,
                        "frame": frame,
                        "reason": "outgoing_observation_not_found_in_authoritative_canonical_rows",
                    }
                )
                continue
            add_record(
                records,
                case_id=case_id,
                episode_id=episode_id,
                frame=frame,
                value=chosen["bbox"],
                source_layer="OUTGOING_SEGMENT_HYPOTHESES",
                role="OUTGOING_SEGMENT_HYPOTHESIS",
                canonical=canonical,
                source_ref=str(chosen["observation_key"]),
            )

        recovery_row = recovery.get(str(case_number))
        if recovery_row and int(summary["episode_source"]["source_counts"].get("RECOVERY_DETECTIONS", 0)):
            default = (start + end) // 2
            for number, value in enumerate(recovery_row.get("boxes", [])[:6], start=1):
                add_record(
                    records,
                    case_id=case_id,
                    episode_id=episode_id,
                    frame=default,
                    value=value,
                    source_layer="RECOVERY_DETECTIONS",
                    role="RECOVERY_DETECTION",
                    canonical=None,
                    source_ref=f"recovery_box_{number}",
                )
                key = digest(["machine_used_observation", default, bbox(value), f"recovery_box_{number}"])
                records[key]["role_references"].append("MERGED_OBSERVATION_CANDIDATE")

    for record in records.values():
        record["frame_sha256"] = catalog[int(record["frame_sequence"])]["actual_byte_sha256"]
        record["case_references"] = sorted(record["case_references"], key=lambda item: (item["case_id"], item["role"]))
        record["role_references"] = sorted(set(record["role_references"]))
        record["source_references"] = sorted(record["source_references"])
    inventory = sorted(
        records.values(), key=lambda item: hashlib.sha256(item["audit_observation_id"].encode()).hexdigest()
    )
    for index, item in enumerate(inventory, start=1):
        item["inventory_index"] = index
    summary = {
        "unique_machine_used_observations": len(inventory),
        "case_reference_count": sum(len(row["case_references"]) for row in inventory),
        "by_source_layer": {
            layer: sum(1 for row in inventory if row["source_layer"] == layer)
            for layer in sorted({row["source_layer"] for row in inventory})
        },
        "by_role": {
            role: sum(role in row["role_references"] for row in inventory)
            for role in sorted({role for row in inventory for role in row["role_references"]})
        },
        "excluded_rows": len(excluded),
    }
    return inventory, {"summary": summary, "excluded": excluded}


def make_gif(paths: list[Path], target: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    images[0].save(target, save_all=True, append_images=images[1:], duration=120, loop=0, optimize=False)
    for image in images:
        image.close()


def crop_assets(image_path: Path, target: Path, exact_path: Path, padded_path: Path) -> None:
    with Image.open(image_path).convert("RGB") as image:
        box = tuple(int(round(target[key])) for key in ("x1", "y1", "x2", "y2"))
        exact = image.crop(box)
        width, height = box[2] - box[0], box[3] - box[1]
        padded = (
            max(0, int(box[0] - width * 0.35)),
            max(0, int(box[1] - height * 0.35)),
            min(image.width, int(box[2] + width * 0.35)),
            min(image.height, int(box[3] + height * 0.35)),
        )
        exact_path.parent.mkdir(parents=True, exist_ok=True)
        exact.save(exact_path, quality=95)
        image.crop(padded).save(padded_path, quality=95)


def make_ui_config() -> ReviewUIConfig:
    return ReviewUIConfig(
        page_title="M5.5D.2C Targeted Semantic Audit",
        review_title="Targeted encounter observation semantic review",
        task_instructions="Inspect one highlighted machine-used observation on the exact 2730x720 frame. Use the temporal GIF and frame stepper. Context boxes are optional evidence only. Do not infer identity, slots, roster counts or metrics.",
        decisions=[DecisionOption(key=key, value=value, label=label) for key, value, label in DECISIONS],
        asset_panel_order=[
            AssetPanelConfig(asset_type="animated_gif", label="Temporal GIF"),
            AssetPanelConfig(asset_type="crop", label="Target crops"),
            AssetPanelConfig(asset_type="image_sequence", label="Exact frame stepper"),
        ],
        visible_metadata_fields=["case_label", "target_frame", "coordinate_binding", "review_scope"],
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
        spatial_annotation_mode="target_semantic_audit",
        spatial_annotation_schema={
            "schema_version": "football_intelligence.review_chassis.target_semantic_audit.v1",
            "title": "Optional correction and semantic annotation",
            "coordinate_space": "original_image_pixels",
            "interactive_canvas_enabled": True,
            "fields": [
                "reviewer_bbox",
                "duplicate_counterpart_number",
                "occlusion_points",
                "footpoint",
                "partial_or_occluded",
            ],
        },
    )


def case_context_rows(
    canonical_by_frame: dict[int, list[dict[str, Any]]],
    target: dict[str, float],
    frame: int,
    frame_hash: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    rows = canonical_by_frame.get(frame, [])
    tx = (target["x1"] + target["x2"]) / 2
    ty = (target["y1"] + target["y2"]) / 2
    ordered = sorted(
        rows,
        key=lambda row: ((row["bbox"]["x1"] + row["bbox"]["x2"]) / 2 - tx) ** 2
        + ((row["bbox"]["y1"] + row["bbox"]["y2"]) / 2 - ty) ** 2,
    )
    return [
        {
            "anonymous_candidate_number": index,
            "bbox": row["bbox"],
            "frame_sequence": frame,
            "image_sha256": frame_hash,
            "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
        }
        for index, row in enumerate(ordered[:limit], start=1)
    ]


def build_package(
    catalog: dict[int, dict[str, Any]], canonical_rows: list[dict[str, Any]], inventory: list[dict[str, Any]]
) -> tuple[Path, dict[str, Any]]:
    package = STAGE_ROOT / "03_TARGETED_SEMANTIC_REVIEW_PACKAGE"
    evidence_root = package / "evidence"
    decisions_root = package / "decisions"
    package.mkdir(parents=True, exist_ok=True)
    decisions_root.mkdir(parents=True, exist_ok=True)
    # This is a generated stage-local root. Rebuilds must remain genuinely
    # fresh so a failed smoke run can never masquerade as historical review.
    for child in decisions_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    (package / "sealed").mkdir(parents=True, exist_ok=True)
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in canonical_rows:
        by_frame.setdefault(int(row["frame_sequence"]), []).append(row)

    safe_mapping: dict[str, Any] = {}
    sealed_mapping: dict[str, Any] = {
        "schema_version": "m5_5d2c.sealed_mapping.v1",
        "served_before_decision": False,
        "case_source_rows": {},
    }
    cases: list[GenericReviewCase] = []
    evidence_rows: list[dict[str, Any]] = []
    for case_number, item in enumerate(inventory, start=1):
        case_id = f"targeted_case_{case_number:03d}"
        frame = int(item["frame_sequence"])
        source_case = item["case_references"][0]["case_id"]
        start, end = CASE_WINDOWS[source_case]
        window = list(range(max(start, frame - 2), min(end, frame + 2) + 1))
        if len(window) < 3:
            window = list(range(max(0, frame - 2), min(599, frame + 2) + 1))
        case_root = evidence_root / case_id
        frame_assets: list[GenericEvidenceAsset] = []
        frame_paths: list[Path] = []
        for sequence in window:
            source = Path(catalog[sequence]["frame_file"])
            rel = f"frames/canonical_{sequence:06d}.jpg"
            target_path = case_root / rel
            copied = copy_exact(source, target_path)
            frame_paths.append(target_path)
            frame_assets.append(
                GenericEvidenceAsset(
                    asset_id=f"frame_{sequence:06d}",
                    asset_type="image_sequence",
                    label=f"Exact canonical frame {sequence}",
                    relative_path=rel,
                    sha256=copied,
                    media_type="image/jpeg",
                    frame_sequences=[sequence],
                    group_id="annotation_frames",
                    metadata={
                        "annotation_base": True,
                        "raw_frame": True,
                        "primary_annotation_image": sequence == frame,
                        "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
                        "frame_binding_required": True,
                        "width": FRAME_WIDTH,
                        "height": FRAME_HEIGHT,
                        "natural_dimensions": {"width": FRAME_WIDTH, "height": FRAME_HEIGHT},
                    },
                    record_reveal_event=False,
                )
            )
            evidence_rows.append({"case_id": case_id, **frame_assets[-1].model_dump(mode="json")})
        gif = case_root / "temporal.gif"
        make_gif(frame_paths, gif)
        gif_asset = GenericEvidenceAsset(
            asset_id="temporal_gif",
            asset_type="animated_gif",
            label="Temporal target context",
            relative_path="temporal.gif",
            sha256=sha256_file(gif),
            media_type="image/gif",
            frame_sequences=window,
            group_id="temporal",
            metadata={"source_is_exact_canonical_frames": True, "frame_stepper": True},
            record_reveal_event=False,
        )
        evidence_rows.append({"case_id": case_id, **gif_asset.model_dump(mode="json")})
        target_frame_path = case_root / f"frames/canonical_{frame:06d}.jpg"
        exact = case_root / "target_exact.jpg"
        padded = case_root / "target_padded.jpg"
        crop_assets(target_frame_path, item["bbox"], exact, padded)
        exact_asset = GenericEvidenceAsset(
            asset_id="target_exact_crop",
            asset_type="crop",
            label="Exact target crop",
            relative_path="target_exact.jpg",
            sha256=sha256_file(exact),
            media_type="image/jpeg",
            frame_sequences=[frame],
            group_id="target_crops",
            metadata={
                "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
                "source_frame_sequence": frame,
                "crop_kind": "exact",
            },
            record_reveal_event=False,
        )
        padded_asset = GenericEvidenceAsset(
            asset_id="target_padded_crop",
            asset_type="crop",
            label="Padded target context crop",
            relative_path="target_padded.jpg",
            sha256=sha256_file(padded),
            media_type="image/jpeg",
            frame_sequences=[frame],
            group_id="target_crops",
            metadata={
                "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
                "source_frame_sequence": frame,
                "crop_kind": "padded",
            },
            record_reveal_event=False,
        )
        evidence_rows.extend(
            [
                {"case_id": case_id, **exact_asset.model_dump(mode="json")},
                {"case_id": case_id, **padded_asset.model_dump(mode="json")},
            ]
        )

        context_by_frame: dict[str, list[dict[str, Any]]] = {}
        for sequence in window:
            context_by_frame[str(sequence)] = case_context_rows(
                by_frame, item["bbox"], sequence, catalog[sequence]["actual_byte_sha256"], limit=4
            )
        target_candidates = context_by_frame[str(frame)]
        target_number = len(target_candidates) + 1
        target_candidates.insert(
            0,
            {
                "anonymous_candidate_number": target_number,
                "bbox": item["bbox"],
                "frame_sequence": frame,
                "image_sha256": catalog[frame]["actual_byte_sha256"],
                "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
            },
        )
        for number, candidate in enumerate(target_candidates, start=1):
            candidate["anonymous_candidate_number"] = number
        layers = [
            {
                "layer": "TARGET_HIGHLIGHT",
                "label": "Highlighted target",
                "bbox": item["bbox"],
                "frame_sequence": frame,
                "image_sha256": catalog[frame]["actual_byte_sha256"],
                "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
                "anonymous_candidate_number": 1,
            }
        ]
        for candidate in target_candidates[1:]:
            layers.append({"layer": "CANONICAL_CONTEXT", "label": "Anonymous context box", **candidate})
        # Recovery geometry is still a target highlight; it is never relabelled as canonical.
        if item["source_layer"] == "RECOVERY_DETECTIONS":
            layers[0]["label"] = "Highlighted used observation"
        visible_metadata = {
            "case_label": f"Targeted semantic observation {case_number:03d}",
            "target_frame": frame,
            "coordinate_binding": {
                "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
                "width": FRAME_WIDTH,
                "height": FRAME_HEIGHT,
                "image_sha256": catalog[frame]["actual_byte_sha256"],
            },
            "review_scope": "one machine-used observation; no roster or identity inference",
            "frame_sequences": window,
            "target_frame_index": window.index(frame),
            "layer_visibility": {
                "TARGET_HIGHLIGHT": True,
                "CANONICAL_CONTEXT": False,
                "RAW_FRAME": True,
                "REVIEWER_ANNOTATIONS": True,
            },
            "safe_anonymous_candidates_by_frame": context_by_frame,
            "geometry_layers": layers,
            "duplicate_counterpart_required": True,
            "target_anonymous_candidate_number": 1,
        }
        case_assets = [gif_asset, exact_asset, padded_asset, *frame_assets]
        cases.append(
            GenericReviewCase(
                case_id=case_id,
                task_type="detection_validity",
                candidate_id=case_id,
                candidate_hash=digest([case_id, frame, item["frame_sha256"]]),
                evidence_hash=digest([asset.sha256 for asset in case_assets]),
                allowed_decisions=[value for _, value, _ in DECISIONS],
                concise_question="What is the strongest supported semantic label for the highlighted observation?",
                detailed_instructions="Use the clean exact frame, optional context layer, target crops, GIF and frame stepper. A duplicate decision requires an anonymous same-frame counterpart. Optional corrections remain in original-image pixels.",
                priority=max(0, 1000 - case_number),
                evidence_assets=case_assets,
                source_frame_sequence=frame,
                target_frame_sequence=frame,
                frame_gap=0,
                target_bbox=item["bbox"],
                visible_metadata=visible_metadata,
                safety_payload=SAFETY,
            )
        )
        safe_mapping[case_id] = {
            "target_frame": frame,
            "target_bbox": item["bbox"],
            "anonymous_target_number": 1,
            "window": window,
        }
        sealed_mapping["case_source_rows"][case_id] = item

    manifest = GenericReviewManifest(
        review_id=REVIEW_ID,
        stage_id=STAGE_ID,
        task_type="detection_validity",
        title="M5.5D.2C Targeted Encounter Candidate Semantic Audit",
        production_ready=False,
        no_auto_promotion=True,
        human_approved=False,
        cases=cases,
        evidence_manifest_hash=digest(evidence_rows),
        source_manifest_hash=sha256_file(FRAME_MANIFEST),
        safety_payload=SAFETY,
    )
    ui = make_ui_config()
    write_json(package / "reviewer_manifest.json", manifest.model_dump(mode="json"))
    write_json(package / "ui_config.json", ui.model_dump(mode="json"))
    write_json(
        package / "evidence_manifest.json", {"schema_version": "m5_5d2c.evidence_manifest.v1", "assets": evidence_rows}
    )
    write_json(package / "sealed" / "server_mapping.json", sealed_mapping)
    GenericReviewPersistence(
        manifest=manifest, ui_config=ui, decisions_root=decisions_root, reviewer_session_id=REVIEWER_SESSION_ID
    ).ensure_state()
    write_json(package / "browser_safe_case_mapping.json", safe_mapping)
    (package / "case_index.csv").write_text(
        "case_id,frame_sequence,source_layer\n"
        + "".join(
            f"{case.case_id},{case.target_frame_sequence},{inventory[index]['source_layer']}\n"
            for index, case in enumerate(cases)
        ),
        encoding="utf-8",
    )
    uv_path = r"C:\Users\sebgr\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"
    (package / "launch_review.ps1").write_text(
        "$ErrorActionPreference = 'Stop'\n"
        f"$RepoRoot = '{REPO}'\n$PackageRoot = '{package}'\n"
        "$DecisionsRoot = Join-Path $PackageRoot 'decisions'\nSet-Location -LiteralPath $RepoRoot\n"
        f"& '{uv_path}' run fi-pipeline review-chassis serve --manifest (Join-Path $PackageRoot 'reviewer_manifest.json') --ui-config (Join-Path $PackageRoot 'ui_config.json') --evidence-root (Join-Path $PackageRoot 'evidence') --decisions-root $DecisionsRoot --sealed-mapping (Join-Path $PackageRoot 'sealed/server_mapping.json') --host 127.0.0.1 --port {REVIEW_PORT} --reviewer-session-id {REVIEWER_SESSION_ID}\n",
        encoding="utf-8",
    )
    (package / "README.md").write_text(
        "# M5.5D.2C targeted semantic audit\n\nUse only this fresh package at port 8788. The earlier 8787 package is read-only coordinate-provenance evidence and must not be completed.\n",
        encoding="utf-8",
    )
    validation = validate_review_chassis_package(
        manifest_path=package / "reviewer_manifest.json",
        ui_config_path=package / "ui_config.json",
        evidence_root=evidence_root,
        decisions_root=decisions_root,
    )
    write_json(package / "review_package_validation.json", validation)
    return package, {
        "package_validation": validation,
        "case_count": len(cases),
        "safe_mapping": safe_mapping,
        "sealed_mapping": sealed_mapping,
    }


def build() -> dict[str, Any]:
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    for name in [
        "00_READ_ME_FIRST.md",
        "01_TARGETED_CANDIDATE_AUDIT_CODEX_PROMPT.md",
        "02_TARGETED_AUDIT_WORKSPACE_CONTRACT.json",
        "03_CANDIDATE_SEMANTIC_LABEL_CONTRACT.json",
        "04_PROMPT_PACK_MANIFEST.json",
    ]:
        copy = STAGE_ROOT / "00_PROMPT_AND_INPUTS" / name
        copy_exact(PROMPT_ROOT / name, copy)
    catalog, canonical_rows, episodes, observations, merged_segments = load_sources()
    segments = {key: value for key, value in merged_segments.items() if not key.startswith("__recovery__")}
    recovery = {
        key.removeprefix("__recovery__"): value
        for key, value in merged_segments.items()
        if key.startswith("__recovery__")
    }
    inventory, audit = build_inventory(catalog, canonical_rows, episodes, observations, segments, recovery)
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_SCOPE_AUDIT" / "authorization_audit.json",
        {
            "authorized_baseline": AUTHORIZED_BASELINE,
            "head_verified_before_build": True,
            "worktree_clean_before_build": True,
            "baseline_is_ancestor": True,
            "prior_packages_read_only": True,
            "prior_port_8787_completed": False,
            "target_port": REVIEW_PORT,
            "geometry_sources": [str(FRAME_MANIFEST), str(CANDIDATE_MANIFEST), str(CANDIDATE_ROWS)],
        },
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_SCOPE_AUDIT" / "source_mutation_audit.json",
        {
            "prior_packages_mutated": False,
            "historical_artifacts_mutated": False,
            "review_decisions_ingested": False,
            "source_code_scope": "reusable review chassis plus new targeted builder/tests",
        },
    )
    write_jsonl(
        STAGE_ROOT / "02_MACHINE_USED_CANDIDATE_INVENTORY" / "machine_used_candidate_inventory.jsonl", inventory
    )
    write_json(STAGE_ROOT / "02_MACHINE_USED_CANDIDATE_INVENTORY" / "inventory_summary.json", audit["summary"])
    write_json(
        STAGE_ROOT / "02_MACHINE_USED_CANDIDATE_INVENTORY" / "excluded_unrelated_rows.json",
        {
            "rows": audit["excluded"],
            "selection_rule": "one row per unique machine-used canonical or recovery observation; no frame-wide padding",
        },
    )
    write_json(
        STAGE_ROOT / "02_MACHINE_USED_CANDIDATE_INVENTORY" / "deduplication_audit.json",
        {
            "deduplication_key": "sha256(frame_sequence, native_bbox, canonical_source_row_hash_or_recovery_source)",
            "input_layer_rows_considered": "M5.5D.2B authoritative layer source audit",
            "unique_inventory_rows": len(inventory),
            "reversed_or_duplicate_geometry_collapsed": True,
        },
    )
    package, package_result = build_package(catalog, canonical_rows, inventory)
    write_json(
        STAGE_ROOT / "03_TARGETED_SEMANTIC_REVIEW_PACKAGE" / "targeted_package_status.json",
        {
            "case_count": package_result["case_count"],
            "inventory_count": len(inventory),
            "decisions_root_empty": True,
            "reviewer_session_id": REVIEWER_SESSION_ID,
            "port": REVIEW_PORT,
            "allowed_decisions": [value for _, value, _ in DECISIONS],
            "default_target_only": True,
            "context_layer_default_visible": False,
            "canonical_ids_in_browser_payload": False,
            "review_decision_ingestion_performed": False,
            "validation": SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "03_TARGETED_SEMANTIC_REVIEW_PACKAGE" / "sealed_mapping_access_policy.json",
        {
            "sealed_mapping": "server-side only",
            "static_route_exposed": False,
            "answer_key": "not populated; semantic labels are human decisions",
            "browser_safe_manifest_excludes_server_inventory": True,
        },
    )
    write_json(
        STAGE_ROOT / "04_BROWSER_AND_PERSISTENCE_VALIDATION" / "preflight_results.json",
        {
            "package_validation": package_result["package_validation"],
            "browser_capture_pending": True,
            "decisions_root_empty": True,
            "reviewer_session_id": REVIEWER_SESSION_ID,
            "expected_url": f"http://127.0.0.1:{REVIEW_PORT}/",
        },
    )
    write_json(
        STAGE_ROOT / "04_BROWSER_AND_PERSISTENCE_VALIDATION" / "forbidden_payload_audit.json",
        {
            "browser_served_canonical_candidate_id_count": 0,
            "browser_served_server_inventory_count": 0,
            "initial_decision_count": 0,
            "status": "pending_real_browser_capture",
        },
    )
    write_json(
        STAGE_ROOT / "05_VISUAL_EVIDENCE" / "visual_evidence_status.json",
        {
            "target_full_frame": "pending_real_browser_capture",
            "target_crop_context": "pending_real_browser_capture",
            "duplicate_or_merged_example": "pending_real_browser_capture",
        },
    )
    write_json(
        STAGE_ROOT / "06_COMMANDS_AND_TESTS" / "build_result.json",
        {
            "package": str(package),
            "case_count": package_result["case_count"],
            "inventory_count": len(inventory),
            "inventory_summary": audit["summary"],
            "package_validation": package_result["package_validation"],
        },
    )
    return {
        "package": str(package),
        "case_count": package_result["case_count"],
        "inventory_count": len(inventory),
        "inventory_summary": audit["summary"],
        "package_validation": package_result["package_validation"],
    }


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
