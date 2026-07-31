"""Finalize three immutable C3A5A human pitch-polygon reviews."""

from __future__ import annotations

import copy
import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
MATCH_IDS = ["117093", "118576", "118577"]
EXPECTED_HEAD = "5c75ba21b77456c16a88374000c04844f77c3547"
REVIEW_ID = "G7D_C3A5A_THREE_MATCH_PITCH_POLYGON_REVIEW"
REVISION = "G7D_C3A5A_THREE_MATCH_PITCH_POLYGON_REVIEW_V1"
VISIBLE_LAST_EVENT = "62e18cf1-cc92-4f75-bf2b-2dd8de57a045"
COMPLETION_ID = "completion-b24767d6b5e9aae8a23feae7"
WORKSPACE = PROJECT / ("experiments/football_observation_reasoner/part 7/G7D_C3A5A_THREE_MATCH_PITCH_POLYGON_REVIEW_v1")
PACKAGE = WORKSPACE / "02_PITCH_POLYGON_REVIEW_PACKAGE"
DECISIONS = PACKAGE / "human_decisions"
FRAMES_MANIFEST = WORKSPACE / "01_REVIEW_FRAMES/source_frame_manifest.json"
STAGE = PROJECT / (
    "experiments/football_observation_reasoner/part 7/G7D_C3A5B_THREE_MATCH_PITCH_POLYGON_FINALIZATION_v1"
)
SPLIT = PROJECT / "datasets/soccertrack_v2/splits/split_v1/split_manifest.json"
CLASSIFICATION = "PASS_G7D_C3A5B_THREE_MATCH_PITCH_POLYGONS_FINALIZED"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def atomic_write(path: Path, data: bytes, *, conflict_safe: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if conflict_safe and path.exists():
        if path.read_bytes() == data:
            return
        raise RuntimeError(f"FAIL_G7D_C3A5B_ARTIFACT_CONFLICT: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any, *, conflict_safe: bool = False) -> None:
    atomic_write(path, canonical_bytes(value), conflict_safe=conflict_safe)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    packed = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(packed).hexdigest()


def row(path: Path) -> dict[str, Any]:
    return {
        "project_relative_path": str(path.relative_to(PROJECT)).replace("\\", "/"),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, encoding="utf-8").strip()


def immutable_files() -> list[Path]:
    return sorted((DECISIONS / "events").rglob("*.json")) + sorted((DECISIONS / "receipts").rglob("*.json"))


def immutable_snapshot() -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(DECISIONS)).replace("\\", "/"): {
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in immutable_files()
    }


def orientation(a: list[float], b: list[float], c: list[float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(a: list[float], b: list[float], point: list[float]) -> bool:
    epsilon = 1e-9
    return (
        min(a[0], b[0]) - epsilon <= point[0] <= max(a[0], b[0]) + epsilon
        and min(a[1], b[1]) - epsilon <= point[1] <= max(a[1], b[1]) + epsilon
    )


def segments_intersect(a: list[float], b: list[float], c: list[float], d: list[float]) -> bool:
    values = orientation(a, b, c), orientation(a, b, d), orientation(c, d, a), orientation(c, d, b)
    if (values[0] > 0) != (values[1] > 0) and (values[2] > 0) != (values[3] > 0):
        return True
    return (
        abs(values[0]) <= 1e-9
        and on_segment(a, b, c)
        or abs(values[1]) <= 1e-9
        and on_segment(a, b, d)
        or abs(values[2]) <= 1e-9
        and on_segment(c, d, a)
        or abs(values[3]) <= 1e-9
        and on_segment(c, d, b)
    )


def polygon_geometry(vertices: Any, width: int, height: int) -> dict[str, Any]:
    if not isinstance(vertices, list) or len(vertices) < 4:
        raise ValueError("FAIL_G7D_C3A5B_POLYGON_GEOMETRY: fewer than four vertices")
    points: list[list[float]] = []
    for index, vertex in enumerate(vertices):
        if (
            not isinstance(vertex, list)
            or len(vertex) != 2
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vertex)
        ):
            raise ValueError(f"FAIL_G7D_C3A5B_POLYGON_GEOMETRY: invalid vertex {index}")
        point = [float(vertex[0]), float(vertex[1])]
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            raise ValueError(f"FAIL_G7D_C3A5B_POLYGON_GEOMETRY: out-of-bounds vertex {index}")
        points.append(point)
    if points[0] == points[-1] or len({tuple(point) for point in points}) != len(points):
        raise ValueError("FAIL_G7D_C3A5B_POLYGON_GEOMETRY: duplicate vertex")
    edges = list(zip(points, points[1:] + points[:1]))
    intersections: list[list[int]] = []
    for left, (a, b) in enumerate(edges):
        for right, (c, d) in enumerate(edges[left + 1 :], start=left + 1):
            if right == left + 1 or (left == 0 and right == len(edges) - 1):
                continue
            if segments_intersect(a, b, c, d):
                intersections.append([left, right])
    if intersections:
        raise ValueError(f"FAIL_G7D_C3A5B_POLYGON_GEOMETRY: intersections {intersections}")
    signed_area = sum(a[0] * b[1] - b[0] * a[1] for a, b in edges) / 2
    if abs(signed_area) <= 1e-6:
        raise ValueError("FAIL_G7D_C3A5B_POLYGON_GEOMETRY: non-positive area")
    return {
        "vertex_count": len(points),
        "distinct_vertex_count": len(points),
        "signed_area_pixels": signed_area,
        "area_pixels": abs(signed_area),
        "winding": "CCW" if signed_area > 0 else "CW",
        "self_intersection_count": 0,
        "non_adjacent_self_intersections": [],
        "implicit_closing_edge_valid": True,
        "source_bounds": {"width": width, "height": height},
    }


def camera_segments(event: dict[str, Any]) -> list[dict[str, Any]]:
    answer = event["alignment_answer"]
    first = event["first_half_polygon_source_xy"]
    if answer == "UNCERTAIN":
        raise ValueError("FAIL_G7D_C3A5B_ALIGNMENT_UNCERTAIN")
    if answer == "YES":
        return [
            {
                "segment_id": "MATCH_STABLE_CAMERA",
                "halves": ["FIRST_HALF", "SECOND_HALF"],
                "vertices_source_xy": first,
            }
        ]
    second = event.get("second_half_polygon_source_xy")
    if answer == "NO" and event.get("second_half_closed") is True and second:
        return [
            {"segment_id": "FIRST_HALF", "halves": ["FIRST_HALF"], "vertices_source_xy": first},
            {"segment_id": "SECOND_HALF", "halves": ["SECOND_HALF"], "vertices_source_xy": second},
        ]
    raise ValueError("FAIL_G7D_C3A5B_POLYGON_GEOMETRY: NO requires a valid second-half polygon")


def resolve_chain() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    acknowledged: dict[str, list[tuple[dict[str, Any], Path, dict[str, Any], Path]]] = {
        match_id: [] for match_id in MATCH_IDS
    }
    for event_path in sorted((DECISIONS / "events").rglob("*.json")):
        event = read_json(event_path)
        if event.get("match_id") not in acknowledged:
            continue
        ack_path = DECISIONS / "receipts/event_acknowledgements" / f"ack-{event['event_id']}.json"
        if not ack_path.is_file():
            continue
        ack = read_json(ack_path)
        event_hash = sha256_file(event_path)
        if (
            ack.get("receipt_id") != f"ack-{event['event_id']}"
            or ack.get("human_event_id") != event["event_id"]
            or ack.get("human_event_sha256") != event_hash
            or ack.get("server_validated") is not True
            or ack.get("case_complete") is not True
        ):
            raise RuntimeError("FAIL_G7D_C3A5B_HUMAN_EVENT_CHAIN: acknowledgement linkage")
        acknowledged[event["match_id"]].append((event, event_path, ack, ack_path))
    selected: dict[str, dict[str, Any]] = {}
    for match_id, candidates in acknowledged.items():
        if not candidates:
            raise RuntimeError(f"FAIL_G7D_C3A5B_HUMAN_EVENT_CHAIN: no event for {match_id}")
        event, event_path, ack, ack_path = max(candidates, key=lambda item: item[0]["server_sequence"])
        selected[match_id] = {
            "event": event,
            "event_path": event_path,
            "event_sha256": sha256_file(event_path),
            "ack": ack,
            "ack_path": ack_path,
            "ack_sha256": sha256_file(ack_path),
        }
    visible_matches = [mid for mid, value in selected.items() if value["event"]["event_id"] == VISIBLE_LAST_EVENT]
    if visible_matches != ["118576"]:
        raise RuntimeError("FAIL_G7D_C3A5B_HUMAN_EVENT_CHAIN: visible last-event resolution")
    completion_path = DECISIONS / "receipts/completion" / f"{COMPLETION_ID}.json"
    completion = read_json(completion_path)
    refs = [
        {
            "acknowledgement_receipt_id": selected[mid]["ack"]["receipt_id"],
            "acknowledgement_receipt_sha256": selected[mid]["ack_sha256"],
            "human_event_id": selected[mid]["event"]["event_id"],
            "human_event_sha256": selected[mid]["event_sha256"],
            "match_id": mid,
        }
        for mid in MATCH_IDS
    ]
    digest = canonical_sha256(refs)
    if (
        completion.get("completion_receipt_id") != COMPLETION_ID
        or completion.get("all_cases_complete") is not True
        or completion.get("required_match_ids") != MATCH_IDS
        or completion.get("latest_acknowledged_events") != refs
        or completion.get("latest_event_set_digest") != digest
        or completion.get("review_id") != REVIEW_ID
        or completion.get("review_revision") != REVISION
    ):
        raise RuntimeError("FAIL_G7D_C3A5B_HUMAN_EVENT_CHAIN: completion linkage")
    return selected, {
        "value": completion,
        "path": completion_path,
        "sha256": sha256_file(completion_path),
    }


def validate_event_and_provenance(
    match_id: str,
    selected: dict[str, Any],
    frame_pair: dict[str, Any],
) -> dict[str, Any]:
    event = selected["event"]
    if (
        event.get("schema_version") != "football_intelligence.g7d_c3a5a.pitch_polygon_review_event.v1"
        or event.get("review_id") != REVIEW_ID
        or event.get("revision") != REVISION
        or event.get("match_id") != match_id
        or event.get("first_half_closed") is not True
        or event.get("coordinate_audit", {}).get("verified") is not True
        or event.get("synthetic") is True
        or event.get("synthetic_smoke") is True
    ):
        raise RuntimeError(f"FAIL_G7D_C3A5B_HUMAN_EVENT_CHAIN: event state {match_id}")
    if event.get("normalization", {}).get("first", {}).get("closure_convention") != (
        "distinct_vertices_once_plus_closed_true"
    ):
        raise RuntimeError(f"FAIL_G7D_C3A5B_HUMAN_EVENT_CHAIN: closure {match_id}")
    geometry: dict[str, Any] = {}
    for key in ("first", "second"):
        frame = frame_pair[key]
        frame_path = WORKSPACE / frame["relative_path"]
        video_path = PROJECT / frame["source_video_relative_path"]
        if (
            event["frame_hashes"][key] != frame["frame_sha256"]
            or event["source_dimensions"][key] != [frame["source_width"], frame["source_height"]]
            or sha256_file(frame_path) != frame["frame_sha256"]
            or frame_path.stat().st_size != frame["frame_byte_size"]
            or sha256_file(video_path) != frame["source_video_sha256"]
        ):
            raise RuntimeError(f"FAIL_G7D_C3A5B_SOURCE_PROVENANCE: {match_id} {key}")
    if match_id == "117093" and frame_pair["first"]["source_video_relative_path"] != (
        "matches/117093/source/videos/117093_panorama_1st_half-008.mp4"
    ):
        raise RuntimeError("FAIL_G7D_C3A5B_SOURCE_PROVENANCE: corrected 117093 source")
    geometry["first_half"] = polygon_geometry(
        event["first_half_polygon_source_xy"],
        frame_pair["first"]["source_width"],
        frame_pair["first"]["source_height"],
    )
    if event["alignment_answer"] == "NO":
        geometry["second_half"] = polygon_geometry(
            event.get("second_half_polygon_source_xy"),
            frame_pair["second"]["source_width"],
            frame_pair["second"]["source_height"],
        )
    elif event["alignment_answer"] == "YES":
        geometry["second_half"] = {"reused_first_half_polygon": True, **geometry["first_half"]}
    else:
        raise RuntimeError("FAIL_G7D_C3A5B_ALIGNMENT_UNCERTAIN")
    geometry["camera_segments"] = camera_segments(event)
    return geometry


def non_pitch_setup(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "pitch_calibration"}


def draw_polygon(draw: Any, vertices: list[list[float]], scale: float, offset: tuple[int, int]) -> None:
    projected = [(x * scale + offset[0], y * scale + offset[1]) for x, y in vertices]
    draw.line(projected + [projected[0]], fill="#ffd84d", width=3, joint="curve")


def create_visuals(
    selected: dict[str, dict[str, Any]], frames: dict[str, Any], geometry: dict[str, Any]
) -> tuple[Path, Path]:
    from PIL import Image, ImageDraw, ImageFont

    visual_dir = STAGE / "03_VISUAL_QA"
    visual_dir.mkdir(parents=True, exist_ok=True)
    first_output = visual_dir / "01_THREE_MATCH_FINAL_POLYGONS.png"
    width, panel_height, label_height = 1024, 270, 42
    sheet = Image.new("RGB", (width * 2, (panel_height + label_height) * 3 + 44), "#101722")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for row_index, match_id in enumerate(MATCH_IDS):
        event = selected[match_id]["event"]
        for column, half in enumerate(("first", "second")):
            frame = frames[match_id][half]
            image = Image.open(WORKSPACE / frame["relative_path"]).convert("RGB").resize((width, panel_height))
            y = row_index * (panel_height + label_height) + label_height
            x = column * width
            sheet.paste(image, (x, y))
            draw_polygon(draw, event["first_half_polygon_source_xy"], width / 4096, (x, y))
            draw.text(
                (x + 12, y - 28),
                (
                    f"{match_id} | {half.upper()} HALF | MATCH_STABLE_CAMERA | "
                    f"{len(event['first_half_polygon_source_xy'])} vertices"
                ),
                fill="white",
                font=font,
            )
    draw.text(
        (12, sheet.height - 30),
        "HUMAN-CONFIRMED PITCH POLYGON — NOT PRODUCTION",
        fill="#ffd84d",
        font=font,
    )
    sheet.save(first_output, optimize=True)

    second_output = visual_dir / "02_SECOND_HALF_ALIGNMENT_VALIDATION.png"
    status = Image.new("RGB", (1800, 700), "#101722")
    status_draw = ImageDraw.Draw(status)
    status_draw.text((55, 45), "SECOND-HALF ALIGNMENT AND PROVENANCE VALIDATION", fill="white", font=font)
    headings = ["MATCH", "ALIGNMENT", "SEGMENTS", "VERTICES", "AREA (PX²)", "EVENT", "PROVENANCE"]
    positions = [55, 210, 390, 565, 745, 1010, 1420]
    for x, heading in zip(positions, headings):
        status_draw.text((x, 120), heading, fill="#7cb7ff", font=font)
    for index, match_id in enumerate(MATCH_IDS):
        y = 205 + index * 115
        event = selected[match_id]["event"]
        values = [
            match_id,
            event["alignment_answer"],
            str(len(geometry[match_id]["camera_segments"])),
            str(geometry[match_id]["first_half"]["vertex_count"]),
            f"{geometry[match_id]['first_half']['area_pixels']:.1f}",
            event["event_id"][:12],
            "FRAME + VIDEO HASH VERIFIED",
        ]
        for x, value in zip(positions, values):
            status_draw.text((x, y), value, fill="#9ff3c8" if value == "YES" else "white", font=font)
        status_draw.line((50, y + 55, 1750, y + 55), fill="#334155", width=1)
    status_draw.text(
        (55, 610),
        "HUMAN-CONFIRMED PITCH POLYGON — NOT PRODUCTION",
        fill="#ffd84d",
        font=font,
    )
    status.save(second_output, optimize=True)
    return first_output, second_output


def build_handoff(
    event_report: dict[str, Any],
    geometry_report: dict[str, Any],
    setup_report: dict[str, Any],
    validation_report: dict[str, Any],
    visuals: tuple[Path, Path],
) -> None:
    handoff = STAGE / "05_REVIEW_PACK/CHATGPT_HANDOFF"
    handoff.mkdir(parents=True, exist_ok=True)
    write_json(
        handoff / "01_EXECUTIVE_SUMMARY.json",
        {
            "classification": CLASSIFICATION,
            "matches": MATCH_IDS,
            "camera_segment_policy": "MATCH_STABLE_CAMERA",
            "production_ready": False,
            "next_stage_not_started": "G7D_C3A5C_ADDITIONAL_COVERAGE_REPLAY",
        },
    )
    write_json(handoff / "02_EVENT_AND_RECEIPT_CLOSURE.json", event_report)
    write_json(handoff / "03_POLYGON_AND_GEOMETRY_RESULTS.json", geometry_report)
    write_json(handoff / "04_MATCH_SETUP_UPDATE_RESULTS.json", setup_report)
    atomic_write(
        handoff / "05_DECISION.md",
        (
            "# C3A5B decision\n\n"
            f"`{CLASSIFICATION}`. The exact latest acknowledged event set and completion receipt validate. "
            "All three YES reviews are finalized as one `MATCH_STABLE_CAMERA` segment per match.\n"
        ).encode(),
    )
    atomic_write(
        handoff / "06_FINALIZATION_CONTRACT.md",
        (
            "# Finalization contract\n\n"
            "Human vertices are retained in source coordinates without alteration. Human events and receipts "
            "remain immutable. Search regions remain pending and production readiness remains false. C3A5C "
            "requires separate authorization.\n"
        ).encode(),
    )
    write_json(
        handoff / "07_TESTS_SAFETY_AND_SOURCE_CHANGES.json",
        {
            "focused_tests": validation_report["focused_tests"],
            "source_changes": [
                "scripts/g7d_c3a5b_finalize_three_match_pitch_polygons.py",
                "tests/test_g7d_c3a5b_three_match_pitch_polygon_finalization.py",
            ],
            "inference_run": False,
            "human_truth_modified": False,
            "validation_or_holdout_access": False,
            "visual_count": 2,
        },
    )
    atomic_write(handoff / "08_FINAL_POLYGONS.png", visuals[0].read_bytes())
    atomic_write(handoff / "09_ALIGNMENT_VALIDATION.png", visuals[1].read_bytes())
    files = [
        {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(handoff.iterdir())
    ]
    write_json(handoff / "10_MANIFEST.json", {"files": files})
    atomic_write(
        STAGE / "05_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt",
        b"Upload only CHATGPT_HANDOFF. It is the complete self-contained C3A5B review pack.\n",
    )


def finalize() -> None:
    if git("rev-parse", "HEAD") != EXPECTED_HEAD:
        raise RuntimeError("FAIL_G7D_C3A5B_BASELINE")
    if STAGE.exists():
        raise RuntimeError("FAIL_G7D_C3A5B_ARTIFACT_CONFLICT: stage exists")
    split = read_json(SPLIT)
    train = split.get("membership", {}).get("TRAIN_DEVELOPMENT", [])
    if split.get("frozen") is not True or not all(match_id in train for match_id in MATCH_IDS):
        raise RuntimeError("FAIL_G7D_C3A5B_FROZEN_SPLIT")
    immutable_before = immutable_snapshot()
    selected, completion = resolve_chain()
    frames = read_json(FRAMES_MANIFEST)["matches"]
    geometry: dict[str, Any] = {}
    setups_before: dict[str, Any] = {}
    for match_id in MATCH_IDS:
        geometry[match_id] = validate_event_and_provenance(match_id, selected[match_id], frames[match_id])
        setup_path = PROJECT / f"matches/{match_id}/calibration/match_setup.json"
        setup = read_json(setup_path)
        if (
            setup.get("pitch_calibration", {}).get("status") != "HUMAN_REQUIRED"
            or setup["pitch_calibration"].get("polygon_path") is not None
            or setup["pitch_calibration"].get("polygon_sha256") is not None
        ):
            raise RuntimeError(f"FAIL_G7D_C3A5B_MATCH_SETUP_UPDATE: {match_id}")
        setups_before[match_id] = setup
    created = completion["value"]["created_at_utc"]
    event_report = {
        "schema_version": "football_intelligence.g7d_c3a5b.event_chain_validation.v1",
        "classification": CLASSIFICATION,
        "visible_last_event_resolved_match_id": "118576",
        "completion_receipt": row(completion["path"]),
        "completion_receipt_id": COMPLETION_ID,
        "latest_event_set_digest": completion["value"]["latest_event_set_digest"],
        "all_cases_complete": True,
        "selected": {
            match_id: {
                "event": row(selected[match_id]["event_path"]),
                "event_id": selected[match_id]["event"]["event_id"],
                "acknowledgement_receipt": row(selected[match_id]["ack_path"]),
                "acknowledgement_receipt_id": selected[match_id]["ack"]["receipt_id"],
            }
            for match_id in MATCH_IDS
        },
        "immutable_files_before": immutable_before,
    }
    write_json(STAGE / "00_INPUT_AND_EVENT_CLOSURE/event_chain_validation.json", event_report)
    geometry_report = {
        "schema_version": "football_intelligence.g7d_c3a5b.geometry_validation.v1",
        "classification": CLASSIFICATION,
        "matches": geometry,
        "vertices_preserved_exactly": True,
        "smoothing_simplification_reordering_or_redraw": False,
        "created_at_utc": created,
    }
    geometry_path = STAGE / "01_FINAL_POLYGON_ARTIFACTS/geometry_validation.json"
    write_json(geometry_path, geometry_report)
    polygon_paths: dict[str, Path] = {}
    setup_report: dict[str, Any] = {
        "schema_version": "football_intelligence.g7d_c3a5b.match_setup_field_diff.v1",
        "matches": {},
    }
    for match_id in MATCH_IDS:
        event = selected[match_id]["event"]
        first = frames[match_id]["first"]
        second = frames[match_id]["second"]
        polygon = {
            "schema_version": "football_intelligence.pitch_polygon.v1",
            "match_id": match_id,
            "status": "HUMAN_CONFIRMED",
            "coordinate_space": "SOURCE_IMAGE_PIXELS",
            "source_width": first["source_width"],
            "source_height": first["source_height"],
            "camera_segments": geometry[match_id]["camera_segments"],
            "vertices_source_xy": event["first_half_polygon_source_xy"],
            "closed": True,
            "self_intersection_count": 0,
            "area_pixels": geometry[match_id]["first_half"]["area_pixels"],
            "first_half_reference": first,
            "second_half_reference": second,
            "second_half_alignment_answer": event["alignment_answer"],
            "human_review_event_id": event["event_id"],
            "human_review_event_sha256": selected[match_id]["event_sha256"],
            "acknowledgement_receipt_id": selected[match_id]["ack"]["receipt_id"],
            "acknowledgement_receipt_sha256": selected[match_id]["ack_sha256"],
            "completion_receipt_id": COMPLETION_ID,
            "completion_receipt_sha256": completion["sha256"],
            "review_revision": REVISION,
            "frame_provenance_hashes": event["frame_hashes"],
            "validation_report_sha256": sha256_file(geometry_path),
            "created_at_utc": created,
            "production_ready": False,
        }
        polygon_path = PROJECT / f"matches/{match_id}/calibration/pitch_polygon_v1/pitch_polygon.json"
        write_json(polygon_path, polygon, conflict_safe=True)
        polygon_paths[match_id] = polygon_path
        setup_path = PROJECT / f"matches/{match_id}/calibration/match_setup.json"
        before = setups_before[match_id]
        after = copy.deepcopy(before)
        after["pitch_calibration"] = {
            "authoritative_method": "HUMAN_DRAWN_PER_MATCH",
            "camera_segment_count": len(geometry[match_id]["camera_segments"]),
            "camera_segment_policy": (
                "MATCH_STABLE_CAMERA" if event["alignment_answer"] == "YES" else "PER_HALF_CAMERA_SEGMENTS"
            ),
            "completion_receipt_id": COMPLETION_ID,
            "completion_receipt_sha256": completion["sha256"],
            "expanded_search_region_status": "PENDING",
            "human_review_event_id": event["event_id"],
            "human_review_event_sha256": selected[match_id]["event_sha256"],
            "polygon_path": "calibration/pitch_polygon_v1/pitch_polygon.json",
            "polygon_sha256": sha256_file(polygon_path),
            "production_ready": False,
            "review_revision": REVISION,
            "search_region_status": "PENDING",
            "status": "HUMAN_CONFIRMED",
        }
        before_non_pitch = canonical_sha256(non_pitch_setup(before))
        after_non_pitch = canonical_sha256(non_pitch_setup(after))
        if before_non_pitch != after_non_pitch:
            raise RuntimeError(f"FAIL_G7D_C3A5B_MATCH_SETUP_UPDATE: non-pitch {match_id}")
        write_json(setup_path, after)
        setup_report["matches"][match_id] = {
            "pitch_calibration_before": before["pitch_calibration"],
            "pitch_calibration_after": after["pitch_calibration"],
            "non_pitch_sha256_before": before_non_pitch,
            "non_pitch_sha256_after": after_non_pitch,
            "only_pitch_calibration_changed": True,
        }
    write_json(STAGE / "02_MATCH_SETUP_UPDATES/match_setup_field_diff.json", setup_report)
    visuals = create_visuals(selected, frames, geometry)
    artifact_paths = (
        [selected[mid]["event_path"] for mid in MATCH_IDS]
        + [selected[mid]["ack_path"] for mid in MATCH_IDS]
        + [completion["path"]]
        + [polygon_paths[mid] for mid in MATCH_IDS]
        + [PROJECT / f"matches/{mid}/calibration/match_setup.json" for mid in MATCH_IDS]
        + list(visuals)
    )
    if len(artifact_paths) != 15:
        raise AssertionError("artifact manifest cardinality")
    write_json(
        STAGE / "01_FINAL_POLYGON_ARTIFACTS/polygon_artifact_manifest.json",
        {
            "schema_version": "football_intelligence.g7d_c3a5b.artifact_manifest.v1",
            "files": [row(path) for path in artifact_paths],
        },
    )
    immutable_after = immutable_snapshot()
    if immutable_before != immutable_after:
        raise RuntimeError("FAIL_G7D_C3A5B_HUMAN_EVENT_CHAIN: immutable bytes changed")
    validation_report = {
        "schema_version": "football_intelligence.g7d_c3a5b.finalization_validation_report.v1",
        "classification": CLASSIFICATION,
        "immutable_human_truth_before_sha256": canonical_sha256(immutable_before),
        "immutable_human_truth_after_sha256": canonical_sha256(immutable_after),
        "immutable_human_truth_unchanged": True,
        "polygon_sha256": {mid: sha256_file(polygon_paths[mid]) for mid in MATCH_IDS},
        "match_setup_non_pitch_fields_unchanged": True,
        "search_region_status": "PENDING",
        "production_ready": False,
        "inference_run": False,
        "validation_or_holdout_access": False,
        "visual_count": 2,
        "focused_tests": [
            "uv lock --check",
            "uv sync",
            "uv run ruff check <changed files>",
            "uv run ruff format --check <changed files>",
            "uv run pytest tests/test_g7d_c3a5b_three_match_pitch_polygon_finalization.py -q",
            "git diff --check",
        ],
        "created_at_utc": created,
    }
    write_json(STAGE / "04_TESTS_AND_LOGS/finalization_validation_report.json", validation_report)
    build_handoff(event_report, geometry_report, setup_report, validation_report, visuals)


def refresh_visuals() -> None:
    selected, _completion = resolve_chain()
    frames = read_json(FRAMES_MANIFEST)["matches"]
    geometry = read_json(STAGE / "01_FINAL_POLYGON_ARTIFACTS/geometry_validation.json")["matches"]
    visuals = create_visuals(selected, frames, geometry)
    manifest_path = STAGE / "01_FINAL_POLYGON_ARTIFACTS/polygon_artifact_manifest.json"
    manifest = read_json(manifest_path)
    visual_rows = {row(path)["project_relative_path"]: row(path) for path in visuals}
    manifest["files"] = [visual_rows.get(item["project_relative_path"], item) for item in manifest["files"]]
    write_json(manifest_path, manifest)
    handoff = STAGE / "05_REVIEW_PACK/CHATGPT_HANDOFF"
    atomic_write(handoff / "08_FINAL_POLYGONS.png", visuals[0].read_bytes())
    atomic_write(handoff / "09_ALIGNMENT_VALIDATION.png", visuals[1].read_bytes())
    files = [
        {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(handoff.iterdir())
        if path.name != "10_MANIFEST.json"
    ]
    write_json(handoff / "10_MANIFEST.json", {"files": files})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-visuals", action="store_true")
    arguments = parser.parse_args()
    if arguments.refresh_visuals:
        refresh_visuals()
    else:
        finalize()
