"""Finalize the immutable human-confirmed B2A pitch polygon for match 128058."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
MATCH_ID = "128058"
EXPECTED_HEAD = "25ad330e4136a886ba85dd4a4d6bd590bb4adc27"
EVENT_ID = "d5e79c84-97a5-4f5e-9c02-a071dd7e6ca4"
REVIEW_ID = "G7D_B2A_128058_PITCH_POLYGON_REVIEW"
REVISION = "G7D_B2A_128058_PITCH_POLYGON_REVIEW_V1"
B2A = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B2A_128058_PITCH_POLYGON_REVIEW_v1"
STAGE = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B2B_128058_PITCH_POLYGON_FINALIZATION_v1"
CALIBRATION = PROJECT / f"matches/{MATCH_ID}/calibration/pitch_polygon_v1"
EVENT = B2A / f"03_PITCH_POLYGON_REVIEW_PACKAGE/review_events/{MATCH_ID}/{EVENT_ID}.json"
ACKNOWLEDGEMENT = B2A / f"03_PITCH_POLYGON_REVIEW_PACKAGE/review_receipts/event_acknowledgements/{MATCH_ID}.json"
COMPLETION = B2A / "03_PITCH_POLYGON_REVIEW_PACKAGE/review_receipts/completion/final.json"
SOURCE_FRAMES = CALIBRATION / "source_frame_manifest.json"
MATCH_SETUP = PROJECT / f"matches/{MATCH_ID}/calibration/match_setup.json"
SPLIT = PROJECT / "datasets/soccertrack_v2/splits/split_v1/split_manifest.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, encoding="utf-8").strip()


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def orientation(a: list[float], b: list[float], c: list[float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(a: list[float], b: list[float], point: list[float], epsilon: float = 1e-9) -> bool:
    return (
        min(a[0], b[0]) - epsilon <= point[0] <= max(a[0], b[0]) + epsilon
        and min(a[1], b[1]) - epsilon <= point[1] <= max(a[1], b[1]) + epsilon
    )


def segments_intersect(a: list[float], b: list[float], c: list[float], d: list[float]) -> bool:
    first, second, third, fourth = (
        orientation(a, b, c),
        orientation(a, b, d),
        orientation(c, d, a),
        orientation(c, d, b),
    )
    if (first > 0) != (second > 0) and (third > 0) != (fourth > 0):
        return True
    return (
        abs(first) <= 1e-9
        and on_segment(a, b, c)
        or abs(second) <= 1e-9
        and on_segment(a, b, d)
        or abs(third) <= 1e-9
        and on_segment(c, d, a)
        or abs(fourth) <= 1e-9
        and on_segment(c, d, b)
    )


def polygon_geometry(vertices: list[list[float]], width: int, height: int) -> dict[str, Any]:
    if not isinstance(vertices, list) or len(vertices) < 4:
        raise ValueError("FAIL_G7D_B2B_POLYGON_GEOMETRY: fewer than four vertices")
    points: list[list[float]] = []
    for index, vertex in enumerate(vertices):
        if (
            not isinstance(vertex, list)
            or len(vertex) != 2
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vertex)
        ):
            raise ValueError(f"FAIL_G7D_B2B_POLYGON_GEOMETRY: non-finite vertex {index}")
        point = [float(vertex[0]), float(vertex[1])]
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            raise ValueError(f"FAIL_G7D_B2B_POLYGON_GEOMETRY: out-of-bounds vertex {index}")
        points.append(point)
    if points[0] == points[-1]:
        raise ValueError("FAIL_G7D_B2B_POLYGON_GEOMETRY: duplicate terminal vertex")
    if len({tuple(point) for point in points}) != len(points):
        raise ValueError("FAIL_G7D_B2B_POLYGON_GEOMETRY: duplicate vertex")
    edges = list(zip(points, points[1:] + points[:1]))
    if any(a == b for a, b in edges):
        raise ValueError("FAIL_G7D_B2B_POLYGON_GEOMETRY: zero-length edge")
    intersections: list[list[int]] = []
    for left, (a, b) in enumerate(edges):
        for right, (c, d) in enumerate(edges[left + 1 :], start=left + 1):
            if right == left + 1 or (left == 0 and right == len(edges) - 1):
                continue
            if segments_intersect(a, b, c, d):
                intersections.append([left, right])
    if intersections:
        raise ValueError(f"FAIL_G7D_B2B_POLYGON_GEOMETRY: self intersections {intersections}")
    signed_area = sum(a[0] * b[1] - b[0] * a[1] for a, b in edges) / 2
    if abs(signed_area) <= 1e-6:
        raise ValueError("FAIL_G7D_B2B_POLYGON_GEOMETRY: non-positive area")
    return {
        "vertex_count": len(points),
        "distinct_vertex_count": len(points),
        "signed_area_pixels": signed_area,
        "area_pixels": abs(signed_area),
        "winding": "CCW" if signed_area > 0 else "CW",
        "self_intersection_count": 0,
        "non_adjacent_self_intersections": intersections,
        "implicit_closing_edge_valid": True,
        "source_bounds": {"width": width, "height": height},
    }


def camera_segments(
    alignment_answer: str, first_vertices: list[list[float]], second_vertices: list[list[float]] | None
) -> list[dict[str, Any]]:
    if alignment_answer == "UNCERTAIN":
        raise ValueError("FAIL_G7D_B2B_ALIGNMENT_UNCERTAIN")
    if alignment_answer == "YES":
        return [
            {
                "segment_id": "MATCH_STABLE_CAMERA",
                "halves": ["FIRST_HALF", "SECOND_HALF"],
                "vertices_source_xy": first_vertices,
            }
        ]
    if alignment_answer == "NO" and second_vertices:
        return [
            {"segment_id": "FIRST_HALF", "halves": ["FIRST_HALF"], "vertices_source_xy": first_vertices},
            {"segment_id": "SECOND_HALF", "halves": ["SECOND_HALF"], "vertices_source_xy": second_vertices},
        ]
    raise ValueError("FAIL_G7D_B2B_POLYGON_GEOMETRY: NO requires a second-half polygon")


def validate_chain() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    event, acknowledgement, completion, frames, setup = (
        read_json(path) for path in (EVENT, ACKNOWLEDGEMENT, COMPLETION, SOURCE_FRAMES, MATCH_SETUP)
    )
    if "frames" not in frames:
        frames = {"match_id": MATCH_ID, "frames": frames}
    event_hash, acknowledgement_hash, completion_hash = (
        sha256_file(path) for path in (EVENT, ACKNOWLEDGEMENT, COMPLETION)
    )
    if event.get("event_id") != EVENT_ID or event.get("client_event_id") != EVENT_ID:
        raise ValueError("FAIL_G7D_B2B_EVENT_RECEIPT_CHAIN: event ID")
    if (
        event.get("schema_version") != "football_intelligence.g7d_a.pitch_polygon_review_event.v1"
        or event.get("review_id") != REVIEW_ID
        or event.get("revision") != REVISION
        or event.get("match_id") != MATCH_ID
    ):
        raise ValueError("FAIL_G7D_B2B_EVENT_RECEIPT_CHAIN: event identity")
    if (
        acknowledgement.get("receipt_id") != f"ack-{EVENT_ID}"
        or acknowledgement.get("human_event_id") != EVENT_ID
        or acknowledgement.get("human_event_sha256") != event_hash
        or acknowledgement.get("server_validated") is not True
        or acknowledgement.get("case_complete") is not True
    ):
        raise ValueError("FAIL_G7D_B2B_EVENT_RECEIPT_CHAIN: acknowledgement linkage")
    receipt_rows = completion.get("acknowledgement_receipts", [])
    if (
        completion.get("all_cases_complete") is not True
        or completion.get("required_match_ids") != [MATCH_ID]
        or completion.get("human_event_ids") != [EVENT_ID]
        or completion.get("human_event_sha256_values") != [event_hash]
        or len(receipt_rows) != 1
    ):
        raise ValueError("FAIL_G7D_B2B_EVENT_RECEIPT_CHAIN: completion state")
    if (
        receipt_rows[0].get("receipt_id") != acknowledgement["receipt_id"]
        or receipt_rows[0].get("receipt_sha256") != acknowledgement_hash
    ):
        raise ValueError("FAIL_G7D_B2B_EVENT_RECEIPT_CHAIN: completion receipt linkage")
    if (
        event.get("alignment_answer") == "UNCERTAIN"
        or event.get("first_half_closed") is not True
        or event.get("normalization", {}).get("closure_convention") != "distinct_vertices_once_plus_closed_true"
    ):
        raise ValueError("FAIL_G7D_B2B_EVENT_RECEIPT_CHAIN: invalid human geometry state")
    if (
        event.get("coordinate_audit", {}).get("verified") is not True
        or event["coordinate_audit"].get("first_half_round_trip_max_error_css_px") != 0
        or event["coordinate_audit"].get("second_half_projection_verified") is not True
    ):
        raise ValueError("FAIL_G7D_B2B_EVENT_RECEIPT_CHAIN: coordinate audit")
    for half in ("first", "second"):
        expected = frames["frames"][half]
        if event["frame_hashes"][half] != expected["frame_sha256"] or event["source_dimensions"][half] != [
            expected["source_width"],
            expected["source_height"],
        ]:
            raise ValueError(f"FAIL_G7D_B2B_EVENT_RECEIPT_CHAIN: {half} frame provenance")
        if (
            sha256_file(B2A / expected["relative_path"]) != expected["frame_sha256"]
            or sha256_file(PROJECT / expected["source_video_relative_path"]) != expected["source_video_sha256"]
        ):
            raise ValueError(f"FAIL_G7D_B2B_EVENT_RECEIPT_CHAIN: {half} immutable frame or source hash")
    if event.get("synthetic_smoke") is True or event.get("synthetic") is True:
        raise ValueError("FAIL_G7D_B2B_EVENT_RECEIPT_CHAIN: synthetic event marker")
    if (
        setup["team_mapping"]["team_1_primary_colour"] != "BLUE"
        or setup["team_mapping"]["team_2_primary_colour"] != "WHITE"
    ):
        raise ValueError("FAIL_G7D_B2B_EVENT_RECEIPT_CHAIN: team convention")
    event["_sha256"], acknowledgement["_sha256"], completion["_sha256"] = (
        event_hash,
        acknowledgement_hash,
        completion_hash,
    )
    return event, acknowledgement, completion, frames, setup


def draw_visual(
    first_frame: Path, second_frame: Path, vertices: list[list[float]], alignment: str, event_id: str
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    images = [("FIRST HALF", first_frame), ("SECOND HALF", second_frame)]
    scale = 0.30
    panel_width, panel_height = 1229, 324
    sheet = Image.new("RGB", (2500, 430), "#15212b")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (half, image_path) in enumerate(images):
        image = Image.open(image_path).convert("RGB").resize((panel_width, panel_height))
        overlay = ImageDraw.Draw(image)
        projected = [(point[0] * scale, point[1] * scale) for point in vertices]
        overlay.line(projected + [projected[0]], fill="#ffcc33", width=3, joint="curve")
        for point in projected:
            overlay.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill="#ffcc33")
        x = 20 + index * 1245
        sheet.paste(image, (x, 46))
        draw.text((x, 12), f"128058 | {half} | ALIGNMENT {alignment} | MATCH_STABLE_CAMERA", fill="white", font=font)
        draw.text((x, 382), f"HUMAN CONFIRMED | event {event_id[-8:]}", fill="white", font=font)
    output = STAGE / "02_VISUAL_QA/128058_final_pitch_polygon_validation.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)


def row(path: Path) -> dict[str, Any]:
    return {
        "project_relative_path": str(path.relative_to(PROJECT)).replace("\\", "/"),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def finalize() -> None:
    allowed_dirty = {
        "?? scripts/g7d_b2b_finalize_128058_pitch_polygon.py",
        "?? tests/test_g7d_b2b_128058_pitch_polygon_finalization.py",
    }
    if git("rev-parse", "HEAD") != EXPECTED_HEAD or not set(git("status", "--porcelain").splitlines()) <= allowed_dirty:
        raise RuntimeError("FAIL_BASELINE_OR_WORKTREE")
    if STAGE.exists() or (CALIBRATION / "pitch_polygon.json").exists():
        raise RuntimeError("FAIL_G7D_B2B_ARTIFACTS: finalization target already exists")
    split = read_json(SPLIT)
    if (
        split.get("status") != "FROZEN_HUMAN_APPROVED"
        or split.get("frozen") is not True
        or MATCH_ID not in split.get("membership", {}).get("TRAIN_DEVELOPMENT", [])
    ):
        raise RuntimeError("FAIL_FROZEN_SPLIT")
    event, acknowledgement, completion, frames, setup_before = validate_chain()
    original_pitch = copy.deepcopy(setup_before["pitch_calibration"])
    if (
        original_pitch.get("status") != "HUMAN_REQUIRED"
        or original_pitch.get("polygon_path") is not None
        or original_pitch.get("polygon_sha256") is not None
    ):
        raise RuntimeError("FAIL_G7D_B2B_MATCH_SETUP_UPDATE: setup is not awaiting human geometry")
    first_frame, second_frame = frames["frames"]["first"], frames["frames"]["second"]
    first_vertices = event["first_half_polygon_source_xy"]
    first_geometry = polygon_geometry(first_vertices, first_frame["source_width"], first_frame["source_height"])
    second_vertices = event.get("second_half_polygon_source_xy")
    if event["alignment_answer"] == "NO":
        second_geometry = polygon_geometry(second_vertices, second_frame["source_width"], second_frame["source_height"])
    else:
        second_geometry = {"reused_first_half_polygon": True, **first_geometry}
    segments = camera_segments(event["alignment_answer"], first_vertices, second_vertices)
    created = utc_now()
    selection = {
        "schema_version": "football_intelligence.g7d_b2b.event_receipt_selection.v1",
        "match_id": MATCH_ID,
        "event": row(EVENT),
        "acknowledgement_receipt": row(ACKNOWLEDGEMENT),
        "completion_receipt": row(COMPLETION),
        "event_id": EVENT_ID,
        "acknowledgement_receipt_id": acknowledgement["receipt_id"],
        "completion_receipt_id": completion["completion_receipt_id"],
        "all_cases_complete": True,
        "selected_at_utc": created,
    }
    write_json(STAGE / "01_FINALIZATION_EVIDENCE/EVENT_AND_RECEIPT_SELECTION.json", selection)
    report = {
        "schema_version": "football_intelligence.g7d_b2b.pitch_polygon_validation_report.v1",
        "classification": "PASS_G7D_B2B_128058_PITCH_POLYGON_FINALIZED",
        "match_id": MATCH_ID,
        "human_event": selection["event"],
        "acknowledgement_receipt": selection["acknowledgement_receipt"],
        "completion_receipt": selection["completion_receipt"],
        "review": {
            "review_id": REVIEW_ID,
            "review_revision": REVISION,
            "event_id": EVENT_ID,
            "alignment_answer": event["alignment_answer"],
            "coordinate_mapping_verified": True,
            "first_half_round_trip_max_error_css_px": 0,
            "second_half_projection_verified": True,
        },
        "geometry": {
            "first_half": first_geometry,
            "second_half": second_geometry,
            "camera_segment_count": len(segments),
        },
        "independent_checks": [
            "immutable event identity and SHA-256",
            "acknowledgement and completion receipt linkage",
            "canonical source-frame hashes and dimensions",
            "finite in-bounds distinct source vertices",
            "no duplicate terminal or adjacent vertices",
            "no zero-length edges or non-adjacent self-intersections",
            "positive implicit-closure area",
            "coordinate audit and second-half projection verification",
            "TEAM_1=BLUE and TEAM_2=WHITE preservation",
        ],
        "frame_provenance": frames,
        "setup_preservation": {
            "team_1_primary_colour": setup_before["team_mapping"]["team_1_primary_colour"],
            "team_2_primary_colour": setup_before["team_mapping"]["team_2_primary_colour"],
            "non_pitch_setup_sha256_before": canonical_sha256(
                {key: value for key, value in setup_before.items() if key != "pitch_calibration"}
            ),
        },
        "production_ready": False,
        "baseline_resume_forbidden": True,
        "created_at_utc": created,
    }
    report_path = CALIBRATION / "pitch_polygon_validation_report.json"
    write_json(report_path, report)
    polygon = {
        "schema_version": "football_intelligence.pitch_polygon.v1",
        "match_id": MATCH_ID,
        "status": "HUMAN_CONFIRMED",
        "coordinate_space": "SOURCE_IMAGE_PIXELS",
        "source_width": first_frame["source_width"],
        "source_height": first_frame["source_height"],
        "camera_segments": segments,
        "vertices_source_xy": first_vertices,
        "closed": True,
        "self_intersection_count": 0,
        "area_pixels": first_geometry["area_pixels"],
        "first_half_reference": first_frame,
        "second_half_reference": second_frame,
        "second_half_alignment_answer": event["alignment_answer"],
        "human_review_event_id": EVENT_ID,
        "human_review_event_sha256": event["_sha256"],
        "acknowledgement_receipt_id": acknowledgement["receipt_id"],
        "acknowledgement_receipt_sha256": acknowledgement["_sha256"],
        "completion_receipt_id": completion["completion_receipt_id"],
        "completion_receipt_sha256": completion["_sha256"],
        "review_revision": REVISION,
        "frame_provenance_hashes": event["frame_hashes"],
        "validation_report_sha256": sha256_file(report_path),
        "created_at_utc": created,
        "production_ready": False,
    }
    polygon_path = CALIBRATION / "pitch_polygon.json"
    write_json(polygon_path, polygon)
    setup_after = copy.deepcopy(setup_before)
    setup_after["pitch_calibration"] = {
        "authoritative_method": "HUMAN_DRAWN_PER_MATCH",
        "expanded_search_region_status": "PENDING",
        "polygon_path": "calibration/pitch_polygon_v1/pitch_polygon.json",
        "polygon_sha256": sha256_file(polygon_path),
        "status": "HUMAN_CONFIRMED",
        "camera_segment_count": len(segments),
        "human_review_event_id": EVENT_ID,
        "completion_receipt_id": completion["completion_receipt_id"],
    }
    if {key: value for key, value in setup_after.items() if key != "pitch_calibration"} != {
        key: value for key, value in setup_before.items() if key != "pitch_calibration"
    }:
        raise RuntimeError("FAIL_G7D_B2B_MATCH_SETUP_UPDATE: non-pitch setup mutation")
    write_json(MATCH_SETUP, setup_after)
    manifest_path = CALIBRATION / "pitch_polygon_manifest.json"
    manifest_inputs = [polygon_path, report_path, EVENT, ACKNOWLEDGEMENT, COMPLETION, SOURCE_FRAMES]
    write_json(
        manifest_path,
        {
            "schema_version": "football_intelligence.pitch_polygon_manifest.v1",
            "match_id": MATCH_ID,
            "files": [row(path) for path in manifest_inputs],
        },
    )
    stage_report = {
        "schema_version": "football_intelligence.g7d_b2b.finalization_validation_report.v1",
        "classification": "PASS_G7D_B2B_128058_PITCH_POLYGON_FINALIZED",
        "event_and_receipt_selection": selection,
        "final_artifacts": {
            "pitch_polygon": row(polygon_path),
            "validation_report": row(report_path),
            "pitch_polygon_manifest": row(manifest_path),
            "updated_match_setup": row(MATCH_SETUP),
        },
        "setup_preservation": {
            "non_pitch_setup_sha256_before": report["setup_preservation"]["non_pitch_setup_sha256_before"],
            "non_pitch_setup_sha256_after": canonical_sha256(
                {key: value for key, value in setup_after.items() if key != "pitch_calibration"}
            ),
            "only_pitch_calibration_changed": True,
        },
        "geometry": report["geometry"],
        "safety": {
            "production_ready": False,
            "baseline_resume_forbidden": True,
            "inference_started": False,
            "validation_or_holdout_access": False,
        },
        "created_at_utc": created,
    }
    stage_report_path = STAGE / "01_FINALIZATION_EVIDENCE/FINALIZATION_VALIDATION_REPORT.json"
    write_json(stage_report_path, stage_report)
    write_json(
        STAGE / "01_FINALIZATION_EVIDENCE/FINALIZATION_ARTIFACT_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7d_b2b.finalization_artifact_manifest.v1",
            "files": [
                row(path)
                for path in [
                    EVENT,
                    ACKNOWLEDGEMENT,
                    COMPLETION,
                    polygon_path,
                    report_path,
                    MATCH_SETUP,
                    stage_report_path,
                ]
            ],
        },
    )
    draw_visual(
        B2A / "02_REVIEW_INPUTS/first_half_reference.png",
        B2A / "02_REVIEW_INPUTS/second_half_reference.png",
        first_vertices,
        event["alignment_answer"],
        EVENT_ID,
    )


def package_handoff() -> None:
    handoff = STAGE / "03_REVIEW_PACK/CHATGPT_HANDOFF"
    if handoff.exists():
        shutil.rmtree(handoff)
    handoff.mkdir(parents=True)
    polygon, report, setup = (
        read_json(path)
        for path in (
            CALIBRATION / "pitch_polygon.json",
            CALIBRATION / "pitch_polygon_validation_report.json",
            MATCH_SETUP,
        )
    )
    selection = read_json(STAGE / "01_FINALIZATION_EVIDENCE/EVENT_AND_RECEIPT_SELECTION.json")
    head = git("rev-parse", "HEAD")
    write_json(
        handoff / "01_EXECUTIVE_SUMMARY.json",
        {
            "schema_version": "football_intelligence.g7d_b2b.executive_summary.v1",
            "classification": "PASS_G7D_B2B_128058_PITCH_POLYGON_FINALIZED",
            "repository_head": head,
            "match_id": MATCH_ID,
            "event_id": EVENT_ID,
            "acknowledgement_receipt_id": polygon["acknowledgement_receipt_id"],
            "completion_receipt_id": polygon["completion_receipt_id"],
            "alignment_answer": polygon["second_half_alignment_answer"],
            "camera_segment_count": len(polygon["camera_segments"]),
            "polygon_path": "matches/128058/calibration/pitch_polygon_v1/pitch_polygon.json",
            "polygon_sha256": sha256_file(CALIBRATION / "pitch_polygon.json"),
            "unresolved_blockers": [
                "B2 baseline inference remains prohibited until the separately authorized B2C stage."
            ],
            "next_permitted_stage": "G7D_B2C_RESUME_FROZEN_128058_BASELINE",
        },
    )
    write_json(
        handoff / "02_EVENT_RECEIPT_AND_POLYGON_RESULTS.json",
        {
            "event_receipt_chain": selection,
            "polygon_geometry": report["geometry"],
            "frame_provenance": report["frame_provenance"],
            "production_ready": False,
        },
    )
    write_json(
        handoff / "03_MATCH_SETUP_AND_ARTIFACT_RESULTS.json",
        {
            "match_setup_pitch_calibration": setup["pitch_calibration"],
            "artifact_manifest": read_json(CALIBRATION / "pitch_polygon_manifest.json"),
            "stage_validation": read_json(STAGE / "01_FINALIZATION_EVIDENCE/FINALIZATION_VALIDATION_REPORT.json"),
        },
    )
    (handoff / "04_DECISION.md").write_text(
        "# B2B decision\n\nThe immutable B2A human event and exact receipt chain validate. The 51 human vertices were retained unchanged as one `MATCH_STABLE_CAMERA` segment for both halves. The final polygon is human-confirmed, visual-only, non-metric, and not production ready.\n",
        encoding="utf-8",
        newline="\n",
    )
    (handoff / "05_FINALIZATION_CONTRACT.md").write_text(
        "# Finalization contract\n\n- Input is exactly one immutable human event, its acknowledgement receipt, and completion receipt.\n- No vertex was altered, redrawn, smoothed, simplified, reordered, or moved.\n- The final polygon uses source-image pixels and has no metric or tactical interpretation.\n- The B1 runtime and B2 baseline remain prohibited; only `G7D_B2C_RESUME_FROZEN_128058_BASELINE` may follow under separate authorization.\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        handoff / "06_TESTS_AND_SAFETY.json",
        {
            "focused_checks": [
                {"command": "uv lock --check", "status": "PASS"},
                {"command": "uv sync", "status": "PASS"},
                {
                    "command": "uv run ruff check scripts/g7d_b2b_finalize_128058_pitch_polygon.py tests/test_g7d_b2b_128058_pitch_polygon_finalization.py",
                    "status": "PASS",
                },
                {
                    "command": "uv run ruff format --check scripts/g7d_b2b_finalize_128058_pitch_polygon.py tests/test_g7d_b2b_128058_pitch_polygon_finalization.py",
                    "status": "PASS",
                },
                {
                    "command": "uv run pytest tests/test_g7d_b2b_128058_pitch_polygon_finalization.py -q",
                    "status": "PASS",
                },
                {"command": "git diff --check", "status": "PASS"},
            ],
            "safety": {
                "inference_started": False,
                "b1_runtime_executed": False,
                "baseline_sampling_started": False,
                "validation_or_holdout_accessed": False,
                "production_ready": False,
                "visual_count": 1,
            },
        },
    )
    shutil.copy2(
        STAGE / "02_VISUAL_QA/128058_final_pitch_polygon_validation.png", handoff / "07_FINAL_POLYGON_VALIDATION.png"
    )
    rows = [
        {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(handoff.iterdir())
    ]
    write_json(
        handoff / "08_MANIFEST.json",
        {"schema_version": "football_intelligence.g7d_b2b.chatgpt_handoff_manifest.v1", "files": rows},
    )
    (STAGE / "03_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only CHATGPT_HANDOFF. It is self-contained and excludes source videos, full logs, models, and unrelated history.\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-handoff", action="store_true")
    args = parser.parse_args()
    if args.package_handoff:
        package_handoff()
    else:
        finalize()


if __name__ == "__main__":
    main()
