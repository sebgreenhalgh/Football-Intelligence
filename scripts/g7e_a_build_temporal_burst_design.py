from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from football_intelligence.temporal_burst_selection import (
    CLASS_PRIORITY,
    MATCHES,
    OFFSETS_SECONDS,
    ONTOLOGY,
    QUOTAS,
    frame_indices_for_centre,
    overlap_count,
    slot_plan,
    validate_burst_records,
    validate_ontology,
)

# Fixed stage paths and contract strings remain directly inspectable.
# ruff: noqa: E501

ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = ROOT / "SoccerTrack-v2"
P6 = ROOT / "experiments/football_observation_reasoner/part 6"
P7 = ROOT / "experiments/football_observation_reasoner/part 7"
STAGE = P7 / "G7E_A_TARGETED_TEMPORAL_BURST_SELECTION_AND_ANNOTATION_DESIGN_v1"
PACK = P7 / "G7E_A_Targeted_Temporal_Burst_Selection_And_Annotation_Design_Codex_Pack"
C3A3 = P7 / "G7D_C3A3_ACTIVE_SANDBOX_PITCH_GATE_INTEGRATION_v1"
C3A5C = P7 / "G7D_C3A5C_ADDITIONAL_COVERAGE_REPLAY_AND_REVIEW_v1"
C3A5D = P7 / "G7D_C3A5D_ADDITIONAL_COVERAGE_FINALIZATION_AND_DEFAULT_DECISION_v1"
C3A6 = P7 / "G7D_C3A6_DEVELOPMENT_ONLY_DEFAULT_ACTIVATION_v1"
C3B = P7 / "G7D_C3B_NESTED_CANDIDATE_SUPPRESSION_SANDBOX_REVIEW_v1"
C3B1 = P7 / "G7D_C3B1_NESTED_REVIEW_FINALIZATION_AND_SAFE_RULE_SELECTION_v1"
C3B2 = P7 / "G7D_C3B2_PERSPECTIVE_NORMALIZED_CANDIDATE_SCALE_SANDBOX_v1"
C2 = P6 / "G7D_C2_R1_RESUME_VISUAL_TRANSFER_DIAGNOSIS_FINALIZATION_v1"
B2C = P6 / "G7D_B2C_RESUME_FROZEN_128058_BASELINE_v1"
B3 = P6 / "G7D_B3_FROZEN_CROSS_MATCH_REPLAY_v1"

EXPECTED_HEAD = "f572fe8fb2ee819548eec0eb09bc57292b56aa81"
PROTOCOL_ID = "G7E_A_BURST_LOCAL_TEMPORAL_OBSERVATION_PROTOCOL_V1"
DECISION = "PASS_G7E_A_TEMPORAL_BURST_SELECTION_AND_ANNOTATION_DESIGN_FROZEN"
FPS = Decimal("25")


@dataclass(frozen=True)
class SourceVideo:
    match_id: str
    half: str
    relative_path: str
    path: Path
    sha256: str
    byte_size: int
    fps: Decimal
    frame_count: int
    width: int
    height: int

    @property
    def duration_seconds(self) -> Decimal:
        return Decimal(self.frame_count) / self.fps


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return path


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "project_relative_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "byte_size": path.stat().st_size,
        "sha256": sha256(path),
    }


def source_artifact(path: Path) -> dict[str, Any]:
    return {
        "project_relative_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "byte_size": path.stat().st_size,
        "sha256": sha256(path),
    }


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def canonical_sources() -> dict[tuple[str, str], SourceVideo]:
    sources: dict[tuple[str, str], SourceVideo] = {}
    for match_id in MATCHES:
        setup = read_json(ROOT / f"matches/{match_id}/calibration/match_setup.json")
        polygon_path = ROOT / f"matches/{match_id}/calibration/pitch_polygon_v1/pitch_polygon.json"
        polygon = read_json(polygon_path)
        if (
            setup["dataset_split"]["proposed_assignment"] != "TRAIN_DEVELOPMENT"
            or not setup["dataset_split"]["frozen"]
            or setup["pitch_calibration"]["status"] != "HUMAN_CONFIRMED"
            or setup["pitch_calibration"]["polygon_sha256"] != sha256(polygon_path)
            or polygon["status"] != "HUMAN_CONFIRMED"
            or polygon["second_half_alignment_answer"] != "YES"
            or len(polygon["camera_segments"]) != 1
            or polygon["camera_segments"][0]["segment_id"] != "MATCH_STABLE_CAMERA"
        ):
            raise RuntimeError(f"FAIL_G7E_A_INPUT_PROVENANCE: match setup/polygon {match_id}")
        source_manifest = read_json(ROOT / f"matches/{match_id}/manifests/source_file_manifest.json")
        for half, key in (("FIRST_HALF", "first_half_reference"), ("SECOND_HALF", "second_half_reference")):
            reference = polygon[key]
            relative_path = reference.get("source_video_relative_path", reference.get("relative_path"))
            expected_hash = reference.get("source_video_sha256", reference.get("source_sha256"))
            entries = [
                row
                for row in source_manifest["files"]
                if row["relative_path"] == relative_path and row["sha256"] == expected_hash
            ]
            if len(entries) != 1 or "panorama" not in relative_path.lower() or "review" in relative_path.lower():
                raise RuntimeError(f"FAIL_G7E_A_INPUT_PROVENANCE: canonical source {match_id}:{half}")
            entry = entries[0]
            path = ROOT / relative_path
            if not path.is_file() or path.stat().st_size != int(entry["byte_size"]):
                raise RuntimeError(f"FAIL_G7E_A_INPUT_PROVENANCE: source bytes {match_id}:{half}")
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError(f"FAIL_G7E_A_FRAME_PROVENANCE: source decode {match_id}:{half}")
            try:
                fps = Decimal(str(capture.get(cv2.CAP_PROP_FPS)))
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            finally:
                capture.release()
            if fps != FPS or frame_count <= 0 or (width, height) != (polygon["source_width"], polygon["source_height"]):
                raise RuntimeError(f"FAIL_G7E_A_FRAME_PROVENANCE: source metadata {match_id}:{half}")
            sources[(match_id, half)] = SourceVideo(
                match_id=match_id,
                half=half,
                relative_path=relative_path,
                path=path,
                sha256=expected_hash,
                byte_size=int(entry["byte_size"]),
                fps=fps,
                frame_count=frame_count,
                width=width,
                height=height,
            )
    if len(sources) != 12:
        raise RuntimeError("FAIL_G7E_A_INPUT_PROVENANCE: twelve canonical half videos")
    return sources


def normalize_frame(row: Mapping[str, Any], sources: Mapping[tuple[str, str], SourceVideo]) -> dict[str, Any]:
    match_id = str(row["match_id"])
    half = str(row["half"])
    source = sources[(match_id, half)]
    if row["source_video_relative_path"] != source.relative_path or row["source_video_sha256"] != source.sha256:
        raise RuntimeError(f"FAIL_G7E_A_INPUT_PROVENANCE: frozen frame source {row['frame_id']}")
    return {
        "anchor_frame_id": str(row["frame_id"]),
        "frame_sha256": str(row["frame_sha256"]),
        "match_id": match_id,
        "half": half,
        "centre_frame_index": int(row["frame_index_zero_based"]),
        "centre_timestamp_seconds": float(Decimal(int(row["frame_index_zero_based"])) / source.fps),
        "source_video_relative_path": source.relative_path,
        "source_video_sha256": source.sha256,
        "source_width": source.width,
        "source_height": source.height,
        "source_frozen_frame_id": str(row["frame_id"]),
        "source_frozen_frame_sha256": str(row["frame_sha256"]),
        "companion": False,
        "companion_anchor_frame_id": None,
        "companion_delta_seconds": None,
    }


def load_frozen_frames(sources: Mapping[tuple[str, str], SourceVideo]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(
        normalize_frame(row, sources)
        for row in read_json(B2C / "02_BASELINE_INPUTS/ordered_sampling_manifest.json")["frames"]
    )
    for match_id in ("117092", "118575"):
        rows.extend(
            normalize_frame(row, sources)
            for row in read_json(B3 / f"02_REPLAY_INPUTS/{match_id}/ordered_sampling_manifest.json")["frames"]
        )
    rows.extend(
        normalize_frame(row, sources)
        for row in read_json(C3A5C / "01_FRAME_REPLAY/frame_and_candidate_manifest.json")["frames"]
    )
    if len(rows) != 144 or Counter(row["match_id"] for row in rows) != {
        "117092": 32,
        "117093": 16,
        "118575": 32,
        "118576": 16,
        "118577": 16,
        "128058": 32,
    }:
        raise RuntimeError("FAIL_G7E_A_INPUT_PROVENANCE: 144 frozen frames")
    if len({row["frame_sha256"] for row in rows}) != 144:
        raise RuntimeError("FAIL_G7E_A_INPUT_PROVENANCE: duplicate frozen frame hashes")
    return rows


def load_retained_candidates(frames: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_sha = {row["frame_sha256"]: row for row in frames}
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    active_path = C3A3 / "04_ACTIVE_OUTPUTS/active_candidate_records.jsonl"
    with active_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            frame = by_sha.get(row["frame_sha256"])
            if frame is None or row["gate_decision"] == "SUPPRESS_SANDBOX":
                raise RuntimeError("FAIL_G7E_A_INPUT_PROVENANCE: active candidate/frame closure")
            candidates[row["frame_sha256"]].append(
                {
                    "candidate_id": row["candidate_local_id"],
                    "source_box_xyxy": row["source_box_xyxy"],
                    "footpoint_xy": row["approximate_footpoint_xy"],
                    "gate_decision": row["gate_decision"],
                    "score": None,
                }
            )
    c3a5c = read_json(C3A5C / "01_FRAME_REPLAY/frame_and_candidate_manifest.json")
    for row in c3a5c["candidates"]:
        if row["gate_decision"] == "SUPPRESS_SANDBOX":
            continue
        frame = by_sha.get(row["frame_sha256"])
        if frame is None:
            raise RuntimeError("FAIL_G7E_A_INPUT_PROVENANCE: additional candidate/frame closure")
        candidates[row["frame_sha256"]].append(
            {
                "candidate_id": row["candidate_local_id"],
                "source_box_xyxy": row["source_box_xyxy"],
                "footpoint_xy": row["approximate_footpoint_xy"],
                "gate_decision": row["gate_decision"],
                "score": row.get("score"),
            }
        )
    if sum(len(rows) for rows in candidates.values()) != 6509 or len(candidates) != 144:
        raise RuntimeError("FAIL_G7E_A_INPUT_PROVENANCE: 6509 retained candidates")
    predictions = {
        row["candidate_id"]: row
        for row in read_jsonl(C3B2 / "03_EXPECTED_HEIGHT_SURFACES/candidate_scale_predictions.jsonl")
    }
    if len(predictions) != 6509:
        raise RuntimeError("FAIL_G7E_A_INPUT_PROVENANCE: H2 prediction closure")
    for frame_sha, rows in candidates.items():
        frame = by_sha[frame_sha]
        for row in rows:
            prediction = predictions.get(row["candidate_id"])
            if prediction is None or prediction["match_id"] != frame["match_id"]:
                raise RuntimeError("FAIL_G7E_A_INPUT_PROVENANCE: H2 candidate join")
            row["h2_expected_height_px"] = prediction.get("expected_height_px")
            row["h2_prediction_available"] = bool(prediction.get("prediction_available"))
            row["h2_support_count"] = int(prediction.get("support_count", 0))
            if row["h2_prediction_available"] and row["h2_expected_height_px"] is not None:
                expected_ratio = float(row["h2_expected_height_px"]) / frame["source_height"]
                row["perspective_band"] = "FAR" if expected_ratio <= 0.045 else "NEAR_MIDDLE"
            else:
                row["perspective_band"] = (
                    "FAR" if float(row["footpoint_xy"][1]) / frame["source_height"] <= 0.45 else "NEAR_MIDDLE"
                )
    return candidates


def add_evidence(
    frames: list[dict[str, Any]], candidates: Mapping[str, list[dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    by_sha = {row["frame_sha256"]: row for row in frames}
    for frame in frames:
        frame["evidence"] = {
            "pair_count": 0,
            "nested_safe_count": 0,
            "nested_must_protect_count": 0,
            "nested_ambiguous_count": 0,
            "missed_mark_count": 0,
            "human_occlusion_or_merge_count": 0,
            "human_official_count": 0,
            "human_goalkeeper_count": 0,
            "human_goalkeeper_endline_positive": 0,
            "human_official_touchline_positive": 0,
            "scene_crowding_high": 0,
            "scene_overlap_high": 0,
            "stable_control_seed": 0,
            "b3_primary_quotas": [],
            "scene_ids": [],
            "mark_ids": [],
            "pair_case_ids": [],
            "pair_inner_candidate_ids": [],
            "human_official_candidate_ids": [],
            "source_event_ids": [],
        }

    pair_geometry_path = C3B / "02_NESTED_PAIR_GEOMETRY/nested_pair_geometry.jsonl"
    with pair_geometry_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            frame = by_sha.get(row["frame_sha256"])
            if frame is None:
                raise RuntimeError("FAIL_G7E_A_INPUT_PROVENANCE: pair geometry frame join")
            frame["evidence"]["pair_count"] += 1

    pair_selection = read_json(C3B / "05_REVIEW_SELECTION/review_pair_selection.json")["cases"]
    pair_truth = read_jsonl(C3B1 / "02_NORMALIZED_PAIR_TRUTH/pair_human_labels.jsonl")
    if len(pair_selection) != 48 or len(pair_truth) != 48:
        raise RuntimeError("FAIL_G7E_A_INPUT_PROVENANCE: pair review closure")
    for truth in pair_truth:
        index = int(truth["case_id"].split("_")[-1]) - 1
        selection = pair_selection[index]
        frame = by_sha.get(selection["frame_sha256"])
        if frame is None or frame["match_id"] != truth["match_id"]:
            raise RuntimeError("FAIL_G7E_A_INPUT_PROVENANCE: pair truth join")
        key = {
            "HUMAN_SAFE_TO_SUPPRESS_INNER": "nested_safe_count",
            "HUMAN_MUST_PROTECT_INNER": "nested_must_protect_count",
            "AMBIGUOUS": "nested_ambiguous_count",
        }[truth["safe_suppression_truth"]]
        frame["evidence"][key] += 1
        frame["evidence"]["pair_case_ids"].append(truth["case_id"])
        frame["evidence"]["pair_inner_candidate_ids"].append(selection["inner_candidate_id"])
        frame["evidence"]["source_event_ids"].append(truth["event_id"])

    c2_scene_rows = read_jsonl(C2 / "01_HUMAN_REVIEW_CLOSURE/scene_human_labels.jsonl")
    c3a5d_scene_rows = read_jsonl(C3A5D / "01_HUMAN_REVIEW_CLOSURE/scene_human_labels.jsonl")
    if len(c2_scene_rows) != 24 or len(c3a5d_scene_rows) != 12:
        raise RuntimeError("FAIL_G7E_A_INPUT_PROVENANCE: 36 scene reviews")
    scene_to_sha: dict[str, str] = {}
    for row in c2_scene_rows:
        frame = by_sha.get(row["frame_sha256"])
        if frame is None:
            raise RuntimeError("FAIL_G7E_A_INPUT_PROVENANCE: C2 scene join")
        scene_to_sha[row["scene_id"]] = row["frame_sha256"]
        evidence = frame["evidence"]
        evidence["scene_ids"].append(row["scene_id"])
        evidence["source_event_ids"].append(row["event_id"])
        review = row["canonical_review"]
        evidence["human_occlusion_or_merge_count"] += int(review["occlusion_burden"] in {"MODERATE", "HIGH"})
        evidence["scene_overlap_high"] += int(review["duplicate_or_overlap_burden"] == "HIGH")
        evidence["scene_crowding_high"] += int("CROWD" in row["scene_category"] or "FAR" in row["scene_category"])
    for row in c3a5d_scene_rows:
        frame = by_sha.get(row["frame_sha256"])
        if frame is None:
            raise RuntimeError("FAIL_G7E_A_INPUT_PROVENANCE: C3A5D scene join")
        scene_to_sha[row["scene_id"]] = row["frame_sha256"]
        evidence = frame["evidence"]
        evidence["scene_ids"].append(row["scene_id"])
        evidence["source_event_ids"].append(row["event_id"])
        review = row["canonical_review"]
        evidence["human_goalkeeper_endline_positive"] += int(review["goalkeeper_endline"] == "YES")
        evidence["human_official_touchline_positive"] += int(review["official_touchline"] == "YES")
        evidence["scene_crowding_high"] += int(review["crowding_overlap"] == "HIGH")
        evidence["scene_overlap_high"] += int(review["crowding_overlap"] == "HIGH")

    marks: list[dict[str, Any]] = []
    for row in read_jsonl(C2 / "01_HUMAN_REVIEW_CLOSURE/missed_person_marks.jsonl"):
        row = dict(row)
        row["frame_sha256"] = scene_to_sha[row["scene_id"]]
        marks.append(row)
    marks.extend(read_jsonl(C3A5D / "01_HUMAN_REVIEW_CLOSURE/missed_person_marks.jsonl"))
    if len(marks) != 25:
        raise RuntimeError("FAIL_G7E_A_INPUT_PROVENANCE: 25 missed marks")
    for row in marks:
        frame = by_sha.get(row["frame_sha256"])
        if frame is None:
            raise RuntimeError("FAIL_G7E_A_INPUT_PROVENANCE: missed mark join")
        frame["evidence"]["missed_mark_count"] += 1
        frame["evidence"]["mark_ids"].append(row["mark_id"])

    label_paths = (
        C2 / "01_HUMAN_REVIEW_CLOSURE/candidate_human_labels.jsonl",
        C3A5D / "01_HUMAN_REVIEW_CLOSURE/candidate_human_labels.jsonl",
    )
    label_count = 0
    for path in label_paths:
        for row in read_jsonl(path):
            label_count += 1
            frame = by_sha.get(row["frame_sha256"])
            if frame is None:
                raise RuntimeError("FAIL_G7E_A_INPUT_PROVENANCE: candidate human label join")
            evidence = frame["evidence"]
            decision = row["canonical_decision"]
            role = decision.get("role")
            if role in {"REFEREE", "OTHER_MATCH_OFFICIAL", "RELEVANT_OFFICIAL"}:
                evidence["human_official_count"] += 1
                evidence["human_official_candidate_ids"].append(row["candidate_local_id"])
            evidence["human_goalkeeper_count"] += int(role == "GOALKEEPER")
            evidence["human_occlusion_or_merge_count"] += int(
                decision.get("occlusion") not in {None, "NONE"}
                or decision.get("proposal_validity") == "MULTIPLE_PEOPLE_MERGED"
            )
            evidence["source_event_ids"].append(row["event_id"])
    if label_count != 252:
        raise RuntimeError("FAIL_G7E_A_INPUT_PROVENANCE: 252 candidate labels")

    b3_shortlist = read_json(B3 / "05_RISK_SHORTLIST/diagnostic_shortlist.json")["scenes"]
    if len(b3_shortlist) != 24:
        raise RuntimeError("FAIL_G7E_A_INPUT_PROVENANCE: B3 shortlist")
    for row in b3_shortlist:
        frame = by_sha.get(row["frame_sha256"])
        if frame is None:
            raise RuntimeError("FAIL_G7E_A_INPUT_PROVENANCE: B3 shortlist frame join")
        frame["evidence"]["b3_primary_quotas"].append(row["primary_quota"])
        frame["evidence"]["stable_control_seed"] += int(row["primary_quota"] == "STABLE_CONTROL")
        frame["evidence"]["scene_crowding_high"] += int(
            row["primary_quota"] in {"HIGH_SCALE_OR_PERSPECTIVE_RESIDUAL", "HIGH_PROPOSAL_OR_OFF_PITCH_BURDEN"}
        )

    additional_shortlist = read_json(C3A5C / "03_SCENE_AND_TARGET_SELECTION/scene_shortlist.json")["scenes"]
    for row in additional_shortlist:
        frame = by_sha.get(row["frame_sha256"])
        if frame is None:
            raise RuntimeError("FAIL_G7E_A_INPUT_PROVENANCE: additional shortlist frame join")
        category = row["selection_category"]
        frame["evidence"]["stable_control_seed"] += int(category == "STABLE_CONTROL")
        frame["evidence"]["scene_overlap_high"] += int(category == "HIGH_DENSITY_OVERLAP")
        frame["evidence"]["scene_crowding_high"] += int(category == "HIGH_DENSITY_OVERLAP")
        frame["evidence"]["scene_ids"].append(row["scene_id"])

    for frame in frames:
        rows = candidates[frame["frame_sha256"]]
        evidence = frame["evidence"]
        evidence["candidate_count"] = len(rows)
        evidence["far_candidate_count"] = sum(row["perspective_band"] == "FAR" for row in rows)
        evidence["near_middle_candidate_count"] = sum(row["perspective_band"] == "NEAR_MIDDLE" for row in rows)
        evidence["far_perspective_burden"] = evidence["far_candidate_count"] / len(rows)
        evidence["endline_proxy_count"] = sum(
            float(row["footpoint_xy"][0]) / frame["source_width"] <= 0.16
            or float(row["footpoint_xy"][0]) / frame["source_width"] >= 0.84
            for row in rows
        )
        evidence["boundary_proxy_count"] = sum(
            float(row["footpoint_xy"][1]) / frame["source_height"] >= 0.55 for row in rows
        )
        evidence["source_event_ids"] = sorted(set(evidence["source_event_ids"]))
        evidence["scene_ids"] = sorted(set(evidence["scene_ids"]))
    return by_sha


def supported_classes(frame: Mapping[str, Any]) -> set[str]:
    evidence = frame["evidence"]
    supported = {"STABLE_OPEN_PLAY_CONTROL"}
    if evidence["pair_count"] or evidence["human_occlusion_or_merge_count"] or evidence["scene_overlap_high"]:
        supported.add("OCCLUSION_OR_MERGE_RISK")
    if evidence["pair_count"] or evidence["nested_safe_count"]:
        supported.add("FRAGMENT_OR_DUPLICATE_RISK")
    if evidence["missed_mark_count"] or evidence["candidate_count"]:
        supported.add("PROPOSAL_MISS_RISK")
    if evidence["far_candidate_count"]:
        supported.add("FAR_SIDE_CROWDING")
    if evidence["human_goalkeeper_endline_positive"] or evidence["endline_proxy_count"]:
        supported.add("GOALMOUTH_OR_ENDLINE_CROWD")
    if (
        evidence["human_official_touchline_positive"]
        or evidence["human_official_count"]
        or evidence["boundary_proxy_count"]
    ):
        supported.add("OFFICIAL_OR_BOUNDARY_CONTINUITY")
    return supported


def evidence_score(frame: Mapping[str, Any], selection_class: str) -> tuple[Any, ...]:
    evidence = frame["evidence"]
    exact = 0
    if selection_class == "OCCLUSION_OR_MERGE_RISK":
        exact = (
            8 * evidence["nested_must_protect_count"]
            + 7 * evidence["nested_ambiguous_count"]
            + 5 * evidence["human_occlusion_or_merge_count"]
            + 3 * evidence["scene_overlap_high"]
        )
    elif selection_class == "FRAGMENT_OR_DUPLICATE_RISK":
        exact = 8 * evidence["nested_safe_count"] + 3 * evidence["pair_count"]
    elif selection_class == "PROPOSAL_MISS_RISK":
        exact = 10 * evidence["missed_mark_count"] + int(
            "LOW_PROPOSAL_OR_CANDIDATE_SUPPLY" in evidence["b3_primary_quotas"]
        )
    elif selection_class == "FAR_SIDE_CROWDING":
        exact = 5 * evidence["scene_crowding_high"] + int(
            "HIGH_SCALE_OR_PERSPECTIVE_RESIDUAL" in evidence["b3_primary_quotas"]
        )
    elif selection_class == "GOALMOUTH_OR_ENDLINE_CROWD":
        exact = 10 * evidence["human_goalkeeper_endline_positive"] + evidence["human_goalkeeper_count"]
    elif selection_class == "OFFICIAL_OR_BOUNDARY_CONTINUITY":
        exact = 10 * evidence["human_official_touchline_positive"] + 5 * evidence["human_official_count"]
    elif selection_class == "STABLE_OPEN_PLAY_CONTROL":
        exact = 10 * evidence["stable_control_seed"]
    stable_score = (
        evidence["stable_control_seed"] * 10
        - evidence["pair_count"]
        - evidence["scene_overlap_high"] * 3
        - evidence["missed_mark_count"] * 4
    )
    overlap_burden = evidence["pair_count"] + 5 * evidence["scene_overlap_high"]
    density = evidence["candidate_count"]
    if selection_class == "STABLE_OPEN_PLAY_CONTROL":
        overlap_burden = -overlap_burden
        density = -abs(density - 55)
    return (
        exact,
        evidence["missed_mark_count"],
        evidence["nested_must_protect_count"] + evidence["nested_ambiguous_count"],
        overlap_burden,
        density,
        evidence["far_perspective_burden"],
        stable_score,
    )


def make_companions(
    frames: list[dict[str, Any]], sources: Mapping[tuple[str, str], SourceVideo]
) -> list[dict[str, Any]]:
    expanded = [dict(frame) for frame in frames]
    for match_id in MATCHES:
        match_frames = [frame for frame in frames if frame["match_id"] == match_id]
        for half in ("FIRST_HALF", "SECOND_HALF"):
            half_frames = [frame for frame in match_frames if frame["half"] == half]
            needed = max(0, 10 - len(half_frames))
            if not needed:
                continue
            source = sources[(match_id, half)]
            existing = {frame["centre_frame_index"] for frame in half_frames}
            ranked = sorted(
                half_frames,
                key=lambda frame: (
                    -max(evidence_score(frame, selection_class)[0] for selection_class in CLASS_PRIORITY),
                    -frame["evidence"]["missed_mark_count"],
                    -frame["evidence"]["nested_must_protect_count"],
                    -frame["evidence"]["nested_ambiguous_count"],
                    frame["centre_timestamp_seconds"],
                    frame["anchor_frame_id"],
                ),
            )
            made = 0
            for anchor in ranked:
                if made == needed:
                    break
                chosen: int | None = None
                for delta in (-30, 30):
                    centre = anchor["centre_frame_index"] + delta
                    indices = frame_indices_for_centre(centre, source.fps)
                    if min(indices) < 0 or max(indices) >= source.frame_count:
                        continue
                    if any(abs(centre - other) <= int(source.fps) for other in existing):
                        continue
                    if any(
                        overlap_count(indices, frame_indices_for_centre(other, source.fps)) > 4 for other in existing
                    ):
                        continue
                    chosen = centre
                    break
                if chosen is None:
                    continue
                companion = dict(anchor)
                companion.update(
                    {
                        "anchor_frame_id": f"{anchor['anchor_frame_id']}__companion_{chosen}",
                        "centre_frame_index": chosen,
                        "centre_timestamp_seconds": float(Decimal(chosen) / source.fps),
                        "companion": True,
                        "companion_anchor_frame_id": anchor["anchor_frame_id"],
                        "companion_delta_seconds": float(Decimal(chosen - anchor["centre_frame_index"]) / source.fps),
                    }
                )
                expanded.append(companion)
                existing.add(chosen)
                made += 1
            if made != needed:
                raise RuntimeError(f"FAIL_G7E_A_SELECTION_COVERAGE: companion centres {match_id}:{half}")
    return expanded


def choose_focus(
    frame: Mapping[str, Any],
    rows: list[dict[str, Any]],
    preferred_perspective: str,
    selection_class: str,
) -> tuple[str, dict[str, Any] | None]:
    evidence = frame["evidence"]
    by_id = {row["candidate_id"]: row for row in rows}
    preferred_ids: list[str] = []
    if selection_class in {"OCCLUSION_OR_MERGE_RISK", "FRAGMENT_OR_DUPLICATE_RISK"}:
        preferred_ids.extend(evidence["pair_inner_candidate_ids"])
    if selection_class == "OFFICIAL_OR_BOUNDARY_CONTINUITY":
        preferred_ids.extend(evidence["human_official_candidate_ids"])
    options = [by_id[candidate_id] for candidate_id in preferred_ids if candidate_id in by_id]
    if not options:
        options = list(rows)
    if preferred_perspective != "ANY":
        filtered = [row for row in options if row["perspective_band"] == preferred_perspective]
        if not filtered:
            filtered = [row for row in rows if row["perspective_band"] == preferred_perspective]
        if filtered:
            options = filtered
    if selection_class == "GOALMOUTH_OR_ENDLINE_CROWD":
        options.sort(
            key=lambda row: (
                min(
                    float(row["footpoint_xy"][0]) / frame["source_width"],
                    1 - float(row["footpoint_xy"][0]) / frame["source_width"],
                ),
                row["candidate_id"],
            )
        )
    elif preferred_perspective == "FAR":
        options.sort(
            key=lambda row: (
                float(row["h2_expected_height_px"] or math.inf) / frame["source_height"],
                row["candidate_id"],
            )
        )
    elif preferred_perspective == "NEAR_MIDDLE":
        options.sort(
            key=lambda row: (
                -(float(row["h2_expected_height_px"] or 0) / frame["source_height"]),
                row["candidate_id"],
            )
        )
    else:
        options.sort(key=lambda row: (-(row["score"] or 0), row["candidate_id"]))
    focus = options[0] if options else None
    if focus is None:
        return ("FAR" if preferred_perspective == "FAR" else "NEAR_MIDDLE"), None
    return focus["perspective_band"], focus


def select_bursts(
    frames: list[dict[str, Any]],
    candidates: Mapping[str, list[dict[str, Any]]],
    sources: Mapping[tuple[str, str], SourceVideo],
    provenance_hashes: Mapping[str, str],
) -> list[dict[str, Any]]:
    pool = make_companions(frames, sources)
    selected: list[dict[str, Any]] = []
    for match_id in MATCHES:
        match_pool = [row for row in pool if row["match_id"] == match_id]
        used_centres: set[tuple[str, int]] = set()
        match_selected: list[dict[str, Any]] = []
        for slot_index, slot in enumerate(slot_plan(), start=1):
            selection_class = slot["selection_class"]
            available = [
                row
                for row in match_pool
                if row["half"] == slot["required_half"] and (row["half"], row["centre_frame_index"]) not in used_centres
            ]
            if not available:
                raise RuntimeError(f"FAIL_G7E_A_SELECTION_COVERAGE: slot {match_id}:{slot_index}")
            preferred = slot["preferred_perspective"]
            perspective_available = []
            if preferred != "ANY":
                for row in available:
                    frame_candidates = candidates[row["frame_sha256"]]
                    if any(candidate["perspective_band"] == preferred for candidate in frame_candidates):
                        perspective_available.append(row)
            ranked_pool = perspective_available or available
            ranked_pool.sort(
                key=lambda row: (
                    *(-value for value in evidence_score(row, selection_class)),
                    int(row["companion"]),
                    row["centre_timestamp_seconds"],
                    row["anchor_frame_id"],
                )
            )
            chosen = ranked_pool[0]
            supported = supported_classes(chosen)
            if selection_class in supported:
                fallback_level = 2 if chosen["companion"] else 0
                source_class = selection_class
                fallback_reason = "COMPANION_TEMPORAL_CONTEXT" if chosen["companion"] else "PREFERRED_FROZEN_EVIDENCE"
            else:
                related = {
                    "OCCLUSION_OR_MERGE_RISK": "FAR_SIDE_CROWDING",
                    "FRAGMENT_OR_DUPLICATE_RISK": "OCCLUSION_OR_MERGE_RISK",
                    "PROPOSAL_MISS_RISK": "FAR_SIDE_CROWDING",
                    "GOALMOUTH_OR_ENDLINE_CROWD": "OCCLUSION_OR_MERGE_RISK",
                    "OFFICIAL_OR_BOUNDARY_CONTINUITY": "STABLE_OPEN_PLAY_CONTROL",
                }.get(selection_class)
                if related in supported:
                    fallback_level, source_class, fallback_reason = 3, related, "PREDECLARED_RELATED_CLASS"
                else:
                    fallback_level, source_class, fallback_reason = (
                        4,
                        "STABLE_OPEN_PLAY_CONTROL",
                        "FROZEN_FRAME_CONTROL",
                    )
            source = sources[(match_id, chosen["half"])]
            indices = frame_indices_for_centre(chosen["centre_frame_index"], source.fps)
            if min(indices) < 0 or max(indices) >= source.frame_count or len(set(indices)) != 9:
                raise RuntimeError(f"FAIL_G7E_A_FRAME_PROVENANCE: burst bounds {match_id}:{slot_index}")
            perspective, focus = choose_focus(
                chosen,
                candidates[chosen["frame_sha256"]],
                preferred,
                selection_class,
            )
            if chosen["companion"]:
                focus_payload: list[dict[str, Any]] = []
            elif focus is None:
                focus_payload = []
            else:
                focus_payload = [
                    {
                        "candidate_id": focus["candidate_id"],
                        "source_box_xyxy": focus["source_box_xyxy"],
                        "footpoint_xy": focus["footpoint_xy"],
                        "h2_expected_height_px": focus["h2_expected_height_px"],
                        "h2_support_count": focus["h2_support_count"],
                    }
                ]
            burst_id = f"g7e_a_{match_id}_{len(match_selected) + 1:02d}"
            blind_payload = {
                "burst_id": burst_id,
                "match_id": match_id,
                "half": chosen["half"],
                "centre_frame_index": chosen["centre_frame_index"],
                "frame_indices_zero_based": list(indices),
                "source_video_sha256": source.sha256,
                "focus_candidates": focus_payload,
                "selection_reason_hidden": True,
                "human_answers_present": False,
            }
            evidence = chosen["evidence"]
            record = {
                "schema_version": "football_intelligence.g7e_a.temporal_burst.v1",
                "burst_id": burst_id,
                "match_id": match_id,
                "half": chosen["half"],
                "centre_timestamp_seconds": float(Decimal(chosen["centre_frame_index"]) / source.fps),
                "centre_frame_index_zero_based": chosen["centre_frame_index"],
                "frame_indices_zero_based": list(indices),
                "relative_offsets_seconds": [float(offset) for offset in OFFSETS_SECONDS],
                "source_video_relative_path": source.relative_path,
                "source_video_sha256": source.sha256,
                "source_width": source.width,
                "source_height": source.height,
                "primary_selection_class": selection_class,
                "secondary_evidence_tags": sorted(
                    {
                        *("NESTED_MUST_PROTECT" for _ in range(int(evidence["nested_must_protect_count"] > 0))),
                        *("NESTED_AMBIGUOUS" for _ in range(int(evidence["nested_ambiguous_count"] > 0))),
                        *("HUMAN_SAFE_FRAGMENT" for _ in range(int(evidence["nested_safe_count"] > 0))),
                        *("MISSED_PERSON_MARK" for _ in range(int(evidence["missed_mark_count"] > 0))),
                        *("HIGH_OVERLAP_PROXY" for _ in range(int(evidence["pair_count"] > 0))),
                        *(
                            "HUMAN_GOALKEEPER_ENDLINE"
                            for _ in range(int(evidence["human_goalkeeper_endline_positive"] > 0))
                        ),
                        *(
                            "HUMAN_OFFICIAL_TOUCHLINE"
                            for _ in range(int(evidence["human_official_touchline_positive"] > 0))
                        ),
                        *("H2_PERSPECTIVE_CONTEXT" for _ in range(1)),
                    }
                ),
                "selection_source_ids": {
                    "frozen_anchor_frame_id": chosen["source_frozen_frame_id"],
                    "pair_case_ids": evidence["pair_case_ids"],
                    "scene_ids": evidence["scene_ids"],
                    "missed_mark_ids": evidence["mark_ids"],
                    "source_event_ids": evidence["source_event_ids"],
                },
                "selection_source_hashes": dict(provenance_hashes),
                "focus_candidates": focus_payload,
                "selection_anchor_focus_candidate_ids": evidence["pair_inner_candidate_ids"],
                "perspective_band": perspective,
                "crowding_overlap_descriptors": {
                    "retained_candidate_count": evidence["candidate_count"],
                    "nested_pair_count": evidence["pair_count"],
                    "far_candidate_fraction": evidence["far_perspective_burden"],
                    "high_scene_overlap_evidence_count": evidence["scene_overlap_high"],
                },
                "h2_context": {
                    "model": "H2_LOCAL_2D_WEIGHTED_MEDIAN",
                    "used_as_context_only": True,
                    "temporal_truth": False,
                },
                "companion": chosen["companion"],
                "companion_label": "COMPANION_TEMPORAL_CONTEXT" if chosen["companion"] else None,
                "companion_anchor_frame_id": chosen["companion_anchor_frame_id"],
                "companion_delta_seconds": chosen["companion_delta_seconds"],
                "fallback_level": fallback_level,
                "fallback_source_class": source_class,
                "fallback_reason": fallback_reason,
                "selection_reason_is_human_truth": False,
                "blind_review_payload_sha256": stable_hash(blind_payload),
                "blind_review_payload_contract": "NO_HUMAN_ANSWERS_OR_MODEL_CONCLUSIONS",
                "production_ready": False,
            }
            match_selected.append(record)
            used_centres.add((chosen["half"], chosen["centre_frame_index"]))
        selected.extend(match_selected)
    validation = validate_burst_records(selected)
    if not validation["valid"]:
        raise RuntimeError(f"FAIL_G7E_A_QUOTA_BALANCE: {validation['errors']}")
    return selected


def representative_bursts(bursts: list[dict[str, Any]]) -> list[str]:
    required_tags = ("NESTED_MUST_PROTECT", "HUMAN_SAFE_FRAGMENT", "MISSED_PERSON_MARK")
    required = {
        *(f"CLASS:{selection_class}" for selection_class in QUOTAS),
        *(f"TAG:{tag}" for tag in required_tags),
        "PERSPECTIVE:FAR",
        "PERSPECTIVE:NEAR_MIDDLE",
        "LIGHTING:DAYLIGHT",
        "LIGHTING:LOW_LIGHT",
        *(f"MATCH:{match_id}" for match_id in MATCHES),
    }
    weights = {
        **{f"CLASS:{selection_class}": 100 for selection_class in QUOTAS},
        **{f"TAG:{tag}": 80 for tag in required_tags},
        "PERSPECTIVE:FAR": 45,
        "PERSPECTIVE:NEAR_MIDDLE": 45,
        "LIGHTING:DAYLIGHT": 45,
        "LIGHTING:LOW_LIGHT": 45,
        **{f"MATCH:{match_id}": 35 for match_id in MATCHES},
    }

    def coverage(row: Mapping[str, Any]) -> set[str]:
        lighting = "LOW_LIGHT" if row["match_id"] == "117092" else "DAYLIGHT"
        return {
            f"CLASS:{row['primary_selection_class']}",
            f"PERSPECTIVE:{row['perspective_band']}",
            f"LIGHTING:{lighting}",
            f"MATCH:{row['match_id']}",
            *(f"TAG:{tag}" for tag in required_tags if tag in row["secondary_evidence_tags"]),
        }

    selected: list[dict[str, Any]] = []
    uncovered = set(required)
    while uncovered and len(selected) < 12:
        selected_ids = {item["burst_id"] for item in selected}
        match_counts = Counter(item["match_id"] for item in selected)
        options = [row for row in bursts if row["burst_id"] not in selected_ids]
        ranked = sorted(
            options,
            key=lambda row: (
                -sum(weights[key] for key in coverage(row) & uncovered),
                match_counts[row["match_id"]],
                row["fallback_level"],
                row["match_id"],
                row["centre_timestamp_seconds"],
                row["burst_id"],
            ),
        )
        if not ranked or not (coverage(ranked[0]) & uncovered):
            break
        selected.append(ranked[0])
        uncovered -= coverage(ranked[0])
    while len(selected) < 12:
        selected_ids = {item["burst_id"] for item in selected}
        match_counts = Counter(item["match_id"] for item in selected)
        selected.append(
            sorted(
                (row for row in bursts if row["burst_id"] not in selected_ids),
                key=lambda row: (
                    match_counts[row["match_id"]],
                    row["fallback_level"],
                    row["match_id"],
                    row["centre_timestamp_seconds"],
                    row["burst_id"],
                ),
            )[0]
        )
    covered = set().union(*(coverage(row) for row in selected))
    if required - covered:
        raise RuntimeError(f"FAIL_G7E_A_VISUALS: representative coverage {sorted(required - covered)}")
    return [row["burst_id"] for row in selected]


def decode_frame_provenance(
    bursts: list[dict[str, Any]],
    sources: Mapping[tuple[str, str], SourceVideo],
    preview_ids: set[str],
    expected_reference_count: int = 1080,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], np.ndarray]]:
    requests_by_source: dict[str, dict[int, list[tuple[dict[str, Any], int]]]] = defaultdict(lambda: defaultdict(list))
    for burst in bursts:
        source = sources[(burst["match_id"], burst["half"])]
        for offset_index, (offset, frame_index) in enumerate(
            zip(OFFSETS_SECONDS, burst["frame_indices_zero_based"], strict=True)
        ):
            requests_by_source[source.relative_path][frame_index].append((burst, offset_index))
    hashes: dict[tuple[str, int], str] = {}
    thumbnails: dict[tuple[str, int], np.ndarray] = {}
    total_sources = len(requests_by_source)
    for source_number, relative_path in enumerate(sorted(requests_by_source), start=1):
        source = next(item for item in sources.values() if item.relative_path == relative_path)
        targets = requests_by_source[relative_path]
        capture = cv2.VideoCapture(str(source.path))
        if not capture.isOpened():
            raise RuntimeError(f"FAIL_G7E_A_FRAME_PROVENANCE: open {relative_path}")
        current_index: int | None = None
        try:
            for frame_index in sorted(targets):
                if current_index is None or frame_index < current_index or frame_index - current_index > 60:
                    if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                        raise RuntimeError(f"FAIL_G7E_A_FRAME_PROVENANCE: seek {relative_path}:{frame_index}")
                    current_index = frame_index
                decoded: np.ndarray | None = None
                while current_index is not None and current_index <= frame_index:
                    okay, candidate = capture.read()
                    actual = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
                    if not okay or actual != current_index:
                        raise RuntimeError(f"FAIL_G7E_A_FRAME_PROVENANCE: exact frame {relative_path}:{frame_index}")
                    if current_index == frame_index:
                        decoded = candidate
                    current_index += 1
                if decoded is None or decoded.shape[:2] != (source.height, source.width):
                    raise RuntimeError(f"FAIL_G7E_A_FRAME_PROVENANCE: dimensions {relative_path}:{frame_index}")
                rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                hashes[(relative_path, frame_index)] = hashlib.sha256(rgb.tobytes(order="C")).hexdigest()
                if any(burst["burst_id"] in preview_ids for burst, _ in targets[frame_index]):
                    width = 150
                    height = max(45, round(source.height * width / source.width))
                    thumbnails[(relative_path, frame_index)] = cv2.resize(
                        decoded,
                        (width, height),
                        interpolation=cv2.INTER_AREA,
                    )
        finally:
            capture.release()
        print(
            f"decoded source {source_number}/{total_sources}: {relative_path} ({len(targets)} unique selected frames)",
            flush=True,
        )

    frame_rows: list[dict[str, Any]] = []
    for burst in bursts:
        source = sources[(burst["match_id"], burst["half"])]
        centre = Decimal(burst["centre_frame_index_zero_based"]) / source.fps
        references: list[str] = []
        for sequence, (offset, frame_index) in enumerate(
            zip(OFFSETS_SECONDS, burst["frame_indices_zero_based"], strict=True)
        ):
            requested = centre + offset
            resolved = Decimal(frame_index) / source.fps
            frame_reference_id = f"{burst['burst_id']}_f{sequence + 1:02d}"
            references.append(frame_reference_id)
            frame_rows.append(
                {
                    "schema_version": "football_intelligence.g7e_a.temporal_frame_reference.v1",
                    "frame_reference_id": frame_reference_id,
                    "burst_id": burst["burst_id"],
                    "burst_frame_sequence": sequence,
                    "match_id": burst["match_id"],
                    "half": burst["half"],
                    "relative_offset_seconds": float(offset),
                    "requested_timestamp_seconds": float(requested),
                    "resolved_timestamp_seconds": float(resolved),
                    "frame_index_zero_based": frame_index,
                    "source_video_relative_path": source.relative_path,
                    "source_video_sha256": source.sha256,
                    "source_video_byte_size": source.byte_size,
                    "source_fps": float(source.fps),
                    "source_frame_count": source.frame_count,
                    "source_width": source.width,
                    "source_height": source.height,
                    "frame_pixel_sha256": hashes[(source.relative_path, frame_index)],
                    "frame_pixel_hash_contract": "SHA256_RGB24_C_CONTIGUOUS_SOURCE_DIMENSIONS",
                    "decoder": f"OPENCV_{cv2.__version__}_EXACT_GLOBAL_FRAME_INDEX",
                    "full_resolution_frame_persisted": False,
                    "production_ready": False,
                }
            )
        burst["frame_reference_ids"] = references
    if len(frame_rows) != expected_reference_count or any(len(row["frame_reference_ids"]) != 9 for row in bursts):
        raise RuntimeError(f"FAIL_G7E_A_FRAME_PROVENANCE: {expected_reference_count} references")
    return frame_rows, thumbnails


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf") if bold else Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf") if bold else Path("C:/Windows/Fonts/segoeui.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def draw_selection_matrix(bursts: list[dict[str, Any]], path: Path) -> None:
    image = Image.new("RGB", (1900, 1120), "#f5f7fb")
    draw = ImageDraw.Draw(image)
    draw.text((60, 35), "G7E-A TEMPORAL SELECTION MATRIX", fill="#14213d", font=font(38, True))
    draw.text((60, 88), "TEMPORAL BURST SELECTION — NOT HUMAN TRUTH", fill="#b45309", font=font(24, True))
    class_labels = {
        "OCCLUSION_OR_MERGE_RISK": "Occlusion / merge",
        "FRAGMENT_OR_DUPLICATE_RISK": "Fragment / duplicate",
        "PROPOSAL_MISS_RISK": "Proposal miss",
        "FAR_SIDE_CROWDING": "Far-side crowding",
        "GOALMOUTH_OR_ENDLINE_CROWD": "Goalmouth / end line",
        "OFFICIAL_OR_BOUNDARY_CONTINUITY": "Official / boundary",
        "STABLE_OPEN_PLAY_CONTROL": "Stable control",
    }
    columns = list(QUOTAS)
    x0, y0, cell_w, cell_h = 250, 185, 220, 92
    for index, selection_class in enumerate(columns):
        x = x0 + index * cell_w
        draw.rounded_rectangle((x, y0 - 45, x + cell_w - 8, y0 + 5), 10, fill="#dbeafe")
        draw.multiline_text(
            (x + 10, y0 - 39), class_labels[selection_class], fill="#1e3a8a", font=font(16, True), spacing=2
        )
    for row_index, match_id in enumerate(MATCHES):
        y = y0 + row_index * cell_h
        match_rows = [row for row in bursts if row["match_id"] == match_id]
        draw.text((65, y + 20), match_id, fill="#111827", font=font(25, True))
        for column_index, selection_class in enumerate(columns):
            count = sum(row["primary_selection_class"] == selection_class for row in match_rows)
            x = x0 + column_index * cell_w
            draw.rounded_rectangle((x, y, x + cell_w - 8, y + 68), 10, fill="#ffffff", outline="#cbd5e1", width=2)
            draw.text((x + 85, y + 16), str(count), fill="#0f766e", font=font(28, True))
        halves = Counter(row["half"] for row in match_rows)
        perspectives = Counter(row["perspective_band"] for row in match_rows)
        fallbacks = Counter(row["fallback_level"] for row in match_rows)
        draw.text(
            (65, y + 55),
            f"H1 {halves['FIRST_HALF']} · H2 {halves['SECOND_HALF']} · FAR {perspectives['FAR']} · NEAR/MID {perspectives['NEAR_MIDDLE']}",
            fill="#475569",
            font=font(14),
        )
        draw.text(
            (x0, y + 70),
            "Fallbacks " + ", ".join(f"L{level}:{fallbacks[level]}" for level in sorted(fallbacks)),
            fill="#64748b",
            font=font(13),
        )
    totals = Counter(row["primary_selection_class"] for row in bursts)
    summary_y = 790
    draw.rounded_rectangle((55, summary_y, 1840, 1055), 18, fill="#14213d")
    draw.text((85, summary_y + 28), "Frozen closure", fill="white", font=font(28, True))
    draw.text(
        (85, summary_y + 80),
        "120 bursts · 20 per match · 9 frames · 1,080 references",
        fill="#bfdbfe",
        font=font(25, True),
    )
    draw.text(
        (85, summary_y + 130),
        "Quota totals: " + "  |  ".join(f"{class_labels[key]} {totals[key]}" for key in columns),
        fill="#e2e8f0",
        font=font(17),
    )
    high_fallbacks = sum(row["fallback_level"] >= 3 for row in bursts)
    companions = sum(bool(row["companion"]) for row in bursts)
    draw.text(
        (85, summary_y + 185),
        f"Companion bursts {companions} · fallback level 3/4 {high_fallbacks} of 120 · production_ready=false",
        fill="#fde68a",
        font=font(20, True),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def draw_burst_strips(
    bursts: list[dict[str, Any]],
    preview_ids: list[str],
    thumbnails: Mapping[tuple[str, int], np.ndarray],
    path: Path,
) -> None:
    by_id = {row["burst_id"]: row for row in bursts}
    image = Image.new("RGB", (1900, 1420), "#0f172a")
    draw = ImageDraw.Draw(image)
    draw.text((40, 25), "TWELVE REPRESENTATIVE NINE-FRAME BURSTS", fill="white", font=font(32, True))
    draw.text((40, 68), "TEMPORAL BURST SELECTION — NOT HUMAN TRUTH", fill="#fbbf24", font=font(21, True))
    for row_index, burst_id in enumerate(preview_ids):
        burst = by_id[burst_id]
        y = 112 + row_index * 106
        lighting = "LOW-LIGHT" if burst["match_id"] == "117092" else "DAYLIGHT"
        evidence_labels = [
            label
            for tag, label in (
                ("NESTED_MUST_PROTECT", "must-protect seed"),
                ("HUMAN_SAFE_FRAGMENT", "safe-fragment seed"),
                ("MISSED_PERSON_MARK", "missed-person seed"),
            )
            if tag in burst["secondary_evidence_tags"]
        ]
        if burst["primary_selection_class"] == "STABLE_OPEN_PLAY_CONTROL":
            evidence_labels.append("stable control")
        label = (
            f"{burst['match_id']}  {burst['half'].replace('_', ' ')}  "
            f"{burst['primary_selection_class'].replace('_', ' ')}  {burst['perspective_band']}  {lighting}"
        )
        draw.text((25, y + 7), label, fill="#e2e8f0", font=font(13, True))
        detail = f"{burst_id} · t={burst['centre_timestamp_seconds']:.2f}s"
        if evidence_labels:
            detail += " · " + ", ".join(evidence_labels)
        draw.text((25, y + 32), detail, fill="#94a3b8", font=font(11))
        for column, frame_index in enumerate(burst["frame_indices_zero_based"]):
            key = (burst["source_video_relative_path"], frame_index)
            frame = thumbnails.get(key)
            if frame is None:
                raise RuntimeError(f"FAIL_G7E_A_VISUALS: missing thumbnail {burst_id}:{frame_index}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tile = Image.fromarray(rgb)
            tile.thumbnail((150, 74))
            x = 475 + column * 156
            tile_y = y
            draw.rounded_rectangle((x - 2, tile_y - 2, x + 152, tile_y + 78), 4, fill="#334155")
            image.paste(tile, (x, tile_y + (74 - tile.height) // 2))
            draw.text((x + 50, tile_y + 77), f"{float(OFFSETS_SECONDS[column]):+.1f}s", fill="#cbd5e1", font=font(11))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def refresh_visuals() -> None:
    bursts = read_jsonl(STAGE / "02_BURST_SELECTION/temporal_burst_manifest.jsonl")
    frame_rows = read_jsonl(STAGE / "02_BURST_SELECTION/temporal_frame_manifest.jsonl")
    input_closure = read_json(STAGE / "00_INPUT_CLOSURE/input_closure.json")
    sources = {
        (row["match_id"], row["half"]): SourceVideo(
            match_id=row["match_id"],
            half=row["half"],
            relative_path=row["relative_path"],
            path=ROOT / row["relative_path"],
            sha256=row["sha256"],
            byte_size=row["byte_size"],
            fps=Decimal(str(row["fps"])),
            frame_count=row["frame_count"],
            width=row["width"],
            height=row["height"],
        )
        for row in input_closure["source_videos"]
    }
    preview_ids = representative_bursts(bursts)
    selected_ids = set(preview_ids)
    preview_bursts = [row for row in bursts if row["burst_id"] in selected_ids]
    decoded_rows, thumbnails = decode_frame_provenance(
        preview_bursts,
        sources,
        selected_ids,
        expected_reference_count=108,
    )
    expected_hashes = {
        (row["source_video_relative_path"], row["frame_index_zero_based"]): row["frame_pixel_sha256"]
        for row in frame_rows
    }
    if any(
        expected_hashes[(row["source_video_relative_path"], row["frame_index_zero_based"])] != row["frame_pixel_sha256"]
        for row in decoded_rows
    ):
        raise RuntimeError("FAIL_G7E_A_VISUALS: preview frame hash mismatch")
    draw_selection_matrix(bursts, STAGE / "04_VISUAL_QA/01_TEMPORAL_SELECTION_MATRIX.png")
    draw_burst_strips(
        bursts,
        preview_ids,
        thumbnails,
        STAGE / "04_VISUAL_QA/02_REPRESENTATIVE_BURST_STRIPS.png",
    )
    refresh_handoff()


def annotation_ontology() -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.g7e_a.temporal_annotation_ontology.v1",
        "protocol_id": PROTOCOL_ID,
        "scope": "BURST_LOCAL_CONTINUITY_ONLY",
        "identity_boundary": {
            "subject_tokens": ["SUBJECT_A", "SUBJECT_B", "SUBJECT_C"],
            "tokens_reset_each_burst": True,
            "permanent_identity": "FORBIDDEN",
            "cross_burst_identity": "FORBIDDEN",
            "cross_match_identity": "FORBIDDEN",
            "track_ids": "FORBIDDEN",
            "shirt_numbers": "FORBIDDEN",
            "contract": "NO_PERMANENT_IDENTITY",
        },
        "enumerations": {key: list(values) for key, values in ONTOLOGY.items()},
        "team_classification": "INTENTIONALLY_EXCLUDED_FIRST_TEMPORAL_WAVE",
        "selection_classes_are_temporal_truth": False,
        "production_ready": False,
    }


def event_schema_draft() -> dict[str, Any]:
    common = {
        "append_only": True,
        "production_ready": {"const": False},
        "review_revision": {"type": "string", "minLength": 1},
    }
    return {
        "schema_version": "football_intelligence.g7e_a.temporal_event_schema_draft.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "DESIGN_ONLY_NOT_IMPLEMENTED",
        "atomic_unit": "ONE_FINAL_ANNOTATION_EVENT_PER_BURST",
        "schemas": {
            "burst_subject": {
                "required": ["subject_token", "role", "participation", "certainty"],
                "subject_token": {"enum": ["SUBJECT_A", "SUBJECT_B", "SUBJECT_C"]},
                "role": {"enum": list(ONTOLOGY["role"])},
                "participation": {"enum": list(ONTOLOGY["participation"])},
                "certainty": {"enum": list(ONTOLOGY["certainty"])},
                "scope": {"const": "BURST_LOCAL_CONTINUITY_ONLY"},
            },
            "frame_subject_observation": {
                "required": [
                    "frame_reference_id",
                    "frame_pixel_sha256",
                    "subject_token",
                    "visibility",
                    "observation_supply",
                    "occlusion_phase",
                    "continuity",
                    "certainty",
                ],
                "visibility": {"enum": list(ONTOLOGY["visibility"])},
                "observation_supply": {"enum": list(ONTOLOGY["observation_supply"])},
                "occlusion_phase": {"enum": list(ONTOLOGY["occlusion_phase"])},
                "continuity": {"enum": list(ONTOLOGY["continuity"])},
                "certainty": {"enum": list(ONTOLOGY["certainty"])},
            },
            "candidate_mapping": {
                "required": ["frame_reference_id", "subject_token", "candidate_ids", "relationship"],
                "candidate_ids": {"type": "array", "items": {"type": "string"}},
                "relationship": {"enum": list(ONTOLOGY["candidate_relationship"])},
                "explicit_no_candidate": "candidate_ids=[] and frame observation supply=NO_CANDIDATE",
            },
            "whole_burst_missed_person_mark": {
                "required": ["mark_id", "frame_reference_id", "source_xy", "role", "certainty"],
                "coordinate_space": {"const": "SOURCE_IMAGE_PIXELS"},
                "role": {"enum": list(ONTOLOGY["role"])},
                "certainty": {"enum": list(ONTOLOGY["certainty"])},
                "cross_burst_identity": {"const": False},
            },
            "burst_annotation_event": {
                "required": [
                    "event_id",
                    "burst_id",
                    "burst_manifest_sha256",
                    "source_frame_hashes",
                    "subjects",
                    "frame_observations",
                    "candidate_mappings",
                    "whole_burst_check_complete",
                    "review_revision",
                    "production_ready",
                ],
                "identity_scope": {"const": "BURST_LOCAL_CONTINUITY_ONLY"},
                "permanent_identity": {"const": False},
                **common,
            },
            "acknowledgement_receipt": {
                "required": ["receipt_id", "event_id", "event_sha256", "persisted_at_utc"],
                "http_200_gate": "receipt atomically persisted before acknowledgement",
                **common,
            },
            "completion_receipt": {
                "required": [
                    "receipt_id",
                    "burst_manifest_sha256",
                    "latest_event_ids",
                    "latest_event_hashes",
                    "burst_count",
                    "all_bursts_complete",
                ],
                "burst_count": {"const": 120},
                "all_bursts_complete": {"const": True},
                "http_200_gate": "completion receipt atomically persisted first",
                **common,
            },
        },
        "server_backed_drafts": {
            "required": True,
            "mutable": True,
            "never_count_as_human_truth": True,
            "refresh_restoration": "exact latest compatible draft",
        },
        "implementation_boundary": "G7E_B_ONLY",
        "production_ready": False,
    }


WORKFLOW = """# G7E-B novice reviewer workflow design

Status: design only. G7E-A does not implement a reviewer or collect annotations.

## Human-facing flow

The reviewer opens one burst at a time with plain language and one question at a time. It starts with a short tutorial explaining that `SUBJECT_A`, `SUBJECT_B`, and `SUBJECT_C` are temporary labels that reset after every burst. Team classification, names, shirt numbers, permanent IDs, and cross-burst continuity are never requested.

1. Load and hash-check all nine source frames before enabling questions.
2. Show a full panorama, a large focus crop, a nine-frame stepper, and a short loop. Candidate overlays are off by default and can be toggled.
3. Find the highlighted subject in the centre frame, or the nearest frame where the subject is clearly visible. If no supplied focus is safe, begin from a source-coordinate point/box mark.
4. For each burst-local subject, ask visibility and candidate-supply questions across the nine frames. Use branching so `NOT_PRESENT`, `OUT_OF_FRAME_OR_LEFT_SCENE`, and `NOT_APPLICABLE` skip irrelevant candidate questions.
5. Ask whether multiple boxes are duplicates, fragments, different people, a correct inner person with a bad outer box, a merged multi-person box, an object/background box, or uncertain.
6. Ask occlusion phase only where visibility changes: entering, occluded, exiting, none, or uncertain.
7. Ask continuity only within this burst: same temporary subject, different subject, cannot tell, or not applicable.
8. Ask role, participation, and certainty once per temporary subject, with `Unknown`/`Not sure` always available.
9. Finish with a whole-burst missed-person sweep. Allow source-coordinate point/box marking on any of the nine frames.
10. Show a plain-language summary. Persist the final burst event atomically, then its acknowledgement receipt, before showing `SAVED — SERVER ACKNOWLEDGED` and advancing.

## Controls and restoration

- Previous/next frame, spacebar loop playback, play speed limited to review-safe presets, overlays on/off, Fit, Zoom, Pan, Reset, and Full screen.
- Draft after every answer to a server-backed draft endpoint; drafts are not immutable truth.
- Refresh restores the first incomplete burst and exact latest compatible draft.
- Final events are append-only. Acknowledgement and completion receipts reference exact event hashes.
- `ALL BURSTS COMPLETE` appears only after the completion receipt for the latest 120-event set is persisted.
- Asset, mapping, runtime, draft, and persistence errors are distinct blocking states.

## Blind-first boundary

The browser receives source-frame references, hashes, focus geometry, and candidate geometry. It does not receive prior human answers, protected evidence tags, model conclusions, or the interpretation of a selection class before acknowledgement.

## Workload controls

Default to one focus subject; add `SUBJECT_B` or `SUBJECT_C` only when the burst visibly requires it. Present a compact nine-cell matrix after the first frame-level answer so unchanged adjacent states can be confirmed together, while each changed state remains explicit. This preserves visibility, supply, occlusion, continuity, role, participation, and certainty within the 2–4 minute median target.
"""


def quota_report(bursts: list[dict[str, Any]]) -> dict[str, Any]:
    per_match: dict[str, Any] = {}
    for match_id in MATCHES:
        rows = [row for row in bursts if row["match_id"] == match_id]
        per_match[match_id] = {
            "burst_count": len(rows),
            "quota_counts": dict(Counter(row["primary_selection_class"] for row in rows)),
            "half_counts": dict(Counter(row["half"] for row in rows)),
            "perspective_counts": dict(Counter(row["perspective_band"] for row in rows)),
            "fallback_counts": {
                str(key): value for key, value in sorted(Counter(row["fallback_level"] for row in rows).items())
            },
            "companion_count": sum(bool(row["companion"]) for row in rows),
        }
    return {
        "schema_version": "football_intelligence.g7e_a.selection_quota_report.v1",
        "burst_count": len(bursts),
        "per_match": per_match,
        "total_quota_counts": dict(Counter(row["primary_selection_class"] for row in bursts)),
        "high_fallback_count": sum(row["fallback_level"] >= 3 for row in bursts),
        "high_fallback_rate": sum(row["fallback_level"] >= 3 for row in bursts) / len(bursts),
        "fallback_threshold_max": 0.15,
        "passes": validate_burst_records(bursts)["valid"],
        "selection_classes_are_human_truth": False,
        "production_ready": False,
    }


def refresh_handoff() -> None:
    handoff = STAGE / "06_REVIEW_PACK/CHATGPT_HANDOFF"
    burst_path = STAGE / "02_BURST_SELECTION/temporal_burst_manifest.jsonl"
    frame_path = STAGE / "02_BURST_SELECTION/temporal_frame_manifest.jsonl"
    quota_path = STAGE / "02_BURST_SELECTION/selection_quota_report.json"
    provenance_path = STAGE / "02_BURST_SELECTION/selection_provenance_report.json"
    ontology_path = STAGE / "03_ANNOTATION_PROTOCOL/temporal_annotation_ontology.json"
    workload_path = STAGE / "03_ANNOTATION_PROTOCOL/human_workload_estimate.json"
    decision_path = STAGE / "01_SELECTION_AND_ANNOTATION_CONTRACT/decision.json"
    tests_path = STAGE / "05_TESTS_AND_LOGS/focused_test_results.json"
    bursts = read_jsonl(burst_path)
    frames = read_jsonl(frame_path)
    quota = read_json(quota_path)
    provenance = read_json(provenance_path)
    workload = read_json(workload_path)
    tests = read_json(tests_path) if tests_path.is_file() else {"status": "PENDING"}
    write_json(
        handoff / "01_EXECUTIVE_SUMMARY.json",
        {
            "decision": DECISION,
            "bursts": len(bursts),
            "bursts_per_match": 20,
            "frame_references": len(frames),
            "frames_per_burst": 9,
            "quota_pass": quota["passes"],
            "frame_pixel_hashes_present": all(len(row["frame_pixel_sha256"]) == 64 for row in frames),
            "identity_scope": "BURST_LOCAL_CONTINUITY_ONLY",
            "human_annotations_created": False,
            "reviewer_implemented": False,
            "production_ready": False,
            "focused_tests": tests,
            "stop_before": "G7E_B_TEMPORAL_REVIEWER_IMPLEMENTATION",
        },
    )
    write_json(
        handoff / "02_INPUT_AND_SELECTION_CONTRACT.json",
        {
            "input_closure": read_json(STAGE / "00_INPUT_CLOSURE/input_closure.json"),
            "selection_contract": read_json(STAGE / "01_SELECTION_AND_ANNOTATION_CONTRACT/selection_contract.json"),
        },
    )
    write_json(
        handoff / "03_BURST_SELECTION_RESULTS.json",
        {
            "manifest": artifact(burst_path),
            "frame_manifest": artifact(frame_path),
            "burst_count": len(bursts),
            "frame_reference_count": len(frames),
            "per_match_counts": dict(Counter(row["match_id"] for row in bursts)),
            "companion_count": sum(bool(row["companion"]) for row in bursts),
            "full_resolution_frames_saved": 0,
        },
    )
    write_json(
        handoff / "04_QUOTA_AND_PROVENANCE_RESULTS.json",
        {
            "quota_report": quota,
            "provenance_summary": provenance,
            "manifest_hashes": read_json(STAGE / "02_BURST_SELECTION/burst_manifest_sha256.json"),
        },
    )
    shutil.copy2(ontology_path, handoff / "05_TEMPORAL_ANNOTATION_ONTOLOGY.json")
    (handoff / "06_REVIEWER_WORKFLOW_AND_WORKLOAD.md").write_text(
        WORKFLOW
        + "\n\n## Estimated workload\n\n"
        + f"Low: {workload['per_burst_minutes']['low']} minutes per burst / {workload['total_hours']['low']} hours total.  "
        + f"Median: {workload['per_burst_minutes']['median']} minutes / {workload['total_hours']['median']} hours.  "
        + f"High: {workload['per_burst_minutes']['high']} minutes / {workload['total_hours']['high']} hours.\n",
        encoding="utf-8",
    )
    (handoff / "07_DECISION.md").write_text(
        "# G7E-A decision\n\n"
        f"`{read_json(decision_path)['decision']}`\n\n"
        "The 120-burst dataset and burst-local annotation protocol are frozen. Selection classes remain risk/proxy labels, not temporal human truth. No reviewer, human annotation, inference, suppression, default change, or permanent identity was created. Stop before G7E-B.\n",
        encoding="utf-8",
    )
    shutil.copy2(STAGE / "04_VISUAL_QA/01_TEMPORAL_SELECTION_MATRIX.png", handoff / "08_SELECTION_MATRIX.png")
    shutil.copy2(STAGE / "04_VISUAL_QA/02_REPRESENTATIVE_BURST_STRIPS.png", handoff / "09_REPRESENTATIVE_BURSTS.png")
    manifest_rows = []
    for path in sorted(handoff.iterdir()):
        if path.name == "10_MANIFEST.json":
            continue
        manifest_rows.append({"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256(path)})
    if len(manifest_rows) != 9:
        raise RuntimeError(f"FAIL_G7E_A_CHATGPT_HANDOFF: expected 9 pre-manifest files, got {len(manifest_rows)}")
    write_json(
        handoff / "10_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7e_a.handoff_manifest.v1",
            "files": manifest_rows,
            "self_hashed": False,
        },
    )
    (STAGE / "06_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only 06_REVIEW_PACK/CHATGPT_HANDOFF. It contains exactly ten self-contained files.\n",
        encoding="utf-8",
    )


def build() -> None:
    if git_head() != EXPECTED_HEAD:
        raise RuntimeError("FAIL_BASELINE_OR_WORKTREE: expected HEAD")
    pack_manifest = read_json(PACK / "04_PACK_MANIFEST.json")
    for row in pack_manifest["files"]:
        path = PACK / row["path"]
        if path.stat().st_size != row["byte_size"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"FAIL_G7E_A_INPUT_PROVENANCE: prompt pack {row['path']}")
    sources = canonical_sources()
    frames = load_frozen_frames(sources)
    candidates = load_retained_candidates(frames)
    add_evidence(frames, candidates)

    input_artifacts = {
        "c3a6_six_match_regression": C3A6 / "02_SIX_MATCH_POLICY_REGRESSION/six_match_policy_regression.json",
        "c3a5d_combined_evidence": C3A5D / "04_COMBINED_DEVELOPMENT_EVIDENCE/combined_six_match_evidence.json",
        "c3b_candidate_input_manifest": C3B / "01_INPUT_AND_PAIR_CLOSURE/candidate_input_manifest.json",
        "c3b_pair_selection": C3B / "05_REVIEW_SELECTION/review_pair_selection.json",
        "c3b_pair_geometry": C3B / "02_NESTED_PAIR_GEOMETRY/nested_pair_geometry.jsonl",
        "c3b1_pair_truth": C3B1 / "02_NORMALIZED_PAIR_TRUTH/pair_human_labels.jsonl",
        "c3b2_h2_selection": C3B2 / "03_EXPECTED_HEIGHT_SURFACES/selected_expected_height_model.json",
        "c3b2_h2_predictions": C3B2 / "03_EXPECTED_HEIGHT_SURFACES/candidate_scale_predictions.jsonl",
        "c2_temporal_need": C2 / "06_CROWDING_AND_TEMPORAL_HYPOTHESIS/temporal_evidence_need.json",
        "b3_shortlist": B3 / "05_RISK_SHORTLIST/diagnostic_shortlist.json",
    }
    provenance_hashes = {name: sha256(path) for name, path in input_artifacts.items()}
    regression = read_json(input_artifacts["c3a6_six_match_regression"])
    combined = read_json(input_artifacts["c3a5d_combined_evidence"])
    truth_counts = Counter(row["safe_suppression_truth"] for row in read_jsonl(input_artifacts["c3b1_pair_truth"]))
    h2 = read_json(C3B2 / "03_EXPECTED_HEIGHT_SURFACES/surface_model_comparison.json")
    if (
        regression["totals"]
        != {
            "control_candidates": 9067,
            "frames": 144,
            "retained_candidates": 6509,
            "suppressed_candidates": 2558,
        }
        or combined["candidate_reviews"]["combined"] != 252
        or combined["whole_scene_reviews"]["combined"] != 36
        or combined["missed_person_neighbourhoods"]["combined_marks"] != 25
        or truth_counts
        != {
            "HUMAN_SAFE_TO_SUPPRESS_INNER": 34,
            "HUMAN_MUST_PROTECT_INNER": 11,
            "AMBIGUOUS": 3,
        }
        or h2["H2_LOCAL_2D_WEIGHTED_MEDIAN"]["coverage"] != 0.9586726071593179
    ):
        raise RuntimeError("FAIL_G7E_A_INPUT_PROVENANCE: frozen closure counts")

    bursts = select_bursts(frames, candidates, sources, provenance_hashes)
    preview_ids = representative_bursts(bursts)
    frame_rows, thumbnails = decode_frame_provenance(bursts, sources, set(preview_ids))

    source_rows = []
    for source in sorted(sources.values(), key=lambda item: (item.match_id, item.half)):
        source_rows.append(
            {
                "match_id": source.match_id,
                "half": source.half,
                "relative_path": source.relative_path,
                "sha256": source.sha256,
                "byte_size": source.byte_size,
                "fps": float(source.fps),
                "frame_count": source.frame_count,
                "duration_seconds": float(source.duration_seconds),
                "width": source.width,
                "height": source.height,
                "validation": "POLYGON_REFERENCE_EQUALS_FROZEN_SOURCE_MANIFEST_AND_FILE_SIZE",
                "source_video_rehashed_this_stage": False,
                "reason": "immutable source hash already bound by frozen manifests; no redundant multi-gigabyte rehash",
            }
        )
    input_closure = {
        "schema_version": "football_intelligence.g7e_a.input_closure.v1",
        "classification": "PASS_G7E_A_INPUT_PROVENANCE",
        "repository_head_before_changes": EXPECTED_HEAD,
        "prompt_pack_valid": True,
        "matches": list(MATCHES),
        "split": "TRAIN_DEVELOPMENT",
        "polygon_status": "HUMAN_CONFIRMED",
        "camera_policy": "MATCH_STABLE_CAMERA",
        "frozen_closure": regression["totals"],
        "human_evidence": {
            "candidate_labels": 252,
            "scene_reviews": 36,
            "missed_person_marks": 25,
            "nested_pair_reviews": 48,
            "nested_safe": 34,
            "nested_must_protect": 11,
            "nested_ambiguous": 3,
        },
        "h2": {
            "model": "H2_LOCAL_2D_WEIGHTED_MEDIAN",
            "valid_coverage": 0.9586726071593179,
            "reference_candidates": 4932,
            "rule_activated": False,
            "use": "PRESERVED_CONTEXT_ONLY",
        },
        "source_videos": source_rows,
        "input_artifacts": {name: source_artifact(path) for name, path in input_artifacts.items()},
        "validation_or_holdout_access": False,
        "production_ready": False,
    }
    write_json(STAGE / "00_INPUT_CLOSURE/input_closure.json", input_closure)
    selection_contract = {
        "schema_version": "football_intelligence.g7e_a.selection_contract.v1",
        "contract_id": "G7E_A_TARGETED_TEMPORAL_BURST_SELECTION_V1",
        "matches": list(MATCHES),
        "bursts": 120,
        "bursts_per_match": 20,
        "frames_per_burst": 9,
        "frame_references": 1080,
        "offset_seconds": [float(offset) for offset in OFFSETS_SECONDS],
        "per_match_quotas": QUOTAS,
        "class_collision_priority": list(CLASS_PRIORITY),
        "selection_source_priority": [
            "HUMAN_MUST_PROTECT_OR_AMBIGUOUS_NESTED_PAIR",
            "HUMAN_SAFE_NESTED_FRAGMENT_OR_DUPLICATE",
            "HUMAN_MISSED_PERSON_MARK",
            "HUMAN_SCENE_EDGE_CASE",
            "C2_B3_FROZEN_DIAGNOSTIC",
            "SIX_MATCH_FROZEN_FRAME_PROXY",
        ],
        "companion_rule": {
            "offset_seconds": 1.2,
            "earlier_first": True,
            "minimum_centre_separation_seconds": 1.0,
            "maximum_shared_frames": 4,
            "maximum_per_match": 6,
        },
        "frame_resolution": "NEAREST_FRAME_ROUND_HALF_UP_AT_CANONICAL_FPS",
        "frame_pixel_hash": "SHA256_RGB24_C_CONTIGUOUS_SOURCE_DIMENSIONS",
        "selection_classes_are_human_truth": False,
        "blind_payload_excludes_human_answers": True,
        "permanent_identity_forbidden": True,
        "team_classification_excluded": True,
        "production_ready": False,
    }
    write_json(STAGE / "01_SELECTION_AND_ANNOTATION_CONTRACT/selection_contract.json", selection_contract)
    write_json(
        STAGE / "01_SELECTION_AND_ANNOTATION_CONTRACT/decision.json",
        {
            "decision": DECISION,
            "burst_manifest_frozen": True,
            "annotation_protocol_frozen": True,
            "human_annotations_created": False,
            "reviewer_implemented": False,
            "inference_run": False,
            "runtime_or_default_changed": False,
            "production_ready": False,
            "stop_before": "G7E_B_TEMPORAL_REVIEWER_IMPLEMENTATION",
        },
    )
    burst_path = write_jsonl(STAGE / "02_BURST_SELECTION/temporal_burst_manifest.jsonl", bursts)
    frame_path = write_jsonl(STAGE / "02_BURST_SELECTION/temporal_frame_manifest.jsonl", frame_rows)
    quota_path = write_json(STAGE / "02_BURST_SELECTION/selection_quota_report.json", quota_report(bursts))
    provenance_path = write_json(
        STAGE / "02_BURST_SELECTION/selection_provenance_report.json",
        {
            "schema_version": "football_intelligence.g7e_a.selection_provenance_report.v1",
            "input_artifact_hashes": provenance_hashes,
            "source_video_count": 12,
            "source_video_paths_and_hashes": source_rows,
            "frozen_anchor_frame_count": 144,
            "selected_bursts": 120,
            "selected_frame_references": 1080,
            "unique_decoded_source_frames": len(
                {(row["source_video_relative_path"], row["frame_index_zero_based"]) for row in frame_rows}
            ),
            "pixel_hash_contract": "SHA256_RGB24_C_CONTIGUOUS_SOURCE_DIMENSIONS",
            "selection_uses_future_temporal_truth": False,
            "protected_human_provenance_not_in_blind_payload": True,
            "source_file_mutation": False,
            "production_ready": False,
        },
    )
    write_json(
        STAGE / "02_BURST_SELECTION/burst_manifest_sha256.json",
        {
            "schema_version": "football_intelligence.g7e_a.burst_manifest_sha256.v1",
            "files": [artifact(path) for path in (burst_path, frame_path, quota_path, provenance_path)],
            "self_hashed": False,
        },
    )

    ontology = annotation_ontology()
    if not validate_ontology(ontology["enumerations"]):
        raise RuntimeError("FAIL_G7E_A_ANNOTATION_PROTOCOL: ontology")
    write_json(STAGE / "03_ANNOTATION_PROTOCOL/temporal_annotation_ontology.json", ontology)
    write_json(STAGE / "03_ANNOTATION_PROTOCOL/temporal_event_schema_draft.json", event_schema_draft())
    workflow_path = STAGE / "03_ANNOTATION_PROTOCOL/reviewer_workflow_design.md"
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.write_text(WORKFLOW, encoding="utf-8")
    write_json(
        STAGE / "03_ANNOTATION_PROTOCOL/human_workload_estimate.json",
        {
            "burst_count": 120,
            "frames_per_burst": 9,
            "focus_subjects_per_burst": [1, 3],
            "per_burst_minutes": {"low": 2.0, "median": 3.0, "high": 4.0},
            "total_hours": {"low": 4.0, "median": 6.0, "high": 8.0},
            "target_met": True,
            "whole_burst_check_each_burst": True,
            "team_classification_questions": 0,
            "identity_questions": 0,
        },
    )

    draw_selection_matrix(bursts, STAGE / "04_VISUAL_QA/01_TEMPORAL_SELECTION_MATRIX.png")
    draw_burst_strips(
        bursts,
        preview_ids,
        thumbnails,
        STAGE / "04_VISUAL_QA/02_REPRESENTATIVE_BURST_STRIPS.png",
    )
    write_json(
        STAGE / "05_TESTS_AND_LOGS/build_validation_report.json",
        {
            "decision": DECISION,
            "burst_validation": validate_burst_records(bursts),
            "frame_references": len(frame_rows),
            "all_frame_pixel_hashes_present": all(len(row["frame_pixel_sha256"]) == 64 for row in frame_rows),
            "full_resolution_images_saved": 0,
            "visual_count": 2,
            "inference_run": False,
            "human_annotation_events_created": False,
            "reviewer_implemented": False,
            "production_ready": False,
        },
    )
    refresh_handoff()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-handoff", action="store_true")
    parser.add_argument("--refresh-visuals", action="store_true")
    args = parser.parse_args()
    if args.refresh_visuals:
        refresh_visuals()
    elif args.refresh_handoff:
        refresh_handoff()
    else:
        build()


if __name__ == "__main__":
    main()
