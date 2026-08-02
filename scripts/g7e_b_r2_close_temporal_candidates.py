"""Close exact frame-local proposal candidates for every frozen G7E temporal frame."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import torch
from football_intelligence.detection_gold.consolidation import consolidate_proposals
from football_intelligence.g7d_b1_foldwise_runtime import frame_local_candidate_id, proposal_view_plan
from football_intelligence.proposal_gate_hook import apply_shadow_hook


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
EXPECTED_HEAD = "d2817306662cfef41b9e403533c0dd1667c4538a"
MATCH_IDS = ("117092", "117093", "118575", "118576", "118577", "128058")
DETECTOR = REPO / "models/model=yolov8m-imgsz=2048.pt"
VALIDATED_SOURCE_VIDEOS: dict[Path, str] = {}
DETECTOR_SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
GATE_ID = "G3_CONSERVATIVE_FAR_OUTSIDE__fixed_08"
STAGE = (
    PROJECT
    / "experiments/football_observation_reasoner/part 7/G7E_B_R2_FULL_TEMPORAL_CANDIDATE_CLOSURE_AND_REVIEWER_REPAIR_v1"
)
G7EA = (
    PROJECT
    / "experiments/football_observation_reasoner/part 7/G7E_A_TARGETED_TEMPORAL_BURST_SELECTION_AND_ANNOTATION_DESIGN_v1"
)
R1 = PROJECT / "experiments/football_observation_reasoner/part 7/G7E_B_R1_SUBJECT_GUIDANCE_AND_ZOOM_USABILITY_REPAIR_v1"
B0 = PROJECT / "experiments/football_observation_reasoner/part 7/G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1"
B1 = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B1_PROPOSAL_CLOSURE_AND_FOLDWISE_RUNTIME_v1"
B2C = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B2C_RESUME_FROZEN_128058_BASELINE_v1"
B3 = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B3_FROZEN_CROSS_MATCH_REPLAY_v1"
C3A1 = PROJECT / "experiments/football_observation_reasoner/part 7/G7D_C3A1_PITCH_GATE_SHADOW_INTEGRATION_REVIEW_v1"
C3A5C = PROJECT / "experiments/football_observation_reasoner/part 7/G7D_C3A5C_ADDITIONAL_COVERAGE_REPLAY_AND_REVIEW_v1"
C3A6 = PROJECT / "experiments/football_observation_reasoner/part 7/G7D_C3A6_DEVELOPMENT_ONLY_DEFAULT_ACTIVATION_v1"
FRAME_MANIFEST = G7EA / "02_BURST_SELECTION/temporal_frame_manifest.jsonl"
GATE_CONTRACT = C3A1 / "01_INTEGRATION_CONTRACT/pitch_gate_shadow_contract.json"
POLICY_CONTRACT = C3A6 / "01_DEVELOPMENT_POLICY_CONTRACT/development_pitch_gate_policy.json"
SPLIT = PROJECT / "datasets/soccertrack_v2/splits/split_v1/split_manifest.json"
SUCCESS = "PASS_G7E_B_R2_FULL_TEMPORAL_CANDIDATE_CLOSURE_READY_FOR_PRACTICE_REVIEW"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(b"".join(canonical_bytes(value) for value in values))
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "project_relative_path": str(path.relative_to(PROJECT)).replace("\\", "/"),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pixel_sha256(decoded_bgr: Any) -> str:
    rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    return hashlib.sha256(rgb.tobytes(order="C")).hexdigest()


def canonical_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["match_id"]),
        str(row["half"]),
        str(row["source_video_sha256"]),
        int(row["frame_index_zero_based"]),
        float(row["resolved_timestamp_seconds"]),
        int(row["source_width"]),
        int(row["source_height"]),
        str(row["frame_pixel_sha256"]),
    )


def source_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["match_id"]),
        str(row["half"]),
        str(row["source_video_sha256"]),
        int(row["frame_index_zero_based"]),
        int(row["source_width"]),
        int(row["source_height"]),
    )


def candidate_ordinal(row: Mapping[str, Any]) -> int:
    if "candidate_ordinal" in row:
        return int(row["candidate_ordinal"])
    return int(str(row["candidate_local_id"]).rsplit("_", 1)[-1])


def r1_event_preflight() -> dict[str, Any]:
    package = R1 / "02_SUBJECT_GUIDANCE_IMPLEMENTATION/temporal_reviewer_r1"
    real_root = package / "human_decisions"
    counts = {
        "human_events": len(list((real_root / "events").glob("*/*.json"))) if (real_root / "events").exists() else 0,
        "acknowledgements": len(list((real_root / "receipts/acknowledgements").glob("*.json")))
        if (real_root / "receipts/acknowledgements").exists()
        else 0,
        "tranche_receipts": len(list((real_root / "receipts/tranche_completion").glob("*.json")))
        if (real_root / "receipts/tranche_completion").exists()
        else 0,
        "completion_receipts": len(list((real_root / "receipts/global_completion").glob("*.json")))
        if (real_root / "receipts/global_completion").exists()
        else 0,
    }
    if any(counts.values()):
        raise RuntimeError("FAIL_G7E_B_R2_REAL_EVENT_PREFLIGHT")
    practice = B0 / "03_TEMPORAL_REVIEWER/practice_decisions"
    practice_files = sorted(path for path in practice.rglob("*.json") if path.is_file()) if practice.exists() else []
    return {
        "r1_decision": read_json(R1 / "decision.json")["decision"],
        "real_human_state": counts,
        "practice_file_count": len(practice_files),
        "practice_files": [artifact(path) for path in practice_files],
        "practice_draft_policy": "INCOMPATIBLE_PRE_R2_DRAFT_REQUIRES_VISIBLE_RESET",
        "passed": not any(counts.values()),
    }


def validate_inputs() -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if head != EXPECTED_HEAD:
        raise RuntimeError("FAIL_G7E_B_R2_RUNTIME_PROVENANCE: HEAD")
    if sha256_file(DETECTOR) != DETECTOR_SHA256:
        raise RuntimeError("FAIL_G7E_B_R2_RUNTIME_PROVENANCE: checkpoint")
    dependency_registry_path = B1 / "01_PROPOSAL_CLOSURE/proposal_dependency_registry.json"
    dependencies = read_json(dependency_registry_path)
    for item in dependencies["artifacts"]:
        if not item.get("required"):
            continue
        path = PROJECT / item["project_relative_path"]
        if not path.is_file() or path.stat().st_size != item["byte_size"] or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"FAIL_G7E_B_R2_RUNTIME_PROVENANCE: {item['logical_name']}")
    runtime_contract_path = B1 / "01_PROPOSAL_CLOSURE/proposal_runtime_contract.json"
    runtime = read_json(runtime_contract_path)
    if runtime["runtime"]["confidence"] != 0.22 or runtime["runtime"]["iou"] != 0.70:
        raise RuntimeError("FAIL_G7E_B_R2_RUNTIME_PROVENANCE: detector thresholds")
    if runtime["runtime"]["checkpoint_sha256"] != DETECTOR_SHA256:
        raise RuntimeError("FAIL_G7E_B_R2_RUNTIME_PROVENANCE: detector contract")
    gate = read_json(GATE_CONTRACT)
    if gate["parent_c3a_gate_id"] != GATE_ID:
        raise RuntimeError("FAIL_G7E_B_R2_RUNTIME_PROVENANCE: gate")
    policy = read_json(POLICY_CONTRACT)
    if policy["policy_id"] != "G7D_C3A6_TRAIN_DEVELOPMENT_PITCH_GATE_DEFAULT_V1":
        raise RuntimeError("FAIL_G7E_B_R2_RUNTIME_PROVENANCE: C3A6 policy")
    split = read_json(SPLIT)
    if split.get("status") != "FROZEN_HUMAN_APPROVED" or split.get("frozen") is not True:
        raise RuntimeError("FAIL_G7E_B_R2_RUNTIME_PROVENANCE: split")
    train = set(split["membership"]["TRAIN_DEVELOPMENT"])
    if set(MATCH_IDS) - train:
        raise RuntimeError("FAIL_G7E_B_R2_RUNTIME_PROVENANCE: development membership")
    polygons: dict[str, Any] = {}
    polygon_hashes: dict[str, str] = {}
    match_reports = []
    for match_id in MATCH_IDS:
        setup_path = PROJECT / f"matches/{match_id}/calibration/match_setup.json"
        setup = read_json(setup_path)
        calibration = setup["pitch_calibration"]
        relative_polygon_path = calibration.get("pitch_polygon_path") or calibration.get("polygon_path")
        if not relative_polygon_path:
            raise RuntimeError(f"FAIL_G7E_B_R2_RUNTIME_PROVENANCE: polygon path {match_id}")
        polygon_path = Path(relative_polygon_path)
        if not polygon_path.is_absolute():
            polygon_path = PROJECT / f"matches/{match_id}" / polygon_path
        polygon_hash = sha256_file(polygon_path)
        expected_polygon_hash = calibration.get("pitch_polygon_sha256") or calibration.get("polygon_sha256")
        if calibration.get("status") != "HUMAN_CONFIRMED" or polygon_hash != expected_polygon_hash:
            raise RuntimeError(f"FAIL_G7E_B_R2_RUNTIME_PROVENANCE: polygon {match_id}")
        polygon = read_json(polygon_path)
        if (
            polygon.get("status") != "HUMAN_CONFIRMED"
            or len(polygon.get("camera_segments", [])) != 1
            or polygon["camera_segments"][0].get("segment_id") != "MATCH_STABLE_CAMERA"
            or polygon.get("production_ready") is not False
        ):
            raise RuntimeError(f"FAIL_G7E_B_R2_RUNTIME_PROVENANCE: polygon geometry {match_id}")
        polygons[match_id] = polygon
        polygon_hashes[match_id] = polygon_hash
        match_reports.append(
            {
                "match_id": match_id,
                "split": "TRAIN_DEVELOPMENT",
                "polygon": artifact(polygon_path),
                "polygon_status": "HUMAN_CONFIRMED",
                "camera_policy": "MATCH_STABLE_CAMERA",
                "production_ready": False,
            }
        )
    event = r1_event_preflight()
    if event["r1_decision"] != "PASS_G7E_B_R1_SUBJECT_GUIDANCE_AND_ZOOM_READY_FOR_PRACTICE_REVIEW":
        raise RuntimeError("FAIL_G7E_B_R2_RUNTIME_PROVENANCE: R1 decision")
    return {
        "head": head,
        "runtime": runtime,
        "runtime_contract": artifact(runtime_contract_path),
        "dependency_registry": artifact(dependency_registry_path),
        "detector_checkpoint": artifact(DETECTOR),
        "gate_contract": artifact(GATE_CONTRACT),
        "policy_contract": artifact(POLICY_CONTRACT),
        "split_manifest": artifact(SPLIT),
        "matches": match_reports,
        "polygons": polygons,
        "polygon_hashes": polygon_hashes,
        "event_preflight": event,
    }


def build_unique_index() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    references = read_jsonl(FRAME_MANIFEST)
    if len(references) != 1080:
        raise RuntimeError("FAIL_G7E_B_R2_UNIQUE_FRAME_INDEX: reference count")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in references:
        groups[canonical_key(row)].append(row)
    if len(groups) != 1044:
        raise RuntimeError("FAIL_G7E_B_R2_UNIQUE_FRAME_INDEX: unique count")
    unique_rows = []
    mapping_rows = []
    for sequence, (key, rows) in enumerate(sorted(groups.items(), key=lambda item: item[0])):
        first = rows[0]
        unique_id = f"temporal_{first['frame_pixel_sha256'][:24]}"
        unique = {
            "schema_version": "football_intelligence.g7e_b_r2.unique_temporal_frame.v1",
            "unique_frame_sequence": sequence,
            "unique_frame_id": unique_id,
            "canonical_key_sha256": sha256_value(list(key)),
            "match_id": first["match_id"],
            "half": first["half"],
            "source_video_relative_path": first["source_video_relative_path"],
            "source_video_sha256": first["source_video_sha256"],
            "frame_index_zero_based": first["frame_index_zero_based"],
            "resolved_timestamp_seconds": first["resolved_timestamp_seconds"],
            "source_width": first["source_width"],
            "source_height": first["source_height"],
            "frame_pixel_sha256": first["frame_pixel_sha256"],
            "frame_reference_count": len(rows),
            "frame_reference_ids": sorted(row["frame_reference_id"] for row in rows),
            "burst_ids": sorted({row["burst_id"] for row in rows}),
        }
        unique_rows.append(unique)
        for row in rows:
            mapping_rows.append(
                {
                    "schema_version": "football_intelligence.g7e_b_r2.frame_reference_mapping.v1",
                    "frame_reference_id": row["frame_reference_id"],
                    "burst_id": row["burst_id"],
                    "burst_frame_sequence": row["burst_frame_sequence"],
                    "unique_frame_id": unique_id,
                    "unique_frame_sequence": sequence,
                    "frame_pixel_sha256": row["frame_pixel_sha256"],
                }
            )
    if len({row["unique_frame_id"] for row in unique_rows}) != 1044:
        raise RuntimeError("FAIL_G7E_B_R2_UNIQUE_FRAME_INDEX: ID collision")
    atomic_jsonl(STAGE / "01_UNIQUE_FRAME_INDEX/unique_temporal_frame_index.jsonl", unique_rows)
    atomic_jsonl(STAGE / "01_UNIQUE_FRAME_INDEX/frame_reference_to_unique_frame.jsonl", mapping_rows)
    atomic_json(
        STAGE / "01_UNIQUE_FRAME_INDEX/unique_frame_index_manifest.json",
        {
            "schema_version": "football_intelligence.g7e_b_r2.unique_frame_index_manifest.v1",
            "canonical_key_fields": [
                "match_id",
                "half",
                "source_video_sha256",
                "frame_index_zero_based",
                "resolved_timestamp_seconds",
                "source_width",
                "source_height",
                "frame_pixel_sha256",
            ],
            "frame_reference_count": len(mapping_rows),
            "unique_frame_count": len(unique_rows),
            "source_manifest": artifact(FRAME_MANIFEST),
            "unique_index_sha256": sha256_file(STAGE / "01_UNIQUE_FRAME_INDEX/unique_temporal_frame_index.jsonl"),
            "reference_mapping_sha256": sha256_file(
                STAGE / "01_UNIQUE_FRAME_INDEX/frame_reference_to_unique_frame.jsonl"
            ),
        },
    )
    return unique_rows, mapping_rows


def old_frame_sources() -> list[tuple[dict[str, Any], str]]:
    rows: list[tuple[dict[str, Any], str]] = []
    for path, label in (
        (B2C / "02_BASELINE_INPUTS/ordered_sampling_manifest.json", "G7D_B2C"),
        (B3 / "02_REPLAY_INPUTS/117092/ordered_sampling_manifest.json", "G7D_B3"),
        (B3 / "02_REPLAY_INPUTS/118575/ordered_sampling_manifest.json", "G7D_B3"),
    ):
        for item in read_json(path)["frames"]:
            rows.append((item, label))
    for item in read_json(C3A5C / "01_FRAME_REPLAY/frame_and_candidate_manifest.json")["frames"]:
        rows.append((item, "G7D_C3A5C"))
    if len(rows) != 144:
        raise RuntimeError("FAIL_G7E_B_R2_REUSE_AUDIT: frozen anchor count")
    return rows


def old_candidates() -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[tuple[str, str], list[dict[str, Any]]]]:
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in (
        B2C / "04_BASELINE_REFERENCE/foldwise_candidate_records.jsonl",
        B3 / "03_REPLAY_RUNTIME/foldwise_candidate_records.jsonl",
    ):
        for row in read_jsonl(path):
            candidates[(str(row["match_id"]), str(row["frame_sha256"]))].append(row)
    c3a5c = read_json(C3A5C / "01_FRAME_REPLAY/frame_and_candidate_manifest.json")
    for row in c3a5c["candidates"]:
        candidates[(str(row["match_id"]), str(row["frame_sha256"]))].append(row)
    decisions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(C3A1 / "02_SHADOW_PARITY/shadow_decisions.jsonl"):
        decisions[(str(row["match_id"]), str(row["frame_sha256"]))].append(row)
    for row in read_jsonl(C3A5C / "02_GATE_RESULTS/gate_decisions.jsonl"):
        decisions[(str(row["match_id"]), str(row["frame_sha256"]))].append(row)
    for values in candidates.values():
        values.sort(key=candidate_ordinal)
    for values in decisions.values():
        values.sort(key=candidate_ordinal)
    return candidates, decisions


def normalized_candidate(row: Mapping[str, Any], unique: Mapping[str, Any], order: int) -> dict[str, Any]:
    provenance = dict(row.get("proposal_provenance", {}))
    score = row.get("score", provenance.get("score"))
    return {
        "schema_version": "football_intelligence.g7e_b_r2.pre_gate_candidate.v1",
        "unique_frame_id": unique["unique_frame_id"],
        "frame_pixel_sha256": unique["frame_pixel_sha256"],
        "candidate_id": row["candidate_local_id"],
        "pre_gate_order": order,
        "source_box_xyxy": [float(value) for value in row["source_box_xyxy"]],
        "footpoint_xy": [float(value) for value in row["approximate_footpoint_xy"]],
        "score": float(score) if score is not None else None,
        "proposal_lineage": provenance,
        "source_detector_record": {
            "checkpoint_sha256": DETECTOR_SHA256,
            "confidence_threshold": 0.22,
            "iou_threshold": 0.70,
        },
        "consolidation_record": {
            "algorithm": "IOU_CONNECTED_COMPONENT_055",
            "merged_gate": True,
            "observation_uuid": row.get("observation_uuid", provenance.get("observation_uuid")),
        },
    }


def write_frame_artifacts(
    unique: Mapping[str, Any],
    pre: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    runtime_source: str,
    runtime_seconds: float,
    source_evidence: Mapping[str, Any],
    view_count: int | None,
) -> dict[str, Any]:
    uid = unique["unique_frame_id"]
    decision_by_id = {str(row["candidate_local_id"]): row for row in decisions}
    if len(decision_by_id) != len(decisions) or set(decision_by_id) != {row["candidate_id"] for row in pre}:
        raise RuntimeError(f"FAIL_G7E_B_R2_CANDIDATE_LINEAGE: {uid}")
    pre_rows = []
    gate_rows = []
    post_rows = []
    retained_order = 0
    for row in pre:
        decision = decision_by_id[row["candidate_id"]]
        gate = {
            "schema_version": "football_intelligence.g7e_b_r2.pitch_gate_decision.v1",
            "unique_frame_id": uid,
            "frame_pixel_sha256": unique["frame_pixel_sha256"],
            "candidate_id": row["candidate_id"],
            "pre_gate_order": row["pre_gate_order"],
            "decision": decision["decision"],
            "reason_codes": decision["reason_codes"],
            "geometry": decision["geometry"],
            "gate_contract_sha256": sha256_file(GATE_CONTRACT),
            "post_gate_retained_order": None,
        }
        enriched = {**row, "pitch_gate_decision": gate["decision"], "post_gate_retained_order": None}
        if gate["decision"] != "SUPPRESS_SANDBOX":
            gate["post_gate_retained_order"] = retained_order
            enriched["post_gate_retained_order"] = retained_order
            post_rows.append(enriched)
            retained_order += 1
        pre_rows.append(dict(row))
        gate_rows.append(gate)
    if len({row["candidate_id"] for row in pre_rows}) != len(pre_rows):
        raise RuntimeError(f"FAIL_G7E_B_R2_DUPLICATE_CANDIDATE_ID: {uid}")
    paths = {
        "pre": STAGE / f"03_TEMPORAL_PROPOSAL_RUNTIME/temporal_pre_gate_candidates/{uid}.json",
        "gate": STAGE / f"03_TEMPORAL_PROPOSAL_RUNTIME/temporal_pitch_gate_decisions/{uid}.json",
        "post": STAGE / f"03_TEMPORAL_PROPOSAL_RUNTIME/temporal_post_gate_candidates/{uid}.json",
        "record": STAGE / f"03_TEMPORAL_PROPOSAL_RUNTIME/frame_completion_records/{uid}.json",
    }
    atomic_json(paths["pre"], {"unique_frame": dict(unique), "candidates": pre_rows})
    atomic_json(paths["gate"], {"unique_frame_id": uid, "decisions": gate_rows})
    atomic_json(paths["post"], {"unique_frame_id": uid, "candidates": post_rows})
    record = {
        "schema_version": "football_intelligence.g7e_b_r2.per_frame_runtime_record.v1",
        "unique_frame_id": uid,
        "frame_pixel_sha256": unique["frame_pixel_sha256"],
        "runtime_source": runtime_source,
        "runtime_completed": True,
        "runtime_execution_count": 0 if runtime_source == "REUSED_HASH_EXACT" else 1,
        "runtime_seconds": runtime_seconds,
        "proposal_view_count": view_count,
        "pre_gate_candidate_count": len(pre_rows),
        "post_gate_candidate_count": len(post_rows),
        "pitch_gate_suppression_count": len(pre_rows) - len(post_rows),
        "source_evidence": dict(source_evidence),
        "pre_gate_artifact": artifact(paths["pre"]),
        "gate_decision_artifact": artifact(paths["gate"]),
        "post_gate_artifact": artifact(paths["post"]),
        "crop_features_executed": False,
        "semantic_folds_executed": False,
    }
    atomic_json(paths["record"], record)
    return record


def audit_and_materialize_reuse(
    unique_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_source = {source_key(row): row for row in unique_rows}
    frame_sources = old_frame_sources()
    candidates_by_frame, decisions_by_frame = old_candidates()
    reused: dict[str, dict[str, Any]] = {}
    audit_rows = []
    for old, label in frame_sources:
        key = (
            str(old["match_id"]),
            str(old["half"]),
            str(old["source_video_sha256"]),
            int(old["frame_index_zero_based"]),
            int(old["source_width"]),
            int(old["source_height"]),
        )
        unique = by_source.get(key)
        if unique is None:
            continue
        image_path = Path(old.get("path") or (PROJECT / old["project_relative_path"]))
        if not image_path.is_absolute():
            image_path = PROJECT / image_path
        decoded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if decoded is None or pixel_sha256(decoded) != unique["frame_pixel_sha256"]:
            raise RuntimeError(f"FAIL_G7E_B_R2_REUSE_AUDIT: frame pixels {unique['unique_frame_id']}")
        frame_hash = old["frame_sha256"]
        candidate_rows = candidates_by_frame[(str(old["match_id"]), str(frame_hash))]
        decision_rows = decisions_by_frame[(str(old["match_id"]), str(frame_hash))]
        if len(candidate_rows) != len(decision_rows):
            raise RuntimeError(f"FAIL_G7E_B_R2_REUSE_AUDIT: candidate parity {unique['unique_frame_id']}")
        pre = [normalized_candidate(row, unique, index) for index, row in enumerate(candidate_rows)]
        record = write_frame_artifacts(
            unique,
            pre,
            decision_rows,
            "REUSED_HASH_EXACT",
            0.0,
            {
                "source_stage": label,
                "source_frame_artifact": artifact(image_path),
                "candidate_artifact_frame_sha256": frame_hash,
                "candidate_id_and_order_preserved": True,
                "compatibility_basis": "C3A1/C3A5C exact frozen proposal and gate parity",
            },
            None,
        )
        reused[unique["unique_frame_id"]] = record
        audit_rows.append(
            {
                "schema_version": "football_intelligence.g7e_b_r2.candidate_reuse.v1",
                "unique_frame_id": unique["unique_frame_id"],
                "reuse_status": "REUSED_HASH_EXACT",
                "source_stage": label,
                "source_frame_pixel_sha256": unique["frame_pixel_sha256"],
                "source_video_sha256": unique["source_video_sha256"],
                "frame_index_zero_based": unique["frame_index_zero_based"],
                "source_dimensions": [unique["source_width"], unique["source_height"]],
                "detector_checkpoint_sha256": DETECTOR_SHA256,
                "proposal_runtime_contract_sha256": sha256_file(
                    B1 / "01_PROPOSAL_CLOSURE/proposal_runtime_contract.json"
                ),
                "gate_contract_sha256": sha256_file(GATE_CONTRACT),
                "pre_gate_candidate_count": record["pre_gate_candidate_count"],
                "post_gate_candidate_count": record["post_gate_candidate_count"],
            }
        )
    missing = [row for row in unique_rows if row["unique_frame_id"] not in reused]
    if len(reused) != 108 or len(missing) != 936:
        raise RuntimeError(f"FAIL_G7E_B_R2_REUSE_AUDIT: expected 108/936, got {len(reused)}/{len(missing)}")
    atomic_jsonl(
        STAGE / "02_EXISTING_CANDIDATE_REUSE/candidate_reuse_index.jsonl",
        sorted(audit_rows, key=lambda row: row["unique_frame_id"]),
    )
    atomic_json(
        STAGE / "02_EXISTING_CANDIDATE_REUSE/reuse_compatibility_report.json",
        {
            "schema_version": "football_intelligence.g7e_b_r2.reuse_compatibility_report.v1",
            "authoritative_frozen_anchor_frames_searched": 144,
            "exact_unique_frames_reused": len(reused),
            "missing_unique_frames": len(missing),
            "compatibility_fields": [
                "frame_pixel_sha256",
                "source_video_sha256",
                "frame_index_zero_based",
                "source_dimensions",
                "detector_checkpoint_sha256",
                "detector_runtime_contract",
                "confidence_iou",
                "preprocessing",
                "proposal_schema",
                "consolidation_contract",
                "candidate_order_contract",
            ],
            "rendered_overlays_used_as_candidate_data": False,
            "post_gate_only_records_reinterpreted": False,
            "passed": True,
        },
    )
    atomic_json(
        STAGE / "02_EXISTING_CANDIDATE_REUSE/missing_unique_frames.json",
        {
            "schema_version": "football_intelligence.g7e_b_r2.missing_unique_frames.v1",
            "frozen_before_new_inference": True,
            "missing_unique_frame_count": len(missing),
            "frames": missing,
        },
    )
    return reused, missing


def gpu_preflight() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("FAIL_G7E_B_R2_CUDA_PREFLIGHT: unavailable")
    name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory
    if "NVIDIA GeForce RTX 5060 Laptop GPU" not in name or memory < int(7.5 * 1024**3):
        raise RuntimeError("FAIL_G7E_B_R2_CUDA_PREFLIGHT: wrong device")
    return {
        "torch_cuda_available": True,
        "device": "cuda:0",
        "device_name": name,
        "total_memory_bytes": memory,
        "detector_dtype": "fp16",
        "detector_batch_size": 1,
        "cpu_or_intel_fallback": False,
    }


def decode_exact(unique: Mapping[str, Any]) -> Any:
    video = PROJECT / unique["source_video_relative_path"]
    expected_video_sha256 = str(unique["source_video_sha256"])
    validated = VALIDATED_SOURCE_VIDEOS.get(video)
    if validated is None:
        validated = sha256_file(video)
        if validated != expected_video_sha256:
            raise RuntimeError(f"FAIL_G7E_B_R2_SOURCE_HASH: {unique['unique_frame_id']}")
        VALIDATED_SOURCE_VIDEOS[video] = validated
    elif validated != expected_video_sha256:
        raise RuntimeError(f"FAIL_G7E_B_R2_SOURCE_HASH_CONTRACT: {unique['unique_frame_id']}")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"FAIL_G7E_B_R2_FRAME_DECODE: {unique['unique_frame_id']}")
    try:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(unique["frame_index_zero_based"])):
            raise RuntimeError(f"FAIL_G7E_B_R2_FRAME_DECODE: seek {unique['unique_frame_id']}")
        okay, decoded = capture.read()
        actual = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
    finally:
        capture.release()
    if not okay or actual != int(unique["frame_index_zero_based"]):
        raise RuntimeError(f"FAIL_G7E_B_R2_FRAME_DECODE: exact index {unique['unique_frame_id']}")
    if decoded.shape[:2] != (int(unique["source_height"]), int(unique["source_width"])):
        raise RuntimeError(f"FAIL_G7E_B_R2_FRAME_DECODE: dimensions {unique['unique_frame_id']}")
    if pixel_sha256(decoded) != unique["frame_pixel_sha256"]:
        raise RuntimeError(f"FAIL_G7E_B_R2_FRAME_DECODE: pixels {unique['unique_frame_id']}")
    return decoded


def query_gpu_snapshot() -> dict[str, Any]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    name, total, used, utilization, temperature = [part.strip() for part in output.split(",")]
    return {
        "name": name,
        "memory_total_mib": int(total),
        "memory_used_mib": int(used),
        "utilization_percent": int(utilization),
        "temperature_celsius": int(temperature),
    }


def run_missing(
    missing: list[dict[str, Any]],
    inputs: Mapping[str, Any],
    *,
    worker_index: int = 0,
    worker_count: int = 1,
) -> list[dict[str, Any]]:
    device = gpu_preflight()
    atomic_json(STAGE / "03_TEMPORAL_PROPOSAL_RUNTIME/runtime_device_preflight.json", device)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(STAGE / "_runtime/ultralytics_config"))
    torch.use_deterministic_algorithms(True, warn_only=True)
    g0 = load_module("g7e_b_r2_g0", REPO / "scripts/build_m5_5g0_detection_forensics.py")
    g6e = load_module("g7e_b_r2_g6e", REPO / "scripts/build_m5_5g6e_c0_reintegration.py")
    gate_hash = sha256_file(GATE_CONTRACT)
    records = []
    started_all = time.perf_counter()
    gpu_before = query_gpu_snapshot()
    torch.cuda.reset_peak_memory_stats(0)
    persistent = STAGE / f"_runtime/persistent_runner_{worker_index:02d}"
    persistent.mkdir(parents=True, exist_ok=True)
    runner = g0.DiagnosticRunner(persistent / "raw.jsonl", persistent / "post.jsonl", persistent / "nms.jsonl")
    try:
        for position, unique in enumerate(missing, start=1):
            records.extend(
                _run_one_missing(unique, position, len(missing), started_all, runner, g6e, gate_hash, inputs)
            )
    finally:
        runner.close()
    for path in (persistent / "raw.jsonl", persistent / "post.jsonl", persistent / "nms.jsonl"):
        path.unlink(missing_ok=True)
    persistent.rmdir()
    atomic_json(
        STAGE / f"03_TEMPORAL_PROPOSAL_RUNTIME/worker_runtime_{worker_index:02d}.json",
        {
            "schema_version": "football_intelligence.g7e_b_r2.worker_runtime.v1",
            "worker_index": worker_index,
            "worker_count": worker_count,
            "assigned_frame_count": len(missing),
            "completed_or_reused_frame_count": len(records),
            "elapsed_seconds": round(time.perf_counter() - started_all, 6),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            "gpu_before": gpu_before,
            "gpu_after": query_gpu_snapshot(),
        },
    )
    return records


def _run_one_missing(
    unique: dict[str, Any],
    position: int,
    total: int,
    started_all: float,
    runner: Any,
    g6e: Any,
    gate_hash: str,
    inputs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    uid = unique["unique_frame_id"]
    completion = STAGE / f"03_TEMPORAL_PROPOSAL_RUNTIME/frame_completion_records/{uid}.json"
    if completion.exists():
        record = read_json(completion)
        if record.get("runtime_completed") is not True or record.get("runtime_execution_count") != 1:
            raise RuntimeError(f"FAIL_G7E_B_R2_PARTIAL_FRAME: {uid}")
        (STAGE / f"03_TEMPORAL_PROPOSAL_RUNTIME/failed_frames/{uid}.json").unlink(missing_ok=True)
        records.append(record)
        return records
    frame_started = time.perf_counter()
    temporary = STAGE / "_runtime/proposals" / uid
    temporary.mkdir(parents=True, exist_ok=True)
    try:
        decoded = decode_exact(unique)
        image_path = temporary / "frame.png"
        if not cv2.imwrite(str(image_path), decoded, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError(f"FAIL_G7E_B_R2_FRAME_WRITE: {uid}")
        image_sha = sha256_file(image_path)
        view_start = len(runner.views)
        post_start = len(runner.post_rows)
        for view in proposal_view_plan(int(unique["source_width"]), int(unique["source_height"])):
            runner.run_view(
                {
                    "image_path": image_path,
                    "image_sha256": image_sha,
                    "frame_sequence": int(unique["unique_frame_sequence"]),
                    "timestamp_seconds": float(unique["resolved_timestamp_seconds"]),
                },
                view_type=view["view_type"],
                view_suffix=view["view_suffix"],
                imgsz=view["imgsz"],
                crop_bounds=view["crop_bounds_panorama_pixels"],
            )
        frame_views = runner.views[view_start:]
        if not all(
            view.get("status") == "PASS" and view.get("nms_replay_exact") and view.get("coordinate_roundtrip_passed")
            for view in frame_views
        ):
            raise RuntimeError(f"FAIL_G7E_B_R2_PROPOSAL_RUNTIME: {uid}")
        post = runner.post_rows[post_start:]
        runtime_by_view = {
            view["inference_view_id"]: {
                **view,
                "c0_family": view["inference_view_type"],
                "cache_provider": "G7E_B_R2_FROZEN_EXACT",
            }
            for view in frame_views
        }
        normalized = [
            {**item, "c0_family": item["inference_view_type"], "cache_provider": "G7E_B_R2_FROZEN_EXACT"}
            for item in post
            if item["inference_view_type"] in {"S0_FULL_PANORAMA_1280", "S3_OVERLAPPING_HIGH_RESOLUTION_TILES"}
        ]
        nodes = g6e.proposal_nodes({image_sha: normalized}, runtime_by_view)[image_sha]
        observations = sorted(
            consolidate_proposals(nodes, "IOU_CONNECTED_COMPONENT_055", apply_merged_gate=True)["observations"],
            key=lambda item: item["observation_uuid"],
        )
        pre = []
        gate_input = []
        for ordinal, observation in enumerate(observations):
            box = observation["box_panorama_pixels"]
            foot = observation["footpoint_proxy_panorama_pixels"]
            candidate_id = frame_local_candidate_id(image_sha, ordinal)
            proposal = {
                "observation_uuid": observation["observation_uuid"],
                "output_state": observation["output_state"],
                "cluster_member_count": len(observation["cluster_member_proposal_uuids"]),
                "source_views": list(observation.get("all_source_view_ids", [])),
                "provenance_hash": observation["provenance_hash"],
                "score": float(observation["score"]),
            }
            pre.append(
                {
                    "schema_version": "football_intelligence.g7e_b_r2.pre_gate_candidate.v1",
                    "unique_frame_id": uid,
                    "frame_pixel_sha256": unique["frame_pixel_sha256"],
                    "candidate_id": candidate_id,
                    "pre_gate_order": ordinal,
                    "source_box_xyxy": [float(box[key]) for key in ("x1", "y1", "x2", "y2")],
                    "footpoint_xy": [float(foot["x"]), float(foot["y"])],
                    "score": float(observation["score"]),
                    "proposal_lineage": proposal,
                    "source_detector_record": {
                        "checkpoint_sha256": DETECTOR_SHA256,
                        "confidence_threshold": 0.22,
                        "iou_threshold": 0.70,
                        "frame_png_sha256": image_sha,
                    },
                    "consolidation_record": {
                        "algorithm": "IOU_CONNECTED_COMPONENT_055",
                        "merged_gate": True,
                        "observation_uuid": observation["observation_uuid"],
                        "raw_consolidation_input_count": len(nodes),
                    },
                }
            )
            gate_input.append(
                {
                    "candidate_local_id": candidate_id,
                    "candidate_ordinal": ordinal,
                    "source_box_xyxy": pre[-1]["source_box_xyxy"],
                    "approximate_footpoint_xy": pre[-1]["footpoint_xy"],
                    "proposal_provenance": proposal,
                }
            )
        polygon = inputs["polygons"][str(unique["match_id"])]
        _, decisions, _ = apply_shadow_hook(
            gate_input,
            {
                "match_id": unique["match_id"],
                "frame_id": uid,
                "frame_sha256": image_sha,
                "source_width": unique["source_width"],
                "source_height": unique["source_height"],
                "polygon_vertices_source_xy": polygon["vertices_source_xy"],
                "polygon_sha256": inputs["polygon_hashes"][str(unique["match_id"])],
            },
            mode="SHADOW",
            gate_contract_sha256=gate_hash,
        )
        record = write_frame_artifacts(
            unique,
            pre,
            decisions,
            "RAN_FROZEN_PROPOSAL_RUNTIME_ONCE",
            round(time.perf_counter() - frame_started, 6),
            {
                "source_video_relative_path": unique["source_video_relative_path"],
                "source_video_sha256": unique["source_video_sha256"],
                "frame_index_zero_based": unique["frame_index_zero_based"],
                "frame_pixel_sha256": unique["frame_pixel_sha256"],
                "temporary_png_sha256": image_sha,
                "persistent_model_instance": True,
            },
            len(frame_views),
        )
        records.append(record)
        image_path.unlink(missing_ok=True)
    except Exception as exc:
        atomic_json(
            STAGE / f"03_TEMPORAL_PROPOSAL_RUNTIME/failed_frames/{uid}.json",
            {
                "unique_frame_id": uid,
                "failure": type(exc).__name__,
                "message": str(exc),
                "partial_candidates_merged": False,
                "retry_with_changed_settings": False,
            },
        )
        raise
    if position == 1 or position % 5 == 0 or position == total:
        elapsed = time.perf_counter() - started_all
        print(f"G7E_B_R2_PROGRESS {position}/{total} elapsed_seconds={elapsed:.1f}", flush=True)
    return records


def finalize(unique_rows: list[dict[str, Any]], mapping_rows: list[dict[str, Any]]) -> dict[str, Any]:
    record_paths = sorted((STAGE / "03_TEMPORAL_PROPOSAL_RUNTIME/frame_completion_records").glob("*.json"))
    records = [read_json(path) for path in record_paths]
    if len(records) != 1044:
        raise RuntimeError(f"FAIL_G7E_B_R2_CLOSURE: {len(records)}/1044 completed")
    by_id = {row["unique_frame_id"]: row for row in records}
    if len(by_id) != 1044:
        raise RuntimeError("FAIL_G7E_B_R2_CLOSURE: duplicate completion")
    for record, record_path in zip(records, record_paths, strict=True):
        pre_path = PROJECT / record["pre_gate_artifact"]["project_relative_path"]
        pre_payload = read_json(pre_path)
        normalized = [
            {
                key: value
                for key, value in candidate.items()
                if key not in ("pitch_gate_decision", "post_gate_retained_order")
            }
            for candidate in pre_payload["candidates"]
        ]
        if normalized != pre_payload["candidates"]:
            atomic_json(pre_path, {**pre_payload, "candidates": normalized})
            record["pre_gate_artifact"] = artifact(pre_path)
            atomic_json(record_path, record)
    atomic_jsonl(
        STAGE / "03_TEMPORAL_PROPOSAL_RUNTIME/per_frame_runtime_records.jsonl",
        sorted(records, key=lambda row: row["unique_frame_id"]),
    )
    new_times = [float(row["runtime_seconds"]) for row in records if row["runtime_source"] != "REUSED_HASH_EXACT"]
    worker_paths = sorted((STAGE / "03_TEMPORAL_PROPOSAL_RUNTIME").glob("worker_runtime_*.json"))
    workers = [read_json(path) for path in worker_paths]
    temperatures = [
        int(worker[key]["temperature_celsius"]) for worker in workers for key in ("gpu_before", "gpu_after")
    ]
    runtime_summary = {
        "schema_version": "football_intelligence.g7e_b_r2.runtime_summary.v1",
        "unique_frames_reused": sum(row["runtime_source"] == "REUSED_HASH_EXACT" for row in records),
        "unique_frames_newly_inferred": sum(row["runtime_source"] != "REUSED_HASH_EXACT" for row in records),
        "total_inference_seconds": round(sum(new_times), 6),
        "median_per_frame_seconds": round(statistics.median(new_times), 6),
        "p95_per_frame_seconds": round(sorted(new_times)[int(0.95 * (len(new_times) - 1))], 6),
        "pre_gate_candidate_count": sum(row["pre_gate_candidate_count"] for row in records),
        "post_gate_candidate_count": sum(row["post_gate_candidate_count"] for row in records),
        "pitch_gate_suppression_count": sum(row["pitch_gate_suppression_count"] for row in records),
        "candidate_artifact_disk_size_bytes": sum(
            int(row[key]["byte_size"])
            for row in records
            for key in ("pre_gate_artifact", "gate_decision_artifact", "post_gate_artifact")
        ),
        "deterministic_worker_count": len(workers),
        "worker_elapsed_seconds": [float(worker["elapsed_seconds"]) for worker in workers],
        "parallel_wall_elapsed_seconds": max((float(worker["elapsed_seconds"]) for worker in workers), default=0.0),
        "cuda_peak_allocated_bytes": max((int(worker["cuda_peak_allocated_bytes"]) for worker in workers), default=0),
        "cuda_peak_reserved_bytes": max((int(worker["cuda_peak_reserved_bytes"]) for worker in workers), default=0),
        "gpu_temperature_before_celsius": [int(worker["gpu_before"]["temperature_celsius"]) for worker in workers],
        "gpu_temperature_after_celsius": [int(worker["gpu_after"]["temperature_celsius"]) for worker in workers],
        "gpu_temperature_min_celsius": min(temperatures, default=None),
        "gpu_temperature_max_celsius": max(temperatures, default=None),
        "crop_features_executed": False,
        "semantic_folds_executed": False,
        "production_ready": False,
    }
    atomic_json(STAGE / "03_TEMPORAL_PROPOSAL_RUNTIME/runtime_summary.json", runtime_summary)
    statuses = []
    for unique in unique_rows:
        record = by_id[unique["unique_frame_id"]]
        post_path = PROJECT / record["post_gate_artifact"]["project_relative_path"]
        state = "VERIFIED_CANDIDATES_AVAILABLE" if record["post_gate_candidate_count"] else "VERIFIED_ZERO_CANDIDATES"
        statuses.append(
            {
                "schema_version": "football_intelligence.g7e_b_r2.candidate_status.v1",
                "unique_frame_id": unique["unique_frame_id"],
                "frame_pixel_sha256": unique["frame_pixel_sha256"],
                "candidate_status": state,
                "runtime_source": record["runtime_source"],
                "runtime_completed": True,
                "pre_gate_candidate_count": record["pre_gate_candidate_count"],
                "post_gate_candidate_count": record["post_gate_candidate_count"],
                "verified_zero_reason": (
                    None
                    if state != "VERIFIED_ZERO_CANDIDATES"
                    else (
                        "PRE_GATE_ZERO"
                        if record["pre_gate_candidate_count"] == 0
                        else "ALL_PRE_GATE_CANDIDATES_SUPPRESSED"
                    )
                ),
                "pre_gate_artifact": record["pre_gate_artifact"],
                "gate_decision_artifact": record["gate_decision_artifact"],
                "post_gate_artifact": record["post_gate_artifact"],
                "post_gate_artifact_hash_valid": sha256_file(post_path) == record["post_gate_artifact"]["sha256"],
                "annotation_allowed": True,
                "box_dependent_answers_enabled": state == "VERIFIED_CANDIDATES_AVAILABLE",
            }
        )
    status_path = STAGE / "04_CANDIDATE_CLOSURE/temporal_candidate_status.jsonl"
    atomic_jsonl(status_path, statuses)
    status_counts = Counter(row["candidate_status"] for row in statuses)
    closure = {
        "schema_version": "football_intelligence.g7e_b_r2.candidate_closure_summary.v1",
        "decision": SUCCESS,
        "unique_frame_count": 1044,
        "verified_unique_frame_count": len(statuses),
        "verified_available_frame_count": status_counts["VERIFIED_CANDIDATES_AVAILABLE"],
        "verified_zero_frame_count": status_counts["VERIFIED_ZERO_CANDIDATES"],
        "candidate_data_unavailable_frame_count": status_counts["CANDIDATE_DATA_UNAVAILABLE"],
        **runtime_summary,
    }
    atomic_json(STAGE / "04_CANDIDATE_CLOSURE/candidate_closure_summary.json", closure)
    atomic_json(
        STAGE / "04_CANDIDATE_CLOSURE/candidate_closure_manifest.json",
        {
            "schema_version": "football_intelligence.g7e_b_r2.candidate_closure_manifest.v1",
            "candidate_status": artifact(status_path),
            "runtime_summary": artifact(STAGE / "03_TEMPORAL_PROPOSAL_RUNTIME/runtime_summary.json"),
            "verified_unique_frame_count": 1044,
            "unavailable_count": 0,
        },
    )
    status_by_id = {row["unique_frame_id"]: row for row in statuses}
    review_mapping = []
    for row in mapping_rows:
        status = status_by_id[row["unique_frame_id"]]
        review_mapping.append(
            {
                **row,
                "candidate_status": status["candidate_status"],
                "post_gate_candidate_count": status["post_gate_candidate_count"],
                "post_gate_artifact": status["post_gate_artifact"],
                "unique_frame_candidate_status_path": str(status_path.relative_to(PROJECT)).replace("\\", "/"),
                "unique_frame_candidate_status_sha256": sha256_file(status_path),
            }
        )
    mapping_path = STAGE / "05_REVIEWER_CANDIDATE_MAPPING/review_frame_candidate_mapping.jsonl"
    atomic_jsonl(mapping_path, review_mapping)
    atomic_json(
        STAGE / "05_REVIEWER_CANDIDATE_MAPPING/review_candidate_asset_manifest.json",
        {
            "schema_version": "football_intelligence.g7e_b_r2.review_candidate_asset_manifest.v1",
            "reference_count": len(review_mapping),
            "unique_frame_count": 1044,
            "candidate_runtime_contract": artifact(B1 / "01_PROPOSAL_CLOSURE/proposal_runtime_contract.json"),
            "candidate_status": artifact(status_path),
            "reference_mapping": artifact(mapping_path),
            "candidate_root": str(
                (STAGE / "03_TEMPORAL_PROPOSAL_RUNTIME/temporal_post_gate_candidates").relative_to(PROJECT)
            ).replace("\\", "/"),
            "gate_decisions_hidden_from_blind_reviewer": True,
        },
    )
    atomic_json(
        STAGE / "05_REVIEWER_CANDIDATE_MAPPING/mapping_validation_report.json",
        {
            "frame_references_mapped": len(review_mapping),
            "expected_frame_references": 1080,
            "unique_frames_verified": 1044,
            "unavailable_references": 0,
            "one_immutable_artifact_per_reused_unique_frame": True,
            "passed": len(review_mapping) == 1080 and not status_counts["CANDIDATE_DATA_UNAVAILABLE"],
        },
    )
    runtime_files = sorted(
        path
        for path in (STAGE / "03_TEMPORAL_PROPOSAL_RUNTIME").rglob("*")
        if path.is_file() and path.suffix in (".json", ".jsonl") and path.name != "runtime_artifact_manifest.json"
    )
    atomic_json(
        STAGE / "03_TEMPORAL_PROPOSAL_RUNTIME/runtime_artifact_manifest.json",
        {
            "schema_version": "football_intelligence.g7e_b_r2.runtime_artifact_manifest.v1",
            "file_count": len(runtime_files),
            "files": [artifact(path) for path in runtime_files],
        },
    )
    return closure


def write_input_evidence(inputs: Mapping[str, Any]) -> None:
    target = STAGE / "00_INPUT_AND_RUNTIME_CLOSURE"
    atomic_json(
        target / "input_closure.json",
        {
            "schema_version": "football_intelligence.g7e_b_r2.input_closure.v1",
            "repository_head": inputs["head"],
            "source_frame_manifest": artifact(FRAME_MANIFEST),
            "split_manifest": inputs["split_manifest"],
            "matches": inputs["matches"],
            "real_human_event_count": 0,
            "production_ready": False,
            "passed": True,
        },
    )
    atomic_json(
        target / "proposal_runtime_resolution.json",
        {
            "runtime_id": "G7D_B1_STATIC_FOLDWISE_RUNTIME_V1_PROPOSAL_PATH_ONLY",
            "runtime_contract": inputs["runtime_contract"],
            "dependency_registry": inputs["dependency_registry"],
            "detector_checkpoint": inputs["detector_checkpoint"],
            "checkpoint_sha256_recomputed": sha256_file(DETECTOR),
            "confidence_threshold": 0.22,
            "iou_threshold": 0.70,
            "preprocessing": inputs["runtime"]["runtime"]["views"],
            "consolidation": inputs["runtime"]["runtime"]["consolidation"],
            "candidate_order": "OBSERVATION_UUID_ASCENDING_THEN_FROZEN_FRAME_LOCAL_ID",
            "crop_features_enabled": False,
            "semantic_folds_enabled": False,
            "passed": True,
        },
    )
    atomic_json(
        target / "pitch_gate_runtime_resolution.json",
        {
            "gate_id": GATE_ID,
            "gate_contract": inputs["gate_contract"],
            "development_policy_contract": inputs["policy_contract"],
            "runtime_population": "POST_C3A6_PITCH_GATE_RETAINED",
            "human_labels_used_at_runtime": False,
            "fail_closed": True,
            "passed": True,
        },
    )
    atomic_json(target / "event_root_preflight.json", inputs["event_preflight"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--infer-only", action="store_true")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    args = parser.parse_args()
    if args.worker_count < 1 or not 0 <= args.worker_index < args.worker_count:
        raise RuntimeError("FAIL_G7E_B_R2_WORKER_SHARD")
    inputs = validate_inputs()
    write_input_evidence(inputs)
    unique_rows, mapping_rows = build_unique_index()
    if args.finalize_only:
        print(finalize(unique_rows, mapping_rows)["decision"])
        return 0
    _, missing = audit_and_materialize_reuse(unique_rows)
    if args.prepare_only:
        print(f"PASS_G7E_B_R2_PREPARED reused=108 missing={len(missing)}")
        return 0
    ordered_missing = sorted(
        missing,
        key=lambda row: (
            str(row["source_video_relative_path"]),
            int(row["frame_index_zero_based"]),
            str(row["unique_frame_id"]),
        ),
    )
    source_videos = sorted({str(row["source_video_relative_path"]) for row in ordered_missing})
    worker_videos = {
        path for position, path in enumerate(source_videos) if position % args.worker_count == args.worker_index
    }
    shard = [row for row in ordered_missing if str(row["source_video_relative_path"]) in worker_videos]
    run_missing(shard, inputs, worker_index=args.worker_index, worker_count=args.worker_count)
    if args.infer_only:
        print(f"PASS_G7E_B_R2_WORKER worker={args.worker_index}/{args.worker_count} " f"assigned={len(shard)}")
        return 0
    print(finalize(unique_rows, mapping_rows)["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
