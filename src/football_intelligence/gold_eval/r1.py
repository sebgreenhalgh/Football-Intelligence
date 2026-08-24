from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from football_intelligence.gold_eval.core import (
    GOLD_CORPUS_VERSION,
    GOLD_SCOPE,
    GoldEvalError,
    canonical_sha256,
    read_json,
    read_jsonl,
    sha256_file,
)
from football_intelligence.gold_eval.evaluation import CANDIDATE_RUN_V2


FRAME_INSTANCE_REGISTRY_REVISION = "G7F_A_R1_FULL_SOURCE_FRAME_REGISTRY_V1"
SOURCE_FRAME_REGISTRY_REVISION = "G7F_A_R1_UNIQUE_SOURCE_FRAME_REGISTRY_V1"
EXPECTED_COUNTS = {
    "bursts": 120,
    "frame_instances": 1080,
    "source_frame_hashes": 1044,
    "old_known_hashes": 892,
    "formerly_omitted_hashes": 152,
    "affected_frame_instances": 155,
    "formerly_blocked_candidates": 7575,
    "frozen_candidates": 49803,
}
REGRESSION_FRAME_SHA256 = "4854dec974a57d5f65177268ef055ee12965e72c54ce147fdfd24c7cb2158b48"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GoldEvalError(message)


def registry_schemas() -> dict[str, dict[str, Any]]:
    sha = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    positive_integer = {"type": "integer", "minimum": 1}
    frame_instance_ref = {
        "type": "object",
        "required": ["burst_id", "frame_sequence", "frame_reference_id"],
        "properties": {
            "burst_id": {"type": "string", "minLength": 1},
            "frame_sequence": {"type": "integer", "minimum": 0, "maximum": 8},
            "frame_reference_id": {"type": "string", "minLength": 1},
        },
        "additionalProperties": False,
    }
    frame_instance = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "G7F-A R1 complete frozen frame-instance registry row",
        "type": "object",
        "required": [
            "registry_revision",
            "gold_corpus_version",
            "gold_scope",
            "burst_id",
            "tranche_id",
            "tranche_position",
            "match_id",
            "half",
            "frame_sequence",
            "frame_reference_id",
            "source_frame_sha256",
            "source_width",
            "source_height",
            "frozen_source_asset_identity",
            "source_event_id",
            "source_event_sha256",
            "source_acknowledgement_id",
            "source_acknowledgement_sha256",
            "source_event_schema_version",
            "source_review_revision",
            "source_review_revision_number",
            "frozen_case_lineage",
            "has_reviewed_subject_point",
            "has_missed_observation_point",
            "historical_frozen_candidate_count",
        ],
        "properties": {
            "registry_revision": {"const": FRAME_INSTANCE_REGISTRY_REVISION},
            "gold_corpus_version": {"const": GOLD_CORPUS_VERSION},
            "gold_scope": {"const": GOLD_SCOPE},
            "burst_id": {"type": "string", "minLength": 1},
            "tranche_id": {"type": "string", "minLength": 1},
            "tranche_position": positive_integer,
            "match_id": {"type": "string", "minLength": 1},
            "half": {"type": "string", "minLength": 1},
            "frame_sequence": {"type": "integer", "minimum": 0, "maximum": 8},
            "frame_reference_id": {"type": "string", "minLength": 1},
            "source_frame_sha256": sha,
            "source_width": positive_integer,
            "source_height": positive_integer,
            "frozen_source_asset_identity": {"type": "object"},
            "source_event_id": {"type": "string", "minLength": 1},
            "source_event_sha256": sha,
            "source_acknowledgement_id": {"type": "string", "minLength": 1},
            "source_acknowledgement_sha256": sha,
            "source_event_schema_version": {"type": "string", "minLength": 1},
            "source_review_revision": {"type": "string", "minLength": 1},
            "source_review_revision_number": {"type": ["integer", "null"]},
            "frozen_case_lineage": {"type": "object"},
            "has_reviewed_subject_point": {"type": "boolean"},
            "has_missed_observation_point": {"type": "boolean"},
            "historical_frozen_candidate_count": {"type": "integer", "minimum": 0},
        },
        "additionalProperties": False,
    }
    source_frame = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "G7F-A R1 unique frozen source-frame registry row",
        "type": "object",
        "required": [
            "registry_revision",
            "gold_corpus_version",
            "gold_scope",
            "source_frame_sha256",
            "source_width",
            "source_height",
            "frame_instances",
            "instance_count",
            "frozen_asset_lineage",
            "frozen_candidate_count",
            "historical_frozen_candidate_count_across_instances",
            "has_reviewed_subject_point",
            "has_missed_observation_point",
        ],
        "properties": {
            "registry_revision": {"const": SOURCE_FRAME_REGISTRY_REVISION},
            "gold_corpus_version": {"const": GOLD_CORPUS_VERSION},
            "gold_scope": {"const": GOLD_SCOPE},
            "source_frame_sha256": sha,
            "source_width": positive_integer,
            "source_height": positive_integer,
            "frame_instances": {"type": "array", "minItems": 1, "items": frame_instance_ref},
            "instance_count": positive_integer,
            "frozen_asset_lineage": {"type": "array", "minItems": 1, "items": {"type": "object"}},
            "frozen_candidate_count": {"type": "integer", "minimum": 0},
            "historical_frozen_candidate_count_across_instances": {"type": "integer", "minimum": 0},
            "has_reviewed_subject_point": {"type": "boolean"},
            "has_missed_observation_point": {"type": "boolean"},
        },
        "additionalProperties": False,
    }
    return {
        "gold_frame_instances.schema.json": frame_instance,
        "gold_source_frame_registry.schema.json": source_frame,
        "future_candidate_run_v2.schema.json": candidate_run_v2_schema(),
    }


def candidate_run_v2_schema() -> dict[str, Any]:
    sha = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    processed = {
        "type": "object",
        "required": ["burst_id", "frame_sequence", "source_frame_sha256", "source_width", "source_height"],
        "properties": {
            "burst_id": {"type": "string", "minLength": 1},
            "frame_sequence": {"type": "integer", "minimum": 0, "maximum": 8},
            "source_frame_sha256": sha,
            "source_width": {"type": "integer", "minimum": 1},
            "source_height": {"type": "integer", "minimum": 1},
            "frame_reference_id": {"type": "string", "minLength": 1},
            "processing_provenance": {"type": "object", "minProperties": 1},
        },
        "additionalProperties": False,
    }
    candidate = {
        "type": "object",
        "required": [
            "source_frame_sha256",
            "source_width",
            "source_height",
            "candidate_id",
            "coordinate_space",
            "box_xyxy",
            "confidence",
            "class_label",
            "view_provenance",
        ],
        "properties": {
            "burst_id": {"type": "string", "minLength": 1},
            "frame_sequence": {"type": "integer", "minimum": 0, "maximum": 8},
            "source_frame_sha256": sha,
            "source_width": {"type": "integer", "minimum": 1},
            "source_height": {"type": "integer", "minimum": 1},
            "candidate_id": {"type": "string", "minLength": 1},
            "coordinate_space": {"type": "string", "minLength": 1},
            "box_xyxy": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "number"},
            },
            "confidence": {"type": "number"},
            "class_label": {"type": "string", "minLength": 1},
            "view_provenance": {"type": "object", "minProperties": 1},
            "transform_to_source": {"type": "object"},
            "runtime_metadata": {"type": ["object", "null"]},
        },
        "dependentRequired": {"burst_id": ["frame_sequence"], "frame_sequence": ["burst_id"]},
        "additionalProperties": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "G7F-A R1 candidate run with explicit processed frame-instance coverage",
        "type": "object",
        "required": [
            "schema_version",
            "run_id",
            "system_id",
            "code_commit",
            "processed_frame_instances",
            "candidates",
        ],
        "properties": {
            "schema_version": {"const": CANDIDATE_RUN_V2},
            "run_id": {"type": "string", "minLength": 1},
            "system_id": {"type": "string", "minLength": 1},
            "code_commit": {"type": "string", "minLength": 1},
            "weight_sha256": {"type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"},
            "run_metadata": {"type": ["object", "null"]},
            "processed_frame_instances": {"type": "array", "items": processed},
            "candidates": {"type": "array", "items": candidate},
        },
        "additionalProperties": False,
    }


def _load_frozen_sources(
    original_workspace: Path, reviewer_package: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    bindings = read_json(original_workspace / "00_SOURCE_FREEZE" / "source_package_bindings.json")
    exact_files = {
        "review_cases.json": "review_cases_sha256",
        "candidate_states_by_reference.json": "candidate_states_by_reference_sha256",
        "tranche_manifest.jsonl": "tranche_manifest_sha256",
        "canonical_reviewer_state_contract.json": "canonical_reviewer_state_contract_sha256",
        "server_action_contract.json": "server_action_contract_sha256",
        "visual_asset_manifest.json": "visual_asset_manifest_sha256",
        "G7E_B_R6_7_2_REAL_REVIEW_RELEASE_GATE.json": "release_gate_sha256",
    }
    for filename, binding_name in exact_files.items():
        path = reviewer_package / filename
        _require(path.is_file(), f"frozen reviewer source missing: {filename}")
        _require(sha256_file(path) == bindings[binding_name], f"frozen reviewer source hash mismatch: {filename}")
    cases = read_json(reviewer_package / "review_cases.json")["cases"]
    states = read_json(reviewer_package / "candidate_states_by_reference.json")["frames"]
    _require(len(cases) == EXPECTED_COUNTS["bursts"], "frozen reviewer must contain exactly 120 cases")
    _require(len(states) == EXPECTED_COUNTS["frame_instances"], "candidate state must contain 1,080 frames")
    return cases, states, bindings


def build_frame_registries(
    original_workspace: Path, reviewer_package: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    original_workspace = original_workspace.resolve()
    reviewer_package = reviewer_package.resolve()
    cases, states, bindings = _load_frozen_sources(original_workspace, reviewer_package)
    gold_dir = original_workspace / "02_NORMALIZED_GOLD"
    bursts = {row["burst_id"]: row for row in read_jsonl(gold_dir / "gold_bursts.jsonl")}
    events = {row["burst_id"]: row for row in read_jsonl(gold_dir / "gold_source_event_index.jsonl")}
    subject_points = {
        (row["burst_id"], row["frame_sequence"])
        for row in read_jsonl(gold_dir / "gold_subject_frames.jsonl")
        if row["human_confirmed_source_coordinate"] is not None
    }
    missed_points = {
        (row["burst_id"], row["frame_sequence"]) for row in read_jsonl(gold_dir / "gold_missed_observations.jsonl")
    }
    old_known_hashes = {
        row["source_frame_sha256"]
        for name in ("gold_subject_frames.jsonl", "gold_missed_observations.jsonl")
        for row in read_jsonl(gold_dir / name)
    }
    _require(len(bursts) == len(events) == len(cases) == 120, "case/Gold burst membership count mismatch")

    instance_rows: list[dict[str, Any]] = []
    candidate_fingerprints: dict[str, str] = {}
    for case in sorted(cases, key=lambda row: row["burst_id"]):
        burst_id = case["burst_id"]
        _require(burst_id in bursts and burst_id in events, f"frozen case absent from original Gold: {burst_id}")
        _require(len(case["frames"]) == 9 and len(case["frame_candidates"]) == 9, f"{burst_id}: nine frames")
        burst = bursts[burst_id]
        event = events[burst_id]
        for sequence, frame in enumerate(case["frames"]):
            _require(frame["burst_frame_sequence"] == sequence, f"{burst_id}: non-canonical frame order")
            frame_reference_id = frame["frame_reference_id"]
            state = states.get(frame_reference_id)
            _require(state is not None, f"{burst_id}/{sequence}: candidate state missing")
            frame_hash = frame["source_frame_pixel_sha256"]
            _require(frame_hash == state["frame_pixel_sha256"], f"{burst_id}/{sequence}: state hash mismatch")
            _require(state["candidates"] == case["frame_candidates"][sequence], f"{burst_id}/{sequence}: candidates")
            _require(frame["source_width"] == case["source_width"], f"{burst_id}/{sequence}: source width")
            _require(frame["source_height"] == case["source_height"], f"{burst_id}/{sequence}: source height")
            fingerprint = canonical_sha256(state["candidates"])
            previous_fingerprint = candidate_fingerprints.setdefault(frame_hash, fingerprint)
            _require(
                previous_fingerprint == fingerprint,
                f"repeated source hash has conflicting candidates: {frame_hash}",
            )
            key = (burst_id, sequence)
            original_mode = frame.get("visual_modes", {}).get("ORIGINAL", {})
            asset_identity = {
                "frame_reference_id": frame_reference_id,
                "canonical_frame_identity": frame["canonical_frame_identity"],
                "resolved_timestamp_seconds": frame["resolved_timestamp_seconds"],
                "relative_offset_seconds": frame["relative_offset_seconds"],
                "focus_sha256": original_mode.get("focus_sha256", frame.get("focus_sha256")),
                "panorama_sha256": original_mode.get("panorama_sha256", frame.get("panorama_sha256")),
            }
            instance_rows.append(
                {
                    "registry_revision": FRAME_INSTANCE_REGISTRY_REVISION,
                    "gold_corpus_version": GOLD_CORPUS_VERSION,
                    "gold_scope": GOLD_SCOPE,
                    "burst_id": burst_id,
                    "tranche_id": burst["tranche_id"],
                    "tranche_position": burst["tranche_position"],
                    "match_id": str(case["match_id"]),
                    "half": case["half"],
                    "frame_sequence": sequence,
                    "frame_reference_id": frame_reference_id,
                    "source_frame_sha256": frame_hash,
                    "source_width": frame["source_width"],
                    "source_height": frame["source_height"],
                    "frozen_source_asset_identity": asset_identity,
                    "source_event_id": event["source_event_id"],
                    "source_event_sha256": event["source_event_sha256"],
                    "source_acknowledgement_id": event["source_acknowledgement_id"],
                    "source_acknowledgement_sha256": event["source_acknowledgement_sha256"],
                    "source_event_schema_version": event["source_event_schema_version"],
                    "source_review_revision": event["source_review_revision"],
                    "source_review_revision_number": event["source_review_revision_number"],
                    "frozen_case_lineage": {
                        "review_id": case["review_id"],
                        "case_schema_version": case["schema_version"],
                        "case_review_revision": case["review_revision"],
                        "burst_manifest_path": case["burst_manifest_path"],
                        "source_manifest_hashes": case["source_manifest_hashes"],
                        "source_candidate_state_file_sha256": bindings["candidate_states_by_reference_sha256"],
                        "source_candidate_state_frame_sha256": canonical_sha256(state),
                        "post_gate_artifact": state["post_gate_artifact"],
                    },
                    "has_reviewed_subject_point": key in subject_points,
                    "has_missed_observation_point": key in missed_points,
                    "historical_frozen_candidate_count": len(state["candidates"]),
                }
            )

    _require(len(instance_rows) == EXPECTED_COUNTS["frame_instances"], "frame-instance registry count")
    _require(len({(row["burst_id"], row["frame_sequence"]) for row in instance_rows}) == len(instance_rows), "keys")
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in instance_rows:
        by_hash[row["source_frame_sha256"]].append(row)
    source_rows: list[dict[str, Any]] = []
    for frame_hash, rows in sorted(by_hash.items()):
        dimensions = {(row["source_width"], row["source_height"]) for row in rows}
        candidate_counts = {row["historical_frozen_candidate_count"] for row in rows}
        _require(len(dimensions) == 1, f"repeated source hash has conflicting dimensions: {frame_hash}")
        _require(len(candidate_counts) == 1, f"repeated source hash has conflicting candidate counts: {frame_hash}")
        source_width, source_height = next(iter(dimensions))
        source_rows.append(
            {
                "registry_revision": SOURCE_FRAME_REGISTRY_REVISION,
                "gold_corpus_version": GOLD_CORPUS_VERSION,
                "gold_scope": GOLD_SCOPE,
                "source_frame_sha256": frame_hash,
                "source_width": source_width,
                "source_height": source_height,
                "frame_instances": [
                    {
                        "burst_id": row["burst_id"],
                        "frame_sequence": row["frame_sequence"],
                        "frame_reference_id": row["frame_reference_id"],
                    }
                    for row in rows
                ],
                "instance_count": len(rows),
                "frozen_asset_lineage": [row["frozen_source_asset_identity"] for row in rows],
                "frozen_candidate_count": next(iter(candidate_counts)),
                "historical_frozen_candidate_count_across_instances": sum(
                    row["historical_frozen_candidate_count"] for row in rows
                ),
                "has_reviewed_subject_point": any(row["has_reviewed_subject_point"] for row in rows),
                "has_missed_observation_point": any(row["has_missed_observation_point"] for row in rows),
            }
        )

    full_hashes = set(by_hash)
    omitted_hashes = full_hashes - old_known_hashes
    affected_rows = [row for row in instance_rows if row["source_frame_sha256"] in omitted_hashes]
    counts = {
        "bursts": len({row["burst_id"] for row in instance_rows}),
        "frame_instances": len(instance_rows),
        "source_frame_hashes": len(source_rows),
        "old_known_hashes": len(old_known_hashes),
        "formerly_omitted_hashes": len(omitted_hashes),
        "affected_frame_instances": len(affected_rows),
        "formerly_blocked_candidates": sum(row["historical_frozen_candidate_count"] for row in affected_rows),
        "frozen_candidates": sum(row["historical_frozen_candidate_count"] for row in instance_rows),
    }
    _require(counts == EXPECTED_COUNTS, f"frozen defect counts mismatch: {counts}")
    _require(old_known_hashes < full_hashes, "old point-derived hashes must be a strict subset")
    _require(REGRESSION_FRAME_SHA256 in omitted_hashes, "formerly rejected regression hash is not reproduced")
    report = {
        "counts": counts,
        "old_known_hashes_are_subset": old_known_hashes < full_hashes,
        "formerly_omitted_hashes": sorted(omitted_hashes),
        "formerly_omitted_regression_hash": REGRESSION_FRAME_SHA256,
        "regression_hash_in_old_authorization": REGRESSION_FRAME_SHA256 in old_known_hashes,
        "regression_hash_in_repaired_authorization": REGRESSION_FRAME_SHA256 in full_hashes,
        "repeated_hash_dimension_conflicts": 0,
        "repeated_hash_candidate_state_conflicts": 0,
    }
    return instance_rows, source_rows, report


def build_frozen_reference_run(
    instance_rows: list[dict[str, Any]], reviewer_package: Path, repository_commit: str
) -> dict[str, Any]:
    states = read_json(reviewer_package / "candidate_states_by_reference.json")["frames"]
    candidate_state_file_sha256 = sha256_file(reviewer_package / "candidate_states_by_reference.json")
    processed: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for row in instance_rows:
        state = states[row["frame_reference_id"]]
        processed.append(
            {
                "burst_id": row["burst_id"],
                "frame_sequence": row["frame_sequence"],
                "frame_reference_id": row["frame_reference_id"],
                "source_frame_sha256": row["source_frame_sha256"],
                "source_width": row["source_width"],
                "source_height": row["source_height"],
                "processing_provenance": {
                    "source_candidate_state_file_sha256": candidate_state_file_sha256,
                    "source_candidate_state_frame_sha256": canonical_sha256(state),
                    "post_gate_artifact": state["post_gate_artifact"],
                },
            }
        )
        for candidate in state["candidates"]:
            candidates.append(
                {
                    "burst_id": row["burst_id"],
                    "frame_sequence": row["frame_sequence"],
                    "source_frame_sha256": row["source_frame_sha256"],
                    "source_width": row["source_width"],
                    "source_height": row["source_height"],
                    "candidate_id": candidate["candidate_id"],
                    "coordinate_space": "SOURCE",
                    "box_xyxy": candidate["source_box_xyxy"],
                    "confidence": candidate["score"],
                    "class_label": "person",
                    "view_provenance": {
                        "frame_reference_id": row["frame_reference_id"],
                        "candidate_state_schema_version": state["schema_version"],
                        "source_candidate_state_file_sha256": candidate_state_file_sha256,
                        "source_candidate_state_frame_sha256": canonical_sha256(state),
                        "post_gate_artifact": state["post_gate_artifact"],
                    },
                    "runtime_metadata": {
                        "footpoint_xy": candidate["footpoint_xy"],
                        "pre_gate_order": candidate["pre_gate_order"],
                        "post_gate_retained_order": candidate["post_gate_retained_order"],
                    },
                }
            )
    _require(len(processed) == EXPECTED_COUNTS["frame_instances"], "frozen v2 processed frame count")
    _require(len(candidates) == EXPECTED_COUNTS["frozen_candidates"], "frozen v2 candidate count")
    return {
        "schema_version": CANDIDATE_RUN_V2,
        "run_id": "FROZEN_G7E_CANDIDATE_REFERENCE",
        "system_id": "FROZEN_G7E_CANDIDATE_REFERENCE",
        "code_commit": repository_commit,
        "weight_sha256": None,
        "run_metadata": {
            "historical_reference": True,
            "detector_inference_executed": False,
            "source_candidate_state_file_sha256": candidate_state_file_sha256,
            "candidate_scope": "FRAME_INSTANCE",
            "production_ready": False,
        },
        "processed_frame_instances": processed,
        "candidates": candidates,
    }


def coverage_summary(
    instance_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]], defect_report: dict[str, Any]
) -> dict[str, Any]:
    point_presence = Counter(
        (
            row["has_reviewed_subject_point"],
            row["has_missed_observation_point"],
        )
        for row in instance_rows
    )
    return {
        "frame_instance_count": len(instance_rows),
        "unique_source_frame_count": len(source_rows),
        "point_presence_by_instance": {
            f"subject_{subject}_missed_{missed}": count for (subject, missed), count in sorted(point_presence.items())
        },
        "defect_reproduction": defect_report,
        "frames_valid_without_reviewed_points": sum(
            not row["has_reviewed_subject_point"] and not row["has_missed_observation_point"] for row in instance_rows
        ),
        "production_ready": False,
    }
