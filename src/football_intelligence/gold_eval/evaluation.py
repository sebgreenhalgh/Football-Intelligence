from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from football_intelligence.gold_eval.core import (
    GoldEvalError,
    UnsupportedMetricError,
    evaluator_code_sha256,
    read_json,
    read_jsonl,
    sha256_file,
    validate_gold,
    write_json,
)


CANDIDATE_RUN_V1 = "football_intelligence.g7f_a.candidate_run.v1"
CANDIDATE_RUN_V2 = "football_intelligence.g7f_a_r1.candidate_run.v2"


def _bindings(workspace: Path, candidate_run_sha256: str | None) -> dict[str, Any]:
    workspace = workspace.resolve()
    validation = validate_gold(workspace)
    r1_binding = workspace / "04_EVALUATION_HARNESS" / "evaluator_binding_r1.json"
    binding_path = (
        r1_binding if r1_binding.is_file() else workspace / "04_EVALUATION_HARNESS" / "evaluator_binding.json"
    )
    binding = read_json(binding_path)
    return {
        "gold_corpus_manifest_sha256": validation["manifest_sha256"],
        "split_manifest_sha256": validation["split_manifest_sha256"],
        "candidate_run_sha256": candidate_run_sha256,
        "evaluator_code_commit": binding["repository_commit"],
        "evaluator_code_sha256": evaluator_code_sha256(),
        "metric_contract_sha256": validation["metric_contract_sha256"],
    }


def require_supported_metric(workspace: Path, metric_name: str) -> str:
    capabilities = read_json(workspace / "04_EVALUATION_HARNESS" / "metric_capabilities.json")
    normalized = metric_name.strip().lower().replace("-", "_").replace(" ", "_")
    exact = {name.lower(): name for name in capabilities["tier_a_exact"]}
    diagnostic = {name.lower(): name for name in capabilities["tier_b_point_support"]}
    unsupported = {name.lower(): reason for name, reason in capabilities["tier_c_unsupported"].items()}
    if normalized in exact:
        return "A_EXACT"
    if normalized in diagnostic:
        return "B_POINT_SUPPORT"
    if normalized in unsupported:
        raise UnsupportedMetricError(
            f"UNSUPPORTED_METRIC: {metric_name}. {unsupported[normalized]} "
            "This metric cannot be emitted from TEMPORAL_OBSERVATION_GOLD."
        )
    raise UnsupportedMetricError(
        f"UNSUPPORTED_OR_UNKNOWN_METRIC: {metric_name}. The metric contract is fail-closed; "
        "declare a supported point-support metric or provide the missing gold ontology."
    )


def _known_frames(workspace: Path) -> dict[str, dict[str, Any]]:
    registry_path = workspace / "02_NORMALIZED_GOLD" / "gold_source_frame_registry.jsonl"
    if not registry_path.is_file():
        raise GoldEvalError(
            "full source-frame registry is required; reviewed-point rows cannot authorize candidate frames"
        )
    rows = read_jsonl(registry_path)
    known: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise GoldEvalError(f"source-frame registry row {index} must be an object")
        required = {"source_frame_sha256", "source_width", "source_height", "instance_count", "frame_instances"}
        missing = required - set(row)
        if missing:
            raise GoldEvalError(f"source-frame registry row {index} missing fields: {sorted(missing)}")
        frame_hash = row["source_frame_sha256"]
        if (
            not isinstance(frame_hash, str)
            or len(frame_hash) != 64
            or any(character not in "0123456789abcdef" for character in frame_hash.lower())
        ):
            raise GoldEvalError(f"source-frame registry row {index} has malformed source hash")
        if frame_hash in known:
            raise GoldEvalError(f"duplicate source-frame registry hash: {frame_hash}")
        if (
            not isinstance(row["source_width"], int)
            or isinstance(row["source_width"], bool)
            or row["source_width"] <= 0
            or not isinstance(row["source_height"], int)
            or isinstance(row["source_height"], bool)
            or row["source_height"] <= 0
        ):
            raise GoldEvalError(f"source-frame registry row {index} has invalid dimensions")
        if (
            not isinstance(row["frame_instances"], list)
            or row["instance_count"] != len(row["frame_instances"])
            or row["instance_count"] <= 0
        ):
            raise GoldEvalError(f"source-frame registry row {index} has invalid instance membership")
        membership: set[tuple[str, int]] = set()
        for member in row["frame_instances"]:
            if (
                not isinstance(member, dict)
                or not isinstance(member.get("burst_id"), str)
                or not member["burst_id"]
                or not isinstance(member.get("frame_sequence"), int)
                or isinstance(member["frame_sequence"], bool)
                or not 0 <= member["frame_sequence"] <= 8
            ):
                raise GoldEvalError(f"source-frame registry row {index} has malformed instance membership")
            membership.add((member["burst_id"], member["frame_sequence"]))
        if len(membership) != row["instance_count"]:
            raise GoldEvalError(f"source-frame registry row {index} has duplicate instance membership")
        current = {
            "source_width": row["source_width"],
            "source_height": row["source_height"],
            "frame_instances": membership,
        }
        known[frame_hash] = current
    return known


def _known_frame_instances(
    workspace: Path, known_frames: dict[str, dict[str, Any]]
) -> dict[tuple[str, int], dict[str, Any]]:
    registry_path = workspace / "02_NORMALIZED_GOLD" / "gold_frame_instances.jsonl"
    if not registry_path.is_file():
        raise GoldEvalError("full frame-instance registry is required for candidate-run coverage")
    known_instances: dict[tuple[str, int], dict[str, Any]] = {}
    required = {
        "burst_id",
        "frame_sequence",
        "source_frame_sha256",
        "source_width",
        "source_height",
    }
    for index, row in enumerate(read_jsonl(registry_path)):
        if not isinstance(row, dict):
            raise GoldEvalError(f"frame-instance registry row {index} must be an object")
        missing = required - set(row)
        if missing:
            raise GoldEvalError(f"frame-instance registry row {index} missing fields: {sorted(missing)}")
        burst_id = row["burst_id"]
        frame_sequence = row["frame_sequence"]
        if not isinstance(burst_id, str) or not burst_id:
            raise GoldEvalError(f"frame-instance registry row {index} has malformed burst_id")
        if not isinstance(frame_sequence, int) or isinstance(frame_sequence, bool) or not 0 <= frame_sequence <= 8:
            raise GoldEvalError(f"frame-instance registry row {index} has malformed frame_sequence")
        key = (burst_id, frame_sequence)
        if key in known_instances:
            raise GoldEvalError(f"duplicate frame-instance registry key: {key}")
        frame_hash = row["source_frame_sha256"]
        if frame_hash not in known_frames:
            raise GoldEvalError(f"frame-instance registry row {index} references unknown source hash")
        expected = known_frames[frame_hash]
        if row["source_width"] != expected["source_width"] or row["source_height"] != expected["source_height"]:
            raise GoldEvalError(f"frame-instance registry row {index} conflicts with source-frame dimensions")
        known_instances[key] = row
    actual_membership: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for key, row in known_instances.items():
        actual_membership[row["source_frame_sha256"]].add(key)
    for frame_hash, expected in known_frames.items():
        if actual_membership[frame_hash] != expected["frame_instances"]:
            raise GoldEvalError(f"source/frame-instance registry membership mismatch: {frame_hash}")
    return known_instances


def _source_box(candidate: dict[str, Any]) -> list[float]:
    box = candidate.get("box_xyxy")
    if (
        not isinstance(box, list)
        or len(box) != 4
        or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in box)
    ):
        raise GoldEvalError("candidate box_xyxy must contain four finite numeric coordinates")
    if candidate.get("coordinate_space") == "SOURCE":
        return [float(value) for value in box]
    transform = candidate.get("transform_to_source")
    required = {"type", "from_space", "to_space", "scale_x", "scale_y", "translate_x", "translate_y"}
    if not isinstance(transform, dict) or set(transform) != required:
        raise GoldEvalError("non-source coordinates require an exact seven-field transform_to_source contract")
    if (
        transform["type"] != "AFFINE_SCALE_TRANSLATE"
        or transform["from_space"] != candidate.get("coordinate_space")
        or transform["to_space"] != "SOURCE"
    ):
        raise GoldEvalError("coordinate transform must map the declared input space exactly to SOURCE")
    sx = float(transform["scale_x"])
    sy = float(transform["scale_y"])
    tx = float(transform["translate_x"])
    ty = float(transform["translate_y"])
    if not all(math.isfinite(value) for value in (sx, sy, tx, ty)) or sx <= 0 or sy <= 0:
        raise GoldEvalError("coordinate transform scale/translation values must be finite and scales positive")
    return [box[0] * sx + tx, box[1] * sy + ty, box[2] * sx + tx, box[3] * sy + ty]


def validate_candidate_run(
    workspace: Path, run_path: Path, *, require_exact_frame_coverage: bool = False
) -> dict[str, Any]:
    workspace = workspace.resolve()
    run_path = run_path.resolve()
    run = read_json(run_path)
    schema_version = run.get("schema_version")
    if schema_version not in {CANDIDATE_RUN_V1, CANDIDATE_RUN_V2}:
        raise GoldEvalError("candidate run schema_version is unsupported")
    is_v2 = schema_version == CANDIDATE_RUN_V2
    required_top = {"schema_version", "run_id", "system_id", "code_commit", "candidates"}
    if is_v2:
        required_top.add("processed_frame_instances")
    allowed_top = required_top | {"weight_sha256", "run_metadata"}
    missing_top = required_top - set(run)
    if missing_top:
        raise GoldEvalError(f"candidate run missing provenance fields: {sorted(missing_top)}")
    if set(run) - allowed_top:
        raise GoldEvalError(f"candidate run contains unknown fields: {sorted(set(run) - allowed_top)}")
    if not all(isinstance(run[field], str) and run[field] for field in ("run_id", "system_id", "code_commit")):
        raise GoldEvalError("run_id, system_id and code_commit must be non-empty strings")
    if not isinstance(run["candidates"], list):
        raise GoldEvalError("candidate run candidates must be an array")
    weight_hash = run.get("weight_sha256")
    if weight_hash is not None and (
        not isinstance(weight_hash, str)
        or len(weight_hash) != 64
        or any(character not in "0123456789abcdef" for character in weight_hash.lower())
    ):
        raise GoldEvalError("weight_sha256 must be null or an exact SHA-256")
    known = _known_frames(workspace)
    known_instances = _known_frame_instances(workspace, known)
    declared_instances: dict[tuple[str, int], dict[str, Any]] = {}
    normalized_processed: list[dict[str, Any]] = []
    if is_v2:
        processed = run["processed_frame_instances"]
        if not isinstance(processed, list):
            raise GoldEvalError("candidate run processed_frame_instances must be an array")
        required_processed = {
            "burst_id",
            "frame_sequence",
            "source_frame_sha256",
            "source_width",
            "source_height",
        }
        allowed_processed = required_processed | {"frame_reference_id", "processing_provenance"}
        for index, row in enumerate(processed):
            if not isinstance(row, dict):
                raise GoldEvalError(f"processed frame instance row {index} must be an object")
            missing = required_processed - set(row)
            if missing:
                raise GoldEvalError(f"processed frame instance row {index} missing fields: {sorted(missing)}")
            if set(row) - allowed_processed:
                raise GoldEvalError(
                    f"processed frame instance row {index} contains unknown fields: "
                    f"{sorted(set(row) - allowed_processed)}"
                )
            burst_id = row["burst_id"]
            frame_sequence = row["frame_sequence"]
            if not isinstance(burst_id, str) or not burst_id:
                raise GoldEvalError(f"processed frame instance row {index} has malformed burst_id")
            if not isinstance(frame_sequence, int) or isinstance(frame_sequence, bool) or not 0 <= frame_sequence <= 8:
                raise GoldEvalError(f"processed frame instance row {index} has malformed frame_sequence")
            key = (burst_id, frame_sequence)
            if key in declared_instances:
                raise GoldEvalError(f"duplicate processed frame instance: {key}")
            expected = known_instances.get(key)
            if expected is None:
                raise GoldEvalError(f"processed frame instance row {index} references unknown instance {key}")
            if (
                row["source_frame_sha256"] != expected["source_frame_sha256"]
                or row["source_width"] != expected["source_width"]
                or row["source_height"] != expected["source_height"]
            ):
                raise GoldEvalError(f"processed frame instance row {index} has wrong burst/frame/hash association")
            if "frame_reference_id" in row and row["frame_reference_id"] != expected.get("frame_reference_id"):
                raise GoldEvalError(f"processed frame instance row {index} has wrong frame_reference_id")
            if "processing_provenance" in row and (
                not isinstance(row["processing_provenance"], dict) or not row["processing_provenance"]
            ):
                raise GoldEvalError(f"processed frame instance row {index} has malformed processing provenance")
            declared_instances[key] = row
            normalized_processed.append(dict(row))
    elif require_exact_frame_coverage:
        raise GoldEvalError("G7F-B exact frame coverage requires candidate-run v2")

    missing_instances = sorted(set(known_instances) - set(declared_instances)) if is_v2 else sorted(known_instances)
    exact_coverage = is_v2 and not missing_instances and len(declared_instances) == len(known_instances)
    if require_exact_frame_coverage and not exact_coverage:
        raise GoldEvalError(
            f"candidate-run v2 exact coverage gate failed: {len(missing_instances)} required frame instances missing"
        )

    normalized_rows = []
    seen: set[tuple[Any, ...]] = set()
    candidate_scope_by_hash: dict[str, set[str]] = defaultdict(set)
    instance_candidate_counts: Counter[tuple[str, int]] = Counter()
    shared_candidate_counts: Counter[str] = Counter()
    required_row = {
        "source_frame_sha256",
        "source_width",
        "source_height",
        "candidate_id",
        "coordinate_space",
        "box_xyxy",
        "confidence",
        "class_label",
        "view_provenance",
    }
    allowed_row = required_row | {"transform_to_source", "runtime_metadata"}
    if is_v2:
        allowed_row |= {"burst_id", "frame_sequence"}
    for index, candidate in enumerate(run["candidates"]):
        if not isinstance(candidate, dict):
            raise GoldEvalError(f"candidate row {index} must be an object")
        missing = required_row - set(candidate)
        if missing:
            raise GoldEvalError(f"candidate row {index} missing fields: {sorted(missing)}")
        if set(candidate) - allowed_row:
            raise GoldEvalError(
                f"candidate row {index} contains unknown fields: {sorted(set(candidate) - allowed_row)}"
            )
        frame_hash = candidate["source_frame_sha256"]
        if frame_hash not in known:
            raise GoldEvalError(f"candidate row {index} references unknown source frame {frame_hash}")
        expected = known[frame_hash]
        if (
            candidate["source_width"] != expected["source_width"]
            or candidate["source_height"] != expected["source_height"]
        ):
            raise GoldEvalError(f"candidate row {index} source dimensions do not match Gold Corpus v1")
        candidate_id = candidate["candidate_id"]
        if not isinstance(candidate_id, str) or not candidate_id:
            raise GoldEvalError(f"candidate row {index} has malformed candidate_id")
        evaluation_frame_keys: list[tuple[str, int]] = []
        candidate_scope = "SOURCE_HASH"
        if is_v2:
            has_burst = "burst_id" in candidate
            has_sequence = "frame_sequence" in candidate
            if has_burst != has_sequence:
                raise GoldEvalError(f"candidate row {index} must declare both burst_id and frame_sequence or neither")
            if has_burst:
                burst_id = candidate["burst_id"]
                frame_sequence = candidate["frame_sequence"]
                if not isinstance(burst_id, str) or not burst_id:
                    raise GoldEvalError(f"candidate row {index} has malformed burst_id")
                if (
                    not isinstance(frame_sequence, int)
                    or isinstance(frame_sequence, bool)
                    or not 0 <= frame_sequence <= 8
                ):
                    raise GoldEvalError(f"candidate row {index} has malformed frame_sequence")
                frame_key = (burst_id, frame_sequence)
                declared = declared_instances.get(frame_key)
                if declared is None:
                    raise GoldEvalError(f"candidate row {index} references an undeclared processed frame instance")
                if declared["source_frame_sha256"] != frame_hash:
                    raise GoldEvalError(f"candidate row {index} has wrong burst/frame/hash association")
                key = ("FRAME_INSTANCE", *frame_key, candidate_id)
                evaluation_frame_keys = [frame_key]
                candidate_scope = "FRAME_INSTANCE"
                instance_candidate_counts[frame_key] += 1
            else:
                evaluation_frame_keys = [
                    key for key, row in declared_instances.items() if row["source_frame_sha256"] == frame_hash
                ]
                if not evaluation_frame_keys:
                    raise GoldEvalError(f"candidate row {index} source hash has no declared processed frame instance")
                key = ("SOURCE_HASH", frame_hash, candidate_id)
                shared_candidate_counts[frame_hash] += 1
            candidate_scope_by_hash[frame_hash].add(candidate_scope)
        else:
            key = (frame_hash, candidate_id)
        if key in seen:
            raise GoldEvalError(f"duplicate candidate ID within frame/run: {candidate_id}")
        seen.add(key)
        if not isinstance(candidate["view_provenance"], dict) or not candidate["view_provenance"]:
            raise GoldEvalError(f"candidate row {index} has malformed view/provenance metadata")
        if not isinstance(candidate["class_label"], str) or not candidate["class_label"]:
            raise GoldEvalError(f"candidate row {index} has malformed class label")
        if not isinstance(candidate["confidence"], (int, float)) or not math.isfinite(candidate["confidence"]):
            raise GoldEvalError(f"candidate row {index} has malformed confidence")
        source_box = _source_box(candidate)
        x1, y1, x2, y2 = source_box
        if not (0 <= x1 <= x2 <= expected["source_width"] and 0 <= y1 <= y2 <= expected["source_height"]):
            raise GoldEvalError(f"candidate row {index} has an out-of-bounds source box")
        normalized_rows.append(
            {
                **candidate,
                "source_box_xyxy": source_box,
                "_candidate_scope": candidate_scope,
                "_evaluation_frame_keys": evaluation_frame_keys,
            }
        )
    mixed_hashes = sorted(frame_hash for frame_hash, scopes in candidate_scope_by_hash.items() if len(scopes) > 1)
    if mixed_hashes:
        raise GoldEvalError(
            "candidate-run v2 cannot mix shared-source and frame-instance candidate scopes for the same hash: "
            f"{mixed_hashes}"
        )

    outcome_counts: Counter[str] = Counter()
    if is_v2:
        for row in normalized_processed:
            key = (row["burst_id"], row["frame_sequence"])
            candidate_count = instance_candidate_counts[key] + shared_candidate_counts[row["source_frame_sha256"]]
            if candidate_count == 0:
                outcome = "PROCESSED_ZERO_CANDIDATES"
            elif shared_candidate_counts[row["source_frame_sha256"]]:
                outcome = "PROCESSED_WITH_SHARED_SOURCE_CANDIDATES"
            else:
                outcome = "PROCESSED_WITH_CANDIDATES"
            row["candidate_count"] = candidate_count
            row["processing_outcome"] = outcome
            outcome_counts[outcome] += 1
    coverage = {
        "state": (
            "EXACT_FULL_FRAME_COVERAGE"
            if exact_coverage
            else "DECLARED_PARTIAL_FRAME_COVERAGE"
            if is_v2
            else "UNDECLARED_OR_PARTIAL_FRAME_COVERAGE"
        ),
        "required_instance_count": len(known_instances),
        "declared_instance_count": len(declared_instances),
        "missing_instance_count": len(missing_instances),
        "outcome_counts": dict(sorted(outcome_counts.items())),
    }
    return {
        "run": run,
        "normalized_candidates": normalized_rows,
        "normalized_processed_frame_instances": normalized_processed,
        "coverage": coverage,
        "run_sha256": sha256_file(run_path),
        "candidate_count": len(normalized_rows),
        "raw_run_preserved": True,
    }


def _contains(box: list[float], point: list[float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _point_metrics(points: list[dict[str, Any]], candidates: dict[Any, list[list[float]]]) -> dict[str, Any]:
    multiplicity = Counter()
    supported = 0
    for point in points:
        count = sum(_contains(box, point["coordinate"]) for box in candidates.get(point["evaluation_frame_key"], []))
        multiplicity[str(count)] += 1
        supported += count > 0
    denominator = len(points)
    return {
        "support_rate": {
            "capability_tier": "B_POINT_SUPPORT",
            "numerator": supported,
            "denominator": denominator,
            "value": round(supported / denominator, 6) if denominator else None,
            "unit": "reviewed_source_points_with_at_least_one_containing_candidate",
        },
        "multiplicity": {
            "capability_tier": "B_POINT_SUPPORT",
            "denominator": denominator,
            "counts": dict(sorted(multiplicity.items(), key=lambda item: int(item[0]))),
            "unit": "containing_candidates_per_reviewed_source_point",
        },
    }


def _group_candidate_metrics(
    subject_points: list[dict[str, Any]], missed_points: list[dict[str, Any]], candidates: dict[Any, list[list[float]]]
) -> dict[str, Any]:
    subject = _point_metrics(subject_points, candidates)
    missed = _point_metrics(missed_points, candidates)
    return {
        "subject_marker_candidate_support_rate": subject["support_rate"],
        "subject_marker_candidate_multiplicity": subject["multiplicity"],
        "candidate_count_near_reviewed_subject_marker": subject["multiplicity"],
        "missed_mark_candidate_support_rate": missed["support_rate"],
    }


def evaluate_candidate_run(
    workspace: Path,
    run_path: Path,
    output_path: Path | None = None,
    *,
    require_exact_frame_coverage: bool = False,
) -> dict[str, Any]:
    validated = validate_candidate_run(workspace, run_path, require_exact_frame_coverage=require_exact_frame_coverage)
    is_v2 = validated["run"]["schema_version"] == CANDIDATE_RUN_V2
    candidates: dict[Any, list[list[float]]] = defaultdict(list)
    for row in validated["normalized_candidates"]:
        if is_v2:
            for key in row["_evaluation_frame_keys"]:
                candidates[key].append(row["source_box_xyxy"])
        else:
            candidates[row["source_frame_sha256"]].append(row["source_box_xyxy"])
    burst_rows = read_jsonl(workspace / "02_NORMALIZED_GOLD" / "gold_bursts.jsonl")
    burst_meta = {row["burst_id"]: row for row in burst_rows}
    subject_rows = read_jsonl(workspace / "02_NORMALIZED_GOLD" / "gold_subject_frames.jsonl")
    missed_rows = read_jsonl(workspace / "02_NORMALIZED_GOLD" / "gold_missed_observations.jsonl")
    subject_points = [
        {
            **row,
            "coordinate": row["human_confirmed_source_coordinate"],
            "evaluation_frame_key": (row["burst_id"], row["frame_sequence"]) if is_v2 else row["source_frame_sha256"],
            "perspective_band": burst_meta[row["burst_id"]]["perspective_band"],
            "primary_selection_class": burst_meta[row["burst_id"]]["primary_selection_class"],
        }
        for row in subject_rows
        if row["human_confirmed_source_coordinate"] is not None
    ]
    missed_points = [
        {
            **row,
            "coordinate": row["source_coordinate"],
            "evaluation_frame_key": (row["burst_id"], row["frame_sequence"]) if is_v2 else row["source_frame_sha256"],
            "perspective_band": burst_meta[row["burst_id"]]["perspective_band"],
            "primary_selection_class": burst_meta[row["burst_id"]]["primary_selection_class"],
        }
        for row in missed_rows
    ]

    def grouped(field: str, values: Iterable[str | int]) -> dict[str, Any]:
        return {
            str(value): _group_candidate_metrics(
                [row for row in subject_points if row[field] == value],
                [row for row in missed_points if row[field] == value],
                candidates,
            )
            for value in values
        }

    matches = sorted({row["match_id"] for row in burst_rows})
    report = {
        "schema_version": (
            "football_intelligence.g7f_a_r1.candidate_evaluation.v2"
            if is_v2
            else "football_intelligence.g7f_a.candidate_evaluation.v1"
        ),
        "run_id": validated["run"]["run_id"],
        "system_id": validated["run"]["system_id"],
        "evaluation_bindings": _bindings(workspace, validated["run_sha256"]),
        "all_120_diagnostic_aggregate": _group_candidate_metrics(subject_points, missed_points, candidates),
        "per_match": grouped("match_id", matches),
        "held_out_match_folds": {
            f"HOLDOUT_MATCH_{match}": {
                "tuning_exposure": "HELD_OUT_INTERNAL",
                **grouped("match_id", [match])[match],
            }
            for match in matches
        },
        "per_perspective": grouped(
            "perspective_band", sorted({row["perspective_band"] for row in subject_points + missed_points})
        ),
        "per_selection_class": grouped(
            "primary_selection_class",
            sorted({row["primary_selection_class"] for row in subject_points + missed_points}),
        ),
        "per_relative_frame_position": grouped("frame_sequence", range(9)),
        "candidate_run_coverage": validated["coverage"],
        "candidate_run_metrics_are_tier_b_point_support_only": True,
        "historical_tier_a_candidate_supply_labels_recomputed": False,
        "raw_candidate_run_preserved": True,
        "identity_inferred": False,
        "precision_or_recall_emitted": False,
        "production_ready": False,
    }
    if output_path is not None:
        write_json(output_path, report)
    return report


def evaluate_frozen_baseline(workspace: Path, output_path: Path | None = None) -> dict[str, Any]:
    baseline_path = workspace / "05_FROZEN_BASELINE" / "baseline_current_candidate_system.json"
    report = {
        "schema_version": "football_intelligence.g7f_a.frozen_baseline_evaluation.v1",
        "evaluation_bindings": _bindings(
            workspace, sha256_file(workspace / "02_NORMALIZED_GOLD" / "gold_selected_candidate_bindings.jsonl")
        ),
        "baseline": read_json(baseline_path),
        "by_match": read_json(workspace / "05_FROZEN_BASELINE" / "baseline_by_match.json"),
        "by_fold": read_json(workspace / "05_FROZEN_BASELINE" / "baseline_by_fold.json"),
        "production_ready": False,
    }
    if output_path is not None:
        write_json(output_path, report)
    return report


def compare_runs(report_paths: list[Path], output_path: Path | None = None) -> dict[str, Any]:
    if len(report_paths) < 2:
        raise GoldEvalError("compare-runs requires at least two candidate evaluation reports")
    reports = [read_json(path) for path in report_paths]
    manifest_hashes = {report["evaluation_bindings"]["gold_corpus_manifest_sha256"] for report in reports}
    split_hashes = {report["evaluation_bindings"]["split_manifest_sha256"] for report in reports}
    if len(manifest_hashes) != 1 or len(split_hashes) != 1:
        raise GoldEvalError("candidate reports do not share exact gold and split manifests")
    metrics = {}
    for report in reports:
        aggregate = report["all_120_diagnostic_aggregate"]
        metrics[report["run_id"]] = {
            name: value["value"] for name, value in aggregate.items() if isinstance(value, dict) and "value" in value
        }
    result = {
        "schema_version": "football_intelligence.g7f_a.run_comparison.v1",
        "evaluation_bindings": {
            "gold_corpus_manifest_sha256": next(iter(manifest_hashes)),
            "split_manifest_sha256": next(iter(split_hashes)),
            "candidate_run_sha256": [report["evaluation_bindings"]["candidate_run_sha256"] for report in reports],
            "candidate_evaluation_report_sha256": [sha256_file(path) for path in report_paths],
            "evaluator_code_commit": reports[0]["evaluation_bindings"]["evaluator_code_commit"],
            "evaluator_code_sha256": evaluator_code_sha256(),
            "metric_contract_sha256": reports[0]["evaluation_bindings"]["metric_contract_sha256"],
        },
        "point_support_metrics_by_run": metrics,
        "promotion_decision": None,
        "production_ready": False,
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def build_error_ledger(workspace: Path, output_path: Path) -> dict[str, Any]:
    source = workspace / "05_FROZEN_BASELINE" / "baseline_error_ledger.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(source.read_bytes())
    return {
        "row_count": len(read_jsonl(source)),
        "sha256": sha256_file(output_path),
        "evaluation_bindings": _bindings(workspace, None),
    }


def render_review_sample(workspace: Path, output_path: Path, limit: int = 24) -> dict[str, Any]:
    ledger = read_jsonl(workspace / "05_FROZEN_BASELINE" / "baseline_error_ledger.jsonl")
    ranked = sorted(
        ledger,
        key=lambda row: (
            row.get("candidate_supply") not in {"NO_CANDIDATE", "MERGED_WITH_OTHER_PEOPLE", "FRAGMENT_ONLY"},
            row["burst_id"],
            row["frame_sequence"],
            row.get("subject_token", ""),
            row.get("mark_id", ""),
        ),
    )[:limit]
    result = {
        "schema_version": "football_intelligence.g7f_a.review_sample.v1",
        "selection": "DETERMINISTIC_HIGH_RISK_THEN_STABLE_KEY",
        "rows": ranked,
        "evaluation_bindings": _bindings(workspace, None),
        "production_ready": False,
    }
    write_json(output_path, result)
    return result
