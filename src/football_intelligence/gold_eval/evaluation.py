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


def _bindings(workspace: Path, candidate_run_sha256: str | None) -> dict[str, Any]:
    workspace = workspace.resolve()
    validation = validate_gold(workspace)
    binding = read_json(workspace / "04_EVALUATION_HARNESS" / "evaluator_binding.json")
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
    rows = read_jsonl(workspace / "02_NORMALIZED_GOLD" / "gold_subject_frames.jsonl")
    missed = read_jsonl(workspace / "02_NORMALIZED_GOLD" / "gold_missed_observations.jsonl")
    known: dict[str, dict[str, Any]] = {}
    for row in [*rows, *missed]:
        frame_hash = row["source_frame_sha256"]
        current = {
            "source_width": row["source_width"],
            "source_height": row["source_height"],
        }
        if frame_hash in known and known[frame_hash] != current:
            raise GoldEvalError(f"conflicting dimensions for source frame {frame_hash}")
        known[frame_hash] = current
    return known


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


def validate_candidate_run(workspace: Path, run_path: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    run_path = run_path.resolve()
    run = read_json(run_path)
    required_top = {"schema_version", "run_id", "system_id", "code_commit", "candidates"}
    allowed_top = required_top | {"weight_sha256", "run_metadata"}
    missing_top = required_top - set(run)
    if missing_top:
        raise GoldEvalError(f"candidate run missing provenance fields: {sorted(missing_top)}")
    if run["schema_version"] != "football_intelligence.g7f_a.candidate_run.v1":
        raise GoldEvalError("candidate run schema_version is unsupported")
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
    normalized_rows = []
    seen: set[tuple[str, str]] = set()
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
        key = (frame_hash, candidate["candidate_id"])
        if key in seen:
            raise GoldEvalError(f"duplicate candidate ID within frame/run: {candidate['candidate_id']}")
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
        normalized_rows.append({**candidate, "source_box_xyxy": source_box})
    return {
        "run": run,
        "normalized_candidates": normalized_rows,
        "run_sha256": sha256_file(run_path),
        "candidate_count": len(normalized_rows),
        "raw_run_preserved": True,
    }


def _contains(box: list[float], point: list[float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _point_metrics(points: list[dict[str, Any]], candidates: dict[str, list[list[float]]]) -> dict[str, Any]:
    multiplicity = Counter()
    supported = 0
    for point in points:
        count = sum(_contains(box, point["coordinate"]) for box in candidates.get(point["source_frame_sha256"], []))
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
    subject_points: list[dict[str, Any]], missed_points: list[dict[str, Any]], candidates: dict[str, list[list[float]]]
) -> dict[str, Any]:
    subject = _point_metrics(subject_points, candidates)
    missed = _point_metrics(missed_points, candidates)
    return {
        "subject_marker_candidate_support_rate": subject["support_rate"],
        "subject_marker_candidate_multiplicity": subject["multiplicity"],
        "candidate_count_near_reviewed_subject_marker": subject["multiplicity"],
        "missed_mark_candidate_support_rate": missed["support_rate"],
    }


def evaluate_candidate_run(workspace: Path, run_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    validated = validate_candidate_run(workspace, run_path)
    candidates: dict[str, list[list[float]]] = defaultdict(list)
    for row in validated["normalized_candidates"]:
        candidates[row["source_frame_sha256"]].append(row["source_box_xyxy"])
    burst_rows = read_jsonl(workspace / "02_NORMALIZED_GOLD" / "gold_bursts.jsonl")
    burst_meta = {row["burst_id"]: row for row in burst_rows}
    subject_rows = read_jsonl(workspace / "02_NORMALIZED_GOLD" / "gold_subject_frames.jsonl")
    missed_rows = read_jsonl(workspace / "02_NORMALIZED_GOLD" / "gold_missed_observations.jsonl")
    subject_points = [
        {
            **row,
            "coordinate": row["human_confirmed_source_coordinate"],
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
        "schema_version": "football_intelligence.g7f_a.candidate_evaluation.v1",
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
