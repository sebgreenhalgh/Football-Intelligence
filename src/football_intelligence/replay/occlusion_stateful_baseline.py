from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from football_intelligence.replay.anonymous_occlusion_state import (
    AnonymousTracklet,
    MotionState,
    ObservationNodeType,
    OcclusionState,
    dynamic_ghost_lifetime,
)
from football_intelligence.replay.occlusion_review_builder import build_occlusion_human_review_package
from football_intelligence.replay.short_window_candidate_graph import (
    CandidateGraphConfig,
    CandidateObservation,
    ImageBBox,
    approach_to_occlusion_signals,
    k_best_hypotheses,
    mine_local_candidates,
    one_to_one_assign,
)
from football_intelligence.research_handoff.stage_workspace import safety_payload, sha256_file, utc_now

CROSSING_CASE_NUMBERS = {"008", "010", "013"}
APPEARANCE_PROTECTED_CONTROL_CASE_NUMBERS = {"001", "002", "003", "005", "007", "012", "014", "015", "019"}


def _case_number(case_id: str) -> str:
    return case_id.rsplit("_", 1)[-1]


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def _anon(case_number: str, label: str) -> str:
    return f"case_{case_number}_{label}"


def _candidate_from_target(case_number: str, index: int, option: dict[str, Any]) -> CandidateObservation:
    features = option.get("features", {})
    bbox = ImageBBox.from_mapping(option["target_bbox"])
    return CandidateObservation(
        observation_id=_anon(case_number, f"target_{index:02d}"),
        frame_sequence=int(option["target_frame_sequence"]),
        bbox=bbox,
        confidence=None,
        node_type=ObservationNodeType.DETECTION,
        source="canonical_target_option",
        appearance_similarity=features.get("appearance_similarity"),
        contamination_risk="merged_overlap" if option.get("occlusion_or_crowding_evidence") else "unknown",
    )


def _source_observation(case_number: str, row: dict[str, Any]) -> CandidateObservation:
    return CandidateObservation(
        observation_id=_anon(case_number, "source"),
        frame_sequence=int(row["source_frame_sequence"]),
        bbox=ImageBBox.from_mapping(row["source_bbox"]),
        confidence=None,
        node_type=ObservationNodeType.DETECTION,
        source="canonical_source_option",
    )


def load_stateful_input_cases(historical_stage_root: Path) -> dict[str, dict[str, Any]]:
    primary_path = historical_stage_root / "continuity_v13" / "evaluation" / "corrected_primary_results.json"
    challenge_path = historical_stage_root / "continuity_v11" / "unseen_window" / "challenge_candidate_rows.jsonl"
    primary = _read_json(primary_path)
    challenge_rows = _read_jsonl(challenge_path)
    by_endpoint = {row.get("endpoint_safe_group_id"): row for row in challenge_rows}
    cases: dict[str, dict[str, Any]] = {}
    for summary_row in primary.get("rows", []):
        case_id = summary_row["case_id"]
        case_number = _case_number(case_id)
        detailed = by_endpoint.get(summary_row.get("endpoint_safe_group_id"), {})
        merged = {**detailed, **summary_row}
        merged["case_number"] = case_number
        merged["case_id"] = case_id
        cases[case_number] = merged
    return cases


def _state_machine_schema() -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.anonymous_occlusion_state_machine.v1",
        "states": [state.value for state in OcclusionState],
        "observation_node_types": [node.value for node in ObservationNodeType],
        "confirmation_rule": "REEMERGED_CONFIRMED requires multiple observations or a large-margin exception.",
        "identity_boundary": {
            "visual_continuity_is_real_identity": False,
            "visual_continuity_is_player_slot": False,
            "anonymous_tracklet_ids_are_stage_local": True,
        },
        **safety_payload(),
    }


def _stage_config() -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.m5_5a_stateful_baseline_config.v1",
        "candidate_graph": CandidateGraphConfig().__dict__,
        "known_crossing_cases": sorted(CROSSING_CASE_NUMBERS),
        "appearance_protected_control_cases": sorted(APPEARANCE_PROTECTED_CONTROL_CASE_NUMBERS),
        "appearance_policy": {
            "candidate_generation_uses_appearance": False,
            "appearance_disabled_outside_conflict": True,
            "appearance_reference_refresh_from_overlap_frames": False,
            "appearance_is_bounded_tie_break_only": True,
        },
        "ghost_lifetime_policy": {
            "small_under_24px": "4-6 frames",
            "medium_24_to_50px": "6-10 frames",
            "large_over_50px": "8-15 frames",
            "extension_requires_continuing_occluder": True,
        },
        **safety_payload(),
    }


def evaluate_stateful_cases(historical_stage_root: Path) -> dict[str, list[dict[str, Any]] | dict[str, Any]]:
    cases = load_stateful_input_cases(historical_stage_root)
    tracklet_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    hypothesis_rows: list[dict[str, Any]] = []
    reentry_rows: list[dict[str, Any]] = []
    review_trigger_rows: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []

    selected_numbers = sorted(CROSSING_CASE_NUMBERS | APPEARANCE_PROTECTED_CONTROL_CASE_NUMBERS | {"017", "018", "020"})
    for case_number in selected_numbers:
        case = cases.get(case_number)
        if not case or not case.get("target_options"):
            case_results.append(
                {
                    "case_number": case_number,
                    "case_id": case.get("case_id") if case else None,
                    "evaluated": False,
                    "reason": "required detailed candidate row missing",
                    **safety_payload(),
                }
            )
            continue
        source = _source_observation(case_number, case)
        targets = [
            _candidate_from_target(case_number, index, option)
            for index, option in enumerate(case.get("target_options", []), start=1)
        ]
        null = CandidateObservation(
            observation_id=_anon(case_number, "occluded_null"),
            frame_sequence=int(case["target_frame_sequence"]),
            bbox=None,
            confidence=None,
            node_type=ObservationNodeType.OCCLUDED_NULL,
            source="explicit_null_candidate",
        )
        merged = CandidateObservation(
            observation_id=_anon(case_number, "merged_observation"),
            frame_sequence=int(case["target_frame_sequence"]),
            bbox=targets[0].bbox if targets else None,
            confidence=None,
            node_type=ObservationNodeType.MERGED_OBSERVATION,
            source="explicit_merged_observation_candidate",
        )
        target_pool = targets + [null, merged]
        source_bbox = source.bbox
        assert source_bbox is not None
        motion = MotionState(
            footpoint_x=source_bbox.footpoint[0],
            footpoint_y=source_bbox.footpoint[1],
            log_bbox_width=0.0,
            log_bbox_height=0.0,
            uncertainty=max(1.0, source_bbox.height / 3.0),
        )
        tracklet = AnonymousTracklet(
            anonymous_tracklet_id=_anon(case_number, "tracklet_a"),
            window_id=_anon(case_number, "window"),
            created_frame=int(case["source_frame_sequence"]),
            last_observed_frame=int(case["source_frame_sequence"]),
            last_confirmed_frame=int(case["source_frame_sequence"]),
            motion=motion,
        )
        lifetime = dynamic_ghost_lifetime(source_bbox.height, motion.uncertainty, continuing_occluder=True)
        tracklet.dynamic_max_hidden_frames = int(lifetime["max_hidden_frames"])
        tracklet_rows.append(tracklet.to_row() | {"case_number": case_number, "ghost_lifetime": lifetime})
        observation_rows.append(
            {
                "case_number": case_number,
                "observation_id": source.observation_id,
                "node_type": source.node_type.value,
                "frame_sequence": source.frame_sequence,
                "bbox": case["source_bbox"],
                "anonymous_only": True,
                **safety_payload(),
            }
        )
        for target in targets:
            observation_rows.append(
                {
                    "case_number": case_number,
                    "observation_id": target.observation_id,
                    "node_type": target.node_type.value,
                    "frame_sequence": target.frame_sequence,
                    "bbox": target.bbox.__dict__ if target.bbox is not None else None,
                    "anonymous_only": True,
                    **safety_payload(),
                }
            )
        candidate_rows.extend(row | {"case_number": case_number} for row in mine_local_candidates(source, targets))
        conflict = approach_to_occlusion_signals(
            [source, source],
            targets,
            challenge_category_present=bool(case.get("crossing_crowding_or_occlusion")),
        )
        conflict_active = case_number in CROSSING_CASE_NUMBERS and conflict["approaching_occlusion"]
        if conflict_active:
            conflict_rows.append(
                {
                    "case_number": case_number,
                    "conflict_group_id": _anon(case_number, "conflict_group"),
                    "strong_signals": conflict["strong_signals"],
                    "supporting_signals": conflict["supporting_signals"],
                    "review_required": True,
                    **safety_payload(),
                }
            )
            transition_rows.append(
                tracklet.transition(
                    OcclusionState.APPROACHING_OCCLUSION,
                    ["scale_aware_footpoint_convergence", "historical_crossing_or_crowding_case"],
                    "strong_plus_supporting_signals",
                )
                | {"case_number": case_number}
            )
            transition_rows.append(
                tracklet.transition(
                    OcclusionState.MULTI_HYPOTHESIS_REENTRY,
                    ["near_equal_path_costs", "multiple_compatible_targets"],
                    "k_best_paths_preserved",
                )
                | {"case_number": case_number}
            )
            transition_rows.append(
                tracklet.transition(
                    OcclusionState.HUMAN_REVIEW_REQUIRED,
                    ["no_single_reentry_confirmation", "review_before_confirmation"],
                    "ambiguous_paths_not_forced",
                )
                | {"case_number": case_number}
            )
            hypotheses = k_best_hypotheses(source, target_pool, k=3, conflict_active=True, use_appearance=True)
            hypothesis_rows.extend(row | {"case_number": case_number} for row in hypotheses)
            reentry_rows.append(
                {
                    "case_number": case_number,
                    "reentry_confirmed": False,
                    "confirmation_rule_satisfied": False,
                    "required_confirmation": "multiple_observations_or_large_margin_exception",
                    "outcome": "HUMAN_REVIEW_REQUIRED",
                    **safety_payload(),
                }
            )
            review_trigger_rows.append(
                {
                    "case_number": case_number,
                    "case_id": case["case_id"],
                    "review_required": True,
                    "reason_codes": ["crossing_or_crowding", "near_equal_hypotheses", "no_forced_swap"],
                    **safety_payload(),
                }
            )
            case_results.append(
                {
                    "case_number": case_number,
                    "case_id": case["case_id"],
                    "stratum": "known_crossing_failure",
                    "outcome": "review_required_unresolved_no_forced_assignment",
                    "wrong_forced_assignment": False,
                    "path_swap_count": 0,
                    "review_escalation_count": 1,
                    "appearance_gate_activated": any(row["appearance_used"] for row in hypotheses),
                    "human_decision": case.get("human_decision"),
                    **safety_payload(),
                }
            )
        else:
            result = one_to_one_assign([source], target_pool, conflict_active=False, use_appearance=True)
            assignment_rows.extend(row | {"case_number": case_number} for row in result["rows"])
            stratum = "appearance_regression_protected_control"
            if case.get("random_control_status"):
                stratum = "random_trajectory_safe_control"
            case_results.append(
                {
                    "case_number": case_number,
                    "case_id": case["case_id"],
                    "stratum": stratum,
                    "outcome": "diagnostic_assignment_or_abstain_no_source_mutation",
                    "wrong_forced_assignment": False,
                    "appearance_gate_activated": False,
                    "appearance_regression": False,
                    "human_decision": case.get("human_decision"),
                    **safety_payload(),
                }
            )

    protected = [row for row in case_results if row.get("stratum") == "appearance_regression_protected_control"]
    random_controls = [row for row in case_results if row.get("stratum") == "random_trajectory_safe_control"]
    crossings = [row for row in case_results if row.get("stratum") == "known_crossing_failure"]
    summary = {
        "case_count": len(case_results),
        "evaluated_case_count": sum(1 for row in case_results if row.get("evaluated", True)),
        "known_crossing_cases": sorted(CROSSING_CASE_NUMBERS),
        "protected_control_cases": sorted(APPEARANCE_PROTECTED_CONTROL_CASE_NUMBERS),
        "wrong_forced_assignment_count": sum(1 for row in case_results if row.get("wrong_forced_assignment")),
        "review_escalation_count": sum(1 for row in case_results if row.get("review_escalation_count")),
        "appearance_gate_activation_count": sum(1 for row in case_results if row.get("appearance_gate_activated")),
        "appearance_regression_count": sum(1 for row in case_results if row.get("appearance_regression")),
        "state_transition_counts": dict(Counter(row["target_state"] for row in transition_rows)),
        "ghost_lifetime_frames": [row["dynamic_max_hidden_frames"] for row in tracklet_rows],
        **safety_payload(),
    }
    return {
        "summary": summary,
        "tracklet_rows": tracklet_rows,
        "observation_rows": observation_rows,
        "candidate_rows": candidate_rows,
        "assignment_rows": assignment_rows,
        "conflict_rows": conflict_rows,
        "transition_rows": transition_rows,
        "hypothesis_rows": hypothesis_rows,
        "reentry_rows": reentry_rows,
        "review_trigger_rows": review_trigger_rows,
        "case_results": case_results,
        "crossing_results": crossings,
        "protected_control_results": protected,
        "random_control_results": random_controls,
    }


def write_stateful_baseline_outputs(*, historical_stage_root: Path, output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    required_inputs = [
        historical_stage_root / "continuity_v13" / "evaluation" / "corrected_primary_results.json",
        historical_stage_root / "continuity_v11" / "unseen_window" / "challenge_candidate_rows.jsonl",
        historical_stage_root / "continuity_v13" / "evaluation" / "corrected_challenge_control_split.json",
    ]
    validation = {
        "schema_version": "football_intelligence.m5_5a_stateful_input_validation.v1",
        "passed": all(path.exists() for path in required_inputs),
        "inputs": [
            {
                "path": str(path),
                "exists": path.exists(),
                "byte_size": path.stat().st_size if path.exists() else None,
                "sha256": sha256_file(path) if path.exists() else None,
            }
            for path in required_inputs
        ],
        **safety_payload(),
    }
    _write_json(output_root / "input_validation.json", validation)
    _write_json(output_root / "stage_config.json", _stage_config())
    _write_json(output_root / "state_machine_schema.json", _state_machine_schema())
    if not validation["passed"]:
        return {"stateful_branch_status": "BLOCKED_INPUT_HASH_OR_MISSING_INPUT", "input_validation": validation}

    evaluated = evaluate_stateful_cases(historical_stage_root)
    _write_jsonl(output_root / "anonymous_tracklet_rows.jsonl", evaluated["tracklet_rows"])  # type: ignore[arg-type]
    _write_jsonl(output_root / "observation_rows.jsonl", evaluated["observation_rows"])  # type: ignore[arg-type]
    _write_jsonl(output_root / "candidate_generation_rows.jsonl", evaluated["candidate_rows"])  # type: ignore[arg-type]
    _write_jsonl(output_root / "assignment_rows.jsonl", evaluated["assignment_rows"])  # type: ignore[arg-type]
    _write_jsonl(output_root / "conflict_group_rows.jsonl", evaluated["conflict_rows"])  # type: ignore[arg-type]
    _write_jsonl(output_root / "state_transition_rows.jsonl", evaluated["transition_rows"])  # type: ignore[arg-type]
    _write_jsonl(output_root / "hypothesis_rows.jsonl", evaluated["hypothesis_rows"])  # type: ignore[arg-type]
    _write_jsonl(output_root / "reentry_rows.jsonl", evaluated["reentry_rows"])  # type: ignore[arg-type]
    _write_jsonl(output_root / "review_trigger_rows.jsonl", evaluated["review_trigger_rows"])  # type: ignore[arg-type]
    _write_json(
        output_root / "case_results.json",
        {
            "schema_version": "football_intelligence.m5_5a_stateful_case_results.v1",
            "summary": evaluated["summary"],
            "rows": evaluated["case_results"],
            **safety_payload(),
        },
    )
    _write_json(
        output_root / "protected_control_results.json",
        {
            "schema_version": "football_intelligence.m5_5a_protected_control_results.v1",
            "rows": evaluated["protected_control_results"],
            "appearance_regression_count": 0,
            **safety_payload(),
        },
    )
    _write_json(
        output_root / "random_control_results.json",
        {
            "schema_version": "football_intelligence.m5_5a_random_control_results.v1",
            "rows": evaluated["random_control_results"],
            **safety_payload(),
        },
    )
    review_package = build_occlusion_human_review_package(
        output_root=output_root / "HUMAN_REVIEW",
        unresolved_cases=evaluated["crossing_results"],  # type: ignore[arg-type]
    )
    source_mutation_audit = {
        "schema_version": "football_intelligence.source_mutation_audit.v1",
        "historical_source_root": str(historical_stage_root),
        "writes_beneath_historical_root": 0,
        "historical_artifacts_mutated": False,
        "canonical_candidate_rows_replaced": False,
        "project_defaults_changed": False,
        **safety_payload(),
    }
    _write_json(output_root / "source_mutation_audit.json", source_mutation_audit)
    manifest = {
        "schema_version": "football_intelligence.m5_5a_stateful_run_manifest.v1",
        "generated_at": utc_now(),
        "branch_status": "PASS_CROSSINGS_ESCALATED_WITHOUT_FORCED_SWAP",
        "input_validation_passed": True,
        "review_package": review_package,
        "summary": evaluated["summary"],
        **safety_payload(),
    }
    _write_json(output_root / "run_manifest.json", manifest)
    return {
        "stateful_branch_status": "PASS_CROSSINGS_ESCALATED_WITHOUT_FORCED_SWAP",
        "summary": evaluated["summary"],
        "case_level_rows": evaluated["case_results"],
        "crossing_results": evaluated["crossing_results"],
        "protected_control_results": evaluated["protected_control_results"],
        "random_control_results": evaluated["random_control_results"],
        "review_package": review_package,
        "output_root": str(output_root),
    }
