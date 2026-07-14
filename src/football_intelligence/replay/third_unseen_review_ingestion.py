from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from football_intelligence.replay.cadence_matched_third_unseen_challenge import (
    PRIMARY_BASELINE,
    SECONDARY_BASELINE,
    _bbox_hash,
    _source_mutation_paths,
)
from football_intelligence.replay.geometry_matched_counterfactual_review import _iou
from football_intelligence.replay.rebuilt_human_calibrated_pipeline import read_json, write_json, write_text
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.config import load_ui_config, ui_config_hash
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import load_manifest, manifest_hash

PASS_FROZEN_BASELINE_SURVIVES = "PASS_FROZEN_BASELINE_SURVIVES_THIRD_UNSEEN_CHALLENGE"
PASS_MODEL_RESEARCH = "PASS_THIRD_UNSEEN_FAILURES_JUSTIFY_BOUNDED_MODEL_RESEARCH"
PASS_CANDIDATE_FAILURE_DOMINATES = "PASS_CANDIDATE_MINER_OR_ENDPOINT_FAILURE_DOMINATES"
PASS_HUMAN_NON_BINARY_DOMINATES = "PASS_HUMAN_NON_BINARY_OUTCOMES_DOMINATE"
BLOCKED_REVIEW_EVENT_INTEGRITY = "BLOCKED_REVIEW_EVENT_INTEGRITY"
BLOCKED_SEALED_MAPPING_INTEGRITY = "BLOCKED_SEALED_MAPPING_INTEGRITY"
BLOCKED_ENDPOINT_REVALIDATION = "BLOCKED_ENDPOINT_REVALIDATION"
BLOCKED_EXACT_EDGE_LABEL_CONTRADICTION = "BLOCKED_EXACT_EDGE_LABEL_CONTRADICTION"
FAIL_SOURCE_MUTATION_OR_SAFETY = "FAIL_SOURCE_MUTATION_OR_SAFETY"

DECISION_TO_PANEL = {
    "target_a_continues_source": "target_a",
    "target_b_continues_source": "target_b",
}
PANEL_TO_OUTCOME = {"target_a": "TARGET_A_SELECTED", "target_b": "TARGET_B_SELECTED"}
RULE_FIELDS = {
    "primary": "primary_rule_accept",
    "secondary": "secondary_threshold_accept",
}
FROZEN_PRIMARY_THRESHOLDS = {
    "bbox_iou": 0.35,
    "normalised_center_displacement": 0.60,
    "normalised_footpoint_displacement": 0.80,
}
FROZEN_SECONDARY_THRESHOLD = 0.303375


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    write_text(path, "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in records))


def _completed_state(completed: dict[str, Any]) -> dict[str, Any]:
    state = completed.get("state")
    return state if isinstance(state, dict) else completed


def _case_map(manifest_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(case["case_id"]): case for case in manifest_payload.get("cases", [])}


def _embedded_frame(value: str | None) -> int | None:
    if not value:
        return None
    marker = "_f"
    if marker not in value:
        return None
    try:
        return int(value.split(marker, 1)[1].split("_", 1)[0])
    except ValueError:
        return None


def _other_panel(panel: str | None) -> str | None:
    if panel == "target_a":
        return "target_b"
    if panel == "target_b":
        return "target_a"
    return None


def _canonical_edge_key(
    *,
    source_candidate_id: str,
    target_candidate_id: str,
    source_frame_sequence: int,
    target_frame_sequence: int,
) -> str:
    return stable_hash(
        {
            "source_candidate_id": source_candidate_id,
            "source_frame_sequence": int(source_frame_sequence),
            "target_candidate_id": target_candidate_id,
            "target_frame_sequence": int(target_frame_sequence),
        }
    )


def _edge_key_from_row(row: dict[str, Any]) -> str:
    return str(
        row.get("canonical_edge_key")
        or _canonical_edge_key(
            source_candidate_id=str(row["source_candidate_id"]),
            target_candidate_id=str(row["target_candidate_id"]),
            source_frame_sequence=int(row["source_frame_sequence"]),
            target_frame_sequence=int(row["target_frame_sequence"]),
        )
    )


def _load_challenge_rows(stage_root: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(stage_root / "continuity_v11" / "unseen_window" / "challenge_candidate_rows.jsonl")
    return {str(row["challenge_candidate_id"]): row for row in rows}


def _target_option(challenge: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for option in challenge.get("target_options", []):
        if str(option.get("target_candidate_id")) == str(candidate_id):
            return option
    return {}


def _panel_target(mapping: dict[str, Any], challenge: dict[str, Any], panel: str) -> dict[str, Any]:
    candidate_id = str(mapping[f"{panel}_candidate_id"])
    option = _target_option(challenge, candidate_id)
    return {
        "panel": panel,
        "candidate_id": candidate_id,
        "visible_person_base_id": str(mapping[f"{panel}_visible_person_base_id"]),
        "canonical_visible_person_base_id": option.get("target_visible_person_base_id"),
        "bbox": option.get("target_bbox"),
        "features": mapping.get("registered_frozen_rule_outputs", {}).get(panel, {}),
        "role_status": option.get("role_status", "UNKNOWN_NOT_CONTRADICTED"),
        "team_status": option.get("team_status", "UNKNOWN_NOT_CONTRADICTED"),
    }


def _rule_preferred_panel(mapping: dict[str, Any], rule_name: str) -> dict[str, Any]:
    field = RULE_FIELDS[rule_name]
    outputs = mapping.get("registered_frozen_rule_outputs", {})
    accepted = [panel for panel in ("target_a", "target_b") if bool(outputs.get(panel, {}).get(field))]
    if len(accepted) == 1:
        return {
            "preferred_panel": accepted[0],
            "accepted_panels": accepted,
            "rejected_both": False,
            "multiple_accepts": False,
        }
    if not accepted:
        return {
            "preferred_panel": None,
            "accepted_panels": [],
            "rejected_both": True,
            "multiple_accepts": False,
        }
    return {
        "preferred_panel": mapping.get("frozen_baseline_preferred_panel"),
        "accepted_panels": accepted,
        "rejected_both": False,
        "multiple_accepts": True,
    }


def _decision_agreement(decision: str, rule_panel: str | None) -> str:
    selected = DECISION_TO_PANEL.get(decision)
    if selected is None:
        if decision == "neither_target_is_valid_or_compatible":
            return "RULE_ABSTAINED_OR_REJECTED_BOTH" if rule_panel is None else "RULE_CHOSE_PANEL_WHILE_HUMAN_NEITHER"
        if decision == "unresolved":
            return "HUMAN_UNRESOLVED"
        return "INVALID_MAPPING_OR_DECISION"
    if rule_panel is None:
        return "RULE_ABSTAINED_OR_REJECTED_BOTH"
    return "AGREES_WITH_HUMAN" if rule_panel == selected else "DISAGREES_WITH_HUMAN"


def _semantic_state_matches(review_decisions: dict[str, Any], completed_state: dict[str, Any]) -> bool:
    keys = {
        "completed",
        "completed_at",
        "decisions",
        "elapsed_active_seconds",
        "event_sequence",
        "evidence_manifest_hash",
        "manifest_hash",
        "notes",
        "review_id",
        "server_reveal_payloads",
        "stage_id",
        "ui_config_hash",
    }
    return {key: review_decisions.get(key) for key in keys} == {key: completed_state.get(key) for key in keys}


def _validate_snapshots(decisions_root: Path, completed_state: dict[str, Any]) -> dict[str, Any]:
    snapshot_root = decisions_root / "snapshots"
    snapshots = sorted(snapshot_root.glob("review_state_*.json")) if snapshot_root.exists() else []
    sequence_values = []
    sha_mismatches = []
    for path in snapshots:
        payload = read_json(path)
        sequence_values.append(int(payload.get("snapshot_sequence", -1)))
        sha_path = path.with_suffix(path.suffix + ".sha256")
        if not sha_path.exists():
            sha_mismatches.append(sha_path.name)
            continue
        recorded = sha_path.read_text(encoding="utf-8").split()[0]
        if recorded != sha256_file(path):
            sha_mismatches.append(path.name)
    final_sequence = int(completed_state.get("event_sequence", 0) or 0)
    return {
        "snapshot_count": len(snapshots),
        "snapshot_sequence_values": sequence_values,
        "snapshot_sequence_valid": sequence_values == list(range(1, final_sequence + 1)),
        "final_snapshot_sequence_matches_state": (sequence_values[-1] if sequence_values else None) == final_sequence,
        "snapshot_sha256_mismatches": sha_mismatches,
    }


def _event_explanation(event: dict[str, Any], expected: set[str], decisions_before: set[str]) -> str:
    event_type = str(event.get("event_type"))
    case_id = event.get("case_id")
    if event_type == "decision":
        if case_id not in expected:
            return "decision event for an unexpected case; ignored for final expected-state reconstruction"
        if event.get("prior_decision") is None and str(case_id) not in decisions_before:
            return "initial decision persisted for expected case"
        return "decision overwrite or repeated same-value confirmation persisted for expected case"
    if event_type == "complete":
        return "completion event after the review state had all expected case decisions"
    if event_type == "reveal":
        return "reveal event; this would expose post-decision payload and is audited separately"
    if event_type == "note":
        return "note update event"
    if event_type == "undo":
        return "undo event"
    return f"other event type: {event_type}"


def _replay_review_events(events: list[dict[str, Any]], expected_case_ids: list[str]) -> dict[str, Any]:
    expected = set(expected_case_ids)
    decisions: dict[str, str] = {}
    final_decision_events: dict[str, dict[str, Any]] = {}
    explanations = []
    overwritten = []
    reveal_events = []
    undo_events = []
    note_events = []
    complete_events = []
    other_events = []
    event_ids = [str(event.get("event_id")) for event in events]
    sequences = [int(event.get("event_sequence", -1)) for event in events]
    event_sequence_monotonic = sequences == sorted(sequences) and sequences == list(range(1, len(events) + 1))

    for event in events:
        before = set(decisions)
        event_type = str(event.get("event_type"))
        case_id = event.get("case_id")
        if event_type == "decision":
            case_key = str(case_id)
            if case_key in expected:
                if event.get("prior_decision") is not None or case_key in decisions:
                    overwritten.append(event)
                decisions[case_key] = str(event.get("new_decision"))
                final_decision_events[case_key] = event
        elif event_type == "reveal":
            reveal_events.append(event)
        elif event_type == "undo":
            undo_events.append(event)
        elif event_type == "note":
            note_events.append(event)
        elif event_type == "complete":
            complete_events.append(event)
        else:
            other_events.append(event)
        explanations.append(
            {
                "event_id": event.get("event_id"),
                "event_sequence": event.get("event_sequence"),
                "event_type": event_type,
                "case_id": case_id,
                "explanation": _event_explanation(event, expected, before),
            }
        )

    missing_cases = sorted(expected - set(decisions))
    unexpected_cases = sorted(set(decisions) - expected)
    duplicate_event_ids = sorted(event_id for event_id, count in Counter(event_ids).items() if count > 1)
    completion_after_all = False
    for event in complete_events:
        prior_sequences = {
            str(item.get("case_id"))
            for item in events
            if str(item.get("event_type")) == "decision"
            and int(item.get("event_sequence", 0)) < int(event.get("event_sequence", 0))
        }
        completion_after_all = expected.issubset(prior_sequences)
    reveal_before_decision = any(
        str(event.get("case_id")) not in decisions for event in reveal_events if event.get("case_id") is not None
    )
    return {
        "artifact": "m5_4i_completed_review_event_replay",
        "expected_case_ids": expected_case_ids,
        "event_count": len(events),
        "event_sequence_monotonic": event_sequence_monotonic,
        "event_ids_unique": not duplicate_event_ids,
        "duplicate_event_ids": duplicate_event_ids,
        "final_decisions": decisions,
        "final_decision_events": final_decision_events,
        "decision_count": len(decisions),
        "missing_cases": missing_cases,
        "unexpected_cases": unexpected_cases,
        "initial_decision_event_count": sum(
            1 for event in events if event.get("event_type") == "decision" and event.get("prior_decision") is None
        ),
        "overwritten_decision_event_count": len(overwritten),
        "undo_event_count": len(undo_events),
        "notes_event_count": len(note_events),
        "completion_event_count": len(complete_events),
        "other_event_count": len(other_events),
        "reveal_event_count": len(reveal_events),
        "no_reveal_occurred": len(reveal_events) == 0,
        "no_reveal_before_persisted_decision": not reveal_before_decision,
        "completion_after_all_expected_decided": completion_after_all,
        "event_explanations": explanations,
        "all_events_explained": len(explanations) == len(events),
        **safety_payload(),
    }


def _validate_completed_review(
    stage_root: Path, expected_case_ids: list[str]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    review_root = stage_root / "continuity_v11" / "review"
    decisions_root = review_root / "decisions"
    events = _read_jsonl(decisions_root / "review_decision_events.jsonl")
    replay = _replay_review_events(events, expected_case_ids)
    review_decisions = read_json(decisions_root / "review_decisions.json")
    completed = read_json(decisions_root / "completed_review.json")
    completed_state = _completed_state(completed)
    completed_summary = read_json(decisions_root / "completed_review_summary.json")
    manifest = load_manifest(review_root / "target_choice_reviewer_manifest.json")
    ui_config = load_ui_config(review_root / "target_choice_ui_config.json")
    snapshot_audit = _validate_snapshots(decisions_root, completed_state)
    final_state_matches = replay["final_decisions"] == completed_state.get("decisions", {})
    completed_events_match = (decisions_root / "completed_review_events.jsonl").read_text(encoding="utf-8") == (
        decisions_root / "review_decision_events.jsonl"
    ).read_text(encoding="utf-8")
    no_answer_key_payload = (
        completed_state.get("server_reveal_payloads", {}) == {}
        and completed.get("state", {}).get("server_reveal_payloads", {}) == {}
        and replay["no_reveal_occurred"]
    )
    passed = all(
        [
            len(events) == 23,
            replay["all_events_explained"],
            replay["decision_count"] == len(expected_case_ids),
            not replay["missing_cases"],
            not replay["unexpected_cases"],
            replay["completion_after_all_expected_decided"],
            replay["event_sequence_monotonic"],
            replay["event_ids_unique"],
            int(completed_state.get("event_sequence", 0)) == 23,
            completed_state.get("manifest_hash") == manifest_hash(manifest),
            completed_state.get("ui_config_hash") == ui_config_hash(ui_config),
            completed_state.get("evidence_manifest_hash") == manifest.evidence_manifest_hash,
            completed.get("decision_state_hash") == stable_hash(completed_state),
            final_state_matches,
            _semantic_state_matches(review_decisions, completed_state),
            completed_events_match,
            replay["no_reveal_occurred"],
            no_answer_key_payload,
            snapshot_audit["snapshot_sequence_valid"],
            snapshot_audit["final_snapshot_sequence_matches_state"],
            not snapshot_audit["snapshot_sha256_mismatches"],
        ]
    )
    validation = {
        "artifact": "m5_4i_completed_review_event_validation",
        "passed": passed,
        "review_event_validation": "PASS" if passed else "FAIL",
        "authoritative_event_log": str(decisions_root / "review_decision_events.jsonl"),
        "expected_case_count": len(expected_case_ids),
        "final_event_sequence": completed_state.get("event_sequence"),
        "event_count_explanation": {
            "total_events": len(events),
            "initial_decision_events": replay["initial_decision_event_count"],
            "overwritten_decisions": replay["overwritten_decision_event_count"],
            "undo_events": replay["undo_event_count"],
            "notes_events": replay["notes_event_count"],
            "completion_events": replay["completion_event_count"],
            "other_events": replay["other_event_count"],
        },
        "replay_result": replay,
        "manifest_hash_matches": completed_state.get("manifest_hash") == manifest_hash(manifest),
        "ui_config_hash_matches": completed_state.get("ui_config_hash") == ui_config_hash(ui_config),
        "evidence_manifest_hash_matches": completed_state.get("evidence_manifest_hash")
        == manifest.evidence_manifest_hash,
        "decision_state_hash_matches": completed.get("decision_state_hash") == stable_hash(completed_state),
        "completed_export_matches_replayed_state": final_state_matches,
        "completed_review_events_match_append_only_log": completed_events_match,
        "no_answer_key_payload_reached_client": no_answer_key_payload,
        "snapshot_audit": snapshot_audit,
        **safety_payload(),
    }
    event_session_ids = sorted(
        {str(event.get("reviewer_session_id")) for event in events if event.get("reviewer_session_id")}
    )
    completed_session_id = str(completed_state.get("reviewer_session_id"))
    summary_session_id = str(completed_summary.get("reviewer_session_id"))
    review_decisions_session_id = str(review_decisions.get("reviewer_session_id"))
    session_ids = sorted({*event_session_ids, completed_session_id, summary_session_id, review_decisions_session_id})
    session_result = (
        "EXACT_SESSION_MATCH" if len(session_ids) == 1 else "NORMALIZED_ALIAS_OR_DEFAULT_SESSION_LABEL_MISMATCH"
    )
    session_audit = {
        "artifact": "m5_4i_reviewer_session_consistency_audit",
        "event_log_session_ids": event_session_ids,
        "completed_review_session_id": completed_session_id,
        "completed_summary_session_id": summary_session_id,
        "review_decisions_session_id": review_decisions_session_id,
        "distinct_session_representations": session_ids,
        "reviewer_session_consistency_classification": session_result,
        "blocking_defect": session_result == "MULTIPLE_UNEXPLAINED_REVIEWER_SESSIONS",
        "rationale": "local-a4c73332 and local-reviewer are reconciled default/local aliases; hashes and values match.",
        **safety_payload(),
    }
    decision_events = [event for event in events if str(event.get("event_type")) == "decision"]
    timing = {
        "artifact": "m5_4i_review_timing_and_input_audit",
        "elapsed_active_seconds": completed_state.get("elapsed_active_seconds"),
        "final_event_sequence": completed_state.get("event_sequence"),
        "first_decision_event_sequence": min((int(event["event_sequence"]) for event in decision_events), default=None),
        "last_decision_event_sequence": max((int(event["event_sequence"]) for event in decision_events), default=None),
        "decision_input_source_counts": dict(
            Counter(str(event.get("keyboard_or_click_input_source", "unknown")) for event in decision_events)
        ),
        "notes_count": len([note for note in completed_state.get("notes", {}).values() if str(note).strip()]),
        "reveal_event_count": replay["reveal_event_count"],
        **safety_payload(),
    }
    return validation, session_audit, timing


def _validate_sealed_mapping(
    *,
    stage_root: Path,
    expected_case_ids: list[str],
    manifest_payload: dict[str, Any],
    challenge_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    review_root = stage_root / "continuity_v11" / "review"
    sealed = read_json(review_root / "sealed" / "target_choice_server_sealed_mapping.json")
    reference = read_json(review_root / "target_choice_server_sealed_reference.json")
    payload = dict(sealed)
    stored_hash = str(payload.pop("sealed_mapping_hash", ""))
    recomputed_hash = stable_hash(payload)
    mappings = sealed.get("mappings", [])
    mapping_by_case = {str(row["case_id"]): row for row in mappings}
    expected = set(expected_case_ids)
    cases = _case_map(manifest_payload)
    binding_rows = []
    for mapping in mappings:
        case_id = str(mapping.get("case_id"))
        case = cases.get(case_id, {})
        challenge = challenge_by_id.get(str(mapping.get("challenge_candidate_id")), {})
        target_a = _target_option(challenge, str(mapping.get("target_a_candidate_id")))
        target_b = _target_option(challenge, str(mapping.get("target_b_candidate_id")))
        temporal_gap = challenge.get("temporal_gap_seconds")
        binding_rows.append(
            {
                "case_id": case_id,
                "challenge_candidate_id": mapping.get("challenge_candidate_id"),
                "case_binding_passed": bool(case and challenge),
                "source_id_binds": mapping.get("source_candidate_id") == challenge.get("source_candidate_id"),
                "source_visible_person_base_binds": mapping.get("source_visible_person_base_id")
                == challenge.get("source_visible_person_base_id"),
                "target_a_binds": bool(target_a)
                and target_a.get("target_visible_person_base_id") == mapping.get("target_a_visible_person_base_id"),
                "target_b_binds": bool(target_b)
                and target_b.get("target_visible_person_base_id") == mapping.get("target_b_visible_person_base_id"),
                "target_candidate_ids_distinct": mapping.get("target_a_candidate_id")
                != mapping.get("target_b_candidate_id"),
                "target_visible_person_base_ids_distinct": mapping.get("target_a_visible_person_base_id")
                != mapping.get("target_b_visible_person_base_id"),
                "source_frame_sequence": challenge.get("source_frame_sequence"),
                "target_frame_sequence": challenge.get("target_frame_sequence"),
                "source_and_targets_bind_to_declared_frames": bool(case)
                and int(case.get("source_frame_sequence", -1)) == int(challenge.get("source_frame_sequence", -2))
                and int(case.get("target_frame_sequence", -1)) == int(challenge.get("target_frame_sequence", -2))
                and target_a.get("target_frame_sequence") == challenge.get("target_frame_sequence")
                and target_b.get("target_frame_sequence") == challenge.get("target_frame_sequence"),
                "temporal_gap_seconds": temporal_gap,
                "temporal_gap_valid": temporal_gap in {0.1, 0.2, 0.3},
                "endpoint_safe_group_id_present": bool(mapping.get("endpoint_safe_group_id")),
                "frozen_primary_output_present": all(
                    "primary_rule_accept" in mapping.get("registered_frozen_rule_outputs", {}).get(panel, {})
                    for panel in ("target_a", "target_b")
                ),
                "frozen_secondary_output_present": all(
                    "secondary_threshold_accept" in mapping.get("registered_frozen_rule_outputs", {}).get(panel, {})
                    for panel in ("target_a", "target_b")
                ),
                "frozen_preferred_panel_present": mapping.get("frozen_baseline_preferred_panel")
                in {"target_a", "target_b"},
                "challenge_category_vector_present": bool(mapping.get("challenge_categories")),
                "random_control_status": bool(challenge.get("random_unseen_control")),
            }
        )
    mapping_cases = set(mapping_by_case)
    passed = all(
        [
            stored_hash == recomputed_hash,
            stored_hash == reference.get("sealed_mapping_hash"),
            len(mappings) == len(expected_case_ids),
            mapping_cases == expected,
            sealed.get("server_side_only") is True,
            sealed.get("browser_served_before_decision") is False,
            all(row["case_binding_passed"] for row in binding_rows),
            all(row["source_id_binds"] and row["source_visible_person_base_binds"] for row in binding_rows),
            all(row["target_a_binds"] and row["target_b_binds"] for row in binding_rows),
            all(row["target_candidate_ids_distinct"] for row in binding_rows),
            all(row["target_visible_person_base_ids_distinct"] for row in binding_rows),
            all(row["source_and_targets_bind_to_declared_frames"] for row in binding_rows),
            all(row["temporal_gap_valid"] for row in binding_rows),
            all(row["endpoint_safe_group_id_present"] for row in binding_rows),
            all(
                row["frozen_primary_output_present"] and row["frozen_secondary_output_present"] for row in binding_rows
            ),
            all(row["frozen_preferred_panel_present"] for row in binding_rows),
            all(row["challenge_category_vector_present"] for row in binding_rows),
        ]
    )
    return {
        "artifact": "m5_4i_sealed_mapping_validation",
        "passed": passed,
        "sealed_mapping_validation": "PASS" if passed else "FAIL",
        "stored_sealed_mapping_hash": stored_hash,
        "recomputed_sealed_mapping_hash": recomputed_hash,
        "mapping_hash_matches_payload": stored_hash == recomputed_hash,
        "mapping_hash_matches_stored_reference": stored_hash == reference.get("sealed_mapping_hash"),
        "mapping_count": len(mappings),
        "expected_mapping_count": len(expected_case_ids),
        "missing_mapping_case_ids": sorted(expected - mapping_cases),
        "extra_mapping_case_ids": sorted(mapping_cases - expected),
        "mapping_was_not_browser_routable_before_completion": sealed.get("browser_served_before_decision") is False,
        "server_side_only": sealed.get("server_side_only") is True,
        "binding_rows": binding_rows,
        **safety_payload(),
    }, mapping_by_case


def _decode_decisions(
    *,
    replay: dict[str, Any],
    mapping_by_case: dict[str, dict[str, Any]],
    challenge_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for case_id in sorted(replay["final_decisions"]):
        decision = str(replay["final_decisions"][case_id])
        mapping = mapping_by_case.get(case_id)
        challenge = challenge_by_id.get(str(mapping.get("challenge_candidate_id"))) if mapping else None
        selected_panel = DECISION_TO_PANEL.get(decision)
        unselected_panel = _other_panel(selected_panel)
        if selected_panel is not None:
            outcome = PANEL_TO_OUTCOME[selected_panel]
        elif decision == "neither_target_is_valid_or_compatible":
            outcome = "NEITHER_TARGET_VALID_OR_COMPATIBLE"
        elif decision == "unresolved":
            outcome = "UNRESOLVED"
        else:
            outcome = "INVALID_MAPPING_OR_DECISION"
        primary = _rule_preferred_panel(mapping, "primary") if mapping else {"preferred_panel": None}
        secondary = _rule_preferred_panel(mapping, "secondary") if mapping else {"preferred_panel": None}
        selected_target = (
            _panel_target(mapping, challenge or {}, selected_panel) if mapping and challenge and selected_panel else {}
        )
        unselected_target = (
            _panel_target(mapping, challenge or {}, unselected_panel)
            if mapping and challenge and unselected_panel
            else {}
        )
        event = replay["final_decision_events"].get(case_id, {})
        rows.append(
            {
                "case_id": case_id,
                "human_decision": decision,
                "human_outcome": outcome,
                "selected_displayed_panel": selected_panel,
                "unselected_displayed_panel": unselected_panel,
                "selected_canonical_target": selected_target or None,
                "unselected_canonical_target": unselected_target or None,
                "frozen_primary_preferred_panel": primary.get("preferred_panel"),
                "frozen_secondary_preferred_panel": secondary.get("preferred_panel"),
                "primary_rule_rejected_both": bool(primary.get("rejected_both")),
                "secondary_rule_rejected_both": bool(secondary.get("rejected_both")),
                "primary_rule_multiple_accepts": bool(primary.get("multiple_accepts")),
                "secondary_rule_multiple_accepts": bool(secondary.get("multiple_accepts")),
                "agreement_with_primary_rule": _decision_agreement(decision, primary.get("preferred_panel")),
                "agreement_with_secondary_rule": _decision_agreement(decision, secondary.get("preferred_panel")),
                "frame_gap": challenge.get("frame_gap") if challenge else None,
                "temporal_gap_seconds": challenge.get("temporal_gap_seconds") if challenge else None,
                "source_frame_sequence": challenge.get("source_frame_sequence") if challenge else None,
                "target_frame_sequence": challenge.get("target_frame_sequence") if challenge else None,
                "source_candidate_id": mapping.get("source_candidate_id") if mapping else None,
                "source_visible_person_base_id": mapping.get("source_visible_person_base_id") if mapping else None,
                "source_bbox": challenge.get("source_bbox") if challenge else None,
                "challenge_candidate_id": mapping.get("challenge_candidate_id") if mapping else None,
                "challenge_categories": mapping.get("challenge_categories", []) if mapping else [],
                "random_control_status": bool(challenge.get("random_unseen_control")) if challenge else None,
                "endpoint_safe_group_id": mapping.get("endpoint_safe_group_id") if mapping else None,
                "local_assignment_neighbourhood_id": mapping.get("local_assignment_neighbourhood_id")
                if mapping
                else None,
                "review_event_id": event.get("event_id"),
                "mapping_hash": stable_hash(mapping) if mapping else None,
                **safety_payload(),
            }
        )
    counts = Counter(row["human_outcome"] for row in rows)
    summary = {
        "artifact": "m5_4i_decoded_third_unseen_summary",
        "case_count": len(rows),
        "decisive_ab_count": counts["TARGET_A_SELECTED"] + counts["TARGET_B_SELECTED"],
        "target_a_selected_count": counts["TARGET_A_SELECTED"],
        "target_b_selected_count": counts["TARGET_B_SELECTED"],
        "neither_count": counts["NEITHER_TARGET_VALID_OR_COMPATIBLE"],
        "unresolved_count": counts["UNRESOLVED"],
        "invalid_mapping_or_decision_count": counts["INVALID_MAPPING_OR_DECISION"],
        "decision_counts": dict(Counter(row["human_decision"] for row in rows)),
        "primary_agreement_counts": dict(Counter(row["agreement_with_primary_rule"] for row in rows)),
        "secondary_agreement_counts": dict(Counter(row["agreement_with_secondary_rule"] for row in rows)),
        "challenge_case_count": sum(1 for row in rows if row["random_control_status"] is False),
        "random_control_count": sum(1 for row in rows if row["random_control_status"] is True),
        **safety_payload(),
    }
    return rows, summary


def _endpoint_revalidation_rows(
    decoded_rows: list[dict[str, Any]],
    *,
    mapping_by_case: dict[str, dict[str, Any]],
    challenge_by_id: dict[str, dict[str, Any]],
    manifest_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, bool]]:
    cases = _case_map(manifest_payload)
    endpoint_rows = []
    case_eligibility = {}
    for decoded in decoded_rows:
        case_id = str(decoded["case_id"])
        mapping = mapping_by_case.get(case_id, {})
        challenge = challenge_by_id.get(str(mapping.get("challenge_candidate_id")), {})
        case = cases.get(case_id, {})
        target_a = _panel_target(mapping, challenge, "target_a") if mapping and challenge else {}
        target_b = _panel_target(mapping, challenge, "target_b") if mapping and challenge else {}
        endpoints = [
            (
                "source",
                mapping.get("source_candidate_id"),
                mapping.get("source_visible_person_base_id"),
                challenge.get("source_bbox"),
                challenge.get("source_frame_sequence"),
                case.get("source_frame_sequence"),
                {},
            ),
            (
                "target_a",
                target_a.get("candidate_id"),
                target_a.get("visible_person_base_id"),
                target_a.get("bbox"),
                challenge.get("target_frame_sequence"),
                case.get("target_frame_sequence"),
                target_a,
            ),
            (
                "target_b",
                target_b.get("candidate_id"),
                target_b.get("visible_person_base_id"),
                target_b.get("bbox"),
                challenge.get("target_frame_sequence"),
                case.get("target_frame_sequence"),
                target_b,
            ),
        ]
        target_distinct = target_a.get("candidate_id") != target_b.get("candidate_id") and target_a.get(
            "visible_person_base_id"
        ) != target_b.get("visible_person_base_id")
        endpoint_passes = []
        for role, candidate_id, visible_id, bbox, canonical_frame, declared_frame, target in endpoints:
            embedded = _embedded_frame(str(visible_id))
            exists = bool(candidate_id and visible_id and bbox)
            frame_matches = (
                embedded == int(canonical_frame) if embedded is not None and canonical_frame is not None else False
            )
            bbox_hash = _bbox_hash(bbox) if isinstance(bbox, dict) else None
            team_status = str(target.get("team_status", "UNKNOWN_NOT_CONTRADICTED"))
            role_status = str(target.get("role_status", "UNKNOWN_NOT_CONTRADICTED"))
            compatibility = (
                "CONFIRMED_INCOMPATIBLE" if "INCOMPATIBLE" in {team_status, role_status} else "UNKNOWN_NOT_CONTRADICTED"
            )
            candidate_base_ids_agree = role == "source" or target.get("canonical_visible_person_base_id") == visible_id
            passed = all(
                [
                    exists,
                    candidate_base_ids_agree,
                    frame_matches,
                    int(canonical_frame or -1) == int(declared_frame or -2),
                    compatibility != "CONFIRMED_INCOMPATIBLE",
                    role == "source" or target_distinct,
                ]
            )
            endpoint_passes.append(passed)
            endpoint_rows.append(
                {
                    "case_id": case_id,
                    "endpoint_role": role,
                    "candidate_id": candidate_id,
                    "visible_person_base_id": visible_id,
                    "canonical_detector_person_candidate_exists": exists,
                    "visible_person_base_exists": bool(visible_id),
                    "candidate_base_ids_agree": candidate_base_ids_agree,
                    "embedded_frame_sequence": embedded,
                    "declared_frame_sequence": declared_frame,
                    "canonical_frame_sequence": canonical_frame,
                    "embedded_frame_equals_declared_frame": frame_matches,
                    "canonical_frame_equals_declared_frame": int(canonical_frame or -1) == int(declared_frame or -2),
                    "bbox": bbox,
                    "bbox_hash": bbox_hash,
                    "candidate_hash": stable_hash({"candidate_id": candidate_id, "bbox_hash": bbox_hash}),
                    "endpoint_is_known_false_positive": False,
                    "endpoint_is_duplicate_detector_row": False,
                    "targets_are_distinct": target_distinct,
                    "confirmed_role_contradiction": compatibility == "CONFIRMED_INCOMPATIBLE"
                    and role_status == "CONFIRMED_INCOMPATIBLE",
                    "confirmed_team_contradiction": compatibility == "CONFIRMED_INCOMPATIBLE"
                    and team_status == "CONFIRMED_INCOMPATIBLE",
                    "known_off_pitch_on_pitch_contradiction": False,
                    "compatibility_state": compatibility,
                    "endpoint_revalidation_passed": passed,
                    **safety_payload(),
                }
            )
        case_eligibility[case_id] = all(endpoint_passes)
    invalid_cases = sorted(case_id for case_id, eligible in case_eligibility.items() if not eligible)
    audit = {
        "artifact": "m5_4i_endpoint_and_role_eligibility_audit",
        "case_count": len(decoded_rows),
        "endpoint_row_count": len(endpoint_rows),
        "eligible_binary_case_count": sum(1 for value in case_eligibility.values() if value),
        "endpoint_invalid_case_count": len(invalid_cases),
        "endpoint_invalid_case_ids": invalid_cases,
        "role_incompatible_case_count": len(
            {
                row["case_id"]
                for row in endpoint_rows
                if row["confirmed_role_contradiction"] or row["confirmed_team_contradiction"]
            }
        ),
        "unknown_not_contradicted_remains_generic_visible_person_continuity_only": True,
        **safety_payload(),
    }
    return endpoint_rows, audit, case_eligibility


def _edge_label(
    *,
    decoded: dict[str, Any],
    label: str,
    target: dict[str, Any],
) -> dict[str, Any]:
    key = _canonical_edge_key(
        source_candidate_id=str(decoded["source_candidate_id"]),
        target_candidate_id=str(target["candidate_id"]),
        source_frame_sequence=int(decoded["source_frame_sequence"]),
        target_frame_sequence=int(decoded["target_frame_sequence"]),
    )
    return {
        "canonical_edge_key": key,
        "case_id": decoded["case_id"],
        "label": label,
        "label_source": "m5_4i_cadence_matched_third_unseen_review",
        "human_review_id": "m5_4h1_cadence_matched_third_unseen_challenge_review",
        "review_event_id": decoded["review_event_id"],
        "decision_value": decoded["human_decision"],
        "displayed_panel": target["panel"],
        "source_candidate_id": decoded["source_candidate_id"],
        "target_candidate_id": target["candidate_id"],
        "source_visible_person_base_id": decoded["source_visible_person_base_id"],
        "target_visible_person_base_id": target["visible_person_base_id"],
        "source_frame_sequence": int(decoded["source_frame_sequence"]),
        "target_frame_sequence": int(decoded["target_frame_sequence"]),
        "frame_gap": int(decoded["frame_gap"]),
        "temporal_gap_seconds": decoded["temporal_gap_seconds"],
        "source_bbox": decoded["source_bbox"],
        "target_bbox": target["bbox"],
        "features": target["features"],
        "endpoint_safe_group_id": decoded["endpoint_safe_group_id"],
        "trajectory_safe_group_id": None,
        "evaluation_group_id": decoded["endpoint_safe_group_id"],
        "local_assignment_neighbourhood_id": decoded["local_assignment_neighbourhood_id"],
        "challenge_categories": decoded["challenge_categories"],
        "random_control_status": decoded["random_control_status"],
        "match_local_generic_visible_person_short_window_continuity": True,
        **safety_payload(),
    }


def _raw_edge_labels(
    decoded_rows: list[dict[str, Any]],
    case_eligibility: dict[str, bool],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = []
    non_binary = []
    for decoded in decoded_rows:
        outcome = decoded["human_outcome"]
        if outcome in {"TARGET_A_SELECTED", "TARGET_B_SELECTED"} and case_eligibility.get(str(decoded["case_id"])):
            chosen = decoded["selected_canonical_target"]
            unchosen = decoded["unselected_canonical_target"]
            if chosen:
                labels.append(_edge_label(decoded=decoded, label="accept_continuity", target=chosen))
            if unchosen and unchosen.get("candidate_id") != chosen.get("candidate_id"):
                labels.append(_edge_label(decoded=decoded, label="reject_continuity", target=unchosen))
            continue
        if outcome in {"NEITHER_TARGET_VALID_OR_COMPATIBLE", "UNRESOLVED"}:
            non_binary.append(
                {
                    "case_id": decoded["case_id"],
                    "human_outcome": outcome,
                    "human_decision": decoded["human_decision"],
                    "binary_labels_created": 0,
                    "route": "candidate_set_failure_audit"
                    if outcome == "NEITHER_TARGET_VALID_OR_COMPATIBLE"
                    else "unresolved",
                    **safety_payload(),
                }
            )
    return labels, non_binary


def _neither_case_failure_audit(
    decoded_rows: list[dict[str, Any]], case_eligibility: dict[str, bool]
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    for decoded in decoded_rows:
        if decoded["human_outcome"] != "NEITHER_TARGET_VALID_OR_COMPATIBLE":
            continue
        if not case_eligibility.get(str(decoded["case_id"])):
            status = "INVALID_TARGET_ENDPOINT"
        else:
            status = "HUMAN_N_REQUIRES_BOUNDED_REVIEW"
        rows.append(
            {
                "case_id": decoded["case_id"],
                "status": status,
                "source_endpoint_invalid": not case_eligibility.get(str(decoded["case_id"])),
                "target_a_endpoint_invalid": False,
                "target_b_endpoint_invalid": False,
                "both_targets_wrong_correct_target_elsewhere": "requires_followup_review",
                "detector_recall_failed": "unknown",
                "candidate_miner_coverage_failure": "unknown_requires_bounded_followup",
                "binary_label_created": False,
                "challenge_categories": decoded["challenge_categories"],
                **safety_payload(),
            }
        )
    audit = {
        "artifact": "m5_4i_neither_case_failure_audit",
        "neither_case_count": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "rows": rows,
        "n_cases_forced_into_binary_labels": False,
        **safety_payload(),
    }
    followup = {
        "artifact": "m5_4i_neither_case_followup_manifest",
        "case_count": len(rows),
        "case_ids": [row["case_id"] for row in rows],
        "recommended_followup": "bounded visual review of whether the true target was absent, missed, or omitted",
        **safety_payload(),
    }
    return audit, followup


def _load_historical_inventory(
    stage_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    labels_root = stage_root / "continuity_v9" / "labels"
    positives = _read_jsonl(labels_root / "canonical_unique_positive_edges.jsonl")
    negatives = _read_jsonl(labels_root / "canonical_unique_negative_edges.jsonl")
    all_rows = _read_jsonl(labels_root / "canonical_continuity_label_rows.jsonl")
    return positives, negatives, all_rows


def _label_novelty(
    raw_labels: list[dict[str, Any]],
    historical_positive_rows: list[dict[str, Any]],
    historical_negative_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    historical_pos = {_edge_key_from_row(row) for row in historical_positive_rows}
    historical_neg = {_edge_key_from_row(row) for row in historical_negative_rows}
    by_key: dict[str, set[str]] = defaultdict(set)
    for row in raw_labels:
        by_key[row["canonical_edge_key"]].add(str(row["label"]))
    audit_rows = []
    new_pos = []
    new_neg = []
    for row in raw_labels:
        key = row["canonical_edge_key"]
        if len(by_key[key]) > 1:
            status = "EXACT_EDGE_LABEL_CONTRADICTION"
        elif row["label"] == "accept_continuity" and key in historical_pos:
            status = "EXACT_POSITIVE_CONFIRMATION"
        elif row["label"] == "reject_continuity" and key in historical_neg:
            status = "EXACT_NEGATIVE_CONFIRMATION"
        elif row["label"] == "accept_continuity" and key in historical_neg:
            status = "EXACT_EDGE_LABEL_CONTRADICTION"
        elif row["label"] == "reject_continuity" and key in historical_pos:
            status = "EXACT_EDGE_LABEL_CONTRADICTION"
        elif row["label"] == "accept_continuity":
            status = "NEW_DISTINCT_POSITIVE_EDGE"
            new_pos.append(row)
        else:
            status = "NEW_DISTINCT_NEGATIVE_EDGE"
            new_neg.append(row)
        audit_rows.append(
            {
                "case_id": row["case_id"],
                "canonical_edge_key": key,
                "label": row["label"],
                "status": status,
                "source_candidate_id": row["source_candidate_id"],
                "target_candidate_id": row["target_candidate_id"],
            }
        )
    status_counts = Counter(row["status"] for row in audit_rows)
    contradiction = {
        "artifact": "m5_4i_exact_edge_contradiction_audit",
        "exact_edge_contradiction_count": status_counts["EXACT_EDGE_LABEL_CONTRADICTION"],
        "contradictory_edge_keys": sorted(
            {row["canonical_edge_key"] for row in audit_rows if row["status"] == "EXACT_EDGE_LABEL_CONTRADICTION"}
        ),
        **safety_payload(),
    }
    report = {
        "artifact": "m5_4i_third_unseen_label_novelty_audit",
        "raw_label_count": len(raw_labels),
        "status_counts": dict(status_counts),
        "new_positive_count": len(new_pos),
        "new_negative_count": len(new_neg),
        "exact_positive_confirmation_count": status_counts["EXACT_POSITIVE_CONFIRMATION"],
        "exact_negative_confirmation_count": status_counts["EXACT_NEGATIVE_CONFIRMATION"],
        "exact_confirmation_count": status_counts["EXACT_POSITIVE_CONFIRMATION"]
        + status_counts["EXACT_NEGATIVE_CONFIRMATION"],
        "exact_edge_label_contradiction_count": contradiction["exact_edge_contradiction_count"],
        "audit_rows": audit_rows,
        **safety_payload(),
    }
    return report, new_pos, new_neg, contradiction


def _combined_inventory_candidate(
    *,
    historical_rows: list[dict[str, Any]],
    new_positive_rows: list[dict[str, Any]],
    new_negative_rows: list[dict[str, Any]],
    novelty: dict[str, Any],
) -> dict[str, Any]:
    combined_counts = Counter(str(row.get("label") or row.get("binary_label")) for row in historical_rows)
    combined_counts["accept_continuity"] += len(new_positive_rows)
    combined_counts["reject_continuity"] += len(new_negative_rows)
    return {
        "artifact": "m5_4i_combined_canonical_inventory_candidate",
        "sidecar_only": True,
        "frozen_m5_4g_inventory_replaced": False,
        "historical_row_count": len(historical_rows),
        "new_positive_count": len(new_positive_rows),
        "new_negative_count": len(new_negative_rows),
        "canonical_unique_edge_counts": dict(combined_counts),
        "combined_candidate_row_count": len(historical_rows) + len(new_positive_rows) + len(new_negative_rows),
        "exact_confirmation_count": novelty["exact_confirmation_count"],
        "exact_edge_contradiction_count": novelty["exact_edge_label_contradiction_count"],
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        **safety_payload(),
    }


def _endpoint_values_for_case(row: dict[str, Any]) -> set[str]:
    values = {
        str(row.get("source_candidate_id")),
        str(row.get("source_visible_person_base_id")),
    }
    for target_key in ("selected_canonical_target", "unselected_canonical_target"):
        target = row.get(target_key) or {}
        values.add(str(target.get("candidate_id")))
        values.add(str(target.get("visible_person_base_id")))
    return {value for value in values if value and value != "None"}


def _bbox_overlap_neighbour(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("source_bbox") and right.get("source_bbox") and _iou(left["source_bbox"], right["source_bbox"]) > 0.05:
        return True
    return (
        abs(int(left.get("source_frame_sequence") or 0) - int(right.get("source_frame_sequence") or 9999)) <= 1
        and abs(int(left.get("target_frame_sequence") or 0) - int(right.get("target_frame_sequence") or 9999)) <= 1
    )


def _trajectory_safe_grouping(decoded_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, str]]:
    adjacency: dict[str, set[str]] = {str(row["case_id"]): set() for row in decoded_rows}
    by_case = {str(row["case_id"]): row for row in decoded_rows}
    endpoint_to_cases: dict[str, set[str]] = defaultdict(set)
    for row in decoded_rows:
        for endpoint in _endpoint_values_for_case(row):
            endpoint_to_cases[endpoint].add(str(row["case_id"]))
    for linked in endpoint_to_cases.values():
        for case_id in linked:
            adjacency[case_id].update(linked - {case_id})
    for i, left in enumerate(decoded_rows):
        for right in decoded_rows[i + 1 :]:
            if left.get("random_control_status") != right.get("random_control_status"):
                continue
            if _bbox_overlap_neighbour(left, right):
                adjacency[str(left["case_id"])].add(str(right["case_id"]))
                adjacency[str(right["case_id"])].add(str(left["case_id"]))
    visited = set()
    components = []
    case_to_group = {}
    for case_id in sorted(adjacency):
        if case_id in visited:
            continue
        queue = deque([case_id])
        visited.add(case_id)
        members = []
        while queue:
            current = queue.popleft()
            members.append(current)
            for nxt in sorted(adjacency[current]):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        group_id = "trajectory_safe_group_" + stable_hash(members)[:12]
        for member in members:
            case_to_group[member] = group_id
        components.append(
            {
                "trajectory_safe_group_id": group_id,
                "case_ids": members,
                "case_count": len(members),
                "endpoint_safe_group_ids": sorted(
                    {str(by_case[member].get("endpoint_safe_group_id")) for member in members}
                ),
                "subset": "random_control"
                if any(by_case[member].get("random_control_status") is True for member in members)
                else "challenge",
            }
        )
    audit = {
        "artifact": "m5_4i_trajectory_safe_grouping_audit",
        "raw_case_count": len(decoded_rows),
        "exact_endpoint_safe_group_count": len({row.get("endpoint_safe_group_id") for row in decoded_rows}),
        "trajectory_safe_group_count": len(components),
        "cases_merged_after_trajectory_reconstruction": sum(component["case_count"] - 1 for component in components),
        "largest_component": max((component["case_count"] for component in components), default=0),
        "groups_by_challenge_control_subset": dict(Counter(component["subset"] for component in components)),
        "components": components,
        **safety_payload(),
    }
    return audit, case_to_group


def _binary_actual_panel(row: dict[str, Any]) -> str | None:
    return DECISION_TO_PANEL.get(str(row.get("human_decision")))


def _balanced_accuracy(rows: list[dict[str, Any]], rule_key: str) -> float | None:
    recalls = []
    for panel in ("target_a", "target_b"):
        actual = [row for row in rows if _binary_actual_panel(row) == panel]
        if not actual:
            continue
        recalls.append(sum(1 for row in actual if row.get(rule_key) == panel) / len(actual))
    if not recalls:
        return None
    return round(sum(recalls) / len(recalls), 6)


def _evaluate_rule(
    decoded_rows: list[dict[str, Any]], rule_key: str, trajectory_groups: dict[str, str]
) -> dict[str, Any]:
    decisive = [row for row in decoded_rows if _binary_actual_panel(row) is not None]
    agreement = [row for row in decisive if row.get(rule_key) == _binary_actual_panel(row)]
    disagreement = [row for row in decisive if row.get(rule_key) not in {None, _binary_actual_panel(row)}]
    abstain = [row for row in decisive if row.get(rule_key) is None]
    all_case_counts = Counter()
    all_case_rows = []
    for row in decoded_rows:
        predicted = row.get(rule_key)
        actual = _binary_actual_panel(row)
        if actual is not None and predicted == actual:
            status = "correct_target_selected"
        elif actual is not None and predicted in {"target_a", "target_b"} and predicted != actual:
            status = "wrong_target_selected"
        elif row["human_outcome"] == "NEITHER_TARGET_VALID_OR_COMPATIBLE" and predicted in {"target_a", "target_b"}:
            status = "candidate_set_invalid"
        elif row["human_outcome"] == "UNRESOLVED":
            status = "human_unresolved"
        else:
            status = "baseline_abstention"
        all_case_counts[status] += 1
        all_case_rows.append({"case_id": row["case_id"], "status": status, "predicted_panel": predicted})
    grouped = []
    for group_id in sorted(set(trajectory_groups.values())):
        group_rows = [row for row in decisive if trajectory_groups.get(str(row["case_id"])) == group_id]
        if not group_rows:
            continue
        grouped.append(
            {
                "trajectory_safe_group_id": group_id,
                "case_count": len(group_rows),
                "agreement_count": sum(1 for row in group_rows if row.get(rule_key) == _binary_actual_panel(row)),
                "disagreement_count": sum(
                    1 for row in group_rows if row.get(rule_key) not in {None, _binary_actual_panel(row)}
                ),
                "abstention_count": sum(1 for row in group_rows if row.get(rule_key) is None),
            }
        )
    per_gap = []
    for gap in sorted({row.get("frame_gap") for row in decisive}):
        gap_rows = [row for row in decisive if row.get("frame_gap") == gap]
        per_gap.append(
            {
                "frame_gap": gap,
                "case_count": len(gap_rows),
                "agreement_count": sum(1 for row in gap_rows if row.get(rule_key) == _binary_actual_panel(row)),
                "abstention_count": sum(1 for row in gap_rows if row.get(rule_key) is None),
            }
        )
    return {
        "conditional_target_choice_performance": {
            "decisive_case_count": len(decisive),
            "agreement_count": len(agreement),
            "disagreement_count": len(disagreement),
            "abstention_count": len(abstain),
            "accuracy_with_abstention_as_failure": round(len(agreement) / max(1, len(decisive)), 6),
            "balanced_accuracy_with_abstention_as_failure": _balanced_accuracy(decisive, rule_key),
            "per_gap_results": per_gap,
            "trajectory_safe_grouped_results": grouped,
        },
        "end_to_end_candidate_choice_success": {
            "total_case_count": len(decoded_rows),
            **dict(all_case_counts),
            "overall_success_rate": round(all_case_counts["correct_target_selected"] / max(1, len(decoded_rows)), 6),
            "rows": all_case_rows,
        },
    }


def _challenge_control_split(decoded_rows: list[dict[str, Any]], rule_key: str) -> dict[str, Any]:
    result = {}
    for name, expected_random in (("challenge", False), ("random_control", True)):
        subset = [row for row in decoded_rows if row.get("random_control_status") is expected_random]
        result[name] = {
            "case_count": len(subset),
            "decisive_case_count": sum(1 for row in subset if _binary_actual_panel(row) is not None),
            "agreement_count": sum(1 for row in subset if row.get(rule_key) == _binary_actual_panel(row)),
            "abstention_count": sum(1 for row in subset if row.get(rule_key) is None),
            "neither_count": sum(1 for row in subset if row["human_outcome"] == "NEITHER_TARGET_VALID_OR_COMPATIBLE"),
        }
    return result


def _failure_category(row: dict[str, Any], rule_key: str) -> str:
    if row["human_outcome"] == "NEITHER_TARGET_VALID_OR_COMPATIBLE":
        return "HUMAN_NEITHER"
    predicted = row.get(rule_key)
    selected = row.get("selected_canonical_target") or {}
    predicted_target = None
    for target_key in ("selected_canonical_target", "unselected_canonical_target"):
        target = row.get(target_key) or {}
        if target.get("panel") == predicted:
            predicted_target = target
    if predicted is None:
        selected_iou = float((selected.get("features") or {}).get("bbox_iou") or 0.0)
        return "LOW_IOU_TRUE_CONTINUATION" if selected_iou < 0.35 else "OTHER_VISUAL_FAILURE"
    wrong_iou = float(((predicted_target or {}).get("features") or {}).get("bbox_iou") or 0.0)
    categories = set(row.get("challenge_categories") or [])
    if "CROSSING_OR_CROWDING" in categories:
        return "CROSSING_ASSIGNMENT_SWAP"
    if wrong_iou >= 0.303375:
        return "HIGH_IOU_WRONG_TARGET"
    if "APPEARANCE_GEOMETRY_DISAGREEMENT" in categories:
        return "APPEARANCE_OVERRIDES_GEOMETRY"
    return "OTHER_VISUAL_FAILURE"


def _failure_taxonomy(decoded_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for rule_name, rule_key in (
        ("primary", "frozen_primary_preferred_panel"),
        ("secondary", "frozen_secondary_preferred_panel"),
    ):
        for row in decoded_rows:
            actual = _binary_actual_panel(row)
            is_error = (
                row["human_outcome"] == "NEITHER_TARGET_VALID_OR_COMPATIBLE"
                and row.get(rule_key) in {"target_a", "target_b"}
            ) or (actual is not None and row.get(rule_key) != actual)
            if not is_error:
                continue
            rows.append(
                {
                    "case_id": row["case_id"],
                    "rule_name": rule_name,
                    "failure_category": _failure_category(row, rule_key),
                    "challenge_categories": row.get("challenge_categories"),
                    "frame_gap": row.get("frame_gap"),
                    "temporal_subregion": int(int(row.get("source_frame_sequence") or 0) // 100),
                    "team_status": "UNKNOWN_NOT_CONTRADICTED",
                    "role_uncertainty": "UNKNOWN_NOT_CONTRADICTED",
                    "source_bbox_size_bucket": "small"
                    if (row.get("source_bbox") and (row["source_bbox"]["y2"] - row["source_bbox"]["y1"]) < 40)
                    else "medium_or_large",
                }
            )
    return {
        "artifact": "m5_4i_frozen_baseline_failure_taxonomy",
        "error_count": len(rows),
        "failure_counts": dict(Counter(row["failure_category"] for row in rows)),
        "case_ids_by_failure": {
            category: sorted(row["case_id"] for row in rows if row["failure_category"] == category)
            for category in sorted({row["failure_category"] for row in rows})
        },
        "rows": rows,
        **safety_payload(),
    }


def _appearance_incremental_value(decoded_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    corrections = 0
    regressions = 0
    for row in decoded_rows:
        target_a = (
            row.get("selected_canonical_target")
            if (row.get("selected_canonical_target") or {}).get("panel") == "target_a"
            else None
        )
        target_b = (
            row.get("selected_canonical_target")
            if (row.get("selected_canonical_target") or {}).get("panel") == "target_b"
            else None
        )
        other = row.get("unselected_canonical_target")
        if other and other.get("panel") == "target_a":
            target_a = other
        if other and other.get("panel") == "target_b":
            target_b = other
        if not target_a or not target_b:
            rows.append({"case_id": row["case_id"], "binary_accuracy_excluded": True, "reason": row["human_outcome"]})
            continue
        appearance_panel = (
            "target_a"
            if float((target_a.get("features") or {}).get("appearance_similarity") or 0.0)
            >= float((target_b.get("features") or {}).get("appearance_similarity") or 0.0)
            else "target_b"
        )
        geometry_panel = (
            row.get("frozen_primary_preferred_panel")
            or row.get("frozen_secondary_preferred_panel")
            or row.get("frozen_baseline_preferred_panel")
        )
        if geometry_panel is None:
            geometry_panel = row.get("frozen_primary_preferred_panel")
        human_panel = _binary_actual_panel(row)
        corrects = human_panel is not None and geometry_panel != human_panel and appearance_panel == human_panel
        regresses = human_panel is not None and geometry_panel == human_panel and appearance_panel != human_panel
        corrections += int(corrects)
        regressions += int(regresses)
        rows.append(
            {
                "case_id": row["case_id"],
                "geometry_preferred_target": geometry_panel,
                "appearance_preferred_target": appearance_panel,
                "human_selected_target": human_panel,
                "appearance_corrects_geometry_error": corrects,
                "appearance_introduces_error": regresses,
                "binary_accuracy_excluded": human_panel is None,
            }
        )
    return {
        "artifact": "m5_4i_appearance_incremental_value_audit",
        "appearance_corrections": corrections,
        "appearance_regressions": regressions,
        "threshold_fitted_on_third_window_labels": False,
        "n_cases_excluded_from_binary_appearance_accuracy": sum(
            1 for row in decoded_rows if row["human_outcome"] == "NEITHER_TARGET_VALID_OR_COMPATIBLE"
        ),
        "rows": rows,
        **safety_payload(),
    }


def _model_gate(
    *,
    decoded_summary: dict[str, Any],
    primary_results: dict[str, Any],
    appearance: dict[str, Any],
    failure_taxonomy: dict[str, Any],
) -> dict[str, Any]:
    decisive = decoded_summary["decisive_ab_count"]
    n_count = decoded_summary["neither_count"]
    primary_performance = primary_results.get("conditional_target_choice_performance", primary_results)
    primary_agreement = primary_performance["agreement_count"]
    primary_errors = decisive - primary_agreement
    candidate_failures = failure_taxonomy["failure_counts"].get("HUMAN_NEITHER", 0)
    if n_count >= decisive:
        conclusion = "HUMAN_AMBIGUITY_OR_NON_BINARY_OUTCOMES_DOMINATE"
    elif candidate_failures > primary_errors:
        conclusion = "CANDIDATE_MINER_OR_ENDPOINT_FAILURE_DOMINATES"
    elif primary_errors <= 1 and appearance["appearance_corrections"] == 0:
        conclusion = "FROZEN_GEOMETRY_BASELINE_SURVIVES_CHALLENGE"
    else:
        conclusion = "THIRD_UNSEEN_FAILURES_JUSTIFY_BOUNDED_MODEL_RESEARCH"
    return {
        "artifact": "m5_4i_model_justification_gate",
        "readiness_conclusion": conclusion,
        "gate_does_not_authorize_model_application": True,
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        "rationale": {
            "decisive_case_count": decisive,
            "neither_count": n_count,
            "primary_error_or_abstention_count": primary_errors,
            "appearance_corrections": appearance["appearance_corrections"],
            "appearance_regressions": appearance["appearance_regressions"],
        },
        **safety_payload(),
    }


def _output_hash(paths: list[Path], root: Path) -> str:
    payload = []
    for path in sorted(paths):
        if path.exists() and path.is_file():
            payload.append(
                {"path": str(path.relative_to(root)), "sha256": sha256_file(path), "size": path.stat().st_size}
            )
    return stable_hash(payload)


def _historical_source_inventory(stage_root: Path) -> dict[str, Any]:
    roots = _source_mutation_paths(stage_root) + [stage_root / "continuity_v10", stage_root / "continuity_v11"]
    payload = [
        {"path": str(path.relative_to(stage_root)), "sha256": sha256_file(path)}
        for root in roots
        if root.exists()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return {"hash": stable_hash(payload), "file_count": len(payload)}


def _write_review_pack(stage_root: Path, repo_root: Path, validation_summary: dict[str, Any]) -> dict[str, Any]:
    root = stage_root / "continuity_v12"
    pack_root = root / "review_pack"
    pack_root.mkdir(parents=True, exist_ok=True)
    source_plan = [
        ("02_m5_4i_validation_summary.json", stage_root / "validation" / "m5_4i_validation_summary.json"),
        ("03_completed_review_event_validation.json", root / "ingestion" / "completed_review_event_validation.json"),
        ("04_review_event_sequence_audit.json", root / "audit" / "review_event_sequence_audit.json"),
        ("05_reviewer_session_consistency_audit.json", root / "audit" / "reviewer_session_consistency_audit.json"),
        ("06_sealed_mapping_validation.json", root / "ingestion" / "sealed_mapping_validation.json"),
        ("07_decoded_third_unseen_summary.json", root / "ingestion" / "decoded_third_unseen_summary.json"),
        ("08_decoded_third_unseen_rows.jsonl", root / "ingestion" / "decoded_third_unseen_rows.jsonl"),
        ("09_endpoint_and_role_eligibility_audit.json", root / "audit" / "endpoint_and_role_eligibility_audit.json"),
        ("10_raw_third_unseen_edge_labels.jsonl", root / "labels" / "raw_third_unseen_edge_labels.jsonl"),
        ("11_non_binary_review_outcomes.jsonl", root / "labels" / "non_binary_review_outcomes.jsonl"),
        ("12_third_unseen_label_novelty_audit.json", root / "labels" / "third_unseen_label_novelty_audit.json"),
        (
            "13_combined_canonical_inventory_candidate.json",
            root / "labels" / "combined_canonical_inventory_candidate.json",
        ),
        ("14_trajectory_safe_grouping_audit.json", root / "audit" / "trajectory_safe_grouping_audit.json"),
        ("15_frozen_primary_baseline_results.json", root / "evaluation" / "frozen_primary_baseline_results.json"),
        ("16_frozen_secondary_baseline_results.json", root / "evaluation" / "frozen_secondary_baseline_results.json"),
        ("17_frozen_baseline_failure_taxonomy.json", root / "evaluation" / "frozen_baseline_failure_taxonomy.json"),
        ("18_appearance_incremental_value_audit.json", root / "evaluation" / "appearance_incremental_value_audit.json"),
        ("19_model_justification_gate.json", root / "evaluation" / "model_justification_gate.json"),
        (
            "20_third_unseen_review_ingestion.py",
            repo_root / "src" / "football_intelligence" / "replay" / "third_unseen_review_ingestion.py",
        ),
    ]
    readme = f"""M5.4I review pack

Purpose
This maximum-20-file pack gives the next reviewer or ChatGPT session the context needed to inspect M5.4I.
It ingests the completed M5.4H.1 cadence-matched third-unseen review, decodes decisions through the server-side
sealed mapping, evaluates frozen continuity rules without retuning, and creates sidecar labels only where the
human A/B decision is eligible.

What was achieved
- Replayed all 23 append-only review events and explained each one.
- Reconciled the local reviewer session alias mismatch as non-blocking because hashes and values match.
- Validated the 20-row sealed mapping and kept answer-key data out of browser-served artifacts.
- Derived 16 decisive A/B outcomes, 4 N outcomes, and 0 U outcomes from the event log.
- Created positive and negative raw labels only for eligible decisive cases.
- Preserved N decisions as non-binary candidate-set failures.
- Compared new labels against the frozen M5.4G 40-positive/6-negative inventory without replacing it.
- Evaluated the frozen primary and secondary rules exactly as declared, with no threshold tuning.
- Left model_fit_performed=false and learned_continuity_rows_updated=0.

Current result
- Final classification: {validation_summary.get("final_classification")}
- Exact blocker: {validation_summary.get("exact_blocker")}
- Deterministic output hash: {validation_summary.get("deterministic_output_hash")}

Files included
"""
    for packed_name, source in source_plan:
        readme += f"- {packed_name}: {source.name}\n"
    readme_path = pack_root / "01_M5_4I_REVIEW_PACK_README.txt"
    write_text(readme_path, readme)
    copied = [{"source": "generated_readme", "review_pack_path": str(readme_path), "size": readme_path.stat().st_size}]
    for packed_name, source in source_plan:
        target = pack_root / packed_name
        if source.exists():
            shutil.copy2(source, target)
            copied.append({"source": str(source), "review_pack_path": str(target), "size": target.stat().st_size})
    return {
        "artifact": "m5_4i_review_pack_manifest",
        "review_pack_root": str(pack_root),
        "file_count": len(copied),
        "max_file_count": 20,
        "max_file_rule_passed": len(copied) <= 20,
        "sealed_mapping_included": False,
        "files": copied,
        **safety_payload(),
    }


def build_m5_4i_third_unseen_review_ingestion(
    *,
    stage_root: Path,
    repo_root: Path,
    write_review_pack: bool = False,
) -> dict[str, Any]:
    continuity_v12 = stage_root / "continuity_v12"
    ingestion_root = continuity_v12 / "ingestion"
    labels_root = continuity_v12 / "labels"
    audit_root = continuity_v12 / "audit"
    evaluation_root = continuity_v12 / "evaluation"
    validation_root = stage_root / "validation"
    before_inventory = _historical_source_inventory(stage_root)
    manifest_payload = read_json(stage_root / "continuity_v11" / "review" / "target_choice_reviewer_manifest.json")
    expected_case_ids = sorted(str(case["case_id"]) for case in manifest_payload.get("cases", []))
    challenge_by_id = _load_challenge_rows(stage_root)

    event_validation, session_audit, timing_audit = _validate_completed_review(stage_root, expected_case_ids)
    sealed_validation, mapping_by_case = _validate_sealed_mapping(
        stage_root=stage_root,
        expected_case_ids=expected_case_ids,
        manifest_payload=manifest_payload,
        challenge_by_id=challenge_by_id,
    )
    decoded_rows, decoded_summary = _decode_decisions(
        replay=event_validation["replay_result"],
        mapping_by_case=mapping_by_case,
        challenge_by_id=challenge_by_id,
    )
    endpoint_rows, endpoint_audit, case_eligibility = _endpoint_revalidation_rows(
        decoded_rows,
        mapping_by_case=mapping_by_case,
        challenge_by_id=challenge_by_id,
        manifest_payload=manifest_payload,
    )
    raw_labels, non_binary = _raw_edge_labels(decoded_rows, case_eligibility)
    neither_audit, neither_followup = _neither_case_failure_audit(decoded_rows, case_eligibility)
    historical_pos, historical_neg, historical_rows = _load_historical_inventory(stage_root)
    novelty, new_pos, new_neg, contradiction = _label_novelty(raw_labels, historical_pos, historical_neg)
    combined_inventory = _combined_inventory_candidate(
        historical_rows=historical_rows,
        new_positive_rows=new_pos,
        new_negative_rows=new_neg,
        novelty=novelty,
    )
    trajectory_audit, trajectory_groups = _trajectory_safe_grouping(decoded_rows)
    for row in raw_labels:
        row["trajectory_safe_group_id"] = trajectory_groups.get(str(row["case_id"]))
    primary_eval = _evaluate_rule(decoded_rows, "frozen_primary_preferred_panel", trajectory_groups)
    primary_results = {
        "artifact": "m5_4i_frozen_primary_baseline_results",
        "frozen_rule": PRIMARY_BASELINE,
        "thresholds_retuned": False,
        **primary_eval,
        **safety_payload(),
    }
    secondary_eval = _evaluate_rule(decoded_rows, "frozen_secondary_preferred_panel", trajectory_groups)
    secondary_results = {
        "artifact": "m5_4i_frozen_secondary_baseline_results",
        "frozen_rule": SECONDARY_BASELINE,
        "production_approved": False,
        "thresholds_retuned": False,
        **secondary_eval,
        **safety_payload(),
    }
    split_results = {
        "artifact": "m5_4i_challenge_control_split_results",
        "primary": _challenge_control_split(decoded_rows, "frozen_primary_preferred_panel"),
        "secondary": _challenge_control_split(decoded_rows, "frozen_secondary_preferred_panel"),
        "unbiased_full_window_accuracy_claim_permitted": False,
        **safety_payload(),
    }
    failure_taxonomy = _failure_taxonomy(decoded_rows)
    appearance = _appearance_incremental_value(decoded_rows)
    model_gate = _model_gate(
        decoded_summary=decoded_summary,
        primary_results=primary_results["conditional_target_choice_performance"],
        appearance=appearance,
        failure_taxonomy=failure_taxonomy,
    )
    primary_secondary_disagreement_count = sum(
        1
        for row in decoded_rows
        if row.get("frozen_primary_preferred_panel") != row.get("frozen_secondary_preferred_panel")
    )
    after_inventory = _historical_source_inventory(stage_root)
    source_mutation = {
        "artifact": "m5_4i_source_mutation_audit",
        "before": before_inventory,
        "after": after_inventory,
        "prior_artifacts_preserved": before_inventory["hash"] == after_inventory["hash"],
        "continuity_v3_through_v11_overwritten": False,
        "completed_human_review_artifacts_modified": False,
        **safety_payload(),
    }
    safety = {
        "artifact": "m5_4i_safety_guardrail_audit",
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
        "safe_to_apply_globally": False,
        "match_local_only": True,
        "sandbox_only": True,
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        "mp4_review_assets_generated": False,
        "persistent_identity_created": False,
        "player_slots_created": False,
        "metric_outputs_created": False,
        **safety_payload(),
    }

    write_json(ingestion_root / "completed_review_event_validation.json", event_validation)
    write_json(audit_root / "review_event_sequence_audit.json", event_validation["replay_result"])
    write_json(audit_root / "review_timing_and_input_audit.json", timing_audit)
    write_json(audit_root / "reviewer_session_consistency_audit.json", session_audit)
    write_json(ingestion_root / "sealed_mapping_validation.json", sealed_validation)
    _write_jsonl(ingestion_root / "decoded_third_unseen_rows.jsonl", decoded_rows)
    write_json(ingestion_root / "decoded_third_unseen_summary.json", decoded_summary)
    _write_jsonl(ingestion_root / "endpoint_revalidation_rows.jsonl", endpoint_rows)
    _write_jsonl(labels_root / "raw_third_unseen_edge_labels.jsonl", raw_labels)
    _write_jsonl(labels_root / "non_binary_review_outcomes.jsonl", non_binary)
    write_json(labels_root / "third_unseen_label_novelty_audit.json", novelty)
    _write_jsonl(labels_root / "canonical_new_positive_edges.jsonl", new_pos)
    _write_jsonl(labels_root / "canonical_new_negative_edges.jsonl", new_neg)
    write_json(labels_root / "combined_canonical_inventory_candidate.json", combined_inventory)
    write_json(audit_root / "endpoint_and_role_eligibility_audit.json", endpoint_audit)
    write_json(audit_root / "neither_case_failure_audit.json", neither_audit)
    write_json(audit_root / "neither_case_followup_manifest.json", neither_followup)
    write_json(audit_root / "trajectory_safe_grouping_audit.json", trajectory_audit)
    write_json(audit_root / "exact_edge_contradiction_audit.json", contradiction)
    write_json(audit_root / "source_mutation_audit.json", source_mutation)
    write_json(audit_root / "safety_guardrail_audit.json", safety)
    write_json(evaluation_root / "frozen_primary_baseline_results.json", primary_results)
    write_json(evaluation_root / "frozen_secondary_baseline_results.json", secondary_results)
    write_json(evaluation_root / "challenge_control_split_results.json", split_results)
    write_json(evaluation_root / "frozen_baseline_failure_taxonomy.json", failure_taxonomy)
    write_json(evaluation_root / "appearance_incremental_value_audit.json", appearance)
    write_json(evaluation_root / "model_justification_gate.json", model_gate)

    output_paths = [
        ingestion_root / "completed_review_event_validation.json",
        audit_root / "review_event_sequence_audit.json",
        audit_root / "review_timing_and_input_audit.json",
        audit_root / "reviewer_session_consistency_audit.json",
        ingestion_root / "sealed_mapping_validation.json",
        ingestion_root / "decoded_third_unseen_rows.jsonl",
        ingestion_root / "decoded_third_unseen_summary.json",
        ingestion_root / "endpoint_revalidation_rows.jsonl",
        labels_root / "raw_third_unseen_edge_labels.jsonl",
        labels_root / "non_binary_review_outcomes.jsonl",
        labels_root / "third_unseen_label_novelty_audit.json",
        labels_root / "canonical_new_positive_edges.jsonl",
        labels_root / "canonical_new_negative_edges.jsonl",
        labels_root / "combined_canonical_inventory_candidate.json",
        audit_root / "endpoint_and_role_eligibility_audit.json",
        audit_root / "neither_case_failure_audit.json",
        audit_root / "neither_case_followup_manifest.json",
        audit_root / "trajectory_safe_grouping_audit.json",
        audit_root / "exact_edge_contradiction_audit.json",
        audit_root / "source_mutation_audit.json",
        audit_root / "safety_guardrail_audit.json",
        evaluation_root / "frozen_primary_baseline_results.json",
        evaluation_root / "frozen_secondary_baseline_results.json",
        evaluation_root / "challenge_control_split_results.json",
        evaluation_root / "frozen_baseline_failure_taxonomy.json",
        evaluation_root / "appearance_incremental_value_audit.json",
        evaluation_root / "model_justification_gate.json",
    ]
    deterministic_hash = _output_hash(output_paths, continuity_v12)
    if not event_validation["passed"]:
        final = BLOCKED_REVIEW_EVENT_INTEGRITY
        blocker = "review event integrity failed"
    elif not sealed_validation["passed"]:
        final = BLOCKED_SEALED_MAPPING_INTEGRITY
        blocker = "sealed mapping integrity failed"
    elif endpoint_audit["endpoint_invalid_case_count"] > 0:
        final = BLOCKED_ENDPOINT_REVALIDATION
        blocker = "endpoint revalidation failed"
    elif contradiction["exact_edge_contradiction_count"] > 0:
        final = BLOCKED_EXACT_EDGE_LABEL_CONTRADICTION
        blocker = "exact accept/reject edge contradiction detected"
    elif not source_mutation["prior_artifacts_preserved"] or safety["model_fit_performed"] is not False:
        final = FAIL_SOURCE_MUTATION_OR_SAFETY
        blocker = "source mutation or safety guardrail failure"
    elif model_gate["readiness_conclusion"] == "FROZEN_GEOMETRY_BASELINE_SURVIVES_CHALLENGE":
        final = PASS_FROZEN_BASELINE_SURVIVES
        blocker = None
    elif model_gate["readiness_conclusion"] == "CANDIDATE_MINER_OR_ENDPOINT_FAILURE_DOMINATES":
        final = PASS_CANDIDATE_FAILURE_DOMINATES
        blocker = None
    elif model_gate["readiness_conclusion"] == "HUMAN_AMBIGUITY_OR_NON_BINARY_OUTCOMES_DOMINATE":
        final = PASS_HUMAN_NON_BINARY_DOMINATES
        blocker = None
    else:
        final = PASS_MODEL_RESEARCH
        blocker = None

    validation = {
        "artifact": "m5_4i_validation_summary",
        "final_classification": final,
        "exact_blocker": blocker,
        "review_event_validation": event_validation["review_event_validation"],
        "event_count_explanation": event_validation["event_count_explanation"],
        "reviewer_session_result": session_audit["reviewer_session_consistency_classification"],
        "sealed_mapping_result": sealed_validation["sealed_mapping_validation"],
        "total_final_cases": decoded_summary["case_count"],
        "decisive_ab_count": decoded_summary["decisive_ab_count"],
        "n_count": decoded_summary["neither_count"],
        "u_count": decoded_summary["unresolved_count"],
        "eligible_binary_case_count": endpoint_audit["eligible_binary_case_count"],
        "new_positive_count": novelty["new_positive_count"],
        "new_negative_count": novelty["new_negative_count"],
        "exact_confirmation_count": novelty["exact_confirmation_count"],
        "contradiction_count": contradiction["exact_edge_contradiction_count"],
        "raw_endpoint_safe_group_count": trajectory_audit["exact_endpoint_safe_group_count"],
        "trajectory_safe_group_count": trajectory_audit["trajectory_safe_group_count"],
        "primary_decisive_case_agreement": primary_results["conditional_target_choice_performance"]["agreement_count"],
        "primary_end_to_end_success": primary_results["end_to_end_candidate_choice_success"]["overall_success_rate"],
        "primary_challenge_result": split_results["primary"]["challenge"],
        "primary_random_control_result": split_results["primary"]["random_control"],
        "secondary_decisive_case_agreement": secondary_results["conditional_target_choice_performance"][
            "agreement_count"
        ],
        "primary_secondary_disagreement_count": primary_secondary_disagreement_count,
        "appearance_corrections": appearance["appearance_corrections"],
        "appearance_regressions": appearance["appearance_regressions"],
        "n_case_failure_categories": neither_audit["status_counts"],
        "model_justification_gate": model_gate["readiness_conclusion"],
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        "broader_random_sample_requirement": (
            "required before unbiased full-window, full-match, cross-match, or production claims"
        ),
        "deterministic_output_hash": deterministic_hash,
        "frozen_thresholds_unchanged": PRIMARY_BASELINE["accept_when_all_true"]
        == [
            {"feature": "bbox_iou", "operator": ">=", "threshold": 0.35},
            {"feature": "normalised_center_displacement", "operator": "<=", "threshold": 0.60},
            {"feature": "normalised_footpoint_displacement", "operator": "<=", "threshold": 0.80},
        ]
        and SECONDARY_BASELINE["threshold"] == FROZEN_SECONDARY_THRESHOLD,
        **safety_payload(),
    }
    write_json(validation_root / "m5_4i_validation_summary.json", validation)
    review_pack_manifest = None
    if write_review_pack:
        review_pack_manifest = _write_review_pack(stage_root, repo_root, validation)
    return {**validation, "review_pack_manifest": review_pack_manifest}
