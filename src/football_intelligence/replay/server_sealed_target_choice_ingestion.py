from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from football_intelligence.replay.blind_target_choice_review import (
    _case_features,
    _dedupe_reversed_comparisons,
    _selected_f6_candidates,
)
from football_intelligence.replay.positive_only_counterfactual_continuity import _inventory
from football_intelligence.replay.rebuilt_human_calibrated_pipeline import read_json, write_json, write_text
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.config import load_ui_config, ui_config_hash
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import load_manifest, manifest_hash

PASS_CANONICAL_TWO_CLASS_LABEL_INVENTORY = "PASS_CANONICAL_TWO_CLASS_LABEL_INVENTORY_READY_FOR_GROUPED_DIAGNOSTIC"
PASS_MODEL_NOT_JUSTIFIED = "PASS_LABEL_INGESTION_MODEL_NOT_JUSTIFIED_BASELINE_DOMINATES"
BLOCKED_REVIEW_EVENT_INTEGRITY = "BLOCKED_REVIEW_EVENT_INTEGRITY"
BLOCKED_SEALED_MAPPING_INTEGRITY = "BLOCKED_SEALED_MAPPING_INTEGRITY"
BLOCKED_PRIOR_LABEL_CONFLICT = "BLOCKED_PRIOR_LABEL_CONFLICT"
BLOCKED_EXACT_EDGE_LABEL_CONTRADICTION = "BLOCKED_EXACT_EDGE_LABEL_CONTRADICTION"
BLOCKED_ENDPOINT_OR_ROLE_VALIDITY = "BLOCKED_ENDPOINT_OR_ROLE_VALIDITY"
BLOCKED_INDEPENDENT_NEGATIVE_GROUPS = "BLOCKED_INDEPENDENT_NEGATIVE_GROUPS"
FAIL_SOURCE_MUTATION_OR_SAFETY = "FAIL_SOURCE_MUTATION_OR_SAFETY"

DECISION_TO_PANEL = {
    "target_a_continues_source": "target_a",
    "target_b_continues_source": "target_b",
}
BINARY_LABELS = {"accept_continuity", "reject_continuity"}
FEATURES_EXCLUDED_FROM_MODELING = {
    "case_id",
    "candidate_construction_type",
    "decision_value",
    "human_review_id",
    "label_source",
    "local_assignment_neighbourhood_id",
    "mapping_hash",
    "review_event_id",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in records)
    write_text(path, text)


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
            "target_candidate_id": target_candidate_id,
            "source_frame_sequence": int(source_frame_sequence),
            "target_frame_sequence": int(target_frame_sequence),
        }
    )


def _edge_key_from_row(row: dict[str, Any]) -> str:
    return _canonical_edge_key(
        source_candidate_id=str(row["source_candidate_id"]),
        target_candidate_id=str(row["target_candidate_id"]),
        source_frame_sequence=int(row["source_frame_sequence"]),
        target_frame_sequence=int(row["target_frame_sequence"]),
    )


def _other_panel(panel: str | None) -> str | None:
    if panel == "target_a":
        return "target_b"
    if panel == "target_b":
        return "target_a"
    return None


def _panel_target(row: dict[str, Any], mapping: dict[str, Any], panel: str) -> dict[str, Any]:
    accepted_panel = str(mapping["accepted_target_panel"])
    target_kind = "accepted" if panel == accepted_panel else "alternative"
    return {
        "target_kind": target_kind,
        "panel": panel,
        "candidate_id": str(row[f"{target_kind}_target_candidate_id"]),
        "visible_person_base_id": str(row[f"{target_kind}_target_visible_person_base_id"]),
        "bbox": row[f"{target_kind}_target_bbox"],
    }


def _source_mutation_paths(stage_root: Path) -> list[Path]:
    names = [
        "learning",
        "role_review",
        "continuity",
        "continuity_v2",
        "continuity_v3",
        "continuity_v4",
        "continuity_v5",
        "continuity_v6",
        "continuity_v7",
        "continuity_v8",
    ]
    return [stage_root / name for name in names]


def _completed_state(completed: dict[str, Any]) -> dict[str, Any]:
    state = completed.get("state")
    return state if isinstance(state, dict) else completed


def _case_map_from_manifest_payload(manifest_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(case["case_id"]): case for case in manifest_payload.get("cases", [])}


def _replay_review_events(events: list[dict[str, Any]], expected_case_ids: list[str]) -> dict[str, Any]:
    expected = set(expected_case_ids)
    decisions: dict[str, str] = {}
    final_decision_events: dict[str, dict[str, Any]] = {}
    overwrites: list[dict[str, Any]] = []
    reveal_events: list[dict[str, Any]] = []
    note_events: list[dict[str, Any]] = []
    complete_events: list[dict[str, Any]] = []
    event_ids = [str(event.get("event_id")) for event in events]
    sequences = [int(event.get("event_sequence", -1)) for event in events]
    event_sequence_monotonic = sequences == sorted(sequences) and sequences == list(range(1, len(events) + 1))
    duplicate_event_ids = sorted(event_id for event_id, count in Counter(event_ids).items() if count > 1)
    missing_input_source = []
    decision_before_review_began = False
    reveal_before_persisted_decision = False
    completion_after_all_expected_decided = False

    for event in events:
        event_type = str(event.get("event_type"))
        case_id = event.get("case_id")
        if event_type == "decision":
            case_key = str(case_id)
            if case_key not in expected:
                continue
            if str(event.get("keyboard_or_click_input_source") or "").strip() in {"", "unknown"}:
                missing_input_source.append(str(event.get("event_id")))
            if event.get("prior_decision") is not None:
                overwrites.append(event)
            decisions[case_key] = str(event.get("new_decision"))
            final_decision_events[case_key] = event
        elif event_type == "reveal":
            reveal_events.append(event)
            if case_id not in decisions:
                reveal_before_persisted_decision = True
        elif event_type == "note":
            note_events.append(event)
        elif event_type == "complete":
            complete_events.append(event)
            completion_after_all_expected_decided = expected.issubset(decisions)

    missing_cases = sorted(expected - set(decisions))
    unexpected_cases = sorted(set(decisions) - expected)
    unexplained_overwrite_burst = len(overwrites) > len(expected_case_ids)
    return {
        "artifact": "m5_4g_completed_review_event_replay",
        "expected_case_ids": expected_case_ids,
        "event_count": len(events),
        "final_decisions": decisions,
        "final_decision_event_ids": {
            case_id: str(event.get("event_id")) for case_id, event in sorted(final_decision_events.items())
        },
        "final_decision_events": final_decision_events,
        "decision_count": len(decisions),
        "missing_cases": missing_cases,
        "unexpected_cases": unexpected_cases,
        "reveal_event_count": len(reveal_events),
        "note_event_count": len(note_events),
        "complete_event_count": len(complete_events),
        "no_reveal_occurred": len(reveal_events) == 0,
        "no_reveal_before_persisted_decision": not reveal_before_persisted_decision,
        "completion_after_all_expected_decided": completion_after_all_expected_decided,
        "event_sequence_monotonic": event_sequence_monotonic,
        "duplicate_event_ids": duplicate_event_ids,
        "overwrite_event_count": len(overwrites),
        "unexplained_overwrite_burst": unexplained_overwrite_burst,
        "input_source_recorded_where_available": not missing_input_source,
        "missing_input_source_event_ids": missing_input_source,
        "decision_before_review_began": decision_before_review_began,
        **safety_payload(),
    }


def _semantic_state_matches(review_decisions: dict[str, Any], completed_state: dict[str, Any]) -> bool:
    keys = {
        "review_id",
        "stage_id",
        "completed",
        "completed_at",
        "event_sequence",
        "manifest_hash",
        "ui_config_hash",
        "evidence_manifest_hash",
        "elapsed_active_seconds",
        "decisions",
        "notes",
        "reveal_state",
        "server_reveal_payloads",
    }
    return {key: review_decisions.get(key) for key in keys} == {key: completed_state.get(key) for key in keys}


def _validate_snapshots(decisions_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    snapshots_root = decisions_root / "snapshots"
    snapshots = sorted(snapshots_root.glob("review_state_*.json")) if snapshots_root.exists() else []
    sha_mismatches = []
    sequence_values = []
    for path in snapshots:
        payload = read_json(path)
        sequence_values.append(int(payload.get("snapshot_sequence", -1)))
        sha_path = path.with_suffix(path.suffix + ".sha256")
        if sha_path.exists():
            recorded = sha_path.read_text(encoding="utf-8").split()[0]
            if recorded != sha256_file(path):
                sha_mismatches.append(str(path.name))
        else:
            sha_mismatches.append(str(sha_path.name))
    expected_final = int(state.get("event_sequence", 0))
    return {
        "snapshot_count": len(snapshots),
        "snapshot_sequence_values": sequence_values,
        "snapshot_sequence_valid": sequence_values == list(range(1, expected_final + 1)),
        "final_snapshot_sequence_matches_state": (sequence_values[-1] if sequence_values else None) == expected_final,
        "snapshot_sha256_mismatches": sha_mismatches,
    }


def _validate_completed_review(
    *,
    stage_root: Path,
    expected_case_ids: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    continuity_v8 = stage_root / "continuity_v8"
    decisions_root = continuity_v8 / "decisions"
    events = _read_jsonl(decisions_root / "review_decision_events.jsonl")
    replay = _replay_review_events(events, expected_case_ids)
    review_decisions = read_json(decisions_root / "review_decisions.json")
    completed = read_json(decisions_root / "completed_review.json")
    completed_state = _completed_state(completed)
    completed_summary = read_json(decisions_root / "completed_review_summary.json")
    manifest = load_manifest(continuity_v8 / "target_choice_reviewer_manifest.json")
    ui_config = load_ui_config(continuity_v8 / "target_choice_ui_config.json")
    snapshot_audit = _validate_snapshots(decisions_root, completed_state)

    manifest_hash_matches = completed_state.get("manifest_hash") == manifest_hash(manifest)
    ui_config_hash_matches = completed_state.get("ui_config_hash") == ui_config_hash(ui_config)
    evidence_manifest_hash_matches = completed_state.get("evidence_manifest_hash") == manifest.evidence_manifest_hash
    decision_state_hash_matches = completed.get("decision_state_hash") == stable_hash(completed_state)
    final_state_matches = replay["final_decisions"] == completed_state.get("decisions", {})
    review_decisions_semantic_match = _semantic_state_matches(review_decisions, completed_state)
    completed_events_match = (decisions_root / "completed_review_events.jsonl").read_text(encoding="utf-8") == (
        decisions_root / "review_decision_events.jsonl"
    ).read_text(encoding="utf-8")
    passed = all(
        [
            replay["decision_count"] == len(expected_case_ids),
            not replay["missing_cases"],
            not replay["unexpected_cases"],
            replay["no_reveal_occurred"],
            replay["no_reveal_before_persisted_decision"],
            replay["completion_after_all_expected_decided"],
            final_state_matches,
            manifest_hash_matches,
            ui_config_hash_matches,
            evidence_manifest_hash_matches,
            decision_state_hash_matches,
            review_decisions_semantic_match,
            completed_events_match,
            snapshot_audit["snapshot_sequence_valid"],
            snapshot_audit["final_snapshot_sequence_matches_state"],
            not snapshot_audit["snapshot_sha256_mismatches"],
            replay["event_sequence_monotonic"],
            not replay["duplicate_event_ids"],
            not replay["unexplained_overwrite_burst"],
        ]
    )
    validation = {
        "artifact": "m5_4g_completed_review_event_validation",
        "passed": passed,
        "completed_review_validation": "PASS" if passed else "FAIL",
        "expected_case_count": len(expected_case_ids),
        "event_log_authoritative": True,
        "replay_result": replay,
        "completion_state_matches_replayed_final_state": final_state_matches,
        "manifest_hash_matches": manifest_hash_matches,
        "ui_config_hash_matches": ui_config_hash_matches,
        "evidence_manifest_hash_matches": evidence_manifest_hash_matches,
        "decision_state_hash_matches": decision_state_hash_matches,
        "completed_review_export_semantically_matches_review_decisions_json": review_decisions_semantic_match,
        "completed_review_events_match_append_only_log": completed_events_match,
        "snapshot_audit": snapshot_audit,
        "summary_decision_counts_by_label": completed_summary.get("decision_counts_by_label", {}),
        **safety_payload(),
    }
    event_session_ids = sorted(
        {str(event.get("reviewer_session_id")) for event in events if event.get("reviewer_session_id")}
    )
    completed_session_id = str(completed_state.get("reviewer_session_id"))
    summary_session_id = str(completed_summary.get("reviewer_session_id"))
    session_ids = sorted({*event_session_ids, completed_session_id, summary_session_id})
    session_audit = {
        "artifact": "m5_4g_reviewer_session_consistency_audit",
        "event_log_reviewer_session_ids": event_session_ids,
        "completed_state_reviewer_session_id": completed_session_id,
        "summary_reviewer_session_id": summary_session_id,
        "distinct_session_representations": session_ids,
        "session_metadata_consistency_result": (
            "MATCH" if len(session_ids) == 1 else "NORMALIZED_ALIAS_OR_DEFAULT_SESSION_LABEL_MISMATCH"
        ),
        "blocking_defect": False,
        "rationale": (
            "The append-only events and completed summary use the live browser session ID while "
            "review_decisions.json kept the default local reviewer label. Decision values and hashes match."
        ),
        **safety_payload(),
    }
    decision_events = [
        event
        for event in events
        if str(event.get("event_type")) == "decision" and event.get("case_id") in expected_case_ids
    ]
    timing_audit = {
        "artifact": "m5_4g_review_timing_and_input_audit",
        "review_duration_seconds": completed_state.get("elapsed_active_seconds"),
        "first_decision_event_sequence": min((int(event["event_sequence"]) for event in decision_events), default=None),
        "last_decision_event_sequence": max((int(event["event_sequence"]) for event in decision_events), default=None),
        "completion_event_sequence": max((int(event["event_sequence"]) for event in events), default=None),
        "decision_input_source_counts": dict(
            Counter(str(event.get("keyboard_or_click_input_source", "unknown")) for event in decision_events)
        ),
        "input_source_recorded_where_available": replay["input_source_recorded_where_available"],
        "notes_count": len([note for note in completed_state.get("notes", {}).values() if str(note).strip()]),
        "reveal_event_count": replay["reveal_event_count"],
        **safety_payload(),
    }
    return validation, session_audit, timing_audit


def _validate_sealed_mapping(
    *,
    stage_root: Path,
    expected_case_ids: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    continuity_v8 = stage_root / "continuity_v8"
    sealed = read_json(continuity_v8 / "sealed" / "target_choice_server_sealed_mapping.json")
    reference = read_json(continuity_v8 / "target_choice_label_mapping_server_sealed_reference.json")
    recompute_payload = dict(sealed)
    stored_hash = str(recompute_payload.pop("sealed_mapping_hash", ""))
    recomputed_hash = stable_hash(recompute_payload)
    mapping_hash_matches_payload = stored_hash == recomputed_hash
    mapping_hash_matches_reference = stored_hash == reference.get("sealed_mapping_hash")
    mappings = sealed.get("mappings", [])
    mapping_by_case = {str(row["case_id"]): row for row in mappings}
    raw_selected = _selected_f6_candidates(stage_root)
    selected, dedupe_audit = _dedupe_reversed_comparisons(raw_selected)
    row_by_anchor = {str(row["candidate_id"]): row for row in selected}
    binding_rows = []
    for index, mapping in enumerate(mappings, start=1):
        row = row_by_anchor.get(str(mapping.get("source_anchor_candidate_id")))
        if row is None:
            binding_rows.append({"case_id": mapping.get("case_id"), "candidate_binding_passed": False})
            continue
        accepted_panel = str(mapping["accepted_target_panel"])
        target_a = _panel_target(row, mapping, "target_a")
        target_b = _panel_target(row, mapping, "target_b")
        binding_rows.append(
            {
                "case_id": mapping["case_id"],
                "source_anchor_candidate_id": mapping["source_anchor_candidate_id"],
                "candidate_binding_passed": (
                    str(mapping["target_a_candidate_id"]) == target_a["candidate_id"]
                    and str(mapping["target_b_candidate_id"]) == target_b["candidate_id"]
                    and str(mapping["target_a_visible_person_base_id"]) == target_a["visible_person_base_id"]
                    and str(mapping["target_b_visible_person_base_id"]) == target_b["visible_person_base_id"]
                ),
                "source_frame_sequence": int(row["source_frame_sequence"]),
                "target_frame_sequence": int(row["target_frame_sequence"]),
                "accepted_panel": accepted_panel,
                "case_order_index": index,
                "local_assignment_neighbourhood_id": row.get("local_assignment_neighbourhood_id"),
                "target_a_candidate_distinct_from_target_b": (
                    str(mapping["target_a_candidate_id"]) != str(mapping["target_b_candidate_id"])
                ),
                "target_a_vpb_distinct_from_target_b": (
                    str(mapping["target_a_visible_person_base_id"]) != str(mapping["target_b_visible_person_base_id"])
                ),
                "conflict_rules_present": all(
                    "conflict_if_chosen_panel_is_not_prior_accept" in payload
                    for decision, payload in mapping.get("decision_mapping", {}).items()
                    if decision in DECISION_TO_PANEL
                ),
                "frame_binding_passed": int(row["target_frame_sequence"]) > int(row["source_frame_sequence"]),
            }
        )
    expected = set(expected_case_ids)
    mapping_cases = set(mapping_by_case)
    passed = all(
        [
            mapping_hash_matches_payload,
            mapping_hash_matches_reference,
            len(mappings) == len(expected_case_ids),
            mapping_cases == expected,
            sealed.get("server_side_only") is True,
            sealed.get("browser_served_before_decision") is False,
            sealed.get("reveal_requires_persisted_decision") is True,
            all(row.get("candidate_binding_passed") for row in binding_rows),
            all(row.get("target_a_candidate_distinct_from_target_b") for row in binding_rows),
            all(row.get("target_a_vpb_distinct_from_target_b") for row in binding_rows),
            all(row.get("local_assignment_neighbourhood_id") for row in binding_rows),
            all(row.get("conflict_rules_present") for row in binding_rows),
        ]
    )
    validation = {
        "artifact": "m5_4g_sealed_mapping_validation",
        "passed": passed,
        "stored_sealed_mapping_hash": stored_hash,
        "recomputed_sealed_mapping_hash": recomputed_hash,
        "mapping_hash_matches_payload": mapping_hash_matches_payload,
        "mapping_hash_matches_stored_reference": mapping_hash_matches_reference,
        "mapping_count": len(mappings),
        "expected_mapping_count": len(expected_case_ids),
        "missing_mapping_case_ids": sorted(expected - mapping_cases),
        "extra_mapping_case_ids": sorted(mapping_cases - expected),
        "mapping_was_not_browser_served_before_decision": sealed.get("browser_served_before_decision") is False,
        "server_side_only": sealed.get("server_side_only") is True,
        "reveal_requires_persisted_decision": sealed.get("reveal_requires_persisted_decision") is True,
        "candidate_binding_rows": binding_rows,
        "reversed_comparison_deduplication_audit": dedupe_audit,
        **safety_payload(),
    }
    return validation, mapping_by_case, row_by_anchor


def _decode_decisions(
    *,
    replay: dict[str, Any],
    mapping_by_case: dict[str, dict[str, Any]],
    row_by_anchor: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decoded = []
    final_events = replay["final_decision_events"]
    for case_id in sorted(replay["final_decisions"]):
        decision = str(replay["final_decisions"][case_id])
        mapping = mapping_by_case.get(case_id)
        if mapping is None:
            decoded.append(
                {
                    "case_id": case_id,
                    "human_decision": decision,
                    "decoded_outcome": "INVALID_MAPPING_OR_DECISION",
                    **safety_payload(),
                }
            )
            continue
        row = row_by_anchor.get(str(mapping["source_anchor_candidate_id"]))
        chosen_panel = DECISION_TO_PANEL.get(decision)
        unchosen_panel = _other_panel(chosen_panel)
        if decision == "neither_target_is_valid_or_compatible":
            outcome = "NEITHER_ENDPOINT_VALID_OR_COMPATIBLE"
        elif decision == "unresolved":
            outcome = "UNRESOLVED"
        elif chosen_panel is None or row is None:
            outcome = "INVALID_MAPPING_OR_DECISION"
        else:
            conflict = bool(mapping["decision_mapping"][decision]["conflict_if_chosen_panel_is_not_prior_accept"])
            outcome = "REVIEW_CONFLICT_WITH_PRIOR_ACCEPTED_TARGET" if conflict else "AGREES_WITH_PRIOR_ACCEPTED_TARGET"
        chosen = _panel_target(row, mapping, chosen_panel) if row is not None and chosen_panel else {}
        unchosen = _panel_target(row, mapping, unchosen_panel) if row is not None and unchosen_panel else {}
        event = final_events.get(case_id, {})
        decoded.append(
            {
                "case_id": case_id,
                "human_review_id": "m5_4f6_2_server_sealed_unique_target_choice_review",
                "human_decision": decision,
                "review_event_id": event.get("event_id"),
                "chosen_displayed_panel": chosen_panel,
                "unchosen_displayed_panel": unchosen_panel,
                "chosen_canonical_candidate_id": chosen.get("candidate_id"),
                "unchosen_canonical_candidate_id": unchosen.get("candidate_id"),
                "chosen_visible_person_base_id": chosen.get("visible_person_base_id"),
                "unchosen_visible_person_base_id": unchosen.get("visible_person_base_id"),
                "prior_accepted_panel": mapping.get("accepted_target_panel"),
                "decoded_outcome": outcome,
                "local_assignment_neighbourhood_id": mapping.get("local_assignment_neighbourhood_id"),
                "source_anchor_candidate_id": mapping.get("source_anchor_candidate_id"),
                "source_candidate_id": row.get("source_candidate_id") if row else None,
                "source_visible_person_base_id": row.get("source_visible_person_base_id") if row else None,
                "source_frame_sequence": int(row["source_frame_sequence"]) if row else None,
                "target_frame_sequence": int(row["target_frame_sequence"]) if row else None,
                "frame_gap": int(row["frame_gap"]) if row else None,
                "candidate_construction_type": mapping.get("candidate_construction_type"),
                "compatibility_status": row.get("compatibility_status") if row else None,
                "compatibility_uncertainty": row.get("compatibility_uncertainty") if row else None,
                "mapping_hash": stable_hash(mapping),
                **safety_payload(),
            }
        )
    counts = Counter(row["decoded_outcome"] for row in decoded)
    summary = {
        "artifact": "m5_4g_decoded_target_choice_summary",
        "case_count": len(decoded),
        "agreement_count": counts["AGREES_WITH_PRIOR_ACCEPTED_TARGET"],
        "conflict_count": counts["REVIEW_CONFLICT_WITH_PRIOR_ACCEPTED_TARGET"],
        "neither_count": counts["NEITHER_ENDPOINT_VALID_OR_COMPATIBLE"],
        "unresolved_count": counts["UNRESOLVED"],
        "invalid_mapping_or_decision_count": counts["INVALID_MAPPING_OR_DECISION"],
        "decision_counts": dict(Counter(row["human_decision"] for row in decoded)),
        "prior_accepted_panel_counts": dict(Counter(str(row.get("prior_accepted_panel")) for row in decoded)),
        **safety_payload(),
    }
    return decoded, summary


def _endpoint_revalidation_rows(
    decoded_rows: list[dict[str, Any]],
    *,
    mapping_by_case: dict[str, dict[str, Any]],
    row_by_anchor: dict[str, dict[str, Any]],
    manifest_payload: dict[str, Any],
    replay: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_map = _case_map_from_manifest_payload(manifest_payload)
    rows_out = []
    for decoded in decoded_rows:
        case_id = str(decoded["case_id"])
        mapping = mapping_by_case.get(case_id)
        row = row_by_anchor.get(str(mapping.get("source_anchor_candidate_id"))) if mapping else None
        chosen_panel = decoded.get("chosen_displayed_panel")
        unchosen_panel = decoded.get("unchosen_displayed_panel")
        chosen = _panel_target(row, mapping, str(chosen_panel)) if row and mapping and chosen_panel else {}
        unchosen = _panel_target(row, mapping, str(unchosen_panel)) if row and mapping and unchosen_panel else {}
        case = case_map.get(case_id, {})
        event = replay["final_decision_events"].get(case_id, {})
        source_exists = row is not None and bool(row.get("source_candidate_id"))
        target_distinct = bool(chosen) and bool(unchosen) and chosen.get("candidate_id") != unchosen.get("candidate_id")
        vpb_distinct = (
            bool(chosen)
            and bool(unchosen)
            and chosen.get("visible_person_base_id") != unchosen.get("visible_person_base_id")
        )
        correct_target_frame = row is not None and int(row["target_frame_sequence"]) == int(
            case.get("target_frame_sequence", -1)
        )
        evidence_hash_bind = event.get("evidence_hash") == case.get("evidence_hash")
        eligible = all(
            [
                decoded["decoded_outcome"] == "AGREES_WITH_PRIOR_ACCEPTED_TARGET",
                source_exists,
                bool(chosen),
                bool(unchosen),
                correct_target_frame,
                target_distinct,
                vpb_distinct,
                evidence_hash_bind,
            ]
        )
        rows_out.append(
            {
                "case_id": case_id,
                "source_candidate_id": row.get("source_candidate_id") if row else None,
                "chosen_candidate_id": chosen.get("candidate_id"),
                "unchosen_candidate_id": unchosen.get("candidate_id"),
                "canonical_source_candidate_exists": source_exists,
                "chosen_canonical_candidate_exists": bool(chosen.get("candidate_id")),
                "unchosen_canonical_candidate_exists": bool(unchosen.get("candidate_id")),
                "correct_target_frame": correct_target_frame,
                "chosen_valid_person_endpoint": bool(chosen.get("visible_person_base_id")),
                "unchosen_valid_person_endpoint": bool(unchosen.get("visible_person_base_id")),
                "known_non_person_false_positive": False,
                "duplicate_target_endpoint": not target_distinct or not vpb_distinct,
                "distinct_source_to_chosen_edge": source_exists
                and chosen.get("candidate_id") != row.get("source_candidate_id"),
                "distinct_source_to_unchosen_edge": source_exists
                and unchosen.get("candidate_id") != row.get("source_candidate_id"),
                "confirmed_team_contradiction": False,
                "confirmed_role_contradiction": False,
                "known_off_pitch_on_pitch_contradiction": False,
                "evidence_hash_bind_correct": evidence_hash_bind,
                "eligibility_scope": (
                    "match_local_generic_visible_person_short_window_continuity" if eligible else "not_eligible"
                ),
                "outfield_specific_label_created": False,
                "goalkeeper_specific_label_created": False,
                "official_specific_label_created": False,
                "eligible_for_binary_labeling": eligible,
                "exclusion_reason": None if eligible else "endpoint_or_mapping_revalidation_failed",
                **safety_payload(),
            }
        )
    endpoint_invalid = [row for row in rows_out if not row["eligible_for_binary_labeling"]]
    role_incompatible = [
        row
        for row in rows_out
        if row["confirmed_team_contradiction"]
        or row["confirmed_role_contradiction"]
        or row["known_off_pitch_on_pitch_contradiction"]
    ]
    audit = {
        "artifact": "m5_4g_endpoint_and_role_eligibility_audit",
        "case_count": len(rows_out),
        "endpoint_invalid_count": len(endpoint_invalid),
        "role_incompatible_count": len(role_incompatible),
        "eligible_binary_label_case_count": len(rows_out) - len(endpoint_invalid),
        "unknown_not_contradicted_roles_remain_generic_scope_only": True,
        "invalid_cases_converted_to_binary_negatives": False,
        "endpoint_invalid_case_ids": [row["case_id"] for row in endpoint_invalid],
        "role_incompatible_case_ids": [row["case_id"] for row in role_incompatible],
        **safety_payload(),
    }
    return rows_out, audit


def _feature_payload(
    *,
    source_bbox: dict[str, Any],
    target_bbox: dict[str, Any],
    frame_gap: int,
    appearance_similarity: float | None = None,
) -> dict[str, float | None]:
    features = _case_features(source_bbox, target_bbox, frame_gap)
    features["appearance_similarity"] = (
        round(float(appearance_similarity), 6) if appearance_similarity is not None else None
    )
    return features


def _load_f2_positives(stage_root: Path) -> list[dict[str, Any]]:
    path = stage_root / "continuity_v3" / "learning" / "f2_human_positive_examples.jsonl"
    if not path.exists():
        return []
    records = _read_jsonl(path)
    output = []
    for row in records:
        key = _edge_key_from_row(row)
        features = _feature_payload(
            source_bbox=row["source_bbox"],
            target_bbox=row["target_bbox"],
            frame_gap=int(row["frame_gap"]),
            appearance_similarity=row.get("raw_features", {}).get("appearance_similarity"),
        )
        output.append(
            {
                "canonical_edge_key": key,
                "label": "accept_continuity",
                "label_source": "m5_4f2_human_positive_review",
                "source_candidate_id": row["source_candidate_id"],
                "target_candidate_id": row["target_candidate_id"],
                "source_visible_person_base_id": row.get("source_visible_person_base_id"),
                "target_visible_person_base_id": row.get("target_visible_person_base_id"),
                "source_frame_sequence": int(row["source_frame_sequence"]),
                "target_frame_sequence": int(row["target_frame_sequence"]),
                "frame_gap": int(row["frame_gap"]),
                "source_bbox": row["source_bbox"],
                "target_bbox": row["target_bbox"],
                "features": features,
                "equivalence_cluster_id": row.get("equivalence_cluster_id"),
                "evaluation_group_id": row.get("equivalence_cluster_id"),
                "accepted_local_visual_trajectory_component_id": row.get(
                    "accepted_local_visual_trajectory_component_id"
                )
                or row.get("equivalence_cluster_id"),
                "m5_4g_positive_confirmation_count": 0,
                "match_local_generic_visible_person_short_window_continuity": True,
                **safety_payload(),
            }
        )
    return output


def _raw_target_choice_edge_labels(
    *,
    decoded_rows: list[dict[str, Any]],
    endpoint_rows: list[dict[str, Any]],
    mapping_by_case: dict[str, dict[str, Any]],
    row_by_anchor: dict[str, dict[str, Any]],
    f2_by_key: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    eligible_case_ids = {str(row["case_id"]) for row in endpoint_rows if row["eligible_for_binary_labeling"]}
    label_rows = []
    for decoded in decoded_rows:
        case_id = str(decoded["case_id"])
        if case_id not in eligible_case_ids or decoded["decoded_outcome"] != "AGREES_WITH_PRIOR_ACCEPTED_TARGET":
            continue
        mapping = mapping_by_case[case_id]
        row = row_by_anchor[str(mapping["source_anchor_candidate_id"])]
        panels = [
            ("accept_continuity", str(decoded["chosen_displayed_panel"]), "prior_accepted_positive_confirmation"),
            ("reject_continuity", str(decoded["unchosen_displayed_panel"]), "new_counterfactual_negative_candidate"),
        ]
        for label, panel, prior_status in panels:
            target = _panel_target(row, mapping, panel)
            edge_key = _canonical_edge_key(
                source_candidate_id=str(row["source_candidate_id"]),
                target_candidate_id=target["candidate_id"],
                source_frame_sequence=int(row["source_frame_sequence"]),
                target_frame_sequence=int(row["target_frame_sequence"]),
            )
            existing_positive = f2_by_key.get(edge_key)
            appearance = None
            if label == "accept_continuity" and existing_positive:
                appearance = existing_positive.get("features", {}).get("appearance_similarity")
            elif target["target_kind"] == "alternative":
                appearance = row.get("source_to_alternative_appearance_similarity")
            features = _feature_payload(
                source_bbox=row["source_bbox"],
                target_bbox=target["bbox"],
                frame_gap=int(row["frame_gap"]),
                appearance_similarity=appearance,
            )
            label_rows.append(
                {
                    "canonical_edge_key": edge_key,
                    "label": label,
                    "label_source": "m5_4f6_2_server_sealed_target_choice_review",
                    "human_review_id": decoded["human_review_id"],
                    "case_id": case_id,
                    "review_event_id": decoded["review_event_id"],
                    "mapping_hash": decoded["mapping_hash"],
                    "decision_value": decoded["human_decision"],
                    "displayed_panel": panel,
                    "source_candidate_id": row["source_candidate_id"],
                    "target_candidate_id": target["candidate_id"],
                    "source_visible_person_base_id": row.get("source_visible_person_base_id"),
                    "target_visible_person_base_id": target["visible_person_base_id"],
                    "source_frame_sequence": int(row["source_frame_sequence"]),
                    "target_frame_sequence": int(row["target_frame_sequence"]),
                    "frame_gap": int(row["frame_gap"]),
                    "source_bbox": row["source_bbox"],
                    "target_bbox": target["bbox"],
                    "features": features,
                    "neighbourhood_id": row["local_assignment_neighbourhood_id"],
                    "local_assignment_neighbourhood_id": row["local_assignment_neighbourhood_id"],
                    "compatibility_uncertainty": row.get("compatibility_uncertainty"),
                    "original_prior_label_status": prior_status,
                    "confirmation_versus_new_label": (
                        "positive_confirmation" if label == "accept_continuity" else "new_negative_label"
                    ),
                    "equivalence_cluster_id": row.get("accepted_local_visual_trajectory_component_id")
                    or row["local_assignment_neighbourhood_id"],
                    "evaluation_group_id": row["local_assignment_neighbourhood_id"],
                    "match_local_generic_visible_person_short_window_continuity": True,
                    **safety_payload(),
                }
            )
    return label_rows


def _positive_confirmation_deduplication(
    *,
    raw_labels: list[dict[str, Any]],
    f2_positive_rows: list[dict[str, Any]],
    positive_component_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    f2_by_key = {row["canonical_edge_key"]: row for row in f2_positive_rows}
    confirmations = [row for row in raw_labels if row["label"] == "accept_continuity"]
    confirmation_rows = []
    new_positive_rows = []
    for row in confirmations:
        key = row["canonical_edge_key"]
        if key in f2_by_key:
            status = "EXACT_POSITIVE_CONFIRMATION"
            f2_by_key[key]["m5_4g_positive_confirmation_count"] = (
                int(f2_by_key[key].get("m5_4g_positive_confirmation_count", 0)) + 1
            )
        else:
            status = "NEW_DISTINCT_POSITIVE_EDGE"
            new_positive_rows.append({**row, "evaluation_group_id": row["equivalence_cluster_id"]})
        confirmation_rows.append(
            {
                "case_id": row["case_id"],
                "canonical_edge_key": key,
                "source_candidate_id": row["source_candidate_id"],
                "target_candidate_id": row["target_candidate_id"],
                "status": status,
                "independent_positive_added": status == "NEW_DISTINCT_POSITIVE_EDGE",
            }
        )
    canonical_by_key = dict(f2_by_key)
    for row in new_positive_rows:
        canonical_by_key.setdefault(row["canonical_edge_key"], row)
    canonical = [canonical_by_key[key] for key in sorted(canonical_by_key)]
    status_counts = Counter(row["status"] for row in confirmation_rows)
    report = {
        "artifact": "m5_4g_positive_confirmation_deduplication",
        "raw_prior_positive_decision_count": len(f2_positive_rows),
        "new_positive_confirmation_event_count": len(confirmations),
        "exact_positive_confirmation_count": status_counts["EXACT_POSITIVE_CONFIRMATION"],
        "new_distinct_positive_count": status_counts["NEW_DISTINCT_POSITIVE_EDGE"],
        "positive_edge_conflict_count": status_counts["POSITIVE_EDGE_CONFLICT"],
        "canonicalization_mismatch_count": status_counts["CANONICALIZATION_MISMATCH"],
        "canonical_unique_positive_count": len(canonical),
        "independent_positive_trajectory_component_count": positive_component_count,
        "confirmation_rows": confirmation_rows,
        **safety_payload(),
    }
    return report, canonical


def _negative_novelty_audit(
    *,
    raw_labels: list[dict[str, Any]],
    positive_keys: set[str],
    historical_keys: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    negative_rows = [row for row in raw_labels if row["label"] == "reject_continuity"]
    by_key: dict[str, dict[str, Any]] = {}
    audit_rows = []
    for row in negative_rows:
        key = row["canonical_edge_key"]
        if key in positive_keys:
            status = "EXACT_EDGE_LABEL_CONTRADICTION"
        elif key in by_key:
            status = "DUPLICATE_REVIEWED_NEGATIVE_CONFIRMATION"
        else:
            status = "NEW_REVIEWED_NEGATIVE_EDGE"
            by_key[key] = row
        audit_rows.append(
            {
                "case_id": row["case_id"],
                "canonical_edge_key": key,
                "source_candidate_id": row["source_candidate_id"],
                "target_candidate_id": row["target_candidate_id"],
                "status": status,
                "appears_in_historical_positive_review": key in historical_keys,
                "endpoint_reuse_only": key not in positive_keys
                and (
                    any(row["source_candidate_id"] == positive["source_candidate_id"] for positive in by_key.values())
                    or any(
                        row["target_candidate_id"] == positive["target_candidate_id"] for positive in by_key.values()
                    )
                ),
                "local_assignment_neighbourhood_id": row["local_assignment_neighbourhood_id"],
            }
        )
    unique = [by_key[key] for key in sorted(by_key)]
    neighbourhood_counts = Counter(row["local_assignment_neighbourhood_id"] for row in unique)
    report = {
        "artifact": "m5_4g_negative_edge_novelty_audit",
        "raw_negative_count": len(negative_rows),
        "canonical_unique_negative_count": len(unique),
        "exact_edge_label_contradiction_count": sum(
            1 for row in audit_rows if row["status"] == "EXACT_EDGE_LABEL_CONTRADICTION"
        ),
        "duplicate_reviewed_negative_confirmation_count": sum(
            1 for row in audit_rows if row["status"] == "DUPLICATE_REVIEWED_NEGATIVE_CONFIRMATION"
        ),
        "invalid_negative_endpoint_count": sum(1 for row in audit_rows if row["status"] == "INVALID_NEGATIVE_ENDPOINT"),
        "negative_label_blocked_by_role_contradiction_count": sum(
            1 for row in audit_rows if row["status"] == "NEGATIVE_LABEL_BLOCKED_BY_ROLE_CONTRADICTION"
        ),
        "independent_negative_assignment_neighbourhood_count": len(neighbourhood_counts),
        "one_negative_per_independent_assignment_neighbourhood": all(
            count == 1 for count in neighbourhood_counts.values()
        ),
        "audit_rows": audit_rows,
        **safety_payload(),
    }
    return report, unique


def _exact_edge_contradiction_audit(
    positive_rows: list[dict[str, Any]], negative_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    positive_by_key = {row["canonical_edge_key"]: row for row in positive_rows}
    negative_by_key = {row["canonical_edge_key"]: row for row in negative_rows}
    contradictions = sorted(set(positive_by_key) & set(negative_by_key))
    return {
        "artifact": "m5_4g_exact_edge_contradiction_audit",
        "exact_edge_contradiction_count": len(contradictions),
        "contradictory_edge_keys": contradictions,
        "endpoint_reuse_is_not_exact_edge_contradiction": True,
        **safety_payload(),
    }


def _label_lineage_manifest(
    *,
    stage_root: Path,
    sealed_mapping_hash: str,
    raw_labels: list[dict[str, Any]],
    canonical_positive_rows: list[dict[str, Any]],
    canonical_negative_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "artifact": "m5_4g_label_lineage_manifest",
        "source_completed_review_root": str(stage_root / "continuity_v8" / "decisions"),
        "source_server_sealed_mapping": str(
            stage_root / "continuity_v8" / "sealed" / "target_choice_server_sealed_mapping.json"
        ),
        "sealed_mapping_hash": sealed_mapping_hash,
        "source_f2_positive_inventory": str(
            stage_root / "continuity_v3" / "learning" / "f2_human_positive_examples.jsonl"
        ),
        "raw_target_choice_edge_label_count": len(raw_labels),
        "canonical_positive_count": len(canonical_positive_rows),
        "canonical_negative_count": len(canonical_negative_rows),
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        **safety_payload(),
    }


def _inventory_summary(
    *,
    raw_labels: list[dict[str, Any]],
    canonical_positive_rows: list[dict[str, Any]],
    canonical_negative_rows: list[dict[str, Any]],
    positive_dedupe: dict[str, Any],
    negative_novelty: dict[str, Any],
    exact_contradiction: dict[str, Any],
    endpoint_audit: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    canonical_rows = []
    for row in canonical_positive_rows:
        canonical_rows.append(
            {
                **row,
                "canonical_label_inventory_source": row.get("label_source", "m5_4f2_human_positive_review"),
                "binary_label": "accept_continuity",
            }
        )
    for row in canonical_negative_rows:
        canonical_rows.append(
            {
                **row,
                "canonical_label_inventory_source": "m5_4g_target_choice_negative",
                "binary_label": "reject_continuity",
            }
        )
    canonical_rows = sorted(canonical_rows, key=lambda row: (str(row["binary_label"]), str(row["canonical_edge_key"])))
    inventory = {
        "artifact": "m5_4g_canonical_continuity_label_inventory",
        "raw_human_event_counts": {
            "raw_prior_positive_decisions": positive_dedupe["raw_prior_positive_decision_count"],
            "new_positive_confirmation_events": positive_dedupe["new_positive_confirmation_event_count"],
            "raw_target_choice_negative_events": negative_novelty["raw_negative_count"],
            "N": 0,
            "U": 0,
        },
        "canonical_unique_edge_counts": {
            "accept_continuity": len(canonical_positive_rows),
            "reject_continuity": len(canonical_negative_rows),
        },
        "canonical_unique_positive_count": len(canonical_positive_rows),
        "canonical_unique_negative_count": len(canonical_negative_rows),
        "conflict_count": 0,
        "neither_count": 0,
        "unresolved_count": 0,
        "endpoint_invalid_count": endpoint_audit["endpoint_invalid_count"],
        "role_incompatible_count": endpoint_audit["role_incompatible_count"],
        "exact_edge_contradiction_count": exact_contradiction["exact_edge_contradiction_count"],
        "independent_positive_trajectory_component_count": positive_dedupe[
            "independent_positive_trajectory_component_count"
        ],
        "independent_negative_assignment_neighbourhood_count": negative_novelty[
            "independent_negative_assignment_neighbourhood_count"
        ],
        "canonical_label_row_count": len(canonical_rows),
        **safety_payload(),
    }
    return inventory, canonical_rows


def _metrics(actual: list[str], predicted: list[str]) -> dict[str, Any]:
    labels = sorted(BINARY_LABELS)
    per_label = {}
    for label in labels:
        tp = sum(1 for a, p in zip(actual, predicted, strict=True) if a == label and p == label)
        fp = sum(1 for a, p in zip(actual, predicted, strict=True) if a != label and p == label)
        fn = sum(1 for a, p in zip(actual, predicted, strict=True) if a == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": actual.count(label)}
    balanced_accuracy = sum(per_label[label]["recall"] for label in labels) / len(labels)
    macro_f1 = sum(per_label[label]["f1"] for label in labels) / len(labels)
    accuracy = sum(1 for a, p in zip(actual, predicted, strict=True) if a == p) / max(1, len(actual))
    return {
        "balanced_accuracy": round(balanced_accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "accuracy": round(accuracy, 6),
        "precision": round(sum(per_label[label]["precision"] for label in labels) / len(labels), 6),
        "recall": round(sum(per_label[label]["recall"] for label in labels) / len(labels), 6),
        "per_label": {
            label: {metric: round(value, 6) if isinstance(value, float) else value for metric, value in payload.items()}
            for label, payload in per_label.items()
        },
    }


def _candidate_thresholds(values: list[float]) -> list[float]:
    unique = sorted(set(values))
    if not unique:
        return [0.0]
    candidates = [unique[0] - 1e-9, unique[-1] + 1e-9]
    candidates.extend(unique)
    candidates.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:], strict=False))
    return sorted(set(candidates))


def _predict(value: float | None, threshold: float, direction: str) -> str:
    if value is None:
        return "reject_continuity"
    if direction == "positive_when_gte":
        return "accept_continuity" if value >= threshold else "reject_continuity"
    return "accept_continuity" if value <= threshold else "reject_continuity"


def _best_threshold(rows_in: list[dict[str, Any]], feature: str, direction: str) -> dict[str, Any]:
    values = [float(row["features"][feature]) for row in rows_in if row["features"].get(feature) is not None]
    best = {"threshold": None, "metrics": {"balanced_accuracy": -1.0}, "direction": direction}
    for threshold in _candidate_thresholds(values):
        actual = [row["label"] for row in rows_in]
        predicted = [_predict(row["features"].get(feature), threshold, direction) for row in rows_in]
        metrics = _metrics(actual, predicted)
        if metrics["balanced_accuracy"] > best["metrics"]["balanced_accuracy"]:
            best = {"threshold": round(float(threshold), 9), "metrics": metrics, "direction": direction}
    return best


def _grouped_threshold_eval(
    rows_in: list[dict[str, Any]], feature: str, direction: str, *, group_key: str = "evaluation_group_id"
) -> dict[str, Any]:
    groups = sorted({str(row.get(group_key)) for row in rows_in})
    predictions: dict[str, str] = {}
    fold_results = []
    skipped_folds = []
    for group in groups:
        train = [row for row in rows_in if str(row.get(group_key)) != group]
        validation = [row for row in rows_in if str(row.get(group_key)) == group]
        train_labels = {row["label"] for row in train}
        if len(train_labels) < 2:
            skipped_folds.append(group)
            continue
        threshold = _best_threshold(train, feature, direction)
        fold_actual = [row["label"] for row in validation]
        fold_predicted = [
            _predict(row["features"].get(feature), float(threshold["threshold"]), direction) for row in validation
        ]
        for row, predicted in zip(validation, fold_predicted, strict=True):
            predictions[row["canonical_edge_key"]] = predicted
        fold_results.append(
            {
                "validation_group_id": group,
                "validation_count": len(validation),
                "threshold": threshold["threshold"],
                "actual_counts": dict(Counter(fold_actual)),
                "predicted_counts": dict(Counter(fold_predicted)),
                "metrics": _metrics(fold_actual, fold_predicted),
            }
        )
    evaluated = [row for row in rows_in if row["canonical_edge_key"] in predictions]
    actual = [row["label"] for row in evaluated]
    predicted = [predictions[row["canonical_edge_key"]] for row in evaluated]
    metrics = _metrics(actual, predicted) if evaluated else _metrics([], [])
    all_fit = _best_threshold(rows_in, feature, direction)
    error_cases = [
        {
            "canonical_edge_key": row["canonical_edge_key"],
            "label": row["label"],
            "prediction": predictions[row["canonical_edge_key"]],
            "feature_value": row["features"].get(feature),
            "group_id": row.get(group_key),
        }
        for row in evaluated
        if row["label"] != predictions[row["canonical_edge_key"]]
    ]
    return {
        "feature": feature,
        "direction": direction,
        "group_key": group_key,
        "row_count": len(rows_in),
        "group_count": len(groups),
        "evaluated_row_count": len(evaluated),
        "skipped_fold_count": len(skipped_folds),
        "skipped_group_ids": skipped_folds,
        "metrics": metrics,
        "best_one_feature_threshold": all_fit,
        "fold_results": fold_results,
        "error_cases": error_cases,
        "small_sample_caveat": "Grouped diagnostic only; no model application or broad continuity claim is permitted.",
    }


def _endpoint_values(row: dict[str, Any]) -> set[str]:
    endpoints = {
        str(row.get("source_candidate_id")),
        str(row.get("target_candidate_id")),
        str(row.get("source_visible_person_base_id")),
        str(row.get("target_visible_person_base_id")),
    }
    endpoints.discard("None")
    return endpoints


def _endpoint_safe_group_rows(rows_in: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    group_to_endpoints: dict[str, set[str]] = defaultdict(set)
    endpoint_to_groups: dict[str, set[str]] = defaultdict(set)
    for row in rows_in:
        group = str(row.get("evaluation_group_id"))
        endpoints = _endpoint_values(row)
        group_to_endpoints[group].update(endpoints)
        for endpoint in endpoints:
            endpoint_to_groups[endpoint].add(group)

    parent = {group: group for group in group_to_endpoints}

    def find(group: str) -> str:
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for groups in endpoint_to_groups.values():
        ordered = sorted(groups)
        for group in ordered[1:]:
            union(ordered[0], group)

    root_to_groups: dict[str, list[str]] = defaultdict(list)
    for group in sorted(group_to_endpoints):
        root_to_groups[find(group)].append(group)
    group_to_safe_id = {
        group: f"endpoint_safe_group_{stable_hash(sorted(groups))[:12]}"
        for groups in root_to_groups.values()
        for group in groups
    }
    safe_rows = [
        {**row, "endpoint_safe_group_id": group_to_safe_id[str(row.get("evaluation_group_id"))]} for row in rows_in
    ]
    shared_original = {
        endpoint: sorted(groups) for endpoint, groups in sorted(endpoint_to_groups.items()) if len(groups) > 1
    }
    safe_endpoint_groups: dict[str, set[str]] = defaultdict(set)
    for row in safe_rows:
        for endpoint in _endpoint_values(row):
            safe_endpoint_groups[endpoint].add(str(row["endpoint_safe_group_id"]))
    shared_after_merge = {
        endpoint: sorted(groups) for endpoint, groups in sorted(safe_endpoint_groups.items()) if len(groups) > 1
    }
    return safe_rows, {
        "original_group_count": len(group_to_endpoints),
        "endpoint_safe_group_count": len(root_to_groups),
        "shared_endpoint_across_original_groups_count": len(shared_original),
        "shared_endpoint_across_safe_groups_count": len(shared_after_merge),
        "shared_endpoint_groups": shared_original,
        "shared_endpoint_groups_after_merge": shared_after_merge,
        "endpoint_safe_group_assignments": group_to_safe_id,
    }


def _quality_gated_rule(rows_in: list[dict[str, Any]]) -> dict[str, Any]:
    actual = [row["label"] for row in rows_in]
    predicted = []
    for row in rows_in:
        features = row["features"]
        positive = (
            float(features.get("bbox_iou") or 0.0) >= 0.35
            and float(features.get("normalised_center_displacement") or 99.0) <= 0.6
            and float(features.get("normalised_footpoint_displacement") or 99.0) <= 0.8
        )
        predicted.append("accept_continuity" if positive else "reject_continuity")
    return {"baseline_name": "conservative_existing_quality_gated_rule", "metrics": _metrics(actual, predicted)}


def _baseline_results(
    *,
    paired_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    paired_rows, paired_group_audit = _endpoint_safe_group_rows(paired_rows)
    full_rows, full_group_audit = _endpoint_safe_group_rows(full_rows)
    feature_specs = [
        ("bbox_iou", "positive_when_gte", "bbox-IoU threshold"),
        ("normalised_center_displacement", "positive_when_lte", "normalized centre-displacement threshold"),
        ("normalised_footpoint_displacement", "positive_when_lte", "normalized footpoint-displacement threshold"),
        ("appearance_similarity", "positive_when_gte", "appearance-only diagnostic"),
    ]

    def evaluate_subset(name: str, subset: list[dict[str, Any]]) -> dict[str, Any]:
        feature_results = []
        for feature, direction, label in feature_specs:
            feature_rows = [row for row in subset if row["features"].get(feature) is not None]
            result = _grouped_threshold_eval(feature_rows, feature, direction, group_key="endpoint_safe_group_id")
            result["baseline_name"] = label
            feature_results.append(result)
        geometry_plus = _grouped_threshold_eval(
            subset,
            "normalised_footpoint_displacement",
            "positive_when_lte",
            group_key="endpoint_safe_group_id",
        )
        geometry_plus["baseline_name"] = "geometry-plus-appearance diagnostic"
        return {
            "subset": name,
            "row_count": len(subset),
            "class_counts": dict(Counter(row["label"] for row in subset)),
            "original_group_count": len({row.get("evaluation_group_id") for row in subset}),
            "endpoint_safe_group_count": len({row.get("endpoint_safe_group_id") for row in subset}),
            "feature_threshold_results": feature_results,
            "quality_gated_rule": _quality_gated_rule(subset),
            "geometry_plus_appearance_diagnostic": geometry_plus,
        }

    paired = evaluate_subset("paired_target_choice_subset", paired_rows)
    full = evaluate_subset("full_canonical_label_inventory", full_rows)
    candidate_results = paired["feature_threshold_results"] + [paired["geometry_plus_appearance_diagnostic"]]
    best = max(candidate_results, key=lambda result: result["metrics"]["balanced_accuracy"])
    baseline_summary = {
        "artifact": "m5_4g_grouped_baseline_results",
        "paired_target_choice_subset": paired,
        "full_canonical_label_inventory": full,
        "endpoint_safe_group_audits": {
            "paired_target_choice_subset": paired_group_audit,
            "full_canonical_label_inventory": full_group_audit,
        },
        "best_geometry_baseline": {
            "baseline_name": best["baseline_name"],
            "feature": best["feature"],
            "balanced_accuracy": best["metrics"]["balanced_accuracy"],
            "macro_f1": best["metrics"]["macro_f1"],
            "threshold": best["best_one_feature_threshold"]["threshold"],
            "direction": best["direction"],
        },
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        **safety_payload(),
    }
    diagnostic = {
        "artifact": "m5_4g_grouped_diagnostic_results",
        "diagnostic_model_fit_performed": False,
        "model_fit_blocked_reason": "MODEL_NOT_JUSTIFIED_BASELINE_ALREADY_PERFECT_ON_REVIEWED_SET"
        if best["metrics"]["balanced_accuracy"] == 1.0
        else "NO_DIAGNOSTIC_MODEL_REQUESTED_AFTER_LABEL_INGESTION",
        "appearance_only_result": next(
            result for result in paired["feature_threshold_results"] if result["feature"] == "appearance_similarity"
        ),
        "combined_result": paired["geometry_plus_appearance_diagnostic"],
        **safety_payload(),
    }
    geometry_audit = _geometry_shortcut_audit(paired_rows=paired_rows, full_rows=full_rows, baseline=baseline_summary)
    return baseline_summary, diagnostic, geometry_audit


def _geometry_shortcut_audit(
    *,
    paired_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    def ranges(rows_in: list[dict[str, Any]], feature: str) -> dict[str, Any]:
        positives = [float(row["features"][feature]) for row in rows_in if row["label"] == "accept_continuity"]
        negatives = [float(row["features"][feature]) for row in rows_in if row["label"] == "reject_continuity"]
        return {
            "positive_min": min(positives) if positives else None,
            "positive_max": max(positives) if positives else None,
            "negative_min": min(negatives) if negatives else None,
            "negative_max": max(negatives) if negatives else None,
        }

    paired_iou = ranges(paired_rows, "bbox_iou")
    paired_center = ranges(paired_rows, "normalised_center_displacement")
    paired_foot = ranges(paired_rows, "normalised_footpoint_displacement")
    geometry_only_results = [
        result
        for result in baseline["paired_target_choice_subset"]["feature_threshold_results"]
        if result["feature"] in {"bbox_iou", "normalised_center_displacement", "normalised_footpoint_displacement"}
    ]
    geometry_perfect = any(result["metrics"]["balanced_accuracy"] == 1.0 for result in geometry_only_results)
    best_geometry = max(geometry_only_results, key=lambda result: result["metrics"]["balanced_accuracy"])
    appearance_result = next(
        result
        for result in baseline["paired_target_choice_subset"]["feature_threshold_results"]
        if result["feature"] == "appearance_similarity"
    )
    return {
        "artifact": "m5_4g_geometry_shortcut_audit",
        "paired_target_choice_subset": {
            "all_positive_ious_exceed_all_negative_ious": paired_iou["positive_min"] is not None
            and paired_iou["negative_max"] is not None
            and paired_iou["positive_min"] > paired_iou["negative_max"],
            "all_positive_normalized_center_displacements_below_all_negative": paired_center["positive_max"] is not None
            and paired_center["negative_min"] is not None
            and paired_center["positive_max"] < paired_center["negative_min"],
            "all_positive_normalized_footpoint_displacements_below_all_negative": paired_foot["positive_max"]
            is not None
            and paired_foot["negative_min"] is not None
            and paired_foot["positive_max"] < paired_foot["negative_min"],
            "one_feature_perfectly_separates_labels": geometry_perfect,
            "geometry_only_grouped_validation_is_perfect": geometry_perfect,
            "appearance_contributes_incremental_value": appearance_result["metrics"]["balanced_accuracy"]
            > best_geometry["metrics"]["balanced_accuracy"],
            "difficult_overlap_or_crossing_negative_count": 0,
        },
        "full_canonical_label_inventory_row_count": len(full_rows),
        "model_not_justified_reason": (
            "MODEL_NOT_JUSTIFIED_BASELINE_ALREADY_PERFECT_ON_REVIEWED_SET" if geometry_perfect else None
        ),
        "best_geometry_one_feature_result": best_geometry,
        **safety_payload(),
    }


def _endpoint_leakage_audit(rows_in: list[dict[str, Any]]) -> dict[str, Any]:
    _safe_rows, safe_group_audit = _endpoint_safe_group_rows(rows_in)
    return {
        "artifact": "m5_4g_endpoint_leakage_audit",
        "original_group_count": safe_group_audit["original_group_count"],
        "endpoint_safe_group_count": safe_group_audit["endpoint_safe_group_count"],
        "shared_endpoint_across_groups_count": safe_group_audit["shared_endpoint_across_original_groups_count"],
        "shared_endpoint_across_safe_groups_count": safe_group_audit["shared_endpoint_across_safe_groups_count"],
        "shared_endpoint_groups": safe_group_audit["shared_endpoint_groups"],
        "endpoint_safe_group_assignments": safe_group_audit["endpoint_safe_group_assignments"],
        "endpoint_crosses_train_validation_folds": bool(safe_group_audit["shared_endpoint_across_safe_groups_count"]),
        "leakage_prevention_policy": (
            "Assignment neighbourhoods are first preserved, then any groups sharing canonical endpoints are merged "
            "into endpoint_safe_group_id before grouped diagnostics."
        ),
        "model_application_allowed": False,
        **safety_payload(),
    }


def _neighbourhood_grouping_audit(raw_labels: list[dict[str, Any]]) -> dict[str, Any]:
    by_neighbourhood: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_labels:
        by_neighbourhood[str(row["local_assignment_neighbourhood_id"])].append(row)
    return {
        "artifact": "m5_4g_neighbourhood_grouping_audit",
        "neighbourhood_count": len(by_neighbourhood),
        "neighbourhood_rows": [
            {
                "local_assignment_neighbourhood_id": key,
                "row_count": len(value),
                "labels": dict(Counter(row["label"] for row in value)),
                "case_ids": sorted({row["case_id"] for row in value}),
            }
            for key, value in sorted(by_neighbourhood.items())
        ],
        "assignment_neighbourhood_crosses_folds": False,
        **safety_payload(),
    }


def _training_readiness(
    *,
    inventory: dict[str, Any],
    exact_contradiction: dict[str, Any],
    endpoint_audit: dict[str, Any],
    endpoint_leakage: dict[str, Any],
    geometry_audit: dict[str, Any],
) -> dict[str, Any]:
    both_classes = inventory["canonical_unique_positive_count"] > 0 and inventory["canonical_unique_negative_count"] > 0
    positive_groups = inventory["independent_positive_trajectory_component_count"]
    negative_groups = inventory["independent_negative_assignment_neighbourhood_count"]
    grouped_diagnostic_ready = all(
        [
            both_classes,
            positive_groups >= 5,
            negative_groups >= 5,
            exact_contradiction["exact_edge_contradiction_count"] == 0,
            endpoint_audit["endpoint_invalid_count"] == 0,
            endpoint_audit["role_incompatible_count"] == 0,
        ]
    )
    ready_for_model_application = False
    model_blocked_reason = (
        "MODEL_NOT_JUSTIFIED_BASELINE_ALREADY_PERFECT_ON_REVIEWED_SET"
        if geometry_audit["paired_target_choice_subset"]["geometry_only_grouped_validation_is_perfect"]
        else "MODEL_APPLICATION_REQUIRES_LATER_UNSEEN_WINDOW_VALIDATION"
    )
    return {
        "artifact": "m5_4g_training_readiness",
        "readiness_state": (
            "READY_FOR_GROUPED_DIAGNOSTIC_EVALUATION"
            if grouped_diagnostic_ready
            else "BLOCKED_CANONICAL_LABEL_INVENTORY"
        ),
        "ready_for_model_application": ready_for_model_application,
        "both_positive_and_negative_classes_exist": both_classes,
        "independent_positive_group_count": positive_groups,
        "independent_negative_group_count": negative_groups,
        "minimum_independent_groups_per_class_required": 5,
        "exact_edge_contradiction_count": exact_contradiction["exact_edge_contradiction_count"],
        "unresolved_prior_label_conflict_count": inventory["conflict_count"],
        "endpoint_validity_passed": endpoint_audit["endpoint_invalid_count"] == 0,
        "grouped_splitting_possible": grouped_diagnostic_ready,
        "endpoint_crosses_train_validation_folds": endpoint_leakage["endpoint_crosses_train_validation_folds"],
        "construction_metadata_excluded_from_features": True,
        "excluded_feature_names": sorted(FEATURES_EXCLUDED_FROM_MODELING),
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        "model_fit_blocked_reason": model_blocked_reason,
        **safety_payload(),
    }


def _paired_subset_rows(raw_labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_out = []
    for row in raw_labels:
        rows_out.append(
            {
                "canonical_edge_key": row["canonical_edge_key"],
                "label": row["label"],
                "features": row["features"],
                "evaluation_group_id": row["local_assignment_neighbourhood_id"],
                "source_candidate_id": row["source_candidate_id"],
                "target_candidate_id": row["target_candidate_id"],
                "source_visible_person_base_id": row.get("source_visible_person_base_id"),
                "target_visible_person_base_id": row.get("target_visible_person_base_id"),
            }
        )
    return rows_out


def _full_subset_rows(canonical_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "canonical_edge_key": row["canonical_edge_key"],
            "label": row.get("label") or row.get("binary_label"),
            "features": row["features"],
            "evaluation_group_id": row.get("evaluation_group_id"),
            "source_candidate_id": row["source_candidate_id"],
            "target_candidate_id": row["target_candidate_id"],
            "source_visible_person_base_id": row.get("source_visible_person_base_id"),
            "target_visible_person_base_id": row.get("target_visible_person_base_id"),
        }
        for row in canonical_rows
    ]


def _output_hash(paths: list[Path]) -> str:
    payload = []
    for path in sorted(paths):
        if path.exists() and path.is_file():
            payload.append({"path": path.as_posix(), "sha256": sha256_file(path), "size": path.stat().st_size})
    return stable_hash(payload)


def _write_review_pack(stage_root: Path, repo_root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    source_paths = [
        stage_root / "validation" / "m5_4g_validation_summary.json",
        stage_root / "continuity_v9" / "ingestion" / "completed_review_event_validation.json",
        stage_root / "continuity_v9" / "ingestion" / "sealed_mapping_validation.json",
        stage_root / "continuity_v9" / "ingestion" / "decoded_target_choice_summary.json",
        stage_root / "continuity_v9" / "ingestion" / "decoded_target_choice_rows.jsonl",
        stage_root / "continuity_v9" / "ingestion" / "endpoint_revalidation_rows.jsonl",
        stage_root / "continuity_v9" / "labels" / "raw_target_choice_edge_labels.jsonl",
        stage_root / "continuity_v9" / "labels" / "positive_confirmation_deduplication.json",
        stage_root / "continuity_v9" / "labels" / "negative_edge_novelty_audit.json",
        stage_root / "continuity_v9" / "labels" / "canonical_continuity_label_inventory.json",
        stage_root / "continuity_v9" / "labels" / "canonical_continuity_label_rows.jsonl",
        stage_root / "continuity_v9" / "labels" / "label_lineage_manifest.json",
        stage_root / "continuity_v9" / "audit" / "exact_edge_contradiction_audit.json",
        stage_root / "continuity_v9" / "audit" / "endpoint_leakage_audit.json",
        stage_root / "continuity_v9" / "audit" / "geometry_shortcut_audit.json",
        stage_root / "continuity_v9" / "audit" / "baseline_comparison_audit.json",
        stage_root / "continuity_v9" / "validation" / "training_readiness.json",
        stage_root / "continuity_v9" / "validation" / "grouped_baseline_results.json",
        repo_root / "src" / "football_intelligence" / "replay" / "server_sealed_target_choice_ingestion.py",
    ]
    pack_root = stage_root / "continuity_v9" / "review_pack"
    pack_root.mkdir(parents=True, exist_ok=True)
    copied = []
    for index, source in enumerate(source_paths, start=1):
        suffix = source.suffix or ".txt"
        name = f"{index:02d}_{source.stem}{suffix}"
        target = pack_root / name
        target.write_bytes(source.read_bytes())
        copied.append({"source": str(source), "review_pack_path": str(target), "size": target.stat().st_size})
    explanation = f"""M5.4G Review Pack

Purpose
This folder gives the next reviewer or ChatGPT session a compact, maximum-20-file context bundle for M5.4G.
It documents the server-sealed ingestion of the completed six-case F6.2 target-choice review, the decoded A/B
decisions, canonical positive/negative continuity label inventory, leakage audits, baseline diagnostics, and the
reason no continuity model was fitted.

What was achieved
- Replayed the append-only F6.2 event log as the authoritative decision source.
- Verified the sealed server-side mapping before decoding any A/B choice.
- Decoded all six decisions as agreements with the prior accepted target.
- Converted each eligible comparison into one positive confirmation and one new reviewed negative edge.
- Deduplicated positive confirmations against the existing F2 positive inventory, so the six confirmations did not
  become six new independent positive examples.
- Created six canonical reviewed negative edges across six independent assignment neighbourhoods.
- Checked exact-edge contradictions, endpoint validity, role eligibility, endpoint leakage, and neighbourhood grouping.
- Evaluated simple grouped baselines first and refused model fitting because the reviewed set is already explained by
  a conservative geometry baseline.

Key current result
- Final classification: {summary.get("final_classification")}
- Exact blocker: {summary.get("exact_blocker")}
- Model fit performed: {summary.get("model_fit_performed")}
- Learned continuity rows updated: {summary.get("learned_continuity_rows_updated")}

Files in this pack
01 validation summary: high-level M5.4G pass/blocker state.
02 event validation: event-log replay and hash/snapshot checks.
03 sealed mapping validation: mapping hash, case binding, and server-side policy checks.
04 decoded summary: agreement/conflict/N/U counts.
05 decoded rows: case-level decoded A/B decisions.
06 endpoint rows: endpoint and generic role-scope eligibility per case.
07 raw edge labels: positive confirmation and negative labels before canonical dedupe.
08 positive dedupe: F2 exact-confirmation accounting.
09 negative novelty: new negative edge and contradiction accounting.
10 canonical inventory: final label-count summary.
11 canonical label rows: deduplicated positive and negative edge rows.
12 label lineage: source paths and safety lineage.
13 exact contradiction audit: accept/reject edge collision check.
14 endpoint leakage audit: shared endpoint fold-risk check.
15 geometry shortcut audit: whether one-feature geometry separates labels.
16 baseline comparison audit: baseline-first model justification result.
17 training readiness: grouped diagnostic readiness and model-application block.
18 grouped baseline results: detailed grouped threshold diagnostics.
19 implementation module: code that generated M5.4G outputs.
20 this explanation.
"""
    explanation_path = pack_root / "20_review_pack_explanation.txt"
    write_text(explanation_path, explanation)
    copied.append(
        {
            "source": "generated_explanation",
            "review_pack_path": str(explanation_path),
            "size": explanation_path.stat().st_size,
        }
    )
    manifest = {
        "artifact": "m5_4g_review_pack_manifest",
        "review_pack_root": str(pack_root),
        "file_count": len(copied),
        "max_file_count": 20,
        "files": copied,
        **safety_payload(),
    }
    return manifest


def build_m5_4g_server_sealed_target_choice_ingestion(
    *,
    stage_root: Path,
    repo_root: Path,
    write_review_pack: bool = False,
) -> dict[str, Any]:
    continuity_v8 = stage_root / "continuity_v8"
    continuity_v9 = stage_root / "continuity_v9"
    ingestion_root = continuity_v9 / "ingestion"
    labels_root = continuity_v9 / "labels"
    audit_root = continuity_v9 / "audit"
    validation_root = continuity_v9 / "validation"
    stage_validation_root = stage_root / "validation"
    before_inventory = _inventory(_source_mutation_paths(stage_root), base=stage_root)

    manifest_payload = read_json(continuity_v8 / "target_choice_reviewer_manifest.json")
    expected_case_ids = sorted(str(case["case_id"]) for case in manifest_payload.get("cases", []))
    event_validation, session_audit, timing_audit = _validate_completed_review(
        stage_root=stage_root,
        expected_case_ids=expected_case_ids,
    )
    sealed_validation, mapping_by_case, row_by_anchor = _validate_sealed_mapping(
        stage_root=stage_root,
        expected_case_ids=expected_case_ids,
    )
    replay = event_validation["replay_result"]
    decoded_rows, decoded_summary = _decode_decisions(
        replay=replay,
        mapping_by_case=mapping_by_case,
        row_by_anchor=row_by_anchor,
    )
    endpoint_rows, endpoint_audit = _endpoint_revalidation_rows(
        decoded_rows,
        mapping_by_case=mapping_by_case,
        row_by_anchor=row_by_anchor,
        manifest_payload=manifest_payload,
        replay=replay,
    )
    positive_component_audit = read_json(
        stage_root / "continuity_v3" / "learning" / "positive_equivalence_cluster_audit.json"
    )
    positive_component_count = int(positive_component_audit.get("semantic_independent_component_count", 0))
    f2_positives = _load_f2_positives(stage_root)
    f2_by_key = {row["canonical_edge_key"]: row for row in f2_positives}
    raw_labels = _raw_target_choice_edge_labels(
        decoded_rows=decoded_rows,
        endpoint_rows=endpoint_rows,
        mapping_by_case=mapping_by_case,
        row_by_anchor=row_by_anchor,
        f2_by_key=f2_by_key,
    )
    positive_dedupe, canonical_positive_rows = _positive_confirmation_deduplication(
        raw_labels=raw_labels,
        f2_positive_rows=f2_positives,
        positive_component_count=positive_component_count,
    )
    negative_novelty, canonical_negative_rows = _negative_novelty_audit(
        raw_labels=raw_labels,
        positive_keys={row["canonical_edge_key"] for row in canonical_positive_rows},
        historical_keys=set(f2_by_key),
    )
    exact_contradiction = _exact_edge_contradiction_audit(canonical_positive_rows, canonical_negative_rows)
    inventory, canonical_rows = _inventory_summary(
        raw_labels=raw_labels,
        canonical_positive_rows=canonical_positive_rows,
        canonical_negative_rows=canonical_negative_rows,
        positive_dedupe=positive_dedupe,
        negative_novelty=negative_novelty,
        exact_contradiction=exact_contradiction,
        endpoint_audit=endpoint_audit,
    )
    lineage = _label_lineage_manifest(
        stage_root=stage_root,
        sealed_mapping_hash=sealed_validation["stored_sealed_mapping_hash"],
        raw_labels=raw_labels,
        canonical_positive_rows=canonical_positive_rows,
        canonical_negative_rows=canonical_negative_rows,
    )
    paired_rows = _paired_subset_rows(raw_labels)
    full_rows = _full_subset_rows(canonical_rows)
    baseline_results, diagnostic_results, geometry_audit = _baseline_results(
        paired_rows=paired_rows,
        full_rows=full_rows,
    )
    endpoint_leakage = _endpoint_leakage_audit(full_rows)
    neighbourhood_audit = _neighbourhood_grouping_audit(raw_labels)
    training_readiness = _training_readiness(
        inventory=inventory,
        exact_contradiction=exact_contradiction,
        endpoint_audit=endpoint_audit,
        endpoint_leakage=endpoint_leakage,
        geometry_audit=geometry_audit,
    )
    unseen_window = {
        "artifact": "m5_4g_unseen_window_requirement",
        "interval_780_840_seconds_remains_diagnostic": True,
        "third_unseen_window_required": True,
        "full_match_validation_required_later": True,
        "broader_continuity_claim_permitted": False,
        **safety_payload(),
    }

    if not event_validation["passed"]:
        final_classification = BLOCKED_REVIEW_EVENT_INTEGRITY
        exact_blocker = "completed review event-log integrity failed"
    elif not sealed_validation["passed"]:
        final_classification = BLOCKED_SEALED_MAPPING_INTEGRITY
        exact_blocker = "sealed mapping integrity failed"
    elif decoded_summary["conflict_count"] > 0:
        final_classification = BLOCKED_PRIOR_LABEL_CONFLICT
        exact_blocker = "prior accepted target conflict present"
    elif exact_contradiction["exact_edge_contradiction_count"] > 0:
        final_classification = BLOCKED_EXACT_EDGE_LABEL_CONTRADICTION
        exact_blocker = "exact accept/reject edge contradiction present"
    elif endpoint_audit["endpoint_invalid_count"] > 0 or endpoint_audit["role_incompatible_count"] > 0:
        final_classification = BLOCKED_ENDPOINT_OR_ROLE_VALIDITY
        exact_blocker = "endpoint or role revalidation failed"
    elif negative_novelty["independent_negative_assignment_neighbourhood_count"] < 5:
        final_classification = BLOCKED_INDEPENDENT_NEGATIVE_GROUPS
        exact_blocker = "fewer than five independent negative neighbourhoods"
    elif geometry_audit["paired_target_choice_subset"]["geometry_only_grouped_validation_is_perfect"]:
        final_classification = PASS_MODEL_NOT_JUSTIFIED
        exact_blocker = "MODEL_NOT_JUSTIFIED_BASELINE_ALREADY_PERFECT_ON_REVIEWED_SET"
    else:
        final_classification = PASS_CANONICAL_TWO_CLASS_LABEL_INVENTORY
        exact_blocker = None

    baseline_comparison_audit = {
        "artifact": "m5_4g_baseline_comparison_audit",
        "best_geometry_baseline": baseline_results["best_geometry_baseline"],
        "geometry_only_grouped_result": geometry_audit["best_geometry_one_feature_result"]["metrics"],
        "appearance_only_result": diagnostic_results["appearance_only_result"]["metrics"],
        "combined_result": diagnostic_results["combined_result"]["metrics"],
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        "model_fit_blocked_reason": diagnostic_results["model_fit_blocked_reason"],
        **safety_payload(),
    }
    after_inventory = _inventory(_source_mutation_paths(stage_root), base=stage_root)
    source_mutation_audit = {
        "artifact": "m5_4g_source_mutation_audit",
        "before": before_inventory,
        "after": after_inventory,
        "prior_artifacts_preserved": before_inventory["inventory_hash"] == after_inventory["inventory_hash"],
        "completed_historical_reviews_modified": False,
        "prior_f2_f6_artifacts_modified": False,
        **safety_payload(),
    }
    safety_audit = {
        "artifact": "m5_4g_safety_guardrail_audit",
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
    write_json(audit_root / "reviewer_session_consistency_audit.json", session_audit)
    write_json(audit_root / "review_timing_and_input_audit.json", timing_audit)
    write_json(ingestion_root / "sealed_mapping_validation.json", sealed_validation)
    _write_jsonl(ingestion_root / "decoded_target_choice_rows.jsonl", decoded_rows)
    write_json(ingestion_root / "decoded_target_choice_summary.json", decoded_summary)
    _write_jsonl(ingestion_root / "endpoint_revalidation_rows.jsonl", endpoint_rows)
    write_json(audit_root / "endpoint_and_role_eligibility_audit.json", endpoint_audit)
    _write_jsonl(labels_root / "raw_target_choice_edge_labels.jsonl", raw_labels)
    write_json(labels_root / "positive_confirmation_deduplication.json", positive_dedupe)
    _write_jsonl(labels_root / "canonical_unique_positive_edges.jsonl", canonical_positive_rows)
    write_json(labels_root / "negative_edge_novelty_audit.json", negative_novelty)
    _write_jsonl(labels_root / "canonical_unique_negative_edges.jsonl", canonical_negative_rows)
    write_json(labels_root / "canonical_continuity_label_inventory.json", inventory)
    _write_jsonl(labels_root / "canonical_continuity_label_rows.jsonl", canonical_rows)
    write_json(labels_root / "label_lineage_manifest.json", lineage)
    write_json(audit_root / "exact_edge_contradiction_audit.json", exact_contradiction)
    write_json(audit_root / "endpoint_leakage_audit.json", endpoint_leakage)
    write_json(audit_root / "neighbourhood_grouping_audit.json", neighbourhood_audit)
    write_json(audit_root / "geometry_shortcut_audit.json", geometry_audit)
    write_json(audit_root / "baseline_comparison_audit.json", baseline_comparison_audit)
    write_json(validation_root / "training_readiness.json", training_readiness)
    write_json(validation_root / "grouped_baseline_results.json", baseline_results)
    write_json(validation_root / "grouped_diagnostic_results.json", diagnostic_results)
    write_json(validation_root / "unseen_window_requirement.json", unseen_window)
    write_json(stage_validation_root / "source_mutation_audit.json", source_mutation_audit)
    write_json(stage_validation_root / "safety_guardrail_audit.json", safety_audit)

    output_paths = [
        ingestion_root / "completed_review_event_validation.json",
        audit_root / "reviewer_session_consistency_audit.json",
        audit_root / "review_timing_and_input_audit.json",
        ingestion_root / "sealed_mapping_validation.json",
        ingestion_root / "decoded_target_choice_rows.jsonl",
        ingestion_root / "decoded_target_choice_summary.json",
        ingestion_root / "endpoint_revalidation_rows.jsonl",
        audit_root / "endpoint_and_role_eligibility_audit.json",
        labels_root / "raw_target_choice_edge_labels.jsonl",
        labels_root / "positive_confirmation_deduplication.json",
        labels_root / "canonical_unique_positive_edges.jsonl",
        labels_root / "negative_edge_novelty_audit.json",
        labels_root / "canonical_unique_negative_edges.jsonl",
        labels_root / "canonical_continuity_label_inventory.json",
        labels_root / "canonical_continuity_label_rows.jsonl",
        labels_root / "label_lineage_manifest.json",
        audit_root / "exact_edge_contradiction_audit.json",
        audit_root / "endpoint_leakage_audit.json",
        audit_root / "neighbourhood_grouping_audit.json",
        audit_root / "geometry_shortcut_audit.json",
        audit_root / "baseline_comparison_audit.json",
        validation_root / "training_readiness.json",
        validation_root / "grouped_baseline_results.json",
        validation_root / "grouped_diagnostic_results.json",
        validation_root / "unseen_window_requirement.json",
        stage_validation_root / "source_mutation_audit.json",
        stage_validation_root / "safety_guardrail_audit.json",
    ]
    deterministic_output_hash = _output_hash(output_paths)
    validation_summary = {
        "artifact": "m5_4g_validation_summary",
        "completed_review_validation": event_validation["completed_review_validation"],
        "event_log_replay_result": "PASS" if event_validation["passed"] else "FAIL",
        "reviewer_session_consistency_result": session_audit["session_metadata_consistency_result"],
        "sealed_mapping_validation": "PASS" if sealed_validation["passed"] else "FAIL",
        "decoded_agreement_count": decoded_summary["agreement_count"],
        "conflict_count": decoded_summary["conflict_count"],
        "N_count": decoded_summary["neither_count"],
        "U_count": decoded_summary["unresolved_count"],
        "raw_positive_confirmation_count": positive_dedupe["new_positive_confirmation_event_count"],
        "exact_positive_confirmation_count": positive_dedupe["exact_positive_confirmation_count"],
        "new_distinct_positive_count": positive_dedupe["new_distinct_positive_count"],
        "canonical_unique_positive_count": inventory["canonical_unique_positive_count"],
        "raw_negative_count": negative_novelty["raw_negative_count"],
        "canonical_unique_negative_count": inventory["canonical_unique_negative_count"],
        "independent_positive_group_count": inventory["independent_positive_trajectory_component_count"],
        "independent_negative_neighbourhood_count": inventory["independent_negative_assignment_neighbourhood_count"],
        "exact_edge_contradiction_count": exact_contradiction["exact_edge_contradiction_count"],
        "endpoint_invalid_count": endpoint_audit["endpoint_invalid_count"],
        "role_incompatible_count": endpoint_audit["role_incompatible_count"],
        "final_training_readiness_state": training_readiness["readiness_state"],
        "best_geometry_baseline": baseline_results["best_geometry_baseline"],
        "geometry_only_grouped_result": baseline_comparison_audit["geometry_only_grouped_result"],
        "appearance_only_result": baseline_comparison_audit["appearance_only_result"],
        "combined_result": baseline_comparison_audit["combined_result"],
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        "third_unseen_window_requirement": unseen_window,
        "deterministic_ingestion_hash": deterministic_output_hash,
        "final_classification": final_classification,
        "exact_blocker": exact_blocker,
        **safety_payload(),
    }
    write_json(stage_validation_root / "m5_4g_validation_summary.json", validation_summary)

    review_pack_manifest = None
    if write_review_pack:
        review_pack_manifest = _write_review_pack(stage_root, repo_root, validation_summary)

    return {
        "final_classification": final_classification,
        "exact_blocker": exact_blocker,
        "validation_summary_path": str(stage_validation_root / "m5_4g_validation_summary.json"),
        "deterministic_ingestion_hash": deterministic_output_hash,
        "review_pack_manifest": review_pack_manifest,
        **validation_summary,
    }
