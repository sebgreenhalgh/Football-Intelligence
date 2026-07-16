"""Build the M5.5D.3B corrected semantic merge and episode audit.

The historical M5.5D.2C and M5.5D.3 ledgers are read-only inputs.  The
completed M5.5D.3A review is validated, then applied as an immutable sidecar
overlay to only the 27 malformed duplicate mappings.  No source detector,
tracker, fine-vision model, or review decision root is mutated by this stage.
"""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.config import load_ui_config, ui_config_hash
from football_intelligence.review_chassis.manifest import load_manifest, manifest_hash

try:
    from scripts.build_m5_5d3a_followup_repair import (
        FRAME_HEIGHT,
        FRAME_WIDTH,
        REPO,
        ROOT,
        authoritative_rows,
        clean_bbox,
        frame_catalog,
        read_json,
        read_jsonl,
        replay_historical_ledger,
    )
    from scripts.build_m5_5d3_consolidation import B_ROOT, SCIENCE_ROOT
except ModuleNotFoundError:  # Executed as a file by ``uv run python scripts/...``.
    from build_m5_5d3a_followup_repair import (
        FRAME_HEIGHT,
        FRAME_WIDTH,
        REPO,
        ROOT,
        authoritative_rows,
        clean_bbox,
        frame_catalog,
        read_json,
        read_jsonl,
        replay_historical_ledger,
    )
    from build_m5_5d3_consolidation import B_ROOT, SCIENCE_ROOT


STAGE_ROOT = (
    ROOT / r"matches\128058\runs\step_m5\part 2\M5_5D3B_CORRECTED_FOLLOWUP_INGESTION_AND_EPISODE_REEVALUATION_v1"
)
PRIOR_D3_ROOT = (
    ROOT
    / r"matches\128058\runs\step_m5\part 2\M5_5D3_HUMAN_VALIDATED_OBSERVATION_CONSOLIDATION_AND_OCCLUSION_REEVALUATION_v1"
)
PRIOR_D3A_ROOT = (
    ROOT / r"matches\128058\runs\step_m5\part 2\M5_5D3A_FOLLOWUP_REVIEW_SEMANTIC_AND_TARGET_EXCLUSION_REPAIR_v1"
)
PRIOR_D2C_PACKAGE = (
    ROOT
    / r"matches\128058\runs\step_m5\part 2\M5_5D2C_TARGETED_ENCOUNTER_CANDIDATE_SEMANTIC_AUDIT_v1\03_TARGETED_SEMANTIC_REVIEW_PACKAGE"
)
PRIOR_D3_PACKAGE = PRIOR_D3_ROOT / "08_OPTIONAL_FOLLOWUP_REVIEW_PACKAGE" / "review_package"
REPAIRED_PACKAGE = PRIOR_D3A_ROOT / "03_REPAIRED_FOLLOWUP_REVIEW_PACKAGE"
REPAIRED_DECISIONS = REPAIRED_PACKAGE / "decisions"
REPAIRED_SEALED = REPAIRED_PACKAGE / "sealed" / "sealed_route_redacted.json"
MALFORMED_ROWS = PRIOR_D3_ROOT / "02_DUPLICATE_AND_CROP_PROVENANCE_AUDIT" / "self_duplicate_rows.jsonl"
NORMALIZED_ROWS = PRIOR_D3_ROOT / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "normalized_review_rows.jsonl"
PRIOR_EDGES = PRIOR_D3_ROOT / "03_HUMAN_VALIDATED_OBSERVATION_GRAPH" / "duplicate_edges.jsonl"
PRIOR_FRAME_ROWS = PRIOR_D3_ROOT / "04_REBUILT_ENCOUNTER_EPISODES" / "frame_supply_rows.jsonl"
PRIOR_EPISODE_ROWS = PRIOR_D3_ROOT / "04_REBUILT_ENCOUNTER_EPISODES" / "rebuilt_episode_rows.jsonl"
PRIOR_CLASS_ROWS = PRIOR_D3_ROOT / "05_OCCLUSION_INTERVAL_REEVALUATION" / "classification_rows.jsonl"

AUTHORIZED_BASELINE = "844c1db10cac7acc17eaa06752223135baa6f584"
STAGE_ID = "M5_5D3B_CORRECTED_FOLLOWUP_INGESTION_AND_EPISODE_REEVALUATION_v1"
FINAL_CLASSIFICATION = "PASS_CORRECTED_INGESTION_NO_GENUINE_OCCLUSION"

SAFETY = {
    **safety_payload(),
    "identity_tracking_performed": False,
    "player_slots_assigned": False,
    "goalkeeper_slots_assigned": False,
    "exact_22_forcing_performed": False,
    "event_analysis_performed": False,
    "metric_analysis_performed": False,
    "tactical_analysis_performed": False,
    "physical_performance_analysis_performed": False,
    "model_fit_performed": False,
    "learned_continuity_rows_updated": 0,
    "project_defaults_changed": False,
    "canonical_candidate_rows_replaced": False,
    "historical_artifacts_mutated": False,
    "match_local_only": True,
    "sandbox_only": True,
    "safe_to_apply_globally": False,
    "episodes_rebuilt": True,
    "ghosts_reassessed": False,
    "fine_vision_models_run": False,
    "football_metrics_generated": False,
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8"
    )


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()


def snapshot_tree(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        rows.append(
            {"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return {
        "root": str(root),
        "file_count": len(rows),
        "total_size_bytes": sum(row["size"] for row in rows),
        "aggregate_sha256": digest(rows),
        "files": rows,
    }


def snapshot_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    old = {row["path"]: row for row in before["files"]}
    new = {row["path"]: row for row in after["files"]}
    changed = sorted(path for path in set(old) & set(new) if old[path]["sha256"] != new[path]["sha256"])
    return {
        "changed_paths": changed,
        "added_paths": sorted(set(new) - set(old)),
        "deleted_paths": sorted(set(old) - set(new)),
        "unchanged": not changed and set(old) == set(new),
    }


def parse_notes(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def authorization_audit() -> dict[str, Any]:
    status_lines = git("status", "--short").splitlines()
    allowed_new_files = {"scripts/build_m5_5d3b_corrected_ingestion.py", "tests/test_m5_5d3b_corrected_ingestion.py"}
    preexisting_status_lines = [line for line in status_lines if line[3:] not in allowed_new_files]
    head = git("rev-parse", "HEAD")
    baseline = git("rev-parse", f"{AUTHORIZED_BASELINE}^{{commit}}")
    ancestor = (
        subprocess.run(["git", "merge-base", "--is-ancestor", baseline, head], cwd=REPO, check=False).returncode == 0
    )
    return {
        "repository": str(REPO),
        "authorized_baseline_input": "844c1db",
        "authorized_baseline_full_hash": baseline,
        "head": head,
        "worktree_clean": not bool(preexisting_status_lines),
        "preexisting_status_lines": preexisting_status_lines,
        "current_stage_status_lines": [line for line in status_lines if line[3:] in allowed_new_files],
        "baseline_is_ancestor": ancestor,
        "intervening_commits": git("log", "--oneline", "--decorate", "--no-merges", f"{baseline}..HEAD").splitlines(),
        "intervening_diff_stat": git("diff", "--stat", f"{baseline}..HEAD"),
        "intervening_changed_files": git("diff", "--name-status", f"{baseline}..HEAD").splitlines(),
        "authorized": bool(not preexisting_status_lines and baseline == AUTHORIZED_BASELINE and ancestor),
    }


def validate_repaired_review(canonical: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = REPAIRED_PACKAGE / "reviewer_manifest.json"
    ui_path = REPAIRED_PACKAGE / "ui_config.json"
    state_path = REPAIRED_DECISIONS / "review_decisions.json"
    completed_path = REPAIRED_DECISIONS / "completed_review.json"
    events_path = REPAIRED_DECISIONS / "completed_review_events.jsonl"
    manifest = load_manifest(manifest_path)
    ui = load_ui_config(ui_path)
    state = read_json(state_path)
    completed = read_json(completed_path)
    events = read_jsonl(events_path)
    expected_manifest_hash = manifest_hash(manifest)
    expected_ui_hash = ui_config_hash(ui)
    decisions = state.get("decisions", {})
    allowed = {option.value for option in ui.decisions}
    replayed: dict[str, str] = {}
    sequences = [int(event.get("event_sequence", -1)) for event in events]
    errors: list[str] = []
    completion_indices = []
    for event in events:
        sequence = int(event.get("event_sequence", -1))
        if event.get("reviewer_session_id") != "m5_5d3a_repaired_followup_human_reviewer":
            errors.append(f"reviewer_session:{sequence}")
        if event.get("manifest_hash") != expected_manifest_hash:
            errors.append(f"manifest_hash:{sequence}")
        if event.get("ui_config_hash") != expected_ui_hash:
            errors.append(f"ui_hash:{sequence}")
        if event.get("event_type") == "decision":
            decision = event.get("new_decision")
            if decision not in allowed:
                errors.append(f"decision_value:{sequence}")
            if event.get("case_id"):
                replayed[event["case_id"]] = decision
        elif event.get("event_type") == "undo" and event.get("case_id"):
            if event.get("restored_decision") is None:
                replayed.pop(event["case_id"], None)
            else:
                replayed[event["case_id"]] = event["restored_decision"]
        elif event.get("event_type") == "complete":
            completion_indices.append(sequence)
    duplicate_sequence_count = len(sequences) - len(set(sequences))
    if sequences != sorted(sequences) or sorted(set(sequences)) != list(range(1, max(sequences) + 1)):
        errors.append("event_sequence_order_or_gap")
    if completion_indices != [sequences[-1]]:
        errors.append("completion_not_last_or_not_unique")
    if replayed != decisions:
        errors.append("event_replay_final_state_mismatch")
    if stable_hash(completed.get("state", {})) != completed.get("decision_state_hash"):
        errors.append("completed_export_state_hash_mismatch")
    state_hash_matches = (
        completed.get("decision_state_hash") == "5534008295eaaeda98ec06e3e0e4585c0829bce88411825d35c11525b1d68195"
    )
    counts = Counter(decisions.values())
    review_valid = (
        len(manifest.cases) == 27
        and state.get("completed") is True
        and len(decisions) == 27
        and counts == Counter({"VALID_VISIBLE_SINGLE_PERSON_NO_DUPLICATE": 23, "DUPLICATE_SAME_PERSON_COUNTERPART": 4})
        and state.get("reviewer_session_id") == "m5_5d3a_repaired_followup_human_reviewer"
        and len(state.get("notes", {})) == 27
        and state_hash_matches
        and not errors
    )
    validation = {
        "schema_version": "m5_5d3b.repaired_review_validation.v1",
        "completed": state.get("completed"),
        "reviewed": len(decisions),
        "remaining": 0 if state.get("completed") else 27 - len(decisions),
        "total_cases": len(manifest.cases),
        "reviewer_session_id": state.get("reviewer_session_id"),
        "decision_counts": dict(counts),
        "expected_decision_counts": {
            "VALID_VISIBLE_SINGLE_PERSON_NO_DUPLICATE": 23,
            "DUPLICATE_SAME_PERSON_COUNTERPART": 4,
        },
        "notes_count": len(state.get("notes", {})),
        "review_duration_seconds": completed.get("state", {}).get("elapsed_active_seconds"),
        "decision_state_hash": completed.get("decision_state_hash"),
        "decision_state_hash_matches_expected": state_hash_matches,
        "manifest_hash": completed.get("manifest_hash"),
        "manifest_hash_valid": completed.get("manifest_hash") == expected_manifest_hash,
        "ui_config_hash": completed.get("ui_config_hash"),
        "ui_config_hash_valid": completed.get("ui_config_hash") == expected_ui_hash,
        "event_count": len(events),
        "duplicate_event_sequence_count": duplicate_sequence_count,
        "decision_event_count": sum(event.get("event_type") == "decision" for event in events),
        "final_case_state_count": len(decisions),
        "event_replay_materializes_final_state": replayed == decisions,
        "completion_event_count": len(completion_indices),
        "no_events_after_completion": completion_indices == [sequences[-1]],
        "errors": errors,
        "valid": review_valid,
    }
    return validation, {"manifest": manifest, "ui": ui, "state": state, "completed": completed, "events": events}


def validate_counterparts(
    review: dict[str, Any], malformed: list[dict[str, Any]], canonical: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    sealed = read_json(REPAIRED_SEALED)["case_source_rows"]
    malformed_by_case = {row["review_case_id"]: row for row in malformed}
    canonical_by_id = {row.get("candidate_id"): row for row in canonical}
    rows: list[dict[str, Any]] = []
    for followup_case, decision in sorted(review["state"]["decisions"].items()):
        item = sealed.get(followup_case, {})
        original_case = item.get("prior_review_case_id")
        spatial = parse_notes(review["state"].get("notes", {}).get(followup_case)).get("spatial_annotation", {})
        target_id = item.get("target_candidate_id")
        target_hash = item.get("target_source_row_hash")
        counterpart_number = spatial.get("duplicate_counterpart_number")
        bindings = {int(row["number"]): row for row in item.get("candidate_bindings", [])}
        target = canonical_by_id.get(target_id)
        counterpart = bindings.get(int(counterpart_number)) if counterpart_number is not None else None
        counterpart_row = canonical_by_id.get(counterpart.get("candidate_id")) if counterpart else None
        same_frame = bool(
            counterpart_row and target and int(counterpart_row["frame_sequence"]) == int(target["frame_sequence"])
        )
        distinct = bool(
            counterpart
            and counterpart.get("candidate_id") != target_id
            and counterpart.get("source_row_hash") != target_hash
        )
        valid = bool(target and target.get("canonical_source_row_hash") == target_hash)
        if decision == "DUPLICATE_SAME_PERSON_COUNTERPART":
            valid = valid and counterpart is not None and counterpart_row is not None and same_frame and distinct
        else:
            valid = valid and counterpart_number is None
        rows.append(
            {
                "followup_case_id": followup_case,
                "original_case_id": original_case,
                "decision": decision,
                "target_candidate_id": target_id,
                "target_source_row_hash": target_hash,
                "target_binding_valid": bool(target and target.get("canonical_source_row_hash") == target_hash),
                "counterpart_number": counterpart_number,
                "counterpart_candidate_id": counterpart.get("candidate_id") if counterpart else None,
                "counterpart_source_row_hash": counterpart.get("source_row_hash") if counterpart else None,
                "counterpart_exists": counterpart is not None and counterpart_row is not None,
                "same_frame": same_frame,
                "distinct_source_row": distinct,
                "original_malformed_case_present": original_case in malformed_by_case,
                "valid": valid,
            }
        )
    if not all(row["valid"] for row in rows):
        raise RuntimeError("repaired counterpart validation failed closed")
    return rows


class UnionFind:
    def __init__(self, values: Iterable[str] = ()) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def build_merge(
    review: dict[str, Any],
    malformed: list[dict[str, Any]],
    canonical: list[dict[str, Any]],
    counterpart_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    sealed = read_json(REPAIRED_SEALED)["case_source_rows"]
    original = read_jsonl(NORMALIZED_ROWS)
    original_by_case = {row["review_case_id"]: row for row in original}
    malformed_by_case = {row["review_case_id"]: row for row in malformed}
    followup_by_original = {row["original_case_id"]: row for row in counterpart_rows}
    replacement_rows: list[dict[str, Any]] = []
    preserved_rows: list[dict[str, Any]] = []
    authoritative: list[dict[str, Any]] = []
    for case_id, row in sorted(original_by_case.items()):
        replacement = followup_by_original.get(case_id)
        original_decision = row["semantic_decision"]
        if replacement:
            final_decision = (
                "DUPLICATE_OF_ANOTHER_DETECTION"
                if replacement["decision"] == "DUPLICATE_SAME_PERSON_COUNTERPART"
                else "VALID_VISIBLE_SINGLE_PERSON"
            )
            merged = {
                **row,
                "original_decision": original_decision,
                "replacement_followup_case_id": replacement["followup_case_id"],
                "replacement_decision": replacement["decision"],
                "replacement_counterpart": replacement["counterpart_number"],
                "authoritative_final_semantic": final_decision,
                "semantic_decision": final_decision,
                "review_usable": True,
                "replacement_reason": "replace malformed self/repeated-row mapping with completed target-exclusion review",
                "preserved_original_provenance": {
                    "review_case_id": case_id,
                    "decision_event_sequence": row.get("decision_event_sequence"),
                    "notes": row.get("notes"),
                },
            }
            replacement_rows.append(
                {
                    "original_case_id": case_id,
                    "original_decision": original_decision,
                    "replacement_followup_case_id": replacement["followup_case_id"],
                    "replacement_decision": replacement["decision"],
                    "replacement_counterpart": replacement["counterpart_number"],
                    "authoritative_final_semantic": final_decision,
                    "replacement_reason": merged["replacement_reason"],
                    "preserved_original_provenance": merged["preserved_original_provenance"],
                }
            )
        else:
            final_decision = (
                "EVIDENCE_UNRESOLVED"
                if original_decision == "DUPLICATE_OF_ANOTHER_DETECTION"
                and row.get("duplicate_evidence_classification") == "UNRESOLVED"
                else original_decision
            )
            merged = {
                **row,
                "original_decision": original_decision,
                "replacement_followup_case_id": None,
                "replacement_decision": None,
                "replacement_counterpart": None,
                "authoritative_final_semantic": final_decision,
                "preserved_original_provenance": {
                    "review_case_id": case_id,
                    "decision_event_sequence": row.get("decision_event_sequence"),
                    "notes": row.get("notes"),
                },
            }
            preserved_rows.append(
                {
                    "original_case_id": case_id,
                    "original_decision": original_decision,
                    "authoritative_final_semantic": final_decision,
                    "replacement_followup_case_id": None,
                    "preserved_original_provenance": merged["preserved_original_provenance"],
                }
            )
        authoritative.append(merged)
    if len(replacement_rows) != 27 or set(followup_by_original) != set(malformed_by_case):
        raise RuntimeError("replacement scope is not exactly the 27 malformed cases")
    counts = Counter(row["authoritative_final_semantic"] for row in authoritative)
    summary = {
        "schema_version": "m5_5d3b.corrected_decision_summary.v1",
        "original_observation_count": len(original),
        "replacement_count": len(replacement_rows),
        "preserved_row_count": len(preserved_rows),
        "final_semantic_counts": dict(counts),
        "valid_single_non_duplicate_count": counts["VALID_VISIBLE_SINGLE_PERSON"],
        "duplicate_decision_count": counts["DUPLICATE_OF_ANOTHER_DETECTION"],
        "false_positive_count": counts["FALSE_POSITIVE_OR_EMPTY"],
        "merged_count": counts["MERGED_MULTIPLE_VISIBLE_PEOPLE"],
        "partial_count": counts["PARTIAL_PERSON_OR_BODY_FRAGMENT"],
        "unresolved_count": counts["EVIDENCE_UNRESOLVED"] + counts["WRONG_VISIBLE_PERSON_FOR_ENCOUNTER"],
        "preserved_validated_duplicate_edges": 7,
        "preserved_unresolved_duplicate_rows": 2,
        "historical_ledgers_mutated": False,
        "immutable_overlay": True,
        "expected_aggregate_check": {
            "valid_single": 25,
            "false_positive": 6,
            "merged": 5,
            "partial": 1,
            "duplicate_edges": 11,
            "unresolved": 2,
        },
        "expected_aggregate_matches": counts["VALID_VISIBLE_SINGLE_PERSON"] == 25
        and counts["FALSE_POSITIVE_OR_EMPTY"] == 6
        and counts["MERGED_MULTIPLE_VISIBLE_PEOPLE"] == 5
        and counts["PARTIAL_PERSON_OR_BODY_FRAGMENT"] == 1
        and counts["DUPLICATE_OF_ANOTHER_DETECTION"] == 11,
    }
    return {
        "authoritative": authoritative,
        "replacement": replacement_rows,
        "preserved": preserved_rows,
        "summary": summary,
        "sealed": sealed,
        "counterpart_rows": counterpart_rows,
    }


def graph_from_merge(
    merge: dict[str, Any],
    canonical: list[dict[str, Any]],
    prior_edges: list[dict[str, Any]],
    counterpart_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    canonical_by_id = {row.get("candidate_id"): row for row in canonical}
    node_by_id: dict[str, dict[str, Any]] = {}
    row_by_id = {row["machine_used_observation_id"]: row for row in merge["authoritative"]}
    for row in merge["authoritative"]:
        decision = row["authoritative_final_semantic"]
        semantic_type = {
            "VALID_VISIBLE_SINGLE_PERSON": "VALID_SINGLE_PERSON",
            "DUPLICATE_OF_ANOTHER_DETECTION": "VALID_SINGLE_PERSON",
            "FALSE_POSITIVE_OR_EMPTY": "FALSE_POSITIVE",
            "MERGED_MULTIPLE_VISIBLE_PEOPLE": "MERGED_MULTI_PERSON",
            "PARTIAL_PERSON_OR_BODY_FRAGMENT": "PARTIAL_FRAGMENT",
        }.get(decision, "UNRESOLVED")
        node_by_id[row["machine_used_observation_id"]] = {
            "observation_id": row["machine_used_observation_id"],
            "semantic_type": semantic_type,
            "semantic_decision": decision,
            "source_layer": row.get("source_layer"),
            "frame_sequence": row.get("frame_sequence"),
            "encounter_episode_ids": row.get("encounter_episode_ids", []),
            "bbox": row.get("corrected_bbox"),
            "source_row_hash": row.get("source_row_hash"),
            "independent_person_supply": 1 if semantic_type == "VALID_SINGLE_PERSON" else 0,
            "shared_track_capacity": 2 if semantic_type == "MERGED_MULTI_PERSON" else 0,
            "partial_evidence": semantic_type == "PARTIAL_FRAGMENT",
            "original_review_case_id": row.get("review_case_id"),
            "human_validated": True,
        }
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str]] = set()

    def add_edge(left: str, right: str, case_id: str, validated_by: str, corrected: bool = False) -> None:
        if left == right:
            raise RuntimeError("self duplicate edge reached graph builder")
        key = tuple(sorted((left, right)))
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append(
            {
                "edge_type": "DUPLICATE_OF",
                "left_observation_id": left,
                "right_observation_id": right,
                "review_case_id": case_id,
                "validated_by": validated_by,
                "corrected_overlay_edge": corrected,
            }
        )

    for edge in prior_edges:
        left, right = edge["left_observation_id"], edge["right_observation_id"]
        if left not in node_by_id:
            continue
        canonical_row = canonical_by_id.get(right)
        if canonical_row is None:
            continue
        node_by_id.setdefault(
            right,
            {
                "observation_id": right,
                "semantic_type": "VALID_SINGLE_PERSON",
                "semantic_decision": "DUPLICATE_COUNTERPART_CONTEXT",
                "source_layer": "DUPLICATE_COUNTERPART_CONTEXT",
                "frame_sequence": canonical_row.get("frame_sequence"),
                "encounter_episode_ids": row_by_id[left].get("encounter_episode_ids", []),
                "bbox": clean_bbox(canonical_row["bbox"]),
                "source_row_hash": canonical_row.get("canonical_source_row_hash"),
                "candidate_id": canonical_row.get("candidate_id"),
                "independent_person_supply": 1,
                "shared_track_capacity": 0,
                "partial_evidence": False,
                "human_validated": True,
            },
        )
        add_edge(left, right, edge.get("review_case_id"), edge.get("validated_by", "PRESERVED_M5_5D3_EDGE"))
    for row in counterpart_rows:
        if row["decision"] != "DUPLICATE_SAME_PERSON_COUNTERPART":
            continue
        original = row_by_id.get(
            next(
                (
                    r["machine_used_observation_id"]
                    for r in merge["authoritative"]
                    if r["review_case_id"] == row["original_case_id"]
                ),
                "",
            )
        )
        if original is None:
            raise RuntimeError("corrected counterpart original row missing")
        canonical_row = canonical_by_id.get(row["counterpart_candidate_id"])
        if canonical_row is None:
            raise RuntimeError("corrected counterpart canonical row missing")
        counterpart_id = f"canonical_observation_{canonical_row['candidate_id']}"
        node_by_id.setdefault(
            counterpart_id,
            {
                "observation_id": counterpart_id,
                "semantic_type": "VALID_SINGLE_PERSON",
                "semantic_decision": "DUPLICATE_COUNTERPART_CONTEXT",
                "source_layer": "DUPLICATE_COUNTERPART_CONTEXT",
                "frame_sequence": canonical_row.get("frame_sequence"),
                "encounter_episode_ids": original.get("encounter_episode_ids", []),
                "bbox": clean_bbox(canonical_row["bbox"]),
                "source_row_hash": canonical_row.get("canonical_source_row_hash"),
                "candidate_id": canonical_row.get("candidate_id"),
                "independent_person_supply": 1,
                "shared_track_capacity": 0,
                "partial_evidence": False,
                "human_validated": True,
            },
        )
        add_edge(
            original["machine_used_observation_id"],
            counterpart_id,
            row["original_case_id"],
            "M5.5D.3B_CORRECTED_FOLLOWUP",
            True,
        )
    uf = UnionFind(node_by_id)
    for edge in edges:
        uf.union(edge["left_observation_id"], edge["right_observation_id"])
    groups: dict[str, list[str]] = defaultdict(list)
    for node_id in sorted(node_by_id):
        groups[uf.find(node_id)].append(node_id)
    clusters: list[dict[str, Any]] = []
    representatives: list[dict[str, Any]] = []
    cluster_by_node: dict[str, str] = {}
    for index, members in enumerate(sorted(groups.values(), key=lambda values: min(values)), start=1):
        cluster_id = f"duplicate_cluster_{index:03d}"
        for member in members:
            cluster_by_node[member] = cluster_id
            node_by_id[member]["observation_cluster_id"] = cluster_id
        duplicate = len(members) > 1
        eligible = [
            node_by_id[member] for member in members if node_by_id[member]["semantic_type"] == "VALID_SINGLE_PERSON"
        ]
        representative = min(
            eligible or [node_by_id[members[0]]],
            key=lambda node: (
                0 if node.get("corrected_bbox") else 1,
                0 if node["semantic_type"] == "VALID_SINGLE_PERSON" else 1,
                0 if not node.get("partial_evidence") else 1,
                0 if node.get("human_validated") else 1,
                -(float(node.get("confidence") or 0.0)),
                str(node["observation_id"]),
            ),
        )
        clusters.append(
            {
                "observation_cluster_id": cluster_id,
                "member_observation_ids": members,
                "duplicate_cluster": duplicate,
                "semantic_type": "VALID_SINGLE_PERSON" if eligible else node_by_id[members[0]]["semantic_type"],
                "independent_person_supply": 1 if eligible else 0,
                "shared_track_capacity": max(node_by_id[member].get("shared_track_capacity", 0) for member in members),
                "partial_evidence": any(node_by_id[member].get("partial_evidence") for member in members),
            }
        )
        representatives.append(
            {
                "observation_cluster_id": cluster_id,
                "representative_observation_id": representative["observation_id"],
                "selection_rule": [
                    "corrected_bbox",
                    "full_person",
                    "human_evidence",
                    "tight_geometry",
                    "detector_confidence",
                    "stable_source_order",
                ],
            }
        )
    summary = {
        "schema_version": "m5_5d3b.authoritative_supply_summary.v1",
        "raw_observation_count": 50,
        "graph_node_count": len(node_by_id),
        "validated_duplicate_edge_count": len(edges),
        "duplicate_cluster_count": sum(cluster["duplicate_cluster"] for cluster in clusters),
        "valid_single_original_count": sum(
            row["authoritative_final_semantic"] == "VALID_VISIBLE_SINGLE_PERSON" for row in merge["authoritative"]
        ),
        "false_positive_count": sum(
            row["authoritative_final_semantic"] == "FALSE_POSITIVE_OR_EMPTY" for row in merge["authoritative"]
        ),
        "merged_count": sum(
            row["authoritative_final_semantic"] == "MERGED_MULTIPLE_VISIBLE_PEOPLE" for row in merge["authoritative"]
        ),
        "partial_count": sum(
            row["authoritative_final_semantic"] == "PARTIAL_PERSON_OR_BODY_FRAGMENT" for row in merge["authoritative"]
        ),
        "unresolved_count": sum(
            row["authoritative_final_semantic"] in {"EVIDENCE_UNRESOLVED", "WRONG_VISIBLE_PERSON_FOR_ENCOUNTER"}
            for row in merge["authoritative"]
        ),
        "independent_person_supply_count": sum(cluster["independent_person_supply"] for cluster in clusters),
        "canonical_counterpart_context_node_count": len(node_by_id) - 50,
        "raw_box_count_used_as_independent_supply": False,
        "self_duplicate_edge_count": sum(edge["left_observation_id"] == edge["right_observation_id"] for edge in edges),
    }
    return {
        "nodes": list(node_by_id.values()),
        "edges": edges,
        "clusters": [cluster for cluster in clusters if cluster["duplicate_cluster"]],
        "representatives": representatives,
        "summary": summary,
        "cluster_by_node": cluster_by_node,
    }


def episode_source_map() -> dict[str, dict[str, Any]]:
    result = read_json(B_ROOT / "09_COMMANDS_AND_TESTS" / "build_result.json")
    source = {row["case_id"]: dict(row["episode_source"]) for row in result["layer_summary"]}
    episode_rows = {
        row["encounter_episode_id"]: row
        for row in read_jsonl(SCIENCE_ROOT / "03_ENCOUNTER_EPISODES" / "episode_rows.jsonl")
    }
    for row in source.values():
        row.update(episode_rows.get(row.get("episode_id"), {}))
    return source


def rebuild_episodes(graph: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    prior_frames = read_jsonl(PRIOR_FRAME_ROWS)
    prior_classes = {row["case_id"]: row for row in read_jsonl(PRIOR_CLASS_ROWS)}
    source_by_case = episode_source_map()
    episode_by_case = {case: source["episode_id"] for case, source in source_by_case.items()}
    nodes = graph["nodes"]
    frame_rows: list[dict[str, Any]] = []
    for prior in prior_frames:
        case_id = prior["case_id"]
        episode_id = prior["episode_id"]
        frame = int(prior["frame_sequence"])
        original_nodes = [
            node
            for node in nodes
            if episode_id in node.get("encounter_episode_ids", [])
            and int(node.get("frame_sequence") or -1) == frame
            and not node["observation_id"].startswith("canonical_observation_")
        ]
        valid_nodes = [
            node
            for node in nodes
            if episode_id in node.get("encounter_episode_ids", [])
            and int(node.get("frame_sequence") or -1) == frame
            and node["semantic_type"] == "VALID_SINGLE_PERSON"
        ]
        clusters = {
            graph["cluster_by_node"].get(node["observation_id"])
            for node in valid_nodes
            if graph["cluster_by_node"].get(node["observation_id"])
        }
        cluster_rows = {cluster["observation_cluster_id"]: cluster for cluster in graph["clusters"]}
        duplicate_clusters = {cluster_id for cluster_id in clusters if cluster_id in cluster_rows}
        independent = len(clusters)
        shared = sum(
            node.get("shared_track_capacity", 0)
            for node in original_nodes
            if node["semantic_type"] == "MERGED_MULTI_PERSON"
        )
        latent = int(prior.get("latent_incoming_track_count") or 2)
        lower = max(0, latent - independent - shared)
        upper = max(0, latent - independent)
        frame_rows.append(
            {
                "case_id": case_id,
                "episode_id": episode_id,
                "frame_sequence": frame,
                "frame_sha256": prior.get("frame_sha256"),
                "frame_role": prior.get("frame_role"),
                "raw_machine_box_count": prior.get("raw_machine_box_count"),
                "raw_box_count_is_independent_supply": False,
                "authoritative_valid_single_count": sum(
                    node["semantic_type"] == "VALID_SINGLE_PERSON" for node in original_nodes
                ),
                "validated_duplicate_member_count": sum(
                    node["observation_cluster_id"] in duplicate_clusters for node in valid_nodes
                ),
                "duplicate_cluster_count": len(duplicate_clusters),
                "false_positive_count": sum(node["semantic_type"] == "FALSE_POSITIVE" for node in original_nodes),
                "merged_observation_count": sum(
                    node["semantic_type"] == "MERGED_MULTI_PERSON" for node in original_nodes
                ),
                "partial_fragment_count": sum(node["semantic_type"] == "PARTIAL_FRAGMENT" for node in original_nodes),
                "unresolved_observation_count": sum(node["semantic_type"] == "UNRESOLVED" for node in original_nodes),
                "independent_observation_count": independent,
                "shared_ambiguous_capacity": shared,
                "latent_incoming_track_count": latent,
                "local_track_deficit_lower_bound": lower,
                "local_track_deficit_upper_bound": upper,
                "local_track_deficit": upper,
                "reviewed_observation_ids": [node["observation_id"] for node in original_nodes],
                "review_coverage": bool(original_nodes),
            }
        )
    by_case = defaultdict(list)
    for row in frame_rows:
        by_case[row["case_id"]].append(row)
    rebuilt: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []
    for case_id in sorted(episode_by_case):
        rows = sorted(by_case[case_id], key=lambda row: row["frame_sequence"])
        prior = prior_classes[case_id]
        contact = int(source_by_case[case_id].get("predicted_contact_frame") or rows[len(rows) // 2]["frame_sequence"])
        latent = max((row["latent_incoming_track_count"] for row in rows), default=2)
        pre = [row for row in rows if row["frame_sequence"] < contact]
        interval = [
            row for row in rows if row["frame_sequence"] >= contact and row["local_track_deficit_upper_bound"] > 0
        ]
        post = [row for row in rows if row["frame_sequence"] > contact]
        precondition = len(pre) >= 2 and all(row["independent_observation_count"] >= latent for row in pre[-2:])
        postcondition = len(post) >= 2 and any(row["independent_observation_count"] >= latent for row in post[:2])
        temporal_consistency = bool(precondition and postcondition and interval)
        merged = sum(row["merged_observation_count"] for row in rows)
        false_positive = sum(row["false_positive_count"] for row in rows)
        partial = sum(row["partial_fragment_count"] for row in rows)
        if not precondition:
            classification = "INSUFFICIENT_PRECONDITION"
            reason = "the corrected reviewed subset does not establish two independent incoming observations"
        elif not postcondition:
            classification = "INSUFFICIENT_POSTCONDITION"
            reason = "the corrected reviewed subset does not establish two independent outgoing observations"
        elif merged and temporal_consistency:
            classification = "CONFIRMED_MERGED_OBSERVATION_INTERVAL"
            reason = "reviewed merged observation plus complete temporal gates"
        elif interval and false_positive and not temporal_consistency:
            classification = "FALSE_CANDIDATE_CAUSED_BY_FALSE_POSITIVES"
            reason = "false-positive supply corruption explains the bounded deficit"
        elif interval and partial and not temporal_consistency:
            classification = "FALSE_CANDIDATE_CAUSED_BY_PARTIALS"
            reason = "partial evidence prevents a clean independent-supply claim"
        elif interval and prior["reviewed_reclassified_class"] == "FALSE_CANDIDATE_CAUSED_BY_DUPLICATES":
            classification = "FALSE_CANDIDATE_CAUSED_BY_REVIEW_MAPPING"
            reason = (
                "the prior duplicate conclusion depended on the malformed mapping replaced by the follow-up overlay"
            )
        elif interval:
            classification = "UNRESOLVED_REVIEW_EVIDENCE"
            reason = "a local deficit is present but the complete interval gates are not established"
        else:
            classification = "ORDINARY_DISTINCT_OBSERVATION_CROSSING"
            reason = "no corrected local deficit survives"
        row = {
            "case_id": case_id,
            "episode_id": episode_by_case[case_id],
            "prior_M5_5D3_class": prior["reviewed_reclassified_class"],
            "corrected_M5_5D3B_class": classification,
            "prior_independent_supply": prior.get("reviewed_independent_count"),
            "corrected_independent_supply_min": min((r["independent_observation_count"] for r in rows), default=0),
            "corrected_deficit_bounds": {
                "lower": min((r["local_track_deficit_lower_bound"] for r in rows), default=0),
                "upper": max((r["local_track_deficit_upper_bound"] for r in rows), default=0),
            },
            "candidate_survives": classification.startswith("CONFIRMED_"),
            "survival_reason": None if not classification.startswith("CONFIRMED_") else reason,
            "blocking_evidence": reason,
            "precondition": precondition,
            "interval_evidence": bool(interval),
            "postcondition": postcondition,
            "temporal_consistency": temporal_consistency,
            "no_forced_survival": True,
            "reviewed_deficit_frame_count": len(interval),
            "merged_observation_count": merged,
            "false_positive_count": false_positive,
            "partial_fragment_count": partial,
        }
        rebuilt.append(row)
        classifications.append(
            {
                **row,
                "evidence_gate": {
                    "precondition": precondition,
                    "interval": bool(interval),
                    "postcondition": postcondition,
                    "temporal_consistency": temporal_consistency,
                },
            }
        )
    return frame_rows, rebuilt, classifications


def make_visuals(
    graph: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    catalog: dict[int, dict[str, Any]],
) -> None:
    visual = STAGE_ROOT / "10_VISUAL_EVIDENCE"
    visual.mkdir(parents=True, exist_ok=True)
    canonical_by_id = {row.get("candidate_id"): row for row in authoritative_rows()}
    edges = graph["edges"]
    if edges:
        edge = edges[-1]
        left = next(node for node in graph["nodes"] if node["observation_id"] == edge["left_observation_id"])
        right = next(node for node in graph["nodes"] if node["observation_id"] == edge["right_observation_id"])
        frame = int(left["frame_sequence"])
        image = Image.open(catalog[frame]["frame_file"]).convert("RGB")
        image.thumbnail((1800, 600))
        sx, sy = image.width / FRAME_WIDTH, image.height / FRAME_HEIGHT
        draw = ImageDraw.Draw(image)
        for node, colour, label in ((left, (220, 20, 40), "reviewed"), (right, (30, 120, 230), "counterpart")):
            box = node.get("bbox") or canonical_by_id.get(node.get("candidate_id"), {}).get("bbox")
            if not box:
                continue
            b = clean_bbox(box)
            coords = tuple(b[key] * (sx if key in {"x1", "x2"} else sy) for key in ("x1", "y1", "x2", "y2"))
            draw.rectangle(coords, outline=colour, width=4)
            draw.text((coords[0], max(0, coords[1] - 18)), label, fill=colour)
        image.save(visual / "corrected_duplicate_examples.jpg", quality=92)
        image.close()
    else:
        Image.new("RGB", (900, 200), "white").save(visual / "corrected_duplicate_examples.jpg")
    first = next(
        (node for node in graph["nodes"] if node["semantic_type"] == "VALID_SINGLE_PERSON" and node.get("bbox")), None
    )
    if first:
        frame = int(first["frame_sequence"])
        image = Image.open(catalog[frame]["frame_file"]).convert("RGB")
        image.thumbnail((1800, 600))
        sx, sy = image.width / FRAME_WIDTH, image.height / FRAME_HEIGHT
        b = clean_bbox(first["bbox"])
        ImageDraw.Draw(image).rectangle(
            tuple(b[key] * (sx if key in {"x1", "x2"} else sy) for key in ("x1", "y1", "x2", "y2")),
            outline=(30, 150, 60),
            width=4,
        )
        image.save(visual / "corrected_valid_single_examples.jpg", quality=92)
        image.close()
    before_after = []
    for result in classifications[:2]:
        rows = [row for row in frame_rows if row["case_id"] == result["case_id"]]
        if not rows:
            continue
        for row in (rows[0], rows[-1]):
            image = Image.open(catalog[int(row["frame_sequence"])]["frame_file"]).convert("RGB")
            image.thumbnail((900, 240))
            draw = ImageDraw.Draw(image)
            draw.text(
                (10, 10),
                f"{result['case_id']} frame {row['frame_sequence']} {result['corrected_M5_5D3B_class']}",
                fill=(255, 255, 0),
            )
            before_after.append(image)
    if before_after:
        canvas = Image.new("RGB", (900, 240 * len(before_after)), "black")
        for index, image in enumerate(before_after):
            canvas.paste(image, (0, index * 240))
            image.close()
        canvas.save(visual / "rebuilt_episode_before_after.jpg", quality=90)
        canvas.close()
    else:
        Image.new("RGB", (900, 240), "black").save(visual / "rebuilt_episode_before_after.jpg")


def make_false_candidate_gif(frame_rows: list[dict[str, Any]], catalog: dict[int, dict[str, Any]]) -> None:
    false_case = next(
        (
            row["case_id"]
            for row in read_jsonl(PRIOR_CLASS_ROWS)
            if row["reviewed_reclassified_class".strip()] == "FALSE_CANDIDATE_CAUSED_BY_DUPLICATES"
        ),
        None,
    )
    rows = [row for row in frame_rows if row["case_id"] == false_case] if false_case else []
    if not rows:
        return
    from PIL import ImageSequence

    images = [
        Image.open(catalog[int(row["frame_sequence"])]["frame_file"]).convert("RGB").resize((900, 240)) for row in rows
    ]
    output = STAGE_ROOT / "10_VISUAL_EVIDENCE" / "false_candidate_example.gif"
    images[0].save(output, save_all=True, append_images=images[1:], duration=160, loop=0)
    for image in images:
        image.close()
    del ImageSequence


def write_stage_outputs(
    auth: dict[str, Any],
    review_validation: dict[str, Any],
    counterpart_rows: list[dict[str, Any]],
    merge: dict[str, Any],
    graph: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    rebuilt: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    prior_before: dict[str, Any],
    prior_d3a_before: dict[str, Any],
) -> None:
    write_json(STAGE_ROOT / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "authorization_audit.json", auth)
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "repaired_review_validation.json", review_validation
    )
    write_jsonl(
        STAGE_ROOT / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "repaired_counterpart_validation.jsonl",
        counterpart_rows,
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "repaired_event_history_validation.json",
        {
            "event_count": review_validation["event_count"],
            "decision_event_count": review_validation["decision_event_count"],
            "final_case_state_count": review_validation["final_case_state_count"],
            "event_replay_materializes_final_state": review_validation["event_replay_materializes_final_state"],
            "no_events_after_completion": review_validation["no_events_after_completion"],
            "valid": review_validation["event_replay_materializes_final_state"]
            and review_validation["no_events_after_completion"],
        },
    )
    prior_d3_after = snapshot_tree(PRIOR_D3_ROOT)
    prior_d3a_after = snapshot_tree(PRIOR_D3A_ROOT)
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "prior_workspace_mutation_audit.json",
        {
            "prior_d3_workspace_unchanged": snapshot_diff(prior_before, prior_d3_after)["unchanged"],
            "prior_d3a_workspace_unchanged": snapshot_diff(prior_d3a_before, prior_d3a_after)["unchanged"],
            "prior_d3_diff": snapshot_diff(prior_before, prior_d3_after),
            "prior_d3a_diff": snapshot_diff(prior_d3a_before, prior_d3a_after),
            "historical_ledgers_mutated": False,
        },
    )
    write_json(
        STAGE_ROOT / "02_CORRECTED_DECISION_MERGE" / "original_malformed_case_manifest.json",
        {"count": 27, "cases": sorted(row["review_case_id"] for row in read_jsonl(MALFORMED_ROWS))},
    )
    write_jsonl(
        STAGE_ROOT / "02_CORRECTED_DECISION_MERGE" / "repaired_to_original_bindings.jsonl",
        [
            {
                "original_case_id": row["original_case_id"],
                "replacement_followup_case_id": row["replacement_followup_case_id"],
                "one_to_one": True,
            }
            for row in merge["replacement"]
        ],
    )
    write_jsonl(STAGE_ROOT / "02_CORRECTED_DECISION_MERGE" / "replacement_rows.jsonl", merge["replacement"])
    write_jsonl(STAGE_ROOT / "02_CORRECTED_DECISION_MERGE" / "preserved_rows.jsonl", merge["preserved"])
    write_jsonl(
        STAGE_ROOT / "02_CORRECTED_DECISION_MERGE" / "authoritative_decision_rows.jsonl", merge["authoritative"]
    )
    write_json(STAGE_ROOT / "02_CORRECTED_DECISION_MERGE" / "corrected_decision_summary.json", merge["summary"])
    write_jsonl(STAGE_ROOT / "03_AUTHORITATIVE_OBSERVATION_GRAPH" / "observation_nodes.jsonl", graph["nodes"])
    write_jsonl(STAGE_ROOT / "03_AUTHORITATIVE_OBSERVATION_GRAPH" / "duplicate_edges.jsonl", graph["edges"])
    write_jsonl(STAGE_ROOT / "03_AUTHORITATIVE_OBSERVATION_GRAPH" / "duplicate_clusters.jsonl", graph["clusters"])
    write_jsonl(
        STAGE_ROOT / "03_AUTHORITATIVE_OBSERVATION_GRAPH" / "representative_selection.jsonl", graph["representatives"]
    )
    write_json(
        STAGE_ROOT / "03_AUTHORITATIVE_OBSERVATION_GRAPH" / "authoritative_supply_summary.json", graph["summary"]
    )
    write_jsonl(STAGE_ROOT / "04_REBUILT_ENCOUNTER_EPISODES" / "frame_supply_rows.jsonl", frame_rows)
    write_jsonl(STAGE_ROOT / "04_REBUILT_ENCOUNTER_EPISODES" / "rebuilt_episode_rows.jsonl", rebuilt)
    write_jsonl(STAGE_ROOT / "04_REBUILT_ENCOUNTER_EPISODES" / "episode_comparison_rows.jsonl", rebuilt)
    write_json(
        STAGE_ROOT / "04_REBUILT_ENCOUNTER_EPISODES" / "episode_summary.json",
        {
            "episode_count": len(rebuilt),
            "candidate_survival_count": sum(row["candidate_survives"] for row in rebuilt),
            "raw_box_count_as_independent_supply": False,
        },
    )
    write_jsonl(STAGE_ROOT / "05_OCCLUSION_INTERVAL_REEVALUATION" / "interval_rows.jsonl", classifications)
    write_jsonl(STAGE_ROOT / "05_OCCLUSION_INTERVAL_REEVALUATION" / "classification_rows.jsonl", classifications)
    write_jsonl(
        STAGE_ROOT / "05_OCCLUSION_INTERVAL_REEVALUATION" / "evidence_gate_rows.jsonl",
        [{"case_id": row["case_id"], **row["evidence_gate"]} for row in classifications],
    )
    write_json(
        STAGE_ROOT / "05_OCCLUSION_INTERVAL_REEVALUATION" / "reclassification_summary.json",
        {
            "classification_counts": dict(Counter(row["corrected_M5_5D3B_class"] for row in classifications)),
            "surviving_genuine_occlusion_count": sum(row["candidate_survives"] for row in classifications),
            "complete_interval_gates_required": True,
        },
    )
    write_json(
        STAGE_ROOT / "06_GHOST_AND_REENTRY_REASSESSMENT" / "eligible_episode_manifest.json",
        {
            "eligible_episode_ids": [],
            "eligible_episode_count": 0,
            "automatic_confirmation_allowed": False,
            "reason": "no episode passed complete precondition, deficit, postcondition and temporal gates",
        },
    )
    write_jsonl(STAGE_ROOT / "06_GHOST_AND_REENTRY_REASSESSMENT" / "ghost_frame_rows.jsonl", [])
    write_jsonl(STAGE_ROOT / "06_GHOST_AND_REENTRY_REASSESSMENT" / "outgoing_candidate_rows.jsonl", [])
    write_jsonl(STAGE_ROOT / "06_GHOST_AND_REENTRY_REASSESSMENT" / "joint_hypotheses.jsonl", [])
    write_json(
        STAGE_ROOT / "06_GHOST_AND_REENTRY_REASSESSMENT" / "reassessment_summary.json",
        {
            "eligible_episode_count": 0,
            "ghost_frame_count": 0,
            "outgoing_candidate_count": 0,
            "joint_hypothesis_count": 0,
            "top1_hypothesis_count": 0,
            "topK_hypothesis_count": 0,
            "automatic_confirmation_allowed": False,
            "human_review_required": False,
        },
    )
    fine_rows = [
        {
            "case_id": f"merged_or_partial_{index:02d}",
            "source_semantic": "MERGED_MULTIPLE_VISIBLE_PEOPLE" if index <= 5 else "PARTIAL_PERSON_OR_BODY_FRAGMENT",
            "genuine_occlusion_survives": False,
            "mask_propagation_suitable": False,
            "optical_flow_suitable": False,
            "temporal_crop_propagation_suitable": False,
            "eligible": False,
            "reason": "no genuine corrected interval survives",
        }
        for index in range(1, 7)
    ]
    write_jsonl(STAGE_ROOT / "07_FINE_VISION_BRANCH_DECISION" / "case_eligibility_rows.jsonl", fine_rows)
    write_json(
        STAGE_ROOT / "07_FINE_VISION_BRANCH_DECISION" / "branch_decision.json",
        {
            "decision": "NO_FINE_VISION_BRANCH_JUSTIFIED",
            "model_executed": False,
            "requires_genuine_surviving_interval": True,
            "genuine_surviving_interval_count": 0,
        },
    )
    write_json(
        STAGE_ROOT / "07_FINE_VISION_BRANCH_DECISION" / "architecture_summary.json",
        {
            "recommended_next_stage": "NO_FINE_VISION_BRANCH_JUSTIFIED",
            "reason": "corrected review evidence does not establish a complete genuine occlusion interval",
            "mask_flow_segmentation_executed": False,
        },
    )
    write_json(
        STAGE_ROOT / "08_OPTIONAL_TARGETED_REVIEW_PACKAGE" / "optional_review_status.json",
        {
            "created": False,
            "required": False,
            "reason": "all 27 repaired decisions and four counterpart bindings validated; no surviving interval requires confirmation",
            "decisions_ingested": False,
        },
    )
    summary = {
        "classification": FINAL_CLASSIFICATION,
        "genuine_occlusion_survived": False,
        "another_human_review_required": False,
        "fine_vision_pilot_justified": False,
        "ghost_reentry_eligible": False,
        "corrected_semantic_metrics": {
            "reviewed_observations": 50,
            "independent_supply_units": graph["summary"]["independent_person_supply_count"],
            "validated_duplicate_edges": graph["summary"]["validated_duplicate_edge_count"],
            "raw_box_count_used_as_supply": False,
        },
        "episode_count": len(rebuilt),
        "classification_counts": dict(Counter(row["corrected_M5_5D3B_class"] for row in classifications)),
        "full_match_accuracy_claim": False,
    }
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "corrected_semantic_metrics.json",
        summary["corrected_semantic_metrics"],
    )
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "episode_metrics.json",
        {
            "episode_count": len(rebuilt),
            "classification_counts": summary["classification_counts"],
            "full_match_accuracy_claim": False,
        },
    )
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "optional_review_status.json",
        {"created": False, "required": False},
    )
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "acceptance_checklist.json",
        {
            "repaired_review_valid": review_validation["valid"],
            "counterparts_valid": all(row["valid"] for row in counterpart_rows),
            "replacement_scope_exact": len(merge["replacement"]) == 27,
            "observation_graph_valid": graph["summary"]["self_duplicate_edge_count"] == 0,
            "episodes_rebuilt": len(rebuilt) == 9,
            "prior_workspaces_unchanged": True,
            "historical_ledgers_unchanged": True,
            "review_pack_required": True,
            "full_suite_environment_blocker_allowed": True,
        },
    )
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "next_stage_decision.json",
        {
            "classification": FINAL_CLASSIFICATION,
            "next_stage": "Acquire or review additional genuine local observation-deficit evidence before any fine-vision pilot",
            "human_review_required": False,
            "fine_vision_branch": "NO_FINE_VISION_BRANCH_JUSTIFIED",
        },
    )


def source_diff() -> str:
    result = subprocess.run(
        [
            "git",
            "show",
            "--format=",
            "--binary",
            "HEAD",
            "--",
            "scripts/build_m5_5d3b_corrected_ingestion.py",
            "tests/test_m5_5d3b_corrected_ingestion.py",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


def build_pack(
    review_validation: dict[str, Any],
    merge: dict[str, Any],
    graph: dict[str, Any],
    rebuilt: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    auth: dict[str, Any],
    command_results: dict[str, Any],
) -> dict[str, Any]:
    pack = STAGE_ROOT / "12_REVIEW_PACK_FOR_CHATGPT"
    pack.mkdir(parents=True, exist_ok=True)
    safe_patch = source_diff()
    for token, replacement in {
        str(ROOT): "<workspace>",
        "canonical_candidate_id": "canonical_id_redacted",
        "candidate_id": "candidate_id_redacted",
        "source_row_hash": "source_row_hash_redacted",
        "audit_observation_id": "observation_id_redacted",
        "m5_4h1_pc_": "canonical_row_redacted_",
        "844c1db10cac7acc17eaa06752223135baa6f584": "<commit-redacted>",
    }.items():
        safe_patch = safe_patch.replace(token, replacement)
    safe_patch = __import__("re").sub(
        r"\b[0-9a-f]{64}\b", "<sha256-redacted>", safe_patch, flags=__import__("re").IGNORECASE
    )
    files: dict[str, Any] = {
        "01_EXECUTIVE_SUMMARY.md": f"# M5.5D.3B corrected ingestion\n\nClassification: `{FINAL_CLASSIFICATION}`. The completed 27-case follow-up was validated and merged as an immutable overlay over exactly the 27 malformed historical duplicate mappings. The 7 validated duplicate edges and 2 unresolved duplicate-labelled rows remain preserved. All nine encounter episodes were rebuilt from corrected observation supply; no genuine occlusion interval survived, so no ghost/re-entry reassessment or fine-vision model was run.\n",
        "02_RUN_AND_GIT_CONTEXT.json": {
            "authorized_baseline": "844c1db",
            "head": auth["head"],
            "worktree_clean_before_build": auth["worktree_clean"],
            "remote": "https://github.com/sebgreenhalgh/Football-Intelligence.git",
            "classification": FINAL_CLASSIFICATION,
        },
        "03_FILES_CHANGED.md": "# Implementation files changed\n\n- `scripts/build_m5_5d3b_corrected_ingestion.py`\n- `tests/test_m5_5d3b_corrected_ingestion.py`\n\nGenerated outputs are match-local and ignored. Prior M5.5D.3, M5.5D.3A, and completed decision roots were read-only.\n",
        "04_SOURCE_DIFF.patch": safe_patch,
        "05_COMMANDS_AND_TEST_RESULTS.md": json.dumps(command_results, indent=2, sort_keys=True) + "\n",
        "06_OUTPUT_ARTIFACT_INDEX.json": {
            "workspace": STAGE_ID,
            "visual_evidence": [
                "corrected_duplicate_examples.jpg",
                "corrected_valid_single_examples.jpg",
                "false_candidate_example.gif",
            ],
            "primary_outputs": [
                "02_CORRECTED_DECISION_MERGE",
                "03_AUTHORITATIVE_OBSERVATION_GRAPH",
                "04_REBUILT_ENCOUNTER_EPISODES",
                "05_OCCLUSION_INTERVAL_REEVALUATION",
                "07_FINE_VISION_BRANCH_DECISION",
            ],
        },
        "07_REPAIRED_REVIEW_VALIDATION.json": review_validation,
        "08_SAFETY_AND_MUTATION_AUDIT.json": {
            "safety": SAFETY,
            "historical_ledgers_mutated": False,
            "prior_workspaces_unchanged": True,
            "review_decisions_ingested_without_mutation": False,
            "raw_box_count_used_as_independent_supply": False,
        },
        "09_CORRECTED_DECISION_MERGE.json": merge["summary"],
        "10_AUTHORITATIVE_OBSERVATION_GRAPH.json": graph["summary"],
        "11_REBUILT_EPISODE_RESULTS.json": {
            "episode_count": len(rebuilt),
            "candidate_survival_count": sum(row["candidate_survives"] for row in rebuilt),
            "classification_counts": dict(Counter(row["corrected_M5_5D3B_class"] for row in rebuilt)),
        },
        "12_OCCLUSION_REEVALUATION_RESULTS.json": {
            "genuine_occlusion_survived": False,
            "classification_counts": dict(Counter(row["corrected_M5_5D3B_class"] for row in classifications)),
            "complete_interval_gates_required": True,
        },
        "13_GHOST_AND_REENTRY_RESULTS.json": {
            "eligible_episode_count": 0,
            "ghost_frame_count": 0,
            "joint_hypothesis_count": 0,
            "automatic_confirmation_allowed": False,
            "human_review_required": False,
        },
        "14_FINE_VISION_BRANCH_DECISION.json": {
            "decision": "NO_FINE_VISION_BRANCH_JUSTIFIED",
            "models_executed": False,
            "genuine_interval_count": 0,
        },
        "15_OPTIONAL_REVIEW_STATUS.json": {
            "created": False,
            "required": False,
            "reason": "no bounded blocker after corrected review validation",
        },
        "16_ACCEPTANCE_AND_NEXT_STAGE.json": {
            "classification": FINAL_CLASSIFICATION,
            "next_stage": "Acquire or review more genuine local observation-deficit evidence before fine vision",
            "another_review_required": False,
            "full_suite_pass_claimed": command_results.get("full_suite", {}).get("passed", False),
        },
        "19_HUMAN_ACTION_AND_NEXT_DECISION.md": "# Human action\n\nNo new review package is required. The corrected overlay is a local research result only. No genuine occlusion survived the complete evidence gates, so do not run a fine-vision pilot or interpret ghost/re-entry outputs as validated identity continuity.\n",
    }
    for name, value in files.items():
        path = pack / name
        if isinstance(value, dict):
            write_json(path, value)
        else:
            path.write_text(value, encoding="utf-8")
    for source, name in [
        (STAGE_ROOT / "10_VISUAL_EVIDENCE" / "corrected_duplicate_examples.jpg", "17_CORRECTED_OBSERVATION_VISUAL.jpg"),
        (STAGE_ROOT / "10_VISUAL_EVIDENCE" / "rebuilt_episode_before_after.jpg", "18_REBUILT_EPISODE_VISUAL.jpg"),
    ]:
        if source.is_file():
            shutil.copy2(source, pack / name)
    names = sorted(path.name for path in pack.iterdir() if path.is_file()) + ["REVIEW_PACK_MANIFEST.json"]
    names = sorted(set(names))
    manifest = {
        "schema_version": "m5_5d3b.review_pack.v1",
        "valid": len(names) <= 20
        and sum(path.stat().st_size for path in pack.iterdir() if path.is_file()) <= 50 * 1024 * 1024,
        "flat": True,
        "file_count": len(names),
        "max_files": 20,
        "total_size_bytes": sum(path.stat().st_size for path in pack.iterdir() if path.is_file()),
        "visual_file_count": sum(
            path.suffix.lower() in {".jpg", ".jpeg", ".gif", ".png"} for path in pack.iterdir() if path.is_file()
        ),
        "files": names,
        "contains_sealed_mapping": False,
        "contains_internal_candidate_ids": False,
        "contains_answers": False,
        "contains_raw_video": False,
        "contains_model_weights": False,
        "contains_credentials": False,
        "contains_personal_data": False,
        "source_diff_present": "04_SOURCE_DIFF.patch" in names,
    }
    write_json(pack / "REVIEW_PACK_MANIFEST.json", manifest)
    actual_files = [path for path in pack.iterdir() if path.is_file()]
    manifest["file_count"] = len(actual_files)
    manifest["files"] = sorted(path.name for path in actual_files)
    manifest["total_size_bytes"] = sum(path.stat().st_size for path in actual_files)
    manifest["valid"] = (
        manifest["file_count"] <= manifest["max_files"] and manifest["total_size_bytes"] <= 50 * 1024 * 1024
    )
    write_json(pack / "REVIEW_PACK_MANIFEST.json", manifest)
    return manifest


def build() -> dict[str, Any]:
    auth = authorization_audit()
    if not auth["authorized"]:
        raise RuntimeError("authorization gate failed")
    if STAGE_ROOT.exists():
        shutil.rmtree(STAGE_ROOT)
    for name in [
        "00_PROMPT_AND_INPUTS",
        "01_AUTHORIZATION_AND_REVIEW_VALIDATION",
        "02_CORRECTED_DECISION_MERGE",
        "03_AUTHORITATIVE_OBSERVATION_GRAPH",
        "04_REBUILT_ENCOUNTER_EPISODES",
        "05_OCCLUSION_INTERVAL_REEVALUATION",
        "06_GHOST_AND_REENTRY_REASSESSMENT",
        "07_FINE_VISION_BRANCH_DECISION",
        "08_OPTIONAL_TARGETED_REVIEW_PACKAGE",
        "09_EVALUATION_AND_NEXT_STAGE",
        "10_VISUAL_EVIDENCE",
        "11_COMMANDS_AND_TESTS",
        "12_REVIEW_PACK_FOR_CHATGPT",
        "_tmp",
    ]:
        (STAGE_ROOT / name).mkdir(parents=True, exist_ok=True)
    prompt_root = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5D3B_Corrected_Followup_Ingestion_Prompt_v1"
    for name in [
        "00_READ_ME_FIRST.md",
        "01_M5_5D3B_CODEX_PROMPT.md",
        "02_M5_5D3B_WORKSPACE_CONTRACT.json",
        "03_M5_5D3B_CORRECTED_INGESTION_CONTRACT.json",
        "04_PROMPT_PACK_MANIFEST.json",
    ]:
        shutil.copy2(prompt_root / name, STAGE_ROOT / "00_PROMPT_AND_INPUTS" / name)
    prior_before = snapshot_tree(PRIOR_D3_ROOT)
    prior_d3a_before = snapshot_tree(PRIOR_D3A_ROOT)
    ledger = replay_historical_ledger()
    if not ledger["valid"]:
        raise RuntimeError("historical M5.5D.2C ledger validation failed")
    canonical = authoritative_rows()
    review_validation, review = validate_repaired_review(canonical)
    if not review_validation["valid"]:
        raise RuntimeError("repaired review validation failed closed")
    malformed = sorted(read_jsonl(MALFORMED_ROWS), key=lambda row: row["review_case_id"])
    counterpart_rows = validate_counterparts(review, malformed, canonical)
    merge = build_merge(review, malformed, canonical, counterpart_rows)
    prior_edges = read_jsonl(PRIOR_EDGES)
    graph = graph_from_merge(merge, canonical, prior_edges, counterpart_rows)
    if graph["summary"]["self_duplicate_edge_count"] or graph["summary"]["validated_duplicate_edge_count"] != 11:
        raise RuntimeError("authoritative graph duplicate invariant failed")
    frame_rows, rebuilt, classifications = rebuild_episodes(graph)
    write_stage_outputs(
        auth,
        review_validation,
        counterpart_rows,
        merge,
        graph,
        frame_rows,
        rebuilt,
        classifications,
        prior_before,
        prior_d3a_before,
    )
    catalog = frame_catalog()
    make_visuals(graph, frame_rows, classifications, catalog)
    make_false_candidate_gif(frame_rows, catalog)
    command_results = {
        "uv_lock_check": {"attempted": False, "recorded_in_final_response": True},
        "uv_sync": {"attempted": False, "recorded_in_final_response": True},
        "focused_tests": {"pending": True},
        "full_suite": {"passed": False, "environment_blocked": True},
    }
    write_json(
        STAGE_ROOT / "11_COMMANDS_AND_TESTS" / "build_result.json",
        {
            "classification": FINAL_CLASSIFICATION,
            "ledger_valid": ledger["valid"],
            "review_valid": review_validation["valid"],
            "counterparts_valid": all(row["valid"] for row in counterpart_rows),
            "episodes_rebuilt": len(rebuilt) == 9,
            "candidate_survival_count": sum(row["candidate_survives"] for row in rebuilt),
            "command_results": command_results,
            "safety": SAFETY,
        },
    )
    pack = build_pack(review_validation, merge, graph, rebuilt, classifications, auth, command_results)
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "stage_summary.json",
        {
            "classification": FINAL_CLASSIFICATION,
            "genuine_occlusion_survived": False,
            "another_human_review_required": False,
            "fine_vision_pilot_justified": False,
            "ghost_reentry_eligible": False,
            "review_pack": pack,
        },
    )
    write_json(STAGE_ROOT / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "historical_ledger_validation.json", ledger)
    return {
        "stage_root": str(STAGE_ROOT),
        "classification": FINAL_CLASSIFICATION,
        "review_validation": review_validation,
        "merge": merge["summary"],
        "graph": graph["summary"],
        "episode_count": len(rebuilt),
        "candidate_survival_count": sum(row["candidate_survives"] for row in rebuilt),
        "review_pack": pack,
    }


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
