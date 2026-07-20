"""Post-result forensics and fresh challenge-benchmark governance.

The spent-holdout helpers in this module are deliberately unable to execute a
tracker.  Gold labels are accepted only by the post-inference invariant audit.
Fresh holdout split access is governed separately so the old spent dataset can
never be mistaken for a reusable validation set.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from football_intelligence.review_chassis.hashing import sha256_file, stable_hash


class SpentHoldoutExecutionError(RuntimeError):
    """Raised before any second scientific execution can start."""


class FreshHoldoutAccessError(RuntimeError):
    """Raised when a fresh sealed split is requested before future unsealing."""


@dataclass(frozen=True)
class SpentResultGuard:
    """Validate an immutable result transaction and reject every re-execution."""

    transaction_path: Path
    expected_transaction_hash: str

    def audit(self) -> dict[str, Any]:
        import json

        transaction = json.loads(self.transaction_path.read_text(encoding="utf-8"))
        recorded_hash = str(transaction.get("transaction_hash", ""))
        immutable = transaction.get("status") == "IMMUTABLE_FIRST_VALID_PRIMARY_RESULT"
        passed = immutable and recorded_hash == self.expected_transaction_hash
        if not passed:
            raise SpentHoldoutExecutionError("spent result transaction failed immutable validation")
        return {
            "schema_version": "football_intelligence.m5_5f1e.spent_result_guard.v1",
            "transaction_path": str(self.transaction_path),
            "transaction_file_sha256": sha256_file(self.transaction_path),
            "transaction_hash": recorded_hash,
            "immutable_first_result": True,
            "scientific_execution_allowed": False,
            "alternate_candidate_scoring_allowed": False,
            "parameter_selection_allowed": False,
            "passed": True,
        }

    def block_scientific_execution(self, operation: str = "spent_holdout_evaluation") -> None:
        self.audit()
        raise SpentHoldoutExecutionError(f"blocked second scientific execution: {operation}")


def contiguous_failure_events(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_fields: Sequence[str] = ("sequence_id", "strand"),
    frame_field: str = "frame_sequence",
) -> list[dict[str, Any]]:
    """Collapse frame-level failures into deterministic contiguous events."""

    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (*[str(row.get(field, "")) for field in group_fields], int(row[frame_field])),
    )
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        grouped[tuple(str(row.get(field, "")) for field in group_fields)].append(row)

    events: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        current: list[dict[str, Any]] = []
        for row in group:
            frame = int(row[frame_field])
            if current and frame != int(current[-1][frame_field]) + 1:
                events.append(_event_row(key, group_fields, current, frame_field))
                current = []
            current.append(row)
        if current:
            events.append(_event_row(key, group_fields, current, frame_field))
    for index, event in enumerate(events, 1):
        event["event_id"] = f"failure_event_{index:03d}"
    return events


def _event_row(
    key: tuple[str, ...],
    group_fields: Sequence[str],
    rows: list[dict[str, Any]],
    frame_field: str,
) -> dict[str, Any]:
    start = int(rows[0][frame_field])
    end = int(rows[-1][frame_field])
    return {
        **dict(zip(group_fields, key, strict=True)),
        "first_failure_frame": start,
        "last_failure_frame": end,
        "duration_frames": end - start + 1,
        "frame_count": len(rows),
        "frame_sequences": [int(row[frame_field]) for row in rows],
        "frame_row_hashes": [stable_hash(row) for row in rows],
    }


def audit_oracle_reachability(
    *,
    graph: Mapping[str, Any],
    gold_paths: Mapping[str, Sequence[str | None]],
    selected_states: Sequence[Mapping[str, Any]],
    global_links: Sequence[Mapping[str, Any]],
    micro_tracklets: Sequence[Mapping[str, Any]],
    purity_splits: Sequence[Mapping[str, Any]],
    authoritative_path_source: str,
    renderer_rows: Sequence[Mapping[str, Any]] | None = None,
    visibility_justifications: Mapping[tuple[str, int], str] | None = None,
    selected_transition_costs: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assert structural oracle properties after inference has completed.

    The caller supplies gold paths after inference.  This function never calls
    an association algorithm and never returns a candidate ranking.
    """

    frames = [int(frame) for frame in graph["allowed_frames"]]
    if any(len(path) != len(frames) for path in gold_paths.values()):
        raise ValueError("gold paths must align exactly with graph frames")
    nodes = {str(row["node_id"]): dict(row) for row in graph["nodes"]}
    feasible_edges = {
        (str(row["source_node_id"]), str(row["target_node_id"])): dict(row)
        for row in graph["edges"]
        if row.get("hard_gate_pass")
    }
    link_edges = {(str(row["source_node_id"]), str(row["target_node_id"])): dict(row) for row in global_links}
    membership = {
        str(node_id): str(tracklet["tracklet_id"])
        for tracklet in micro_tracklets
        for node_id in tracklet.get("node_ids", [])
    }
    state_by_frame = {int(row["frame_sequence"]): row for row in selected_states}
    renderer = {(str(row.get("strand")), int(row.get("frame_sequence", -1))): row for row in (renderer_rows or [])}
    visibility_justifications = visibility_justifications or {}
    selected_transition_costs = selected_transition_costs or {}

    transition_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    one_to_one_rows: list[dict[str, Any]] = []
    materialization_rows: list[dict[str, Any]] = []
    for strand, path in sorted(gold_paths.items()):
        previous: str | None = None
        for index, node_id in enumerate(path):
            frame = frames[index]
            if node_id is not None and str(node_id) not in nodes:
                raise ValueError(f"gold node is absent from graph: {node_id}")
            if previous is not None and node_id is not None:
                edge = feasible_edges.get((previous, str(node_id)))
                link = link_edges.get((previous, str(node_id)))
                cost_row = selected_transition_costs.get((strand, frame), {})
                transition_rows.append(
                    {
                        "strand": strand,
                        "source_frame_sequence": int(nodes[previous]["frame_sequence"]),
                        "target_frame_sequence": frame,
                        "source_node_id": previous,
                        "target_node_id": str(node_id),
                        "hard_graph_edge_reachable": edge is not None,
                        "global_link_reachable": link is not None,
                        "source_tracklet_id": membership.get(previous),
                        "target_tracklet_id": membership.get(str(node_id)),
                        "gold_link_cost": link.get("link_cost") if link else None,
                        "selected_transition_cost": cost_row.get("selected_cost"),
                        "null_state_cost": cost_row.get("null_cost"),
                        "selected_state": cost_row.get("selected_state"),
                    }
                )
            selected = state_by_frame.get(frame, {}).get(strand, {})
            if node_id is not None and not selected.get("node_id"):
                justification = visibility_justifications.get((strand, frame))
                null_rows.append(
                    {
                        "strand": strand,
                        "frame_sequence": frame,
                        "gold_node_id": str(node_id),
                        "selected_state": selected.get("state"),
                        "visibility_justification": justification,
                        "unjustified_null": justification is None,
                    }
                )
            if selected.get("node_id"):
                key = (strand, frame)
                rendered_node = renderer.get(key, {}).get("node_id") if renderer else selected.get("node_id")
                materialization_rows.append(
                    {
                        "strand": strand,
                        "frame_sequence": frame,
                        "selected_node_id": selected.get("node_id"),
                        "renderer_node_id": rendered_node,
                        "materialized": rendered_node == selected.get("node_id"),
                    }
                )
            previous = str(node_id) if node_id is not None else previous

    for index, frame in enumerate(frames):
        left = gold_paths.get("A", [None] * len(frames))[index]
        right = gold_paths.get("B", [None] * len(frames))[index]
        one_to_one_rows.append(
            {
                "frame_sequence": frame,
                "A_node_id": left,
                "B_node_id": right,
                "distinct_when_both_visible": left is None or right is None or left != right,
            }
        )

    path_reachable = all(row["hard_graph_edge_reachable"] for row in transition_rows)
    global_reachable = all(row["global_link_reachable"] for row in transition_rows)
    forward_reachable = _reachable_gold_nodes(transition_rows, direction="forward")
    backward_reachable = _reachable_gold_nodes(transition_rows, direction="backward")
    bidirectional = path_reachable and forward_reachable and backward_reachable
    split_orphaned = any(
        row["hard_graph_edge_reachable"] and not row["global_link_reachable"] for row in transition_rows
    ) and bool(purity_splits)
    global_consumed = authoritative_path_source == "POST_PURITY_JOINT_TRACKLET_DAG"
    invariant_rows = [
        _invariant("GOLD_PATH_REACHABLE", path_reachable, transition_rows),
        _invariant("SEED_TO_END_REACHABILITY", path_reachable, transition_rows),
        _invariant("BIDIRECTIONAL_REACHABILITY", bidirectional, transition_rows),
        _invariant(
            "NULL_NOT_PREFERRED_WHEN_UNAMBIGUOUS",
            not any(row["unjustified_null"] for row in null_rows),
            null_rows,
        ),
        _invariant("PURITY_SPLIT_CANNOT_ORPHAN_GOLD", not split_orphaned, transition_rows),
        _invariant("GLOBAL_LINKER_CONSUMED", global_consumed and global_reachable, transition_rows),
        _invariant(
            "ONE_TO_ONE_GOLD_PAIR",
            all(row["distinct_when_both_visible"] for row in one_to_one_rows),
            one_to_one_rows,
        ),
        _invariant(
            "STATE_MATERIALIZATION_COMPLETE",
            all(row["materialized"] for row in materialization_rows),
            materialization_rows,
        ),
    ]
    return {
        "schema_version": "football_intelligence.m5_5f1e.oracle_invariant_result.v1",
        "sequence_id": graph.get("sequence_id"),
        "graph_hash": graph.get("graph_hash"),
        "post_inference_assertion_only": True,
        "tracker_called": False,
        "gold_supplied_to_tracker": False,
        "transition_rows": transition_rows,
        "null_preference_rows": null_rows,
        "forward_gold_reachability": forward_reachable,
        "backward_gold_reachability": backward_reachable,
        "invariants": invariant_rows,
        "all_passed": all(row["passed"] for row in invariant_rows),
    }


def _reachable_gold_nodes(rows: Sequence[Mapping[str, Any]], *, direction: str) -> bool:
    """Check the ordered gold chain in either direction without running inference."""

    if direction not in {"forward", "backward"}:
        raise ValueError("direction must be forward or backward")
    ordered = sorted(
        rows,
        key=lambda row: (str(row["strand"]), int(row["target_frame_sequence"])),
        reverse=direction == "backward",
    )
    return all(bool(row.get("hard_graph_edge_reachable")) for row in ordered)


def _invariant(invariant_id: str, passed: bool, evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "invariant_id": invariant_id,
        "passed": bool(passed),
        "evidence_count": len(evidence),
        "failing_evidence_count": sum(
            1
            for row in evidence
            if row.get("hard_graph_edge_reachable") is False
            or row.get("global_link_reachable") is False
            or row.get("unjustified_null") is True
            or row.get("distinct_when_both_visible") is False
            or row.get("materialized") is False
        ),
    }


def preflight_challenge_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Apply non-semantic, machine-only eligibility gates to a fresh sequence."""

    reasons = []
    frame_count = len(candidate.get("frames", []))
    if not 13 <= frame_count <= 19:
        reasons.append("FRAME_COUNT_OUT_OF_RANGE")
    if len(set(candidate.get("seed_observation_ids", []))) != 2:
        reasons.append("TWO_DISTINCT_SEEDS_REQUIRED")
    if not candidate.get("seeds_on_pitch", False):
        reasons.append("ON_PITCH_SEED_GATE_FAILED")
    if not candidate.get("full_panorama_evidence", False):
        reasons.append("FULL_PANORAMA_EVIDENCE_MISSING")
    if not candidate.get("source_provenance_complete", False):
        reasons.append("SOURCE_PROVENANCE_INCOMPLETE")
    if candidate.get("true_occlusion_suspected", False):
        reasons.append("TRUE_OCCLUSION_EXCLUDED")
    if candidate.get("prior_window_overlap", False):
        reasons.append("PRIOR_REVIEW_WINDOW_OVERLAP")
    if candidate.get("event_cluster_duplicate", False):
        reasons.append("DUPLICATE_EVENT_CLUSTER")
    if candidate.get("evidence_route_failure", False):
        reasons.append("EVIDENCE_ROUTE_FAILURE")
    return {
        "candidate_key": candidate.get("candidate_key"),
        "passed": not reasons,
        "rejection_reasons": reasons,
        "machine_only": True,
        "human_label_inferred": False,
    }


def assign_hidden_splits(
    candidates: Sequence[Mapping[str, Any]],
    *,
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Deterministically assign event-cluster-disjoint hidden splits."""

    count = len(candidates)
    if count < 24:
        raise ValueError("fresh challenge benchmark requires at least 24 candidates")
    target = (
        {"challenge_development": 16, "challenge_validation": 8, "new_sealed_holdout": 8}
        if count >= 32
        else {
            "challenge_development": count - 2 * max(6, round(count * 0.25)),
            "challenge_validation": max(6, round(count * 0.25)),
            "new_sealed_holdout": max(6, round(count * 0.25)),
        }
    )
    ordered = sorted(
        (dict(row) for row in candidates),
        key=lambda row: stable_hash(
            {
                "seed": seed,
                "event_cluster_id": row["event_cluster_id"],
                "challenge_tags": sorted(row.get("challenge_tags", [])),
            }
        ),
    )
    remaining = dict(target)
    tag_counts: dict[str, Counter[str]] = {name: Counter() for name in target}
    assignments = []
    for row in ordered:
        tags = tuple(sorted(str(value) for value in row.get("challenge_tags", [])))
        eligible = [name for name, capacity in remaining.items() if capacity > 0]
        split = min(
            eligible,
            key=lambda name: (
                sum(tag_counts[name][tag] for tag in tags),
                -remaining[name] / target[name],
                stable_hash({"seed": seed, "candidate": row["candidate_key"], "split": name}),
            ),
        )
        assignments.append(
            {
                "candidate_key": row["candidate_key"],
                "event_cluster_id": row["event_cluster_id"],
                "hidden_split": split,
                "assignment_hash": stable_hash(
                    {"seed": seed, "candidate_key": row["candidate_key"], "hidden_split": split}
                ),
            }
        )
        remaining[split] -= 1
        tag_counts[split].update(tags)
    return assignments, target


@dataclass(frozen=True)
class FreshHoldoutResolver:
    """Reject access until a future frozen candidate has a valid unseal grant."""

    sealed_manifest_path: Path
    access_state_path: Path

    def require_future_unseal(self, *, frozen_candidate_hash: str | None, unseal_grant: str | None) -> None:
        import json

        state = json.loads(self.access_state_path.read_text(encoding="utf-8"))
        if (
            not frozen_candidate_hash
            or not unseal_grant
            or not state.get("future_unseal_authorized", False)
            or state.get("unseal_count", 0) != 1
            or state.get("frozen_candidate_hash") != frozen_candidate_hash
            or state.get("unseal_grant") != unseal_grant
        ):
            raise FreshHoldoutAccessError("new sealed holdout remains inaccessible before a future freeze and unseal")

    def negative_audit(self) -> dict[str, Any]:
        blocked = 0
        for candidate, grant in ((None, None), ("candidate", None), (None, "grant"), ("candidate", "grant")):
            try:
                self.require_future_unseal(frozen_candidate_hash=candidate, unseal_grant=grant)
            except FreshHoldoutAccessError:
                blocked += 1
        return {
            "schema_version": "football_intelligence.m5_5f1e.fresh_holdout_negative_access.v1",
            "attempt_count": 4,
            "blocked_count": blocked,
            "sealed_manifest_sha256": sha256_file(self.sealed_manifest_path),
            "semantic_content_returned": False,
            "passed": blocked == 4,
        }
