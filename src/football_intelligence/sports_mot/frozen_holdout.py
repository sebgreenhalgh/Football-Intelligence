"""One-time sealed-holdout governance for the frozen P-MHSAG candidate.

The module deliberately contains no tracker implementation.  It binds an
already-frozen candidate to a preregistration, grants one semantic access
session, and commits the first valid result transaction immutably.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from football_intelligence.review_chassis.hashing import sha256_file, stable_hash


class HoldoutGovernanceError(RuntimeError):
    """Raised when a sealed-holdout governance invariant is violated."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HoldoutGovernanceError(f"expected JSON object: {path}")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True).encode() + b"\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))


def write_json_exclusive(path: Path, value: Any) -> None:
    """Create a JSON artifact exactly once and durably close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise HoldoutGovernanceError(f"immutable artifact already exists: {path}") from exc
    try:
        payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True).encode() + b"\n"
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def file_hash_rows(paths: Iterable[Path], *, relative_to: Path | None = None) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(paths, key=lambda value: str(value).lower()):
        if not path.is_file():
            raise HoldoutGovernanceError(f"required file is missing: {path}")
        display = path.relative_to(relative_to).as_posix() if relative_to else str(path)
        rows.append({"path": display, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def validate_preregistration(
    preregistration: dict[str, Any],
    *,
    expected_candidate_manifest_hash: str,
    expected_configuration_hash: str,
) -> dict[str, Any]:
    required = {
        "frozen_candidate_manifest_hash",
        "exact_execution_command",
        "oracle_mode_command",
        "detector_mode_command",
        "legacy_focal_supplementary_command",
        "holdout_sequence_count",
        "expected_output_schemas",
        "machine_hard_gates",
        "failure_attribution_rules",
        "one_time_access_policy",
        "same_config_retry_policy",
        "pre_registered_shadow_stress_matrix",
        "conditional_visual_audit_policy",
        "no_retune_statement",
        "configuration_hash",
        "execution_harness_source_hashes",
    }
    missing = sorted(required - preregistration.keys())
    errors = []
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    if preregistration.get("frozen_candidate_manifest_hash") != expected_candidate_manifest_hash:
        errors.append("candidate manifest hash mismatch")
    if preregistration.get("configuration_hash") != expected_configuration_hash:
        errors.append("configuration hash mismatch")
    if preregistration.get("holdout_sequence_count") != 8:
        errors.append("holdout sequence count must remain opaque value 8")
    if preregistration.get("no_retune_statement") is not True:
        errors.append("no-retune statement is not affirmative")
    return {
        "passed": not errors,
        "errors": errors,
        "required_field_count": len(required),
        "pre_registration_hash": stable_hash(preregistration),
    }


@dataclass(frozen=True)
class OneTimeSemanticAccessController:
    """Gate semantic holdout access behind an exclusive unseal event."""

    event_path: Path
    access_state_path: Path

    @property
    def unseal_count(self) -> int:
        return int(self.event_path.is_file())

    def require_sealed(self) -> None:
        if self.unseal_count != 0:
            raise HoldoutGovernanceError("semantic holdout has already been unsealed")

    def reject_pre_unseal_access(self, resolver: Callable[[], Any]) -> dict[str, Any]:
        """Prove a resolver cannot execute while the event is absent."""
        try:
            self.require_authorized_session()
            resolver()
        except HoldoutGovernanceError as exc:
            return {"passed": True, "resolver_called": False, "error": str(exc)}
        raise HoldoutGovernanceError("pre-unseal semantic resolver unexpectedly executed")

    def require_authorized_session(self) -> dict[str, Any]:
        if not self.event_path.is_file():
            raise HoldoutGovernanceError("semantic holdout access requires HOLDOUT_UNSEALED event")
        event = read_json(self.event_path)
        if event.get("event_type") != "HOLDOUT_UNSEALED" or event.get("unseal_count_after") != 1:
            raise HoldoutGovernanceError("invalid holdout unseal event")
        return event

    def unseal(
        self,
        *,
        preregistration: dict[str, Any],
        preregistration_hash: str,
        candidate_manifest: dict[str, Any],
        candidate_manifest_hash: str,
        sealed_manifest_hash: str,
        sealed_container_hash: str,
        actor: dict[str, Any],
        resolver: Callable[[], Any],
    ) -> Any:
        self.require_sealed()
        if stable_hash(preregistration) != preregistration_hash:
            raise HoldoutGovernanceError("pre-registration hash mismatch")
        if stable_hash(candidate_manifest) != candidate_manifest_hash:
            raise HoldoutGovernanceError("frozen candidate manifest hash mismatch")
        validation = validate_preregistration(
            preregistration,
            expected_candidate_manifest_hash=candidate_manifest_hash,
            expected_configuration_hash=str(candidate_manifest.get("configuration_hash")),
        )
        if not validation["passed"]:
            raise HoldoutGovernanceError("invalid pre-registration: " + "; ".join(validation["errors"]))
        event = {
            "schema_version": "football_intelligence.m5_5f1d.holdout_unseal_event.v1",
            "event_type": "HOLDOUT_UNSEALED",
            "occurred_at": utc_now(),
            "unseal_count_before": 0,
            "unseal_count_after": 1,
            "pre_registration_hash": preregistration_hash,
            "frozen_candidate_manifest_hash": candidate_manifest_hash,
            "configuration_hash": candidate_manifest["configuration_hash"],
            "candidate_source_commit": candidate_manifest["candidate_source_commit"],
            "sealed_manifest_hash": sealed_manifest_hash,
            "sealed_container_hash": sealed_container_hash,
            "actor": actor,
            "single_semantic_access_session": True,
            "retuning_after_unseal_forbidden": True,
        }
        event["event_hash"] = stable_hash(event)
        # The event is the semantic point of no return.  It is durable before
        # the resolver can read a label, image binding, or sequence ID.
        write_json_exclusive(self.event_path, event)
        write_json_exclusive(
            self.access_state_path,
            {
                "schema_version": "football_intelligence.m5_5f1d.holdout_access_state.v1",
                "unseal_count": 1,
                "event_hash": event["event_hash"],
                "semantic_access_authorized": True,
            },
        )
        return resolver()


def compare_determinism_runs(runs: list[dict[str, Any]], *, tolerance: float = 1e-6) -> dict[str, Any]:
    if len(runs) < 2:
        raise HoldoutGovernanceError("at least two determinism runs are required")
    reference = runs[0]
    discrete_keys = (
        "strand_states",
        "observation_source_choices",
        "error_attribution",
        "fully_exact_sequences",
        "graph_hashes",
        "descriptor_cache_hash",
        "configuration_hash",
    )
    mismatches = []
    for index, run in enumerate(runs[1:], start=2):
        for key in discrete_keys:
            if run.get(key) != reference.get(key):
                mismatches.append({"run": index, "field": key})
    reference_costs = [float(value) for value in reference.get("joint_path_costs", [])]
    max_delta = 0.0
    for run in runs[1:]:
        costs = [float(value) for value in run.get("joint_path_costs", [])]
        if len(costs) != len(reference_costs):
            mismatches.append({"run": runs.index(run) + 1, "field": "joint_path_cost_count"})
            continue
        max_delta = max([max_delta, *(abs(a - b) for a, b in zip(reference_costs, costs))])
    passed = not mismatches and max_delta <= tolerance
    return {
        "schema_version": "football_intelligence.m5_5f1d.development_canary_comparison.v1",
        "run_count": len(runs),
        "discrete_fields_compared": list(discrete_keys),
        "mismatches": mismatches,
        "maximum_floating_cost_delta": max_delta,
        "floating_cost_tolerance": tolerance,
        "passed": passed,
    }


def evaluate_machine_gates(oracle: dict[str, Any], detector: dict[str, Any]) -> dict[str, Any]:
    om = oracle["metrics"]
    dm = detector["metrics"]
    oracle_checks = {
        "eight_of_eight_exact": om.get("fully_exact_sequences") == 8,
        "identity_switches_zero": om.get("identity_switches") == 0,
        "false_continuations_zero": om.get("false_continuations") == 0,
        "losses_zero": om.get("strand_losses_when_supply_available") == 0,
        "double_assignments_zero": om.get("double_assignments") == 0,
        "provenance_failures_zero": om.get("provenance_failures") == 0,
    }
    nonexact = int(dm.get("sequence_count", 0)) - int(dm.get("fully_exact_sequences", 0))
    disallowed_nonexact = [
        row
        for row in detector.get("frame_attribution_rows", [])
        if row.get("outcome") not in {"CORRECT_CONTINUATION", "SAFE_ABSTENTION", "SAFE_ABSTENTION_NO_SUPPLY"}
    ]
    detector_checks = {
        "identity_switches_zero": dm.get("identity_switches") == 0,
        "false_continuations_zero": dm.get("false_continuations") == 0,
        "losses_with_supply_zero": dm.get("strand_losses_when_supply_available") == 0,
        "off_pitch_assignments_zero": dm.get("off_pitch_assignments") == 0,
        "double_assignments_zero": dm.get("double_assignments") == 0,
        "provenance_failures_zero": dm.get("provenance_failures") == 0,
        "at_least_seven_exact": int(dm.get("fully_exact_sequences", 0)) >= 7,
        "at_most_one_nonexact": nonexact <= 1,
        "only_safe_nonexact_reason": not disallowed_nonexact,
    }
    passed = all(oracle_checks.values()) and all(detector_checks.values())
    return {
        "schema_version": "football_intelligence.m5_5f1d.machine_gate_checklist.v1",
        "oracle_checks": oracle_checks,
        "detector_checks": detector_checks,
        "detector_nonexact_sequence_count": nonexact,
        "disallowed_nonexact_rows": disallowed_nonexact,
        "passed": passed,
    }


@dataclass(frozen=True)
class ImmutablePrimaryResultTransaction:
    root: Path

    @property
    def transaction_path(self) -> Path:
        return self.root / "primary_result_transaction.json"

    def commit(self, artifacts: dict[str, dict[str, Any]], *, context: dict[str, Any]) -> dict[str, Any]:
        if self.transaction_path.exists():
            raise HoldoutGovernanceError("a scientifically valid primary result already exists")
        expected = {
            "oracle_holdout_results.json",
            "detector_holdout_results.json",
            "legacy_focal_holdout_results.json",
        }
        if set(artifacts) != expected:
            raise HoldoutGovernanceError("primary result transaction artifact set is incomplete")
        rows = []
        for name in sorted(artifacts):
            path = self.root / name
            write_json_exclusive(path, artifacts[name])
            rows.append({"path": name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
        transaction = {
            "schema_version": "football_intelligence.m5_5f1d.primary_result_transaction.v1",
            "committed_at": utc_now(),
            "status": "IMMUTABLE_FIRST_VALID_PRIMARY_RESULT",
            "artifacts": rows,
            "context": context,
            "scientific_underperformance_retry_allowed": False,
        }
        transaction["transaction_hash"] = stable_hash(transaction)
        write_json_exclusive(self.transaction_path, transaction)
        return transaction

    def validate(self) -> dict[str, Any]:
        if not self.transaction_path.is_file():
            return {"passed": False, "errors": ["transaction missing"]}
        transaction = read_json(self.transaction_path)
        errors = []
        for row in transaction.get("artifacts", []):
            path = self.root / str(row["path"])
            if not path.is_file() or sha256_file(path) != row["sha256"]:
                errors.append(f"artifact mismatch: {row['path']}")
        expected_hash = stable_hash({key: value for key, value in transaction.items() if key != "transaction_hash"})
        if transaction.get("transaction_hash") != expected_hash:
            errors.append("transaction hash mismatch")
        return {"passed": not errors, "errors": errors, "transaction_hash": transaction.get("transaction_hash")}


def retry_policy(reason: str, *, sequence_score_committed: bool, valid_result_exists: bool) -> dict[str, Any]:
    permitted_reasons = {"CUDA_OOM", "PROCESS_CRASH", "CORRUPT_OUTPUT_WRITE"}
    allowed = reason in permitted_reasons and not sequence_score_committed and not valid_result_exists
    return {
        "reason": reason,
        "same_config_retry_allowed": allowed,
        "scientific_underperformance_retry_allowed": False,
        "candidate_or_parameter_change_allowed": False,
    }
