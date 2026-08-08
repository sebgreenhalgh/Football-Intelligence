"""G7E-B deterministic tranche assignment and temporal-review persistence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import mimetypes
import os
import shutil
import tempfile
import threading
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qs, unquote, urlparse

from football_intelligence.g7e_b_r5_reviewer_state import (
    R5_CONTRACT_ID,
    R5_EVENT_SCHEMA,
    R5_REVIEW_REVISION,
    R5_WORKING_DRAFT_SCHEMA,
    compile_final_event,
    initialize_working_draft,
    load_contract,
    validate_working_draft,
)
from football_intelligence.g7e_b_r6_action_reducer import (
    ACTION_TYPES as R6_ACTION_TYPES,
    R6_ACTION_RECEIPT_SCHEMA,
    R6_CONTRACT_NAME,
    R6_REVIEW_REVISION,
    apply_action as apply_r6_action,
    compile_final_event as compile_r6_final_event,
    initialize_r6_draft,
)
from football_intelligence.temporal_burst_selection import CLASS_PRIORITY, MATCHES, QUOTAS
from football_intelligence.temporal_reviewer.contracts import contained_path, validate_action_envelope
from football_intelligence.temporal_reviewer.http_server import (
    RequestBodyTooLarge,
    UnsupportedMediaType,
    read_json_request,
)
from football_intelligence.temporal_reviewer.persistence import (
    ActionTransaction,
    recover_action_transactions,
    replace_with_retry,
)

REVIEW_ID = "G7E_B_TEMPORAL_BURST_REVIEW"
REVIEW_REVISION = "G7E_B_TEMPORAL_BURST_REVIEW_V1"
R1_REVIEW_REVISION = "G7E_B_R1_SUBJECT_GUIDANCE_AND_ZOOM_REPAIR_V1"
R2_REVIEW_REVISION = "G7E_B_R2_FULL_TEMPORAL_CANDIDATE_CLOSURE_V1"
R3_REVIEW_REVISION = "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_V1"
R4_REVIEW_REVISION = "G7E_B_R4_CANDIDATE_RELATIONSHIP_BRANCH_INTEGRITY_V1"
SUPPORTED_REVIEW_REVISIONS = (
    REVIEW_REVISION,
    R1_REVIEW_REVISION,
    R2_REVIEW_REVISION,
    R3_REVIEW_REVISION,
    R4_REVIEW_REVISION,
    R5_REVIEW_REVISION,
    R6_REVIEW_REVISION,
)
PROTOCOL_ID = "G7E_A_BURST_LOCAL_TEMPORAL_OBSERVATION_PROTOCOL_V1"
R6_3_RELEASE_REVISION = "G7E_B_R6_3_FAST_ACTION_AND_STALE_DRAFT_RECOVERY_V1"
TRANCHES = tuple(f"TRANCHE_{index}" for index in range(1, 7))

MATCH_ROTATION: dict[str, dict[str, int]] = {
    "TRANCHE_1": {"117092": 4, "117093": 4, "118575": 3, "118576": 3, "118577": 3, "128058": 3},
    "TRANCHE_2": {"117092": 3, "117093": 3, "118575": 4, "118576": 4, "118577": 3, "128058": 3},
    "TRANCHE_3": {"117092": 3, "117093": 3, "118575": 3, "118576": 3, "118577": 4, "128058": 4},
    "TRANCHE_4": {"117092": 4, "117093": 3, "118575": 4, "118576": 3, "118577": 3, "128058": 3},
    "TRANCHE_5": {"117092": 3, "117093": 4, "118575": 3, "118576": 3, "118577": 4, "128058": 3},
    "TRANCHE_6": {"117092": 3, "117093": 3, "118575": 3, "118576": 4, "118577": 3, "128058": 4},
}

VISIBILITY = (
    "VISIBLE_COMPLETE",
    "VISIBLE_PARTIAL",
    "FULLY_OCCLUDED_EXPECTED_PRESENT",
    "OUT_OF_FRAME_OR_LEFT_SCENE",
    "NOT_PRESENT",
    "UNCERTAIN",
)
SUPPLY = (
    "ONE_USEFUL_CANDIDATE",
    "MULTIPLE_CANDIDATES",
    "MERGED_WITH_OTHER_PEOPLE",
    "FRAGMENT_ONLY",
    "NO_CANDIDATE",
    "NOT_APPLICABLE",
    "UNCERTAIN",
)
RELATIONSHIPS = (
    "SAME_PERSON_DUPLICATES",
    "SAME_PERSON_FRAGMENTS",
    "DIFFERENT_PEOPLE",
    "CORRECT_INNER_BAD_OUTER",
    "MERGED_MULTI_PERSON",
    "OBJECT_OR_BACKGROUND",
    "SUBJECT_BODY_FRAGMENT",
    "UNCERTAIN",
)
OCCLUSION_PHASES = ("NONE", "ENTERING_OCCLUSION", "OCCLUDED", "EXITING_OCCLUSION", "UNCERTAIN")
CONTINUITY = ("SAME_BURST_LOCAL_SUBJECT", "DIFFERENT_SUBJECT", "CANNOT_TELL", "NOT_APPLICABLE")
ROLES = ("OUTFIELD_PLAYER", "GOALKEEPER", "RELEVANT_OFFICIAL", "OTHER_PERSON", "UNKNOWN_ROLE")
PARTICIPATION = ("ACTIVE_IN_MATCH", "WARMING_OR_INACTIVE", "NOT_PLAYER_OR_OFFICIAL", "UNKNOWN_PARTICIPATION")
CERTAINTY = ("CERTAIN", "PROBABLE", "NOT_SURE")
SUBJECT_TOKENS = ("SUBJECT_A", "SUBJECT_B", "SUBJECT_C")
FRAME_LOCAL_ACTION_TYPES = (
    "SUBJECT_LOCATION",
    "APPROXIMATE_HIDDEN_LOCATION",
    "CANDIDATE_SELECTION",
    "MISSED_PERSON_MARK",
)


class ReviewValidationError(ValueError):
    """Structured, field-level validation failure safe for the review UI."""

    def __init__(self, errors: list[dict[str, Any]], error_code: str = "FRAME_BINDING_VALIDATION_FAILED"):
        if not errors:
            raise ValueError("structured validation requires at least one error")
        self.errors = errors
        self.error_code = error_code
        super().__init__(str(errors[0]["message"]))


class StaleDraftError(ReviewValidationError):
    """Strict optimistic-concurrency rejection carrying canonical state for resync."""

    def __init__(self, error_code: str, draft: Mapping[str, Any]):
        self.canonical_draft = copy.deepcopy(dict(draft))
        super().__init__(
            [
                {
                    "error_code": error_code,
                    "field": "draft",
                    "message": f"{error_code}: the canonical server draft was returned.",
                }
            ],
            error_code,
        )


class InterruptedAcknowledgement(RuntimeError):
    """Acceptance-only interruption after event persistence and before acknowledgement."""

    def __init__(self, event_id: str):
        self.event_id = event_id
        super().__init__("event persisted; acknowledgement intentionally interrupted")


def relevant_visibility_for_supply(value: Any) -> bool:
    return value in ("VISIBLE_COMPLETE", "VISIBLE_PARTIAL", "FULLY_OCCLUDED_EXPECTED_PRESENT", "UNCERTAIN")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_digest(value: Any) -> str:
    packed = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _perfect_matching(remaining: Mapping[str, Mapping[str, int]]) -> dict[str, str]:
    result: dict[str, str] = {}

    def search(index: int, used: set[str]) -> bool:
        if index == len(TRANCHES):
            return True
        tranche_id = TRANCHES[index]
        options = sorted(
            (match_id for match_id in MATCHES if remaining[tranche_id][match_id] > 0 and match_id not in used),
            key=lambda match_id: (-remaining[tranche_id][match_id], match_id),
        )
        for match_id in options:
            result[tranche_id] = match_id
            if search(index + 1, used | {match_id}):
                return True
        result.pop(tranche_id, None)
        return False

    if not search(0, set()):
        raise ValueError("match rotation cannot be decomposed into deterministic perfect matchings")
    return result


def _matching_decomposition() -> list[dict[str, str]]:
    remaining = {tranche: dict(counts) for tranche, counts in MATCH_ROTATION.items()}
    matchings: list[dict[str, str]] = []
    for _ in range(20):
        matching = _perfect_matching(remaining)
        matchings.append(dict(matching))
        for tranche_id, match_id in matching.items():
            remaining[tranche_id][match_id] -= 1
    if any(value for counts in remaining.values() for value in counts.values()):
        raise ValueError("match rotation decomposition left unmatched capacity")
    return matchings


def _balance_errors(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    by_tranche: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_tranche[str(row["tranche_id"])].append(row)
    errors: list[str] = []
    for tranche_id in TRANCHES:
        tranche_rows = by_tranche[tranche_id]
        classes = Counter(str(row["primary_selection_class"]) for row in tranche_rows)
        matches = Counter(str(row["match_id"]) for row in tranche_rows)
        halves = Counter(str(row["half"]) for row in tranche_rows)
        perspectives = Counter(str(row["perspective_band"]) for row in tranche_rows)
        if len(tranche_rows) != 20:
            errors.append(f"{tranche_id}:count={len(tranche_rows)}")
        if classes != Counter(QUOTAS):
            errors.append(f"{tranche_id}:classes={dict(classes)}")
        if matches != Counter(MATCH_ROTATION[tranche_id]):
            errors.append(f"{tranche_id}:matches={dict(matches)}")
        if halves["FIRST_HALF"] < 8 or halves["SECOND_HALF"] < 8:
            errors.append(f"{tranche_id}:halves={dict(halves)}")
        if perspectives["FAR"] < 5 or perspectives["NEAR_MIDDLE"] < 5:
            errors.append(f"{tranche_id}:perspectives={dict(perspectives)}")
        if matches["117092"] < 1:
            errors.append(f"{tranche_id}:low_light")
    tranche_one = by_tranche["TRANCHE_1"]
    tags = {tag for row in tranche_one for tag in row.get("secondary_evidence_tags", [])}
    for tag in ("NESTED_MUST_PROTECT", "HUMAN_SAFE_FRAGMENT", "MISSED_PERSON_MARK"):
        if tag not in tags:
            errors.append(f"TRANCHE_1:missing_seed={tag}")
    return errors


def deterministic_tranche_assignment(bursts: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Assign the exact frozen bursts with a deterministic constrained hash ranking."""
    rows = [dict(row) for row in bursts]
    if len(rows) != 120 or Counter(str(row["match_id"]) for row in rows) != Counter({match: 20 for match in MATCHES}):
        raise ValueError("frozen burst closure mismatch")
    matchings = _matching_decomposition()
    class_units = [selection_class for selection_class in CLASS_PRIORITY for _ in range(QUOTAS[selection_class])]
    if len(class_units) != len(matchings):
        raise AssertionError("class unit decomposition mismatch")
    occurrence: dict[tuple[str, str], list[str]] = defaultdict(list)
    for matching, selection_class in zip(matchings, class_units, strict=True):
        for tranche_id, match_id in matching.items():
            occurrence[(match_id, selection_class)].append(tranche_id)
    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pools[(str(row["match_id"]), str(row["primary_selection_class"]))].append(row)
    for key, pool in pools.items():
        if len(pool) != len(occurrence[key]):
            raise ValueError(f"class/match pool mismatch: {key}")

    for attempt in range(10000):
        assigned: list[dict[str, Any]] = []
        for key in sorted(pools):
            ranked = sorted(
                pools[key],
                key=lambda row: (
                    hashlib.sha256(f"G7E_B|{attempt}|{row['burst_id']}".encode()).hexdigest(),
                    row["burst_id"],
                ),
            )
            for tranche_id, row in zip(sorted(occurrence[key]), ranked, strict=True):
                assigned.append({**row, "tranche_id": tranche_id})
        if not _balance_errors(assigned):
            ordered: list[dict[str, Any]] = []
            for tranche_id in TRANCHES:
                tranche_rows = [row for row in assigned if row["tranche_id"] == tranche_id]
                tranche_rows.sort(
                    key=lambda row: (
                        hashlib.sha256(f"{tranche_id}|ORDER|{row['burst_id']}".encode()).hexdigest(),
                        row["burst_id"],
                    )
                )
                for position, row in enumerate(tranche_rows, start=1):
                    ordered.append({**row, "tranche_position": position, "assignment_attempt": attempt})
            return ordered
    raise ValueError("no deterministic tranche assignment satisfied all frozen constraints")


def validate_tranche_assignment(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(rows)
    errors = _balance_errors(records)
    if len(records) != 120 or len({str(row["burst_id"]) for row in records}) != 120:
        errors.append("burst uniqueness")
    positions = Counter((str(row["tranche_id"]), int(row["tranche_position"])) for row in records)
    if len(positions) != 120 or any(count != 1 for count in positions.values()):
        errors.append("tranche positions")
    return {"valid": not errors, "errors": errors}


def source_to_display(
    source_xy: tuple[float, float],
    source_size: tuple[float, float],
    viewport_size: tuple[float, float],
    zoom: float = 1.0,
    pan: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    source_width, source_height = source_size
    viewport_width, viewport_height = viewport_size
    scale = min(viewport_width / source_width, viewport_height / source_height) * zoom
    left = (viewport_width - source_width * scale) / 2 + pan[0]
    top = (viewport_height - source_height * scale) / 2 + pan[1]
    return left + source_xy[0] * scale, top + source_xy[1] * scale


def display_to_source(
    display_xy: tuple[float, float],
    source_size: tuple[float, float],
    viewport_size: tuple[float, float],
    zoom: float = 1.0,
    pan: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    source_width, source_height = source_size
    viewport_width, viewport_height = viewport_size
    scale = min(viewport_width / source_width, viewport_height / source_height) * zoom
    left = (viewport_width - source_width * scale) / 2 + pan[0]
    top = (viewport_height - source_height * scale) / 2 + pan[1]
    return (display_xy[0] - left) / scale, (display_xy[1] - top) / scale


def _latest_sequence(paths: Iterable[Path]) -> int:
    highest = 0
    for path in paths:
        try:
            highest = max(highest, int(read_json(path).get("server_sequence", 0)))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return highest


class TemporalReviewStore:
    """Server-backed drafts and immutable burst/tranche/global receipts."""

    def __init__(
        self,
        package: Path,
        decisions_root: Path | None = None,
        practice_root: Path | None = None,
        acceptance_mode: bool = False,
        action_transaction_fail_after: str | None = None,
    ):
        self.package = package.resolve()
        self.decisions = (decisions_root or package / "human_decisions").resolve()
        self.practice = (practice_root or package / "practice_decisions").resolve()
        self.acceptance_mode = acceptance_mode
        self.action_transaction_fail_after = action_transaction_fail_after
        self.lock = threading.RLock()
        self.action_recovery: dict[str, dict[str, int]] = {}
        build_manifest_path = self.package / "build_manifest.json"
        self.release_revision = (
            read_json(build_manifest_path).get("release_revision") if build_manifest_path.is_file() else None
        )
        review = read_json(self.package / "review_cases.json")
        practice = read_json(self.package / "practice_cases.json")
        self.review_id = str(review.get("review_id", REVIEW_ID))
        self.review_revision = str(review.get("review_revision", REVIEW_REVISION))
        if self.review_id != REVIEW_ID or self.review_revision not in SUPPORTED_REVIEW_REVISIONS:
            raise ValueError("unsupported temporal-review identity or revision")
        self.cases = list(review["cases"])
        self.practice_cases = list(practice["cases"])
        candidate_state_path = self.package / "candidate_states_by_reference.json"
        candidate_state_rows = read_json(candidate_state_path) if candidate_state_path.is_file() else {"frames": {}}
        self.candidate_states_by_reference = dict(candidate_state_rows.get("frames", {}))
        relationship_path = self.package / "relationship_compatibility.json"
        self.relationship_contract = read_json(relationship_path) if relationship_path.is_file() else None
        self.canonical_contract: dict[str, Any] | None = None
        self.canonical_contract_sha256: str | None = None
        self.action_contract: dict[str, Any] | None = None
        self.action_contract_sha256: str | None = None
        if self.review_revision in (R5_REVIEW_REVISION, R6_REVIEW_REVISION):
            self.canonical_contract, self.canonical_contract_sha256 = load_contract(
                self.package / "canonical_reviewer_state_contract.json"
            )
            self.relationship_contract = self.canonical_contract["relationship_compatibility"]
        if self.review_revision == R6_REVIEW_REVISION:
            action_contract_path = self.package / "server_action_contract.json"
            self.action_contract = read_json(action_contract_path)
            self.action_contract_sha256 = sha256_file(action_contract_path)
            if self.action_contract.get("contract_name") != R6_CONTRACT_NAME:
                raise ValueError("R6 server-action contract identity mismatch")
            if self.action_contract.get("review_revision") != R6_REVIEW_REVISION:
                raise ValueError("R6 server-action contract revision mismatch")
        if self.review_revision == R4_REVIEW_REVISION:
            if not isinstance(self.relationship_contract, Mapping):
                raise ValueError("R4 relationship-compatibility contract is missing")
            if self.relationship_contract.get("review_revision") != R4_REVIEW_REVISION:
                raise ValueError("R4 relationship-compatibility revision mismatch")
            self.relationship_compatibility_sha256 = sha256_file(relationship_path)
        elif self.review_revision in (R5_REVIEW_REVISION, R6_REVIEW_REVISION):
            self.relationship_compatibility_sha256 = self.canonical_contract_sha256
        else:
            self.relationship_compatibility_sha256 = None
        self.by_id = {case["burst_id"]: case for case in self.cases}
        self.practice_by_id = {case["burst_id"]: case for case in self.practice_cases}
        self.by_tranche = {
            tranche_id: sorted(
                (case for case in self.cases if case["tranche_id"] == tranche_id),
                key=lambda case: case["tranche_position"],
            )
            for tranche_id in TRANCHES
        }
        if len(self.cases) != 120 or any(len(cases) != 20 for cases in self.by_tranche.values()):
            raise ValueError("review-case cardinality mismatch")
        if len(self.practice_cases) != 3:
            raise ValueError("practice-case cardinality mismatch")
        self.tranche_manifest_sha256 = sha256_file(self.package / "tranche_manifest.jsonl")
        self._cached_release_gate_status = self._verify_release_gate()
        if (
            not self.acceptance_mode
            and self.review_revision in (R5_REVIEW_REVISION, R6_REVIEW_REVISION)
            and self._cached_release_gate_status.get("valid") is not True
        ):
            failures = ",".join(self._cached_release_gate_status.get("failures", []))
            raise RuntimeError(f"REAL_REVIEW_RELEASE_GATE_INVALID_AT_STARTUP:{failures}")
        self.action_recovery = {
            "real": recover_action_transactions(self.decisions),
            "practice": recover_action_transactions(self.practice),
        }

    def _verify_release_gate(self) -> dict[str, Any]:
        if self.review_revision == R6_REVIEW_REVISION:
            if self.release_revision == R6_3_RELEASE_REVISION:
                gate_path = self.package / "G7E_B_R6_3_REAL_REVIEW_RELEASE_GATE.json"
                failures: list[str] = []
                gate: Mapping[str, Any] = {}
                if not gate_path.is_file():
                    failures.append("R6_3_RELEASE_GATE_MISSING")
                else:
                    gate = read_json(gate_path)
                    if gate.get("schema_version") != "football_intelligence.g7e_b_r6_3.real_review_release_gate.v1":
                        failures.append("R6_3_RELEASE_GATE_SCHEMA_MISMATCH")
                    if (
                        gate.get("release_classification")
                        != "PASS_G7E_B_R6_3_FAST_ACTION_AND_STALE_RECOVERY_READY_FOR_TRANCHE_1_RESUME"
                    ):
                        failures.append("R6_3_RELEASE_CLASSIFICATION_NOT_PASSED")
                    if gate.get("review_protocol_revision") != R6_REVIEW_REVISION:
                        failures.append("R6_3_REVIEW_PROTOCOL_REVISION_MISMATCH")
                    if gate.get("action_contract_sha256") != self.action_contract_sha256:
                        failures.append("R6_3_ACTION_CONTRACT_HASH_MISMATCH")
                    expected_files = gate.get("reviewer_file_sha256", {})
                    if not isinstance(expected_files, Mapping) or not expected_files:
                        failures.append("R6_3_RELEASE_REVIEWER_HASHES_MISSING")
                    else:
                        for relative, expected in expected_files.items():
                            path = self.package / str(relative)
                            if not path.is_file() or sha256_file(path) != expected:
                                failures.append(f"R6_3_RELEASE_REVIEWER_HASH_MISMATCH:{relative}")
                return {
                    "required": True,
                    "valid": not failures,
                    "failures": failures,
                    "release_classification": gate.get("release_classification"),
                    "gate_path": str(gate_path),
                    "gate_sha256": sha256_file(gate_path) if gate_path.is_file() else None,
                    "production_ready": False,
                }
            if self.release_revision == "G7E_B_R6_2_PRECISION_ZOOM_PAN_COORDINATE_SAFE_MARKING_V1":
                gate_path = self.package / "G7E_B_R6_2_REAL_REVIEW_RELEASE_GATE.json"
                failures: list[str] = []
                gate: Mapping[str, Any] = {}
                if not gate_path.is_file():
                    failures.append("R6_2_RELEASE_GATE_MISSING")
                else:
                    gate = read_json(gate_path)
                    if gate.get("schema_version") != "football_intelligence.g7e_b_r6_2.real_review_release_gate.v1":
                        failures.append("R6_2_RELEASE_GATE_SCHEMA_MISMATCH")
                    if (
                        gate.get("release_classification")
                        != "PASS_G7E_B_R6_2_PRECISION_ZOOM_PAN_READY_FOR_TRANCHE_1_RESUME"
                    ):
                        failures.append("R6_2_RELEASE_CLASSIFICATION_NOT_PASSED")
                    if gate.get("review_protocol_revision") != R6_REVIEW_REVISION:
                        failures.append("R6_2_REVIEW_PROTOCOL_REVISION_MISMATCH")
                    if gate.get("action_contract_sha256") != self.action_contract_sha256:
                        failures.append("R6_2_ACTION_CONTRACT_HASH_MISMATCH")
                    expected_files = gate.get("reviewer_file_sha256", {})
                    if not isinstance(expected_files, Mapping) or not expected_files:
                        failures.append("R6_2_RELEASE_REVIEWER_HASHES_MISSING")
                    else:
                        for relative, expected in expected_files.items():
                            path = self.package / str(relative)
                            if not path.is_file() or sha256_file(path) != expected:
                                failures.append(f"R6_2_RELEASE_REVIEWER_HASH_MISMATCH:{relative}")
                return {
                    "required": True,
                    "valid": not failures,
                    "failures": failures,
                    "release_classification": gate.get("release_classification"),
                    "gate_path": str(gate_path),
                    "gate_sha256": sha256_file(gate_path) if gate_path.is_file() else None,
                    "production_ready": False,
                }
            if self.release_revision == "G7E_B_R6_1_FINAL_BYTE_VISUAL_RUNTIME_CLOSURE_V1":
                gate_path = self.package / "G7E_B_R6_1_REAL_REVIEW_RELEASE_GATE.json"
                failures: list[str] = []
                gate: Mapping[str, Any] = {}
                if not gate_path.is_file():
                    failures.append("R6_1_RELEASE_GATE_MISSING")
                else:
                    gate = read_json(gate_path)
                    if gate.get("schema_version") != "football_intelligence.g7e_b_r6_1.real_review_release_gate.v1":
                        failures.append("R6_1_RELEASE_GATE_SCHEMA_MISMATCH")
                    if (
                        gate.get("release_classification")
                        != "PASS_G7E_B_R6_1_FINAL_BYTE_VISUAL_AND_RUNTIME_CLOSURE_READY_FOR_TRANCHE_1_RESUME"
                    ):
                        failures.append("R6_1_RELEASE_CLASSIFICATION_NOT_PASSED")
                    if gate.get("review_protocol_revision") != R6_REVIEW_REVISION:
                        failures.append("R6_1_REVIEW_PROTOCOL_REVISION_MISMATCH")
                    if gate.get("action_contract_sha256") != self.action_contract_sha256:
                        failures.append("R6_1_ACTION_CONTRACT_HASH_MISMATCH")
                    expected_files = gate.get("reviewer_file_sha256", {})
                    if not isinstance(expected_files, Mapping) or not expected_files:
                        failures.append("R6_1_RELEASE_REVIEWER_HASHES_MISSING")
                    else:
                        for relative, expected in expected_files.items():
                            path = self.package / str(relative)
                            if not path.is_file() or sha256_file(path) != expected:
                                failures.append(f"R6_1_RELEASE_REVIEWER_HASH_MISMATCH:{relative}")
                return {
                    "required": True,
                    "valid": not failures,
                    "failures": failures,
                    "release_classification": gate.get("release_classification"),
                    "gate_path": str(gate_path),
                    "gate_sha256": sha256_file(gate_path) if gate_path.is_file() else None,
                    "production_ready": False,
                }
            gate_path = self.package / "G7E_B_R6_REAL_REVIEW_RELEASE_GATE.json"
            failures: list[str] = []
            gate: Mapping[str, Any] = {}
            if not gate_path.is_file():
                failures.append("R6_RELEASE_GATE_MISSING")
            else:
                gate = read_json(gate_path)
                if gate.get("schema_version") != "football_intelligence.g7e_b_r6.real_review_release_gate.v1":
                    failures.append("R6_RELEASE_GATE_SCHEMA_MISMATCH")
                if (
                    gate.get("release_classification")
                    != "PASS_G7E_B_R6_SERVER_AUTHORITATIVE_REVIEWER_READY_FOR_FAILED_BURST_RESUME"
                ):
                    failures.append("R6_RELEASE_CLASSIFICATION_NOT_PASSED")
                if gate.get("review_revision") != R6_REVIEW_REVISION:
                    failures.append("R6_RELEASE_REVIEW_REVISION_MISMATCH")
                if gate.get("server_action_contract_sha256") != self.action_contract_sha256:
                    failures.append("R6_ACTION_CONTRACT_HASH_MISMATCH")
                expected_files = gate.get("reviewer_file_sha256", {})
                if not isinstance(expected_files, Mapping) or not expected_files:
                    failures.append("R6_RELEASE_REVIEWER_HASHES_MISSING")
                else:
                    for relative, expected in expected_files.items():
                        path = self.package / str(relative)
                        if not path.is_file() or sha256_file(path) != expected:
                            failures.append(f"R6_RELEASE_REVIEWER_HASH_MISMATCH:{relative}")
            return {
                "required": True,
                "valid": not failures,
                "failures": failures,
                "release_classification": gate.get("release_classification"),
                "gate_path": str(gate_path),
                "gate_sha256": sha256_file(gate_path) if gate_path.is_file() else None,
                "r5_release_gate_revoked": bool(gate.get("r5_release_gate_revocation_sha256")),
                "production_ready": False,
            }
        if self.review_revision != R5_REVIEW_REVISION:
            return {"required": False, "valid": True, "release_classification": None}
        gate_path = self.package / "G7E_B_R5_REAL_REVIEW_RELEASE_GATE.json"
        failures: list[str] = []
        gate: Mapping[str, Any] = {}
        if not gate_path.is_file():
            failures.append("RELEASE_GATE_MISSING")
        else:
            gate = read_json(gate_path)
            if gate.get("schema_version") != "football_intelligence.g7e_b_r5.real_review_release_gate.v1":
                failures.append("RELEASE_GATE_SCHEMA_MISMATCH")
            if (
                gate.get("release_classification")
                != "PASS_G7E_B_R5_REVIEWER_RELEASE_CANDIDATE_READY_FOR_REAL_TRANCHE_RESUME"
            ):
                failures.append("RELEASE_CLASSIFICATION_NOT_PASSED")
            if gate.get("canonical_contract_id") != R5_CONTRACT_ID:
                failures.append("RELEASE_CONTRACT_ID_MISMATCH")
            if gate.get("canonical_contract_sha256") != self.canonical_contract_sha256:
                failures.append("RELEASE_CONTRACT_HASH_MISMATCH")
            if gate.get("review_revision") != R5_REVIEW_REVISION:
                failures.append("RELEASE_REVIEW_REVISION_MISMATCH")
            if gate.get("production_ready") is not False:
                failures.append("RELEASE_PRODUCTION_FLAG_INVALID")
            if gate.get("corpus", {}).get("bursts") != 120 or gate.get("corpus", {}).get("frame_references") != 1080:
                failures.append("RELEASE_CORPUS_CARDINALITY_MISMATCH")
            expected_files = gate.get("reviewer_file_sha256", {})
            if not isinstance(expected_files, Mapping) or not expected_files:
                failures.append("RELEASE_REVIEWER_HASHES_MISSING")
            else:
                for relative, expected in expected_files.items():
                    path = self.package / str(relative)
                    if not path.is_file() or sha256_file(path) != expected:
                        failures.append(f"RELEASE_REVIEWER_HASH_MISMATCH:{relative}")
        return {
            "required": True,
            "valid": not failures,
            "failures": failures,
            "release_classification": gate.get("release_classification"),
            "gate_path": str(gate_path),
            "gate_sha256": sha256_file(gate_path) if gate_path.is_file() else None,
            "production_ready": False,
        }

    def r5_release_gate_status(self) -> dict[str, Any]:
        """Return the immutable startup verdict without rehashing the package."""

        return copy.deepcopy(self._cached_release_gate_status)

    def _require_r5_real_release(self, mode: str) -> None:
        if (
            self.review_revision not in (R5_REVIEW_REVISION, R6_REVIEW_REVISION)
            or mode != "real"
            or self.acceptance_mode
        ):
            return
        status = self.r5_release_gate_status()
        if status["valid"] is not True:
            raise ReviewValidationError(
                [
                    self._structured_error(
                        code="REAL_REVIEW_TEMPORARILY_LOCKED",
                        field="release_gate",
                        message=(
                            "Real review is temporarily locked. Completed progress remains read-only; "
                            "practice and temporary acceptance are still available."
                        ),
                        question_id="original_focus",
                    )
                ],
                "REAL_REVIEW_TEMPORARILY_LOCKED",
            )

    @staticmethod
    def _frame_identity(case: Mapping[str, Any], sequence: int) -> dict[str, Any]:
        frame = case["frames"][sequence]
        identity = frame.get("canonical_frame_identity")
        if not isinstance(identity, Mapping):
            raise ValueError("canonical frame identity is unavailable")
        return dict(identity)

    @staticmethod
    def _structured_error(
        *,
        code: str,
        field: str,
        message: str,
        subject_token: str | None = None,
        question_id: str | None = None,
        expected: Mapping[str, Any] | None = None,
        observed: Mapping[str, Any] | None = None,
        array_location: str | None = None,
    ) -> dict[str, Any]:
        return {
            "error_code": code,
            "field": field,
            "subject_token": subject_token,
            "question_id": question_id,
            "observation_frame_id": expected.get("frame_id") if expected else None,
            "record_frame_id": observed.get("frame_id") if observed else None,
            "unique_frame_id": expected.get("unique_frame_id") if expected else None,
            "array_location": array_location,
            "message": message,
            "suggested_correction_route": question_id,
        }

    def relationship_compatibility(
        self,
        *,
        case: Mapping[str, Any],
        subject_token: str,
        subject_index: int,
        sequence: int,
        observation: Mapping[str, Any],
        final: bool,
    ) -> dict[str, Any]:
        """Evaluate one frame against the package's canonical relationship matrix."""
        if self.review_revision not in (R4_REVIEW_REVISION, R5_REVIEW_REVISION) or not isinstance(
            self.relationship_contract, Mapping
        ):
            raise ValueError("relationship compatibility is unavailable")
        aliases = self.relationship_contract.get("legacy_supply_aliases", {})
        raw_supply = observation.get("observation_supply")
        supply = aliases.get(raw_supply, raw_supply)
        states = self.relationship_contract["supply_states"]
        state = states.get(supply)
        expected = self._frame_identity(case, sequence)
        supply_question = f"subject_{subject_index}_supply_{sequence}"
        relationship_question = f"subject_{subject_index}_relationship_{sequence}"
        selected = observation.get("selected_candidate_ids", [])
        selected = list(selected) if isinstance(selected, list) else []
        relationship = observation.get("candidate_relationship")
        errors: list[dict[str, Any]] = []

        def add_error(code: str, field: str, message: str, route: str) -> None:
            errors.append(
                {
                    **self._structured_error(
                        code=code,
                        field=field,
                        message=message,
                        subject_token=subject_token,
                        question_id=route,
                        expected=expected,
                        array_location=f"subjects[{subject_index}].frame_observations[{sequence}]",
                    ),
                    "candidate_supply_state": supply,
                    "selected_candidate_count": len(selected),
                    "selected_candidate_ids": selected,
                    "stored_relationship": relationship,
                    "allowed_relationships": (
                        self.relationship_contract["question_families"]
                        .get(state.get("question_family") if state else "", {})
                        .get("allowed_relationships", [])
                    ),
                    "correction_route": route,
                }
            )

        if state is None:
            add_error(
                "UNKNOWN_CANDIDATE_SUPPLY_STATE",
                "observation_supply",
                f"{raw_supply!r} is not a supported candidate-supply answer.",
                supply_question,
            )
            return {
                "supply_state": supply,
                "selected_candidate_count": len(selected),
                "selected_candidate_ids": selected,
                "relationship_applicable": False,
                "question_family": None,
                "relationship_question_id": None,
                "allowed_relationships": [],
                "next_valid_question": supply_question,
                "errors": errors,
            }

        minimum = int(state["minimum_selected_count"])
        maximum = state.get("maximum_selected_count")
        if maximum is not None and len(selected) > int(maximum):
            add_error(
                "CANDIDATE_CARDINALITY_MISMATCH",
                "selected_candidate_ids",
                f"{supply.replace('_', ' ').title()} permits at most {maximum} selected box(es).",
                supply_question,
            )
        if final and len(selected) < minimum:
            add_error(
                "CANDIDATE_CARDINALITY_MISMATCH",
                "selected_candidate_ids",
                f"{supply.replace('_', ' ').title()} requires at least {minimum} selected box(es).",
                supply_question,
            )
        applicable = bool(state["relationship_applicable"])
        family = state.get("question_family")
        family_contract = self.relationship_contract["question_families"].get(family, {})
        allowed = list(family_contract.get("allowed_relationships", []))
        if not applicable:
            if relationship != state["canonical_relationship"]:
                add_error(
                    "RELATIONSHIP_NOT_ALLOWED_FOR_SUPPLY_BRANCH",
                    "candidate_relationship",
                    "This frame does not use a relationship follow-up; the stored value must be NOT_APPLICABLE.",
                    supply_question,
                )
            if (
                observation.get("relationship_question_id") is not None
                or observation.get("relationship_branch_family") is not None
            ):
                add_error(
                    "STALE_HIDDEN_RELATIONSHIP_BRANCH",
                    "relationship_question_id",
                    "A hidden relationship branch remained after the upstream answer changed.",
                    supply_question,
                )
        else:
            if (
                observation.get("relationship_question_id") != relationship_question
                or observation.get("relationship_branch_family") != family
            ):
                add_error(
                    "RELATIONSHIP_BRANCH_BINDING_MISMATCH",
                    "relationship_question_id",
                    "The relationship answer is not bound to this exact subject, frame, and branch.",
                    relationship_question,
                )
            if relationship is None:
                if final:
                    add_error(
                        "RELATIONSHIP_REQUIRED_BUT_MISSING",
                        "candidate_relationship",
                        "This candidate-supply answer requires one frame-specific relationship follow-up.",
                        relationship_question,
                    )
            elif relationship not in allowed:
                add_error(
                    "RELATIONSHIP_INCOMPATIBLE_WITH_BRANCH",
                    "candidate_relationship",
                    "The stored relationship does not belong to the active frame-specific branch.",
                    relationship_question,
                )
            else:
                relationship_rule = self.relationship_contract["relationship_states"].get(relationship)
                relationship_minimum = int(relationship_rule["minimum_selected_count"])
                if len(selected) < relationship_minimum:
                    add_error(
                        "RELATIONSHIP_CANDIDATE_CARDINALITY_MISMATCH",
                        "candidate_relationship",
                        f"{relationship.replace('_', ' ').title()} requires at least "
                        f"{relationship_minimum} selected boxes.",
                        relationship_question,
                    )
        next_question = relationship_question if applicable and not errors else supply_question
        return {
            "supply_state": supply,
            "selected_candidate_count": len(selected),
            "selected_candidate_ids": selected,
            "relationship_applicable": applicable,
            "minimum_selected_count": minimum,
            "maximum_selected_count": maximum,
            "canonical_relationship": state.get("canonical_relationship"),
            "question_family": family,
            "relationship_question_id": relationship_question if applicable else None,
            "allowed_relationships": allowed,
            "plain_english_question": family_contract.get("question"),
            "next_valid_question": next_question,
            "errors": errors,
        }

    def _validate_r4_relationships(
        self, payload: Mapping[str, Any], case: Mapping[str, Any], *, final: bool
    ) -> list[dict[str, Any]]:
        errors: list[dict[str, Any]] = []
        for subject_index, subject in enumerate(payload.get("subjects", [])):
            token = str(subject.get("subject_token", ""))
            for sequence, observation in enumerate(subject.get("frame_observations", [])):
                result = self.relationship_compatibility(
                    case=case,
                    subject_token=token,
                    subject_index=subject_index,
                    sequence=sequence,
                    observation=observation,
                    final=final,
                )
                errors.extend(result["errors"])
        if errors:
            raise ReviewValidationError(errors, "CANDIDATE_RELATIONSHIP_VALIDATION_FAILED")
        return errors

    def _validate_r3_frame_bindings(
        self,
        payload: Mapping[str, Any],
        case: Mapping[str, Any],
        *,
        final: bool,
    ) -> None:
        errors: list[dict[str, Any]] = []
        candidates = {
            candidate["candidate_id"]: candidate for rows in case.get("frame_candidates", []) for candidate in rows
        }
        candidate_frames = {
            candidate["candidate_id"]: sequence
            for sequence, rows in enumerate(case.get("frame_candidates", []))
            for candidate in rows
        }
        subjects = payload.get("subjects", [])
        for subject_index, subject in enumerate(subjects):
            token = subject.get("subject_token")
            observations = subject.get("frame_observations", [])
            if len(observations) != 9:
                errors.append(
                    self._structured_error(
                        code="FRAME_OBSERVATION_CARDINALITY",
                        field="frame_observations",
                        message="Each subject must have exactly nine frame observations.",
                        subject_token=token,
                        array_location=f"subjects[{subject_index}].frame_observations",
                    )
                )
                continue
            for sequence, observation in enumerate(observations):
                expected = self._frame_identity(case, sequence)
                observed = observation.get("canonical_frame_identity")
                question_id = f"subject_{subject_index}_location_{sequence}"
                location = f"subjects[{subject_index}].frame_observations[{sequence}]"
                if observed != expected or observation.get("frame_reference_id") != expected["frame_id"]:
                    errors.append(
                        self._structured_error(
                            code="SUBJECT_LOCATION_FRAME_MISMATCH",
                            field="canonical_frame_identity",
                            message=f"Subject {token} frame {sequence + 1} is not bound to its exact source frame.",
                            subject_token=token,
                            question_id=question_id,
                            expected=expected,
                            observed=observed if isinstance(observed, Mapping) else None,
                            array_location=location,
                        )
                    )
                visibility = observation.get("visibility")
                x = observation.get("subject_location_source_x")
                y = observation.get("subject_location_source_y")
                has_location = isinstance(x, (int, float)) and isinstance(y, (int, float))
                binding = observation.get("location_binding")
                if has_location:
                    if not 0 <= x <= case["source_width"] or not 0 <= y <= case["source_height"]:
                        errors.append(
                            self._structured_error(
                                code="SOURCE_COORDINATE_OUT_OF_BOUNDS",
                                field="subject_location_source_xy",
                                message="The subject point is outside the source frame.",
                                subject_token=token,
                                question_id=question_id,
                                expected=expected,
                                array_location=location,
                            )
                        )
                    expected_action = (
                        "APPROXIMATE_HIDDEN_LOCATION"
                        if observation.get("approximate_hidden_location")
                        else "SUBJECT_LOCATION"
                    )
                    if (
                        not isinstance(binding, Mapping)
                        or binding.get("action_type") != expected_action
                        or binding.get("canonical_frame_identity") != expected
                        or binding.get("question_id") != question_id
                        or binding.get("source_xy") != [x, y]
                    ):
                        errors.append(
                            self._structured_error(
                                code="LOCATION_BINDING_MISMATCH",
                                field="location_binding",
                                message=(
                                    "The stored subject point is not bound to the frame and question "
                                    "that created it."
                                ),
                                subject_token=token,
                                question_id=question_id,
                                expected=expected,
                                observed=binding.get("canonical_frame_identity")
                                if isinstance(binding, Mapping)
                                else None,
                                array_location=location,
                            )
                        )
                if final and visibility in ("VISIBLE_COMPLETE", "VISIBLE_PARTIAL") and not has_location:
                    errors.append(
                        self._structured_error(
                            code="MISSING_REQUIRED_LOCATION",
                            field="subject_location_source_xy",
                            message="A visible or partly visible subject needs one confirmed point.",
                            subject_token=token,
                            question_id=question_id,
                            expected=expected,
                            array_location=location,
                        )
                    )
                if visibility in ("OUT_OF_FRAME_OR_LEFT_SCENE", "NOT_PRESENT") and has_location:
                    errors.append(
                        self._structured_error(
                            code="LOCATION_NOT_ALLOWED_FOR_VISIBILITY",
                            field="subject_location_source_xy",
                            message="A subject marked absent or out of frame cannot retain a location.",
                            subject_token=token,
                            question_id=question_id,
                            expected=expected,
                            array_location=location,
                        )
                    )
                selected = observation.get("selected_candidate_ids", [])
                supply_question = f"subject_{subject_index}_supply_{sequence}"
                selection_binding = observation.get("candidate_selection_binding")
                if (
                    not isinstance(selection_binding, Mapping)
                    or selection_binding.get("action_type") != "CANDIDATE_SELECTION"
                    or selection_binding.get("canonical_frame_identity") != expected
                    or selection_binding.get("question_id") != supply_question
                    or selection_binding.get("selected_candidate_ids") != selected
                ):
                    errors.append(
                        self._structured_error(
                            code="CANDIDATE_SELECTION_FRAME_MISMATCH",
                            field="candidate_selection_binding",
                            message="Candidate selections must be bound to the exact frame where those boxes exist.",
                            subject_token=token,
                            question_id=supply_question,
                            expected=expected,
                            observed=selection_binding.get("canonical_frame_identity")
                            if isinstance(selection_binding, Mapping)
                            else None,
                            array_location=location,
                        )
                    )
                if len(selected) != len(set(selected)):
                    errors.append(
                        self._structured_error(
                            code="DUPLICATE_CANDIDATE_SELECTION",
                            field="selected_candidate_ids",
                            message="The same candidate box cannot be selected twice.",
                            subject_token=token,
                            question_id=supply_question,
                            expected=expected,
                            array_location=location,
                        )
                    )
                for candidate_id in selected:
                    if candidate_id not in candidates or candidate_frames[candidate_id] != sequence:
                        errors.append(
                            self._structured_error(
                                code="CANDIDATE_NOT_IN_EXACT_FRAME",
                                field="selected_candidate_ids",
                                message=f"Candidate {candidate_id} does not belong to this exact frame.",
                                subject_token=token,
                                question_id=supply_question,
                                expected=expected,
                                array_location=location,
                            )
                        )
                requires_selection = observation.get("observation_supply") in (
                    "ONE_USEFUL_CANDIDATE",
                    "MULTIPLE_CANDIDATES",
                    "MERGED_WITH_OTHER_PEOPLE",
                    "FRAGMENT_ONLY",
                )
                if final and requires_selection and not selected:
                    errors.append(
                        self._structured_error(
                            code="CANDIDATE_CARDINALITY_MISMATCH",
                            field="selected_candidate_ids",
                            message="This answer requires at least one selected candidate box.",
                            subject_token=token,
                            question_id=supply_question,
                            expected=expected,
                            array_location=location,
                        )
                    )
        for mapping_index, mapping in enumerate(payload.get("candidate_mappings", [])):
            sequence = mapping.get("frame_sequence")
            expected = self._frame_identity(case, sequence) if isinstance(sequence, int) and 0 <= sequence < 9 else None
            observed = mapping.get("canonical_frame_identity")
            candidate_id = mapping.get("candidate_id")
            if (
                expected is None
                or observed != expected
                or mapping.get("frame_reference_id") != expected["frame_id"]
                or candidate_id not in candidates
                or candidate_frames[candidate_id] != sequence
            ):
                errors.append(
                    self._structured_error(
                        code="CANDIDATE_MAPPING_FRAME_MISMATCH",
                        field="canonical_frame_identity",
                        message="Candidate mapping does not bind the selected box to its exact source frame.",
                        subject_token=mapping.get("subject_token"),
                        question_id=(
                            f"subject_{SUBJECT_TOKENS.index(mapping['subject_token'])}_supply_{sequence}"
                            if mapping.get("subject_token") in SUBJECT_TOKENS and isinstance(sequence, int)
                            else None
                        ),
                        expected=expected,
                        observed=observed if isinstance(observed, Mapping) else None,
                        array_location=f"candidate_mappings[{mapping_index}]",
                    )
                )
        for mark_index, mark in enumerate(
            payload.get("whole_burst_missed_person_marks", payload.get("missed_person_marks", []))
        ):
            sequence = mark.get("frame_sequence")
            expected = self._frame_identity(case, sequence) if isinstance(sequence, int) and 0 <= sequence < 9 else None
            observed = mark.get("canonical_frame_identity")
            if expected is None or observed != expected or mark.get("frame_reference_id") != expected["frame_id"]:
                errors.append(
                    self._structured_error(
                        code="MISSED_PERSON_MARK_FRAME_MISMATCH",
                        field="canonical_frame_identity",
                        message="The missed-person mark is not bound to the exact frame where it was placed.",
                        question_id="missed_mark",
                        expected=expected,
                        observed=observed if isinstance(observed, Mapping) else None,
                        array_location=f"missed_person_marks[{mark_index}]",
                    )
                )
            binding = mark.get("mark_binding")
            if expected and (
                not isinstance(binding, Mapping)
                or binding.get("action_type") != "MISSED_PERSON_MARK"
                or binding.get("canonical_frame_identity") != expected
                or binding.get("question_id") != "missed_mark"
                or binding.get("source_xy") != mark.get("source_xy")
            ):
                errors.append(
                    self._structured_error(
                        code="MISSED_PERSON_MARK_BINDING_MISMATCH",
                        field="mark_binding",
                        message="The missed-person mark is missing its immutable click-frame binding.",
                        question_id="missed_mark",
                        expected=expected,
                        observed=binding.get("canonical_frame_identity") if isinstance(binding, Mapping) else None,
                        array_location=f"missed_person_marks[{mark_index}]",
                    )
                )
        if errors:
            raise ReviewValidationError(errors)

    def _root(self, mode: str) -> Path:
        if mode == "practice":
            return self.practice
        if mode != "real":
            raise ValueError("invalid review mode")
        return self.decisions

    def _event_paths(self, mode: str = "real") -> list[Path]:
        return sorted((self._root(mode) / "events").glob("*/*.json"))

    def _ack_path(self, root: Path, event_id: str) -> Path:
        return root / "receipts/acknowledgements" / f"ack-{event_id}.json"

    def latest_events(self, mode: str = "real") -> dict[str, dict[str, Any]]:
        root = self._root(mode)
        latest: dict[str, dict[str, Any]] = {}
        for path in self._event_paths(mode):
            event = read_json(path)
            ack_path = self._ack_path(root, event["event_id"])
            if not ack_path.is_file():
                continue
            ack = read_json(ack_path)
            if ack.get("event_sha256") != sha256_file(path) or ack.get("server_validated") is not True:
                raise ValueError("acknowledgement linkage failure")
            burst_id = str(event["burst_id"])
            current = latest.get(burst_id)
            if current is None or int(event["server_sequence"]) > int(current["server_sequence"]):
                latest[burst_id] = event
        return latest

    def acknowledged_event(self, mode: str, event_id: str) -> dict[str, Any]:
        """Return one hash-verified immutable event for read-only restoration."""
        root = self._root(mode)
        matches = [path for path in self._event_paths(mode) if path.stem == event_id]
        if len(matches) != 1:
            raise ValueError("acknowledged event not found")
        event_path = matches[0]
        event = read_json(event_path)
        ack_path = self._ack_path(root, event_id)
        if not ack_path.is_file():
            raise ValueError("event is not acknowledged")
        acknowledgement = read_json(ack_path)
        if (
            acknowledgement.get("event_id") != event_id
            or acknowledgement.get("event_sha256") != sha256_file(event_path)
            or acknowledgement.get("server_validated") is not True
        ):
            raise ValueError("acknowledgement linkage failure")
        case = (self.practice_by_id if mode == "practice" else self.by_id).get(str(event.get("burst_id")))
        if case is None:
            raise ValueError("acknowledged event case is unavailable")
        return {
            "ok": True,
            "read_only": True,
            "event": event,
            "acknowledgement": acknowledgement,
            "case": case,
            "production_ready": False,
        }

    def _event_reference(self, event: Mapping[str, Any]) -> dict[str, Any]:
        root = self.decisions
        event_path = root / "events" / str(event["tranche_id"]) / f"{event['event_id']}.json"
        ack_path = self._ack_path(root, str(event["event_id"]))
        return {
            "burst_id": event["burst_id"],
            "event_id": event["event_id"],
            "event_sha256": sha256_file(event_path),
            "acknowledgement_receipt_id": f"ack-{event['event_id']}",
            "acknowledgement_receipt_sha256": sha256_file(ack_path),
        }

    def current_tranche_receipt(self, tranche_id: str, create: bool = False) -> dict[str, Any] | None:
        if tranche_id not in TRANCHES:
            raise ValueError("unknown tranche")
        latest = self.latest_events("real")
        cases = self.by_tranche[tranche_id]
        if any(case["burst_id"] not in latest for case in cases):
            return None
        references = [self._event_reference(latest[case["burst_id"]]) for case in cases]
        digest = canonical_digest(references)
        receipt_id = f"tranche-{tranche_id.removeprefix('TRANCHE_').lower()}-{digest[:24]}"
        path = self.decisions / "receipts/tranche_completion" / f"{receipt_id}.json"
        if path.is_file():
            payload = read_json(path)
            if payload.get("latest_event_set_digest") != digest:
                raise ValueError("tranche receipt digest mismatch")
            return payload
        if not create:
            return None
        payload = {
            "schema_version": "football_intelligence.g7e_b.tranche_completion_receipt.v1",
            "tranche_completion_receipt_id": receipt_id,
            "review_id": self.review_id,
            "review_revision": self.review_revision,
            "tranche_id": tranche_id,
            "tranche_manifest_sha256": self.tranche_manifest_sha256,
            "latest_event_set_digest": digest,
            "latest_acknowledged_events": references,
            "event_count": 20,
            "all_tranche_cases_complete": True,
            "created_at_utc": utc_now(),
            "production_ready": False,
        }
        atomic_write(path, canonical_bytes(payload))
        return payload

    def current_global_receipt(self, create: bool = False) -> dict[str, Any] | None:
        tranche_receipts = [self.current_tranche_receipt(tranche_id, create=False) for tranche_id in TRANCHES]
        if any(receipt is None for receipt in tranche_receipts):
            return None
        latest = self.latest_events("real")
        if len(latest) != 120:
            return None
        event_refs = [self._event_reference(latest[case["burst_id"]]) for case in self.cases]
        tranche_refs = []
        for receipt in tranche_receipts:
            assert receipt is not None
            path = self.decisions / "receipts/tranche_completion" / f"{receipt['tranche_completion_receipt_id']}.json"
            tranche_refs.append(
                {
                    "tranche_id": receipt["tranche_id"],
                    "tranche_completion_receipt_id": receipt["tranche_completion_receipt_id"],
                    "tranche_completion_receipt_sha256": sha256_file(path),
                }
            )
        digest = canonical_digest({"events": event_refs, "tranches": tranche_refs})
        receipt_id = f"global-{digest[:24]}"
        path = self.decisions / "receipts/global_completion" / f"{receipt_id}.json"
        if path.is_file():
            payload = read_json(path)
            if payload.get("latest_event_set_digest") != digest:
                raise ValueError("global receipt digest mismatch")
            return payload
        if not create:
            return None
        payload = {
            "schema_version": "football_intelligence.g7e_b.global_completion_receipt.v1",
            "global_completion_receipt_id": receipt_id,
            "review_id": self.review_id,
            "review_revision": self.review_revision,
            "latest_event_set_digest": digest,
            "latest_acknowledged_events": event_refs,
            "current_tranche_receipts": tranche_refs,
            "event_count": 120,
            "tranche_receipt_count": 6,
            "all_cases_complete": True,
            "created_at_utc": utc_now(),
            "production_ready": False,
        }
        atomic_write(path, canonical_bytes(payload))
        return payload

    def _unlocked(self) -> list[str]:
        path = self.decisions / "control/unlocked_tranches.json"
        if not path.is_file():
            return ["TRANCHE_1"]
        payload = read_json(path)
        values = [value for value in payload.get("unlocked_tranches", []) if value in TRANCHES]
        return values or ["TRANCHE_1"]

    def unlock_next(self, tranche_id: str) -> dict[str, Any]:
        if tranche_id not in TRANCHES[:-1]:
            raise ValueError("no later tranche")
        if self.current_tranche_receipt(tranche_id, create=False) is None:
            raise ValueError("current tranche is not complete")
        next_id = TRANCHES[TRANCHES.index(tranche_id) + 1]
        unlocked = self._unlocked()
        expected_previous = set(TRANCHES[: TRANCHES.index(next_id)])
        if not expected_previous <= set(unlocked):
            raise ValueError("tranche unlock sequence invalid")
        if next_id not in unlocked:
            unlocked.append(next_id)
            atomic_write(
                self.decisions / "control/unlocked_tranches.json",
                canonical_bytes(
                    {
                        "schema_version": "football_intelligence.g7e_b.tranche_unlock_state.v1",
                        "unlocked_tranches": unlocked,
                        "updated_at_utc": utc_now(),
                    }
                ),
            )
        return {"ok": True, "unlocked_tranches": unlocked, "next_tranche_id": next_id}

    def draft(self, mode: str, burst_id: str) -> dict[str, Any] | None:
        path = self._root(mode) / "drafts" / f"{burst_id}.json"
        if not path.is_file():
            return None
        payload = read_json(path)
        if payload.get("review_revision") != self.review_revision or payload.get("burst_id") != burst_id:
            return None
        return payload

    def incompatible_draft(self, mode: str, burst_id: str) -> dict[str, Any] | None:
        """Describe an older draft without migrating or modifying it."""
        path = self._root(mode) / "drafts" / f"{burst_id}.json"
        if not path.is_file():
            return None
        payload = read_json(path)
        if payload.get("review_revision") == self.review_revision:
            return None
        return {
            "burst_id": burst_id,
            "stored_review_revision": payload.get("review_revision"),
            "required_review_revision": self.review_revision,
            "reset_required": True,
            "silently_migrated": False,
        }

    def save_draft(self, payload: Mapping[str, Any], mode: str) -> dict[str, Any]:
        cases = self.practice_by_id if mode == "practice" else self.by_id
        burst_id = str(payload.get("burst_id", ""))
        if burst_id not in cases:
            raise ValueError("unknown burst draft")
        if self.review_revision == R6_REVIEW_REVISION:
            raise ReviewValidationError(
                [
                    self._structured_error(
                        code="DIRECT_DRAFT_MUTATION_FORBIDDEN",
                        field="draft",
                        message="R6 accepts versioned browser actions only.",
                        question_id=str(payload.get("current_question", "")),
                    )
                ],
                "DIRECT_DRAFT_MUTATION_FORBIDDEN",
            )
        if self.review_revision == R5_REVIEW_REVISION:
            self._require_r5_real_release(mode)
            return self._save_r5_draft(payload, mode, cases[burst_id])
        if self.review_revision in (R3_REVIEW_REVISION, R4_REVIEW_REVISION):
            return self._save_r3_draft(payload, mode, cases[burst_id])
        document = {
            "schema_version": "football_intelligence.g7e_b.temporal_review_draft.v1",
            "review_id": self.review_id,
            "review_revision": self.review_revision,
            "mode": mode,
            "burst_id": burst_id,
            "tranche_id": cases[burst_id].get("tranche_id"),
            "current_question": str(payload.get("current_question", "focus_confirmation")),
            "current_frame_sequence": int(payload.get("current_frame_sequence", 4)),
            "playback_speed": float(payload.get("playback_speed", 1.0)),
            "answers": payload.get("answers", {}),
            "subjects": payload.get("subjects", []),
            "candidate_mappings": payload.get("candidate_mappings", []),
            "missed_person_marks": payload.get("missed_person_marks", []),
            "source_manifest_hashes": cases[burst_id]["source_manifest_hashes"],
            "updated_at_utc": utc_now(),
            "production_ready": False,
        }
        if self.review_revision == R2_REVIEW_REVISION:
            case = cases[burst_id]
            if any(
                state.get("candidate_status") == "CANDIDATE_DATA_UNAVAILABLE"
                for state in case["per_frame_candidate_states"]
            ):
                raise ValueError("candidate data unavailable for an exact frame")
            document.update(
                {
                    "candidate_runtime_contract": case["candidate_runtime_contract"],
                    "unique_frame_candidate_status": case["unique_frame_candidate_status"],
                    "per_frame_candidate_states": case["per_frame_candidate_states"],
                }
            )
        self._validate_draft(document)
        atomic_write(self._root(mode) / "drafts" / f"{burst_id}.json", canonical_bytes(document))
        return document

    def initialize_draft(self, mode: str, burst_id: str) -> dict[str, Any]:
        if self.review_revision not in (R5_REVIEW_REVISION, R6_REVIEW_REVISION):
            raise ValueError("canonical draft initialization requires R5 or R6")
        self._require_r5_real_release(mode)
        cases = self.practice_by_id if mode == "practice" else self.by_id
        case = cases.get(burst_id)
        if case is None:
            raise ValueError("unknown burst draft")
        existing = self.draft(mode, burst_id)
        if existing is not None:
            return existing
        if self.canonical_contract is None or self.canonical_contract_sha256 is None:
            raise ValueError("R5 canonical contract is unavailable")
        if self.review_revision == R6_REVIEW_REVISION:
            if self.action_contract_sha256 is None:
                raise ValueError("R6 server-action contract is unavailable")
            initial = initialize_r6_draft(
                case,
                mode,
                self.canonical_contract,
                self.canonical_contract_sha256,
                self.action_contract_sha256,
            )
            return self._persist_r6_draft(initial, mode, expected_increment=1)
        initial = initialize_working_draft(case, mode, self.canonical_contract, self.canonical_contract_sha256)
        return self._save_r5_draft(initial, mode, case)

    @staticmethod
    def _r6_content_digest(document: Mapping[str, Any]) -> str:
        content = copy.deepcopy(dict(document))
        for field in ("draft_content_sha256", "optimistic_lock_token", "server_file_sha256"):
            content.pop(field, None)
        return canonical_digest(content)

    def _persist_r6_draft(
        self,
        document: Mapping[str, Any],
        mode: str,
        *,
        expected_increment: int,
        write: bool = True,
    ) -> dict[str, Any]:
        draft = copy.deepcopy(dict(document))
        burst_id = str(draft["burst_id"])
        path = contained_path(self._root(mode), "drafts", f"{burst_id}.json")
        draft["draft_version"] = int(draft.get("draft_version", 0)) + expected_increment
        draft["updated_at_utc"] = utc_now()
        draft["draft_content_sha256"] = self._r6_content_digest(draft)
        draft["optimistic_lock_token"] = canonical_digest(
            {
                "review_revision": R6_REVIEW_REVISION,
                "burst_id": burst_id,
                "draft_version": draft["draft_version"],
                "draft_content_sha256": draft["draft_content_sha256"],
            }
        )
        data = canonical_bytes(draft)
        if write:
            atomic_write(path, data)
        draft["server_file_sha256"] = hashlib.sha256(data).hexdigest()
        return draft

    def apply_browser_action(self, payload: Mapping[str, Any], mode: str) -> dict[str, Any]:
        if self.review_revision != R6_REVIEW_REVISION:
            raise ValueError("server-authoritative actions require R6")
        self._require_r5_real_release(mode)
        cases = self.practice_by_id if mode == "practice" else self.by_id
        burst_id = str(payload.get("burst_id", ""))
        case = cases.get(burst_id)
        if case is None:
            raise ValueError("unknown R6 burst action")
        if self.canonical_contract is None or self.action_contract_sha256 is None:
            raise ValueError("R6 contracts are unavailable")
        root = self._root(mode)
        action_id, idempotency_key = validate_action_envelope(payload, R6_ACTION_TYPES)
        action_bytes = canonical_bytes(dict(payload))
        semantic_payload = {
            key: payload[key]
            for key in (
                "review_revision",
                "contract_hash",
                "mode",
                "tranche_id",
                "burst_id",
                "question_instance_key",
                "action_type",
                "payload",
            )
        }
        action_envelope_sha256 = hashlib.sha256(action_bytes).hexdigest()
        action_semantic_sha256 = hashlib.sha256(canonical_bytes(semantic_payload)).hexdigest()
        ledger_path = contained_path(root, "action_idempotency", f"{idempotency_key}.json")
        if ledger_path.is_file():
            ledger = read_json(ledger_path)
            if ledger.get("action_id") != payload.get("action_id"):
                raise ValueError("action idempotency key is bound to another action")
            if ledger.get("action_semantic_sha256") != action_semantic_sha256:
                raise ValueError("action ID is already bound to different semantic content")
            restored = self.draft(mode, burst_id)
            if restored is None or restored.get("draft_content_sha256") != ledger.get("result_draft_sha256"):
                raise ValueError("idempotent action result no longer matches canonical draft")
            return {
                "ok": True,
                "idempotent_replay": True,
                "action_receipt_id": ledger["action_receipt_id"],
                "draft": restored,
            }
        current = self.draft(mode, burst_id)
        if current is None:
            current = self.initialize_draft(mode, burst_id)
        if int(payload.get("expected_draft_revision", -1)) != int(current.get("draft_version", -2)):
            raise StaleDraftError("STALE_DRAFT_REVISION", current)
        if payload.get("expected_draft_sha256") != current.get("draft_content_sha256"):
            raise StaleDraftError("STALE_DRAFT_HASH", current)
        current.pop("server_file_sha256", None)
        reduced = apply_r6_action(
            current,
            payload,
            case,
            self.canonical_contract,
            self.action_contract_sha256,
        )
        canonical_noop = (
            payload.get("action_type") == "COMPLETE_MISSED_PERSON_MARKING"
            and current.get("missed_marking_complete") is True
        )
        if canonical_noop:
            persisted = copy.deepcopy(current)
            persisted["server_file_sha256"] = hashlib.sha256(canonical_bytes(persisted)).hexdigest()
        else:
            persisted = self._persist_r6_draft(reduced, mode, expected_increment=1, write=False)
        receipt = {
            "schema_version": R6_ACTION_RECEIPT_SCHEMA,
            "receipt_id": f"action-ack-{action_id}",
            "action_id": action_id,
            "idempotency_key": idempotency_key,
            "review_revision": R6_REVIEW_REVISION,
            "mode": mode,
            "burst_id": burst_id,
            "question_instance_key": payload.get("question_instance_key"),
            "action_type": payload.get("action_type"),
            "result_draft_revision": persisted["draft_version"],
            "result_draft_sha256": persisted["draft_content_sha256"],
            "canonical_noop": canonical_noop,
            "server_validated": True,
            "created_at_utc": utc_now(),
            "production_ready": False,
        }
        receipt_bytes = canonical_bytes(receipt)
        ledger = {
            "schema_version": "football_intelligence.g7e_b_r6.action_idempotency.v1",
            "idempotency_key": idempotency_key,
            "action_id": action_id,
            "action_receipt_id": receipt["receipt_id"],
            "action_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "action_envelope_sha256": action_envelope_sha256,
            "action_semantic_sha256": action_semantic_sha256,
            "result_draft_revision": persisted["draft_version"],
            "result_draft_sha256": persisted["draft_content_sha256"],
            "created_at_utc": utc_now(),
            "production_ready": False,
        }
        persisted_for_disk = copy.deepcopy(persisted)
        persisted_for_disk.pop("server_file_sha256", None)
        ActionTransaction(root, action_id).commit(
            draft_relative=f"drafts/{burst_id}.json",
            draft_bytes=canonical_bytes(persisted_for_disk),
            receipt_relative=f"receipts/actions/{receipt['receipt_id']}.json",
            receipt_bytes=receipt_bytes,
            ledger_relative=f"action_idempotency/{idempotency_key}.json",
            ledger_bytes=canonical_bytes(ledger),
            transaction_context={
                "previous_draft_revision": current["draft_version"],
                "previous_draft_sha256": current["draft_content_sha256"],
                "action_envelope_sha256": action_envelope_sha256,
                "action_semantic_sha256": action_semantic_sha256,
                "next_draft_revision": persisted["draft_version"],
                "next_draft_sha256": persisted["draft_content_sha256"],
            },
            fail_after=self.action_transaction_fail_after,
        )
        return {
            "ok": True,
            "idempotent_replay": False,
            "canonical_noop": canonical_noop,
            "action_receipt_id": receipt["receipt_id"],
            "draft": persisted,
        }

    def _save_r5_draft(
        self,
        payload: Mapping[str, Any],
        mode: str,
        case: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.canonical_contract is None or self.canonical_contract_sha256 is None:
            raise ValueError("R5 canonical contract is unavailable")
        burst_id = str(payload["burst_id"])
        path = self._root(mode) / "drafts" / f"{burst_id}.json"
        current = read_json(path) if path.is_file() else None
        expected_version = int(current.get("draft_version", 0)) if current else 0
        expected_token = current.get("optimistic_lock_token") if current else None
        supplied_version = int(payload.get("draft_version", 0))
        supplied_token = payload.get("optimistic_lock_token")
        if supplied_version != expected_version or supplied_token != expected_token:
            raise ReviewValidationError(
                [
                    self._structured_error(
                        code="STALE_DRAFT_VERSION",
                        field="optimistic_lock_token",
                        message="A newer server-backed draft exists. Reload before saving this answer.",
                        question_id=str(payload.get("current_question", "")),
                    )
                ],
                "STALE_DRAFT_VERSION",
            )
        document = {
            "schema_version": R5_WORKING_DRAFT_SCHEMA,
            "review_id": self.review_id,
            "review_revision": R5_REVIEW_REVISION,
            "mode": mode,
            "burst_id": burst_id,
            "tranche_id": case.get("tranche_id"),
            "canonical_contract_id": R5_CONTRACT_ID,
            "canonical_contract_sha256": self.canonical_contract_sha256,
            "current_question_instance_key": payload.get("current_question_instance_key"),
            "current_question": str(payload.get("current_question", "original_focus")),
            "current_frame_sequence": int(payload.get("current_frame_sequence", 4)),
            "playback_speed": float(payload.get("playback_speed", 1.0)),
            "question_lifecycle": copy.deepcopy(payload.get("question_lifecycle", {})),
            "answered_domain_values": copy.deepcopy(payload.get("answered_domain_values", {})),
            "pending_edit": copy.deepcopy(payload.get("pending_edit", {})),
            "answers": copy.deepcopy(payload.get("answers", {})),
            "subjects": copy.deepcopy(payload.get("subjects", [])),
            "candidate_mappings": copy.deepcopy(payload.get("candidate_mappings", [])),
            "missed_person_marks": copy.deepcopy(payload.get("missed_person_marks", [])),
            "click_transactions": copy.deepcopy(payload.get("click_transactions", [])),
            "action_journal": copy.deepcopy(payload.get("action_journal", [])),
            "branch_invalidation_journal": copy.deepcopy(payload.get("branch_invalidation_journal", [])),
            "prior_final_save_error": copy.deepcopy(payload.get("prior_final_save_error")),
            "targeted_correction": copy.deepcopy(payload.get("targeted_correction")),
            "source_manifest_hashes": copy.deepcopy(case["source_manifest_hashes"]),
            "candidate_runtime_contract": copy.deepcopy(case["candidate_runtime_contract"]),
            "unique_frame_candidate_status": copy.deepcopy(case["unique_frame_candidate_status"]),
            "per_frame_candidate_states": copy.deepcopy(case["per_frame_candidate_states"]),
            "draft_version": expected_version + 1,
            "updated_at_utc": utc_now(),
            "acceptance_temporary": bool(payload.get("acceptance_temporary", False)),
            "migration_record": copy.deepcopy(payload.get("migration_record")),
            "production_ready": False,
        }
        errors = validate_working_draft(
            document,
            self.canonical_contract,
            self.canonical_contract_sha256,
            "DRAFT_SHAPE",
            case,
        )
        errors.extend(
            validate_working_draft(
                document,
                self.canonical_contract,
                self.canonical_contract_sha256,
                "DRAFT_PROGRESS",
                case,
            )
        )
        if errors:
            unique = {canonical_digest(error): error for error in errors}
            raise ReviewValidationError(list(unique.values()), "DRAFT_SCHEMA_MISMATCH")
        document["draft_content_sha256"] = canonical_digest(document)
        document["optimistic_lock_token"] = canonical_digest(
            {
                "review_revision": R5_REVIEW_REVISION,
                "burst_id": burst_id,
                "draft_version": document["draft_version"],
                "draft_content_sha256": document["draft_content_sha256"],
            }
        )
        atomic_write(path, canonical_bytes(document))
        document["server_file_sha256"] = sha256_file(path)
        return document

    def _save_r3_draft(
        self,
        payload: Mapping[str, Any],
        mode: str,
        case: Mapping[str, Any],
    ) -> dict[str, Any]:
        burst_id = str(payload["burst_id"])
        path = self._root(mode) / "drafts" / f"{burst_id}.json"
        current = read_json(path) if path.is_file() else None
        expected_version = int(current.get("draft_version", 0)) if current else 0
        expected_token = current.get("optimistic_lock_token") if current else None
        supplied_version = int(payload.get("draft_version", 0))
        supplied_token = payload.get("optimistic_lock_token")
        if supplied_version != expected_version or supplied_token != expected_token:
            raise ReviewValidationError(
                [
                    self._structured_error(
                        code="STALE_DRAFT_VERSION",
                        field="optimistic_lock_token",
                        message="A newer server-backed draft exists. Reload before saving this answer.",
                        question_id=str(payload.get("current_question", "")),
                    )
                ],
                "STALE_DRAFT_VERSION",
            )
        document = {
            "schema_version": (
                "football_intelligence.g7e_b_r4.temporal_review_draft.v1"
                if self.review_revision == R4_REVIEW_REVISION
                else "football_intelligence.g7e_b_r3.temporal_review_draft.v1"
            ),
            "review_id": self.review_id,
            "review_revision": self.review_revision,
            "mode": mode,
            "burst_id": burst_id,
            "tranche_id": case.get("tranche_id"),
            "current_question": str(payload.get("current_question", "original_focus")),
            "current_frame_sequence": int(payload.get("current_frame_sequence", 4)),
            "playback_speed": float(payload.get("playback_speed", 1.0)),
            "answers": payload.get("answers", {}),
            "subjects": payload.get("subjects", []),
            "candidate_mappings": payload.get("candidate_mappings", []),
            "missed_person_marks": payload.get("missed_person_marks", []),
            "click_transactions": payload.get("click_transactions", []),
            "action_journal": payload.get("action_journal", []),
            "prior_final_save_error": payload.get("prior_final_save_error"),
            "targeted_correction": payload.get("targeted_correction"),
            "real_draft_recovery": payload.get("real_draft_recovery"),
            "source_manifest_hashes": case["source_manifest_hashes"],
            "candidate_runtime_contract": case["candidate_runtime_contract"],
            "unique_frame_candidate_status": case["unique_frame_candidate_status"],
            "per_frame_candidate_states": case["per_frame_candidate_states"],
            "draft_version": expected_version + 1,
            "updated_at_utc": utc_now(),
            "production_ready": False,
        }
        self._validate_draft(document)
        self._validate_r3_frame_bindings(document, case, final=False)
        if self.review_revision == R4_REVIEW_REVISION:
            self._validate_r4_relationships(document, case, final=False)
        document["draft_content_sha256"] = canonical_digest(document)
        document["optimistic_lock_token"] = canonical_digest(
            {
                "review_revision": self.review_revision,
                "burst_id": burst_id,
                "draft_version": document["draft_version"],
                "draft_content_sha256": document["draft_content_sha256"],
            }
        )
        atomic_write(path, canonical_bytes(document))
        document["server_file_sha256"] = sha256_file(path)
        return document

    def _validate_draft(self, draft: Mapping[str, Any]) -> None:
        subjects = draft.get("subjects", [])
        if not isinstance(subjects, list) or len(subjects) > 3:
            raise ValueError("at most three burst-local subjects")
        tokens = [subject.get("subject_token") for subject in subjects]
        if len(tokens) != len(set(tokens)) or any(token not in SUBJECT_TOKENS for token in tokens):
            raise ValueError("invalid burst-local subject tokens")
        for mark in draft.get("missed_person_marks", []):
            xy = mark.get("source_xy")
            if not isinstance(xy, list) or len(xy) != 2 or not all(isinstance(value, (int, float)) for value in xy):
                raise ValueError("invalid source-coordinate missed-person mark")

    def _validate_event(self, payload: Mapping[str, Any], mode: str) -> tuple[dict[str, Any], Mapping[str, Any]]:
        cases = self.practice_by_id if mode == "practice" else self.by_id
        burst_id = str(payload.get("burst_id", ""))
        if burst_id not in cases:
            raise ValueError("unknown burst event")
        case = cases[burst_id]
        if self.review_revision == R5_REVIEW_REVISION:
            if payload.get("schema_version") != R5_EVENT_SCHEMA:
                raise ValueError("R5 immutable event schema mismatch")
            if payload.get("canonical_contract_sha256") != self.canonical_contract_sha256:
                raise ValueError("R5 immutable event contract hash mismatch")
            if payload.get("compiled_from_working_draft") is not True:
                raise ValueError("R5 immutable event was not compiled from a working draft")
            self._validate_r3_frame_bindings(payload, case, final=True)
            self._validate_r4_relationships(payload, case, final=True)
            event, validated_case = self._validate_r1_event(payload, mode, case)
            event.update(
                {
                    "schema_version": R5_EVENT_SCHEMA,
                    "canonical_contract_id": R5_CONTRACT_ID,
                    "canonical_contract_sha256": self.canonical_contract_sha256,
                    "question_lifecycle_sha256": payload.get("question_lifecycle_sha256"),
                    "compiled_from_working_draft": True,
                    "final_validation_profile": "IMMUTABLE_EVENT",
                }
            )
            return event, validated_case
        if self.review_revision in (R3_REVIEW_REVISION, R4_REVIEW_REVISION):
            self._validate_r3_frame_bindings(payload, case, final=True)
            if self.review_revision == R4_REVIEW_REVISION:
                self._validate_r4_relationships(payload, case, final=True)
            return self._validate_r1_event(payload, mode, case)
        if self.review_revision in (R1_REVIEW_REVISION, R2_REVIEW_REVISION):
            return self._validate_r1_event(payload, mode, case)
        focus_answer = payload.get("focus_answer")
        if focus_answer not in ("ONE_PERSON", "MULTIPLE_PEOPLE", "NO_RELEVANT_PERSON", "NOT_SURE"):
            raise ValueError("invalid focus answer")
        subjects = payload.get("subjects", [])
        if not isinstance(subjects, list) or len(subjects) > 3:
            raise ValueError("invalid subject count")
        tokens = [subject.get("subject_token") for subject in subjects]
        if tokens != list(SUBJECT_TOKENS[: len(tokens)]):
            raise ValueError("subjects must use ordered burst-local tokens")
        if focus_answer == "ONE_PERSON" and len(subjects) < 1:
            raise ValueError("one-person focus requires a subject")
        if focus_answer == "MULTIPLE_PEOPLE" and len(subjects) < 2:
            raise ValueError("multiple-person focus requires two subjects")
        if focus_answer in ("NO_RELEVANT_PERSON", "NOT_SURE") and subjects:
            raise ValueError("minimal focus branch must not create subjects")
        for subject in subjects:
            if subject.get("role") not in ROLES or subject.get("participation") not in PARTICIPATION:
                raise ValueError("invalid role or participation")
            if subject.get("certainty") not in CERTAINTY:
                raise ValueError("invalid certainty")
            if subject.get("candidate_relationship") not in (*RELATIONSHIPS, "NOT_APPLICABLE"):
                raise ValueError("invalid candidate relationship")
            observations = subject.get("frame_observations", [])
            if len(observations) != 9:
                raise ValueError("each subject requires nine frame observations")
            for observation in observations:
                if (
                    observation.get("visibility") not in VISIBILITY
                    or observation.get("observation_supply") not in SUPPLY
                ):
                    raise ValueError("invalid visibility or supply")
                if observation.get("occlusion_phase") not in OCCLUSION_PHASES:
                    raise ValueError("invalid occlusion phase")
            if subject.get("continuity") not in CONTINUITY:
                raise ValueError("invalid continuity")
            anchor_sequence = subject.get("anchor_frame_sequence")
            anchor_xy = subject.get("anchor_source_xy")
            if not isinstance(anchor_sequence, int) or not 0 <= anchor_sequence < 9:
                raise ValueError("invalid subject anchor frame")
            if (
                not isinstance(anchor_xy, list)
                or len(anchor_xy) != 2
                or not all(isinstance(value, (int, float)) for value in anchor_xy)
                or not 0 <= anchor_xy[0] <= case["source_width"]
                or not 0 <= anchor_xy[1] <= case["source_height"]
            ):
                raise ValueError("invalid subject source-coordinate anchor")
        candidates = {candidate["candidate_id"]: candidate for candidate in case["candidates"]}
        for mapping in payload.get("candidate_mappings", []):
            candidate = candidates.get(mapping.get("candidate_id"))
            if candidate is None or mapping.get("source_box_xyxy") != candidate["source_box_xyxy"]:
                raise ValueError("candidate mapping does not bind a frozen source box")
            if (
                mapping.get("frame_sequence") != 4
                or mapping.get("frame_reference_id") != case["frames"][4]["frame_reference_id"]
            ):
                raise ValueError("candidate mapping frame mismatch")
            if mapping.get("subject_token") not in tokens:
                raise ValueError("candidate mapping subject mismatch")
        missed_answer = payload.get("whole_burst_missed_person_answer")
        if missed_answer not in ("YES", "NO", "NOT_SURE"):
            raise ValueError("invalid whole-burst missed-person answer")
        marks = payload.get("whole_burst_missed_person_marks", [])
        if missed_answer == "YES" and not marks:
            raise ValueError("yes missed-person answer requires a source-coordinate mark")
        if missed_answer != "YES" and marks:
            raise ValueError("missed-person marks require a yes answer")
        for mark in marks:
            sequence = mark.get("frame_sequence")
            xy = mark.get("source_xy")
            if not isinstance(sequence, int) or not 0 <= sequence < 9:
                raise ValueError("invalid missed-person frame")
            if mark.get("frame_reference_id") != case["frames"][sequence]["frame_reference_id"]:
                raise ValueError("missed-person frame-reference mismatch")
            if (
                not isinstance(xy, list)
                or len(xy) != 2
                or not all(isinstance(value, (int, float)) for value in xy)
                or not 0 <= xy[0] <= case["source_width"]
                or not 0 <= xy[1] <= case["source_height"]
            ):
                raise ValueError("invalid missed-person source coordinate")
        if payload.get("source_frame_hashes") != [frame["source_frame_pixel_sha256"] for frame in case["frames"]]:
            raise ValueError("source-frame hash mismatch")
        latest = self.latest_events(mode)
        supersedes_event_id = payload.get("supersedes_event_id")
        if burst_id in latest and supersedes_event_id != latest[burst_id]["event_id"]:
            raise ValueError("superseding edit must reference the exact latest event")
        if burst_id not in latest and supersedes_event_id is not None:
            raise ValueError("first event cannot supersede another event")
        event = {
            "schema_version": "football_intelligence.g7e_b.burst_annotation_event.v1",
            "review_id": self.review_id,
            "review_revision": self.review_revision,
            "protocol_id": PROTOCOL_ID,
            "event_id": str(uuid.uuid4()),
            "mode": mode,
            "tranche_id": case.get("tranche_id"),
            "burst_id": burst_id,
            "burst_manifest_path": case["burst_manifest_path"],
            "burst_manifest_sha256": case["source_manifest_hashes"]["temporal_burst_manifest_sha256"],
            "source_frame_hashes": payload["source_frame_hashes"],
            "focus_answer": focus_answer,
            "subjects": subjects,
            "candidate_mappings": payload.get("candidate_mappings", []),
            "whole_burst_missed_person_answer": payload.get("whole_burst_missed_person_answer"),
            "whole_burst_missed_person_marks": marks,
            "summary_confirmed": payload.get("summary_confirmed") is True,
            "supersedes_event_id": supersedes_event_id,
            "acceptance_temporary": payload.get("acceptance_temporary") is True,
            "created_at_utc": utc_now(),
            "production_ready": False,
        }
        if not event["summary_confirmed"]:
            raise ValueError("summary confirmation required")
        return event, case

    def _validate_r1_event(
        self, payload: Mapping[str, Any], mode: str, case: Mapping[str, Any]
    ) -> tuple[dict[str, Any], Mapping[str, Any]]:
        """Validate R1 human-guided locations and frame-local candidate evidence."""
        focus_answer = payload.get("original_focus_box_answer")
        focus_values = (
            "ONE_RELEVANT_MATCH_PERSON",
            "PART_OF_ONE_RELEVANT_MATCH_PERSON",
            "MORE_THAN_ONE_RELEVANT_PERSON",
            "NO_RELEVANT_PERSON",
            "NOT_SURE",
        )
        if focus_answer not in focus_values:
            raise ValueError("invalid original focus box answer")
        subjects = payload.get("subjects", [])
        if not isinstance(subjects, list) or len(subjects) > 3:
            raise ValueError("invalid subject count")
        tokens = [subject.get("subject_token") for subject in subjects]
        if tokens != list(SUBJECT_TOKENS[: len(tokens)]):
            raise ValueError("subjects must use ordered burst-local tokens")
        if focus_answer in ("ONE_RELEVANT_MATCH_PERSON", "PART_OF_ONE_RELEVANT_MATCH_PERSON") and not subjects:
            raise ValueError("yellow-box person requires Subject A")
        allowed_definition_sources = (
            "YELLOW_ORIGINAL_FOCUS_CANDIDATE",
            "YELLOW_MULTI_PERSON_HUMAN_SELECTION",
            "BLUE_CONTEXT_HUMAN_SELECTION",
            "UNCERTAIN_HUMAN_SELECTION",
        )
        frame_candidates = case.get("frame_candidates", [[] for _ in range(9)])
        if len(frame_candidates) != 9:
            raise ValueError("frame-candidate closure mismatch")
        all_candidates = {
            (sequence, candidate["candidate_id"]): candidate
            for sequence, candidates in enumerate(frame_candidates)
            for candidate in candidates
        }
        expected_mappings: set[tuple[str, int, str]] = set()
        frame_states = case.get("per_frame_candidate_states", [])
        if self.review_revision in (
            R2_REVIEW_REVISION,
            R3_REVIEW_REVISION,
            R4_REVIEW_REVISION,
            R5_REVIEW_REVISION,
        ):
            if len(frame_states) != 9:
                raise ValueError("R2 candidate-state closure mismatch")
            if any(row.get("candidate_status") == "CANDIDATE_DATA_UNAVAILABLE" for row in frame_states):
                raise ValueError("candidate data unavailable for an exact frame")
            if payload.get("candidate_runtime_contract") != case.get("candidate_runtime_contract"):
                raise ValueError("candidate runtime contract mismatch")
            if payload.get("unique_frame_candidate_status") != case.get("unique_frame_candidate_status"):
                raise ValueError("unique-frame candidate-status mismatch")
            if payload.get("per_frame_candidate_states") != frame_states:
                raise ValueError("per-frame candidate-state mismatch")
        allowed_supply = (
            (*SUPPLY, "NO_USEFUL_BOX", "NOT_SURE")
            if self.review_revision in (R2_REVIEW_REVISION, R3_REVIEW_REVISION)
            else SUPPLY
        )
        for subject in subjects:
            if subject.get("subject_definition_source") not in allowed_definition_sources:
                raise ValueError("invalid subject definition source")
            if subject.get("role") not in ROLES or subject.get("participation") not in PARTICIPATION:
                raise ValueError("invalid role or participation")
            if subject.get("certainty") not in CERTAINTY:
                raise ValueError("invalid certainty")
            if self.review_revision not in (R4_REVIEW_REVISION, R5_REVIEW_REVISION) and subject.get(
                "candidate_relationship"
            ) not in (
                *RELATIONSHIPS,
                "NOT_APPLICABLE",
            ):
                raise ValueError("invalid candidate relationship")
            if subject.get("continuity") not in CONTINUITY:
                raise ValueError("invalid continuity")
            marker_review = subject.get("marker_continuity_confirmation")
            if marker_review not in ("SAME_SUBJECT_CONFIRMED", "CANNOT_TELL"):
                raise ValueError("marker continuity review required")
            anchor_sequence = subject.get("anchor_frame_sequence")
            anchor_xy = subject.get("anchor_source_xy")
            if not isinstance(anchor_sequence, int) or not 0 <= anchor_sequence < 9:
                raise ValueError("invalid subject anchor frame")
            self._validate_source_xy(anchor_xy, case, "subject anchor")
            observations = subject.get("frame_observations", [])
            if len(observations) != 9:
                raise ValueError("each subject requires nine frame observations")
            for sequence, observation in enumerate(observations):
                if observation.get("frame_reference_id") != case["frames"][sequence]["frame_reference_id"]:
                    raise ValueError("subject location frame mismatch")
                visibility = observation.get("visibility")
                supply = observation.get("observation_supply")
                if visibility not in VISIBILITY or supply not in allowed_supply:
                    raise ValueError("invalid visibility or supply")
                if observation.get("occlusion_phase") not in OCCLUSION_PHASES:
                    raise ValueError("invalid occlusion phase")
                x = observation.get("subject_location_source_x")
                y = observation.get("subject_location_source_y")
                has_point = isinstance(x, (int, float)) and isinstance(y, (int, float))
                if visibility in ("VISIBLE_COMPLETE", "VISIBLE_PARTIAL"):
                    if not has_point or observation.get("human_confirmed") is not True:
                        raise ValueError("visible subject requires a human-confirmed location")
                elif visibility in ("OUT_OF_FRAME_OR_LEFT_SCENE", "NOT_PRESENT") and has_point:
                    raise ValueError("absent subject must not have a location")
                if has_point:
                    self._validate_source_xy([x, y], case, "frame subject location")
                if (
                    observation.get("approximate_hidden_location") is True
                    and visibility != "FULLY_OCCLUDED_EXPECTED_PRESENT"
                ):
                    raise ValueError("approximate hidden point requires hidden visibility")
                if visibility in ("OUT_OF_FRAME_OR_LEFT_SCENE", "NOT_PRESENT") and supply != "NOT_APPLICABLE":
                    raise ValueError("absent frame supply must be NOT_APPLICABLE")
                if (
                    self.review_revision
                    in (R2_REVIEW_REVISION, R3_REVIEW_REVISION, R4_REVIEW_REVISION, R5_REVIEW_REVISION)
                    and relevant_visibility_for_supply(visibility)
                    and frame_states[sequence]["candidate_status"] == "VERIFIED_ZERO_CANDIDATES"
                    and supply
                    not in (
                        ("NO_CANDIDATE", "UNCERTAIN")
                        if self.review_revision in (R4_REVIEW_REVISION, R5_REVIEW_REVISION)
                        else ("NO_USEFUL_BOX", "NOT_SURE")
                    )
                ):
                    raise ValueError("verified-zero frame permits only no useful box or not sure")
                selected = observation.get("selected_candidate_ids", [])
                if not isinstance(selected, list) or len(selected) != len(set(selected)):
                    raise ValueError("invalid selected candidate IDs")
                needs_selection = supply in (
                    "ONE_USEFUL_CANDIDATE",
                    "MULTIPLE_CANDIDATES",
                    "MERGED_WITH_OTHER_PEOPLE",
                    "FRAGMENT_ONLY",
                )
                if needs_selection and not selected:
                    raise ValueError("candidate supply answer requires a selected box")
                if not needs_selection and selected:
                    raise ValueError("candidate selections do not match supply answer")
                for candidate_id in selected:
                    if (sequence, candidate_id) not in all_candidates:
                        raise ValueError("selected candidate is unavailable in this frame")
                    expected_mappings.add((str(subject["subject_token"]), sequence, str(candidate_id)))
        candidate_mappings = payload.get("candidate_mappings", [])
        actual_mappings: set[tuple[str, int, str]] = set()
        for mapping in candidate_mappings:
            sequence = mapping.get("frame_sequence")
            key = (sequence, mapping.get("candidate_id"))
            candidate = all_candidates.get(key)
            if candidate is None or mapping.get("source_box_xyxy") != candidate["source_box_xyxy"]:
                raise ValueError("candidate mapping does not bind a frozen frame-local box")
            if mapping.get("frame_reference_id") != case["frames"][sequence]["frame_reference_id"]:
                raise ValueError("candidate mapping frame mismatch")
            actual_mappings.add((str(mapping.get("subject_token")), int(sequence), str(mapping.get("candidate_id"))))
        if actual_mappings != expected_mappings:
            raise ValueError("candidate mappings do not match frame observations")
        self._validate_missed_people(payload, case)
        if payload.get("source_frame_hashes") != [frame["source_frame_pixel_sha256"] for frame in case["frames"]]:
            raise ValueError("source-frame hash mismatch")
        latest = self.latest_events(mode)
        burst_id = str(payload["burst_id"])
        supersedes_event_id = payload.get("supersedes_event_id")
        if burst_id in latest and supersedes_event_id != latest[burst_id]["event_id"]:
            raise ValueError("superseding edit must reference the exact latest event")
        if burst_id not in latest and supersedes_event_id is not None:
            raise ValueError("first event cannot supersede another event")
        normalized_focus = {
            "ONE_RELEVANT_MATCH_PERSON": "ONE_PERSON",
            "PART_OF_ONE_RELEVANT_MATCH_PERSON": "ONE_PERSON",
            "MORE_THAN_ONE_RELEVANT_PERSON": "MULTIPLE_PEOPLE",
            "NO_RELEVANT_PERSON": "NO_RELEVANT_PERSON",
            "NOT_SURE": "NOT_SURE",
        }[focus_answer]
        event = {
            "schema_version": "football_intelligence.g7e_b_r1.burst_annotation_event.v1",
            "review_id": self.review_id,
            "review_revision": self.review_revision,
            "protocol_id": PROTOCOL_ID,
            "event_id": str(uuid.uuid4()),
            "mode": mode,
            "tranche_id": case.get("tranche_id"),
            "burst_id": burst_id,
            "burst_manifest_path": case["burst_manifest_path"],
            "burst_manifest_sha256": case["source_manifest_hashes"]["temporal_burst_manifest_sha256"],
            "source_frame_hashes": payload["source_frame_hashes"],
            "focus_answer": normalized_focus,
            "original_focus_box_answer": focus_answer,
            "context_subject_answer": payload.get("context_subject_answer", "NOT_APPLICABLE"),
            "subjects": subjects,
            "candidate_mappings": candidate_mappings,
            "whole_burst_missed_person_answer": payload.get("whole_burst_missed_person_answer"),
            "whole_burst_missed_person_marks": payload.get("whole_burst_missed_person_marks", []),
            "summary_confirmed": payload.get("summary_confirmed") is True,
            "supersedes_event_id": supersedes_event_id,
            "acceptance_temporary": payload.get("acceptance_temporary") is True,
            "created_at_utc": utc_now(),
            "production_ready": False,
        }
        if self.review_revision in (
            R2_REVIEW_REVISION,
            R3_REVIEW_REVISION,
            R4_REVIEW_REVISION,
            R5_REVIEW_REVISION,
        ):
            event.update(
                {
                    "schema_version": (
                        R5_EVENT_SCHEMA
                        if self.review_revision == R5_REVIEW_REVISION
                        else (
                            "football_intelligence.g7e_b_r4.burst_annotation_event.v1"
                            if self.review_revision == R4_REVIEW_REVISION
                            else (
                                "football_intelligence.g7e_b_r3.burst_annotation_event.v1"
                                if self.review_revision == R3_REVIEW_REVISION
                                else "football_intelligence.g7e_b_r2.burst_annotation_event.v1"
                            )
                        )
                    ),
                    "candidate_runtime_contract": case["candidate_runtime_contract"],
                    "unique_frame_candidate_status": case["unique_frame_candidate_status"],
                    "per_frame_candidate_states": case["per_frame_candidate_states"],
                    "candidate_supply_interpretation": [
                        {
                            "frame_reference_id": state["frame_reference_id"],
                            "candidate_status": state["candidate_status"],
                            "available_candidate_count": state["post_gate_candidate_count"],
                            "selected_candidate_ids": sorted(
                                {
                                    candidate_id
                                    for subject in subjects
                                    for observation in subject["frame_observations"]
                                    if observation["frame_reference_id"] == state["frame_reference_id"]
                                    for candidate_id in observation.get("selected_candidate_ids", [])
                                }
                            ),
                        }
                        for state in frame_states
                    ],
                }
            )
        if self.review_revision in (R3_REVIEW_REVISION, R4_REVIEW_REVISION, R5_REVIEW_REVISION):
            event.update(
                {
                    "draft_version": payload.get("draft_version"),
                    "draft_content_sha256": payload.get("draft_content_sha256"),
                    "click_transactions": payload.get("click_transactions", []),
                    "frame_binding_validation": "PASSED",
                }
            )
        if self.review_revision in (R4_REVIEW_REVISION, R5_REVIEW_REVISION):
            event.update(
                {
                    "relationship_compatibility_sha256": self.relationship_compatibility_sha256,
                    "relationship_branch_validation": "PASSED",
                }
            )
        if not event["summary_confirmed"]:
            raise ValueError("summary confirmation required")
        return event, case

    @staticmethod
    def _validate_source_xy(xy: Any, case: Mapping[str, Any], label: str) -> None:
        if (
            not isinstance(xy, list)
            or len(xy) != 2
            or not all(isinstance(value, (int, float)) for value in xy)
            or not 0 <= xy[0] <= case["source_width"]
            or not 0 <= xy[1] <= case["source_height"]
        ):
            raise ValueError(f"invalid {label} source coordinate")

    def _validate_missed_people(self, payload: Mapping[str, Any], case: Mapping[str, Any]) -> None:
        answer = payload.get("whole_burst_missed_person_answer")
        marks = payload.get("whole_burst_missed_person_marks", [])
        if answer not in ("YES", "NO", "NOT_SURE"):
            raise ValueError("invalid whole-burst missed-person answer")
        if answer == "YES" and not marks:
            raise ValueError("yes missed-person answer requires a source-coordinate mark")
        if answer != "YES" and marks:
            raise ValueError("missed-person marks require a yes answer")
        for mark in marks:
            sequence = mark.get("frame_sequence")
            if not isinstance(sequence, int) or not 0 <= sequence < 9:
                raise ValueError("invalid missed-person frame")
            if mark.get("frame_reference_id") != case["frames"][sequence]["frame_reference_id"]:
                raise ValueError("missed-person frame-reference mismatch")
            self._validate_source_xy(mark.get("source_xy"), case, "missed-person")

    def _r3_draft_reference(
        self,
        payload: Mapping[str, Any],
        mode: str,
        case: Mapping[str, Any],
    ) -> dict[str, Any]:
        path = self._root(mode) / "drafts" / f"{case['burst_id']}.json"
        if not path.is_file():
            raise ReviewValidationError(
                [
                    self._structured_error(
                        code="SERVER_BACKED_DRAFT_REQUIRED",
                        field="draft",
                        message="The server-backed draft is missing. Restore the draft before final save.",
                        question_id="summary",
                    )
                ],
                "SERVER_BACKED_DRAFT_REQUIRED",
            )
        draft = read_json(path)
        supplied = (
            payload.get("draft_version"),
            payload.get("draft_content_sha256"),
            payload.get("optimistic_lock_token"),
        )
        expected = (
            draft.get("draft_version"),
            draft.get("draft_content_sha256"),
            draft.get("optimistic_lock_token"),
        )
        if supplied != expected:
            raise ReviewValidationError(
                [
                    self._structured_error(
                        code="STALE_FINAL_SAVE_DRAFT",
                        field="draft_version",
                        message="Final save does not reference the latest acknowledged draft.",
                        question_id="summary",
                    )
                ],
                "STALE_FINAL_SAVE_DRAFT",
            )
        comparisons = (
            (payload.get("subjects", []), draft.get("subjects", []), "subjects"),
            (payload.get("candidate_mappings", []), draft.get("candidate_mappings", []), "candidate_mappings"),
            (
                payload.get("whole_burst_missed_person_marks", []),
                draft.get("missed_person_marks", []),
                "whole_burst_missed_person_marks",
            ),
            (
                payload.get("original_focus_box_answer"),
                draft.get("answers", {}).get("original_focus_box_answer"),
                "original_focus_box_answer",
            ),
            (
                payload.get("whole_burst_missed_person_answer"),
                draft.get("answers", {}).get("missed_check"),
                "whole_burst_missed_person_answer",
            ),
        )
        mismatch = next(
            (field for supplied_value, stored_value, field in comparisons if supplied_value != stored_value), None
        )
        if mismatch:
            raise ReviewValidationError(
                [
                    self._structured_error(
                        code="FINAL_EVENT_DIFFERS_FROM_DRAFT",
                        field=mismatch,
                        message="Final save differs from the latest acknowledged server draft.",
                        question_id="summary",
                    )
                ],
                "FINAL_EVENT_DIFFERS_FROM_DRAFT",
            )
        return draft

    def _r3_preflight_inputs(
        self,
        payload: Mapping[str, Any],
        mode: str,
    ) -> tuple[dict[str, Any], Mapping[str, Any], dict[str, Any]]:
        cases = self.practice_by_id if mode == "practice" else self.by_id
        burst_id = str(payload.get("burst_id", ""))
        if burst_id not in cases:
            raise ValueError("unknown burst event")
        case = cases[burst_id]
        event, _ = self._validate_event(payload, mode)
        draft = self._r3_draft_reference(payload, mode, case)
        identity_inputs = {
            "review_revision": self.review_revision,
            "burst_id": burst_id,
            "draft_version": draft["draft_version"],
            "draft_content_sha256": draft["draft_content_sha256"],
        }
        base_digest = canonical_digest(identity_inputs)
        proposed_event_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.review_revision}:{base_digest}"))
        idempotency_key = canonical_digest({**identity_inputs, "proposed_event_id": proposed_event_id})
        event["event_id"] = proposed_event_id
        event["idempotency_key"] = idempotency_key
        event["draft_version"] = draft["draft_version"]
        event["draft_content_sha256"] = draft["draft_content_sha256"]
        return (
            event,
            case,
            {
                "proposed_event_id": proposed_event_id,
                "idempotency_key": idempotency_key,
                "draft_version": draft["draft_version"],
                "draft_content_sha256": draft["draft_content_sha256"],
            },
        )

    def _r5_preflight_inputs(
        self,
        payload: Mapping[str, Any],
        mode: str,
    ) -> tuple[dict[str, Any], Mapping[str, Any], dict[str, Any]]:
        if self.canonical_contract is None or self.canonical_contract_sha256 is None:
            raise ValueError("R5 canonical contract is unavailable")
        self._require_r5_real_release(mode)
        cases = self.practice_by_id if mode == "practice" else self.by_id
        burst_id = str(payload.get("burst_id", ""))
        case = cases.get(burst_id)
        if case is None:
            raise ValueError("unknown burst event")
        path = self._root(mode) / "drafts" / f"{burst_id}.json"
        if not path.is_file():
            raise ReviewValidationError(
                [
                    self._structured_error(
                        code="SERVER_BACKED_DRAFT_REQUIRED",
                        field="draft",
                        message="The server-backed working draft is missing.",
                        question_id="summary",
                    )
                ],
                "SERVER_BACKED_DRAFT_REQUIRED",
            )
        draft = read_json(path)
        expected = (
            draft.get("draft_version"),
            draft.get("draft_content_sha256"),
            draft.get("optimistic_lock_token"),
        )
        supplied = (
            payload.get("draft_version"),
            payload.get("draft_content_sha256"),
            payload.get("optimistic_lock_token"),
        )
        if supplied != expected:
            raise ReviewValidationError(
                [
                    self._structured_error(
                        code="STALE_FINAL_SAVE_DRAFT",
                        field="draft_version",
                        message="Final save does not reference the latest server-backed working draft.",
                        question_id="summary",
                    )
                ],
                "STALE_FINAL_SAVE_DRAFT",
            )
        event, errors = compile_final_event(
            draft,
            self.canonical_contract,
            self.canonical_contract_sha256,
            case,
        )
        if errors or event is None:
            raise ReviewValidationError(errors, "FINAL_EVENT_COMPILATION_FAILED")
        event, _ = self._validate_event(event, mode)
        event.pop("created_at_utc", None)
        event.pop("event_id", None)
        identity_inputs = {
            "review_revision": R5_REVIEW_REVISION,
            "canonical_contract_sha256": self.canonical_contract_sha256,
            "burst_id": burst_id,
            "draft_version": draft["draft_version"],
            "draft_content_sha256": draft["draft_content_sha256"],
            "compiled_event_sha256": canonical_digest(event),
        }
        base_digest = canonical_digest(identity_inputs)
        proposed_event_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{R5_REVIEW_REVISION}:{base_digest}"))
        idempotency_key = canonical_digest({**identity_inputs, "proposed_event_id": proposed_event_id})
        event["event_id"] = proposed_event_id
        event["idempotency_key"] = idempotency_key
        event["draft_version"] = draft["draft_version"]
        event["draft_content_sha256"] = draft["draft_content_sha256"]
        return (
            event,
            case,
            {
                "proposed_event_id": proposed_event_id,
                "idempotency_key": idempotency_key,
                "draft_version": draft["draft_version"],
                "draft_content_sha256": draft["draft_content_sha256"],
                "compiled_event_sha256": canonical_digest(event),
            },
        )

    def _r6_preflight_inputs(
        self,
        payload: Mapping[str, Any],
        mode: str,
    ) -> tuple[dict[str, Any], Mapping[str, Any], dict[str, Any]]:
        if (
            self.canonical_contract is None
            or self.canonical_contract_sha256 is None
            or self.action_contract_sha256 is None
        ):
            raise ValueError("R6 contracts are unavailable")
        self._require_r5_real_release(mode)
        cases = self.practice_by_id if mode == "practice" else self.by_id
        burst_id = str(payload.get("burst_id", ""))
        case = cases.get(burst_id)
        if case is None:
            raise ValueError("unknown R6 burst event")
        path = self._root(mode) / "drafts" / f"{burst_id}.json"
        if not path.is_file():
            raise ReviewValidationError(
                [
                    self._structured_error(
                        code="SERVER_BACKED_DRAFT_REQUIRED",
                        field="draft",
                        message="The canonical R6 server draft is missing.",
                        question_id="summary",
                    )
                ],
                "SERVER_BACKED_DRAFT_REQUIRED",
            )
        draft = read_json(path)
        expected = (
            draft.get("draft_version"),
            draft.get("draft_content_sha256"),
            draft.get("optimistic_lock_token"),
        )
        supplied = (
            payload.get("draft_version"),
            payload.get("draft_content_sha256"),
            payload.get("optimistic_lock_token"),
        )
        if supplied != expected:
            raise ReviewValidationError(
                [
                    self._structured_error(
                        code="STALE_FINAL_SAVE_DRAFT",
                        field="draft_version",
                        message="Final save does not reference the latest canonical R6 draft.",
                        question_id="summary",
                    )
                ],
                "STALE_FINAL_SAVE_DRAFT",
            )
        event, errors = compile_r6_final_event(
            draft,
            self.canonical_contract,
            self.canonical_contract_sha256,
            self.action_contract_sha256,
            case,
        )
        if errors or event is None:
            raise ReviewValidationError(errors, "FINAL_EVENT_COMPILATION_FAILED")
        identity_inputs = {
            "review_revision": R6_REVIEW_REVISION,
            "server_action_contract_sha256": self.action_contract_sha256,
            "burst_id": burst_id,
            "draft_version": draft["draft_version"],
            "draft_content_sha256": draft["draft_content_sha256"],
            "compiled_event_sha256": canonical_digest(event),
        }
        base_digest = canonical_digest(identity_inputs)
        proposed_event_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{R6_REVIEW_REVISION}:{base_digest}"))
        idempotency_key = canonical_digest({**identity_inputs, "proposed_event_id": proposed_event_id})
        event["event_id"] = proposed_event_id
        event["idempotency_key"] = idempotency_key
        return (
            event,
            case,
            {
                "proposed_event_id": proposed_event_id,
                "idempotency_key": idempotency_key,
                "draft_version": draft["draft_version"],
                "draft_content_sha256": draft["draft_content_sha256"],
                "compiled_event_sha256": canonical_digest(event),
            },
        )

    def _r3_error_path(self, mode: str, burst_id: str) -> Path:
        return self._root(mode) / "status" / f"final_save_error_{burst_id}.json"

    def final_save_preflight(self, payload: Mapping[str, Any], mode: str) -> dict[str, Any]:
        if self.review_revision not in (
            R3_REVIEW_REVISION,
            R4_REVIEW_REVISION,
            R5_REVIEW_REVISION,
            R6_REVIEW_REVISION,
        ):
            raise ValueError("final-save preflight requires the R3, R4, R5, or R6 reviewer")
        self._require_r5_real_release(mode)
        try:
            if self.review_revision == R6_REVIEW_REVISION:
                _, _, inputs = self._r6_preflight_inputs(payload, mode)
            elif self.review_revision == R5_REVIEW_REVISION:
                _, _, inputs = self._r5_preflight_inputs(payload, mode)
            else:
                _, _, inputs = self._r3_preflight_inputs(payload, mode)
        except ReviewValidationError as exc:
            burst_id = str(payload.get("burst_id", "unknown"))
            error_state = {
                "schema_version": (
                    "football_intelligence.g7e_b_r6.final_save_error.v1"
                    if self.review_revision == R6_REVIEW_REVISION
                    else (
                        "football_intelligence.g7e_b_r5.final_save_error.v1"
                        if self.review_revision == R5_REVIEW_REVISION
                        else (
                            "football_intelligence.g7e_b_r4.final_save_error.v1"
                            if self.review_revision == R4_REVIEW_REVISION
                            else "football_intelligence.g7e_b_r3.final_save_error.v1"
                        )
                    )
                ),
                "review_revision": self.review_revision,
                "burst_id": burst_id,
                "mode": mode,
                "error_code": exc.error_code,
                "errors": exc.errors,
                "created_at_utc": utc_now(),
                "production_ready": False,
            }
            atomic_write(self._r3_error_path(mode, burst_id), canonical_bytes(error_state))
            return {"ok": False, "status": "FINAL_SAVE_ERROR", **error_state}
        return {
            "ok": True,
            "status": "READY_TO_PERSIST",
            **inputs,
            "validation_error_count": 0,
            "production_ready": False,
        }

    def _r3_acknowledge_event(
        self,
        *,
        root: Path,
        event: Mapping[str, Any],
        event_path: Path,
        case: Mapping[str, Any],
    ) -> dict[str, Any]:
        receipt = {
            "schema_version": (
                "football_intelligence.g7e_b_r6.event_acknowledgement_receipt.v1"
                if self.review_revision == R6_REVIEW_REVISION
                else (
                    "football_intelligence.g7e_b_r5.event_acknowledgement_receipt.v1"
                    if self.review_revision == R5_REVIEW_REVISION
                    else (
                        "football_intelligence.g7e_b_r4.event_acknowledgement_receipt.v1"
                        if self.review_revision == R4_REVIEW_REVISION
                        else "football_intelligence.g7e_b_r3.event_acknowledgement_receipt.v1"
                    )
                )
            ),
            "receipt_id": f"ack-{event['event_id']}",
            "review_id": self.review_id,
            "review_revision": self.review_revision,
            "mode": event["mode"],
            "tranche_id": case.get("tranche_id"),
            "burst_id": event["burst_id"],
            "event_id": event["event_id"],
            "idempotency_key": event["idempotency_key"],
            "event_relative_path": str(event_path.relative_to(root)).replace("\\", "/"),
            "event_byte_size": event_path.stat().st_size,
            "event_sha256": sha256_file(event_path),
            "server_validated": True,
            "case_complete": True,
            "created_at_utc": utc_now(),
            "production_ready": False,
        }
        ack_path = self._ack_path(root, str(event["event_id"]))
        if ack_path.is_file():
            existing = read_json(ack_path)
            if (
                existing.get("event_sha256") != receipt["event_sha256"]
                or existing.get("idempotency_key") != receipt["idempotency_key"]
            ):
                raise ValueError("existing acknowledgement does not bind the exact event")
            return existing
        atomic_write(ack_path, canonical_bytes(receipt))
        if read_json(ack_path).get("event_sha256") != sha256_file(event_path):
            raise ValueError("acknowledgement hash verification failed")
        return receipt

    def _save_r3_event(self, payload: Mapping[str, Any], mode: str) -> dict[str, Any]:
        supplied_key = str(payload.get("idempotency_key", ""))
        if supplied_key and (
            len(supplied_key) != 64 or any(character not in "0123456789abcdef" for character in supplied_key)
        ):
            raise ValueError("invalid idempotency key")
        root = self._root(mode)
        existing_ledger_path = root / "idempotency" / f"{supplied_key}.json"
        if supplied_key and existing_ledger_path.is_file():
            ledger = read_json(existing_ledger_path)
            if ledger.get("idempotency_key") != supplied_key or ledger.get("event_id") != payload.get(
                "proposed_event_id"
            ):
                raise ValueError("idempotency record does not match this final-save request")
            event_path = root / str(ledger["event_relative_path"])
            if not event_path.is_file() or sha256_file(event_path) != ledger.get("event_sha256"):
                raise ValueError("persisted idempotent event failed hash validation")
            event = read_json(event_path)
            cases = self.practice_by_id if mode == "practice" else self.by_id
            case = cases.get(str(event.get("burst_id")))
            if case is None:
                raise ValueError("persisted event references an unknown burst")
            receipt = self._r3_acknowledge_event(root=root, event=event, event_path=event_path, case=case)
            ledger.update(
                {
                    "status": "SERVER_ACKNOWLEDGED",
                    "acknowledgement_receipt_id": receipt["receipt_id"],
                    "acknowledgement_receipt_sha256": sha256_file(self._ack_path(root, str(event["event_id"]))),
                    "updated_at_utc": utc_now(),
                }
            )
            atomic_write(existing_ledger_path, canonical_bytes(ledger))
            draft_path = root / "drafts" / f"{event['burst_id']}.json"
            if draft_path.is_file():
                draft_path.unlink()
            tranche_receipt = (
                self.current_tranche_receipt(str(case["tranche_id"]), create=True) if mode == "real" else None
            )
            global_receipt = self.current_global_receipt(create=True) if tranche_receipt is not None else None
            return {
                "ok": True,
                "saved": True,
                "status": "SERVER_ACKNOWLEDGED",
                "event_id": event["event_id"],
                "acknowledgement_receipt_id": receipt["receipt_id"],
                "idempotency_key": supplied_key,
                "recovered_existing_event": True,
                "duplicate_event_created": False,
                "tranche_complete": tranche_receipt is not None,
                "tranche_completion_receipt_id": (
                    tranche_receipt["tranche_completion_receipt_id"] if tranche_receipt else None
                ),
                "all_cases_complete": global_receipt is not None,
                "global_completion_receipt_id": (
                    global_receipt["global_completion_receipt_id"] if global_receipt else None
                ),
                "production_ready": False,
            }
        if self.review_revision == R6_REVIEW_REVISION:
            event, case, inputs = self._r6_preflight_inputs(payload, mode)
        elif self.review_revision == R5_REVIEW_REVISION:
            event, case, inputs = self._r5_preflight_inputs(payload, mode)
        else:
            event, case, inputs = self._r3_preflight_inputs(payload, mode)
        if payload.get("proposed_event_id") != inputs["proposed_event_id"]:
            raise ReviewValidationError(
                [
                    self._structured_error(
                        code="PROPOSED_EVENT_ID_MISMATCH",
                        field="proposed_event_id",
                        message="Final save must use the event ID returned by preflight.",
                        question_id="summary",
                    )
                ],
                "PROPOSED_EVENT_ID_MISMATCH",
            )
        if payload.get("idempotency_key") != inputs["idempotency_key"]:
            raise ReviewValidationError(
                [
                    self._structured_error(
                        code="IDEMPOTENCY_KEY_MISMATCH",
                        field="idempotency_key",
                        message="Final save must use the idempotency key returned by preflight.",
                        question_id="summary",
                    )
                ],
                "IDEMPOTENCY_KEY_MISMATCH",
            )
        if mode == "real" and case["tranche_id"] not in self._unlocked():
            raise ValueError("tranche is locked")
        tranche_dir = str(case.get("tranche_id") or "PRACTICE")
        event_path = root / "events" / tranche_dir / f"{event['event_id']}.json"
        ledger_path = root / "idempotency" / f"{inputs['idempotency_key']}.json"
        recovered = event_path.is_file()
        if recovered:
            event = read_json(event_path)
            if event.get("idempotency_key") != inputs["idempotency_key"]:
                raise ValueError("proposed event ID is already bound to another request")
        else:
            event["server_sequence"] = _latest_sequence(self._event_paths(mode)) + 1
            event["created_at_utc"] = utc_now()
            atomic_write(event_path, canonical_bytes(event))
        ledger = {
            "schema_version": (
                "football_intelligence.g7e_b_r6.idempotency_record.v1"
                if self.review_revision == R6_REVIEW_REVISION
                else (
                    "football_intelligence.g7e_b_r5.idempotency_record.v1"
                    if self.review_revision == R5_REVIEW_REVISION
                    else (
                        "football_intelligence.g7e_b_r4.idempotency_record.v1"
                        if self.review_revision == R4_REVIEW_REVISION
                        else "football_intelligence.g7e_b_r3.idempotency_record.v1"
                    )
                )
            ),
            "idempotency_key": inputs["idempotency_key"],
            "event_id": event["event_id"],
            "event_relative_path": str(event_path.relative_to(root)).replace("\\", "/"),
            "event_sha256": sha256_file(event_path),
            "status": "EVENT_PERSISTED",
            "updated_at_utc": utc_now(),
            "production_ready": False,
        }
        atomic_write(ledger_path, canonical_bytes(ledger))
        if payload.get("simulate_interrupt_after_event") is True:
            if not self.acceptance_mode:
                raise ValueError("interruption simulation is acceptance-only")
            raise InterruptedAcknowledgement(str(event["event_id"]))
        receipt = self._r3_acknowledge_event(root=root, event=event, event_path=event_path, case=case)
        ledger.update(
            {
                "status": "SERVER_ACKNOWLEDGED",
                "acknowledgement_receipt_id": receipt["receipt_id"],
                "acknowledgement_receipt_sha256": sha256_file(self._ack_path(root, str(event["event_id"]))),
                "updated_at_utc": utc_now(),
            }
        )
        atomic_write(ledger_path, canonical_bytes(ledger))
        draft_path = root / "drafts" / f"{event['burst_id']}.json"
        if draft_path.is_file():
            draft_path.unlink()
        tranche_receipt = self.current_tranche_receipt(str(case["tranche_id"]), create=True) if mode == "real" else None
        global_receipt = self.current_global_receipt(create=True) if tranche_receipt is not None else None
        return {
            "ok": True,
            "saved": True,
            "status": "SERVER_ACKNOWLEDGED",
            "event_id": event["event_id"],
            "acknowledgement_receipt_id": receipt["receipt_id"],
            "idempotency_key": inputs["idempotency_key"],
            "recovered_existing_event": recovered,
            "duplicate_event_created": False,
            "tranche_complete": tranche_receipt is not None,
            "tranche_completion_receipt_id": (
                tranche_receipt["tranche_completion_receipt_id"] if tranche_receipt else None
            ),
            "all_cases_complete": global_receipt is not None,
            "global_completion_receipt_id": (
                global_receipt["global_completion_receipt_id"] if global_receipt else None
            ),
            "production_ready": False,
        }

    def save_event(self, payload: Mapping[str, Any], mode: str = "real") -> dict[str, Any]:
        with self.lock:
            self._require_r5_real_release(mode)
            if self.review_revision in (
                R3_REVIEW_REVISION,
                R4_REVIEW_REVISION,
                R5_REVIEW_REVISION,
                R6_REVIEW_REVISION,
            ):
                return self._save_r3_event(payload, mode)
            event, case = self._validate_event(payload, mode)
            root = self._root(mode)
            if mode == "real" and case["tranche_id"] not in self._unlocked():
                raise ValueError("tranche is locked")
            event["server_sequence"] = _latest_sequence(self._event_paths(mode)) + 1
            tranche_dir = str(case.get("tranche_id") or "PRACTICE")
            event_path = root / "events" / tranche_dir / f"{event['event_id']}.json"
            atomic_write(event_path, canonical_bytes(event))
            receipt = {
                "schema_version": "football_intelligence.g7e_b.event_acknowledgement_receipt.v1",
                "receipt_id": f"ack-{event['event_id']}",
                "review_id": self.review_id,
                "review_revision": self.review_revision,
                "mode": mode,
                "tranche_id": case.get("tranche_id"),
                "burst_id": event["burst_id"],
                "event_id": event["event_id"],
                "event_relative_path": str(event_path.relative_to(root)).replace("\\", "/"),
                "event_byte_size": event_path.stat().st_size,
                "event_sha256": sha256_file(event_path),
                "server_validated": True,
                "case_complete": True,
                "created_at_utc": utc_now(),
                "production_ready": False,
            }
            ack_path = self._ack_path(root, event["event_id"])
            atomic_write(ack_path, canonical_bytes(receipt))
            draft_path = root / "drafts" / f"{event['burst_id']}.json"
            if draft_path.is_file():
                draft_path.unlink()
            tranche_receipt = None
            global_receipt = None
            if mode == "real":
                tranche_receipt = self.current_tranche_receipt(str(case["tranche_id"]), create=True)
                if tranche_receipt is not None:
                    global_receipt = self.current_global_receipt(create=True)
            return {
                "ok": True,
                "saved": True,
                "event_id": event["event_id"],
                "acknowledgement_receipt_id": receipt["receipt_id"],
                "tranche_complete": tranche_receipt is not None,
                "tranche_completion_receipt_id": (
                    tranche_receipt["tranche_completion_receipt_id"] if tranche_receipt else None
                ),
                "all_cases_complete": global_receipt is not None,
                "global_completion_receipt_id": (
                    global_receipt["global_completion_receipt_id"] if global_receipt else None
                ),
            }

    def final_save_status(self, mode: str, idempotency_key: str) -> dict[str, Any]:
        if self.review_revision not in (
            R3_REVIEW_REVISION,
            R4_REVIEW_REVISION,
            R5_REVIEW_REVISION,
            R6_REVIEW_REVISION,
        ):
            raise ValueError("final-save status requires the R3, R4, R5, or R6 reviewer")
        if len(idempotency_key) != 64 or any(character not in "0123456789abcdef" for character in idempotency_key):
            raise ValueError("invalid idempotency key")
        path = self._root(mode) / "idempotency" / f"{idempotency_key}.json"
        if not path.is_file():
            return {"ok": True, "status": "NOT_PERSISTED", "idempotency_key": idempotency_key}
        ledger = read_json(path)
        return {
            "ok": True,
            "status": ledger["status"],
            "idempotency_key": idempotency_key,
            "event_id": ledger.get("event_id"),
            "acknowledgement_receipt_id": ledger.get("acknowledgement_receipt_id"),
            "production_ready": False,
        }

    def state(self, mode: str = "real", requested_tranche: str | None = None) -> dict[str, Any]:
        if mode == "practice":
            latest = self.latest_events("practice")
            first = next((case for case in self.practice_cases if case["burst_id"] not in latest), None)
            final_error_path = (
                self._r3_error_path(mode, first["burst_id"])
                if first
                and self.review_revision
                in (R3_REVIEW_REVISION, R4_REVIEW_REVISION, R5_REVIEW_REVISION, R6_REVIEW_REVISION)
                else None
            )
            return {
                "review_id": self.review_id,
                "review_revision": self.review_revision,
                "mode": "practice",
                "practice": True,
                "completed_count": len(latest),
                "total_count": 3,
                "first_incomplete_burst_id": first["burst_id"] if first else None,
                "all_practice_complete": len(latest) == 3,
                "draft": self.draft("practice", first["burst_id"]) if first else None,
                "incompatible_draft": self.incompatible_draft("practice", first["burst_id"]) if first else None,
                "final_save_error": (
                    read_json(final_error_path) if final_error_path is not None and final_error_path.is_file() else None
                ),
                "human_truth": False,
            }
        latest = self.latest_events("real")
        unlocked = self._unlocked()
        tranche_id = requested_tranche if requested_tranche in unlocked else unlocked[-1]
        cases = self.by_tranche[tranche_id]
        completed_cases = [case for case in cases if case["burst_id"] in latest]
        receipt = self.current_tranche_receipt(tranche_id, create=False)
        global_receipt = self.current_global_receipt(create=False)
        first = next((case for case in cases if case["burst_id"] not in latest), None)
        final_error_path = (
            self._r3_error_path(mode, first["burst_id"])
            if first
            and self.review_revision in (R3_REVIEW_REVISION, R4_REVIEW_REVISION, R5_REVIEW_REVISION, R6_REVIEW_REVISION)
            else None
        )
        last_event = max(
            (latest[case["burst_id"]] for case in completed_cases), key=lambda row: row["server_sequence"], default=None
        )
        return {
            "review_id": self.review_id,
            "review_revision": self.review_revision,
            "mode": "real",
            "tranche_id": tranche_id,
            "unlocked_tranches": unlocked,
            "completed_count": len(completed_cases),
            "total_count": 20,
            "first_incomplete_burst_id": first["burst_id"] if first else None,
            "draft": self.draft("real", first["burst_id"]) if first else None,
            "incompatible_draft": self.incompatible_draft("real", first["burst_id"]) if first else None,
            "final_save_error": (
                read_json(final_error_path) if final_error_path is not None and final_error_path.is_file() else None
            ),
            "tranche_complete": receipt is not None,
            "tranche_completion_receipt_id": receipt["tranche_completion_receipt_id"] if receipt else None,
            "last_event_id": last_event["event_id"] if last_event else None,
            "all_cases_complete": global_receipt is not None,
            "global_completion_receipt_id": (
                global_receipt["global_completion_receipt_id"] if global_receipt else None
            ),
            "editable": receipt is None and (self.acceptance_mode or self.r5_release_gate_status()["valid"]),
            "release_gate": self.r5_release_gate_status(),
        }

    def reset_practice(self) -> dict[str, Any]:
        if self.practice.is_dir():
            shutil.rmtree(self.practice)
        return {"ok": True, "practice_reset": True, "human_event_count": len(self.latest_events("real"))}

    def acceptance_event(self, case: Mapping[str, Any], branch: str = "simple") -> dict[str, Any]:
        if self.review_revision in (R1_REVIEW_REVISION, R2_REVIEW_REVISION, R3_REVIEW_REVISION):
            return self._r1_acceptance_event(case, branch)
        visibility = ["VISIBLE_COMPLETE"] * 9
        supply = ["ONE_USEFUL_CANDIDATE"] * 9
        phases = ["NONE"] * 9
        continuity = "NOT_APPLICABLE"
        if branch == "occlusion":
            visibility = [
                "VISIBLE_COMPLETE",
                "VISIBLE_COMPLETE",
                "VISIBLE_PARTIAL",
                "FULLY_OCCLUDED_EXPECTED_PRESENT",
                "FULLY_OCCLUDED_EXPECTED_PRESENT",
                "VISIBLE_PARTIAL",
                "VISIBLE_COMPLETE",
                "VISIBLE_COMPLETE",
                "VISIBLE_COMPLETE",
            ]
            supply = [
                "ONE_USEFUL_CANDIDATE",
                "ONE_USEFUL_CANDIDATE",
                "FRAGMENT_ONLY",
                "NO_CANDIDATE",
                "NO_CANDIDATE",
                "FRAGMENT_ONLY",
                "ONE_USEFUL_CANDIDATE",
                "ONE_USEFUL_CANDIDATE",
                "ONE_USEFUL_CANDIDATE",
            ]
            phases = [
                "NONE",
                "NONE",
                "ENTERING_OCCLUSION",
                "OCCLUDED",
                "OCCLUDED",
                "EXITING_OCCLUSION",
                "NONE",
                "NONE",
                "NONE",
            ]
            continuity = "SAME_BURST_LOCAL_SUBJECT"
        subjects = []
        if branch != "no_focus":
            subject_count = 2 if branch == "multiple" else 1
            for subject_index in range(subject_count):
                subjects.append(
                    {
                        "subject_token": SUBJECT_TOKENS[subject_index],
                        "anchor_frame_sequence": 4,
                        "anchor_source_xy": [
                            case["source_width"] * (0.5 + subject_index * 0.02),
                            case["source_height"] * 0.5,
                        ],
                        "frame_observations": [
                            {
                                "frame_reference_id": frame["frame_reference_id"],
                                "visibility": visibility[index],
                                "observation_supply": supply[index],
                                "occlusion_phase": phases[index],
                            }
                            for index, frame in enumerate(case["frames"])
                        ],
                        "candidate_relationship": "SAME_PERSON_FRAGMENTS"
                        if branch == "occlusion"
                        else "NOT_APPLICABLE",
                        "continuity": continuity,
                        "role": "OUTFIELD_PLAYER",
                        "participation": "ACTIVE_IN_MATCH",
                        "certainty": "PROBABLE",
                    }
                )
        return {
            "burst_id": case["burst_id"],
            "focus_answer": "NO_RELEVANT_PERSON"
            if branch == "no_focus"
            else ("MULTIPLE_PEOPLE" if branch == "multiple" else "ONE_PERSON"),
            "subjects": subjects,
            "candidate_mappings": [],
            "whole_burst_missed_person_answer": "NO",
            "whole_burst_missed_person_marks": [],
            "source_frame_hashes": [frame["source_frame_pixel_sha256"] for frame in case["frames"]],
            "summary_confirmed": True,
            "acceptance_temporary": True,
        }

    def _r1_acceptance_event(self, case: Mapping[str, Any], branch: str) -> dict[str, Any]:
        visibility = ["VISIBLE_COMPLETE"] * 9
        if branch == "occlusion":
            visibility = [
                "VISIBLE_COMPLETE",
                "VISIBLE_COMPLETE",
                "VISIBLE_PARTIAL",
                "FULLY_OCCLUDED_EXPECTED_PRESENT",
                "FULLY_OCCLUDED_EXPECTED_PRESENT",
                "VISIBLE_PARTIAL",
                "VISIBLE_COMPLETE",
                "VISIBLE_COMPLETE",
                "VISIBLE_COMPLETE",
            ]
        subject_count = 0 if branch == "no_focus" else (2 if branch == "multiple" else 1)
        subjects = []
        for subject_index in range(subject_count):
            observations = []
            for sequence, frame in enumerate(case["frames"]):
                state = visibility[sequence]
                point = [
                    case["source_width"] * (0.48 + subject_index * 0.04),
                    case["source_height"] * 0.5,
                ]
                observation = {
                    "frame_reference_id": frame["frame_reference_id"],
                    "visibility": state,
                    "subject_location_source_x": point[0] if state in ("VISIBLE_COMPLETE", "VISIBLE_PARTIAL") else None,
                    "subject_location_source_y": point[1] if state in ("VISIBLE_COMPLETE", "VISIBLE_PARTIAL") else None,
                    "human_confirmed": state in ("VISIBLE_COMPLETE", "VISIBLE_PARTIAL"),
                    "approximate_hidden_location": False,
                    "observation_supply": (
                        "NO_USEFUL_BOX"
                        if self.review_revision in (R2_REVIEW_REVISION, R3_REVIEW_REVISION)
                        else "NO_CANDIDATE"
                    ),
                    "selected_candidate_ids": [],
                    "occlusion_phase": "OCCLUDED" if state == "FULLY_OCCLUDED_EXPECTED_PRESENT" else "NONE",
                }
                if self.review_revision == R3_REVIEW_REVISION:
                    identity = self._frame_identity(case, sequence)
                    observation["canonical_frame_identity"] = identity
                    observation["candidate_selection_binding"] = {
                        "action_type": "CANDIDATE_SELECTION",
                        "canonical_frame_identity": identity,
                        "question_id": f"subject_{subject_index}_supply_{sequence}",
                        "selected_candidate_ids": [],
                    }
                    if observation["subject_location_source_x"] is not None:
                        observation["location_binding"] = {
                            "action_type": (
                                "APPROXIMATE_HIDDEN_LOCATION"
                                if observation["approximate_hidden_location"]
                                else "SUBJECT_LOCATION"
                            ),
                            "canonical_frame_identity": identity,
                            "question_id": f"subject_{subject_index}_location_{sequence}",
                            "source_xy": [
                                observation["subject_location_source_x"],
                                observation["subject_location_source_y"],
                            ],
                            "binding_provenance": "ACCEPTANCE_TEMPORARY",
                        }
                observations.append(observation)
            subjects.append(
                {
                    "subject_token": SUBJECT_TOKENS[subject_index],
                    "subject_definition_source": "YELLOW_MULTI_PERSON_HUMAN_SELECTION"
                    if branch == "multiple"
                    else "YELLOW_ORIGINAL_FOCUS_CANDIDATE",
                    "anchor_frame_sequence": 4,
                    "anchor_source_xy": [
                        case["source_width"] * (0.48 + subject_index * 0.04),
                        case["source_height"] * 0.5,
                    ],
                    "frame_observations": observations,
                    "marker_continuity_confirmation": "SAME_SUBJECT_CONFIRMED",
                    "candidate_relationship": "NOT_APPLICABLE",
                    "continuity": "SAME_BURST_LOCAL_SUBJECT" if branch == "occlusion" else "NOT_APPLICABLE",
                    "role": "OUTFIELD_PLAYER",
                    "participation": "ACTIVE_IN_MATCH",
                    "certainty": "PROBABLE",
                }
            )
        payload = {
            "burst_id": case["burst_id"],
            "original_focus_box_answer": "NO_RELEVANT_PERSON"
            if branch == "no_focus"
            else ("MORE_THAN_ONE_RELEVANT_PERSON" if branch == "multiple" else "ONE_RELEVANT_MATCH_PERSON"),
            "context_subject_answer": "NO" if branch == "no_focus" else "NOT_APPLICABLE",
            "subjects": subjects,
            "candidate_mappings": [],
            "whole_burst_missed_person_answer": "NO",
            "whole_burst_missed_person_marks": [],
            "source_frame_hashes": [frame["source_frame_pixel_sha256"] for frame in case["frames"]],
            "summary_confirmed": True,
            "acceptance_temporary": True,
        }
        if self.review_revision in (R2_REVIEW_REVISION, R3_REVIEW_REVISION):
            payload.update(
                {
                    "candidate_runtime_contract": case["candidate_runtime_contract"],
                    "unique_frame_candidate_status": case["unique_frame_candidate_status"],
                    "per_frame_candidate_states": case["per_frame_candidate_states"],
                }
            )
        return payload

    def complete_acceptance_tranche(self, tranche_id: str) -> dict[str, Any]:
        if not self.acceptance_mode:
            raise ValueError("acceptance endpoint disabled")
        for index, case in enumerate(self.by_tranche[tranche_id]):
            if case["burst_id"] not in self.latest_events("real"):
                branch = ("no_focus", "simple", "occlusion", "multiple")[index % 4]
                if self.review_revision == R3_REVIEW_REVISION:
                    self._save_r3_acceptance_event(case, branch, "real")
                else:
                    self.save_event(self.acceptance_event(case, branch), "real")
        receipt = self.current_tranche_receipt(tranche_id, create=False)
        return {
            "ok": receipt is not None,
            "tranche_id": tranche_id,
            "tranche_completion_receipt_id": receipt["tranche_completion_receipt_id"] if receipt else None,
        }

    def complete_acceptance_practice(self) -> dict[str, Any]:
        if not self.acceptance_mode:
            raise ValueError("acceptance endpoint disabled")
        branches = ("simple", "occlusion", "multiple")
        for case, branch in zip(self.practice_cases, branches, strict=True):
            if case["burst_id"] not in self.latest_events("practice"):
                if self.review_revision == R3_REVIEW_REVISION:
                    self._save_r3_acceptance_event(case, branch, "practice")
                else:
                    self.save_event(self.acceptance_event(case, branch), "practice")
        return {
            "ok": True,
            "practice_event_count": len(self.latest_events("practice")),
            "human_event_count": len(self.latest_events("real")),
        }

    def _save_r3_acceptance_event(self, case: Mapping[str, Any], branch: str, mode: str) -> dict[str, Any]:
        if not self.acceptance_mode or self.review_revision != R3_REVIEW_REVISION:
            raise ValueError("R3 acceptance event helper is disabled")
        event = self.acceptance_event(case, branch)
        event["mode"] = mode
        draft = self.save_draft(
            {
                "mode": mode,
                "burst_id": case["burst_id"],
                "current_question": "summary",
                "current_frame_sequence": 4,
                "playback_speed": 1.0,
                "answers": {
                    "original_focus_box_answer": event["original_focus_box_answer"],
                    "context_subject_answer": event["context_subject_answer"],
                    "missed_check": event["whole_burst_missed_person_answer"],
                },
                "subjects": event["subjects"],
                "candidate_mappings": event["candidate_mappings"],
                "missed_person_marks": event["whole_burst_missed_person_marks"],
                "click_transactions": [],
                "action_journal": [{"action": "ACCEPTANCE_TEMPORARY_EVENT"}],
                "draft_version": 0,
                "optimistic_lock_token": None,
            },
            mode,
        )
        event.update(
            {
                "draft_version": draft["draft_version"],
                "draft_content_sha256": draft["draft_content_sha256"],
                "optimistic_lock_token": draft["optimistic_lock_token"],
                "click_transactions": [],
            }
        )
        preflight = self.final_save_preflight(event, mode)
        if preflight.get("status") != "READY_TO_PERSIST":
            raise ValueError("R3 acceptance event preflight failed")
        event["proposed_event_id"] = preflight["proposed_event_id"]
        event["idempotency_key"] = preflight["idempotency_key"]
        return self.save_event(event, mode)


def create_server(
    package: Path,
    decisions_root: Path | None = None,
    practice_root: Path | None = None,
    asset_root: Path | None = None,
    port: int = 8818,
    acceptance_mode: bool = False,
) -> ThreadingHTTPServer:
    package = package.resolve()
    resolved_asset_root = (asset_root or package / "assets").resolve()
    store = TemporalReviewStore(package, decisions_root, practice_root, acceptance_mode)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def send_bytes(self, status: int, data: bytes, content_type: str) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return

        def send_json(self, status: int, payload: Any) -> None:
            self.send_bytes(status, canonical_bytes(payload), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            route = parsed.path
            query = parse_qs(parsed.query)
            if route == "/":
                return self.send_bytes(200, (package / "index.html").read_bytes(), "text/html; charset=utf-8")
            if route == "/review.js":
                return self.send_bytes(200, (package / "review.js").read_bytes(), "text/javascript; charset=utf-8")
            if route == "/viewport_transform.js":
                return self.send_bytes(
                    200,
                    (package / "viewport_transform.js").read_bytes(),
                    "text/javascript; charset=utf-8",
                )
            if route == "/review.css":
                return self.send_bytes(200, (package / "review.css").read_bytes(), "text/css; charset=utf-8")
            if route == "/generated_client_contract.js":
                return self.send_bytes(
                    200,
                    (package / "generated_client_contract.js").read_bytes(),
                    "text/javascript; charset=utf-8",
                )
            if route == "/api/bootstrap":
                mode = query.get("mode", ["real"])[0]
                tranche_id = query.get("tranche", [None])[0]
                cases = store.practice_cases if mode == "practice" else store.cases
                return self.send_json(
                    200,
                    {
                        "acceptance_temporary": store.acceptance_mode,
                        "cases": cases,
                        "state": store.state(mode, tranche_id),
                        "relationship_compatibility": store.relationship_contract,
                        "relationship_compatibility_sha256": store.relationship_compatibility_sha256,
                        "canonical_contract": store.canonical_contract,
                        "canonical_contract_sha256": store.canonical_contract_sha256,
                        "server_action_contract": store.action_contract,
                        "server_action_contract_sha256": store.action_contract_sha256,
                        "release_gate": store.r5_release_gate_status(),
                    },
                )
            if route == "/api/review-state":
                mode = query.get("mode", ["real"])[0]
                tranche_id = query.get("tranche", [None])[0]
                state = store.state(mode, tranche_id)
                print(
                    f"GET /api/review-state HTTP 200 mode={mode} burst={state.get('first_incomplete_burst_id')}",
                    flush=True,
                )
                return self.send_json(200, state)
            if route == "/api/final-save-status":
                mode = query.get("mode", ["real"])[0]
                key = query.get("idempotency_key", [""])[0]
                return self.send_json(200, store.final_save_status(mode, key))
            if route == "/api/acknowledged-event":
                mode = query.get("mode", ["real"])[0]
                event_id = query.get("event_id", [""])[0]
                return self.send_json(200, store.acknowledged_event(mode, event_id))
            if route == "/api/completed":
                tranche_id = query.get("tranche", [""])[0]
                if tranche_id not in TRANCHES or store.current_tranche_receipt(tranche_id, create=False) is None:
                    return self.send_json(409, {"error": "tranche is not complete"})
                latest = store.latest_events("real")
                events = [latest[case["burst_id"]] for case in store.by_tranche[tranche_id]]
                return self.send_json(200, {"tranche_id": tranche_id, "read_only": True, "events": events})
            if route.startswith("/api/candidate-state/"):
                frame_reference_id = unquote(route.removeprefix("/api/candidate-state/"))
                state = store.candidate_states_by_reference.get(frame_reference_id)
                if state is None:
                    return self.send_json(404, {"error": "candidate state not found"})
                return self.send_json(200, state)
            if route.startswith("/assets/"):
                relative = Path(unquote(route.removeprefix("/assets/")))
                try:
                    candidate = contained_path(resolved_asset_root, relative)
                except ValueError:
                    return self.send_json(404, {"error": "asset not found"})
                if not candidate.is_file():
                    return self.send_json(404, {"error": "asset not found"})
                mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
                return self.send_bytes(200, candidate.read_bytes(), mime)
            if route.startswith("/review-assets/"):
                relative = Path(unquote(route.removeprefix("/review-assets/")))
                visual_root = contained_path(package, "review_assets")
                try:
                    candidate = contained_path(visual_root, relative)
                except ValueError:
                    return self.send_json(404, {"error": "review asset not found"})
                if not candidate.is_file():
                    return self.send_json(404, {"error": "review asset not found"})
                mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
                if not mime.startswith("image/"):
                    return self.send_json(415, {"error": "review asset is not an image"})
                return self.send_bytes(200, candidate.read_bytes(), mime)
            return self.send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = read_json_request(self)
                route = urlparse(self.path).path
                mode = str(payload.get("mode", "real"))
                if route == "/api/relationship-compatibility":
                    cases = store.practice_by_id if mode == "practice" else store.by_id
                    case = cases.get(str(payload.get("burst_id", "")))
                    if case is None:
                        raise ValueError("unknown burst relationship check")
                    result = store.relationship_compatibility(
                        case=case,
                        subject_token=str(payload.get("subject_token", "")),
                        subject_index=int(payload.get("subject_index", -1)),
                        sequence=int(payload.get("frame_sequence", -1)),
                        observation=payload.get("observation", {}),
                        final=payload.get("final") is True,
                    )
                    return self.send_json(200, {"ok": not result["errors"], **result})
                if route == "/api/draft":
                    draft = store.save_draft(payload, mode)
                    print(
                        "POST /api/draft HTTP 200 "
                        f"mode={mode} burst={draft['burst_id']} draft_revision={draft.get('draft_version')}",
                        flush=True,
                    )
                    return self.send_json(200, {"ok": True, "draft": draft})
                if route == "/api/action":
                    with store.lock:
                        result = store.apply_browser_action(payload, mode)
                    print(
                        "POST /api/action HTTP 200 "
                        f"mode={mode} burst={payload.get('burst_id')} action={payload.get('action_type')} "
                        f"draft_revision={result['draft'].get('draft_version')} "
                        f"receipt={result.get('action_receipt_id')}",
                        flush=True,
                    )
                    return self.send_json(200, result)
                if route == "/api/initialize-draft":
                    draft = store.initialize_draft(mode, str(payload.get("burst_id", "")))
                    print(
                        "POST /api/initialize-draft HTTP 200 "
                        f"mode={mode} burst={draft['burst_id']} draft_revision={draft.get('draft_version')}",
                        flush=True,
                    )
                    return self.send_json(200, {"ok": True, "draft": draft})
                if route == "/api/final-save-preflight":
                    result = store.final_save_preflight(payload, mode)
                    status = 200 if result.get("ok") else 422
                    print(
                        "POST /api/final-save-preflight "
                        f"HTTP {status} mode={mode} burst={payload.get('burst_id')} "
                        f"draft_revision={payload.get('draft_version')} "
                        f"error_code={result.get('error_code')}",
                        flush=True,
                    )
                    return self.send_json(status, result)
                if route == "/api/save":
                    result = store.save_event(payload, mode)
                    print(
                        "POST /api/save HTTP 200 "
                        f"mode={mode} burst={payload.get('burst_id')} "
                        f"draft_revision={payload.get('draft_version')} event_id={result.get('event_id')} "
                        f"acknowledgement_id={result.get('acknowledgement_receipt_id')}",
                        flush=True,
                    )
                    return self.send_json(200, result)
                if route == "/api/practice/reset":
                    return self.send_json(200, store.reset_practice())
                if route == "/api/tranche/start-next":
                    return self.send_json(200, store.unlock_next(str(payload.get("tranche_id", ""))))
                if route == "/api/acceptance/complete-tranche":
                    return self.send_json(200, store.complete_acceptance_tranche(str(payload.get("tranche_id", ""))))
                if route == "/api/acceptance/complete-practice":
                    return self.send_json(200, store.complete_acceptance_practice())
                return self.send_json(404, {"error": "not found"})
            except RequestBodyTooLarge as exc:
                return self.send_json(413, {"ok": False, "error_code": "REQUEST_BODY_TOO_LARGE", "error": str(exc)})
            except UnsupportedMediaType as exc:
                return self.send_json(415, {"ok": False, "error_code": "UNSUPPORTED_MEDIA_TYPE", "error": str(exc)})
            except InterruptedAcknowledgement as exc:
                print(
                    f"POST /api/save HTTP 503 event_id={exc.event_id} error_code=ACKNOWLEDGEMENT_INTERRUPTED",
                    flush=True,
                )
                return self.send_json(
                    503,
                    {
                        "ok": False,
                        "error_code": "ACKNOWLEDGEMENT_INTERRUPTED",
                        "error": str(exc),
                        "event_id": exc.event_id,
                        "retry_same_idempotency_key": True,
                    },
                )
            except StaleDraftError as exc:
                canonical = exc.canonical_draft
                return self.send_json(
                    409,
                    {
                        "ok": False,
                        "error_code": exc.error_code,
                        "error": str(exc),
                        "errors": exc.errors,
                        "canonical_draft": canonical,
                        "canonical_metadata": {
                            "burst_id": canonical.get("burst_id"),
                            "tranche_id": canonical.get("tranche_id"),
                            "current_question": canonical.get("current_question"),
                            "current_question_instance_key": canonical.get("current_question_instance_key"),
                            "draft_version": canonical.get("draft_version"),
                            "draft_content_sha256": canonical.get("draft_content_sha256"),
                            "optimistic_lock_token": canonical.get("optimistic_lock_token"),
                        },
                        "rejected_action_replayed": False,
                    },
                )
            except ReviewValidationError as exc:
                print(
                    f"POST {urlparse(self.path).path} HTTP 409 error_code={exc.error_code}",
                    flush=True,
                )
                return self.send_json(
                    409,
                    {
                        "ok": False,
                        "error_code": exc.error_code,
                        "error": str(exc),
                        "errors": exc.errors,
                    },
                )
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                return self.send_json(400, {"ok": False, "error": str(exc)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(
    package: Path,
    decisions_root: Path | None = None,
    practice_root: Path | None = None,
    asset_root: Path | None = None,
    port: int = 8818,
    acceptance_mode: bool = False,
) -> None:
    server = create_server(package, decisions_root, practice_root, asset_root, port, acceptance_mode)
    print(f"G7E-B temporal reviewer: http://127.0.0.1:{server.server_port}/", flush=True)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--decisions-root", type=Path)
    parser.add_argument("--practice-root", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--port", type=int, default=8818)
    parser.add_argument("--acceptance-mode", action="store_true")
    args = parser.parse_args()
    serve(args.package, args.decisions_root, args.practice_root, args.asset_root, args.port, args.acceptance_mode)


if __name__ == "__main__":
    main()
