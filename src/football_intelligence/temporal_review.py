"""G7E-B deterministic tranche assignment and temporal-review persistence."""

from __future__ import annotations

import argparse
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

from football_intelligence.temporal_burst_selection import CLASS_PRIORITY, MATCHES, QUOTAS

REVIEW_ID = "G7E_B_TEMPORAL_BURST_REVIEW"
REVIEW_REVISION = "G7E_B_TEMPORAL_BURST_REVIEW_V1"
R1_REVIEW_REVISION = "G7E_B_R1_SUBJECT_GUIDANCE_AND_ZOOM_REPAIR_V1"
SUPPORTED_REVIEW_REVISIONS = (REVIEW_REVISION, R1_REVIEW_REVISION)
PROTOCOL_ID = "G7E_A_BURST_LOCAL_TEMPORAL_OBSERVATION_PROTOCOL_V1"
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
    "UNCERTAIN",
)
OCCLUSION_PHASES = ("NONE", "ENTERING_OCCLUSION", "OCCLUDED", "EXITING_OCCLUSION", "UNCERTAIN")
CONTINUITY = ("SAME_BURST_LOCAL_SUBJECT", "DIFFERENT_SUBJECT", "CANNOT_TELL", "NOT_APPLICABLE")
ROLES = ("OUTFIELD_PLAYER", "GOALKEEPER", "RELEVANT_OFFICIAL", "OTHER_PERSON", "UNKNOWN_ROLE")
PARTICIPATION = ("ACTIVE_IN_MATCH", "WARMING_OR_INACTIVE", "NOT_PLAYER_OR_OFFICIAL", "UNKNOWN_PARTICIPATION")
CERTAINTY = ("CERTAIN", "PROBABLE", "NOT_SURE")
SUBJECT_TOKENS = ("SUBJECT_A", "SUBJECT_B", "SUBJECT_C")


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
        os.replace(temporary, path)
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
    ):
        self.package = package.resolve()
        self.decisions = (decisions_root or package / "human_decisions").resolve()
        self.practice = (practice_root or package / "practice_decisions").resolve()
        self.acceptance_mode = acceptance_mode
        self.lock = threading.RLock()
        review = read_json(self.package / "review_cases.json")
        practice = read_json(self.package / "practice_cases.json")
        self.review_id = str(review.get("review_id", REVIEW_ID))
        self.review_revision = str(review.get("review_revision", REVIEW_REVISION))
        if self.review_id != REVIEW_ID or self.review_revision not in SUPPORTED_REVIEW_REVISIONS:
            raise ValueError("unsupported temporal-review identity or revision")
        self.cases = list(review["cases"])
        self.practice_cases = list(practice["cases"])
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
        self._validate_draft(document)
        atomic_write(self._root(mode) / "drafts" / f"{burst_id}.json", canonical_bytes(document))
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
        if self.review_revision == R1_REVIEW_REVISION:
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
        for subject in subjects:
            if subject.get("subject_definition_source") not in allowed_definition_sources:
                raise ValueError("invalid subject definition source")
            if subject.get("role") not in ROLES or subject.get("participation") not in PARTICIPATION:
                raise ValueError("invalid role or participation")
            if subject.get("certainty") not in CERTAINTY:
                raise ValueError("invalid certainty")
            if subject.get("candidate_relationship") not in (*RELATIONSHIPS, "NOT_APPLICABLE"):
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
                if visibility not in VISIBILITY or supply not in SUPPLY:
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

    def save_event(self, payload: Mapping[str, Any], mode: str = "real") -> dict[str, Any]:
        with self.lock:
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

    def state(self, mode: str = "real", requested_tranche: str | None = None) -> dict[str, Any]:
        if mode == "practice":
            latest = self.latest_events("practice")
            first = next((case for case in self.practice_cases if case["burst_id"] not in latest), None)
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
            "tranche_complete": receipt is not None,
            "tranche_completion_receipt_id": receipt["tranche_completion_receipt_id"] if receipt else None,
            "last_event_id": last_event["event_id"] if last_event else None,
            "all_cases_complete": global_receipt is not None,
            "global_completion_receipt_id": (
                global_receipt["global_completion_receipt_id"] if global_receipt else None
            ),
            "editable": receipt is None,
        }

    def reset_practice(self) -> dict[str, Any]:
        if self.practice.is_dir():
            shutil.rmtree(self.practice)
        return {"ok": True, "practice_reset": True, "human_event_count": len(self.latest_events("real"))}

    def acceptance_event(self, case: Mapping[str, Any], branch: str = "simple") -> dict[str, Any]:
        if self.review_revision == R1_REVIEW_REVISION:
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
                observations.append(
                    {
                        "frame_reference_id": frame["frame_reference_id"],
                        "visibility": state,
                        "subject_location_source_x": point[0]
                        if state in ("VISIBLE_COMPLETE", "VISIBLE_PARTIAL")
                        else None,
                        "subject_location_source_y": point[1]
                        if state in ("VISIBLE_COMPLETE", "VISIBLE_PARTIAL")
                        else None,
                        "human_confirmed": state in ("VISIBLE_COMPLETE", "VISIBLE_PARTIAL"),
                        "approximate_hidden_location": False,
                        "observation_supply": "NO_CANDIDATE",
                        "selected_candidate_ids": [],
                        "occlusion_phase": "OCCLUDED" if state == "FULLY_OCCLUDED_EXPECTED_PRESENT" else "NONE",
                    }
                )
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
        return {
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

    def complete_acceptance_tranche(self, tranche_id: str) -> dict[str, Any]:
        if not self.acceptance_mode:
            raise ValueError("acceptance endpoint disabled")
        for index, case in enumerate(self.by_tranche[tranche_id]):
            if case["burst_id"] not in self.latest_events("real"):
                branch = ("no_focus", "simple", "occlusion", "multiple")[index % 4]
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
                self.save_event(self.acceptance_event(case, branch), "practice")
        return {
            "ok": True,
            "practice_event_count": len(self.latest_events("practice")),
            "human_event_count": len(self.latest_events("real")),
        }


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
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

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
            if route == "/review.css":
                return self.send_bytes(200, (package / "review.css").read_bytes(), "text/css; charset=utf-8")
            if route == "/api/bootstrap":
                mode = query.get("mode", ["real"])[0]
                tranche_id = query.get("tranche", [None])[0]
                cases = store.practice_cases if mode == "practice" else store.cases
                return self.send_json(200, {"cases": cases, "state": store.state(mode, tranche_id)})
            if route == "/api/completed":
                tranche_id = query.get("tranche", [""])[0]
                if tranche_id not in TRANCHES or store.current_tranche_receipt(tranche_id, create=False) is None:
                    return self.send_json(409, {"error": "tranche is not complete"})
                latest = store.latest_events("real")
                events = [latest[case["burst_id"]] for case in store.by_tranche[tranche_id]]
                return self.send_json(200, {"tranche_id": tranche_id, "read_only": True, "events": events})
            if route.startswith("/assets/"):
                relative = Path(unquote(route.removeprefix("/assets/")))
                candidate = (resolved_asset_root / relative).resolve()
                if resolved_asset_root not in candidate.parents or not candidate.is_file():
                    return self.send_json(404, {"error": "asset not found"})
                mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
                return self.send_bytes(200, candidate.read_bytes(), mime)
            return self.send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                route = urlparse(self.path).path
                mode = str(payload.get("mode", "real"))
                if route == "/api/draft":
                    return self.send_json(200, {"ok": True, "draft": store.save_draft(payload, mode)})
                if route == "/api/save":
                    return self.send_json(200, store.save_event(payload, mode))
                if route == "/api/practice/reset":
                    return self.send_json(200, store.reset_practice())
                if route == "/api/tranche/start-next":
                    return self.send_json(200, store.unlock_next(str(payload.get("tranche_id", ""))))
                if route == "/api/acceptance/complete-tranche":
                    return self.send_json(200, store.complete_acceptance_tranche(str(payload.get("tranche_id", ""))))
                if route == "/api/acceptance/complete-practice":
                    return self.send_json(200, store.complete_acceptance_practice())
                return self.send_json(404, {"error": "not found"})
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
