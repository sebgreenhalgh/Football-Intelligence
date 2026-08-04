"""Genuine persisted draft, action, event, and receipt invariant scanner."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from football_intelligence.g7e_b_r6_action_reducer import applicable_question_sequence, validate_r6_invariants
from football_intelligence.temporal_reviewer.contracts import canonical_action_uuid, contained_path
from football_intelligence.temporal_reviewer.persistence import finalized_draft_target_is_superseded


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"persisted JSON is not an object: {path}")
    return value


class PersistedInvariantScanner:
    """Inspect actual files under one reviewer root; never synthesize drafts."""

    def __init__(self, store: Any, mode: str):
        self.store = store
        self.mode = mode
        self.root = store._root(mode).resolve()
        self.cases = store.practice_by_id if mode == "practice" else store.by_id
        self.discrepancies: list[dict[str, Any]] = []
        self.counts: Counter[str] = Counter()

    def error(self, kind: str, path: Path, detail: str) -> None:
        self.discrepancies.append(
            {"kind": kind, "relative_path": path.relative_to(self.root).as_posix(), "detail": detail}
        )

    def scan_drafts(self) -> None:
        directory = contained_path(self.root, "drafts")
        for path in sorted(directory.glob("*.json")) if directory.exists() else []:
            self.counts["drafts"] += 1
            draft = _read(path)
            case = self.cases.get(str(draft.get("burst_id", "")))
            if case is None:
                self.error("UNKNOWN_DRAFT_BURST", path, "draft burst is absent from the frozen case set")
                continue
            answers = draft.get("answered_domain_values", {})
            lifecycle = draft.get("question_lifecycle", {})
            if self.store.canonical_contract is not None:
                for detail in validate_r6_invariants(draft, self.store.canonical_contract):
                    self.error("R6_DRAFT_INVARIANT_FAILURE", path, detail)
            for key in answers:
                self.counts["domain_answers"] += 1
                if lifecycle.get(key) != "ANSWERED":
                    self.error("ANSWER_WITHOUT_ANSWERED_LIFECYCLE", path, key)
            for key, state in lifecycle.items():
                self.counts["question_lifecycles"] += 1
                if state == "ANSWERED" and key not in answers and not key.endswith("|summary"):
                    self.error("ANSWERED_LIFECYCLE_WITHOUT_DOMAIN_VALUE", path, key)
            if self.store.canonical_contract is not None:
                applicable = set(applicable_question_sequence(draft, self.store.canonical_contract))
                for key, value in answers.items():
                    if key not in applicable:
                        self.error("HIDDEN_STALE_ANSWER", path, f"{key}={value!r}")
                if draft.get("summary_ready"):
                    for key in applicable:
                        if key.endswith("|summary"):
                            continue
                        if lifecycle.get(key) not in {"ANSWERED", "SKIPPED_NOT_APPLICABLE"}:
                            self.error("SUMMARY_FIELD_NOT_RESOLVED", path, key)
            for subject in draft.get("subjects", []):
                for sequence, observation in enumerate(subject.get("frame_observations", [])):
                    frozen_ids = {row["candidate_id"] for row in case["frame_candidates"][sequence]}
                    for candidate_id in observation.get("selected_candidate_ids", []):
                        self.counts["selected_candidate_ids"] += 1
                        if candidate_id not in frozen_ids:
                            self.error("CANDIDATE_FRAME_BINDING_MISMATCH", path, candidate_id)
                    for x_key, y_key in (("subject_location_source_x", "subject_location_source_y"),):
                        if x_key in observation or y_key in observation:
                            x, y = observation.get(x_key), observation.get(y_key)
                            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                                self.error("INVALID_SUBJECT_COORDINATE", path, f"frame {sequence}")
                            elif not 0 <= x <= case["source_width"] or not 0 <= y <= case["source_height"]:
                                self.error("SUBJECT_COORDINATE_OUT_OF_BOUNDS", path, f"frame {sequence}")
            for mark in draft.get("missed_person_marks", []):
                self.counts["missed_person_marks"] += 1
                sequence = mark.get("frame_sequence")
                point = mark.get("source_xy")
                if not isinstance(sequence, int) or not 0 <= sequence < 9:
                    self.error("MISSED_MARK_FRAME_BINDING_MISMATCH", path, str(sequence))
                if not isinstance(point, list) or len(point) != 2:
                    self.error("INVALID_MISSED_MARK_COORDINATE", path, repr(point))
                elif not 0 <= point[0] <= case["source_width"] or not 0 <= point[1] <= case["source_height"]:
                    self.error("MISSED_MARK_COORDINATE_OUT_OF_BOUNDS", path, repr(point))

    def scan_actions(self) -> None:
        receipt_dir = contained_path(self.root, "receipts", "actions")
        ledger_dir = contained_path(self.root, "action_idempotency")
        receipts = sorted(receipt_dir.glob("*.json")) if receipt_dir.exists() else []
        ledgers = sorted(ledger_dir.glob("*.json")) if ledger_dir.exists() else []
        receipt_ids: list[str] = []
        ledger_ids: list[str] = []
        for path in receipts:
            self.counts["action_receipts"] += 1
            receipt = _read(path)
            try:
                action_id = canonical_action_uuid(receipt.get("action_id"), "action receipt action_id")
            except ValueError as exc:
                self.error("INVALID_ACTION_RECEIPT_ID", path, str(exc))
                continue
            receipt_ids.append(action_id)
            ledger = contained_path(self.root, "action_idempotency", f"{action_id}.json")
            if not ledger.is_file():
                self.error("ACTION_RECEIPT_WITHOUT_LEDGER", path, action_id)
        for path in ledgers:
            self.counts["action_ledgers"] += 1
            ledger = _read(path)
            try:
                action_id = canonical_action_uuid(ledger.get("action_id"), "action ledger action_id")
                key = canonical_action_uuid(ledger.get("idempotency_key"), "action ledger idempotency_key")
            except ValueError as exc:
                self.error("INVALID_ACTION_LEDGER_ID", path, str(exc))
                continue
            ledger_ids.append(action_id)
            if key != action_id:
                self.error("ACTION_LEDGER_KEY_MISMATCH", path, action_id)
            transaction_path = contained_path(self.root, "action_transactions", f"{action_id}.json")
            if transaction_path.is_file():
                if not ledger.get("action_semantic_sha256") or not ledger.get("action_envelope_sha256"):
                    self.error("R6_1_ACTION_LEDGER_PROVENANCE_MISSING", path, action_id)
            elif not ledger.get("action_semantic_sha256"):
                self.counts["legacy_action_ledgers_without_semantic_hash"] += 1
            receipt = contained_path(self.root, "receipts", "actions", f"action-ack-{action_id}.json")
            if not receipt.is_file():
                self.error("ACTION_LEDGER_WITHOUT_RECEIPT", path, action_id)
            elif ledger.get("action_receipt_sha256") != hashlib.sha256(receipt.read_bytes()).hexdigest():
                self.error("ACTION_LEDGER_RECEIPT_HASH_MISMATCH", path, action_id)
        for values, label in ((receipt_ids, "RECEIPT"), (ledger_ids, "LEDGER")):
            for duplicate in (value for value, count in Counter(values).items() if count != 1):
                self.error(f"DUPLICATE_ACTION_{label}_ID", self.root, duplicate)

        transaction_dir = contained_path(self.root, "action_transactions")
        for path in sorted(transaction_dir.glob("*.json")) if transaction_dir.exists() else []:
            self.counts["action_transactions"] += 1
            transaction = _read(path)
            if transaction.get("state") != "COMMITTED":
                self.error("UNCOMMITTED_ACTION_TRANSACTION", path, str(transaction.get("state")))
            targets = transaction.get("targets", [])
            if not isinstance(targets, list) or len(targets) != 3:
                self.error("ACTION_TRANSACTION_TARGET_CARDINALITY", path, repr(targets))
                continue
            for target in targets:
                try:
                    destination = contained_path(self.root, str(target["relative_path"]))
                except (KeyError, TypeError, ValueError) as exc:
                    self.error("ACTION_TRANSACTION_TARGET_PATH", path, str(exc))
                    continue
                if not destination.is_file():
                    if finalized_draft_target_is_superseded(self.root, target):
                        self.counts["finalized_draft_transaction_targets"] += 1
                    else:
                        self.error("ACTION_TRANSACTION_TARGET_MISSING", path, str(target.get("relative_path")))
                elif hashlib.sha256(destination.read_bytes()).hexdigest() != target.get("sha256"):
                    if target.get("label") == "draft":
                        current = _read(destination)
                        if int(current.get("draft_version", -1)) <= int(transaction.get("next_draft_revision", -1)):
                            self.error("ACTION_TRANSACTION_DRAFT_HISTORY_MISMATCH", path, destination.name)
                    else:
                        self.error("ACTION_TRANSACTION_TARGET_HASH_MISMATCH", path, str(target.get("relative_path")))

    def _validate_event_provenance(self, event: dict[str, Any], path: Path) -> None:
        burst_id = str(event.get("burst_id", ""))
        case = self.cases.get(burst_id)
        if case is None:
            self.error("UNKNOWN_EVENT_BURST", path, burst_id)
            return
        expected_hashes = [str(frame["source_frame_pixel_sha256"]) for frame in case["frames"]]
        if event.get("source_frame_hashes") != expected_hashes:
            self.error("EVENT_SOURCE_FRAME_HASH_MISMATCH", path, burst_id)
        candidate_sets = [
            {str(candidate["candidate_id"]) for candidate in frame_candidates}
            for frame_candidates in case["frame_candidates"]
        ]
        for mapping in event.get("candidate_mappings", []):
            sequence = mapping.get("frame_sequence")
            candidate_id = str(mapping.get("candidate_id", ""))
            if not isinstance(sequence, int) or not 0 <= sequence < len(candidate_sets):
                self.error("EVENT_CANDIDATE_FRAME_SEQUENCE", path, repr(sequence))
            elif candidate_id not in candidate_sets[sequence]:
                self.error("EVENT_CANDIDATE_FRAME_BINDING", path, candidate_id)
        for subject in event.get("subjects", []):
            for sequence, observation in enumerate(subject.get("frame_observations", [])):
                if sequence >= len(candidate_sets):
                    self.error("EVENT_SUBJECT_FRAME_SEQUENCE", path, repr(sequence))
                    continue
                for candidate_id in observation.get("selected_candidate_ids", []):
                    if str(candidate_id) not in candidate_sets[sequence]:
                        self.error("EVENT_SELECTED_CANDIDATE_FRAME_BINDING", path, str(candidate_id))
        for mark in event.get("whole_burst_missed_person_marks", []):
            sequence = mark.get("frame_sequence")
            point = mark.get("source_xy")
            if not isinstance(sequence, int) or not 0 <= sequence < len(expected_hashes):
                self.error("EVENT_MISSED_MARK_FRAME_BINDING", path, repr(sequence))
            if not isinstance(point, list) or len(point) != 2:
                self.error("EVENT_MISSED_MARK_COORDINATE", path, repr(point))
            elif not 0 <= point[0] <= case["source_width"] or not 0 <= point[1] <= case["source_height"]:
                self.error("EVENT_MISSED_MARK_OUT_OF_BOUNDS", path, repr(point))

    def scan_events(self) -> None:
        event_directory = contained_path(self.root, "events")
        event_paths = sorted(event_directory.glob("**/*.json")) if event_directory.exists() else []
        seen: Counter[str] = Counter()
        acknowledgement_ids: Counter[str] = Counter()
        for path in event_paths:
            self.counts["events"] += 1
            event = _read(path)
            event_id = str(event.get("event_id", ""))
            seen[event_id] += 1
            acknowledgement_directory = contained_path(self.root, "receipts", "acknowledgements")
            acknowledgements = list(acknowledgement_directory.glob(f"ack-{event_id}.json"))
            if len(acknowledgements) != 1:
                self.error("EVENT_ACKNOWLEDGEMENT_CARDINALITY", path, f"{event_id}: {len(acknowledgements)}")
            else:
                self.counts["acknowledgements"] += 1
                acknowledgement = _read(acknowledgements[0])
                acknowledgement_ids[str(acknowledgement.get("receipt_id", ""))] += 1
                if acknowledgement.get("event_id") != event_id:
                    self.error("ACKNOWLEDGEMENT_EVENT_MISMATCH", acknowledgements[0], event_id)
                if acknowledgement.get("event_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
                    self.error("ACKNOWLEDGEMENT_EVENT_HASH_MISMATCH", acknowledgements[0], event_id)
                if acknowledgement.get("event_byte_size") != path.stat().st_size:
                    self.error("ACKNOWLEDGEMENT_EVENT_SIZE_MISMATCH", acknowledgements[0], event_id)
                if acknowledgement.get("event_relative_path") != path.relative_to(self.root).as_posix():
                    self.error("ACKNOWLEDGEMENT_EVENT_PATH_MISMATCH", acknowledgements[0], event_id)
                valid_complete = (
                    acknowledgement.get("server_validated") is True and acknowledgement.get("case_complete") is True
                )
                if not valid_complete:
                    self.error("ACKNOWLEDGEMENT_NOT_VALID_COMPLETE", acknowledgements[0], event_id)
            self._validate_event_provenance(event, path)
        for event_id, count in seen.items():
            if count != 1:
                self.error("DUPLICATE_EVENT_ID", self.root, f"{event_id}: {count}")
        acknowledgement_directory = contained_path(self.root, "receipts", "acknowledgements")
        all_acknowledgements = (
            sorted(acknowledgement_directory.glob("ack-*.json")) if acknowledgement_directory.exists() else []
        )
        if len(all_acknowledgements) != self.counts["acknowledgements"]:
            self.error(
                "ORPHAN_ACKNOWLEDGEMENT",
                self.root,
                f"files={len(all_acknowledgements)} matched={self.counts['acknowledgements']}",
            )
        for receipt_id, count in acknowledgement_ids.items():
            if count != 1:
                self.error("DUPLICATE_ACKNOWLEDGEMENT_ID", self.root, f"{receipt_id}: {count}")

    def scan_completion_receipts(self) -> None:
        if self.mode != "real":
            return
        tranche_directory = contained_path(self.root, "receipts", "tranche_completion")
        global_directory = contained_path(self.root, "receipts", "global_completion")
        self.counts["tranche_receipt_files"] = (
            len(list(tranche_directory.glob("*.json"))) if tranche_directory.exists() else 0
        )
        self.counts["global_receipt_files"] = (
            len(list(global_directory.glob("*.json"))) if global_directory.exists() else 0
        )
        for tranche_id in sorted({str(case["tranche_id"]) for case in self.cases.values()}):
            try:
                receipt = self.store.current_tranche_receipt(tranche_id, create=False)
            except Exception as exc:
                self.error("CURRENT_TRANCHE_RECEIPT_INVALID", self.root, f"{tranche_id}: {exc}")
                continue
            if receipt is not None:
                self.counts["current_tranche_receipts"] += 1
        try:
            global_receipt = self.store.current_global_receipt(create=False)
        except Exception as exc:
            self.error("CURRENT_GLOBAL_RECEIPT_INVALID", self.root, str(exc))
        else:
            if global_receipt is not None:
                self.counts["current_global_receipts"] += 1

    def run(self) -> dict[str, Any]:
        self.scan_drafts()
        self.scan_actions()
        self.scan_events()
        self.scan_completion_receipts()
        return {
            "schema_version": "football_intelligence.g7e_b_r6_1.persisted_invariant_scan.v1",
            "mode": self.mode,
            "root": str(self.root),
            "inspected_counts": dict(sorted(self.counts.items())),
            "discrepancy_count": len(self.discrepancies),
            "discrepancies": self.discrepancies,
            "passed": not self.discrepancies,
            "synthetic_completed_drafts_used": False,
            "production_ready": False,
        }


def scan_persisted_invariants(store: Any, mode: str = "real") -> dict[str, Any]:
    return PersistedInvariantScanner(store, mode).run()
