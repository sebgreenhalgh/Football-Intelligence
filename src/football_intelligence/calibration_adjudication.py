"""Append-only superseding adjudication for the six G7F-C calibration frames.

Original FIRST_PASS events and acknowledgements are read as immutable parents.
This module writes only below the separate ``calibration_adjudication`` root.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from football_intelligence.dense_person_gold import (
    ACK_SCHEMA as FIRST_PASS_ACK_SCHEMA,
    COMPLETION_ASSERTION,
    EVENT_SCHEMA as FIRST_PASS_EVENT_SCHEMA,
    DensePersonConflictError,
    DensePersonValidationError,
    _atomic_json,
    canonical_json_bytes,
    sha256_bytes,
    validate_and_canonicalize_document,
)
from football_intelligence.review_chassis.hashing import stable_hash


CALIBRATION_IDS = tuple(f"DG-{index:03d}" for index in range(1, 7))
PASS_KIND = "CALIBRATION_ADJUDICATION"
EVENT_SCHEMA = "football_intelligence.g7f_c.calibration_superseding_annotation.v1"
ACK_SCHEMA = "football_intelligence.g7f_c.calibration_superseding_acknowledgement.v1"
ADJUDICATION_REASON = "CALIBRATION_VISUAL_QA_MATERIAL_OMISSION_REPAIR"
ADJUDICATION_ASSERTION = (
    "I re-reviewed the full image, addressed the calibration audit findings, "
    "and annotated every individually evaluable visible human."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DensePersonValidationError(f"JSON object required: {path}")
    return value


def _logical_hash(value: Mapping[str, Any], excluded: set[str]) -> str:
    return sha256_bytes(canonical_json_bytes({key: child for key, child in value.items() if key not in excluded}))


@dataclass(frozen=True)
class OriginalCalibrationParent:
    image_id: str
    event_path: Path
    acknowledgement_path: Path
    event: dict[str, Any]
    acknowledgement: dict[str, Any]
    event_file_sha256: str
    acknowledgement_file_sha256: str

    @property
    def event_id(self) -> str:
        return str(self.event["event_id"])

    @property
    def event_sha256(self) -> str:
        return str(self.event["event_sha256"])

    @property
    def acknowledgement_sha256(self) -> str:
        return str(self.acknowledgement["acknowledgement_sha256"])


def load_original_parent(
    original_decisions_root: Path,
    frame: Mapping[str, Any],
    *,
    expected_event_file_sha256: str | None = None,
    expected_ack_file_sha256: str | None = None,
) -> OriginalCalibrationParent:
    """Load and validate one immutable FIRST_PASS parent without writing it."""

    image_id = str(frame.get("anonymous_dense_image_id", ""))
    if image_id not in CALIBRATION_IDS or frame.get("selection_status") != "CALIBRATION_ONLY":
        raise DensePersonValidationError("adjudication parent must be one of DG-001..DG-006 CALIBRATION_ONLY")
    root = original_decisions_root.resolve()
    event_path = root / "events" / f"first_pass__{image_id}.json"
    ack_path = root / "acknowledgements" / f"first_pass__{image_id}.json"
    if not event_path.is_file() or not ack_path.is_file():
        raise DensePersonValidationError(f"finalized FIRST_PASS event and acknowledgement required for {image_id}")
    event_file_sha = sha256_file(event_path)
    ack_file_sha = sha256_file(ack_path)
    if expected_event_file_sha256 is not None and event_file_sha != expected_event_file_sha256:
        raise DensePersonValidationError(f"immutable FIRST_PASS event bytes changed for {image_id}")
    if expected_ack_file_sha256 is not None and ack_file_sha != expected_ack_file_sha256:
        raise DensePersonValidationError(f"immutable FIRST_PASS acknowledgement bytes changed for {image_id}")

    event, acknowledgement = _read_json(event_path), _read_json(ack_path)
    if event.get("schema_version") != FIRST_PASS_EVENT_SCHEMA:
        raise DensePersonValidationError(f"unexpected FIRST_PASS event schema for {image_id}")
    if event.get("pass_kind") != "FIRST_PASS" or event.get("selection_status") != "CALIBRATION_ONLY":
        raise DensePersonValidationError(f"invalid FIRST_PASS calibration lineage for {image_id}")
    if event.get("anonymous_dense_image_id") != image_id or event.get("immutable") is not True:
        raise DensePersonValidationError(f"invalid immutable FIRST_PASS identity for {image_id}")
    for key in ("source_frame_sha256", "source_width", "source_height"):
        if event.get(key) != frame.get(key):
            raise DensePersonValidationError(f"FIRST_PASS source binding mismatch for {image_id}: {key}")
    expected_event_hash = _logical_hash(event, {"event_id", "event_sha256"})
    if event.get("event_sha256") != expected_event_hash:
        raise DensePersonValidationError(f"FIRST_PASS logical event hash mismatch for {image_id}")
    if event.get("event_id") != f"dense-person-{expected_event_hash[:24]}":
        raise DensePersonValidationError(f"FIRST_PASS event ID mismatch for {image_id}")

    if acknowledgement.get("schema_version") != FIRST_PASS_ACK_SCHEMA:
        raise DensePersonValidationError(f"unexpected FIRST_PASS acknowledgement schema for {image_id}")
    if acknowledgement.get("event_id") != event["event_id"]:
        raise DensePersonValidationError(f"FIRST_PASS acknowledgement event ID mismatch for {image_id}")
    if acknowledgement.get("event_sha256") != event["event_sha256"]:
        raise DensePersonValidationError(f"FIRST_PASS acknowledgement event hash mismatch for {image_id}")
    if acknowledgement.get("anonymous_dense_image_id") != image_id:
        raise DensePersonValidationError(f"FIRST_PASS acknowledgement image mismatch for {image_id}")
    if acknowledgement.get("pass_kind") != "FIRST_PASS" or acknowledgement.get("status") != "IMMUTABLY_FINALIZED":
        raise DensePersonValidationError(f"FIRST_PASS acknowledgement status mismatch for {image_id}")
    expected_ack_hash = _logical_hash(acknowledgement, {"acknowledgement_id", "acknowledgement_sha256"})
    if acknowledgement.get("acknowledgement_sha256") != expected_ack_hash:
        raise DensePersonValidationError(f"FIRST_PASS acknowledgement logical hash mismatch for {image_id}")

    return OriginalCalibrationParent(
        image_id=image_id,
        event_path=event_path,
        acknowledgement_path=ack_path,
        event=event,
        acknowledgement=acknowledgement,
        event_file_sha256=event_file_sha,
        acknowledgement_file_sha256=ack_file_sha,
    )


def editable_seed(parent: OriginalCalibrationParent) -> dict[str, Any]:
    """Create a deterministic editable seed while resetting current strip state."""

    annotation = parent.event["annotation"]
    return editable_document(annotation, reset_strips=True)


def editable_document(annotation: Mapping[str, Any], *, reset_strips: bool = False) -> dict[str, Any]:
    """Convert canonical stored geometry to the reviewer's editable representation."""

    people = [
        {
            "instance_id": person["instance_id"],
            "relevance": person["relevance"],
            "visible_mask_components": copy.deepcopy(person["canonical_components"]),
        }
        for person in annotation["people"]
    ]
    ignore_regions = [
        {
            "ignore_region_id": region["ignore_region_id"],
            "reason": region["reason"],
            "polygon": copy.deepcopy(region["canonical_components"][0]),
        }
        for region in annotation["ignore_regions"]
    ]
    return {
        "people": people,
        "ignore_regions": ignore_regions,
        "reviewed_exhaustiveness_strips": (
            [] if reset_strips else copy.deepcopy(annotation["reviewed_exhaustiveness_strips"])
        ),
        "unfinished_polygon": None,
        "completion_assertion": annotation["completion_assertion"],
    }


def parent_provenance(parent: OriginalCalibrationParent) -> dict[str, Any]:
    annotation = parent.event["annotation"]
    return {
        "original_first_pass_event_id": parent.event_id,
        "original_first_pass_event_sha256": parent.event_sha256,
        "original_first_pass_event_file_sha256": parent.event_file_sha256,
        "original_first_pass_acknowledgement_id": parent.acknowledgement["acknowledgement_id"],
        "original_first_pass_ack_sha256": parent.acknowledgement_sha256,
        "original_first_pass_ack_file_sha256": parent.acknowledgement_file_sha256,
        "original_annotation_sha256": sha256_bytes(canonical_json_bytes(annotation)),
        "original_reviewed_exhaustiveness_strips": copy.deepcopy(annotation["reviewed_exhaustiveness_strips"]),
        "original_strip_state": copy.deepcopy(annotation["strip_state"]),
        "original_completion_assertion": annotation["completion_assertion"],
    }


def _metadata(value: Any) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise DensePersonValidationError("adjudication metadata must be an object")
    assertion = value.get("adjudication_assertion")
    addressed = value.get("repair_checklist_addressed", False)
    if assertion not in {None, ADJUDICATION_ASSERTION}:
        raise DensePersonValidationError("adjudication assertion must be exact")
    if not isinstance(addressed, bool):
        raise DensePersonValidationError("repair checklist state must be boolean")
    return {"adjudication_assertion": assertion, "repair_checklist_addressed": addressed}


def build_adjudication_event(
    frame: Mapping[str, Any],
    document: Mapping[str, Any],
    adjudication_metadata: Mapping[str, Any],
    *,
    parent: OriginalCalibrationParent,
    binding_hashes: Mapping[str, str],
    reviewer_release: str,
    audit_checklist_sha256: str,
    final_revision: int,
    adjudication_sequence: int = 1,
    supersedes_event_id: str | None = None,
    supersedes_event_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _metadata(adjudication_metadata)
    if metadata["adjudication_assertion"] != ADJUDICATION_ASSERTION:
        raise DensePersonValidationError("exact adjudication assertion required")
    if metadata["repair_checklist_addressed"] is not True:
        raise DensePersonValidationError("audit repair checklist must be marked addressed")
    canonical = validate_and_canonicalize_document(document, frame)
    provenance = parent_provenance(parent)
    payload = {
        "schema_version": EVENT_SCHEMA,
        "anonymous_dense_image_id": frame["anonymous_dense_image_id"],
        "pass_kind": PASS_KIND,
        "selection_status": "CALIBRATION_ONLY",
        "source_frame_sha256": frame["source_frame_sha256"],
        "source_width": frame["source_width"],
        "source_height": frame["source_height"],
        "all_frame_instance_lineage": frame["all_frame_instance_lineage"],
        "binding_hashes": dict(sorted(binding_hashes.items())),
        "reviewer_release": reviewer_release,
        "final_revision": final_revision,
        "adjudication_sequence": adjudication_sequence,
        "adjudication_reason": ADJUDICATION_REASON,
        "supersedes_event_id": supersedes_event_id or parent.event_id,
        "supersedes_event_sha256": supersedes_event_sha256 or parent.event_sha256,
        "parent_provenance": provenance,
        "audit_checklist_sha256": audit_checklist_sha256,
        "annotation": canonical,
        "adjudication_metadata": metadata,
        "server_validation": {
            "candidate_data_used": False,
            "source_hash_bound": True,
            "source_coordinates_validated": True,
            "thresholds_enforced": True,
            "all_eight_adjudication_strips_reviewed": True,
            "exact_dense_gold_completion_assertion": canonical["completion_assertion"] == COMPLETION_ASSERTION,
            "exact_adjudication_assertion": True,
            "parent_bindings_exact": True,
            "repair_checklist_addressed": True,
        },
        "immutable": True,
        "production_ready": False,
    }
    event_hash = sha256_bytes(canonical_json_bytes(payload))
    event = {
        **payload,
        "event_id": f"calibration-adjudication-{event_hash[:24]}",
        "event_sha256": event_hash,
    }
    ack_payload = {
        "schema_version": ACK_SCHEMA,
        "event_id": event["event_id"],
        "event_sha256": event_hash,
        "anonymous_dense_image_id": frame["anonymous_dense_image_id"],
        "pass_kind": PASS_KIND,
        "adjudication_sequence": adjudication_sequence,
        "status": "IMMUTABLY_FINALIZED",
    }
    acknowledgement = {
        **ack_payload,
        "acknowledgement_id": f"ack-{stable_hash(ack_payload)[:24]}",
        "acknowledgement_sha256": sha256_bytes(canonical_json_bytes(ack_payload)),
    }
    return event, acknowledgement


def validate_adjudication_pair(
    event_path: Path,
    acknowledgement_path: Path,
    *,
    frame: Mapping[str, Any],
    parent: OriginalCalibrationParent,
    expected_binding_hashes: Mapping[str, str] | None = None,
    expected_adjudication_sequence: int = 1,
    expected_supersedes_event_id: str | None = None,
    expected_supersedes_event_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    event, acknowledgement = _read_json(event_path), _read_json(acknowledgement_path)
    image_id = str(frame["anonymous_dense_image_id"])
    if event.get("schema_version") != EVENT_SCHEMA or event.get("pass_kind") != PASS_KIND:
        raise DensePersonValidationError(f"invalid adjudication event schema for {image_id}")
    if event.get("anonymous_dense_image_id") != image_id or event.get("selection_status") != "CALIBRATION_ONLY":
        raise DensePersonValidationError(f"invalid adjudication image binding for {image_id}")
    if event.get("immutable") is not True or event.get("production_ready") is not False:
        raise DensePersonValidationError(f"invalid adjudication immutability state for {image_id}")
    if event.get("adjudication_sequence") != expected_adjudication_sequence:
        raise DensePersonValidationError(f"invalid adjudication sequence for {image_id}")
    supersedes_id = expected_supersedes_event_id or parent.event_id
    supersedes_sha = expected_supersedes_event_sha256 or parent.event_sha256
    if event.get("supersedes_event_id") != supersedes_id:
        raise DensePersonValidationError(f"invalid adjudication parent ID for {image_id}")
    if event.get("supersedes_event_sha256") != supersedes_sha:
        raise DensePersonValidationError(f"invalid adjudication parent hash for {image_id}")
    provenance = event.get("parent_provenance")
    if provenance != parent_provenance(parent):
        raise DensePersonValidationError(f"invalid adjudication parent provenance for {image_id}")
    for key in ("source_frame_sha256", "source_width", "source_height"):
        if event.get(key) != frame.get(key):
            raise DensePersonValidationError(f"adjudication source binding mismatch for {image_id}: {key}")
    expected_bindings = dict(sorted(expected_binding_hashes.items())) if expected_binding_hashes is not None else None
    if expected_bindings is not None and event.get("binding_hashes") != expected_bindings:
        raise DensePersonValidationError(f"adjudication frozen contract binding mismatch for {image_id}")
    expected_event_hash = _logical_hash(event, {"event_id", "event_sha256"})
    if event.get("event_sha256") != expected_event_hash:
        raise DensePersonValidationError(f"adjudication event hash mismatch for {image_id}")
    if event.get("event_id") != f"calibration-adjudication-{expected_event_hash[:24]}":
        raise DensePersonValidationError(f"adjudication event ID mismatch for {image_id}")
    if event.get("adjudication_metadata") != {
        "adjudication_assertion": ADJUDICATION_ASSERTION,
        "repair_checklist_addressed": True,
    }:
        raise DensePersonValidationError(f"adjudication completion metadata mismatch for {image_id}")
    canonical = validate_and_canonicalize_document(editable_document(event["annotation"]), frame)
    if canonical != event["annotation"]:
        raise DensePersonValidationError(f"adjudication annotation is not canonical for {image_id}")

    if acknowledgement.get("schema_version") != ACK_SCHEMA:
        raise DensePersonValidationError(f"invalid adjudication acknowledgement schema for {image_id}")
    if acknowledgement.get("event_id") != event["event_id"]:
        raise DensePersonValidationError(f"adjudication acknowledgement event mismatch for {image_id}")
    if acknowledgement.get("event_sha256") != event["event_sha256"]:
        raise DensePersonValidationError(f"adjudication acknowledgement hash mismatch for {image_id}")
    if acknowledgement.get("anonymous_dense_image_id") != image_id:
        raise DensePersonValidationError(f"adjudication acknowledgement image mismatch for {image_id}")
    if acknowledgement.get("adjudication_sequence") != expected_adjudication_sequence:
        raise DensePersonValidationError(f"adjudication acknowledgement sequence mismatch for {image_id}")
    if acknowledgement.get("pass_kind") != PASS_KIND or acknowledgement.get("status") != "IMMUTABLY_FINALIZED":
        raise DensePersonValidationError(f"adjudication acknowledgement status mismatch for {image_id}")
    expected_ack_hash = _logical_hash(acknowledgement, {"acknowledgement_id", "acknowledgement_sha256"})
    if acknowledgement.get("acknowledgement_sha256") != expected_ack_hash:
        raise DensePersonValidationError(f"adjudication acknowledgement logical hash mismatch for {image_id}")
    return event, acknowledgement


class CalibrationAdjudicationStore:
    """Revisioned store writing only the isolated adjudication namespace."""

    def __init__(
        self,
        root: Path,
        *,
        original_decisions_root: Path,
        frames: Mapping[str, Mapping[str, Any]],
        binding_hashes: Mapping[str, str],
        reviewer_release: str,
        audit_checklists: Mapping[str, list[Mapping[str, Any]]],
        audit_checklist_sha256: str,
        frozen_parent_file_hashes: Mapping[str, Mapping[str, str]] | None = None,
        adjudication_sequence: int = 1,
    ) -> None:
        if set(frames) != set(CALIBRATION_IDS):
            raise DensePersonValidationError("adjudication queue must contain exactly DG-001..DG-006")
        self.root = root.resolve()
        self.original_decisions_root = original_decisions_root.resolve()
        self.frames = {str(key): dict(value) for key, value in frames.items()}
        self.binding_hashes = dict(binding_hashes)
        self.reviewer_release = reviewer_release
        self.audit_checklists = {key: copy.deepcopy(list(value)) for key, value in audit_checklists.items()}
        self.audit_checklist_sha256 = audit_checklist_sha256
        self.frozen_parent_file_hashes = copy.deepcopy(dict(frozen_parent_file_hashes or {}))
        if not isinstance(adjudication_sequence, int) or adjudication_sequence < 1:
            raise DensePersonValidationError("adjudication sequence must be a positive integer")
        self.adjudication_sequence = adjudication_sequence
        self._lock = threading.RLock()
        self.parents = {image_id: self._load_parent(image_id) for image_id in CALIBRATION_IDS}

    def _load_parent(self, image_id: str) -> OriginalCalibrationParent:
        expected = self.frozen_parent_file_hashes.get(image_id, {})
        return load_original_parent(
            self.original_decisions_root,
            self.frames[image_id],
            expected_event_file_sha256=expected.get("event_file_sha256"),
            expected_ack_file_sha256=expected.get("acknowledgement_file_sha256"),
        )

    def _key(self, image_id: str, sequence: int | None = None) -> str:
        if image_id not in CALIBRATION_IDS:
            raise DensePersonValidationError("unknown calibration adjudication image")
        value = self.adjudication_sequence if sequence is None else sequence
        return f"calibration_adjudication_{value:03d}__{image_id}"

    def _chain_event(self, image_id: str, sequence: int) -> tuple[dict[str, Any], dict[str, Any]]:
        if sequence < 1:
            raise DensePersonValidationError("adjudication chain sequence must be positive")
        if sequence == 1:
            supersedes_id = self.parents[image_id].event_id
            supersedes_sha = self.parents[image_id].event_sha256
        else:
            previous, _ = self._chain_event(image_id, sequence - 1)
            supersedes_id = previous["event_id"]
            supersedes_sha = previous["event_sha256"]
        key = self._key(image_id, sequence)
        event_path, ack_path = self._event_path(key), self._ack_path(key)
        if not event_path.is_file() or not ack_path.is_file():
            raise DensePersonValidationError(f"adjudication {sequence} parent pair required for {image_id}")
        return validate_adjudication_pair(
            event_path,
            ack_path,
            frame=self.frames[image_id],
            parent=self.parents[image_id],
            expected_binding_hashes=self.binding_hashes,
            expected_adjudication_sequence=sequence,
            expected_supersedes_event_id=supersedes_id,
            expected_supersedes_event_sha256=supersedes_sha,
        )

    def _superseded_event(self, image_id: str) -> dict[str, Any]:
        if self.adjudication_sequence == 1:
            return self.parents[image_id].event
        return self._chain_event(image_id, self.adjudication_sequence - 1)[0]

    def _draft_path(self, key: str) -> Path:
        return self.root / "drafts" / f"{key}.json"

    def _event_path(self, key: str) -> Path:
        return self.root / "events" / f"{key}.json"

    def _ack_path(self, key: str) -> Path:
        return self.root / "acknowledgements" / f"{key}.json"

    def _action_path(self, action_id: str) -> Path:
        return self.root / "action_receipts" / f"{action_id}.json"

    def state(self, image_id: str) -> dict[str, Any]:
        key = self._key(image_id)
        event_path, ack_path, draft_path = self._event_path(key), self._ack_path(key), self._draft_path(key)
        parent = self._load_parent(image_id)
        superseded = self._superseded_event(image_id)
        if event_path.is_file() or ack_path.is_file():
            if not event_path.is_file() or not ack_path.is_file():
                raise DensePersonValidationError(f"incomplete immutable adjudication pair for {image_id}")
            event, _ = validate_adjudication_pair(
                event_path,
                ack_path,
                frame=self.frames[image_id],
                parent=parent,
                expected_binding_hashes=self.binding_hashes,
                expected_adjudication_sequence=self.adjudication_sequence,
                expected_supersedes_event_id=superseded["event_id"],
                expected_supersedes_event_sha256=superseded["event_sha256"],
            )
            return {
                "anonymous_dense_image_id": image_id,
                "pass_kind": PASS_KIND,
                "revision": int(event["final_revision"]),
                "finalized": True,
                "read_only": True,
                "document": event["annotation"],
                "adjudication_metadata": event["adjudication_metadata"],
                "parent_provenance": event["parent_provenance"],
                "repair_checklist": copy.deepcopy(self.audit_checklists[image_id]),
            }
        if draft_path.is_file():
            draft = _read_json(draft_path)
            return {**draft, "finalized": False, "read_only": False}
        return {
            "anonymous_dense_image_id": image_id,
            "pass_kind": PASS_KIND,
            "revision": 0,
            "finalized": False,
            "read_only": False,
            "document": editable_document(superseded["annotation"], reset_strips=True),
            "adjudication_metadata": {
                "adjudication_assertion": None,
                "repair_checklist_addressed": False,
            },
            "parent_provenance": parent_provenance(parent),
            "repair_checklist": copy.deepcopy(self.audit_checklists[image_id]),
            "seeded_from_immutable_first_pass": self.adjudication_sequence == 1,
            "seeded_from_superseding_event_id": superseded["event_id"],
            "adjudication_strip_state_reset": True,
        }

    def apply_action(self, action: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            action_id = str(action.get("action_id", "")).strip()
            image_id = str(action.get("anonymous_dense_image_id", ""))
            pass_kind = str(action.get("pass_kind", PASS_KIND))
            action_type = str(action.get("action_type", ""))
            if not action_id:
                raise DensePersonValidationError("action_id required")
            if pass_kind != PASS_KIND:
                raise DensePersonValidationError("action pass kind differs from calibration adjudication")
            request_hash = sha256_bytes(canonical_json_bytes(dict(action)))
            receipt_path = self._action_path(action_id)
            if receipt_path.is_file():
                receipt = _read_json(receipt_path)
                if receipt["request_sha256"] != request_hash:
                    raise DensePersonConflictError("ACTION_ID_REUSE", "action ID reused with different payload")
                return receipt["response"]
            current = self.state(image_id)
            expected_revision = action.get("expected_revision")
            if not isinstance(expected_revision, int) or expected_revision != current["revision"]:
                raise DensePersonConflictError(
                    "STALE_REVISION",
                    "expected revision differs from server state",
                    current_revision=current["revision"],
                )
            if action_type not in {"SAVE_DRAFT", "FINALIZE"}:
                raise DensePersonValidationError("calibration adjudication supports SAVE_DRAFT and FINALIZE only")
            if current["finalized"]:
                raise DensePersonConflictError("FINALIZED_READ_ONLY", "finalized adjudication is immutable")
            document = action.get("document")
            if not isinstance(document, Mapping):
                raise DensePersonValidationError("annotation document required")
            metadata = _metadata(action.get("adjudication_metadata"))
            new_revision = current["revision"] + 1
            parent = self._load_parent(image_id)
            superseded = self._superseded_event(image_id)
            if action_type == "SAVE_DRAFT":
                response = {
                    "anonymous_dense_image_id": image_id,
                    "pass_kind": PASS_KIND,
                    "revision": new_revision,
                    "finalized": False,
                    "read_only": False,
                    "document": copy.deepcopy(dict(document)),
                    "adjudication_metadata": metadata,
                    "parent_provenance": parent_provenance(parent),
                    "repair_checklist": copy.deepcopy(self.audit_checklists[image_id]),
                }
                _atomic_json(self._draft_path(self._key(image_id)), response)
            else:
                event, acknowledgement = build_adjudication_event(
                    self.frames[image_id],
                    document,
                    metadata,
                    parent=parent,
                    binding_hashes=self.binding_hashes,
                    reviewer_release=self.reviewer_release,
                    audit_checklist_sha256=self.audit_checklist_sha256,
                    final_revision=new_revision,
                    adjudication_sequence=self.adjudication_sequence,
                    supersedes_event_id=superseded["event_id"],
                    supersedes_event_sha256=superseded["event_sha256"],
                )
                event_path, ack_path = self._event_path(self._key(image_id)), self._ack_path(self._key(image_id))
                if event_path.exists() or ack_path.exists():
                    raise DensePersonConflictError("EVENT_ALREADY_EXISTS", "immutable adjudication already exists")
                _atomic_json(event_path, event)
                _atomic_json(ack_path, acknowledgement)
                response = {
                    "anonymous_dense_image_id": image_id,
                    "pass_kind": PASS_KIND,
                    "revision": new_revision,
                    "finalized": True,
                    "read_only": True,
                    "event_id": event["event_id"],
                    "event_sha256": event["event_sha256"],
                    "acknowledgement": acknowledgement,
                }
            _atomic_json(
                receipt_path,
                {
                    "schema_version": "football_intelligence.g7f_c.calibration_adjudication_action_receipt.v1",
                    "request_sha256": request_hash,
                    "response": response,
                },
            )
            return response
