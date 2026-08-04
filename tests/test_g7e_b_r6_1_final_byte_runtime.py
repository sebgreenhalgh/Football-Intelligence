"""Focused R6.1 contracts, transaction recovery, visual, and compatibility tests."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import uuid

import cv2
import numpy as np
import pytest

from football_intelligence.g7e_b_r6_action_reducer import ACTION_TYPES, R6_REVIEW_REVISION
from football_intelligence.temporal_reviewer.contracts import (
    MAX_JSON_BODY_BYTES,
    canonical_action_uuid,
    contained_path,
    validate_action_envelope,
)
from football_intelligence.temporal_reviewer.http_server import (
    RequestBodyTooLarge,
    UnsupportedMediaType,
    read_json_request,
)
from football_intelligence.temporal_reviewer.persistence import ActionTransaction, recover_action_transactions
from football_intelligence.temporal_reviewer.reducer import apply_action as modular_apply_action
from football_intelligence.temporal_reviewer.visual import enhance_review_image


def action(action_id: str | None = None) -> dict[str, object]:
    action_id = action_id or str(uuid.uuid4())
    return {
        "schema_version": "football_intelligence.g7e_b_r6.browser_action.v1",
        "action_id": action_id,
        "idempotency_key": action_id,
        "review_revision": R6_REVIEW_REVISION,
        "contract_hash": "0" * 64,
        "mode": "practice",
        "tranche_id": "PRACTICE",
        "burst_id": "burst",
        "expected_draft_revision": 1,
        "expected_draft_sha256": "1" * 64,
        "question_instance_key": "burst|original_focus",
        "action_type": "ANSWER_QUESTION",
        "payload": {"value": "NO_RELEVANT_PERSON"},
        "client_timestamp": "2026-08-04T00:00:00Z",
    }


def test_action_identity_requires_equal_canonical_uuids() -> None:
    envelope = action()
    expected = envelope["action_id"]
    assert validate_action_envelope(envelope, ACTION_TYPES) == (expected, expected)
    altered = copy.deepcopy(envelope)
    altered["idempotency_key"] = str(uuid.uuid4())
    with pytest.raises(ValueError, match="must equal"):
        validate_action_envelope(altered, ACTION_TYPES)
    uppercase = copy.deepcopy(envelope)
    uppercase["action_id"] = str(expected).upper()
    uppercase["idempotency_key"] = str(expected).upper()
    with pytest.raises(ValueError, match="lowercase canonical"):
        validate_action_envelope(uppercase, ACTION_TYPES)
    assert "CLEAR_ANSWER" not in ACTION_TYPES


def test_contained_path_rejects_absolute_parent_and_traversal(tmp_path: Path) -> None:
    assert contained_path(tmp_path, "receipts", "ok.json").parent.name == "receipts"
    for value in ("../escape.json", "sub/../../escape.json", Path("C:/escape.json")):
        with pytest.raises(ValueError):
            contained_path(tmp_path, value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("action_id", "../../receipt"),
        ("idempotency_key", "00000000-0000-4000-8000-000000000001/extra"),
        ("burst_id", "../burst"),
        ("burst_id", "C:\\private\\burst"),
        ("question_instance_key", "burst|question\x00hidden"),
        ("contract_hash", "A" * 64),
    ],
)
def test_action_envelope_rejects_path_and_encoding_attacks(field: str, value: str) -> None:
    envelope = action()
    envelope[field] = value
    if field == "action_id":
        envelope["idempotency_key"] = value
    with pytest.raises(ValueError):
        validate_action_envelope(envelope, ACTION_TYPES)


@pytest.mark.parametrize("failure", ["draft", "action_receipt", "idempotency_ledger"])
def test_action_transaction_recovers_after_every_subwrite(tmp_path: Path, failure: str) -> None:
    root = tmp_path / "decisions"
    action_id = str(uuid.uuid4())
    transaction = ActionTransaction(root, action_id)
    kwargs = {
        "draft_relative": "drafts/burst.json",
        "draft_bytes": b'{"draft":1}\n',
        "receipt_relative": f"receipts/actions/action-ack-{action_id}.json",
        "receipt_bytes": b'{"receipt":1}\n',
        "ledger_relative": f"action_idempotency/{action_id}.json",
        "ledger_bytes": b'{"ledger":1}\n',
        "transaction_context": {
            "previous_draft_revision": 1,
            "previous_draft_sha256": "0" * 64,
            "action_envelope_sha256": "1" * 64,
            "action_semantic_sha256": "2" * 64,
            "next_draft_revision": 2,
            "next_draft_sha256": "3" * 64,
        },
    }
    with pytest.raises(RuntimeError, match="SIMULATED_ACTION_TRANSACTION_INTERRUPTION"):
        transaction.commit(**kwargs, fail_after=failure)
    result = recover_action_transactions(root)
    assert result["recovered"] == 1
    assert (root / "drafts/burst.json").read_bytes() == kwargs["draft_bytes"]
    assert (root / f"receipts/actions/action-ack-{action_id}.json").read_bytes() == kwargs["receipt_bytes"]
    assert (root / f"action_idempotency/{action_id}.json").read_bytes() == kwargs["ledger_bytes"]
    assert recover_action_transactions(root)["recovered"] == 0
    journal = json.loads((root / f"action_transactions/{action_id}.json").read_text(encoding="utf-8"))
    assert journal["state"] == "COMMITTED"
    assert journal["previous_draft_revision"] == 1
    assert journal["next_draft_revision"] == 2


def test_finalized_acknowledged_event_legitimately_supersedes_action_draft(tmp_path: Path) -> None:
    root = tmp_path / "decisions"
    action_id = str(uuid.uuid4())
    draft_bytes = b'{"burst_id":"burst","draft_version":2}\n'
    receipt_relative = f"receipts/actions/action-ack-{action_id}.json"
    ledger_relative = f"action_idempotency/{action_id}.json"
    ActionTransaction(root, action_id).commit(
        draft_relative="drafts/burst.json",
        draft_bytes=draft_bytes,
        receipt_relative=receipt_relative,
        receipt_bytes=b'{"receipt":1}\n',
        ledger_relative=ledger_relative,
        ledger_bytes=b'{"ledger":1}\n',
        transaction_context={
            "previous_draft_revision": 1,
            "previous_draft_sha256": "0" * 64,
            "action_envelope_sha256": "1" * 64,
            "action_semantic_sha256": "2" * 64,
            "next_draft_revision": 2,
            "next_draft_sha256": "3" * 64,
        },
    )
    (root / "drafts/burst.json").unlink()
    event_id = str(uuid.uuid4())
    event_bytes = (json.dumps({"burst_id": "burst", "event_id": event_id}, sort_keys=True) + "\n").encode()
    event_path = root / f"events/TRANCHE_1/{event_id}.json"
    event_path.parent.mkdir(parents=True)
    event_path.write_bytes(event_bytes)
    acknowledgement_path = root / f"receipts/acknowledgements/ack-{event_id}.json"
    acknowledgement_path.parent.mkdir(parents=True)
    acknowledgement_path.write_text(
        json.dumps(
            {
                "event_id": event_id,
                "event_sha256": hashlib.sha256(event_bytes).hexdigest(),
                "server_validated": True,
                "case_complete": True,
            }
        ),
        encoding="utf-8",
    )
    assert recover_action_transactions(root) == {"inspected": 1, "recovered": 0, "committed": 1}
    acknowledgement_path.unlink()
    with pytest.raises(ValueError, match="target is missing"):
        recover_action_transactions(root)


class Request:
    def __init__(self, body: bytes, content_type: str = "application/json; charset=utf-8"):
        self.headers = {"Content-Type": content_type, "Content-Length": str(len(body))}
        self.rfile = io.BytesIO(body)


def test_http_json_request_is_strict_and_bounded() -> None:
    assert read_json_request(Request(b'{"ok":true}')) == {"ok": True}
    with pytest.raises(UnsupportedMediaType):
        read_json_request(Request(b"{}", "text/plain"))
    oversized = Request(b"{}")
    oversized.headers["Content-Length"] = str(MAX_JSON_BODY_BYTES + 1)
    with pytest.raises(RequestBodyTooLarge):
        read_json_request(oversized)
    with pytest.raises(ValueError, match="object"):
        read_json_request(Request(b"[]"))


def test_visual_enhancement_preserves_geometry_and_is_deterministic() -> None:
    image = np.zeros((108, 409, 3), dtype=np.uint8)
    image[:, :, 0] = 32
    image[:, :, 1] = np.linspace(18, 92, image.shape[1], dtype=np.uint8)
    image[:, :, 2] = 50
    metrics = {"luminance_median": 42.0}
    first, parameters = enhance_review_image(image, metrics)
    second, second_parameters = enhance_review_image(image, metrics)
    assert first.shape == image.shape
    assert np.array_equal(first, second)
    assert parameters == second_parameters
    assert parameters["geometry_operation"] == "NONE"
    assert float(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY).mean()) > float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())


def test_modular_reducer_is_the_compatibility_equivalent() -> None:
    from football_intelligence.g7e_b_r6_action_reducer import apply_action

    assert modular_apply_action is apply_action
    assert canonical_action_uuid("00000000-0000-4000-8000-000000000001", "test") == (
        "00000000-0000-4000-8000-000000000001"
    )


def test_action_contract_json_has_no_partial_clear_action() -> None:
    contract = json.loads(
        Path("src/football_intelligence/g7e_b_r6_server_action_contract.json").read_text(encoding="utf-8")
    )
    assert "CLEAR_ANSWER" not in contract["action_types"]


def test_canonical_docs_are_utf8_current_and_public_identity_is_explicit() -> None:
    paths = [
        Path("AGENTS.md"),
        Path("README.md"),
        Path("CONTRIBUTING.md"),
        Path("docs/football_intelligence/CURRENT_STATE.md"),
        Path("docs/football_intelligence/DATA_LAYOUT.md"),
        Path("docs/football_intelligence/NEXT_STAGE.md"),
    ]
    corrupt = ("â€", "Ã¢", "ï¿½", "\ufffd")
    documents = {path: path.read_text(encoding="utf-8") for path in paths}
    assert all(not any(token in text for token in corrupt) for text in documents.values())
    current = documents[Path("docs/football_intelligence/CURRENT_STATE.md")]
    next_stage = documents[Path("docs/football_intelligence/NEXT_STAGE.md")]
    assert "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1" not in current + next_stage
    assert "R6.1" in current and "launch" in next_stage.lower()
    assert documents[Path("README.md")].startswith("# Football Intelligence Infrastructure")
    assert "upstream" in documents[Path("README.md")].lower()
    status_paths = (
        Path("README.md"),
        Path("docs/football_intelligence/CURRENT_STATE.md"),
        Path("docs/football_intelligence/NEXT_STAGE.md"),
    )
    assert all("production_ready=false" in documents[path] for path in status_paths)
