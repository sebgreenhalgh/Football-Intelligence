from __future__ import annotations

import json
import shutil
import subprocess
import threading
import urllib.request
from pathlib import Path

import pytest

from football_intelligence import g7d_c1_r1_novice_review as r1
from football_intelligence.g7d_c1_r8_latest_completion_receipt import (
    HISTORICAL_RECEIPT_ID,
    REPOSITORY_BASELINE,
    append_current_completion_receipt,
    create_server,
    resolve_current_completion_receipt,
    resolve_latest_event_set,
)

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
PACKAGE = (
    PROJECT
    / "experiments"
    / "football_observation_reasoner"
    / "part 6"
    / "G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_v1"
    / "02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE"
)
LATEST_ID = "8e145c713516fb829dc8f32bfe0ecea2"
LATEST_HASH = "6445b04f14bd211f1ebbd8033711a9c3cea8aa43d73f113e4b959e9e93262ab5"
LATEST_ACK_HASH = "2ac52a1c79fa01dd01f57cc1d2e510eef4efb6c2ee1b748b5dada8a8ca167844"
SUPERSEDED_ID = "d6cff7afef94bad7d411d659dacb0e2d"


def _copy_protocol_package(destination: Path) -> Path:
    destination.mkdir()
    for name in ("review_cases.json", "scene_candidate_overlays.json", "target_box_calibration_status.json"):
        shutil.copy2(PACKAGE / name, destination / name)
    shutil.copytree(PACKAGE / "review_events", destination / "review_events")
    receipts = destination / "review_receipts"
    shutil.copytree(PACKAGE / "review_receipts" / "acknowledgements", receipts / "acknowledgements")
    (receipts / "completion").mkdir()
    shutil.copy2(PACKAGE / "review_receipts" / "completion" / "final.json", receipts / "completion" / "final.json")
    return destination


def test_expected_baseline_is_ancestor_and_input_counts_are_exact() -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", REPOSITORY_BASELINE, "HEAD"],
        cwd=Path(__file__).parents[1],
        check=False,
    )
    assert result.returncode == 0
    assert len(list((PACKAGE / "review_events" / "candidate").glob("*.json"))) == 193
    assert len(list((PACKAGE / "review_events" / "scene").glob("*.json"))) == 24


def test_exact_latest_event_set_and_acknowledgements() -> None:
    latest = resolve_latest_event_set(PACKAGE)
    assert (len(latest.candidate_events), len(latest.scene_events), latest.event_count) == (192, 24, 216)
    selected = {row["event_id"] for row in [*latest.candidate_events, *latest.scene_events]}
    assert LATEST_ID in selected
    assert SUPERSEDED_ID not in selected
    s01t01 = next(row for row in latest.candidate_events if row["identity"] == "s01t01")
    assert s01t01["event_sha256"] == LATEST_HASH
    assert s01t01["acknowledgement_sha256"] == LATEST_ACK_HASH
    assert len(latest.acknowledgement_receipts) == 216


def test_append_is_immutable_idempotent_and_resolves_current(tmp_path: Path) -> None:
    package = _copy_protocol_package(tmp_path / "package")
    old_event = package / "review_events" / "candidate" / f"{SUPERSEDED_ID}.json"
    latest_event = package / "review_events" / "candidate" / f"{LATEST_ID}.json"
    historical = package / "review_receipts" / "completion" / "final.json"
    before = {path: (path.read_bytes(), r1.sha256_file(path)) for path in (old_event, latest_event, historical)}

    first_path, first = append_current_completion_receipt(package)
    first_bytes = first_path.read_bytes()
    second_path, second = append_current_completion_receipt(package)
    resolved_path, resolved = resolve_current_completion_receipt(package)

    assert first_path == second_path == resolved_path
    assert first == second == resolved
    assert first_path.read_bytes() == first_bytes
    assert first["supersedes_completion_receipt_id"] == HISTORICAL_RECEIPT_ID
    ids = {row["event_id"] for row in [*first["candidate_events"], *first["scene_events"]]}
    assert len(ids) == 216 and LATEST_ID in ids and SUPERSEDED_ID not in ids
    assert len(first["acknowledgement_receipts"]) == 216
    for path, (content, digest) in before.items():
        assert path.read_bytes() == content
        assert r1.sha256_file(path) == digest


def test_future_superseding_save_refreshes_receipt_before_http_200(tmp_path: Path) -> None:
    package = _copy_protocol_package(tmp_path / "package")
    _, prior = append_current_completion_receipt(package)
    latest_event = json.loads(
        (package / "review_events" / "candidate" / f"{LATEST_ID}.json").read_text(encoding="utf-8")
    )
    payload = {**latest_event["payload"], "idempotency_key": "r8-future-superseding-edit-test"}
    server = create_server(package, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/api/save",
            data=r1.canonical_bytes(payload),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:  # noqa: S310 - bounded loopback test
            body = json.loads(response.read())
            assert response.status == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert body["all_cases_complete"] is True
    assert body["event_id"] != body["current_completion_receipt_id"]
    assert body["current_completion_receipt_id"] != prior["completion_receipt_id"]
    path, current = resolve_current_completion_receipt(package)
    assert path.relative_to(package).as_posix() == body["current_completion_receipt_relative_path"]
    assert current["completion_receipt_id"] == body["current_completion_receipt_id"]
    assert body["event_id"] in {row["event_id"] for row in current["candidate_events"]}


def test_current_resolver_rejects_missing_and_ambiguous_receipts(tmp_path: Path) -> None:
    package = _copy_protocol_package(tmp_path / "package")
    with pytest.raises(RuntimeError, match="CURRENT_COMPLETION_RECEIPT_MISSING"):
        resolve_current_completion_receipt(package)
    path, receipt = append_current_completion_receipt(package)
    duplicate = path.with_name("completion-r8-duplicate-current.json")
    duplicate.write_bytes(r1.canonical_bytes({**receipt, "completion_receipt_id": duplicate.stem}))
    with pytest.raises(RuntimeError, match="CURRENT_COMPLETION_RECEIPT_AMBIGUOUS"):
        resolve_current_completion_receipt(package)
