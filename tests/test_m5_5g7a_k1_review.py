from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from football_intelligence.football_observation_reasoner.k1_review import (
    CERTAINTY_VALUES,
    DEFAULT_REVIEWER_SESSION_ID,
    GUIDANCE,
    HOST,
    INDEXEDDB_NAMESPACE,
    KIT_VALUES,
    K1CaseSpec,
    K1ContextFrame,
    K1ReviewPersistence,
    K1ServerConfig,
    K1StateDivergenceError,
    PARTICIPATION_VALUES,
    PITCH_VALUES,
    PORT,
    REVIEW_ID,
    ROLE_VALUES,
    TARGET_STRATA,
    TEAM_VALUES,
    build_k1_package,
    load_k1_package,
    validate_k1_package,
)
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.hashing import sha256_file


def _image(path: Path, colour: tuple[int, int, int], size: tuple[int, int] = (120, 80)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)
    return path


def _build_package(tmp_path: Path, *, case_count: int = 3) -> Path:
    inputs = tmp_path / "inputs"
    context_previous = _image(inputs / "previous.png", (20, 60, 110))
    context_next = _image(inputs / "next.png", (110, 50, 20))
    specs: list[K1CaseSpec] = []
    for index in range(case_count):
        source = _image(inputs / f"source_{index}.png", (20 + index * 20, 100, 40))
        crop = _image(inputs / f"crop_{index}.png", (40, 20 + index * 20, 100), size=(24, 40))
        specs.append(
            K1CaseSpec(
                case_id=f"k1_case_{index:03d}",
                source_group_id=f"source_group_{index:03d}",
                source_image_path=source,
                crop_image_path=crop,
                target_box={"x1": 10, "y1": 8, "x2": 34, "y2": 48},
                context_frames=(
                    K1ContextFrame("Previous", context_previous),
                    K1ContextFrame("Next", context_next),
                ),
                claimed_source_sha256=sha256_file(source),
                claimed_crop_sha256=sha256_file(crop),
            )
        )
    package = tmp_path / "12_SUPPLEMENTARY_REVIEW_PACKAGE"
    shortfalls = dict(TARGET_STRATA)
    team_1_examples = (case_count + 1) // 2
    team_2_examples = case_count // 2
    shortfalls["team_1_goalkeeper"] -= team_1_examples
    shortfalls["team_2_goalkeeper"] -= team_2_examples
    build_k1_package(
        package_root=package,
        cases=specs,
        stage_id="M5_5G7A_TEST_STAGE",
        selection_spec_sha256="a" * 64,
        quota_shortfalls=shortfalls,
        repo_root=Path(__file__).resolve().parents[1],
    )
    return package


def _store(package: Path) -> K1ReviewPersistence:
    manifest, ui_config = load_k1_package(package)
    return K1ReviewPersistence(
        manifest=manifest,
        ui_config=ui_config,
        decisions_root=package / "decisions",
        reviewer_session_id=DEFAULT_REVIEWER_SESSION_ID,
    )


def _annotation(
    case: dict[str, Any],
    *,
    role: str = "OUTFIELD_PLAYER",
    team: str = "TEAM_1",
    kit: str = "MATCH_OUTFIELD_KIT",
    pitch: str = "ON_PITCH",
    participation: str = "ACTIVE_ON_PITCH",
    certainty: str = "CERTAIN",
) -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.m5_5g7a.k1_annotation.v1",
        "role": role,
        "team_affiliation": team,
        "kit_state": kit,
        "pitch_state": pitch,
        "participation_state": participation,
        "certainty": certainty,
        "source_frame_sha256": case["source_frame_sha256"],
        "target_crop_sha256": case["target_crop_sha256"],
        "target_binding_sha256": case["target_binding_sha256"],
    }


def _save_payload(
    store: K1ReviewPersistence,
    case: dict[str, Any],
    annotation: dict[str, Any],
    *,
    expected_hash: str | None = None,
    event_id: str | None = None,
) -> dict[str, Any]:
    identifier = event_id or str(uuid.uuid4())
    return {
        "event_type": "K1_CASE_SAVED",
        "review_id": REVIEW_ID,
        "reviewer_session_id": DEFAULT_REVIEWER_SESSION_ID,
        "case_id": case["case_id"],
        "annotation": annotation,
        "client_event_id": identifier,
        "idempotency_key": identifier,
        "expected_server_state_hash": expected_hash or store.state()["server_state_hash"],
        "elapsed_active_seconds": 3,
    }


def _completion_payload(store: K1ReviewPersistence, *, event_id: str | None = None) -> dict[str, Any]:
    identifier = event_id or str(uuid.uuid4())
    return {
        "event_type": "K1_REVIEW_COMPLETED",
        "review_id": REVIEW_ID,
        "reviewer_session_id": DEFAULT_REVIEWER_SESSION_ID,
        "client_event_id": identifier,
        "idempotency_key": identifier,
        "expected_server_state_hash": store.state()["server_state_hash"],
        "pending_outbox_events": 0,
        "unresolved_draft_count": 0,
        "evidence_blocker_count": 0,
        "unresolved_divergence": False,
    }


def test_package_is_target_only_answer_free_and_fresh(tmp_path: Path) -> None:
    package = _build_package(tmp_path)
    validation = validate_k1_package(package)
    manifest, config = load_k1_package(package)

    assert validation["passed"] is True
    assert manifest["review_id"] == REVIEW_ID
    assert manifest["target_count"] == 3
    assert manifest["source_group_count"] == 3
    assert manifest["selection_frozen_before_human_answers"] is True
    assert manifest["quota_shortfalls"]["team_1_goalkeeper"] == 10
    assert config["indexeddb_namespace"] == INDEXEDDB_NAMESPACE
    assert config["fresh_indexeddb_namespace"] is True
    assert config["prior_indexeddb_namespace_import_forbidden"] is True
    assert [field["name"] for question in config["questions"] for field in question["fields"]] == [
        "role",
        "team_affiliation",
        "kit_state",
        "pitch_state",
        "participation_state",
        "certainty",
    ]
    assert config["questions"][0]["fields"][0]["options"] == list(ROLE_VALUES)
    assert config["questions"][1]["fields"][0]["options"] == list(TEAM_VALUES)
    assert config["questions"][2]["fields"][0]["options"] == list(KIT_VALUES)
    assert config["questions"][3]["fields"][0]["options"] == list(PITCH_VALUES)
    assert config["questions"][3]["fields"][1]["options"] == list(PARTICIPATION_VALUES)
    assert config["questions"][4]["fields"][0]["options"] == list(CERTAINTY_VALUES)

    serialized = json.dumps(manifest).lower()
    assert "expected_answer" not in serialized
    assert "hidden_answer" not in serialized
    assert "identity_id" not in serialized
    for case in manifest["cases"]:
        assert case["target_only"] is True
        assert case["target"]["highlight_label"] == "TARGET"
        assert "targets" not in case
        assert case["current_frame"]["authoritative"] is True
        assert all(frame["context_only"] and not frame["authoritative"] for frame in case["context_frames"])
        evidence = package / "evidence" / case["case_id"]
        assert sha256_file(evidence / case["current_frame"]["relative_path"]) == case["source_frame_sha256"]
        assert sha256_file(evidence / case["target_crop"]["relative_path"]) == case["target_crop_sha256"]

    html = (package / "index.html").read_text(encoding="utf-8")
    assert "Label the highlighted target person only. Other people are context." in html
    assert "Current frame is authoritative" in html
    assert all(line in html for line in GUIDANCE)
    assert 'content: "TARGET"' in html
    launcher = (package / "launch_team_role_kit_review.ps1").read_text(encoding="utf-8")
    assert f"-LocalPort {PORT}" in launcher
    assert f"--host {HOST} --port {PORT}" in launcher
    assert "will not move ports" in launcher
    assert list((package / "decisions").iterdir()) == []


def test_generated_javascript_is_syntax_valid_and_has_durable_outbox(tmp_path: Path) -> None:
    package = _build_package(tmp_path, case_count=1)
    javascript = (package / "app.js").read_text(encoding="utf-8")
    assert 'createObjectStore("drafts"' in javascript
    assert 'createObjectStore("outbox"' in javascript
    assert "expected_server_state_hash" in javascript
    assert "Saved after server acknowledgement." in javascript
    assert 'await dbDelete("outbox"' in javascript
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    completed = subprocess.run([node, "--check", str(package / "app.js")], text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_axes_remain_independent_for_goalkeepers_and_warmup_players(tmp_path: Path) -> None:
    package = _build_package(tmp_path)
    store = _store(package)
    cases = store.manifest["cases"]
    annotations = [
        _annotation(cases[0], role="GOALKEEPER", team="TEAM_1", kit="MATCH_OUTFIELD_KIT"),
        _annotation(cases[1], role="GOALKEEPER", team="TEAM_2", kit="MATCH_GOALKEEPER_KIT"),
        _annotation(
            cases[2],
            role="OUTFIELD_PLAYER",
            team="TEAM_1",
            kit="WARMUP_OR_BIB",
            pitch="OFF_PITCH",
            participation="OFF_PITCH_SUBSTITUTE_OR_WARMING",
        ),
    ]
    for case, annotation in zip(cases, annotations, strict=True):
        store.save_event(_save_payload(store, case, annotation))
    state = store.state()
    saved = [state["annotations"][case["case_id"]]["annotation"] for case in cases]
    assert (saved[0]["role"], saved[0]["team_affiliation"], saved[0]["kit_state"]) == (
        "GOALKEEPER",
        "TEAM_1",
        "MATCH_OUTFIELD_KIT",
    )
    assert (saved[1]["role"], saved[1]["team_affiliation"]) == ("GOALKEEPER", "TEAM_2")
    assert saved[2]["role"] == "OUTFIELD_PLAYER"
    assert saved[2]["team_affiliation"] == "TEAM_1"
    assert saved[2]["kit_state"] == "WARMUP_OR_BIB"
    assert saved[2]["participation_state"] == "OFF_PITCH_SUBSTITUTE_OR_WARMING"

    invalid = _annotation(cases[0])
    invalid.pop("team_affiliation")
    with pytest.raises(ValueError, match="exactly the independent answer axes"):
        store.save_event(_save_payload(store, cases[0], invalid))


def test_idempotency_divergence_and_restart_replay(tmp_path: Path) -> None:
    package = _build_package(tmp_path, case_count=1)
    store = _store(package)
    case = store.manifest["cases"][0]
    initial_hash = store.state()["server_state_hash"]
    identifier = str(uuid.uuid4())
    payload = _save_payload(store, case, _annotation(case), expected_hash=initial_hash, event_id=identifier)
    saved = store.save_event(payload)
    assert saved["ack"]["idempotent_retry"] is False
    retried = store.save_event(payload)
    assert retried["ack"]["idempotent_retry"] is True
    assert retried["state"]["event_sequence"] == 1

    divergent = _save_payload(
        store,
        case,
        _annotation(case, certainty="PROBABLE"),
        expected_hash=initial_hash,
    )
    with pytest.raises(K1StateDivergenceError):
        store.save_event(divergent)

    store.state_path.unlink()
    restarted = _store(package)
    recovered = restarted.recover_authoritative_state()
    assert recovered["event_count"] == 1
    assert recovered["replayed_from_event_ledger"] is True
    assert recovered["state"]["annotations"][case["case_id"]]["annotation"]["certainty"] == "CERTAIN"
    assert recovered["state"]["server_state_hash"] == saved["state"]["server_state_hash"]


def test_completion_is_rollback_safe_idempotent_and_immutable(tmp_path: Path) -> None:
    package = _build_package(tmp_path, case_count=2)
    store = _store(package)
    for case in store.manifest["cases"]:
        store.save_event(_save_payload(store, case, _annotation(case)))

    blocked = _completion_payload(store)
    blocked["pending_outbox_events"] = 1
    with pytest.raises(ValueError, match="blocked"):
        store.complete(blocked)

    completion_id = str(uuid.uuid4())
    payload = _completion_payload(store, event_id=completion_id)
    with pytest.raises(OSError, match="injected interrupted"):
        store.complete(payload, fail_after_replace=1)
    assert store.state()["completed"] is False
    assert len(store.events_path.read_text(encoding="utf-8").splitlines()) == 2
    assert validate_completion_bundle(package / "decisions")["passed"] is False

    completed = store.complete(payload)
    assert completed["state"]["completed"] is True
    assert completed["state"]["event_sequence"] == 3
    assert validate_completion_bundle(package / "decisions")["passed"] is True
    summary = json.loads((package / "decisions" / "completed_review_summary.json").read_text(encoding="utf-8"))
    assert summary["k1"]["exact_target_count"] == 2
    assert summary["k1"]["source_group_count"] == 2
    assert summary["k1"]["selection_spec_sha256"] == "a" * 64
    assert summary["k1"]["quota_shortfalls"]["team_2_goalkeeper"] == 11
    assert summary["k1"]["axis_label_counts"]["role"] == {"OUTFIELD_PLAYER": 2}

    retried = store.complete(payload)
    assert retried["ack"]["idempotent_retry"] is True
    assert retried["state"]["event_sequence"] == 3
    with pytest.raises(ValueError, match="immutable"):
        store.save_event(_save_payload(store, store.manifest["cases"][0], _annotation(store.manifest["cases"][0])))

    (package / "decisions" / "completed_review_summary.json").unlink()
    restarted = _store(package)
    assert restarted.state()["completed"] is True
    assert validate_completion_bundle(package / "decisions")["passed"] is True


def test_rebuild_refuses_real_decisions_and_server_address_is_fixed(tmp_path: Path) -> None:
    package = _build_package(tmp_path, case_count=1)
    store = _store(package)
    case = store.manifest["cases"][0]
    store.save_event(_save_payload(store, case, _annotation(case)))

    with pytest.raises(ValueError, match="refusing to rebuild"):
        build_k1_package(
            package_root=package,
            cases=[
                K1CaseSpec(
                    case_id="replacement",
                    source_group_id="replacement-group",
                    source_image_path=tmp_path / "inputs" / "source_0.png",
                    crop_image_path=tmp_path / "inputs" / "crop_0.png",
                    target_box={"x1": 1, "y1": 1, "x2": 10, "y2": 20},
                )
            ],
            stage_id="M5_5G7A_TEST_STAGE",
            selection_spec_sha256="b" * 64,
            quota_shortfalls=TARGET_STRATA,
        )
    with pytest.raises(ValueError, match="fixed"):
        K1ServerConfig(package_root=package, port=PORT + 1)
    assert K1ServerConfig(package_root=package).host == HOST
    assert K1ServerConfig(package_root=package).port == PORT
