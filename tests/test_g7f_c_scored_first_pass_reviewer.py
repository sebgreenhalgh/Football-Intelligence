from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from football_intelligence.dense_person_gold import (
    COMPLETION_ASSERTION,
    IGNORE_REASONS,
    RELEVANCE_CLASSES,
    DensePersonConflictError,
    coco_uncompressed_rle,
)
from football_intelligence.dense_person_reviewer import (
    REVIEW_FRAME_KEYS,
    DensePersonHTTPServer,
    DensePersonReviewerConfig,
    scan_blind_payload,
)
from scripts.g7f_c_run_scored_first_pass_reviewer import scored_queue


def frame(image_id: str, status: str, position: int) -> dict[str, object]:
    return {
        "anonymous_dense_image_id": image_id,
        "selection_status": status,
        "source_frame_sha256": f"{position:064x}",
        "source_width": 64,
        "source_height": 48,
        "review_queue_position": position,
        "all_frame_instance_lineage": [{"burst_id": "b", "frame_sequence": position}],
        "candidate_counts": {"forbidden": 99},
        "composite_disagreement_score": 1.0,
        "match_id": "must-not-reach-review-server",
    }


def selection() -> dict[str, object]:
    rows = [frame(f"DG-{index:03d}", "CALIBRATION_ONLY", index) for index in range(1, 7)]
    rows.extend(frame(f"DG-{index:03d}", "SCORED_DENSE_GOLD", index) for index in range(7, 55))
    return {"images": list(reversed(rows))}


def config(tmp_path: Path) -> DensePersonReviewerConfig:
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection()), encoding="utf-8")
    return DensePersonReviewerConfig(
        selection_manifest_path=selection_path,
        assets_root=tmp_path / "assets",
        decisions_root=tmp_path / "decisions",
        binding_hashes={"selection_manifest_sha256": "a" * 64},
        reviewer_release="G7F_C_SCORED_DENSE_PERSON_REVIEWER_R1",
        reveal_payload_path=None,
        allowed_selection_statuses=("SCORED_DENSE_GOLD",),
        candidate_reveal_enabled=False,
        port=0,
    )


def legacy_rle(mask: np.ndarray) -> dict[str, object]:
    flattened = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    counts: list[int] = []
    current, run = 0, 0
    for value in flattened:
        bit = int(value > 0)
        if bit == current:
            run += 1
        else:
            counts.append(run)
            current = bit
            run = 1
    counts.append(run)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((7, 9), dtype=np.uint8),
        np.ones((7, 9), dtype=np.uint8),
        np.eye(9, dtype=np.uint8)[:7],
        np.random.default_rng(71).integers(0, 3, size=(31, 43), dtype=np.uint8),
    ],
)
def test_vectorized_rle_is_exactly_legacy_equivalent(mask: np.ndarray) -> None:
    assert coco_uncompressed_rle(mask) == legacy_rle(mask)


def test_queue_is_exactly_48_scored_rows_in_frozen_order() -> None:
    rows = scored_queue(selection())
    assert len(rows) == 48
    assert [row["review_queue_position"] for row in rows] == list(range(7, 55))
    assert all(row["selection_status"] == "SCORED_DENSE_GOLD" for row in rows)
    assert len({row["source_frame_sha256"] for row in rows}) == 48


def test_scored_bootstrap_is_candidate_blind_and_server_frames_are_sanitized(tmp_path: Path) -> None:
    server = DensePersonHTTPServer(config(tmp_path))
    try:
        bootstrap = server.blind_bootstrap()
        assert len(bootstrap["queue"]) == 48
        assert {row["workflow_group"] for row in bootstrap["queue"]} == {"SCORED"}
        assert bootstrap["candidate_blind"] is True
        assert bootstrap["tools"]["post_finalize_comparison"] is False
        assert scan_blind_payload(bootstrap) == []
        assert all(set(row) == set(REVIEW_FRAME_KEYS) for row in server.frames.values())
        assert server.store.reveal_payloads == {}
    finally:
        server.server_close()


def test_every_never_started_scored_frame_is_blank(tmp_path: Path) -> None:
    server = DensePersonHTTPServer(config(tmp_path))
    try:
        for image_id in server.frames:
            state = server.store.state(image_id, "FIRST_PASS")
            assert state["revision"] == 0
            assert state["finalized"] is False
            assert state["document"] == {
                "people": [],
                "ignore_regions": [],
                "reviewed_exhaustiveness_strips": [],
                "unfinished_polygon": None,
                "completion_assertion": None,
            }
    finally:
        server.server_close()


def test_disabled_comparison_is_rejected_at_http_boundary(tmp_path: Path) -> None:
    server = DensePersonHTTPServer(config(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = json.dumps(
            {
                "action_id": "no-reveal",
                "action_type": "REVEAL_CANDIDATES",
                "anonymous_dense_image_id": "DG-007",
                "pass_kind": "FIRST_PASS",
                "expected_revision": 0,
            }
        ).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/api/action",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5)
        assert caught.value.code == 422
        payload = json.loads(caught.value.read())
        assert payload["error_code"] == "VALIDATION_ERROR"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_temp_scored_finalize_is_idempotent_and_creates_one_event_ack(tmp_path: Path) -> None:
    server = DensePersonHTTPServer(config(tmp_path))
    document = {
        "people": [
            {
                "instance_id": "person-001",
                "relevance": "MATCH_RELEVANT",
                "visible_mask_components": [
                    [{"x": 10, "y": 10}, {"x": 18, "y": 10}, {"x": 18, "y": 25}, {"x": 10, "y": 25}]
                ],
            }
        ],
        "ignore_regions": [],
        "reviewed_exhaustiveness_strips": list(range(8)),
        "unfinished_polygon": None,
        "completion_assertion": COMPLETION_ASSERTION,
    }
    action = {
        "action_id": "finalize-once",
        "action_type": "FINALIZE",
        "anonymous_dense_image_id": "DG-007",
        "pass_kind": "FIRST_PASS",
        "expected_revision": 0,
        "document": document,
    }
    try:
        first = server.store.apply_action(action)
        assert server.store.apply_action(action) == first
        with pytest.raises(DensePersonConflictError):
            server.store.apply_action({**action, "action_id": "second-click", "expected_revision": 1})
        decisions = config(tmp_path).decisions_root
        assert len(list((decisions / "events").glob("*.json"))) == 1
        assert len(list((decisions / "acknowledgements").glob("*.json"))) == 1
        event = json.loads((decisions / "events/first_pass__DG-007.json").read_text())
        assert event["pass_kind"] == "FIRST_PASS"
        assert event["selection_status"] == "SCORED_DENSE_GOLD"
        assert event["server_validation"]["candidate_data_used"] is False
        assert set(RELEVANCE_CLASSES) == {
            "MATCH_RELEVANT",
            "NON_MATCH_RELEVANT",
            "RELEVANCE_UNCERTAIN",
        }
        assert len(IGNORE_REASONS) == 5
    finally:
        server.server_close()


def test_draft_resume_is_image_bound_without_cross_image_leakage(tmp_path: Path) -> None:
    server = DensePersonHTTPServer(config(tmp_path))
    draft = {
        "people": [],
        "ignore_regions": [],
        "reviewed_exhaustiveness_strips": [0],
        "unfinished_polygon": None,
        "completion_assertion": None,
    }
    try:
        saved = server.store.apply_action(
            {
                "action_id": "save-dg007",
                "action_type": "SAVE_DRAFT",
                "anonymous_dense_image_id": "DG-007",
                "pass_kind": "FIRST_PASS",
                "expected_revision": 0,
                "document": draft,
            }
        )
        assert server.store.state("DG-007", "FIRST_PASS")["document"] == saved["document"]
        assert server.store.state("DG-008", "FIRST_PASS")["document"]["reviewed_exhaustiveness_strips"] == []
    finally:
        server.server_close()
