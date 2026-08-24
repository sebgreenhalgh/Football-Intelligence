from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from football_intelligence.dense_person_gold import (
    COMPLETION_ASSERTION,
    DensePersonConflictError,
    DensePersonDecisionStore,
    DensePersonValidationError,
    canonical_json_bytes,
    canonical_mask_geometry,
    candidate_ignored,
    decode_coco_uncompressed_rle,
    evaluate_dense_boxes,
    sha256_bytes,
    validate_and_canonicalize_document,
)
from football_intelligence.dense_person_reviewer import scan_blind_payload
from football_intelligence.dense_person_reviewer import DensePersonHTTPServer, DensePersonReviewerConfig


def polygon(x1: int, y1: int, x2: int, y2: int) -> list[dict[str, int]]:
    return [{"x": x1, "y": y1}, {"x": x2, "y": y1}, {"x": x2, "y": y2}, {"x": x1, "y": y2}]


def frame(image_id: str = "DG-001") -> dict[str, object]:
    return {
        "anonymous_dense_image_id": image_id,
        "selection_status": "SCORED_DENSE_GOLD",
        "source_frame_sha256": "a" * 64,
        "source_width": 64,
        "source_height": 48,
        "all_frame_instance_lineage": [{"burst_id": "burst", "frame_sequence": 1}],
    }


def complete_document() -> dict[str, object]:
    return {
        "people": [
            {
                "instance_id": "person-001",
                "relevance": "MATCH_RELEVANT",
                "visible_mask_components": [polygon(10, 8, 18, 24), polygon(21, 16, 23, 22)],
            }
        ],
        "ignore_regions": [
            {
                "ignore_region_id": "ignore-001",
                "reason": "SEVERE_VISUAL_ARTIFACT",
                "polygon": polygon(40, 4, 50, 18),
            }
        ],
        "reviewed_exhaustiveness_strips": list(range(8)),
        "unfinished_polygon": None,
        "completion_assertion": COMPLETION_ASSERTION,
    }


def test_mask_geometry_is_canonical_and_round_trips_rle() -> None:
    components = [polygon(20, 10, 24, 20), polygon(2, 3, 8, 15)]
    rotated_reversed = [list(reversed(component[2:] + component[:2])) for component in reversed(components)]
    left = canonical_mask_geometry(components, width=32, height=24)
    right = canonical_mask_geometry(rotated_reversed, width=32, height=24)
    assert left == right
    mask = decode_coco_uncompressed_rle(left["coco_uncompressed_rle"])
    assert int(mask.sum()) == left["visible_mask_area_px"]
    assert sha256_bytes(mask.tobytes(order="C")) == left["binary_mask_sha256"]
    assert left["derived_visible_box_xyxy"] == {"x1": 2, "y1": 3, "x2": 25, "y2": 21}


def test_finalization_requires_valid_people_strips_and_exact_assertion() -> None:
    canonical = validate_and_canonicalize_document(complete_document(), frame())
    assert len(canonical["strip_state"]) == 8
    assert canonical["people"][0]["component_count"] == 2
    assert canonical["people"][0]["derived_visible_box_height_px"] >= 6

    invalid = complete_document()
    invalid["reviewed_exhaustiveness_strips"] = list(range(7))
    with pytest.raises(DensePersonValidationError, match="eight"):
        validate_and_canonicalize_document(invalid, frame())

    invalid = complete_document()
    invalid["people"][0]["visible_mask_components"] = [polygon(1, 1, 5, 4)]
    with pytest.raises(DensePersonValidationError, match="height below 6"):
        validate_and_canonicalize_document(invalid, frame())

    invalid = complete_document()
    invalid["people"][0]["relevance"] = ""
    with pytest.raises(DensePersonValidationError, match="relevance"):
        validate_and_canonicalize_document(invalid, frame())

    zero_person = complete_document()
    zero_person["people"] = []
    assert validate_and_canonicalize_document(zero_person, frame())["zero_person_frame_allowed"] is True


def test_store_is_revisioned_idempotent_immutable_and_reveal_gated(tmp_path: Path) -> None:
    image = frame()
    store = DensePersonDecisionStore(
        tmp_path,
        frames={"DG-001": image},
        binding_hashes={"selection_manifest_sha256": "b" * 64},
        reviewer_release="reviewer-v1",
        reveal_payloads={"DG-001": {"runs": {"local": [{"candidate_id": "c1"}]}}},
    )
    assert scan_blind_payload(store.state("DG-001")) == []
    draft_action = {
        "action_id": "save-1",
        "action_type": "SAVE_DRAFT",
        "anonymous_dense_image_id": "DG-001",
        "pass_kind": "FIRST_PASS",
        "expected_revision": 0,
        "document": complete_document(),
    }
    saved = store.apply_action(draft_action)
    assert saved["revision"] == 1
    assert store.apply_action(draft_action) == saved
    assert scan_blind_payload(store.state("DG-001")) == []

    with pytest.raises(DensePersonConflictError) as stale:
        store.apply_action({**draft_action, "action_id": "save-stale"})
    assert stale.value.code == "STALE_REVISION"
    with pytest.raises(DensePersonConflictError) as early_reveal:
        store.apply_action(
            {
                "action_id": "reveal-early",
                "action_type": "REVEAL_CANDIDATES",
                "anonymous_dense_image_id": "DG-001",
                "pass_kind": "FIRST_PASS",
                "expected_revision": 1,
            }
        )
    assert early_reveal.value.code == "REVEAL_BEFORE_FINALIZATION"

    finalized = store.apply_action(
        {
            "action_id": "final-1",
            "action_type": "FINALIZE",
            "anonymous_dense_image_id": "DG-001",
            "pass_kind": "FIRST_PASS",
            "expected_revision": 1,
            "document": complete_document(),
        }
    )
    assert finalized["revision"] == 2
    event_path = tmp_path / "events/first_pass__DG-001.json"
    event_bytes = event_path.read_bytes()
    event = json.loads(event_bytes)
    event_payload = {key: value for key, value in event.items() if key not in {"event_id", "event_sha256"}}
    assert event["event_sha256"] == sha256_bytes(canonical_json_bytes(event_payload))
    assert store.state("DG-001")["document"] == event["annotation"]

    reveal = store.apply_action(
        {
            "action_id": "reveal-after",
            "action_type": "REVEAL_CANDIDATES",
            "anonymous_dense_image_id": "DG-001",
            "pass_kind": "FIRST_PASS",
            "expected_revision": 2,
        }
    )
    assert reveal["read_only"] is True
    assert reveal["candidate_comparison"]["runs"]["local"][0]["candidate_id"] == "c1"
    assert event_path.read_bytes() == event_bytes

    with pytest.raises(DensePersonConflictError) as immutable:
        store.apply_action({**draft_action, "action_id": "save-after", "expected_revision": 2})
    assert immutable.value.code == "FINALIZED_READ_ONLY"


def test_candidate_blind_payload_scanner_is_recursive() -> None:
    safe = {"candidate_blind": True, "queue": [{"anonymous_dense_image_id": "DG-001"}]}
    assert scan_blind_payload(safe) == []
    violations = scan_blind_payload({"queue": [{"subject_marker_score": 1}], "historical": {"x": 2}})
    assert violations == ["historical", "queue.0.subject_marker_score"]


def test_blind_repeat_queue_remains_sealed_and_uses_independent_ids(tmp_path: Path) -> None:
    selection = {"images": []}
    reveal = {"reveal_payloads": {}}
    for index in range(54):
        image_id = f"DG-{index + 1:03d}"
        selection["images"].append(
            {
                **frame(image_id),
                "source_frame_sha256": f"{index:064x}",
                "review_queue_position": index + 1,
            }
        )
        reveal["reveal_payloads"][image_id] = {"runs": {}}
    repeat = {
        "rows": [
            {"sealed_repeat_id": f"BR-{index + 1:03d}", "anonymous_dense_image_id": f"DG-{index + 1:03d}"}
            for index in range(6)
        ]
    }
    selection_path, reveal_path, repeat_path = (
        tmp_path / "selection.json",
        tmp_path / "reveal.json",
        tmp_path / "repeat.json",
    )
    for path, payload in ((selection_path, selection), (reveal_path, reveal), (repeat_path, repeat)):
        path.write_text(json.dumps(payload), encoding="utf-8")
    config = DensePersonReviewerConfig(
        selection_manifest_path=selection_path,
        assets_root=tmp_path / "assets",
        decisions_root=tmp_path / "decisions",
        binding_hashes={},
        reviewer_release="reviewer-v1",
        reveal_payload_path=reveal_path,
        pass_kind="BLIND_REPEAT",
        repeat_manifest_path=repeat_path,
        require_completed_first_pass=True,
        port=0,
    )
    with pytest.raises(RuntimeError, match="remain sealed"):
        DensePersonHTTPServer(config)
    for folder in ("events", "acknowledgements"):
        target = config.decisions_root / folder
        target.mkdir(parents=True)
        for index in range(54):
            (target / f"first_pass__DG-{index + 1:03d}.json").write_text("{}", encoding="utf-8")
    server = DensePersonHTTPServer(config)
    try:
        bootstrap = server.blind_bootstrap()
        assert bootstrap["pass_kind"] == "BLIND_REPEAT"
        assert [row["anonymous_dense_image_id"] for row in bootstrap["queue"]] == [
            f"BR-{index + 1:03d}" for index in range(6)
        ]
        assert scan_blind_payload(bootstrap) == []
        assert all("DG-" not in json.dumps(row) for row in bootstrap["queue"])
    finally:
        server.server_close()


def _metric_person(box: list[int], relevance: str = "MATCH_RELEVANT") -> dict[str, object]:
    geometry = canonical_mask_geometry([polygon(*box)], width=100, height=80)
    return {"instance_id": str(box), "relevance": relevance, **geometry}


def test_dense_metric_contract_duplicate_ignore_merge_and_calibration_exclusion() -> None:
    people = [_metric_person([10, 10, 25, 40]), _metric_person([30, 10, 45, 40])]
    ignore = canonical_mask_geometry([polygon(70, 10, 90, 40)], width=100, height=80)
    scored = {
        "selection_status": "SCORED_DENSE_GOLD",
        "source_frame_sha256": "1" * 64,
        "source_width": 100,
        "source_height": 80,
        "people": people,
        "ignore_regions": [ignore],
        "candidates": [
            {"candidate_id": "p1", "confidence": 0.99, "box_xyxy": [10, 10, 26, 41]},
            {"candidate_id": "p2", "confidence": 0.98, "box_xyxy": [30, 10, 46, 41]},
            {"candidate_id": "duplicate", "confidence": 0.50, "box_xyxy": [10, 10, 25, 40]},
            {"candidate_id": "merge", "confidence": 0.40, "box_xyxy": [10, 10, 46, 41]},
            {"candidate_id": "ignored", "confidence": 0.30, "box_xyxy": [72, 12, 88, 35]},
        ],
    }
    calibration = {
        **scored,
        "selection_status": "CALIBRATION_ONLY",
        "source_frame_sha256": "2" * 64,
        "candidates": [],
    }
    result = evaluate_dense_boxes([scored, calibration])
    assert result["sample_size_images"] == 1
    assert result["evaluable_person_denominator"] == 2
    assert result["AP50"] == 1.0
    assert result["recall_50"] == 1.0
    assert result["DUPLICATE_CANDIDATE_AT_IOU50"] == 1
    assert result["MULTI_PERSON_CANDIDATE_MASK_COVERAGE_030"] == 1
    assert result["by_iou_threshold"]["0.50"]["ignored_candidates"] == 1


def test_ignore_rule_uses_center_or_half_box_intersection() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    assert candidate_ignored([0, 0, 12, 12], mask)  # center inside
    assert candidate_ignored([4, 4, 16, 16], mask)  # at least half covered
    assert not candidate_ignored([0, 0, 6, 6], mask)
