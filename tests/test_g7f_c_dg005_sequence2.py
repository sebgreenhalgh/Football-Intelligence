from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import g7f_c_run_calibration_adjudication_reaudit as reaudit

from football_intelligence.calibration_adjudication import (
    ADJUDICATION_ASSERTION,
    CALIBRATION_IDS,
    PASS_KIND,
    CalibrationAdjudicationStore,
    sha256_file,
)
from football_intelligence.dense_person_gold import (
    COMPLETION_ASSERTION,
    DensePersonValidationError,
    build_final_event,
)
from football_intelligence.dense_person_reviewer import scan_blind_payload
from football_intelligence.dg005_sequence2_reviewer import (
    GUIDANCE_ACTION,
    GUIDANCE_PRIMARY,
    REVIEWER_RELEASE,
    TARGET_IMAGE_ID,
    TARGET_PERSON_ID,
    DG005Sequence2HTTPServer,
    DG005Sequence2ReviewerConfig,
    sequence2_truth_delta,
)


REPO = Path(__file__).resolve().parents[1]
STATIC = REPO / "src/football_intelligence/dg005_sequence2_reviewer_static"


def polygon(x1: int, y1: int, x2: int, y2: int) -> list[dict[str, int]]:
    return [{"x": x1, "y": y1}, {"x": x2, "y": y1}, {"x": x2, "y": y2}, {"x": x1, "y": y2}]


def frame(image_id: str, position: int) -> dict[str, object]:
    return {
        "anonymous_dense_image_id": image_id,
        "selection_status": "CALIBRATION_ONLY",
        "source_frame_sha256": f"{position:064x}",
        "source_width": 96,
        "source_height": 64,
        "review_queue_position": position,
        "all_frame_instance_lineage": [{"burst_id": "burst", "frame_sequence": position}],
    }


def document(image_id: str) -> dict[str, object]:
    people = [
        {
            "instance_id": "person-001",
            "relevance": "MATCH_RELEVANT",
            "visible_mask_components": [polygon(5, 5, 15, 30)],
        }
    ]
    if image_id == TARGET_IMAGE_ID:
        people.append(
            {
                "instance_id": TARGET_PERSON_ID,
                "relevance": "NON_MATCH_RELEVANT",
                "visible_mask_components": [polygon(40, 8, 50, 38)],
            }
        )
    return {
        "people": people,
        "ignore_regions": [],
        "reviewed_exhaustiveness_strips": list(range(8)),
        "unfinished_polygon": None,
        "completion_assertion": COMPLETION_ASSERTION,
    }


def build_fixture(tmp_path: Path) -> tuple[DG005Sequence2ReviewerConfig, Path, dict[str, object]]:
    frames = {image_id: frame(image_id, index) for index, image_id in enumerate(CALIBRATION_IDS, start=1)}
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"images": list(frames.values())}), encoding="utf-8")
    root = tmp_path / "decisions"
    frozen = {}
    for image_id in CALIBRATION_IDS:
        event, acknowledgement = build_final_event(
            frames[image_id],
            document(image_id),
            binding_hashes={"selection_manifest_sha256": "a" * 64},
            reviewer_release="G7F_C_DENSE_PERSON_REVIEWER_R2",
            pass_kind="FIRST_PASS",
            final_revision=7,
        )
        event_path = root / "events" / f"first_pass__{image_id}.json"
        acknowledgement_path = root / "acknowledgements" / f"first_pass__{image_id}.json"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        acknowledgement_path.parent.mkdir(parents=True, exist_ok=True)
        event_path.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        acknowledgement_path.write_text(json.dumps(acknowledgement, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        frozen[image_id] = {
            "event_file_sha256": sha256_file(event_path),
            "acknowledgement_file_sha256": sha256_file(acknowledgement_path),
        }
    checklists = {image_id: [] for image_id in CALIBRATION_IDS}
    first_store = CalibrationAdjudicationStore(
        root / "calibration_adjudication",
        original_decisions_root=root,
        frames=frames,
        binding_hashes={"selection_manifest_sha256": "a" * 64},
        reviewer_release="G7F_C_CALIBRATION_ADJUDICATION_REVIEWER_R2",
        audit_checklists=checklists,
        audit_checklist_sha256="b" * 64,
        frozen_parent_file_hashes=frozen,
    )
    metadata = {
        "adjudication_assertion": ADJUDICATION_ASSERTION,
        "repair_checklist_addressed": True,
    }
    for image_id in CALIBRATION_IDS:
        state = first_store.state(image_id)
        state["document"]["reviewed_exhaustiveness_strips"] = list(range(8))
        first_store.apply_action(
            {
                "action_id": f"sequence1-{image_id}",
                "action_type": "FINALIZE",
                "anonymous_dense_image_id": image_id,
                "pass_kind": PASS_KIND,
                "expected_revision": 0,
                "document": state["document"],
                "adjudication_metadata": metadata,
            }
        )
    sequence1_event_path = root / "calibration_adjudication/events/calibration_adjudication_001__DG-005.json"
    sequence1_ack_path = root / "calibration_adjudication/acknowledgements/calibration_adjudication_001__DG-005.json"
    sequence1_event = json.loads(sequence1_event_path.read_text(encoding="utf-8"))
    config = DG005Sequence2ReviewerConfig(
        selection_manifest_path=selection,
        assets_root=tmp_path / "assets",
        original_decisions_root=root,
        adjudication_root=root / "calibration_adjudication",
        binding_hashes={"selection_manifest_sha256": "a" * 64},
        audit_checklist_sha256="c" * 64,
        frozen_parent_file_hashes=frozen,
        expected_sequence1_event_id=sequence1_event["event_id"],
        expected_sequence1_event_sha256=sequence1_event["event_sha256"],
        expected_sequence1_event_file_sha256=sha256_file(sequence1_event_path),
        expected_sequence1_ack_file_sha256=sha256_file(sequence1_ack_path),
        port=0,
    )
    return config, root, sequence1_event


def target(document_value: dict[str, object]) -> dict[str, object]:
    return next(row for row in document_value["people"] if row["instance_id"] == TARGET_PERSON_ID)


def action(state: dict[str, object], action_id: str, action_type: str = "SAVE_DRAFT") -> dict[str, object]:
    return {
        "action_id": action_id,
        "action_type": action_type,
        "anonymous_dense_image_id": TARGET_IMAGE_ID,
        "pass_kind": PASS_KIND,
        "expected_revision": state["revision"],
        "document": copy.deepcopy(state["document"]),
        "adjudication_metadata": copy.deepcopy(state["adjudication_metadata"]),
    }


def test_queue_is_candidate_free_dg005_only_and_starts_unmodified_at_zero_strips(tmp_path: Path) -> None:
    config, _, _ = build_fixture(tmp_path)
    server = DG005Sequence2HTTPServer(config)
    try:
        bootstrap = server.bootstrap()
        state = server.store.state(TARGET_IMAGE_ID)
    finally:
        server.server_close()
    assert [row["anonymous_dense_image_id"] for row in bootstrap["queue"]] == [TARGET_IMAGE_ID]
    assert bootstrap["reviewer_release"] == REVIEWER_RELEASE
    assert bootstrap["guidance"] == [GUIDANCE_PRIMARY, GUIDANCE_ACTION]
    assert scan_blind_payload(bootstrap) == []
    assert bootstrap["tools"] == {
        "visible_mask_polygon": False,
        "multiple_components": False,
        "vertex_edit_delete": False,
        "ignore_region_polygon": False,
        "add_person": False,
        "delete_person": False,
        "delete_ignore_region": False,
        "relevance_target": TARGET_PERSON_ID,
        "zoom_pan": True,
        "exhaustiveness_strips": 8,
    }
    assert target(state["document"])["relevance"] == "NON_MATCH_RELEVANT"
    assert state["document"]["reviewed_exhaustiveness_strips"] == []
    assert state["document"]["completion_assertion"] is None
    assert state["adjudication_metadata"] == {
        "adjudication_assertion": None,
        "repair_checklist_addressed": False,
    }


def test_static_ui_has_only_relevance_and_confirmation_controls() -> None:
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "state.bootstrap.guidance[0]" in script
    assert "state.bootstrap.guidance[1]" in script
    for required in (
        "targetNonMatch",
        "targetMatch",
        "stripButtons",
        "officialConfirmed",
        "denseAssertion",
        "adjudicationAssertion",
        "finalizeSequence2",
    ):
        assert f'id="{required}"' in page
    for forbidden in (
        "newPerson",
        "addComponent",
        "newIgnore",
        "finishPolygon",
        "deletePerson",
        "deleteIgnore",
        "editRelevance",
    ):
        assert f'id="{forbidden}"' not in page


@pytest.mark.parametrize("mutation", ["geometry", "other_relevance", "add", "delete", "ignore", "target_id"])
def test_server_rejects_every_unauthorized_truth_mutation(tmp_path: Path, mutation: str) -> None:
    config, root, _ = build_fixture(tmp_path)
    server = DG005Sequence2HTTPServer(config)
    sequence1_before = {path: path.read_bytes() for path in (root / "calibration_adjudication").rglob("*001__*.json")}
    try:
        state = server.store.state(TARGET_IMAGE_ID)
        request = action(state, f"crafted-{mutation}")
        people = request["document"]["people"]
        if mutation == "geometry":
            people[0]["visible_mask_components"][0][0]["x"] += 1
        elif mutation == "other_relevance":
            people[0]["relevance"] = "NON_MATCH_RELEVANT"
        elif mutation == "add":
            people.append(copy.deepcopy(people[0]))
        elif mutation == "delete":
            people.pop(0)
        elif mutation == "ignore":
            request["document"]["ignore_regions"] = [
                {"ignore_region_id": "ignore-001", "reason": "OTHER", "polygon": polygon(1, 1, 3, 3)}
            ]
        else:
            target(request["document"])["instance_id"] = "person-999"
        with pytest.raises(DensePersonValidationError):
            server.store.apply_action(request)
        assert server.store.state(TARGET_IMAGE_ID)["revision"] == 0
    finally:
        server.server_close()
    assert {path: path.read_bytes() for path in sequence1_before} == sequence1_before


def test_finalize_requires_human_delta_strips_and_assertions_then_is_idempotent(tmp_path: Path) -> None:
    config, root, sequence1_event = build_fixture(tmp_path)
    server = DG005Sequence2HTTPServer(config)
    sequence1_before = {path: path.read_bytes() for path in (root / "calibration_adjudication").rglob("*001__*.json")}
    try:
        state = server.store.state(TARGET_IMAGE_ID)
        with pytest.raises(DensePersonValidationError, match="all 8|MATCH_RELEVANT|assertion"):
            server.store.apply_action(action(state, "too-early", "FINALIZE"))
        request = action(state, "final-fixed", "FINALIZE")
        target(request["document"])["relevance"] = "MATCH_RELEVANT"
        request["document"]["reviewed_exhaustiveness_strips"] = list(range(8))
        request["document"]["completion_assertion"] = COMPLETION_ASSERTION
        request["adjudication_metadata"] = {
            "adjudication_assertion": ADJUDICATION_ASSERTION,
            "repair_checklist_addressed": True,
        }
        first = server.store.apply_action(request)
        second = server.store.apply_action(request)
        assert first == second
        finalized = server.store.state(TARGET_IMAGE_ID)
        assert finalized["finalized"] is True
        assert finalized["truth_delta"] == [
            {
                "field": "person-030.relevance",
                "from": "NON_MATCH_RELEVANT",
                "to": "MATCH_RELEVANT",
            }
        ]
        sequence2_event = json.loads(
            (root / "calibration_adjudication/events/calibration_adjudication_002__DG-005.json").read_text(
                encoding="utf-8"
            )
        )
        assert sequence2_event["supersedes_event_id"] == sequence1_event["event_id"]
        assert sequence2_event["supersedes_event_sha256"] == sequence1_event["event_sha256"]
        assert sequence2_event["reviewer_release"] == REVIEWER_RELEASE
        assert (
            sequence2_truth_delta(sequence1_event["annotation"], sequence2_event["annotation"])
            == finalized["truth_delta"]
        )
        selection = json.loads(config.selection_manifest_path.read_text(encoding="utf-8"))
        frames = {row["anonymous_dense_image_id"]: row for row in selection["images"]}
        frozen = {
            image_id: {
                "event_file_sha256": sha256_file(root / "events" / f"first_pass__{image_id}.json"),
                "acknowledgement_file_sha256": sha256_file(root / "acknowledgements" / f"first_pass__{image_id}.json"),
            }
            for image_id in CALIBRATION_IDS
        }
        chains, missing = reaudit.validated_adjudication_chains(
            adjudication_root=root / "calibration_adjudication",
            original_root=root,
            frames=frames,
            frozen=frozen,
            bindings={"selection_manifest_sha256": "a" * 64},
        )
        assert missing == []
        assert {image_id: len(rows) for image_id, rows in chains.items()} == {
            **{image_id: 1 for image_id in CALIBRATION_IDS},
            TARGET_IMAGE_ID: 2,
        }
        assert chains[TARGET_IMAGE_ID][-1]["event"]["event_id"] == sequence2_event["event_id"]
    finally:
        server.server_close()
    assert {path: path.read_bytes() for path in sequence1_before} == sequence1_before
