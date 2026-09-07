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
from football_intelligence.dg004_sequence2_reviewer import (
    GUIDANCE_ACTION,
    GUIDANCE_PRIMARY,
    REVIEWER_RELEASE,
    TARGET_IMAGE_ID,
    DG004Sequence2HTTPServer,
    DG004Sequence2ReviewerConfig,
    derive_next_person_id,
    sequence2_truth_delta,
)
from football_intelligence.dg005_sequence2_reviewer import REVIEWER_RELEASE as DG005_SEQUENCE2_RELEASE


REPO = Path(__file__).resolve().parents[1]
STATIC = REPO / "src/football_intelligence/dg004_sequence2_reviewer_static"


def polygon(x1: int, y1: int, x2: int, y2: int) -> list[dict[str, int]]:
    return [{"x": x1, "y": y1}, {"x": x2, "y": y1}, {"x": x2, "y": y2}, {"x": x1, "y": y2}]


def frame(image_id: str, position: int) -> dict[str, object]:
    return {
        "anonymous_dense_image_id": image_id,
        "selection_status": "CALIBRATION_ONLY",
        "source_frame_sha256": f"{position:064x}",
        "source_width": 128,
        "source_height": 96,
        "review_queue_position": position,
        "all_frame_instance_lineage": [{"burst_id": "burst", "frame_sequence": position}],
    }


def document(image_id: str) -> dict[str, object]:
    people = [
        {
            "instance_id": "person-001",
            "relevance": "MATCH_RELEVANT",
            "visible_mask_components": [polygon(5, 5, 15, 35)],
        },
        {
            "instance_id": "person-002",
            "relevance": "NON_MATCH_RELEVANT",
            "visible_mask_components": [polygon(20, 7, 30, 40)],
        },
    ]
    if image_id == "DG-005":
        people.append(
            {
                "instance_id": "person-030",
                "relevance": "NON_MATCH_RELEVANT",
                "visible_mask_components": [polygon(40, 8, 50, 42)],
            }
        )
    return {
        "people": people,
        "ignore_regions": [],
        "reviewed_exhaustiveness_strips": list(range(8)),
        "unfinished_polygon": None,
        "completion_assertion": COMPLETION_ASSERTION,
    }


def metadata() -> dict[str, object]:
    return {
        "adjudication_assertion": ADJUDICATION_ASSERTION,
        "repair_checklist_addressed": True,
    }


def build_fixture(
    tmp_path: Path, *, include_dg005_sequence2: bool = False
) -> tuple[DG004Sequence2ReviewerConfig, Path, dict[str, object], str]:
    frames = {image_id: frame(image_id, index) for index, image_id in enumerate(CALIBRATION_IDS, start=1)}
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"images": list(frames.values())}), encoding="utf-8")
    root = tmp_path / "TEMP_decisions"
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
        ack_path = root / "acknowledgements" / f"first_pass__{image_id}.json"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        ack_path.parent.mkdir(parents=True, exist_ok=True)
        event_path.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        ack_path.write_text(json.dumps(acknowledgement, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        frozen[image_id] = {
            "event_file_sha256": sha256_file(event_path),
            "acknowledgement_file_sha256": sha256_file(ack_path),
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
                "adjudication_metadata": metadata(),
            }
        )
    if include_dg005_sequence2:
        dg005_store = CalibrationAdjudicationStore(
            root / "calibration_adjudication",
            original_decisions_root=root,
            frames=frames,
            binding_hashes={"selection_manifest_sha256": "a" * 64},
            reviewer_release=DG005_SEQUENCE2_RELEASE,
            audit_checklists=checklists,
            audit_checklist_sha256="c" * 64,
            frozen_parent_file_hashes=frozen,
            adjudication_sequence=2,
        )
        dg005_state = dg005_store.state("DG-005")
        next(person for person in dg005_state["document"]["people"] if person["instance_id"] == "person-030")[
            "relevance"
        ] = "MATCH_RELEVANT"
        dg005_state["document"]["reviewed_exhaustiveness_strips"] = list(range(8))
        dg005_store.apply_action(
            {
                "action_id": "sequence2-DG-005",
                "action_type": "FINALIZE",
                "anonymous_dense_image_id": "DG-005",
                "pass_kind": PASS_KIND,
                "expected_revision": 0,
                "document": dg005_state["document"],
                "adjudication_metadata": metadata(),
            }
        )
    sequence1_event_path = root / "calibration_adjudication/events/calibration_adjudication_001__DG-004.json"
    sequence1_ack_path = root / "calibration_adjudication/acknowledgements/calibration_adjudication_001__DG-004.json"
    sequence1_event = json.loads(sequence1_event_path.read_text(encoding="utf-8"))
    new_person_id = derive_next_person_id(sequence1_event["annotation"])
    config = DG004Sequence2ReviewerConfig(
        selection_manifest_path=selection,
        assets_root=tmp_path / "assets",
        original_decisions_root=root,
        adjudication_root=root / "calibration_adjudication",
        binding_hashes={"selection_manifest_sha256": "a" * 64},
        audit_checklist_sha256="d" * 64,
        frozen_parent_file_hashes=frozen,
        expected_sequence1_event_id=sequence1_event["event_id"],
        expected_sequence1_event_sha256=sequence1_event["event_sha256"],
        expected_sequence1_event_file_sha256=sha256_file(sequence1_event_path),
        expected_sequence1_ack_file_sha256=sha256_file(sequence1_ack_path),
        expected_new_person_id=new_person_id,
        port=0,
    )
    return config, root, sequence1_event, new_person_id


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


def add_target(request: dict[str, object], new_person_id: str, *, relevance: str | None = None) -> None:
    request["document"]["people"].append(
        {
            "instance_id": new_person_id,
            "relevance": relevance,
            "visible_mask_components": [polygon(70, 12, 80, 48)],
        }
    )


def test_queue_is_candidate_free_dg004_only_and_seed_resets_fresh_state(tmp_path: Path) -> None:
    config, _, sequence1_event, new_person_id = build_fixture(tmp_path)
    server = DG004Sequence2HTTPServer(config)
    try:
        bootstrap = server.bootstrap()
        state = server.store.state(TARGET_IMAGE_ID)
    finally:
        server.server_close()
    assert [row["anonymous_dense_image_id"] for row in bootstrap["queue"]] == [TARGET_IMAGE_ID]
    assert bootstrap["reviewer_release"] == REVIEWER_RELEASE
    assert bootstrap["guidance"] == [GUIDANCE_PRIMARY, GUIDANCE_ACTION]
    assert bootstrap["repair"] == {
        "anonymous_dense_image_id": TARGET_IMAGE_ID,
        "strip_index": 2,
        "new_person_id": new_person_id,
    }
    assert scan_blind_payload(bootstrap) == []
    assert bootstrap["automatic_segmentation"] is False
    assert state["document"]["people"] == [
        {
            "instance_id": person["instance_id"],
            "relevance": person["relevance"],
            "visible_mask_components": person["canonical_components"],
        }
        for person in sequence1_event["annotation"]["people"]
    ]
    assert state["document"]["reviewed_exhaustiveness_strips"] == []
    assert state["document"]["completion_assertion"] is None
    assert state["adjudication_metadata"] == {
        "adjudication_assertion": None,
        "repair_checklist_addressed": False,
    }


def test_next_person_id_is_derived_and_frozen() -> None:
    annotation = {"people": [{"instance_id": "person-001"}, {"instance_id": "person-009"}]}
    assert derive_next_person_id(annotation) == "person-010"
    with pytest.raises(DensePersonValidationError, match="person-NNN"):
        derive_next_person_id({"people": [{"instance_id": "person-x"}]})


def test_static_ui_exposes_only_new_person_geometry_relevance_and_confirmations() -> None:
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    for required in (
        "panEdit",
        "drawNewPerson",
        "addComponent",
        "finishPolygon",
        "targetNonMatch",
        "targetMatch",
        "stripButtons",
        "omissionConfirmed",
        "denseAssertion",
        "adjudicationAssertion",
        "finalizeSequence2",
    ):
        assert f'id="{required}"' in page
    for forbidden in ("newIgnore", "deletePerson", "deleteIgnore", "editVertices", "candidateReveal"):
        assert f'id="{forbidden}"' not in page
    assert 'PAN_EDIT: "PAN_EDIT"' in script
    assert 'event.code === "Space"' in script


def test_one_new_person_and_multiple_visible_components_can_be_saved(tmp_path: Path) -> None:
    config, _, _, new_person_id = build_fixture(tmp_path)
    server = DG004Sequence2HTTPServer(config)
    try:
        state = server.store.state(TARGET_IMAGE_ID)
        first = action(state, "add-new")
        add_target(first, new_person_id)
        saved = server.store.apply_action(first)
        target = saved["document"]["people"][-1]
        assert target["instance_id"] == new_person_id
        assert target["relevance"] is None
        second = action(saved, "add-component")
        second["document"]["people"][-1]["visible_mask_components"].append(polygon(84, 14, 90, 30))
        second["document"]["people"][-1]["relevance"] = "NON_MATCH_RELEVANT"
        saved = server.store.apply_action(second)
        assert len(saved["document"]["people"][-1]["visible_mask_components"]) == 2
        assert saved["document"]["people"][-1]["relevance"] == "NON_MATCH_RELEVANT"
    finally:
        server.server_close()


@pytest.mark.parametrize(
    "mutation",
    ["existing_geometry", "existing_relevance", "delete", "ignore", "wrong_new_id", "second_new", "reorder"],
)
def test_server_rejects_every_unauthorized_truth_mutation(tmp_path: Path, mutation: str) -> None:
    config, root, _, new_person_id = build_fixture(tmp_path)
    server = DG004Sequence2HTTPServer(config)
    protected = {
        path: path.read_bytes()
        for path in (root / "calibration_adjudication").rglob("*.json")
        if "_001__" in path.name or "_002__DG-005" in path.name
    }
    try:
        state = server.store.state(TARGET_IMAGE_ID)
        request = action(state, f"crafted-{mutation}")
        people = request["document"]["people"]
        if mutation == "existing_geometry":
            people[0]["visible_mask_components"][0][0]["x"] += 1
        elif mutation == "existing_relevance":
            people[0]["relevance"] = "NON_MATCH_RELEVANT"
        elif mutation == "delete":
            people.pop(0)
        elif mutation == "ignore":
            request["document"]["ignore_regions"] = [
                {"ignore_region_id": "ignore-001", "reason": "OTHER", "polygon": polygon(1, 1, 3, 3)}
            ]
        elif mutation == "wrong_new_id":
            add_target(request, "person-999")
        elif mutation == "second_new":
            add_target(request, new_person_id)
            request["document"]["people"].append(
                {
                    "instance_id": "person-999",
                    "relevance": None,
                    "visible_mask_components": [polygon(90, 10, 100, 40)],
                }
            )
        else:
            people[0], people[1] = people[1], people[0]
        with pytest.raises(DensePersonValidationError):
            server.store.apply_action(request)
        assert server.store.state(TARGET_IMAGE_ID)["revision"] == 0
    finally:
        server.server_close()
    assert {path: path.read_bytes() for path in protected} == protected


@pytest.mark.parametrize("missing", ["person", "relevance", "strips", "assertions"])
def test_finalize_requires_the_complete_human_delta(tmp_path: Path, missing: str) -> None:
    config, _, _, new_person_id = build_fixture(tmp_path)
    server = DG004Sequence2HTTPServer(config)
    try:
        state = server.store.state(TARGET_IMAGE_ID)
        request = action(state, f"missing-{missing}", "FINALIZE")
        if missing != "person":
            add_target(request, new_person_id, relevance=None if missing == "relevance" else "MATCH_RELEVANT")
        if missing != "strips":
            request["document"]["reviewed_exhaustiveness_strips"] = list(range(8))
        if missing != "assertions":
            request["document"]["completion_assertion"] = COMPLETION_ASSERTION
            request["adjudication_metadata"] = metadata()
        with pytest.raises(DensePersonValidationError):
            server.store.apply_action(request)
    finally:
        server.server_close()


def test_successful_sequence2_is_exact_append_only_delta_and_double_finalize_safe(tmp_path: Path) -> None:
    config, root, sequence1_event, new_person_id = build_fixture(tmp_path, include_dg005_sequence2=True)
    protected = {
        path: path.read_bytes()
        for path in (root / "calibration_adjudication").rglob("*.json")
        if "_001__" in path.name or "_002__DG-005" in path.name
    }
    server = DG004Sequence2HTTPServer(config)
    try:
        state = server.store.state(TARGET_IMAGE_ID)
        request = action(state, "final-one-person", "FINALIZE")
        add_target(request, new_person_id, relevance="NON_MATCH_RELEVANT")
        request["document"]["people"][-1]["visible_mask_components"].append(polygon(84, 14, 90, 30))
        request["document"]["reviewed_exhaustiveness_strips"] = list(range(8))
        request["document"]["completion_assertion"] = COMPLETION_ASSERTION
        request["adjudication_metadata"] = metadata()
        first = server.store.apply_action(request)
        second = server.store.apply_action(request)
        assert first == second
        finalized = server.store.state(TARGET_IMAGE_ID)
        assert finalized["finalized"] is True
        assert finalized["truth_delta"][0]["change"] == "ADD_VISIBLE_PERSON"
        assert finalized["truth_delta"][0]["instance_id"] == new_person_id
        event_path = root / "calibration_adjudication/events/calibration_adjudication_002__DG-004.json"
        event = json.loads(event_path.read_text(encoding="utf-8"))
        assert event["supersedes_event_id"] == sequence1_event["event_id"]
        assert event["supersedes_event_sha256"] == sequence1_event["event_sha256"]
        assert event["reviewer_release"] == REVIEWER_RELEASE
        assert (
            sequence2_truth_delta(
                sequence1_event["annotation"], event["annotation"], expected_new_person_id=new_person_id
            )
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
        assert sum(len(rows) for rows in chains.values()) == 8
        assert len(chains["DG-004"]) == 2
        assert len(chains["DG-005"]) == 2
    finally:
        server.server_close()
    assert {path: path.read_bytes() for path in protected} == protected
