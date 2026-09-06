from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_intelligence.calibration_adjudication import (
    ADJUDICATION_ASSERTION,
    CALIBRATION_IDS,
    PASS_KIND,
    CalibrationAdjudicationStore,
    editable_seed,
    load_original_parent,
    sha256_file,
    validate_adjudication_pair,
)
from football_intelligence.calibration_adjudication_reviewer import (
    SCOPE_REMINDER,
    SCOPE_REMINDER_SECOND_LINE,
    CalibrationAdjudicationHTTPServer,
    CalibrationAdjudicationReviewerConfig,
)
from football_intelligence.dense_person_gold import (
    COMPLETION_ASSERTION,
    DensePersonConflictError,
    build_final_event,
)
from football_intelligence.dense_person_reviewer import scan_blind_payload


REPO = Path(__file__).resolve().parents[1]
STATIC = REPO / "src/football_intelligence/dense_person_reviewer_static"


def polygon(x1: int, y1: int, x2: int, y2: int) -> list[dict[str, int]]:
    return [{"x": x1, "y": y1}, {"x": x2, "y": y1}, {"x": x2, "y": y2}, {"x": x1, "y": y2}]


def frame(image_id: str, position: int) -> dict[str, object]:
    return {
        "anonymous_dense_image_id": image_id,
        "selection_status": "CALIBRATION_ONLY",
        "source_frame_sha256": f"{position:064x}",
        "source_width": 64,
        "source_height": 48,
        "review_queue_position": position,
        "all_frame_instance_lineage": [{"burst_id": "burst", "frame_sequence": position}],
    }


def complete_document(instance_id: str, relevance: str = "MATCH_RELEVANT") -> dict[str, object]:
    return {
        "people": [
            {
                "instance_id": instance_id,
                "relevance": relevance,
                "visible_mask_components": [polygon(10, 8, 18, 24)],
            }
        ],
        "ignore_regions": [],
        "reviewed_exhaustiveness_strips": list(range(8)),
        "unfinished_polygon": None,
        "completion_assertion": COMPLETION_ASSERTION,
    }


def build_originals(tmp_path: Path) -> tuple[dict[str, dict[str, object]], Path, dict[str, dict[str, str]]]:
    frames = {image_id: frame(image_id, index) for index, image_id in enumerate(CALIBRATION_IDS, start=1)}
    root = tmp_path / "original"
    frozen = {}
    for image_id in CALIBRATION_IDS:
        instance_id = "person-030" if image_id == "DG-005" else "person-001"
        relevance = "NON_MATCH_RELEVANT" if image_id == "DG-005" else "MATCH_RELEVANT"
        event, ack = build_final_event(
            frames[image_id],
            complete_document(instance_id, relevance),
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
        ack_path.write_text(json.dumps(ack, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        frozen[image_id] = {
            "event_file_sha256": sha256_file(event_path),
            "acknowledgement_file_sha256": sha256_file(ack_path),
        }
    return frames, root, frozen


def checklist() -> dict[str, list[dict[str, object]]]:
    return {
        image_id: [
            {
                "instance_or_ignore_id": None,
                "source_region": f"repair region {image_id}",
                "issue": "A visible human was omitted.",
                "required_correction_type": "Add a visible-only person mask.",
            }
        ]
        for image_id in CALIBRATION_IDS
    }


def make_store(tmp_path: Path) -> tuple[CalibrationAdjudicationStore, dict[str, dict[str, object]], Path, Path]:
    frames, original, frozen = build_originals(tmp_path)
    adjudication = tmp_path / "adjudication"
    store = CalibrationAdjudicationStore(
        adjudication,
        original_decisions_root=original,
        frames=frames,
        binding_hashes={"selection_manifest_sha256": "a" * 64},
        reviewer_release="G7F_C_CALIBRATION_ADJUDICATION_REVIEWER_R1",
        audit_checklists=checklist(),
        audit_checklist_sha256="b" * 64,
        frozen_parent_file_hashes=frozen,
    )
    return store, frames, original, adjudication


def test_seed_is_parent_exact_for_truth_but_resets_adjudication_progress(tmp_path: Path) -> None:
    store, frames, original, adjudication = make_store(tmp_path)
    original_bytes = {path: path.read_bytes() for path in original.rglob("*.json")}
    for image_id in CALIBRATION_IDS:
        parent = load_original_parent(original, frames[image_id])
        state = store.state(image_id)
        expected = editable_seed(parent)
        assert state["document"] == expected
        assert state["document"]["reviewed_exhaustiveness_strips"] == []
        assert state["document"]["completion_assertion"] == COMPLETION_ASSERTION
        assert state["parent_provenance"]["original_reviewed_exhaustiveness_strips"] == list(range(8))
        assert state["adjudication_metadata"] == {
            "adjudication_assertion": None,
            "repair_checklist_addressed": False,
        }
    assert store.state("DG-005")["document"]["people"][0]["relevance"] == "NON_MATCH_RELEVANT"
    assert not adjudication.exists()
    assert {path: path.read_bytes() for path in original.rglob("*.json")} == original_bytes


def test_finalization_is_append_only_exact_and_parent_bound(tmp_path: Path) -> None:
    store, frames, original, adjudication = make_store(tmp_path)
    original_bytes = {path: path.read_bytes() for path in original.rglob("*.json")}
    state = store.state("DG-001")
    document = state["document"]
    document["people"].append(
        {
            "instance_id": "person-002",
            "relevance": "NON_MATCH_RELEVANT",
            "visible_mask_components": [polygon(30, 12, 38, 30)],
        }
    )
    document["reviewed_exhaustiveness_strips"] = list(range(8))
    result = store.apply_action(
        {
            "action_id": "final-001",
            "action_type": "FINALIZE",
            "anonymous_dense_image_id": "DG-001",
            "pass_kind": PASS_KIND,
            "expected_revision": 0,
            "document": document,
            "adjudication_metadata": {
                "adjudication_assertion": ADJUDICATION_ASSERTION,
                "repair_checklist_addressed": True,
            },
        }
    )
    assert result["finalized"] is True
    event_path = adjudication / "events/calibration_adjudication_001__DG-001.json"
    ack_path = adjudication / "acknowledgements/calibration_adjudication_001__DG-001.json"
    parent = load_original_parent(original, frames["DG-001"])
    event, ack = validate_adjudication_pair(
        event_path,
        ack_path,
        frame=frames["DG-001"],
        parent=parent,
        expected_binding_hashes={"selection_manifest_sha256": "a" * 64},
    )
    assert event["supersedes_event_id"] == parent.event_id
    assert event["supersedes_event_sha256"] == parent.event_sha256
    assert event["annotation"]["people"][0] == parent.event["annotation"]["people"][0]
    assert len(event["annotation"]["people"]) == 2
    assert ack["event_sha256"] == event["event_sha256"]
    assert store.state("DG-001")["read_only"] is True
    with pytest.raises(DensePersonConflictError, match="immutable"):
        store.apply_action(
            {
                "action_id": "after-final",
                "action_type": "SAVE_DRAFT",
                "anonymous_dense_image_id": "DG-001",
                "pass_kind": PASS_KIND,
                "expected_revision": 1,
                "document": document,
            }
        )
    assert {path: path.read_bytes() for path in original.rglob("*.json")} == original_bytes


def test_second_adjudication_seeds_latest_truth_and_extends_chain(tmp_path: Path) -> None:
    store, frames, original, adjudication = make_store(tmp_path)
    first_document = store.state("DG-001")["document"]
    first_document["reviewed_exhaustiveness_strips"] = list(range(8))
    metadata = {
        "adjudication_assertion": ADJUDICATION_ASSERTION,
        "repair_checklist_addressed": True,
    }
    store.apply_action(
        {
            "action_id": "first-final",
            "action_type": "FINALIZE",
            "anonymous_dense_image_id": "DG-001",
            "pass_kind": PASS_KIND,
            "expected_revision": 0,
            "document": first_document,
            "adjudication_metadata": metadata,
        }
    )
    frozen = {
        image_id: {
            "event_file_sha256": sha256_file(original / "events" / f"first_pass__{image_id}.json"),
            "acknowledgement_file_sha256": sha256_file(original / "acknowledgements" / f"first_pass__{image_id}.json"),
        }
        for image_id in CALIBRATION_IDS
    }
    second_store = CalibrationAdjudicationStore(
        adjudication,
        original_decisions_root=original,
        frames=frames,
        binding_hashes={"selection_manifest_sha256": "a" * 64},
        reviewer_release="G7F_C_CALIBRATION_ADJUDICATION_REVIEWER_R1",
        audit_checklists=checklist(),
        audit_checklist_sha256="b" * 64,
        frozen_parent_file_hashes=frozen,
        adjudication_sequence=2,
    )
    state = second_store.state("DG-001")
    assert state["document"]["reviewed_exhaustiveness_strips"] == []
    assert state["seeded_from_immutable_first_pass"] is False
    state["document"]["reviewed_exhaustiveness_strips"] = list(range(8))
    second_store.apply_action(
        {
            "action_id": "second-final",
            "action_type": "FINALIZE",
            "anonymous_dense_image_id": "DG-001",
            "pass_kind": PASS_KIND,
            "expected_revision": 0,
            "document": state["document"],
            "adjudication_metadata": metadata,
        }
    )
    first = json.loads((adjudication / "events/calibration_adjudication_001__DG-001.json").read_text(encoding="utf-8"))
    second = json.loads((adjudication / "events/calibration_adjudication_002__DG-001.json").read_text(encoding="utf-8"))
    assert second["adjudication_sequence"] == 2
    assert second["supersedes_event_id"] == first["event_id"]
    assert second["supersedes_event_sha256"] == first["event_sha256"]


def test_reviewer_is_six_frame_candidate_free_and_shows_exact_audit_guidance(tmp_path: Path) -> None:
    frames, original, frozen = build_originals(tmp_path)
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps({"images": [*frames.values(), {**frame("DG-007", 7), "selection_status": "SCORED_DENSE_GOLD"}]}),
        encoding="utf-8",
    )
    assessment = {
        "images": [
            {
                "anonymous_dense_image_id": image_id,
                "visual_status": "REPAIR_REQUIRED",
                "blocking_findings": findings,
            }
            for image_id, findings in checklist().items()
        ]
    }
    assessment_path = tmp_path / "assessment.json"
    assessment_path.write_text(json.dumps(assessment), encoding="utf-8")
    server = CalibrationAdjudicationHTTPServer(
        CalibrationAdjudicationReviewerConfig(
            selection_manifest_path=selection_path,
            assets_root=tmp_path / "assets",
            original_decisions_root=original,
            adjudication_root=tmp_path / "adjudication",
            binding_hashes={},
            reviewer_release="G7F_C_CALIBRATION_ADJUDICATION_REVIEWER_R1",
            audit_assessment_path=assessment_path,
            audit_checklist_sha256="b" * 64,
            frozen_parent_file_hashes=frozen,
            port=0,
        )
    )
    try:
        bootstrap = server.bootstrap()
        assert [row["anonymous_dense_image_id"] for row in bootstrap["queue"]] == list(CALIBRATION_IDS)
        assert bootstrap["pass_kind"] == PASS_KIND
        assert bootstrap["scope_reminder"] == [SCOPE_REMINDER, SCOPE_REMINDER_SECOND_LINE]
        assert bootstrap["queue"][0]["repair_checklist"] == checklist()["DG-001"]
        assert scan_blind_payload(bootstrap) == []
    finally:
        server.server_close()


def test_static_reviewer_supports_adjudication_metadata_and_relevance_editing() -> None:
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    styles = (STATIC / "styles.css").read_text(encoding="utf-8")
    for element_id in (
        "scopeReminder",
        "auditChecklistPanel",
        "repairChecklistAddressed",
        "adjudicationAssertion",
        "editRelevance",
    ):
        assert f'id="{element_id}"' in page
    assert ADJUDICATION_ASSERTION in script
    assert "CALIBRATION_ADJUDICATION" in script
    assert "const metadataSnapshot = clone(state.adjudicationMetadata)" in script
    assert "adjudication_metadata: metadataSnapshot" in script
    assert "state.document.people[index].relevance" in script
    assert "body.has-scope-reminder main" in styles


def test_new_event_and_ack_contracts_are_distinct_from_frozen_first_pass() -> None:
    event_schema = json.loads(
        (REPO / "src/football_intelligence/g7f_c_calibration_adjudication_event.schema.json").read_text()
    )
    ack_schema = json.loads(
        (REPO / "src/football_intelligence/g7f_c_calibration_adjudication_acknowledgement.schema.json").read_text()
    )
    assert event_schema["$id"] == "football_intelligence.g7f_c.calibration_superseding_annotation.v1"
    assert ack_schema["$id"] == "football_intelligence.g7f_c.calibration_superseding_acknowledgement.v1"
    assert event_schema["properties"]["pass_kind"]["const"] == PASS_KIND
