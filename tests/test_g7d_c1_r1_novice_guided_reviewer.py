from __future__ import annotations

import json
from pathlib import Path

from football_intelligence.g7d_c1_r1_novice_review import (
    DRAFT_SCHEMA,
    EVENT_SCHEMA,
    REVIEW_ID,
    REVISION,
    ReviewStore,
    sha256_file,
)

PROJECT = Path(__file__).resolve().parents[2]
STAGE = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_v1"
PACKAGE = STAGE / "02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE"
EVIDENCE = STAGE / "05_R1_NOVICE_GUIDED_REVIEWER_USABILITY_OVERHAUL"
HANDOFF = STAGE / "06_R1_REVIEW_PACK/CHATGPT_HANDOFF"


def candidate_payload(target_id: str, scene_id: str, key: str, decision: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": EVENT_SCHEMA,
        "review_id": REVIEW_ID,
        "revision": REVISION,
        "event_type": "candidate",
        "target_id": target_id,
        "scene_id": scene_id,
        "idempotency_key": key,
        "decision": decision,
    }


def clean_person() -> dict[str, object]:
    return {
        "proposal_validity": "CLEAN_SINGLE_PERSON",
        "role": "OUTFIELD_PLAYER",
        "team": "TEAM_1",
        "participation": "ACTIVE",
        "pitch_state": "ON_PITCH",
        "occlusion": "NONE",
        "box_quality": "GOOD_SINGLE_PERSON_BOX",
        "certainty": "CERTAIN",
        "notes": "",
    }


def make_store(tmp_path: Path) -> ReviewStore:
    cases = {
        "review_revision": REVISION,
        "cases": [
            {
                "scene_id": "scene_01",
                "match_id": "118575",
                "source_width": 100,
                "source_height": 50,
                "asset_name": "frame.png",
                "targets": [
                    {"target_id": "a", "source_box_xyxy": [1, 1, 10, 20]},
                    {"target_id": "b", "source_box_xyxy": [10, 1, 20, 20]},
                ],
            }
        ],
    }
    (tmp_path / "review_cases.json").write_text(json.dumps(cases), encoding="utf-8")
    return ReviewStore(tmp_path)


def test_server_draft_restore_immutable_truth_and_superseding_edits(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    draft = {
        "schema_version": DRAFT_SCHEMA,
        "review_id": REVIEW_ID,
        "revision": REVISION,
        "draft_type": "candidate",
        "scene_id": "scene_01",
        "target_id": "a",
        "step_index": 3,
        "answers": {"proposal_validity": "CLEAN_SINGLE_PERSON", "role": "OUTFIELD_PLAYER"},
        "missed_people_source_xy": [],
        "idempotency_key": "draft-a",
    }
    status, result = store.save_draft(draft)
    assert status == 200 and result["status"] == "Progress saved"
    assert store.state()["drafts"]["a"]["step_index"] == 3

    first_payload = candidate_payload("a", "scene_01", "final-a-1", clean_person())
    status, first = store.save(first_payload)
    assert status == 200 and first["status"] == "SAVED — SERVER ACKNOWLEDGED"
    event_path = tmp_path / "review_events" / "candidate" / f"{first['event_id']}.json"
    first_hash = sha256_file(event_path)
    status, repeat = store.save(first_payload)
    assert status == 200 and repeat["event_id"] == first["event_id"] and repeat["restored_idempotently"]
    assert sha256_file(event_path) == first_hash

    changed = clean_person() | {"certainty": "PROBABLE"}
    status, second = store.save(candidate_payload("a", "scene_01", "final-a-2", changed))
    assert status == 200 and second["supersedes_event_id"] == first["event_id"]
    assert event_path.is_file() and sha256_file(event_path) == first_hash
    assert store.state()["saved_candidates"]["a"]["event_id"] == second["event_id"]


def test_schema_validation_branch_defaults_duplicate_and_scene_gating(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    invalid_no_person = {
        "proposal_validity": "NO_PERSON_BACKGROUND_OR_OBJECT",
        "role": "OUTFIELD_PLAYER",
        "team": "TEAM_1",
        "participation": "ACTIVE",
        "pitch_state": "UNCERTAIN",
        "occlusion": "NONE",
        "box_quality": "GOOD_SINGLE_PERSON_BOX",
        "certainty": "CERTAIN",
    }
    status, result = store.save(candidate_payload("a", "scene_01", "bad", invalid_no_person))
    assert status == 422 and result["error_code"] == "IMPOSSIBLE_COMBINATION"

    duplicate = {
        "proposal_validity": "DUPLICATE_OF_ANOTHER_CANDIDATE",
        "role": "UNKNOWN_PERSON_ROLE",
        "team": "UNKNOWN_TEAM",
        "participation": "UNKNOWN",
        "pitch_state": "UNCERTAIN",
        "occlusion": "UNCERTAIN",
        "box_quality": "UNCERTAIN",
        "certainty": "PROBABLE",
        "duplicate_of_target_id": "b",
    }
    assert store.save(candidate_payload("a", "scene_01", "dup", duplicate))[0] == 200
    scene = {
        "schema_version": EVENT_SCHEMA,
        "review_id": REVIEW_ID,
        "revision": REVISION,
        "event_type": "scene",
        "scene_id": "scene_01",
        "idempotency_key": "scene-1",
        "review": {
            "full_frame_coverage_confirmed": True,
            "missed_people_source_xy": [],
            "off_pitch_proposal_burden": "LOW",
            "duplicate_or_overlap_burden": "MODERATE",
            "occlusion_burden": "NONE",
            "bottlenecks": ["NO_CLEAR_BOTTLENECK"],
        },
    }
    status, result = store.save(scene)
    assert status == 422 and result["error_code"] == "CANDIDATE_ACKNOWLEDGEMENTS_REQUIRED"
    assert store.save(candidate_payload("b", "scene_01", "b-final", clean_person()))[0] == 200
    assert store.save(scene)[0] == 200


def test_built_wizard_preserves_scope_and_hides_expert_form() -> None:
    document = json.loads((PACKAGE / "review_cases.json").read_text(encoding="utf-8"))
    assert document["review_revision"] == REVISION
    assert len(document["cases"]) == 24
    assert sum(len(case["targets"]) for case in document["cases"]) == 192
    assert {
        case["match_id"]: case["team_colours"]
        for case in document["cases"]
        if case["scene_id"] in {"scene_01_118575_118575_first_half_13", "scene_13_117092_117092_first_half_07"}
    } == {
        "118575": {"TEAM_1": "GREY", "TEAM_2": "BLUE"},
        "117092": {"TEAM_1": "BLUE", "TEAM_2": "WHITE"},
    }
    html = (PACKAGE / "index.html").read_text(encoding="utf-8")
    assert html.count('id="questionTitle"') == 1
    assert all(
        token not in html
        for token in ("proposal_validity", "pitch_state", "CLEAN_SINGLE_PERSON", "GOOD_SINGLE_PERSON_BOX")
    )
    assert "Show other boxes" in html and "Closer look" in html and "Show tutorial again" in html
    css = (PACKAGE / "styles.css").read_text(encoding="utf-8")
    assert "font: 18px" in css and "min-height: 66px" in css and "@media (max-width: 1100px)" in css
    javascript = (PACKAGE / "app.js").read_text(encoding="utf-8")
    for token in (
        "saveDraft",
        "Progress saved",
        "DUPLICATE_OF_ANOTHER_CANDIDATE",
        "markMissedPerson",
        "Backspace",
        "SAVED — SERVER ACKNOWLEDGED",
    ):
        assert token in javascript
    assert "fold_outputs" not in javascript and "top_probability" not in javascript


def test_evidence_two_visuals_and_self_excluding_handoff_manifest() -> None:
    preservation = json.loads((EVIDENCE / "target_and_asset_preservation.json").read_text(encoding="utf-8"))
    assert preservation["scene_count"] == 24 and preservation["target_count"] == 192
    assert preservation["frames_boxes_ids_and_selection_reasons_unchanged"]
    visuals = sorted((EVIDENCE / "visual_qa").glob("*.png"))
    assert [path.name for path in visuals] == ["01_candidate_wizard.png", "02_scene_wizard.png"]
    manifest = json.loads((HANDOFF / "10_MANIFEST.json").read_text(encoding="utf-8"))
    assert len(list(HANDOFF.iterdir())) == 10 and len(manifest["files"]) == 9 and not manifest["self_hashed"]
    for row in manifest["files"]:
        path = HANDOFF / row["filename"]
        assert path.stat().st_size == row["byte_size"] and sha256_file(path) == row["sha256"]
