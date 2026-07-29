from __future__ import annotations

import json
from pathlib import Path

from football_intelligence.g7d_c1_visual_diagnosis_review import REVIEW_ID, REVISION, ReviewStore, sha256_file


def payload(event_type: str, **values: object) -> dict[str, object]:
    return {
        "schema_version": "football_intelligence.g7d_c1.human_visual_diagnosis_event.v1",
        "review_id": REVIEW_ID,
        "revision": REVISION,
        "event_type": event_type,
        "idempotency_key": f"test-{event_type}-{values.get('target_id', values.get('scene_id'))}",
        **values,
    }


def test_atomic_acknowledgement_and_completion_are_gated(tmp_path: Path) -> None:
    case = {
        "scene_id": "scene_01",
        "source_width": 100,
        "source_height": 50,
        "asset_name": "x.png",
        "targets": [{"target_id": "target_01", "source_box_xyxy": [1, 1, 10, 10]}],
    }
    (tmp_path / "review_cases.json").write_text(json.dumps({"cases": [case]}), encoding="utf-8")
    store = ReviewStore(tmp_path)
    status, result = store.complete({"review_id": REVIEW_ID, "revision": REVISION})
    assert status == 409 and result["error_code"] == "COMPLETION_GATED"
    decision = {
        "proposal_validity": "CLEAN_SINGLE_PERSON",
        "role": "OUTFIELD_PLAYER",
        "team": "TEAM_1",
        "participation": "ACTIVE",
        "pitch_state": "ON_PITCH",
        "occlusion": "NONE",
        "box_quality": "GOOD_SINGLE_PERSON_BOX",
        "certainty": "CERTAIN",
    }
    status, result = store.save(payload("candidate", target_id="target_01", scene_id="scene_01", decision=decision))
    assert status == 200 and result["status"] == "SAVED — SERVER ACKNOWLEDGED"
    assert (tmp_path / "review_receipts" / "acknowledgements" / f"ack-{result['event_id']}.json").is_file()
    review = {
        "full_frame_coverage_confirmed": True,
        "missed_people_source_xy": [
            {"source_xy": [10.5, 12.5], "role": "UNKNOWN_RELEVANT_PERSON", "certainty": "PROBABLE"}
        ],
        "off_pitch_proposal_burden": "LOW",
        "duplicate_or_overlap_burden": "LOW",
        "occlusion_burden": "LOW",
        "bottlenecks": ["SCALE_OR_PERSPECTIVE"],
    }
    status, _ = store.save(payload("scene", scene_id="scene_01", review=review))
    assert status == 200
    status, result = store.complete({"review_id": REVIEW_ID, "revision": REVISION})
    assert status == 200 and result["status"] == "ALL CASES COMPLETE"
    assert (tmp_path / "review_receipts" / "completion" / "final.json").is_file()


def test_duplicate_requires_same_scene_reference(tmp_path: Path) -> None:
    cases = [
        {
            "scene_id": "scene_01",
            "source_width": 10,
            "source_height": 10,
            "asset_name": "x.png",
            "targets": [
                {"target_id": "a", "source_box_xyxy": [0, 0, 1, 1]},
                {"target_id": "b", "source_box_xyxy": [1, 1, 2, 2]},
            ],
        }
    ]
    (tmp_path / "review_cases.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")
    decision = {
        "proposal_validity": "DUPLICATE_OF_ANOTHER_CANDIDATE",
        "duplicate_of_target_id": "b",
        "role": "UNKNOWN_PERSON_ROLE",
        "team": "UNKNOWN_TEAM",
        "participation": "UNKNOWN",
        "pitch_state": "UNCERTAIN",
        "occlusion": "UNCERTAIN",
        "box_quality": "UNCERTAIN",
        "certainty": "UNCERTAIN",
    }
    status, result = ReviewStore(tmp_path).save(
        payload("candidate", target_id="a", scene_id="scene_01", decision=decision)
    )
    assert status == 200 and result["ok"]


def test_built_package_is_blind_and_complete() -> None:
    project = Path(__file__).resolve().parents[2]
    stage = (
        project / "experiments/football_observation_reasoner/part 6/G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_v1"
    )
    cases = json.loads((stage / "02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE/review_cases.json").read_text(encoding="utf-8"))[
        "cases"
    ]
    assert len(cases) == 24 and sum(len(case["targets"]) for case in cases) == 192
    assert {case["match_id"] for case in cases} == {"118575", "117092"}
    browser = (stage / "02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE/review_cases.json").read_text(encoding="utf-8").lower()
    assert all(
        word not in browser
        for word in ("fold_outputs", "selection_metrics", "slot_index", "entropy", "top_probability")
    )
    handoff = stage / "04_REVIEW_PACK/CHATGPT_HANDOFF"
    manifest = json.loads((handoff / "10_MANIFEST.json").read_text(encoding="utf-8"))["files"]
    assert len(manifest) == 9 and all(
        (handoff / row["filename"]).stat().st_size == row["byte_size"]
        and sha256_file(handoff / row["filename"]) == row["sha256"]
        for row in manifest
    )
