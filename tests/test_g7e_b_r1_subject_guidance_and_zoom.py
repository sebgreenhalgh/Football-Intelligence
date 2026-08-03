from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from football_intelligence.temporal_review import R1_REVIEW_REVISION, TemporalReviewStore, sha256_file

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPOSITORY = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
G7E_A = PART7 / "G7E_A_TARGETED_TEMPORAL_BURST_SELECTION_AND_ANNOTATION_DESIGN_v1"
G7E_B = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1"
STAGE = PART7 / "G7E_B_R1_SUBJECT_GUIDANCE_AND_ZOOM_USABILITY_REPAIR_v1"
PACKAGE = STAGE / "02_SUBJECT_GUIDANCE_IMPLEMENTATION/temporal_reviewer_r1"
OLD_PACKAGE = G7E_B / "03_TEMPORAL_REVIEWER"

BURST_SHA256 = "619b6847fdde14ae13ec8a2618ac90c7ac9fc7f4d7445336bfde529e5746909d"
FRAME_SHA256 = "96688c685cc495a05af4c70003ea02b3e3f5b2dd66cc2c4813e581b10c42723d"
TRANCHE_SHA256 = "bdb9e0c2a54718124467600c2af2ebede70eda4f386c03b7374f9191dfa29466"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_input_provenance_and_zero_real_event_preflight() -> None:
    provenance = read_json(STAGE / "00_INPUT_AND_EVENT_CLOSURE/input_provenance.json")
    preflight = read_json(STAGE / "00_INPUT_AND_EVENT_CLOSURE/event_root_preflight.json")
    assert provenance["classification"] == "PASS_G7E_B_R1_INPUT_PROVENANCE"
    assert provenance["repository_head"] == "f3243226c77c28323d78f8d00eb745f6980cde50"
    assert sha256_file(G7E_A / "02_BURST_SELECTION/temporal_burst_manifest.jsonl") == BURST_SHA256
    assert sha256_file(G7E_A / "02_BURST_SELECTION/temporal_frame_manifest.jsonl") == FRAME_SHA256
    assert sha256_file(G7E_B / "01_TRANCHE_CONTRACT/tranche_manifest.jsonl") == TRANCHE_SHA256
    assert preflight["real_event_count"] == 0
    assert preflight["real_acknowledgement_count"] == 0
    assert preflight["real_tranche_receipt_count"] == 0
    assert preflight["real_global_receipt_count"] == 0
    assert preflight["old_practice_draft_count"] == 1
    for path, digest in preflight["old_practice_draft_hashes"].items():
        historical = Path(path)
        assert len(digest) == 64
        if historical.is_file():
            assert sha256_file(historical) == digest


def test_exact_frozen_closure_and_no_asset_duplication() -> None:
    cases = read_json(PACKAGE / "review_cases.json")
    practices = read_json(PACKAGE / "practice_cases.json")
    old_cases = read_json(OLD_PACKAGE / "review_cases.json")
    assert cases["case_count"] == 120 and practices["case_count"] == 3
    assert Counter(row["tranche_id"] for row in cases["cases"]) == Counter({f"TRANCHE_{i}": 20 for i in range(1, 7)})
    assert [row["burst_id"] for row in cases["cases"]] == [row["burst_id"] for row in old_cases["cases"]]
    assert [row["frames"] for row in cases["cases"]] == [row["frames"] for row in old_cases["cases"]]
    assert all(len(row["frames"]) == 9 and len(row["frame_candidates"]) == 9 for row in cases["cases"])
    assert all(row["frame_candidates"][4] == row["candidates"] for row in cases["cases"])
    assert all(not row["frame_candidates"][index] for row in cases["cases"] for index in (*range(4), *range(5, 9)))
    package_manifest = read_json(STAGE / "02_SUBJECT_GUIDANCE_IMPLEMENTATION/package_manifest.json")
    assert package_manifest["asset_corpus_duplicated"] is False
    assert not (PACKAGE / "assets").exists()
    assert Path(package_manifest["asset_root"]) == OLD_PACKAGE / "assets"


def test_yellow_blue_subject_semantics_and_branch_contract() -> None:
    branch = read_json(PACKAGE / "reviewer_branch_contract.json")
    html = (PACKAGE / "index.html").read_text(encoding="utf-8")
    script = (PACKAGE / "review.js").read_text(encoding="utf-8")
    assert branch["first_question"] == "What does the yellow original focus box contain?"
    assert branch["yellow_box_first"] and branch["blue_context_is_not_counted"]
    for text in (
        "ORIGINAL FOCUS CANDIDATE",
        "CONTEXT AREA",
        "HUMAN-CONFIRMED subject location",
        "MODEL CANDIDATES — not identity",
        "What does the yellow original focus box contain?",
    ):
        assert text in html or text in script
    for value in (
        "ONE_RELEVANT_MATCH_PERSON",
        "PART_OF_ONE_RELEVANT_MATCH_PERSON",
        "MORE_THAN_ONE_RELEVANT_PERSON",
        "NO_RELEVANT_PERSON",
        "NOT_SURE",
    ):
        assert value in script
    assert "localStorage" not in script
    assert "team classification" not in script.lower()


def test_frame_by_frame_human_markers_and_candidate_supply_contract() -> None:
    script = (PACKAGE / "review.js").read_text(encoding="utf-8")
    marker = read_json(STAGE / "02_SUBJECT_GUIDANCE_IMPLEMENTATION/subject_marker_contract.json")
    assert marker["human_click_only"] and marker["automatic_trajectory"] is False
    assert marker["per_frame_locations"] == 9
    for token in (
        "Where is Subject ${subjectLetter(index)} in Frame ${frame + 1}?",
        "subject_location_source_x",
        "subject_location_source_y",
        "approximate_hidden_location",
        "marker_continuity_confirmation",
        "Which box evidence belongs to Subject ${subjectLetter(index)} in Frame ${frame + 1}?",
        "selected_candidate_ids",
        "NOT_APPLICABLE",
    ):
        assert token in script
    assert "Do not copy subject coordinates" not in script  # behavior is implemented structurally, not as a fake notice
    assert "row.subject_location_source_x = null" in script


def test_zoom_controls_transform_contract_and_visual_tokens() -> None:
    zoom = read_json(STAGE / "03_ZOOM_AND_COORDINATE_REPAIR/zoom_transform_contract.json")
    script = (PACKAGE / "review.js").read_text(encoding="utf-8")
    style = (PACKAGE / "review.css").read_text(encoding="utf-8")
    html = (PACKAGE / "index.html").read_text(encoding="utf-8")
    assert zoom["zoom_multiplier"] == {"minimum": 1.0, "maximum": 12.0}
    assert zoom["cursor_anchored_wheel_zoom"]
    assert zoom["lock_view_across_frames_default"]
    assert "Math.min(12" in script and "requestAnimationFrame" in script
    assert "displayToSource(sourceToDisplay(point" in script
    for control in ("Fit", "Zoom to Subject A", "Full screen", "Lock view across frames"):
        assert control in html
    for token in ("--navy:#111a33", "--surface:#fff", "--mint:#2cc9a0", "--blue:#5068e8", "--radius:22px"):
        assert token in style
    assert "prefers-reduced-motion" in style and "min-height:44px" in style


def test_old_practice_draft_is_rejected_not_migrated(tmp_path: Path) -> None:
    old_practice = tmp_path / "old_practice"
    draft_path = old_practice / "drafts/g7e_a_118576_01.json"
    draft_path.parent.mkdir(parents=True)
    draft_path.write_text(
        json.dumps(
            {
                "schema_version": "football_intelligence.g7e_b.temporal_review_draft.v1",
                "review_revision": "G7E_B_TEMPORAL_BURST_REVIEW_V1",
                "burst_id": "g7e_a_118576_01",
            }
        ),
        encoding="utf-8",
    )
    before = {path: sha256_file(path) for path in old_practice.glob("drafts/*.json")}
    store = TemporalReviewStore(PACKAGE, tmp_path / "real", old_practice, acceptance_mode=True)
    state = store.state("practice")
    assert state["draft"] is None
    assert state["incompatible_draft"] == {
        "burst_id": "g7e_a_118576_01",
        "stored_review_revision": "G7E_B_TEMPORAL_BURST_REVIEW_V1",
        "required_review_revision": R1_REVIEW_REVISION,
        "reset_required": True,
        "silently_migrated": False,
    }
    assert {path: sha256_file(path) for path in old_practice.glob("drafts/*.json")} == before


def test_r1_server_validation_and_receipt_chain(tmp_path: Path) -> None:
    store = TemporalReviewStore(PACKAGE, tmp_path / "real", tmp_path / "practice", acceptance_mode=True)
    case = store.by_tranche["TRANCHE_1"][0]
    payload = store.acceptance_event(case, "occlusion")
    result = store.save_event(payload, "real")
    assert result["acknowledgement_receipt_id"] == f"ack-{result['event_id']}"
    event_path = next((tmp_path / "real/events/TRANCHE_1").glob("*.json"))
    event = read_json(event_path)
    assert event["review_revision"] == R1_REVIEW_REVISION
    assert event["original_focus_box_answer"] == "ONE_RELEVANT_MATCH_PERSON"
    assert all(len(subject["frame_observations"]) == 9 for subject in event["subjects"])
    ack = read_json(tmp_path / f"real/receipts/acknowledgements/ack-{result['event_id']}.json")
    assert ack["event_sha256"] == sha256_file(event_path) and ack["server_validated"]


def test_r1_visible_location_and_candidate_selection_fail_closed(tmp_path: Path) -> None:
    store = TemporalReviewStore(PACKAGE, tmp_path / "real", tmp_path / "practice", acceptance_mode=True)
    case = store.by_tranche["TRANCHE_1"][0]
    payload = store.acceptance_event(case, "simple")
    payload["subjects"][0]["frame_observations"][0]["subject_location_source_x"] = None
    with pytest.raises(ValueError, match="human-confirmed location"):
        store.save_event(payload, "real")
    payload = store.acceptance_event(case, "simple")
    payload["subjects"][0]["frame_observations"][0]["observation_supply"] = "ONE_USEFUL_CANDIDATE"
    with pytest.raises(ValueError, match="selected box"):
        store.save_event(payload, "real")


def test_real_browser_acceptance_three_visual_cap_and_no_human_truth() -> None:
    report = read_json(STAGE / "04_BROWSER_ACCEPTANCE/browser_acceptance_report.json")
    assert report["classification"] == "PASS_G7E_B_R1_BROWSER_ACCEPTANCE"
    assert report["browser"] == "INSTALLED_MICROSOFT_EDGE"
    assert report["actual_local_server"] == "http://127.0.0.1:8818/"
    assert report["real_frozen_football_assets_visible"]
    assert report["uncaught_javascript_errors"] == []
    assert report["real_human_events_before"] == report["real_human_events_after"] == 0
    assert report["coordinate_round_trip_max_source_px_per_axis"] <= 0.5
    assert report["coordinate_round_trip_max_display_css_px_per_axis"] <= 1.0
    assert report["visual_transform_cached_p95_ms"] <= 16.0
    visuals = sorted((STAGE / "05_VISUAL_QA").glob("*.png"))
    assert [path.name for path in visuals] == [
        "01_CLARIFIED_FOCUS_AND_SUBJECT_A.png",
        "02_FRAME_BY_FRAME_SUBJECT_AND_CANDIDATES.png",
        "03_ZOOM_PAN_AND_SUBJECT_VIEW.png",
    ]
    assert all(path.stat().st_size > 100_000 for path in visuals)
    assert not (PACKAGE / "human_decisions").exists()


def test_handoff_is_exactly_twelve_self_contained_files() -> None:
    handoff = STAGE / "07_REVIEW_PACK/CHATGPT_HANDOFF"
    files = sorted(path for path in handoff.iterdir() if path.is_file())
    assert len(files) == 12
    manifest = read_json(handoff / "12_MANIFEST.json")
    assert len(manifest["files"]) == 11
    assert "12_MANIFEST.json" not in {row["filename"] for row in manifest["files"]}
    for row in manifest["files"]:
        path = handoff / row["filename"]
        assert path.stat().st_size == row["byte_size"]
        assert sha256_file(path) == row["sha256"]
    assert read_json(handoff / "01_EXECUTIVE_SUMMARY.json")["decision"] == (
        "PASS_G7E_B_R1_SUBJECT_GUIDANCE_AND_ZOOM_READY_FOR_PRACTICE_REVIEW"
    )
    digest = hashlib.sha256((handoff / "08_DECISION.md").read_bytes()).hexdigest()
    assert len(digest) == 64
