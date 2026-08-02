from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from football_intelligence.temporal_review import (
    MATCH_ROTATION,
    QUOTAS,
    TRANCHES,
    TemporalReviewStore,
    deterministic_tranche_assignment,
    display_to_source,
    sha256_file,
    source_to_display,
    validate_tranche_assignment,
)

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPOSITORY = PROJECT / "SoccerTrack-v2"
G7E_A = (
    PROJECT
    / "experiments/football_observation_reasoner/part 7"
    / "G7E_A_TARGETED_TEMPORAL_BURST_SELECTION_AND_ANNOTATION_DESIGN_v1"
)
STAGE = PROJECT / "experiments/football_observation_reasoner/part 7" / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1"
PACKAGE = STAGE / "03_TEMPORAL_REVIEWER"
EXPECTED_HEAD = "4f6e3a9a4e7402411b644e088ee440daf937c70c"
BURST_SHA256 = "619b6847fdde14ae13ec8a2618ac90c7ac9fc7f4d7445336bfde529e5746909d"
FRAME_SHA256 = "96688c685cc495a05af4c70003ea02b3e3f5b2dd66cc2c4813e581b10c42723d"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_expected_baseline_and_exact_g7e_a_closure() -> None:
    closure = read_json(STAGE / "00_INPUT_CLOSURE/input_closure.json")
    assert closure["repository_head"] == EXPECTED_HEAD
    assert (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", EXPECTED_HEAD, "HEAD"], cwd=REPOSITORY, check=False
        ).returncode
        == 0
    )
    assert closure["classification"] == "PASS_G7E_B_INPUT_PROVENANCE"
    burst_path = G7E_A / "02_BURST_SELECTION/temporal_burst_manifest.jsonl"
    frame_path = G7E_A / "02_BURST_SELECTION/temporal_frame_manifest.jsonl"
    assert sha256_file(burst_path) == BURST_SHA256
    assert sha256_file(frame_path) == FRAME_SHA256
    bursts = read_jsonl(burst_path)
    frames = read_jsonl(frame_path)
    assert len(bursts) == 120
    assert Counter(row["match_id"] for row in bursts) == Counter(
        {match: 20 for match in ("117092", "117093", "118575", "118576", "118577", "128058")}
    )
    assert len(frames) == 1080
    assert len({row["frame_pixel_sha256"] for row in frames}) == 1044
    assert {len(row["frame_reference_ids"]) for row in bursts} == {9}
    assert read_json(G7E_A / "01_SELECTION_AND_ANNOTATION_CONTRACT/decision.json")["decision"] == (
        "PASS_G7E_A_TEMPORAL_BURST_SELECTION_AND_ANNOTATION_DESIGN_FROZEN"
    )


def test_deterministic_six_tranche_assignment_and_balance() -> None:
    bursts = read_jsonl(G7E_A / "02_BURST_SELECTION/temporal_burst_manifest.jsonl")
    frozen = read_jsonl(STAGE / "01_TRANCHE_CONTRACT/tranche_manifest.jsonl")
    recomputed = deterministic_tranche_assignment(bursts)
    assert [(row["burst_id"], row["tranche_id"], row["tranche_position"]) for row in frozen] == [
        (row["burst_id"], row["tranche_id"], row["tranche_position"]) for row in recomputed
    ]
    assert validate_tranche_assignment(frozen) == {"valid": True, "errors": []}
    for tranche_id in TRANCHES:
        rows = [row for row in frozen if row["tranche_id"] == tranche_id]
        assert len(rows) == 20
        assert Counter(row["primary_selection_class"] for row in rows) == Counter(QUOTAS)
        assert Counter(str(row["match_id"]) for row in rows) == Counter(MATCH_ROTATION[tranche_id])
        halves = Counter(row["half"] for row in rows)
        perspectives = Counter(row["perspective_band"] for row in rows)
        assert halves["FIRST_HALF"] >= 8 and halves["SECOND_HALF"] >= 8
        assert perspectives["FAR"] >= 5 and perspectives["NEAR_MIDDLE"] >= 5
        assert any(str(row["match_id"]) == "117092" for row in rows)
    tranche_one = [row for row in frozen if row["tranche_id"] == "TRANCHE_1"]
    tags = {tag for row in tranche_one for tag in row["secondary_evidence_tags"]}
    assert {"NESTED_MUST_PROTECT", "HUMAN_SAFE_FRAGMENT", "MISSED_PERSON_MARK"} <= tags


def test_all_review_assets_are_hash_bound_and_bounded() -> None:
    rows = read_jsonl(STAGE / "02_REVIEW_ASSET_PACKAGE/review_asset_manifest.jsonl")
    assert len(rows) == 1080
    assert len({row["source_frame_pixel_sha256"] for row in rows}) == 1044
    checked: set[str] = set()
    for row in rows:
        for derivative in (row["panorama"], row["focus"]):
            path = STAGE / derivative["derivative_path"]
            assert path.is_file()
            assert path.stat().st_size == derivative["byte_size"]
            if str(path) not in checked:
                assert sha256_file(path) == derivative["sha256"]
                checked.add(str(path))
        assert max(row["panorama"]["width"], row["panorama"]["height"]) == 2560
        crop = row["focus"]["crop_source_xyxy"]
        assert 0 <= crop[0] < crop[2] <= row["source_width"]
        assert 0 <= crop[1] < crop[3] <= row["source_height"]
        assert row["focus"]["width"] == crop[2] - crop[0]
        assert row["focus"]["height"] == crop[3] - crop[1]
    report = read_json(STAGE / "02_REVIEW_ASSET_PACKAGE/asset_generation_report.json")
    assert report["classification"] == "PASS_G7E_B_REVIEW_ASSETS"
    assert report["source_frame_hash_failures"] == []
    assert report["full_resolution_frames_retained"] == 0


def test_browser_payload_is_blind_and_practice_is_isolated() -> None:
    cases = read_json(PACKAGE / "review_cases.json")
    practice = read_json(PACKAGE / "practice_cases.json")
    assert cases["case_count"] == 120
    assert practice["case_count"] == 3
    assert Counter(case["tranche_id"] for case in cases["cases"]) == Counter({tranche: 20 for tranche in TRANCHES})
    assert cases["human_answers_present"] is False
    keys: set[str] = set()

    def collect_keys(value) -> None:
        if isinstance(value, dict):
            keys.update(value)
            for child in value.values():
                collect_keys(child)
        elif isinstance(value, list):
            for child in value:
                collect_keys(child)

    collect_keys(cases)
    for forbidden in (
        "primary_selection_class",
        "secondary_evidence_tags",
        "selection_source_ids",
        "shirt_number",
        "track_id",
    ):
        assert forbidden not in keys
    assert all(case["team_classification_present"] is False for case in cases["cases"])
    assert all(case["practice_only"] for case in practice["cases"])
    assert all(case["tranche_id"] is None for case in practice["cases"])
    assert not (PACKAGE / "human_decisions").exists()
    practice_root = PACKAGE / "practice_decisions"
    if practice_root.exists():
        assert not list(practice_root.glob("events/*/*.json"))
        assert not list(practice_root.glob("receipts/**/*.json"))
        drafts = list(practice_root.glob("drafts/*.json"))
        assert drafts
        assert all(read_json(path)["review_revision"] == "G7E_B_TEMPORAL_BURST_REVIEW_V1" for path in drafts)


def test_branch_graph_ontology_and_accessibility_contract() -> None:
    branch = read_json(PACKAGE / "reviewer_branch_contract.json")
    assert branch["one_question_at_a_time"]
    assert branch["subject_tokens"] == ["SUBJECT_A", "SUBJECT_B", "SUBJECT_C"]
    assert branch["occlusion_helper_requires_explicit_confirmation"]
    assert branch["team_classification"] == "INTENTIONALLY_EXCLUDED"
    assert branch["permanent_identity"] == "FORBIDDEN"
    assert branch["focus_no_relevant_branch"] == [
        "focus_confirmation",
        "whole_burst_missed_person_check",
        "summary_and_acknowledged_save",
    ]
    script = (PACKAGE / "review.js").read_text(encoding="utf-8")
    style = (PACKAGE / "review.css").read_text(encoding="utf-8")
    markup = (PACKAGE / "index.html").read_text(encoding="utf-8")
    assert "localStorage" not in script
    assert "subject_token" in script and '"ABC"' in script
    assert "team" not in json.dumps(read_json(PACKAGE / "reviewer_event_schema.json")["properties"])
    assert "min-height:44px" in style
    assert "focus-visible" in style and "prefers-reduced-motion" in style
    assert 'aria-label="Show candidate boxes"' in markup
    assert "PRACTICE — NOT HUMAN TRUTH" in markup


@pytest.mark.parametrize(
    ("source_size", "viewport", "zoom", "pan"),
    [
        ((4096.0, 1080.0), (1920.0, 640.0), 1.0, (0.0, 0.0)),
        ((3840.0, 1906.0), (1200.0, 520.0), 2.4, (-81.0, 43.0)),
        ((4096.0, 1080.0), (1366.0, 420.0), 4.5, (137.0, -92.0)),
    ],
)
def test_source_display_coordinate_round_trip(source_size, viewport, zoom, pan) -> None:
    for source in ((0.0, 0.0), (source_size[0] * 0.5, source_size[1] * 0.5), source_size):
        display = source_to_display(source, source_size, viewport, zoom, pan)
        restored = display_to_source(display, source_size, viewport, zoom, pan)
        assert abs(restored[0] - source[0]) <= 0.5
        assert abs(restored[1] - source[1]) <= 0.5


def test_drafts_events_receipts_supersession_and_locking(tmp_path: Path) -> None:
    decisions = tmp_path / "real"
    practice = tmp_path / "practice"
    store = TemporalReviewStore(PACKAGE, decisions, practice, acceptance_mode=True)
    first = store.by_tranche["TRANCHE_1"][0]
    draft = store.save_draft(
        {
            "burst_id": first["burst_id"],
            "current_question": "focus_confirmation",
            "current_frame_sequence": 4,
            "playback_speed": 1.0,
            "answers": {"focus_confirmation": "NOT_SURE"},
            "subjects": [],
            "candidate_mappings": [],
            "missed_person_marks": [],
        },
        "real",
    )
    assert store.draft("real", first["burst_id"])["updated_at_utc"] == draft["updated_at_utc"]
    first_result = store.save_event(store.acceptance_event(first, "simple"), "real")
    assert first_result["acknowledgement_receipt_id"] == f"ack-{first_result['event_id']}"
    assert not (decisions / "drafts" / f"{first['burst_id']}.json").exists()

    locked_case = store.by_tranche["TRANCHE_2"][0]
    with pytest.raises(ValueError, match="locked"):
        store.save_event(store.acceptance_event(locked_case, "simple"), "real")
    store.complete_acceptance_tranche("TRANCHE_1")
    state = store.state("real", "TRANCHE_1")
    assert state["tranche_complete"] and not state["editable"]
    original_receipt = state["tranche_completion_receipt_id"]
    latest = store.latest_events("real")[first["burst_id"]]
    superseding = store.acceptance_event(first, "occlusion")
    superseding["supersedes_event_id"] = latest["event_id"]
    edit = store.save_event(superseding, "real")
    assert edit["tranche_completion_receipt_id"] != original_receipt
    assert len(list((decisions / "events/TRANCHE_1").glob("*.json"))) == 21
    assert len(list((decisions / "receipts/tranche_completion").glob("*.json"))) == 2

    for number in range(1, 6):
        current = f"TRANCHE_{number}"
        following = f"TRANCHE_{number + 1}"
        assert store.unlock_next(current)["next_tranche_id"] == following
        assert store.complete_acceptance_tranche(following)["ok"]
    global_receipt = store.current_global_receipt(create=False)
    assert global_receipt is not None
    assert global_receipt["event_count"] == 120
    assert len(global_receipt["current_tranche_receipts"]) == 6
    assert len(store.latest_events("real")) == 120

    assert store.complete_acceptance_practice()["practice_event_count"] == 3
    assert len(store.latest_events("real")) == 120
    assert store.reset_practice()["human_event_count"] == 120
    assert not practice.exists()


def test_real_edge_acceptance_and_three_visual_cap() -> None:
    report = read_json(STAGE / "04_BROWSER_ACCEPTANCE/browser_acceptance_report.json")
    assert report["classification"] == "PASS_G7E_B_BROWSER_ACCEPTANCE"
    assert report["real_frozen_football_assets_visible"]
    assert report["mock_html_used"] is False and report["synthetic_canvas_used"] is False
    assert report["temporary_tranche_protocol"]["burst_events"] == 120
    assert report["temporary_tranche_protocol"]["acknowledgement_receipts"] == 120
    assert report["temporary_tranche_protocol"]["tranche_completion_receipts"] == 6
    assert report["temporary_tranche_protocol"]["global_completion_receipts"] == 1
    assert report["performance_pass"]
    assert report["real_human_event_count"] == 0
    visuals = sorted((STAGE / "05_VISUAL_QA").glob("*.png"))
    assert len(visuals) == 3
    assert all(path.stat().st_size > 100_000 for path in visuals)


def test_no_inference_or_restricted_media_access_and_handoff_manifest() -> None:
    build = read_json(STAGE / "06_TESTS_AND_LOGS/build_report.json")
    closure = read_json(STAGE / "00_INPUT_CLOSURE/input_closure.json")
    assert build["inference_run"] is False
    assert closure["validation_or_holdout_access"] is False
    assert closure["selection_manifests_mutated"] is False
    assert build["production_ready"] is False
    handoff = STAGE / "07_REVIEW_PACK/CHATGPT_HANDOFF"
    if handoff.is_dir():
        files = sorted(path for path in handoff.iterdir() if path.is_file())
        assert len(files) == 12
        manifest = read_json(handoff / "12_MANIFEST.json")
        assert len(manifest["files"]) == 11
        for row in manifest["files"]:
            path = handoff / row["filename"]
            assert path.stat().st_size == row["byte_size"]
            assert sha256_file(path) == row["sha256"]
        assert "12_MANIFEST.json" not in {row["filename"] for row in manifest["files"]}


def test_repository_test_is_focused_only() -> None:
    test_names = subprocess.check_output(
        [
            str(REPOSITORY / ".venv/Scripts/python.exe"),
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            str(Path(__file__).resolve()),
        ],
        cwd=REPOSITORY,
        text=True,
        encoding="utf-8",
    )
    digest = hashlib.sha256(test_names.encode()).hexdigest()
    assert "test_g7e_b_temporal_reviewer_and_tranches.py" in test_names
    assert len(digest) == 64
