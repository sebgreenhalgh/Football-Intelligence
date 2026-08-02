"""Focused G7E-B R4 relationship-integrity and real-draft recovery tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from football_intelligence.temporal_review import R4_REVIEW_REVISION, TemporalReviewStore

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
STAGE = PART7 / "G7E_B_R4_CANDIDATE_RELATIONSHIP_BRANCH_INTEGRITY_AND_REAL_DRAFT_RECOVERY_v1"
R3 = PART7 / "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1"
PACKAGE = STAGE / "02_BRANCH_COMPATIBILITY_ENGINE/temporal_reviewer_r4"
REAL_ROOT = R3 / "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
REAL_DRAFT = REAL_ROOT / "drafts/g7e_a_117093_10.json"
BACKUP = STAGE / "00_INPUT_EVENT_AND_REAL_DRAFT_CLOSURE/forensic_backups/g7e_a_117093_10.original.json"
EXPECTED_BASELINE = "6357aff3d030b38ff879bb25281d9f2823c68925"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_exact_post_r3_baseline_and_forensic_closure() -> None:
    baseline = read_json(STAGE / "00_INPUT_EVENT_AND_REAL_DRAFT_CLOSURE/baseline_resolution.json")
    backup = read_json(STAGE / "00_INPUT_EVENT_AND_REAL_DRAFT_CLOSURE/real_draft_forensic_backup_manifest.json")
    preflight = read_json(STAGE / "00_INPUT_EVENT_AND_REAL_DRAFT_CLOSURE/real_root_preflight.json")
    assert baseline["resolved_post_r3_commit"] == baseline["origin_main"] == EXPECTED_BASELINE
    assert backup["sha256"] == sha256(BACKUP) == "f91ecc55945946efacd65c037dc94a2da4ad7661ab8e9148440eaf3de39d49c5"
    assert backup["byte_size"] == BACKUP.stat().st_size == 69808
    assert backup["draft_version"] == 75
    assert len(preflight["events"]) == len(preflight["acknowledgements"]) == 0
    assert len(preflight["tranche_receipts"]) == len(preflight["global_receipts"]) == 0


def test_exact_failure_and_full_nine_frame_forensic_index() -> None:
    reproduction = read_json(STAGE / "01_FAILURE_REPRODUCTION_AND_FORENSICS/failure_reproduction.json")
    root_cause = read_json(STAGE / "01_FAILURE_REPRODUCTION_AND_FORENSICS/root_cause.json")
    invalid = read_json(STAGE / "01_FAILURE_REPRODUCTION_AND_FORENSICS/invalid_relationship_index.json")
    rows = [
        json.loads(line)
        for line in (STAGE / "01_FAILURE_REPRODUCTION_AND_FORENSICS/candidate_relationship_forensic_table.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert reproduction["reproduced"] is True
    assert reproduction["response"]["error"] == "invalid candidate relationship"
    assert root_cause["classification"] == "GENERIC_SINGLE_BOX_RELATIONSHIP_BUG"
    assert len(rows) == invalid["invalid_count"] == 9
    assert {row["candidate_supply_answer"] for row in rows} == {"ONE_USEFUL_CANDIDATE"}
    assert {row["selected_candidate_count"] for row in rows} == {1}
    assert all(len(row["selected_candidate_ids"]) == len(row["selected_candidate_frame_ids"]) == 1 for row in rows)
    assert all(row["selected_candidate_frame_ids"] == [row["observation_frame_id"]] for row in rows)


def _check(
    store: TemporalReviewStore,
    case: dict,
    row: dict,
    supply: str,
    selected: list[str],
    relationship: str | None,
    family: str | None,
) -> dict:
    observation = json.loads(json.dumps(row))
    observation.update(
        {
            "observation_supply": supply,
            "selected_candidate_ids": selected,
            "candidate_relationship": relationship,
            "relationship_question_id": "subject_0_relationship_0" if family else None,
            "relationship_branch_family": family,
        }
    )
    return store.relationship_compatibility(
        case=case,
        subject_token="SUBJECT_A",
        subject_index=0,
        sequence=0,
        observation=observation,
        final=True,
    )


def test_shared_matrix_cardinality_and_branch_specific_relationships(tmp_path: Path) -> None:
    store = TemporalReviewStore(PACKAGE, tmp_path / "real", tmp_path / "practice")
    assert store.review_revision == R4_REVIEW_REVISION
    case = store.by_id["g7e_a_117093_10"]
    row = read_json(REAL_DRAFT)["subjects"][0]["frame_observations"][0]
    ids = [candidate["candidate_id"] for candidate in case["frame_candidates"][0]][:2]
    canonical = (
        ("ONE_USEFUL_CANDIDATE", ids[:1], "NOT_APPLICABLE", None),
        ("MULTIPLE_CANDIDATES", ids, "SAME_PERSON_DUPLICATES", "MULTIPLE_BOX_RELATIONSHIP"),
        ("MULTIPLE_CANDIDATES", ids, "SAME_PERSON_FRAGMENTS", "MULTIPLE_BOX_RELATIONSHIP"),
        ("MULTIPLE_CANDIDATES", ids, "DIFFERENT_PEOPLE", "MULTIPLE_BOX_RELATIONSHIP"),
        ("MULTIPLE_CANDIDATES", ids, "CORRECT_INNER_BAD_OUTER", "MULTIPLE_BOX_RELATIONSHIP"),
        ("MERGED_WITH_OTHER_PEOPLE", ids[:1], "MERGED_MULTI_PERSON", "SINGLE_MERGED_BOX_CONFIRMATION"),
        ("FRAGMENT_ONLY", ids[:1], "SUBJECT_BODY_FRAGMENT", "FRAGMENT_MEANING"),
        ("NO_CANDIDATE", [], "NOT_APPLICABLE", None),
        ("UNCERTAIN", [], "NOT_APPLICABLE", None),
        ("NOT_APPLICABLE", [], "NOT_APPLICABLE", None),
    )
    assert all(not _check(store, case, row, *values)["errors"] for values in canonical)
    assert _check(store, case, row, "ONE_USEFUL_CANDIDATE", ids, "NOT_APPLICABLE", None)["errors"]
    assert _check(store, case, row, "MULTIPLE_CANDIDATES", ids[:1], None, "MULTIPLE_BOX_RELATIONSHIP")["errors"]
    merged = store.relationship_contract["question_families"]["SINGLE_MERGED_BOX_CONFIRMATION"]
    fragment = store.relationship_contract["question_families"]["FRAGMENT_MEANING"]
    assert "selected boxes related" not in merged["question"].lower()
    assert "one box cover Subject {subject}" in merged["question"]
    assert fragment["question"] == "What does the selected fragment box represent?"


def test_candidate_ids_are_frame_local_and_candidate_artifacts_are_unchanged() -> None:
    old = read_json(R3 / "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/review_cases.json")
    new = read_json(PACKAGE / "review_cases.json")
    assert len(old["cases"]) == len(new["cases"]) == 120
    for before, after in zip(old["cases"], new["cases"], strict=True):
        assert before["frames"] == after["frames"]
        assert before["frame_candidates"] == after["frame_candidates"]
        assert before["per_frame_candidate_states"] == after["per_frame_candidate_states"]
    states = read_json(PACKAGE / "candidate_states_by_reference.json")["frames"].values()
    unique = {row["unique_frame_id"]: row for row in states}
    assert len(unique) == 1044
    assert sum(row["post_gate_candidate_count"] for row in unique.values()) == 48062


def test_real_draft_migration_preserves_human_truth_and_creates_no_event() -> None:
    record = read_json(STAGE / "03_REAL_DRAFT_RECOVERY/real_draft_migration_record.json")
    draft = read_json(REAL_DRAFT)
    assert draft["review_revision"] == R4_REVIEW_REVISION
    assert draft["draft_version"] == record["after_draft_version"] == 76
    assert sha256(REAL_DRAFT) == record["after_sha256"]
    assert record["human_answers_changed"] == record["source_coordinates_changed"] == 0
    assert record["candidate_selections_changed"] == record["missed_person_marks_changed"] == 0
    assert record["real_event_created"] is record["real_acknowledgement_created"] is False
    assert record["burst_2_started"] is False
    assert len(record["changes"]) == 10
    assert all(row["candidate_relationship"] == "NOT_APPLICABLE" for row in draft["subjects"][0]["frame_observations"])
    assert all(row["relationship_question_id"] is None for row in draft["subjects"][0]["frame_observations"])
    assert len(list(REAL_ROOT.glob("events/*/*.json"))) == 0
    assert len(list(REAL_ROOT.glob("receipts/acknowledgements/*.json"))) == 0


def test_invalidation_preflight_targeted_routes_and_atomic_temp_save() -> None:
    invalidation = read_json(STAGE / "02_BRANCH_COMPATIBILITY_ENGINE/branch_invalidation_results.json")
    preflight = read_json(STAGE / "04_FINAL_SAVE_AND_TARGETED_CORRECTION/final_save_preflight_results.json")
    browser = read_json(STAGE / "05_BROWSER_ACCEPTANCE/browser_acceptance_report.json")
    assert invalidation["hidden_stale_answers_remaining"] == invalidation["human_answers_replaced"] == 0
    assert invalidation["branch_invalidation_journal_entries_added"] >= 9
    assert preflight["preflight"]["status"] == "READY_TO_PERSIST"
    assert preflight["event_count"] == preflight["acknowledgement_count"] == 1
    assert preflight["duplicate_save"]["recovered_existing_event"] is True
    failure = browser["exact_stale_relationship_reproduction"]
    assert failure["count"] == 9
    assert {row["correction_route"] for row in failure["errors"]} == {f"subject_0_supply_{index}" for index in range(9)}
    assert browser["upstream_invalidation"]["relation"] == "NOT_APPLICABLE"
    assert browser["refresh_restoration"]["relationship"] == "NOT_APPLICABLE"
    assert browser["temporary_counts"]["events"] == browser["temporary_counts"]["acknowledgements"] == 1
    assert browser["double_save_duplicate_events"] == 0
    assert browser["acknowledged_event_read_only_reload"]["readOnly"] is True
    assert browser["acknowledged_event_read_only_reload"]["continueDisabled"] is True
    assert browser["actual_real_event_and_acknowledgement_counts_increased"] is False
    assert browser["burst_2_started"] is False


def test_real_edge_visual_cap_and_live_content() -> None:
    report = read_json(STAGE / "05_BROWSER_ACCEPTANCE/browser_acceptance_report.json")
    visuals = sorted((STAGE / "06_VISUAL_QA").glob("*.png"))
    assert report["decision"] == "PASS_G7E_B_R4_REAL_EDGE_ACCEPTANCE"
    assert report["browser"] == "Microsoft Edge"
    assert report["preview_watermark"] == "R4 REVIEWER PREVIEW — NO NEW HUMAN TRUTH"
    assert len(visuals) == len(report["visuals"]) == 3
    assert all(row["real_browser_football_and_overlay_gate"] == "PASS" for row in report["visuals"])


def test_scope_launcher_and_zero_inference() -> None:
    launcher = (STAGE / "launch_temporal_burst_review_r4.ps1").read_text(encoding="utf-8")
    instructions = (STAGE / "HUMAN_RESUME_INSTRUCTIONS.md").read_text(encoding="utf-8")
    assert "--port 8818" in launcher and "temporal_reviewer_r4" in launcher
    assert "g7e_a_117093_10" in instructions and "Save burst" in instructions
    implementation = read_json(STAGE / "02_BRANCH_COMPATIBILITY_ENGINE/relationship_engine_implementation.json")
    assert implementation["detector_or_temporal_inference_run"] is False
    assert implementation["project_defaults_changed"] is False
    assert implementation["validation_or_holdout_accessed"] is False


def test_chatgpt_handoff_exact_twelve_file_manifest() -> None:
    handoff = STAGE / "08_REVIEW_PACK/CHATGPT_HANDOFF"
    manifest = read_json(handoff / "12_MANIFEST.json")
    files = sorted(path.name for path in handoff.iterdir() if path.is_file())
    assert len(files) == 12
    assert files == [
        f"{index:02d}_{name}"
        for index, name in enumerate(
            (
                "EXECUTIVE_SUMMARY.json",
                "BASELINE_EVENT_AND_REAL_DRAFT_PREFLIGHT.json",
                "FAILURE_REPRODUCTION_AND_ROOT_CAUSE.json",
                "BRANCH_COMPATIBILITY_AND_INVALIDATION.json",
                "REAL_DRAFT_RECOVERY_RESULTS.json",
                "FINAL_SAVE_AND_TARGETED_CORRECTION.json",
                "BROWSER_AND_REGRESSION_ACCEPTANCE.json",
                "DECISION.md",
                "AFFECTED_FRAME.png",
                "BRANCH_SPECIFIC_QUESTION.png",
                "REAL_DRAFT_READY.png",
                "MANIFEST.json",
            ),
            start=1,
        )
    ]
    assert len(manifest["files"]) == 11
    for row in manifest["files"]:
        path = handoff / row["filename"]
        assert path.stat().st_size == row["byte_size"]
        assert sha256(path) == row["sha256"]
    assert (STAGE / "08_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt").is_file()
    assert read_json(STAGE / "decision.json")["decision"] == (
        "PASS_G7E_B_R4_CANDIDATE_RELATIONSHIP_INTEGRITY_READY_FOR_REAL_DRAFT_RESUME"
    )
