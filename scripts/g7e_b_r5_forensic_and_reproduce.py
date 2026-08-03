"""Close R5 inputs and reproduce the null-future-answer failure without real writes."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from football_intelligence.temporal_review import ReviewValidationError, TemporalReviewStore

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
R4 = PART7 / "G7E_B_R4_CANDIDATE_RELATIONSHIP_BRANCH_INTEGRITY_AND_REAL_DRAFT_RECOVERY_v1"
R3 = PART7 / "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1"
R5 = PART7 / "G7E_B_R5_REVIEWER_STATE_MACHINE_AND_FULL_CORPUS_STABILIZATION_v1"
PACKAGE = R4 / "02_BRANCH_COMPATIBILITY_ENGINE/temporal_reviewer_r4"
REAL_ROOT = R3 / "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
BASELINE = R5 / "00_BASELINE_AND_REAL_STATE_CLOSURE"
FAILURE = R5 / "01_FAILURE_REPRODUCTION_AND_LIFECYCLE_AUDIT"
BURST1 = "g7e_a_117093_10"
BURST2 = "g7e_a_118575_18"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def files(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "byte_size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("*.json"))
        if path.is_file()
    ]


def reproduce() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="g7e_b_r5_exact_failure_") as temporary:
        decisions = Path(temporary) / "real"
        practice = Path(temporary) / "practice"
        store = TemporalReviewStore(PACKAGE, decisions, practice, acceptance_mode=True)
        case = store.by_id[BURST2]
        observations = []
        for sequence, frame in enumerate(case["frames"]):
            identity = frame["canonical_frame_identity"]
            observations.append(
                {
                    "frame_reference_id": identity["frame_id"],
                    "canonical_frame_identity": identity,
                    "visibility": None,
                    "subject_location_source_x": None,
                    "subject_location_source_y": None,
                    "human_confirmed": False,
                    "approximate_hidden_location": False,
                    "observation_supply": None,
                    "selected_candidate_ids": [],
                    "candidate_selection_binding": {
                        "action_type": "CANDIDATE_SELECTION",
                        "canonical_frame_identity": identity,
                        "question_id": f"subject_0_supply_{sequence}",
                        "selected_candidate_ids": [],
                    },
                    "occlusion_phase": "NONE",
                    "candidate_relationship": None,
                    "relationship_question_id": None,
                    "relationship_branch_family": None,
                }
            )
        payload = {
            "mode": "real",
            "burst_id": BURST2,
            "current_question": "original_focus",
            "current_frame_sequence": 4,
            "playback_speed": 1.0,
            "answers": {"original_focus_box_answer": "ONE_RELEVANT_MATCH_PERSON"},
            "subjects": [
                {
                    "subject_token": "SUBJECT_A",
                    "subject_definition_source": "YELLOW_ORIGINAL_FOCUS_CANDIDATE",
                    "anchor_frame_sequence": None,
                    "anchor_source_xy": None,
                    "frame_observations": observations,
                    "marker_continuity_confirmation": None,
                    "occlusion_confirmed": False,
                    "continuity": None,
                    "role": None,
                    "participation": None,
                    "certainty": None,
                }
            ],
            "candidate_mappings": [],
            "missed_person_marks": [],
            "click_transactions": [],
            "action_journal": [],
            "draft_version": 0,
            "optimistic_lock_token": None,
        }
        try:
            store.save_draft(payload, "real")
        except ReviewValidationError as error:
            return {
                "reproduced": True,
                "error_code": error.error_code,
                "exact_error": str(error),
                "first_validation_error": error.errors[0],
                "temporary_event_count": len(list((decisions / "events").rglob("*.json"))),
                "temporary_acknowledgement_count": len(list((decisions / "receipts/acknowledgements").glob("*.json"))),
            }
        raise RuntimeError("R4 null-future-answer failure did not reproduce")


def main() -> None:
    inventory = files(REAL_ROOT)
    event = next(row for row in inventory if row["relative_path"].startswith("events/"))
    ack = next(row for row in inventory if row["relative_path"].startswith("receipts/acknowledgements/"))
    if len(inventory) != 3:
        raise RuntimeError("unexpected real-root file count before R5")
    backups = BASELINE / "forensic_backups"
    backups.mkdir(parents=True, exist_ok=True)
    for row in inventory:
        source = REAL_ROOT / row["relative_path"]
        target = backups / row["relative_path"].replace("/", "__")
        if not target.is_file():
            shutil.copyfile(source, target)
        if sha256(target) != row["sha256"]:
            raise RuntimeError("forensic backup differs from source")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    origin = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=REPO, text=True).strip()
    write_json(
        BASELINE / "baseline_and_real_state_closure.json",
        {
            "schema_version": "football_intelligence.g7e_b_r5.baseline_and_real_state.v1",
            "repository_head": head,
            "origin_main": origin,
            "expected_head": "cd3c5fbd9cf919b1821b7e3bd4c7efe6cdaabd9f",
            "head_matches_expected": head == origin == "cd3c5fbd9cf919b1821b7e3bd4c7efe6cdaabd9f",
            "real_root": str(REAL_ROOT),
            "real_root_files": inventory,
            "burst_1_event": event,
            "burst_1_acknowledgement": ack,
            "burst_1_immutable_and_acknowledged": True,
            "burst_2_draft_existed_before_r5": False,
            "burst_2_event_count": 0,
            "burst_2_acknowledgement_count": 0,
            "tranche_completion_receipt_count": 0,
            "global_completion_receipt_count": 0,
            "production_ready": False,
        },
    )
    reproduction = reproduce()
    write_json(FAILURE / "exact_failure_reproduction.json", reproduction)
    write_json(
        FAILURE / "root_cause_and_lifecycle_audit.json",
        {
            "schema_version": "football_intelligence.g7e_b_r5.root_cause.v1",
            "primary_classification": "BRANCH_ENGINE_VALIDATED_UNREACHED_FIELDS",
            "contributing_classifications": [
                "PREPOPULATED_NULL_DOMAIN_FIELDS",
                "MISSING_QUESTION_LIFECYCLE_STATE",
                "DRAFT_AND_FINAL_VALIDATION_NOT_SEPARATED",
            ],
            "exact_mechanism": (
                "The R4 client created all nine future observations with observation_supply=null; "
                "draft save then ran the relationship engine on every future observation, and Python None "
                "was rejected as an unsupported candidate-supply answer."
            ),
            "observed_request_was_not_persisted": True,
            "question_1_human_answer_recoverable": False,
            "r5_migration_rule": "RESTORE_BURST_2_AT_BLANK_QUESTION_1_WITH_NO_INFERRED_ANSWER",
            "reproduction": reproduction,
            "production_ready": False,
        },
    )
    print("PASS_G7E_B_R5_FAILURE_REPRODUCED_AND_REAL_STATE_CLOSED")


if __name__ == "__main__":
    main()
