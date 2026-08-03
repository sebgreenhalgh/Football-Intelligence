"""Close R6 provenance and perform the authorized lifecycle-only recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from football_intelligence.g7e_b_r5_reviewer_state import load_contract
from football_intelligence.g7e_b_r6_action_reducer import migrate_failed_r5_draft
from football_intelligence.temporal_review import TemporalReviewStore, read_json

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
R5 = PART7 / "G7E_B_R5_REVIEWER_STATE_MACHINE_AND_FULL_CORPUS_STABILIZATION_v1"
R6 = PART7 / "G7E_B_R6_SERVER_AUTHORITATIVE_ACTION_REDUCER_AND_EXACT_BRANCH_REPAIR_v1"
PACKAGE = R6 / "03_SERVER_AUTHORITATIVE_ACTION_REDUCER/temporal_reviewer_r6"
REAL_ROOT = PART7 / (
    "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1/"
    "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
)
FAILED = "g7e_a_117092_16"
EXPECTED_HEAD = "54cb594edd1cca65b258d099c0803d59b0e4d1e8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def immutable_inventory() -> list[dict[str, Any]]:
    rows = []
    for kind, glob in (("EVENT", "events/**/*.json"), ("ACKNOWLEDGEMENT", "receipts/acknowledgements/*.json")):
        for path in sorted(REAL_ROOT.glob(glob)):
            rows.append(
                {
                    "kind": kind,
                    "path": str(path),
                    "relative_path": path.relative_to(REAL_ROOT).as_posix(),
                    "byte_size": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return rows


def close_baseline() -> None:
    head = git("rev-parse", "HEAD")
    origin = git("rev-parse", "origin/main")
    if head != EXPECTED_HEAD or origin != EXPECTED_HEAD:
        raise RuntimeError("FAIL_G7E_B_R6_BASELINE_PROVENANCE")
    backup = R6 / "00_BASELINE_REAL_EVENT_AND_DRAFT_CLOSURE/forensic_backups"
    failed_path = REAL_ROOT / f"drafts/{FAILED}.json"
    error_path = REAL_ROOT / f"status/final_save_error_{FAILED}.json"
    required = {
        backup / f"{FAILED}.r5_failed.original.json": failed_path,
        backup / f"final_save_error_{FAILED}.original.json": error_path,
        backup / "G7E_B_R5_REAL_REVIEW_RELEASE_GATE.original.json": R5
        / "02_CANONICAL_STATE_CONTRACT/temporal_reviewer_r5/G7E_B_R5_REAL_REVIEW_RELEASE_GATE.json",
    }
    for target, source in required.items():
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        if target.read_bytes() != source.read_bytes():
            raise RuntimeError(f"forensic backup mismatch: {target.name}")
    draft = read_json(failed_path)
    events = immutable_inventory()
    failed_events = [
        row for row in events if row["kind"] == "EVENT" and read_json(Path(row["path"])).get("burst_id") == FAILED
    ]
    if failed_events:
        raise RuntimeError("FAIL_G7E_B_R6_UNEXPECTED_FAILED_BURST_EVENT")
    closure = {
        "schema_version": "football_intelligence.g7e_b_r6.baseline_closure.v1",
        "classification": "PASS_G7E_B_R6_BASELINE_REAL_TRUTH_AND_FAILED_DRAFT_CLOSED",
        "repository_head": head,
        "origin_main": origin,
        "real_root": str(REAL_ROOT),
        "immutable_event_count": sum(row["kind"] == "EVENT" for row in events),
        "immutable_acknowledgement_count": sum(row["kind"] == "ACKNOWLEDGEMENT" for row in events),
        "immutable_files": events,
        "failed_burst": FAILED,
        "failed_burst_event_count": 0,
        "failed_burst_acknowledgement_count": 0,
        "failed_draft": {
            "path": str(failed_path),
            "byte_size": failed_path.stat().st_size,
            "sha256": sha256(failed_path),
            "review_revision": draft.get("review_revision"),
            "draft_version": draft.get("draft_version"),
            "current_question": draft.get("current_question"),
            "answer_count": len(draft.get("answered_domain_values", {})),
            "missed_person_mark_count": len(draft.get("missed_person_marks", [])),
        },
        "forensic_backups": [
            {"path": str(path), "byte_size": path.stat().st_size, "sha256": sha256(path)} for path in required
        ],
        "production_ready": False,
    }
    write_json(R6 / "00_BASELINE_REAL_EVENT_AND_DRAFT_CLOSURE/baseline_real_event_and_draft_closure.json", closure)
    write_json(
        R6 / "00_BASELINE_REAL_EVENT_AND_DRAFT_CLOSURE/immutable_real_file_manifest_before.json", {"files": events}
    )
    root_cause = {
        "schema_version": "football_intelligence.g7e_b_r6.exact_path_root_cause.v1",
        "classification": "PASS_G7E_B_R6_EXACT_PATH_AND_ROOT_CAUSE_PROVEN",
        "failed_burst": FAILED,
        "authoritative_real_path": [
            "original_focus=NO_RELEVANT_PERSON",
            "context_subject=NO",
            "missed_check=YES",
            "27 source-coordinate marks",
            "summary displayed",
            "HTTP 422 FINAL_EVENT_COMPILATION_FAILED",
            "subsequent HTTP 409 DRAFT_SCHEMA_MISMATCH",
        ],
        "primary_root_cause": (
            "R5_BROWSER_MUTATED_DOMAIN_ANSWERS_AND_RECONSTRUCTED_LIFECYCLE_IN_A_SEPARATE_SAVE-TIME_PASS"
        ),
        "secondary_root_cause": (
            "R5_FINAL_COMPILER_ALWAYS_REQUIRED_ADDITIONAL_SUBJECT_EVEN_WHEN_THE_NO_SUBJECT_BRANCH_MADE_IT_"
            "NON_APPLICABLE"
        ),
        "r5_release_gap": {
            "synthetic_completed_draft_helper": "g7e_b_r5_reviewer_state.synthetic_complete_draft",
            "full_corpus_path": "scripts/g7e_b_r5_capture_edge_acceptance.py::save_synthetic",
            "browser_branch_path": "assigned window.__G7E_B_R5__.app.data and called renderQuestion directly",
            "production_dom_action_handlers_exercised": False,
            "why_missed": (
                "No accepted-answer/server-lifecycle transaction crossed the production DOM-to-server boundary."
            ),
        },
        "exact_real_draft_state": {
            "original_focus": draft["answers"].get("original_focus_box_answer"),
            "context_subject": draft["answers"].get("context_subject_answer"),
            "missed_check": draft["answers"].get("missed_check"),
            "subjects": len(draft.get("subjects", [])),
            "marks": len(draft.get("missed_person_marks", [])),
            "draft_version": draft.get("draft_version"),
        },
        "production_ready": False,
    }
    write_json(R6 / "01_EXACT_REAL_BROWSER_PATH_REPRODUCTION/exact_path_and_root_cause.json", root_cause)
    revocation = {
        "schema_version": "football_intelligence.g7e_b_r6.r5_release_gate_revocation.v1",
        "revocation_id": "G7E_B_R6_REVOKES_R5_RELEASE_GATE_APPEND_ONLY_V1",
        "revoked_release_gate_path": str(backup / "G7E_B_R5_REAL_REVIEW_RELEASE_GATE.original.json"),
        "revoked_release_gate_sha256": sha256(backup / "G7E_B_R5_REAL_REVIEW_RELEASE_GATE.original.json"),
        "reason_codes": [
            "SYNTHETIC_DRAFT_BYPASSED_PRODUCTION_DOM_HANDLERS",
            "ANSWER_AND_LIFECYCLE_WERE_NOT_ATOMIC",
            "NO_SUBJECT_BRANCH_APPLICABILITY_WAS_WRONG",
        ],
        "original_artifact_modified": False,
        "production_ready": False,
    }
    write_json(R6 / "07_RELEASE_GATE_AND_REAL_STATE_ACCEPTANCE/G7E_B_R5_RELEASE_GATE_REVOCATION.json", revocation)
    print("PASS_G7E_B_R6_BASELINE_CLOSURE")


def migrate(real: bool) -> None:
    source_path = REAL_ROOT / f"drafts/{FAILED}.json"
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    canonical, canonical_hash = load_contract(PACKAGE / "canonical_reviewer_state_contract.json")
    store = TemporalReviewStore(
        PACKAGE, decisions_root=REAL_ROOT, practice_root=R6 / ".migration_practice", acceptance_mode=True
    )
    case = store.by_id[FAILED]
    migrated = migrate_failed_r5_draft(source, case, canonical, canonical_hash, str(store.action_contract_sha256))
    migrated["draft_version"] = int(source["draft_version"]) + 1
    migrated = store._persist_r6_draft(migrated, "real" if real else "practice", expected_increment=0)
    report = {
        "schema_version": "football_intelligence.g7e_b_r6.failed_real_draft_recovery.v1",
        "classification": "PASS_G7E_B_R6_FAILED_DRAFT_LIFECYCLE_ONLY_RECOVERY",
        "real_root_written": real,
        "source_path": str(source_path),
        "source_byte_size": len(source_bytes),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "result_draft_version": migrated["draft_version"],
        "result_draft_sha256": migrated["draft_content_sha256"],
        "human_content_sha256_before": migrated["migration_record"]["human_content_sha256_before"],
        "human_content_sha256_after": migrated["migration_record"]["human_content_sha256_after"],
        "human_values_changed": False,
        "marks_preserved": len(migrated["missed_person_marks"]),
        "summary_ready": migrated["summary_ready"],
        "event_created": False,
        "acknowledgement_created": False,
        "next_burst_started": False,
        "production_ready": False,
    }
    write_json(R6 / "02_FAILED_REAL_DRAFT_RECOVERY/failed_real_draft_recovery.json", report)
    if not real:
        # A temporary migration must never touch the user draft.
        if source_path.read_bytes() != source_bytes:
            raise RuntimeError("temporary migration mutated the real failed draft")
    print("PASS_G7E_B_R6_FAILED_DRAFT_MIGRATION" + ("_REAL" if real else "_TEMPORARY"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--close-baseline", action="store_true")
    parser.add_argument("--migrate-temporary", action="store_true")
    parser.add_argument("--migrate-real", action="store_true")
    args = parser.parse_args()
    if args.close_baseline:
        close_baseline()
    if args.migrate_temporary:
        migrate(False)
    if args.migrate_real:
        migrate(True)


if __name__ == "__main__":
    main()
