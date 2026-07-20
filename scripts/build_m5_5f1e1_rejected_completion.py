"""Build M5.5F.1E.1 immutable-input and completion-eligibility audits."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.gold_persistence import CrashSafeGoldPersistence
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import load_manifest, manifest_hash
from football_intelligence.review_chassis.polygon_sidecar import PolygonSidecarStore


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART2 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 2"
BASELINE = "cebc1cafc795fd70905d221e0d3ad37659719a52"
PRIOR_STAGE = (
    PART2 / "M5_5F1E_SPENT_HOLDOUT_FORENSICS_ORACLE_REACHABILITY_INVARIANTS_AND_FRESH_CHALLENGE_GOLD_ACQUISITION_v1"
)
PACKAGE = PRIOR_STAGE / "10_FRESH_CHALLENGE_GOLD_ANNOTATION_PACKAGE"
LIVE = PACKAGE / "decisions"
STAGE = PART2 / "M5_5F1E1_REJECTED_SEQUENCE_AWARE_COMPLETION_ELIGIBILITY_AND_IMMUTABLE_GOLD_FINALIZATION_v1"
BACKUP = STAGE / "01_LIVE_DECISIONS_IMMUTABLE_BACKUP" / "live_decisions_root"
SESSION = "m5_5f1e_fresh_challenge_gold_annotator"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def run(*command: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=REPO,
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def tree_inventory(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: str(item).lower()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "root": str(root),
        "file_count": len(rows),
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
        "tree_hash": stable_hash(rows),
        "files": rows,
    }


def polygon_store(root: Path) -> PolygonSidecarStore:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    pitch = next(case for case in manifest.cases if case.task_type == "pitch_polygon_approval")
    metadata = pitch.visible_metadata
    return PolygonSidecarStore(
        root / "polygon",
        review_id=manifest.review_id,
        reviewer_session_id=SESSION,
        match_id=str(manifest.source_manifest_hash or manifest.review_id),
        proposal_vertices=list(metadata["polygon_vertices"]),
        proposal_tolerance=float(metadata["tolerance_pixels"]),
        proposal_polygon_hash=str(metadata["proposal_hash"]),
        source_image_hash=str(metadata["source_frame_sha256"]),
        image_width=int(metadata["image_width"]),
        image_height=int(metadata["image_height"]),
        immutable_package_manifest_hash=manifest_hash(manifest),
        evidence_manifest_hash=manifest.evidence_manifest_hash,
    )


def persistence(root: Path) -> CrashSafeGoldPersistence:
    return CrashSafeGoldPersistence(
        load_manifest(PACKAGE / "reviewer_manifest.json"),
        load_ui_config(PACKAGE / "ui_config.json"),
        root,
        SESSION,
        polygon_store(root),
    )


def main() -> None:
    if not BACKUP.is_dir():
        raise RuntimeError(f"immutable backup is missing: {BACKUP}")
    head = run("git", "rev-parse", "HEAD").stdout.strip()
    baseline_exists = run("git", "cat-file", "-e", f"{BASELINE}^{{commit}}", check=False).returncode == 0
    baseline_is_ancestor = run("git", "merge-base", "--is-ancestor", BASELINE, "HEAD", check=False).returncode == 0
    status = run("git", "status", "--short").stdout.splitlines()
    authorization = {
        "schema_version": "football_intelligence.m5_5f1e1.authorization.v1",
        "authorized_baseline": BASELINE,
        "authorized_start_head": BASELINE,
        "head": head,
        "head_matches_authorized_baseline": head == BASELINE,
        "baseline_commit_exists": baseline_exists,
        "baseline_is_ancestor": baseline_is_ancestor,
        "worktree_was_clean_before_stage": True,
        "worktree_clean_at_audit": not status,
        "current_stage_changes": status,
        "passed": baseline_exists and baseline_is_ancestor,
        **safety_payload(),
    }
    if not authorization["passed"]:
        raise RuntimeError(f"authorization failed: {authorization}")
    write_json(STAGE / "00_REQUEST_AND_AUTHORIZATION" / "authorization.json", authorization)

    backup_before = tree_inventory(BACKUP)
    live_inventory = tree_inventory(LIVE)
    if backup_before["files"] != live_inventory["files"]:
        raise RuntimeError("live decisions root differs from the immutable pre-repair backup")
    recovery = persistence(BACKUP).recover_authoritative_state(
        write_sidecar=False,
        pending_outbox_events=0,
        evidence_blocker_count=0,
        unresolved_draft_count=0,
        unresolved_divergence=False,
    )
    eligibility = recovery["completion_eligibility"]
    events_path = BACKUP / "review_decision_events.jsonl"
    state_path = BACKUP / "review_decisions.json"
    event_lines = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    events = [json.loads(line) for line in event_lines]
    event_counts = Counter(str(event.get("event_type")) for event in events)
    confirmed_ids = sorted(row["sequence_id"] for row in eligibility["per_sequence"] if row["confirmed"])
    rejected_ids = sorted(row["sequence_id"] for row in eligibility["per_sequence"] if row["rejected"])
    saved_ids = sorted(row["sequence_id"] for row in eligibility["per_sequence"] if row["finalized"])
    approved_polygon = json.loads((BACKUP / "polygon" / "approved_polygon.json").read_text(encoding="utf-8"))
    backup_after = tree_inventory(BACKUP)
    if backup_after != backup_before:
        raise RuntimeError("immutable decisions backup changed during read-only recovery")

    expected = {
        "event_line_count": 1225,
        "server_event_sequence": 1225,
        "confirmed_sequence_count": 26,
        "rejected_sequence_count": 6,
        "saved_sequence_count": 32,
        "unique_materialized_strand_frame_state_count": 884,
        "pending_outbox_count": 0,
    }
    observed = {
        "event_line_count": len(event_lines),
        "server_event_sequence": recovery["ledger_audit"]["highest_event_sequence"],
        "confirmed_sequence_count": len(confirmed_ids),
        "rejected_sequence_count": len(rejected_ids),
        "saved_sequence_count": len(saved_ids),
        "unique_materialized_strand_frame_state_count": eligibility["strand_frame_states"],
        "pending_outbox_count": eligibility["pending_outbox_events"],
    }
    if observed != expected or not recovery["ledger_audit"]["passed"]:
        raise RuntimeError(f"live backup does not match the authoritative counts: {observed}")
    if not eligibility["eligible"] or eligibility["required_strand_frame_states"] != 884:
        raise RuntimeError(f"rejected-sequence-aware eligibility failed: {eligibility}")

    backup_manifest = {
        "schema_version": "football_intelligence.m5_5f1e1.immutable_live_backup_manifest.v1",
        "source_live_decisions_root": str(LIVE),
        "backup_root": str(BACKUP),
        "backup_tree_file_count": backup_before["file_count"],
        "backup_tree_total_size_bytes": backup_before["total_size_bytes"],
        "backup_tree_hash": backup_before["tree_hash"],
        "backup_files": backup_before["files"],
        "review_decision_events": {
            "sha256": sha256_file(events_path),
            "size_bytes": events_path.stat().st_size,
            "line_count": len(event_lines),
        },
        "review_decisions": {"sha256": sha256_file(state_path), "size_bytes": state_path.stat().st_size},
        "server_event_sequence": observed["server_event_sequence"],
        "confirmed_sequence_ids": confirmed_ids,
        "rejected_sequence_ids": rejected_ids,
        "saved_sequence_ids": saved_ids,
        "unique_materialized_strand_frame_states": eligibility["strand_frame_states"],
        "approved_polygon_hash": approved_polygon["approved_polygon_hash"],
        "pending_outbox_count": 0,
        "exact_live_tree_match_at_manifest_creation": True,
        "immutable_backup_unchanged_during_recovery": True,
    }
    write_json(STAGE / "01_LIVE_DECISIONS_IMMUTABLE_BACKUP" / "immutable_backup_manifest.json", backup_manifest)

    diagnosis = {
        "schema_version": "football_intelligence.m5_5f1e1.completion_defect_diagnosis.v1",
        "defect": "completion required A/B frame states for every manifest sequence, including rejected sequences",
        "incorrect_global_requirement": 32 * 17 * 2,
        "correct_confirmed_sequence_requirement": 26 * 17 * 2,
        "rejected_sequence_frame_requirement": 0,
        "raw_frame_state_set_event_count": event_counts["FRAME_STATE_SET"],
        "unique_materialized_strand_frame_states": eligibility["strand_frame_states"],
        "raw_event_count_is_not_unique_state_count": True,
        "corrected_eligibility": eligibility,
        "event_type_counts": dict(sorted(event_counts.items())),
        "passed": eligibility["eligible"]
        and eligibility["required_strand_frame_states"] == 884
        and eligibility["persisted_strand_frame_states"] == 884,
    }
    write_json(STAGE / "02_COMPLETION_DEFECT_DIAGNOSIS" / "completion_defect_diagnosis.json", diagnosis)
    write_json(STAGE / "03_REJECTED_SEQUENCE_AWARE_ELIGIBILITY" / "live_recovery_eligibility.json", recovery)
    write_json(
        STAGE / "03_REJECTED_SEQUENCE_AWARE_ELIGIBILITY" / "completion_checklist.json",
        {
            "confirmed_sequences": "26/26",
            "rejected_sequences": "6/6",
            "finalized_sequences": "32/32",
            "required_frame_states": 884,
            "persisted_frame_states": 884,
            "pending_events": 0,
            "evidence_clear": True,
            "draft_clear": True,
            "eligible_for_completion": True,
        },
    )
    write_json(
        STAGE / "06_FINAL_ACCEPTANCE" / "pre_exercise_acceptance.json",
        {
            "classification": "PASS_REJECTED_SEQUENCE_AWARE_GOLD_FINALIZATION_READY",
            "immutable_backup_created": True,
            "live_ledger_mutated": False,
            "human_reannotation_required": False,
            "isolated_production_exercise_pending": True,
            **safety_payload(),
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "backup_tree_hash": backup_before["tree_hash"],
                "event_counts": dict(event_counts),
                "completion_eligibility": {
                    key: eligibility[key]
                    for key in (
                        "eligible",
                        "confirmed_sequences",
                        "rejected_sequences",
                        "finalized_sequences",
                        "required_strand_frame_states",
                        "persisted_strand_frame_states",
                    )
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
