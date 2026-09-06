"""Freeze inputs and finalize the G7F-C calibration adjudication Phase-A release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from football_intelligence.calibration_adjudication import CALIBRATION_IDS, load_original_parent, sha256_file
from g7f_c_run_calibration_adjudication_reviewer import (
    BASELINE,
    EXPECTED_HASHES,
    EXPECTED_INTERPRETER,
    REVIEWER_RELEASE,
    git,
    read_json,
    roots,
    verify_frozen_contracts,
)


PASS = "PASS_G7F_C_CALIBRATION_SUPERSEDING_ADJUDICATION_REVIEWER_READY_FOR_HUMAN_REPAIR"
INCOMPLETE = "HOLD_G7F_C_CALIBRATION_ADJUDICATION_INCOMPLETE"
BASE_SITE_PACKAGES = Path(r"C:\Users\sebgr\anaconda3\Lib\site-packages")
ACCEPTED_R2_PREFLIGHT_WORKTREE = [
    "?? scripts/g7f_c_r2_edge_navigation_acceptance.js",
    "?? scripts/g7f_c_r2_run_acceptance.py",
    "?? tests/test_g7f_c_r2_reviewer_navigation.py",
]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def inventory(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    rows = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    ordered = "\n".join(f"{row['relative_path']}\t{row['byte_size']}\t{row['sha256']}" for row in rows).encode()
    categories = Counter(row["relative_path"].split("/", 1)[0] for row in rows)
    return {
        "file_count": len(rows),
        "total_bytes": sum(row["byte_size"] for row in rows),
        "ordered_inventory_sha256": hashlib.sha256(ordered).hexdigest(),
        "categories": dict(sorted(categories.items())),
        "files": rows,
    }


def selection_frames(source: Path) -> dict[str, dict[str, Any]]:
    selection = read_json(source / "01_SELECTION/dense_gold_selection_manifest.json")
    frames = {
        row["anonymous_dense_image_id"]: row
        for row in selection["images"]
        if row["anonymous_dense_image_id"] in CALIBRATION_IDS
    }
    if set(frames) != set(CALIBRATION_IDS):
        raise RuntimeError("selection does not contain exactly DG-001..DG-006")
    return frames


def freeze(repo: Path) -> dict[str, Any]:
    paths = roots(repo)
    head, origin = git(repo, "rev-parse", "HEAD"), git(repo, "rev-parse", "origin/main")
    if head != BASELINE or origin != BASELINE:
        raise RuntimeError(f"required prechange baseline mismatch: HEAD={head} origin/main={origin}")
    frozen_contracts = verify_frozen_contracts(paths["source"])
    decision_root = paths["source"] / "06_DENSE_DECISIONS"
    if (decision_root / "calibration_adjudication").exists():
        raise RuntimeError("real calibration adjudication namespace existed before Phase A")
    original_inventory = inventory(decision_root)
    frames = selection_frames(paths["source"])
    parents = {}
    for image_id in CALIBRATION_IDS:
        parent = load_original_parent(decision_root, frames[image_id])
        parents[image_id] = {
            "event_id": parent.event_id,
            "event_sha256": parent.event_sha256,
            "event_file_sha256": parent.event_file_sha256,
            "event_byte_size": parent.event_path.stat().st_size,
            "acknowledgement_id": parent.acknowledgement["acknowledgement_id"],
            "acknowledgement_sha256": parent.acknowledgement_sha256,
            "acknowledgement_file_sha256": parent.acknowledgement_file_sha256,
            "acknowledgement_byte_size": parent.acknowledgement_path.stat().st_size,
            "people": len(parent.event["annotation"]["people"]),
            "ignore_regions": len(parent.event["annotation"]["ignore_regions"]),
            "reviewed_strips": parent.event["annotation"]["reviewed_exhaustiveness_strips"],
        }
    event_names = [path.name for path in (decision_root / "events").glob("*.json")]
    ack_names = [path.name for path in (decision_root / "acknowledgements").glob("*.json")]
    if len(event_names) != 6 or len(ack_names) != 6:
        raise RuntimeError("expected exactly six original finalized calibration events and acknowledgements")
    if any(not name.startswith("first_pass__DG-00") for name in (*event_names, *ack_names)):
        raise RuntimeError("scored or blind-repeat finalized event exists")
    payload = {
        "schema_version": "football_intelligence.g7f_c.calibration_adjudication.original_freeze.v1",
        "captured_before_real_adjudication_writes": True,
        "repository": {
            "path": str(repo),
            "head": head,
            "origin_main": origin,
            "required_baseline": BASELINE,
            "initial_preexisting_worktree_entries": ACCEPTED_R2_PREFLIGHT_WORKTREE,
            "initial_preexisting_entries_are_accepted_r2_regression_support": True,
            "worktree_at_freeze_record_persistence": git(repo, "status", "--short").splitlines(),
        },
        "decision_root": str(decision_root),
        **original_inventory,
        "calibration_parents": parents,
        "six_first_pass_calibration_events": True,
        "six_exact_acknowledgements": True,
        "scored_finalized_events": 0,
        "blind_repeat_events": 0,
        "real_adjudication_namespace_existed": False,
        "frozen_contracts": frozen_contracts,
        "production_ready": False,
    }
    write_json(paths["stage"] / "01_FREEZE/original_real_decisions_inventory.json", payload)
    audit_files = {}
    for relative in (
        "03_MANUAL_VISUAL_ASSESSMENT.json",
        "07_QA_ASSETS/qa_asset_manifest.json",
        "10_REVIEW_PACK/CHATGPT_HANDOFF/04_CALIBRATION_VISUAL_QA.md",
    ):
        path = paths["audit"] / relative
        audit_files[relative] = {"byte_size": path.stat().st_size, "sha256": sha256_file(path)}
    write_json(
        paths["stage"] / "01_FREEZE/audit_evidence_bindings.json",
        {"files": audit_files, "candidate_data_used": False, "production_ready": False},
    )
    return payload


def run_focused_tests(repo: Path, stage: Path) -> dict[str, Any]:
    args = [
        "tests/test_g7f_c_calibration_adjudication.py",
        "tests/test_g7f_c_dense_person_gold.py",
        "tests/test_g7f_c_r1_reviewer_interaction.py",
        "tests/test_g7f_c_r2_reviewer_navigation.py",
        "-q",
        "-p",
        "no:cacheprovider",
        f"--basetemp={stage / '07_ACCEPTANCE/pytest_final'}",
    ]
    code = (
        "import json,sys; "
        f"sys.path.append({str(BASE_SITE_PACKAGES)!r}); "
        "import pytest; "
        f"raise SystemExit(pytest.main(json.loads({json.dumps(json.dumps(args))})))"
    )
    result = subprocess.run(
        [str(EXPECTED_INTERPRETER), "-c", code],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo / "src")},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    log = result.stdout + result.stderr
    (stage / "07_ACCEPTANCE/focused_tests.log").write_text(log, encoding="utf-8", newline="\n")
    if result.returncode:
        raise RuntimeError(f"focused tests failed:\n{log}")
    return {"exit_code": 0, "summary": result.stdout.strip().splitlines()[-1], "interpreter": str(EXPECTED_INTERPRETER)}


def run_launcher_check(repo: Path, stage: Path) -> dict[str, Any]:
    launcher = stage / "05_REVIEWER/launch_calibration_adjudication_reviewer.ps1"
    result = subprocess.run(
        [
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(launcher),
            "-CheckOnly",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    (stage / "07_ACCEPTANCE/launcher_check.log").write_text(
        result.stdout + result.stderr, encoding="utf-8", newline="\n"
    )
    if result.returncode:
        raise RuntimeError(f"launcher check failed:\n{result.stdout}\n{result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def run_reaudit_block(repo: Path, stage: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            str(EXPECTED_INTERPRETER),
            str(repo / "scripts/g7f_c_run_calibration_adjudication_reaudit.py"),
            "status",
            "--repo",
            str(repo),
        ],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": f"{repo / 'src'};{repo / 'scripts'}"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    (stage / "07_ACCEPTANCE/reaudit_blocking_state.log").write_text(
        result.stdout + result.stderr, encoding="utf-8", newline="\n"
    )
    value = json.loads(result.stdout)
    if result.returncode != 2 or value.get("decision") != INCOMPLETE or value.get("valid_event_ack_pairs") != 0:
        raise RuntimeError(f"re-audit did not block before human adjudication: {result.stdout} {result.stderr}")
    return value


def release_files(repo: Path, stage: Path) -> list[Path]:
    return [
        repo / "src/football_intelligence/calibration_adjudication.py",
        repo / "src/football_intelligence/calibration_adjudication_reviewer.py",
        repo / "src/football_intelligence/g7f_c_calibration_adjudication_event.schema.json",
        repo / "src/football_intelligence/g7f_c_calibration_adjudication_acknowledgement.schema.json",
        repo / "src/football_intelligence/dense_person_reviewer.py",
        repo / "src/football_intelligence/dense_person_reviewer_static/index.html",
        repo / "src/football_intelligence/dense_person_reviewer_static/app.js",
        repo / "src/football_intelligence/dense_person_reviewer_static/styles.css",
        repo / "scripts/g7f_c_run_calibration_adjudication_reviewer.py",
        repo / "scripts/g7f_c_run_calibration_adjudication_reaudit.py",
        stage / "05_REVIEWER/launch_calibration_adjudication_reviewer.ps1",
        stage / "05_REVIEWER/run_calibration_adjudication_reaudit.ps1",
        stage / "05_REVIEWER/REQUIREMENTS.md",
        stage / "05_REVIEWER/PHASE_B_VISUAL_ASSESSMENT_TEMPLATE.json",
    ]


def build_release(repo: Path, stage: Path, commit: str) -> tuple[dict[str, Any], dict[str, str]]:
    audit = roots(repo)["audit"]
    files = release_files(repo, stage)
    release = {
        "schema_version": "football_intelligence.g7f_c.calibration_adjudication_reviewer_release.v1",
        "reviewer_release": REVIEWER_RELEASE,
        "required_baseline": BASELINE,
        "repository_commit": commit,
        "workflow_authorization": "CALIBRATION_ADJUDICATION_DG_001_THROUGH_DG_006_ONLY",
        "files": [{"path": str(path), "byte_size": path.stat().st_size, "sha256": sha256_file(path)} for path in files],
        "scored_annotation_authorized": False,
        "production_ready": False,
    }
    release_path = stage / "05_REVIEWER/reviewer_release_manifest.json"
    write_json(release_path, release)
    bindings = {
        "selection_manifest_sha256": EXPECTED_HASHES["01_SELECTION/dense_gold_selection_manifest.json"],
        "ontology_sha256": EXPECTED_HASHES["02_ONTOLOGY/dense_person_gold_ontology.json"],
        "metric_protocol_sha256": EXPECTED_HASHES["03_METRICS/dense_gold_metric_protocol.json"],
        "first_pass_event_schema_sha256": EXPECTED_HASHES["04_SCHEMAS/dense_person_frame_annotation.schema.json"],
        "first_pass_ack_schema_sha256": EXPECTED_HASHES["04_SCHEMAS/dense_person_frame_acknowledgement.schema.json"],
        "adjudication_event_schema_sha256": sha256_file(
            repo / "src/football_intelligence/g7f_c_calibration_adjudication_event.schema.json"
        ),
        "adjudication_ack_schema_sha256": sha256_file(
            repo / "src/football_intelligence/g7f_c_calibration_adjudication_acknowledgement.schema.json"
        ),
        "audit_manual_visual_assessment_sha256": sha256_file(audit / "03_MANUAL_VISUAL_ASSESSMENT.json"),
        "audit_qa_asset_manifest_sha256": sha256_file(audit / "07_QA_ASSETS/qa_asset_manifest.json"),
        "audit_visual_qa_handoff_sha256": sha256_file(
            audit / "10_REVIEW_PACK/CHATGPT_HANDOFF/04_CALIBRATION_VISUAL_QA.md"
        ),
        "reviewer_release_manifest_sha256": sha256_file(release_path),
        "repository_commit": commit,
    }
    write_json(stage / "05_REVIEWER/reviewer_binding_hashes.json", bindings)
    return release, bindings


def build_handoff(
    repo: Path,
    stage: Path,
    *,
    commit: str,
    inventory_before: dict[str, Any],
    release: dict[str, Any],
    bindings: dict[str, str],
    tests: dict[str, Any],
    launcher: dict[str, Any],
    reaudit: dict[str, Any],
) -> Path:
    handoff = stage / "10_REVIEW_PACK/CHATGPT_HANDOFF"
    handoff.mkdir(parents=True, exist_ok=True)
    after = inventory(roots(repo)["source"] / "06_DENSE_DECISIONS")
    before_summary = {key: inventory_before[key] for key in ("file_count", "total_bytes", "ordered_inventory_sha256")}
    after_summary = {key: after[key] for key in ("file_count", "total_bytes", "ordered_inventory_sha256")}
    if before_summary != after_summary:
        raise RuntimeError("real decisions changed during Phase-A engineering")
    write_json(
        handoff / "00_EXECUTIVE_SUMMARY.json",
        {
            "stage": "G7F_C_CALIBRATION_SUPERSEDING_ADJUDICATION_AND_REAUDIT",
            "phase": "A",
            "decision": PASS,
            "reviewer_release": REVIEWER_RELEASE,
            "repository_commit": commit,
            "human_repair_started": False,
            "real_adjudication_events_created": 0,
            "fresh_proficiency_images": 0,
            "scored_annotation_authorized": False,
            "scored_annotation_started": False,
            "production_ready": False,
        },
    )
    write_json(
        handoff / "01_ORIGINAL_CALIBRATION_IMMUTABILITY.json",
        {
            "before": before_summary,
            "after": after_summary,
            "byte_identical": True,
            "calibration_parents": inventory_before["calibration_parents"],
            "drafts_action_receipts_and_reveal_logs_preserved": True,
            "production_ready": False,
        },
    )
    write_json(
        handoff / "02_ADJUDICATION_EVENT_AND_PRECEDENCE_CONTRACT.json",
        {
            "event_schema": "football_intelligence.g7f_c.calibration_superseding_annotation.v1",
            "ack_schema": "football_intelligence.g7f_c.calibration_superseding_acknowledgement.v1",
            "pass_kind": "CALIBRATION_ADJUDICATION",
            "namespace": str(roots(repo)["source"] / "06_DENSE_DECISIONS/calibration_adjudication"),
            "correction_mode": "APPEND_ONLY_COMPLETE_CORRECTED_STATE",
            "precedence": {
                "without_valid_acknowledged_adjudication": "FIRST_PASS",
                "with_one_valid_acknowledged_adjudication": "LATEST_VALID_ADJUDICATION",
                "scored_images": "NO_ADJUDICATION_PRECEDENCE",
            },
            "bindings": bindings,
            "production_ready": False,
        },
    )
    assessment = read_json(roots(repo)["audit"] / "03_MANUAL_VISUAL_ASSESSMENT.json")
    write_json(
        handoff / "03_AUDIT_REPAIR_CHECKLIST.json",
        {
            "source_sha256": bindings["audit_manual_visual_assessment_sha256"],
            "images": assessment["images"],
            "do_not_redraw_good_masks_unnecessarily": True,
            "fresh_proficiency_images": 0,
            "production_ready": False,
        },
    )
    (handoff / "04_ADJUDICATION_REVIEWER_AND_SCOPE_GUIDANCE.md").write_text(
        "# Adjudication reviewer and scope guidance\n\n"
        "The reviewer is candidate-free and shows only DG-001 through DG-006. Each image starts from its exact "
        "FIRST_PASS people, components, relevance, and ignore geometry, while adjudication strip progress starts "
        "at 0/8.\n\n"
        "**ALL visible people count: pitch + touchlines + benches/technical areas + foreground/background + "
        "frame edges.**\n\n"
        "Annotate the person first; classify relevance second. Retain good "
        "existing masks, address the exact audit checklist, and re-review all eight strips. `production_ready=false`\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        handoff / "05_TEST_AND_EDGE_ACCEPTANCE.json",
        {
            "focused_tests": tests,
            "adjudication_live_edge": read_json(stage / "07_ACCEPTANCE/live_edge_acceptance.json"),
            "r1_r2_live_edge_regressions": read_json(stage / "07_ACCEPTANCE/r1_r2_live_edge_regressions.json"),
            "launcher_check": launcher,
            "temporary_decisions_only": True,
            "real_decisions_byte_identical": True,
            "production_ready": False,
        },
    )
    (handoff / "06_LAUNCH_AND_HUMAN_REPAIR_INSTRUCTIONS.md").write_text(
        "# Launch and human repair\n\n"
        "Run `05_REVIEWER\\launch_calibration_adjudication_reviewer.ps1` manually. The launcher uses only "
        "`C:\\Users\\sebgr\\anaconda3\\envs\\fi-reviewer\\python.exe`; `-CheckOnly` validates without serving.\n\n"
        "Repair DG-001 through DG-006 using the on-screen checklist. DG-005 person-030 begins unchanged; relabel it "
        "MATCH_RELEVANT only after confirming the assistant referee. Re-review 8/8 strips and check both assertions. "
        "Do not launch scored annotation. `production_ready=false`\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        handoff / "07_REAUDIT_COMMAND_AND_BLOCKING_STATE.json",
        {
            "command": str(stage / "05_REVIEWER/run_calibration_adjudication_reaudit.ps1"),
            "current_state": reaudit,
            "qa_when_complete": {"full_overlays": 6, "strip_views": 48, "person_crop_montages": 6},
            "candidate_data_allowed": False,
            "phase_b_not_run": True,
            "production_ready": False,
        },
    )
    (handoff / "08_DECISION.md").write_text(
        f"# Decision\n\n`{PASS}`\n\n"
        "The six-frame adjudication reviewer is released for manual repair. No real adjudication event was created, "
        "the reviewer was not started, Phase B remains blocked, and scored annotation is unauthorized.\n\n"
        "`production_ready=false`\n",
        encoding="utf-8",
        newline="\n",
    )
    payload_files = sorted(path for path in handoff.iterdir() if path.is_file() and path.name != "10_MANIFEST.json")
    if len(payload_files) != 9:
        raise RuntimeError(f"Phase-A handoff must contain exactly nine payload files: {payload_files}")
    write_json(
        handoff / "10_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7f_c.calibration_adjudication_phase_a_handoff.v1",
            "files": [
                {"name": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
                for path in payload_files
            ],
            "file_count_excluding_manifest": len(payload_files),
            "decision": PASS,
            "production_ready": False,
        },
    )
    return handoff


def release(repo: Path) -> dict[str, Any]:
    paths = roots(repo)
    freeze_path = paths["stage"] / "01_FREEZE/original_real_decisions_inventory.json"
    if not freeze_path.is_file():
        raise RuntimeError("original decision freeze is missing")
    frozen = read_json(freeze_path)
    commit = git(repo, "rev-parse", "HEAD")
    origin = git(repo, "rev-parse", "origin/main")
    if commit != origin or git(repo, "status", "--porcelain"):
        raise RuntimeError("engineering PASS requires a clean repository with HEAD == origin/main")
    ancestry = subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE, commit], cwd=repo, check=False)
    if ancestry.returncode:
        raise RuntimeError("release commit does not descend from the required baseline")
    release_manifest, bindings = build_release(repo, paths["stage"], commit)
    tests = run_focused_tests(repo, paths["stage"])
    launcher = run_launcher_check(repo, paths["stage"])
    reaudit = run_reaudit_block(repo, paths["stage"])
    handoff = build_handoff(
        repo,
        paths["stage"],
        commit=commit,
        inventory_before=frozen,
        release=release_manifest,
        bindings=bindings,
        tests=tests,
        launcher=launcher,
        reaudit=reaudit,
    )
    return {
        "decision": PASS,
        "reviewer_release": REVIEWER_RELEASE,
        "repository_commit": commit,
        "handoff": str(handoff),
        "real_adjudication_events_created": 0,
        "reviewer_started": False,
        "phase_b_state": reaudit["decision"],
        "scored_annotation_authorized": False,
        "scored_annotation_started": False,
        "production_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "release"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = freeze(args.repo.resolve()) if args.phase == "freeze" else release(args.repo.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
