"""Check, serve, accept, or release the DG-005 sequence-2 relevance reviewer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from football_intelligence.calibration_adjudication import CALIBRATION_IDS, sha256_file
from football_intelligence.dg005_sequence2_reviewer import (
    ADJUDICATION_SEQUENCE,
    REVIEWER_RELEASE,
    TARGET_IMAGE_ID,
    TARGET_PERSON_ID,
    DG005Sequence2HTTPServer,
    DG005Sequence2ReviewerConfig,
    run_server,
)
from g7f_c_run_calibration_adjudication_reviewer_r2 import (
    read_json,
    verify_frozen_contracts,
    verify_original_inventory,
)


BASELINE = "770d5b35466d6e4fb68d45abd907adef2c7dff73"
EXPECTED_INTERPRETER = Path(r"C:\Users\sebgr\anaconda3\envs\fi-reviewer\python.exe")
EXPECTED_R2_MANIFEST_SHA256 = "871d558913aafd59f844162e98b3946ee66580a6a6a59ec39510a8f39b1faa40"
PHASE_A_PASS = "PASS_G7F_C_DG005_SEQUENCE2_RELEVANCE_ADJUDICATION_REVIEWER_READY_FOR_HUMAN_CORRECTION"
SEQUENCE2_HOLD = "HOLD_G7F_C_DG005_SEQUENCE2_RELEVANCE_ADJUDICATION_INCOMPLETE"
EXPECTED_RELEASE_PATHS = frozenset(
    {
        "scripts/g7f_c_run_calibration_adjudication_reaudit.py",
        "scripts/g7f_c_run_dg005_sequence2_reviewer.py",
        "scripts/g7f_c_dg005_sequence2_edge_acceptance.js",
        "src/football_intelligence/dg005_sequence2_reviewer.py",
        "src/football_intelligence/dg005_sequence2_reviewer_static/app.js",
        "src/football_intelligence/dg005_sequence2_reviewer_static/index.html",
        "src/football_intelligence/dg005_sequence2_reviewer_static/styles.css",
        "tests/test_g7f_c_dg005_sequence2.py",
        "tests/test_g7f_c_phase_b_r2_compatibility.py",
    }
)


def roots(repo: Path) -> dict[str, Path]:
    part9 = repo.parent / "experiments/football_observation_reasoner/part 9"
    return {
        "repo": repo,
        "source": part9 / "G7F_C_DENSE_PERSON_GOLD_DISCRIMINATION_SET_v1",
        "audit": part9 / "G7F_C_CALIBRATION_6_HUMAN_ANNOTATION_QUALITY_GATE_v1",
        "phase_a": part9 / "G7F_C_CALIBRATION_SUPERSEDING_ADJUDICATION_AND_REAUDIT_v1",
        "r2_stage": part9 / "G7F_C_ADJUDICATION_R1_NAVIGATION_STATE_DESYNC_REPAIR_v1",
        "compatibility_stage": part9 / "G7F_C_PHASE_B_R2_RELEASE_COMPATIBILITY_AND_CLOSURE_REPAIR_v1",
        "stage": part9 / "G7F_C_DG005_SEQUENCE2_RELEVANCE_ADJUDICATION_AND_PHASE_B_CLOSURE_v1",
    }


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def decisions_inventory(root: Path) -> dict[str, Any]:
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
    return {
        "file_count": len(rows),
        "total_bytes": sum(row["byte_size"] for row in rows),
        "ordered_inventory_sha256": hashlib.sha256(ordered).hexdigest(),
        "files": rows,
    }


def verify_environment() -> dict[str, str]:
    if Path(sys.executable).resolve() != EXPECTED_INTERPRETER.resolve():
        raise RuntimeError(f"use the fi-reviewer interpreter: {EXPECTED_INTERPRETER}; found {sys.executable}")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"DG-005 sequence 2 requires Python 3.12; found {sys.version.split()[0]}")
    try:
        import cv2
        import numpy
    except ImportError as exc:
        raise RuntimeError(f"missing reviewer dependency: {exc.name}") from exc
    return {"python": sys.version.split()[0], "numpy": numpy.__version__, "opencv_python": cv2.__version__}


def load_frames(source: Path) -> dict[str, dict[str, Any]]:
    selection = read_json(source / "01_SELECTION/dense_gold_selection_manifest.json")
    frames = {
        row["anonymous_dense_image_id"]: row
        for row in selection["images"]
        if row["anonymous_dense_image_id"] in CALIBRATION_IDS
    }
    if set(frames) != set(CALIBRATION_IDS):
        raise RuntimeError("frozen selection must contain exactly DG-001..DG-006")
    return frames


def stable_truth_bindings(paths: dict[str, Path]) -> dict[str, str]:
    manifest_path = paths["r2_stage"] / "05_REVIEWER/reviewer_release_manifest.json"
    if sha256_file(manifest_path) != EXPECTED_R2_MANIFEST_SHA256:
        raise RuntimeError("accepted R2 reviewer release manifest changed")
    manifest = read_json(manifest_path)
    for row in manifest.get("files", []):
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size != row["byte_size"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"accepted R2 reviewer release file changed: {path}")
    binding = read_json(paths["r2_stage"] / "05_REVIEWER/reviewer_binding_hashes.json")
    truth = binding.get("adjudication_truth_bindings")
    if not isinstance(truth, dict) or binding.get("reviewer_release_manifest_sha256") != EXPECTED_R2_MANIFEST_SHA256:
        raise RuntimeError("accepted R2 stable truth binding is invalid")
    if truth != read_json(paths["phase_a"] / "05_REVIEWER/reviewer_binding_hashes.json"):
        raise RuntimeError("R2 stable truth bindings differ from Phase A")
    return {str(key): str(value) for key, value in truth.items()}


def sequence1_binding(paths: dict[str, Path]) -> dict[str, Any]:
    root = paths["source"] / "06_DENSE_DECISIONS/calibration_adjudication"
    event_path = root / "events/calibration_adjudication_001__DG-005.json"
    ack_path = root / "acknowledgements/calibration_adjudication_001__DG-005.json"
    event, ack = read_json(event_path), read_json(ack_path)
    person = [row for row in event["annotation"]["people"] if row["instance_id"] == TARGET_PERSON_ID]
    if (
        event.get("adjudication_sequence") != 1
        or event.get("anonymous_dense_image_id") != TARGET_IMAGE_ID
        or len(person) != 1
        or person[0].get("relevance") != "NON_MATCH_RELEVANT"
        or ack.get("event_id") != event.get("event_id")
        or ack.get("event_sha256") != event.get("event_sha256")
    ):
        raise RuntimeError("DG-005 sequence-1 relevance parent is invalid")
    return {
        "anonymous_dense_image_id": TARGET_IMAGE_ID,
        "adjudication_sequence": 1,
        "event_id": event["event_id"],
        "event_sha256": event["event_sha256"],
        "event_file_sha256": sha256_file(event_path),
        "event_byte_size": event_path.stat().st_size,
        "acknowledgement_id": ack["acknowledgement_id"],
        "acknowledgement_sha256": ack["acknowledgement_sha256"],
        "acknowledgement_file_sha256": sha256_file(ack_path),
        "acknowledgement_byte_size": ack_path.stat().st_size,
        "reviewer_release": event["reviewer_release"],
        "people": len(event["annotation"]["people"]),
        "ignore_regions": len(event["annotation"]["ignore_regions"]),
        "person_030_relevance": person[0]["relevance"],
        "person_030_binary_mask_sha256": person[0]["binary_mask_sha256"],
        "person_030_rle_sha256": person[0]["rle_sha256"],
        "person_030_derived_visible_box_xyxy": person[0]["derived_visible_box_xyxy"],
    }


def verify_release(paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = paths["stage"] / "05_REVIEWER/reviewer_release_manifest.json"
    binding_path = paths["stage"] / "05_REVIEWER/reviewer_binding_hashes.json"
    manifest = read_json(manifest_path)
    binding = read_json(binding_path)
    commit = str(manifest.get("repository_commit", ""))
    head, origin = git(paths["repo"], "rev-parse", "HEAD"), git(paths["repo"], "rev-parse", "origin/main")
    if not commit or head != commit or origin != commit or git(paths["repo"], "status", "--porcelain"):
        raise RuntimeError(f"sequence-2 reviewer requires clean HEAD == origin/main == {commit}")
    if manifest.get("required_baseline") != BASELINE or manifest.get("reviewer_release") != REVIEWER_RELEASE:
        raise RuntimeError("sequence-2 reviewer release identity or baseline mismatch")
    for row in manifest.get("files", []):
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size != row["byte_size"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"sequence-2 reviewer release file mismatch: {path}")
    if binding.get("reviewer_release_manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("sequence-2 reviewer binding does not match its release manifest")
    if binding.get("dg005_sequence1_parent") != sequence1_binding(paths):
        raise RuntimeError("sequence-2 reviewer parent binding differs from immutable DG-005 sequence 1")
    if binding.get("adjudication_truth_bindings") != stable_truth_bindings(paths):
        raise RuntimeError("sequence-2 reviewer stable truth bindings changed")
    return manifest, binding


def reviewer_config(
    repo: Path,
    *,
    port: int = 8793,
    decisions_root: Path | None = None,
    assets_root: Path | None = None,
    require_release: bool = True,
) -> DG005Sequence2ReviewerConfig:
    paths = roots(repo)
    verify_frozen_contracts(paths["source"])
    bindings = stable_truth_bindings(paths)
    release_binding = (
        verify_release(paths)[1]
        if require_release
        else {
            "dg005_sequence1_parent": sequence1_binding(paths),
            "audit_visual_assessment_sha256": sha256_file(
                paths["phase_a"] / "11_PHASE_B/04_CANDIDATE_FREE_VISUAL_ASSESSMENT.json"
            ),
        }
    )
    frozen = verify_original_inventory(
        {
            "phase_a": paths["phase_a"],
            "source": paths["source"],
        }
    )
    parent = release_binding["dg005_sequence1_parent"]
    decision_root = decisions_root or paths["source"] / "06_DENSE_DECISIONS"
    return DG005Sequence2ReviewerConfig(
        selection_manifest_path=paths["source"] / "01_SELECTION/dense_gold_selection_manifest.json",
        assets_root=assets_root or paths["source"] / "05_REVIEWER/assets",
        original_decisions_root=decision_root,
        adjudication_root=decision_root / "calibration_adjudication",
        binding_hashes=bindings,
        audit_checklist_sha256=release_binding["audit_visual_assessment_sha256"],
        frozen_parent_file_hashes=frozen,
        expected_sequence1_event_id=parent["event_id"],
        expected_sequence1_event_sha256=parent["event_sha256"],
        expected_sequence1_event_file_sha256=parent["event_file_sha256"],
        expected_sequence1_ack_file_sha256=parent["acknowledgement_file_sha256"],
        port=port,
    )


def check(config: DG005Sequence2ReviewerConfig) -> dict[str, Any]:
    server = DG005Sequence2HTTPServer(config)
    try:
        bootstrap = server.bootstrap()
        target_state = server.store.state(TARGET_IMAGE_ID)
    finally:
        server.server_close()
    target = [row for row in target_state["document"]["people"] if row["instance_id"] == TARGET_PERSON_ID]
    if [row["anonymous_dense_image_id"] for row in bootstrap["queue"]] != [TARGET_IMAGE_ID] or len(target) != 1:
        raise RuntimeError("sequence-2 queue or target is not exact")
    if not target_state["finalized"] and target_state["revision"] == 0:
        if target[0]["relevance"] != "NON_MATCH_RELEVANT":
            raise RuntimeError("sequence-2 initial target was changed automatically")
        if target_state["document"]["reviewed_exhaustiveness_strips"]:
            raise RuntimeError("sequence-2 strip confirmation did not reset to 0/8")
    return {
        "decision": PHASE_A_PASS,
        "reviewer_release": REVIEWER_RELEASE,
        "queue": [TARGET_IMAGE_ID],
        "adjudication_sequence": ADJUDICATION_SEQUENCE,
        "target": TARGET_PERSON_ID,
        "target_relevance": target[0]["relevance"],
        "reviewed_strips": target_state["document"]["reviewed_exhaustiveness_strips"],
        "finalized": target_state["finalized"],
        "candidate_data_used": False,
        "drawing_edit_delete_controls": False,
        "scored_annotation_authorized": False,
        "scored_annotation_started": False,
        "production_ready": False,
    }


def copy_truth_to_temp(source_root: Path, temp_root: Path) -> None:
    for folder in ("events", "acknowledgements"):
        for image_id in CALIBRATION_IDS:
            source = source_root / folder / f"first_pass__{image_id}.json"
            target = temp_root / folder / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            source = (
                source_root / "calibration_adjudication" / folder / f"calibration_adjudication_001__{image_id}.json"
            )
            target = temp_root / "calibration_adjudication" / folder / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def run_acceptance(repo: Path, output: Path) -> dict[str, Any]:
    paths = roots(repo)
    output.mkdir(parents=True, exist_ok=True)
    real_root = paths["source"] / "06_DENSE_DECISIONS"
    before = decisions_inventory(real_root)
    with tempfile.TemporaryDirectory(prefix="g7f_c_dg005_seq2_", dir=output) as temporary:
        temp_root = Path(temporary) / "decisions"
        copy_truth_to_temp(real_root, temp_root)
        config = reviewer_config(repo, port=0, decisions_root=temp_root, require_release=False)
        server = DG005Sequence2HTTPServer(config)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        screenshot = output / "live_edge_sequence2_acceptance.png"
        try:
            command = [
                "node",
                str(repo / "scripts/g7f_c_dg005_sequence2_edge_acceptance.js"),
                f"http://127.0.0.1:{server.server_address[1]}/",
                str(screenshot),
            ]
            try:
                result = subprocess.run(
                    command,
                    cwd=repo,
                    capture_output=True,
                    text=True,
                    timeout=240,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "live Edge sequence-2 acceptance timed out; " f"stdout={exc.stdout!r}; stderr={exc.stderr!r}"
                ) from exc
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        if result.returncode:
            raise RuntimeError(f"live Edge sequence-2 acceptance failed:\n{result.stdout}\n{result.stderr}")
        browser = json.loads(result.stdout)
        events = list((temp_root / "calibration_adjudication/events").glob("calibration_adjudication_002__*.json"))
        acknowledgements = list(
            (temp_root / "calibration_adjudication/acknowledgements").glob("calibration_adjudication_002__*.json")
        )
        if len(events) != 1 or len(acknowledgements) != 1 or "DG-005" not in events[0].name:
            raise RuntimeError("acceptance must create exactly one temporary DG-005 sequence-2 pair")
    after = decisions_inventory(real_root)
    if before != after:
        raise RuntimeError("sequence-2 acceptance changed real human decisions")
    payload = {
        "browser": browser,
        "temporary_dg005_sequence2_event_ack_pairs": 1,
        "real_decisions_before": {
            key: before[key] for key in ("file_count", "total_bytes", "ordered_inventory_sha256")
        },
        "real_decisions_after": {key: after[key] for key in ("file_count", "total_bytes", "ordered_inventory_sha256")},
        "real_decisions_byte_identical": True,
        "candidate_data_used": False,
        "scored_annotation_authorized": False,
        "production_ready": False,
    }
    write_json(output / "live_edge_sequence2_acceptance.json", payload)
    return payload


def verify_release_commit(repo: Path) -> str:
    head, origin = git(repo, "rev-parse", "HEAD"), git(repo, "rev-parse", "origin/main")
    if head != origin or git(repo, "status", "--porcelain"):
        raise RuntimeError("release requires a clean repository with HEAD == origin/main")
    if git(repo, "merge-base", BASELINE, head) != BASELINE:
        raise RuntimeError("required baseline is not an ancestor of the release commit")
    changed = frozenset(filter(None, git(repo, "diff", "--name-only", BASELINE, head).splitlines()))
    commits = int(git(repo, "rev-list", "--count", f"{BASELINE}..{head}"))
    if changed != EXPECTED_RELEASE_PATHS or commits != 1:
        raise RuntimeError(f"release is not the exact one-commit sequence-2 change: {commits}, {sorted(changed)}")
    return head


def write_launchers(repo: Path, stage: Path) -> None:
    reviewer = stage / "05_REVIEWER/launch_dg005_sequence2_relevance_adjudication_reviewer.ps1"
    reviewer.parent.mkdir(parents=True, exist_ok=True)
    reviewer.write_text(
        "param([switch]$CheckOnly, [int]$Port = 8793)\n"
        "$ErrorActionPreference = 'Stop'\n"
        "$Python = 'C:\\Users\\sebgr\\anaconda3\\envs\\fi-reviewer\\python.exe'\n"
        f"$Repo = '{repo}'\n"
        'if (-not (Test-Path -LiteralPath $Python)) { throw "Required fi-reviewer Python is missing: $Python" }\n'
        '$env:PYTHONPATH = "$Repo\\src;$Repo\\scripts"\n'
        "$Phase = if ($CheckOnly) { 'check' } else { 'serve' }\n"
        '& $Python "$Repo\\scripts\\g7f_c_run_dg005_sequence2_reviewer.py" $Phase --repo $Repo --port $Port\n'
        'if ($LASTEXITCODE -ne 0) { throw "DG-005 sequence-2 reviewer failed with exit code $LASTEXITCODE" }\n',
        encoding="utf-8",
        newline="\n",
    )
    closure = stage / "05_REVIEWER/run_phase_b_closure.ps1"
    closure.write_text(
        "param([ValidateSet('status','prepare','finalize')][string]$Phase = 'status', [string]$VisualAssessment)\n"
        "$ErrorActionPreference = 'Stop'\n"
        "$Python = 'C:\\Users\\sebgr\\anaconda3\\envs\\fi-reviewer\\python.exe'\n"
        f"$Repo = '{repo}'\n"
        'if (-not (Test-Path -LiteralPath $Python)) { throw "Required fi-reviewer Python is missing: $Python" }\n'
        '$env:PYTHONPATH = "$Repo\\src;$Repo\\scripts"\n'
        "$Args = @($Phase, '--repo', $Repo)\n"
        "if ($Phase -eq 'finalize') {\n"
        "  if (-not $VisualAssessment) { throw 'finalize requires -VisualAssessment' }\n"
        "  $Args += @('--visual-assessment', $VisualAssessment)\n"
        "}\n"
        '& $Python "$Repo\\scripts\\g7f_c_run_calibration_adjudication_reaudit.py" @Args\n'
        "exit $LASTEXITCODE\n",
        encoding="utf-8",
        newline="\n",
    )
    (stage / "05_REVIEWER/REQUIREMENTS.md").write_text(
        "# Requirements\n\n"
        "Use only `C:\\Users\\sebgr\\anaconda3\\envs\\fi-reviewer\\python.exe`. The launcher fails closed "
        "if Python 3.12, NumPy, OpenCV, frozen contracts, the exact Git release, or the immutable DG-005 "
        "sequence-1 parent cannot be verified.\n\n"
        "The real queue contains DG-005 only. It is candidate-free, relevance-only, and does not start "
        "scored annotation. `production_ready=false`\n",
        encoding="utf-8",
        newline="\n",
    )


def release_files(repo: Path, stage: Path) -> list[Path]:
    return [repo / path for path in sorted(EXPECTED_RELEASE_PATHS) if not path.startswith("tests/")] + [
        stage / "05_REVIEWER/launch_dg005_sequence2_relevance_adjudication_reviewer.ps1",
        stage / "05_REVIEWER/run_phase_b_closure.ps1",
        stage / "05_REVIEWER/REQUIREMENTS.md",
    ]


def run_closure_status(repo: Path) -> dict[str, Any]:
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
        timeout=420,
        check=False,
    )
    value = json.loads(result.stdout)
    if result.returncode != 2 or value.get("decision") != SEQUENCE2_HOLD:
        raise RuntimeError(f"Phase-B closure did not block on missing DG-005 sequence 2: {result.stdout}")
    return value


def build_handoff(
    paths: dict[str, Path],
    *,
    commit: str,
    before: dict[str, Any],
    after: dict[str, Any],
    parent: dict[str, Any],
    acceptance: dict[str, Any],
    closure: dict[str, Any],
    release_manifest: dict[str, Any],
) -> Path:
    handoff = paths["stage"] / "10_REVIEW_PACK/CHATGPT_HANDOFF"
    handoff.mkdir(parents=True, exist_ok=True)
    summary = lambda value: {  # noqa: E731
        key: value[key] for key in ("file_count", "total_bytes", "ordered_inventory_sha256")
    }
    write_json(
        handoff / "00_EXECUTIVE_SUMMARY.json",
        {
            "decision": PHASE_A_PASS,
            "reviewer_release": REVIEWER_RELEASE,
            "repository_commit": commit,
            "queue": [TARGET_IMAGE_ID],
            "human_correction_started": False,
            "real_sequence2_events_created": 0,
            "phase_b_decision": closure["decision"],
            "scored_annotation_authorized": False,
            "scored_annotation_started": False,
            "production_ready": False,
        },
    )
    write_json(
        handoff / "01_REAL_DECISIONS_BEFORE_AFTER_INTEGRITY.json",
        {
            "before": summary(before),
            "after": summary(after),
            "byte_identical": before == after,
            "production_ready": False,
        },
    )
    write_json(handoff / "02_DG005_SEQUENCE1_PARENT_BINDING.json", {**parent, "immutable": True})
    write_json(
        handoff / "03_SEQUENCE2_DELTA_CONTRACT.json",
        {
            "only_permitted_annotation_truth_delta": {
                "field": "person-030.relevance",
                "from": "NON_MATCH_RELEVANT",
                "to": "MATCH_RELEVANT",
            },
            "geometry_changes_permitted": False,
            "other_metadata_truth_changes_permitted": False,
            "sequence2_strip_state_initial": [],
            "sequence2_strip_state_required_for_finalize": list(range(8)),
            "server_enforced": True,
            "production_ready": False,
        },
    )
    (handoff / "04_TARGETED_REVIEWER_AND_SERVER_GUARDS.md").write_text(
        "# Targeted reviewer and server guards\n\n"
        "The candidate-free queue contains DG-005 only and exposes no draw, geometry-edit, add, or delete controls. "
        "The server compares every person ID, component, ignore region, and non-target relevance value against the "
        "immutable sequence-1 event before writing. Crafted unauthorized requests fail with HTTP 422. Finalization "
        "also canonicalizes the document and requires an exact one-field truth delta.\n\n"
        f"Release manifest files: {len(release_manifest['files'])}. `production_ready=false`\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        handoff / "05_TEST_AND_EDGE_ACCEPTANCE.json",
        {
            "live_edge": acceptance,
            "focused_test_log": str(paths["stage"] / "07_ACCEPTANCE/focused_tests.log"),
            "temporary_decisions_only": True,
            "sequence1_bytes_unchanged": True,
            "real_decisions_byte_identical": True,
            "production_ready": False,
        },
    )
    (handoff / "06_HUMAN_CORRECTION_INSTRUCTIONS.md").write_text(
        "# Human correction\n\n"
        "1. Run `05_REVIEWER\\launch_dg005_sequence2_relevance_adjudication_reviewer.ps1`.\n"
        "2. Confirm that DG-005 person-030 is the visible flag-carrying assistant referee.\n"
        "3. Set only person-030 to `MATCH_RELEVANT`.\n"
        "4. Re-review and confirm all eight strips.\n"
        "5. Check the official confirmation and both exact completion assertions.\n"
        "6. Finalize the immutable sequence-2 event, then stop the reviewer.\n"
        "7. Run `05_REVIEWER\\run_phase_b_closure.ps1 -Phase status`; do not start scored annotation.\n\n"
        "`production_ready=false`\n",
        encoding="utf-8",
        newline="\n",
    )
    (handoff / "07_DECISION.md").write_text(
        f"# Decision\n\n`{PHASE_A_PASS}`\n\n"
        f"Phase B remains `{SEQUENCE2_HOLD}` until explicit human correction.\n\n"
        "SCORED_ANNOTATION_AUTHORIZED=false  \nSCORED_ANNOTATION_STARTED=false  \nproduction_ready=false\n",
        encoding="utf-8",
        newline="\n",
    )
    payload = sorted(path for path in handoff.iterdir() if path.is_file() and path.name != "09_MANIFEST.json")
    if len(payload) != 8:
        raise RuntimeError(f"sequence-2 handoff must contain exactly eight payload files: {payload}")
    write_json(
        handoff / "09_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7f_c.dg005_sequence2_phase_a_handoff.v1",
            "files": [
                {"name": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)} for path in payload
            ],
            "file_count_excluding_manifest": 8,
            "decision": PHASE_A_PASS,
            "production_ready": False,
        },
    )
    return handoff


def finalize_release(repo: Path) -> dict[str, Any]:
    paths = roots(repo)
    commit = verify_release_commit(repo)
    verify_frozen_contracts(paths["source"])
    truth = stable_truth_bindings(paths)
    parent = sequence1_binding(paths)
    before = read_json(paths["stage"] / "01_FREEZE/real_decisions_before_engineering.json")
    current = decisions_inventory(paths["source"] / "06_DENSE_DECISIONS")
    if before != current:
        raise RuntimeError("real decisions changed during sequence-2 engineering")
    acceptance = read_json(paths["stage"] / "07_ACCEPTANCE/live_edge_sequence2_acceptance.json")
    if acceptance.get("real_decisions_byte_identical") is not True:
        raise RuntimeError("live Edge acceptance did not preserve real decisions")
    write_launchers(repo, paths["stage"])
    assessment = paths["phase_a"] / "11_PHASE_B/04_CANDIDATE_FREE_VISUAL_ASSESSMENT.json"
    release_manifest = {
        "schema_version": "football_intelligence.g7f_c.dg005_sequence2_reviewer_release.v1",
        "reviewer_release": REVIEWER_RELEASE,
        "required_baseline": BASELINE,
        "repository_commit": commit,
        "workflow_authorization": "DG_005_SEQUENCE_2_PERSON_030_RELEVANCE_ONLY",
        "files": [
            {"path": str(path), "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in release_files(repo, paths["stage"])
        ],
        "scored_annotation_authorized": False,
        "production_ready": False,
    }
    manifest_path = paths["stage"] / "05_REVIEWER/reviewer_release_manifest.json"
    write_json(manifest_path, release_manifest)
    binding = {
        "adjudication_truth_bindings": truth,
        "dg005_sequence1_parent": parent,
        "audit_visual_assessment_sha256": sha256_file(assessment),
        "reviewer_release_manifest_sha256": sha256_file(manifest_path),
        "repository_commit": commit,
        "production_ready": False,
    }
    write_json(paths["stage"] / "05_REVIEWER/reviewer_binding_hashes.json", binding)
    closure = run_closure_status(repo)
    launcher = subprocess.run(
        [
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(paths["stage"] / "05_REVIEWER/launch_dg005_sequence2_relevance_adjudication_reviewer.ps1"),
            "-CheckOnly",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    (paths["stage"] / "07_ACCEPTANCE/launcher_check.log").write_text(
        launcher.stdout + launcher.stderr, encoding="utf-8", newline="\n"
    )
    if launcher.returncode:
        raise RuntimeError(f"sequence-2 launcher check failed: {launcher.stdout} {launcher.stderr}")
    after = decisions_inventory(paths["source"] / "06_DENSE_DECISIONS")
    if after != before:
        raise RuntimeError("release checks changed real human decisions")
    handoff = build_handoff(
        paths,
        commit=commit,
        before=before,
        after=after,
        parent=parent,
        acceptance=acceptance,
        closure=closure,
        release_manifest=release_manifest,
    )
    write_json(paths["stage"] / "01_FREEZE/real_decisions_after_engineering.json", after)
    return {
        "decision": PHASE_A_PASS,
        "reviewer_release": REVIEWER_RELEASE,
        "repository_commit": commit,
        "release_manifest_sha256": sha256_file(manifest_path),
        "closure_decision": closure["decision"],
        "handoff": str(handoff),
        "real_decisions_byte_identical": True,
        "human_correction_started": False,
        "scored_annotation_authorized": False,
        "scored_annotation_started": False,
        "production_ready": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("check", "serve", "acceptance", "release"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--port", type=int, default=8793)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    environment = verify_environment()
    repo = args.repo.resolve()
    if args.phase == "acceptance":
        if args.output is None:
            raise RuntimeError("acceptance requires --output")
        result = run_acceptance(repo, args.output.resolve())
    elif args.phase == "release":
        result = finalize_release(repo)
    else:
        config = reviewer_config(repo, port=args.port)
        if args.phase == "check":
            result = check(config)
        else:
            print(f"DG-005 sequence-2 relevance adjudication reviewer: http://{config.host}:{config.port}/")
            print("Candidate-free DG-005 only. Scored annotation is not authorized. production_ready=false")
            run_server(config)
            return 0
    print(json.dumps({**result, "environment": environment}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
