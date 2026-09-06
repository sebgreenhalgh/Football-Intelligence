"""Run G7F-C R1 interaction, browser, launcher, and data-safety acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pydantic

from football_intelligence.dense_person_gold import COMPLETION_ASSERTION, DensePersonConflictError
from football_intelligence.dense_person_reviewer import (
    DensePersonHTTPServer,
    DensePersonReviewerConfig,
    scan_blind_payload,
)
from football_intelligence.gold_eval.core import inventory_tree
from g7f_c_build_dense_person_reviewer import roots as g7f_c_roots
from g7f_c_build_dense_person_reviewer import verify_upstream


PASS = "PASS_G7F_C_R1_REVIEWER_INTERACTION_AND_UI_REPAIR_READY_FOR_CALIBRATION_RESUME"
RELEASE = "G7F_C_DENSE_PERSON_REVIEWER_R1"
BASELINE_COMMIT = "664a8cc65999b3e17c02654e20b143226137279f"
FI_PYTHON = Path(r"C:\Users\sebgr\anaconda3\envs\fi-reviewer\python.exe")
BASE_SITE_PACKAGES = Path(r"C:\Users\sebgr\anaconda3\Lib\site-packages")
FROZEN = {
    "selection_manifest_sha256": (
        "01_SELECTION/dense_gold_selection_manifest.json",
        "f454e0e93ec2cb01f5edeba1b6545a9e607a26ec8ec58528bccc7f5e12618eef",
    ),
    "ontology_sha256": (
        "02_ONTOLOGY/dense_person_gold_ontology.json",
        "410e7a704ed750fc6e08b064d08578075c751f556d2439f7e15a350430beac5b",
    ),
    "metric_protocol_sha256": (
        "03_METRICS/dense_gold_metric_protocol.json",
        "c28a9d467719dae1944abe2594080e45300b40d46d79e68d636ee1601c8262cd",
    ),
    "event_schema_sha256": (
        "04_SCHEMAS/dense_person_frame_annotation.schema.json",
        "0449998705cffec7d718f7c00cb2249f8b6a35f933a7768aa76db12ea20e8d0c",
    ),
    "ack_schema_sha256": (
        "04_SCHEMAS/dense_person_frame_acknowledgement.schema.json",
        "cc902281a2f5fa14f97a3ad186c8bd565ed748855c6fd445cf35f906cbf94d75",
    ),
}


def paths(repo: Path) -> dict[str, Path]:
    part9 = repo.parent / "experiments/football_observation_reasoner/part 9"
    return {
        "repo": repo,
        "source": part9 / "G7F_C_DENSE_PERSON_GOLD_DISCRIMINATION_SET_v1",
        "repair": part9 / "G7F_C_R1_REVIEWER_INTERACTION_AND_UI_REPAIR_v1",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def verify_frozen(source: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, (relative, expected) in FROZEN.items():
        target = source / relative
        actual = sha256_file(target)
        if actual != expected:
            raise RuntimeError(f"frozen contract changed: {relative}={actual}")
        result[name] = {
            "path": str(target),
            "sha256": actual,
            "expected_sha256": expected,
            "byte_identical": True,
        }
    selection = read_json(source / FROZEN["selection_manifest_sha256"][0])
    counts = Counter(row["selection_status"] for row in selection["images"])
    if counts != Counter({"SCORED_DENSE_GOLD": 48, "CALIBRATION_ONLY": 6}):
        raise RuntimeError(f"frozen selection counts changed: {counts}")
    result["selection_counts"] = {"scored": 48, "calibration": 6, "total": 54}
    result["production_ready"] = False
    return result


def normalize_inventory_files(inventory: dict[str, Any]) -> dict[str, tuple[int, str]]:
    normalized = {}
    for row in inventory.get("files", []):
        relative = row.get("relative_path", row.get("path"))
        size = row.get("byte_size", row.get("bytes"))
        normalized[str(relative)] = (int(size), str(row["sha256"]))
    return normalized


def decision_categories(inventory: dict[str, Any]) -> dict[str, dict[str, int]]:
    categories = {
        "drafts": {"file_count": 0, "total_bytes": 0},
        "events": {"file_count": 0, "total_bytes": 0},
        "acknowledgements": {"file_count": 0, "total_bytes": 0},
        "reveal_logs": {"file_count": 0, "total_bytes": 0},
        "action_receipts": {"file_count": 0, "total_bytes": 0},
    }
    for relative, (size, _) in normalize_inventory_files(inventory).items():
        category = relative.split("/", 1)[0]
        if category in categories:
            categories[category]["file_count"] += 1
            categories[category]["total_bytes"] += size
    return categories


def real_decision_integrity(all_paths: dict[str, Path]) -> dict[str, Any]:
    prechange = read_json(all_paths["repair"] / "01_REAL_DECISIONS_BEFORE_AFTER_INTEGRITY.json")
    before = prechange["before"]
    real_root = all_paths["source"] / "06_DENSE_DECISIONS"
    after = inventory_tree(real_root, include_files=True)
    before_files = normalize_inventory_files(before)
    after_files = normalize_inventory_files(after)
    byte_identical = before_files == after_files
    if not byte_identical:
        added = sorted(after_files.keys() - before_files.keys())
        removed = sorted(before_files.keys() - after_files.keys())
        changed = sorted(
            key for key in before_files.keys() & after_files.keys() if before_files[key] != after_files[key]
        )
        raise RuntimeError(f"real decision integrity failure: added={added}, removed={removed}, changed={changed}")

    draft_path = real_root / "drafts/first_pass__DG-001.json"
    draft_before = draft_path.read_bytes()
    config = reviewer_config(all_paths, decisions_root=real_root)
    server = DensePersonHTTPServer(config)
    try:
        loaded = server.store.state("DG-001", "FIRST_PASS")
    finally:
        server.server_close()
    if draft_path.read_bytes() != draft_before:
        raise RuntimeError("loading the real V1 draft changed its bytes")
    raw_draft = json.loads(draft_before)
    if loaded["document"] != raw_draft["document"] or loaded["revision"] != raw_draft["revision"]:
        raise RuntimeError("R1 did not load the V1 draft identically")

    return {
        "schema_version": "football_intelligence.g7f_c_r1.real_decisions_integrity.v1",
        "real_decisions_root": str(real_root),
        "captured_before_code_changes": True,
        "before": {
            **before,
            "categories": decision_categories(before),
        },
        "after": {
            **after,
            "categories": decision_categories(after),
        },
        "byte_identical": True,
        "v1_draft_loaded_identically_without_write": True,
        "existing_finalized_events_immutable": True,
        "production_ready": False,
    }


def original_bindings(source: Path) -> dict[str, str]:
    return read_json(source / "05_REVIEWER/reviewer_binding_hashes.json")


def reviewer_config(
    all_paths: dict[str, Path], *, decisions_root: Path, port: int = 0, bindings: dict[str, str] | None = None
) -> DensePersonReviewerConfig:
    source = all_paths["source"]
    return DensePersonReviewerConfig(
        selection_manifest_path=source / "01_SELECTION/dense_gold_selection_manifest.json",
        assets_root=source / "05_REVIEWER/assets",
        decisions_root=decisions_root,
        binding_hashes=bindings or original_bindings(source),
        reviewer_release=RELEASE,
        reveal_payload_path=source / "05_REVIEWER/sealed_candidate_reveal_payloads.json",
        pass_kind="FIRST_PASS",
        allowed_selection_statuses=("CALIBRATION_ONLY",),
        port=port,
    )


def run_focused_tests(all_paths: dict[str, Path], temp_root: Path) -> dict[str, Any]:
    if not FI_PYTHON.is_file():
        raise RuntimeError(f"required reviewer interpreter is missing: {FI_PYTHON}")
    args = [
        "tests/test_g7f_c_r1_reviewer_interaction.py",
        "tests/test_g7f_c_dense_person_gold.py",
        "-q",
        f"--basetemp={temp_root / 'pytest_temp'}",
        "-o",
        f"cache_dir={temp_root / 'pytest_cache'}",
    ]
    code = (
        "import json,sys; "
        f"sys.path.append({str(BASE_SITE_PACKAGES)!r}); "
        "import pytest; "
        f"raise SystemExit(pytest.main(json.loads({json.dumps(json.dumps(args))})))"
    )
    env = {**os.environ, "PYTHONPATH": str(all_paths["repo"] / "src")}
    result = subprocess.run(
        [str(FI_PYTHON), "-c", code],
        cwd=all_paths["repo"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"focused tests failed:\n{result.stdout}\n{result.stderr}")
    return {"exit_code": result.returncode, "stdout": result.stdout.strip(), "interpreter": str(FI_PYTHON)}


def run_edge_acceptance(all_paths: dict[str, Path], temp_root: Path, screenshot_path: Path) -> dict[str, Any]:
    decisions = temp_root / "browser_decisions"
    server = DensePersonHTTPServer(reviewer_config(all_paths, decisions_root=decisions, port=0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    try:
        result = subprocess.run(
            [
                "node",
                str(all_paths["repo"] / "scripts/g7f_c_r1_edge_acceptance.js"),
                f"http://127.0.0.1:{port}/",
                str(screenshot_path),
            ],
            cwd=all_paths["repo"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"live Edge acceptance failed:\n{result.stdout}\n{result.stderr}")
        payload = json.loads(result.stdout)
        if not payload.get("passed"):
            raise RuntimeError("live Edge acceptance did not pass")
        payload["screenshot"] = {
            "path": str(screenshot_path),
            "sha256": sha256_file(screenshot_path),
            "byte_size": screenshot_path.stat().st_size,
        }
        return payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def polygon(x1: int, y1: int, x2: int, y2: int) -> list[dict[str, int]]:
    return [{"x": x1, "y": y1}, {"x": x2, "y": y1}, {"x": x2, "y": y2}, {"x": x1, "y": y2}]


def api_immutability_acceptance(all_paths: dict[str, Path], decisions: Path) -> dict[str, Any]:
    server = DensePersonHTTPServer(reviewer_config(all_paths, decisions_root=decisions))
    try:
        bootstrap = server.blind_bootstrap()
        if scan_blind_payload(bootstrap):
            raise RuntimeError("candidate-blind bootstrap regression")
        if len(bootstrap["queue"]) != 6 or any(row["workflow_group"] != "CALIBRATION" for row in bootstrap["queue"]):
            raise RuntimeError("R1 reviewer must expose calibration only")
        image_id = bootstrap["queue"][0]["anonymous_dense_image_id"]
        if scan_blind_payload(server.store.state(image_id, "FIRST_PASS")):
            raise RuntimeError("candidate-blind state regression")
        document = {
            "people": [
                {
                    "instance_id": "person-001",
                    "relevance": "MATCH_RELEVANT",
                    "visible_mask_components": [polygon(100, 100, 130, 170)],
                }
            ],
            "ignore_regions": [],
            "reviewed_exhaustiveness_strips": list(range(8)),
            "unfinished_polygon": None,
            "completion_assertion": COMPLETION_ASSERTION,
        }
        saved = server.store.apply_action(
            {
                "action_id": "r1-api-save",
                "action_type": "SAVE_DRAFT",
                "anonymous_dense_image_id": image_id,
                "pass_kind": "FIRST_PASS",
                "expected_revision": 0,
                "document": document,
            }
        )
        finalized = server.store.apply_action(
            {
                "action_id": "r1-api-finalize",
                "action_type": "FINALIZE",
                "anonymous_dense_image_id": image_id,
                "pass_kind": "FIRST_PASS",
                "expected_revision": saved["revision"],
                "document": document,
            }
        )
        event_path = decisions / "events" / f"first_pass__{image_id}.json"
        ack_path = decisions / "acknowledgements" / f"first_pass__{image_id}.json"
        event_before = event_path.read_bytes()
        ack_before = ack_path.read_bytes()
        reloaded = server.store.state(image_id, "FIRST_PASS")
        reveal = server.store.apply_action(
            {
                "action_id": "r1-api-reveal",
                "action_type": "REVEAL_CANDIDATES",
                "anonymous_dense_image_id": image_id,
                "pass_kind": "FIRST_PASS",
                "expected_revision": finalized["revision"],
            }
        )
        try:
            server.store.apply_action(
                {
                    "action_id": "r1-api-illegal-save",
                    "action_type": "SAVE_DRAFT",
                    "anonymous_dense_image_id": image_id,
                    "pass_kind": "FIRST_PASS",
                    "expected_revision": finalized["revision"],
                    "document": document,
                }
            )
        except DensePersonConflictError as error:
            immutable_code = error.code
        else:
            raise RuntimeError("finalized frame accepted a mutation")
        if event_path.read_bytes() != event_before or ack_path.read_bytes() != ack_before:
            raise RuntimeError("reveal or reload changed finalized event/ack bytes")
        return {
            "candidate_blind_bootstrap_and_state": True,
            "calibration_only_queue_count": 6,
            "scored_annotation_authorized": False,
            "finalized_state_read_only": reloaded["read_only"],
            "post_final_reveal_read_only": reveal["read_only"],
            "event_and_ack_byte_identical_after_reveal": True,
            "finalized_mutation_error_code": immutable_code,
        }
    finally:
        server.server_close()


def build_release(all_paths: dict[str, Path], commit: str) -> tuple[dict[str, Any], dict[str, str]]:
    repo = all_paths["repo"]
    repair = all_paths["repair"]
    launcher = repair / "05_REVIEWER/launch_calibration_reviewer_r1.ps1"
    requirements = repair / "05_REVIEWER/REQUIREMENTS.md"
    release_files = [
        repo / "src/football_intelligence/dense_person_gold.py",
        repo / "src/football_intelligence/dense_person_reviewer.py",
        repo / "src/football_intelligence/dense_person_reviewer_static/index.html",
        repo / "src/football_intelligence/dense_person_reviewer_static/app.js",
        repo / "src/football_intelligence/dense_person_reviewer_static/styles.css",
        repo / "scripts/g7f_c_r1_run_reviewer.py",
        launcher,
        requirements,
    ]
    release = {
        "reviewer_release": RELEASE,
        "repository_commit": commit,
        "baseline_commit": BASELINE_COMMIT,
        "workflow_authorization": "CALIBRATION_ONLY",
        "files": [
            {
                "path": str(path),
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in release_files
        ],
        "production_ready": False,
    }
    release_path = repair / "05_REVIEWER/reviewer_release_manifest.json"
    write_json(release_path, release)
    bindings = original_bindings(all_paths["source"])
    bindings["reviewer_release_manifest_sha256"] = sha256_file(release_path)
    write_json(repair / "05_REVIEWER/reviewer_binding_hashes.json", bindings)
    return release, bindings


def launcher_acceptance(all_paths: dict[str, Path]) -> dict[str, Any]:
    launcher = all_paths["repair"] / "05_REVIEWER/launch_calibration_reviewer_r1.ps1"
    source = launcher.read_text(encoding="utf-8")
    required_command = r'C:\Users\sebgr\anaconda3\envs\fi-reviewer\python.exe -m pip install "pydantic>=2.10.3,<3"'
    checks = {
        "explicit_fi_reviewer_interpreter": str(FI_PYTHON) in source,
        "broken_dot_venv_absent": ".venv" not in source,
        "explicit_pythonpath": "$env:PYTHONPATH" in source,
        "pydantic_install_guidance_exact": required_command in source,
        "does_not_install_packages": "pip install"
        not in "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("throw")),
    }
    if not all(checks.values()):
        raise RuntimeError(f"launcher acceptance failed: {checks}")
    check = subprocess.run(
        [
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(launcher),
            "-CheckOnly",
        ],
        cwd=all_paths["repo"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if check.returncode:
        raise RuntimeError(f"reviewer launcher check failed: {check.stdout}\n{check.stderr}")
    runtime_check = json.loads(check.stdout.strip().splitlines()[-1])
    return {
        "launcher_path": str(launcher),
        "launcher_sha256": sha256_file(launcher),
        "checks": checks,
        "runtime_check": runtime_check,
        "python": {"executable": str(FI_PYTHON), "version": sys.version.split()[0]},
        "dependencies": {
            "hashlib": "stdlib",
            "numpy": np.__version__,
            "opencv_python": cv2.__version__,
            "pydantic": pydantic.__version__,
            "football_intelligence.dense_person_reviewer": "imported",
        },
        "required_pydantic_install_command": required_command,
        "packages_installed_by_acceptance": False,
        "production_ready": False,
    }


def write_handoff(
    all_paths: dict[str, Path],
    *,
    frozen: dict[str, Any],
    real_integrity: dict[str, Any],
    tests: dict[str, Any],
    environment: dict[str, Any],
    release: dict[str, Any],
    bindings: dict[str, str],
    upstream: dict[str, Any],
    commit: str,
) -> Path:
    handoff = all_paths["repair"] / "10_REVIEW_PACK/CHATGPT_HANDOFF"
    handoff.mkdir(parents=True, exist_ok=True)
    upstream_before = read_json(all_paths["source"] / "00_SOURCE_FREEZE/upstream_substrate_integrity_after.json")
    write_json(
        handoff / "00_EXECUTIVE_SUMMARY.json",
        {
            "stage": "G7F_C_R1_REVIEWER_INTERACTION_AND_UI_REPAIR",
            "decision": PASS,
            "primary_defect_repaired": True,
            "default_mode": "PAN_EDIT",
            "real_decisions_byte_identical": True,
            "calibration_may_resume": True,
            "annotation_resumed_automatically": False,
            "scored_annotation_authorized": False,
            "production_ready": False,
            "repository_commit": commit,
        },
    )
    write_json(handoff / "01_REAL_DECISIONS_BEFORE_AFTER_INTEGRITY.json", real_integrity)
    write_json(
        handoff / "02_FROZEN_CONTRACT_HASHES.json",
        {
            "frozen_contracts": frozen,
            "upstream_before": upstream_before,
            "upstream_after": upstream,
            "upstream_and_human_sources_verified": True,
            "production_ready": False,
        },
    )
    write_json(
        handoff / "03_INTERACTION_STATE_MACHINE_CONTRACT.json",
        {
            "persistent_modes": [
                "PAN_EDIT",
                "DRAW_PERSON",
                "ADD_VISIBLE_COMPONENT",
                "DRAW_IGNORE_REGION",
            ],
            "optional_transient_mode": "TEMPORARY_PAN",
            "default_mode": "PAN_EDIT",
            "vertex_addition_modes": ["DRAW_PERSON", "ADD_VISIBLE_COMPONENT", "DRAW_IGNORE_REGION"],
            "pan_edit_can_add_vertices": False,
            "returns_to_pan_edit_after": ["FINISH", "ESCAPE_OR_CANCEL", "IMAGE_LOAD", "FINALIZATION"],
            "space_hold": {"effect": "TEMPORARY_PAN", "preserves_working_polygon": True, "adds_vertices": False},
            "working_polygon_blocks": ["PREVIOUS", "NEXT", "IMAGE_DROPDOWN", "FINALIZE"],
            "minimum_commit_points": 3,
            "empty_person_or_ignore_created_on_mode_entry": False,
            "production_ready": False,
        },
    )
    (handoff / "04_UI_REDESIGN_AND_SHORTCUTS.md").write_text(
        "# G7F-C R1 UI redesign and shortcuts\n\n"
        "The reviewer now uses a compact top identity/navigation bar, an always-visible annotation toolbar, "
        "a persistent mode badge, a canvas-local view toolbar, and a single context inspector. Annotation "
        "tools and camera controls are visually and behaviorally separate.\n\n"
        "- Pan / Edit is the default and never creates vertices.\n"
        "- Draw person, Add visible part, and Ignore region are explicit drawing modes.\n"
        "- Finish and Cancel are always present and become available during drawing.\n"
        "- Space temporarily pans without changing the persistent mode or working points.\n"
        "- Selected people and ignore regions receive a strong highlight and explicit confirmed deletion.\n"
        "- Eight strip buttons show an explicit `x/8 reviewed` counter.\n"
        "- View controls are Fit width, Fit height, Zoom +/-, and Reset view.\n\n"
        "Shortcuts: `P` draw person, `C` add visible part, `I` ignore region, `V` or `A` Pan/Edit, "
        "`Enter` finish, `Escape` cancel, hold `Space` temporary pan, `Z` undo, and `Y` redo.\n\n"
        "Scored annotation remains unauthorized. `production_ready=false`\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(handoff / "05_ENVIRONMENT_AND_LAUNCHER_REPORT.json", environment)
    write_json(handoff / "06_TEST_AND_BROWSER_ACCEPTANCE.json", tests)
    write_json(
        handoff / "07_REVIEWER_R1_RELEASE_BINDINGS.json",
        {
            "reviewer_release": RELEASE,
            "release_manifest": release,
            "binding_hashes": bindings,
            "release_manifest_sha256": sha256_file(all_paths["repair"] / "05_REVIEWER/reviewer_release_manifest.json"),
            "v1_drafts_loadable": True,
            "v1_finalized_events_remain_valid": True,
            "workflow_authorization": "CALIBRATION_ONLY",
            "production_ready": False,
        },
    )
    (handoff / "08_DECISION.md").write_text(
        f"# Decision\n\n`{PASS}`\n\n"
        "Calibration may resume only by an operator explicitly running the R1 launcher. "
        "This repair did not resume annotation. Scored annotation remains unauthorized.\n\n"
        "`production_ready=false`\n",
        encoding="utf-8",
        newline="\n",
    )
    payload_files = sorted(path for path in handoff.iterdir() if path.is_file() and path.name != "10_MANIFEST.json")
    if len(payload_files) != 9:
        raise RuntimeError(f"handoff must contain exactly nine payload files before manifest: {payload_files}")
    write_json(
        handoff / "10_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7f_c_r1.handoff_manifest.v1",
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


def run(phase: str, repo: Path) -> dict[str, Any]:
    all_paths = paths(repo.resolve())
    frozen = verify_frozen(all_paths["source"])
    repair = all_paths["repair"]
    acceptance_root = repair / "07_ACCEPTANCE"
    acceptance_root.mkdir(parents=True, exist_ok=True)
    screenshot = acceptance_root / "reviewer_r1_initial.png"
    with tempfile.TemporaryDirectory(prefix="g7f_c_r1_", dir=acceptance_root) as temp_name:
        temp_root = Path(temp_name)
        focused = run_focused_tests(all_paths, temp_root)
        browser = run_edge_acceptance(all_paths, temp_root, screenshot)
        api = api_immutability_acceptance(all_paths, temp_root / "api_decisions")
    real_integrity = real_decision_integrity(all_paths)
    tests = {
        "focused_tests": focused,
        "live_edge_acceptance": browser,
        "api_and_immutability_acceptance": api,
        "mandatory_interaction_checks_passed": all(row["passed"] for row in browser["checks"]),
        "production_ready": False,
    }
    if phase == "preflight":
        return {
            "phase": phase,
            "decision": "PREFLIGHT_PASS",
            "focused": focused,
            "edge_check_count": len(browser["checks"]),
            "real_decisions_byte_identical": real_integrity["byte_identical"],
            "production_ready": False,
        }

    status = git(repo, "status", "--porcelain")
    head = git(repo, "rev-parse", "HEAD")
    origin = git(repo, "rev-parse", "origin/main")
    if status or head != origin:
        raise RuntimeError(f"final repository gate failed: dirty={bool(status)} head={head} origin/main={origin}")
    if subprocess.call(["git", "merge-base", "--is-ancestor", BASELINE_COMMIT, head], cwd=repo):
        raise RuntimeError("expected baseline is not an ancestor of the repair commit")
    upstream = verify_upstream(g7f_c_roots(repo))
    release, bindings = build_release(all_paths, head)
    environment = launcher_acceptance(all_paths)
    handoff = write_handoff(
        all_paths,
        frozen=frozen,
        real_integrity=real_integrity,
        tests=tests,
        environment=environment,
        release=release,
        bindings=bindings,
        upstream=upstream,
        commit=head,
    )
    return {
        "decision": PASS,
        "handoff": str(handoff),
        "repository_commit": head,
        "real_decisions_byte_identical": True,
        "annotation_resumed_automatically": False,
        "scored_annotation_authorized": False,
        "production_ready": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "finalize"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(json.dumps(run(args.phase, args.repo), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
