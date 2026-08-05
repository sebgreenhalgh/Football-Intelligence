"""Create the final-byte R6.2 release gate, closure evidence, and compact handoff."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
import urllib.request

from football_intelligence import g7e_b_r6_action_reducer, temporal_review

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
STAGE = PROJECT / (
    "experiments/football_observation_reasoner/part 8/" "G7E_B_R6_2_PRECISION_ZOOM_PAN_AND_COORDINATE_SAFE_MARKING_v1"
)
PACKAGE = STAGE / "03_PRECISION_NAVIGATION_IMPLEMENTATION/temporal_reviewer_r6_2"
REAL_ROOT = PART7 / (
    "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1/"
    "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
)
ASSET_ROOT = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/assets"
BEFORE = STAGE / "00_BASELINE_AND_REAL_STATE_FREEZE/real_state_file_manifest_before.json"
PASS = "PASS_G7E_B_R6_2_PRECISION_ZOOM_PAN_READY_FOR_TRANCHE_1_RESUME"
GATE_NAME = "G7E_B_R6_2_REAL_REVIEW_RELEASE_GATE.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=REPO, text=True).strip()


def require(path: Path, classification: str) -> dict[str, Any]:
    value = read(path)
    if value.get("classification") != classification:
        raise RuntimeError(f"required classification absent from {path}: {value.get('classification')}")
    return value


def real_closure() -> dict[str, Any]:
    before = read(BEFORE)
    rows = [
        {
            "relative_path": path.relative_to(REAL_ROOT).as_posix(),
            "byte_size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(REAL_ROOT.rglob("*"))
        if path.is_file()
    ]
    digest = sha256_bytes(
        "".join(f"{row['relative_path']}\t{row['byte_size']}\t{row['sha256']}\n" for row in rows).encode()
    )
    expected_rows = {
        row["relative_path"]: {key: row[key] for key in ("relative_path", "byte_size", "sha256")}
        for row in before["files"]
    }
    actual_rows = {row["relative_path"]: row for row in rows}
    mismatches = [
        {"relative_path": relative, "before": expected_rows.get(relative), "after": actual_rows.get(relative)}
        for relative in sorted(set(expected_rows) | set(actual_rows))
        if expected_rows.get(relative) != actual_rows.get(relative)
    ]
    if mismatches or digest != before["ordered_file_set_sha256"]:
        raise RuntimeError(f"real human-decision state changed during R6.2: {mismatches[:5]}")
    active = next(row["metadata"] for row in before["files"] if row.get("category") == "drafts")
    return {
        "schema_version": "football_intelligence.g7e_b_r6_2.real_state_closure.v1",
        "classification": "PASS_G7E_B_R6_2_REAL_STATE_BYTE_IDENTICAL",
        "file_count_before": before["file_count"],
        "file_count_after": len(rows),
        "ordered_file_set_sha256_before": before["ordered_file_set_sha256"],
        "ordered_file_set_sha256_after": digest,
        "mismatches": mismatches,
        "active_draft": active,
        "new_real_files": 0,
        "production_ready": False,
    }


def exact_bytes(commit: str) -> dict[str, Any]:
    direct = {
        "browser": (
            "src/football_intelligence/g7e_b_r6_temporal_review.js",
            "review.js",
        ),
        "viewport": (
            "src/football_intelligence/g7e_b_r6_2_viewport.js",
            "viewport_transform.js",
        ),
        "server": (
            "src/football_intelligence/temporal_review.py",
            "runtime_source_snapshot/football_intelligence/temporal_review.py",
        ),
        "reducer": (
            "src/football_intelligence/g7e_b_r6_action_reducer.py",
            "runtime_source_snapshot/football_intelligence/g7e_b_r6_action_reducer.py",
        ),
        "action_contract": (
            "src/football_intelligence/g7e_b_r6_server_action_contract.json",
            "server_action_contract.json",
        ),
    }
    direct["reviewer_state"] = (
        "src/football_intelligence/g7e_b_r5_reviewer_state.py",
        "runtime_source_snapshot/football_intelligence/g7e_b_r5_reviewer_state.py",
    )
    for source in sorted((REPO / "src/football_intelligence/temporal_reviewer").glob("*.py")):
        direct[f"runtime_{source.stem}"] = (
            source.relative_to(REPO).as_posix(),
            f"runtime_source_snapshot/football_intelligence/temporal_reviewer/{source.name}",
        )
    result: dict[str, Any] = {}
    for label, (repository_path, package_path) in direct.items():
        working = (REPO / repository_path).read_bytes()
        committed = subprocess.check_output(["git", "show", f"{commit}:{repository_path}"], cwd=REPO)
        packaged = (PACKAGE / package_path).read_bytes()
        hashes = {
            "working_tree_sha256": sha256_bytes(working),
            "git_object_sha256": sha256_bytes(committed),
            "packaged_sha256": sha256_bytes(packaged),
        }
        if len(set(hashes.values())) != 1:
            raise RuntimeError(f"final-byte mismatch for {label}: {hashes}")
        result[label] = hashes
    imported = {
        "server": sha256(Path(inspect.getfile(temporal_review))),
        "reducer": sha256(Path(inspect.getfile(g7e_b_r6_action_reducer))),
    }
    if imported["server"] != result["server"]["working_tree_sha256"]:
        raise RuntimeError("imported server is not the final committed source")
    if imported["reducer"] != result["reducer"]["working_tree_sha256"]:
        raise RuntimeError("imported reducer is not the final committed source")
    css_tail = (REPO / "src/football_intelligence/g7e_b_r6_2_temporal_review.css").read_bytes()
    if not (PACKAGE / "review.css").read_bytes().endswith(css_tail + b"\n"):
        raise RuntimeError("packaged CSS is not composed from the final R6.2 source")
    html_source = (REPO / "src/football_intelligence/g7e_b_r2_temporal_review.html").read_text(encoding="utf-8")
    expected_html = (
        html_source.replace(
            '<html lang="en">',
            '<html lang="en" data-release-revision="G7E_B_R6_2_PRECISION_ZOOM_PAN_COORDINATE_SAFE_MARKING_V1">',
            1,
        )
        .replace("R2 REVIEWER PREVIEW", "R6.1 FINAL-BYTE REVIEW PREVIEW")
        .replace(
            '<script src="/review.js"></script>',
            '<script src="/generated_client_contract.js"></script>\n  <script src="/review.js"></script>',
        )
    )
    if expected_html.encode() != (PACKAGE / "index.html").read_bytes():
        raise RuntimeError("packaged HTML is not the deterministic final-source transformation")
    result["html"] = {"deterministic_composition_sha256": sha256_bytes(expected_html.encode())}
    result["css"] = {
        "package_sha256": sha256(PACKAGE / "review.css"),
        "r6_2_source_sha256": sha256(REPO / "src/football_intelligence/g7e_b_r6_2_temporal_review.css"),
    }
    result["imported_module_sha256"] = imported
    return result


def package_manifest() -> dict[str, Any]:
    rows = [
        {
            "relative_path": path.relative_to(PACKAGE).as_posix(),
            "byte_size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(PACKAGE.rglob("*"))
        if path.is_file() and path.name != "package_manifest.json"
    ]
    return {
        "schema_version": "football_intelligence.g7e_b_r6_2.package_manifest.v1",
        "files": rows,
        "self_hash_excluded": True,
        "production_ready": False,
    }


def served_hashes() -> dict[str, str]:
    parent = STAGE / "11_RELEASE_GATE"
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="g7e_b_r6_2_served_", dir=parent))
    stream = (temporary / "server.log").open("wb")
    process = subprocess.Popen(
        [
            sys.executable,
            str(PACKAGE / "review_server.py"),
            "--package",
            str(PACKAGE),
            "--asset-root",
            str(ASSET_ROOT),
            "--decisions-root",
            str(temporary / "real"),
            "--practice-root",
            str(temporary / "practice"),
            "--port",
            "8826",
            "--acceptance-mode",
        ],
        cwd=REPO,
        stdout=stream,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        for _ in range(300):
            try:
                if urllib.request.urlopen("http://127.0.0.1:8826/", timeout=1).status == 200:
                    break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("final served-byte server did not start")
        mapping = {
            "index.html": "/",
            "review.js": "/review.js",
            "review.css": "/review.css",
            "viewport_transform.js": "/viewport_transform.js",
            "generated_client_contract.js": "/generated_client_contract.js",
        }
        result = {}
        for filename, route in mapping.items():
            data = urllib.request.urlopen(f"http://127.0.0.1:8826{route}", timeout=10).read()
            if data != (PACKAGE / filename).read_bytes():
                raise RuntimeError(f"served bytes differ from package: {filename}")
            result[filename] = sha256_bytes(data)
        return result
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        stream.close()
        shutil.rmtree(temporary, ignore_errors=True)


def mandatory_evidence(commit: str) -> tuple[dict[str, Any], dict[str, str]]:
    requirements = {
        "edge": (
            STAGE / "06_REAL_EDGE_AND_COORDINATE_ACCEPTANCE/edge_coordinate_and_navigation_acceptance.json",
            "PASS_G7E_B_R6_2_EDGE_COORDINATE_AND_NAVIGATION_ACCEPTANCE",
        ),
        "production": (
            STAGE
            / (
                "07_PRODUCTION_CHALLENGE_AND_120_BURST_AUDIT/04_PRODUCTION_PATH_CHALLENGE_SUITE/"
                "production_path_challenge_results.json"
            ),
            "PASS_G7E_B_R6_PRODUCTION_PATH_CHALLENGE",
        ),
        "marking": (
            STAGE
            / (
                "07_PRODUCTION_CHALLENGE_AND_120_BURST_AUDIT/04_PRODUCTION_PATH_CHALLENGE_SUITE/"
                "r6_2_marking_and_branch_acceptance.json"
            ),
            "PASS_G7E_B_R6_2_MARKING_AND_ALL_BRANCH_ACCEPTANCE",
        ),
        "corpus": (
            STAGE
            / (
                "07_PRODUCTION_CHALLENGE_AND_120_BURST_AUDIT/05_FULL_120_BURST_BROWSER_AUDIT/"
                "full_120_burst_browser_audit.json"
            ),
            "PASS_G7E_B_R6_FULL_120_BURST_PRODUCTION_DOM_AUDIT",
        ),
        "fault": (
            STAGE
            / "07_PRODUCTION_CHALLENGE_AND_120_BURST_AUDIT/06_FAULT_AND_RACE_CHALLENGE/fault_and_race_results.json",
            "PASS_G7E_B_R6_FAULT_AND_RACE_CHALLENGE",
        ),
        "tests": (
            STAGE / "09_FOCUSED_REGRESSION_TESTS/focused_test_report.json",
            "PASS_G7E_B_R6_2_FOCUSED_AND_INHERITED_TESTS",
        ),
    }
    documents = {name: require(path, classification) for name, (path, classification) in requirements.items()}
    if documents["tests"].get("git_commit") != commit:
        raise RuntimeError("focused tests are not bound to the final commit")
    bundle = sha256(PACKAGE / "review.js")
    for name in ("edge", "production", "marking", "corpus", "fault"):
        recorded = documents[name].get("production_browser_bundle_sha256") or documents[name].get(
            "package_review_js_sha256"
        )
        if recorded != bundle:
            raise RuntimeError(f"{name} evidence is not bound to the final browser bundle")
    hashes = {name: sha256(path) for name, (path, _) in requirements.items()}
    return documents, hashes


def prepare_gate() -> None:
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise RuntimeError("tracked worktree must be clean before final-byte gate creation")
    closure = real_closure()
    write(STAGE / "10_REAL_STATE_ZERO_MUTATION_AND_RESUME/real_state_closing_hash_comparison.json", closure)
    evidence, evidence_hashes = mandatory_evidence(commit)
    equivalence = {
        "schema_version": "football_intelligence.g7e_b_r6_2.final_byte_equivalence.v1",
        "classification": "PASS_G7E_B_R6_2_FINAL_BYTE_EQUIVALENCE",
        "git_commit": commit,
        "source_package_import_equivalence": exact_bytes(commit),
        "mandatory_evidence_sha256": evidence_hashes,
        "served_hashes": {},
        "production_ready": False,
    }
    gate_files = tuple(
        path.relative_to(PACKAGE).as_posix()
        for path in sorted(PACKAGE.rglob("*"))
        if path.is_file() and path.name not in {GATE_NAME, "package_manifest.json"}
    )
    gate = {
        "schema_version": "football_intelligence.g7e_b_r6_2.real_review_release_gate.v1",
        "release_classification": PASS,
        "git_commit": commit,
        "review_protocol_revision": temporal_review.R6_REVIEW_REVISION,
        "action_contract_sha256": sha256(PACKAGE / "server_action_contract.json"),
        "reviewer_file_sha256": {relative: sha256(PACKAGE / relative) for relative in gate_files},
        "mandatory_evidence_sha256": evidence_hashes,
        "real_state_ordered_file_set_sha256": closure["ordered_file_set_sha256_after"],
        "active_burst": closure["active_draft"]["burst_id"],
        "active_question": closure["active_draft"]["current_question"],
        "active_question_instance_key": closure["active_draft"]["current_question_instance_key"],
        "active_draft_revision": closure["active_draft"]["draft_version"],
        "human_answer_changed": False,
        "real_state_mutations": 0,
        "production_ready": False,
    }
    write(PACKAGE / GATE_NAME, gate)
    write(PACKAGE / "package_manifest.json", package_manifest())
    equivalence["served_hashes"] = served_hashes()
    equivalence["release_gate_sha256"] = sha256(PACKAGE / GATE_NAME)
    equivalence["package_manifest_sha256"] = sha256(PACKAGE / "package_manifest.json")
    write(STAGE / "11_RELEASE_GATE/final_byte_equivalence.json", equivalence)
    write(STAGE / f"11_RELEASE_GATE/{GATE_NAME}", gate)
    write(
        STAGE / "02_SOURCE_PATH_DIAGNOSIS/source_path_diagnosis.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6_2.source_path_diagnosis.v1",
            "classification": "PASS_G7E_B_R6_2_SOURCE_PATH_DIAGNOSIS",
            "root_cause": "The prior reviewer exposed only fixed-centre zoom controls and no movable viewport state.",
            "implemented_boundary": (
                "One canonical normalized focal-point transform drives panorama and Closer look independently."
            ),
            "source_truth_changed": False,
            "production_ready": False,
        },
    )
    write(
        STAGE / "04_VIEWPORT_TRANSFORM_AND_COORDINATE_SPECIFICATION/viewport_transform_design.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6_2.viewport_transform.v1",
            "classification": "PASS_G7E_B_R6_2_CANONICAL_VIEWPORT_TRANSFORM",
            "state": ["zoom", "focalX", "focalY", "panMode"],
            "source_mapping": (
                "CSS client point -> canonical source through the rendered image rectangle; "
                "DPR affects backing pixels only."
            ),
            "lock_view": "normalized focal coordinates and zoom copied across all nine frames",
            "annotation_guard": (
                "5 CSS-pixel click-versus-drag threshold; Pan, Space-pan, and middle-pan never annotate"
            ),
            "production_ready": False,
        },
    )
    write(
        STAGE / "05_IMPLEMENTATION_MANIFEST/implementation_manifest.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6_2.implementation_manifest.v1",
            "classification": "PASS_G7E_B_R6_2_IMPLEMENTATION_HASH_BOUND",
            "git_commit": commit,
            "changed_paths": git("diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines(),
            "package_build": evidence.get("tests", {}).get("classification"),
            "source_truth_changed": False,
            "human_truth_changed": False,
            "production_ready": False,
        },
    )
    print("PASS_G7E_B_R6_2_RELEASE_GATE_PREPARED")


def build_handoff(context: dict[str, Any]) -> None:
    root = STAGE / "12_REVIEW_PACK/CHATGPT_HANDOFF"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    items = {
        "00_EXECUTIVE_SUMMARY.json": context["executive"],
        "01_SCOPE_AND_BASELINE.json": read(STAGE / "00_BASELINE_AND_REAL_STATE_FREEZE/baseline_and_real_state.json"),
        "02_REAL_STATE_CLOSURE.json": context["closure"],
        "03_NAVIGATION_IMPLEMENTATION.json": read(
            STAGE / "04_VIEWPORT_TRANSFORM_AND_COORDINATE_SPECIFICATION/viewport_transform_design.json"
        ),
        "04_COORDINATE_ACCEPTANCE.json": context["evidence"]["edge"],
        "05_PRODUCTION_PATH_AND_120.json": {
            "production_path": context["evidence"]["production"],
            "marking_and_branches": context["evidence"]["marking"],
            "full_120_bursts": context["evidence"]["corpus"],
        },
        "06_FAULT_AND_REGRESSION.json": {
            "fault_and_race": context["evidence"]["fault"],
            "focused_tests": context["evidence"]["tests"],
        },
        "07_FINAL_BYTE_EQUIVALENCE.json": context["equivalence"],
        "08_REAL_RESUME.json": context["resume"],
        "09_RELEASE_GATE.json": read(PACKAGE / GATE_NAME),
        "10_CHANGE_MANIFEST.json": read(STAGE / "05_IMPLEMENTATION_MANIFEST/implementation_manifest.json"),
        "13_REVIEWER_BUILD.json": read(PACKAGE / "build_manifest.json"),
    }
    for filename, value in items.items():
        write(root / filename, value)
    contact_sheet = STAGE / "06_REAL_EDGE_AND_COORDINATE_ACCEPTANCE/coordinate_navigation_contact_sheet.png"
    paused = STAGE / "10_REAL_STATE_ZERO_MUTATION_AND_RESUME/real_reviewer_exact_paused_draft.png"
    shutil.copyfile(contact_sheet, root / "11_COORDINATE_NAVIGATION_CONTACT_SHEET.png")
    shutil.copyfile(paused, root / "12_REAL_PAUSED_DRAFT.png")
    shutil.copyfile(STAGE / "HUMAN_RESUME_INSTRUCTIONS.md", root / "14_HUMAN_RESUME_INSTRUCTIONS.md")
    rows = [
        {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "15_MANIFEST.json"
    ]
    if len(rows) > 19:
        raise RuntimeError("handoff exceeds the 20-file cap")
    write(
        root / "15_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6_2.handoff_manifest.v1",
            "files": rows,
            "self_hash_excluded": True,
        },
    )


def finalize() -> None:
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise RuntimeError("tracked worktree must remain clean during closure")
    closure = real_closure()
    evidence, _ = mandatory_evidence(commit)
    equivalence = require(
        STAGE / "11_RELEASE_GATE/final_byte_equivalence.json",
        "PASS_G7E_B_R6_2_FINAL_BYTE_EQUIVALENCE",
    )
    resume = require(
        STAGE / "10_REAL_STATE_ZERO_MUTATION_AND_RESUME/real_resume_edge_acceptance.json",
        "PASS_G7E_B_R6_2_REAL_REVIEWER_EXACT_DRAFT_RESTORED",
    )
    if resume.get("git_commit") != commit or resume.get("real_root_mutations") != 0:
        raise RuntimeError("real resume is not bound to the final commit and byte-identical root")
    executive = {
        "schema_version": "football_intelligence.g7e_b_r6_2.executive_summary.v1",
        "classification": PASS,
        "git_commit": commit,
        "real_state_file_count": closure["file_count_after"],
        "real_state_ordered_sha256": closure["ordered_file_set_sha256_after"],
        "active_burst": closure["active_draft"]["burst_id"],
        "active_question": closure["active_draft"]["current_question"],
        "active_draft_revision": closure["active_draft"]["draft_version"],
        "real_state_mutations": 0,
        "reviewer_answered_by_codex": False,
        "production_ready": False,
    }
    build_handoff(
        {
            "executive": executive,
            "closure": closure,
            "evidence": evidence,
            "equivalence": equivalence,
            "resume": resume,
        }
    )
    write(STAGE / "11_RELEASE_GATE/release_decision.json", executive)
    for work in STAGE.rglob("_browser_work"):
        shutil.rmtree(work, ignore_errors=True)
    print(PASS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-gate", action="store_true")
    arguments = parser.parse_args()
    if arguments.prepare_gate:
        prepare_gate()
    else:
        finalize()


if __name__ == "__main__":
    main()
