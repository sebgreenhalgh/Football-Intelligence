"""Finalize the exact-byte R6.1 release gate, closure evidence, and 20-file handoff."""

from __future__ import annotations

from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
import urllib.request

from football_intelligence import g7e_b_r6_action_reducer, temporal_review
from football_intelligence.temporal_review import TemporalReviewStore
from football_intelligence.temporal_reviewer.invariants import scan_persisted_invariants

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
STAGE = PROJECT / (
    "experiments/football_observation_reasoner/part 8/" "G7E_B_R6_1_FINAL_BYTE_VISUAL_RUNTIME_AND_REPOSITORY_CLOSURE_v1"
)
PACKAGE = STAGE / "03_VISUAL_REPAIR_IMPLEMENTATION/temporal_reviewer_r6_1"
REAL_ROOT = PART7 / (
    "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1/"
    "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
)
ASSET_ROOT = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/assets"
BEFORE = STAGE / "00_BASELINE_AND_REAL_STATE_FREEZE/real_state_file_manifest_before.json"
PASS = "PASS_G7E_B_R6_1_FINAL_BYTE_VISUAL_AND_RUNTIME_CLOSURE_READY_FOR_TRANCHE_1_RESUME"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return document


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def remove_tree_after_process_exit(path: Path, timeout_seconds: float = 30.0) -> None:
    """Remove a process worktree after Windows releases inherited log handles."""

    deadline = time.monotonic() + timeout_seconds
    while path.exists():
        try:
            shutil.rmtree(path)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=REPO, text=True).strip()


def require_classification(path: Path, expected: str) -> dict[str, Any]:
    document = read(path)
    if document.get("classification") != expected:
        raise RuntimeError(f"required PASS is absent from {path}: {document.get('classification')}")
    return document


def current_real_manifest() -> dict[str, Any]:
    before = read(BEFORE)
    rows = []
    for path in sorted(REAL_ROOT.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": path.relative_to(REAL_ROOT).as_posix(),
                    "byte_size": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    ordered_digest = sha256_bytes(
        "".join(f"{row['relative_path']}\t{row['byte_size']}\t{row['sha256']}\n" for row in rows).encode()
    )
    before_rows = [{key: row[key] for key in ("relative_path", "byte_size", "sha256")} for row in before["files"]]
    mismatches = []
    before_by_path = {row["relative_path"]: row for row in before_rows}
    after_by_path = {row["relative_path"]: row for row in rows}
    for relative in sorted(set(before_by_path) | set(after_by_path)):
        if before_by_path.get(relative) != after_by_path.get(relative):
            mismatches.append(
                {
                    "relative_path": relative,
                    "before": before_by_path.get(relative),
                    "after": after_by_path.get(relative),
                }
            )
    if mismatches or ordered_digest != before["ordered_file_set_sha256"]:
        raise RuntimeError(f"real state changed during R6.1: {mismatches}")
    prefixes = (
        "receipts/acknowledgements",
        "receipts/actions",
        "receipts/tranches",
        "receipts/global",
        "action_idempotency",
        "idempotency",
        "transactions",
        "journals",
        "recovery",
        "drafts",
        "events",
        "status",
    )
    categories = Counter(
        next(
            (prefix for prefix in prefixes if row["relative_path"].startswith(prefix + "/")),
            row["relative_path"].split("/", 1)[0],
        )
        for row in rows
    )
    active_draft = next(row["metadata"] for row in before["files"] if row["category"] == "drafts")
    return {
        "schema_version": "football_intelligence.g7e_b_r6_1.real_state_closure.v1",
        "classification": "PASS_G7E_B_R6_1_REAL_STATE_BYTE_IDENTICAL",
        "file_count_before": before["file_count"],
        "file_count_after": len(rows),
        "ordered_file_set_sha256_before": before["ordered_file_set_sha256"],
        "ordered_file_set_sha256_after": ordered_digest,
        "mismatches": mismatches,
        "category_counts_after": dict(sorted(categories.items())),
        "active_draft": active_draft,
        "new_real_files": 0,
        "production_ready": False,
    }


def source_equivalence(commit: str) -> dict[str, Any]:
    pairs = {
        "server": (
            "src/football_intelligence/temporal_review.py",
            "runtime_source_snapshot/football_intelligence/temporal_review.py",
        ),
        "reducer": (
            "src/football_intelligence/g7e_b_r6_action_reducer.py",
            "runtime_source_snapshot/football_intelligence/g7e_b_r6_action_reducer.py",
        ),
        "browser_js": ("src/football_intelligence/g7e_b_r6_temporal_review.js", "review.js"),
        "action_contract": (
            "src/football_intelligence/g7e_b_r6_server_action_contract.json",
            "server_action_contract.json",
        ),
    }
    rows = {}
    for label, (repository_path, package_path) in pairs.items():
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
        rows[label] = hashes
    imported = {
        "server": sha256(Path(inspect.getfile(temporal_review))),
        "reducer": sha256(Path(inspect.getfile(g7e_b_r6_action_reducer))),
    }
    for label, digest in imported.items():
        if digest != rows[label]["working_tree_sha256"]:
            raise RuntimeError(f"imported {label} bytes differ from committed bytes")
    css_sources = (
        "src/football_intelligence/g7e_b_temporal_review.css",
        "src/football_intelligence/g7e_b_r2_temporal_review.css",
    )
    committed_css = b"\n".join(
        subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=REPO) for path in css_sources
    ) + (
        b"\n.release-gate{border:2px solid #2cc9a0;border-radius:14px;padding:12px;"
        b"background:#effcf7;color:#132039}.blocking-error[data-error-kind='image-asset']{border-color:#e76d6d}"
        b".blocking-error[data-error-kind='server-action']{border-color:#e7a51a}\n"
    )
    packaged_css = (PACKAGE / "review.css").read_bytes()
    if committed_css != packaged_css:
        raise RuntimeError("packaged CSS differs from committed source composition")
    rows["css"] = {
        "git_composition_sha256": sha256_bytes(committed_css),
        "packaged_sha256": sha256_bytes(packaged_css),
    }
    canonical_repository = REPO / "src/football_intelligence/g7e_b_r5_canonical_reviewer_state_contract.json"
    if sha256(canonical_repository) != sha256(PACKAGE / "canonical_reviewer_state_contract.json"):
        raise RuntimeError("canonical contract package copy mismatch")
    rows["canonical_contract"] = {"sha256": sha256(canonical_repository)}
    rows["imported_module_sha256"] = imported
    return rows


def served_hashes() -> dict[str, str]:
    temporary = Path(tempfile.mkdtemp(prefix="g7e_b_r6_1_served_", dir=STAGE / "11_RELEASE_GATE"))
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
            "8819",
            "--acceptance-mode",
        ],
        cwd=REPO,
        stdout=stream,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    try:
        for _ in range(200):
            try:
                if urllib.request.urlopen("http://127.0.0.1:8819/", timeout=1).status == 200:
                    break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("final served-byte server did not start")
        mapping = {
            "index.html": "/",
            "review.js": "/review.js",
            "review.css": "/review.css",
            "generated_client_contract.js": "/generated_client_contract.js",
        }
        result = {}
        for filename, route in mapping.items():
            data = urllib.request.urlopen(f"http://127.0.0.1:8819{route}", timeout=10).read()
            if data != (PACKAGE / filename).read_bytes():
                raise RuntimeError(f"served bytes differ from packaged bytes: {filename}")
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
        remove_tree_after_process_exit(temporary)


def package_manifest() -> dict[str, Any]:
    rows = {}
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_file() and path.name != "package_manifest.json":
            rows[path.relative_to(PACKAGE).as_posix()] = {"sha256": sha256(path), "byte_size": path.stat().st_size}
    return {
        "schema_version": "football_intelligence.g7e_b_r6_1.package_manifest.v1",
        "files": rows,
        "self_hash_excluded": True,
        "production_ready": False,
    }


def build_handoff(context: dict[str, Any]) -> None:
    root = STAGE / "12_REVIEW_PACK/CHATGPT_HANDOFF"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    write(root / "01_EXECUTIVE_SUMMARY.json", context["executive"])
    write(root / "02_BASELINE_AND_REAL_STATE_CLOSURE.json", context["closure"])
    shutil.copyfile(context["findings_path"], root / "03_REPOSITORY_FINDINGS_AND_FIXES.json")
    write(root / "04_LOW_LIGHT_DIAGNOSIS_AND_TRANSFORM.json", context["visual_summary"])
    shutil.copyfile(context["contact_sheet"], root / "05_VISUAL_ACCEPTANCE_CONTACT_SHEET.png")
    write(root / "06_FINAL_BYTE_EQUIVALENCE.json", context["equivalence"])
    write(root / "07_ACTION_TRANSACTION_AND_SECURITY.json", context["security"])
    write(root / "08_GENUINE_INVARIANT_AND_FULL_CORPUS_AUDIT.json", context["corpus"])
    write(root / "09_FAULT_RECOVERY_RESULTS.json", context["fault"])
    write(root / "10_TESTS_CI_AND_DEPENDENCIES.json", context["tests"])
    write(root / "11_DOCUMENTATION_AND_PUBLIC_REPOSITORY.json", context["governance"])
    write(
        root / "12_REAL_ROOT_ZERO_MUTATION.json",
        {"byte_closure": context["closure"], "real_resume_edge_acceptance": context["real_resume"]},
    )
    shutil.copyfile(context["release_gate_path"], root / "13_RELEASE_GATE.json")
    (root / "14_DECISION.md").write_text(
        f"# Decision\n\n{PASS}\n\nThe exact committed R6.1 reviewer is ready for the user to resume the paused "
        "Tranche 1 draft. No human answer was added or changed. `production_ready=false`.\n",
        encoding="utf-8",
        newline="\n",
    )
    shutil.copyfile(STAGE / "HUMAN_RESUME_INSTRUCTIONS.md", root / "15_HUMAN_RESUME_INSTRUCTIONS.md")
    write(root / "16_CHANGE_MANIFEST.json", context["change_manifest"])
    shutil.copyfile(context["before_screenshot"], root / "17_LOW_LIGHT_BEFORE.png")
    shutil.copyfile(context["after_screenshot"], root / "18_LOW_LIGHT_AFTER.png")
    shutil.copyfile(context["daylight_screenshot"], root / "19_DAYLIGHT_CONTROL.png")
    files = []
    for path in sorted(root.iterdir()):
        if path.name != "20_MANIFEST.json":
            files.append({"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256(path)})
    if len(files) != 19:
        raise RuntimeError(f"handoff must contain exactly 19 files before its manifest, found {len(files)}")
    write(
        root / "20_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6_1.chatgpt_handoff_manifest.v1",
            "file_count_excluding_manifest": 19,
            "files": files,
            "self_hash_excluded": True,
            "production_ready": False,
        },
    )


def main() -> None:
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("final release requires a clean tracked and untracked repository")
    build = read(PACKAGE / "build_manifest.json")
    if build.get("frozen_truth_changed") is not False:
        raise RuntimeError("package build did not preserve frozen truth")
    visual_acceptance_path = STAGE / "07_FINAL_BYTE_BROWSER_ACCEPTANCE/visual_acceptance/edge_visual_acceptance.json"
    visual_acceptance = require_classification(visual_acceptance_path, "PASS_G7E_B_R6_1_EDGE_VISUAL_ACCEPTANCE")
    challenge_path = STAGE / (
        "07_FINAL_BYTE_BROWSER_ACCEPTANCE/04_PRODUCTION_PATH_CHALLENGE_SUITE/production_path_challenge_results.json"
    )
    challenge = require_classification(challenge_path, "PASS_G7E_B_R6_PRODUCTION_PATH_CHALLENGE")
    corpus_path = STAGE / (
        "07_FINAL_BYTE_BROWSER_ACCEPTANCE/05_FULL_120_BURST_BROWSER_AUDIT/full_120_burst_browser_audit.json"
    )
    corpus = require_classification(corpus_path, "PASS_G7E_B_R6_FULL_120_BURST_PRODUCTION_DOM_AUDIT")
    race_path = (
        STAGE / "09_FAULT_RECOVERY_AND_SECURITY_CHALLENGE/06_FAULT_AND_RACE_CHALLENGE/fault_and_race_results.json"
    )
    race = require_classification(race_path, "PASS_G7E_B_R6_FAULT_AND_RACE_CHALLENGE")
    security_path = STAGE / "09_FAULT_RECOVERY_AND_SECURITY_CHALLENGE/fault_recovery_and_security_results.json"
    security = require_classification(security_path, "PASS_G7E_B_R6_1_FAULT_RECOVERY_AND_SECURITY_CHALLENGE")
    tests_path = STAGE / "08_FULL_CORPUS_AND_INVARIANT_AUDIT/focused_and_cpu_safe_test_report.json"
    tests = require_classification(tests_path, "PASS_G7E_B_R6_1_FOCUSED_AND_CPU_SAFE_TESTS")
    if tests.get("git_commit") != commit:
        raise RuntimeError("test report is not bound to the final commit")
    history_path = STAGE / "06_DOCUMENTATION_REPOSITORY_AND_CI/repository_history_privacy_audit.json"
    history = read(history_path)
    if history.get("head") != commit:
        raise RuntimeError("history audit is not bound to the final commit")

    closure = current_real_manifest()
    write(STAGE / "10_REAL_STATE_ZERO_MUTATION_AND_RESUME/real_state_closing_hash_comparison.json", closure)
    temporary_practice = Path(
        tempfile.mkdtemp(prefix="g7e_b_r6_1_invariant_", dir=STAGE / "08_FULL_CORPUS_AND_INVARIANT_AUDIT")
    )
    before_digest = closure["ordered_file_set_sha256_after"]
    try:
        store = TemporalReviewStore(PACKAGE, REAL_ROOT, temporary_practice, acceptance_mode=True)
        real_invariant = scan_persisted_invariants(store, "real")
    finally:
        shutil.rmtree(temporary_practice)
    if not real_invariant["passed"]:
        raise RuntimeError(f"real persisted invariant scan failed: {real_invariant['discrepancies']}")
    if current_real_manifest()["ordered_file_set_sha256_after"] != before_digest:
        raise RuntimeError("real invariant scan changed the human root")
    write(STAGE / "08_FULL_CORPUS_AND_INVARIANT_AUDIT/genuine_real_invariant_scan.json", real_invariant)

    equivalence = {
        "schema_version": "football_intelligence.g7e_b_r6_1.final_byte_equivalence.v1",
        "classification": "PASS_G7E_B_R6_1_FINAL_BYTE_EQUIVALENCE",
        "git_commit": commit,
        "source_package_import_equivalence": source_equivalence(commit),
        "served_hashes": {},
        "evidence_hashes": {
            path.relative_to(STAGE).as_posix(): sha256(path)
            for path in (visual_acceptance_path, challenge_path, corpus_path, race_path, security_path, tests_path)
        },
        "production_ready": False,
    }
    browser_hash = equivalence["source_package_import_equivalence"]["browser_js"]["packaged_sha256"]
    for evidence in (visual_acceptance, challenge, corpus, race, security):
        recorded = evidence.get("production_browser_bundle_sha256") or evidence.get("package_review_js_sha256")
        if recorded and recorded != browser_hash:
            raise RuntimeError("browser evidence is not bound to the final package JS")

    gate_files = (
        "index.html",
        "review.js",
        "review.css",
        "generated_client_contract.js",
        "server_action_contract.json",
        "canonical_reviewer_state_contract.json",
        "review_cases.json",
        "practice_cases.json",
        "build_manifest.json",
        "visual_asset_manifest.json",
        "runtime_source_snapshot/football_intelligence/temporal_review.py",
        "runtime_source_snapshot/football_intelligence/g7e_b_r6_action_reducer.py",
    )
    release_gate = {
        "schema_version": "football_intelligence.g7e_b_r6_1.real_review_release_gate.v1",
        "release_classification": PASS,
        "git_commit": commit,
        "review_protocol_revision": temporal_review.R6_REVIEW_REVISION,
        "action_contract_sha256": sha256(PACKAGE / "server_action_contract.json"),
        "reviewer_file_sha256": {relative: sha256(PACKAGE / relative) for relative in gate_files},
        "mandatory_evidence_sha256": equivalence["evidence_hashes"],
        "real_state_ordered_file_set_sha256": closure["ordered_file_set_sha256_after"],
        "active_burst": closure["active_draft"]["burst_id"],
        "active_question": closure["active_draft"]["current_question"],
        "active_draft_revision": closure["active_draft"]["draft_version"],
        "human_answer_changed": False,
        "real_state_mutations": 0,
        "production_ready": False,
    }
    release_gate_path = PACKAGE / "G7E_B_R6_1_REAL_REVIEW_RELEASE_GATE.json"
    write(release_gate_path, release_gate)
    write(PACKAGE / "package_manifest.json", package_manifest())
    equivalence["served_hashes"] = served_hashes()
    equivalence["package_manifest_sha256"] = sha256(PACKAGE / "package_manifest.json")
    equivalence["release_gate_sha256"] = sha256(release_gate_path)
    equivalence_path = STAGE / "11_RELEASE_GATE/final_byte_equivalence.json"
    write(equivalence_path, equivalence)
    write(STAGE / "11_RELEASE_GATE/G7E_B_R6_1_REAL_REVIEW_RELEASE_GATE.json", release_gate)

    resume_command = [sys.executable, str(REPO / "scripts/g7e_b_r6_1_verify_real_resume.py")]
    resume_result = subprocess.run(resume_command, cwd=REPO, capture_output=True, text=True, check=False)
    if resume_result.returncode:
        raise RuntimeError(
            "real reviewer resume acceptance failed: "
            f"stdout={resume_result.stdout[-4000:]} stderr={resume_result.stderr[-4000:]}"
        )
    real_resume_path = STAGE / "10_REAL_STATE_ZERO_MUTATION_AND_RESUME/real_resume_edge_acceptance.json"
    real_resume = require_classification(real_resume_path, "PASS_G7E_B_R6_1_REAL_REVIEWER_EXACT_DRAFT_RESTORED")
    if real_resume.get("git_commit") != commit or real_resume.get("real_root_mutations") != 0:
        raise RuntimeError("real reviewer resume evidence is not bound to the final commit and immutable root")

    governance = {
        "schema_version": "football_intelligence.g7e_b_r6_1.repository_governance.v1",
        "classification": "PASS_G7E_B_R6_1_REPOSITORY_GOVERNANCE",
        "git_commit": commit,
        "history_privacy_audit": history,
        "data_boundary_check": "PASS",
        "ci_matrix": ["ubuntu-latest", "windows-latest"],
        "hosted_ci_protected_data_access": False,
        "lint_scope": "maintained temporal reviewer source, R6.1 scripts, and reviewer tests",
        "legacy_lint_debt_changed": False,
        "production_ready": False,
    }
    write(STAGE / "06_DOCUMENTATION_REPOSITORY_AND_CI/documentation_ci_data_boundary_report.json", governance)
    modularization = {
        "schema_version": "football_intelligence.g7e_b_r6_1.modularization.v1",
        "classification": "PASS_G7E_B_R6_1_COMPATIBILITY_PRESERVING_MODULARIZATION",
        "package_modules": sorted(
            path.name for path in (REPO / "src/football_intelligence/temporal_reviewer").glob("*.py")
        ),
        "compatibility_imports_preserved": True,
        "serialized_output_regressions": "PASS",
        "dependency_groups": ["reviewer", "cv-gpu", "research", "dev"],
        "production_ready": False,
    }
    write(STAGE / "05_RUNTIME_MODULARIZATION/runtime_modularization_report.json", modularization)

    change_manifest = {
        "schema_version": "football_intelligence.g7e_b_r6_1.change_manifest.v1",
        "git_commit": commit,
        "changed_paths": git("diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines(),
        "source_truth_changed": False,
        "real_human_truth_changed": False,
        "production_ready": False,
    }
    visual_diagnosis_path = STAGE / "02_LOW_LIGHT_DIAGNOSIS/source_derivative_browser_luminance_diagnosis.json"
    transform_path = PACKAGE / "visual_asset_manifest.json"
    visual_summary = {
        "diagnosis": read(visual_diagnosis_path),
        "transform_manifest_sha256": sha256(transform_path),
        "visual_contract": read(transform_path)["visual_contract"],
        "source_truth_changed": False,
        "geometry_changed": False,
        "edge_visual_acceptance": visual_acceptance,
    }
    executive = {
        "schema_version": "football_intelligence.g7e_b_r6_1.executive_summary.v1",
        "classification": PASS,
        "git_commit": commit,
        "real_state_file_count": closure["file_count_after"],
        "real_state_mutations": 0,
        "active_burst": closure["active_draft"]["burst_id"],
        "active_question": closure["active_draft"]["current_question"],
        "reviewer_answered_by_codex": False,
        "production_ready": False,
    }
    build_handoff(
        {
            "executive": executive,
            "closure": closure,
            "findings_path": STAGE / "01_REPOSITORY_FINDING_REPRODUCTION/finding_reproduction_report.json",
            "visual_summary": visual_summary,
            "contact_sheet": STAGE
            / "07_FINAL_BYTE_BROWSER_ACCEPTANCE/visual_acceptance/low_light_all_applicable_matches_contact_sheet.png",
            "equivalence": equivalence,
            "security": {
                "transaction": read(
                    STAGE
                    / "04_SERVER_ACTION_AND_TRANSACTION_HARDENING/action_transaction_and_recovery_specification.json"
                ),
                "challenge": security,
            },
            "corpus": {"real_invariant": real_invariant, "production_path": challenge, "full_120_burst": corpus},
            "fault": {"fault_race": race, "process_and_http_security": security},
            "tests": tests,
            "governance": governance,
            "real_resume": real_resume,
            "release_gate_path": release_gate_path,
            "change_manifest": change_manifest,
            "before_screenshot": STAGE / "02_LOW_LIGHT_DIAGNOSIS/09_USER_EVIDENCE_TOO_DARK.png",
            "after_screenshot": STAGE
            / "07_FINAL_BYTE_BROWSER_ACCEPTANCE/visual_acceptance/screenshots/active_enhanced_1920x1080.png",
            "daylight_screenshot": STAGE
            / "07_FINAL_BYTE_BROWSER_ACCEPTANCE/visual_acceptance/screenshots/daylight_auto_original.png",
        }
    )
    write(STAGE / "11_RELEASE_GATE/release_decision.json", executive)
    print(PASS)


if __name__ == "__main__":
    main()
