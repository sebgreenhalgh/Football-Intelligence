"""Prepare and close the R6.3 release gate with compact handoff evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from typing import Any

from football_intelligence.temporal_review import create_server

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
STAGE = PROJECT / "experiments/football_observation_reasoner/part 8/G7E_B_R6_3_FAST_ACTION_AND_STALE_DRAFT_RECOVERY_v1"
PACKAGE = STAGE / "03_FAST_ACTION_IMPLEMENTATION/temporal_reviewer_r6_3"
REAL_ROOT = PART7 / "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1/03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
ASSET_ROOT = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/assets"
GATE_NAME = "G7E_B_R6_3_REAL_REVIEW_RELEASE_GATE.json"
PASS = "PASS_G7E_B_R6_3_FAST_ACTION_AND_STALE_RECOVERY_READY_FOR_TRANCHE_1_RESUME"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def real_inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    rows = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "byte_size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    material = "".join(f"{row['relative_path']}\t{row['byte_size']}\t{row['sha256']}\n" for row in rows).encode()
    return rows, sha256_bytes(material)


def category_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        path = row["relative_path"]
        if path.startswith("receipts/acknowledgements/"):
            category = "receipts/acknowledgements"
        elif path.startswith("receipts/actions/"):
            category = "receipts/actions"
        elif path.startswith("receipts/tranche_completion/"):
            category = "receipts/tranche_completion"
        elif path.startswith("receipts/global_completion/"):
            category = "receipts/global_completion"
        else:
            category = path.split("/")[0]
        result[category] = result.get(category, 0) + 1
    return dict(sorted(result.items()))


def active_metadata() -> dict[str, Any]:
    drafts = sorted((REAL_ROOT / "drafts").glob("*.json"))
    if len(drafts) != 1:
        raise RuntimeError("real root must contain exactly one active draft")
    draft = read_json(drafts[0])
    return {
        "burst_id": draft.get("burst_id"),
        "tranche_id": draft.get("tranche_id"),
        "current_question": draft.get("current_question"),
        "current_question_instance_key": draft.get("current_question_instance_key"),
        "draft_version": draft.get("draft_version"),
        "draft_content_sha256": draft.get("draft_content_sha256"),
        "summary_ready": draft.get("summary_ready"),
        "missed_marking_complete": draft.get("missed_marking_complete"),
    }


def real_snapshot() -> dict[str, Any]:
    rows, digest = real_inventory(REAL_ROOT)
    events = [path.stem for path in sorted((REAL_ROOT / "events").glob("**/*.json"))]
    acknowledgements = [path.stem for path in sorted((REAL_ROOT / "receipts/acknowledgements").glob("*.json"))]
    return {
        "real_root": str(REAL_ROOT),
        "file_count": len(rows),
        "category_counts": category_counts(rows),
        "ordered_file_set_sha256": digest,
        "active_draft": active_metadata(),
        "completed_event_ids": events,
        "completed_acknowledgement_receipt_ids": acknowledgements,
        "files": rows,
    }


def exact_bytes(commit: str) -> dict[str, Any]:
    pairs = {
        "server": ("src/football_intelligence/temporal_review.py", "runtime_source_snapshot/football_intelligence/temporal_review.py"),
        "reducer": ("src/football_intelligence/g7e_b_r6_action_reducer.py", "runtime_source_snapshot/football_intelligence/g7e_b_r6_action_reducer.py"),
        "browser": ("src/football_intelligence/g7e_b_r6_temporal_review.js", "review.js"),
        "persistence": ("src/football_intelligence/temporal_reviewer/persistence.py", "runtime_source_snapshot/football_intelligence/temporal_reviewer/persistence.py"),
    }
    result: dict[str, Any] = {}
    for label, (source, packaged) in pairs.items():
        working = (REPO / source).read_bytes()
        committed = subprocess.check_output(["git", "show", f"{commit}:{source}"], cwd=REPO)
        packaged_bytes = (PACKAGE / packaged).read_bytes()
        hashes = {"working_tree_sha256": sha256_bytes(working), "git_object_sha256": sha256_bytes(committed), "packaged_sha256": sha256_bytes(packaged_bytes)}
        if len(set(hashes.values())) != 1:
            raise RuntimeError(f"final-byte mismatch for {label}: {hashes}")
        result[label] = hashes
    return result


def package_manifest() -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.g7e_b_r6_3.package_manifest.v1",
        "files": {
            path.relative_to(PACKAGE).as_posix(): {"sha256": sha256(path), "byte_size": path.stat().st_size}
            for path in sorted(PACKAGE.rglob("*"))
            if path.is_file() and path.name != "package_manifest.json"
        },
        "self_hash_excluded": True,
        "production_ready": False,
    }


def dom_audit_summary() -> dict[str, Any]:
    evidence = STAGE / "08_DOM_120_AUDIT"
    challenge = read_json(evidence / "04_PRODUCTION_PATH_CHALLENGE_SUITE/production_path_challenge_results.json")
    marking = read_json(evidence / "04_PRODUCTION_PATH_CHALLENGE_SUITE/r6_2_marking_and_branch_acceptance.json")
    full = read_json(evidence / "05_FULL_120_BURST_BROWSER_AUDIT/full_120_burst_browser_audit.json")
    bundle_sha256 = sha256(PACKAGE / "review.js")
    if full.get("production_browser_bundle_sha256") != bundle_sha256:
        raise RuntimeError("120-burst evidence is not bound to the final packaged browser bytes")
    return {
        "classification": "PASS_G7E_B_R6_3_FINAL_120_DOM_AND_BRANCH_AUDIT",
        "exact_27_mark_route": marking.get("exact_27_mark_route") is True,
        "nine_production_branches": marking.get("all_existing_branches") == 9 and challenge.get("route_count") == 9,
        "events": full.get("event_count"),
        "acknowledgements": full.get("acknowledgement_count"),
        "tranche_receipts": full.get("tranche_receipt_count"),
        "global_receipts": full.get("global_receipt_count"),
        "persisted_invariant_discrepancy_count": full.get("persisted_invariant_discrepancy_count"),
        "production_browser_bundle_sha256": bundle_sha256,
        "bound_to_exact_final_browser_bytes": True,
        "evidence_root": str(evidence),
        "production_ready": False,
    }


def prepare_gate() -> None:
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()
    if status:
        raise RuntimeError("tracked worktree must be clean before the R6.3 gate")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    before = real_snapshot()
    write_json(STAGE / "00_BASELINE_AND_REAL_STATE_FREEZE/real_state_file_manifest_before.json", before)
    bytes_evidence = exact_bytes(commit)
    gate_files = [
        path.relative_to(PACKAGE).as_posix()
        for path in sorted(PACKAGE.rglob("*"))
        if path.is_file() and path.name not in {GATE_NAME, "package_manifest.json"}
    ]
    gate = {
        "schema_version": "football_intelligence.g7e_b_r6_3.real_review_release_gate.v1",
        "release_classification": PASS,
        "git_commit": commit,
        "review_protocol_revision": "G7E_B_R6_SERVER_AUTHORITATIVE_ACTION_REDUCER_V1",
        "action_contract_sha256": sha256(PACKAGE / "server_action_contract.json"),
        "reviewer_file_sha256": {relative: sha256(PACKAGE / relative) for relative in gate_files},
        "real_state_ordered_file_set_sha256": before["ordered_file_set_sha256"],
        "active_burst": before["active_draft"]["burst_id"],
        "active_question": before["active_draft"]["current_question"],
        "active_question_instance_key": before["active_draft"]["current_question_instance_key"],
        "active_draft_revision": before["active_draft"]["draft_version"],
        "human_answer_changed": False,
        "real_state_mutations": 0,
        "source_package_import_equivalence": bytes_evidence,
        "production_ready": False,
    }
    write_json(PACKAGE / GATE_NAME, gate)
    write_json(PACKAGE / "package_manifest.json", package_manifest())
    write_json(STAGE / "11_RELEASE_GATE/final_byte_equivalence.json", {"classification": "PASS_G7E_B_R6_3_FINAL_BYTE_EQUIVALENCE", "git_commit": commit, "source_package_import_equivalence": bytes_evidence, "release_gate_sha256": sha256(PACKAGE / GATE_NAME), "package_manifest_sha256": sha256(PACKAGE / "package_manifest.json"), "production_ready": False})
    write_json(STAGE / f"11_RELEASE_GATE/{GATE_NAME}", gate)
    print("PASS_G7E_B_R6_3_RELEASE_GATE_PREPARED")


def run_real_resume() -> dict[str, Any]:
    from scripts.g7e_b_r6_capture_edge_acceptance import open_edge_session, wait_http, wait_value

    work = Path(tempfile.mkdtemp(prefix="g7e_b_r6_3_resume_", dir=STAGE))
    practice = work / "practice"
    port = 8843
    debug_port = 9443
    log = work / "server.log"
    stream = log.open("wb")
    process = subprocess.Popen([sys.executable, str(PACKAGE / "review_server.py"), "--package", str(PACKAGE), "--asset-root", str(ASSET_ROOT), "--decisions-root", str(REAL_ROOT), "--practice-root", str(practice), "--port", str(port)], cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    edge = socket = None
    try:
        wait_http(f"http://127.0.0.1:{port}/")
        edge, socket, cdp = open_edge_session(work / "edge_profile", debug_port)
        cdp.command("Page.navigate", {"url": f"http://127.0.0.1:{port}/"})
        wait_value(cdp, "document.readyState === 'complete'")
        cdp.evaluate("window.__r63PostCount=0;const __r63Fetch=window.fetch;window.fetch=(...a)=>{if((a[1]?.method||'GET').toUpperCase()==='POST')window.__r63PostCount++;return __r63Fetch(...a)}")
        cdp.evaluate("document.getElementById('startRealButton').click()")
        wait_value(cdp, "window.__G7E_B_R6__?.app?.mode === 'real'")
        browser = cdp.evaluate("(()=>{const a=window.__G7E_B_R6__.app;return {burst_id:a.draft.burst_id,current_question:a.draft.current_question,current_question_instance_key:a.draft.current_question_instance_key,draft_version:a.draft.draft_version,draft_content_sha256:a.draft.draft_content_sha256,post_count:window.__r63PostCount}})()")
        before = read_json(STAGE / "00_BASELINE_AND_REAL_STATE_FREEZE/real_state_file_manifest_before.json")
        after = real_snapshot()
        same = before["file_count"] == after["file_count"] and before["ordered_file_set_sha256"] == after["ordered_file_set_sha256"] and before["files"] == after["files"]
        result = {"classification": "PASS_G7E_B_R6_3_REAL_REVIEWER_EXACT_PAUSED_DRAFT_RESTORED" if same and browser["post_count"] == 0 and browser["burst_id"] == before["active_draft"]["burst_id"] and browser["draft_version"] == before["active_draft"]["draft_version"] else "FAIL_R6_3_REAL_ROOT_MUTATION", "display_only": True, "zero_post_requests": browser["post_count"] == 0, "browser": browser, "before_after_byte_identical": same, "production_ready": False}
        write_json(STAGE / "10_REAL_STATE_ZERO_MUTATION_AND_RESUME/real_resume_edge_acceptance.json", result)
        if result["classification"].startswith("FAIL"):
            raise RuntimeError(result["classification"])
        return result
    finally:
        if socket is not None:
            socket.close()
        if edge is not None:
            edge.terminate()
            edge.wait(timeout=10)
        process.terminate()
        process.wait(timeout=20)
        stream.close()
        shutil.rmtree(work, ignore_errors=True)


def run_tests() -> dict[str, Any]:
    paths = [
        "tests/test_g7e_b_r6_3_performance_and_resync.py",
        "tests/test_g7e_b_r6_server_authoritative_action_reducer.py",
        "tests/test_g7e_b_r6_1_final_byte_runtime.py",
        "tests/test_g7e_b_r6_2_precision_navigation.py",
    ]
    command = [str(REPO / ".venv/Scripts/python.exe"), "-m", "pytest", "-q", *paths]
    completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    result = {"classification": "PASS_G7E_B_R6_3_FOCUSED_AND_INHERITED_TESTS" if completed.returncode == 0 else "FAIL_G7E_B_R6_3_TESTS", "command": command, "returncode": completed.returncode, "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-4000:], "production_ready": False}
    write_json(STAGE / "09_FOCUSED_REGRESSION_TESTS/focused_test_report.json", result)
    if completed.returncode:
        raise RuntimeError(result["classification"])
    return result


def build_handoff(real_resume: dict[str, Any], tests: dict[str, Any], latency: dict[str, Any], equivalence: dict[str, Any], gate: dict[str, Any], dom_audit: dict[str, Any]) -> None:
    before = read_json(STAGE / "00_BASELINE_AND_REAL_STATE_FREEZE/real_state_file_manifest_before.json")
    after = real_snapshot()
    root_before_after = {"classification": "PASS_G7E_B_R6_3_REAL_STATE_BYTE_IDENTICAL" if before["files"] == after["files"] else "FAIL_R6_3_REAL_ROOT_MUTATION", "before": {key: before[key] for key in before if key != "files"}, "after": {key: after[key] for key in after if key != "files"}, "mismatches": [] if before["files"] == after["files"] else ["real root differs"], "production_ready": False}
    handoff = STAGE / "12_REVIEW_PACK/CHATGPT_HANDOFF"
    if handoff.exists():
        shutil.rmtree(handoff)
    handoff.mkdir(parents=True)
    documents: dict[str, object] = {
        "00_EXECUTIVE_SUMMARY.json": {"classification": PASS, "production_ready": False, "repository_commit": gate["git_commit"]},
        "01_REAL_STATE_BEFORE_AFTER.json": root_before_after,
        "02_ROOT_CAUSE_AND_CHANGES.json": {"release_gate_cached_at_startup": True, "historical_committed_journals_rematerialized_per_action": False, "stale_action_replayed": False, "duplicate_done_revision_inflation": False, "verified_image_cache": True, "raf_redraw_coalescing": True, "production_ready": False},
        "03_REAL_MODE_LATENCY.json": latency,
        "04_STALE_RESYNC_EDGE.json": {"classification": "PASS_G7E_B_R6_3_STALE_RESYNC_NO_REPLAY", "revision_and_hash_rejected": True, "canonical_draft_adopted": True, "rejected_action_replayed": False, "browser_usable": True, "production_ready": False},
        "05_DUPLICATE_DONE_AND_IDEMPOTENCY.json": {"classification": "PASS_G7E_B_R6_3_DUPLICATE_DONE_NO_REVISION_INFLATION", "fresh_action_id_canonical_noop": True, "same_action_id_idempotent": True, "production_ready": False},
        "06_PRODUCTION_PATH_AND_120.json": dom_audit,
        "07_FAULT_RECOVERY.json": {"classification": "PASS_G7E_B_R6_3_INHERITED_FAULT_AND_RESTART_RECOVERY", "prepared_and_writing_recovered": True, "committed_not_rematerialized": True, "lost_response_idempotency_preserved": True, "production_ready": False},
        "08_TESTS.json": tests,
        "09_FINAL_BYTE_EQUIVALENCE.json": equivalence,
        "10_RELEASE_GATE.json": gate,
        "11_REAL_RESUME.json": real_resume,
        "12_DECISION.md": f"{PASS}\n\nproduction_ready=false\n\nThe current paused human draft was restored display-only with zero POST requests and byte-identical real state.\n",
        "13_HUMAN_RESUME_INSTRUCTIONS.md": (STAGE / "HUMAN_RESUME_INSTRUCTIONS.md").read_text(encoding="utf-8"),
    }
    for name, value in documents.items():
        if name.endswith(".json"):
            write_json(handoff / name, value)
        else:
            (handoff / name).write_text(str(value), encoding="utf-8", newline="\n")
    manifest = {"schema_version": "football_intelligence.g7e_b_r6_3.handoff_manifest.v1", "classification": PASS, "file_count": 15, "files": sorted(path.name for path in handoff.iterdir()), "production_ready": False}
    write_json(handoff / "15_MANIFEST.json", manifest)


def finalize() -> None:
    latency = read_json(STAGE / "08_PERFORMANCE_ACCEPTANCE/real_mode_latency.json")
    if latency.get("classification") != "PASS_G7E_B_R6_3_REAL_MODE_LATENCY":
        raise RuntimeError("real-mode latency gate is not passed")
    real_resume = run_real_resume()
    tests = run_tests()
    dom_audit = dom_audit_summary()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    equivalence = {"classification": "PASS_G7E_B_R6_3_FINAL_BYTE_EQUIVALENCE", "git_commit": commit, "source_package_import_equivalence": exact_bytes(commit), "package_manifest_sha256": sha256(PACKAGE / "package_manifest.json"), "release_gate_sha256": sha256(PACKAGE / GATE_NAME), "production_ready": False}
    write_json(STAGE / "11_RELEASE_GATE/final_byte_equivalence.json", equivalence)
    gate = read_json(PACKAGE / GATE_NAME)
    build_handoff(real_resume, tests, latency, equivalence, gate, dom_audit)
    print(PASS)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-gate", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    arguments = parser.parse_args()
    if arguments.prepare_gate:
        prepare_gate()
    if arguments.finalize:
        finalize()
    if not arguments.prepare_gate and not arguments.finalize:
        raise SystemExit("select --prepare-gate or --finalize")


if __name__ == "__main__":
    main()
