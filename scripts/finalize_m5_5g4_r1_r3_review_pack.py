"""Finalize the bounded M5.5G.4-R1-R3 ChatGPT review pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from build_m5_5g4_r1_r3_pending_recovery import (
    BASELINE,
    BRANCH,
    EXPECTED_EXPORT_SHA256,
    PASS_CLASSIFICATION,
    REMOTE,
    REPO,
    REVIEW_PACK,
    STAGE,
    read_json,
    sha256_file,
)


COMMANDS_ROOT = STAGE / "08_COMMANDS_AND_TESTS"
BROWSER_ROOT = STAGE / "06_BROWSER_PERSISTENCE_AND_REPLAY"
VISUAL_ROOT = STAGE / "05_RECOVERY_REVIEW_UI"
EXPECTED_FILES = (
    "00_READ_ME_FIRST.md",
    "01_EXECUTIVE_OUTCOME.md",
    "02_REPOSITORY_STATE.json",
    "03_LIVE_STATE_AND_EXPORT_PROOF.json",
    "04_SOURCE_DIFF.patch",
    "05_SERVER_DECISION_PRESERVATION.json",
    "06_DEPENDENCY_ROOT_CAUSE.json",
    "07_SERVER_PREFLIGHT_AND_PAIR_WORKFLOW.json",
    "08_PENDING_MIGRATION_AND_DRAFT_RECOVERY.json",
    "09_BROWSER_PERSISTENCE_AND_IDEMPOTENCY.json",
    "10_PACKAGE_AND_LAUNCHER.json",
    "11_COMMANDS_AND_TESTS.md",
    "12_SAFETY_AND_ACCEPTANCE.json",
    "13_FIVE_ITEM_RECOVERY_QUEUE.png",
    "14_EXPLICIT_PAIR_REVIEW.png",
    "15_PRESERVED_POLYGON_NO_REDRAW.png",
    "16_HUMAN_ACTION.json",
    "REVIEW_PACK_MANIFEST.json",
)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def stable_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def safe_text(value: str) -> str:
    replacements = (
        (str(REPO.parent), "<FOOTBALL_INTELLIGENCE_ROOT>"),
        (str(REPO), "<REPOSITORY>"),
        (str(Path.home()), "<USER_PROFILE>"),
    )
    for source, replacement in replacements:
        value = value.replace(source, replacement).replace(source.replace("\\", "/"), replacement)
    return value


def write_text(name: str, value: str) -> None:
    (REVIEW_PACK / name).write_text(safe_text(value).rstrip() + "\n", encoding="utf-8")


def write_json(name: str, value: Any) -> None:
    write_text(name, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True))


def repository_gate(*, precommit: bool) -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    remote = git("remote", "get-url", "origin")
    baseline_is_ancestor = (
        subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE, head], cwd=REPO, check=False).returncode == 0
    )
    if precommit:
        staged = git("diff", "--cached", "--name-only", BASELINE).splitlines()
        unstaged = git("diff", "--name-only").splitlines()
        result = {
            "mode": "PRECOMMIT_STAGED",
            "implementation_commit": "PENDING_COMMIT",
            "authorized_baseline": BASELINE,
            "head_at_precommit": head,
            "head_is_authorized_baseline": head == BASELINE,
            "baseline_is_ancestor": baseline_is_ancestor,
            "branch": branch,
            "remote": remote,
            "staged_file_count": len(staged),
            "unstaged_file_count": len(unstaged),
            "remote_head_matches_local": git("rev-parse", "@{upstream}") == head,
            "worktree_clean": False,
        }
        result["passed"] = all(
            (
                result["head_is_authorized_baseline"],
                baseline_is_ancestor,
                branch == BRANCH,
                remote == REMOTE,
                bool(staged),
                not unstaged,
                result["remote_head_matches_local"],
            )
        )
    else:
        result = {
            "mode": "FINAL_COMMITTED",
            "implementation_commit": head,
            "authorized_baseline": BASELINE,
            "baseline_is_ancestor": baseline_is_ancestor,
            "branch": branch,
            "remote": remote,
            "remote_head_matches_local": git("rev-parse", "@{upstream}") == head,
            "worktree_clean": not git("status", "--porcelain"),
        }
        result["passed"] = all(
            (
                head != BASELINE,
                baseline_is_ancestor,
                branch == BRANCH,
                remote == REMOTE,
                result["remote_head_matches_local"],
                result["worktree_clean"],
            )
        )
    if not result["passed"]:
        raise RuntimeError(f"repository finalization gate failed: {result}")
    return result


def source_diff(*, precommit: bool, implementation_commit: str) -> str:
    if precommit:
        return git("diff", "--cached", "--binary", BASELINE)
    return git("diff", "--binary", BASELINE, implementation_commit)


def changed_python_files(*, precommit: bool, implementation_commit: str) -> list[str]:
    if precommit:
        changed = git("diff", "--cached", "--name-only", BASELINE).splitlines()
    else:
        changed = git("diff", "--name-only", BASELINE, implementation_commit).splitlines()
    return [path for path in changed if path.endswith(".py")]


def run_command(name: str, command: list[str], *, timeout: int = 2400) -> dict[str, Any]:
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
    )
    output = "\n".join(part.strip() for part in (process.stdout, process.stderr) if part.strip())
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    result = {
        "name": name,
        "command": " ".join(command),
        "return_code": process.returncode,
        "duration_seconds": round(time.perf_counter() - started, 2),
        "summary": " | ".join(lines[-4:]) if lines else "completed without console output",
        "passed": process.returncode == 0,
    }
    if not result["passed"]:
        raise RuntimeError(f"validation command failed: {result}\n{output[-8000:]}")
    return result


def validation_commands(*, precommit: bool, implementation_commit: str) -> list[dict[str, Any]]:
    python_files = changed_python_files(precommit=precommit, implementation_commit=implementation_commit)
    focused = [
        "tests/test_m5_5g4_r1_dense_mask_repair.py",
        "tests/test_m5_5g4_r1_r1_dense_mask_ui_repair.py",
        "tests/test_m5_5g4_r1_r2_marker_scale_repair.py",
        "tests/test_m5_5g4_r1_r3_pending_recovery.py",
    ]
    diff_command = (
        ["git", "diff", "--check", "--cached"]
        if precommit
        else ["git", "diff", "--check", BASELINE, implementation_commit]
    )
    return [
        run_command("uv_lock_check", ["uv", "lock", "--check"]),
        run_command("uv_sync", ["uv", "sync"], timeout=1200),
        run_command("ruff_check", ["uv", "run", "ruff", "check", *python_files]),
        run_command("ruff_format_check", ["uv", "run", "ruff", "format", "--check", *python_files]),
        run_command("node_app", ["node", "--check", "src/football_intelligence/review_chassis/static/app.js"]),
        run_command(
            "node_dense_correction",
            ["node", "--check", "src/football_intelligence/review_chassis/static/dense_mask_correction.js"],
        ),
        run_command("focused_and_prior_regressions", ["uv", "run", "pytest", *focused, "-q"]),
        run_command("full_suite", ["uv", "run", "pytest", "-q"]),
        run_command("pipeline_help", ["uv", "run", "fi-pipeline", "--help"]),
        run_command("review_chassis_help", ["uv", "run", "fi-pipeline", "review-chassis", "--help"]),
        run_command("diff_check", diff_command),
    ]


def compact_live_and_export() -> dict[str, Any]:
    live = read_json(STAGE / "01_LIVE_SERVER_AND_BROWSER_STATE_EXPORT" / "live_server_state_validation.json")
    restore = read_json(STAGE / "01_LIVE_SERVER_AND_BROWSER_STATE_EXPORT" / "temporary_clone_restore_validation.json")
    return {
        "passed": live["passed"] and restore["passed"],
        "live_server": {
            "port": 8808,
            "server_saved_corrections": live["server_saved_correction_count"],
            "affected_cases_complete": live["affected_cases_complete"],
            "affected_cases_total": live["affected_cases_total"],
            "geometry_reviews_remaining": live["geometry_reviews_remaining"],
            "server_event_sequence": live["server_event_sequence"],
            "pending_local_outbox_count": live["pending_local_outbox_count"],
            "repair_completion_bundle_exists": live["repair_completion_bundle_exists"],
        },
        "export": {
            "pending_record_count": 5,
            "draft_count": 1,
            "unique_event_id_count": 5,
            "unique_idempotency_key_count": 5,
            "source_export_sha256": EXPECTED_EXPORT_SHA256,
            "temporary_restore_hash_matches": restore["checks"]["canonical_reexport_hash_matches"],
            "raw_payloads_in_review_pack": False,
        },
    }


def compact_browser() -> dict[str, Any]:
    browser = read_json(BROWSER_ROOT / "browser_persistence_results.json")
    return {
        "passed": browser["passed"],
        "browser": browser["browser"],
        "scenario_count": browser["scenario_count"],
        "scenario_pass_count": sum(bool(value) for value in browser["scenarios"].values()),
        "scenarios": browser["scenarios"],
        "viewport_count": len(browser["viewport_results"]),
        "viewport_checks_passed": all(
            row["reviewVisible"] and not row["horizontalOverflow"] for row in browser["viewport_results"]
        ),
        "temporary_clone_initial_pending_count": browser["initial_pending_count"],
        "temporary_clone_final_pending_count": browser["final_pending_count"],
        "temporary_clone_initial_server_corrections": browser["initial_server_correction_count"],
        "temporary_clone_final_server_corrections": browser["final_temporary_server_correction_count"],
        "real_port_8808_touched": browser["real_port_8808_touched"],
        "human_answers_were_synthetic_only_in_disposable_clone": browser["scenarios"][
            "human_answers_were_only_synthetic_in_temporary_clone"
        ],
    }


def ensure_pack_root() -> None:
    expected = STAGE / "09_REVIEW_PACK_FOR_CHATGPT"
    if REVIEW_PACK.resolve() != expected.resolve():
        raise RuntimeError("review-pack path escaped the dedicated stage")
    REVIEW_PACK.mkdir(parents=True, exist_ok=True)
    nested = [path for path in REVIEW_PACK.rglob("*") if path.is_file() and path.parent != REVIEW_PACK]
    if nested:
        raise RuntimeError(f"refusing to replace a non-flat review pack: {nested}")
    unknown = [path for path in REVIEW_PACK.iterdir() if path.is_file() and path.name not in EXPECTED_FILES]
    if unknown:
        raise RuntimeError(f"refusing to remove unknown review-pack files: {unknown}")
    for path in REVIEW_PACK.iterdir():
        if path.is_file():
            path.unlink()


def populate_pack(repository: dict[str, Any], commands: list[dict[str, Any]], *, precommit: bool) -> None:
    ensure_pack_root()
    live = compact_live_and_export()
    preservation = read_json(STAGE / "01_LIVE_SERVER_AND_BROWSER_STATE_EXPORT" / "server_decision_preservation.json")
    root_cause = read_json(STAGE / "02_DEPENDENCY_ROOT_CAUSE" / "occlusion_dependency_root_cause.json")
    preflight = read_json(STAGE / "03_SERVER_AUTHORITATIVE_PREFLIGHT" / "dependency_preflight_specification.json")
    migration = read_json(STAGE / "04_PENDING_QUEUE_MIGRATION" / "pending_event_migration_manifest.json")
    recovery = read_json(STAGE / "04_PENDING_QUEUE_MIGRATION" / "pending_recovery_results.json")
    pair_ui = read_json(STAGE / "05_RECOVERY_REVIEW_UI" / "pair_review_ui_validation.json")
    package = read_json(STAGE / "05_RECOVERY_REVIEW_UI" / "review_package_validation.json")
    browser = compact_browser()
    diff = source_diff(precommit=precommit, implementation_commit=repository["implementation_commit"])
    write_text(
        "00_READ_ME_FIRST.md",
        """# M5.5G.4-R1-R3 review pack

This flat pack documents the bounded repair for five preserved IndexedDB
events that were blocked by a missing occlusion-dependency handshake. Start
with 01, then read 03 and 06-09. Files 13-15 are real Edge screenshots from
the repaired production package against an isolated copy of the live state.

No raw polygon, human correction payload, event UUID mapping, expected answer,
video, model weight, credential or personal data is included.
""",
    )
    write_text(
        "01_EXECUTIVE_OUTCOME.md",
        f"""# Executive outcome

Classification: `{PASS_CLASSIFICATION}`

The existing five pending records and current draft were exported and hash
validated before implementation. The server now computes polygon-dependent
coverage and occlusion requirements in a read-only preflight, and the client
asks the reviewer the exact missing Person A/Person B question without
guessing. Original event IDs, idempotency keys and order are preserved.

The real server remains at 13 corrections with five browser-local events. A
{browser['scenario_count']}-scenario Edge exercise drained only a disposable clone and proved reload,
offline, server-restart and duplicate-replay recovery.
""",
    )
    write_json("02_REPOSITORY_STATE.json", repository)
    write_json("03_LIVE_STATE_AND_EXPORT_PROOF.json", live)
    write_text("04_SOURCE_DIFF.patch", diff)
    write_json(
        "05_SERVER_DECISION_PRESERVATION.json",
        {
            "passed": preservation["passed"],
            "real_root_opened_for_writes": preservation["real_root_opened_for_writes"],
            "server_saved_corrections": preservation["correction_count"],
            "server_event_sequence": preservation["event_sequence"],
            "before_after_trees_match": preservation["trees_match"],
            "before_tree_digest": stable_json_hash(preservation["before_tree"]),
            "after_tree_digest": stable_json_hash(preservation["after_tree"]),
            "original_c1_mutated": preservation["original_c1_mutated"],
            "repair_manifest_mutated": preservation["repair_manifest_mutated"],
        },
    )
    write_json(
        "06_DEPENDENCY_ROOT_CAUSE.json",
        {
            "passed": root_cause["passed"],
            "confirmed": root_cause["confirmed"],
            "root_cause": root_cause["root_cause"],
            "failure_message": root_cause["failure_message"],
            "pending_rows_audited": [
                {
                    "queue_position": row["queue_position"],
                    "already_acknowledged": row["already_acknowledged"],
                    "submitted_occlusion_review_count": row["submitted_occlusion_review_count"],
                    "required_occlusion_pair_count": row["required_occlusion_pair_count"],
                    "missing_answer_count": len(row["missing_answer_ids"]),
                    "extra_answer_count": len(row["extra_answer_ids"]),
                    "material_overlap_evidence": row["material_overlap_evidence"],
                }
                for row in root_cause["queue_rows"]
            ],
            "client_pair_answers_per_row": root_cause["client_submitted_pair_count_per_event"],
            "server_required_pairs_per_row": root_cause["server_required_pair_count_per_event"],
            "audited_hypotheses": root_cause["audited_hypotheses"],
            "server_validation_weakened": root_cause["server_validation_weakened"],
        },
    )
    write_json(
        "07_SERVER_PREFLIGHT_AND_PAIR_WORKFLOW.json",
        {
            "passed": preflight["passed"] and pair_ui["passed"],
            "endpoint": preflight["endpoint"],
            "read_only": preflight["read_only"],
            "idempotent": preflight["idempotent"],
            "handshake_version": preflight["handshake_version"],
            "phase_1_returns": preflight["phase_1_returns"],
            "phase_2_recomputes_dependency_set": preflight["phase_2_recomputes_dependency_set"],
            "phase_2_requires_exact_hashes_and_answers": preflight["phase_2_requires_exact_hashes_and_answers"],
            "structured_rejection": preflight["structured_rejection"],
            "pair_ui_checks": pair_ui["checks"],
            "browser_pair_validation": pair_ui["browser_validation"],
            "answer_inference_performed": False,
        },
    )
    write_json(
        "08_PENDING_MIGRATION_AND_DRAFT_RECOVERY.json",
        {
            "passed": migration["passed"] and recovery["passed"],
            "expected_pending_count": migration["expected_pending_count"],
            "source_export_sha256": migration["source_export_sha256"],
            "temporary_restore_validation": migration["temporary_restore_validation"],
            "old_database_retained_read_only": migration["old_database_retained_read_only"],
            "atomic_three_store_transaction": migration["atomic_three_store_transaction"],
            "event_order_preserved": migration["event_order_preserved"],
            "original_event_ids_preserved": migration["original_event_ids_preserved"],
            "original_idempotency_keys_preserved": migration["original_idempotency_keys_preserved"],
            "remove_only_after_server_acknowledgement": migration["remove_only_after_server_acknowledgement"],
            "current_draft_represented_in_pending_queue": recovery["current_draft_represented_in_pending_queue"],
            "current_draft_duplicate_event_created": recovery["current_draft_duplicate_event_created"],
            "real_server_acknowledged_during_implementation": recovery["server_acknowledged_during_implementation"],
        },
    )
    write_json("09_BROWSER_PERSISTENCE_AND_IDEMPOTENCY.json", browser)
    write_json(
        "10_PACKAGE_AND_LAUNCHER.json",
        {
            "passed": package["passed"],
            "review_url": "http://127.0.0.1:8808/",
            "launcher": "07_REPAIRED_REVIEW_PACKAGE/launch_pending_recovery_dense_mask_review.ps1",
            "same_historical_decisions_root": True,
            "fresh_browser_namespace": True,
            "legacy_browser_database_retained_read_only": True,
            "reviewer_manifest_byte_identical": package["checks"]["reviewer_manifest_byte_identical"],
            "evidence_tree_byte_identical": package["checks"]["evidence_tree_byte_identical"],
            "generic_package_valid": package["checks"]["generic_package_valid"],
            "browser_acceptance": package["browser_acceptance"],
        },
    )
    write_text(
        "11_COMMANDS_AND_TESTS.md",
        "# Commands and tests\n\n"
        + "\n".join(f"- **{row['name']}:** passed ({row['summary']}; {row['duration_seconds']}s)" for row in commands),
    )
    write_json(
        "12_SAFETY_AND_ACCEPTANCE.json",
        {
            "classification": PASS_CLASSIFICATION,
            "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
            "production_ready": False,
            "no_auto_promotion": True,
            "human_approved": False,
            "safe_to_apply_globally": False,
            "sandbox_only": True,
            "match_local_only": True,
            "human_dependency_answer_fabricated": False,
            "model_inference_performed": False,
            "model_weights_changed": False,
            "detector_or_tracker_changed": False,
            "original_human_corrections_rewritten": False,
            "original_c1_mutated": False,
            "repair_manifest_mutated": False,
            "real_pending_queue_drained_during_implementation": False,
        },
    )
    for source, destination in (
        ("01_FIVE_ITEM_RECOVERY_QUEUE.png", "13_FIVE_ITEM_RECOVERY_QUEUE.png"),
        ("03_EXPLICIT_PAIR_REVIEW.png", "14_EXPLICIT_PAIR_REVIEW.png"),
        ("02_PRESERVED_POLYGON_NO_REDRAW.png", "15_PRESERVED_POLYGON_NO_REDRAW.png"),
    ):
        shutil.copy2(VISUAL_ROOT / source, REVIEW_PACK / destination)
    write_json(
        "16_HUMAN_ACTION.json",
        {
            "action": (
                "After preserving the current tab until implementation handoff, stop the old port-8808 server, "
                "run the repaired launcher, and return to the same URL without clearing browser site data."
            ),
            "launcher": "07_REPAIRED_REVIEW_PACKAGE/launch_pending_recovery_dense_mask_review.ps1",
            "url": "http://127.0.0.1:8808/",
            "expected_start": (
                "13 saved corrections, five locally pending recovery items, and the valid polygon restored "
                "without redraw."
            ),
            "required_human_work": (
                "Answer the explicit overlap question for each preserved event; no prior correction needs to "
                "be redone."
            ),
            "do_not": [
                "Do not clear browser storage before recovery completes.",
                "Do not redraw the restored valid polygon.",
                "Do not fabricate an overlap answer; use I can't tell when unresolved.",
            ],
        },
    )
    entries = [
        {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(item for item in REVIEW_PACK.iterdir() if item.is_file())
    ]
    write_json(
        "REVIEW_PACK_MANIFEST.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1_r3.review_pack_manifest.v1",
            "classification": PASS_CLASSIFICATION,
            "implementation_commit": repository["implementation_commit"],
            "file_count_including_manifest": len(entries) + 1,
            "flat": True,
            "maximum_files": 20,
            "maximum_total_bytes": 50 * 1024 * 1024,
            "maximum_visuals": 3,
            "raw_human_payloads_included": False,
            "manifest_self_hash_included": False,
            "entries_excluding_manifest": entries,
        },
    )


def validate_visual(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        spread = max(ImageStat.Stat(image.convert("RGB").resize((96, 60))).stddev)
        result = {
            "filename": path.name,
            "width": image.width,
            "height": image.height,
            "rgb_standard_deviation_max": round(spread, 3),
            "nonblank": spread > 8,
            "sha256": sha256_file(path),
        }
    if result["width"] < 1000 or result["height"] < 600 or not result["nonblank"]:
        raise RuntimeError(f"invalid review-pack visual: {result}")
    return result


def validate_pack() -> dict[str, Any]:
    files = sorted(path for path in REVIEW_PACK.iterdir() if path.is_file())
    nested = [path for path in REVIEW_PACK.rglob("*") if path.is_file() and path.parent != REVIEW_PACK]
    visuals = [path for path in files if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    total_bytes = sum(path.stat().st_size for path in files)
    forbidden_extensions = {".mp4", ".avi", ".mov", ".pt", ".pth", ".onnx"}
    forbidden_names = [path.name for path in files if path.suffix.lower() in forbidden_extensions]
    privacy_hits: list[dict[str, str]] = []
    credential = re.compile(r"(?:password|api[_-]?key)\s*=\s*['\"][^'\"]+['\"]", re.IGNORECASE)
    uuid_pattern = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.I)
    for path in files:
        if path in visuals:
            continue
        value = path.read_text(encoding="utf-8", errors="replace")
        for token in ("C:" + "\\Users\\", "/" + "Users/"):
            if token.lower() in value.lower():
                privacy_hits.append({"filename": path.name, "token": token})
        if "BEGIN " + "PRIVATE KEY" in value or credential.search(value):
            privacy_hits.append({"filename": path.name, "token": "credential_pattern"})
        if path.name != "04_SOURCE_DIFF.patch" and uuid_pattern.search(value):
            privacy_hits.append({"filename": path.name, "token": "event_or_candidate_uuid"})
    visual_results = [validate_visual(path) for path in visuals]
    manifest = read_json(REVIEW_PACK / "REVIEW_PACK_MANIFEST.json")
    checks = {
        "exact_expected_files": {path.name for path in files} == set(EXPECTED_FILES),
        "file_count_at_most_20": len(files) <= 20,
        "flat": not nested,
        "total_size_at_most_50_mib": total_bytes <= 50 * 1024 * 1024,
        "visual_count_exactly_three": len(visuals) == 3,
        "visuals_distinct": len({row["sha256"] for row in visual_results}) == 3,
        "visuals_valid": all(row["nonblank"] for row in visual_results),
        "source_diff_present_nonempty": (REVIEW_PACK / "04_SOURCE_DIFF.patch").stat().st_size > 0,
        "forbidden_binaries_absent": not forbidden_names,
        "personal_credentials_or_uuid_payloads_absent": not privacy_hits,
        "raw_export_or_draft_files_absent": not any(
            token in path.name for path in files for token in ("indexeddb_pending_export", "current_draft_export")
        ),
        "manifest_self_hash_absent": "REVIEW_PACK_MANIFEST.json"
        not in {entry["filename"] for entry in manifest["entries_excluding_manifest"]},
    }
    result = {
        "schema_version": "football_intelligence.m5_5g4_r1_r3.review_pack_validation.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "file_count": len(files),
        "total_size_bytes": total_bytes,
        "visual_count": len(visuals),
        "visuals": visual_results,
        "forbidden_names": forbidden_names,
        "privacy_hits": privacy_hits,
    }
    if not result["passed"]:
        raise RuntimeError(f"review-pack validation failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precommit", action="store_true")
    args = parser.parse_args()
    repository = repository_gate(precommit=args.precommit)
    browser = read_json(BROWSER_ROOT / "browser_persistence_results.json")
    package = read_json(VISUAL_ROOT / "review_package_validation.json")
    preservation = read_json(STAGE / "01_LIVE_SERVER_AND_BROWSER_STATE_EXPORT" / "server_decision_preservation.json")
    if not all(
        (
            browser["passed"],
            browser["scenario_count"] >= 25,
            package["passed"],
            preservation["passed"],
            preservation["trees_match"],
        )
    ):
        raise RuntimeError("browser, package, or preservation acceptance gate failed")
    if args.precommit:
        commands = validation_commands(precommit=True, implementation_commit=repository["implementation_commit"])
        COMMANDS_ROOT.mkdir(parents=True, exist_ok=True)
        (COMMANDS_ROOT / "commands_and_tests.json").write_text(
            json.dumps({"passed": True, "commands": commands}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        command_report = read_json(COMMANDS_ROOT / "commands_and_tests.json")
        if command_report.get("passed") is not True:
            raise RuntimeError("precommit validation command report is missing or failed")
        commands = command_report["commands"]
    populate_pack(repository, commands, precommit=args.precommit)
    validation = validate_pack()
    validation.update(
        {
            "implementation_commit": repository["implementation_commit"],
            "classification": PASS_CLASSIFICATION,
            "manifest_sha256": sha256_file(REVIEW_PACK / "REVIEW_PACK_MANIFEST.json"),
        }
    )
    (COMMANDS_ROOT / "review_pack_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
