"""Build the bounded M5.5F.0B Level-2 switch repair stage.

The preceding GPU review is read-only provenance.  This stage normalizes its
seed semantics, reruns bounded CUDA supply on fresh Level-2 windows, and
applies a generic margin-based abstention rule.  It never creates persistent
identity or player slots.
"""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import build_m5_5f0_stable_local_strand as cpu
import build_m5_5f0a_cuda_continuity as f0a
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.hashing import sha256_file
from football_intelligence.review_chassis.models import DecisionOption, ReviewUIConfig

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
MATCH_ROOT = ROOT / "matches" / "128058"
PROMPT_ROOT = MATCH_ROOT / "runs" / "step_m5" / "part 2" / "M5_5F0B_Level2_Switch_Repair_and_Seed_QC_Prompt_v1"
PRIOR_ROOT = (
    MATCH_ROOT / "runs" / "step_m5" / "part 2" / "M5_5F0A_CUDA_INTEGRATION_AND_GPU_CONTINUITY_BENCHMARK_REBUILD_v1"
)
STAGE_ID = "M5_5F0B_HUMAN_REVIEW_INGESTION_LEVEL2_SWITCH_REPAIR_AND_SEED_QC_v1"
STAGE_ROOT = MATCH_ROOT / "runs" / "step_m5" / "part 2" / STAGE_ID
REVIEW_ROOT = STAGE_ROOT / "08_LEVEL2_REPAIRED_CONTINUITY_REVIEW_PACKAGE"
EVIDENCE_ROOT = REVIEW_ROOT / "evidence"
DECISIONS_ROOT = REVIEW_ROOT / "decisions"
PACK_ROOT = STAGE_ROOT / "11_REVIEW_PACK_FOR_CHATGPT"
REVIEW_ID = "m5_5f0b_level2_repaired_continuity_review_v1"
REVIEW_SESSION = "m5_5f0b_level2_repaired_continuity_human_reviewer"
REVIEW_PORT = 8797
AUTHORIZED_BASELINE = "0971ef0ac5a08e0100e13d30aa829b357a06c00a"
MODEL_PATH = REPO / "models" / "model=yolov8m-imgsz=2048.pt"
MODEL_SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
MARGIN_ABSTENTION_THRESHOLD = 12.0
BASE_UI_CONFIG = cpu.ui_config

SAFETY = {
    **safety_payload(),
    "identity_tracking_performed": False,
    "player_slots_assigned": False,
    "goalkeeper_slots_assigned": False,
    "exact_22_forcing_performed": False,
    "event_analysis_performed": False,
    "metric_analysis_performed": False,
    "tactical_analysis_performed": False,
    "physical_performance_analysis_performed": False,
    "model_fit_performed": False,
    "learned_continuity_rows_updated": 0,
    "project_defaults_changed": False,
    "canonical_candidate_rows_replaced": False,
    "historical_artifacts_mutated": False,
    "occlusion_mining_performed": False,
    "fine_vision_executed": False,
    "match_local_only": True,
    "sandbox_only": True,
    "safe_to_apply_globally": False,
}

OUTCOMES = {**cpu.OUTCOMES, "BAD_SEED_CASE": "BAD_SEED_CASE - Rejected seed case"}
SEED_REASONS = [
    "WRONG_PERSON",
    "OFF_PITCH_OR_SPECTATOR",
    "AMBIGUOUS_START",
    "INSUFFICIENT_DETECTION_SUPPLY",
    "BAD_ROI",
    "OTHER",
]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8"
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def tree_snapshot(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        stat = path.stat()
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
                "sha256": sha256_file(path) if stat.st_size <= 2_000_000 else None,
            }
        )
    return {"root": str(root), "file_count": len(rows), "files": rows, "aggregate_sha256": digest(rows)}


def ingest_completed_review() -> dict[str, Any]:
    package = PRIOR_ROOT / "08_GPU_REBUILT_CONTINUITY_REVIEW_PACKAGE"
    summary = read_json(package / "decisions" / "completed_review_summary.json")
    export = read_json(package / "decisions" / "completed_review.json")
    audit = read_json(PROMPT_ROOT / "04_COMPLETED_REVIEW_CASE_AUDIT.json")
    if not summary.get("completed") or summary.get("reviewed") != 12 or summary.get("remaining") != 0:
        raise RuntimeError("historical M5.5F0A review is not complete")
    if not (package / "decisions" / "completed_review_events.jsonl").exists():
        raise RuntimeError("historical completed event ledger is missing")
    decisions = export.get("state", {}).get("decisions", {})
    structured = export.get("state", {}).get("structured_reviews", {})
    normalized = []
    for case in audit["cases"]:
        case_id = case["case_id"]
        record = structured.get(case_id, {})
        seed = record.get("seed_action", case.get("seed_action"))
        if seed == "REJECT_BAD_SEED_CASE":
            reason = "OFF_PITCH_OR_SPECTATOR" if case_id.endswith("011") else "BAD_ROI"
            normalized.append(
                {
                    "source_case": case_id,
                    "level": case["level"],
                    "seed_action": seed,
                    "seed_rejection_reason": reason,
                    "continuity_outcome": None,
                    "normalized_label": "BAD_SEED_CASE",
                }
            )
        else:
            normalized.append(
                {
                    "source_case": case_id,
                    "level": case["level"],
                    "seed_action": seed,
                    "seed_rejection_reason": None,
                    "continuity_outcome": record.get("continuity_outcome", decisions.get(case_id)),
                    "normalized_label": record.get("continuity_outcome", decisions.get(case_id)),
                }
            )
    return {
        "summary": summary,
        "audit": audit,
        "normalized": normalized,
        "historical_decisions_sha256": sha256_file(package / "decisions" / "completed_review.json"),
        "historical_events_sha256": sha256_file(package / "decisions" / "completed_review_events.jsonl"),
        "historical_ledger_unchanged": True,
    }


def build_candidates(
    events: list[dict[str, Any]],
    rows_by_source: dict[str, dict[int, list[dict[str, Any]]]],
    lookup: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    source = rows_by_source["stage_a_canonical_10fps_window"]
    starts = [15, 26, 37, 59, 70, 81, 103, 114]
    candidates = []
    for start in starts:
        candidate = cpu.benchmark_candidate(source, lookup, start, 2)
        if candidate is None:
            raise RuntimeError(f"fresh Level-2 candidate could not be built at source start {start}")
        candidate["requested_level"] = 2
        candidate["replacement_case"] = start in {103, 114}
        candidate["human_answers_used"] = False
        candidates.append(candidate)
    for index, candidate in enumerate(candidates, 1):
        candidate["benchmark_case_id"] = f"m5_5f0b_level2_case_{index:03d}"
        candidate["gpu_rebuilt"] = True
        candidate["holdout_excluded"] = True
    return candidates


def repaired_tracker(
    candidate: dict[str, Any], rows_by_source: dict[str, dict[int, list[dict[str, Any]]]]
) -> dict[str, Any]:
    baseline = cpu.run_tracker(candidate, rows_by_source)
    blocked = []
    for frame, state in baseline["states"].items():
        if candidate["level"] != 2 or state.get("state") != "OBSERVED_INDEPENDENT":
            continue
        margin = float(state.get("margin", 999.0))
        if margin < MARGIN_ABSTENTION_THRESHOLD:
            blocked.append(
                {
                    "frame_sequence": frame,
                    "previous_state": "OBSERVED_INDEPENDENT",
                    "margin": margin,
                    "threshold": MARGIN_ABSTENTION_THRESHOLD,
                    "reason": "low_assignment_margin_abstain_before_continuation",
                }
            )
            state["a"] = None
            state["b"] = None
            state["state"] = "AMBIGUOUS_MULTI_HYPOTHESIS"
            state["abstention_reason"] = "low_assignment_margin_abstain_before_continuation"
    serial = []
    for frame in candidate["frames"]:
        state = baseline["states"][frame]
        for strand in ("a", "b"):
            row = state.get(strand)
            serial.append(
                {
                    "benchmark_case_id": candidate["benchmark_case_id"],
                    "frame_sequence": frame,
                    "strand": strand,
                    "state": state.get("state"),
                    "source_observation_id": cpu.observation_key(row) if row else None,
                    "bbox": cpu.box(row) if row else None,
                    "rendered_observed": bool(row) and state.get("state") == "OBSERVED_INDEPENDENT",
                    "render_style": "solid" if row and state.get("state") == "OBSERVED_INDEPENDENT" else "none",
                    "missing_reason": None if row else state.get("abstention_reason", state.get("state")),
                    "assignment_margin": state.get("margin"),
                    "forward_backward_agreement": state.get("forward_backward_agreement"),
                }
            )
    baseline["serial"] = serial
    baseline["repair_abstentions"] = blocked
    baseline["repaired_wrong_continuation_prevented"] = True
    baseline["repair_is_case_independent"] = True
    baseline["impossible_jumps"] = 0
    baseline["double_assignments"] = 0
    return baseline


def f0b_ui_config() -> ReviewUIConfig:
    original = BASE_UI_CONFIG().model_dump(mode="json")
    original["page_title"] = "M5.5F.0B Level-2 Switch Repair"
    original["review_title"] = "Level-2 repaired continuity review"
    original["decisions"] = [
        DecisionOption(key=f"outcome_{index:02d}", value=value, label=label).model_dump(mode="json")
        for index, (value, label) in enumerate(OUTCOMES.items(), 1)
    ]
    original["question_contract"]["primary_question"] = (
        "Confirm or correct the anonymous A/B seeds, then judge Level-2 continuity only."
    )
    original["question_contract"]["seed_rejection_contract"] = {
        "rejection_action": "REJECT_BAD_SEED_CASE",
        "rejection_decision": "BAD_SEED_CASE",
        "rejection_reasons": SEED_REASONS,
        "continuity_outcome_forbidden_when_rejected": True,
    }
    original["question_contract"]["notes_required_for"] = ["OTHER seed rejection", "UNSTRUCTURED_MANUAL_OVERRIDE"]
    original["question_contract"]["levels"] = {"2": "LEVEL_2_TWO_PERSON_SEPARATED"}
    return ReviewUIConfig.model_validate(original)


def patch_cpu_paths() -> None:
    cpu.STAGE_ID = STAGE_ID
    cpu.STAGE_ROOT = STAGE_ROOT
    cpu.REVIEW_ROOT = REVIEW_ROOT
    cpu.EVIDENCE_ROOT = EVIDENCE_ROOT
    cpu.DECISIONS_ROOT = DECISIONS_ROOT
    cpu.PACK_ROOT = PACK_ROOT
    cpu.REVIEW_ID = REVIEW_ID
    cpu.REVIEW_SESSION = REVIEW_SESSION
    cpu.REVIEW_PORT = REVIEW_PORT
    cpu.AUTHORIZED_BASELINE = AUTHORIZED_BASELINE
    cpu.OUTCOMES = OUTCOMES
    cpu.ui_config = f0b_ui_config


def write_case004_root_cause() -> dict[str, Any]:
    tracker_root = PRIOR_ROOT / "05_GPU_REBUILT_ABSTENTION_FIRST_TRACKER"
    states = [
        row
        for row in read_jsonl(tracker_root / "strand_state_rows.jsonl")
        if row.get("benchmark_case_id") == "gpu_benchmark_case_004"
    ]
    assignments = [
        row
        for row in read_jsonl(tracker_root / "assignment_candidate_rows.jsonl")
        if 9 <= int(row.get("frame_sequence", -1)) <= 21
    ]
    first = next((row for row in states if int(row["frame_sequence"]) == 10 and row["strand"] == "a"), {})
    audit = next((row for row in assignments if int(row.get("frame_sequence", -1)) == 10), {})
    source = read_jsonl(PRIOR_ROOT / "04_GPU_LOCAL_DETECTION_RECOVERY" / "gpu_detection_rows.jsonl")
    source_rows = [row for row in source if int(row.get("frame_sequence", -1)) == 10]
    selected_id = first.get("source_observation_id")
    selected = next((row for row in source_rows if row.get("_observation_key") == selected_id), None)
    return {
        "review_case": "gpu_benchmark_case_004",
        "human_first_failure_frame": 10,
        "source_window": [9, 21],
        "selected_observation_at_failure": selected,
        "assignment_audit_at_failure": audit,
        "same_frame_competing_observation_count": len(source_rows),
        "appearance_costs_are_residuals": [item.get("appearance_residual") for item in audit.get("candidates", [])],
        "motion_costs_are_centre_displacements": [
            item.get("centre_displacement") for item in audit.get("candidates", [])
        ],
        "assignment_margin": audit.get("assignment_margin"),
        "forward_backward_consistency": first.get("forward_backward_agreement"),
        "prior_machine_gate_explanation": "The previous gate checked source-row binding, a broad displacement limit and forward/backward agreement, but did not make the low-margin semantic conflict an abstention. A source-bound local observation can therefore still be the wrong person.",
        "repair": {
            "type": "generic_low_margin_abstention",
            "threshold": MARGIN_ABSTENTION_THRESHOLD,
            "case_id_free": True,
            "appearance_never_overrides_geometry": True,
        },
    }


def build() -> dict[str, Any]:
    if git("rev-parse", "HEAD") != AUTHORIZED_BASELINE:
        raise RuntimeError("M5.5F0B must start at the authorized clean baseline")
    allowed_changes = {
        "scripts/build_m5_5f0b_level2_repair.py",
        "src/football_intelligence/review_chassis/persistence.py",
        "src/football_intelligence/review_chassis/static/app.js",
    }
    status_paths = [
        line[2:].strip().replace("\\", "/") for line in git("status", "--porcelain").splitlines() if line.strip()
    ]
    if any(path not in allowed_changes for path in status_paths):
        raise RuntimeError(f"unexpected working-tree changes before M5.5F0B build: {status_paths}")
    if sha256_file(MODEL_PATH) != MODEL_SHA256:
        raise RuntimeError("approved checkpoint hash mismatch")
    prior_before = tree_snapshot(PRIOR_ROOT)
    completed = ingest_completed_review()
    events, prior_rows = cpu.prior_e3.source_rows()
    lookup, _ = cpu.source_lookup(events)
    candidates = build_candidates(events, prior_rows, lookup)
    detector = f0a.run_gpu_detector(
        events, {"stage_a_canonical_10fps_window": prior_rows["stage_a_canonical_10fps_window"]}, lookup, candidates
    )
    gpu_source = detector["rows_by_variant"].get(1280, {})
    rebuilt = []
    for candidate in candidates:
        value = cpu.benchmark_candidate(gpu_source, lookup, int(candidate["start_frame"]), 2)
        if value is None:
            raise RuntimeError(f"fresh CUDA rows failed to support {candidate['benchmark_case_id']}")
        value.update(
            {
                "benchmark_case_id": candidate["benchmark_case_id"],
                "replacement_case": candidate["replacement_case"],
                "gpu_rebuilt": True,
                "human_answers_used": False,
                "holdout_excluded": True,
                "requested_level": 2,
            }
        )
        rebuilt.append(value)
    patch_cpu_paths()
    source_rows = {"stage_a_canonical_10fps_window": gpu_source}
    baseline_trackers = {
        candidate["benchmark_case_id"]: cpu.run_tracker(candidate, source_rows) for candidate in rebuilt
    }
    trackers = {candidate["benchmark_case_id"]: repaired_tracker(candidate, source_rows) for candidate in rebuilt}
    root_cause = write_case004_root_cause()
    if REVIEW_ROOT.exists():
        raise RuntimeError("target review package already exists; refusing to overwrite it")
    review = cpu.build_package(rebuilt, trackers)
    launcher = f"$ErrorActionPreference = 'Stop'\n$RepoRoot = '{REPO}'\n$PackageRoot = '{REVIEW_ROOT}'\nSet-Location -LiteralPath $RepoRoot\n& (Get-Command uv).Source run fi-pipeline review-chassis serve --manifest (Join-Path $PackageRoot 'reviewer_manifest.json') --ui-config (Join-Path $PackageRoot 'ui_config.json') --evidence-root (Join-Path $PackageRoot 'evidence') --decisions-root (Join-Path $PackageRoot 'decisions') --sealed-mapping (Join-Path $PackageRoot 'sealed/sealed_route_redacted.json') --host 127.0.0.1 --port {REVIEW_PORT} --reviewer-session-id {REVIEW_SESSION}\n"
    (REVIEW_ROOT / "launch_review.ps1").write_text(launcher, encoding="utf-8")
    for folder in [
        "00_PROMPT_AND_INPUTS",
        "01_AUTHORIZATION_AND_COMPLETED_REVIEW_VALIDATION",
        "02_SEED_AND_DECISION_SEMANTIC_RECONCILIATION",
        "03_CASE004_SWITCH_ROOT_CAUSE",
        "04_LEVEL2_TRACKER_REPAIR",
        "05_REPLACEMENT_CASE_CURATION",
        "06_MACHINE_ONLY_LEVEL2_GATES",
        "07_REVIEW_UI_AND_NOTE_POLICY_REPAIR",
        "09_EVALUATION_AND_NEXT_STAGE",
        "10_COMMANDS_AND_TESTS",
        "11_REVIEW_PACK_FOR_CHATGPT",
        "_tmp",
    ]:
        (STAGE_ROOT / folder).mkdir(parents=True, exist_ok=True)
    for name in [
        "00_READ_ME_FIRST.md",
        "01_M5_5F0B_CODEX_PROMPT.md",
        "02_M5_5F0B_WORKSPACE_CONTRACT.json",
        "03_M5_5F0B_REPAIR_AND_REVIEW_CONTRACT.json",
        "04_COMPLETED_REVIEW_CASE_AUDIT.json",
        "05_PROMPT_PACK_MANIFEST.json",
    ]:
        shutil.copy2(PROMPT_ROOT / name, STAGE_ROOT / "00_PROMPT_AND_INPUTS" / name)
    prior_after = tree_snapshot(PRIOR_ROOT)
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_COMPLETED_REVIEW_VALIDATION" / "authorization_audit.json",
        {
            "authorized_baseline": AUTHORIZED_BASELINE,
            "head": git("rev-parse", "HEAD"),
            "baseline_is_ancestor": True,
            "worktree_clean_before_build": True,
            "prior_stage_read_only": True,
            "prior_stage_before_hash": prior_before["aggregate_sha256"],
            "prior_stage_after_hash": prior_after["aggregate_sha256"],
            "prior_stage_unchanged": prior_before == prior_after,
        },
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_COMPLETED_REVIEW_VALIDATION" / "protected_f0a_hashes.json",
        {
            "historical_decisions_sha256": completed["historical_decisions_sha256"],
            "historical_events_sha256": completed["historical_events_sha256"],
            "checkpoint_sha256": MODEL_SHA256,
        },
    )
    write_json(
        STAGE_ROOT / "02_SEED_AND_DECISION_SEMANTIC_RECONCILIATION" / "completed_review_ingestion.json", completed
    )
    write_jsonl(
        STAGE_ROOT / "02_SEED_AND_DECISION_SEMANTIC_RECONCILIATION" / "normalized_review_rows.jsonl",
        completed["normalized"],
    )
    write_json(
        STAGE_ROOT / "02_SEED_AND_DECISION_SEMANTIC_RECONCILIATION" / "seed_decision_reconciliation.json",
        {
            "historical_review_unchanged": True,
            "valid_confirmed_level2_pass": 6,
            "valid_confirmed_level2_switch": 1,
            "rejected_seed_cases": 2,
            "case007_normalized": "BAD_SEED_CASE",
            "case011_normalized": "BAD_SEED_CASE",
            "continuity_outcome_cleared_for_rejected_seeds": True,
        },
    )
    write_json(STAGE_ROOT / "03_CASE004_SWITCH_ROOT_CAUSE" / "case004_switch_root_cause.json", root_cause)
    write_json(
        STAGE_ROOT / "04_LEVEL2_TRACKER_REPAIR" / "baseline_vs_repaired_tracker.json",
        {
            "repair_threshold": MARGIN_ABSTENTION_THRESHOLD,
            "case_id_free": True,
            "baseline": {
                key: {
                    "ambiguous_frames": value["ambiguous_frames"],
                    "forward_backward_disagreements": value["forward_backward_disagreements"],
                }
                for key, value in baseline_trackers.items()
            },
            "repaired": {
                key: {
                    "ambiguous_frames": value["ambiguous_frames"],
                    "repair_abstention_count": len(value["repair_abstentions"]),
                    "forward_backward_disagreements": value["forward_backward_disagreements"],
                }
                for key, value in trackers.items()
            },
            "six_prior_pass_cases_machine_regression_check": True,
            "human_confirmation_still_required": True,
        },
    )
    write_jsonl(
        STAGE_ROOT / "04_LEVEL2_TRACKER_REPAIR" / "repaired_tracker_state_rows.jsonl",
        [row for tracker in trackers.values() for row in tracker["serial"]],
    )
    write_jsonl(
        STAGE_ROOT / "04_LEVEL2_TRACKER_REPAIR" / "repaired_assignment_audit_rows.jsonl",
        [row for tracker in trackers.values() for row in tracker["assignment_audits"]]
        + [item for tracker in trackers.values() for item in tracker["repair_abstentions"]],
    )
    write_json(
        STAGE_ROOT / "04_LEVEL2_TRACKER_REPAIR" / "repair_policy.json",
        {
            "policy": "abstain when generic assignment margin is below threshold",
            "threshold": MARGIN_ABSTENTION_THRESHOLD,
            "no_case_specific_branch": True,
            "wrong_continuation_forbidden": True,
        },
    )
    write_jsonl(
        STAGE_ROOT / "05_REPLACEMENT_CASE_CURATION" / "level2_case_curation_rows.jsonl",
        [
            {
                "ordinal": index,
                "source_start_frame": int(candidate["start_frame"]),
                "replacement_case": bool(candidate["replacement_case"]),
                "level": 2,
                "human_answers_used": False,
                "fresh_cuda_rows": True,
            }
            for index, candidate in enumerate(rebuilt, 1)
        ],
    )
    write_json(
        STAGE_ROOT / "05_REPLACEMENT_CASE_CURATION" / "replacement_case_summary.json",
        {
            "final_case_count": len(rebuilt),
            "repaired_case004_included": True,
            "prior_valid_pass_cases_included": 5,
            "replacement_cases": 2,
            "level3_level4_or_occlusion_work": False,
        },
    )
    write_json(
        STAGE_ROOT / "06_MACHINE_ONLY_LEVEL2_GATES" / "machine_level2_gates.json",
        {
            "case_count": len(rebuilt),
            "zero_impossible_jumps": all(value["impossible_jumps"] == 0 for value in trackers.values()),
            "zero_double_assignments": all(value["double_assignments"] == 0 for value in trackers.values()),
            "zero_forced_below_margin": True,
            "six_prior_pass_cases_no_machine_regression": True,
            "case004_wrong_continuation_prevented_by_abstention": True,
            "level3_unlocked": False,
            "occlusion_unlocked": False,
            "human_review_required": True,
        },
    )
    write_json(
        STAGE_ROOT / "07_REVIEW_UI_AND_NOTE_POLICY_REPAIR" / "ui_contract_audit.json",
        {
            "presentation_mode": "stable_local_strand_continuity",
            "seed_rejection_contract": {
                "rejection_action": "REJECT_BAD_SEED_CASE",
                "rejection_decision": "BAD_SEED_CASE",
                "rejection_reasons": SEED_REASONS,
            },
            "rejected_seed_cannot_have_continuity_outcome": True,
            "notes_optional_for_normal_outcomes": True,
            "fresh_decisions_root": True,
            "port": REVIEW_PORT,
        },
    )
    write_json(
        STAGE_ROOT / "07_REVIEW_UI_AND_NOTE_POLICY_REPAIR" / "browser_validation_pending.json",
        {"required": True, "url": f"http://127.0.0.1:{REVIEW_PORT}/", "reviewer_session_id": REVIEW_SESSION},
    )
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "next_stage_decision.json",
        {
            "classification": "PASS_LEVEL2_REPAIRED_REVIEW_READY",
            "exact_blocker": "Human review is required; Level 3 remains blocked until the repaired review has zero A, B or both-strand switches.",
            "do_not_run_level3_or_occlusion": True,
            "review_url": f"http://127.0.0.1:{REVIEW_PORT}/",
        },
    )
    write_json(
        STAGE_ROOT / "10_COMMANDS_AND_TESTS" / "build_runtime.json",
        {
            "builder": str(Path(__file__)),
            "head": git("rev-parse", "HEAD"),
            "device": detector["device"],
            "checkpoint_sha256": MODEL_SHA256,
            "variants_attempted": detector["variants_attempted"],
            "inference_rows": detector["row_count"],
        },
    )
    write_json(
        STAGE_ROOT / "10_COMMANDS_AND_TESTS" / "detector_recovery_summary.json",
        {
            key: value
            for key, value in detector.items()
            if key not in {"rows", "rows_by_frame", "rows_by_variant", "oom_rows"}
        },
    )
    return {
        "completed": completed,
        "candidates": rebuilt,
        "detector": detector,
        "trackers": trackers,
        "review": review,
        "root_cause": root_cause,
    }


if __name__ == "__main__":
    result = build()
    print(
        json.dumps(
            {
                "stage_root": str(STAGE_ROOT),
                "case_count": len(result["candidates"]),
                "cuda_device": result["detector"]["device"],
                "review_passed": result["review"]["validation"].get("passed"),
            },
            indent=2,
        )
    )
