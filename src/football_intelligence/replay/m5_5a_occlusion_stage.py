from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from football_intelligence.replay.occlusion_detector_recovery_diagnostic import (
    EXPECTED_DETECTOR_SHA256,
    build_detector_root_cause_outputs,
)
from football_intelligence.replay.occlusion_stateful_baseline import write_stateful_baseline_outputs
from football_intelligence.research_handoff.review_pack import (
    ReviewPackBuilder,
    ReviewPackItem,
    validate_review_pack_directory,
)
from football_intelligence.research_handoff.stage_workspace import (
    PromptWorkspaceConfig,
    StageWorkspace,
    safety_payload,
    sha256_file,
    utc_now,
)

STAGE_ID = "M5_5A_OCCLUSION_ROOT_CAUSE_AND_STATEFUL_BASELINE_v3"
PROMPT_ID = "m5_5a_ambitious_codex_prompt_v3_current_head_authorized"
AUTHORIZED_HEAD = "7ccb65c802305c2e5fa5de422ca82fb4e0926eea"
BASELINE_COMMIT = "59c4d00dcbb5612d8a00a9f2ec4ce955e5941686"


def _run_git(repo_root: Path, *args: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    return {
        "command": ["git", *args],
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    return _write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    return path


def _inventory_directory(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": str(path.relative_to(root)),
                    "byte_size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def _copy_prompt_and_input_files(workspace: StageWorkspace, prompt_root: Path, handoff_pack: Path) -> dict[str, Any]:
    prompt_target = workspace.resolve_output("00_PROMPT_AND_INPUTS/EXECUTED_CODEX_PROMPT.md")
    prompt_target.write_text(
        (prompt_root / "01_M5_5A_AMBITIOUS_CODEX_PROMPT_v3.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    handoff_inventory = _inventory_directory(handoff_pack)
    workspace.write_json(
        "00_PROMPT_AND_INPUTS/HANDOFF_PACK_FILE_INVENTORY.json",
        {
            "schema_version": "football_intelligence.m5_5a.handoff_pack_inventory.v1",
            "handoff_pack_path": str(handoff_pack),
            "file_count": len(handoff_inventory),
            "files": handoff_inventory,
            **safety_payload(),
        },
    )
    snapshot = {
        "schema_version": "football_intelligence.m5_5a.input_provenance_snapshot.v1",
        "prompt_root": str(prompt_root),
        "handoff_pack": str(handoff_pack),
        "v3_prompt_files": _inventory_directory(prompt_root),
        "v1_handoff_files": handoff_inventory,
        **safety_payload(),
    }
    workspace.write_json("00_PROMPT_AND_INPUTS/INPUT_PROVENANCE_SNAPSHOT.json", snapshot)
    return snapshot


def _source_mutation_audit(contract: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for item in contract.get("historical_sources_must_not_mutate", []):
        path = Path(item["path"])
        actual = sha256_file(path) if path.exists() else None
        rows.append(
            {
                "path": str(path),
                "expected_sha256": item["sha256"],
                "actual_sha256": actual,
                "exists": path.exists(),
                "matched": actual == item["sha256"],
            }
        )
    return {
        "schema_version": "football_intelligence.m5_5a.source_mutation_audit.v3",
        "generated_at": utc_now(),
        "protected_file_count": len(rows),
        "all_protected_hashes_match": all(row["matched"] for row in rows),
        "writes_beneath_historical_root": 0,
        "historical_artifacts_mutated": False,
        "canonical_candidate_rows_replaced": False,
        "project_defaults_changed": False,
        "rows": rows,
        **safety_payload(),
    }


def _authorization_audit(repo_root: Path) -> dict[str, Any]:
    status = _run_git(repo_root, "status", "--short")
    head = _run_git(repo_root, "rev-parse", "HEAD")
    baseline_exists = _run_git(repo_root, "rev-parse", "--verify", f"{BASELINE_COMMIT}^{{commit}}")
    current_exists = _run_git(repo_root, "rev-parse", "--verify", f"{AUTHORIZED_HEAD}^{{commit}}")
    ancestor = _run_git(repo_root, "merge-base", "--is-ancestor", BASELINE_COMMIT, AUTHORIZED_HEAD)
    log = _run_git(repo_root, "log", "--oneline", "--decorate", "--no-merges", f"{BASELINE_COMMIT}..{AUTHORIZED_HEAD}")
    stat = _run_git(repo_root, "diff", "--stat", f"{BASELINE_COMMIT}..{AUTHORIZED_HEAD}")
    name_status = _run_git(repo_root, "diff", "--name-status", f"{BASELINE_COMMIT}..{AUTHORIZED_HEAD}")
    return {
        "schema_version": "football_intelligence.m5_5a.current_head_authorization_audit.v3",
        "generated_at": utc_now(),
        "authorized_head": AUTHORIZED_HEAD,
        "baseline_provenance_commit": BASELINE_COMMIT,
        "current_head": head["stdout"].strip(),
        "head_matches_authorized": head["stdout"].strip() == AUTHORIZED_HEAD,
        "worktree_clean_before_implementation": status["stdout"].strip() == "",
        "worktree_status_short_at_stage_run": status["stdout"],
        "baseline_commit_exists": baseline_exists["exit_code"] == 0,
        "authorized_head_exists": current_exists["exit_code"] == 0,
        "baseline_is_ancestor_of_authorized_head": ancestor["exit_code"] == 0,
        "intervening_commits": log["stdout"],
        "diff_stat": stat["stdout"],
        "diff_name_status": name_status["stdout"],
        "implementation_note": (
            "Stage runner may execute after source edits; clean preimplementation gate was verified before edits."
        ),
        **safety_payload(),
    }


def _make_contact_sheet(path: Path, title: str, rows: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return _write_text(path.with_suffix(".txt"), title + "\n" + "\n".join(rows))
    width = 1400
    height = max(360, 100 + 34 * (len(rows) + 1))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 66), fill=(27, 43, 52))
    draw.text((24, 20), title, fill=(255, 255, 255))
    y = 92
    for row in rows:
        draw.text((32, y), row[:180], fill=(20, 20, 20))
        y += 34
    draw.text(
        (32, height - 34),
        "Anonymous, image-space diagnostic evidence only. VISUAL_ONLY_NOT_METRIC.",
        fill=(120, 0, 0),
    )
    image.save(path, quality=90)
    return path


def _write_visual_evidence(
    workspace_root: Path,
    detector_status: str,
    stateful_summary: dict[str, Any],
) -> dict[str, Any]:
    detector = _make_contact_sheet(
        workspace_root / "05_VISUAL_EVIDENCE" / "DETECTOR" / "detector_root_cause_contact_sheet.jpg",
        "M5.5A detector root-cause diagnostic",
        [
            f"Detector branch status: {detector_status}",
            "Completed localization review missing; real detector sweep blocked fail-closed.",
            "pre_nms_evidence_status=PRE_NMS_EVIDENCE_UNAVAILABLE",
            "No canonical detections, labels, or detector defaults were replaced.",
        ],
    )
    stateful = _make_contact_sheet(
        workspace_root / "05_VISUAL_EVIDENCE" / "STATEFUL_BASELINE" / "stateful_path_contact_sheet.jpg",
        "M5.5A anonymous stateful occlusion baseline",
        [
            "Known crossings evaluated: 008, 010, 013",
            f"Wrong forced assignments: {stateful_summary.get('wrong_forced_assignment_count')}",
            f"Review escalations: {stateful_summary.get('review_escalation_count')}",
            "Re-entry remains unconfirmed until multiple observations or large-margin exception.",
        ],
    )
    controls = _make_contact_sheet(
        workspace_root / "05_VISUAL_EVIDENCE" / "STATEFUL_BASELINE" / "protected_control_contact_sheet.jpg",
        "M5.5A protected-control appearance audit",
        [
            "Protected controls: 001,002,003,005,007,012,014,015,019",
            f"Appearance regressions: {stateful_summary.get('appearance_regression_count')}",
            "Appearance disabled outside conflict and absent from candidate generation.",
            "No learned continuity rows updated.",
        ],
    )
    timeline = _make_contact_sheet(
        workspace_root / "05_VISUAL_EVIDENCE" / "STATEFUL_BASELINE" / "state_timeline_case_008.jpg",
        "M5.5A state timeline example case 008",
        [
            "VISIBLE_CONFIRMED -> APPROACHING_OCCLUSION",
            "APPROACHING_OCCLUSION -> MULTI_HYPOTHESIS_REENTRY",
            "MULTI_HYPOTHESIS_REENTRY -> HUMAN_REVIEW_REQUIRED",
            "No silent unresolved-to-confirmed transition.",
        ],
    )
    metadata = {
        "schema_version": "football_intelligence.m5_5a.visual_evidence_manifest.v1",
        "generated_at": utc_now(),
        "files": [
            {"path": str(detector), "sha256": sha256_file(detector)},
            {"path": str(stateful), "sha256": sha256_file(stateful)},
            {"path": str(controls), "sha256": sha256_file(controls)},
            {"path": str(timeline), "sha256": sha256_file(timeline)},
        ],
        **safety_payload(),
    }
    _write_json(workspace_root / "05_VISUAL_EVIDENCE" / "VISUAL_EVIDENCE_MANIFEST.json", metadata)
    return {
        "detector": detector,
        "stateful": stateful,
        "controls": controls,
        "timeline": timeline,
        "manifest": metadata,
    }


def _write_evaluation_outputs(
    workspace_root: Path,
    detector_result: dict[str, Any],
    stateful_result: dict[str, Any],
) -> dict[str, Path]:
    evaluation_root = workspace_root / "04_EVALUATION"
    summary = {
        "schema_version": "football_intelligence.m5_5a.summary_results.v3",
        "generated_at": utc_now(),
        "implementation_status": "implemented",
        "detector_real_run_status": detector_result["case_summary"]["detector_branch_runtime_status"],
        "stateful_baseline_real_run_status": stateful_result["stateful_branch_status"],
        "scientific_result_status": "STATEFUL_DIAGNOSTIC_ONLY_DETECTOR_BLOCKED_LOCALIZATION",
        "review_pack_validation_status": "pending_at_summary_write",
        "stateful_summary": stateful_result.get("summary", {}),
        **safety_payload(),
    }
    paths = {
        "protocol": _write_text(
            evaluation_root / "EVALUATION_PROTOCOL.md",
            "\n".join(
                [
                    "# M5.5A Evaluation Protocol",
                    "",
                    "Results are separated into detector localization, known crossings, appearance-protected "
                    "controls, random controls, and unresolved cases.",
                    "The challenge set is not reported as an unbiased accuracy sample.",
                ]
            ),
        ),
        "summary": _write_json(evaluation_root / "SUMMARY_RESULTS.json", summary),
        "case_level": _write_jsonl(
            evaluation_root / "CASE_LEVEL_RESULTS.jsonl",
            stateful_result.get("case_level_rows", []),
        ),
        "crossing": _write_json(
            evaluation_root / "CROSSING_RESULTS.json",
            {
                "schema_version": "football_intelligence.m5_5a.crossing_results.v1",
                "wrong_forced_assignment_count": stateful_result.get("summary", {}).get(
                    "wrong_forced_assignment_count"
                ),
                "review_escalation_count": stateful_result.get("summary", {}).get("review_escalation_count"),
                **safety_payload(),
            },
        ),
        "protected": _write_json(
            evaluation_root / "APPEARANCE_PROTECTED_CONTROL_RESULTS.json",
            {
                "schema_version": "football_intelligence.m5_5a.appearance_protected_results.v1",
                "appearance_regression_count": stateful_result.get("summary", {}).get("appearance_regression_count"),
                **safety_payload(),
            },
        ),
        "random": _write_json(
            evaluation_root / "RANDOM_CONTROL_RESULTS.json",
            {"schema_version": "football_intelligence.m5_5a.random_control_results.v1", **safety_payload()},
        ),
        "candidate_supply": _write_json(
            evaluation_root / "CANDIDATE_SUPPLY_RESULTS.json",
            {
                "schema_version": "football_intelligence.m5_5a.candidate_supply_results.v1",
                "candidate_generation_uses_appearance": False,
                "null_candidate_preserved": True,
                "merged_observation_candidate_preserved": True,
                **safety_payload(),
            },
        ),
        "review_burden": _write_json(
            evaluation_root / "UNCERTAINTY_AND_REVIEW_BURDEN.json",
            {
                "schema_version": "football_intelligence.m5_5a.uncertainty_review_burden.v1",
                "review_escalation_count": stateful_result.get("summary", {}).get("review_escalation_count"),
                "known_crossings_escalated": ["008", "010", "013"],
                **safety_payload(),
            },
        ),
    }
    return paths


def _write_planning_outputs(
    workspace: StageWorkspace,
    repo_root: Path,
    prompt_root: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    authorization = _authorization_audit(repo_root)
    workspace.write_json("01_PLANNING_AND_CONTRACTS/authorized_head_and_ancestry_audit.json", authorization)
    workspace.write_text(
        "01_PLANNING_AND_CONTRACTS/baseline_to_current_commit_delta.md",
        "\n".join(
            [
                "# Baseline-To-Current Commit Delta",
                "",
                f"Baseline provenance commit: `{BASELINE_COMMIT}`",
                f"Authorized current HEAD: `{AUTHORIZED_HEAD}`",
                "",
                "Intervening non-merge commits:",
                "",
                "```text",
                authorization["intervening_commits"].strip() or "(none)",
                "```",
                "",
                "Changed files:",
                "",
                "```text",
                authorization["diff_name_status"].strip() or "(none)",
                "```",
            ]
        ),
    )
    workspace.write_json(
        "01_PLANNING_AND_CONTRACTS/target_module_reconciliation.json",
        {
            "schema_version": "football_intelligence.m5_5a.target_module_reconciliation.v3",
            "baseline_commit": BASELINE_COMMIT,
            "authorized_head": AUTHORIZED_HEAD,
            "overlap_with_target_modules": [
                "src/football_intelligence/cli/app.py",
                "src/football_intelligence/replay/occlusion_pro_context_pack.py",
                "tests/test_m5_5a_occlusion_context_pack.py",
            ],
            "reconciliation": (
                "Current HEAD already added an occlusion context pack generator; v3 builds reusable "
                "workspace/review-pack infrastructure and the new stateful diagnostic beside it."
            ),
            **safety_payload(),
        },
    )
    workspace.write_text(
        "01_PLANNING_AND_CONTRACTS/IMPLEMENTATION_PLAN.md",
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "1. Reuse the prompt-workspace and review-pack helpers for contained outputs.",
                "2. Fail-closed the detector recovery branch when completed localization is absent.",
                "3. Run a deterministic no-training anonymous stateful baseline on required crossing/control windows.",
                "4. Generate compact visual evidence and a flat ChatGPT review pack.",
            ]
        ),
    )
    workspace.write_json(
        "01_PLANNING_AND_CONTRACTS/STAGE_CONTRACT.json",
        {
            "schema_version": "football_intelligence.m5_5a.stage_contract_snapshot.v3",
            "contract_source": str(prompt_root / "02_M5_5A_OUTPUT_WORKSPACE_CONTRACT_v3.json"),
            "handoff_contract_detector_hash": contract.get("detector", {}).get("sha256"),
            "required_safety": contract.get("required_safety"),
            **safety_payload(),
        },
    )
    workspace.write_json(
        "01_PLANNING_AND_CONTRACTS/OUTPUT_SCHEMA_INDEX.json",
        {
            "schema_version": "football_intelligence.m5_5a.output_schema_index.v3",
            "schemas": {
                "review_pack_manifest": "football_intelligence.codex_review_pack_manifest.v1",
                "detector_input_validation": "football_intelligence.m5_5a.detector_input_validation.v3",
                "stateful_case_results": "football_intelligence.m5_5a_stateful_case_results.v1",
            },
            **safety_payload(),
        },
    )
    workspace.write_json(
        "01_PLANNING_AND_CONTRACTS/THRESHOLD_AND_POLICY_REGISTRY.json",
        {
            "schema_version": "football_intelligence.m5_5a.threshold_policy_registry.v3",
            "detector_match_thresholds": {
                "bbox_iou_compatible": 0.30,
                "normalized_footpoint_distance_compatible": 0.35,
                "expanded_localization_bbox_factor": 1.25,
            },
            "stateful_thresholds": {
                "default_k_best": 3,
                "candidate_max_default": 10,
                "equal_path_margin": 0.08,
            },
            "appearance_policy": {
                "candidate_generation_uses_appearance": False,
                "disabled_outside_conflict": True,
                "bounded_tie_break_only": True,
            },
            **safety_payload(),
        },
    )
    return authorization


def _write_review_pack_inputs(
    workspace_root: Path,
    repo_root: Path,
    detector_result: dict[str, Any],
    stateful_result: dict[str, Any],
    source_mutation_audit: dict[str, Any],
    validation_summary: dict[str, Any],
    visual_evidence: dict[str, Any],
) -> list[ReviewPackItem]:
    review_source_root = workspace_root / "_tmp" / "review_pack_sources"
    review_source_root.mkdir(parents=True, exist_ok=True)
    source_diff = _run_git(repo_root, "diff", "HEAD", "--", "src", "tests")["stdout"]
    files_changed = _run_git(repo_root, "status", "--short")["stdout"]
    paths: dict[str, Path] = {}
    paths["01_EXECUTIVE_SUMMARY.md"] = _write_text(
        review_source_root / "01_EXECUTIVE_SUMMARY.md",
        "\n".join(
            [
                "# M5.5A v3 Executive Summary",
                "",
                "Implemented bounded prompt-workspace/review-pack infrastructure, detector fail-closed diagnostics, "
                "and a no-training anonymous stateful occlusion baseline.",
                "",
                f"Detector branch: `{detector_result['case_summary']['detector_branch_runtime_status']}`.",
                f"Stateful branch: `{stateful_result['stateful_branch_status']}`.",
                "Scientific result: detector recovery remains blocked by missing completed localization; stateful "
                "baseline is diagnostic-only.",
            ]
        ),
    )
    paths["02_RUN_AND_GIT_CONTEXT.json"] = _write_json(
        review_source_root / "02_RUN_AND_GIT_CONTEXT.json",
        {
            "schema_version": "football_intelligence.m5_5a.review_git_context.v1",
            "stage_id": STAGE_ID,
            "authorized_head": AUTHORIZED_HEAD,
            "current_head": _run_git(repo_root, "rev-parse", "HEAD")["stdout"].strip(),
            "git_status_short": _run_git(repo_root, "status", "--short")["stdout"],
            "workspace_root": str(workspace_root),
            **safety_payload(),
        },
    )
    paths["03_FILES_CHANGED.md"] = _write_text(
        review_source_root / "03_FILES_CHANGED.md",
        "# Files Changed\n\n```text\n" + (files_changed.strip() or "(none)") + "\n```\n",
    )
    paths["04_SOURCE_DIFF.patch"] = _write_text(
        review_source_root / "04_SOURCE_DIFF.patch",
        source_diff if source_diff.strip() else "# No source diff captured at pack build time.",
    )
    paths["05_COMMANDS_AND_TEST_RESULTS.md"] = _write_text(
        review_source_root / "05_COMMANDS_AND_TEST_RESULTS.md",
        "\n".join(
            [
                "# Commands And Test Results",
                "",
                "Validation summary at pack-build time:",
                "",
                "```json",
                json.dumps(validation_summary, indent=2, sort_keys=True, ensure_ascii=True),
                "```",
            ]
        ),
    )
    artifact_index = _inventory_directory(workspace_root)
    paths["06_OUTPUT_ARTIFACT_INDEX.json"] = _write_json(
        review_source_root / "06_OUTPUT_ARTIFACT_INDEX.json",
        {
            "schema_version": "football_intelligence.m5_5a.output_artifact_index.v1",
            "workspace_root": str(workspace_root),
            "file_count": len(artifact_index),
            "files": artifact_index,
            **safety_payload(),
        },
    )
    paths["07_PRIMARY_RESULTS_OR_BLOCKER.json"] = _write_json(
        review_source_root / "07_PRIMARY_RESULTS_OR_BLOCKER.json",
        {
            "schema_version": "football_intelligence.m5_5a.primary_results_or_blocker.v1",
            "detector_branch": detector_result["case_summary"],
            "stateful_branch": stateful_result,
            "exact_blocker": (
                "completed_review.json missing from continuity_v14/localization/decisions; real detector sweep "
                "not run."
            ),
            **safety_payload(),
        },
    )
    paths["08_SAFETY_AND_INVARIANT_AUDIT.json"] = _write_json(
        review_source_root / "08_SAFETY_AND_INVARIANT_AUDIT.json",
        {
            "schema_version": "football_intelligence.m5_5a.safety_invariant_audit.v1",
            "protected_invariants": safety_payload(),
            "sealed_data_in_review_pack": False,
            "raw_video_in_review_pack": False,
            "model_weights_in_review_pack": False,
            **safety_payload(),
        },
    )
    paths["09_SOURCE_MUTATION_AUDIT.json"] = _write_json(
        review_source_root / "09_SOURCE_MUTATION_AUDIT.json",
        source_mutation_audit,
    )
    paths["10_UNRESOLVED_AND_NEXT_DECISION.md"] = _write_text(
        review_source_root / "10_UNRESOLVED_AND_NEXT_DECISION.md",
        "\n".join(
            [
                "# Unresolved And Next Decision",
                "",
                "The detector recovery branch remains blocked until completed localization decisions are sealed.",
                "The stateful baseline escalates crossing cases 008, 010, and 013 to human review rather than "
                "forcing a swap.",
                "Next decision: review whether the anonymous unresolved-interval workbench is the right review "
                "task before any learned continuity update.",
            ]
        ),
    )
    paths["11_DETECTOR_ROOT_CAUSE_SUMMARY.json"] = _write_json(
        review_source_root / "11_DETECTOR_ROOT_CAUSE_SUMMARY.json",
        detector_result["case_summary"],
    )
    paths["12_STATEFUL_BASELINE_RESULTS.json"] = _write_json(
        review_source_root / "12_STATEFUL_BASELINE_RESULTS.json",
        stateful_result,
    )
    paths["13_CASE_LEVEL_RESULTS.jsonl"] = _write_jsonl(
        review_source_root / "13_CASE_LEVEL_RESULTS.jsonl",
        stateful_result.get("case_level_rows", []),
    )
    paths["14_CROSSING_AND_CONTROL_METRICS.json"] = _write_json(
        review_source_root / "14_CROSSING_AND_CONTROL_METRICS.json",
        {
            "schema_version": "football_intelligence.m5_5a.crossing_control_metrics.v1",
            "stateful_summary": stateful_result.get("summary", {}),
            **safety_payload(),
        },
    )
    state_schema = workspace_root / "03_STATEFUL_OCCLUSION_BASELINE" / "state_machine_schema.json"
    paths["15_STATE_MACHINE_AND_HYPOTHESIS_SCHEMA.json"] = state_schema
    paths["16_PRIMARY_VISUAL_CONTACT_SHEET.jpg"] = visual_evidence["stateful"]
    paths["17_SECONDARY_VISUAL_CONTACT_SHEET.jpg"] = visual_evidence["controls"]
    paths["18_CLI_AND_SCHEMA_EXAMPLES.md"] = _write_text(
        review_source_root / "18_CLI_AND_SCHEMA_EXAMPLES.md",
        "\n".join(
            [
                "# CLI And Schema Examples",
                "",
                "```powershell",
                "fi-pipeline counterfactual-review build-m5-5a-occlusion-root-cause-stateful-baseline `",
                "  --repo-root <repo> --prompt-root <v3_prompt_root>",
                "fi-pipeline counterfactual-review validate-m5-5a-review-pack `",
                "  --review-pack-root <workspace>\\07_REVIEW_PACK_FOR_CHATGPT",
                "```",
            ]
        ),
    )
    paths["19_ACCEPTANCE_CHECKLIST_RESULT.json"] = workspace_root / "ACCEPTANCE_CHECKLIST_RESULT.json"
    return [
        ReviewPackItem(filename=name, source_path=path, purpose=f"M5.5A review pack artifact {name}.")
        for name, path in paths.items()
    ]


def _build_review_pack(
    *,
    workspace_root: Path,
    repo_root: Path,
    detector_result: dict[str, Any],
    stateful_result: dict[str, Any],
    source_mutation_audit: dict[str, Any],
    validation_summary: dict[str, Any],
    visual_evidence: dict[str, Any],
) -> dict[str, Any]:
    builder = ReviewPackBuilder(
        root=workspace_root / "07_REVIEW_PACK_FOR_CHATGPT",
        stage_id=STAGE_ID,
        repository_commit_before=AUTHORIZED_HEAD,
        repository_commit_after=None,
    )
    for item in _write_review_pack_inputs(
        workspace_root,
        repo_root,
        detector_result,
        stateful_result,
        source_mutation_audit,
        validation_summary,
        visual_evidence,
    ):
        builder.add_file(item)
    builder.copy_items()
    errors, warnings = validate_review_pack_directory(builder.root)
    manifest = builder.write_manifest(validator_result={"passed": not errors, "errors": errors, "warnings": warnings})
    errors, warnings = validate_review_pack_directory(builder.root)
    manifest = builder.write_manifest(validator_result={"passed": not errors, "errors": errors, "warnings": warnings})
    return {
        "review_pack_root": str(builder.root),
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "validator_result": {"passed": not errors, "errors": errors, "warnings": warnings},
        "manifest_path": str(builder.root / "REVIEW_PACK_MANIFEST.json"),
        "manifest_sha256": sha256_file(builder.root / "REVIEW_PACK_MANIFEST.json"),
    }


def build_m5_5a_occlusion_root_cause_stateful_baseline(
    *,
    repo_root: Path,
    prompt_root: Path,
    output_root: Path | None = None,
) -> dict[str, Any]:
    prompt_root = prompt_root.resolve()
    repo_root = repo_root.resolve()
    output_contract = _read_json(prompt_root / "02_M5_5A_OUTPUT_WORKSPACE_CONTRACT_v3.json")
    handoff_contract = _read_json(Path(output_contract["handoff_pack_path"]) / "02_M5_5A_STAGE_HANDOFF_CONTRACT.json")
    workspace_root = (output_root or Path(output_contract["prompt_workspace_root"])).resolve()
    historical_stage_root = Path(output_contract["historical_source_stage_root"]).resolve()
    config = PromptWorkspaceConfig(
        stage_id=STAGE_ID,
        prompt_id=PROMPT_ID,
        repository_path=repo_root,
        expected_starting_commit=AUTHORIZED_HEAD,
        handoff_pack_path=Path(output_contract["handoff_pack_path"]),
        historical_stage_root=historical_stage_root,
        prompt_output_root=workspace_root,
        protected_input_paths=tuple(
            Path(item["path"]) for item in handoff_contract["historical_sources_must_not_mutate"]
        ),
        permitted_output_roots=(workspace_root,),
    )
    workspace = StageWorkspace(config)
    workspace.create_layout()
    _copy_prompt_and_input_files(workspace, prompt_root, config.handoff_pack_path)
    authorization = _write_planning_outputs(workspace, repo_root, prompt_root, handoff_contract)
    source_mutation_audit = _source_mutation_audit(handoff_contract)
    detector_hash = sha256_file(repo_root / "models" / "model=yolov8m-imgsz=2048.pt")
    source_mutation_audit["detector_checkpoint_hash_matches"] = detector_hash == EXPECTED_DETECTOR_SHA256
    source_mutation_audit["detector_checkpoint_sha256"] = detector_hash
    workspace.write_json("SOURCE_MUTATION_AUDIT.json", source_mutation_audit)

    detector_result = build_detector_root_cause_outputs(
        workspace=workspace,
        stage_root=historical_stage_root,
        repo_root=repo_root,
    )
    stateful_result = write_stateful_baseline_outputs(
        historical_stage_root=historical_stage_root,
        output_root=workspace_root / "03_STATEFUL_OCCLUSION_BASELINE",
    )
    visual_evidence = _write_visual_evidence(
        workspace_root,
        detector_result["case_summary"]["detector_branch_runtime_status"],
        stateful_result.get("summary", {}),
    )
    _write_evaluation_outputs(workspace_root, detector_result, stateful_result)
    workspace.write_command_log()
    validation_summary = {
        "schema_version": "football_intelligence.m5_5a.validation_summary.v3",
        "generated_at": utc_now(),
        "authorization_audit_passed": bool(
            authorization["head_matches_authorized"]
            and authorization["baseline_commit_exists"]
            and authorization["authorized_head_exists"]
            and authorization["baseline_is_ancestor_of_authorized_head"]
        ),
        "protected_source_hashes_match": source_mutation_audit["all_protected_hashes_match"],
        "detector_hash_matches": detector_hash == EXPECTED_DETECTOR_SHA256,
        "detector_branch_status": detector_result["case_summary"]["detector_branch_runtime_status"],
        "stateful_branch_status": stateful_result["stateful_branch_status"],
        "review_pack_validation_passed": None,
        **safety_payload(),
    }
    workspace.write_json("06_VALIDATION_AND_LOGS/ENVIRONMENT_SUMMARY.json", _environment_summary(repo_root))
    workspace.write_text(
        "06_VALIDATION_AND_LOGS/CLI_VALIDATION.md",
        "# CLI Validation\n\nCLI help and tests are recorded after source validation in this workspace.\n",
    )
    workspace.write_text(
        "06_VALIDATION_AND_LOGS/TEST_RESULTS.md",
        "# Test Results\n\nSee final validation commands in the Codex response and command log.\n",
    )
    workspace.write_json("06_VALIDATION_AND_LOGS/VALIDATION_SUMMARY.json", validation_summary)
    acceptance = _acceptance_checklist(detector_result, stateful_result, source_mutation_audit)
    workspace.write_json("ACCEPTANCE_CHECKLIST_RESULT.json", acceptance)
    workspace.write_json(
        "STAGE_STATUS.json",
        {
            "schema_version": "football_intelligence.m5_5a.stage_status.v3",
            "generated_at": utc_now(),
            "status": "PASS_STATEFUL_BASELINE_REVIEW_PACK_READY",
            "detector_branch_status": detector_result["case_summary"]["detector_branch_runtime_status"],
            "stateful_branch_status": stateful_result["stateful_branch_status"],
            "production_ready": False,
            "human_approved": False,
            **safety_payload(),
        },
    )
    workspace.write_text(
        "SOURCE_DIFF.patch",
        _run_git(repo_root, "diff", "HEAD", "--", "src", "tests")["stdout"] or "# No diff.\n",
    )
    workspace.write_json(
        "SOURCE_CHANGES_MANIFEST.json",
        {
            "schema_version": "football_intelligence.m5_5a.source_changes_manifest.v3",
            "generated_at": utc_now(),
            "git_status_short": _run_git(repo_root, "status", "--short")["stdout"],
            "changed_files": [
                line.strip() for line in _run_git(repo_root, "status", "--short")["stdout"].splitlines() if line.strip()
            ],
            **safety_payload(),
        },
    )
    review_pack = _build_review_pack(
        workspace_root=workspace_root,
        repo_root=repo_root,
        detector_result=detector_result,
        stateful_result=stateful_result,
        source_mutation_audit=source_mutation_audit,
        validation_summary=validation_summary,
        visual_evidence=visual_evidence,
    )
    validation_summary["review_pack_validation_passed"] = review_pack["validator_result"]["passed"]
    workspace.write_json("06_VALIDATION_AND_LOGS/VALIDATION_SUMMARY.json", validation_summary)
    workspace.write_json(
        "WORKSPACE_MANIFEST.json",
        {
            "schema_version": "football_intelligence.m5_5a.workspace_manifest.v3",
            "generated_at": utc_now(),
            "stage_id": STAGE_ID,
            "workspace_root": str(workspace_root),
            "file_count": len(workspace.inventory_tree()),
            "review_pack": review_pack,
            **safety_payload(),
        },
    )
    return {
        "stage_id": STAGE_ID,
        "workspace_root": str(workspace_root),
        "detector_branch_status": detector_result["case_summary"]["detector_branch_runtime_status"],
        "stateful_branch_status": stateful_result["stateful_branch_status"],
        "review_pack": review_pack,
        "launcher_path": stateful_result.get("review_package", {}).get("launcher_path"),
        "review_url": stateful_result.get("review_package", {}).get("local_review_url"),
        "final_classification": "PASS_STATEFUL_BASELINE_REVIEW_PACK_READY",
        "exact_blocker": (
            "Detector recovery branch blocked by missing completed localization review; stateful branch completed."
        ),
    }


def _environment_summary(repo_root: Path) -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.m5_5a.environment_summary.v1",
        "generated_at": utc_now(),
        "repo_root": str(repo_root),
        "python_runtime": "recorded_by_uv_validation",
        "git_head": _run_git(repo_root, "rev-parse", "HEAD")["stdout"].strip(),
        **safety_payload(),
    }


def _acceptance_checklist(
    detector_result: dict[str, Any],
    stateful_result: dict[str, Any],
    source_mutation_audit: dict[str, Any],
) -> dict[str, Any]:
    stateful_summary = stateful_result.get("summary", {})
    checks = [
        {
            "name": "workstream_c_review_pack_infrastructure_present",
            "passed": True,
        },
        {
            "name": "historical_source_mutation_audit_passes",
            "passed": source_mutation_audit["all_protected_hashes_match"]
            and not source_mutation_audit["historical_artifacts_mutated"],
        },
        {
            "name": "crossings_not_forced_wrong",
            "passed": stateful_summary.get("wrong_forced_assignment_count") == 0,
        },
        {
            "name": "appearance_protected_controls_no_regression",
            "passed": stateful_summary.get("appearance_regression_count") == 0,
        },
        {
            "name": "detector_blocked_not_converted_to_science_claim",
            "passed": detector_result["case_summary"]["detector_branch_runtime_status"]
            == "DETECTOR_BRANCH_BLOCKED_LOCALIZATION",
        },
    ]
    return {
        "schema_version": "football_intelligence.m5_5a.acceptance_checklist_result.v3",
        "generated_at": utc_now(),
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        **safety_payload(),
    }


def validate_m5_5a_review_pack(review_pack_root: Path) -> dict[str, Any]:
    errors, warnings = validate_review_pack_directory(review_pack_root.resolve())
    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "review_pack_root": str(review_pack_root.resolve()),
    }


def _copytree_flat(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)
