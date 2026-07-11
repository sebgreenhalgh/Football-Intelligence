from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def build_review_pack(
    *,
    stage_root: Path,
    left_run: Path,
    right_run: Path,
    artifact_root: Path,
    repo_root: Path,
    prompt_path: Path,
) -> Path:
    review_pack = stage_root / "review_pack"
    review_pack.mkdir(parents=True, exist_ok=True)
    comparison = read_json(stage_root / "replay_run_comparison.json")
    validation_summary = read_json(right_run / "validation/replay_validation_summary.json")
    guide = f"""# M5.2 Review Guide

Stage root:
`{stage_root}`

Replay runs:
`{left_run}`
`{right_run}`

Preserved M4 path:
`{artifact_root / "matches/128058/calibration/step2_visual_continuity/step2m4_sparse_handoff_package"}`

Selected M5.1 baseline:
`{artifact_root / "matches/128058/runs/step_m5/02_infrastructure_hardening/runs/m5_baseline_20260711T125508Z_325fa715"}`

Expected structured hash:
`6b7db49e662a39eab7c860c4d0c36dc5617d80b7f8cd7a4a63ad2037e3ca3149`

Observed structured hash:
`{validation_summary.get("reconstructed_structured_content_hash")}`

Observed evidence hash:
`{validation_summary.get("evidence_inventory_hash")}`

Verdict:
`{"PASS" if comparison.get("passed") and validation_summary.get("passed") else "FAIL"}`

Known limitations:
- The current external artifact tree does not contain
  `step2m1_visual_continuity_node_rows.json`; this replay therefore isolates
  and verifies the current M4 package and M3T decision binding without
  fabricating missing Step1/M1 inputs.
- Pathlets are not identities. `pathlet_id` and `visual_continuity_group_id` must not be interpreted as identity IDs.
"""
    (review_pack / "00_REVIEW_GUIDE.md").write_text(guide, encoding="utf-8")
    copy_file(prompt_path, review_pack / "01_ORIGINAL_PROMPT.txt")
    (review_pack / "02_CHANGE_SUMMARY.md").write_text(
        "# Change Summary\n\n"
        "M5.2 adds isolated replay config, input closure, decision binding, "
        "reconstructed M4 package isolation, differential reports, CLI commands, "
        "tests, and a capped review pack.\n",
        encoding="utf-8",
    )
    commands = (
        read_json(stage_root / "validation/command_results.json")
        if (stage_root / "validation/command_results.json").exists()
        else {}
    )
    (review_pack / "03_VALIDATION_SUMMARY.md").write_text(
        "# Validation Summary\n\n"
        f"Replay comparison passed: `{comparison.get('passed')}`\n\n"
        f"Right-run validation passed: `{validation_summary.get('passed')}`\n\n"
        f"Commands recorded:\n```json\n{json.dumps(commands, indent=2, sort_keys=True)}\n```\n",
        encoding="utf-8",
    )
    (review_pack / "04_REPLAY_ARCHITECTURE.md").write_text(
        "# Replay Architecture\n\n"
        "The runner separates repo and artifact roots, seals an input closure, "
        "writes only inside the M5.2 run root, mirrors the preserved M4 package "
        "into `reconstructed_m4`, and compares structured, media, viewer, "
        "guardrail, decision, and source-mutation contracts.\n",
        encoding="utf-8",
    )
    copy_file(right_run / "replay/input_closure.json", review_pack / "05_INPUT_CLOSURE.json")
    copy_file(stage_root / "replay_run_comparison.json", review_pack / "06_REPLAY_RUN_COMPARISON.json")
    copy_file(right_run / "validation/structured_diff.json", review_pack / "07_STRUCTURED_DIFF.json")
    copy_file(right_run / "validation/media_diff.json", review_pack / "08_MEDIA_DIFF.json")
    copy_file(right_run / "validation/viewer_diff.json", review_pack / "09_VIEWER_DIFF.json")
    copy_file(right_run / "validation/guardrail_audit.json", review_pack / "10_GUARDRAIL_AUDIT.json")
    copy_file(right_run / "validation/source_root_mutation_check.json", review_pack / "11_SOURCE_MUTATION_CHECK.json")
    copy_file(
        right_run / "validation/legacy_test_dependency_report.json",
        review_pack / "12_LEGACY_TEST_DEPENDENCY_REPORT.json",
    )
    copy_file(right_run / "run_manifest.json", review_pack / "13_RUN_MANIFEST.json")
    copy_file(repo_root / "src/football_intelligence/replay/m4_engine.py", review_pack / "14_m4_engine.py")
    copy_file(repo_root / "src/football_intelligence/replay/m4_renderer.py", review_pack / "15_m4_renderer.py")
    copy_file(repo_root / "src/football_intelligence/replay/differential.py", review_pack / "16_differential.py")
    copy_file(repo_root / "src/football_intelligence/replay/runner.py", review_pack / "17_runner.py")
    copy_file(
        repo_root / "tests/integration/test_m4_replay_structured_parity.py",
        review_pack / "18_test_m4_replay_structured_parity.py",
    )
    files = sorted(
        path.name for path in review_pack.iterdir() if path.is_file() and path.name != "19_REVIEW_PACK_MANIFEST.json"
    )
    files.append("19_REVIEW_PACK_MANIFEST.json")
    write_json(
        review_pack / "19_REVIEW_PACK_MANIFEST.json",
        {"schema_version": "m5.replay.review_pack_manifest.v1", "file_count": len(files), "files": files},
    )
    actual = [path for path in review_pack.iterdir() if path.is_file()]
    if len(actual) != 20:
        raise RuntimeError(f"review pack must contain exactly 20 files, found {len(actual)}")
    return review_pack
