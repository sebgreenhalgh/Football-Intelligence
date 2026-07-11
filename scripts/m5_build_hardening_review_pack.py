from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from football_intelligence.validation.baseline_integrity import compare_baseline_runs, read_json  # noqa: E402


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the capped M5.1 ChatGPT review pack.")
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--left-run", required=True, type=Path)
    parser.add_argument("--right-run", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    args = parser.parse_args()

    stage_root = args.artifact_root / "matches/128058/runs/step_m5/02_infrastructure_hardening"
    review_pack = stage_root / "review_pack"
    review_pack.mkdir(parents=True, exist_ok=True)

    comparison = compare_baseline_runs(args.left_run, args.right_run)
    historical = read_json(stage_root / "validation/historical_relocated_run_assessment.json")
    guardrail = read_json(args.right_run / "validation/guardrail_audit.json")
    registry = read_json(args.right_run / "validation/registry_integrity_report.json")
    structured = read_json(args.right_run / "baseline/m4_structured_fingerprints.json")

    guide = f"""# M5.1 Review Guide

Stage root:
`{stage_root}`

Canonical run parent:
`{stage_root / "runs"}`

Review focus:
- repo_root and artifact_root are separate.
- run_uri locations are canonical and validated against artifact_root.
- moved M5.0 runs are assessed as historical relocated captures.
- registry finalization rejects duplicate IDs, missing parents, cycles, stale hashes, and self-parents.
- semantic fingerprinting uses explicit runtime-field policy.
- M4 structured fingerprints are parsed from source JSON/JSONL.GZ without rebuilding M4.
- safety fields fail closed when missing or mismatched.
"""
    (review_pack / "00_REVIEW_GUIDE.md").write_text(guide, encoding="utf-8")
    copy_file(args.prompt, review_pack / "01_ORIGINAL_PROMPT.txt")
    (review_pack / "02_CHANGE_SUMMARY.md").write_text(
        "# Change Summary\n\n"
        "M5.1 hardens roots, manifests, registries, fingerprints, guardrails, CLI validation, "
        "quarantine handling, and canonical external baseline capture.\n",
        encoding="utf-8",
    )
    (review_pack / "03_VALIDATION_SUMMARY.md").write_text(
        "# Validation Summary\n\n"
        "See canonical run validation outputs and command transcript in the final agent response.\n",
        encoding="utf-8",
    )
    (review_pack / "04_PATH_ROOT_MODEL.md").write_text(
        "# Path Root Model\n\n"
        "`repo_root` resolves source/config/test files. `artifact_root` resolves `matches/...` artifacts "
        "and canonical run outputs.\n",
        encoding="utf-8",
    )
    write_json(review_pack / "05_HISTORICAL_RELOCATED_RUN_ASSESSMENT.json", historical)
    write_json(review_pack / "06_CANONICAL_CAPTURE_COMPARISON.json", comparison)
    write_json(review_pack / "07_GUARDRAIL_AUDIT.json", guardrail)
    write_json(review_pack / "08_REGISTRY_INTEGRITY_REPORT.json", registry)
    write_json(review_pack / "09_M4_STRUCTURED_FINGERPRINTS.json", structured)

    sources = [
        ("src/football_intelligence/core/path_roots.py", "10_core_path_roots.py"),
        ("src/football_intelligence/core/config.py", "11_core_config.py"),
        ("src/football_intelligence/core/run_context.py", "12_core_run_context.py"),
        ("src/football_intelligence/core/artifact_registry.py", "13_core_artifact_registry.py"),
        ("src/football_intelligence/core/fingerprints.py", "14_core_fingerprints.py"),
        ("src/football_intelligence/core/guardrails.py", "15_core_guardrails.py"),
        ("src/football_intelligence/core/manifest.py", "16_core_manifest.py"),
        ("src/football_intelligence/cli/app.py", "17_cli_app.py"),
        ("tests/integration/test_external_artifact_root.py", "18_test_external_artifact_root.py"),
    ]
    for source, name in sources:
        copy_file(REPO_ROOT / source, review_pack / name)

    manifest = {
        "schema_version": "m5.review_pack_manifest.v1",
        "file_count": 20,
        "files": sorted(
            path.name
            for path in review_pack.iterdir()
            if path.is_file() and path.name != "19_REVIEW_PACK_MANIFEST.json"
        ),
    }
    manifest["files"].append("19_REVIEW_PACK_MANIFEST.json")
    write_json(review_pack / "19_REVIEW_PACK_MANIFEST.json", manifest)
    actual = [path for path in review_pack.iterdir() if path.is_file()]
    if len(actual) != 20:
        raise RuntimeError(f"review pack must contain exactly 20 files, found {len(actual)}")
    print(review_pack.as_posix())


if __name__ == "__main__":
    main()
