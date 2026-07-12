from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.core.fingerprints import sha256_file


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_blind_review_pack(
    *,
    stage_root: Path,
    repo_root: Path,
    prompt_path: Path,
) -> Path:
    pack = stage_root / "review_pack"
    pack.mkdir(parents=True, exist_ok=True)
    guide = (
        "# M5.3 Blind Window Review Pack\n\n"
        "This pack contains exactly 20 files for review. It is VISUAL_ONLY_NOT_METRIC and is not production-ready.\n"
    )
    (pack / "00_REVIEW_GUIDE.md").write_text(guide, encoding="utf-8")
    copy_file(prompt_path, pack / "01_ORIGINAL_PROMPT.txt")
    copy_file(stage_root / "prior_stage_closure/M5_2R_FINAL_CLOSURE.md", pack / "02_M5_2R_FINAL_CLOSURE.md")
    copy_file(stage_root / "selection/blind_window_selection.json", pack / "03_BLIND_WINDOW_SELECTION.json")
    copy_file(stage_root / "selection/blind_window_selection_seal.json", pack / "04_SELECTION_SEAL.json")
    copy_file(stage_root / "source/source_video_manifest.json", pack / "05_SOURCE_VIDEO_MANIFEST.json")
    copy_file(stage_root / "source/artifact_retention_contract.json", pack / "06_ARTIFACT_RETENTION_CONTRACT.json")
    copy_file(
        stage_root / "validation/frame_extraction_repeatability.json",
        pack / "07_FRAME_EXTRACTION_REPEATABILITY.json",
    )
    copy_file(stage_root / "frames/extraction_a/frame_manifest.json", pack / "08_CANONICAL_FRAME_MANIFEST.json")
    copy_file(
        stage_root / "pipeline/frozen_configuration_manifest.json",
        pack / "09_FROZEN_CONFIGURATION_MANIFEST.json",
    )
    copy_file(stage_root / "pipeline/input_closure.json", pack / "10_INPUT_CLOSURE.json")
    copy_file(stage_root / "validation/blind_run_comparison.json", pack / "11_BLIND_RUN_COMPARISON.json")
    copy_file(stage_root / "validation/blind_generalization_report.md", pack / "12_BLIND_GENERALIZATION_REPORT.md")
    copy_file(stage_root / "review/blind_review_candidate_summary.json", pack / "13_REVIEW_CANDIDATE_SUMMARY.json")
    copy_file(stage_root / "review/blind_review_selection_audit.json", pack / "14_REVIEW_SELECTION_AUDIT.json")
    copy_file(stage_root / "review/blind_review_ui_manifest.json", pack / "15_BLIND_REVIEW_UI_MANIFEST.json")
    copy_file(
        repo_root / "src/football_intelligence/replay/blind_window_selection.py",
        pack / "16_blind_window_selection.py",
    )
    copy_file(
        repo_root / "src/football_intelligence/replay/blind_window_extractor.py",
        pack / "17_blind_window_extractor.py",
    )
    copy_file(repo_root / "tests/test_blind_window_selection.py", pack / "18_test_blind_window_selection.py")
    files = sorted(path for path in pack.iterdir() if path.is_file())
    manifest = {
        "schema_version": "m5.blind_window.review_pack_manifest.v1",
        "created_at": utc_now(),
        "file_count": len(files) + 1,
        "expected_file_count": 20,
        "exactly_20_files": len(files) + 1 == 20,
        "files": [
            {"name": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in files
            if path.name != "19_REVIEW_PACK_MANIFEST.json"
        ],
    }
    write_json(pack / "19_REVIEW_PACK_MANIFEST.json", manifest)
    final_files = [path for path in pack.iterdir() if path.is_file()]
    if len(final_files) != 20:
        raise RuntimeError(
            f"review pack must contain exactly 20 files, found {len(final_files)}; "
            "remove or quarantine unexpected review-pack files outside this command before rebuilding"
        )
    return pack
