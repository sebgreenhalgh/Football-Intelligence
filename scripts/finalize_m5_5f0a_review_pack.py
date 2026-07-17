"""Generate and validate the flat maximum-20-file M5.5F0A handoff pack."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
STAGE = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 2"
    / "M5_5F0A_CUDA_INTEGRATION_AND_GPU_CONTINUITY_BENCHMARK_REBUILD_v1"
)
PACK = STAGE / "11_REVIEW_PACK_FOR_CHATGPT"
REVIEW = STAGE / "08_GPU_REBUILT_CONTINUITY_REVIEW_PACKAGE"
BASELINE = "4a62125853992dfd4424b5404e382aed2b8f7ba9"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False).stdout


def copy_json(name: str, source: Path) -> None:
    text = source.read_text(encoding="utf-8").replace("C:\\Users\\sebgr", "<REDACTED_USER>")
    (PACK / name).write_text(text, encoding="utf-8")


def make_detector_contact_sheet() -> None:
    candidates = sorted(REVIEW.glob("evidence/*/focal/all_*.png"))[:4]
    if not candidates:
        raise RuntimeError("no real GPU detector evidence available for the review pack")
    tiles = []
    for path in candidates:
        with Image.open(path) as image:
            tile = image.convert("RGB")
            tile.thumbnail((720, 360))
            canvas = Image.new("RGB", (720, 400), (12, 20, 34))
            canvas.paste(tile, ((720 - tile.width) // 2, 24))
            ImageDraw.Draw(canvas).text((12, 8), path.parent.parent.name + " / " + path.name, fill=(240, 240, 240))
            tiles.append(canvas)
    sheet = Image.new("RGB", (1440, 800), (12, 20, 34))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 2) * 720, (index // 2) * 400))
    sheet.save(PACK / "17_GPU_DETECTION_COMPARISON.jpg", quality=86, optimize=True)


def main() -> None:
    PACK.mkdir(parents=True, exist_ok=True)
    for path in PACK.iterdir():
        if path.is_file():
            path.unlink()
    copy_json(
        "07_GPU_AND_PRIMARY_ENVIRONMENT.json",
        STAGE / "01_AUTHORIZATION_AND_GPU_PREFLIGHT" / "primary_environment_after.json",
    )
    copy_json(
        "08_CUDA_DEPENDENCY_INTEGRATION.json",
        STAGE / "02_PROJECT_CUDA_DEPENDENCY_INTEGRATION" / "dependency_integration_summary.json",
    )
    copy_json(
        "09_GPU_CHECKPOINT_AND_INFERENCE.json",
        STAGE / "03_GPU_SMOKE_AND_CHECKPOINT_VALIDATION" / "ultralytics_cuda_smoke.json",
    )
    copy_json(
        "10_GPU_DETECTION_RECOVERY.json", STAGE / "04_GPU_LOCAL_DETECTION_RECOVERY" / "detection_recovery_summary.json"
    )
    copy_json("11_GPU_TRACKER_REBUILD.json", STAGE / "05_GPU_REBUILT_ABSTENTION_FIRST_TRACKER" / "tracker_summary.json")
    copy_json(
        "12_BENCHMARK_AND_LEVEL4_RESULTS.json", STAGE / "06_BENCHMARK_REBUILD_AND_LEVEL4_SEARCH" / "level_summary.json"
    )
    copy_json(
        "13_CPU_GPU_COMPARISON.json", STAGE / "07_MACHINE_AND_CPU_GPU_COMPARISON" / "cpu_gpu_detection_comparison.json"
    )
    copy_json("14_REVIEW_PACKAGE_STATUS.json", REVIEW / "review_package_validation.json")
    copy_json("16_ACCEPTANCE_AND_NEXT_STAGE.json", STAGE / "09_EVALUATION_AND_NEXT_STAGE" / "next_stage_decision.json")
    safety = read_json(STAGE / "07_MACHINE_AND_CPU_GPU_COMPARISON" / "acceptance_checklist.json")
    (PACK / "15_SAFETY_AND_MUTATION_AUDIT.json").write_text(
        json.dumps(safety, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    status = git("status", "--short")
    diff = git("diff", "--binary", BASELINE, "HEAD") or git("diff", "--binary")
    diff = diff.replace("C:\\Users\\sebgr", "<REDACTED_USER>")
    (PACK / "04_SOURCE_DIFF.patch").write_text(diff, encoding="utf-8")
    (PACK / "02_RUN_AND_GIT_CONTEXT.json").write_text(
        json.dumps(
            {
                "authorized_baseline": BASELINE,
                "head": git("rev-parse", "HEAD").strip(),
                "worktree_status_redacted": status.replace("C:\\Users\\sebgr", "<REDACTED_USER>"),
                "port_8796": True,
                "port_8795_final_review_forbidden": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    changed = git("diff", "--name-status", BASELINE, "HEAD") or git("diff", "--name-status")
    (PACK / "03_FILES_CHANGED.md").write_text(
        "# Files changed\n\n```text\n"
        + changed
        + "```\n\nOnly repository source/dependency files are changed; match-local outputs are outside Git.\n",
        encoding="utf-8",
    )
    test_results = STAGE / "10_COMMANDS_AND_TESTS" / "test_results.json"
    (PACK / "05_COMMANDS_AND_TEST_RESULTS.md").write_text(
        (
            test_results.read_text(encoding="utf-8")
            if test_results.exists()
            else "# Tests\n\nFinal test record pending.\n"
        ),
        encoding="utf-8",
    )
    outputs = sorted(str(path.relative_to(STAGE)).replace("\\", "/") for path in STAGE.rglob("*") if path.is_file())
    (PACK / "06_OUTPUT_ARTIFACT_INDEX.json").write_text(
        json.dumps(
            {
                "stage_root": "M5_5F0A_CUDA_INTEGRATION_AND_GPU_CONTINUITY_BENCHMARK_REBUILD_v1",
                "file_count": len(outputs),
                "sample_outputs": outputs[:80],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (PACK / "01_EXECUTIVE_SUMMARY.md").write_text(
        "# M5.5F0A GPU continuity rebuild\n\n"
        "A fresh 12-case review package was rebuilt from CUDA detector output. "
        "The primary environment is CUDA-enabled through `pyproject.toml` and `uv.lock`; "
        "the approved checkpoint ran on `cuda:0` with FP16. The benchmark contains "
        "3 Level 1 and 9 Level 2 cases, with no defensible Level 3 or Level 4 cases "
        "after the bounded 1536/2048 search. This is a limited-supply, match-local "
        "visual benchmark; no identity, metrics, occlusion mining or learned rows "
        "were produced.\n",
        encoding="utf-8",
    )
    (PACK / "19_HUMAN_REVIEW_INSTRUCTIONS.md").write_text(
        "# Human review\n\n"
        "Do not use port 8795. Use port 8796 only. First confirm, swap, correct or "
        "reject the temporary anonymous A/B seeds, then review continuity. Notes are "
        "optional for structured outcomes. Do not infer persistent player identity, "
        "player slots, metrics or occlusion truth. Do not return to occlusion until "
        "this completed GPU continuity review passes.\n",
        encoding="utf-8",
    )
    make_detector_contact_sheet()
    ui = STAGE / "10_COMMANDS_AND_TESTS" / "browser_evidence" / "gpu_benchmark_review_ui.png"
    shutil.copy2(ui, PACK / "18_BENCHMARK_REVIEW_UI.png")
    mandatory = [
        "REVIEW_PACK_MANIFEST.json",
        "01_EXECUTIVE_SUMMARY.md",
        "02_RUN_AND_GIT_CONTEXT.json",
        "03_FILES_CHANGED.md",
        "04_SOURCE_DIFF.patch",
        "05_COMMANDS_AND_TEST_RESULTS.md",
        "06_OUTPUT_ARTIFACT_INDEX.json",
        "07_GPU_AND_PRIMARY_ENVIRONMENT.json",
        "08_CUDA_DEPENDENCY_INTEGRATION.json",
        "09_GPU_CHECKPOINT_AND_INFERENCE.json",
        "10_GPU_DETECTION_RECOVERY.json",
        "11_GPU_TRACKER_REBUILD.json",
        "12_BENCHMARK_AND_LEVEL4_RESULTS.json",
        "13_CPU_GPU_COMPARISON.json",
        "14_REVIEW_PACKAGE_STATUS.json",
        "15_SAFETY_AND_MUTATION_AUDIT.json",
        "16_ACCEPTANCE_AND_NEXT_STAGE.json",
        "17_GPU_DETECTION_COMPARISON.jpg",
        "18_BENCHMARK_REVIEW_UI.png",
        "19_HUMAN_REVIEW_INSTRUCTIONS.md",
    ]
    manifest = {
        "schema_version": "football_intelligence.m5_5f0a.review_pack.v1",
        "stage_id": "M5_5F0A_CUDA_INTEGRATION_AND_GPU_CONTINUITY_BENCHMARK_REBUILD_v1",
        "maximum_file_count": 20,
        "maximum_total_bytes": 52428800,
        "maximum_visual_files": 3,
        "files": [
            {"filename": name, "bytes": PACK.joinpath(name).stat().st_size, "sha256": digest(PACK / name)}
            for name in mandatory
            if name != "REVIEW_PACK_MANIFEST.json"
        ],
    }
    (PACK / "REVIEW_PACK_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = list(PACK.iterdir())
    visual = [path for path in files if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif"}]
    total = sum(path.stat().st_size for path in files)
    assert (
        len(files) == 20
        and total <= 52428800
        and len(visual) <= 3
        and (PACK / "04_SOURCE_DIFF.patch").stat().st_size >= 0
    )
    print(
        json.dumps(
            {
                "pack": str(PACK),
                "file_count": len(files),
                "total_bytes": total,
                "visual_files": [path.name for path in visual],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
