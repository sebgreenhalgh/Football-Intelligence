from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).parents[1]
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
PACKAGE = STAGE / "08_GPU_REBUILT_CONTINUITY_REVIEW_PACKAGE"
PACK = STAGE / "11_REVIEW_PACK_FOR_CHATGPT"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_cuda_dependency_configuration_is_explicit_and_locked() -> None:
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    lock = (REPO / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "pytorch-cu130"' in pyproject
    assert 'url = "https://download.pytorch.org/whl/cu130"' in pyproject
    assert "torch==2.12.1" in pyproject and "torchvision==0.27.1" in pyproject
    assert 'version = "2.12.1+cu130"' in lock and 'version = "0.27.1+cu130"' in lock
    assert "https://download.pytorch.org/whl/cu130" in lock


def test_primary_cuda_and_checkpoint_outputs_pass() -> None:
    environment = read_json(STAGE / "01_AUTHORIZATION_AND_GPU_PREFLIGHT" / "primary_environment_after.json")
    smoke = read_json(STAGE / "03_GPU_SMOKE_AND_CHECKPOINT_VALIDATION" / "ultralytics_cuda_smoke.json")
    checkpoint = read_json(STAGE / "03_GPU_SMOKE_AND_CHECKPOINT_VALIDATION" / "checkpoint_validation.json")
    assert environment["cuda_available"] is True
    assert "RTX 5060" in environment["gpu_name"]
    assert smoke["model_device"] == "cuda:0" and smoke["no_cpu_fallback"] is True
    assert checkpoint["sha256"] == "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"


def test_gpu_rebuild_is_fresh_and_safe() -> None:
    tracker = read_json(STAGE / "05_GPU_REBUILT_ABSTENTION_FIRST_TRACKER" / "tracker_summary.json")
    level = read_json(STAGE / "06_BENCHMARK_REBUILD_AND_LEVEL4_SEARCH" / "level_summary.json")
    package = read_json(PACKAGE / "review_package_validation.json")
    state = read_json(PACKAGE / "decisions" / "review_decisions.json")
    assert tracker["rebuilt_from_gpu_rows"] is True and tracker["stale_m5_5f0_rows_reused"] is False
    assert tracker["double_assignments"] == 0 and tracker["forced_below_margin"] == 0
    assert level["case_count"] == 12 and level["human_answers_used"] is False
    assert package["passed"] is True and state["decisions"] == {}


def test_review_pack_is_flat_bounded_and_has_source_diff() -> None:
    files = list(PACK.iterdir())
    assert len(files) == 20
    assert sum(path.stat().st_size for path in files) <= 50 * 1024 * 1024
    assert len([path for path in files if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif"}]) <= 3
    assert (PACK / "04_SOURCE_DIFF.patch").exists()
    assert (PACK / "18_BENCHMARK_REVIEW_UI.png").exists()
