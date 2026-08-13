"""R6.7.1 real-mode release-gate closure tests."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from football_intelligence.temporal_review import (
    R6_7_RELEASE_CLASSIFICATION,
    R6_7_RELEASE_GATE_NAME,
    R6_7_RELEASE_REVISION,
    TemporalReviewStore,
)


PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
STAGE = PROJECT / "experiments/football_observation_reasoner/part 8/G7E_B_R6_7_1_REAL_MODE_RELEASE_GATE_CLOSURE_v1"
BUILDER = STAGE / "build_r6_7_1.py"
PACKAGE = STAGE / "03_REAL_MODE_RELEASE_GATE_IMPLEMENTATION/temporal_reviewer_r6_7_1"


@pytest.fixture(scope="module", autouse=True)
def fresh_package() -> None:
    subprocess.run([sys.executable, str(BUILDER)], check=True)


def make_store(package: Path, tmp_path: Path) -> TemporalReviewStore:
    return TemporalReviewStore(package, tmp_path / "real", tmp_path / "practice", acceptance_mode=False)


def copied_package(tmp_path: Path) -> Path:
    target = tmp_path / "package"
    shutil.copytree(PACKAGE, target)
    return target


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def test_fresh_r6_7_package_has_first_class_release_metadata_and_starts_normally(tmp_path: Path) -> None:
    build = json.loads((PACKAGE / "build_manifest.json").read_text(encoding="utf-8"))
    gate = json.loads((PACKAGE / R6_7_RELEASE_GATE_NAME).read_text(encoding="utf-8"))

    store = make_store(PACKAGE, tmp_path)

    assert build["release_revision"] == R6_7_RELEASE_REVISION
    assert gate["release_classification"] == R6_7_RELEASE_CLASSIFICATION
    assert gate["runtime_source_snapshot_sha256"]
    assert store._cached_release_gate_status["valid"] is True


@pytest.mark.parametrize(
    "mutation",
    ["release_revision", "classification", "action_contract", "runtime_python", "reviewer_js", "delete_gate"],
)
def test_r6_7_release_gate_fails_closed_for_each_bound_surface(tmp_path: Path, mutation: str) -> None:
    package = copied_package(tmp_path)
    if mutation == "release_revision":
        manifest_path = package / "build_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["release_revision"] = "INVALID_RELEASE_REVISION"
        write_json(manifest_path, manifest)
    elif mutation == "classification":
        gate_path = package / R6_7_RELEASE_GATE_NAME
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        gate["release_classification"] = "NOT_A_PASS"
        write_json(gate_path, gate)
    elif mutation == "action_contract":
        gate_path = package / R6_7_RELEASE_GATE_NAME
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        gate["action_contract_sha256"] = "0" * 64
        write_json(gate_path, gate)
    elif mutation == "runtime_python":
        path = package / "runtime_source_snapshot/football_intelligence/g7e_b_r6_action_reducer.py"
        path.write_bytes(path.read_bytes() + b"\n# temporary gate mutation\n")
    elif mutation == "reviewer_js":
        path = package / "review.js"
        path.write_bytes(path.read_bytes() + b"\n// temporary gate mutation\n")
    else:
        (package / R6_7_RELEASE_GATE_NAME).unlink()

    with pytest.raises(RuntimeError, match="REAL_REVIEW_RELEASE_GATE_INVALID_AT_STARTUP"):
        make_store(package, tmp_path)


def test_existing_release_gate_paths_remain_explicit() -> None:
    source = (PROJECT / "SoccerTrack-v2/src/football_intelligence/temporal_review.py").read_text(encoding="utf-8")

    assert "G7E_B_R6_3_REAL_REVIEW_RELEASE_GATE.json" in source
    assert "G7E_B_R6_2_REAL_REVIEW_RELEASE_GATE.json" in source
    assert "G7E_B_R6_1_REAL_REVIEW_RELEASE_GATE.json" in source
    assert R6_7_RELEASE_GATE_NAME in source
