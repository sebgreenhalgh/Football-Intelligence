"""Check or serve the atomic-state G7F-C calibration adjudication reviewer R2."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from football_intelligence.calibration_adjudication import CALIBRATION_IDS, PASS_KIND
from football_intelligence.calibration_adjudication_reviewer import (
    CalibrationAdjudicationHTTPServer,
    CalibrationAdjudicationReviewerConfig,
    run_server,
)


REVIEWER_RELEASE = "G7F_C_CALIBRATION_ADJUDICATION_REVIEWER_R2"
EXPECTED_INTERPRETER = Path(r"C:\Users\sebgr\anaconda3\envs\fi-reviewer\python.exe")
BASELINE = "548f1d17bb75632670fb576e3614eb341db133df"
EXPECTED_HASHES = {
    "01_SELECTION/dense_gold_selection_manifest.json": (
        "f454e0e93ec2cb01f5edeba1b6545a9e607a26ec8ec58528bccc7f5e12618eef"
    ),
    "02_ONTOLOGY/dense_person_gold_ontology.json": "410e7a704ed750fc6e08b064d08578075c751f556d2439f7e15a350430beac5b",
    "03_METRICS/dense_gold_metric_protocol.json": "c28a9d467719dae1944abe2594080e45300b40d46d79e68d636ee1601c8262cd",
    "04_SCHEMAS/dense_person_frame_annotation.schema.json": (
        "0449998705cffec7d718f7c00cb2249f8b6a35f933a7768aa76db12ea20e8d0c"
    ),
    "04_SCHEMAS/dense_person_frame_acknowledgement.schema.json": (
        "cc902281a2f5fa14f97a3ad186c8bd565ed748855c6fd445cf35f906cbf94d75"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def roots(repo: Path) -> dict[str, Path]:
    part9 = repo.parent / "experiments/football_observation_reasoner/part 9"
    return {
        "repo": repo,
        "source": part9 / "G7F_C_DENSE_PERSON_GOLD_DISCRIMINATION_SET_v1",
        "audit": part9 / "G7F_C_CALIBRATION_6_HUMAN_ANNOTATION_QUALITY_GATE_v1",
        "phase_a": part9 / "G7F_C_CALIBRATION_SUPERSEDING_ADJUDICATION_AND_REAUDIT_v1",
        "stage": part9 / "G7F_C_ADJUDICATION_R1_NAVIGATION_STATE_DESYNC_REPAIR_v1",
    }


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def verify_environment() -> dict[str, str]:
    if Path(sys.executable).resolve() != EXPECTED_INTERPRETER.resolve():
        raise RuntimeError(f"use the fi-reviewer interpreter: {EXPECTED_INTERPRETER}; found {sys.executable}")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"G7F-C adjudication requires Python 3.12; found {sys.version.split()[0]}")
    try:
        import cv2
        import numpy
    except ImportError as exc:
        raise RuntimeError(f"missing reviewer dependency: {exc.name}") from exc
    return {"python": sys.version.split()[0], "numpy": numpy.__version__, "opencv_python": cv2.__version__}


def verify_frozen_contracts(source: Path) -> dict[str, str]:
    actual = {relative: sha256_file(source / relative) for relative in EXPECTED_HASHES}
    if actual != EXPECTED_HASHES:
        raise RuntimeError(f"frozen G7F-C contract hash mismatch: {actual}")
    return actual


def verify_release(paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, str]]:
    stage, repo = paths["stage"], paths["repo"]
    release = read_json(stage / "05_REVIEWER/reviewer_release_manifest.json")
    commit = str(release.get("repository_commit", ""))
    head = git(repo, "rev-parse", "HEAD")
    origin = git(repo, "rev-parse", "origin/main")
    if not commit or head != commit or origin != commit:
        raise RuntimeError(f"R2 reviewer requires HEAD == origin/main == {commit}; found {head}, {origin}")
    if release.get("required_baseline") != BASELINE or release.get("reviewer_release") != REVIEWER_RELEASE:
        raise RuntimeError("R2 reviewer release identity or baseline binding mismatch")
    for row in release.get("files", []):
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size != row["byte_size"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"R2 reviewer release file mismatch: {path}")
    release_bindings = read_json(stage / "05_REVIEWER/reviewer_binding_hashes.json")
    truth_bindings = release_bindings.get("adjudication_truth_bindings")
    if not isinstance(truth_bindings, dict):
        raise RuntimeError("R2 release does not contain immutable adjudication truth bindings")
    if release_bindings.get("reviewer_release_manifest_sha256") != sha256_file(
        stage / "05_REVIEWER/reviewer_release_manifest.json"
    ):
        raise RuntimeError("R2 reviewer release manifest hash mismatch")
    return release, {str(key): str(value) for key, value in truth_bindings.items()}


def verify_original_inventory(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    inventory = read_json(paths["phase_a"] / "01_FREEZE/original_real_decisions_inventory.json")
    decision_root = paths["source"] / "06_DENSE_DECISIONS"
    for row in inventory["files"]:
        relative = Path(row["relative_path"])
        path = decision_root / relative
        if not path.is_file() or path.stat().st_size != row["byte_size"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"immutable original decision artifact changed: {relative}")
    return {
        image_id: {
            "event_file_sha256": row["event_file_sha256"],
            "acknowledgement_file_sha256": row["acknowledgement_file_sha256"],
        }
        for image_id, row in inventory["calibration_parents"].items()
    }


def reviewer_config(repo: Path, *, port: int = 8792) -> CalibrationAdjudicationReviewerConfig:
    paths = roots(repo)
    verify_frozen_contracts(paths["source"])
    _, bindings = verify_release(paths)
    frozen_parents = verify_original_inventory(paths)
    assessment = paths["audit"] / "03_MANUAL_VISUAL_ASSESSMENT.json"
    expected_assessment_sha = bindings["audit_manual_visual_assessment_sha256"]
    if sha256_file(assessment) != expected_assessment_sha:
        raise RuntimeError("calibration audit checklist bytes changed")
    return CalibrationAdjudicationReviewerConfig(
        selection_manifest_path=paths["source"] / "01_SELECTION/dense_gold_selection_manifest.json",
        assets_root=paths["source"] / "05_REVIEWER/assets",
        original_decisions_root=paths["source"] / "06_DENSE_DECISIONS",
        adjudication_root=paths["source"] / "06_DENSE_DECISIONS/calibration_adjudication",
        binding_hashes=bindings,
        reviewer_release=REVIEWER_RELEASE,
        audit_assessment_path=assessment,
        audit_checklist_sha256=expected_assessment_sha,
        frozen_parent_file_hashes=frozen_parents,
        port=port,
    )


def check(config: CalibrationAdjudicationReviewerConfig) -> dict[str, Any]:
    server = CalibrationAdjudicationHTTPServer(config)
    try:
        bootstrap = server.bootstrap()
        states = [server.store.state(image_id) for image_id in CALIBRATION_IDS]
    finally:
        server.server_close()
    if [row["anonymous_dense_image_id"] for row in bootstrap["queue"]] != list(CALIBRATION_IDS):
        raise RuntimeError("adjudication queue is not exactly DG-001..DG-006")
    return {
        "reviewer_release": REVIEWER_RELEASE,
        "workflow": PASS_KIND,
        "queue": list(CALIBRATION_IDS),
        "finalized_adjudications": sum(bool(row["finalized"]) for row in states),
        "mutable_drafts": [
            row["anonymous_dense_image_id"] for row in states if not row["finalized"] and row["revision"] > 0
        ],
        "scored_annotation_authorized": False,
        "scored_annotation_started": False,
        "production_ready": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("check", "serve"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--port", type=int, default=8792)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    environment = verify_environment()
    config = reviewer_config(args.repo.resolve(), port=args.port)
    if args.phase == "check":
        print(json.dumps({**check(config), "environment": environment}, sort_keys=True))
        return 0
    print(f"G7F-C calibration adjudication reviewer R2: http://{config.host}:{config.port}/")
    print("Only DG-001..DG-006 are available. Scored annotation is not authorized. production_ready=false")
    run_server(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
